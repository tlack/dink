"""Intensity inverter: graded phrasings of one change, with values computed in code.

Intensifiers outside the generator's list ("hot as hell", "super bright", "crank it") were
unstable between runs: the model has no graded sense of amount. For each adjustable device
(an action that sets a field with a real range) and direction, one call writes phrasings at
four levels:
  small  "a bit warmer", "just a touch cooler"            -> 8% of the range
  plain  "warmer please", "it's cold in here"             -> 20%
  big    "way warmer", "crank the heat", "hot as hell"    -> 45%
  max    "as hot as it goes", "all the way up"            -> the end of the range
Values come from code (the current value +- the level's share of the range, clamped), not
the LLM, so amounts are consistent everywhere. A second call classifies each phrasing's
direction and level without seeing the first; only matches are kept. Each utterance carries
the whole ladder ({level: value}), so the converter can offer the other levels as decoys.

Devices come from the LLM worlds plus small smart-home rooms (thermostat, lights, fan,
blinds), where the unstable hard cases live. Exact hard-case prompts are never kept.

  uv run python -u -m dink.llm.intensity --worlds 10 --out data/llm/intensity_pilot.jsonl
  uv run python -u -m dink.llm.intensity --workers 6
"""
import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .openrouter import Budget, Client
from .worlds import _STR, _obj, _schema, ask, clean, fmt_actions, fmt_state

OUT = Path("data/llm/intensity.jsonl")
LEVELS = {"small": 0.08, "plain": 0.2, "big": 0.45, "max": None}

WRITE_SYSTEM = """You write test data for a tiny model inside a gadget. You get the gadget's current state and
a numbered list of changes, each "raise" or "lower" one setting. For each change write things a
person might say, at four strengths:
- "small": a slight change ("a bit warmer", "just a touch dimmer", "nudge the fan up")
- "plain": an ordinary change with no strength word ("warmer please", "dim the lights", or a
  plain complaint: "it's cold in here")
- "big": a strong change ("way warmer", "crank the heat", "i'm freezing", "hot as hell")
- "max": all the way to the limit ("as hot as it goes", "full blast", "all the way down")
Write 3 phrasings per strength: vary the wording (commands, complaints, wishes, slang), keep
the direction clear, and don't reuse the same intensifier. Lowercase, at most 60 characters,
no quotes, no digits.

Reply with a JSON object whose "changes" list has one entry per change: "i" (its number) and
"small", "plain", "big", "max" (3 phrasings each)."""

JUDGE_SYSTEM = """You judge requests to a gadget. You get its current state and numbered requests. For each,
say which way it asks to move its setting ("up" or "down") and how strongly: "small" (a
slight change), "plain" (an ordinary change, no strength given), "big" (a strong change) or
"max" (all the way to the limit).

Reply with a JSON object whose "labels" list has one entry per request: "i", "direction",
"strength"."""

WRITE_USER = """State: {state}
Actions: {actions}
Changes:
{changes}"""

JUDGE_USER = """State: {state}
Actions: {actions}
Requests:
{requests}"""

WRITE_SCHEMA = _schema("intensity", _obj(changes={"type": "array", "items": _obj(
    i={"type": "integer"}, small={"type": "array", "items": _STR}, plain={"type": "array", "items": _STR},
    big={"type": "array", "items": _STR}, max={"type": "array", "items": _STR})}))
JUDGE_SCHEMA = _schema("judge", _obj(labels={"type": "array", "items": _obj(
    i={"type": "integer"}, direction={"type": "string", "enum": ["up", "down"]},
    strength={"type": "string", "enum": list(LEVELS)})}))


def ladder(a, cur, up):
    """{level: value} for one direction; integer settings take whole steps."""
    lo, hi = a["lo"], a["hi"]
    span = hi - lo
    whole = all(float(x).is_integer() for x in (lo, hi, cur))
    out = {}
    for lvl, share in LEVELS.items():
        v = (hi if up else lo) if share is None else min(hi, max(lo, cur + (1 if up else -1) * share * span))
        v = round(v) if whole or span >= 10 else round(v, 2)
        out[lvl] = v
    return out


def home_rooms():
    """Small smart-home "worlds": one room's thermostat, lights, fan and blinds."""
    rng = random.Random(3)
    rooms = ["bedroom", "kitchen", "living room", "office", "bathroom", "nursery", "basement", "guest room"]
    out = []
    for room in rooms:
        for i in range(3):
            r = room.replace(" ", "_")
            state = [{"name": f"{r}_thermostat", "value": rng.randint(62, 78), "lo": 60, "hi": 85, "unit": "F"},
                     {"name": f"{r}_light", "value": round(rng.uniform(0.2, 0.8), 2), "lo": 0, "hi": 1, "unit": "level"},
                     {"name": f"{r}_fan", "value": rng.randint(1, 2), "lo": 0, "hi": 3, "unit": "speed"},
                     {"name": f"{r}_blinds", "value": round(rng.uniform(0.2, 0.8), 2), "lo": 0, "hi": 1, "unit": "open"}]
            actions = [{"name": f"set_{f['name']}", "lo": f["lo"], "hi": f["hi"], "unit": f["unit"], "sets": f["name"]}
                       for f in state]
            out.append({"setting": f"Smart home {room} {i + 1}", "about": f"A {room} with a thermostat, lights, a fan and blinds.",
                        "area": "smart home", "state": state, "actions": actions, "utterances": []})
    return out


def changes_for(w, rng, n=3):
    """Up to n (action, field, direction) changes with room to move."""
    st = {f["name"]: f for f in w["state"]}
    out = []
    acts = [a for a in w["actions"] if a["lo"] is not None and a["sets"] in st and a["hi"] - a["lo"] > 1
            or a["lo"] is not None and a["sets"] in st and not float(st[a["sets"]]["value"]).is_integer()]
    rng.shuffle(acts)
    for a in acts:
        cur = st[a["sets"]]["value"]
        for up in rng.sample([True, False], 2):
            room = (a["hi"] - cur) if up else (cur - a["lo"])
            lad = ladder(a, cur, up)
            distinct = len(set(lad.values())) == len(lad) and cur not in lad.values()
            if room >= 0.5 * (a["hi"] - a["lo"]) and distinct:   # four different amounts, all a real change
                out.append((a, st[a["sets"]], up))
                break
        if len(out) >= n:
            break
    return out


def run_world(c, w, hard_prompts, rng):
    ch = changes_for(w, rng)
    if not ch:
        return []
    listing = "\n".join(f"{i + 1}. {'raise' if up else 'lower'} {f['name']} (now {f['value']:g}, "
                        f"{a['lo']:g}..{a['hi']:g}) with {a['name']}" for i, (a, f, up) in enumerate(ch))
    got = ask(c, WRITE_USER.format(state=fmt_state(w["state"]), actions=fmt_actions(w["actions"]), changes=listing),
              WRITE_SYSTEM, max_tokens=4000, temperature=0.9, response_format=WRITE_SCHEMA)
    cands = []
    for item in got.get("changes") or []:
        i = item.get("i")
        if not isinstance(i, int) or not 1 <= i <= len(ch):
            continue
        a, f, up = ch[i - 1]
        lad = ladder(a, f["value"], up)
        for lvl in LEVELS:
            for say in (item.get(lvl) or [])[:3]:
                say = clean(say)
                if say and say not in hard_prompts and not any(c.isdigit() for c in say):
                    cands.append({"say": say, "level": lvl, "up": up, "action": a["name"], "ladder": lad})
    if not cands:
        return []
    listing = "\n".join(f"{i + 1}. {x['say']}" for i, x in enumerate(cands))
    lab = ask(c, JUDGE_USER.format(state=fmt_state(w["state"]), actions=fmt_actions(w["actions"]), requests=listing),
              JUDGE_SYSTEM, max_tokens=4000, temperature=0.2, response_format=JUDGE_SCHEMA)
    second = {x.get("i"): x for x in lab.get("labels") or [] if isinstance(x, dict)}
    out = []
    for i, x in enumerate(cands):
        j = second.get(i + 1) or {}
        ag = j.get("direction") == ("up" if x["up"] else "down") and j.get("strength") == x["level"]
        out.append({"kind": "int_" + x["level"], "say": x["say"], "source": "intensity", "action": x["action"],
                    "value": x["ladder"][x["level"]], "ladder": x["ladder"], "agree": ag, "flag": None,
                    "judged": {"direction": j.get("direction"), "strength": j.get("strength")}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/llm/worlds.jsonl")
    ap.add_argument("--worlds", type=int, default=0, help="only the first N LLM worlds (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-requests", type=int, default=3000)
    ap.add_argument("--max-spend", type=float, default=10.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from ..evals.hard import CASES, GATE
    hard_prompts = {p for p, _ in CASES} | {p for p, _, _ in GATE}
    from ..gen.worlds import load
    worlds = load(a.src)
    if a.worlds:
        worlds = worlds[:a.worlds]
    worlds = home_rooms() + worlds
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["setting"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [w for w in worlds if w["setting"] not in done]
    print(f"{len(done)} done; {len(todo)} to do ({sum(w['area'] == 'smart home' for w in todo)} smart-home rooms)", flush=True)
    c = Client.parasail(max_requests=a.max_requests, max_spend_per_day=a.max_spend)
    c.reasoning = "off"
    lock, stats = threading.Lock(), {"worlds": 0, "utts": 0, "agree": 0}
    with ThreadPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(run_world, c, w, hard_prompts, random.Random(hash(w["setting"]) & 0xffff)): w for w in todo}
        for j in as_completed(jobs):
            w = jobs[j]
            try:
                utts = j.result()
            except Budget as e:
                print(f"stopping: {e}")
                pool.shutdown(cancel_futures=True)
                break
            except RuntimeError as e:
                print(f"  error {w['setting']}: {e}")
                continue
            rec = {k: w[k] for k in ("setting", "about", "area", "state", "actions")}
            rec["utterances"] = utts
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                stats["worlds"] += 1
                stats["utts"] += len(utts)
                stats["agree"] += sum(u["agree"] for u in utts)
                if stats["worlds"] % 50 == 0 or stats["worlds"] <= 4:
                    print(f"  {stats['worlds']} worlds, {stats['utts']} phrasings, "
                          f"{stats['agree'] / max(1, stats['utts']):.0%} agree", flush=True)
    print(f"{stats} -> {out}\n{c.summary()}")


if __name__ == "__main__":
    main()
