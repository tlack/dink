"""Inverter: three contrast variants for each validated request in a world, independently relabeled.

The model's remaining failures are framing: the same key word means opposite actions
("it's too cold" = warmer, "i want it cold" = cooler; "it's too dark" vs "make it dark"), and
negation. For each kept request of the kinds below (dink/gen/worlds.py validation), one call
writes three variants, keeping the request's key words where it can:
  opposite   same framing, the other direction or state ("a bit warmer" -> "a bit cooler")
  reframe    complaint <-> desire about the same quality, same key word
             ("it's too cold in here" <-> "i want it cold in here")
  negate     tells the gadget NOT to do it, or that no change is wanted -> "none"
             ("don't make it warmer", "leave the pump alone")
A second, independent call labels the variants without seeing the first labels; only agreeing
variants are used (dink/gen/worlds.py checks them like any other utterance). Prompts are
static-first for the provider cache; strict JSON schemas; reasoning off.

  uv run python -u -m dink.llm.invert --worlds 20 --out data/llm/inverted_pilot.jsonl   # pilot
  uv run python -u -m dink.llm.invert --workers 6                                      # all worlds

Resumable: worlds already in --out are skipped, and every reply is cached on disk.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .openrouter import Budget, Client
from .worlds import _NNUM, _STR, _obj, _schema, agree, ask, check_label, clean, fmt_actions, fmt_state

OUT = Path("data/llm/inverted.jsonl")
KINDS = ["direct", "relative_small", "relative_big", "complaint", "desire", "slang", "no_value"]
TYPES = ["opposite", "reframe", "negate"]

INVERT_SYSTEM = """You write contrast test data for a tiny model inside a gadget. You get the gadget's current
state (numeric fields with ranges), its actions, and numbered requests with their right answers.

For each request write exactly three variants. Keep the request's key words wherever you can:
the point is that similar words lead to different answers.
- "opposite": the same kind of sentence asking for the other direction or state
  ("a bit warmer" -> "a bit cooler", "turn on the pump" -> "turn off the pump").
- "reframe": switch between describing a problem and stating a wish about the same quality,
  keeping its key word. A problem implies the fix away from it ("it's too cold in here" =
  warmer); a wish asks for it ("i want it cold in here" = cooler). If the request is neither,
  turn it into one of them.
- "negate": tells the gadget not to do it, or that no change is wanted ("don't make it
  warmer", "leave the pump alone", "no need to open the vent"). Its answer is always "none".

Each variant has the right answer for the current state: exactly one action with the absolute
value it sets (inside the action's lo..hi), or "none" if no listed action fits (including a
change past lo or hi). Relative amounts come from the current value of the field the action
"sets": "a bit" is about 5 to 10 percent of the range, plain about 15 to 25 percent, "a lot"
about 30 to 60 percent. Switching on or fully open = hi, off or closed = lo. Actions without a
value take value null. All variants lowercase, at most 60 characters, no quotes.

Reply with a JSON object whose "items" list has one entry per request: "i" (the request's
number) and "variants" (three, each with "type", "say", "action" and "value")."""

INVERT_USER = """State: {state}
Actions: {actions}
Requests:
{requests}"""

LABEL_SYSTEM = """You label test data for a tiny model inside a gadget. You get the gadget's current state
(numeric fields with ranges), its actions, and numbered requests people said to it.

For each request, give the right answer: exactly one action with the absolute value it sets
(inside the action's lo..hi), or "none" when no listed action fits: a question, something
unrelated, a device or action that isn't listed, a change past lo or hi, a number outside the
range, or a request NOT to do something or to leave things as they are ("don't ...", "leave
... alone"). Relative amounts come from the current value of the field the action "sets": "a
bit" is about 5 to 10 percent of the range, plain about 15 to 25 percent, "a lot" about 30 to
60 percent. A described problem implies the fix away from it ("it's too cold" = warmer); a
stated wish asks for it ("i want it cold" = cooler). Switching on or fully open = hi, off or
closed = lo. Actions without a value take value null.

Reply with a JSON object whose "labels" list has one entry per request: "i" (its number),
"action" (a name or "none") and "value"."""

INVERT_SCHEMA = _schema("invert", _obj(items={"type": "array", "items": _obj(
    i={"type": "integer"}, variants={"type": "array", "items": _obj(type=_STR, say=_STR, action=_STR, value=_NNUM)})}))
LABEL_SCHEMA = _schema("labels", _obj(labels={"type": "array", "items": _obj(
    i={"type": "integer"}, action=_STR, value=_NNUM)}))


def answer(u):
    """The right answer as the prompt shows it: "none", "start_cycle()", "set_speed(40)"."""
    if u["action"] == "none":
        return "none"
    v = u.get("value")
    return f"{u['action']}({'' if v is None else format(v, 'g')})"


def invert_world(c, w, requests):
    """-> list of inverted utterances (with agreement) for one world."""
    state, actions = w["state"], w["actions"]
    listing = "\n".join(f"{i + 1}. {u['say']}   -> {answer(u)}" for i, u in enumerate(requests))
    got = ask(c, INVERT_USER.format(state=fmt_state(state), actions=fmt_actions(actions), requests=listing),
              INVERT_SYSTEM, max_tokens=6000, temperature=0.8, response_format=INVERT_SCHEMA)
    variants = []
    for item in got.get("items") or []:
        i = item.get("i")
        if not isinstance(i, int) or not 1 <= i <= len(requests):
            continue
        for v in (item.get("variants") or [])[:3]:
            say = clean(v.get("say"))
            if v.get("type") in TYPES and say and check_label(v, actions)[0] and say != requests[i - 1]["say"]:
                variants.append({"parent": requests[i - 1], "type": v["type"], "say": say,
                                 "action": v["action"], "value": v.get("value")})
    if not variants:
        return []
    listing = "\n".join(f"{i + 1}. {v['say']}" for i, v in enumerate(variants))
    lab = ask(c, INVERT_USER.format(state=fmt_state(state), actions=fmt_actions(actions), requests=listing),
              LABEL_SYSTEM, max_tokens=4000, temperature=0.3, response_format=LABEL_SCHEMA)
    second = {x.get("i"): x for x in lab.get("labels") or [] if isinstance(x, dict)}
    out = []
    for i, v in enumerate(variants):
        s2 = second.get(i + 1)
        ok2 = s2 is not None and check_label(s2, actions)[0]
        ag = ok2 and agree(v, s2, actions, state)
        if v["type"] == "negate":               # the rule, not a judgment call
            ag = ag and v["action"] == "none"
        out.append({"kind": "inv_" + v["type"], "say": v["say"], "source": "invert", "action": v["action"],
                    "value": v["value"], "agree": ag, "flag": None, "parent_kind": v["parent"]["kind"],
                    "parent": v["parent"]["say"],
                    **({"replay": {"action": s2["action"], "value": s2.get("value")}} if ok2 else {})})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/llm/worlds.jsonl")
    ap.add_argument("--worlds", type=int, default=0, help="only the first N worlds (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-requests", type=int, default=3000)
    ap.add_argument("--max-spend", type=float, default=10.0, help="dollars per day")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from ..gen.worlds import load, validate
    worlds = load(a.src)
    if a.worlds:
        worlds = worlds[:a.worlds]
    keep, _ = validate(worlds)
    by_world = {}
    for w, u, _ in keep:
        if u["source"] == "request" and u["kind"] in KINDS:
            by_world.setdefault(w["setting"], (w, []))[1].append(u)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["setting"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [(w, reqs) for name, (w, reqs) in by_world.items() if name not in done]
    print(f"{len(done)} worlds already in {out}; {len(todo)} to invert", flush=True)
    c = Client.parasail(max_requests=a.max_requests, max_spend_per_day=a.max_spend)
    c.reasoning = "off"
    lock = threading.Lock()
    stats = {"worlds": 0, "variants": 0, "agree": 0}
    with ThreadPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(invert_world, c, w, reqs): w for w, reqs in todo}
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
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps({"setting": w["setting"], "area": w["area"], "utterances": utts}) + "\n")
                stats["worlds"] += 1
                stats["variants"] += len(utts)
                stats["agree"] += sum(u["agree"] for u in utts)
                if stats["worlds"] % 25 == 0 or stats["worlds"] <= 5:
                    print(f"  {stats['worlds']} worlds, {stats['variants']} variants, "
                          f"{stats['agree'] / max(1, stats['variants']):.0%} agree", flush=True)
    print(f"{stats} -> {out}\n{c.summary()}")


if __name__ == "__main__":
    main()
