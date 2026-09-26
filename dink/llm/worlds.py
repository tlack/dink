"""LLM-invented worlds: new settings, devices and labeled requests beyond the smart home.

Three LLM stages, then checks in code. Every prompt puts its fixed instructions first (the
system message, cached by the provider) and only the per-call part last.

  1. settings  For a rotating area ("boats", "aquariums", ...), invent a few concrete settings,
               excluding the ones already made there (variety comes from the areas, the
               exclusions, and the temperature).
  2. world     For one setting: a state (3-8 fields, each with a value and declared range),
               actions (2-6, each with a numeric range or none, and the state field it changes),
               and 12 requests of required kinds, each labeled with an action and value, or none.
  3. replay    A second, independent call labels the same requests without seeing the first
               labels, and adds 4 rewordings of each (same meaning) and 1 "flip" (similar
               wording, different right answer: opposite direction, complaint vs desire, a
               question instead of a command ...).

Checks in code: schema, names, ranges, prompt length, agreement between the two labelings,
and numbers in a request against its label. Records keep everything, flags included, so
the converter to Dink scenarios (later) can choose what to trust.

  uv run python -u -m dink.llm.worlds --settings 20 --out data/llm/worlds_pilot.jsonl   # pilot
  uv run python -u -m dink.llm.worlds --settings 600 --workers 4                       # collect

Resumable: settings already in --out are skipped, and every reply is cached on disk.
"""
import argparse
import json
import random
import re
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..data.numbers import extract
from .openrouter import Budget, Client

OUT = Path("data/llm/worlds.jsonl")
AREAS = [
    "boats and marine", "RVs and camper vans", "cars and trucks", "bicycles and e-scooters", "aircraft cabins",
    "trains and buses", "greenhouses", "gardens and irrigation", "farms and livestock barns", "beekeeping",
    "aquariums and ponds", "pet care", "baby and nursery", "elder care at home", "home medical devices",
    "kitchen appliances", "coffee and brewing", "restaurant kitchens", "laundry", "bathrooms and spas",
    "saunas and hot tubs", "swimming pools", "workshops and power tools", "3D printers and CNC machines",
    "home labs and science", "server rooms", "music studios and audio gear", "stage and event lighting",
    "classrooms", "offices and meeting rooms", "retail shops", "warehouses", "cold storage",
    "hotels and rentals", "hospitals and clinics", "gyms and fitness equipment", "sports venues",
    "camping and outdoor gear", "off-grid solar and batteries", "EV charging", "weather stations",
    "telescopes and observatories", "ham radio and electronics benches", "toys and hobby robots",
    "drones", "wearables", "wine cellars", "food trucks", "ice rinks", "vertical farms",
]

SETTINGS_SYSTEM = """You help build training data for a tiny model that runs inside small gadgets (a voice box,
a remote, a wall panel, a machine's control board). It turns one spoken request into one action
on the gadget, or into "no action".

Invent settings for it: a concrete place, vehicle, appliance or system where a small controller
can read a few values (sensors, current settings) and adjust or switch a few things. Prefer
ordinary, specific settings that real people use; make them different from each other. Avoid
smart-home rooms with lights, thermostats, fans and blinds: we already have those.
Each setting needs at least 3 readable values and at least 2 adjustable things.

Reply with a JSON object whose "settings" list has, for each setting, a "name" (3 to 6
words) and "about" (one sentence: who uses it and what it controls)."""

SETTINGS_USER = """Area: {area}
Write {n} settings in this area.
Already used in this area (write different ones): {exclude}"""

WORLD_SYSTEM = """You design test worlds for a tiny model that runs inside a gadget. The gadget knows its
current state (named numeric values, each with a declared range) and a list of actions. When a
person says something, the right answer is exactly one action with a value, or "none".

Given a setting, write a JSON object with:

"state": 3 to 8 fields, each with a name, its current value, its range lo..hi and a unit.
  Numbers only; on/off or open/closed is 0..1 (unit "on/off"). Include 1 or 2 fields that most
  requests don't need (a battery level, the time, an outside reading).
"actions": 2 to 6 actions, each with a name, a range lo..hi and unit for the value it sets,
  and "sets": the name of the state field it changes (null if none). An action with no value
  (start a program, sound an alarm) has lo, hi, unit and sets all null.
  Most state fields that can be changed should have an action; some fields are read-only.
"requests": exactly 12, one of each kind, in this order:
  direct (an absolute setting or switching on/off, no number), number (says a number, with or
  without a unit), relative_small ("a bit more ..."), relative_big ("way more ..."),
  complaint (describes a problem, implies the fix: "it's too hot" = cooler), desire (says the
  wanted condition: "i want it hot" = warmer), slang (casual or unusual words for the thing),
  question (asks about status; needs no action), unrelated (something this gadget can't be
  about), missing (a reasonable request for a device or action that is NOT in the list),
  limit (asks to push an adjustable field further in one direction, as if it were already at
  its lo or hi: "even brighter", "any colder?"; give that field as "limit_field" and
  "limit_up": true when pushing past hi, false past lo), and
  no_value (uses an action that has no value; if there is none, another direct request).
  Each request has its kind, what the person says ("say"), the right "action" (an action
  name, or "none") and the absolute "value" that action sets (not the change), or null.
  "limit_field" and "limit_up" are null except for the limit request.

Rules:
- names: lowercase snake_case, 2 to 4 descriptive words, at most 32 characters; action names
  start with a verb (set_, turn_on_, open_, start_, ...); state names describe what is measured.
- values: inside lo..hi. Relative amounts from the current value of the "sets" field: "a bit"
  is about 5 to 10 percent of the range, plain about 15 to 25 percent, "a lot" about 30 to 60
  percent, never past lo or hi. Switching on / fully open = hi, off / closed = lo.
- "none" when no listed action fits: questions, unrelated or missing requests, limit
  requests (the value can't go further), and numbers outside the action's range.
- "say": what a person would really say to it, lowercase, at most 60 characters, no quotes.
  Numbers may be digits or words. Vary the sentence shapes."""

WORLD_USER = """Setting: {name}
About: {about}"""

REPLAY_SYSTEM = """You check and extend test data for a tiny model inside a gadget. You get the gadget's
current state (numeric fields with ranges), its actions, and numbered requests people said to it.

For each request, decide the right answer: exactly one action with the absolute value it sets
(inside the action's lo..hi), or "none" when no listed action fits (a question, something
unrelated, a device or action that isn't listed, a change past lo or hi, or a number outside
the action's range). Relative amounts
come from the current value of the field the action "sets": "a bit" is about 5 to 10 percent
of the range, plain about 15 to 25 percent, "a lot" about 30 to 60 percent. Switching on or
fully open = hi, off or closed = lo. Actions without a value take value null.

Also write, for each request:
- "same": 4 different ways a person might say it with exactly the same right answer (same
  action, same amount). Vary the words and shapes: terse, polite, slang, indirect.
- "flip": 1 request with similar words but a different right answer: the opposite direction,
  a complaint turned into a desire ("it's too dark" vs "make it dark"), a command turned into a
  question, or another action. Give its own "action" and "value".
All requests lowercase, at most 60 characters, no quotes.

Reply with a JSON object whose "labels" list has one entry per request: "i" (the request's
number), "action", "value", "same" (the 4 rewordings) and "flip" (with its own "say",
"action" and "value")."""

REPLAY_USER = """State: {state}
Actions: {actions}
Requests:
{requests}"""

# Strict JSON schemas (structured outputs): the provider enforces the shape, so the prompts
# need no example replies to copy from.
_NUM, _NNUM, _STR, _NSTR = {"type": "number"}, {"type": ["number", "null"]}, {"type": "string"}, {"type": ["string", "null"]}


def _obj(**props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _schema(name, schema):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


SETTINGS_SCHEMA = _schema("settings", _obj(settings={"type": "array", "items": _obj(name=_STR, about=_STR)}))
WORLD_SCHEMA = _schema("world", _obj(
    state={"type": "array", "items": _obj(name=_STR, value=_NUM, lo=_NUM, hi=_NUM, unit=_STR)},
    actions={"type": "array", "items": _obj(name=_STR, lo=_NNUM, hi=_NNUM, unit=_NSTR, sets=_NSTR)},
    requests={"type": "array", "items": _obj(kind=_STR, say=_STR, action=_STR, value=_NNUM,
                                             limit_field=_NSTR, limit_up={"type": ["boolean", "null"]})}))
REPLAY_SCHEMA = _schema("replay", _obj(labels={"type": "array", "items": _obj(
    i={"type": "integer"}, action=_STR, value=_NNUM, same={"type": "array", "items": _STR},
    flip=_obj(say=_STR, action=_STR, value=_NNUM))}))

NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
KINDS = ["direct", "number", "relative_small", "relative_big", "complaint", "desire", "slang", "question",
         "unrelated", "missing", "limit", "no_value"]


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_world(w):
    """-> (state, actions, requests, problems). Drops what can't be used; problems explain."""
    probs = []
    state, names = [], set()
    for f in (w.get("state") or [])[:8]:
        if not (isinstance(f, dict) and NAME.match(str(f.get("name", ""))) and num(f.get("value"))
                and num(f.get("lo")) and num(f.get("hi")) and f["lo"] < f["hi"] and f["lo"] <= f["value"] <= f["hi"]
                and f["name"] not in names):
            probs.append(f"bad state field {json.dumps(f)[:80]}")
            continue
        names.add(f["name"])
        state.append({k: f.get(k) for k in ("name", "value", "lo", "hi", "unit")})
    actions, anames = [], set()
    for a in (w.get("actions") or [])[:6]:
        if not (isinstance(a, dict) and NAME.match(str(a.get("name", ""))) and a["name"] not in anames):
            probs.append(f"bad action {json.dumps(a)[:80]}")
            continue
        has = num(a.get("lo")) and num(a.get("hi"))
        if has and not a["lo"] < a["hi"]:
            probs.append(f"bad range {a['name']}")
            continue
        sets = a.get("sets") if a.get("sets") in names else None
        anames.add(a["name"])
        actions.append({"name": a["name"], "lo": a["lo"] if has else None, "hi": a["hi"] if has else None,
                        "unit": a.get("unit") if has else None, "sets": sets})
    if len(state) < 3 or len(actions) < 2:
        probs.append("too few state fields or actions")
    requests = []
    for r in w.get("requests") or []:
        ok, why = check_label(r, actions)
        say = clean(r.get("say")) if isinstance(r, dict) else ""
        if not say or not ok:
            probs.append(f"bad request {json.dumps(r)[:80]} ({why or 'say'})")
            continue
        req = {"kind": r.get("kind"), "say": say, "action": r["action"], "value": r.get("value")}
        if r.get("kind") == "limit":
            if r.get("limit_field") not in names or not isinstance(r.get("limit_up"), bool):
                probs.append(f"limit request without a usable field: {say}")
                continue
            req["limit"] = {"field": r["limit_field"], "up": r["limit_up"]}      # converter sets it to lo/hi
        requests.append(req)
    return state, actions, requests, probs


def clean(s):
    if not isinstance(s, str):
        return ""
    s = " ".join(s.lower().replace('"', "").split())
    return s if 0 < len(s.encode()) <= 64 else ""


def check_label(r, actions):
    if not isinstance(r, dict):
        return False, "not an object"
    act = r.get("action")
    if act == "none":
        return r.get("value") is None or True, ""
    a = next((x for x in actions if x["name"] == act), None)
    if a is None:
        return False, f"unknown action {act}"
    if a["lo"] is None:
        return True, ""
    v = r.get("value")
    if not num(v) or not a["lo"] <= v <= a["hi"]:
        return False, f"value {v} outside {a['lo']}..{a['hi']}"
    return True, ""


def agree(first, second, actions, state):
    """Do two labelings of one request agree? Same action; for valued actions the values are
    within 10% of the range of each other, or move the same way from the current value."""
    if first["action"] != second["action"]:
        return False
    a = next((x for x in actions if x["name"] == first["action"]), None)
    if a is None or a["lo"] is None:
        return True
    v1, v2 = first["value"], second["value"]
    if abs(v1 - v2) <= 0.1 * (a["hi"] - a["lo"]):
        return True
    cur = next((f["value"] for f in state if f["name"] == a["sets"]), None)
    return cur is not None and (v1 - cur) * (v2 - cur) > 0


def number_flag(say, action, value, actions):
    """A request that says a number in its action's range, labeled with a different value."""
    a = next((x for x in actions if x["name"] == action), None)
    if a is None or a["lo"] is None or value is None:
        return None
    for v, kind in extract(say):
        if kind == "raw" and a["lo"] <= v <= a["hi"] and abs(v - value) > 1e-6:
            return f"says {v:g}, labeled {value:g}"
    return None


def fmt_state(state):
    return "; ".join(f"{f['name']}={f['value']:g} ({f['lo']:g}..{f['hi']:g}{' ' + f['unit'] if f.get('unit') else ''})"
                     for f in state)


def fmt_actions(actions):
    out = []
    for a in actions:
        if a["lo"] is None:
            out.append(f"{a['name']}() [no value]")
        else:
            out.append(f"{a['name']}({a['lo']:g}..{a['hi']:g}{' ' + a['unit'] if a.get('unit') else ''})"
                       + (f" sets {a['sets']}" if a["sets"] else ""))
    return "; ".join(out)


def ask(c, user, system, **kw):
    return c.json_obj(user, system=system, reasoning_effort=getattr(c, "reasoning", "low"), **kw)


def build(c, setting, area, model):
    """Stages 2 and 3 for one setting -> a record (or None)."""
    w = ask(c, WORLD_USER.format(name=setting["name"], about=setting["about"]), WORLD_SYSTEM,
            max_tokens=8000, temperature=0.9, response_format=WORLD_SCHEMA)
    state, actions, requests, probs = check_world(w)
    if len(state) < 3 or len(actions) < 2 or len(requests) < 6:
        return {"setting": setting["name"], "area": area, "failed": probs[:6]}
    def line(i, r):
        lim = r.get("limit")
        note = f"   ({lim['field']} is already at its {'maximum' if lim['up'] else 'minimum'})" if lim else ""
        return f"{i + 1}. {r['say']}{note}"
    listing = "\n".join(line(i, r) for i, r in enumerate(requests))
    rep = ask(c, REPLAY_USER.format(state=fmt_state(state), actions=fmt_actions(actions), requests=listing),
              REPLAY_SYSTEM, max_tokens=12000, temperature=0.7, response_format=REPLAY_SCHEMA)
    labels = {x.get("i"): x for x in rep.get("labels") or [] if isinstance(x, dict)}
    utts = []
    for i, r in enumerate(requests):
        second = labels.get(i + 1)
        ok2 = second is not None and check_label(second, actions)[0]
        ag = ok2 and agree(r, second, actions, state)
        base = {"kind": r["kind"], "action": r["action"], "value": r["value"], "agree": ag}
        if "limit" in r:
            base["limit"] = r["limit"]
        if ok2:
            base["replay"] = {"action": second["action"], "value": second.get("value")}
        utts.append(dict(base, say=r["say"], source="request", flag=number_flag(r["say"], r["action"], r["value"], actions)))
        if not ok2:
            continue
        seen = {r["say"]}
        for s in (second.get("same") or [])[:4]:            # rewordings take the replay's label
            s = clean(s)
            if s and s not in seen:
                seen.add(s)
                utts.append({"kind": r["kind"], "say": s, "source": "same", "action": second["action"],
                             "value": second.get("value"), "agree": ag, **({"limit": r["limit"]} if "limit" in r else {}),
                             "flag": number_flag(s, second["action"], second.get("value"), actions)})
        f = second.get("flip")
        if isinstance(f, dict) and clean(f.get("say")) and check_label(f, actions)[0]:
            utts.append({"kind": r["kind"], "say": clean(f["say"]), "source": "flip", "action": f["action"],
                         "value": f.get("value"), "agree": None,
                         "flag": number_flag(clean(f["say"]), f["action"], f.get("value"), actions)})
    return {"setting": setting["name"], "about": setting["about"], "area": area, "state": state,
            "actions": actions, "utterances": utts, "problems": probs, "model": model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=int, default=20, help="stop after this many settings in --out")
    ap.add_argument("--per-area", type=int, default=5, help="settings asked for per area per round")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-requests", type=int, default=3000)
    ap.add_argument("--max-spend", type=float, default=5.0, help="dollars per day")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasoning", default="off", help='"off", or an effort: "low", "medium", ...')
    a = ap.parse_args()
    c = Client.parasail(max_requests=a.max_requests, max_spend_per_day=a.max_spend)
    c.reasoning = a.reasoning
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["setting"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    print(f"{len(done)} settings already in {out}", flush=True)
    rng = random.Random(a.seed)
    areas = AREAS[:]
    rng.shuffle(areas)
    used = {ar: [] for ar in areas}
    write = threading.Lock()
    n_ok = [0]
    stop = [False]

    def save(rec):
        with write:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if "failed" not in rec:
                n_ok[0] += 1
                u = rec["utterances"]
                ag = sum(1 for x in u if x["source"] == "request" and x["agree"])
                nreq = sum(1 for x in u if x["source"] == "request")
                print(f"  {rec['area'][:24]:24s} {rec['setting'][:40]:40s} {len(u):3d} utterances, "
                      f"{ag}/{nreq} requests agree", flush=True)
            else:
                print(f"  FAILED {rec['setting'][:40]}: {rec['failed'][:2]}", flush=True)

    rounds = 0
    with ThreadPoolExecutor(a.workers) as pool:
        while len(done) < a.settings and not stop[0] and rounds < 20:
            rounds += 1
            jobs = []
            for area in areas:
                if len(done) + len(jobs) >= a.settings:
                    break
                try:
                    got = ask(c, SETTINGS_USER.format(area=area, n=a.per_area,
                                                      exclude="; ".join(used[area]) or "(none yet)"),
                              SETTINGS_SYSTEM, max_tokens=3000, temperature=1.0,
                              response_format=SETTINGS_SCHEMA)
                except Budget as e:
                    print(f"stopping: {e}")
                    stop[0] = True
                    break
                for s in got.get("settings") or []:
                    if not (isinstance(s, dict) and isinstance(s.get("name"), str) and isinstance(s.get("about"), str)):
                        continue
                    used[area].append(s["name"])
                    if s["name"] in done or len(done) + len(jobs) >= a.settings:
                        continue
                    done.add(s["name"])
                    jobs.append(pool.submit(build, c, s, area, c.model))
            for j in as_completed(jobs):
                try:
                    rec = j.result()
                except Budget as e:
                    print(f"stopping: {e}")
                    stop[0] = True
                    continue
                except RuntimeError as e:
                    print(f"  error: {e}")
                    continue
                if rec:
                    save(rec)
    print(f"{n_ok[0]} new settings -> {out}\n{c.summary()}")


if __name__ == "__main__":
    main()
