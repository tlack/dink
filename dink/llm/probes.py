"""Write direction probes with the LLM: each a small, made-up gadget of its own plus one request
and what a right answer looks like (which action, which way, how much).

The point is "most correct in the right direction": the label is the action and the way it
should move. The amount is coarse (any / small / big / end / a number said), and scored by
rank among the options offered, not by exact value (dink/evals/direction.py).

  uv run python -m dink.llm.probes                     # -> data/probes/raw.jsonl
  uv run python -m dink.llm.probes --keep              # checked cases -> dink/evals/direction_cases.jsonl

Cases are reviewed by hand before they're kept: the LLM gets direction wrong sometimes too.
"""
import argparse
import json
import random
import re
from pathlib import Path

from .openrouter import Client

RAW = Path("data/probes/raw.jsonl")
KEEP = Path("dink/evals/direction_cases.jsonl")

_NUM, _NNUM, _STR, _NSTR = {"type": "number"}, {"type": ["number", "null"]}, {"type": "string"}, {"type": ["string", "null"]}


def _obj(**props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


SCHEMA = {"type": "json_schema", "json_schema": {"name": "probes", "strict": True, "schema": _obj(cases={
    "type": "array", "items": _obj(
        setting=_STR,
        state={"type": "array", "items": _obj(name=_STR, value=_NUM, lo=_NUM, hi=_NUM)},
        actions={"type": "array", "items": _obj(name=_STR, lo=_NNUM, hi=_NNUM, sets=_NSTR)},
        say=_STR, action=_STR, dir={"type": "string", "enum": ["up", "down", "none"]},
        amount={"type": "string", "enum": ["any", "small", "big", "end", "number"]},
        number=_NNUM, why=_STR)})}}

SYSTEM = """You write test cases for a tiny model that controls one gadget from one spoken request.
The model sees the gadget's state (numeric fields with a current value and a range lo..hi) and
its actions, and picks one action with one value, or nothing.

Each case invents its own small gadget:
- 2 to 4 state fields: snake_case names, numeric value, lo, hi (value inside lo..hi). Toggles
  are 0..1. Some fields are read-only sensors (no action sets them).
- 3 to 5 actions: snake_case names. A valued action has lo/hi equal to the range of the state
  field it sets, and "sets" names that field. A plain action (a scene, a cycle, a one-shot) has
  lo, hi and sets all null. Include at least one distractor action that is plausible but wrong.
- Name things the way real firmware or Home Assistant would; vary the styles (set_x, x_level,
  short names like fan_spd). Names at most 32 characters.

Then one request ("say", at most 60 characters, lowercase, how a person would talk) and the
right answer:
- action: the right action's name, or "none" when the right answer is to do nothing (a
  question, small talk, a request no action can do).
- dir: "up" or "down" for the right action's value relative to its field's current value;
  "none" for plain actions and for "none".
- amount: "any" (any move that way is fine), "small" (a little), "big" (a lot, but not
  necessarily all the way), "end" (all the way: max, min, off, fully), "number" (the request
  says the value; put it in "number"). For plain actions and "none" use "any".
- number: the value for amount "number", else null.
- why: one short sentence on why this is the answer.

Be careful with direction: think about what the person wants to happen, not the words they
use. "i'm hot" means cooler. "it's too bright" means dimmer. "make it dark" means off.
Check that the current value leaves room to move that way, unless the case is about a limit.
Reply with JSON: {"cases": [...]}."""

THEMES = {
    "complaint": "The person states a problem, not a command; the fix moves a value opposite to the complaint (\"it's freezing in here\", \"the music is way too loud\", \"this water is lukewarm\"). Mix amounts any/small/big.",
    "desire": "The person says the result they want (\"i want it cold in here\", \"make it really dark\", \"i want the water hotter\"). Some mean a full move (end), some just a direction.",
    "a_little": "Requests for a small change: \"a little\", \"a touch\", \"slightly\", \"a smidge\", \"just a bit\", \"one notch\". Amount small. Vary current values, including near the middle and near an end with room left.",
    "a_lot": "Requests for a big change that is not all the way: \"a lot\", \"way more\", \"crank it\", \"much louder\", \"really turn it up\". Amount big.",
    "end": "Requests for all the way: \"max\", \"full blast\", \"all the way down\", \"kill it\", \"off\", \"lowest setting\", \"wide open\". Amount end.",
    "more_less": "Bare comparatives using the thing's everyday noun, not the action's name: \"more jets\", \"less bass\", \"faster\", \"slower\", \"more air\", \"less steam\", \"higher\", \"lower\". Amount any.",
    "question": "Questions and remarks that only look like commands and need no action: \"is it dark outside yet\", \"how hot is the pool\", \"did the fan turn on\", \"the lights are nice\". Action none. Mention a device or quantity the gadget has.",
    "limit": "The value is already at or near the end of its range and the person asks to go further, or asks to go back the other way. If it is at the end and they ask to go further, the answer is still that action and direction (the scorer accepts doing nothing there). Mix small and any amounts.",
    "small_range": "Actions with small whole-number ranges (0..3 fan levels, 1..5 speed, 0..2 modes, 1..4 heat settings). Mix a little / a lot / more / less / max. Current values anywhere, including the ends.",
    "lookalike": "Gadgets with two or more similar valued actions (two zones, front and rear, upper and lower, left and right). The request names one of them indirectly. Mix directions.",
    "indirect": "Indirect or negated wording: \"not so bright\", \"tone it down\", \"it's blinding\", \"i can barely hear it\", \"don't let it get any colder\", \"ease off the pressure\". Amount any or small.",
    "number": "Requests that say a value, in different ways: \"set it to 4\", \"72 degrees\", \"half way\", \"level two\", \"40 percent\". Amount number, with the value as the action would take it (a percent on a 0..1 field is 0.4).",
}

SETTINGS = """home thermostats, ceiling fans, dimmable lights, blinds, aquariums, greenhouses, saunas,
hot tubs, pool pumps, ovens, sous vide cookers, coffee makers, espresso machines, kettles, grills,
smokers, dehumidifiers, air purifiers, space heaters, car climate, car seats, e-bikes, treadmills,
massage chairs, hospital beds, incubators, 3D printers, laser cutters, CNC spindles, kilns,
soldering stations, drones, boats, RVs, tractors, irrigation, chicken coops, beehives, terrariums,
wine coolers, fridges, freezers, ice makers, dishwashers, washing machines, dryers, humidifiers,
fireplaces, water heaters, garage doors, gates, speakers, amplifiers, mixers, stage lights,
projectors, monitors, telescopes, microscopes, cameras, gimbals, robot vacuums, lawn mowers,
snow blowers, pressure washers, compressors, generators, solar inverters, EV chargers,
bathtubs, showers, bidets, heated floors, electric blankets, baby monitors, white noise machines"""


def generate(c, theme, k, n=12, seed=0):
    rng = random.Random(f"{theme}{k}{seed}")
    pick = rng.sample([s.strip() for s in SETTINGS.replace("\n", " ").split(",") if s.strip()], 6)
    user = (f"Theme: {theme}. {THEMES[theme]}\n\nWrite {n} cases, each with a different gadget. "
            f"Use these settings (or close cousins): {', '.join(pick)}.")
    out = c.json_obj(user, system=SYSTEM, max_tokens=12000, temperature=0.8, response_format=SCHEMA,
                     reasoning_effort="low")
    return [dict(x, theme=theme) for x in out.get("cases", [])]


NAME = re.compile(r"^[a-z][a-z0-9_.]{1,31}$")


def check(x):
    """Problems with a case (empty if usable)."""
    p = []
    st = {f["name"]: f for f in x["state"]}
    acts = {a["name"]: a for a in x["actions"]}
    for n in list(st) + list(acts):
        if not NAME.match(n):
            p.append(f"name {n!r}")
    for f in x["state"]:
        if not f["lo"] < f["hi"] or not f["lo"] <= f["value"] <= f["hi"]:
            p.append(f"field {f['name']} range")
    for a in x["actions"]:
        if a["lo"] is None:
            if a["sets"] is not None or a["hi"] is not None:
                p.append(f"plain action {a['name']} half-valued")
            continue
        f = st.get(a["sets"])
        if f is None:
            p.append(f"action {a['name']} sets missing field")
        elif (f["lo"], f["hi"]) != (a["lo"], a["hi"]):
            p.append(f"action {a['name']} range differs from its field")
    if len(x["say"].encode()) > 64:
        p.append("say too long")
    if len(x["actions"]) < 2 or len(x["actions"]) > 7 or len(x["state"]) > 6:
        p.append("size")
    if x["action"] == "none":
        if x["dir"] != "none":
            p.append("none with a direction")
        return p
    a = acts.get(x["action"])
    if a is None:
        return p + ["unknown action"]
    if a["lo"] is None:
        if x["dir"] != "none":
            p.append("plain action with a direction")
        return p
    if x["dir"] == "none" and x["amount"] != "number":
        p.append("valued action without a direction")
    if x["amount"] == "number":
        if x["number"] is None or not a["lo"] <= x["number"] <= a["hi"]:
            p.append("number missing or out of range")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-theme", type=int, default=2, help="calls per theme (12 cases each)")
    ap.add_argument("--keep", action="store_true", help="write checked cases from data/probes/raw.jsonl")
    a = ap.parse_args()
    if a.keep:
        rows = [json.loads(l) for l in RAW.read_text().splitlines()]
        good = [r for r in rows if not check(r)]
        KEEP.write_text("".join(json.dumps(r) + "\n" for r in good))
        print(f"kept {len(good)}/{len(rows)} -> {KEEP}")
        return
    c = Client.parasail(max_requests=60)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for theme in THEMES:
        for k in range(a.per_theme):
            got = generate(c, theme, k)
            bad = [(x["say"], check(x)) for x in got if check(x)]
            print(f"{theme:12s} {k}: {len(got)} cases, {len(bad)} rejected {bad[:2]}", flush=True)
            rows += got
    RAW.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} cases -> {RAW}")


if __name__ == "__main__":
    main()
