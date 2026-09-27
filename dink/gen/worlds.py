"""LLM-invented worlds (dink/llm/worlds.py) -> validated Dink scenarios.

Validation (python -m dink.gen.worlds --validate) re-checks every utterance in code, on top
of the collector's checks, and reports what it would drop and why:
  kind rules      questions / unrelated / missing / limit requests must be "none"; no_value
                  requests must use an action without a value; relative, complaint and
                  desire requests must change a field (not re-send its current value)
  amounts         "a bit" moves 0-20% of the range, "a lot" at least 12% (or to lo / hi)
  agreement       the replay's independent label agrees (requests and their rewordings)
  numbers         if a request says numbers inside its action's range, the label is one of
                  them (numbers that are part of a name, "zone 1", "co2", don't count
                  when another said number matches)
  conflicts       the same words with two different labels in one world
  on / off        an on / start / open action for "off / stop / close" words (or the
                  reverse), unless it's a valued switch sent to the matching end
  flips           no second opinion: they must also move the way plain direction words say
                  ("slower" down), and a limit request's flip uses the limit state
  schema          names, ranges, lengths (Scenario.check)

Conversion: each kept utterance becomes a scenario: the world's state (with the limit field
at lo / hi for limit requests and their flips), and tools made from the actions: the right
action with near-miss values (the current value, the opposite move, a smaller and a bigger
move), plus the other actions at plausible values, up to 8 slots. Names are restyled:
snake_case, Title Case or camelCase (camel is held out by configs/home_split_v4.json).

  uv run python -m dink.gen.worlds --validate
  uv run python -m dink.gen.worlds --worlds 250 --variants 2 --out data/worlds/pool.jsonl
"""
import argparse
import collections
import json
import random
import zlib
from pathlib import Path

from ..data.numbers import extract
from ..data.scenario import MAX_SLOTS, Field, Scenario
from . import names as N
from .bank import annotate
from .ordinal import offer, right as ranked

# Relative requests whose strength picks among the offered options by rank (dink/gen/ordinal.py)
ORDINAL = {"relative_small": "small", "relative_big": "big", "int_small": "small", "int_plain": "plain",
           "int_big": "big", "int_max": "max"}

SRC = Path("data/llm/worlds.jsonl")
UP = {"warmer", "hotter", "higher", "faster", "brighter", "stronger", "louder", "more", "increase", "raise",
      "up", "boost", "crank", "wider"}
DOWN = {"cooler", "colder", "lower", "slower", "dimmer", "weaker", "quieter", "decrease", "reduce", "down",
        "dim", "narrower"}
NEGATORS = {"too", "less", "not", "no", "don't", "dont", "isn't", "stop", "instead", "than", "enough", "so"}
ON_VERBS = ("turn_on", "start", "open", "enable", "activate", "engage", "unlock")
OFF_VERBS = ("turn_off", "stop", "close", "disable", "deactivate", "shut", "lock")
OFF_WORDS = {"off", "stop", "close", "shut", "disable", "deactivate", "kill", "cancel", "end"}
ON_WORDS = {"on", "start", "open", "enable", "activate", "begin"}
NONE_KINDS = {"question", "unrelated", "missing", "limit"}
MOVE_KINDS = {"relative_small", "relative_big", "complaint", "desire"}
STYLES = ["long", "title", "camel"]
STYLE_WEIGHTS = [0.45, 0.45, 0.1]     # camel is a held-out style: keep its share (lost to test_styles) small


def load(path=SRC):
    return [r for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())
            if "failed" not in r]


def state_for(world, u):
    """The world's state, with a limit request's field set to its lo or hi."""
    st = {f["name"]: dict(f) for f in world["state"]}
    lim = u.get("limit")
    if lim and lim["field"] in st:
        f = st[lim["field"]]
        f["value"] = f["hi"] if lim["up"] else f["lo"]
    return st


def said_direction(say):
    """+1 / -1 when the words name one direction plainly (no negation or complaint framing)."""
    words = set(say.replace("?", " ").replace(",", " ").replace(".", " ").split())
    if words & NEGATORS:
        return 0
    up, down = bool(words & UP), bool(words & DOWN)
    return 1 if up and not down else -1 if down and not up else 0


def polarity_clash(say, a, value):
    """An on/start/open action used for "off" words (or the reverse), unless it's a valued
    switch sent to the matching end ("turn_on_heater=0" for "turn off the heater" is fine)."""
    say = say.replace("kick off", "start").replace("kicks off", "starts").replace("set off", "trigger")
    words = set(say.replace("?", " ").replace(",", " ").replace(".", " ").split())
    name = a["name"]
    if name.startswith(ON_VERBS) and words & OFF_WORDS and not words & ON_WORDS - {"on"}:
        return a["lo"] is None or value == a["hi"]
    if name.startswith(OFF_VERBS) and words & ON_WORDS and not words & OFF_WORDS:
        return a["lo"] is None or value == a["hi"]
    return False


def problems(world, u, parent=None):
    """Reasons to drop one utterance (empty list = keep)."""
    acts = {a["name"]: a for a in world["actions"]}
    out = []
    kind, src = u["kind"], u["source"]
    a = acts.get(u["action"])
    if u["action"] != "none" and a is None:
        return ["unknown action"]
    if u["source"] in ("request", "same", "invert", "intensity") and not u.get("agree"):
        out.append("replay disagrees")
    if a is not None and a["lo"] is not None and u["kind"] in ("number", "direct"):
        said = [v for v, k in extract(u["say"]) if k == "raw" and a["lo"] <= v <= a["hi"]]
        said += [a["lo"] + v * (a["hi"] - a["lo"]) for v, k in extract(u["say"]) if k == "level"]
        if said and all(abs(v - u["value"]) > 1e-6 for v in said):
            out.append("number mismatch")
    if a is not None and polarity_clash(u["say"], a, u.get("value")):
        out.append("on/off action against its words")
    if src == "flip" and a is not None and a["lo"] is not None and a["sets"]:
        # Flips have no second labeling. Only for them: plain direction words must agree with the
        # move. (Not applied elsewhere: "make it cooler" -> fan up is right, and that's the
        # world knowledge we want kept.)
        cur = state_for(world, u).get(a["sets"], {}).get("value")
        d = said_direction(u["say"])
        if cur is not None and d and (u["value"] - cur) * d < 0:
            out.append("moves against its words")
    if src != "flip":
        if kind in NONE_KINDS and u["action"] != "none":
            out.append(f"{kind} not none")
        if kind == "no_value" and (a is None or a["lo"] is not None):
            out.append("no_value uses a valued action")
        if kind in MOVE_KINDS and a is not None and a["lo"] is not None:
            st = state_for(world, u)
            cur = st.get(a["sets"], {}).get("value") if a["sets"] else None
            if cur is None:
                out.append("move without a field it sets")
            else:
                span = a["hi"] - a["lo"]
                frac = abs(u["value"] - cur) / span
                if frac < 1e-9:
                    out.append("move re-sends the current value")
                elif kind == "relative_small" and frac > 0.2:
                    out.append("'a bit' moves over 20%")
                elif kind == "relative_big" and frac < 0.12 and u["value"] not in (a["lo"], a["hi"]):
                    out.append("'a lot' moves under 12%")
    return out


def validate(worlds):
    """-> [(world, utterance)] to keep, and a report Counter of drop reasons."""
    keep, why = [], collections.Counter()
    for w in worlds:
        utts = w["utterances"]
        labels = collections.defaultdict(set)
        for u in utts:
            labels[u["say"]].add((u["action"], u.get("value")))
        parent = None
        for u in utts:
            if u["source"] == "request":
                parent = u
            if u["source"] == "flip" and parent is not None and parent.get("limit"):
                u = dict(u, limit=parent["limit"])          # labeled with the field at its limit
            p = problems(w, u, parent if u["source"] == "flip" else None)
            if len(labels[u["say"]]) > 1:
                p.append("same words, two labels")
            if p:
                why.update(p)
            else:
                keep.append((w, u, parent))
    return keep, why


# ---------------------------------------------------------------- conversion
def restyle(name, style):
    words = name.split("_")
    return N.join(style, words) if style in ("title", "camel") else name


def is_int(x):
    return x is None or float(x).is_integer()


def rnd(v, a):
    """Clamp and round like the action's own values: whole steps for integer settings (modes,
    levels, on/off) and wide ranges, two decimals otherwise."""
    v = min(a["hi"], max(a["lo"], v))
    return round(v) if a.get("int") or a["hi"] - a["lo"] >= 10 else round(v, 2)


def mark_int(world):
    """An action takes whole steps when its range, its field's value and every label for it
    are integers (0..1 switches, 1..5 levels, 0..4 modes)."""
    st = {f["name"]: f["value"] for f in world["state"]}
    for a in world["actions"]:
        if a["lo"] is None:
            continue
        labels = [u["value"] for u in world["utterances"] if u["action"] == a["name"]]
        switch = a["hi"] - a["lo"] <= 1
        a["int"] = all(is_int(x) for x in [a["lo"], a["hi"]] + labels + ([] if switch else [st.get(a["sets"])]))


def candidates(a, cur, target, rng):
    """Near-miss values for the right action around its target.

    The range ends are added only sometimes: always offering lo and hi as decoys taught the
    model that an end value is usually wrong ("close the gate" picked 0.16 over 0). When the
    target IS an end, a near-end decoy teaches "fully" = the end, not almost."""
    span = a["hi"] - a["lo"]
    base = cur if cur is not None else rng.uniform(a["lo"], a["hi"])
    d = target - base if abs(target - base) > 1e-9 else span * 0.2
    raw = [base - d, base + 0.35 * d, base + 2.2 * d, base]
    raw += [e for e in (a["lo"], a["hi"]) if rng.random() < 0.3]
    if target == a["lo"]:
        raw.append(a["lo"] + rng.uniform(0.1, 0.2) * span)
    elif target == a["hi"]:
        raw.append(a["hi"] - rng.uniform(0.1, 0.2) * span)
    out = []
    for v in raw:
        if not a["lo"] <= v <= a["hi"]:
            continue                    # dropped, not clamped: clamping piled decoys onto the ends
        v = rnd(v, a)
        if abs(v - target) >= 0.08 * span and all(abs(v - o) >= 0.05 * span for o in out):
            out.append(v)
    rng.shuffle(out)
    return out


def scenario(world, u, rng, style):
    acts = {a["name"]: a for a in world["actions"]}
    st = state_for(world, u)
    tools, rights = [], []
    right = acts.get(u["action"])

    def add(a, v):
        nm = restyle(a["name"], style)
        if a["lo"] is None:
            tools.append(Field(nm, 0, 0, 0, arg=False))
        else:
            tools.append(Field(nm, v, a["lo"], a["hi"]))

    cur_right = st.get(right["sets"], {}).get("value") if right is not None and right.get("sets") else None
    if right is not None and u["kind"] in ORDINAL and right["lo"] is not None and cur_right is not None \
            and abs(u["value"] - cur_right) > 1e-9:
        # strength picks by rank among coarse offered values; several can be right
        up = u["value"] > cur_right
        moves = offer(cur_right, right["lo"], right["hi"], up, rng, whole=right.get("int", False))
        back = offer(cur_right, right["lo"], right["hi"], not up, rng, whole=right.get("int", False))
        for v in moves + back[:rng.randint(0, 1)]:
            add(right, v)
        rights = ranked(len(moves), ORDINAL[u["kind"]])
        if not rights:
            return None
    elif right is not None:
        target_v = None if right["lo"] is None else rnd(u["value"], right)
        if target_v is not None and abs(target_v - u["value"]) > 1e-9:
            return None                          # a fractional value on a range we round to whole steps
        add(right, target_v)
        rights = [0]
        if right["lo"] is not None:
            cur = cur_right
            if u.get("ladder"):                  # intensity: the other strengths are the decoys
                decoys = [v for v in dict.fromkeys(u["ladder"].values()) if v != target_v]
                plain = u["ladder"]["plain"]
                if cur is not None and right["lo"] <= 2 * cur - plain <= right["hi"]:
                    decoys.append(rnd(2 * cur - plain, right))       # the opposite direction
                rng.shuffle(decoys)
                decoys = decoys[:rng.randint(2, 3)]
            else:
                decoys = candidates(right, cur, target_v, rng)[:rng.randint(1, 3)]
            for v in decoys:
                add(right, v)
    others = [a for a in world["actions"] if a is not right]
    rng.shuffle(others)
    for a in others:
        if len(tools) >= MAX_SLOTS:
            break
        cur = st.get(a["sets"], {}).get("value") if a["sets"] else None
        if a["lo"] is None:
            add(a, None)
            continue
        vals = candidates(a, cur, rnd(rng.uniform(a["lo"], a["hi"]), a), rng)[:rng.randint(0, 1)]
        for v in [rnd(rng.uniform(a["lo"], a["hi"]), a)] + vals:
            if len(tools) < MAX_SLOTS:
                add(a, v)
    order = list(range(len(tools)))
    rng.shuffle(order)
    tools = [tools[i] for i in order]
    target = [0.0] * (len(tools) + 1)
    for i in rights:
        target[order.index(i)] = 1.0 / len(rights)
    if not rights:
        target[len(tools)] = 1.0
    state = [Field(restyle(f["name"], style), f["value"], f["lo"], f["hi"]) for f in st.values()][:MAX_SLOTS]
    rng.shuffle(state)
    needed = [restyle(n, style) for n in ([right["sets"]] if right is not None and right.get("sets") else [])
              + ([u["limit"]["field"]] if u.get("limit") else [])]
    words, templates = annotate(u["say"])       # keeps the smart-home word / template holdouts honest
    uses = {"words": words, "templates": templates, "style": style, "units": None,
            "n_state": len(state), "n_tools": len(tools), "rooms": [], "kinds": [],
            "area": world["area"], "world": world["setting"]}
    s = Scenario(state, u["say"], tools, target,
                 {"intent": "world", "cls": "w_" + u["kind"], "source": u["source"], "needed": needed,
                  "chunks": [[u["say"], "core"]], "uses": uses})
    try:
        return s.check()
    except AssertionError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--extra", default=None, help="inverted utterances to merge in (dink/llm/invert.py output)")
    ap.add_argument("--sources", default=None, help="convert only these sources, e.g. 'invert'")
    ap.add_argument("--validate", action="store_true", help="report only")
    ap.add_argument("--worlds", type=int, default=250, help="how many worlds to convert (a seeded sample)")
    ap.add_argument("--variants", type=int, default=2, help="scenarios per utterance (different tools, styles)")
    ap.add_argument("--out", default="data/worlds/pool.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    worlds = load(a.src)
    if a.extra:                                  # inverted variants join their world's utterances
        extra = {}
        for line in Path(a.extra).read_text().splitlines():
            r = json.loads(line)
            extra.setdefault(r["setting"], []).extend(r["utterances"])
        for w in worlds:
            w["utterances"] = w["utterances"] + extra.get(w["setting"], [])
    for w in worlds:
        mark_int(w)
    rng = random.Random(a.seed)
    if not a.validate:
        worlds = rng.sample(worlds, min(a.worlds, len(worlds)))
    keep, why = validate(worlds)
    total = sum(len(w["utterances"]) for w in worlds)
    print(f"{len(worlds)} worlds, {total} utterances: keep {len(keep)} ({len(keep) / total:.0%})")
    for k, v in why.most_common():
        print(f"  drop: {k:36s} {v}")
    if a.validate:
        return
    if a.sources:
        keep = [k for k in keep if k[1]["source"] in a.sources.split(",")]
    out, bad = [], 0
    for w, u, _ in keep:
        for _ in range(a.variants):
            s = scenario(w, u, rng, rng.choices(STYLES, STYLE_WEIGHTS)[0])
            if s is None:
                bad += 1
            else:
                out.append(s)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(s.to_json() + "\n" for s in out))
    print(f"wrote {len(out)} scenarios ({bad} failed conversion) -> {a.out}")


if __name__ == "__main__":
    main()
