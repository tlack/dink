"""Home Assistant's voice templates (github.com/OHF-Voice/intents, CC-BY-4.0) -> prompt-only
Dink scenarios.

These are the hand-written English sentence templates Home Assistant's voice control runs on,
for 40-odd intents (turn on / off, set brightness / temperature / position / volume, media,
timers, lists, weather, time ...). They come with no device state, so every scenario here has
an empty state: Dink must decide from the request and the tool names alone.

  training   sentences sampled from each template block (slot lists filtered to the block's
             required device type), labeled by Home Assistant's own recognizer (hassil), with
             a random area as the speaker's "here"
  testing    the repo's own hand-written test sentences with their stated slots:
             test_ha (intents seen in training; exact duplicates are removed from training)
             test_ha_unseen (HELD_OUT intents: never in training)

Each request becomes one tool call: domain-aware verbs (covers open / close, locks lock /
unlock, others turn on / off), targets from the entity name, or area + device kind. Distractor
tools are hard negatives: the same target with other verbs, and the same verb on other
targets. About 12% of scenarios drop the right tool (answer: nothing relevant), and "never
mind" requests are always nothing relevant.

  git clone --depth 1 https://github.com/OHF-Voice/intents.git data/ha/intents
  uv run python -m dink.gen.ha --out data/ha
"""
import argparse
import collections
import itertools
import random
import re
import sys
from pathlib import Path

from ..data.scenario import MAX_SLOTS, PROMPT_BYTES, Field, Scenario
from .bank import annotate
from .names import join

REPO = Path("data/ha/intents")
HELD_OUT = {"HassFanSetSpeed", "HassMediaPrevious", "HassVacuumReturnToBase", "HassGetCurrentDate"}
SKIP = {"HassBroadcast", "HassRespond", "HassMediaSearchAndPlay", "HassVacuumCleanArea", "HassClimateSetFanMode"}
STYLES, STYLE_WEIGHTS = ["long", "title", "camel"], [0.45, 0.45, 0.1]   # camel is a held-out style

DOMAIN_WORD = {"light": "lights", "fan": "fan", "cover": "blinds", "climate": "thermostat", "media_player": "speaker",
               "switch": "switch", "lock": "lock", "valve": "valve", "vacuum": "vacuum", "lawn_mower": "mower",
               "scene": "scene", "script": "script", "sensor": "sensor", "binary_sensor": "sensor", "todo": "list"}
DEVICE_CLASS_WORD = {"window": "windows", "blind": "blinds", "curtain": "curtains", "shade": "shades",
                     "garage": "garage door", "garage_door": "garage door", "door": "door", "gate": "gate",
                     "awning": "awning", "shutter": "shutters", "damper": "damper", "water": "water valve",
                     "gas": "gas valve", "outlet": "outlet", "tv": "tv", "speaker": "speaker", "receiver": "receiver"}
VERBS = {"cover": ("open", "close"), "valve": ("open", "close"), "lock": ("lock", "unlock"),
         "scene": ("activate", "deactivate"), "script": ("run", "stop")}
NO_VALUE = {"HassMediaPause": "pause", "HassMediaUnpause": "resume", "HassMediaNext": "skip_track_on",
            "HassMediaPrevious": "previous_track_on", "HassMediaPlayerMute": "mute", "HassMediaPlayerUnmute": "unmute",
            "HassVacuumStart": "start", "HassVacuumReturnToBase": "dock", "HassLawnMowerStartMowing": "start",
            "HassLawnMowerDock": "dock"}
FIXED = {"HassGetWeather": "get_weather", "HassGetCurrentTime": "get_time", "HassGetCurrentDate": "get_date",
         "HassCancelAllTimers": "cancel_all_timers", "HassCancelTimer": "cancel_timer", "HassPauseTimer": "pause_timer",
         "HassUnpauseTimer": "resume_timer", "HassTimerStatus": "get_timer_status",
         "HassIncreaseTimer": "add_time_to_timer", "HassDecreaseTimer": "take_time_off_timer",
         "HassShoppingListAddItem": "add_to_shopping_list", "HassShoppingListCompleteItem": "check_off_shopping_item",
         "HassListAddItem": "add_to_list", "HassListCompleteItem": "check_off_list_item",
         "HassListRemoveItem": "remove_from_list"}
DEFAULT_TARGET = {"HassMediaPause": "media", "HassMediaUnpause": "media", "HassMediaNext": "media",
                  "HassMediaPrevious": "media", "HassMediaPlayerMute": "media", "HassMediaPlayerUnmute": "media",
                  "HassVacuumStart": "vacuum", "HassVacuumReturnToBase": "vacuum", "HassLawnMowerStartMowing": "mower",
                  "HassLawnMowerDock": "mower", "HassSetVolume": "volume", "HassSetVolumeRelative": "volume",
                  "HassLightSet": "lights", "HassClimateSetTemperature": "thermostat",
                  "HassClimateGetTemperature": "temperature", "HassFanSetSpeed": "fan", "HassSetPosition": "blinds"}
TIMER_MAX = {"hours": 100, "minutes": 180, "seconds": 600}


def snake(text):
    return "_".join(re.findall(r"[a-z0-9]+", str(text).lower()))


class Labeler:
    """Maps (intent, slots) to Dink tool specs: (name, value, lo, hi) or (name, None)."""

    def __init__(self, fixtures):
        self.domain_of = {e["name"].lower(): e["id"].split(".")[0] for e in fixtures.get("entities", [])
                          if isinstance(e.get("name"), str)}

    def target(self, intent, slots, domain):
        dc = slots.get("device_class")
        kind = DEVICE_CLASS_WORD.get(dc) if dc else DOMAIN_WORD.get(domain) if domain else None
        if "name" in slots:
            return snake(slots["name"])
        where = slots.get("area") or slots.get("floor")
        if where and str(where).lower() not in slots.get("_text", "").lower():
            where = None        # the area came from the speaker's "here", not the words: unknowable
                                # without state, so the tool doesn't name it
        kind = kind or DEFAULT_TARGET.get(intent)
        if where and kind:
            return snake(f"{where} {kind}")
        return snake(kind) if kind else None

    def tool(self, intent, slots):
        domain = slots.get("domain")
        if isinstance(domain, list):
            domain = domain[0] if domain else None
        if not domain and "name" in slots:
            domain = self.domain_of.get(str(slots["name"]).lower())
        if intent in FIXED:
            return (FIXED[intent], None), None
        t = self.target(intent, slots, domain)
        if not t:
            return None, None
        if intent in ("HassTurnOn", "HassTurnOff"):
            on, off = VERBS.get(domain, ("turn_on", "turn_off"))
            return (f"{on if intent == 'HassTurnOn' else off}_{t}", None), (on, off, t)
        if intent in NO_VALUE:
            return (f"{NO_VALUE[intent]}_{t}", None), None
        if intent == "HassGetState":
            return (f"check_{t}", None), None
        if intent == "HassClimateGetTemperature":
            return (f"get_{t}" if t.endswith("temperature") else f"get_{t}_temperature", None), None
        if intent == "HassSetVolumeRelative":
            step = slots.get("volume_step")
            if step in ("up", "down"):
                return (f"turn_{step}_{t}" if t.endswith("volume") else f"turn_{step}_{t}_volume", None), None
            if isinstance(step, (int, float)) and 1 <= step <= 100:   # "turn the volume up by 10%"
                words = set(re.findall(r"[a-z]+", slots.get("_text", "").lower()))
                up = bool(words & {"up", "increase", "raise", "louder", "higher"})
                down = bool(words & {"down", "decrease", "lower", "quieter", "reduce"})
                if up != down:
                    what = t if t.endswith("volume") else f"{t}_volume"
                    return (f"{'raise' if up else 'lower'}_{what}_by", float(step), 1, 100), None
            return None, None
        if intent == "HassStartTimer":
            units = [u for u in TIMER_MAX if u in slots]
            if len(units) != 1:
                return None, None
            v = float(slots[units[0]])
            if not 1 <= v <= TIMER_MAX[units[0]]:
                return None, None
            return (f"start_timer_{units[0]}", v, 1, TIMER_MAX[units[0]]), None
        valued = {"HassLightSet": ("brightness", "brightness"), "HassFanSetSpeed": ("percentage", "speed"),
                  "HassSetPosition": ("position", "position"), "HassSetVolume": ("volume_level", "volume")}
        if intent in valued:
            slot, what = valued[intent]
            if slot not in slots:
                return None, None
            name = f"set_{t}" if t.endswith(what) else f"set_{t}_{what}"
            return (name, float(slots[slot]), 0, 100), None
        if intent == "HassClimateSetTemperature" and "temperature" in slots:
            v = float(slots["temperature"])
            lo, hi = (10, 32) if v <= 35 else (50, 90)
            if not lo <= v <= hi:
                return None, None
            name = "set_temperature" if t == "thermostat" else f"set_{t}_temperature"
            return (name, v, lo, hi), None
        return None, None


def near_values(v, lo, hi, rng, n):
    span = hi - lo
    out = []
    for c in [v - 0.3 * span, v + 0.3 * span, v - 0.15 * span, v + 0.15 * span, lo, hi, rng.uniform(lo, hi)]:
        c = round(min(hi, max(lo, c)))
        if abs(c - v) >= 0.08 * span and all(abs(c - o) >= 0.05 * span for o in out):
            out.append(c)
    rng.shuffle(out)
    return out[:n]


def restyle(name, style):
    return join(style, name.split("_")) if style in ("title", "camel") else name


def build(intent, text, spec, pairs, pool, rng, none_share=0.12, source="ha"):
    """One prompt-only scenario. spec: (name, value, lo, hi) | (name, None); pairs: (on, off, target)."""
    if len(text.encode()) > PROMPT_BYTES:
        return None
    style = rng.choices(STYLES, STYLE_WEIGHTS)[0]
    tools, right = [], None
    if intent != "HassNevermind":
        name = spec[0]
        if len(spec) == 4:
            _, v, lo, hi = spec
            tools.append(Field(name, v, lo, hi))
            tools += [Field(name, c, lo, hi) for c in near_values(v, lo, hi, rng, rng.randint(1, 3))]
        else:
            tools.append(Field(name, 0, 0, 0, arg=False))
        right = 0
        if pairs:                                  # the same device, the other verb
            on, off, t = pairs
            other = f"{off}_{t}" if name.startswith(on) else f"{on}_{t}"
            tools.append(Field(other, 0, 0, 0, arg=False))
    for d in rng.sample(pool, len(pool)):
        if len(tools) >= rng.randint(3, MAX_SLOTS):
            break
        if right is not None and d[0] == spec[0]:
            continue
        if any(t.name == d[0] for t in tools):
            continue
        tools.append(Field(d[0], d[1], d[2], d[3]) if len(d) == 4 else Field(d[0], 0, 0, 0, arg=False))
    if right is not None and rng.random() < none_share:  # the right action isn't offered
        tools = [t for t in tools if t.name != spec[0]]
        right = None
    tools = tools[:MAX_SLOTS]
    if not tools or any(len(restyle(t.name, style).encode()) > 40 for t in tools):
        return None
    order = list(range(len(tools)))
    rng.shuffle(order)
    tools = [Field(restyle(tools[i].name, style), tools[i].value, tools[i].lo, tools[i].hi, tools[i].arg) for i in order]
    target = [0.0] * (len(tools) + 1)
    target[order.index(right) if right is not None else len(tools)] = 1.0
    words, templates = annotate(text.lower())
    uses = {"words": words, "templates": templates, "style": style, "units": None, "n_state": 0,
            "n_tools": len(tools), "rooms": [], "kinds": [], "area": "home assistant"}   # intents are held
                                                                   # out by ha.py itself, not the splitter
    s = Scenario([], text.lower(), tools, target,
                 {"intent": "ha", "cls": "ha_" + intent[4:], "source": source, "needed": [],
                  "chunks": [[text.lower(), "core"]], "uses": uses})
    try:
        return s.check()
    except AssertionError:
        return None


def load_repo():
    sys.path.insert(0, str(REPO.resolve()))
    import os
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        from script.intentfest.util import load_fixtures, load_intents_dict
        from shared import get_slot_lists
        from hassil import Intents
        from hassil.intents import TextSlotList
        intents = Intents.from_dict(load_intents_dict("en"))
        fixtures = load_fixtures("en")
        # the intents' own lists (temperatures, speeds, volumes, timer lengths, cover classes),
        # the fixtures' devices / areas / floors, and example values for free-text items
        lists = dict(intents.slot_lists)
        lists.update(get_slot_lists(fixtures))
        items = ["milk", "eggs", "batteries", "dog food", "call the plumber", "paper towels", "coffee"]
        for name in ("shopping_list_item", "todo_list_item", "item"):
            lists.setdefault(name, TextSlotList.from_strings(items))
    finally:
        os.chdir(cwd)
    return intents, fixtures, lists


def block_lists(lists, requires):
    """Slot lists restricted to the entities a template block can address."""
    import copy
    doms = requires.get("domain") if requires else None
    if not doms or "name" not in lists:
        return lists
    doms = [doms] if isinstance(doms, str) else doms
    out = dict(lists)
    nl = copy.copy(lists["name"])
    nl.values = [v for v in lists["name"].values if (v.context or {}).get("domain") in doms]
    out["name"] = nl
    return out if nl.values else None


def sample_training(intents, lists, rng, per_template):
    """-> [(intent, text, slots)] sampled from the templates, labeled by recognition."""
    from hassil import recognize_all
    from hassil.sample import sample_sentence
    areas = [v.text_in.text for v in lists["area"].values] if "area" in lists else ["Kitchen"]
    out = []
    for name, intent in intents.intents.items():
        if name in SKIP:
            continue
        for dat in intent.data:
            bl = block_lists(lists, dat.requires_context)
            if bl is None:
                continue
            for sent in dat.sentences:
                try:
                    texts = list(itertools.islice(sample_sentence(sent, bl, intents.expansion_rules, language="en"), 400))
                except Exception:
                    continue
                texts = [t for t in texts if "{" not in t]          # a wildcard list left unfilled
                for text in rng.sample(texts, min(per_template, len(texts))):
                    ctx = {"area": rng.choice(areas)}
                    for r in recognize_all(text, intents, slot_lists=lists, intent_context=ctx):
                        if r.intent.name == name:
                            out.append((name, text, {k: v.value for k, v in r.entities.items()}))
                            break
    return out


def load_tests():
    import yaml
    out = []
    for f in sorted((REPO / "tests" / "en").glob("*/*.yaml")):
        intent = f.parent.name
        d = yaml.safe_load(f.read_text()) or {}
        for t in d.get("tests", []):
            for text in t.get("sentences", []):
                out.append((intent, text, dict(t.get("slots") or {})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ha")
    ap.add_argument("--per-template", type=int, default=6)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=600, help="training sentences per intent, at most")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    intents, fixtures, lists = load_repo()
    lab = Labeler(fixtures)
    tests = load_tests()
    test_texts = {t.lower() for _, t, _ in tests}
    train = sample_training(intents, lists, rng, a.per_template)
    print(f"{len(train)} sampled + recognized sentences; {len(tests)} hand-written test sentences")

    labeled = collections.defaultdict(list)            # (split, intent) -> [(text, spec, pairs)]
    dropped = collections.Counter()
    for split, rows in (("train", train), ("test", tests)):
        for intent, text, slots in rows:
            if intent in SKIP:
                continue
            if split == "train" and (intent in HELD_OUT or text.lower() in test_texts):
                dropped["held out or a test sentence"] += 1
                continue
            if intent == "HassNevermind":
                labeled[(split, intent)].append((text, ("none",), None))
                continue
            spec, pairs = lab.tool(intent, dict(slots, _text=text))
            if spec is None:
                dropped[f"unmapped {intent}"] += 1
                continue
            labeled[(split, intent)].append((text, spec, pairs))
    for key in [k for k in labeled if k[0] == "train"]:           # timers and state checks dominate
        rng.shuffle(labeled[key])
        labeled[key] = labeled[key][:a.cap]
    trained = {intent for (split, intent), rows in labeled.items() if split == "train" and rows}
    pool = list({spec for (split, _), rows in labeled.items() if split == "train"
                 for _, spec, _ in rows if spec[0] != "none"})
    pool_all = list({spec for rows in labeled.values() for _, spec, _ in rows if spec[0] != "none"})
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sets = collections.defaultdict(list)
    for (split, intent), rows in labeled.items():
        for text, spec, pairs in rows:
            n = a.variants if split == "train" else 1
            for _ in range(n):
                s = build(intent, text, spec, pairs, pool if split == "train" else pool_all, rng)
                if s is None:
                    continue
                if split == "train":
                    sets["pool"].append(s)
                else:
                    sets["test_ha" if intent in trained else "test_ha_unseen"].append(s)
    for name, items in sets.items():
        (outdir / f"{name}.jsonl").write_text("".join(s.to_json() + "\n" for s in items))
        print(f"  {name:15s} {len(items)}  ({len({s.meta['cls'] for s in items})} intents)")
    for k, v in dropped.most_common(12):
        print(f"  dropped: {k:40s} {v}")


if __name__ == "__main__":
    main()
