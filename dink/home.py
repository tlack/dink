"""Smart-home scenario generator (stage 1): devices, values, offered tools, the request,
and the right answer. Answer-preserving surface changes (slot order, typos, fillers,
speech noise, moving prompt pieces) live in augment.py (stage 2).

Every word list and template carries a split tag, "train" or "heldout", so evaluation
sets can use phrasings, rooms, naming styles and units that training never saw.
"""
import random
from dataclasses import dataclass, field, replace

from .scenario import MAX_SLOTS, PROMPT_BYTES, Field, Scenario


# ---------------------------------------------------------------- configuration
@dataclass
class Split:
    """What an evaluation set holds out. The default is the training distribution."""
    words: str = "train"        # synonym / phrase lists
    templates: str = "train"    # prompt templates
    rooms: str = "train"        # "heldout": scenarios use rooms never trained (the den)
    naming: str = "train"       # "heldout": a tool naming style never trained
    units: str = "F"            # "C": thermostats in Celsius (unseen ranges)
    state_slots: tuple = (2, 6)
    tool_slots: tuple = (2, 6)
    only_none: bool = False     # generate only "nothing relevant" cases


def pick(rng, options, split):
    """options: [(item, split_tag)]. Items of the requested split, falling back to train."""
    pool = [o for o, s in options if s == split] or [o for o, s in options if s == "train"]
    return rng.choice(pool)


# ---------------------------------------------------------------- the house
@dataclass
class Room:
    code: str                   # used in names, <= 3 bytes
    words: list                 # [(phrase, split)]


ROOMS = [
    Room("lr", [("living room", "train"), ("lounge", "train"), ("family room", "heldout")]),
    Room("kit", [("kitchen", "train")]),
    Room("bed", [("bedroom", "train"), ("master bedroom", "heldout")]),
    Room("ofc", [("office", "train"), ("study", "heldout")]),
    Room("bth", [("bathroom", "train"), ("bath", "train"), ("washroom", "heldout")]),
]
HELDOUT_ROOMS = [Room("den", [("den", "train")])]


@dataclass
class Kind:
    key: str                    # name suffix, 4 bytes: lite / temp / fan / blnd
    lo: float
    hi: float
    steps: dict                 # intensity -> step in value units
    integer: bool = False
    nouns: list = field(default_factory=list)       # [(phrase, split)]
    up: list = field(default_factory=list)          # comparative words, value going up
    down: list = field(default_factory=list)


KINDS = {
    "lite": Kind("lite", 0, 1, {"small": 0.15, "default": 0.3, "big": 0.6},
                 nouns=[("lights", "train"), ("light", "train"), ("lamps", "heldout")],
                 up=[("brighter", "train"), ("lighter", "train"), ("more lit", "heldout")],
                 down=[("dimmer", "train"), ("darker", "train"), ("less bright", "heldout")]),
    "temp": Kind("temp", 60, 85, {"small": 1, "default": 3, "big": 6}, integer=True,
                 nouns=[("thermostat", "train"), ("heat", "train"), ("temperature", "train"),
                        ("ac", "train"), ("climate", "heldout")],
                 up=[("warmer", "train"), ("hotter", "train"), ("toastier", "heldout")],
                 down=[("cooler", "train"), ("colder", "train"), ("chillier", "heldout")]),
    "fan": Kind("fan", 0, 3, {"small": 1, "default": 1, "big": 2}, integer=True,
                nouns=[("fan", "train"), ("ceiling fan", "heldout")],
                up=[("faster", "train"), ("stronger", "train"), ("harder", "heldout")],
                down=[("slower", "train"), ("gentler", "heldout")]),
    "blnd": Kind("blnd", 0, 1, {"small": 0.2, "default": 0.5, "big": 1.0},
                 nouns=[("blinds", "train"), ("shades", "train"), ("curtains", "heldout")],
                 up=[("more open", "train"), ("wider", "heldout")],
                 down=[("more closed", "train"), ("less open", "heldout")]),
}
CELSIUS = replace(KINDS["temp"], lo=16, hi=30, steps={"small": 1, "default": 2, "big": 3})

# Whole-house binary devices: value 1 = locked / open.
HOUSE = {
    "front": dict(canon="frnt_lck", names={"same": "frnt_lck", "swap": "lck_frnt", "set": "set_flck",
                                           "dot": "frnt.lk"},
                  words=[("front door", "train"), ("main door", "heldout")], lock=True),
    "back": dict(canon="back_lck", names={"same": "back_lck", "swap": "lck_back", "set": "set_blck",
                                          "dot": "back.lk"},
                 words=[("back door", "train"), ("rear door", "heldout")], lock=True),
    "garage": dict(canon="gar_door", names={"same": "gar_door", "swap": "door_gar", "set": "set_gdor",
                                            "dot": "gar.dr"},
                   words=[("garage door", "train"), ("garage", "train"), ("carport door", "heldout")],
                   lock=False),
}
SCENES = {  # tool name -> [(phrase, split)]
    "goodnite": [("goodnight", "train"), ("time for bed", "train"), ("nighty night", "heldout")],
    "away": [("we're leaving", "train"), ("i'm heading out", "train"), ("see you later house", "heldout")],
    "all_off": [("turn everything off", "train"), ("shut it all down", "heldout")],
}
DISTRACTORS = [  # state fields that never matter for the request
    ("out_tmp", -10, 110), ("humid", 0, 100), ("hour", 0, 23), ("aqi", 0, 300),
    ("batt_pct", 0, 100), ("co2_ppm", 400, 2000),
]

TRAIN_STYLES = ["same", "swap", "set"]
HELDOUT_STYLES = ["dot"]


def tool_name(style, room, kind):
    if style == "same":
        return f"{room}_{kind}"
    if style == "swap":
        return f"{kind}_{room}"
    if style == "set":
        return f"set_{room}{kind[0]}"
    if style == "dot":
        return f"{room}.{kind[:3]}"
    raise ValueError(style)


# ---------------------------------------------------------------- prompt templates
# {dev} device noun, {room} "kitchen " or "", {loc} trailing " in the kitchen" (a movable
# chunk), {cmp} comparative word, {int} intensity prefix ("a little "), {pol} politeness.
INTENSITY = {
    "small": [("a little ", "train"), ("slightly ", "train"), ("a bit ", "train"), ("a touch ", "heldout")],
    "big": [("a lot ", "train"), ("way ", "train"), ("much ", "train"), ("a whole lot ", "heldout")],
    "default": [("", "train")],
}
POLITE = [("", "train"), ("", "train"), ("please ", "train"), ("could you ", "train"),
          ("hey ", "train"), ("can you ", "train"), ("would you mind ", "heldout")]

RELATIVE = [  # direction-neutral templates; {cmp} carries the direction
    ("{pol}make the {room}{dev} {int}{cmp}", "train"),
    ("{pol}make it {int}{cmp}{loc}", "train"),
    ("{pol}set the {room}{dev} {int}{cmp}", "train"),
    ("i want the {room}{dev} {int}{cmp}", "train"),
    ("{room}{dev} {int}{cmp}", "train"),
    ("{int}{cmp} {room}{dev} please", "heldout"),
    ("{pol}get the {room}{dev} {int}{cmp}", "heldout"),
]
TURN = {  # absolute on/off style requests per (kind, on?)
    ("lite", True): [("{pol}turn on the {room}{dev}", "train"), ("{pol}switch on the {room}{dev}", "train"),
                     ("{room}{dev} on", "train"), ("lights on{loc}", "train"),
                     ("{pol}flip the {room}{dev} on", "heldout")],
    ("lite", False): [("{pol}turn off the {room}{dev}", "train"), ("{pol}switch off the {room}{dev}", "train"),
                      ("{room}{dev} off", "train"), ("kill the {room}{dev}", "train"),
                      ("{pol}shut off the {room}{dev}", "heldout")],
    ("fan", True): [("{pol}turn on the {room}{dev}", "train"), ("start the {room}{dev}", "train"),
                    ("{pol}fire up the {room}{dev}", "heldout")],
    ("fan", False): [("{pol}turn off the {room}{dev}", "train"), ("stop the {room}{dev}", "train"),
                     ("{pol}shut down the {room}{dev}", "heldout")],
    ("blnd", True): [("{pol}open the {room}{dev}", "train"), ("{room}{dev} up", "train"),
                     ("{pol}raise the {room}{dev}", "heldout")],
    ("blnd", False): [("{pol}close the {room}{dev}", "train"), ("{room}{dev} down", "train"),
                      ("{pol}lower the {room}{dev}", "heldout")],
}
INDIRECT = {  # (kind, up?) -> [(text, intensity, split)] -- the speaker's condition implies the change
    ("temp", True): [("i'm cold{loc}", "default", "train"), ("i'm freezing{loc}", "big", "train"),
                     ("it's a bit chilly{loc}", "small", "train"), ("it's cold{loc}", "default", "train"),
                     ("brr it's frigid{loc}", "big", "heldout")],
    ("temp", False): [("i'm hot{loc}", "default", "train"), ("it's so hot{loc}", "default", "train"),
                      ("it's stuffy{loc}", "small", "train"), ("i'm burning up{loc}", "big", "train"),
                      ("i'm roasting{loc}", "big", "heldout")],
    ("lite", True): [("it's too dark{loc}", "default", "train"), ("i can't see{loc}", "big", "train"),
                     ("it's gloomy{loc}", "default", "heldout")],
    ("lite", False): [("it's too bright{loc}", "default", "train"), ("the light hurts my eyes", "big", "train"),
                      ("the glare is awful{loc}", "default", "heldout")],
}
NUMERIC = {  # absolute numeric requests; {num} rendered per kind
    "temp": [("{pol}set the {room}{dev} to {num}", "train"), ("{room}{dev} {num} degrees", "train"),
             ("make it {num}{loc}", "train"), ("{pol}put the {room}{dev} at {num}", "heldout")],
    "lite": [("{pol}set the {room}{dev} to {num} percent", "train"), ("{room}{dev} at {num} percent", "train"),
             ("{pol}dim the {room}{dev} to {num} percent", "heldout")],
    "fan": [("{pol}set the {room}{dev} to {num}", "train"), ("{room}{dev} speed {num}", "train"),
            ("{pol}put the {room}{dev} on {num}", "heldout")],
}
LOCKS = {
    ("lock", True): [("{pol}lock the {dev}", "train"), ("lock up the {dev}", "train"),
                     ("{pol}secure the {dev}", "heldout")],
    ("lock", False): [("{pol}unlock the {dev}", "train"), ("let me in the {dev}", "train"),
                      ("{pol}unbolt the {dev}", "heldout")],
    ("door", True): [("{pol}open the {dev}", "train"), ("{dev} up", "train"), ("{pol}raise the {dev}", "heldout")],
    ("door", False): [("{pol}close the {dev}", "train"), ("{dev} down", "train"), ("{pol}shut the {dev}", "heldout")],
}
OFF_DOMAIN = [("what time is it", "train"), ("play some jazz", "train"), ("tell me a joke", "train"),
              ("add milk to the shopping list", "train"), ("what's the weather tomorrow", "train"),
              ("call mom", "train"), ("set a timer for ten minutes", "train"), ("how are you", "train"),
              ("who won the game last night", "heldout"), ("order a pizza", "heldout"),
              ("remind me to water the plants", "heldout"), ("what's on my calendar", "heldout")]
QUESTIONS = [("is the {dev} locked", "train"), ("are the {room}{dev} on", "train"),
             ("what's the temperature{loc}", "train"), ("is the {dev} open", "train"),
             ("did i leave the {room}{dev} on", "heldout")]


# ---------------------------------------------------------------- building blocks
def candidate_values(kind: Kind, cur, rng, want=None):
    """What a device harness would offer for a numeric tool: a spread of values around
    the current one and across the range (never the current value itself)."""
    if kind.integer:
        span = kind.hi - kind.lo
        deltas = [1, 2, 3, 5, 8] if span > 6 else [1, 2, 3]
        vals = {cur + d for d in deltas} | {cur - d for d in deltas} | {kind.lo, kind.hi}
    else:
        vals = {0.0, 1.0, 0.5} | {round(cur + d, 2) for d in (0.15, 0.3, 0.6, -0.15, -0.3, -0.6)}
    if want is not None:
        vals.add(want)
    vals = {v for v in vals if kind.lo <= v <= kind.hi and abs(v - cur) > 1e-6}
    return sorted(vals)


def best_move(values, cur, target_value, direction):
    """The candidate a person would pick: moves in the asked direction (if any), and lands
    closest to the target value. None if no candidate qualifies."""
    ok = [v for v in values if (direction == 0 or (v - cur) * direction > 0)]
    if not ok:
        return None
    return min(ok, key=lambda v: (abs(v - target_value), abs(v - cur)))


@dataclass
class Device:
    room: str                   # room code, or a HOUSE key for whole-house devices
    kind: str                   # KINDS key, "lock" or "door"
    canon: str                  # state field name
    lo: float
    hi: float
    value: float
    kobj: Kind = None
    words: list = None          # for house devices


class HomeGen:
    def __init__(self, split: Split = None, seed: int = 0):
        self.sp = split or Split()
        self.rng = random.Random(seed)

    # -- helpers
    def w(self, options):
        return pick(self.rng, options, self.sp.words)

    def t(self, options):
        return pick(self.rng, options, self.sp.templates)

    def kind(self, key):
        return CELSIUS if key == "temp" and self.sp.units == "C" else KINDS[key]

    def rooms(self):
        return HELDOUT_ROOMS + self.rng.sample(ROOMS, 2) if self.sp.rooms == "heldout" else ROOMS

    def random_value(self, k: Kind):
        if k.integer:
            return self.rng.randint(int(k.lo), int(k.hi))
        return round(self.rng.choice([0.0, 0.0, 1.0, self.rng.uniform(0, 1)]), 2)

    def build_house(self):
        """A random set of devices with current values."""
        rng, devs = self.rng, []
        rooms = self.rooms()
        n_rooms = rng.randint(1, min(4, len(rooms)))
        chosen = rooms[:1] + rng.sample(rooms[1:], n_rooms - 1) if self.sp.rooms == "heldout" \
            else rng.sample(rooms, n_rooms)
        for r in chosen:
            for key in rng.sample(list(KINDS), rng.randint(1, len(KINDS))):
                k = self.kind(key)
                devs.append(Device(r.code, key, f"{r.code}_{key}", k.lo, k.hi, self.random_value(k), kobj=k))
        for key in rng.sample(list(HOUSE), rng.randint(0, len(HOUSE))):
            h = HOUSE[key]
            devs.append(Device(key, "lock" if h["lock"] else "door", h["canon"], 0, 1,
                               float(rng.randint(0, 1)), words=h["words"]))
        return devs, chosen

    def room_words(self, code):
        for r in ROOMS + HELDOUT_ROOMS:
            if r.code == code:
                return self.w(r.words)
        raise KeyError(code)

    # -- the main entry point
    def sample(self) -> Scenario:
        for _ in range(200):
            s = self._try()
            if s is not None:
                return s.check()
        raise RuntimeError("could not build a scenario")

    INTENTS = [  # (name, weight)
        ("relative", 4.0), ("turn", 2.0), ("indirect", 1.5), ("numeric", 1.0), ("lock", 1.2),
        ("scene", 0.6), ("none_offdomain", 0.8), ("none_question", 0.4), ("none_missing", 0.8),
        ("none_limit", 0.5), ("none_noise", 0.2),
    ]

    def _try(self):
        rng = self.rng
        intents = [(n, w) for n, w in self.INTENTS if not self.sp.only_none or n.startswith("none")]
        intent = rng.choices([n for n, _ in intents], [w for _, w in intents])[0]
        devs, rooms = self.build_house()
        style = rng.choice(HELDOUT_STYLES if self.sp.naming == "heldout" else TRAIN_STYLES)
        n_tools = rng.randint(*self.sp.tool_slots)
        n_state = rng.randint(*self.sp.state_slots)
        return getattr(self, "_" + intent.split("_")[0])(intent, devs, rooms, style, n_tools, n_state)

    # ---------------------------------------------------------------- tools and state
    def tools_for(self, d: Device, style, values):
        name = HOUSE[d.room]["names"][style] if d.room in HOUSE else tool_name(style, d.room, d.kind)
        return [Field(name, v, d.lo, d.hi) for v in values]

    def assemble(self, devs, style, n_tools, n_state, must_tools, extra_devs, needed_state, occ=None):
        """Fill the tool and state slots: required tools first, then distractor tools from
        other devices (same kind in other rooms = hard negatives), then state fields."""
        rng = self.rng
        tools = list(must_tools)
        pool = []
        for d in extra_devs:
            vals = [0.0, 1.0] if d.kobj is None else candidate_values(d.kobj, d.value, rng)
            pool += [(d, v) for v in vals]
        rng.shuffle(pool)
        for d, v in pool:
            if len(tools) >= min(n_tools, MAX_SLOTS):
                break
            f = self.tools_for(d, style, [v])[0]
            if all((f.name, f.value) != (t.name, t.value) for t in tools):
                tools.append(f)
        if rng.random() < 0.3 and len(tools) < n_tools:          # a scene tool as a distractor
            scene = rng.choice(list(SCENES))
            if all(t.name != scene for t in tools):
                tools.append(Field(scene, 0, 0, 0, arg=False))
        if len(tools) > min(MAX_SLOTS, self.sp.tool_slots[1]):
            return None, None
        # state: needed fields, occupancy, then other devices, then distractors
        state = {}
        for d in needed_state:
            state[d.canon] = Field(d.canon, d.value, d.lo, d.hi)
        for code, v in (occ or {}).items():
            state[f"occ_{code}"] = Field(f"occ_{code}", v, 0, 1)
        others = [d for d in devs if d.canon not in state]
        rng.shuffle(others)
        for d in others:
            if len(state) >= n_state:
                break
            state[d.canon] = Field(d.canon, d.value, d.lo, d.hi)
        for name, lo, hi in rng.sample(DISTRACTORS, len(DISTRACTORS)):
            if len(state) >= n_state:
                break
            state[name] = Field(name, float(rng.randint(lo, hi)), lo, hi)
        if len(state) > min(MAX_SLOTS, self.sp.state_slots[1]):
            return None, None
        return tools, list(state.values())

    def render(self, template, dev="", room_code=None, cmp="", intensity="default", num="", mention_room=True):
        """Fill a template. Returns (prompt, chunks); chunks mark the pieces augment.py may move."""
        room = loc = ""
        if room_code and mention_room and "{loc}" not in template and "{room}" not in template:
            return None, None                   # the label depends on a room this template can't name
        if room_code and mention_room:
            words = self.room_words(room_code)
            if "{loc}" in template and "{room}" not in template:
                loc = " in the " + words
            else:
                room = words + " "
        pol = self.w(POLITE) if "{pol}" in template else ""   # only where a command allows it
        text = template.format(pol="", dev=dev, room=room, loc="", cmp=cmp,
                               int=self.w(INTENSITY[intensity]), num=num)
        chunks = []
        if pol:
            chunks.append([pol.strip(), "polite"])
        chunks.append([" ".join(text.split()), "core"])
        if loc:
            chunks.append([loc.strip(), "loc"])
        prompt = " ".join(c for c, _ in chunks)
        if len(prompt.encode()) > PROMPT_BYTES:
            return None, None
        return prompt, chunks

    def finish(self, intent, cls, tools, state, target_idx, prompt, chunks, room=None, ambiguous=False,
               needed=()):
        """target_idx: {tool index: weight} (normalized here), or {} for "nothing relevant"."""
        if tools is None or prompt is None:
            return None
        order = list(range(len(tools)))
        self.rng.shuffle(order)
        tools = [tools[i] for i in order]
        inv = {old: new for new, old in enumerate(order)}
        target = [0.0] * (len(tools) + 1)
        total = sum(target_idx.values())
        for i, wgt in target_idx.items():
            target[inv[i]] = wgt / total
        if not target_idx:
            target[-1] = 1.0
        self.rng.shuffle(state)
        return Scenario(state, prompt, tools, target,
                        {"intent": intent, "cls": cls, "room": room, "ambiguous": ambiguous,
                         "needed": list(needed), "chunks": chunks, "split": vars(self.sp)})

    # ---------------------------------------------------------------- room resolution
    def resolve(self, devs, rooms_with_kind):
        """Decide whether the prompt names the room. Without a name, occupancy sensors may
        pick one room; otherwise the request is ambiguous across the plausible rooms."""
        rng = self.rng
        if self.sp.rooms == "heldout":          # the point of that set: requests about the new room
            return True, {}, None
        mention = rng.random() < 0.65 or len(rooms_with_kind) == 1 and rng.random() < 0.5
        occ = {}
        if not mention and len(rooms_with_kind) > 1 and rng.random() < 0.6:
            here = rng.choice(rooms_with_kind)
            occ = {r: float(r == here) for r in rooms_with_kind}
            return mention, occ, [here]
        if not mention and len(rooms_with_kind) > 1:
            return mention, occ, list(rooms_with_kind)          # ambiguous
        return mention, occ, None

    # ---------------------------------------------------------------- intents
    def _pick_device(self, devs, kinds):
        cands = [d for d in devs if d.kind in kinds and d.room not in HOUSE]
        if self.sp.rooms == "heldout":
            cands = [d for d in cands if d.room in {r.code for r in HELDOUT_ROOMS}]
        return self.rng.choice(cands) if cands else None

    def _relative(self, intent, devs, rooms, style, n_tools, n_state, force_limit=False):
        rng = self.rng
        d = self._pick_device(devs, list(KINDS))
        if d is None:
            return None
        k = d.kobj
        up = rng.random() < 0.5
        direction = 1 if up else -1
        intensity = rng.choice(["small", "default", "default", "big"])
        same_kind = [x for x in devs if x.kind == d.kind and x.room not in HOUSE]
        if force_limit:                         # name the room, and put that device at its limit
            mention, occ, plausible = True, {}, None
        else:
            mention, occ, plausible = self.resolve(devs, [x.room for x in same_kind])
        cands = [x for x in same_kind if plausible is None and x is d or plausible and x.room in plausible]
        for x in cands:
            self._movable(x, up, at_limit=force_limit)
        must, target, needed = [], {}, []
        for x in cands:
            want = x.value + direction * x.kobj.steps[intensity]
            vals = candidate_values(x.kobj, x.value, rng)
            best = best_move(vals, x.value, want, direction)
            keep = self._subset(vals, best, x.value, rng)
            base = len(must)
            must += self.tools_for(x, style, keep)
            if best is not None:
                target[base + keep.index(best)] = 1.0
            needed.append(x)
        others = [x for x in devs if x not in cands]
        tools, state = self.assemble(devs, style, n_tools, n_state, must, others, needed, occ)
        cmp = self.w(k.up if up else k.down)
        prompt, chunks = self.render(self.t(RELATIVE), dev=self.w(k.nouns), room_code=d.room, cmp=cmp,
                                     intensity=intensity, mention_room=mention)
        cls = "limit" if force_limit else ("relative_" + intensity)
        s = self.finish(intent, cls, tools, state, target, prompt, chunks,
                        room=",".join(sorted({x.room for x in cands})) or d.room,
                        ambiguous=len(target) > 1, needed=[x.canon for x in needed])
        if s:
            s.meta["dir"] = direction
        return s

    def _movable(self, x, up, at_limit=False):
        """Put a device at its limit in the asked direction, or make sure it can move."""
        k = x.kobj
        if at_limit:
            x.value = k.hi if up else k.lo
        elif (up and x.value >= k.hi) or (not up and x.value <= k.lo):
            v = k.lo + (k.hi - k.lo) * self.rng.uniform(0.2, 0.8)
            x.value = float(round(v)) if k.integer else round(v, 2)

    def _subset(self, vals, best, cur, rng):
        """Offer 1-4 candidates, always including the right one (if any): the harness would
        offer several, and near misses make the task real."""
        k = rng.randint(1, 4)
        others = [v for v in vals if v != best]
        rng.shuffle(others)
        keep = ([best] if best is not None else []) + others[: k - (best is not None)]
        return sorted(keep) if keep else sorted(vals)[:1]

    def _turn(self, intent, devs, rooms, style, n_tools, n_state):
        rng = self.rng
        d = self._pick_device(devs, ["lite", "fan", "blnd"])
        if d is None:
            return None
        on = rng.random() < 0.5
        k = d.kobj
        same_kind = [x for x in devs if x.kind == d.kind and x.room not in HOUSE]
        mention, occ, plausible = self.resolve(devs, [x.room for x in same_kind])
        cands = [x for x in same_kind if plausible is None and x is d or plausible and x.room in plausible]
        must, target = [], {}
        for x in cands:
            want = (2 if x.kind == "fan" else x.kobj.hi) if on else x.kobj.lo
            vals = candidate_values(x.kobj, x.value, rng, want=want if x.value != want else None)
            best = best_move(vals, x.value, want, 0) if x.value != want else None
            if x.value == want:                 # already there: re-sending the same value is right
                vals = sorted(set(vals) | {want})
                best = want
            keep = self._subset(vals, best, x.value, rng)
            base = len(must)
            must += [Field(f.name, f.value, f.lo, f.hi) for f in self.tools_for(x, style, keep)]
            target[base + keep.index(best)] = 1.0
        others = [x for x in devs if x not in cands]
        tools, state = self.assemble(devs, style, n_tools, n_state, must, others, cands, occ)
        prompt, chunks = self.render(self.t(TURN[(d.kind, on)]), dev=self.w(k.nouns), room_code=d.room,
                                     mention_room=mention)
        return self.finish(intent, "turn", tools, state, target, prompt, chunks,
                           room=",".join(sorted({x.room for x in cands})) or d.room,
                           ambiguous=len(target) > 1, needed=[x.canon for x in cands])

    def _indirect(self, intent, devs, rooms, style, n_tools, n_state):
        rng = self.rng
        d = self._pick_device(devs, ["temp", "lite"])
        if d is None:
            return None
        up = rng.random() < 0.5
        text, intensity = pick(rng, [((t, i), s) for t, i, s in INDIRECT[(d.kind, up)]], self.sp.templates)
        direction = 1 if up else -1
        same_kind = [x for x in devs if x.kind == d.kind and x.room not in HOUSE]
        mention, occ, plausible = self.resolve(devs, [x.room for x in same_kind])
        cands = [x for x in same_kind if plausible is None and x is d or plausible and x.room in plausible]
        for x in cands:
            self._movable(x, up)
        must, target = [], {}
        for x in cands:
            want = x.value + direction * x.kobj.steps[intensity]
            vals = candidate_values(x.kobj, x.value, rng)
            best = best_move(vals, x.value, want, direction)
            keep = self._subset(vals, best, x.value, rng)
            base = len(must)
            must += self.tools_for(x, style, keep)
            if best is not None:
                target[base + keep.index(best)] = 1.0
        others = [x for x in devs if x not in cands]
        tools, state = self.assemble(devs, style, n_tools, n_state, must, others, cands, occ)
        prompt, chunks = self.render(text, room_code=d.room, mention_room=mention)
        s = self.finish(intent, "indirect_" + intensity, tools, state, target, prompt, chunks,
                        room=",".join(sorted({x.room for x in cands})) or d.room,
                        ambiguous=len(target) > 1, needed=[x.canon for x in cands])
        if s:
            s.meta["dir"] = direction
        return s

    def _numeric(self, intent, devs, rooms, style, n_tools, n_state):
        rng = self.rng
        d = self._pick_device(devs, ["temp", "lite", "fan"])
        if d is None:
            return None
        k = d.kobj
        if d.kind == "lite":
            want = rng.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9])
            num = str(round(want * 100))
        else:
            want = rng.randint(int(k.lo), int(k.hi))
            num = str(want)
        if abs(d.value - want) < 1e-6:
            d.value = k.lo if want != k.lo else k.hi
        vals = candidate_values(k, d.value, rng, want=want)
        keep = self._subset(vals, want, d.value, rng)
        must = self.tools_for(d, style, keep)
        target = {keep.index(want): 1.0}
        others = [x for x in devs if x is not d]
        tools, state = self.assemble(devs, style, n_tools, n_state, must, others, [d])
        same_kind_rooms = {x.room for x in devs if x.kind == d.kind and x.room not in HOUSE}
        prompt, chunks = self.render(self.t(NUMERIC[d.kind]), dev=self.w(k.nouns), room_code=d.room, num=num,
                                     mention_room=(self.sp.rooms == "heldout" or len(same_kind_rooms) > 1
                                                   or rng.random() < 0.5))
        return self.finish(intent, "numeric", tools, state, target, prompt, chunks, room=d.room,
                           needed=[d.canon])

    def _lock(self, intent, devs, rooms, style, n_tools, n_state):
        rng = self.rng
        house = [d for d in devs if d.room in HOUSE]
        if not house:
            return None
        d = rng.choice(house)
        on = rng.random() < 0.5                  # lock / open garage
        want = 1.0 if on else 0.0
        must = self.tools_for(d, style, [0.0, 1.0] if rng.random() < 0.7 else [want])
        target = {[t.value for t in must].index(want): 1.0}
        others = [x for x in devs if x is not d]
        tools, state = self.assemble(devs, style, n_tools, n_state, must, others, [d])
        prompt, chunks = self.render(self.t(LOCKS[(d.kind, on)]), dev=self.w(d.words))
        return self.finish(intent, "lock", tools, state, target, prompt, chunks, room=d.room, needed=[d.canon])

    def _scene(self, intent, devs, rooms, style, n_tools, n_state):
        rng = self.rng
        scene = rng.choice(list(SCENES))
        must = [Field(scene, 0, 0, 0, arg=False)]
        other_scenes = [s for s in SCENES if s != scene]
        if rng.random() < 0.5:
            must.append(Field(rng.choice(other_scenes), 0, 0, 0, arg=False))
        tools, state = self.assemble(devs, style, n_tools, n_state, must, devs, [])
        prompt, chunks = self.render(self.w(SCENES[scene]))
        return self.finish(intent, "scene", tools, state, {0: 1.0}, prompt, chunks)

    # ---------------------------------------------------------------- "nothing relevant"
    def _none(self, intent, devs, rooms, style, n_tools, n_state):
        rng = self.rng
        kind = intent.split("_", 1)[1]
        if kind == "limit":                     # a relative request that can't go any further
            return self._relative(intent, devs, rooms, style, n_tools, n_state, force_limit=True)
        if kind == "missing":                   # a sensible request for a device not offered
            d = self._pick_device(devs, list(KINDS))
            if d is None:
                return None
            k = d.kobj
            up = rng.random() < 0.5
            others = [x for x in devs if x.kind != d.kind]
            if not others:
                return None
            tools, state = self.assemble(devs, style, n_tools, n_state, [], others, [d])
            if not tools:
                return None
            prompt, chunks = self.render(self.t(RELATIVE), dev=self.w(k.nouns), room_code=d.room,
                                         cmp=self.w(k.up if up else k.down),
                                         intensity=rng.choice(["small", "default", "big"]))
            return self.finish(intent, "none_missing", tools, state, {}, prompt, chunks, room=d.room)
        tools, state = self.assemble(devs, style, n_tools, n_state, [], devs, [])
        if not tools:
            return None
        if kind == "offdomain":
            prompt, chunks = self.render(self.w(OFF_DOMAIN))
        elif kind == "question":
            d = rng.choice(devs)
            words = self.w(d.words) if d.words else self.w(d.kobj.nouns)
            prompt, chunks = self.render(self.t(QUESTIONS), dev=words,
                                         room_code=None if d.room in HOUSE else d.room)
        else:                                   # noise
            letters = "abcdefghijklmnopqrstuvwxyz     "
            prompt = "".join(rng.choice(letters) for _ in range(rng.randint(3, 30))).strip() or "hmm"
            chunks = [[prompt, "core"]]
        return self.finish(intent, "none_" + kind, tools, state, {}, prompt, chunks)
