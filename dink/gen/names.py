"""How the smart-home generator names tools and state fields, in several styles.

Short styles (the first experiments; 8 bytes or less):
  same  kit_lite     swap  lite_kit     set  set_kitl     dot  kit.lit
Long styles (descriptive names, closer to real systems and to what a text teacher knows):
  long      bedroom_light         long_set  set_bedroom_light (state: bedroom_light)
  ha        light.bedroom         (Home Assistant entity ids: climate.*, cover.*_blinds, lock.*)
  title     Bedroom Light         camel     bedroomLight

In short styles state fields keep the internal names (kit_lite, occ_kit, out_tmp) while tools
vary; in long styles state fields are named in the same style. Every name any style can
produce is in LOOKUP (name -> (room or house key, kind)), so nothing parses names.
"""
ROOM_WORDS = {"lr": ["living", "room"], "kit": ["kitchen"], "bed": ["bedroom"], "ofc": ["office"],
              "bth": ["bathroom"], "den": ["den"]}
KIND_WORDS = {"lite": ["light"], "temp": ["thermostat"], "fan": ["fan"], "blnd": ["blinds"]}
HA_DOMAIN = {"lite": "light", "temp": "climate", "fan": "fan", "blnd": "cover"}
HOUSE_WORDS = {"front": ["front", "door", "lock"], "back": ["back", "door", "lock"], "garage": ["garage", "door"]}
HA_HOUSE = {"front": "lock.front_door", "back": "lock.back_door", "garage": "cover.garage_door"}
HOUSE_SHORT = {  # canon (state) and the short tool names
    "front": {"canon": "frnt_lck", "same": "frnt_lck", "swap": "lck_frnt", "set": "set_flck", "dot": "frnt.lk"},
    "back": {"canon": "back_lck", "same": "back_lck", "swap": "lck_back", "set": "set_blck", "dot": "back.lk"},
    "garage": {"canon": "gar_door", "same": "gar_door", "swap": "door_gar", "set": "set_gdor", "dot": "gar.dr"},
}
SCENE_WORDS = {"goodnite": ["goodnight"], "away": ["away", "mode"], "all_off": ["all", "off"]}
DISTRACTOR_WORDS = {"out_tmp": ["outdoor", "temperature"], "humid": ["humidity"], "hour": ["hour", "of", "day"],
                    "aqi": ["air", "quality"], "batt_pct": ["battery", "percent"], "co2_ppm": ["co2", "ppm"]}

SHORT = ["same", "swap", "set", "dot"]
LONG = ["long", "long_set", "ha", "title", "camel"]
STYLES = SHORT + LONG
MAX_NAME = 40


def join(style, words):
    if style == "title":
        return " ".join(w.capitalize() for w in words)
    if style == "camel":
        return words[0] + "".join(w.capitalize() for w in words[1:])
    return "_".join(words)


def tool(style, room, kind):
    if style == "same":
        return f"{room}_{kind}"
    if style == "swap":
        return f"{kind}_{room}"
    if style == "set":
        return f"set_{room}{kind[0]}"
    if style == "dot":
        return f"{room}.{kind[:3]}"
    if style == "ha":
        return f"{HA_DOMAIN[kind]}.{join(style, ROOM_WORDS[room])}" + ("_blinds" if kind == "blnd" else "")
    name = join(style, ROOM_WORDS[room] + KIND_WORDS[kind])
    return "set_" + name if style == "long_set" else name


def state(style, room, kind):
    if style in SHORT:
        return f"{room}_{kind}"
    return tool("long" if style == "long_set" else style, room, kind)


def house_tool(style, key):
    if style in SHORT:
        return HOUSE_SHORT[key][style]
    if style == "ha":
        return HA_HOUSE[key]
    name = join(style, HOUSE_WORDS[key])
    return "set_" + name if style == "long_set" else name


def house_state(style, key):
    if style in SHORT:
        return HOUSE_SHORT[key]["canon"]
    return house_tool("long" if style == "long_set" else style, key)


def occupancy(style, room):
    if style in SHORT:
        return f"occ_{room}"
    if style == "ha":
        return f"binary_sensor.{join(style, ROOM_WORDS[room])}_occupancy"
    return join(style, ROOM_WORDS[room] + ["occupied"])


def distractor(style, key):
    if style in SHORT:
        return key
    return ("sensor." if style == "ha" else "") + join("long" if style == "ha" else style, DISTRACTOR_WORDS[key])


def scene(style, key):
    if style in SHORT:
        return key
    if style == "ha":
        return "scene." + join("long", SCENE_WORDS[key])
    return join("long" if style == "long_set" else style, SCENE_WORDS[key])


def _lookup():
    out = {}
    for st in STYLES:
        for room in ROOM_WORDS:
            for kind in KIND_WORDS:
                out[tool(st, room, kind)] = out[state(st, room, kind)] = (room, kind)
            out[occupancy(st, room)] = (room, "occ")
        for key in HOUSE_WORDS:
            k = "lock" if key != "garage" else "door"
            out[house_tool(st, key)] = out[house_state(st, key)] = (key, k)
        for key in SCENE_WORDS:
            out[scene(st, key)] = (key, "scene")
        for key in DISTRACTOR_WORDS:
            out[distractor(st, key)] = (None, None)
    assert max(len(n.encode()) for n in out) <= MAX_NAME
    return out


LOOKUP = _lookup()


def parse(name):
    """(room or house key, kind) for any generated name; (None, None) if unknown."""
    return LOOKUP.get(name, (None, None))
