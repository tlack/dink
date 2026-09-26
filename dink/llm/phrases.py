"""A bank of LLM-written prompts, keyed by meaning, for the generator to use as Dink prompts.

The generator (dink/gen/home.py) builds the house, tools and right answer; when it has a bank
it swaps its template prompt for a bank phrasing with the same key (same intent, device kind,
direction, amount, and the same room, or none). So LLM text becomes labeled training data
without the LLM ever seeing a tool list.

The prompts aim at what the first Dink runs got wrong, described by kind rather than by
quoting test prompts: unusual synonyms for devices and rooms, softened or negated modifiers
("not so", "less", "a notch"), complaints that imply the opposite change, questions that are
not requests, and off-topic requests. Each call also gets a speaker persona, so replies
don't collapse into the same few phrasings.

Keys (JSON lists):
  ["relative", kind, dir, intensity, room|null]   ["indirect", kind, dir, intensity, room|null]
  ["turn", kind, on, room|null]   ["lock", device, on]   ["scene", name]
  ["question"]   ["offdomain"]

  uv run python -u -m dink.llm.phrases --max-requests 170
"""
import argparse
import json
import random
from pathlib import Path

from ..gen import home as H
from .openrouter import Budget, Client

OUT = Path("data/llm/phrases.jsonl")
AMOUNT = {"small": "by a small amount", "default": "by a normal amount", "big": "by a large amount"}
DIRECTION = {"lite": ("brighter", "dimmer"), "temp": ("warmer", "cooler"),
             "fan": ("faster", "slower"), "blnd": ("more open", "more closed")}
DEVICE = {"lite": "lights", "temp": "thermostat", "fan": "fan", "blnd": "blinds"}
PERSONAS = [
    "a busy parent, short and practical", "a teenager who uses slang", "an older person, polite and wordy",
    "someone terse who uses 2 to 4 words", "a non-native english speaker", "someone rambling mid-conversation",
    "someone annoyed", "someone half asleep", "speech-to-text output: no punctuation, small word errors",
    "someone joking around",
]

COMMON = """Write {n} different things {persona} might say to a smart home voice assistant.
{task}

Variety matters more than anything. Mix:
- everyday synonyms and nicknames for the device and the room, not just the obvious word
- softened, negated or comparative wording ("not so ...", "less ...", "a notch ...", "tone down ...")
- complaints or descriptions of how the speaker feels, when they clearly imply the request
- different sentence shapes: commands, questions, fragments, "i need ...", "can we ..."
Rules: {rules}
lowercase, under 60 characters each, no quotes, no numbering, no digits.
Reply with only a JSON array of objects: {{"t": "<what they say>", "room": true or false}}
where room says whether the sentence names or clearly refers to a specific room."""


def seeds(rng, repeats=3):
    rooms = [r for r in H.ROOMS if r.code != "den"]
    out = []

    def room_rule(room):
        return (f"the room is the {room.words[0]}; name it (any natural way) in about half, and leave the room "
                "out entirely in the rest")

    for key in H.KINDS:
        for d in (1, -1):
            for inten in ("small", "default", "big"):
                for room in rng.sample(rooms, repeats):
                    task = (f"They want the {room.words[0]} {DEVICE[key]} {DIRECTION[key][0 if d > 0 else 1]}, "
                            f"changed {AMOUNT[inten]} (keep that amount: do not make a small change big, "
                            "or a big one small).")
                    out.append({"type": "relative", "kind": key, "dir": d, "intensity": inten,
                                "room": room.code, "task": task, "rules": room_rule(room)})
    for key in ("temp", "lite"):
        for d in (1, -1):
            for inten in ("small", "default", "big"):
                for room in rng.sample(rooms, repeats):
                    task = (f"They only describe how they feel or what is wrong, and never name the device or say "
                            f"what to change, but it clearly means the {room.words[0]} should be "
                            f"{DIRECTION[key][0 if d > 0 else 1]} {AMOUNT[inten]} "
                            f"({'mild' if inten == 'small' else 'strong' if inten == 'big' else 'plain'} wording).")
                    out.append({"type": "indirect", "kind": key, "dir": d, "intensity": inten,
                                "room": room.code, "task": task, "rules": room_rule(room)})
    for key in ("lite", "fan", "blnd"):
        for on in (True, False):
            for room in rng.sample(rooms, repeats):
                state = {"lite": ("on", "off"), "fan": ("on", "off"), "blnd": ("fully open", "fully closed")}[key]
                task = f"They want the {room.words[0]} {DEVICE[key]} turned {state[0 if on else 1]}."
                out.append({"type": "turn", "kind": key, "on": on, "room": room.code, "task": task,
                            "rules": room_rule(room)})
    for name, h in H.HOUSE.items():
        for on in (True, False):
            verb = ("locked", "unlocked") if h["lock"] else ("opened", "closed")
            for _ in range(2):
                out.append({"type": "lock", "device": name, "on": on,
                            "task": f"They want the {h['words'][0]} {verb[0 if on else 1]}.",
                            "rules": "no room is involved"})
    scene_meaning = {"goodnite": "they are going to bed and want the house set for the night",
                     "away": "they are leaving the house and want it set for being away",
                     "all_off": "they want every device in the house turned off"}
    for scene, meaning in scene_meaning.items():
        for _ in range(2):
            out.append({"type": "scene", "scene": scene, "task": f"It means: {meaning}.",
                        "rules": "no room is involved"})
    for key, dev in list(DEVICE.items()) + [("door", "doors and locks")]:
        for _ in range(2):
            out.append({"type": "question", "task": (
                f"They ask about or comment on the {dev} (status, history, how it works, whether something "
                "is on), but do NOT ask for anything to be changed."),
                "rules": "name a room in some of them"})
    for _ in range(10):
        out.append({"type": "offdomain", "task": (
            "They ask for something a home assistant might do that is NOT controlling lights, "
            "temperature, fans, blinds, doors or locks: music, timers, shopping, weather, cooking, "
            "messages, trivia. Include some that mention hot, cold, light, dark, open or close in "
            "another sense (e.g. about food, the weather, a store's hours)."),
            "rules": "any wording"})
    return out


def key_of(seed, room_named):
    t = seed["type"]
    room = seed.get("room") if room_named else None
    if t in ("relative", "indirect"):
        return [t, seed["kind"], seed["dir"], seed["intensity"], room]
    if t == "turn":
        return [t, seed["kind"], seed["on"], room]
    if t == "lock":
        return [t, seed["device"], seed["on"]]
    if t == "scene":
        return [t, seed["scene"]]
    return [t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-requests", type=int, default=20)
    ap.add_argument("--per-seed", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c = Client(max_requests=a.max_requests)
    rng = random.Random(a.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")
    seen, n_rows, bad, empty_in_a_row = set(), 0, 0, 0
    for i, s in enumerate(seeds(rng)):
        persona = PERSONAS[i % len(PERSONAS)]
        prompt = COMMON.format(n=a.per_seed, persona=persona, task=s["task"], rules=s["rules"])
        try:
            items = c.json_list(prompt, max_tokens=4000)
        except (Budget, RuntimeError) as e:
            print(f"stopping: {e}")
            break
        new = []
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("t"), str):
                bad += 1
                continue
            t = " ".join(it["t"].lower().replace('"', "").split())
            if not t or len(t.encode()) > 64 or any(ch.isdigit() for ch in t) or t in seen:
                bad += 1
                continue
            seen.add(t)
            named = bool(it.get("room")) and s["type"] not in ("lock", "scene", "question", "offdomain")
            new.append({"text": t, "key": key_of(s, named), "persona": persona, "model": c.model})
        with open(OUT, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in new))
        n_rows += len(new)
        print(f"[{i}] {s['type']:9s} {json.dumps(key_of(s, True)):40s} +{len(new):3d}  ({persona})", flush=True)
        empty_in_a_row = 0 if new else empty_in_a_row + 1
        if empty_in_a_row >= 5:
            print("stopping: 5 seeds in a row gave nothing usable (check data/llm/live.log)")
            break
    print(f"{n_rows} phrasings from {c.requests} new requests + {c.cache_hits} cached; "
          f"{bad} dropped items -> {OUT}")


if __name__ == "__main__":
    main()
