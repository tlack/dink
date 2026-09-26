"""Ask an LLM for varied phrasings of smart-home requests whose meaning we already know.

Each seed is a meaning ("the bedroom fan, a lot slower"); the LLM writes ways a person
might say it. Every phrasing keeps its seed's meaning, so it can later serve as a labeled
Dink prompt, not just distillation text. Capped and cached (see openrouter.py), so the job
can be stopped and re-run: finished seeds come from the cache for free.

  uv run python -m dink.llm.paraphrase --max-requests 20
"""
import argparse
import json
import random
from pathlib import Path

from ..gen import home as H
from .openrouter import Budget, Client

OUT = Path("data/llm/paraphrases.jsonl")
INTENSITY = {"small": "a little", "default": "", "big": "a lot"}
DIRECTION = {"lite": ("brighter", "dimmer"), "temp": ("warmer", "cooler"),
             "fan": ("faster", "slower"), "blnd": ("more open", "more closed")}

PROMPT = """Write {n} different things a person might say to a smart home voice assistant that mean exactly this:
{meaning}

Rules:
- keep exactly the same meaning and the same amount of change (do not make "a little" into "a lot", or drop it)
- vary the wording and sentence structure: casual, terse, polite, indirect ("it's too dark in here")
- {room_rule}
- lowercase, under 60 characters each, no quotes, no numbering
Reply with only a JSON array of strings."""


def seeds(rng):
    """Meanings to paraphrase: relative changes, on/off, locks, scenes."""
    out = []
    rooms = [r for r in H.ROOMS if r.code != "den"]
    for key, k in H.KINDS.items():
        for up in (True, False):
            for inten in ("small", "default", "big"):
                room = rng.choice(rooms)
                word = DIRECTION[key][0 if up else 1]
                amount = INTENSITY[inten]
                out.append({"type": "relative", "kind": key, "dir": 1 if up else -1, "intensity": inten,
                            "room": room.code,
                            "meaning": f"the {room.words[0]} {k.nouns[0]}, {amount + ' ' if amount else ''}{word}"})
    for key in ("lite", "fan", "blnd"):
        for on in (True, False):
            room = rng.choice(rooms)
            state = {"lite": ("on", "off"), "fan": ("on", "off"), "blnd": ("open", "closed")}[key][0 if on else 1]
            out.append({"type": "turn", "kind": key, "on": on, "room": room.code,
                        "meaning": f"the {room.words[0]} {H.KINDS[key].nouns[0]} turned {state}"
                        if key != "blnd" else f"the {room.words[0]} blinds {state}"})
    for name, h in H.HOUSE.items():
        for on in (True, False):
            verb = ("lock", "unlock") if h["lock"] else ("open", "close")
            out.append({"type": "lock", "device": name, "on": on,
                        "meaning": f"{verb[0 if on else 1]} the {h['words'][0]}"})
    for scene, phrases in H.SCENES.items():
        out.append({"type": "scene", "scene": scene, "meaning": f"run the '{phrases[0]}' routine"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-requests", type=int, default=20)
    ap.add_argument("--per-seed", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c = Client(max_requests=a.max_requests)
    rng = random.Random(a.seed)
    rows, bad, empty_in_a_row = [], 0, 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")
    for s in seeds(rng):
        room_rule = ("mention the room in about half of them, and leave it out in the rest"
                     if "room" in s else "no room is involved")
        try:
            items = c.json_list(PROMPT.format(n=a.per_seed, meaning=s["meaning"], room_rule=room_rule),
                                max_tokens=2500)
        except (Budget, RuntimeError) as e:
            print(f"stopping: {e}")
            break
        ok = [t.strip().lower() for t in items if isinstance(t, str) and 0 < len(t.strip().encode()) <= 64]
        bad += len(items) - len(ok) if items else 1
        new = [{"text": t, "seed": s, "model": c.model} for t in dict.fromkeys(ok)]
        rows += new
        with open(OUT, "a") as f:                       # written as it arrives
            f.write("".join(json.dumps(r) + "\n" for r in new))
        empty_in_a_row = 0 if new else empty_in_a_row + 1
        if empty_in_a_row >= 5:
            print("stopping: 5 seeds in a row gave nothing usable (check data/llm/live.log)")
            break
    print(f"{len(rows)} phrasings from {c.requests} new requests + {c.cache_hits} cached; "
          f"{bad} unusable replies/items -> {OUT}")


if __name__ == "__main__":
    main()
