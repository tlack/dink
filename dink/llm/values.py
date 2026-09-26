"""Ask an LLM what numeric device states mean in words ("a thermostat at 83 of 60-85 °F"
-> "pretty warm", "nearly maxed out"), across many device types and values.

These pairs teach Dink's value path that 0.92 on a thermostat sits near "warm". The
number itself is kept out of the descriptions, so they must carry meaning in words.
Offline augmentation (dink/teacher/values_corpus.py) reuses each description set for
nearby values. Capped and cached like every LLM job (see openrouter.py).

  uv run python -m dink.llm.values --max-requests 100
"""
import argparse
import json
from pathlib import Path

from .openrouter import Budget, Client

OUT = Path("data/llm/values.jsonl")

# (type id, noun, lo, hi, unit, integer?, held out of training?)
TYPES = [
    ("thermo_f", "thermostat setpoint", 60, 85, "°F", True, False),
    ("thermo_c", "thermostat setpoint", 16, 30, "°C", True, False),
    ("light", "light brightness", 0, 100, "%", True, False),
    ("fan", "fan speed", 0, 3, "(0 = off, 3 = max)", True, False),
    ("blinds", "window blinds opening", 0, 100, "% open", True, False),
    ("volume", "speaker volume", 0, 100, "%", True, False),
    ("humidity", "room humidity", 0, 100, "%", True, False),
    ("room_temp", "room temperature", 50, 100, "°F", True, False),
    ("oven", "oven temperature", 150, 500, "°F", True, True),
    ("battery", "battery charge", 0, 100, "%", True, True),
]
LEVELS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

PROMPT = """A {noun} that ranges from {lo} to {hi} {unit} is currently at {v} {unit}.

Write {n} short descriptions of this state, the way a person would say or think it about a
{noun}: how it feels or what it means in practice, whether it is low, medium or high, near a
limit, off, or at maximum. Mix three styles: terse labels of two or three words; full
sentences that name the device; and effects or consequences a person would notice.

Rules: use words that fit a {noun} specifically; do not write the number {v} or any other
number; lowercase; under 60 characters each; no quotes or numbering.
Reply with only a JSON array of strings."""


def fmt(v, integer):
    return str(int(round(v))) if integer else f"{v:g}"


def seeds():
    for tid, noun, lo, hi, unit, integer, held in TYPES:
        for lv in LEVELS:
            v = lo + lv * (hi - lo)
            if integer:
                v = round(v)
            yield {"type": tid, "noun": noun, "lo": lo, "hi": hi, "unit": unit, "value": v,
                   "level": (v - lo) / (hi - lo), "held_out": held}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-requests", type=int, default=100)
    ap.add_argument("--per-seed", type=int, default=12)
    a = ap.parse_args()
    c = Client(max_requests=a.max_requests)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")
    n_rows = empty_in_a_row = 0
    for s in seeds():
        v = fmt(s["value"], True)
        prompt = PROMPT.format(noun=s["noun"], lo=s["lo"], hi=s["hi"], unit=s["unit"], v=v, n=a.per_seed)
        try:
            items = c.json_list(prompt, max_tokens=2500)
        except (Budget, RuntimeError) as e:
            print(f"stopping: {e}")
            break
        ok = [t.strip().lower() for t in items if isinstance(t, str)
              and 0 < len(t.strip().encode()) <= 64 and not any(ch.isdigit() for ch in t)]
        with open(OUT, "a") as f:
            for t in dict.fromkeys(ok):
                f.write(json.dumps({"text": t, **s, "model": c.model}) + "\n")
        n_rows += len(ok)
        empty_in_a_row = 0 if ok else empty_in_a_row + 1
        if empty_in_a_row >= 5:
            print("stopping: 5 seeds in a row gave nothing usable (check data/llm/live.log)")
            break
    print(f"{n_rows} descriptions from {c.requests} new requests + {c.cache_hits} cached -> {OUT}")


if __name__ == "__main__":
    main()
