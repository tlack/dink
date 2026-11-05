"""Ask Dink about a gadget it has never seen.

  uv run python examples/quickstart.py
  uv run python examples/quickstart.py "it's freezing in here"     # your own request

A gadget is its state (named values with ranges) and its actions. An action with a range
sets a value; one without (lo = hi = None) just runs. The harness turns the gadget and the
request into Dink's options in two passes: first which action and which way, then how much.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dink.harness import decide
from dink.play import load_model

MODEL = Path(__file__).resolve().parents[1] / "models" / "dink-v16"

gadget = {
    "state": [
        {"name": "water_temperature", "value": 102, "lo": 80, "hi": 104},
        {"name": "ph_level", "value": 7.4, "lo": 6.8, "hi": 8.2},
        {"name": "sanitizer_level", "value": 2.5, "lo": 0, "hi": 10},
        {"name": "jet_pump_speed", "value": 1, "lo": 0, "hi": 3},
        {"name": "ambient_air_temp", "value": 68, "lo": 40, "hi": 100},
    ],
    "actions": [
        {"name": "set_temperature", "lo": 80, "hi": 104, "sets": "water_temperature"},
        {"name": "set_ph", "lo": 6.8, "hi": 8.2, "sets": "ph_level"},
        {"name": "increase_sanitizer", "lo": 0, "hi": 10, "sets": "sanitizer_level"},
        {"name": "set_jet_speed", "lo": 0, "hi": 3, "sets": "jet_pump_speed"},
        {"name": "turn_on_ozone", "lo": None, "hi": None, "sets": None},
    ],
}

requests = sys.argv[1:] or [
    "turn on the ozone",                    # an action without a value
    "set jet speed to 2",                   # a number said: offered as-is
    "bump the sanitizer up a bit",          # a relative amount: pass 2 picks how much
    "the water is way too hot",             # a complaint: which action, which way, how much
    "i want it hotter",
    "what's the ph right now?",             # a question, not a command
    "turn on the underwater light",         # a device this gadget doesn't have
    "play some jazz",                       # unrelated
    "more jets",                            # an everyday word: the action is set_jet_speed
]

model = load_model(MODEL, "cpu")
for prompt in requests:
    r = decide(model, gadget, prompt)
    if r["pick"] is None:
        answer = "(nothing relevant)"
    elif r["pick"].arg:
        answer = f"{r['pick'].name}({r['pick'].value:g})"
    else:
        answer = f"{r['pick'].name}()"
    print(f"{prompt!r:48} -> {answer:28} {r['p']:5.0%}{'  (unsure)' if r.get('unsure') else ''}")
