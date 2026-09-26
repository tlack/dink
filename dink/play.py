"""Drive a trained Dink by hand: give it a state, tools and a prompt, see what it picks.

One shot:
  uv run python -m dink.play \\
      --state "bedroom_thermostat=70  bedroom_light=0.3  outdoor_temperature=95:-10..110" \\
      --tools "set_bedroom_thermostat=71  set_bedroom_thermostat=73  set_bedroom_thermostat=76 \\
               set_bedroom_light=0  set_bedroom_light=1  goodnight" \\
      "it's a bit chilly in here"

Interactive (keeps the house loaded; type prompts, or :help):
  uv run python -m dink.play -i
  uv run python -m dink.play -i --random          # start from a held-out test scenario

Item syntax, space- or comma-separated:  name=value[:lo..hi]   or a bare name for a tool
with no argument (a scene). Names can't contain spaces here (use _). Without a range, one is
guessed from words in the name: thermostat/temp/climate/heat -> 60..85, light/lamp and
blinds/shades/cover -> 0..1, fan -> 0..3, lock/door -> 0..1. The generator's styles
(dink/gen/names.py): bedroom_light, set_bedroom_light, light.bedroom, kit_lite, set_kitl ...
Rooms: living room, kitchen, bedroom, office, bathroom (den was held out).
"""
import argparse
import random
import re
from pathlib import Path

import torch

from .data.collate import collate
from .data.numbers import extract
from .data.scenario import Field, Scenario
from .gen.home import KINDS
from .gen.names import parse
from .model.dink import SLOTS, Dink

RANGES = {k: (v.lo, v.hi) for k, v in KINDS.items()}
HELP = """commands:
  <any text>          ask the model (the prompt)
  :state <items>      replace the state          :tools <items>   replace the tools
  :add state|tools <items>                       :show            show the house
  :random             load a random held-out test scenario (shows its right answer)
  :quit"""


GUESS = [(("thermostat", "temp", "climate", "heat", "_t"), RANGES["temp"]), (("fan",), RANGES["fan"]),
         (("light", "lamp", "lite", "blind", "shade", "curtain", "cover", "lock", "door", "lck"), (0.0, 1.0))]


def guess_range(name):
    _, k = parse(name)
    if k in RANGES:
        return RANGES[k]
    if k in ("lock", "door", "occ"):
        return 0.0, 1.0
    low = name.lower()
    for words, rng in GUESS:
        if any(w in low for w in words):
            return rng
    raise ValueError(f"no range for {name!r}: write it as {name}=<value>:<lo>..<hi>")


def parse_items(text):
    out = []
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"([^=]+)=(-?[\d.]+)(?::(-?[\d.]+)\.\.(-?[\d.]+))?", tok)
        if not m:
            out.append(Field(tok, 0, 0, 0, arg=False))
            continue
        name, v = m.group(1), float(m.group(2))
        lo, hi = (float(m.group(3)), float(m.group(4))) if m.group(3) else guess_range(name)
        out.append(Field(name, v, lo, hi))
    if len(out) > SLOTS:
        raise ValueError(f"at most {SLOTS} items (got {len(out)})")
    return out


def fmt(f):
    return f"{f.name}()" if not f.arg else f"{f.name}={f.value:g} [{f.lo:g}..{f.hi:g}]"


def load_model(run, device):
    ck = torch.load(Path(run) / "model.pt", weights_only=False, map_location=device)
    m = Dink(width=ck["args"]["width"], scenario=ck["args"].get("scenario", False)).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def ask(model, state, tools, prompt, answer=None):
    s = Scenario(state, prompt, tools, [0.0] * len(tools) + [1.0])
    x = collate([s])
    dev = next(model.parameters()).device
    p = model({k: v.to(dev) for k, v in x.items()}).softmax(-1)[0].cpu()
    probs = p[:len(tools)].tolist() + [p[SLOTS].item()]
    nums = extract(prompt)
    print(f"\n  prompt : {prompt!r}" + (f"   numbers seen: {nums}" if nums else ""))
    print(f"  state  : {'  '.join(fmt(f) for f in state) or '(empty)'}")
    best = max(range(len(probs)), key=probs.__getitem__)
    rows = [(fmt(t), probs[i], i) for i, t in enumerate(tools)] + [("(nothing relevant)", probs[-1], len(tools))]
    for label, pr, i in sorted(rows, key=lambda r: -r[1]):
        mark = "->" if i == best else "  "
        right = "  <- right answer" if answer is not None and answer[i] >= max(answer) - 1e-6 else ""
        print(f"  {mark} {label:30s} {pr:6.1%}  {'#' * round(pr * 30)}{right}")
    print()


def random_case(split):
    files = sorted(Path(split).glob("test_*.jsonl"))
    f = random.choice(files)
    s = Scenario.from_json(random.choice(f.read_text().splitlines()))
    print(f"  loaded from {f.stem} (class {s.meta.get('cls')})")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--run", default="runs/dink/all_inv")
    ap.add_argument("--state", default="")
    ap.add_argument("--tools", default="")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--random", action="store_true", help="start from a random held-out test scenario")
    ap.add_argument("--split", default="data/home/split_v6")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model = load_model(a.run, a.device)
    state, tools, answer = parse_items(a.state), parse_items(a.tools), None
    if a.random:
        s = random_case(a.split)
        state, tools, answer = s.state, s.tools, s.target
        ask(model, state, tools, s.prompt, answer)
    if a.prompt:
        if not tools:
            ap.error("give --tools (and usually --state)")
        ask(model, state, tools, a.prompt)
    if not a.interactive:
        return
    print(HELP)
    while True:
        try:
            line = input("dink> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if not line:
                continue
            if line in (":q", ":quit", ":exit"):
                return
            if line == ":help":
                print(HELP)
            elif line == ":show":
                print("  state:", "  ".join(fmt(f) for f in state) or "(empty)")
                print("  tools:", "  ".join(fmt(f) for f in tools) or "(empty)")
            elif line == ":random":
                s = random_case(a.split)
                state, tools, answer = s.state, s.tools, s.target
                ask(model, state, tools, s.prompt, answer)
            elif line.startswith(":state"):
                state, answer = parse_items(line[6:]), None
            elif line.startswith(":tools"):
                tools, answer = parse_items(line[6:]), None
            elif line.startswith(":add"):
                which, _, items = line[4:].strip().partition(" ")
                new = parse_items(items)
                if which == "state":
                    state = (state + new)[:SLOTS]
                else:
                    tools = (tools + new)[:SLOTS]
                answer = None
            elif line.startswith(":"):
                print("unknown command; :help")
            elif not tools:
                print("no tools yet: :tools <items>")
            else:
                ask(model, state, tools, line, None)
                answer = None
        except ValueError as e:
            print(f"  {e}")


if __name__ == "__main__":
    main()
