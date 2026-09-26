"""Hand-written hard cases, scored by every training run (and by dink.evaluate).

They target failures seen when driving the model by hand: the same word meaning opposite
actions depending on framing ("i'm hot" = cooler, "i want it hot" = warmer; "it's too
dark" = brighter, "make it dark" = off), intensifiers outside the generator's list ("as
hell", "super", "crank"), and requests that only look like commands. Each case exists in
long names (bedroom_thermostat) and short ones (bed_temp / set_bedt).

  uv run python -m dink.evals.hard --run runs/dink/big_llm
"""
import argparse

from ..data.scenario import Field, Scenario

# (prompt, right answers as tool keys); tool keys index TOOLS below
CASES = [
    ("i want it hot as hell", ["t76"]),
    ("crank the heat way up", ["t76"]),
    ("make it a little warmer", ["t71"]),
    ("warmer please", ["t73"]),
    ("i'm hot", ["t67"]),
    ("i'm freezing", ["t76"]),
    ("i want it cold in here", ["t67"]),
    ("make it really dark in here", ["l0"]),
    ("it's too dark in here", ["l0.6"]),
    ("make it super bright", ["l1"]),
    ("lights off", ["l0"]),
    ("goodnight", ["scene"]),
    ("what's the temperature in here", ["none"]),
    ("is it dark outside yet", ["none"]),
]
# key -> (long name, short name, value, lo, hi)
TOOLS = {"t71": ("set_bedroom_thermostat", "set_bedt", 71, 60, 85),
         "t73": ("set_bedroom_thermostat", "set_bedt", 73, 60, 85),
         "t76": ("set_bedroom_thermostat", "set_bedt", 76, 60, 85),
         "t67": ("set_bedroom_thermostat", "set_bedt", 67, 60, 85),
         "l0": ("set_bedroom_light", "set_bedl", 0, 0, 1),
         "l0.6": ("set_bedroom_light", "set_bedl", 0.6, 0, 1),
         "l1": ("set_bedroom_light", "set_bedl", 1, 0, 1)}
SCENE = ("goodnight", "goodnite")
STATE = [("bedroom_thermostat", "bed_temp", 70, 60, 85), ("bedroom_light", "bed_lite", 0.3, 0, 1),
         ("outdoor_temperature", "out_tmp", 55, -10, 110)]


def cases():
    out = []
    for style, i in (("long", 0), ("short", 1)):
        state = [Field(f[i], f[2], f[3], f[4]) for f in STATE]
        keys = list(TOOLS) + ["scene"]
        tools = [Field(TOOLS[k][i], TOOLS[k][2], TOOLS[k][3], TOOLS[k][4]) for k in TOOLS] + \
                [Field(SCENE[i], 0, 0, 0, arg=False)]
        for prompt, right in CASES:
            target = [0.0] * (len(tools) + 1)
            for r in right:
                target[len(tools) if r == "none" else keys.index(r)] = 1.0 / len(right)
            out.append(Scenario(state, prompt, tools, target, {"cls": f"hard_{style}"}).check())
    return out


def main():
    import torch
    from ..data.collate import collate
    from ..play import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/dink/big_llm")
    a = ap.parse_args()
    model = load_model(a.run, "cpu")
    items = cases()
    with torch.no_grad():
        p = model(collate(items)).softmax(-1)
    ok = 0
    for s, pr in zip(items, p):
        n = len(s.tools)
        pick = int(pr.argmax())
        pick = n if pick >= n else pick
        right = s.target[pick] > 0
        ok += right
        name = "(nothing)" if pick == n else f"{s.tools[pick].name}={s.tools[pick].value:g}"
        want = [("(nothing)" if i == n else f"{s.tools[i].name}={s.tools[i].value:g}")
                for i, t in enumerate(s.target) if t > 0]
        print(f"{'ok ' if right else 'BAD'} {s.meta['cls'][5:]:5s} {s.prompt!r:34s} -> {name:28s}"
              + ("" if right else f"  (want {', '.join(want)})"))
    print(f"{ok}/{len(items)} right")


if __name__ == "__main__":
    main()
