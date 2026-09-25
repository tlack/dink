"""Generate smart-home datasets: a training set and held-out evaluation sets.

  uv run python -m dink.gen                  # data/home/{train,eval_*}.jsonl
  uv run python -m dink.gen --show 12        # print examples instead
  uv run python -m dink.gen --stats          # label / class distributions of the train set
"""
import argparse
import collections
import random
from pathlib import Path

from . import augment as A
from .home import HomeGen, Split

TRAIN = Split()
EVAL_SETS = {  # name -> (split, stage-2 recipe applied once, or None)
    "indist": (Split(), None),
    "h_range": (Split(units="C"), None),
    "h_template": (Split(templates="heldout"), None),
    "h_words": (Split(words="heldout"), None),
    "h_rooms": (Split(rooms="heldout"), None),
    "h_naming": (Split(naming="heldout"), None),
    "h_slots": (Split(state_slots=(7, 8), tool_slots=(7, 8)), None),
    "h_none": (Split(only_none=True, words="heldout"), None),
    "h_noise": (Split(), [(A.asr_noise, 1.0), (A.typos, 1.0), (A.fillers, 0.5)]),
}


def generate(split, n, seed, recipe=None):
    g = HomeGen(split, seed)
    rng = random.Random(seed + 1)
    out = []
    for _ in range(n):
        s = g.sample()
        if recipe:
            s = A.augment(s, rng, recipe)
        out.append(s)
    return out


def write(path, scenarios):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s.to_json() + "\n" for s in scenarios))


def stats(scenarios):
    c = collections.Counter(s.meta["cls"] for s in scenarios)
    none = sum(s.label == len(s.tools) for s in scenarios)
    amb = sum(bool(s.meta.get("ambiguous")) for s in scenarios)
    print(f"{len(scenarios)} scenarios: nothing-relevant {none} ({100 * none / len(scenarios):.0f}%), "
          f"ambiguous {amb} ({100 * amb / len(scenarios):.0f}%)")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v}")
    lens = sorted(len(s.prompt) for s in scenarios)
    tools = collections.Counter(len(s.tools) for s in scenarios)
    state = collections.Counter(len(s.state) for s in scenarios)
    print(f"  prompt length median {lens[len(lens) // 2]}, max {lens[-1]} bytes")
    print(f"  tools per scenario {dict(sorted(tools.items()))}")
    print(f"  state fields       {dict(sorted(state.items()))}")
    pos = collections.Counter(s.label for s in scenarios if s.label < len(s.tools))
    print(f"  answer slot        {dict(sorted(pos.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/home")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=0, help="print N examples (optionally --set NAME)")
    ap.add_argument("--set", default="train")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.show or a.stats:
        split, recipe = (TRAIN, None) if a.set == "train" else EVAL_SETS[a.set]
        items = generate(split, max(a.show, 1000 if a.stats else 0), a.seed, recipe)
        for s in items[: a.show]:
            print(s.show() + "\n")
        if a.stats:
            stats(items)
        return
    out = Path(a.out)
    write(out / "train.jsonl", generate(TRAIN, a.n_train, a.seed))
    for i, (name, (split, recipe)) in enumerate(EVAL_SETS.items()):
        write(out / f"eval_{name}.jsonl", generate(split, a.n_eval, 1000 + 97 * i, recipe))
    print(f"wrote {out}/train.jsonl ({a.n_train}) and {len(EVAL_SETS)} eval sets ({a.n_eval} each)")


if __name__ == "__main__":
    main()
