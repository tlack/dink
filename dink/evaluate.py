"""Score a trained Dink checkpoint on a split's test sets.

  uv run python -m dink.evaluate runs/dink/big_tmpl --split data/home/split_v2
"""
import argparse
from pathlib import Path

import torch

from .model.dink import Dink
from .train import evaluate, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", required=True)
    ap.add_argument("--sets", default="test_*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ck = torch.load(Path(a.run) / "model.pt", weights_only=False, map_location=a.device)
    model = Dink(width=ck["args"]["width"]).to(a.device)
    model.load_state_dict(ck["state"])
    print(f"{'set':16s} {'acc':>6s} {'p(right)':>9s} {'nothing P/R':>12s}")
    for p in sorted(Path(a.split).glob(a.sets + ".jsonl")):
        r = evaluate(model, load(p))
        print(f"{p.stem:16s} {r['acc']:6.3f} {r['p_right']:9.3f} {r['nothing_precision']:6.2f}/{r['nothing_recall']:.2f}")


if __name__ == "__main__":
    main()
