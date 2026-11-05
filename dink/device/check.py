"""Accuracy of the device arithmetic (float NumPy reference, int8 trunk) on held-out sets.

  uv run python -m dink.device.check models/dink-v16 --n 200
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from ..data.numbers import extract
from ..data.scenario import Scenario
from . import reference as R
from .export import Trunk, calibration_texts, load


def args_of(s):
    return ([(f.name, f.value, f.lo, f.hi) for f in s.state], s.prompt,
            [(f.name, f.value, f.lo, f.hi, f.arg) for f in s.tools],
            [(v, k == "level") for v, k in extract(s.prompt)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="data/home/split_v13")
    ap.add_argument("--sets", default="test_areas,test_worlds,test_words,test_llm,test_nostate")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    t = load(a.run)
    trunk = Trunk(t, calibration_texts())
    fl, q8 = R.Reference(t), R.Reference(t, trunk)
    names_f, names_q = {}, {}
    print(f"{'set':14s} {'n':>4s} {'float':>6s} {'int8':>6s} {'same pick':>9s} {'max |dp|':>8s}")
    for name in a.sets.split(","):
        f = Path(a.split) / f"{name}.jsonl"
        if not f.exists():
            continue
        lines = f.read_text().splitlines()
        rows = random.Random(0).sample(lines, min(a.n, len(lines)))
        ok_f = ok_q = same = 0
        dp = 0.0
        for line in rows:
            s = Scenario.from_json(line)
            pf = fl.score(*args_of(s), names=names_f)
            pq = q8.score(*args_of(s), names=names_q)
            best = max(s.target)
            ok_f += s.target[int(pf.argmax())] >= best - 1e-6
            ok_q += s.target[int(pq.argmax())] >= best - 1e-6
            same += pf.argmax() == pq.argmax()
            dp = max(dp, float(np.abs(pf - pq).max()))
        n = len(rows)
        print(f"{name:14s} {n:4d} {ok_f / n:6.3f} {ok_q / n:6.3f} {same / n:9.3f} {dp:8.3f}")
    print(f"largest accumulator {q8.max_acc} (limit {1 << 19})")


if __name__ == "__main__":
    main()
