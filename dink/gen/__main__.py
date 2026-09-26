"""Generate a pool of scenarios. Splitting into train / val / test is dink/data/split.py.

  uv run python -m dink.gen --n 20000 --out data/home/pool.jsonl
  uv run python -m dink.gen --show 12        # print examples instead
  uv run python -m dink.gen --stats          # class mix of 2000 samples
"""
import argparse
import collections
from pathlib import Path

from .home import GenConfig, HomeGen


def generate(n, seed, cfg=None, bank=None):
    g = HomeGen(cfg or GenConfig(), seed, bank)
    return [g.sample() for _ in range(n)]


def stats(scenarios):
    c = collections.Counter(s.meta["cls"] for s in scenarios)
    none = sum(s.label == len(s.tools) for s in scenarios)
    amb = sum(bool(s.meta.get("ambiguous")) for s in scenarios)
    print(f"{len(scenarios)} scenarios: nothing-relevant {none} ({100 * none / len(scenarios):.0f}%), "
          f"ambiguous {amb} ({100 * amb / len(scenarios):.0f}%)")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v}")
    lens = sorted(len(s.prompt) for s in scenarios)
    print(f"  prompt length median {lens[len(lens) // 2]}, max {lens[-1]} bytes")
    for key in ("n_tools", "n_state", "style", "units"):
        print(f"  {key:8s} {dict(sorted(collections.Counter(s.meta['uses'][key] for s in scenarios).items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/home/pool.jsonl")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--bank", default=None, help="LLM phrase bank (data/llm/phrases.jsonl)")
    ap.add_argument("--bank-p", type=float, default=0.5, help="share of prompts taken from the bank")
    a = ap.parse_args()
    cfg = GenConfig(bank_p=a.bank_p)
    from .bank import load
    bank = load(a.bank) if a.bank else None
    if a.show or a.stats:
        items = generate(max(a.show, 2000 if a.stats else 0), a.seed, cfg, bank)
        for s in items[: a.show]:
            print(s.show() + "\n")
        if a.stats:
            stats(items)
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(s.to_json() + "\n" for s in generate(a.n, a.seed, cfg, bank)))
    print(f"wrote {a.n} scenarios to {out}")


if __name__ == "__main__":
    main()
