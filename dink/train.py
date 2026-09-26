"""Train Dink on a split (dink/data/split.py) and evaluate on every test set.

  uv run python -u -m dink.train --init distilled --ckpt runs/text/tv128/model.pt --name d_distilled
  uv run python -u -m dink.train --init scratch --name d_scratch

Each step draws base examples and augments them afresh (dink/gen/augment.py). Loss: cross
entropy against the (possibly split) target distribution over 8 tools + nothing relevant.
"""
import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data.collate import collate
from .data.scenario import Scenario
from .gen import augment as A
from .model.dink import SLOTS, Dink


def load(path):
    return [Scenario.from_json(l) for l in Path(path).read_text().splitlines() if l.strip()]


def is_nothing(s):
    return s.label == len(s.tools)


@torch.no_grad()
def evaluate(model, items, batch=256):
    """Accuracy (a pick is right if it is one of the best answers), nothing-relevant
    precision / recall, accuracy per request class, mean probability of the right answer."""
    model.eval()
    ok = tp = fp = fn = 0
    per = collections.defaultdict(lambda: [0, 0])
    p_right = 0.0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = collate(chunk)
        prob = model(x).softmax(-1)
        pred = prob.argmax(-1)
        for s, p, pr, t in zip(chunk, pred.tolist(), prob, x["target"]):
            right = bool(t[p] >= t.max() - 1e-6)
            ok += right
            per[s.meta["cls"]][0] += right
            per[s.meta["cls"]][1] += 1
            p_right += float((pr * t).sum())
            said_nothing, is_n = p == SLOTS, is_nothing(s)
            tp += said_nothing and is_n
            fp += said_nothing and not is_n
            fn += (not said_nothing) and is_n
    n = len(items)
    return {"acc": ok / n, "p_right": p_right / n,
            "nothing_precision": tp / max(1, tp + fp), "nothing_recall": tp / max(1, tp + fn),
            "by_class": {k: round(a / b, 3) for k, (a, b) in sorted(per.items())}, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/home/split")
    ap.add_argument("--init", choices=["scratch", "distilled"], default="distilled")
    ap.add_argument("--ckpt", default="runs/text/tv128/model.pt")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pre-lr", type=float, default=0.3, help="lr multiplier for distilled T and F")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    split = Path(a.split)
    train, val = load(split / "train.jsonl"), load(split / "val.jsonl")
    tests = {p.stem: load(p) for p in sorted(split.glob("test_*.jsonl"))}
    noise = [(A.asr_noise, 1.0), (A.typos, 1.0), (A.fillers, 0.5)]
    nrng = random.Random(123)
    tests["test_noise"] = [A.augment(s, nrng, noise) for s in tests["test_indist"]]

    model = Dink(width=a.width)
    if a.init == "distilled":
        model.load_distilled(a.ckpt)
    pre = list(model.text.parameters()) + list(model.fuse.parameters())
    pre_ids = {id(p) for p in pre}
    rest = [p for p in model.parameters() if id(p) not in pre_ids]
    mult = a.pre_lr if a.init == "distilled" else 1.0
    opt = torch.optim.AdamW([{"params": pre, "lr": a.lr * mult}, {"params": rest, "lr": a.lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr * mult, a.lr], total_steps=a.steps, pct_start=0.05)
    n_params = sum(p.numel() for p in model.parameters())
    name = a.name or f"dink_{a.init}"
    out = Path("runs/dink") / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {n_params / 1e3:.0f}K parameters, init {a.init}, {len(train)} train / {len(val)} val", flush=True)
    history = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        batch = [A.augment(rng.choice(train), rng) for _ in range(a.batch)]
        x = collate(batch)
        logp = F.log_softmax(model(x), -1)
        loss = -(x["target"] * logp).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step in (1, 100) or step % a.eval_every == 0:
            v = evaluate(model, val)
            history.append({"step": step, "loss": loss.item(), "val_acc": v["acc"]})
            print(f"step {step:5d}  loss {loss.item():.3f}  val acc {v['acc']:.3f}  "
                  f"nothing P/R {v['nothing_precision']:.2f}/{v['nothing_recall']:.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)
    torch.save({"state": model.state_dict(), "args": vars(a)}, out / "model.pt")
    results = {"val": evaluate(model, val)}
    for k, items in tests.items():
        results[k] = evaluate(model, items)
    print(f"\n{'set':16s} {'acc':>6s} {'p(right)':>9s} {'nothing P/R':>12s}")
    for k, r in results.items():
        print(f"{k:16s} {r['acc']:6.3f} {r['p_right']:9.3f} {r['nothing_precision']:6.2f}/{r['nothing_recall']:.2f}")
    (out / "metrics.json").write_text(json.dumps({"args": vars(a), "params": n_params, "history": history,
                                                  "results": results}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
