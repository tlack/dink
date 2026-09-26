"""Train Dink on a split (dink/data/split.py) and evaluate on every test set.

  uv run python -u -m dink.train --init distilled --ckpt runs/text/tv128/model.pt --name d_distilled
  uv run python -u -m dink.train --init scratch --name d_scratch
  uv run python -u -m dink.train --split data/home/split_v2 --batch 512 --steps 20000 --name big   # GPU

On a GPU, background workers augment and collate batches (the Python side is the
bottleneck). A resumable checkpoint is written every --ckpt-every steps (--resume picks it
up); the log line shows the GPU temperature, since long runs have overheated this laptop.

Each step draws base examples and augments them afresh (dink/gen/augment.py). Loss: cross
entropy against the (possibly split) target distribution over 8 tools + nothing relevant.
"""
import argparse
import collections
import dataclasses
import json
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data.collate import collate
from .data.scenario import Scenario
from .gen import augment as A
from .model.dink import SLOTS, Dink


def load(path):
    """Scenarios from a .jsonl file; records that fail Scenario.check (no tools, bad targets,
    values out of range, ...) are skipped and counted, never trained on."""
    out, bad = [], 0
    for line in Path(path).read_text().splitlines():
        if line.strip():
            s = Scenario.from_json(line)
            try:
                out.append(s.check())
            except AssertionError:
                bad += 1
    if bad:
        print(f"  {path}: skipped {bad} invalid records", flush=True)
    return out


def is_nothing(s):
    return s.label == len(s.tools)


class Batches(torch.utils.data.IterableDataset):
    """Endless augmented, collated batches; each worker has its own random stream."""

    def __init__(self, train, batch, seed, scenario_drop=0.0):
        self.train, self.batch, self.seed, self.scenario_drop = train, batch, seed, scenario_drop

    def draw(self, rng):
        s = A.augment(rng.choice(self.train), rng)
        if s.scenario and rng.random() < self.scenario_drop:     # the model must work without one too
            s = dataclasses.replace(s, scenario="")
        return s

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        while True:
            yield collate([self.draw(rng) for _ in range(self.batch)])


def to(x, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in x.items()}


def gpu_celsius():
    """GPU temperature, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_temp():
    c = gpu_celsius()
    return f"  gpu {c}C" if c is not None else ""


def cool_down(hot, cool):
    """This laptop's fan sometimes doesn't spin up, and hot runs have crashed it twice:
    above `hot` C, pause training until the GPU is back under `cool` C."""
    c = gpu_celsius()
    if c is None or c < hot:
        return
    t0 = time.time()
    print(f"  gpu {c}C >= {hot}C: pausing until {cool}C", flush=True)
    while c is not None and c > cool:
        time.sleep(5)
        c = gpu_celsius()
    print(f"  resumed after {time.time() - t0:.0f}s at {c}C", flush=True)


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
        prob = model(to(x, next(model.parameters()).device)).softmax(-1).cpu()
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
    ap.add_argument("--freeze", action="store_true",
                    help="keep distilled T and F fixed (the prompt's pooling queries still train)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4, help="batch-building processes (0 = in the main process)")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--scenario", action="store_true",
                    help="give the model each gadget's scenario description (dink/gen/scenario_text.py)")
    ap.add_argument("--scenario-drop", type=float, default=0.3, help="share of training draws without it")
    ap.add_argument("--also", default=None,
                    help="another split (same holdouts) whose test sets are scored too, as old:<set>")
    ap.add_argument("--resume", action="store_true", help="continue from runs/dink/<name>/ckpt.pt")
    ap.add_argument("--gpu-hot", type=int, default=80, help="pause training at this GPU temperature (C)")
    ap.add_argument("--gpu-cool", type=int, default=70, help="...until it is back under this one")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    split = Path(a.split)
    train, val = load(split / "train.jsonl"), load(split / "val.jsonl")
    tests = {p.stem: load(p) for p in sorted(split.glob("test_*.jsonl"))}
    noise = [(A.asr_noise, 1.0), (A.typos, 1.0), (A.fillers, 0.5)]
    nrng = random.Random(123)
    tests["test_noise"] = [A.augment(s, nrng, noise) for s in tests["test_indist"]]
    from .evals.hard import cases
    tests["hard"] = cases()
    if a.also:
        tests |= {"old:" + p.stem: load(p) for p in sorted(Path(a.also).glob("test_*.jsonl"))}
    if a.scenario:
        from .gen.scenario_text import ScenarioText
        st, srng = ScenarioText(), random.Random(7)
        for items in [train, val] + list(tests.values()):
            for s in items:
                st.fill(s, srng)

    model = Dink(width=a.width, scenario=a.scenario)
    if a.init == "distilled":
        model.load_distilled(a.ckpt)
    model.to(a.device)
    pre = [p for p in list(model.text.parameters()) + list(model.fuse.parameters()) if p is not model.text.query]
    pre_ids = {id(p) for p in pre}
    rest = [p for p in model.parameters() if id(p) not in pre_ids]
    if a.freeze:
        for p in pre:
            p.requires_grad_(False)
        opt = torch.optim.AdamW(rest, lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    else:
        mult = a.pre_lr if a.init == "distilled" else 1.0
        opt = torch.optim.AdamW([{"params": pre, "lr": a.lr * mult}, {"params": rest, "lr": a.lr}],
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr * mult, a.lr], total_steps=a.steps, pct_start=0.05)
    n_params = sum(p.numel() for p in model.parameters())
    name = a.name or f"dink_{a.init}"
    out = Path("runs/dink") / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {n_params / 1e3:.0f}K parameters, init {a.init}, {len(train)} train / {len(val)} val", flush=True)
    history, start = [], 1
    if a.resume and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, start = ck["history"], ck["step"] + 1
        print(f"resumed at step {start}", flush=True)
    loader = iter(torch.utils.data.DataLoader(Batches(train, a.batch, a.seed + start,
                                                                      a.scenario_drop if a.scenario else 0.0), batch_size=None,
                                              num_workers=a.workers, pin_memory=a.device != "cpu",
                                              persistent_workers=a.workers > 0))
    t0 = time.time()
    for step in range(start, a.steps + 1):
        if a.device != "cpu" and step % 25 == 0:
            cool_down(a.gpu_hot, a.gpu_cool)
        model.train()
        x = to(next(loader), a.device)
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
                  f"nothing P/R {v['nothing_precision']:.2f}/{v['nothing_recall']:.2f}  ({time.time() - t0:.0f}s)"
                  + gpu_temp(), flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0:
            torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "history": history, "step": step, "args": vars(a)}, out / "ckpt.pt")
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
