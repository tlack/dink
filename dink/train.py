"""Train Dink on a split (dink/data/split.py) and evaluate on every test set.

  uv run python -u -m dink.train --init distilled --ckpt runs/text/tv128/model.pt --name d_distilled
  uv run python -u -m dink.train --init scratch --name d_scratch
  uv run python -u -m dink.train --split data/home/split_v2 --batch 512 --steps 20000 --name big   # GPU

On a GPU, background workers augment and collate batches (the Python side is the
bottleneck). A resumable checkpoint is written every --ckpt-every steps (--resume picks it
up); the log line shows the GPU temperature, since long runs have overheated this laptop.

Each step draws base examples and augments them afresh (dink/gen/augment.py). Loss: cross
entropy against the (possibly split) target distribution over 8 tools + nothing relevant.

Options added for v14 experiments:
  --anchor W   keep the text encoder near the teacher while Dink trains: the distillation
               loss on the teacher corpus (dink/teacher/distill.py), weight W
  --match W    and the teacher's request-to-name rankings over LLM-world names
               (dink/teacher/matching.py; held-out areas and worlds never used), weight W
  --dir-credit F  options on the right action in the right direction get at least F of the
               best option's weight (dink/data/direction.py); requests with numbers excepted
  --thin P     in a share P of draws, one action keeps only some of its options, so how
               many options an action has says nothing about whether it's the answer
  --loops K    run the mixer up to K times (dink/model/dink.py Loop); each step trains a
               random count from 1 to K; --no-inject leaves out re-adding the mixer input
"""
import argparse
import collections
import dataclasses
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.collate import collate
from .data.direction import credit_direction, right_direction, thin_options
from .data.scenario import Scenario
from .evals.nostate import nostate_tests, state_free, strip_state
from .gen import augment as A
from .model.dink import SLOTS, Dink
from .model.text import to_bytes


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

    def __init__(self, train, batch, seed, scenario_drop=0.0, no_state=0.0, mix=None, thin=0.0):
        self.train, self.batch, self.seed, self.scenario_drop = train, batch, seed, scenario_drop
        self.thin = thin
        self.no_state = no_state
        self.groups, self.weights = None, None
        if mix:                                 # fixed share of each batch per source
            by = collections.defaultdict(list)
            for s in train:
                by[source_of(s)].append(s)
            self.groups = [by[k] for k in mix if by[k]]
            self.weights = [mix[k] for k in mix if by[k]]

    def draw(self, rng):
        pool = rng.choices(self.groups, self.weights)[0] if self.groups else self.train
        s = A.augment(rng.choice(pool), rng)
        if self.thin and rng.random() < self.thin:
            s = thin_options(s, rng)
        if self.no_state and s.state and state_free(s) and rng.random() < self.no_state:
            s = strip_state(s)                  # prompt-only: only where the answer can't depend on state
        if s.scenario and rng.random() < self.scenario_drop:     # the model must work without one too
            s = dataclasses.replace(s, scenario="")
        return s

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        while True:
            yield collate([self.draw(rng) for _ in range(self.batch)])


def rank_loss(logits, target, margin=1.0):
    """Pairwise ranking over graded targets: for every pair where option i is more right than
    option j, softplus(margin - (z_i - z_j)), weighted by how much more right. A near-miss
    between two acceptable answers costs little; ranking a wrong one high costs a lot."""
    diff = target[:, :, None] - target[:, None, :]              # [B, i, j] > 0: i better than j
    gap = logits[:, :, None] - logits[:, None, :]
    w = diff.clamp(min=0)
    return (w * F.softplus(margin - gap)).sum((1, 2)).div(w.sum((1, 2)).clamp(min=1e-9)).mean()


def source_of(s):
    """Data source, for --mix: the smart home generator, LLM worlds, or Home Assistant."""
    intent = s.meta.get("intent")
    return "ha" if intent == "ha" else "world" if intent == "world" else "home"


class Anchor:
    """Teacher targets for the text encoder during Dink training (--anchor, --match)."""

    def __init__(self, ckpt, dev, rng, match=False):
        from .teacher.distill import noisy
        from .teacher import matching as M
        self.dev, self.rng, self.noisy, self.spaced = dev, rng, noisy, M.spaced
        ck = torch.load(ckpt, weights_only=False)
        mu, W = ck["pca_mu"], ck["pca_W"]
        rows = [json.loads(l) for l in Path("data/teacher/corpus.jsonl").read_text().splitlines()]
        E = np.load("data/teacher/teacher.npy").astype(np.float32)
        keep = [k for k, r in enumerate(rows) if r["split"] == "train"]
        self.texts = [rows[k]["text"] for k in keep]
        Z = (E[keep] - mu) @ W
        self.Z = torch.from_numpy((Z / np.linalg.norm(Z, axis=1, keepdims=True)).astype(np.float32)).to(dev)
        self.E = torch.from_numpy(E[keep]).to(dev)
        self.groups = None
        if match:
            meta, ME = M.load()
            self.idx = {t: i for i, t in enumerate(meta["texts"])}
            self.ME = torch.from_numpy(ME).to(dev)
            self.groups = [g for g in meta["groups"] if g["split"] == "train"]

    def emb(self, text, texts):
        return F.normalize(text(to_bytes(texts).to(self.dev))[:, 0], dim=-1)

    def corpus_loss(self, text, n=256):
        """distill.py's loss: cosine to the teacher's PCA vector + similarity structure."""
        idx = self.rng.sample(range(len(self.texts)), n)
        s = self.emb(text, [self.noisy(self.texts[k], self.rng) for k in idx])
        e = self.E[idx]
        return (1 - (s * self.Z[idx]).sum(-1)).mean() + F.mse_loss(s @ s.T, e @ e.T)

    def match_loss(self, text, n_groups=32, per=8, temp=0.05):
        """For each request, the student's softmax over its world's names should match the
        teacher's (KL). Names go in as written (snake_case etc.); the teacher saw them spaced."""
        reqs, names, spans = [], [], []
        for g in self.rng.sample(self.groups, n_groups):
            rs = self.rng.sample(g["requests"], min(per, len(g["requests"])))
            spans.append((len(reqs), len(reqs) + len(rs), len(names), len(names) + len(g["names"])))
            reqs += [r["say"] for r in rs]
            names += g["names"]
        R, N = self.emb(text, [self.noisy(t, self.rng) for t in reqs]), self.emb(text, names)
        TR = self.ME[[self.idx[t] for t in reqs]]
        TN = self.ME[[self.idx[self.spaced(t)] for t in names]]
        loss = 0.0
        for r0, r1, n0, n1 in spans:
            t = F.softmax(TR[r0:r1] @ TN[n0:n1].T / temp, -1)
            loss = loss + F.kl_div(F.log_softmax(R[r0:r1] @ N[n0:n1].T / temp, -1), t, reduction="batchmean")
        return loss / len(spans)


def save_atomic(obj, path):
    """Write to a temporary file, then rename: a shutdown mid-write keeps the old checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


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
    ok = good = dirn = tp = fp = fn = 0
    per = collections.defaultdict(lambda: [0, 0])
    p_right = 0.0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = collate(chunk)
        prob = model(to(x, next(model.parameters()).device)).softmax(-1).cpu()
        pred = prob.argmax(-1)
        for s, p, pr, t in zip(chunk, pred.tolist(), prob, x["target"]):
            right = bool(t[p] >= t.max() - 1e-6)      # the best answer (or one of equally best)
            ok += right
            good += bool(t[p] > 0)                   # acceptable: not clearly wrong
            dirn += right_direction(s, p if p < SLOTS else len(s.tools))   # right action and way
            per[s.meta["cls"]][0] += right
            per[s.meta["cls"]][1] += 1
            p_right += float((pr * t).sum())
            said_nothing, is_n = p == SLOTS, is_nothing(s)
            tp += said_nothing and is_n
            fp += said_nothing and not is_n
            fn += (not said_nothing) and is_n
    n = len(items)
    return {"acc": ok / n, "ok": good / n, "dir": dirn / n, "p_right": p_right / n,
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
    ap.add_argument("--no-state", type=float, default=0.0,
                    help="share of state-free training draws with the whole state removed (dink/evals/nostate.py)")
    ap.add_argument("--extra-test", nargs="*", default=[], help="more test files, scored as test sets")
    ap.add_argument("--raw-magnitude", action="store_true",
                    help="also feed the raw value's log magnitude (the first models did; off since 2026-09-27)")
    ap.add_argument("--rank-loss", type=float, default=0.0,
                    help="weight of the pairwise ranking loss: more-right options above less-right ones")
    ap.add_argument("--mix", default=None, help='per-source batch shares, e.g. "home=0.4,world=0.45,ha=0.15"')
    ap.add_argument("--also", default=None,
                    help="another split (same holdouts) whose test sets are scored too, as old:<set>")
    ap.add_argument("--resume", action="store_true", help="continue from runs/dink/<name>/ckpt.pt")
    ap.add_argument("--gpu-hot", type=int, default=80, help="pause training at this GPU temperature (C)")
    ap.add_argument("--gpu-cool", type=int, default=70, help="...until it is back under this one")
    ap.add_argument("--anchor", type=float, default=0.0, help="weight of the teacher-corpus distillation loss")
    ap.add_argument("--match", type=float, default=0.0, help="weight of the request-to-name matching loss")
    ap.add_argument("--dir-credit", type=float, default=0.0,
                    help="min share of the best weight for same-action, same-direction options")
    ap.add_argument("--thin", type=float, default=0.0, help="share of draws with one action's options thinned")
    ap.add_argument("--loops", type=int, default=1, help="max mixer passes (1 = the plain mixer)")
    ap.add_argument("--no-inject", action="store_true", help="looped mixer without re-adding its input")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    split = Path(a.split)
    train, val = load(split / "train.jsonl"), load(split / "val.jsonl")
    if a.dir_credit:
        train = [credit_direction(s, a.dir_credit) for s in train]
    tests = {p.stem: load(p) for p in sorted(split.glob("test_*.jsonl"))}
    noise = [(A.asr_noise, 1.0), (A.typos, 1.0), (A.fillers, 0.5)]
    nrng = random.Random(123)
    tests["test_noise"] = [A.augment(s, nrng, noise) for s in tests["test_indist"]]
    tests["test_nostate"] = nostate_tests(tests)
    for p in a.extra_test:
        tests[Path(p).stem] = load(p)
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

    mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))} if a.mix else None
    if mix:
        have = collections.Counter(source_of(s) for s in train)
        print("mix:", {k: f"{v:.2f} of batches ({have[k]} rows)" for k, v in mix.items()}, flush=True)
    model = Dink(width=a.width, scenario=a.scenario, raw_magnitude=a.raw_magnitude, loops=a.loops,
                 inject=not a.no_inject)
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
    arng = random.Random(a.seed * 7919 + start)
    anchor = Anchor(a.ckpt, a.device, arng, match=a.match > 0) if (a.anchor or a.match) else None
    lrng = random.Random(a.seed * 31 + start)
    loader = iter(torch.utils.data.DataLoader(Batches(train, a.batch, a.seed + start,
                                                                      a.scenario_drop if a.scenario else 0.0,
                                                                      a.no_state, mix, a.thin), batch_size=None,
                                              num_workers=a.workers, pin_memory=a.device != "cpu",
                                              persistent_workers=a.workers > 0))
    t0 = t_last = time.time()
    for step in range(start, a.steps + 1):
        if a.device != "cpu" and step % 25 == 0:
            cool_down(a.gpu_hot, a.gpu_cool)
        model.train()
        model.loops = lrng.randint(1, a.loops)
        x = to(next(loader), a.device)
        model_logits = model(x)
        logp = F.log_softmax(model_logits, -1)
        loss = -(x["target"] * logp).sum(-1).mean()
        if a.rank_loss:
            loss = loss + a.rank_loss * rank_loss(model_logits, x["target"])
        if a.anchor:
            loss = loss + a.anchor * anchor.corpus_loss(model.text)
        if a.match:
            loss = loss + a.match * anchor.match_loss(model.text)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        model.loops = a.loops
        if step % 200 == 0:
            print(f"  step {step}: {200 / (time.time() - t_last):.2f} steps/s", flush=True)
            t_last = time.time()
        if step in (1, 100) or step % a.eval_every == 0:
            v = evaluate(model, val)
            history.append({"step": step, "loss": loss.item(), "val_acc": v["acc"]})
            print(f"step {step:5d}  loss {loss.item():.3f}  val acc {v['acc']:.3f}  "
                  f"nothing P/R {v['nothing_precision']:.2f}/{v['nothing_recall']:.2f}  ({time.time() - t0:.0f}s)"
                  + gpu_temp(), flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0:
            save_atomic({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "history": history, "step": step, "args": vars(a)}, out / "ckpt.pt")
    save_atomic({"state": model.state_dict(), "args": vars(a)}, out / "model.pt")
    model.loops = a.loops
    results = {"val": evaluate(model, val)}
    for k, items in tests.items():
        results[k] = evaluate(model, items)
    for k in range(1, a.loops):                  # the same weights at fewer passes
        model.loops = k
        for t in ("val", "test_areas", "test_worlds", "test_llm", "hard"):
            items = val if t == "val" else tests.get(t)
            if items:
                results[f"{t}@{k}"] = evaluate(model, items)
    model.loops = a.loops
    try:                                         # request-to-name matching (dink/teacher/matching.py)
        from .teacher import matching as M
        from .teacher.distill import Student
        meta, _ = M.load()
        stu = Student(model.text.cpu())
        for sp in ("areas", "worlds"):
            acc, n, tub, _ = M.accuracy(stu, [g for g in meta["groups"] if g["split"] == sp])
            results[f"match_{sp}"] = {"acc": acc, "n": n, "hot_tub": tub}
        model.text.to(a.device)
    except FileNotFoundError:
        pass
    print(f"\n{'set':16s} {'best':>6s} {'ok':>6s} {'dir':>6s} {'p(right)':>9s} {'nothing P/R':>12s}")
    for k, r in results.items():
        if k.startswith("match_"):
            print(f"{k:16s} {r['acc']:6.3f}  (hot tub {r['hot_tub']}/10)")
            continue
        print(f"{k:16s} {r['acc']:6.3f} {r['ok']:6.3f} {r['dir']:6.3f} {r['p_right']:9.3f} "
              f"{r['nothing_precision']:6.2f}/{r['nothing_recall']:.2f}")
    (out / "metrics.json").write_text(json.dumps({"args": vars(a), "params": n_params, "history": history,
                                                  "results": results}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
