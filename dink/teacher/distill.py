"""Distill the teacher into the tiny text encoder.

The same text goes to both. Targets: the teacher's vector reduced to 64 dims by PCA
(cosine loss), and the teacher's similarity structure within each batch (MSE between
the student's and the teacher's cosine-similarity matrices). Inputs get light noise
(typos, case, dropped letters) while targets stay clean, so the student also learns to
read messy text.

  uv run python -m dink.teacher.distill --width 128 --epochs 30 --name t128
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..model.text import TextEncoder, to_bytes
from . import probe

DATA = Path("data/teacher")


def noisy(t, rng, p=0.35):
    """Light, meaning-preserving noise on a training text."""
    if rng.random() > p:
        return t
    s = list(t)
    for i in range(len(s) - 1, -1, -1):
        r = rng.random()
        if s[i] != " " and r < 0.03:
            del s[i]
        elif r < 0.05 and i + 1 < len(s):
            s[i], s[i + 1] = s[i + 1], s[i]
    t = "".join(s) or t
    if rng.random() < 0.15:
        t = t.upper() if rng.random() < 0.3 else t.capitalize()
    return t


class Student:
    """The trained encoder as a probe-compatible embedder (normalized numpy vectors)."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), 1024):
            v = self.model(to_bytes(texts[i:i + 1024]))[:, 0]
            out.append(F.normalize(v, dim=-1).numpy())
        return np.concatenate(out)


def pca(E, k):
    mu = E.mean(0)
    _, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    var = (S[:k] ** 2).sum() / (S ** 2).sum()
    return mu, Vt[:k].T, var


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def evaluate(model, texts, E, rng, n=4000):
    stu = Student(model)
    S = stu(texts)
    i = np.array([rng.randrange(len(texts)) for _ in range(n)])
    j = np.array([rng.randrange(len(texts)) for _ in range(n)])
    return spearman((S[i] * S[j]).sum(1), (E[i] * E[j]).sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rel", type=float, default=1.0, help="weight of the similarity-structure loss")
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    rng = random.Random(0)

    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text().splitlines()]
    E = np.load(DATA / "teacher.npy").astype(np.float32)
    tr = [k for k, r in enumerate(rows) if r["split"] == "train"]
    ev = [k for k, r in enumerate(rows) if r["split"] == "eval"]
    mu, W, var = pca(E[tr], a.out_dim)
    Z = (E - mu) @ W
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    print(f"{len(tr)} train / {len(ev)} eval texts; PCA to {a.out_dim} dims keeps {100 * var:.0f}% of variance")

    model = TextEncoder(width=a.width, out=a.out_dim, blocks=a.blocks)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"student: width {a.width}, {a.blocks} blocks, {n_params / 1e3:.0f}K parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    texts = [r["text"] for r in rows]
    Et, Zt = torch.from_numpy(E), torch.from_numpy(Z.astype(np.float32))
    ev_texts, ev_E = [texts[k] for k in ev], E[ev]
    name = a.name or f"t{a.width}"
    out = Path("runs/text") / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"step 0: eval similarity agreement (Spearman) {evaluate(model, ev_texts, ev_E, rng):.3f}")
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        model.train()
        order = tr[:]
        rng.shuffle(order)
        tot = 0.0
        for b0 in range(0, len(order) - a.batch + 1, a.batch):
            idx = order[b0:b0 + a.batch]
            x = to_bytes([noisy(texts[k], rng) for k in idx])
            s = F.normalize(model(x)[:, 0], dim=-1)
            z, e = Zt[idx], Et[idx]
            l_vec = (1 - (s * z).sum(-1)).mean()
            l_rel = F.mse_loss(s @ s.T, e @ e.T)
            loss = l_vec + a.rel * l_rel
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
        if ep % 5 == 4 or ep == a.epochs - 1:
            sp = evaluate(model, ev_texts, ev_E, rng)
            print(f"epoch {ep + 1:3d}  loss {tot / (len(order) // a.batch):.4f}  "
                  f"eval Spearman {sp:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "args": vars(a), "pca_mu": mu, "pca_W": W}, out / "model.pt")
    print("probes (student):")
    res = probe.run(Student(model))
    res.update(eval_spearman=float(evaluate(model, ev_texts, ev_E, rng)), params=n_params)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
