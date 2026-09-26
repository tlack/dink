"""Distill the teacher into the text encoder T and the value fuse layer F.

Text: the same text goes to teacher and student; the student matches the teacher's vector
(PCA to 64 dims, cosine loss) and its similarity structure within each batch. Inputs get
light noise (typos, case) while targets stay clean. The corpus is data/teacher/corpus.jsonl
plus the value descriptions and number texts (values_corpus.py).

Items (--values): F(T(name), value features) must match the teacher's mean vector of the
LLM descriptions of that state ("thermostat at 83 of 60..85" -> "warm setting"), with
jittered values. Held-out device types (oven, battery) are never trained.

  uv run python -u -m dink.teacher.distill --width 128 --epochs 30 --values --name tv128
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..model.items import Fuse, value_features
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
    """Trained T (+ F) as probe-compatible embedders (normalized numpy vectors)."""

    def __init__(self, text, fuse=None):
        self.text, self.fuse = text.eval(), fuse.eval() if fuse is not None else None

    @torch.no_grad()
    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), 1024):
            out.append(F.normalize(self.text(to_bytes(texts[i:i + 1024]))[:, 0], dim=-1).numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def items(self, items):
        return F.normalize(fuse_items(self.text, self.fuse, items), dim=-1).numpy()


def fuse_items(text, fuse, items):
    names = text(to_bytes([i["name"] for i in items]))[:, 0]
    t = lambda k: torch.tensor([float(i[k]) for i in items])
    return fuse(names, value_features(t("value"), t("lo"), t("hi")))


def pca(E, k):
    mu = E.mean(0)
    _, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    return mu, Vt[:k].T, (S[:k] ** 2).sum() / (S ** 2).sum()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def evaluate(model, texts, E, rng, n=4000):
    S = Student(model)(texts)
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
    ap.add_argument("--item-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rel", type=float, default=1.0, help="weight of the similarity-structure loss")
    ap.add_argument("--values", action="store_true", help="add value texts and train the fuse layer")
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    rng = random.Random(0)

    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text().splitlines()]
    texts = [r["text"] for r in rows]
    E = np.load(DATA / "teacher.npy").astype(np.float32)
    tr = [k for k, r in enumerate(rows) if r["split"] == "train"]
    ev = [k for k, r in enumerate(rows) if r["split"] == "eval"]
    items, item_targets = [], None
    if a.values:
        vt = json.loads((DATA / "values_texts.json").read_text())
        VE = np.load(DATA / "values_teacher.npy").astype(np.float32)
        vtexts = vt["descriptions"] + vt["numbers"]
        pos = {t: len(texts) + i for i, t in enumerate(vtexts)}
        tr += list(range(len(texts), len(texts) + len(vtexts)))
        texts += vtexts
        E = np.concatenate([E, VE])
        items = [json.loads(l) for l in (DATA / "values_items.jsonl").read_text().splitlines()]
        items = [i for i in items if not i["held_out"]]
        item_targets = np.stack([E[[pos[t] for t in i["texts"]]].mean(0) for i in items])
    mu, W, var = pca(E[tr], a.out_dim)
    proj = lambda X: (lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True))((X - mu) @ W)
    Zt = torch.from_numpy(proj(E).astype(np.float32))
    Et = torch.from_numpy(E)
    print(f"{len(tr)} train / {len(ev)} eval texts, {len(items)} items; "
          f"PCA to {a.out_dim} dims keeps {100 * var:.0f}% of variance")

    text = TextEncoder(width=a.width, out=a.out_dim, blocks=a.blocks)
    fuse = Fuse(dim=a.out_dim) if a.values else None
    params = list(text.parameters()) + (list(fuse.parameters()) if fuse else [])
    n_params = sum(p.numel() for p in params)
    print(f"student: width {a.width}, {a.blocks} blocks{', + fuse' if fuse else ''}: {n_params / 1e3:.0f}K parameters")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    if items:
        IT_z = torch.from_numpy(proj(item_targets).astype(np.float32))
        IT_e = torch.from_numpy(item_targets / np.linalg.norm(item_targets, axis=1, keepdims=True))
    ev_texts, ev_E = [texts[k] for k in ev], E[ev]
    name = a.name or f"t{a.width}{'v' if a.values else ''}"
    out = Path("runs/text") / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"step 0: eval similarity agreement (Spearman) {evaluate(text, ev_texts, ev_E, rng):.3f}", flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        text.train()
        if fuse:
            fuse.train()
        order = tr[:]
        rng.shuffle(order)
        tot = tot_items = 0.0
        for b0 in range(0, len(order) - a.batch + 1, a.batch):
            idx = order[b0:b0 + a.batch]
            s = F.normalize(text(to_bytes([noisy(texts[k], rng) for k in idx]))[:, 0], dim=-1)
            e = Et[idx]
            loss = (1 - (s * Zt[idx]).sum(-1)).mean() + a.rel * F.mse_loss(s @ s.T, e @ e.T)
            if items:
                ib = rng.sample(range(len(items)), a.item_batch)
                batch_items = [dict(items[k], name=noisy(items[k]["name"], rng)) for k in ib]
                f = F.normalize(fuse_items(text, fuse, batch_items), dim=-1)
                ie = IT_e[ib]
                l_items = (1 - (f * IT_z[ib]).sum(-1)).mean() + a.rel * F.mse_loss(f @ f.T, ie @ ie.T)
                loss = loss + l_items
                tot_items += l_items.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == a.epochs - 1:
            nb = len(order) // a.batch
            print(f"epoch {ep + 1:3d}  loss {tot / nb:.4f}{f'  (items {tot_items / nb:.4f})' if items else ''}  "
                  f"eval Spearman {evaluate(text, ev_texts, ev_E, rng):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"text": text.state_dict(), "fuse": fuse.state_dict() if fuse else None, "args": vars(a),
                "pca_mu": mu, "pca_W": W}, out / "model.pt")
    print("probes (student):")
    stu = Student(text, fuse)
    res = probe.run(stu, stu.items if fuse else None)
    res.update(eval_spearman=float(evaluate(text, ev_texts, ev_E, rng)), params=n_params)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
