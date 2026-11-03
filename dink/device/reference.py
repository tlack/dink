"""Dink's forward pass in NumPy, one query at a time: the spec for the C engine.

It reads the same tensors the device file holds (dink/device/export.py), in the same
order of operations as the C code, so a mismatch between C and PyTorch can be traced to a
step. The text encoder's conv trunk runs either in float or as MMRT's int8 arithmetic
(bit-exact with BabyTalk's mmrt/mmrt_ref.c), which is what the ESP32-S3 runs.

Folds that change nothing mathematically:
  - attention pooling: softmax(query . (K x + b)) = softmax((K^T query) . x), since the
    query . b term is the same at every position. So the 128x128 key conv becomes one
    128-vector per query.
  - the depthwise conv's bias moves into the next 1x1 conv's bias (no ReLU between them).
"""
import math

import numpy as np

SLOTS, N_NUM, PROMPT_TOKENS = 8, 3, 4
NAME_LEN, PROMPT_LEN = 40, 64
BINS = 16
_HALF = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.65, 1.0]
REL_EDGES = np.array([-e for e in reversed(_HALF)] + [0.0] + _HALF, np.float32)


def text_bytes(text, n):
    """UTF-8, zero padded; longer texts keep their end (as dink.model.text.to_bytes)."""
    b = text.encode()[-n:]
    return np.frombuffer(b + b"\0" * (n - len(b)), np.uint8)


# ---- MMRT int8 arithmetic (mmrt/mmrt_ref.c) ----

def rshift_rnd(v, s):
    v = v.astype(np.int64)
    return (v + (1 << (s - 1))) >> s if s > 0 else v * (1 << -s)


def sat8(v):
    return np.clip(v, -128, 127).astype(np.int8)


def mmrt_conv1x1(x, w, b, shift, relu):
    """x [T][C] int8, w [N][C] int8, b [N] int32 -> [T][N] int8; also the largest |acc|."""
    acc = x.astype(np.int32) @ w.astype(np.int32).T
    if b is not None:
        acc = acc + b
    v = rshift_rnd(acc, shift)
    if relu:
        v = np.maximum(v, 0)
    return sat8(v), int(np.abs(acc).max())


def mmrt_dwconv(x, w, shift, pad):
    """x [T][C], w [C][K] -> [T][C] (stride 1, zero padding)."""
    T, C = x.shape
    K = w.shape[1]
    xp = np.zeros((T + 2 * pad, C), np.int32)
    xp[pad:pad + T] = x
    acc = sum(xp[k:k + T] * w[:, k].astype(np.int32) for k in range(K))
    return sat8(rshift_rnd(acc, shift)), int(np.abs(acc).max())


def mmrt_add(a, b, e_a, e_b, e_out):
    e0 = min(e_a, e_b)
    v = (a.astype(np.int64) << (e_a - e0)) + (b.astype(np.int64) << (e_b - e0))
    return sat8(rshift_rnd(v, e_out - e0))


def quantize(x, e):
    return sat8(np.floor(x / 2.0 ** e + 0.5))       # round half up, like the runtime


# ---- the network ----

class Reference:
    def __init__(self, t, trunk=None):
        """t: name -> float32 array (export.py's tensors). trunk: export.Trunk for int8, or
        None for the float trunk."""
        self.t, self.trunk = t, trunk
        self.max_acc = 0

    def trunk_float(self, b):
        t = self.t
        x = t["emb"][b] + t["pos"][: len(b)]
        x = x @ t["inp_w"].T + t["inp_b"]
        for i in range(2):
            T, C = x.shape
            xp = np.zeros((T + 4, C), np.float32)
            xp[2:2 + T] = x
            d = sum(xp[k:k + T] * t[f"dw{i}_w"][:, k] for k in range(5))
            x = x + np.maximum(d @ t[f"pw{i}_w"].T + t[f"pw{i}_b"], 0)
        return x

    def trunk_int8(self, b):
        t, q = self.t, self.trunk
        x0 = quantize(t["emb"][b] + t["pos"][: len(b)], q.e["x0"])
        x, a = mmrt_conv1x1(x0, q.w["inp"], q.b["inp"], q.shift["inp"], False)
        accs = [a]
        for i in range(2):
            d, a1 = mmrt_dwconv(x, q.w[f"dw{i}"], q.shift[f"dw{i}"], 2)
            p, a2 = mmrt_conv1x1(d, q.w[f"pw{i}"], q.b[f"pw{i}"], q.shift[f"pw{i}"], True)
            x = mmrt_add(x, p, q.e[f"x{i + 1}"], q.e[f"p{i}"], q.e[f"x{i + 2}"])
            accs += [a1, a2]
        self.max_acc = max(self.max_acc, *accs)
        return x.astype(np.float32) * 2.0 ** q.e["x3"]

    def text(self, b, queries):
        """b: byte array -> [queries][64]."""
        x = self.trunk_int8(b) if self.trunk else self.trunk_float(b)
        t = self.t
        s = t["pool_u"][:queries] @ x.T                               # [q][L]
        s = np.where(b[None, :] > 0, s, -1e4)
        s = np.exp(s - s.max(1, keepdims=True))
        att = s / s.sum(1, keepdims=True)
        pooled = att @ x
        mu = pooled.mean(1, keepdims=True)
        var = ((pooled - mu) ** 2).mean(1, keepdims=True)
        h = (pooled - mu) / np.sqrt(var + 1e-5) * t["tnorm_w"] + t["tnorm_b"]
        return h @ t["tout_w"].T + t["tout_b"]

    def name(self, name):
        return self.text(text_bytes(name, NAME_LEN), 1)[0]

    @staticmethod
    def level(v, lo, hi):
        return min(1.0, max(0.0, (v - lo) / (hi - lo))) if hi > lo else 0.0

    def fuse(self, vec, level, has):
        t = self.t
        ple = np.clip((level - np.arange(BINS) / BINS) * BINS, 0, 1) * has
        f = np.concatenate([vec, ple, [has]]).astype(np.float32)
        return np.maximum(f @ t["fuse0_w"].T + t["fuse0_b"], 0) @ t["fuse1_w"].T + t["fuse1_b"]

    def score(self, state, prompt, tools, numbers, names=None):
        """state: [(name, value, lo, hi)], tools: [(name, value, lo, hi, arg)],
        numbers: [(value, is_level)] (dink.data.numbers.extract) -> probabilities, tools + nothing."""
        t = self.t
        names = names if names is not None else {}
        vec = lambda n: names[n] if n in names else names.setdefault(n, self.name(n))
        sn = [vec(s[0]) for s in state]
        tn = [vec(x[0]) for x in tools]
        s_lvl = [self.level(v, lo, hi) for _, v, lo, hi in state]
        st = [self.fuse(sn[i], s_lvl[i], 1.0) + t["role"][0] for i in range(len(state))]
        toks_tool = []
        for j, (n, v, lo, hi, arg) in enumerate(tools):
            if not arg:
                v, lo, hi = 0.0, 0.0, 1.0
            lvl = self.level(v, lo, hi) if arg else 0.0
            tok = self.fuse(tn[j], lvl, 1.0 if arg else 0.0) + t["role"][1]
            # relation step
            if state:
                q = t["rel_q"] @ tn[j]
                a = np.array([q @ (t["rel_k"] @ s) for s in sn]) / 8.0
                a = np.exp(a - a.max())
                a /= a.sum()
                attended, match = float(a @ np.array(s_lvl)), float(a.max())
            else:
                attended, match = 0.0, 0.0
            delta = (lvl - attended) * (1.0 if arg else 0.0)
            lo_e, w_e = REL_EDGES[:-1], REL_EDGES[1:] - REL_EDGES[:-1]
            rel = np.concatenate([np.clip((delta - lo_e) / w_e, 0, 1), [abs(delta), match]])
            tok = tok + t["rel_out_w"] @ rel + t["rel_out_b"]
            # prompt numbers in this tool's range
            nf = np.zeros(N_NUM * 5, np.float32)
            if arg:
                for k, (nv, is_level) in enumerate(numbers[:N_NUM]):
                    l = nv if is_level else (nv - lo) / max(hi - lo, 1e-9)
                    inr = 1.0 if -1e-6 <= l <= 1 + 1e-6 else 0.0
                    dist = abs(lvl - min(1.0, max(0.0, l)))
                    nf[k * 5:k * 5 + 5] = [1.0, inr, min(2.0, max(-1.0, l)) / 2, dist, (1.0 if dist < 1e-3 else 0.0) * inr]
            tok = tok + t["num_out_w"] @ nf + t["num_out_b"]
            toks_tool.append(tok)
        pr = self.text(text_bytes(prompt, PROMPT_LEN), PROMPT_TOKENS) + t["role"][2]
        h = np.stack(st + list(pr) + toks_tool + [t["nothing"]]).astype(np.float32)
        for i in range(2):
            h = self.layer(h, i)
        h = self.ln(h, t["norm_w"], t["norm_b"])
        first = len(state) + PROMPT_TOKENS
        logits = h[first:] @ t["score_w"][0] + t["score_b"][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    @staticmethod
    def ln(x, w, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * w + b

    def layer(self, x, i):
        t, p = self.t, f"m{i}_"
        h = self.ln(x, t[p + "ln1_w"], t[p + "ln1_b"])
        qkv = h @ t[p + "in_w"].T + t[p + "in_b"]
        q, k, v = np.split(qkv, 3, axis=1)
        heads = []
        for hd in range(4):
            sl = slice(hd * 16, hd * 16 + 16)
            a = q[:, sl] @ k[:, sl].T / 4.0
            a = np.exp(a - a.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            heads.append(a @ v[:, sl])
        x = x + np.concatenate(heads, 1) @ t[p + "out_w"].T + t[p + "out_b"]
        h = self.ln(x, t[p + "ln2_w"], t[p + "ln2_b"])
        return x + np.maximum(h @ t[p + "ff1_w"].T + t[p + "ff1_b"], 0) @ t[p + "ff2_w"].T + t[p + "ff2_b"]
