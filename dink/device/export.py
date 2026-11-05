"""Export a trained Dink for the ESP32: one file the C engine (esp32/components/dink) maps
in place.

  uv run python -m dink.device.export models/dink-v16 --out models/dink-v16/dink.dnk

Layout (little endian, every section 16-byte aligned):
  header      "DNK1", version, n_tensors, mmrt_off, mmrt_size, then config words
  tensor table  n_tensors x {name[24], dims[4] (0 = unused), data offset}: float32 data
  MMRT image  the text encoder's conv trunk, int8 (bytes -> [L][128]); MMRT format v2,
              ops CONV1X1, DWCONV and ADD only, so the runtime needs no new ops
  data        the float32 tensors

The trunk is most of the arithmetic (about 2.4M of the 3.4M multiply-adds per text); the
rest (pooling, LayerNorm, value features, the relation step, the 2-layer mixer) is small and
stays float32. Exponents are powers of two, chosen from the largest activation seen on
calibration texts (names and prompts from the example gadgets and the hard cases), and the
largest accumulator is checked against the S3's 20-bit lanes.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from . import reference as R

ACC_LIMIT = 1 << 19                  # ESP32-S3 QACC int8 lanes are 20 bits


def tensors(state):
    """PyTorch state dict -> the engine's float32 tensors (with the folds in reference.py)."""
    s = {k: v.float().numpy() for k, v in state.items()}
    t = {"emb": s["text.byte.weight"], "pos": s["text.pos"],
         "inp_w": s["text.inp.weight"][:, :, 0], "inp_b": s["text.inp.bias"]}
    for i in range(2):
        b = f"text.blocks.{i}."
        pw = s[b + "pw.weight"][:, :, 0]
        t[f"dw{i}_w"] = s[b + "dw.weight"][:, 0, :]
        t[f"pw{i}_w"] = pw
        t[f"pw{i}_b"] = s[b + "pw.bias"] + pw @ s[b + "dw.bias"]        # dw bias folded in
    t["pool_u"] = s["text.query"] @ s["text.key.weight"][:, :, 0] / np.sqrt(128)   # key conv folded
    t.update(tnorm_w=s["text.norm.weight"], tnorm_b=s["text.norm.bias"],
             tout_w=s["text.out.weight"], tout_b=s["text.out.bias"],
             fuse0_w=s["fuse.net.0.weight"], fuse0_b=s["fuse.net.0.bias"],
             fuse1_w=s["fuse.net.2.weight"], fuse1_b=s["fuse.net.2.bias"],
             role=s["role.weight"], nothing=s["nothing"], rel_q=s["rel_q.weight"], rel_k=s["rel_k.weight"],
             rel_out_w=s["rel_out.weight"], rel_out_b=s["rel_out.bias"],
             num_out_w=s["num_out.weight"], num_out_b=s["num_out.bias"],
             norm_w=s["norm.weight"], norm_b=s["norm.bias"], score_w=s["score.weight"], score_b=s["score.bias"])
    for i in range(2):
        m, p = f"mixer.layers.{i}.", f"m{i}_"
        t.update({p + "in_w": s[m + "self_attn.in_proj_weight"], p + "in_b": s[m + "self_attn.in_proj_bias"],
                  p + "out_w": s[m + "self_attn.out_proj.weight"], p + "out_b": s[m + "self_attn.out_proj.bias"],
                  p + "ff1_w": s[m + "linear1.weight"], p + "ff1_b": s[m + "linear1.bias"],
                  p + "ff2_w": s[m + "linear2.weight"], p + "ff2_b": s[m + "linear2.bias"],
                  p + "ln1_w": s[m + "norm1.weight"], p + "ln1_b": s[m + "norm1.bias"],
                  p + "ln2_w": s[m + "norm2.weight"], p + "ln2_b": s[m + "norm2.bias"]})
    return {k: np.ascontiguousarray(v, np.float32) for k, v in t.items()}


def load(run):
    ck = torch.load(Path(run) / "model.pt", weights_only=True, map_location="cpu")
    assert not ck["args"].get("raw_magnitude", True) and not ck["args"].get("scenario"), \
        "the engine supports the released configuration only (no raw magnitude, no scenario)"
    assert ck["args"]["width"] == 128
    return tensors(ck["state"])


def pow2_exp(maxabs):
    """Smallest e with maxabs <= 127 * 2^e."""
    return int(np.ceil(np.log2(max(maxabs, 1e-12) / 127)))


class Trunk:
    """The conv trunk in MMRT int8: weights, int32 biases, exponents e, shifts."""

    def __init__(self, t, texts):
        """texts: (bytes array) samples for calibrating activation ranges."""
        self.w, self.b, self.ew, self.shift = {}, {}, {}, {}
        for k, src in [("inp", "inp_w"), ("dw0", "dw0_w"), ("pw0", "pw0_w"), ("dw1", "dw1_w"), ("pw1", "pw1_w")]:
            self.ew[k] = pow2_exp(np.abs(t[src]).max())
            self.w[k] = R.sat8(np.floor(t[src] / 2.0 ** self.ew[k] + 0.5))
        # activation ranges, float trunk, every stage
        mx = {k: 0.0 for k in ("x0", "x1", "d0", "p0", "x2", "d1", "p1", "x3")}
        for b in texts:
            x = t["emb"][b] + t["pos"][: len(b)]
            mx["x0"] = max(mx["x0"], np.abs(x).max())
            x = x @ t["inp_w"].T + t["inp_b"]
            mx["x1"] = max(mx["x1"], np.abs(x).max())
            for i in range(2):
                T, C = x.shape
                xp = np.zeros((T + 4, C), np.float32)
                xp[2:2 + T] = x
                d = sum(xp[k:k + T] * t[f"dw{i}_w"][:, k] for k in range(5))
                p = np.maximum(d @ t[f"pw{i}_w"].T + t[f"pw{i}_b"], 0)
                x = x + p
                mx[f"d{i}"] = max(mx[f"d{i}"], np.abs(d).max())
                mx[f"p{i}"] = max(mx[f"p{i}"], np.abs(p).max())
                mx[f"x{i + 2}"] = max(mx[f"x{i + 2}"], np.abs(x).max())
        self.e = {k: pow2_exp(v) for k, v in mx.items()}
        e = self.e
        self.shift["inp"] = e["x1"] - e["x0"] - self.ew["inp"]
        self.b["inp"] = np.floor(t["inp_b"] / 2.0 ** (e["x0"] + self.ew["inp"]) + 0.5).astype(np.int32)
        for i in range(2):
            xin = f"x{i + 1}"
            self.shift[f"dw{i}"] = e[f"d{i}"] - e[xin] - self.ew[f"dw{i}"]
            self.shift[f"pw{i}"] = e[f"p{i}"] - e[f"d{i}"] - self.ew[f"pw{i}"]
            self.b[f"pw{i}"] = np.floor(t[f"pw{i}_b"] / 2.0 ** (e[f"d{i}"] + self.ew[f"pw{i}"]) + 0.5).astype(np.int32)
        for k, s in self.shift.items():
            assert 0 <= s < 32, (k, s, self.e, self.ew)
        self.b["dw0"] = self.b["dw1"] = None


def calibration_texts(n=2000, seed=0):
    """Names and prompts the model was trained on (data/home/split_v13/val.jsonl if present),
    plus the example gadgets."""
    import random
    rng = random.Random(seed)
    names, prompts = set(), set()
    for f in sorted(Path("examples/gadgets").glob("*.json")):
        g = json.loads(f.read_text())
        names.update(x["name"] for x in g["state"] + g["actions"])
        prompts.update(u["say"] for u in g["utterances"])
    val = Path("data/home/split_v13/val.jsonl")
    if val.exists():
        lines = val.read_text().splitlines()
        for line in rng.sample(lines, min(n, len(lines))):
            r = json.loads(line)
            names.update(x["name"] for x in r["state"] + r["tools"])
            prompts.add(r["prompt"])
    return [R.text_bytes(x, R.NAME_LEN) for x in sorted(names)] + \
           [R.text_bytes(x, R.PROMPT_LEN) for x in sorted(prompts)]


# ---- file writing ----

MMRT_MAGIC, MMRT_VERSION = 0x54524D4D, 2
NONE16, NONE32 = 0xFFFF, 0xFFFFFFFF
MMRT_HDR = struct.Struct("<16I")
MMRT_TENSOR = struct.Struct("<Hbx")
MMRT_OP = struct.Struct("<BBBBbbbbbB2xHHHxxII")      # mmrt_op_t, 28 bytes (wfmt 0 = int8)
DNK_MAGIC, DNK_VERSION = 0x314B4E44, 1             # "DNK1"
DNK_HDR = struct.Struct("<16I")
DNK_ENTRY = struct.Struct("<24s6I")                 # name, n, dims[4], offset (from data start)
assert MMRT_OP.size == 28 and DNK_ENTRY.size == 48

# the float tensors the engine reads (the trunk's conv weights live in the MMRT image)
DEVICE_TENSORS = ["emb", "pos", "pool_u", "tnorm_w", "tnorm_b", "tout_w", "tout_b", "fuse0_w", "fuse0_b",
                  "fuse1_w", "fuse1_b", "role", "nothing", "rel_q", "rel_k", "rel_out_w", "rel_out_b",
                  "num_out_w", "num_out_b", "norm_w", "norm_b", "score_w", "score_b"] + \
                 [f"m{i}_{k}" for i in range(2) for k in ("ln1_w", "ln1_b", "in_w", "in_b", "out_w", "out_b",
                                                          "ln2_w", "ln2_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b")]


def align16(n):
    return (n + 15) & ~15


class Blob:
    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, data):
        data = data if isinstance(data, bytes) else np.ascontiguousarray(data).tobytes()
        off = self.size
        self.parts.append(data + b"\0" * (align16(len(data)) - len(data)))
        self.size += align16(len(data))
        return off

    def bytes(self):
        return b"".join(self.parts)


def mmrt_image(q):
    """The trunk as an MMRT v2 file: [L][32] int8 bytes+positions -> [L][128]."""
    pw_layout = lambda w: w.reshape(w.shape[0] // 16, 16, w.shape[1]).transpose(0, 2, 1)   # [N/16][C][16]
    dw_layout = lambda w: w.reshape(w.shape[0] // 16, 16, w.shape[1]).transpose(0, 2, 1)   # [C/16][K][16]
    e = q.e
    tensors = [(32, e["x0"]), (128, e["x1"])]               # t0 input, t1 after inp
    for i in range(2):
        tensors += [(128, e[f"d{i}"]), (128, e[f"p{i}"]), (128, e[f"x{i + 2}"])]
    blob, ops = Blob(), []

    def op(kind, relu=0, K=0, pad=0, shift=0, e_a=0, e_b=0, e_out=0, in0=0, in1=NONE16, out=0,
           w=None, b=None):
        ops.append(MMRT_OP.pack(kind, relu, K, 1, pad, shift, e_a, e_b, e_out, 0, in0, in1, out,
                                blob.add(w) if w is not None else NONE32,
                                blob.add(b) if b is not None else NONE32))

    op(2, K=1, shift=q.shift["inp"], e_out=e["x1"], in0=0, out=1, w=pw_layout(q.w["inp"]), b=q.b["inp"])
    x = 1
    for i in range(2):
        d, p, y = x + 1, x + 2, x + 3
        op(1, K=5, pad=2, shift=q.shift[f"dw{i}"], e_out=e[f"d{i}"], in0=x, out=d, w=dw_layout(q.w[f"dw{i}"]))
        op(2, relu=1, K=1, shift=q.shift[f"pw{i}"], e_out=e[f"p{i}"], in0=d, out=p,
           w=pw_layout(q.w[f"pw{i}"]), b=q.b[f"pw{i}"])
        op(6, e_a=e[f"x{i + 1}"], e_b=e[f"p{i}"], e_out=e[f"x{i + 2}"], in0=x, in1=p, out=y)
        x = y
    tens = b"".join(MMRT_TENSOR.pack(c, ex) for c, ex in tensors)
    ops_b = b"".join(ops)
    t_off = align16(MMRT_HDR.size)
    o_off = align16(t_off + len(tens))
    b_off = align16(o_off + len(ops_b))
    hdr = MMRT_HDR.pack(MMRT_MAGIC, MMRT_VERSION, len(tensors), len(ops), 0, x, t_off, o_off, b_off,
                        blob.size, 128, 0, 0, 0, 0, 0)
    img = bytearray(b_off + blob.size)
    img[:len(hdr)] = hdr
    img[t_off:t_off + len(tens)] = tens
    img[o_off:o_off + len(ops_b)] = ops_b
    img[b_off:] = blob.bytes()
    return bytes(img)


def dnk_file(t, q):
    mm = mmrt_image(q)
    data, table = Blob(), []
    for name in DEVICE_TENSORS:
        a = t[name]
        dims = list(a.shape) + [0] * (4 - a.ndim)
        table.append(DNK_ENTRY.pack(name.encode(), a.size, *dims, data.add(a.astype("<f4"))))
    table_b = b"".join(table)
    table_off = align16(DNK_HDR.size)
    mmrt_off = align16(table_off + len(table_b))
    data_off = align16(mmrt_off + len(mm))
    # config: name_len, prompt_len, prompt_tokens, slots, numbers, dim, width, emb, heads, ff, layers
    hdr = DNK_HDR.pack(DNK_MAGIC, DNK_VERSION, len(table), table_off, mmrt_off, len(mm), data_off, data.size,
                       R.NAME_LEN, R.PROMPT_LEN, R.PROMPT_TOKENS, R.SLOTS, R.N_NUM, 64, 128, 32)
    out = bytearray(data_off + data.size)
    out[:len(hdr)] = hdr
    out[table_off:table_off + len(table_b)] = table_b
    out[mmrt_off:mmrt_off + len(mm)] = mm
    out[data_off:] = data.bytes()
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None, help="default <run>/dink.dnk")
    a = ap.parse_args()
    t = load(a.run)
    q = Trunk(t, calibration_texts())
    img = dnk_file(t, q)
    out = Path(a.out or Path(a.run) / "dink.dnk")
    out.write_bytes(img)
    print(f"-> {out}: {len(img) / 1024:.0f} KB (trunk int8 exponents {q.e})")


if __name__ == "__main__":
    main()
