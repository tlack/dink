"""Text encoder T: bytes -> vectors. Used for the prompt (every query) and for state/tool
names (once per device setup, cached). See docs/architecture.md.

Layer types match BabyTalk's speech model (1x1 and depthwise 1D convolutions), so MMRT's
existing ESP32-S3 kernels can run it; widths are multiples of 16 for the vector unit.
"""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    """Depthwise conv (k taps along the text) + 1x1 conv, ReLU, residual."""

    def __init__(self, width, k):
        super().__init__()
        self.dw = nn.Conv1d(width, width, k, padding=k // 2, groups=width)
        self.pw = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + F.relu(self.pw(self.dw(x)))


class TextEncoder(nn.Module):
    def __init__(self, width=128, emb=32, out=64, max_len=64, blocks=2, k=5, queries=1):
        super().__init__()
        self.byte = nn.Embedding(256, emb)
        self.pos = nn.Parameter(torch.zeros(max_len, emb))
        self.inp = nn.Conv1d(emb, width, 1)
        self.blocks = nn.ModuleList(Block(width, k) for _ in range(blocks))
        self.query = nn.Parameter(torch.randn(queries, width) * width ** -0.5)
        self.key = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out)

    def forward(self, b):
        """b: [B, L] byte values (0 = padding). Returns [B, queries, out]."""
        mask = b > 0
        x = self.byte(b.long()) + self.pos[: b.shape[1]]
        x = self.inp(x.transpose(1, 2))
        for blk in self.blocks:
            x = blk(x)
        keys = self.key(x)                                              # [B, W, L]
        att = torch.einsum("qw,bwl->bql", self.query, keys) / x.shape[1] ** 0.5
        att = att.masked_fill(~mask[:, None, :], -1e4).softmax(-1)
        pooled = torch.einsum("bql,bwl->bqw", att, x)
        return self.out(self.norm(pooled))


def to_bytes(texts, max_len=64):
    """Texts -> [N, max_len] uint8 tensor (UTF-8, zero padded, longer texts keep their end)."""
    out = torch.zeros(len(texts), max_len, dtype=torch.uint8)
    for i, t in enumerate(texts):
        b = t.encode()[-max_len:]
        out[i, : len(b)] = torch.tensor(list(b), dtype=torch.uint8)
    return out
