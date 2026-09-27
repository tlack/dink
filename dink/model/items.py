"""Value encoder V and fuse F: (name vector, value, lo, hi) -> a value-aware item token.

V turns a value into features: 16 piecewise-linear bins of its normalized position in
[lo, hi] (Gorishniy et al. 2022) and a has-value flag, plus (raw=True, the first models only)
a signed log magnitude of the raw value. Dropped since 2026-09-27: it let the model
memorize absolute amounts; values now enter only as their 0..1 position in their range. F mixes
those with the name vector in a small MLP, so "thermostat at 83 of 60..85" can mean
"warm" before any later layer. See docs/architecture.md.
"""
import torch
from torch import nn

BINS = 16
N_FEATS = BINS + 2


def value_features(value, lo, hi, has=None, raw=True):
    """value, lo, hi: float tensors [...]; returns [..., 18] (raw) or [..., 17]."""
    span = (hi - lo).clamp(min=1e-9)
    level = ((value - lo) / span).clamp(0, 1)
    edges = torch.arange(BINS, dtype=value.dtype, device=value.device) / BINS
    ple = ((level[..., None] - edges) * BINS).clamp(0, 1)
    mag = torch.sign(value) * torch.log1p(value.abs()) / 8
    has = torch.ones_like(value) if has is None else has.float()
    if not raw:
        return torch.cat([ple * has[..., None], has[..., None]], -1)
    return torch.cat([ple * has[..., None], (mag * has)[..., None], has[..., None]], -1)


class Fuse(nn.Module):
    def __init__(self, dim=64, hidden=128, n_feats=N_FEATS):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + n_feats, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, name_vec, feats):
        return self.net(torch.cat([name_vec, feats], -1))
