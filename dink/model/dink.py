"""The Dink network: state + prompt + tools -> one probability per tool + "nothing relevant".
See docs/architecture.md for the diagram.

  text encoder T   names (1 vector each; cached on a device) and the prompt (4 tokens)
  value path V+F   (name vector, value, lo, hi) -> value-aware item tokens
  relation step    each tool finds its state field (attention over name vectors) and gets
                   the difference between its value and that field's, as features
  prompt numbers   numbers pulled from the prompt, compared with each tool in the tool's
                   own range: in range?, level, distance to the candidate value, exact match
  mixer            2 pre-norm attention blocks over 8 state + 4 prompt + 8 tool + 1 nothing
  scorer           one logit per tool token and for the nothing token; empty slots masked
"""
import torch
from torch import nn
import torch.nn.functional as F

from .items import Fuse, value_features
from .text import TextEncoder

SLOTS = 8
N_NUM = 3
PROMPT_TOKENS = 4
REL_BINS = 16


def ple_signed(x, bins=REL_BINS):
    """Piecewise-linear encoding of x in [-1, 1] (as (x + 1) / 2 in [0, 1])."""
    level = ((x + 1) / 2).clamp(0, 1)
    edges = torch.arange(bins, dtype=x.dtype, device=x.device) / bins
    return ((level[..., None] - edges) * bins).clamp(0, 1)


class Dink(nn.Module):
    def __init__(self, width=128, dim=64, blocks=2, heads=4, ff=256, dropout=0.1):
        super().__init__()
        self.text = TextEncoder(width=width, out=dim, queries=PROMPT_TOKENS)
        self.fuse = Fuse(dim=dim)
        self.role = nn.Embedding(3, dim)                       # state, prompt, tool
        self.nothing = nn.Parameter(torch.randn(dim) * 0.02)
        self.rel_q = nn.Linear(dim, dim, bias=False)
        self.rel_k = nn.Linear(dim, dim, bias=False)
        self.rel_out = nn.Linear(REL_BINS + 2, dim)            # delta bins, |delta|, match strength
        self.num_out = nn.Linear(N_NUM * 5, dim)               # per number: present, in range, level, distance, exact
        layer = nn.TransformerEncoderLayer(dim, heads, ff, dropout, batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def load_distilled(self, path):
        """Start T and F from a distilled checkpoint (runs/text/<name>/model.pt). The distilled
        encoder has one pooling query; the prompt's four start from it plus a little noise."""
        ck = torch.load(path, weights_only=False)
        st = dict(ck["text"])
        q = st.pop("query")
        missing = self.text.load_state_dict(st, strict=False)
        assert set(missing.missing_keys) == {"query"}, missing
        with torch.no_grad():
            self.text.query.copy_(q.repeat(PROMPT_TOKENS, 1) + 0.01 * torch.randn_like(self.text.query))
        if ck.get("fuse"):
            self.fuse.load_state_dict(ck["fuse"])

    def names(self, b):
        """[N, 8, L] name bytes -> [N, 8, dim] (one vector per name: pooling query 0)."""
        n, s, l = b.shape
        return self.text(b.reshape(n * s, l))[:, 0].reshape(n, s, -1)

    def forward(self, x):
        state_name, tool_name = self.names(x["state_name"]), self.names(x["tool_name"])
        sv = value_features(x["state_val"], x["state_lo"], x["state_hi"])
        tv = value_features(x["tool_val"], x["tool_lo"], x["tool_hi"], x["tool_arg"])
        state = self.fuse(state_name, sv) + self.role.weight[0]
        tool = self.fuse(tool_name, tv) + self.role.weight[1]

        # relation step: each tool attends over the state fields by name
        att = torch.einsum("btd,bsd->bts", self.rel_q(tool_name), self.rel_k(state_name)) / tool_name.shape[-1] ** 0.5
        att = att.masked_fill(~x["state_mask"][:, None, :], -1e4).softmax(-1)
        state_level = x["state_level"]                                  # [B, 8] normalized 0..1
        attended = torch.einsum("bts,bs->bt", att, state_level)
        delta = (x["tool_level"] - attended) * x["tool_arg"].float()
        match = att.max(-1).values * x["state_mask"].any(-1, keepdim=True).float()
        tool = tool + self.rel_out(torch.cat([ple_signed(delta), delta.abs()[..., None], match[..., None]], -1))

        # prompt numbers, each read in each tool's own range
        nv, nk, npres = x["num_val"], x["num_level_kind"], x["num_mask"].float()   # [B, 3]
        lo, hi = x["tool_lo"][..., None], x["tool_hi"][..., None]                   # [B, 8, 1]
        span = (hi - lo).clamp(min=1e-9)
        lvl = torch.where(nk[:, None, :], nv[:, None, :], (nv[:, None, :] - lo) / span)   # [B, 8, 3]
        in_range = ((lvl >= -1e-6) & (lvl <= 1 + 1e-6)).float()
        dist = (x["tool_level"][..., None] - lvl.clamp(0, 1)).abs()
        exact = (dist < 1e-3).float() * in_range
        feats = torch.stack([npres[:, None, :].expand_as(lvl), in_range, lvl.clamp(-1, 2) / 2, dist, exact], -1)
        feats = feats * npres[:, None, :, None] * x["tool_arg"][..., None, None].float()
        tool = tool + self.num_out(feats.flatten(-2))

        prompt = self.text(x["prompt"]) + self.role.weight[2]                     # [B, 4, dim]
        B = prompt.shape[0]
        tokens = torch.cat([state, prompt, tool, self.nothing.expand(B, 1, -1)], 1)
        pad = torch.cat([~x["state_mask"], torch.zeros(B, PROMPT_TOKENS, dtype=torch.bool, device=prompt.device),
                         ~x["tool_mask"], torch.zeros(B, 1, dtype=torch.bool, device=prompt.device)], 1)
        h = self.norm(self.mixer(tokens, src_key_padding_mask=pad))
        logits = self.score(h[:, SLOTS + PROMPT_TOKENS:]).squeeze(-1)             # [B, 9]: 8 tools + nothing
        mask = torch.cat([x["tool_mask"], torch.ones(B, 1, dtype=torch.bool, device=prompt.device)], 1)
        return logits.masked_fill(~mask, -1e4)
