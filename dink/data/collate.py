"""Scenarios -> the tensors dink.model.dink.Dink takes."""
import torch

from ..model.dink import N_NUM, SLOTS
from ..model.text import to_bytes
from .numbers import extract

NAME_LEN = 16       # bytes per name the model reads (names are <= 8 bytes in the smart home)


def collate(scenarios, name_len=NAME_LEN):
    B = len(scenarios)
    z = lambda *shape, dt=torch.float32: torch.zeros(*shape, dtype=dt)
    x = {"state_name": torch.zeros(B, SLOTS, name_len, dtype=torch.uint8),
         "tool_name": torch.zeros(B, SLOTS, name_len, dtype=torch.uint8),
         "state_val": z(B, SLOTS), "state_lo": z(B, SLOTS), "state_hi": z(B, SLOTS).fill_(1),
         "state_level": z(B, SLOTS), "state_mask": z(B, SLOTS, dt=torch.bool),
         "tool_val": z(B, SLOTS), "tool_lo": z(B, SLOTS), "tool_hi": z(B, SLOTS).fill_(1),
         "tool_level": z(B, SLOTS), "tool_arg": z(B, SLOTS, dt=torch.bool), "tool_mask": z(B, SLOTS, dt=torch.bool),
         "num_val": z(B, N_NUM), "num_level_kind": z(B, N_NUM, dt=torch.bool), "num_mask": z(B, N_NUM, dt=torch.bool),
         "target": z(B, SLOTS + 1)}
    prompts = []
    for i, s in enumerate(scenarios):
        for j, f in enumerate(s.state[:SLOTS]):
            x["state_name"][i, j] = to_bytes([f.name], name_len)[0]
            x["state_val"][i, j], x["state_lo"][i, j], x["state_hi"][i, j] = f.value, f.lo, f.hi
            x["state_level"][i, j] = f.norm()
            x["state_mask"][i, j] = True
        for j, f in enumerate(s.tools[:SLOTS]):
            x["tool_name"][i, j] = to_bytes([f.name], name_len)[0]
            if f.arg:
                x["tool_val"][i, j], x["tool_lo"][i, j], x["tool_hi"][i, j] = f.value, f.lo, f.hi
                x["tool_level"][i, j] = f.norm()
            x["tool_arg"][i, j] = f.arg
            x["tool_mask"][i, j] = True
        for j, (v, kind) in enumerate(extract(s.prompt)[:N_NUM]):
            x["num_val"][i, j], x["num_level_kind"][i, j], x["num_mask"][i, j] = v, kind == "level", True
        n = len(s.tools)
        x["target"][i, :n] = torch.tensor(s.target[:n])
        x["target"][i, SLOTS] = s.target[n]
        prompts.append(s.prompt)
    x["prompt"] = to_bytes(prompts)
    return x
