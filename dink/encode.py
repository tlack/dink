"""Scenario -> the fixed-size arrays the network (and the device) sees."""
import math

import numpy as np

from .scenario import MAX_SLOTS, NAME_BYTES, PROMPT_BYTES, Scenario


def name_bytes(name: str) -> np.ndarray:
    b = name.encode()[:NAME_BYTES]
    out = np.zeros(NAME_BYTES, np.uint8)
    out[: len(b)] = np.frombuffer(b, np.uint8)
    return out


def prompt_bytes(prompt: str) -> np.ndarray:
    b = prompt.encode()[-PROMPT_BYTES:]                 # longer prompts keep their end
    out = np.zeros(PROMPT_BYTES, np.uint8)
    out[: len(b)] = np.frombuffer(b, np.uint8)
    return out


def magnitude(v: float) -> float:
    """Signed log magnitude of the raw value, for the rare cases absolute size matters."""
    return math.copysign(math.log1p(abs(v)), v) / 8.0


def encode(s: Scenario) -> dict:
    """Arrays (one example): names uint8, normalized values, masks, prompt, target."""
    n_s, n_t = len(s.state), len(s.tools)
    e = {
        "state_name": np.zeros((MAX_SLOTS, NAME_BYTES), np.uint8),
        "state_val": np.zeros((MAX_SLOTS, 2), np.float32),      # [normalized 0..1, magnitude]
        "state_mask": np.zeros(MAX_SLOTS, bool),
        "tool_name": np.zeros((MAX_SLOTS, NAME_BYTES), np.uint8),
        "tool_val": np.zeros((MAX_SLOTS, 3), np.float32),       # [normalized, magnitude, has-arg]
        "tool_mask": np.zeros(MAX_SLOTS, bool),
        "prompt": prompt_bytes(s.prompt),
        "target": np.zeros(MAX_SLOTS + 1, np.float32),         # 8 tool slots + nothing relevant
    }
    for i, f in enumerate(s.state):
        e["state_name"][i] = name_bytes(f.name)
        e["state_val"][i] = (f.norm(), magnitude(f.value))
        e["state_mask"][i] = True
    for i, f in enumerate(s.tools):
        e["tool_name"][i] = name_bytes(f.name)
        e["tool_val"][i] = (f.norm(), magnitude(f.value) if f.arg else 0.0, float(f.arg))
        e["tool_mask"][i] = True
    e["target"][:n_t] = s.target[:n_t]
    e["target"][MAX_SLOTS] = s.target[n_t]
    return e


def batch(scenarios) -> dict:
    encoded = [encode(s) for s in scenarios]
    return {k: np.stack([e[k] for e in encoded]) for k in encoded[0]}
