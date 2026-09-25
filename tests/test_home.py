import random

import numpy as np
import pytest

from dink import augment as A
from dink.encode import batch
from dink.gen import EVAL_SETS, TRAIN, generate
from dink.home import HomeGen, Split
from dink.scenario import Scenario


def canon(tool_name):
    """Map any naming style back to the canonical state name (room + kind)."""
    kinds = {"lite": "lite", "lit": "lite", "l": "lite", "temp": "temp", "tem": "temp", "t": "temp",
             "fan": "fan", "f": "fan", "blnd": "blnd", "bln": "blnd", "b": "blnd"}
    for sep in ("_", "."):
        if sep in tool_name:
            a, b = tool_name.split(sep, 1)
            if a == "set":                                  # set_kitl
                return f"{b[:-1]}_{kinds[b[-1]]}"
            if a in kinds and len(a) == 4 or a == "fan":    # lite_kit
                return f"{b}_{kinds[a]}"
            return f"{a}_{kinds.get(b, b)}"
    return tool_name


@pytest.fixture(scope="module")
def train():
    return generate(TRAIN, 3000, 7)


def test_invariants_all_sets():
    for name, (split, recipe) in EVAL_SETS.items():
        for s in generate(split, 300, 11, recipe):
            s.check()
            lo, hi = split.tool_slots
            assert len(s.tools) <= hi, name


def test_relative_moves_the_right_way(train):
    checked = 0
    for s in train:
        if s.meta["cls"].startswith(("relative", "indirect")) and not s.meta["ambiguous"] \
                and s.label < len(s.tools):
            t = s.tools[s.label]
            cur = {f.name: f.value for f in s.state}[canon(t.name)]
            assert (t.value - cur) * s.meta["dir"] > 0, s.show()
            checked += 1
    assert checked > 300


def test_limit_and_none_cases_answer_nothing(train):
    for s in train:
        if s.meta["cls"] == "limit" or s.meta["cls"].startswith("none"):
            assert s.label == len(s.tools), s.show()


def test_class_mix(train):
    none = sum(s.label == len(s.tools) for s in train) / len(train)
    amb = sum(bool(s.meta["ambiguous"]) for s in train) / len(train)
    assert 0.1 < none < 0.35
    assert 0.02 < amb < 0.3


def test_augment_preserves_answer(train):
    rng = random.Random(3)
    def dist(x):
        d = {(t.name, t.value): p for t, p in zip(x.tools, x.target) if p > 0}
        d["nothing"] = x.target[-1]
        return d
    for s in train[:500]:
        assert dist(A.augment(s, rng)) == dist(s)


def test_heldout_sets_differ():
    rooms = generate(EVAL_SETS["h_rooms"][0], 100, 5)
    den = [s for s in rooms if s.meta["room"] == "den"]
    assert len(den) > 40
    assert all(any(f.name.startswith("den") for f in s.state + s.tools) and "den" in s.prompt for s in den)
    naming = generate(EVAL_SETS["h_naming"][0], 100, 5)
    assert all("." in t.name for s in naming for t in s.tools if t.arg and "_" not in t.name[:3])
    celsius = generate(EVAL_SETS["h_range"][0], 100, 5)
    temps = [f for s in celsius for f in s.state if f.name.endswith("_temp")]
    assert temps and all(f.lo == 16 and f.hi == 30 for f in temps)


def test_json_roundtrip_and_encode(train):
    s = train[0]
    assert Scenario.from_json(s.to_json()).to_json() == s.to_json()
    b = batch(train[:64])
    assert b["prompt"].shape == (64, 64) and b["tool_name"].shape == (64, 8, 8)
    assert np.allclose(b["target"].sum(1), 1)
