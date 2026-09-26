import json
import random

import numpy as np
import pytest

from dink.data.encode import batch
from dink.data.scenario import Scenario
from dink.data.split import AXES, split, touches
from dink.gen import augment as A
from dink.gen.__main__ import generate
from dink.gen.home import split_name

CFG = json.load(open("configs/home_split.json"))


@pytest.fixture(scope="module")
def pool():
    return generate(6000, 7)


def test_invariants(pool):
    for s in pool:
        s.check()
        assert set(s.meta["uses"]) >= {"words", "templates", "rooms", "kinds", "style", "units"}


def test_relative_moves_the_right_way(pool):
    checked = 0
    for s in pool:
        if s.meta["cls"].startswith(("relative", "indirect")) and not s.meta["ambiguous"] \
                and s.label < len(s.tools):
            t = s.tools[s.label]
            room, kind = split_name(t.name)
            cur = {f.name: f.value for f in s.state}[f"{room}_{kind}"]
            assert (t.value - cur) * s.meta["dir"] > 0, s.show()
            checked += 1
    assert checked > 500


def test_limit_and_none_cases_answer_nothing(pool):
    for s in pool:
        if s.meta["cls"] == "limit" or s.meta["cls"].startswith("none"):
            assert s.label == len(s.tools), s.show()


def test_class_mix(pool):
    none = sum(s.label == len(s.tools) for s in pool) / len(pool)
    amb = sum(bool(s.meta["ambiguous"]) for s in pool) / len(pool)
    assert 0.1 < none < 0.35 and 0.02 < amb < 0.3


def test_augment_preserves_answer(pool):
    rng = random.Random(3)

    def dist(x):
        d = {(t.name, t.value): p for t, p in zip(x.tools, x.target) if p > 0}
        d["nothing"] = x.target[-1]
        return d
    for s in pool[:800]:
        assert dist(A.augment(s, rng)) == dist(s)


def test_split_no_leaks_and_all_axes(pool):
    sets, held = split(pool, CFG, seed=0, n_train=1000)
    assert 200 < len(sets["train"]) <= 1000 and sets["val"] and sets["test_indist"]
    for axis in ("words", "templates", "rooms", "styles", "units", "slots"):
        assert len(sets.get("test_" + axis, [])) > 30, axis
    heldset = {a: set(v) for a, v in held.items()}
    for s in sets["train"] + sets["val"]:
        u = s.meta["uses"]
        assert not set(u["words"]) & heldset["words"]
        assert not set(u["templates"]) & heldset["templates"]
        assert "den" not in u["rooms"] and u["style"] != "dot" and u["units"] != "C"
        assert u["n_state"] <= 6 and u["n_tools"] <= 6


def test_sampled_holdouts_are_reproducible(pool):
    cfg = dict(CFG, sample={"words": 3, "kinds": 1})
    a = split(pool, cfg, seed=5)[1]
    b = split(pool, cfg, seed=5)[1]
    assert a == b and len(a["kinds"]) == 1
    sets = split(pool, cfg, seed=5)[0]
    assert all(a["kinds"][0] not in s.meta["uses"]["kinds"] for s in sets["train"])


def test_json_roundtrip_and_encode(pool):
    s = pool[0]
    assert Scenario.from_json(s.to_json()).to_json() == s.to_json()
    b = batch(pool[:64])
    assert b["prompt"].shape == (64, 64) and b["tool_name"].shape == (64, 8, 8)
    assert np.allclose(b["target"].sum(1), 1)


def test_bank_phrases_are_annotated_and_held_out():
    from dink.data.split import split
    from dink.gen.bank import phrase_id
    from dink.gen.home import GenConfig, HomeGen
    bank = {("scene", s): [f"please {s} now", "shut it all down for me"] for s in ("goodnite", "away", "all_off")}
    bank[("offdomain",)] = ["what's the capital of peru", "sing me a song"]
    g = HomeGen(GenConfig(bank_p=1.0), seed=0, bank=bank)
    items = [g.sample() for _ in range(3000)]
    used = [s for s in items if s.meta["uses"].get("llm")]
    assert used and all(s.meta["uses"]["llm"] == [phrase_id(s.prompt)] for s in used)
    cfg = {"words": ["shut it all down"], "llm_holdout": 0.5, "fractions": {"val": 0.1, "test_indist": 0.1}}
    sets, _ = split(items, cfg, seed=0)
    for s in sets["train"] + sets["val"]:
        assert "shut it all down" not in s.prompt
    assert sets.get("test_llm")
