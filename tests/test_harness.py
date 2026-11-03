from pathlib import Path

import pytest

from dink.gen.ordinal import nice
from dink.harness import decide, ladder
from dink.play import load_model

MODEL = Path(__file__).resolve().parents[1] / "models" / "dink-v13"

TUB = {
    "state": [{"name": "water_temperature", "value": 102, "lo": 80, "hi": 104},
              {"name": "jet_pump_speed", "value": 1, "lo": 0, "hi": 3}],
    "actions": [{"name": "set_temperature", "lo": 80, "hi": 104, "sets": "water_temperature"},
                {"name": "set_jet_speed", "lo": 0, "hi": 3, "sets": "jet_pump_speed"},
                {"name": "turn_on_ozone", "lo": None, "hi": None, "sets": None}],
}


def test_nice_reaches_range_ends():
    assert nice(8.2, 6.8, 8.2, False) == 8.2
    assert nice(97, 3, 97, True) == 97
    assert nice(3.2, 3, 97, True) == 3
    assert nice(7.7, 6.8, 8.2, False) == 7.7


def test_ladder_keeps_said_numbers_without_current_value():
    g = {"state": [], "actions": [{"name": "set_level", "lo": 0, "hi": 100, "sets": None}]}
    assert 37 in ladder(g, g["actions"][0], "set the level to 37")


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL, "cpu")


@pytest.mark.parametrize("prompt, name, value", [
    ("turn on the ozone", "turn_on_ozone", None),
    ("set jet speed to 2", "set_jet_speed", 2),
    ("i want it hotter", "set_temperature", 104),
    ("play some jazz", None, None),
])
def test_released_model(model, prompt, name, value):
    r = decide(model, TUB, prompt)
    if name is None:
        assert r["pick"] is None
    else:
        assert r["pick"].name == name
        if value is not None:
            assert r["pick"].value == value
