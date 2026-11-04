import random

from dink.data.direction import credit_direction, right_direction, same_direction, thin_options
from dink.data.scenario import Field, Scenario


def scen(prompt="warmer please", target=(1, 0, 0, 0, 0)):
    state = [Field("thermostat", 70, 60, 85), Field("fan", 1, 0, 3)]
    tools = [Field("thermostat", 74, 60, 85), Field("thermostat", 85, 60, 85),
             Field("thermostat", 64, 60, 85), Field("fan", 2, 0, 3)]
    return Scenario(state, prompt, tools, list(target), {"needed": ["thermostat"]}).check()


def test_same_direction():
    assert same_direction(scen()) == [0, 1]                 # 74 and 85 go up from 70; 64 goes down


def test_credit_direction():
    t = credit_direction(scen(), 0.25).target
    assert t[0] > t[1] > 0 and t[2] == t[3] == t[4] == 0 and abs(sum(t) - 1) < 1e-9
    assert credit_direction(scen("set it to 74"), 0.25).target == scen().target   # a number said: exact
    sized = scen("a little warmer")
    sized.meta["cls"] = "relative_small"
    assert credit_direction(sized, 0.25).target == sized.target                   # a size named: exact


def test_right_direction():
    s = scen()
    assert right_direction(s, 0) and right_direction(s, 1)
    assert not right_direction(s, 2) and not right_direction(s, 3) and not right_direction(s, 4)


def test_thin_options_keeps_the_answer():
    rng = random.Random(0)
    for _ in range(200):
        t = thin_options(scen(), rng).check()
        best = t.tools[t.label]
        assert (best.name, best.value) == ("thermostat", 74)
        assert {x.name for x in t.tools} == {"thermostat", "fan"}
