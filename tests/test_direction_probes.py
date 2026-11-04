"""The direction-probe scorer: "a little" is judged by rank among the options offered."""
import json

from dink.data.scenario import Field
from dink.evals.direction import CASES, judge
from dink.llm.probes import check


def fan(v, say, d, amount, action="set_fan_level"):
    return {"state": [{"name": "fan_level", "value": v, "lo": 1, "hi": 3}],
            "actions": [{"name": "set_fan_level", "lo": 1, "hi": 3, "sets": "fan_level"},
                        {"name": "fan_down", "lo": 1, "hi": 3, "sets": "fan_level"},
                        {"name": "sleep_mode", "lo": None, "hi": None, "sets": None}],
            "say": say, "action": action, "dir": d, "amount": amount, "number": None}


def out(pick, offered):
    return {"pick": pick, "pass1": [(Field("set_fan_level", v, 1, 3), 0.1) for v in offered], "pass2": None}


def F(v, name="set_fan_level"):
    return Field(name, v, 1, 3)


def test_a_little_from_the_top_is_the_next_step_not_the_bottom():
    c = fan(3, "turn the fan down a little", "down", "small")
    assert judge(c, out(F(2), [2, 1]))[:2] == (True, True)
    assert judge(c, out(F(1), [2, 1]))[:2] == (True, False)      # right way, too far


def test_a_little_at_the_bottom_nothing_is_fine():
    c = fan(1, "turn the fan down a little", "down", "small")
    assert judge(c, out(None, [2, 3]))[:2] == (True, True)
    assert judge(c, out(F(2), [2, 3]))[3] == "way"


def test_only_option_that_way_is_fine_for_any_amount():
    c = fan(2, "a little more fan", "up", "small")
    assert judge(c, out(F(3), [3, 1]))[:2] == (True, True)


def test_end_and_big():
    assert judge(fan(1, "max fan", "up", "end"), out(F(2), [2, 3]))[:2] == (True, False)
    assert judge(fan(1, "max fan", "up", "end"), out(F(3), [2, 3]))[:2] == (True, True)
    assert judge(fan(1, "way more fan", "up", "big"), out(F(2), [2, 3]))[:2] == (True, False)


def test_another_action_on_the_same_field_counts():
    c = fan(3, "turn the fan down", "down", "any")
    assert judge(c, out(F(2, "fan_down"), [2]))[:2] == (True, True)


def test_questions_want_nothing():
    c = fan(2, "is the fan on", "none", "any", action="none")
    assert judge(c, out(None, [1, 3]))[:2] == (True, True)
    assert judge(c, out(F(3), [1, 3]))[3] == "acted"


def test_kept_cases_are_well_formed():
    rows = [json.loads(l) for l in CASES.read_text().splitlines()]
    assert len(rows) >= 150 and not [r["say"] for r in rows if check(r)]
