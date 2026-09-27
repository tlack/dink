"""Prompt-only mode: which requests can be answered with no state at all, and a test set of them.

Absolute requests (switch on / off, set to a number, lock, run a scene, a no-value action),
negations and the "nothing relevant" kinds have the same right answer whatever the state is.
Relative requests, complaints, desires and limit requests don't: they need the current value.
Training (train.py --no-state) removes the whole state from some draws of the first group only;
test_nostate is the first group of the held-out tests with their state removed.
"""
import dataclasses

STATE_FREE = {
    # smart home
    "turn", "numeric", "lock", "scene", "none_offdomain", "none_question", "none_noise", "none_missing",
    # LLM worlds
    "w_direct", "w_number", "w_no_value", "w_question", "w_unrelated", "w_missing", "w_inv_negate",
}


def state_free(s):
    return s.meta.get("cls") in STATE_FREE or s.meta.get("intent") == "ha"


def strip_state(s):
    return dataclasses.replace(s, state=[])


def nostate_tests(tests, sets=("test_indist", "test_worlds", "test_areas")):
    """The state-free scenarios of the given test sets, with the state removed."""
    return [strip_state(s) for name in sets for s in tests.get(name, []) if state_free(s)]
