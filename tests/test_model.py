import torch

from dink.data.collate import collate
from dink.data.numbers import extract
from dink.gen.__main__ import generate
from dink.model.dink import SLOTS, Dink


def test_extract_numbers():
    assert extract("set it to seventy two") == [(72.0, "raw")]
    assert extract("set the lights to 40 percent") == [(0.4, "level")]
    assert extract("bedroom to seventy and fan on two") == [(70.0, "raw"), (2.0, "raw")]
    assert extract("dim the lights to half") == [(0.5, "level")]
    assert extract("set to a hundred and five") == [(105.0, "raw")]
    assert extract("i am full") == [] and extract("turn on the lights") == []


def test_forward_and_masking():
    items = generate(16, 3)
    x = collate(items)
    out = Dink()(x)
    assert out.shape == (16, SLOTS + 1)
    for i, s in enumerate(items):                 # empty tool slots can never be chosen
        assert (out[i, len(s.tools):SLOTS] < -1e3).all()
    assert torch.allclose(x["target"].sum(1), torch.ones(16))
