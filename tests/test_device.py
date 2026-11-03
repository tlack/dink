"""The device spec (dink/device/reference.py) against PyTorch, and the exported file's layout."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from dink.data.numbers import extract
from dink.device.export import DNK_HDR, load
from dink.device.reference import Reference
from dink.harness import fields, pass1_tools, score
from dink.play import load_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models" / "dink-v13"


@pytest.fixture(scope="module")
def models():
    return load_model(RUN, "cpu"), Reference(load(RUN))


def test_reference_matches_pytorch(models):
    model, ref = models
    g = json.loads((ROOT / "examples" / "gadgets" / "hot_tub.json").read_text())
    for u in g["utterances"][:10]:
        p = u["say"]
        st, tools = fields(g), pass1_tools(model, g, p)
        pt = np.array(score(model, st, p, tools))
        pr = ref.score([(s.name, s.value, s.lo, s.hi) for s in st], p,
                       [(x.name, x.value, x.lo, x.hi, x.arg) for x in tools],
                       [(v, k == "level") for v, k in extract(p)])
        assert np.abs(pt - pr).max() < 1e-4, p


def test_shipped_device_file():
    img = (RUN / "dink.dnk").read_bytes()
    h = DNK_HDR.unpack_from(img)
    assert h[0] == 0x314B4E44 and h[6] + h[7] == len(img)
    assert struct.unpack_from("<I", img, h[4])[0] == 0x54524D4D       # the MMRT trunk inside
