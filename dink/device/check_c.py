"""Check the C engine (esp32/components/dink, built for the PC) against this repo:
number extraction against dink.data.numbers, dink_score against the NumPy reference with
the int8 trunk, and dink_decide against dink.harness with the PyTorch model.

  esp32/host/build.sh
  uv run python -m dink.device.check_c esp32/host/libdink.so models/dink-v13
"""
import argparse
import ctypes as C
import json
import random
from pathlib import Path

import numpy as np

from ..data.numbers import extract
from ..data.scenario import Scenario
from . import reference as R
from .check import args_of
from .export import Trunk, calibration_texts, load

DIM = 64


class Name(C.Structure):
    _fields_ = [("vec", C.c_float * DIM), ("q", C.c_float * DIM), ("k", C.c_float * DIM)]


class Prompt(C.Structure):
    _fields_ = [("tok", (C.c_float * DIM) * 4), ("num_val", C.c_float * 3), ("num_level", C.c_uint8 * 3),
                ("n_num", C.c_uint8)]


class Item(C.Structure):
    _fields_ = [("name", C.POINTER(Name)), ("value", C.c_float), ("lo", C.c_float), ("hi", C.c_float),
                ("arg", C.c_uint8)]


class FieldS(C.Structure):
    _fields_ = [("name", C.c_char_p), ("value", C.c_float), ("lo", C.c_float), ("hi", C.c_float)]


class Action(C.Structure):
    _fields_ = [("name", C.c_char_p), ("lo", C.c_float), ("hi", C.c_float), ("has_value", C.c_uint8),
                ("sets", C.c_char_p)]


class Gadget(C.Structure):
    _fields_ = [("fields", C.POINTER(FieldS)), ("n_fields", C.c_int), ("actions", C.POINTER(Action)),
                ("n_actions", C.c_int), ("field_names", C.POINTER(Name)), ("action_names", C.POINTER(Name)),
                ("action_rel", C.c_void_p)]


class Decision(C.Structure):
    _fields_ = [("action", C.c_int), ("value", C.c_float), ("p", C.c_float), ("unsure", C.c_uint8),
                ("passes", C.c_uint8)]


class Engine:
    def __init__(self, lib, dnk):
        self.lib = C.CDLL(str(lib))
        libc = C.CDLL(None)
        libc.malloc.restype, libc.malloc.argtypes = C.c_void_p, [C.c_size_t]
        libc.free.argtypes = [C.c_void_p]
        L = self.lib
        L.dink_open.restype = C.c_void_p
        L.dink_open.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p]
        for f in ("dink_encode_name", "dink_encode_prompt", "dink_score", "dink_decide", "dink_gadget_prepare"):
            getattr(L, f).argtypes = None
        self.image = C.create_string_buffer(Path(dnk).read_bytes())
        self.h = L.dink_open(self.image, len(self.image) - 1, C.cast(libc.malloc, C.c_void_p),
                             C.cast(libc.free, C.c_void_p))
        assert self.h, "dink_open failed"
        self.names = {}

    def name(self, s):
        if s not in self.names:
            n = Name()
            assert self.lib.dink_encode_name(C.c_void_p(self.h), s.encode(), C.byref(n)) == 0
            self.names[s] = n
        return self.names[s]

    def numbers(self, prompt):
        v, lv = (C.c_float * 3)(), (C.c_uint8 * 3)()
        n = self.lib.dink_numbers(prompt.encode(), v, lv, 3)
        return [(float(v[i]), bool(lv[i])) for i in range(n)]

    def score(self, state, prompt, tools):
        p = Prompt()
        assert self.lib.dink_encode_prompt(C.c_void_p(self.h), prompt.encode(), C.byref(p)) == 0
        st = (Item * 8)(*[Item(C.pointer(self.name(n)), v, lo, hi, 1) for n, v, lo, hi in state])
        tl = (Item * 8)(*[Item(C.pointer(self.name(n)), v, lo, hi, int(a)) for n, v, lo, hi, a in tools])
        out = (C.c_float * 9)()
        assert self.lib.dink_score(C.c_void_p(self.h), st, len(state), C.byref(p), tl, len(tools), out) == 0
        return np.array(out[:len(tools) + 1])

    def gadget(self, g):
        keep = []
        enc = lambda s: keep.append(s.encode()) or keep[-1]
        fields = (FieldS * len(g["state"]))(*[FieldS(enc(f["name"]), f["value"], f["lo"], f["hi"]) for f in g["state"]])
        acts = (Action * len(g["actions"]))(*[
            Action(enc(a["name"]), a["lo"] or 0, a["hi"] or 0, a["lo"] is not None,
                   enc(a["sets"]) if a.get("sets") else None) for a in g["actions"]])
        gs = Gadget(fields, len(g["state"]), acts, len(g["actions"]))
        gs._keep = (keep, fields, acts)
        assert self.lib.dink_gadget_prepare(C.c_void_p(self.h), C.byref(gs)) == 0
        return gs

    def decide(self, gs, request):
        d = Decision()
        assert self.lib.dink_decide(C.c_void_p(self.h), C.byref(gs), request.encode(), C.byref(d)) == 0
        return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lib")
    ap.add_argument("run")
    ap.add_argument("--split", default="data/home/split_v13")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    eng = Engine(a.lib, Path(a.run) / "dink.dnk")
    t = load(a.run)
    ref = R.Reference(t, Trunk(t, calibration_texts()))

    # 1. numbers, on every prompt we have
    prompts = set()
    for f in sorted(Path(a.split).glob("test_*.jsonl")):
        prompts.update(json.loads(l)["prompt"] for l in f.read_text().splitlines())
    for f in sorted(Path("examples/gadgets").glob("*.json")):
        prompts.update(u["say"] for u in json.loads(f.read_text())["utterances"])
    prompts |= {"set it to seventy two", "one hundred and five degrees", "a hundred percent", "half way",
                "turn it to max", "i'm full", "set to 1e5", "-3 degrees", "twenty five point", "50% please"}
    bad = [(p, extract(p), eng.numbers(p)) for p in sorted(prompts)
           if [(round(v, 4), k == "level") for v, k in extract(p)] != [(round(v, 4), k) for v, k in eng.numbers(p)]]
    print(f"numbers: {len(prompts) - len(bad)}/{len(prompts)} prompts agree")
    for b in bad[:5]:
        print("   ", b)

    # 2. the network, against the reference (int8 trunk)
    worst, n, same = 0.0, 0, 0
    for f in sorted(Path(a.split).glob("test_*.jsonl")):
        lines = f.read_text().splitlines()
        for line in random.Random(0).sample(lines, min(a.n // 5 + 1, len(lines))):
            s = Scenario.from_json(line)
            st, p, tools, nums = args_of(s)
            pc, pr = eng.score(st, p, tools), ref.score(st, p, tools, nums)
            worst = max(worst, float(np.abs(pc - pr).max()))
            same += pc.argmax() == pr.argmax()
            n += 1
    print(f"dink_score: {n} scenarios, same pick {same}/{n}, largest probability difference {worst:.2e}")

    # 3. the harness, against dink.harness with the PyTorch model
    from ..harness import decide
    from ..play import load_model
    model = load_model(a.run, "cpu")
    agree = total = 0
    for f in sorted(Path("examples/gadgets").glob("*.json")):
        g = json.loads(f.read_text())
        gs = eng.gadget(g)
        for u in g["utterances"]:
            py, c = decide(model, g, u["say"]), eng.decide(gs, u["say"])
            pa = None if py["pick"] is None else py["pick"].name
            ca = None if c.action < 0 else g["actions"][c.action]["name"]
            ok = pa == ca and (pa is None or not py["pick"].arg or abs(py["pick"].value - c.value) < 1e-3)
            agree += ok
            total += 1
            if not ok:
                pv = "" if py["pick"] is None or not py["pick"].arg else f"={py['pick'].value:g}"
                cv = "" if ca is None or g["actions"][c.action]["lo"] is None else f"={c.value:g}"
                print(f"    {f.stem}: {u['say']!r}: python {pa}{pv}  C {ca}{cv}")
    print(f"dink_decide: {agree}/{total} requests give the same action and value as the PyTorch harness")


if __name__ == "__main__":
    main()
