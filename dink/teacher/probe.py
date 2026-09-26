"""Probes for a text embedder (the teacher, and later our distilled student).

1. held-out words: does "make the thermostat chillier" sit closer to the trained
   same-direction phrasings (colder, cooler) than to the opposite ones (warmer, hotter)?
   Covers held-out comparatives, device nouns and indirect phrases (configs/home_split.json).
2. values in words: "the light is at full brightness" closer to "it's bright" than "it's dark".
3. abstention: a request's best similarity to the offered tools' descriptions should be
   higher for device requests than for off-domain ones (AUC; arXiv 2609.18672 reports
   0.806 for multilingual-e5-base on its own 70-action task).

  uv run python -m dink.teacher.probe [--model intfloat/multilingual-e5-base]
"""
import argparse
import json

import numpy as np

from ..gen import home as H
from ..gen.__main__ import generate

HELD = json.load(open("configs/home_split.json"))


class SentenceEmbedder:
    """A sentence-transformers model; e5 models expect a "query: " prefix."""

    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name, device="cpu")
        self.prefix = "query: " if "e5" in name else ""

    def __call__(self, texts):
        return self.model.encode([self.prefix + t for t in texts], batch_size=128,
                                 normalize_embeddings=True, convert_to_numpy=True)


def probe_directions(embed):
    """Held-out word vs trained same-direction and opposite-direction phrasings."""
    held = set(HELD["words"]) | set(HELD["templates"])
    cases = []
    for key, k in H.KINDS.items():
        noun = k.nouns[0]
        for up in (True, False):
            same, other = (k.up, k.down) if up else (k.down, k.up)
            for w in same:
                if w in held:
                    cases.append((f"make the {noun} {w}",
                                  [f"make the {noun} {x}" for x in same if x not in held],
                                  [f"make the {noun} {x}" for x in other if x not in held]))
    for (key, up), items in H.INDIRECT.items():
        other_items = H.INDIRECT[(key, not up)]
        for text, _ in items:
            if text in held:
                strip = lambda t: t.replace("{loc}", "")
                cases.append((strip(text), [strip(t) for t, _ in items if t not in held],
                              [strip(t) for t, _ in other_items if t not in held]))
    for key, k in H.KINDS.items():                 # held-out nouns vs other kinds' nouns
        for n in k.nouns:
            if n in held:
                cmp = k.up[0]
                cases.append((f"make the {n} {cmp}", [f"make the {x} {cmp}" for x in k.nouns if x not in held],
                              [f"make the {o.nouns[0]} {cmp}" for kk, o in H.KINDS.items() if kk != key]))
    return score_triplets(embed, cases), len(cases)


VALUE_CASES = [
    ("the light is at full brightness", ["it's bright in here", "the lights are on"], ["it's dark in here", "the lights are off"]),
    ("the light is at zero", ["the lights are off", "it's dark in here"], ["the lights are on", "it's bright in here"]),
    ("the thermostat is set very low", ["it's cold in here", "it's chilly"], ["it's hot in here", "it's warm"]),
    ("the thermostat is set very high", ["it's hot in here", "it's warm"], ["it's cold in here", "it's chilly"]),
    ("the fan is at maximum speed", ["the fan is running fast", "strong breeze"], ["the fan is off", "no air moving"]),
    ("the fan is at zero", ["the fan is off", "no air moving"], ["the fan is running fast", "strong breeze"]),
    ("the blinds are fully open", ["sunlight coming in", "the window is uncovered"], ["the blinds are shut", "the room is covered"]),
    ("the blinds are fully closed", ["the blinds are shut", "the room is covered"], ["sunlight coming in", "the window is uncovered"]),
    ("the front door lock is at one", ["the door is locked", "secured"], ["the door is unlocked", "open to anyone"]),
    ("the garage door is at zero", ["the garage is closed", "shut"], ["the garage is open", "wide open"]),
]


def score_triplets(embed, cases):
    texts = sorted({t for a, p, n in cases for t in [a, *p, *n]})
    idx = {t: i for i, t in enumerate(texts)}
    E = embed(texts)
    ok = 0
    for a, p, n in cases:
        if not p or not n:
            continue
        sp = np.mean([E[idx[a]] @ E[idx[x]] for x in p])
        sn = np.mean([E[idx[a]] @ E[idx[x]] for x in n])
        ok += sp > sn
    return ok / max(1, sum(1 for _, p, n in cases if p and n))


def describe(name):
    """Plain words for a tool name, as a device's setup would provide them."""
    room, kind = H.split_name(name)
    if room:
        words = next(r.words[0] for r in H.ROOMS if r.code == room)
        return f"{words} {H.KINDS[kind].nouns[0]}"
    for key, h in H.HOUSE.items():
        if name in h["names"].values():
            return h["words"][0] + (" lock" if h["lock"] else "")
    return {"goodnite": "goodnight scene", "away": "away from home scene", "all_off": "turn everything off"}.get(name, name)


def probe_abstention(embed, n=1500):
    from sklearn.metrics import roc_auc_score
    items = [s for s in generate(n, 99) if s.meta["cls"] in ("none_offdomain", "none_noise")
             or not s.meta["cls"].startswith("none") and s.meta["cls"] != "limit"]
    scores, labels = [], []
    texts = sorted({s.prompt for s in items} | {describe(t.name) for s in items for t in s.tools})
    idx = {t: i for i, t in enumerate(texts)}
    E = embed(texts)
    for s in items:
        best = max(E[idx[s.prompt]] @ E[idx[describe(t.name)]] for t in s.tools)
        scores.append(best)
        labels.append(0 if s.meta["cls"].startswith("none") else 1)
    return roc_auc_score(labels, scores), sum(labels), len(labels) - sum(labels)


def run(embed):
    d, nd = probe_directions(embed)
    v = score_triplets(embed, VALUE_CASES)
    auc, n_in, n_out = probe_abstention(embed)
    print(f"  held-out word direction  {d:.2f}  ({nd} cases)")
    print(f"  values in words          {v:.2f}  ({len(VALUE_CASES)} cases)")
    print(f"  abstention AUC           {auc:.3f}  ({n_in} device requests vs {n_out} off-domain)")
    return {"directions": d, "values": v, "abstention_auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["intfloat/multilingual-e5-base"])
    a = ap.parse_args()
    for m in a.model:
        print(m)
        run(SentenceEmbedder(m))


if __name__ == "__main__":
    main()
