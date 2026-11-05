"""Request-to-name matching: what the teacher knows that the student lost.

Dink has to tie a request to the right name among a gadget's actions and fields ("more jets"
-> set_jet_speed, not increase_sanitizer). The teacher does this well; the distilled student
keeps topics but weights verbs over nouns. This module builds matching groups for
distillation and measures matching accuracy.

Groups come from the LLM worlds (data/llm/worlds.jsonl): for each world, its requests (with
rewordings and flips) and its candidate names (actions and state fields). Targets are the
teacher's similarities, not the LLM labels, so the signal carries over to any names.
Held out exactly as Dink's split (configs/home_split_v4.json): the 5 test_areas areas and
the test_worlds worlds are evaluation only, so Dink's test sets stay clean.

  uv run python -m dink.teacher.matching --embed        # data/teacher/match.json + match_teacher.npy
  uv run python -m dink.teacher.matching --eval runs/text/tv128/model.pt models/dink-v16/model.pt
"""
import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np

DATA = Path("data/teacher")
WORLDS = Path("data/llm/worlds.jsonl")
TEACHER = "Qwen/Qwen3-Embedding-0.6B"
SKIP = {"unrelated", "missing"}         # no right name in the world: the ranking means nothing
SPLIT = json.loads(Path("configs/home_split_v4.json").read_text())

# Today's misses on the example hot tub, plus their fixed twins: a quick, readable check.
HOT_TUB = (["set_temperature", "set_ph", "increase_sanitizer", "set_jet_speed", "turn_on_ozone"], [
    ("more jets", "set_jet_speed"), ("crank the jets up", "set_jet_speed"), ("less jets", "set_jet_speed"),
    ("faster jets", "set_jet_speed"), ("more heat", "set_temperature"), ("warmer water", "set_temperature"),
    ("the water is way too hot", "set_temperature"), ("raise the ph", "set_ph"),
    ("more sanitizer", "increase_sanitizer"), ("turn on the ozone", "turn_on_ozone")])


def spaced(name):
    """set_jet_speed / setJetSpeed / Set Jet Speed -> 'set jet speed' (what the teacher sees)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ").replace("-", " ").lower().strip()


def split_of(w):
    """'areas' / 'worlds' (held out, as in dink/data/split.py) or 'train'."""
    if w["area"] in SPLIT["areas"]:
        return "areas"
    if zlib.crc32(w["setting"].encode()) % 1000 < SPLIT["world_holdout"] * 1000:
        return "worlds"
    return "train"


def build():
    worlds = [json.loads(l) for l in WORLDS.read_text().splitlines()]
    held = sorted(SPLIT["areas"])
    groups = []
    for w in worlds:
        actions = [a["name"] for a in w["actions"]]
        names = list(dict.fromkeys(actions + [f["name"] for f in w["state"]]))
        reqs = []
        for u in w["utterances"]:
            if u["kind"] in SKIP or not u.get("say") or len(u["say"].encode()) > 64:
                continue
            label = u["action"] if u.get("agree") and u.get("action") in actions else None
            reqs.append({"say": u["say"], "label": label, "kind": u["kind"]})
        if reqs and len(names) >= 3:
            groups.append({"setting": w["setting"], "area": w["area"], "split": split_of(w),
                           "actions": actions, "names": names, "requests": reqs})
    return groups, sorted(held)


def texts_of(groups):
    """Every text the teacher embeds: requests as said, names spaced."""
    out = {}
    for g in groups:
        for r in g["requests"]:
            out.setdefault(r["say"], len(out))
        for n in g["names"]:
            out.setdefault(spaced(n), len(out))
    return out


def embed(texts, batch=128):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(TEACHER, device="cuda")
    return st.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True).astype(np.float16)


def load():
    meta = json.loads((DATA / "match.json").read_text())
    return meta, np.load(DATA / "match_teacher.npy").astype(np.float32)


def accuracy(embedder, groups, style=spaced):
    """Top-1: the request's nearest action name is its labeled action. embedder(texts) ->
    normalized vectors. Also the hot tub check."""
    hits = n = 0
    for g in groups:
        rs = [r for r in g["requests"] if r["label"]]
        if not rs:
            continue
        R = embedder([r["say"] for r in rs])
        A = embedder([style(a) for a in g["actions"]])
        pick = (R @ A.T).argmax(1)
        hits += sum(g["actions"][p] == r["label"] for p, r in zip(pick, rs))
        n += len(rs)
    acts, cases = HOT_TUB
    R, A = embedder([c[0] for c in cases]), embedder([style(a) for a in acts])
    tub = [acts[p] == c[1] for p, c in zip((R @ A.T).argmax(1), cases)]
    return hits / max(n, 1), n, sum(tub), [c[0] for c, ok in zip(cases, tub) if not ok]


class Teacher:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer(TEACHER, device="cuda")

    def __call__(self, texts):
        return self.st.encode(texts, batch_size=128, normalize_embeddings=True)


def student(ckpt):
    """A text encoder from a distill checkpoint (runs/text/*/model.pt) or a Dink model.pt."""
    import torch
    from .distill import Student
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    if "text" not in ck:                                  # a whole Dink model
        from ..play import load_model
        return Student(load_model(Path(ckpt).parent, "cpu").text)
    from ..model.text import TextEncoder
    a = ck["args"]
    t = TextEncoder(width=a["width"], out=a["out_dim"], blocks=a["blocks"])
    t.load_state_dict(ck["text"])
    return Student(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed", action="store_true", help="build the groups and embed them with the teacher")
    ap.add_argument("--eval", nargs="*", default=None, help="checkpoints to score ('teacher' for the teacher)")
    a = ap.parse_args()
    if a.embed:
        groups, held = build()
        idx = texts_of(groups)
        E = embed(list(idx))
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "match.json").write_text(json.dumps({"held_areas": held, "texts": list(idx), "groups": groups}))
        np.save(DATA / "match_teacher.npy", E)
        n = {k: sum(g["split"] == k for g in groups) for k in ("train", "areas", "worlds")}
        print(f"{len(groups)} worlds ({n}; held-out areas: {', '.join(held)}), "
              f"{sum(len(g['requests']) for g in groups)} requests, {len(idx)} texts")
    if a.eval is not None:
        meta, _ = load()
        by = {k: [g for g in meta["groups"] if g["split"] == k] for k in ("areas", "worlds", "train")}
        for c in a.eval:
            emb = Teacher() if c == "teacher" else student(c)
            res = {k: accuracy(emb, gs) for k, gs in by.items()}
            _, _, tub, miss = res["areas"]
            print(f"{c:36} " + "  ".join(f"{k} {r[0]:.3f} ({r[1]})" for k, r in res.items()) +
                  f"  hot tub {tub}/10  misses: {', '.join(miss) or '-'}", flush=True)


if __name__ == "__main__":
    main()
