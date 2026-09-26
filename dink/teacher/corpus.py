"""Build the distillation corpus and embed it with the teacher.

Text only, no labels: the teacher's vector for each text is the target the student learns
to reproduce. Sources:
- public voice-assistant utterances: MASSIVE English (CC-BY-4.0), CLINC150
- the 20,000 most common English words (wordfreq)
- device vocabulary from the smart-home generator: names and verbalized values
  ("kitchen thermostat set low"), built only from training-split words
- prompts of the training split (data/home/split/train.jsonl), if present

Held-out words (configs/home_split.json) never appear in the text we write ourselves, so
test_words still measures knowledge the decision training never saw.

  uv run python -m dink.teacher.corpus                       # data/teacher/corpus.jsonl
  uv run python -m dink.teacher.corpus --embed               # + teacher vectors (slow, CPU)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from ..gen import home as H

OUT = Path("data/teacher")
TEACHER = "Qwen/Qwen3-Embedding-0.6B"
HELD = json.load(open("configs/home_split.json"))
HELD_WORDS = set(HELD["words"]) | set(HELD["templates"]) | {"den"}

LEVELS = {  # verbalized positions on each kind's range
    "lite": ["off", "at zero", "very dim", "dim", "at half brightness", "bright", "at full brightness"],
    "temp": ["set to the minimum", "set low", "set cool", "set to the middle", "set warm", "set high",
             "set to the maximum"],
    "fan": ["off", "on low", "on medium", "on high", "at full speed"],
    "blnd": ["fully closed", "mostly closed", "half open", "mostly open", "fully open"],
}
HOUSE_LEVELS = {True: ["locked", "unlocked"], False: ["open", "closed", "half open"]}


def device_texts():
    ok = lambda w: w not in HELD_WORDS
    out = set()
    rooms = [(r.code, [w for w in r.words if ok(w)]) for r in H.ROOMS if r.code != "den"]
    for key, k in H.KINDS.items():
        nouns = [n for n in k.nouns if ok(n)]
        for code, words in rooms:
            for w in words:
                for n in nouns:
                    out.add(f"{w} {n}")
                    for lv in LEVELS[key]:
                        out.add(f"the {w} {n} is {lv}")
        for n in nouns:
            for lv in LEVELS[key]:
                out.add(f"the {n} is {lv}")
            for cmp in [c for c in k.up + k.down if ok(c)]:
                out.add(f"make the {n} {cmp}")
    for h in H.HOUSE.values():
        for w in [w for w in h["words"] if ok(w)]:
            out.add(w)
            for lv in HOUSE_LEVELS[h["lock"]]:
                out.add(f"the {w} is {lv}")
    for phrases in H.SCENES.values():
        out |= {p for p in phrases if ok(p)}
    return sorted(out)


def public_texts():
    from datasets import load_dataset
    texts = []
    m = load_dataset("mteb/amazon_massive_intent", "en")
    for split in m.values():
        texts += split["text"]
    c = load_dataset("clinc/clinc_oos", "plus")
    for split in c.values():
        texts += split["text"]
    return texts


def build():
    from wordfreq import top_n_list
    parts = {"public": public_texts(), "words": top_n_list("en", 20000), "device": device_texts()}
    train = Path("data/home/split/train.jsonl")
    if train.exists():
        parts["prompts"] = [json.loads(l)["prompt"] for l in train.read_text().splitlines()]
    seen, rows = set(), []
    for src, texts in parts.items():
        for t in texts:
            t = " ".join(str(t).lower().split())
            if t and len(t.encode()) <= 64 and t not in seen:
                seen.add(t)
                rows.append({"text": t, "src": src})
    random.Random(0).shuffle(rows)
    for i, r in enumerate(rows):                        # 3% held out to evaluate the student
        r["split"] = "eval" if i % 33 == 0 else "train"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {}
    for r in rows:
        counts[r["src"]] = counts.get(r["src"], 0) + 1
    print(f"{len(rows)} texts (<= 64 bytes, deduplicated):", counts)
    return rows


def embed(rows, model=TEACHER, batch=64):
    """Teacher vectors, normalized, float16, in corpus order. Resumes from chunk files."""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model, device="cpu")
    texts = [r["text"] for r in rows]
    chunk = 4096
    parts = []
    t0 = time.time()
    for i in range(0, len(texts), chunk):
        f = OUT / f"emb_{i // chunk:04d}.npy"
        if not f.exists():
            e = st.encode(texts[i:i + chunk], batch_size=batch, normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float16)
            np.save(f, e)
        parts.append(np.load(f))
        done = min(len(texts), i + chunk)
        rate = done / max(1e-6, time.time() - t0)
        print(f"  {done}/{len(texts)} embedded ({rate:.0f}/s)", flush=True)
    E = np.concatenate(parts)
    np.save(OUT / "teacher.npy", E)
    (OUT / "teacher.json").write_text(json.dumps({"model": model, "n": len(texts), "dim": E.shape[1]}))
    print("teacher vectors:", E.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args()
    rows = build()
    if a.embed:
        embed(rows)


if __name__ == "__main__":
    main()
