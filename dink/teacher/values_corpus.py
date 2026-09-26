"""Value data for distillation: what numeric states mean in words.

- items: (device name, value, lo, hi) -> the teacher's mean vector of that state's LLM
  descriptions (data/llm/values.jsonl). Each description set is reused for jittered nearby
  values (offline augmentation), so 83 of 60..85 and 82.4 share "warm setting" targets.
- number texts: templated requests with numbers ("set the thermostat to 84"), EVEN numbers
  only; odd numbers are kept back for the probe.
Held-out device types (oven, battery) appear in neither; they are probe-only.

  uv run python -m dink.teacher.values_corpus          # data/teacher/values_*.{jsonl,npy}
"""
import json
import random
from pathlib import Path

import numpy as np

from ..llm.values import TYPES

OUT = Path("data/teacher")
TEACHER = "Qwen/Qwen3-Embedding-0.6B"

NAMES = {  # device names a setup might use, per value type (training-split rooms only)
    "thermo_f": ["thermostat", "kitchen thermostat", "living room thermostat", "bedroom thermostat",
                 "heat setpoint", "ac setpoint", "office thermostat"],
    "thermo_c": ["thermostat", "bedroom thermostat", "heat setpoint", "office thermostat"],
    "light": ["light", "kitchen lights", "bedroom light", "light brightness", "office lamp", "lounge lights"],
    "fan": ["fan", "ceiling fan", "bedroom fan", "fan speed", "office fan"],
    "blinds": ["blinds", "kitchen blinds", "bedroom shades", "window blinds"],
    "volume": ["speaker volume", "volume", "living room speaker", "kitchen speaker"],
    "humidity": ["humidity", "bathroom humidity", "room humidity"],
    "room_temp": ["room temperature", "kitchen temperature", "bedroom temperature sensor"],
    "oven": ["oven", "oven temperature"],
    "battery": ["battery", "phone battery", "battery charge"],
}
NUMBER_TEMPLATES = {
    "thermo_f": ["set the thermostat to {v}", "make it {v} degrees", "thermostat {v}", "heat to {v}",
                 "set the temperature to {v} degrees"],
    "thermo_c": ["set the thermostat to {v} celsius", "make it {v} degrees celsius"],
    "light": ["set the lights to {v} percent", "brightness {v} percent", "dim the lights to {v} percent"],
    "fan": ["set the fan to {v}", "fan speed {v}"],
    "blinds": ["open the blinds to {v} percent", "blinds at {v} percent"],
    "volume": ["set the volume to {v}", "volume {v} percent", "turn the speaker to {v}"],
    "humidity": ["set humidity to {v} percent"],
    "room_temp": [],
}


def load_descriptions():
    rows = [json.loads(l) for l in Path("data/llm/values.jsonl").read_text().splitlines()]
    groups = {}
    for r in rows:
        groups.setdefault((r["type"], r["value"]), {"seed": r, "texts": []})["texts"].append(r["text"])
    return groups


def build(jitter=0.04, per_seed=24, seed=0):
    rng = random.Random(seed)
    types = {t[0]: t for t in TYPES}
    groups = load_descriptions()
    desc_texts = sorted({t for g in groups.values() for t in g["texts"]})
    items = []
    for (tid, v), g in sorted(groups.items()):
        _, noun, lo, hi, unit, integer, held = types[tid]
        for k in range(per_seed):
            vj = v if k == 0 else min(hi, max(lo, v + rng.uniform(-jitter, jitter) * (hi - lo)))
            items.append({"type": tid, "name": rng.choice(NAMES[tid]), "value": round(vj, 3), "lo": lo, "hi": hi,
                          "anchor": v, "held_out": held, "texts": g["texts"]})
    numbers = []
    for tid, temps in NUMBER_TEMPLATES.items():
        _, noun, lo, hi, unit, integer, held = types[tid]
        for v in range(int(lo), int(hi) + 1):
            if v % 2 == 0:                      # odd numbers are probe-only
                numbers += [t.format(v=v) for t in temps]
    return desc_texts, items, sorted(set(numbers))


def main():
    from sentence_transformers import SentenceTransformer
    desc_texts, items, numbers = build()
    texts = desc_texts + numbers
    print(f"{len(desc_texts)} descriptions, {len(numbers)} number texts, {len(items)} items "
          f"({sum(i['held_out'] for i in items)} held out)")
    st = SentenceTransformer(TEACHER, device="cpu")
    E = st.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=False).astype(np.float16)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "values_teacher.npy", E)
    (OUT / "values_texts.json").write_text(json.dumps({"descriptions": desc_texts, "numbers": numbers}))
    (OUT / "values_items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print("saved", E.shape)


if __name__ == "__main__":
    main()
