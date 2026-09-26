"""Group split: route generated records to train / val / test sets by what they used.

Generators record their choices in meta["uses"] (words, templates, rooms, device kinds,
naming style, units, slot counts). A record that touches a held-out value on exactly one
axis goes to that axis's test set (test_words, test_rooms, ...); one touching several goes
to test_mixed; records touching none are split into train / val / test_indist. So a
held-out word, room or naming style never appears anywhere in training.

What is held out comes from the config: fixed lists, plus optionally values sampled at
random from the pool ("sample": {"words": 3, "kinds": 1}), seeded and written to the
manifest so a run can be reproduced.

  uv run python -m dink.data.split data/home/pool.jsonl --config configs/home_split.json \\
      --out data/home/split --n-train 1000
"""
import argparse
import collections
import json
import random
import zlib
from pathlib import Path

from .scenario import Scenario

AXES = ["words", "templates", "rooms", "kinds", "styles", "units", "slots", "llm", "areas", "worlds"]


def values(uses, axis):
    """The values a record used on one axis (a set)."""
    if axis in ("words", "templates", "rooms", "kinds"):
        return {v for v in uses.get(axis, []) if v and v.strip()}
    if axis == "styles":
        return {uses.get("style")}
    if axis == "units":
        return {uses.get("units")}
    if axis == "areas":
        return {uses.get("area")} - {None}
    return set()


def llm_held(phrase_id, fraction):
    """LLM phrasings (dink/gen/bank.py ids) held out by a stable hash: test_llm."""
    return int(phrase_id, 16) % 1000 < fraction * 1000


def touches(uses, axis, held, slots):
    if axis == "llm":
        return any(llm_held(p, held["llm_fraction"]) for p in uses.get("llm", []))
    if axis == "worlds":                # LLM-invented worlds (dink/gen/worlds.py) held out by name hash
        w = uses.get("world")
        return bool(w) and uses.get("area") not in held["areas"] and \
            zlib.crc32(w.encode()) % 1000 < held["world_fraction"] * 1000
    if axis == "slots":
        if uses.get("world"):          # the slot-count test is about the smart home; worlds use all 8
            return False
        return uses.get("n_state", 0) > slots["max_state"] or uses.get("n_tools", 0) > slots["max_tools"]
    return bool(values(uses, axis) & held[axis])


def choose_holdouts(records, cfg, rng):
    """Fixed lists from the config, plus sampled values. Words and templates used by more
    than 20% of records are never sampled: holding one out would starve training."""
    held = {a: set(cfg.get(a, [])) for a in AXES if a not in ("slots", "llm", "worlds")}
    counts = {a: collections.Counter() for a in held}
    for s in records:
        for a in held:
            counts[a].update(values(s.meta["uses"], a))
    for axis, n in cfg.get("sample", {}).items():
        common_ok = axis in ("rooms", "kinds", "styles", "units")   # holding one out is the point
        pool = sorted(v for v, c in counts[axis].items()
                      if v not in held[axis] and (common_ok or c <= 0.2 * len(records)))
        held[axis] |= set(rng.sample(pool, min(n, len(pool))))
    return held


def split(records, cfg, seed=0, n_train=None):
    rng = random.Random(seed)
    held = choose_holdouts(records, cfg, rng)
    held["llm_fraction"] = cfg.get("llm_holdout", 0.0)
    held["world_fraction"] = cfg.get("world_holdout", 0.0)
    slots = cfg.get("slots", {"max_state": 8, "max_tools": 8})
    sets = collections.defaultdict(list)
    clean = []
    for s in records:
        hit = [a for a in AXES if touches(s.meta["uses"], a, held, slots)]
        if not hit:
            clean.append(s)
        else:
            sets["test_" + hit[0] if len(hit) == 1 else "test_mixed"].append(s)
    rng.shuffle(clean)
    fr = cfg.get("fractions", {"val": 0.1, "test_indist": 0.1})
    n_val, n_test = int(fr["val"] * len(clean)), int(fr["test_indist"] * len(clean))
    sets["val"] = clean[:n_val]
    sets["test_indist"] = clean[n_val:n_val + n_test]
    sets["train"] = clean[n_val + n_test:][:n_train]
    if n_train and len(sets["train"]) < n_train:
        print(f"warning: only {len(sets['train'])} clean records for training (asked {n_train});"
              " generate a bigger pool")
    cap = cfg.get("max_test")
    if cap:
        for name in list(sets):
            if name.startswith("test_"):
                rng.shuffle(sets[name])
                sets[name] = sets[name][:cap]
    # leak check: nothing in train or val touches a held-out value
    for name in ("train", "val"):
        for s in sets[name]:
            assert not any(touches(s.meta["uses"], a, held, slots) for a in AXES), name
    return dict(sets), {a: sorted(v) for a, v in held.items() if not a.endswith("_fraction")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=None, help="cap the training set (base examples)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    records = [Scenario.from_json(l) for l in Path(a.pool).read_text().splitlines() if l.strip()]
    sets, held = split(records, cfg, a.seed, a.n_train)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in sorted(sets.items()):
        (out / f"{name}.jsonl").write_text("".join(s.to_json() + "\n" for s in items))
    manifest = {"pool": a.pool, "config": a.config, "seed": a.seed, "n_train_cap": a.n_train,
                "held_out": held, "llm_holdout": cfg.get("llm_holdout", 0.0),
                "world_holdout": cfg.get("world_holdout", 0.0), "counts": {k: len(v) for k, v in sorted(sets.items())}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for k, v in sorted(sets.items()):
        print(f"  {k:14s} {len(v)}")
    sampled = {ax: v for ax, v in held.items() if ax in cfg.get("sample", {})}
    if sampled:
        print("  sampled holdouts:", {ax: [x for x in v if x not in cfg.get(ax, [])] for ax, v in sampled.items()})


if __name__ == "__main__":
    main()
