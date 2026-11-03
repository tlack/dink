# Scenario generation

> Written for the first generator (the smart home, `dink/gen/home.py`). Commands and sizes
> here are from that stage; the released model also trains on LLM-invented gadgets and Home
> Assistant templates and uses `configs/home_split_v4.json`. See the README and
> [results.md](results.md) for the current recipe.

Training data for Dink comes from scenario generators: code that builds a situation,
a request, the tools on offer, and the provably right answer. The first one models a
smart home (`dink/gen/home.py`).

## Three pieces

1. **Generators (`dink/gen/`)**, e.g. `dink/gen/home.py`, decide meaning: which devices
   exist, their values, which tools are offered (including near misses), what the user
   asks for, and the right answer, computed from rules. A generator holds nothing out: it
   produces every variant and records its choices in `meta["uses"]` (words, templates,
   rooms, device kinds, naming style, units, slot counts).
2. **The splitter (`dink/data/split.py`)** routes records by those choices (a group
   split). Anything touching a held-out value on one axis goes to that axis's test set;
   records touching none become train / val / test_indist. Held-out values come from a
   config (`configs/home_split.json`) and/or are sampled at random from the pool
   (`"sample": {"words": 3, "kinds": 1}`), seeded and recorded in `manifest.json`.
3. **Augmentations (`dink/gen/augment.py`)** make answer-preserving changes each time a
   scenario is drawn for training: reorder tools and state fields, drop unneeded state
   fields, move prompt pieces ("in the kitchen, make it warmer"), filler words, case and
   punctuation, typos, and speech-recognition-style errors. Not smart-home specific.

The generator records which prompt pieces may move (`meta["chunks"]`) and which state
fields the answer depends on (`meta["needed"]`), so augmentation never breaks an answer.
Tests check this, and check that nothing held out leaks into train or val.

## What the smart home covers

| challenge | example | how it's modeled |
|---|---|---|
| which room | "turn on the light" with lights in 3 rooms | room words in the prompt; occupancy sensors (`occ_kit` = 1) in the state; if nothing disambiguates, probability is split across the plausible rooms |
| relative requests | "a little warmer", "way brighter" | step size by intensity; the answer is the offered value that moves the right way and lands closest |
| absolute requests | "turn on the fan", "set the heat to 72" | on/off targets; numbers in the prompt (a separate class, `numeric`) |
| indirect requests | "i'm freezing", "it's too dark in here" | the intensity is implied ("freezing" = big step) |
| already at a limit | "brighter" at 100% | nothing relevant (`limit`) |
| device not offered | "open the blinds", no blinds tool | nothing relevant (`none_missing`) |
| not an action | "is the front door locked?", "play some jazz", noise | nothing relevant |
| near misses | kitchen vs bedroom light; `temp 78` vs `temp 70` | several tools of the same kind, several candidate values |
| locks and doors | "lock the front door", "close the garage" | 0/1 values |
| scenes | "goodnight", "we're leaving" | tools without a value |
| naming conventions | `kit_lite`, `lite_kit`, `set_kitl` | three training styles, one held out (`kit.lit`) |

Devices: lights and blinds (0..1), thermostats (60..85 °F), fans (0..3) in up to four of
six rooms (the den is normally held out), plus front/back locks and a garage door. Distractor state fields (outdoor
temperature, humidity, hour, air quality, battery, CO2) never matter to the answer.

Current training mix (1,000 examples): 22% "nothing relevant", 4% ambiguous; prompts up to
54 bytes; 1-6 tools and 2-6 state fields per scenario.

## Evaluation sets

From `configs/home_split.json` (500 each at most):

| set | what's new to the model |
|---|---|
| `val` | nothing: for model selection during training |
| `test_indist` | nothing: fresh records from the training distribution |
| `test_words` | synonyms never trained ("climate", "chillier", "family room") |
| `test_templates` | prompt templates never trained |
| `test_rooms` | a room never trained (the den) |
| `test_styles` | a tool naming style never trained (`kit.lit`) |
| `test_units` | thermostats in Celsius (16..30) |
| `test_slots` | 7-8 state fields or tools (training uses at most 6) |
| `test_kinds` | (optional, sampled) a whole device kind never trained |
| `test_mixed` | records touching held-out values on several axes |

"Nothing relevant" precision/recall is reported on every set rather than kept as a set of
its own. Speech-style noise is applied at evaluation time (augment.py) to any set.

Only ~6% of generated records touch no held-out value, so the pool is large (60,000
records, 9 s) to leave 1,000 clean training examples.

## Commands

```bash
uv run python -m dink.gen --n 60000 --out data/home/pool.jsonl     # generate
uv run python -m dink.data.split data/home/pool.jsonl --config configs/home_split.json \
    --out data/home/split --n-train 1000                            # split
uv run python -m dink.gen --show 10                                 # look at examples
uv run python -m dink.gen --stats                                   # class mix
uv run pytest                                                       # checks
```
