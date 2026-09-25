# Scenario generation

Training data for Dink comes from scenario generators: code that builds a situation,
a request, the tools on offer, and the provably right answer. The first one models a
smart home (`dink/home.py`).

## Two stages

1. **Stage 1, the generator (`dink/home.py`)** decides meaning: which devices exist,
   their current values, which tools are offered (including near misses), what the user
   asks for, and the right answer. The answer is computed from rules, never guessed from
   the text.
2. **Stage 2, augmentation (`dink/augment.py`)** makes answer-preserving changes each time
   a scenario is drawn for training: reorder tools and state fields, drop unneeded state
   fields, move prompt pieces ("in the kitchen, make it warmer"), filler words, case and
   punctuation, typos, and speech-recognition-style errors. These functions are not
   smart-home specific; future generators reuse them.

The generator records which prompt pieces may move (`meta["chunks"]`) and which state
fields the answer depends on (`meta["needed"]`), so stage 2 never breaks an answer. A test
checks this on every run.

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
five rooms, plus front/back locks and a garage door. Distractor state fields (outdoor
temperature, humidity, hour, air quality, battery, CO2) never matter to the answer.

Current training mix (1,000 examples): 22% "nothing relevant", 4% ambiguous; prompts up to
54 bytes; 1-6 tools and 2-6 state fields per scenario.

## Held-out evaluation sets

Every word list and template carries a split tag, so evaluation can use things training
never saw. `uv run python -m dink.gen` writes all of them to `data/home/`.

| set | what's new |
|---|---|
| `indist` | nothing: fresh samples from the training distribution |
| `h_range` | thermostats in Celsius (16..30) |
| `h_template` | prompt templates never trained |
| `h_words` | synonyms never trained ("climate", "chillier", "family room") |
| `h_rooms` | a room never trained (the den), always named in the prompt |
| `h_naming` | a tool naming style never trained (`kit.lit`) |
| `h_slots` | 7-8 state fields and tools (training uses at most 6) |
| `h_none` | "nothing relevant" cases only, with unseen off-domain phrases |
| `h_noise` | speech-recognition errors, typos and fillers on every prompt |

## Commands

```bash
uv run python -m dink.gen                     # write data/home/*.jsonl
uv run python -m dink.gen --show 10           # look at examples (--set h_rooms, ...)
uv run python -m dink.gen --stats             # class mix of the training distribution
uv run pytest                                 # invariants, answer checks, augmentation safety
```
