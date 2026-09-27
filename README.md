# Dink

A tiny decision model for microcontrollers. A gadget gives it its current **state** (named
values with declared ranges), a spoken **request**, and up to 8 offered **actions**; Dink
returns one probability per action plus "nothing relevant". 204K parameters and about 6M
multiply-adds per query, sized for an ESP32-S3. It never generates text or numbers: the
device harness offers candidate values ("set the thermostat to 72 / 75 / 79") and Dink
picks one.

Named after Peter Dinklage. Sibling of [BabyTalk](../esp32-micromodels) (on-device speech to
text), which feeds it requests.

- Design and plans: [PLAN.md](PLAN.md)
- Network diagram: [docs/architecture.md](docs/architecture.md) (Graphviz source in `docs/architecture.dot`)
- Every experiment and its numbers: [docs/results.md](docs/results.md)

## Status (2026-09-27)

Trained on the GPU from about 156k scenarios, drawn from three families of sources (below).
The best finished run so far (`runs/dink/all_v9`) scores, on held-out data:

| test | accuracy |
|---|---|
| gadgets in 5 whole areas never seen in training (beekeeping, saunas, drones, clinics, studios) | 0.91 |
| unseen gadgets in known areas | 0.89 |
| Home Assistant's own hand-written test sentences | 0.97 |
| smart home, held-out words / templates / naming styles | 0.88 / 0.94 / 0.96 |
| prompt-only (no state at all) | 0.96 |

In progress: "ordinal" strength ("a bit" / "a lot" pick among the offered values by rank,
with nearly-right answers not punished) and graded targets with a ranking loss.

## Quick start

```sh
uv sync                                        # Python deps (uv, never pip); torch is the CUDA 13.0 build
uv run pytest                                  # tests

uv run python -m dink.play -i                  # drive the newest model by hand (:random, :tools, :state)
uv run python -m dink.evaluate runs/dink/all_v9 --split data/home/split_v10
uv run python -m dink.evals.hard --run runs/dink/all_v9       # the hand-written hard cases
```

Rebuilding data and training (the LLM steps need an OpenRouter key in `.env`; see below):

```sh
uv run python -m dink.gen --n 400000 --bank data/llm/phrases.jsonl --out data/home/pool.jsonl
uv run python -m dink.gen.worlds --extra data/llm/inverted.jsonl --out data/worlds/pool.jsonl
uv run python -m dink.gen.ha --out data/ha      # needs the Home Assistant intents repo, see below
uv run python -m dink.data.split <pool> --config configs/home_split_v4.json --out data/home/split
uv run python -u -m dink.train --split data/home/split --steps 40000 --no-state 0.3 \
    --mix "home=0.4,world=0.45,ha=0.15" --name mine
```

`data/` and `runs/` are git-ignored: all data is rebuilt from the code, the external sources
below, and the cached LLM replies.

## Repository map

| path | what |
|---|---|
| `dink/model/` | the network: byte text encoder, value fuse, relation step, mixer, scorer |
| `dink/gen/home.py`, `names.py`, `ordinal.py` | the hand-written smart-home scenario generator |
| `dink/gen/worlds.py`, `ha.py`, `bank.py` | converters: LLM worlds, Home Assistant templates, LLM phrase bank |
| `dink/llm/` | OpenRouter client and the LLM data jobs (phrases, worlds, inverter, intensity) |
| `dink/teacher/` | teacher probes and distillation of the text encoder |
| `dink/data/` | scenario format, group splitter, collation, number extraction |
| `dink/train.py`, `evaluate.py`, `play.py`, `evals/` | training, scoring, hand driving, hard cases |

## External data and models

What Dink is built from, besides the code in this repo. None of it is redistributed here: every
source is downloaded (or queried) by the scripts that use it, and each keeps its own license.

### Datasets

| source | used for | used by | license |
|---|---|---|---|
| [Home Assistant intents](https://github.com/OHF-Voice/intents) (OHF-Voice, English sentence templates, fixtures and test sentences) | ~7.5k prompt-only training scenarios over 26 intents, sampled from HA's voice templates; HA's 900 hand-written test sentences as the `test_ha` / `test_ha_unseen` sets | `dink/gen/ha.py` (expects a clone at `data/ha/intents`) | CC-BY-4.0 |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) (English, via `mteb/amazon_massive_intent`) | assistant requests in the text-encoder distillation corpus | `dink/teacher/corpus.py` | CC-BY-4.0 (the mteb copy is tagged Apache-2.0) |
| [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) ("plus" config) | assistant and out-of-scope requests in the distillation corpus | `dink/teacher/corpus.py` | CC-BY-3.0 |
| [wordfreq](https://github.com/rspeer/wordfreq) word list | the 20,000 most common English words in the distillation corpus | `dink/teacher/corpus.py` | code Apache-2.0; word data CC-BY-SA-4.0 |

### Models and libraries

| source | used for | license |
|---|---|---|
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | the teacher: Dink's text encoder is distilled from its embeddings (training only; it never ships in Dink) | Apache-2.0 |
| multilingual-e5-base, bge-small-en-v1.5 | compared as teachers in the probes only (`dink/teacher/probe.py`) | MIT |
| [hassil](https://github.com/OHF-Voice/hassil) | expands and recognizes Home Assistant's templates to label sentences | Apache-2.0 |

### LLM-generated data (via OpenRouter)

Generated here, cached on disk (`data/llm_cache/`), logged (`data/llm/calls.jsonl`), and
checked in code before use. The replies are ours to use; the models' terms apply to generation.

| job | model | output | cost |
|---|---|---|---|
| phrase bank (`dink/llm/phrases.py`) | Space Bunny Alpha (a free stealth test model) | 4,770 smart-home phrasings with known meanings | free |
| value descriptions (`dink/llm/values.py`) | Space Bunny Alpha | ~1k value descriptions for distillation | free |
| LLM worlds (`dink/llm/worlds.py`) | DeepSeek V4.1 Flash on Parasail | 1,000 invented gadgets in 50 areas, 68,765 labeled utterances | $2.92 |
| inverter (`dink/llm/invert.py`) | DeepSeek V4.1 Flash on Parasail | 19,380 contrast variants (opposite, complaint/desire, negation) | $1.75 |
| intensity (`dink/llm/intensity.py`) | DeepSeek V4.1 Flash on Parasail | 28,480 graded-strength phrasings | $1.30 |

Every LLM label is checked in code (schema, ranges, agreement between two independent
labelings, on/off wording, direction words, numbers said vs. labeled); see `dink/gen/worlds.py`.

### Considered, not yet used

Schema-Guided Dialogue (unseen services in its test split), MultiWOZ 2.2, Taskmaster-2/3,
API-Bank, APIBench, ToolBench, TaskBench, NESTFUL, When2Call, `acon96/Home-Assistant-Requests`,
EB-ALFRED, Fluent Speech Commands, Mind2Web. Notes on fit are in PLAN.md ("Later ideas").

## License

The code in this repository is released under the [MIT license](LICENSE), copyright
Thomas Lackner. The external data and models listed above keep their own licenses, and
none of them are redistributed here.

## AI authorship

Dink was built by Thomas Lackner with heavy assistance from Claude Opus 5.5, Anthropic's AI
model, working in Claude Code. Claude wrote most of the code, the data generators and
converters, the LLM prompts and their validation, and the training and evaluation tooling;
it ran the experiments and wrote up the results. Thomas set the direction and design,
questioned results, tested the model by hand, and decided what to try next. Much of the
training data is itself AI-generated (see "LLM-generated data" above), checked in code
before use.

## Credentials

`.env` (git-ignored) holds `OPENROUTER_KEY` and a default `OPENROUTER_MODEL`. The data jobs
pick their model explicitly (`Client.parasail()`, `Client.groq()`) and stop at hard caps:
requests per run, requests per day, dollars per day, and consecutive failures.
