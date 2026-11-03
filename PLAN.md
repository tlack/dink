# Dink: a tiny decision model for microcontrollers

> Working design notes, written before the first experiment and updated along the way. Some
> questions here were settled later; [docs/results.md](docs/results.md) has what actually
> happened. The on-device parts (sections 4, 5, 8, 9) are plans: Dink runs in PyTorch today.

Dink picks what a device should do. Given a small structured **state**, a short text
**prompt** (typed, or transcribed on-device by [BabyTalk](https://github.com/tlack/babytalk)), and up to 8 **tools**, it points
at the best tool, or answers "nothing relevant". It is sized to run in milliseconds on an ESP32-S3.

```
state:  {"ac_set", 80, 60..85}  {"room_t", 88, 50..100}  {"humid", 40, 0..100}
prompt: "make it a little colder"
tools:  {"set_temp", 70, 60..85}  {"set_temp", 78, 60..85}  {"set_temp", 84, 60..85}  {"set_fan", 3, 0..3}
->      tool 1 (set_temp 78)        p = 0.91
```

Named after Peter Dinklage: small, and punches far above its size.

## 1. Why this shape

- **Pointing, not generating.** The answer is an index into options the device supplied, so
  it is always a valid action. Probabilities come for free (softmax over options), plus a
  "nothing relevant" option so it can decline. This is the same core idea as Jev/Laya
  (score each option in place), without a 400M-parameter language model underneath.
- **No tokenizer.** Names and prompt are raw bytes in fixed-size fields, so memory and
  compute are fixed and known in advance.
- **Numbers travel separately from text.** Every state field and tool argument is
  `(name, value, min, max)`. Values are normalized to 0..1 by their declared range, so
  "a little colder" means the same small negative step on a thermostat (60..85) or a
  dimmer (0..1). Byte-level digits would force a tiny model to learn place value.
- **The device lists numeric choices.** Dink never predicts an integer. The harness expands
  a numeric tool into candidates (around the current value, or across the range), and Dink
  picks one. Precision = how many candidates the harness offers.

## 2. Input format (v0)

| field | size | contents |
|---|---|---|
| state | 8 slots | name (up to 64 bytes, e.g. "kitchen thermostat"), value, min, max; empty slots masked |
| prompt | 64 bytes | lowercase text, NUL padded (longer prompts: last 64 bytes) |
| tools | 8 slots | name (up to 64 bytes), argument value, min, max, has-argument flag; empty slots masked |
| output | 9 logits | one per tool slot + "nothing relevant"; softmax over the unmasked ones |

Names are encoded once, when a device's state fields and tool list are set up, and their
vectors are cached; only the prompt and the current values are processed per query. So
names can be long and descriptive rather than 8-byte abbreviations.

Normalized value = clip((v - min) / (max - min), 0, 1). Categorical values (mode 1/2/3)
should be text in the name for now ("mode_eco"), not numbers.

On a device this is just a C struct (~300 bytes). An Erlang/Elixir host (AtomVM) can pass
`{:state, [{"ac_set", 80, 60, 85}, ...]}` terms straight through.

## 3. Network (v1, D = 64)

See [`docs/architecture.md`](docs/architecture.md) for the diagram. In short:

- **One text encoder T** (byte embedding, depthwise + 1x1 convolutions, attention pooling)
  for everything textual: the prompt every query (4 tokens), names once per setup (1 vector
  each, cached). The same layer types as BabyTalk's speech model, so MMRT runs them.
- **Value encoder V + fuse F:** each normalized value becomes 18 features (16 piecewise-linear
  bins, magnitude, has-value) and is mixed with its name vector in a small MLP, so an item
  token can mean "thermostat, set low" before any later layer sees it (early fusion).
- **Relation step:** each tool finds its state field by attention over name vectors, and
  gets Δ = tool value − state value as extra features.
- **Mixer:** 21 tokens (8 state, 4 prompt, 8 tool, 1 "nothing"), 2 attention blocks (or one
  block applied 3 times), then a scorer and a masked softmax over 9 outputs.

~190 K parameters, ~6 M multiply-adds per query, ~10 ms on the ESP32-S3 (estimate).

## 4. Shaping it for the ESP32 (wide, not deep)

The S3's vector unit does 16 int8 multiply-adds per instruction. It is fed well when a
layer's weights sit in fast memory and are reused across many rows, and badly when the
network is a long chain of tiny layers (each layer pays fixed costs: call, staging, sync
between cores). So:

- **Few, wide layers.** Every width is a multiple of 16 (64, 128, 256). Prefer one 128-wide
  layer over two 64-wide ones in sequence.
- **Batch items as rows.** All 16 names go through the name encoder in one matrix multiply;
  all 21 tokens through each mixer layer at once. Each weight is then reused 16-21 times.
- **Depth by reuse, not by new layers** (the Tiny Recursive Model idea): if more depth
  helps, apply the same mixer block 2-4 times. Same weights, so they stay in SRAM/cache;
  more compute, no more storage. This is an ablation in the first experiment.
- **Split across both cores** by output channels, as MMRT already does.
- **Keep the whole model under ~256 KB** so it lives in PSRAM (or partly SRAM) next to the
  speech model, with no flash traffic per decision.

## 5. Low-bit weights (later, not in the first experiment)

- Train in float first. Then quantization-aware training to int8, then try ternary
  ({-1, 0, +1}, BitNet b1.58 style) for the name encoder, prompt encoder and FFNs, keeping
  the value encoders and scorer at int8 (numbers need the precision).
- On the S3, MMRT's kernels are int8: ternary weights would be unpacked to int8 while
  staging, like BabyTalk's 4-bit weights. So ternary saves storage and memory traffic,
  not multiply time. That matters most for domain experts streamed from flash (section 8).
- "Vague" low-precision attention (e.g. sign-binarized queries and keys) is interesting
  as a regularizer, but with 21 tokens attention costs almost nothing, so it buys no speed.
  Test it as an ablation only if the float model overfits.

## 6. First experiment: prove the concept on one hand-coded domain

**Goal:** show that the structured format + pointer head learns a small domain from a
modest dataset, generalizes to held-out variations, and beats a bytes-only version on
numbers. Train on CPU (the model is tiny; the laptop GPU is optional and runs hot).

### Domain: a smart home

Built: `dink/gen/home.py` (+ the splitter `dink/data/split.py`), described in [`docs/scenarios.md`](docs/scenarios.md). Lights,
blinds, thermostats and fans in up to four rooms, locks and a garage door, scenes, and
distractor sensors. It covers room ambiguity (resolved by room words or occupancy
sensors, or split probability when neither helps), relative / absolute / indirect /
numeric requests, requests at a device's limit, missing devices, questions, off-domain
requests, near-miss tools, and several tool naming styles. Stage 2 (`dink/gen/augment.py`) adds answer-preserving variations each time a scenario is drawn.

**Data:** 1,000 base examples, plus augmentation applied every time an example is drawn.

### Held-out evaluation sets (500 each)

| set | what changes | expectation |
|---|---|---|
| `indist` | fresh samples from the training distribution | ~98% |
| `h_range` | thermostats in Celsius (16..30) | should hold: tests normalization |
| `h_template` | prompt templates never trained | should mostly hold |
| `h_slots` | 7-8 state fields and tools (training uses at most 6) | should hold |
| `h_none` | "nothing relevant" only, unseen off-domain phrases | should mostly hold |
| `h_noise` | speech-recognition errors, typos, fillers | partial |
| `h_rooms` | a room never trained (the den) | partial: the name bytes must match the prompt |
| `h_naming` | a tool naming style never trained (`kit.lit`) | expected weak: motivates teacher init |
| `h_words` | synonyms never trained ("climate", "chillier") | expected weak: same |

Metrics: top-1 accuracy, "nothing relevant" precision/recall, calibration error, and a
"right magnitude" rate for numeric picks (chose the small step when asked for "a little").

### Comparisons

- **A: bytes only**: numbers rendered into the name bytes (`set_t 78`), no value channel.
- **B: structured** (section 3).
- Ablations of B: no relation features; raw scalar instead of piecewise-linear encoding;
  mixer depth 1 vs 2 vs one block applied 3 times; D = 64 vs 128.

**Success:** loss clearly falling by step 100 (sanity); B >= 95% in-distribution and
>= 90% on H-range / H-template / H-slots / H-none; B clearly beats A on numeric picks.
If B cannot learn one hand-coded domain, stop and rethink before generating more data.

## 7. Then: world knowledge and more domains

- **Teacher initialization (next step):** before decision training, distill a sentence-embedding
  teacher into the text encoder: the same text goes to both, and the student learns to
  match the teacher's vector (cut to 64 dims) and its similarity structure within a batch.
  Also distill fused item tokens toward the teacher's embedding of verbalized items
  ("kitchen thermostat, set low", "light at full brightness"), so values carry meaning early.
- **Choosing the teacher by test:** candidates EmbeddingGemma-300M, Qwen3-Embedding-0.6B,
  bge-small-en-v1.5, multilingual-e5-base, scored on our own probes: matching requests to
  tool descriptions, separating in-catalog from out-of-catalog requests (abstention is the
  hard part: "Selection Is Retrieval, Abstention Is Not", arXiv 2609.18672), and placing
  verbalized values near the right words.
- **More domains via OpenRouter:** have a frontier model propose domains, fields, ranges,
  tools and prompt phrasings as structured specs; we turn each spec into a generator like
  the hand-coded one. The LLM supplies breadth; code supplies volume and exact labels.
- **"Nothing relevant" at scale:** cross-domain mismatches (thermostat prompt, only media
  tools offered) come free once there are several domains.

## 8. Then: domain experts across storage tiers

- Train one generalist on all domains; fork a copy per domain and fine-tune it deeply
  (Branch-Train-Merge style: one expert per query, not per token).
- Split: the **trunk** (name, value, prompt encoders + a router) stays resident; each
  **expert** (mixer + scorer, ~100-150 KB) is swappable. The router's labels are free: the
  generator knows each example's domain.
- Storage (measured on the BabyTalk board): experts in **flash** are read in place, no
  loading step (~32 MB/s, so ~5 ms per 150 KB expert); PSRAM 88 MB/s; experts on **SD**
  are copied into PSRAM on first use (not into flash: slow to write, and it wears).
- Wrong-route safety: if the router is unsure, try the top two experts; if an expert says
  "nothing relevant", try the next.

## 9. Then: on the device

Export to MMRT (int8), add the attention op, run on the ESP32-S3, and wire
BabyTalk speech -> Dink -> action, with BabyTalk's voice saying "sorry, which did you
mean?" when Dink answers "nothing relevant" or is unsure.

## Later ideas

- **Primitive vision through the state:** state fields as image statistics (average
  brightness, per-channel color, quadrant hues, edge-orientation counts) from a camera
  frame, e.g. car vs. plane. Needs more state slots (16-64) and good hand-made features.

- **Enumerated values** ("play country music"): no ordering, so never map categories to
  numbers. Options: (a) one tool per category (`play_country`), simplest but uses slots;
  (b) text-valued arguments: the argument is text ("country") encoded by the text encoder
  and cached, fused like a numeric value; (c) two passes: pick `play_music`, then run Dink
  again with the genres as its options (pages of 8 for long lists). Leaning (b) + (c).
  Not settled or built yet; (a) is the workaround today.

- **Beyond the microcontroller: Jev-style workloads** (noted 2026-09-26; parked until the
  gadget case works). Dink is a small decision model for any problem, not just physical
  control. Mapping:
  - tools = labels as no-value actions (`spam`, `route_to_code_model`, `flag_harassment`);
  - state = the numeric metadata text classifiers ignore (sender reputation, account age,
    link count, length, hour);
  - "nothing relevant" = abstain / escalate to a bigger model;
  - labels are encoded names, so a trained Dink can pick among labels it has never seen.
    That's the Jev claim worth proving.

  Order when we get there:
  1. Intent classification with held-out intents: CLINC150 / MASSIVE (already downloaded),
     Banking77; sample 8 intents per example.
  2. A skill router built from the 1,000 LLM worlds (tools = world names; free).
  3. Spam: UCI SMS Spam.
  4. Moderation (Jigsaw / Civil Comments) and LLM routing (arena preference data). These
     need longer prompts, 256 bytes: linear cost in the conv encoder, 4x per query on the
     ESP32.

  Verify each dataset before relying on it. Open question: capacity. Does one 204K model
  improve from mixing domains or run out of room? Measure it on the gadget test sets, with a
  wider model as the fallback.

- **Cascades of Dinks** (noted 2026-09-26). The first Dink picks a skill; later Dinks pick
  parameters. This handles more than 8 options (category, then subcategory). Tools needn't
  be actuators: a severity level, a confidence band, a response budget are candidate values
  within declared ranges, which Dink already supports. Modeling such decompositions is also
  a source of more diverse training inputs. On the device, stages run in sequence
  (about 2x compute per decision).

## Open questions

- Prompts with explicit numbers ("set it to 72"): **answered, extract them.** Even the
  600M-parameter teacher places "set the thermostat to 83" no nearer "warm" than "cold"
  (0.51 on 59 unseen numbers), so distillation cannot teach digits. The harness should pull
  numbers out of the prompt into numeric slots (2-3). A prompt number has no range of its
  own: it is interpreted per candidate tool, against that tool's declared range, in the
  relation step. Per tool and prompt number: in range?, level within the range, distance to
  the candidate's value (fraction of the range), exact match. "set it to 2" then points at
  the fan (0..3) and away from the thermostat (60..85) with no device named. Units: "50
  percent" / "half" are levels (0.5 of any range); bare numbers and "degrees" are raw values.
  Number words from speech ("seventy two") are converted to digits first.
- Is 8 bytes enough for names? (We think yes, now that values travel separately.)
- How much does the prompt encoder need to see word order vs. just which words appear?

## References

- Jev (TypeSafe AI), and its open counterparts [Laya](https://github.com/NandhaKishorM/laya)
  (per-option `[MASK]` markers + scorer) and [Open-Jev](https://github.com/Zefan-Cai/Open-Jev).
- [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
  (Tiny Recursive Model), and [Probabilistic TRM](https://arxiv.org/abs/2605.19943) (2026).
- [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)
  (piecewise-linear and periodic encodings); xVal (Golkar et al., 2023); FoNE, Fourier number embeddings (Zhou et al.).
- [BitNetMCU](https://github.com/cpldcpu/BitNetMCU): ternary/low-bit networks on
  microcontrollers.
- Branch-Train-Merge (Li et al., 2022) and Branch-Train-MiX (Sukhbaatar et al., 2024).
- [BabyTalk](https://github.com/tlack/babytalk), the author's companion project for on-device
  speech to text on the ESP32-S3: the MMRT runtime, storage-tier measurements, STT/TTS.
