# Dink: a tiny decision model for microcontrollers

Dink picks what a device should do. Given a small structured **state**, a short text
**prompt** (typed, or transcribed on-device by BabyTalk), and up to 8 **tools**, it points
at the best tool, or answers "nothing relevant". It runs in milliseconds on an ESP32-S3.

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
| state | 8 slots | name (8 bytes, NUL padded), value, min, max; empty slots masked |
| prompt | 64 bytes | lowercase text, NUL padded (longer prompts: last 64 bytes) |
| tools | 8 slots | name (8 bytes), argument value, min, max, has-argument flag; empty slots masked |
| output | 9 logits | one per tool slot + "nothing relevant"; softmax over the unmasked ones |

Normalized value = clip((v - min) / (max - min), 0, 1). Categorical values (mode 1/2/3)
should be text in the name for now ("mode_eco"), not numbers.

On a device this is just a C struct (~300 bytes). An Erlang/Elixir host (AtomVM) can pass
`{:state, [{"ac_set", 80, 60, 85}, ...]}` terms straight through.

## 3. Network (v0, D = 64)

```
names (8 state + 8 tool, 8 bytes each) --byte embed 16--> concat 128 --> Linear 128x128 --> ReLU --> Linear 128x64  = name vec
values (normalized)  --> piecewise-linear encoding, 16 bins + "present" flag --> Linear 17x64                      = value vec
item token = name vec + value vec + type embedding (state / tool / nothing)

prompt 64 bytes --byte embed 32 + position--> [1x1 32->128] --> 2 x ( depthwise conv k5 + 1x1 128x128 + ReLU )
                --> attention pooling with 4 learned queries --> 4 prompt tokens x 64

relation features (per tool): tool name vec . state name vecs --> softmax --> attended state value
                --> delta = tool value - attended value --> PLE(delta in -1..1) --> Linear --> added to the tool token

mixer: 21 tokens (8 state + 4 prompt + 8 tool + 1 "nothing") x 64
       2 blocks: attention (4 heads x 16) + FFN 64 -> 256 -> 64, pre-norm
scorer: Linear 64 -> 1 on the 8 tool tokens and the "nothing" token --> masked softmax
```

| part | parameters | multiply-adds per decision |
|---|---|---|
| name encoder (shared, 16 names) | 29 K | 0.39 M |
| value + relation encoders | 2 K | 0.03 M |
| prompt encoder (conv) + pooling | 66 K | 3.5 M |
| mixer, 2 blocks | 99 K | 2.2 M |
| scorer, type/NONE embeddings | 1 K | < 0.01 M |
| **total** | **~200 K (~200 KB int8)** | **~6 M** |

At the 1-3 billion multiply-adds per second MMRT sustains on the S3, that is a few
milliseconds of math; fixed per-layer overheads will likely dominate, so plan on ~10 ms.

The prompt encoder is deliberately the same depthwise + 1x1 pattern as the BabyTalk speech
model: it runs on MMRT's existing kernels. The mixer's attention (21 tokens) is new but tiny.

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

- **Teacher initialization:** before decision training, train the name and prompt encoders
  to reproduce a sentence-embedding model's vectors (projected down to 64 floats) over lots
  of short phrases and identifiers. Target: H-words and H-names go from failing to passing.
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

## Open questions

- Prompts with explicit numbers ("set it to 72"): read the digits as bytes, or have the
  harness extract numbers from the prompt into a numeric slot? (Plan: try both in the
  second experiment.)
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
- BabyTalk (`~/work/esp32-micromodels`): MMRT runtime, storage-tier measurements, STT/TTS.
