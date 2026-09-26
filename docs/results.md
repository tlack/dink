# Results log

## 2026-09-25: teacher choice

Probes in `dink/teacher/probe.py` (smart-home vocabulary; held-out words from
`configs/home_split.json`).

| teacher | held-out word direction (16) | values in words (10) | abstention AUC (1,296) |
|---|---|---|---|
| multilingual-e5-base | 0.81 | 0.80 | 0.955 |
| bge-small-en-v1.5 | 0.81 | 0.90 | 0.974 |
| **Qwen3-Embedding-0.6B** | **1.00** | **1.00** | **0.988** |

## 2026-09-25: first distillation into the text encoder

Corpus: 58,505 texts (MASSIVE + CLINC150 37.6k, 19.8k common words, 565 device texts,
568 training prompts), Qwen3-Embedding-0.6B vectors, PCA to 64 dims (keeps 56% of the
variance). 30 epochs, cosine + in-batch similarity loss, light input noise. CPU.

| student | parameters | similarity agreement (Spearman, 1,773 held-out texts) | word direction | values | abstention AUC | time |
|---|---|---|---|---|---|---|
| width 128 | 74K | 0.641 | 0.81 | 0.40 | 0.984 | 8 min |
| width 256 | 236K | 0.695 | 0.62 | 0.40 | 0.993 | 15 min |

Readings:
- Both keep the teacher's in-domain vs off-domain separation (abstention AUC 0.98-0.99).
- Overall similarity is partly kept (0.64-0.70 rank agreement). Width 256 is better there.
- Word direction flips with width on only 16 cases: too few to tell the widths apart.
- Values in words fails (4/10, below chance): only 565 device texts, few of them value
  descriptions. The student has not learned "at full brightness" ~ "bright".

Next: larger probes (LLM-written, meaning-labeled), more device and value text in the
corpus, and the real test: Dink trained with this encoder vs from scratch on test_words.

## 2026-09-25: value distillation (T + F)

Same corpus plus 916 value descriptions and 2,016 value items (`dink/teacher/values_corpus.py`;
oven and battery held out as types).

| student | parameters | agreement | word direction | values in words | abstention AUC | item values (known / held-out types) |
|---|---|---|---|---|---|---|
| tv128 | 93K | 0.641 | 0.69 | 0.50 | 0.989 | 0.975 / 1.00 |
| tv256 | 255K | 0.687 | 0.81 | 0.40 | 0.997 | 0.977 / 1.00 |

The fused items order values correctly, even for unseen types. Values written as words still fail.

## 2026-09-25: first Dink runs (204K parameters, 3,000 steps, batch 64)

Accuracy = the pick is one of the best answers. `s1` = seed 1.

**1,000 training scenarios** (`data/home/split`)

| set | scratch | scratch s1 | distilled tv128 | distilled s1 | frozen tv128 | frozen tv256 |
|---|---|---|---|---|---|---|
| val | 0.889 | 0.897 | 0.897 | 0.907 | 0.801 | 0.841 |
| test_indist | 0.909 | 0.922 | 0.861 | 0.924 | 0.761 | 0.844 |
| test_words | 0.672 | 0.636 | 0.648 | 0.682 | 0.594 | 0.634 |
| test_templates | 0.654 | 0.694 | 0.630 | 0.616 | 0.568 | 0.660 |
| test_mixed | 0.640 | 0.652 | 0.624 | 0.634 | 0.562 | 0.620 |

**3,178 training scenarios** (`data/home/split3k`, every record that avoids all holdouts;
same val and test sets as above; delta bins dense near zero)

| set | scratch | scratch s1 | distilled tv128 | distilled s1 | scratch w256 | distilled tv256 |
|---|---|---|---|---|---|---|
| val | 0.947 | 0.927 | 0.980 | 0.960 | 0.922 | 0.980 |
| test_indist | 0.972 | 0.927 | 0.980 | 0.975 | 0.922 | 0.977 |
| test_units | 0.954 | 0.928 | 0.976 | 0.960 | 0.926 | 0.960 |
| test_rooms | 0.922 | 0.900 | 0.930 | 0.926 | 0.904 | 0.942 |
| test_slots | 0.938 | 0.922 | 0.956 | 0.928 | 0.886 | 0.950 |
| test_noise | 0.965 | 0.924 | 0.955 | 0.960 | 0.907 | 0.960 |
| test_words | 0.704 | 0.686 | 0.744 | 0.706 | 0.732 | 0.736 |
| test_templates | 0.724 | 0.632 | 0.704 | 0.712 | 0.654 | 0.700 |
| test_mixed | 0.726 | 0.656 | 0.734 | 0.674 | 0.718 | 0.732 |

Readings:
- Seed noise is ±0.03-0.06 per test set, so single runs mislead. At 1k the distilled start is
  no better than scratch.
- Freezing the distilled T and F is worse everywhere, including held-out words. Fine-tuning is
  not erasing useful knowledge; the features alone are not enough.
- 3x data is the big lever. It lifts everything, and it mostly fixes magnitude ("a bit" vs
  "a lot"; relative_small 0.47 -> 0.74 in-distribution). The denser delta bins did nothing
  measurable at 1k.
- At 3k the distilled start is consistently ahead in-distribution (0.96-0.98 vs 0.92-0.95)
  and steadier across seeds. On held-out language the edge is small (words 0.71-0.74 vs 0.69-0.73).
- Held-out language plateaus near 0.70-0.74. The typical error is abstaining on an unfamiliar
  phrasing: "nothing relevant" precision is 0.57-0.64 on test_words and 0.25-0.50 on templates.

**Why distillation helps so little on new words.** Probe: for each of 17 held-out phrasings,
which of 15 training phrasings is nearest?
- teacher 11/17; tv128 7/17; tv256 7/17
- The students keep one-word synonyms: toastier->warmer, secure->lock, carport->garage,
  nighty night->good night.
- They lose composed phrases: "less bright" -> brighter, "more lit" -> dimmer, "curtains wider"
  -> fan.
- Even the teacher sends indirect requests to the right topic but the wrong action:
  "i'm roasting" -> warmer, "brr it's frigid" -> cooler. Embedding similarity is topic, not
  intent, so indirect_* has to come from training data, not from the teacher.

Next:
- More varied phrasing in the Dink training data itself (LLM paraphrases; currently 200 lines).
- A bigger pool, so that 10k+ records avoid all holdouts.
- Distillation pairs that teach composition ("less X" vs "more X").

## 2026-09-25: more data, on the GPU; LLM phrasings as prompts

GPU (RTX 5080 laptop): batch 512, 10,000 steps, distilled tv128 start, ~11 min per run.
Both pools have 400k records split with the same holdouts. `old:` = the exact test sets
from the runs above (`data/home/split`).

- **big_tmpl**: hand templates only (`split_big`, 21,855 train).
- **big_llm**: half of all prompts swapped for LLM phrasings of the same meaning (`split_v2`,
  31,485 train, 3,324 distinct LLM phrasings in train). The bank is `dink/llm/phrases.py`:
  4,770 phrasings, 164 requests, with rotating speaker personas. Phrases that use a
  held-out word or template are routed to the test sets, and 15% of phrasings are held
  out as test_llm.

| set | best 3k run (b3k_distilled) | big_tmpl | big_llm |
|---|---|---|---|
| old:test_indist | 0.980 | 0.995 | 1.000 |
| old:test_words | 0.744 | 0.778 | **0.846** |
| old:test_templates | 0.704 | 0.736 | **0.952** |
| old:test_mixed | 0.734 | 0.748 | **0.806** |
| test_llm (held-out LLM phrasings) | | 0.544 | **0.894** |

Readings:
- 7x more template data: in-distribution ~0.99, but new wording barely moves (words 0.74 -> 0.78).
- The template-only model gets 0.54 on held-out LLM phrasings. Trained on templates, it
  handles only about half of natural wording.
- LLM phrasings: held-out templates 0.74 -> 0.95, words 0.78 -> 0.85, LLM phrasings 0.54 -> 0.89.
  Variety in the training prompts is the lever, not volume.
- Caveat: the annotator catches held-out wording only literally. "Toasty" can still reach
  training while "toastier" is held out, so test_words and test_templates now measure
  "this exact wording never seen" more than "this idea never seen".
- The bank has some label noise, e.g. "why is the front door unlocked?" filed as *lock it*.

## 2026-09-26: descriptive names; LLM-invented worlds

**Descriptive names** (`big_names`, split_v3: nine naming styles, `camel` held out). This run
survived a laptop crash through `--resume`; `train.py` now pauses at 80C.
Language results match big_llm (test_llm 0.872 vs 0.894, words 0.830 vs 0.840). The unseen
long style is handled: test_styles 0.966.

**Worlds** (`dink/llm/worlds.py`): DeepSeek V4.1 Flash on Parasail, reasoning off, strict
JSON schemas.
- Collected 1,000 settings in 50 areas, 68,765 utterances, for $2.92.
- Two independent labelings agree on 96.7% of requests.

`dink/gen/worlds.py` validates in code and keeps 96%. Dropped: the two labelings disagree
(1,791), same words with two labels, number mismatches, "a bit" moving over 20%, on/off
actions against their words, and flips moving against plain direction words.

250 worlds became 32k scenarios (2 variants each). Split v4 adds two holdouts: 5 whole areas
(test_areas) and 10% of the worlds in the other areas (test_worlds). 18k world scenarios
went into training alongside the smart home.

| set | big_names (no worlds) | big_worlds |
|---|---|---|
| test_worlds (unseen settings, known areas) | 0.442 | **0.848** |
| test_areas (5 unseen areas) | 0.436 | **0.776** |
| smart home old:test_words | 0.830 | 0.846 |
| smart home old:test_templates | 0.918 | 0.944 |
| smart home old:test_llm | 0.872 | 0.832 |
| hard cases | 20/28 | 20/28 |

Readings:
- 250 worlds nearly double accuracy on unseen settings. The smart-home sets hold, and
  test_llm drops a little.
- Accuracy on unseen areas by request kind: numbers 1.00, questions 0.98, unrelated 0.96,
  "a bit" 0.76, "a lot" 0.65, direct 0.68, slang 0.67, desire 0.64, complaint 0.50.
- Complaint errors: 21% abstain (a complaint doesn't name the device), 20% right tool but
  wrong value, 11% wrong tool (world knowledge: "the image is too dark" -> exposure).
- The framing hard cases haven't moved. The world data's desire / complaint examples don't
  yet carry over to the smart home's "make it dark".

## 2026-09-26: all 1,000 worlds

`all_worlds`: all 1,000 worlds -> 129,578 scenarios (73,603 in training, 70% of it), with the
same holdouts (split_v5: the same 5 unseen areas; worlds held out by the same name hash).
20,000 steps, double the earlier runs, since there is about twice the data. So the gain mixes
data and training length. 24 thermal pauses of 10-18 s; peak 81C.

| set | big_worlds (250 worlds, 10k steps) | all_worlds (1,000 worlds, 20k steps) |
|---|---|---|
| test_areas (5 unseen areas) | 0.798 | **0.892** |
| test_worlds (unseen settings) | 0.814 | **0.894** |
| smart home old:test_words | 0.846 | 0.876 |
| smart home old:test_templates | 0.944 | 0.938 |
| smart home old:test_llm | 0.832 | 0.822 |
| hard cases | 20/28 | 21/28 |

(big_worlds is re-scored here on split_v5's test sets, which sample more worlds.)

Unseen areas by request kind: complaint 0.50 -> 0.73, desire 0.64 -> 0.78, slang 0.67 -> 0.81,
direct 0.68 -> 0.90, "a lot" 0.65 -> 0.84, "a bit" 0.76 -> 0.95; numbers, questions and
unrelated requests 1.00.

Still wrong in the smart home: "i want it cold in here" -> warmer (the complaint reading of
"cold"), "hot as hell" and "super bright" are under-scaled, and "make it really dark" is fixed
in long names only.

## 2026-09-26: scenario descriptions: no effect, dropped

`all_worlds_scn` = all_worlds (same split, seed and 20k steps) plus a scenario description:
- worlds: "<setting>. <about>";
- smart-home houses: a generated description of their rooms and devices.

It is encoded by T (extended to 128 bytes) into 2 extra mixer tokens, and removed from 30% of
training draws. It adds 2,112 parameters.

| set | without | with scenario | same model, scenario removed at test |
|---|---|---|---|
| test_areas | 0.892 | 0.904 | 0.904 |
| test_worlds | 0.894 | 0.872 | 0.878 |
| test_llm | 0.858 | 0.890 | 0.886 |
| test_words | 0.856 | 0.878 | 0.876 |

The last column settles it. Removing the scenario from the trained model changes accuracy by
at most 0.008, so the model doesn't use the tokens. Differences between the two runs are
training noise: the extra tokens and dropout change the random stream. Likely reason: the
state and action names already carry the context (`chamber_temperature_celsius`,
`set_cooling_power`), and the description adds little beyond them.

Decision: scenarios off. The code stays (`--scenario`, off by default) and costs nothing
unless enabled. No LLM spend was involved.

## 2026-09-26: inverter (contrast variants)

`dink/llm/invert.py`: for each validated request of 7 kinds (direct, a bit, a lot,
complaint, desire, slang, no-value), three variants that keep the key words:
- opposite direction;
- complaint <-> desire reframe ("it's too cold" <-> "i want it cold");
- negation, which always means "none".

A second call labels the variants independently. Results:
- 19,380 variants, 94% agree, $1.75 for all 1,000 worlds;
- 31k scenarios after the code checks, 18k of them in training (7.4k negate, 6.0k reframe,
  4.8k opposite).

`all_inv` uses the same settings as all_worlds (20k steps, split_v6 = split_v5 + inverted
variants, same holdouts). Both models are scored on split_v6's test sets, which now include
unseen inverted variants. 30 thermal pauses.

| set | all_worlds | all_inv |
|---|---|---|
| test_areas (unseen areas) | 0.874 | **0.896** |
| test_worlds (unseen settings) | 0.880 | **0.900** |
| test_templates | 0.902 | 0.934 |
| test_styles | 0.904 | 0.954 |
| test_words | 0.852 | 0.864 |
| test_llm | 0.852 | 0.838 |
| hard cases | 21/28 | **25/28** |

Readings:
- Framing is mostly fixed. "i want it cold in here" -> cooler now works in both name
  styles, as does "make it super bright" -> full.
- Unseen areas by kind: complaint 0.73 -> 0.80, desire 0.78 -> 0.83, slang 0.81 -> 0.90,
  negations 1.00, opposite / reframe variants 0.80.
- Still wrong: "hot as hell" under-scales (+1 to +3 instead of +6), and "make it really dark"
  (long names) now flips back to lights on. It had been right in all_worlds, so this one case
  is unstable between runs.
- test_llm dips 0.014. That's within seed noise; watch it.
