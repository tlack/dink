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
