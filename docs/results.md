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
