# Model history

How Dink got here, one line per step. The numbers, tables and full readings are in the lab
notebook, [results.md](results.md). Scores from different rows are often on different test
splits, so compare within a row's own entry, not down this page.

| date | run | what changed | what we learned |
|---|---|---|---|
| 2026-09-25 | | [teacher choice](results.md#2026-09-25-teacher-choice) | Qwen3-Embedding-0.6B beats e5 and bge on held-out word direction and abstention |
| 2026-09-25 | tv128 | [text encoder distilled](results.md#2026-09-25-first-distillation-into-the-text-encoder) from the teacher, [plus value descriptions](results.md#2026-09-25-value-distillation-t--f) | 74K parameters keep the teacher's in-domain vs off-domain split, but lose composed phrases ("less bright" -> brighter) |
| 2026-09-25 | b3k_* | [first Dink runs](results.md#2026-09-25-first-dink-runs-204k-parameters-3000-steps-batch-64): 204K parameters, 1,000 scenarios | held-out words and templates are the hard part |
| 2026-09-25 | big_tmpl, big_llm | [more data, on the GPU; LLM phrasings](results.md#2026-09-25-more-data-on-the-gpu-llm-phrasings-as-prompts) | variety in the training prompts is the lever, not volume: held-out LLM phrasings 0.54 -> 0.89 |
| 2026-09-26 | big_names, big_worlds | [descriptive names; LLM-invented gadgets](results.md#2026-09-26-descriptive-names-llm-invented-worlds) | gadgets from whole unseen areas become learnable |
| 2026-09-26 | all_worlds | [all 1,000 LLM gadgets](results.md#2026-09-26-all-1000-worlds) | unseen areas 0.80 -> 0.89 |
| 2026-09-26 | all_worlds_scn | [scenario descriptions](results.md#2026-09-26-scenario-descriptions-no-effect-dropped) | no effect; dropped |
| 2026-09-26 | all_inv | [contrast variants](results.md#2026-09-26-inverter-contrast-variants) (opposite, complaint, negation) | helps unseen areas |
| 2026-09-26 | all_v7 | [end-value decoys](results.md#2026-09-26-end-value-decoy-fix) | "close it" picks the end, not a near-end value |
| 2026-09-27 | all_v8 | [Home Assistant templates, prompt-only mode](results.md#2026-09-27-home-assistant-templates--prompt-only-mode) | works with no state at all |
| 2026-09-27 | all_v9 | [intensity ladder, per-source balancing, 40k steps](results.md#2026-09-27-intensity-ladder--per-source-balancing-40k-steps) | unseen areas 0.79 -> 0.91; balancing the sources stops the smart-home slide |
| 2026-09-27 | all_v10, all_v11 | [ordinal strength](results.md#2026-09-27-ordinal-strength-and-the-option-count-shortcut-it-created) | taught a shortcut: direction read off how many options point each way |
| 2026-09-27 | all_v12 | [balanced options, graded targets, ranking loss](results.md#2026-09-27-balanced-ordinal-data--graded-targets--ranking-loss) | the shortcut is gone; graded "close enough" amounts |
| 2026-09-28 | **dink-v13** (all_v13) | [judged amounts cleaned](results.md#2026-09-28-judged-amounts-cleaned-ordinal-from-both-labelings) | first public release |
| 2026-11-03 | v14_* | [request-to-name matching, encoder anchor, looped mixer](results.md#2026-11-03-request-to-name-matching-encoder-anchor-looped-mixer-v14) | looping gives nothing at this size; the anchor fixes name matching in the encoder but not end to end; the harness's whole-number rounding fixed |
| 2026-11-04 | v15_* | [direction credit, option thinning](results.md#2026-11-04-direction-credit--option-thinning-v15) | thinning fixes "more jets" |
| 2026-11-04 | v16_* | [thinning alone vs softer credit](results.md#2026-11-04-softer-credit-vs-thinning-alone-v16) | thinning does the work; credit adds nothing measurable |
| 2026-11-04 | | [direction probes](results.md#2026-11-04-direction-probes-264-made-up-gadgets-run-through-the-harness): 264 made-up gadgets | direction is mostly right; picking the right device is the main weakness; an action offered once: v13 46%, v16 77% |
| 2026-11-05 | **dink-v16** (v16_thin) | released | [model card](../models/dink-v16/README.md) |

Old releases stay in git: `git checkout dink-v13 -- models/dink-v13` restores v13's files
(the tag marks the last commit that shipped it).
