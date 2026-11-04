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

## 2026-09-26: end-value decoy fix

Found by hand-testing: "close the entrance gate" with {0, 0.16} offered picked 0.16. In the
world data, an end value (lo or hi) of a non-switch setting was the right answer 5% of the
time vs 12% for middle values. The cause: lo and hi were always added as decoys, and
overshooting decoys were clamped onto the ends. Fix in `dink/gen/worlds.py`:
- ends are added as decoys 30% of the time;
- when the target is an end, a near-end decoy is offered;
- out-of-range decoys are dropped instead of clamped.
Balance after: ends right 12.8%, middle 10.7%.

`all_v7`: same settings as all_inv (20k steps), split_v7, which is the same data
reconverted. 143 thermal pauses, 79 minutes paused (the fan was probably not at max).
Both models are scored on split_v7's test sets.

| set | all_inv | all_v7 |
|---|---|---|
| test_areas | 0.908 | 0.914 |
| test_worlds | 0.924 | **0.948** |
| test_templates | 0.906 | 0.916 |
| test_words | 0.884 | 0.864 |
| test_llm | 0.836 | 0.808 |
| test_styles | 0.940 | 0.920 |
| test_mixed | 0.844 | 0.824 |
| hard cases (32, incl. 4 range-end cases) | 28/32 | 28/32 |

Readings:
- The gate is fixed: all 4 range-end cases pass, and "make it really dark" passes in both
  name styles.
- The smart-home language sets slip 2-3 points (llm, words, styles, mixed), in the same
  direction on all four. This may be seed noise (±0.03), or the world data crowding the
  smart home. It needs a second seed to tell.
- The intensity cases are unstable between runs. all_v7 sends "i want it hot as hell" to
  *cooler* (67) and "make it super bright" to goodnight / off; all_inv got the direction
  right. The model has no stable reading of intensifiers outside the generator's list
  ("as hell", "super", "really").

## 2026-09-27: Home Assistant templates + prompt-only mode

`dink/gen/ha.py`: Home Assistant's own English voice templates (OHF-Voice/intents,
CC-BY-4.0) -> 7.5k prompt-only scenarios over 26 intents (4.9k of them in training). They're
sampled per template block, labeled by HA's recognizer (hassil), and use hard negatives (same
device with the other verb, same verb on another device). Test sets:
- test_ha: HA's hand-written test sentences for the 26 trained intents (809);
- test_ha_unseen: 11 intents never trained (91).
Prompt-only: `--no-state 0.3` removes the whole state from 30% of draws of state-independent
request kinds (dink/evals/nostate.py); test_nostate scores that mode.

`all_v8`: split_v8 (= split_v7 + HA), 20k steps. 168 thermal pauses; the run took 2.8 h
instead of about 36 min.

| set | all_v7 | all_v8 |
|---|---|---|
| test_ha (HA's hand-written sentences) | 0.551 | **0.956** |
| test_ha_unseen (11 unseen HA intents) | 0.582 | **0.780** |
| test_nostate (prompt-only) | 0.945 | 0.957 |
| test_areas / test_worlds | 0.900 / 0.870 | 0.900 / 0.876 |
| test_llm (smart home) | 0.836 | 0.820 |
| test_words (smart home) | 0.864 | 0.850 |
| hard cases | 28/32 | 27/32 |

Readings:
- A small amount of HA data (4% of training) takes HA's own test sentences from 0.55 to 0.96.
- Unseen HA intents: fan speed, timers, vacuum-to-base, never-mind 1.00; previous track 0.86.
  List and shopping items 0-0.5, and "today's date" 0. These need a text argument (the item)
  or have no near analogue in training.
- Prompt-only was already good (0.945), and nudged up.
- Smart-home language keeps slipping as other data grows. Across the last three runs:
  test_llm 0.858 -> 0.836 -> 0.820, test_words 0.876 -> 0.864 -> 0.850. The smart home is
  now 26% of training. It needs balancing (sampling weights per source) or more smart-home
  phrasing data.

## 2026-09-27: intensity ladder + per-source balancing, 40k steps

`dink/llm/intensity.py`: the LLM writes phrasings at four strengths per device and direction;
code sets the values (8% / 20% / 45% of the range / the end); an independent judge must agree
on each phrasing's strength.
- 28,480 phrasings over 1,000 worlds and 24 smart-home rooms, 98.5% agree, $1.30;
- 55.7k scenarios, whose decoys are the other strengths.

`all_v9`: split_v9 (= v8 + intensity), 40k steps, `--mix home=0.4,world=0.45,ha=0.15`,
`--no-state 0.3`. 244 thermal pauses, 93 min paused, 2.5 h in total (fan pinned).

| set (split_v9 tests) | all_v8 | all_v9 |
|---|---|---|
| test_areas | 0.786 | **0.910** |
| test_worlds | 0.836 | **0.890** |
| test_words (smart home) | 0.802 | **0.876** |
| test_llm (smart home) | 0.836 | 0.856 |
| test_templates | 0.902 | 0.944 |
| test_styles | 0.862 | 0.958 |
| test_indist | 0.876 | 0.980 |
| test_nostate | 0.957 | 0.958 |
| test_ha | 0.956 | 0.967 |
| test_ha_unseen | 0.780 | 0.725 (n=91) |
| old:test_words / old:test_llm (split_v3 sets) | 0.834 / 0.826 | 0.876 / 0.852 |
| hard cases | 27/32 | 27/32 |

Readings:
- Balancing stopped the smart-home slide and reversed it (words 0.80 -> 0.88).
- Intensity on unseen areas: small 0.96, plain 0.96, big 0.93, max 1.00.
- But the smart-home intensity hard cases still fail, for two reasons found by probing:
  1. The model learned exact amounts. "make it way warmer" -> 76 is right, but "make it as
     hot as it goes" with the end (85) not offered -> 73, or 67 with 27%. In training, "max"
     always had the end on offer, so the model has no rule for "most extreme option offered".
  2. Sources disagree on what "a lot" means: the generator uses +6 F on a thermostat (24%)
     and +60% on lights, the LLM worlds prompts 30-60%, the intensity ladder 45%. "super
     bright" from 0.3 -> 0.6, the nearest to the ladder's 0.75; the hard case, on the
     generator's scale, expects 1.
- The gate case {0, 0.5} failed this run (0.5 for "close"); it's unstable across runs.
- test_ha_unseen dips on only 91 sentences.

## 2026-09-27: ordinal strength, and the option-count shortcut it created

`all_v10`: ordinal labels (`dink/gen/ordinal.py`). Relative requests pick among coarse offered
values by rank ("a bit" = smallest move, plain = middle, "a lot" = upper half, "all the way"
= most extreme offered), every acceptable answer is right, and the raw log-magnitude input is
gone. Split v10, 40k steps, balanced mix. 181 thermal pauses.

On split_v10 tests it looked excellent vs all_v9 (unseen areas 0.812 -> 0.910, smart-home
LLM phrasings 0.730 -> 0.812, held-out words 0.756 -> 0.846). But the hard cases fell to 22/32,
with wrong *directions* ("i'm hot" -> warmer).

Cause: the v10 generator offered 2-4 options in the asked direction and 0-1 the other way,
so the model read the direction off the counts. Probe (thermostat at 70):

| offered | "i'm hot" | "i'm freezing" |
|---|---|---|
| 3 warmer, 1 cooler | 73 (wrong) | 76 |
| 1 warmer, 3 cooler | 64 | 61 (wrong) |
| 2 and 2 | 67 | 64 (wrong) |

The test sets had the same imbalance, so they couldn't see it. Fixes:
- `offer_both()` gives both directions equal counts whenever the device can move both ways;
- `offer()` never returns nothing for a movable device (a 0..3 fan could get zero tools);
- limit requests weigh 1.5 instead of 0.5, so one-sided option sets (a device at an end)
  still need the words to be read;
- hard cases gain six count probes (38 in total).

all_v10 on the balanced split_v12 tests (ok = any acceptable pick):
unseen areas 0.898, worlds 0.870, llm 0.780, words 0.816, templates 0.906.
Hard cases: all_v10 27/38 (fails 3 count probes); all_v9 32/38 (right directions everywhere).
The graded run (`all_v11`) was stopped: it trained on the same leaky data.

## 2026-09-27: balanced ordinal data + graded targets + ranking loss

`all_v12`: split_v12 (ordinal, equal option counts both ways), graded right answers (best fit
1, other acceptable 0.5), `--rank-loss 0.5`, 40k steps, balanced mix. 195 thermal pauses.
Both models scored on split_v12's balanced tests. ok = any acceptable pick; best = the
top-weighted pick.

| set | all_v10 ok | all_v12 ok | all_v10 best | all_v12 best |
|---|---|---|---|---|
| test_areas | 0.898 | **0.930** | 0.892 | 0.926 |
| test_worlds | 0.870 | **0.914** | 0.866 | 0.906 |
| test_llm | 0.780 | **0.882** | 0.750 | 0.872 |
| test_words | 0.816 | **0.858** | 0.794 | 0.836 |
| test_templates | 0.906 | 0.938 | 0.860 | 0.908 |
| test_styles | 0.908 | 0.942 | 0.896 | 0.934 |
| test_indist | 0.920 | 0.974 | 0.912 | 0.974 |
| test_ha_unseen | 0.714 | 0.780 | | |
| hard cases (38) | 27 | **30** | | |

Readings:
- Every held-out set improves. The gain combines the balanced data and the graded targets
  with the ranking loss; this pair of runs can't separate the two.
- All six option-count probes pass: the direction now comes from the words.
- "best" sits close to "ok": when the model is acceptable, it usually also picks the
  best-fitting option.
- Still wrong: "hot as hell" -> 73 (plain; want the biggest, 76); "make it a little warmer"
  -> 73 (want the smallest, 71); "make it really dark" -> lights on; "is it dark outside yet"
  -> goodnight (a question taken as a scene).

## 2026-09-28: judged amounts cleaned (ordinal from both labelings)

Found by hand: "decrease fountain height way more" was stored as 1.3 (a small drop); the
second labeling said 0.5, but agreement compared directions only. For judged moves
(complaint, desire, slang, inverted variants), the two labelings' amounts differed by more
than 20% of the range in 6-15% of cases. Fix (`dink/gen/worlds.py`, `amount_strength`):
- judged moves take a strength (a bit / plain / a lot / all the way) from the average of both
  labelings' shares of the room left, and pick by rank;
- requests whose labelings are two strengths apart are dropped with their rewordings
  (about 2,250 utterances), and 235 more because the labelings moved in opposite directions;
- inverted variants of "a bit" / "a lot" keep their parent's strength.

`all_v13`: split_v13 (cleaned), otherwise as all_v12 (graded + rank loss 0.5, 40k steps).
153 thermal pauses. Both models scored on split_v13's tests.

| set | all_v12 ok | all_v13 ok |
|---|---|---|
| test_areas | 0.914 | 0.914 |
| test_worlds | 0.914 | 0.918 |
| test_llm | 0.878 | 0.888 |
| test_words | 0.836 | 0.832 |
| test_templates | 0.910 | 0.924 |
| test_styles | 0.912 | 0.932 |
| test_mixed | 0.860 | 0.890 |
| test_ha / test_ha_unseen | 0.964 / 0.780 | 0.939 / 0.769 |
| hard cases (38) | 30 | **37** |

Readings:
- Hard cases 37/38: "hot as hell", "really dark", "is it dark outside yet" and all count probes
  now pass. The one miss is "make it a little warmer" (long names) -> 73; the smallest move,
  71, was wanted.
- Held-out sets flat to slightly up (within noise); HA dips 2.5 points (watch).
- Weakest kinds in unseen areas: inverted reframes 0.65, complaints 0.72, desires 0.80.

## 2026-11-03: request-to-name matching, encoder anchor, looped mixer (v14)

Trigger: on the example hot tub, "more jets" never picks `set_jet_speed` (0..3, at 1).

**Matching probe** (`dink/teacher/matching.py`): does a request's nearest action name (query-0
cosine) match its LLM label? LLM worlds, held out exactly as split_v13 (areas: 3,603 requests;
worlds: 2,418), plus 10 hot tub requests.

| encoder | held-out areas | held-out worlds | hot tub |
|---|---|---|---|
| Qwen3-Embedding-0.6B (teacher) | 0.880 | 0.891 | 10/10 |
| tv128 (distilled, before Dink training) | 0.788 | 0.805 | 10/10 |
| dink-v13's encoder | 0.849 | 0.849 | 5/10 |

Dink training improves matching overall but forgets some words ("jets", "heat").

**Runs** (v13's recipe, 40k steps, rented RTX 6000 Ada):
- `v14_anchor`: + the distillation loss on the teacher corpus (0.5) and a request-to-name
  matching loss (0.5: per request, KL to the teacher's softmax over its world's names, T=0.05).
- `v14_loop`: the mixer looped up to 4x with shared weights, after arXiv 2610.12582 (NVIDIA):
  pass 1 is the plain mixer; later passes re-add the input, take a zero-initialized
  (pass, budget) scale/shift, and move the state a gated step toward the mixer output.
  Training draws 1-4 passes per step. 211K parameters.
- `v14_loop_noinj`: the same without re-adding the input.

| set | v13 | anchor | loop | loop, no inject |
|---|---|---|---|---|
| val | 0.961 | 0.957 | 0.963 | 0.963 |
| test_areas | 0.906 | 0.904 | 0.902 | 0.902 |
| test_worlds | 0.904 | 0.904 | 0.900 | 0.906 |
| test_words | 0.820 | 0.846 | 0.856 | 0.838 |
| test_llm | 0.862 | 0.858 | 0.810 | 0.834 |
| test_ha / test_ha_unseen | 0.939 / 0.769 | 0.967 / 0.703 | 0.953 / 0.648 | 0.954 / 0.747 |
| hard cases (38) | 37 | 32 | 37 | 38 |
| matching: areas / hot tub | 0.849 / 5 | **0.877 / 10** | 0.856 / 8 | 0.841 / 7 |

Looped models at fewer passes (same weights): test_areas 0.904 / 0.900 / 0.902 / 0.902 at
1-4 passes; val 0.958 -> 0.963. **Looping does nothing here**: the mixer is not short of
compute; the misses are knowledge (words, names), which more passes can't supply. Re-adding
the input makes no difference either, as 2610.12582 reports. Not pursued (it would cost
ESP32 time for nothing).

The anchor keeps the encoder's matching (hot tub 10/10) but the full model still sends "more
jets" to the sanitizer. Why, found by holding everything else fixed (anchor model, jets at 1
of 0..3):

| jet options offered | p(jets) for "more jets" |
|---|---|
| [2] (what the harness offered) | 0.00 |
| [2, 3] | 0.30 |
| [2, 0] | 0.94 |

**The option-count shortcut is back, inverted.** Since 2026-09-27 training offers the right
action in balanced pairs (as many options up as down), so the model learned "an action
offered once, or lopsided, is not the answer". On whole-number ranges the harness's 35% step
rounds back to the current value (0..3 at 1: 0.65 -> 1) and is dropped, leaving one option.
And "less jets" (1 -> 0, the only way down, a full move) loses to a moderate move on another
device: graded targets give an off-size move on the right action 0, the same as a wrong one.

Fixes: the harness now keeps a 1-unit step each way on whole-number ranges (v13: "crank the
jets up" and "faster jets" now right; examples 170 -> 175 of 288 matching the LLM label).
Training options for v15 (`dink/data/direction.py`): `--dir-credit` (right action, right
direction gets at least 0.25 of the best weight, unless the request says a number), `--thin`
(one action, right or distractor, keeps only some options), and a `dir` score in every
evaluation (right action and direction).

## 2026-11-04: direction credit + option thinning (v15)

`v15_dir`: v13's recipe + `--dir-credit 0.25 --thin 0.3`. `v15_dir_anchor`: + `--anchor 0.5
--match 0.5`. "dir" = right action and right direction (`dink/data/direction.py`).

| set (best / dir) | v13 | v15_dir | v15_dir_anchor |
|---|---|---|---|
| test_areas | 0.906 / 0.946 | 0.898 / 0.946 | 0.910 / 0.958 |
| test_worlds | 0.904 / 0.944 | 0.900 / 0.932 | 0.904 / 0.944 |
| test_words | 0.820 / 0.900 | 0.812 / 0.898 | 0.810 / 0.864 |
| test_styles | 0.922 / 0.972 | 0.882 / 0.946 | 0.894 / 0.956 |
| test_llm | 0.862 / 0.924 | 0.856 / 0.902 | 0.862 / 0.914 |
| test_mixed | 0.878 / 0.916 | 0.882 / 0.920 | 0.844 / 0.878 |
| test_ha / test_ha_unseen | 0.939 / 0.769 | 0.957 / 0.703 | 0.965 / 0.736 |
| hard cases (38) | 37 | 30 | 28 |
| example gadgets, action = LLM label (288, harness fixed) | 175 | **188** | 181 |
| hot tub jets requests (5) | 2 | **5** | 2 |

Readings:
- `v15_dir` fixes the jets end to end ("more", "less", "max", "crank", "faster jets") and
  gains 13 of 288 on the example gadgets. Held-out sets are flat on both scores (within noise).
- The cost is precision at the extremes. Hard-case misses: "super bright" -> 0.6 (1 wanted),
  "a little warmer" -> 73 (71), "warmer please" -> 76 (73), and two that leave the right
  direction: "make it really dark in here" with short names (bed_lite) -> 1, and "is it
  dark outside yet" -> goodnight. A 0.25 floor tells the model an off-size move is a quarter
  right, and "super"/"really" lose their pull toward the range end.
- The anchor on top doesn't help end to end (jets 2/5, hard 28/38).
- The hard cases have no meta["needed"], so their dir score equals best; worth adding.

## 2026-11-04: softer credit vs thinning alone (v16)

`v16_dir`: `--dir-credit 0.1` (now skipped for requests that name a size: classes ending
_small/_big/_max) + `--thin 0.3`. `v16_thin`: `--thin 0.3` only. Hard cases now carry
meta["needed"], so their dir score means something (v13: best 37/38, dir 38/38).

| set (best / dir) | v13 | v15_dir | v16_dir | v16_thin |
|---|---|---|---|---|
| test_areas | 0.906 / 0.946 | 0.898 / 0.946 | 0.868 / 0.920 | 0.878 / 0.920 |
| test_worlds | 0.904 / 0.944 | 0.900 / 0.932 | 0.900 / 0.934 | 0.900 / 0.930 |
| test_words | 0.820 / 0.900 | 0.812 / 0.898 | 0.834 / 0.914 | 0.848 / 0.910 |
| test_styles | 0.922 / 0.972 | 0.882 / 0.946 | 0.924 / 0.970 | 0.902 / 0.956 |
| test_llm | 0.862 / 0.924 | 0.856 / 0.902 | 0.844 / 0.918 | 0.824 / 0.888 |
| test_ha / test_ha_unseen | 0.939 / 0.769 | 0.957 / 0.703 | 0.960 / 0.747 | 0.959 / 0.846 |
| hard cases (38), best / dir | 37 / 38 | 30 / 30* | 30 / 35 | 31 / 34 |
| example gadgets (288) | 175 | **188** | 176 | 180 |
| hot tub jets (5) | 2 | 5 | 5 | 4 |

\* v15 was scored before the hard cases had meta["needed"].

Readings:
- **Thinning is what fixes the jets** (alone: 4/5; it misses "max jets"). The credit adds
  "max jets" and nothing measurable elsewhere.
- **The hard-case losses come with thinning, not credit**: the same misses in every
  thinning run. Most are amounts read by size rather than rank: "a little warmer" (70 on
  60..85, offered 71/73/76) -> 73, "warmer please" -> 76. By IDEAL fractions (small 0.15,
  plain 0.35 of the room) 73 and 76 are the closer sizes; the hard cases expect rank (the
  smallest / middle option). Thinning breaks rank as a cue, so the model reads size.
- The wrong-direction misses are "make it really dark in here" -> light 1 (both name
  styles; light at 0.3, offered 0 / 0.6 / 1) and "is it dark outside yet" -> goodnight.
- Held-out sets move within ±0.03 both ways (test_areas down, words and HA up): single
  seeds can't separate these runs. Decisions here rest on 19 hand-written prompts and 5
  jets requests.

## 2026-11-04: direction probes (264 made-up gadgets, run through the harness)

The hard cases are 19 prompts on one bedroom. `dink/evals/direction.py` scores 264 cases,
each with its own small gadget (`dink/evals/direction_cases.jsonl`): 243 written by the LLM
(`dink/llm/probes.py`, 12 themes, ~70 kinds of device, reviewed by hand: 33 of 276 dropped
as wrong or ambiguous, 6 fixed) plus 21 by hand (the 1..3 fan, the hot tub, the bedroom).
Each runs through the two-pass harness, as on a device. Scoring follows "most correct in
the right direction":

- **dir**: the right action (or any action setting the same field), moved the right way;
  nothing for questions, and nothing is fine when the value is already at the end asked for.
- **full**: dir, and the amount fits by *rank* among the options offered that way: "a
  little" is not the farthest option (on 1..3 at 3, "a little lower" is 2, not 1; at 1,
  nothing is fine), "a lot" is not the nearest, "all the way" is the end, a number said is
  that number.
- **lone**: the raw model on pass-1 options with the right action offered once (its
  other-way option dropped), 218 cases.

| (264 cases) | v13 | v14_anchor | v15_dir | v16_dir | v16_thin |
|---|---|---|---|---|---|
| dir | 0.81 | 0.80 | 0.84 | 0.83 | 0.84 |
| full | 0.80 | 0.78 | 0.82 | 0.81 | 0.81 |
| lone | **0.46** | 0.48 | 0.76 | 0.74 | **0.77** |
| misses: wrong device | 34 | 33 | 23 | 31 | 27 |
| misses: right device, wrong way | 7 | 9 | 11 | 4 | 4 |
| misses: did nothing / acted on a question | 7 / 2 | 7 / 3 | 5 / 4 | 6 / 4 | 9 / 3 |
| misses: amount | 4 | 5 | 4 | 4 | 6 |

Paired against v13 on the same cases: dir +24 / -17 for v16_thin (not significant), lone
+75 / -9 (clearly real; v15_dir +74 / -9, v16_dir +69 / -8).

Readings:
- **Direction is mostly fine; the device is the problem.** Wrong device is 60-70% of the
  misses in every run, and the harness offered the right action every time (trimming to 8
  options never dropped it). Examples: "this pace is a bit much" -> incline, "this water is
  lukewarm" -> water level, "the reds are too cold" -> the white-wine zone, "the top rack
  is drowning" -> lower rinse, "it's blazing in there" -> ignite.
- **Thinning fixes a real shortcut**: v13 picks an action offered once only 46% of the time.
  That shortcut is what "more jets" hit; it's general, not about jets.
- The hard-case regressions from thinning (amounts read by size) don't show here: scored by
  rank, amounts are 4-6 misses in every run.
- Right-way misses left on the right device are few and scattered: "less smoke" -> up,
  "the glasses come out soaking wet" -> dry time 0, "make it really dark in here" -> 1.
