# dink-v13

The released Dink model: training run `all_v13` (2026-09-27), 40,000 steps.

| | |
|---|---|
| parameters | 204,289 (width 128, dim 64), float32 |
| inputs | state: up to 8 fields (name, value, lo, hi); request: up to 64 bytes; up to 8 options |
| output | one probability per option, plus "nothing relevant" |
| files | `model.pt`: `{"state": state_dict, "args": training args}`; loads with `torch.load(..., weights_only=True)`. `metrics.json`: training curve and every test score below |
| sha256 (`model.pt`) | `cd5051dd4d8605457ad68f5b73bc03aab08840f39d5ff9343d3b85f89497fb9e` |
| `dink.dnk` | the ESP32 file (623 KB): text-encoder convolutions in int8 for MMRT, the rest float32. Made by `uv run python -m dink.device.export models/dink-v13`; its int8 scales are calibrated on the validation split, so re-exporting without `data/` calibrates on the example gadgets only and may differ slightly. sha256 `63df7c7ef69072d0e4e347e1bfd67512b69dc99421fe3e724d084a798b29c65f` |

```python
from dink.play import load_model
from dink.harness import decide
model = load_model("models/dink-v13", "cpu")
```

## Training

About 154k training scenarios, sampled 40% hand-written smart-home generator, 45% LLM-invented gadgets
(1,000 gadgets in 50 areas), 15% Home Assistant voice templates. 30% of examples drop the
state (prompt-only). The text encoder starts from a distillation of Qwen3-Embedding-0.6B.
Graded targets ("close enough" amounts count as partly right) with a ranking loss (0.5).
Full command in `metrics.json` (`args`); history in [docs/results.md](../../docs/results.md).

## Test scores

"best" = picked the best answer; "ok" = picked any acceptable answer. 500 cases per set
unless noted.

| held-out set | best | ok |
|---|---|---|
| LLM gadgets from 5 areas never seen in training | 0.906 | 0.914 |
| LLM gadgets, unseen gadgets in known areas | 0.904 | 0.918 |
| smart home: held-out words / templates / naming styles | 0.820 / 0.902 / 0.922 | 0.832 / 0.924 / 0.932 |
| smart home: held-out rooms / units / slot counts | 0.948 / 0.990 / 0.968 | 0.966 / 0.992 / 0.972 |
| smart home: LLM-written phrasings | 0.862 | 0.888 |
| prompt-only, no state (622) | 0.966 | 0.966 |
| Home Assistant's hand-written test sentences (809) | 0.939 | 0.939 |
| Home Assistant intents never trained on (91) | 0.769 | 0.769 |
| hand-written hard cases (38) | 0.974 | 0.974 |

## Known weaknesses

- Words it never learned: on the example hot tub, "more jets" never picks `set_jet_speed`
  (it picks the sanitizer, or "nothing relevant" when only jets and ozone are offered).
  Saying the number works: "set jet speed to 2" is right.
- Complaints and desires ("the pressure is creeping too high") are its weakest phrasing:
  0.72 / 0.80 on unseen areas.
- Amounts: "a bit" sometimes picks a medium step. Training counts one strength step off as
  acceptable, so the scores above don't penalize it.
- A number for a missing device can land on a present one: "set the feed rate to 100" on a
  heater with only `set_fuel_gas_flow` sets the fuel gas to 100.
- Calibration has never been measured: treat "100%" as "confident", not "certain".
