# dink-v16

The released Dink model: training run `v16_thin` (2026-11-04), 40,000 steps. Same network
and data as dink-v13, plus option thinning: in 30% of training examples one action keeps
only some of its options, so "offered once" or "lopsided" is no longer a tell. History of
every version: [docs/history.md](../../docs/history.md).

| | |
|---|---|
| parameters | 204,289 (width 128, dim 64), float32 |
| inputs | state: up to 8 fields (name, value, lo, hi); request: up to 64 bytes; up to 8 options |
| output | one probability per option, plus "nothing relevant" |
| files | `model.pt`: `{"state": state_dict, "args": training args}`; loads with `torch.load(..., weights_only=True)`. `metrics.json`: training curve and every test score below |
| sha256 (`model.pt`) | `315a632a480cb9b63fcc194c1a2d72374d8df23994eb1fe35de0544977a21bc9` |
| `dink.dnk` | the ESP32 file (623 KB): text-encoder convolutions in int8 for MMRT, the rest float32. Made by `uv run python -m dink.device.export models/dink-v16`; its int8 scales are calibrated on the validation split, so re-exporting without `data/` calibrates on the example gadgets only and may differ slightly. sha256 `4e6600b5ac2a59651330691a17a1c34782ae692375c4419d99f1550c063ab158` |

```python
from dink.play import load_model
from dink.harness import decide
model = load_model("models/dink-v16", "cpu")
```

## Training

About 154k training scenarios, sampled 40% hand-written smart-home generator, 45% LLM-invented gadgets
(1,000 gadgets in 50 areas), 15% Home Assistant voice templates. 30% of examples drop the
state (prompt-only). The text encoder starts from a distillation of Qwen3-Embedding-0.6B.
Graded targets ("close enough" amounts count as partly right) with a ranking loss (0.5), and
`--thin 0.3`. Full command in `metrics.json` (`args`); history in
[docs/results.md](../../docs/results.md).

## Test scores

"best" = picked the best answer; "ok" = picked any acceptable answer; "dir" = the right
action, moved the right way from its current value. 500 cases per set unless noted.

| held-out set | best | ok | dir |
|---|---|---|---|
| LLM gadgets from 5 areas never seen in training | 0.878 | 0.884 | 0.920 |
| LLM gadgets, unseen gadgets in known areas | 0.900 | 0.916 | 0.930 |
| smart home: held-out words / templates / naming styles | 0.848 / 0.904 / 0.902 | 0.868 / 0.916 / 0.918 | 0.910 / 0.960 / 0.956 |
| smart home: held-out rooms / units / slot counts | 0.960 / 0.990 / 0.970 | 0.976 / 0.994 / 0.976 | 0.984 / 1.000 / 0.980 |
| smart home: LLM-written phrasings | 0.824 | 0.840 | 0.888 |
| prompt-only, no state (622) | 0.966 | 0.966 | 0.966 |
| Home Assistant's hand-written test sentences (809) | 0.959 | 0.959 | 0.959 |
| Home Assistant intents never trained on (91) | 0.846 | 0.846 | 0.846 |
| direction probes: 264 made-up gadgets through the harness (`dink.evals.direction`) | | 0.81 | 0.84 |
| hand-written hard cases (38) | 0.816 | 0.816 | 0.895 |

Against dink-v13, the held-out sets move within ±0.03 either way (the noise between
training runs). Clear changes: an action offered only once is picked 77% of the time (v13:
46%), "more jets" on the example hot tub is right, Home Assistant's unseen intents went from
0.77 to 0.85, and the hard cases went from 37 to 31 of 38. Most of those are amounts: "a
little warmer" offered 71 / 73 / 76 picks 73.

## Known weaknesses

- **The right device** is most of what it gets wrong: 27 of the 44 direction-probe misses
  pick another action ("this pace is a bit much" -> incline, "the reds are too cold" -> the
  white-wine zone). Direction on the right device is rarely wrong (4 of 264).
- **Jets and sanitizer on the example hot tub** nearly tie: in PyTorch "make the sanitizer
  way stronger" goes to the jets, and on the device's int8 text encoder some jets requests
  go to the sanitizer. The C engine gives PyTorch's answer on 259 of 288 example requests
  (v13: 281); most of the rest are this pair or rounding of the amount.
- Complaints and desires ("the pressure is creeping too high") are its weakest phrasing:
  0.67 / 0.72 on unseen areas.
- A number for a missing device can land on a present one: "set the feed rate to 100" on the
  refinery heater sets the air damper to 100.
- Calibration has never been measured: treat "100%" as "confident", not "certain".
