# Dink on the ESP32-S3

Dink's C engine and a test firmware. Dink builds on **MMRT**, the int8 inference runtime from
[BabyTalk](https://github.com/tlack/babytalk) (on-device speech to text), pulled in as the
`babytalk` submodule:

- the text encoder's convolutions are a `.mmrt` graph, run by MMRT's ESP32-S3 vector kernels;
- the dense float32 layers (pooling, the item encoder, the 2-layer attention mixer) use MMRT's
  `mmrt_f32.h` ops, which read each weight row from flash once per layer;
- Dink's own code ([`components/dink/`](components/dink)) is the wiring, the `.dnk` file, the
  two-pass harness and number extraction, ported line for line from `dink/harness.py` and
  `dink/data/numbers.py`.

## Measured on an ESP32-S3

ESP32-S3 at 240 MHz, 16 MB flash, no PSRAM. The model file (623 KB) is read in place from
flash. Measured with dink-v13; dink-v16 is the same network, so timing and memory are the
same, but its answers on the board haven't been re-checked yet (the PC build of the same C
engine has, below).

| | |
|---|---|
| one request, end to end (both harness passes) | **92 ms** average, 123 ms worst, over 44 requests |
| encode a request (64 bytes, MMRT int8) | 15 ms (109 ms with MMRT's portable C) |
| encode a name (once per gadget, then cached) | 9 ms |
| score 8 options against 8 state fields (float32) | 57 ms |
| working memory | 59 KB of internal RAM, plus MMRT's 72 KB staging buffers |
| app size | 250 KB |

Answers matched the PC (dink-v13): on 44 requests to the four example gadgets (`examples/gadgets/`), the
board gave the same action and value as the PyTorch harness on 42. The two differences come
from the int8 text encoder (one of them lands on the LLM's labeled answer where PyTorch
didn't) and from float rounding (3.82 vs 3.83). MMRT's S3 kernels give bit-identical results
to its portable reference on Dink's layers.

```
dink> set the water to one hundred degrees
  -> set_temperature=100  100%  (68 ms, 1 pass)
dink> make it cooler
  -> set_temperature=89  55%  (119 ms, 2 passes)
dink> turn on the ozone
  -> turn_on_ozone()  100%  (69 ms, 1 pass)
dink> what's the weather
  -> (nothing)  100%  (69 ms, 1 pass)
```

## Build and run

Needs ESP-IDF v5.5 and an ESP32-S3 with 4 MB+ flash.

```bash
git submodule update --init esp32/babytalk
export IDF_PATH=~/esp/esp-idf                         # your ESP-IDF v5.5
cd esp32/firmware
./idf.sh build
BOARD_MAC=aa:bb:cc:dd:ee:ff ./flash.sh                # app + models/dink-v16/dink.dnk at 0x210000
./idf.sh -p /dev/ttyACM0 monitor
```

`flash.sh` goes through BabyTalk's `tools/flash.sh`, which reads the board's MAC first and
refuses to write to any other board (`esptool.py --port /dev/ttyACM0 read_mac` shows it).

The firmware runs every example request, printing each answer next to the LLM's label and
PyTorch's answer, then reads requests from the USB console (`:g N` switches gadget). The
example gadgets are compiled in from `firmware/main/gadgets.h`, made by
`uv run python -m dink.device.c_gadgets models/dink-v16 --out esp32/firmware/main/gadgets.h`.

## Using it in your app

Add `components/dink` and `babytalk/components/mmrt` to your project's
`EXTRA_COMPONENT_DIRS`, put `dink.dnk` in a flash partition, then:

```c
#include "dink.h"

dink_t *d = dink_open(image, size, alloc16, release);   // image: the mmapped partition

dink_field_t fields[] = {{"water_temperature", 102, 80, 104}, {"jet_pump_speed", 1, 0, 3}};
dink_action_t actions[] = {
    {"set_temperature", 80, 104, 1, "water_temperature"},
    {"set_jet_speed", 0, 3, 1, "jet_pump_speed"},
    {"turn_on_ozone", 0, 0, 0, NULL},
};
dink_gadget_t g = {.fields = fields, .n_fields = 2, .actions = actions, .n_actions = 3};
dink_gadget_prepare(d, &g);                              // encodes the names once

fields[0].value = read_temperature();                    // values can change between requests
dink_decision_t r;
dink_decide(d, &g, "it's way too hot", &r);              // r.action (-1 = nothing), r.value, r.p, r.unsure
```

`alloc16` must return 16-byte aligned memory (MMRT's vector kernels), e.g.
`heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL)`. `dink_score()` scores options you choose
yourself, if you'd rather not use the two-pass harness.

## Checking the C against Python

```bash
esp32/host/build.sh                                       # the engine + MMRT, for the PC
uv run python -m dink.device.check_c esp32/host/libdink.so models/dink-v16
```

| check | result |
|---|---|
| number extraction, every test prompt | 4,899 / 4,899 agree |
| the network, against `dink/device/reference.py` with the same int8 arithmetic | 341 / 341 picks, probabilities within 1e-5 |
| the two-pass harness, against PyTorch | 259 / 288 requests (dink-v16; v13: 281). Most differences are jets vs sanitizer on the hot tub, a near-tie that int8 rounding tips, or the amount rounded differently |

## Next

- The float32 layers are now limited by arithmetic, at about 6 cycles per multiply-add.
  esp-dsp's dot products, or int8 weights for the mixer on MMRT's int8 kernels, should
  roughly halve a request.
- Pass 2 (how much) only matters for relative amounts; requests that say a number already
  skip it.
