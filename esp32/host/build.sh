#!/usr/bin/env bash
# Build Dink's engine and MMRT's portable kernels as a PC shared library, for the parity check:
#   esp32/host/build.sh && uv run python -m dink.device.check_c esp32/host/libdink.so models/dink-v13
set -e
H="$(cd "$(dirname "$0")" && pwd)"
M="$H/../babytalk/mmrt"
D="$H/../components/dink"
[ -f "$M/mmrt_f32.c" ] || { echo "MMRT not found: git submodule update --init esp32/babytalk" >&2; exit 2; }
gcc -O2 -Wall -Wextra -shared -fPIC -I"$M" -I"$D" -o "$H/libdink.so" \
    "$M/mmrt.c" "$M/mmrt_ref.c" "$M/mmrt_port.c" "$M/mmrt_f32.c" "$D/dink.c" "$D/dink_harness.c" -lm
echo "-> $H/libdink.so"
