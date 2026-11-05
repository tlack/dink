#!/usr/bin/env bash
# Flash the Dink test app and the model through BabyTalk's tools/flash.sh, which reads the
# board's MAC first and refuses to write to any other board: set BOARD_MAC=aa:bb:cc:dd:ee:ff
# (esptool.py --port /dev/ttyACM0 read_mac). PORT=/dev/ttyACM1 picks another port.
#   ./flash.sh [path/to/dink.dnk]      default: models/dink-v16/dink.dnk
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$(realpath "${1:-$HERE/../../models/dink-v16/dink.dnk}")"
[ -f "$MODEL" ] || { echo "no model at $MODEL" >&2; exit 2; }
[ -f "$HERE/build/flash_args" ] || { echo "build first: ./idf.sh build" >&2; exit 2; }
[ -n "$IDF_PATH" ] && . "$IDF_PATH/export.sh" >/dev/null 2>&1
cd "$HERE/build"
# shellcheck disable=SC2046
exec "$HERE/../babytalk/tools/flash.sh" $(tr '\n' ' ' < flash_args) 0x210000 "$MODEL"
