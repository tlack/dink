#!/usr/bin/env bash
# idf.py wrapper: activates ESP-IDF (v5.5; set IDF_PATH), then passes all args through.
#   ./idf.sh build
#   ./idf.sh -p /dev/ttyACM0 monitor
set -e
[ -n "$IDF_PATH" ] || { echo "set IDF_PATH to your ESP-IDF v5.5 checkout" >&2; exit 2; }
. "$IDF_PATH/export.sh" >/dev/null 2>&1
cd "$(dirname "$0")"
exec idf.py "$@"
