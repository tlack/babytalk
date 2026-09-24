#!/usr/bin/env bash
# idf.py wrapper: activates ESP-IDF, then passes all args through.
#   ./idf.sh build
#   ./idf.sh -p /dev/ttyACM0 flash monitor
# IDF_PATH defaults to the v5.5.1 tree the lvgl_micropython build already uses.
set -e
export IDF_PATH="${IDF_PATH:-$HOME/build/lvgl_micropython/lib/esp-idf}"
. "$IDF_PATH/export.sh" >/dev/null 2>&1
cd "$(dirname "$0")"
exec idf.py "$@"
