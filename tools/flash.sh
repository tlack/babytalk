#!/usr/bin/env bash
# Flash only the intended board: reads the chip's MAC first and refuses to write if it isn't
# the board configured for this repo (`mac = ...` in board.conf, or $BOARD_MAC). Several
# boards may take turns on the same serial port -- this is the guard against flashing the
# wrong one.
#
#   tools/flash.sh 0x0 ~/build/atomvm-out/atomvm-babytalk.img 0x490000 models/citrinet256_int4.mmrt
#   PORT=/dev/ttyACM1 tools/flash.sh 0xA90000 atomvm/apps/listen_demo/listen_demo.avm
#
# The arguments are esptool write_flash's: offset/file pairs (and write_flash options).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-/dev/ttyACM0}"
want="${BOARD_MAC:-$(python3 -c "import sys; sys.path.insert(0, '$HERE'); from board_config import conf_value; print(conf_value('mac') or '')")}"
if [ -z "$want" ]; then
    echo "no board MAC configured: put \`mac = aa:bb:cc:dd:ee:ff\` in board.conf (see board.conf.example)" >&2
    echo "or set BOARD_MAC; get it with: esptool.py --port $PORT read_mac" >&2
    exit 2
fi
[ $# -ge 2 ] || { sed -n '2,11p' "$0" >&2; exit 2; }
command -v esptool.py >/dev/null || [ -n "$IDF_PATH" ] || . "${IDF_PATH:-$HOME/build/lvgl_micropython/lib/esp-idf}/export.sh" >/dev/null
have=$(python -m esptool --chip esp32s3 -p "$PORT" read_mac 2>/dev/null | sed -n 's/^MAC: *//p' | head -1)
norm() { echo "$1" | tr 'A-F' 'a-f' | tr -d ' '; }
if [ -z "$have" ]; then
    echo "could not read a MAC on $PORT (no board, or the port is busy): not flashing" >&2
    exit 3
fi
if [ "$(norm "$have")" != "$(norm "$want")" ]; then
    echo "WRONG BOARD on $PORT: MAC $have, expected $want -- not flashing" >&2
    exit 4
fi
echo "board $have on $PORT: flashing"
python -m esptool --chip esp32s3 -p "$PORT" -b 921600 write_flash "$@"
