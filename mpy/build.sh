#!/usr/bin/env bash
# Build MicroPython + camera API (+ mp_jpeg) + stt for the Waveshare ESP32-S3-CAM.
#
#   mpy/build.sh [FW_DIR]      FW_DIR (default ~/build/sentry-fw) holds micropython/ (v1.27.0
#                              with mpy/patches/micropython-i2s-mck.patch applied) and,
#                              optionally, micropython-camera-API/ + mp_jpeg/ (camera module).
# IDF_PATH defaults to an ESP-IDF v5.5.1 tree. Output: FW_DIR/out/firmware-stt.bin (flash at 0x0)
# plus the model image to flash at 0x410000.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
FW="${1:-$HOME/build/sentry-fw}"
export IDF_PATH="${IDF_PATH:-$HOME/build/dstike-fw/esp-idf}"
. "$IDF_PATH/export.sh" >/dev/null
PORT="$FW/micropython/ports/esp32"
# Optional TTS: sanoTTS checkout (github.com/Ampixa/sanoTTS), copied and patched by
# components/sanotts/prepare.sh.
SANOTTS_DIR="${SANOTTS_DIR:-$HOME/build/tts/sanoTTS}"
TTS_ARGS=()
EXTRA="$HERE/../components"      # shared with the AtomVM firmware: mmrt, sram_pool, stt_engine, sanotts
[ -f "$FW/micropython-camera-API/micropython.cmake" ] && EXTRA="$FW/micropython-camera-API;$EXTRA"
if "$HERE/../components/sanotts/prepare.sh" "$SANOTTS_DIR" "$FW/sanotts_src"; then
    TTS_ARGS=(-D SANOTTS_SRC="$FW/sanotts_src")
fi
BOARD=SENTRY_S3_STT
cp "$HERE/boards/$BOARD/partitions-stt.csv" "$PORT/partitions-sentry-stt.csv"
make -C "$FW/micropython/mpy-cross" -j8 >/dev/null
cd "$PORT"
idf.py -B "build-$BOARD" \
    -D MICROPY_BOARD=$BOARD -D MICROPY_BOARD_DIR="$HERE/boards/$BOARD" \
    -D USER_C_MODULES="$HERE/usermods.cmake" -D CAMERA_API_DIR="$FW/micropython-camera-API" \
    -D EXTRA_COMPONENT_DIRS="$EXTRA" "${TTS_ARGS[@]}" \
    build
mkdir -p "$FW/out"
B="build-$BOARD"
python -m esptool --chip esp32s3 merge_bin -o "$FW/out/firmware-stt.bin" --flash_mode keep --flash_freq keep --flash_size keep \
    0x0 "$B/bootloader/bootloader.bin" 0x8000 "$B/partition_table/partition-table.bin" \
    0x10000 "$B/micropython.bin"
# int8-model layout (10MB model partition): same app, flash this table at 0x8000 on top
python "$IDF_PATH/components/partition_table/gen_esp32part.py" \
    "$HERE/boards/$BOARD/partitions-stt-int8.csv" "$FW/out/partition-table-int8.bin" >/dev/null
ls -l "$FW/out/firmware-stt.bin" "$B/micropython.bin" "$FW/out/partition-table-int8.bin"
