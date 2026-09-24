#!/usr/bin/env bash
# Build MicroPython + camera API (+ mp_jpeg) + stt for the Waveshare ESP32-S3-CAM.
#
#   mpy/build.sh [FW_DIR]      FW_DIR (default ~/build/sentry-fw) holds micropython/ (v1.27.0,
#                              I2S-MCK patch), micropython-camera-API/ (fb_location patch),
#                              mp_jpeg/ -- see watchtower firmware/waveshare_s3_cam/BUILD.md.
# IDF_PATH defaults to an ESP-IDF v5.5.1 tree. Output: FW_DIR/out/firmware-stt.bin (flash at 0x0)
# plus the model image to flash at 0x310000.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
FW="${1:-$HOME/build/sentry-fw}"
export IDF_PATH="${IDF_PATH:-$HOME/build/dstike-fw/esp-idf}"
. "$IDF_PATH/export.sh" >/dev/null
PORT="$FW/micropython/ports/esp32"
BOARD=SENTRY_S3_STT
cp "$HERE/boards/$BOARD/partitions-stt.csv" "$PORT/partitions-sentry-stt.csv"
make -C "$FW/micropython/mpy-cross" -j8 >/dev/null
cd "$PORT"
idf.py -B "build-$BOARD" \
    -D MICROPY_BOARD=$BOARD -D MICROPY_BOARD_DIR="$HERE/boards/$BOARD" \
    -D USER_C_MODULES="$HERE/usermods.cmake" -D CAMERA_API_DIR="$FW/micropython-camera-API" \
    -D EXTRA_COMPONENT_DIRS="$FW/micropython-camera-API;$HERE/components" \
    build
mkdir -p "$FW/out"
B="build-$BOARD"
python -m esptool --chip esp32s3 merge_bin -o "$FW/out/firmware-stt.bin" --flash_mode keep --flash_freq keep --flash_size keep \
    0x0 "$B/bootloader/bootloader.bin" 0x8000 "$B/partition_table/partition-table.bin" \
    0x10000 "$B/micropython.bin"
ls -l "$FW/out/firmware-stt.bin" "$B/micropython.bin"
