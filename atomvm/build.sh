#!/usr/bin/env bash
# Build AtomVM v0.7.0-beta.0 with the BabyTalk components, for an ESP32-S3 or an ESP32-P4.
#
#   atomvm/build.sh [TARGET]      (or TARGET=... atomvm/build.sh)
#
#   esp32s3          the default: any ESP32-S3 with 16 MB flash and 8 MB octal PSRAM (Waveshare ESP32-S3-CAM)
#   esp32p4_pre_c6   ESP32-P4 silicon before v3.0 + an ESP32-C6 for WiFi (Waveshare ESP32-P4-WIFI6 so far)
#   esp32p4_c6       ESP32-P4 v3.0 or later + an ESP32-C6 for WiFi
#   esp32p4_pre      ESP32-P4 before v3.0, no WiFi
#   esp32p4          ESP32-P4 v3.0 or later, no WiFi
# (the P4 names are AtomVM's own presets; esptool and the boot log show the chip's revision)
#
# -> $OUT/atomvm-babytalk.img (flash at 0x0: bootloader, partition table, AtomVM, boot.avm).
# AVM_DIR holds an AtomVM checkout at the pinned tag (default ~/build/atomvm-<tag>[-<TARGET>]:
# one tree per release and target, as switching either rebuilds everything);
# OUT_DIR the results (default ~/build/atomvm-out, ~/build/atomvm-out-<TARGET>); IDF_PATH an
# ESP-IDF 5.5 tree. The Erlang/Elixir boot libraries (boot.avm) come from AtomVM's release
# image for the same target, so no host build of AtomVM (and no gperf) is needed.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${AVM_TAG:-v0.7.0-beta.0}"    # AVM_TAG: another AtomVM release (patches/ must still apply)
TARGET="${1:-${TARGET:-esp32s3}}"
case "$TARGET" in
    esp32s3) CHIP=esp32s3; SUFFIX="" ;;
    esp32p4|esp32p4_pre|esp32p4_c6|esp32p4_pre_c6) CHIP=esp32p4; SUFFIX="-$TARGET" ;;
    *) echo "unknown target '$TARGET': esp32s3, esp32p4_pre_c6, esp32p4_c6, esp32p4_pre or esp32p4" >&2; exit 2 ;;
esac
AVM="${AVM_DIR:-$HOME/build/atomvm-$TAG$SUFFIX}"
OUT="${OUT_DIR:-$HOME/build/atomvm-out$SUFFIX}"
export IDF_PATH="${IDF_PATH:-$HOME/build/lvgl_micropython/lib/esp-idf}"
. "$IDF_PATH/export.sh" >/dev/null

[ -d "$AVM" ] || git clone --depth 1 --branch "$TAG" https://github.com/atomvm/AtomVM.git "$AVM"
[ "$(git -C "$AVM" describe --tags)" = "$TAG" ] || { echo "$AVM is not at $TAG (set AVM_DIR to another directory)"; exit 1; }

# Local fixes to AtomVM (atomvm/patches; atomvm/README.md, "AtomVM patches"), applied once
for p in "$HERE"/patches/*.patch; do
    git -C "$AVM" apply --reverse --check "$p" 2>/dev/null || git -C "$AVM" apply "$p"
done

E="$AVM/src/platforms/esp32"
for c in "$HERE"/components/*/; do ln -sfn "$c" "$E/components/$(basename "$c")"; done
# Engine components shared with the MicroPython firmware (repo-root components/)
for c in mmrt sram_pool stt_engine sanotts; do ln -sfn "$HERE/../components/$c" "$E/components/$c"; done
# Our partition table (AtomVM's CMakeLists would otherwise impose its own: patches/0004)
TABLE=partitions-babytalk.csv
cp "$HERE/$TABLE" "$E/"
if [ "$CHIP" = esp32p4 ]; then
    # AtomVM's own recipe (its CI): the variant's preset, and for a C6 co-processor
    # esp_wifi_remote (bringing esp_hosted), so AtomVM's network driver reaches the C6's WiFi
    # through the usual esp_wifi API
    cp "$E/sdkconfig.defaults.$TARGET" "$E/sdkconfig.defaults.babytalk-variant"
    case "$TARGET" in
        *_c6) cp "$E/components/avm_builtins/idf_component.yml.esp32p4_wifi_remote" "$E/components/avm_builtins/idf_component.yml" ;;
    esac
fi

# The defaults: AtomVM's (ESP-IDF adds its .<chip> file), the P4 variant's preset, ours
# (ESP-IDF adds sdkconfig.babytalk.<chip>). Re-applied whenever ours change (set-target
# regenerates sdkconfig from the defaults).
DEFAULTS="sdkconfig.defaults"
[ "$CHIP" = esp32p4 ] && DEFAULTS="$DEFAULTS;sdkconfig.defaults.babytalk-variant"
DEFAULTS="$DEFAULTS;$HERE/sdkconfig.babytalk"
cd "$E"
stamp="$(echo "$TARGET" | cat - "$HERE"/sdkconfig.babytalk* | sha1sum | cut -c1-12)"
if [ ! -f sdkconfig ] || [ "$(cat .babytalk-overlay 2>/dev/null)" != "$stamp" ]; then
    rm -f sdkconfig
    idf.py -D SDKCONFIG_DEFAULTS="$DEFAULTS" -D AVM_PARTITION_TABLE_FILENAME="$TABLE" set-target "$CHIP"
    echo "$stamp" > .babytalk-overlay
fi
# Text to speech: sanoTTS sources (github.com/Ampixa/sanoTTS) copied and patched; its static
# buffers go to PSRAM (internal RAM is short next to AtomVM + WiFi)
mkdir -p "$OUT"
TTS_ARGS=()
rc=0; "$HERE/../components/sanotts/prepare.sh" "${SANOTTS_DIR:-$HOME/build/tts/sanoTTS}" "$OUT/sanotts_src" || rc=$?
case $rc in
    0) TTS_ARGS=(-D SANOTTS_SRC="$OUT/sanotts_src" -D SANOTTS_BSS_PSRAM=1) ;;
    1) echo "no sanoTTS checkout: building without text to speech (babytalk:say/1 will return {error, -4})" >&2 ;;
    *) exit $rc ;;                                   # wrong sanoTTS commit
esac
idf.py -D SDKCONFIG_DEFAULTS="$DEFAULTS" -D AVM_PARTITION_TABLE_FILENAME="$TABLE" "${TTS_ARGS[@]}" build

# boot.avm: the Erlang + Elixir standard libraries, cut from AtomVM's release image for this
# target (where that image's own partition table puts boot.avm)
REL="$OUT/AtomVM-$TARGET-elixir-$TAG.img"
[ -s "$REL" ] || curl -fsL -o "$REL" \
    "https://github.com/atomvm/AtomVM/releases/download/$TAG/AtomVM-$TARGET-elixir-$TAG.img"
python3 - "$REL" "$OUT/boot.avm" <<'PY'
import struct, sys
img = open(sys.argv[1], "rb").read()
# The image starts where its bootloader goes: flash 0x0 on the S3, 0x2000 on the P4. The
# partition table (32-byte entries) is at flash 0x8000 either way.
start = 0 if img[0x8000:0x8002] == b"\xaa\x50" else 0x2000
for i in range(0x8000 - start, 0x8C00 - start, 32):
    magic, _type, _sub, off, size, label = struct.unpack("<HBBII16s", img[i:i + 28])
    if magic != 0x50AA:
        sys.exit("no boot.avm partition in the release image")
    if label.rstrip(b"\0") == b"boot.avm":
        break
boot = img[off - start:off - start + size].rstrip(b"\xff")
assert len(boot) > 1000, "boot.avm is empty in the release image"
open(sys.argv[2], "wb").write(boot)
print(f"boot.avm: {len(boot)} bytes (at {off:#x} in the release image)")
PY

# One image from 0x0: ESP-IDF's own flash layout (the bootloader at 0x0 on the S3, 0x2000 on
# the P4), plus boot.avm at our partition table's offset
BOOT_OFF="$(awk -F, '$1 ~ /^boot\.avm/ {gsub(/ /, "", $4); print $4}' "$HERE/partitions-babytalk.csv")"
(cd build && python -m esptool --chip "$CHIP" merge_bin -o "$OUT/atomvm-babytalk.img" \
    @flash_args "$BOOT_OFF" "$OUT/boot.avm")
ls -l "$OUT/atomvm-babytalk.img" build/atomvm-esp32.bin
echo "flash: esptool.py --chip $CHIP write_flash 0x0 $OUT/atomvm-babytalk.img   (the model goes to 0x490000, apps to main.avm at 0xA90000)"
