#!/usr/bin/env bash
# Build AtomVM v0.7.0-alpha.1 for the ESP32-S3 with the BabyTalk components.
#
#   atomvm/build.sh            -> ~/build/atomvm-out/atomvm-babytalk.img (flash at 0x0)
#
# AVM_DIR (default ~/build/atomvm) holds an AtomVM checkout at the pinned tag; IDF_PATH
# defaults to the ESP-IDF 5.5 tree the rest of this repo builds with. The Erlang/Elixir boot
# libraries (boot.avm) come from the official esp32s3 Elixir release image, so no host build
# of AtomVM (and no gperf) is needed.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG=v0.7.0-alpha.1
AVM="${AVM_DIR:-$HOME/build/atomvm}"
OUT="${OUT_DIR:-$HOME/build/atomvm-out}"
export IDF_PATH="${IDF_PATH:-$HOME/build/lvgl_micropython/lib/esp-idf}"
. "$IDF_PATH/export.sh" >/dev/null

[ -d "$AVM" ] || git clone --depth 1 --branch "$TAG" https://github.com/atomvm/AtomVM.git "$AVM"
[ "$(git -C "$AVM" describe --tags)" = "$TAG" ] || { echo "$AVM is not at $TAG"; exit 1; }

# Local fixes to AtomVM (atomvm/patches), applied once
for p in "$HERE"/patches/*.patch; do
    git -C "$AVM" apply --reverse --check "$p" 2>/dev/null || git -C "$AVM" apply "$p"
done

E="$AVM/src/platforms/esp32"
for c in "$HERE"/components/*/; do ln -sfn "$c" "$E/components/$(basename "$c")"; done
# Engine components shared with the MicroPython firmware (repo-root components/)
for c in mmrt sram_pool stt_engine sanotts; do ln -sfn "$HERE/../components/$c" "$E/components/$c"; done
cp "$HERE/partitions-babytalk.csv" "$E/"

# Re-apply our overlay whenever it changes (set-target regenerates sdkconfig from defaults)
cd "$E"
stamp="$(sha1sum "$HERE/sdkconfig.babytalk" | cut -c1-12)"
if [ ! -f sdkconfig ] || [ "$(cat .babytalk-overlay 2>/dev/null)" != "$stamp" ]; then
    rm -f sdkconfig
    idf.py -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;$HERE/sdkconfig.babytalk" set-target esp32s3
    echo "$stamp" > .babytalk-overlay
fi
# Text to speech: sanoTTS sources (github.com/Ampixa/sanoTTS) copied and patched; its static
# buffers go to PSRAM (internal RAM is short next to AtomVM + WiFi)
mkdir -p "$OUT"
TTS_ARGS=()
rc=0; "$HERE/../components/sanotts/prepare.sh" "${SANOTTS_DIR:-$HOME/build/tts/sanoTTS}" "$OUT/sanotts_src" || rc=$?
case $rc in
    0) TTS_ARGS=(-D SANOTTS_SRC="$OUT/sanotts_src" -D SANOTTS_BSS_PSRAM=1) ;;
    1) echo "no sanoTTS checkout: building without text to speech" >&2 ;;
    *) exit $rc ;;                                   # wrong sanoTTS commit
esac
idf.py -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;$HERE/sdkconfig.babytalk" "${TTS_ARGS[@]}" build

# boot.avm: the Erlang + Elixir standard libraries, cut from the release image (at 0x1D0000 there)
mkdir -p "$OUT"
REL="$OUT/AtomVM-esp32s3-elixir-$TAG.img"
[ -f "$REL" ] || curl -sL -o "$REL" \
    "https://github.com/atomvm/AtomVM/releases/download/$TAG/AtomVM-esp32s3-elixir-$TAG.img"
python3 - "$REL" "$OUT/boot.avm" <<'PY'
import sys
img = open(sys.argv[1], "rb").read()
boot = img[0x1D0000:]
assert boot[:4] != b"\xff\xff\xff\xff" and len(boot) > 1000, "no boot.avm at 0x1D0000 in the release image"
open(sys.argv[2], "wb").write(boot)
print(f"boot.avm: {len(boot)} bytes")
PY

python -m esptool --chip esp32s3 merge_bin -o "$OUT/atomvm-babytalk.img" \
    --flash_mode keep --flash_freq keep --flash_size keep \
    0x0 build/bootloader/bootloader.bin 0x8000 build/partition_table/partition-table.bin \
    0x10000 build/atomvm-esp32.bin 0x410000 "$OUT/boot.avm"
ls -l "$OUT/atomvm-babytalk.img" build/atomvm-esp32.bin
echo "flash: esptool.py --chip esp32s3 write_flash 0x0 $OUT/atomvm-babytalk.img   (the model goes to 0x490000, apps to main.avm at 0xA90000)"
