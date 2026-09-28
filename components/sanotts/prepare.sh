#!/usr/bin/env bash
# Copy sanoTTS's ESPHome component sources (github.com/Ampixa/sanoTTS) to $2 and patch them for
# BabyTalk: the runtime is serial here (snt_scratch_id() == 0), so one scratch bank instead of
# two saves ~40 KB of RAM. Exit 1 (and copy nothing) when there is no checkout at $1.
#   components/sanotts/prepare.sh ~/build/tts/sanoTTS <out dir> [voice]
# voice: nano (the default: the component's own int8 voice, 294k parameters), heart (sanoTTS's
# 2.27M-parameter voice, requantized from its float32 export to sanoTTS's int8 format into
# $2/voice/ by tools/sanotts_voice.py) or heart4 (the same in BabyTalk's 4-bit format: smaller,
# slower on the P4, a little rougher). heart and heart4: ESP32-P4 only (docs/TTS_VOICES.md).
#
# The checkout must be at SANOTTS_COMMIT: the commit whose per-file licences were checked
# (components/sanotts/LICENSES.md). Another commit may carry other terms -- re-check them,
# then update both. SANOTTS_ALLOW_ANY_COMMIT=1 skips the check (local experiments only).
set -e
SANOTTS_COMMIT=18e26b2b365bff41d211e516b0760021451438f1
SRC="$1/esphome/components/sanotts"
[ -f "$SRC/snt_nano.c" ] || exit 1
have=$(git -C "$1" rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$have" != "$SANOTTS_COMMIT" ] && [ -z "$SANOTTS_ALLOW_ANY_COMMIT" ]; then
    echo "sanoTTS checkout $1 is at $have, not the audited $SANOTTS_COMMIT:" >&2
    echo "  git -C $1 fetch origin $SANOTTS_COMMIT && git -C $1 checkout $SANOTTS_COMMIT" >&2
    echo "  (see components/sanotts/LICENSES.md; SANOTTS_ALLOW_ANY_COMMIT=1 to override)" >&2
    exit 2
fi
VOICE="${3:-nano}"
case "$VOICE" in nano|heart|heart4) ;; *) echo "unknown voice $VOICE (nano, heart, heart4)" >&2; exit 2 ;; esac
rm -rf "$2" && cp -r "$SRC" "$2"
if [ "$VOICE" = nano ]; then   # (heart and heart4 run on both cores: snt_port_p4.c)
    sed -i 's/^static NanoScratch g_scr\[2\];/static NanoScratch g_scr[1];/' "$2/snt_nano.c"
    grep -q 'g_scr\[1\]' "$2/snt_nano.c"
fi
# speaking pace as a runtime setting (tts_set_length_scale) instead of the constant 1.0
sed -i 's|^#define LENGTH_SCALE 1.0f .*|float snt_length_scale = 1.0f;  /* BabyTalk: runtime pace */\n#define LENGTH_SCALE snt_length_scale|' "$2/snt_nano.c"
grep -q '^float snt_length_scale' "$2/snt_nano.c"
# on the ESP32-P4 BabyTalk's own int8 kernels (snt_kernels_esp32p4.c) replace the scalar ones,
# as sanoTTS's snt_kernels_esp32s3.c does on the S3
sed -i 's/^#if !SANOTTS_S3_SIMD$/#if !SANOTTS_S3_SIMD \&\& !SANOTTS_P4_SIMD/' "$2/snt_kernels_ref.c"
[ "$(grep -c '^#if !SANOTTS_S3_SIMD && !SANOTTS_P4_SIMD$' "$2/snt_kernels_ref.c")" = 2 ]
# snt_nano.patch, BabyTalk's additions to the runtime, both off unless asked for: the 4-bit
# weight format (NANO_WEIGHT_FORMAT 2, snt_q4.h; SNT_NANO_W_Q4) and the decoder's per-frame
# stages (embedding, pointwise layers, head) on several frames per weight fetch (SNT_NANO_PW_BATCH)
patch -s -d "$2" -p1 < "$(dirname "$0")/snt_nano.patch"
grep -q 'defined(SNT_NANO_W_Q4)' "$2/snt_nano.c"
grep -q 'SNT_NANO_PW_BATCH > 1' "$2/snt_nano.c"
if [ "$VOICE" != nano ]; then
    V="$1/web/voices/heart"
    # the float32 export the browser voice ships, checked against its own meta.json
    for f in front dec; do
        want=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['${f}_sha256'])" "$V/meta.json")
        file=$([ $f = front ] && echo front_f32.bin || echo model_f32.bin)
        [ "$(sha256sum "$V/$file" | cut -c1-64)" = "$want" ] || { echo "$V/$file: sha256 mismatch" >&2; exit 2; }
    done
    uv run --quiet --no-project --with numpy python "$(dirname "$0")/../../tools/sanotts_voice.py" \
        "$1/mcu/models/en_us_r227f32/nano_q8_meta.h" "$V/front_f32.bin" "$V/model_f32.bin" "$2/voice" --fmt "$([ "$VOICE" = heart4 ] && echo q4 || echo int8)"
    cp "$2/voice/nano_q8_meta.h" "$2/nano_q8_meta.h"   # snt_nano.c includes the one beside it
fi
