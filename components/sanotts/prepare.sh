#!/usr/bin/env bash
# Copy sanoTTS's ESPHome component sources (github.com/Ampixa/sanoTTS) to $2 and patch them for
# BabyTalk: the runtime is serial here (snt_scratch_id() == 0), so one scratch bank instead of
# two saves ~40 KB of RAM. Exit 1 (and copy nothing) when there is no checkout at $1.
#   components/sanotts/prepare.sh ~/build/tts/sanoTTS <out dir>
set -e
SRC="$1/esphome/components/sanotts"
[ -f "$SRC/snt_nano.c" ] || exit 1
rm -rf "$2" && cp -r "$SRC" "$2"
sed -i 's/^static NanoScratch g_scr\[2\];/static NanoScratch g_scr[1];/' "$2/snt_nano.c"
grep -q 'g_scr\[1\]' "$2/snt_nano.c"
