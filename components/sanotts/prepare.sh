#!/usr/bin/env bash
# Copy sanoTTS's ESPHome component sources (github.com/Ampixa/sanoTTS) to $2 and patch them for
# BabyTalk: the runtime is serial here (snt_scratch_id() == 0), so one scratch bank instead of
# two saves ~40 KB of RAM. Exit 1 (and copy nothing) when there is no checkout at $1.
#   components/sanotts/prepare.sh ~/build/tts/sanoTTS <out dir>
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
rm -rf "$2" && cp -r "$SRC" "$2"
sed -i 's/^static NanoScratch g_scr\[2\];/static NanoScratch g_scr[1];/' "$2/snt_nano.c"
grep -q 'g_scr\[1\]' "$2/snt_nano.c"
# speaking pace as a runtime setting (tts_set_length_scale) instead of the constant 1.0
sed -i 's|^#define LENGTH_SCALE 1.0f .*|float snt_length_scale = 1.0f;  /* BabyTalk: runtime pace */\n#define LENGTH_SCALE snt_length_scale|' "$2/snt_nano.c"
grep -q '^float snt_length_scale' "$2/snt_nano.c"
