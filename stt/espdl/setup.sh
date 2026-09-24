#!/usr/bin/env bash
# Recreate stt/components/esp-dl: pristine ESP-DL 3.3.11 from the Espressif component
# registry + our patches (espdl/patches/*.patch). The patched copy is git-ignored;
# only the patches are tracked. Re-run after editing a patch.
#
# To change a patch: edit components/esp-dl, then
#   espdl/setup.sh --diff > espdl/patches/0001-micromodels.patch
set -euo pipefail
cd "$(dirname "$0")/.."
VER=3.3.11
URL=https://components-file.espressif.com/components/espressif/esp-dl/$VER/espressif__esp-dl-v$VER.zip
CACHE=${XDG_CACHE_HOME:-$HOME/.cache}/micromodels
ZIP=$CACHE/esp-dl-$VER.zip
PRISTINE=$CACHE/esp-dl-$VER
mkdir -p "$CACHE"
[[ -f $ZIP ]] || curl -sfL -o "$ZIP" "$URL"
if [[ ! -d $PRISTINE ]]; then
    mkdir -p "$PRISTINE" && (cd "$PRISTINE" && unzip -q "$ZIP")
fi
if [[ ${1:-} == --diff ]]; then
    diff -ruN --exclude='*.a' --exclude=CHECKSUMS.json "$PRISTINE" components/esp-dl | sed "s|$PRISTINE|a|; s|components/esp-dl|b|" || true
    exit 0
fi
rm -rf components/esp-dl
cp -a "$PRISTINE" components/esp-dl
for p in espdl/patches/*.patch; do
    patch -s -p1 -d components/esp-dl < "$p"
done
echo "components/esp-dl: ESP-DL $VER + $(ls espdl/patches/*.patch | wc -l) patch(es)"
