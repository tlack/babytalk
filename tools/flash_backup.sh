#!/usr/bin/env bash
# Full-flash backup of a board in ROM download mode, verified on-chip (MD5).
#
#   tools/flash_backup.sh <name>        e.g. tools/flash_backup.sh my-board
#   WAIT=300 tools/flash_backup.sh ...   seconds to wait for download mode (default 120)
#
# Put the board in download mode: hold BOOT, tap RESET (or unplug, hold
# BOOT, replug). It then enumerates as 303a:1001. Under WSL this script attaches
# that device via usbipd.exe (the device must already be `usbipd bind`-ed, which
# needs admin once per device).
#
# Output: backups/<name>-<date>.bin (+ .sha256). Contains the node's secrets
# (node.json token, WiFi creds) -- backups/ is git-ignored, keep it that way.
#
# Restore:  esptool.py --port /dev/ttyACM0 write_flash 0x0 backups/<file>.bin
set -euo pipefail
name="${1:?usage: $0 <name>}"
port="${PORT:-/dev/ttyACM0}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/backups/$name-$(date +%Y%m%d-%H%M%S).bin"
mkdir -p "$root/backups"

if command -v usbipd.exe >/dev/null; then
    echo "waiting for 303a:1001 (ROM download mode) on the Windows side..."
    for _ in $(seq "${WAIT:-120}"); do
        line=$(usbipd.exe list 2>/dev/null | tr -d '\r' | grep '303a:1001' || true)
        if [[ -n "$line" ]]; then
            busid=$(awk '{print $1}' <<<"$line")
            if ! grep -q Attached <<<"$line"; then
                echo "attaching busid $busid"
                usbipd.exe attach --wsl --busid "$busid"
            fi
            break
        fi
        sleep 1
    done
    [[ -n "${busid:-}" ]] || { echo "no 303a:1001 device appeared"; exit 1; }
fi

for _ in $(seq 30); do [[ -e "$port" ]] && break; sleep 1; done
[[ -e "$port" ]] || { echo "$port never appeared"; exit 1; }

esptool.py --port "$port" --before no_reset --after no_reset flash_id
esptool.py --port "$port" --before no_reset --after no_reset read_flash 0 ALL "$out"
# The chip hashes its own flash and compares with the file: as strong as a second
# read, without a second transfer (reads crawl at ~11KB/s through usbipd).
if esptool.py --port "$port" --before no_reset --after no_reset verify_flash 0 "$out"; then
    (cd "$(dirname "$out")" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
    echo "OK: $out ($(du -h "$out" | cut -f1)), verified against the chip"
else
    mv "$out" "$out.BAD"
    echo "VERIFY FAILED -- do not trust $out.BAD"
    exit 1
fi
echo "board left in download mode; press RESET (or replug) to boot it normally"
