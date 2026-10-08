#!/usr/bin/env bash
# A hands-free recording session in the truck: the laptop reads prompts aloud in its TTS
# voices, the LilyGO T-LoRa Pager (BabyTalk firmware + atomvm/apps/mic_stream) streams its
# mic over USB, and capture.py cuts and saves each clip. Nothing to press: Ctrl-C ends it early.
#
#   field/truck_session.sh [minutes] [condition] [noise description]
#   field/truck_session.sh 45 truck-drive "diesel, doors off, traffic, light music"
#
# Before: the Pager attached to WSL (Windows: usbipd attach --wsl --busid <id> --auto-attach),
# the laptop set not to sleep, and its volume where you want it for the whole drive.
set -e
cd "$(dirname "$0")"
MINUTES="${1:-45}"
CONDITION="${2:-truck-drive}"
NOISE="${3:-diesel, doors off, traffic, light music}"
uv run capture.py --link usb --port "${PORT:-/dev/ttyACM0}" \
    --board lilygo-t-lora-pager --mic es8311 \
    --auto --minutes "$MINUTES" \
    --condition "$CONDITION" --noise "$NOISE" \
    --notes "TTS from the laptop speaker; the Pager moved around the cab during the drive"
