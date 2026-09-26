# stt/ -- the plain ESP-IDF test firmware

A standalone ESP-IDF app (no MicroPython, no AtomVM) for developing and measuring the speech
engine on the board: Citrinet-256 on MMRT, weights read in place from the `model` partition,
driven over WiFi TCP (port 5555) by the laptop-side tools. It is where the engine is proven
before the firmwares pick it up. Not to be confused with `mpy/stt/`, which is the MicroPython
`stt` *module* (a thin binding over `components/stt_engine/`).

Commands (see the top of `main/main.cpp`): `pcm` (transcribe PCM sent from the laptop),
`feats` (bit-exact check against the host run of the same C code), `listen` (record from the
board's mic and transcribe), `wake` (wake-phrase mode), `info`.

Shared with the other firmwares: `main/stt_core.c` (log-mel front end + CTC decoding),
`main/kws.c` (wake-phrase scoring), `main/mic.c` (ES7210 capture) -- used by
`components/stt_engine/` and the AtomVM build.

Build and flash (ESP-IDF 5.5, WiFi credentials in the git-ignored `main/wifi_secrets.h`, see
`main/wifi_secrets.h.example`):

    stt/idf.sh build flash         # then flash a model to 0x410000 (partitions.csv)

Laptop side: `tools/stt.py`, `tools/listen.py`, `tools/wake.py`, `tools/mmrt_cmd.py`,
`export/mmrt_check_device.py`, `export/check_frontend_device.py`; the board's address comes
from `--host`, `$STT_HOST` or `board.conf` (see `board.conf.example`).

`RESULTS.md` here is the first on-device run (2026-09-23, on ESP-DL) and is kept as history.
