# sanoTTS licensing, file by file

BabyTalk's text to speech compiles sanoTTS's ESPHome component sources
(`esphome/components/sanotts/` in [Ampixa/sanoTTS](https://github.com/Ampixa/sanoTTS)). **None
of that code is in this repository**: `prepare.sh` copies it from your own checkout at build
time, and refuses any commit but the one audited here:

    18e26b2b365bff41d211e516b0760021451438f1   (2026-09-24, "Merge pull request #17")

The code in this repository (`tts_engine.c`, `tts_engine.h`, the build files) is ours, MIT.
The question is only what a **firmware binary** built with text to speech contains.

## What sanoTTS says

sanoTTS as a whole is **GPL-3.0-or-later** (`LICENSE`), because its original G2P used
espeak-ng. `LICENSE.MIT` makes the inference runtime **MIT**, but it lists its files
"exhaustively", and those are the copies under `mcu/src/`, `mcu/include/` and `mobile/` --
not the ESPHome component we compile. Separately, `esphome/g2p/lex/README.md` states that the
lexicon G2P is espeak-free and Apache-2.0 ("Nothing here is GPL").

## The files, at 18e26b2

| File (compiled) | In-file licence | sanoTTS docs | Status for a binary |
|---|---|---|---|
| `snt_nano.c` | `SPDX: MIT` | runtime (the `mcu/src` copy is MIT-listed; this copy differs) | MIT |
| `snt_matvec_esp32s3_asm.c` | `SPDX: MIT` | runtime | MIT |
| `snt_port_esphome.c` | `SPDX: MIT` | runtime | MIT |
| `sanotts_model_data.c` (voice weights) | `SPDX: MIT` | weights: distilled from Apache-2.0 / MIT teachers | MIT |
| `snt_kernels_esp32s3.c` | none | not listed in `LICENSE.MIT` | **unclear: GPL-3.0 by the repo default** |
| `snt_kernels_ref.c` | none | the `mcu/src` copy is MIT-listed; this copy differs | **unclear: GPL-3.0 by the repo default** |
| `nano_lex_g2p.c` | none | G2P README: port of misaki (Apache-2.0), "nothing here is GPL" | **unclear**: documented Apache-2.0, no in-file grant |
| `nano_lex_tables.c` (dictionary) | none | misaki `us_gold` data, Apache-2.0, pinned + sha256 in sanoTTS | Apache-2.0 (misaki) |
| headers `snt_nano.h`, `snt_port.h`, `sanotts_model_data.h` | `SPDX: MIT` | runtime | MIT |
| headers `snt_arch.h`, `nano_q8_meta.h` (generated offsets), `nano_lex_g2p.h`, `nano_lex_tables.h` | none | `mcu/models/*/nano_q8_meta.h` is MIT-listed; the others unlisted | follow their `.c` files above |

The intent upstream is plainly permissive for everything we use: no espeak-ng code is
compiled. But three files carry no licence of their own and fall outside the MIT file list,
so on the letter of the repository they are GPL-3.0-or-later (so are four headers without one).

## The heart voice (ESP32-P4)

`prepare.sh ... heart` (or `heart4`) makes a voice from the checkout's `web/voices/heart/`
(float32 weights, 2.27M parameters). sanoTTS's README lists only its runtime as MIT, so those
weights are GPL-3.0-or-later by the repository default. They are not compiled into the app but
flashed to their own partition (`voice.bin`); a device carrying them is in the same position as
the firmware below. BabyTalk's own files for it (`snt_q4.*`, `snt_port_p4.c`, `snt_nano.patch`,
`tools/sanotts_voice.py`) are MIT.

## What this means

- **This repository (source)**: MIT, unaffected. Nothing from sanoTTS is committed here.
- **Building for yourself**: no issue.
- **Distributing a firmware binary that includes text to speech**: until the three files
  above get an explicit permissive licence upstream, treat that binary as GPL-3.0-or-later
  (ship it with the corresponding source and the GPL text), or build it without text to
  speech (no sanoTTS checkout -> the firmware builds without it). Binaries with speech to
  text only are unaffected.
- **Fix**: ask Ampixa to add SPDX headers to the ESPHome component files (or list them in
  `LICENSE.MIT`), matching what their docs already say; then re-audit, update the pinned
  commit in `prepare.sh`, and update this table.

This is a record of what the files say, not legal advice.
