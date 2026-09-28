# Text-to-speech voices

BabyTalk speaks with [sanoTTS](https://github.com/Ampixa/sanoTTS) voices. There are three to
choose from when building; the choice is made at build time (`prepare.sh <sanoTTS> <out> [voice]`,
or `SANOTTS_VOICE=` for the Watchtower P4 firmware).

| voice | parameters | weights | quality (SCOREQ) | speed on the ESP32-P4 | where |
|---|---|---|---|---|---|
| `nano` (default) | 294k | 0.34 MB, int8 | 2.13 | 0.23x real time | every board: compiled into the app |
| **`heart`** | 2.27M | 2.4 MB, int8 | **3.36** | **0.51-0.58x real time** | ESP32-P4: the `voice` flash partition |
| `heart4` | 2.27M | 2.4 MB, 4-bit | 3.08 | 1.75-1.81x real time | ESP32-P4 |

"x real time" is synthesis time over audio length, measured on the board (`atomvm/apps/speed_bench`),
so 0.5x speaks 5 s of audio after 2.5 s of work. SCOREQ is a no-reference quality predictor, the one
sanoTTS reports its voices with (sanoTTS's own figure for heart is 3.48, for heart-nano 2.29); ours
are over the same eight sentences for every voice, rendered by the C runtime. Listen before you
trust a number: to our ears heart is plainly the better voice, and heart4 sounded fine too.

`heart` is sanoTTS's best-sounding voice, the one its browser demo ships as float32 (9 MB): the same
recipe as nano (duration, acoustic model, vocoder), eight times the parameters. BabyTalk's G2P gives
it the same phoneme ids as nano. It needs the ESP32-P4: the S3 would take about 2x real time.

## Building and flashing heart (Watchtower P4 firmware)

    SANOTTS_VOICE=heart node_avm/firmware/p4/build.sh      # -> $OUT/voice.bin as well
    python -m esptool --chip esp32p4 write_flash 0x8000 partition-table.bin \
        0x10000 atomvm-esp32.bin 0xD00000 voice.bin

`prepare.sh` makes the voice from the checkout's float32 export (`web/voices/heart/`, checked
against its `meta.json` hashes) with `tools/sanotts_voice.py`, which needs numpy (run through
`uv`). `voice.bin` goes to the `voice` partition (3 MB at 0xD00000); the firmware reads it into
PSRAM at the first speech (0.27 s, once), and refuses a partition that holds another voice or
format than the one it was built for. The firmware and the voice go together: the runtime's
shapes are compiled in from the voice's header.

Keep partitions below 16 MB. Above it the flash needs 32-bit addresses, which ESP-IDF 5.5 reads only
with an experimental option: esptool wrote a voice at 28 MB fine, and the app's read came back
from 12 MB.

## What it took

**int8 with sanoTTS's own format.** sanoTTS gated heart out of int8: its waveform correlation with
the float reference fell to 0.951 over its eight golden sentences, below the 0.98 every nano voice
is held to, so it ships heart as float32. The correlation gate is strict (it compares waveforms,
phase and all); SCOREQ gives int8 heart 3.36 against 3.37 for float32. (Scales per 32 weights
instead of per row pass the gate, 0.989, but need a runtime change and sound no different.)

**Batching frames.** sanoTTS's runtime multiplies one frame at a time. Heart's decoder layers
(a pointwise pair of 221 KB per block, the head 197 KB) don't fit the P4's 128 KB cache, so every
frame fetched every matrix from PSRAM again: 1.8x real time, memory-bound (the int8 kernel does
837 M multiply-adds/s on cached weights and 124 M/s streaming them from PSRAM). `snt_nano.patch`
runs the embedding, the pointwise layers and the head over 8 frames (the head: its pair) per
weight fetch, in blocks of 8 rows, so a block is fetched once and read from L1 for the other
frames. Same arithmetic, so the same audio, byte for byte. 1.8x -> 1.06x.

**Both cores.** sanoTTS's ESP32 port runs serially (on the S3 the second core didn't help).
`snt_port_p4.c` splits every parallel range with a helper task on the other core, as the runtime
was designed for; the second scratch bank lives in PSRAM. 1.06x -> 0.67x, then batching the
embedding and the head: 0.56x.

**4-bit (`heart4`).** 4-bit weights with a float scale per 16 inputs, each chosen to minimise its
group's rounding error (`snt_q4.h`, `snt_q4.c`, `snt_q4_esp32p4.S`). The acoustic model is the
fragile part: one of its layers in 4-bit alone takes the correlation to ~0.88. The vocoder's ten
pointwise layers barely notice one by one. All of it in 4-bit: correlation 0.72, SCOREQ 3.08. It
is slower than int8 here: the unpacking and per-group scaling cost more than the memory it saves,
once batching has made the work compute-bound. Its rows keep the int8 rows' size, so it saves
nothing in flash yet (packed tightly it would be ~1.4 MB). Nano in 4-bit falls from SCOREQ 2.13
to 1.86 for ~130 KB: not worth it on the S3.

## Measuring a voice

The quality numbers come from sanoTTS's golden fixtures (`mcu/test/fixtures/en_us_r227f32`: eight
sentences with frozen durations and the float PyTorch waveforms) run through the C runtime on
the PC, with a copy of `mcu/test/nano_golden_main.c` that also writes each sentence's PCM, and
SCOREQ (`pip install scoreq`, synthetic domain, no reference) over those files. The board timings
and PCM hashes: `atomvm/apps/speed_bench`.

## Licence

heart's weights come from sanoTTS's `web/voices/heart/`, which sanoTTS's README doesn't list under
its MIT runtime; by the repository default they are GPL-3.0-or-later, like the three component
files `components/sanotts/LICENSES.md` flags. A firmware built with them is to be treated as
GPL-3.0-or-later (it already is, with any sanoTTS voice, until upstream clarifies). Nothing from
sanoTTS is committed here: the voice is made from your own checkout at build time.
