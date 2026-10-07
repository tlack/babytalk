# Text-to-speech voices

BabyTalk speaks with [sanoTTS](https://github.com/Ampixa/sanoTTS) voices. There are three to
choose from, at build time: `SANOTTS_VOICE=` for `atomvm/build.sh`, or the third argument of
`components/sanotts/prepare.sh` in your own build.

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

## Building and flashing heart

With the AtomVM firmware, on an ESP32-P4:

    SANOTTS_VOICE=heart PARTITIONS=atomvm/partitions-babytalk-voice.csv atomvm/build.sh esp32p4_pre_c6
    esptool.py --chip esp32p4 write_flash 0x0 ~/build/atomvm-out-esp32p4_pre_c6/atomvm-babytalk.img \
        0x490000 models/citrinet256_int4.mmrt 0xD00000 ~/build/atomvm-out-esp32p4_pre_c6/voice.bin

How the pieces fit:

1. **`components/sanotts/prepare.sh <sanoTTS checkout> <out> heart`** copies sanoTTS's C sources
   into `<out>` and patches them. For heart it also builds the voice: it takes the float32 weights
   the browser demo ships (`web/voices/heart/`, checked against their `meta.json` hashes), and
   `tools/sanotts_voice.py` (numpy, run through `uv`) converts them to int8, writing
   `<out>/voice/voice.bin` and a header with the voice's layer sizes.
2. **The `sanotts` component** (`components/sanotts/CMakeLists.txt`) is built with
   `-D SANOTTS_SRC=<out>`. When it finds `voice/voice.bin` there, it compiles the runtime for that
   voice (its layer sizes from the header) and reads the weights from a flash partition labelled
   `voice` at run time, rather than compiling them into the firmware as it does nano's.
3. **`atomvm/build.sh`** does both steps for you, then copies `voice.bin` next to the firmware
   image and prints where to flash it. It refuses heart on an ESP32-S3, and refuses a partition
   table without a `voice` partition.

The partition table belongs to whoever builds the firmware: `atomvm/partitions-babytalk.csv` has
no voice partition, so `PARTITIONS=` picks another. `atomvm/partitions-babytalk-voice.csv` is
the same table with `main.avm` (your app) cut to 2.4 MB to make room for a 3 MB `voice` partition
at 0xD00000. In your own ESP-IDF app, add a data partition labelled `voice` (any subtype; we use
0x41) of at least 2.4 MB.

The firmware reads `voice.bin` into PSRAM at the first speech (0.27 s, once), and refuses a
partition that holds another voice or format than the one it was built for. The firmware and
the voice go together: flash both after every rebuild that changes the voice.

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
