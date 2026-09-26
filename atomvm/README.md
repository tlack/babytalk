# BabyTalk and MMRT for AtomVM (Erlang / Elixir on the ESP32-S3)

On-device speech to text, text to speech, a wake phrase, microphone and speaker, plus the
int8/int4 vector kernels underneath, for [AtomVM](https://github.com/atomvm/AtomVM) **v0.7.0-alpha.1** on the ESP32-S3.
Everything runs on the board: no cloud, no network needed (WiFi is only for the test tools).

```erlang
{ok, Pcm} = babytalk:record(3),                               % 3 s from the board's mic
{ok, Text, _Info} = babytalk:transcribe_sync(Pcm, 10000),     % "what's the weather like today"
ok = babytalk:speak([<<"You said: ">>, Text]),                 % out of the board's speaker

W = mmrt:pack({int8, Weights}, 64, 256),                      % 64 x 256 matrix, packed once
{int8, Y} = mmrt:matvec(W, {int8, X}, 7).                     % SIMD, bit-exact
```

```elixir
{:ok, text, _info} = BabyTalk.transcribe_sync(pcm, 10_000)
```

Two libraries, usable separately:

| | What | Module | Needs |
|---|---|---|---|
| **MMRT** | int8/int4 matvec, matmul, dot, requant, top-k... on the S3's SIMD unit | `mmrt` / `MMRT` | nothing board-specific |
| **BabyTalk** | speech to text, text to speech, wake phrase, mic + speaker | `babytalk`, `babytalk_listener` / `BabyTalk` | the model partition; our audio drivers (below) |

Status: all of it works on the Waveshare ESP32-S3-CAM -- say "wake up tomato face", wait for
the chime, say something, and the board says it back (`apps/listen_demo`).

## Device drivers: we bring our own

AtomVM's driver catalog is still growing, and it has **no I2S driver** at all, so BabyTalk
ships its own C drivers for the audio chips and exposes them to Erlang as NIFs. They are
written for one board, the **Waveshare ESP32-S3-CAM**:

| Chip | Role | Bus | Driver | Status |
|---|---|---|---|---|
| **Everest ES7210** | 4-channel microphone ADC (we use mic 1) | I2C `0x40` (setup) + I2S in, 16 kHz, with MCLK | `board_audio.c` | working |
| **WCH CH32V003** | small RISC-V MCU acting as an 8-bit I2C IO expander; gates the audio rail (P6) and the speaker amp (P4) | I2C `0x24` (write-only latch: we keep a shadow) | `board_audio.c` | working |
| **Everest ES8311** | mono speaker codec (DAC) | I2C `0x18` + I2S out, any rate (MCLK = 256 x Fs) | `components/atomvm_babytalk/board_audio.c` | working |
| **NS4150B** | 3 W class-D speaker amp | enable = expander P4, on only while playing | `board_audio.c` | working |

All four sit behind one module, `board_audio.c`, because the two codecs share I2S port 0's
clock pins: the port is either capturing or playing, never both. These use ESP-IDF 5.5's current drivers (`i2c_master`, `i2s_std`). ESP-IDF aborts at boot if
the legacy I2C driver is linked next to the new one, so this firmware **turns off AtomVM's own
I2C** (`CONFIG_AVM_ENABLE_I2C_PORT_DRIVER=n`, `..._RESOURCE_NIFS=n`): Erlang code gets no
`i2c` module on this build, and C owns the bus. (Revisit when AtomVM moves to the new I2C
driver.) Another board needs its own pins and codec setup in `board_audio.c`; everything above the
driver (engines, NIFs, Erlang) is board-independent. MMRT has no drivers and runs on any
ESP32-S3 AtomVM build.

## OTP structure

The C side is deliberately small: NIFs start work and report back with **messages**, and the
supervision, state and policy are Erlang.

- `babytalk:transcribe/1` returns `{ok, Ref}` at once; inference runs on a FreeRTOS task
  and the result arrives as `{babytalk, Ref, {ok, Text, Info}}`. (AtomVM has no dirty NIFs,
  and a scheduler must never block for a second.)
- `babytalk:listen/1` streams the mic as `{babytalk_mic, Ref, Pcm}` chunks, gapless, even
  while a transcription runs; `stop_listening/0` ends it with `{babytalk_mic, Ref, stopped}`.
- `babytalk_listener` is a **gen_server** that owns the microphone *and* the speaker (one
  I2S port) and runs the conversation:

  | Phase | Chime on entry | What it does |
  |---|---|---|
  | wake | *ready* (D5 -> A5) | scores the last 4 s against the wake phrase about once a second |
  | message | *wake* (A5 -> D6) | records until 700 ms of quiet after speech (or 6 s) |
  | idle | *decoding* (G5 -> C5) | transcribes, reports `{command, Text}`, speaks any reply, then *ready* again |

  For dialogs, `ask/3` speaks a prompt, chimes and reports the `{answer, Text}` (recorded
  until silence, or for a fixed time); `wake_on/2` installs a wake phrase at run time; the
  `greeting` option is spoken after the wake chime. Replies go through `babytalk_listener:say/2`, which pauses the mic while speaking (so it
  doesn't hear itself); they are spoken 10% slower than the voice's own pace
  (`length_scale`), which is easier to follow. Events to the subscriber: `{heard, Text, Score}`, `{wake, Text,
  Score}`, `{message_end, silence | max_length, Levels}`, `{command, Text}`, `{said, Text}`.
  Put it under a supervisor: if the mic or engine fails it crashes, and the restart
  re-opens everything.
- `apps/listen_demo` shows the shape: a `rest_for_one` supervisor over a responder
  (the subscriber, which answers "You said: ...") and the listener.

```
listen_demo_sup (rest_for_one)
|-- listen_demo_responder    handle_info({babytalk_listener, {command, Text}}, ...) -> say/2
`-- babytalk_listener        mic + speaker: wake -> message -> transcribe -> reply
```

One engine (transcription and synthesis) and one audio port (mic or speaker): a second
concurrent call gets `{error, busy}`, so put a single owner process (like the listener) in
front of them.

## API

### `mmrt` (Erlang) / `MMRT` (Elixir)

Tagged binaries: `{int8, Bin}`, `{int4, Bin}` (signed nibbles, low first), `{int32, Bin}`
(little-endian). `pack/3` turns a row-major matrix into
`{int8_m16 | cb4_m16, Rows, Cols, MaxRowL1, Bin}` for the SIMD kernel (padded to multiples
of 16; the Bin is an ordinary binary you can keep, send, or build on the host).

| Call | Result |
|---|---|
| `pack({int8 \| int4, RowMajor}, Rows, Cols)` | packed matrix (int4 -> half the bytes) |
| `matvec(M, {int8, X})` | `{int32, _}` exact accumulators |
| `matvec(M, {int8, X}, Shift)` / `matmul(M, {int8, Rows}, Shift)` | `{int8, _}`: sat8(round_half_up(acc >> Shift)) |
| `dot(A, B)` | integer, any element types |
| `add/2`, `relu/1`, `requant(Int32, Shift)`, `argmax/1`, `top_k(V, K)` | elementwise / reductions |
| `to_int4/1`, `from_int4/1` | conversions |

matvec/matmul use the SIMD unit when the accumulators provably fit its 20-bit lanes
(MaxRowL1 x max|x| < 2^19) and an exact C path otherwise: same results either way. Each call
runs on the calling scheduler and is capped at 4M multiply-adds (`error(too_big)` above that).

### `babytalk` (Erlang) / `BabyTalk` (Elixir)

| Call | Result |
|---|---|
| `transcribe(Pcm)` | `{ok, Ref}`, then `{babytalk, Ref, {ok, Text, Info} \| {error, Code}}` |
| `transcribe_sync(Pcm, Timeout)` | `{ok, Text, Info}` |
| `listen(ChunkMs)` / `stop_listening()` | mic stream, see above |
| `record(Secs)` | `{ok, Pcm}` |
| `say(Text[, #{length_scale => S}])` / `say_sync(...)` | `{ok, Ref}`, then `{babytalk, Ref, {ok, Pcm24k, Info}}` (sanoTTS, 24 kHz; S 1.0 = the voice's pace, 1.1 = 10% slower) |
| `play(Pcm, Rate[, Volume])` | `{ok, Ref}`, then `{babytalk_play, Ref, done}`; Volume 0..100 (DAC, 75 = 0 dB, default 76: much above ~80 the speech clips) |
| `tones([{Hz, Ms}], Volume)` | chimes made on the board, played like `play` |
| `speak(Text[, Volume])` | `ok` once said out loud (say + play) |
| `rms(Pcm)` | loudness 0..32768 |
| `phrase([Spelling])` | `{ok, Sequences}`: the wake phrase `Info`'s score is measured against |
| `cache(Bytes)` | keep weights in PSRAM (~20 ms faster per MB per transcription) |
| `info()`, `heap_info()` | model and memory facts |

Audio is 16 kHz mono signed 16-bit PCM. `Info` has `score` (0.0 = the phrase is the best
reading; the demo wakes at >= -21), `span`, `frames`, `fe_ms`, `model_ms`, `dec_ms`.

## Build and flash

Needs ESP-IDF 5.5 (the repo's `~/build/lvgl_micropython/lib/esp-idf`), Erlang/OTP 26+ and
Elixir 1.17+ (for the `exatomvm` mix tasks).

```bash
atomvm/build.sh                     # AtomVM v0.7.0-alpha.1 + our components -> ~/build/atomvm-out/atomvm-babytalk.img
esptool.py --chip esp32s3 write_flash 0x0 ~/build/atomvm-out/atomvm-babytalk.img \
    0x490000 models/citrinet256_int4.mmrt          # firmware + Erlang/Elixir libs, then the model

cd atomvm/apps/listen_demo && mix deps.get && mix atomvm.packbeam
esptool.py --chip esp32s3 write_flash 0xA90000 listen_demo.avm
```

If boards share a serial port, flash with `tools/flash.sh` instead of esptool: same
offset/file arguments, but it reads the chip's MAC first and refuses to write unless it matches
the `mac` in `board.conf`.

`build.sh` clones the pinned AtomVM tag, applies `patches/`, links in `components/` (ours and
the shared `mmrt`, `sram_pool`, `stt_engine`, `sanotts` from the repo root), prepares the
sanoTTS sources from a checkout (`SANOTTS_DIR`, default `~/build/tts/sanoTTS`; without one the
firmware builds without text to speech), applies
`sdkconfig.babytalk` and builds. The Erlang/Elixir standard libraries (`boot.avm`) come from
the official release image, so no host build of AtomVM is needed.

| Partition | Offset | Size | |
|---|---|---|---|
| factory | 0x10000 | 4 MB | AtomVM + NIFs + sanoTTS (its dictionary and voice: ~1.9 MB) |
| boot.avm | 0x410000 | 512 KB | Erlang + Elixir libraries |
| model | 0x490000 | 6 MB | the int4 `.mmrt` model |
| main.avm | 0xA90000 | 5.4 MB | your app: last, so it takes whatever flash is left |

Apps: `apps/mmrt_test` (MMRT test vectors from `gen_vectors.py` + benchmarks),
`apps/stt_server` + `tools/stt_client.py` (send WAVs over WiFi, get transcripts back; the
board's address comes from `--host`, `$STT_HOST` or `board.conf`, see `board.conf.example`),
`apps/tts_demo` (speech out, speed, PCM hashes), `apps/listen_demo` (the full conversation),
`apps/wakeword_demo` (teach it your own wake phrase by voice -- "Did you say ...?" -- then
talk to it; the phrase is stored as the model's own spelling of how you said it). Apps that use WiFi need
`gen_wifi_creds.sh <app dir>`, which writes a git-ignored `wifi_creds.erl`.

## Measured (Waveshare ESP32-S3-CAM, 240 MHz, octal PSRAM)

- **Same transcripts as MicroPython**: 8 field clips x 4 runs, 32/32 identical to the
  MicroPython firmware on the same PCM; a 7.7 s clip takes 1.33 s (MicroPython: 1.41 s).
- **MMRT**: 256x256 int8 matvec 453 us (145 MMAC/s, limited by PSRAM bandwidth); matmul with
  16 rows 720 MMAC/s; int4 466 MMAC/s; the exact C fallback 22 MMAC/s. For scale: a dot
  product in plain Erlang over lists runs at 0.19 MMAC/s.
- **Text to speech**: RTF 0.27 (a 4.76 s sentence in 1.27 s; MicroPython 0.25-0.26) with
  sanoTTS's ~104 KB of static buffers moved to PSRAM (`components/sanotts/linker.lf`); its
  84 KB arena shares the engine's internal SRAM pool with transcription. Playback: 0 DMA
  underruns.
- **Internal RAM**: 116 KB free at start (after the engines' 84.5 KB shared pool is
  reserved), 42 KB after WiFi joins, ~31 KB during inference.

## Things we learned (AtomVM specifics)

- NIFs bind only to **external** calls: a module calling its own NIF stub locally gets the
  Erlang stub (`undef`). Call `?MODULE:fun(...)` and export it.
- NIF arguments are **not GC roots**: after `memory_ensure_free`, re-read binaries only
  through `memory_ensure_free_with_roots(ctx, n, argc, argv, ...)`.
- `gen_tcp:recv(S, 0)` returns whatever has arrived: buffer it. Long uploads occasionally
  end in `{error, closed}` (2 of 34 clips of 100-330 KB): be ready to resend.
- Building big binaries element by element is very slow (a 64 KB comprehension took
  minutes; matching 256 bytes one at a time, 80 ms): use `binary:copy/2`, iolists, or NIFs.
- Reserve large internal-RAM blocks in the NIF collection's init callback, which runs
  before any Erlang code and so before WiFi fragments the heap.
- AtomVM's default WiFi settings keep ~30 KB more internal RAM than MicroPython's;
  `sdkconfig.babytalk` trims them (10 static RX buffers, BA window 6, WiFi code out of IRAM).

## Risks and limits

AtomVM v0.7 is an alpha (the project recommends v0.6 for production). While a transcription
runs, MMRT keeps both cores busy (its helper task on core 0), so Erlang code and WiFi respond
more slowly for those ~0.2 s per second of audio. The mic and the speaker take turns (one
I2S port), so the board can't hear you while it talks. The end of a message is found by
loudness, so a noisy room may run to the 6 s cap.
