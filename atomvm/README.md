# BabyTalk and MMRT for AtomVM (Erlang / Elixir on the ESP32-S3)

On-device speech to text, a wake phrase and a microphone, plus the int8/int4 vector kernels
underneath, for [AtomVM](https://github.com/atomvm/AtomVM) **v0.7.0-alpha.1** on the ESP32-S3.
Everything runs on the board: no cloud, no network needed (WiFi is only for the test tools).

```erlang
{ok, Pcm} = babytalk:record(3),                               % 3 s from the board's mic
{ok, Text, _Info} = babytalk:transcribe_sync(Pcm, 10000),     % "what's the weather like today"

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
| **BabyTalk** | speech to text, wake phrase, microphone | `babytalk`, `babytalk_listener` / `BabyTalk` | the model partition; our audio drivers (below) |

Status: speech to text, wake phrase and microphone work on the Waveshare ESP32-S3-CAM. Text to
speech (which works under MicroPython) is not ported yet: it needs ~104 KB more internal RAM
than AtomVM + WiFi leave, and a driver for the speaker codec.

## Device drivers: we bring our own

AtomVM's driver catalog is still growing, and it has **no I2S driver** at all, so BabyTalk
ships its own C drivers for the audio chips and exposes them to Erlang as NIFs. They are
written for one board, the **Waveshare ESP32-S3-CAM**:

| Chip | Role | Bus | Driver | Status |
|---|---|---|---|---|
| **Everest ES7210** | 4-channel microphone ADC (we use mic 1) | I2C `0x40` (setup) + I2S in, 16 kHz, with MCLK | `stt/main/mic.c` | working |
| **WCH CH32V003** | small RISC-V MCU acting as an 8-bit I2C IO expander; gates the audio rail (P6) and the speaker amp (P4) | I2C `0x24` | `stt/main/mic.c` (rail only) | working |
| **Everest ES8311** | mono speaker codec (DAC) | I2C + I2S out | not yet (MicroPython has `mpy/drivers/es8311.py`) | for TTS |
| **NS4150B** | 3 W class-D speaker amp | enable = expander P4 | not yet | for TTS |

These use ESP-IDF 5.5's current drivers (`i2c_master`, `i2s_std`). ESP-IDF aborts at boot if
the legacy I2C driver is linked next to the new one, so this firmware **turns off AtomVM's own
I2C** (`CONFIG_AVM_ENABLE_I2C_PORT_DRIVER=n`, `..._RESOURCE_NIFS=n`): Erlang code gets no
`i2c` module on this build, and C owns the bus. (Revisit when AtomVM moves to the new I2C
driver.) Another board needs its own pins and codec setup in `mic.c`; everything above the
driver (engine, NIFs, Erlang) is board-independent. MMRT has no drivers and runs on any
ESP32-S3 AtomVM build.

## OTP structure

The C side is deliberately small: NIFs start work and report back with **messages**, and the
supervision, state and policy are Erlang.

- `babytalk:transcribe/1` returns `{ok, Ref}` at once; inference runs on a FreeRTOS task
  and the result arrives as `{babytalk, Ref, {ok, Text, Info}}`. (AtomVM has no dirty NIFs,
  and a scheduler must never block for a second.)
- `babytalk:listen/1` streams the mic as `{babytalk_mic, Ref, Pcm}` chunks, gapless, even
  while a transcription runs; `stop_listening/0` ends it with `{babytalk_mic, Ref, stopped}`.
- `babytalk_listener` is a **gen_server** that owns the mic stream, scores a sliding window
  against the wake phrase about once a second, transcribes the command after a wake, and
  sends events to a subscriber:
  `{babytalk_listener, {heard, Text, Score} | {wake, Text, Score} | {command, Text}}`.
  Put it under a supervisor: if the mic or engine fails it crashes, and the restart
  re-opens the stream.
- `apps/listen_demo` shows the shape: a `rest_for_one` supervisor over a printer
  (the subscriber) and the listener.

```
listen_demo_sup (rest_for_one)
|-- listen_demo_printer      handle_info({babytalk_listener, Event}, ...)
`-- babytalk_listener        mic stream -> window -> wake -> command
```

One engine and one microphone: a second concurrent `transcribe` or `listen` gets
`{error, busy}`, so put a single owner process (like the listener) in front of them.

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
esptool.py --chip esp32s3 write_flash 0x390000 listen_demo.avm
```

`build.sh` clones the pinned AtomVM tag, applies `patches/`, links in `components/` (ours and
the shared `mmrt`, `sram_pool`, `stt_engine` from the repo root), applies
`sdkconfig.babytalk` and builds. The Erlang/Elixir standard libraries (`boot.avm`) come from
the official release image, so no host build of AtomVM is needed.

| Partition | Offset | Size | |
|---|---|---|---|
| factory | 0x10000 | 3 MB | AtomVM + NIFs |
| boot.avm | 0x310000 | 512 KB | Erlang + Elixir libraries |
| main.avm | 0x390000 | 1 MB | your app |
| model | 0x490000 | 6 MB | the int4 `.mmrt` model |

Apps: `apps/mmrt_test` (MMRT test vectors from `gen_vectors.py` + benchmarks),
`apps/stt_server` + `tools/stt_client.py` (send WAVs over WiFi, get transcripts back),
`apps/listen_demo` (wake phrase + command). Apps that use WiFi need
`gen_wifi_creds.sh <app dir>`, which writes a git-ignored `wifi_creds.erl`.

## Measured (Waveshare ESP32-S3-CAM, 240 MHz, octal PSRAM)

- **Same transcripts as MicroPython**: 8 field clips x 4 runs, 32/32 identical to the
  MicroPython firmware on the same PCM; a 7.7 s clip takes 1.33 s (MicroPython: 1.41 s).
- **MMRT**: 256x256 int8 matvec 453 us (145 MMAC/s, limited by PSRAM bandwidth); matmul with
  16 rows 720 MMAC/s; int4 466 MMAC/s; the exact C fallback 22 MMAC/s. For scale: a dot
  product in plain Erlang over lists runs at 0.19 MMAC/s.
- **Internal RAM**: 112 KB free at start (after the engine's 84.5 KB pool is reserved),
  42 KB after WiFi joins, ~31 KB during inference.

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

## Risks

AtomVM v0.7 is an alpha (the project recommends v0.6 for production). While a transcription
runs, MMRT keeps both cores busy (its helper task on core 0), so Erlang code and WiFi respond
more slowly for those ~0.2 s per second of audio.
