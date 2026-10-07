# BabyTalk and MMRT for AtomVM (Erlang / Elixir on the ESP32-S3 and ESP32-P4)

On-device speech to text, text to speech, a wake phrase, microphone and speaker, plus the
int8/int4 vector kernels underneath, for [AtomVM](https://github.com/atomvm/AtomVM)
**v0.7.0-beta.0** on the ESP32-S3 and the ESP32-P4. Everything runs on the board: no cloud,
no network needed (WiFi is only for the test tools).

```erlang
ok = babytalk:audio_config(waveshare_s3_cam),                  % which pins and chips (a preset, or your own)
{ok, Pcm} = babytalk:record(3),                               % 3 s from the board's mic
{ok, Text, _Info} = babytalk:transcribe_sync(Pcm, 10000),     % "what's the weather like today"
ok = babytalk:speak([<<"You said: ">>, Text]),                 % out of the board's speaker

W = mmrt:pack({int8, Weights}, 64, 256),                      % 64 x 256 matrix, packed once
{int8, Y} = mmrt:matvec(W, {int8, X}, 7).                     % SIMD, bit-exact
```

```elixir
:ok = BabyTalk.audio_config(%{board: :waveshare_p4_wifi6, amp: {:gpio, 53}})
{:ok, text, _info} = BabyTalk.transcribe_sync(pcm, 10_000)
```

**Contents:** [What's here](#whats-here) · [Quick start](#quick-start) ·
[From Elixir](#from-elixir) ·
[Boards and pins](#boards-and-pins) · [OTP structure](#otp-structure) ·
[API reference](#api-reference): [`babytalk`](#babytalk--babytalk),
[`babytalk_listener`](#babytalk_listener--babytalklistener), [`mmrt`](#mmrt--mmrt), [errors](#error-reference) ·
[Build and flash](#build-and-flash) · [Example apps](#example-apps) · [Measured](#measured) ·
[AtomVM lessons](#things-we-learned-atomvm-specifics) · [Risks and limits](#risks-and-limits)

## What's here

Two libraries, usable separately:

| | What | Modules | Needs |
|---|---|---|---|
| **MMRT** | int8/int4 matvec, matmul, dot, requant, top-k... on the chip's SIMD unit | `mmrt` / `MMRT` | an ESP32-S3 or ESP32-P4 |
| **BabyTalk** | speech to text, text to speech, wake phrase, mic + speaker, a conversation `gen_server` | `babytalk`, `babytalk_listener` / `BabyTalk`, `BabyTalk.Listener` | the `model` partition; a board with supported audio chips ([Boards and pins](#boards-and-pins)) |

| Where | What |
|---|---|
| `lib/babytalk/`, `lib/mmrt/` | the Erlang modules (and Elixir wrappers in `lib/*/lib/`) |
| `components/atomvm_babytalk/` | the BabyTalk NIFs and audio driver (C, an ESP-IDF component) |
| `components/atomvm_mmrt/` | the MMRT NIFs |
| `../components/` | engines shared with the MicroPython firmware: `mmrt`, `stt_engine`, `sanotts`, `sram_pool` |
| `apps/` | example and test apps ([Example apps](#example-apps)) |
| `build.sh`, `sdkconfig.babytalk`, `partitions-babytalk.csv`, `patches/` | the firmware build |

**Status.** All of it works on the **Waveshare ESP32-S3-CAM**: say "wake up tomato face", wait
for the chime, say something, and the board says it back (`apps/listen_demo`). It also runs
in Watchtower, our network of sensor and messaging nodes, on four boards, next to
WiFi, a camera or a screen:

| Board | Audio | How | Speech |
|---|---|---|---|
| Waveshare ESP32-S3-CAM | ES7210 mic, ES8311 speaker | built-in driver, default board | to text and back; wake phrase, hands free |
| Waveshare ESP32-P4-WIFI6 ([notes](../docs/BOARD_WAVESHARE_P4_WIFI6.md)) | ES8311 mic and speaker | built-in driver, default board | to text and back; the larger `heart` voice ([TTS_VOICES.md](../docs/TTS_VOICES.md)) |
| LilyGO T-LoRa Pager | ES8311 mic and speaker, NS4150B amp on an XL9555 expander | its own driver ([Other audio chips](#other-audio-chips)) | to text and back; push to talk and wake phrase |
| Elecrow CrowPanel Advance 7.0" | PDM mic | its own driver | to text only (no speaker fitted): a mic key on the on-screen keyboard |

Those firmwares pick their board in Kconfig. Changing the board from Erlang
(`audio_config/1`) is tested on the PC but not yet on a board.

## Quick start

Needs ESP-IDF 5.5, Erlang/OTP 26+ and Elixir 1.17+ (for the `exatomvm` mix tasks).

```bash
# optional, for text to speech: sanoTTS at the commit we audited (build.sh refuses any other).
# Without it the firmware has speech to text only, and babytalk:say/1 returns {error, -4}.
git clone https://github.com/Ampixa/sanoTTS ~/build/tts/sanoTTS     # or anywhere: SANOTTS_DIR=...
git -C ~/build/tts/sanoTTS checkout 18e26b2b365bff41d211e516b0760021451438f1

atomvm/build.sh                                   # ESP32-S3 -> ~/build/atomvm-out/atomvm-babytalk.img
esptool.py --chip esp32s3 write_flash 0x0 ~/build/atomvm-out/atomvm-babytalk.img \
    0x490000 models/citrinet256_int4.mmrt         # firmware + Erlang/Elixir libs, then the model

atomvm/build.sh esp32p4_pre_c6                    # or an ESP32-P4 + C6 (Waveshare ESP32-P4-WIFI6)
esptool.py --chip esp32p4 write_flash 0x0 ~/build/atomvm-out-esp32p4_pre_c6/atomvm-babytalk.img \
    0x490000 models/citrinet256_int4.mmrt

cd atomvm/apps/listen_demo && mix deps.get && mix atomvm.packbeam
esptool.py --chip esp32s3 write_flash 0xA90000 listen_demo.avm
```

Say "wake up tomato face", wait for the second chime, say something short.

A firmware image with text to speech should be treated as GPL-3.0 if you distribute it: three
sanoTTS files carry no licence ([../components/sanotts/LICENSES.md](../components/sanotts/LICENSES.md)).

**In your own app**, compile the library sources along with yours. With Mix + exatomvm:

```elixir
def project do
  [app: :my_app, version: "0.1.0", elixir: "~> 1.17",
   erlc_paths: ["src", "path/to/atomvm/lib/babytalk/src"],        # + lib/mmrt/src for MMRT
   # the Elixir wrappers; an absolute path (Mix won't compile .ex files from a relative
   # path outside the project)
   elixirc_paths: ["lib", Path.expand("path/to/atomvm/lib/babytalk/lib", __DIR__)],
   deps: [{:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false},
          {:atomvm, "~> 0.7.0-beta.0", runtime: false}],   # the release's API, to check against
   atomvm: [start: :my_app, flash_offset: 0xA90000]]
end
```

The NIFs are compiled into the firmware; the `.erl` files give them names and types and hold
the Erlang-side logic (the listener, the configuration maps).

## From Elixir

Every Erlang module has an Elixir wrapper that delegates to it one for one: `BabyTalk`
(`:babytalk`), `BabyTalk.Listener` (`:babytalk_listener`) and `MMRT` (`:mmrt`). Arguments,
results and messages are the same as in the [API reference](#api-reference), with Elixir
syntax: Erlang's `<<"text">>` is `"text"`, and atoms get a colon (`{:ok, text, info}`).

```elixir
# Record 3 s, transcribe it, say it back
:ok = BabyTalk.audio_config(:waveshare_s3_cam)          # or a map of your own pins
{:ok, pcm} = BabyTalk.record(3)
{:ok, text, info} = BabyTalk.transcribe_sync(pcm, 10_000)
IO.puts("heard: " <> text <> " in #{info[:model_ms]} ms")
:ok = BabyTalk.speak(["You said: ", text])

# Asynchronous: the result arrives as a message
{:ok, ref} = BabyTalk.transcribe(pcm)
receive do
  {:babytalk, ^ref, {:ok, text, _info}} -> text
  {:babytalk, ^ref, {:error, code}} -> {:error, code}
end

# Stream the mic in 250 ms chunks
{:ok, ref} = BabyTalk.listen(250)
receive do
  {:babytalk_mic, ^ref, chunk} -> IO.puts("level #{BabyTalk.rms(chunk)}")
end
BabyTalk.stop_listening()

# A chime: {Hz, ms} notes, volume 70
{:ok, ref} = BabyTalk.tones([{587, 90}, {0, 30}, {880, 140}], 70)
receive do
  {:babytalk_play, ^ref, :done} -> :ok
end
```

**A voice assistant under a supervisor.** `BabyTalk.Listener` has a `child_spec/1`, so it
goes straight into a children list. It registers itself as `:babytalk_listener`; its events go
to `notify`:

```elixir
defmodule MyApp.Responder do
  use GenServer
  def start_link(_), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)
  def init(nil), do: {:ok, nil}
  def handle_call(_req, _from, s), do: {:reply, :ok, s}    # see "Elixir on AtomVM" below
  def handle_cast(_req, s), do: {:noreply, s}

  def handle_info({:babytalk_listener, {:command, text}}, s) do
    BabyTalk.Listener.say(:babytalk_listener, ["You said: ", text])
    {:noreply, s}
  end
  def handle_info({:babytalk_listener, _event}, s), do: {:noreply, s}
end

children = [
  MyApp.Responder,
  {BabyTalk.Listener, %{notify: MyApp.Responder, spellings: ["hey computer"], threshold: -21.0}}
]
Supervisor.start_link(children, strategy: :rest_for_one)
```

The whole app, with a few commands: [`apps/voice_commands`](apps/voice_commands).

**MMRT** from Elixir:

```elixir
w = MMRT.pack({:int8, weights}, 64, 256)          # 64 outputs x 256 inputs, packed once
{:int8, h} = MMRT.matvec(w, {:int8, x}, 7)
[{best, _score} | _] = MMRT.top_k({:int8, h}, 3)
```

**Elixir on AtomVM.** AtomVM runs a subset of Elixir's standard library (`exavmlib`):
`Enum`, `Map`, `Keyword`, `List`, `IO`, `GenServer`, `Supervisor` and string interpolation are
there; **`String` is not**. Match text with `:binary` (`:binary.match(text, "lights") !=
:nomatch`, `:binary.split/2`). And `use GenServer` compiles in default `handle_call/3` and
`handle_cast/2` that call `:erlang.phash2/2`, which AtomVM lacks: define both yourself, as
above. `mix atomvm.packbeam` warns about calls the release doesn't have; take the warnings
seriously.

## Boards and pins

AtomVM has **no I2S driver**, so BabyTalk ships its own C audio driver
(`components/atomvm_babytalk/board_audio.c`) and exposes it as NIFs. It is **configured at run
time**: build the firmware once per chip (ESP32-S3 or ESP32-P4) and pick the pins and parts
from your app. A different board needs no rebuild, as long as it uses these parts:

| Part | Role | Driven over | Setting |
|---|---|---|---|
| **Everest ES7210** | 4-channel microphone ADC (one channel used) | I2C (setup) + I2S in | `mic => #{codec => es7210}` |
| **Everest ES8311** | mono codec: its DAC for the speaker, and/or its ADC for a mic | I2C + I2S | `speaker => #{codec => es8311}`, `mic => #{codec => es8311}` |
| speaker amp enable (NS4150B and the like) | on only while playing (no hiss, no pop) | a GPIO, or an I2C IO expander pin | `amp => ...` |
| audio power rail (optional) | on from bring-up | the same | `power => ...` |
| IO expanders | **WCH CH32V003** as Waveshare programs it (8 pins, write-only latch: we keep a shadow), **PCA9555 / TCA9555 / XL9555** (16 pins) | I2C | `{ch32v003, Addr, Pin}`, `{pca9555, Addr, Pin}` |

The ESP32 is always the I2S master with MCLK = 256 x Fs, so each codec needs one fixed set
of clock dividers for every sample rate. The mic and the speaker share one I2S port and take
turns: the board is either listening or playing, never both. PCA9555 support is written to
the datasheet but hasn't been used on a board yet.

### Presets

| Preset | Board | Mic | Speaker | Amp / rail | I2C (SDA, SCL) | I2S (MCLK, BCLK, WS, out, in) |
|---|---|---|---|---|---|---|
| `waveshare_s3_cam` | Waveshare ESP32-S3-CAM ([notes](../docs/BOARD_WAVESHARE_S3_CAM.md)) | ES7210 at `0x40` | ES8311 at `0x18` | CH32V003 at `0x24`: amp pin 4, rail pin 6 | 8, 7 | 10, 11, 12, 14, 13 |
| `waveshare_p4_wifi6` | Waveshare ESP32-P4-WIFI6 ([notes](../docs/BOARD_WAVESHARE_P4_WIFI6.md)) | ES8311 at `0x18` (its ADC) | the same ES8311 | GPIO 53 / none | 7, 8 | 13, 12, 10, 9, 11 |

The firmware starts out assuming its default board (Kconfig `BABYTALK_BOARD`: the S3-CAM on
an S3, the P4-WIFI6 on a P4), but **touches no pin** until the first `listen`, `play` or
`tones`, or an `audio_config/1` call. So configure first, then use audio.

### Configuring your board

```erlang
ok = babytalk:audio_config(waveshare_p4_wifi6).                           % a preset
ok = babytalk:audio_config(#{board => waveshare_p4_wifi6, amp => {gpio, 46}}).   % a preset, changed
ok = babytalk:audio_config(#{i2s => #{din => 21}}).                       % change the board in use
{ok, Cfg} = babytalk:audio_config().                                      % what's in force
{ok, Cfg} = babytalk:audio_preset(waveshare_s3_cam).                      % a starting point
[waveshare_s3_cam, waveshare_p4_wifi6] = babytalk:audio_presets().
```

A change map is merged **section by section** into the board in use (or into `board`'s
preset): `#{i2s => #{din => 21}}` keeps the other I2S pins. The full form, with every key:

```erlang
#{i2c     => #{port => 0, sda => 8, scl => 7, speed_hz => 100000},
  i2s     => #{port => 0, mclk => 10, bclk => 11, ws => 12, dout => 14, din => 13},
  mic     => #{codec => es7210, address => 16#40, gain => default, slot => left},
  speaker => #{codec => es8311, address => 16#18},
  amp     => {ch32v003, 16#24, 4},
  power   => {ch32v003, 16#24, 6}}
```

| Key | Values |
|---|---|
| `i2c.port`, `i2s.port` | the controller number (0, 1...) |
| `sda`, `scl`, `mclk`, `bclk`, `ws`, `dout`, `din` | GPIO numbers; `none` for `dout` without a speaker, `din` without a mic, `mclk` if unused |
| `i2c.speed_hz` | 10000..1000000 (default 100000) |
| `mic.codec` | `es7210`, `es8311` or `none` (also: `mic => none`) |
| `mic.gain` | `default`, or the codec's own step: ES7210 0..14 (PGA), ES8311 0..7 (6 dB each). Defaults: ES7210 14, ES8311 4 |
| `mic.slot` | `left` or `right`: the I2S slot the mic is on |
| `speaker.codec` | `es8311` or `none` (also: `speaker => none`) |
| `address` | the chip's 7-bit I2C address |
| `amp`, `power` | `none` · `{gpio, Pin}` · `{ch32v003, Addr, Pin}` · `{pca9555, Addr, Pin}`, each optionally with `active_low` as a last element: `{gpio, 46, active_low}` |

With one ES8311 as both mic and speaker (same address in both), it's set up for both
directions at once; an ES8311 as speaker only gets the DAC-only setup.

`audio_config/1` releases the old pins, brings the new board up right away and waits for it,
so a wrong pin or address shows immediately:

| Result | Meaning |
|---|---|
| `ok` | the chips answered and are set up |
| `{error, {unknown_key, Where}}` | a key that doesn't exist, e.g. `[i2s, dat]` |
| `{error, {bad_value, Where}}` | not a valid value, or not a valid pin on this chip (e.g. GPIO 53 on an S3): `[i2s, din]`, `[amp]` |
| `{error, {unknown_board, Name}}` | no such preset |
| `{error, i2c_bus}` | the I2C bus couldn't be opened on those pins |
| `{error, switch}` | the amp or rail switch failed (an expander didn't answer) |
| `{error, mic_codec}` / `{error, speaker_codec}` | that codec didn't answer at that address |
| `{error, busy}` | the mic or speaker is in use: stop listening first |
| `{error, timeout}` | no answer from the audio task in 5 s |
| `{error, custom_board}` | this firmware has its own C driver (below) |

In an OTP app, put the board in the listener's options instead (`audio => ...`, see
[`babytalk_listener`](#babytalk_listener--babytalklistener)); it's applied before the listener first listens,
and a failure stops the listener with `{audio_config, Reason}`.

**Finding your pins.** The board's schematic or vendor BSP header lists the codec's I2C
address and the I2S pins; the amp enable ("PA", "PA_CTRL", "SPK_EN") is usually a GPIO or an
expander pin. If your board uses the same parts as a preset, start from the preset and change
what differs.

**Sharing the I2C bus.** If something already opened an I2C bus on that port (a camera's
SCCB, a display, a touch controller), BabyTalk adds its chips to that bus and never closes it;
the pins are then whatever that driver chose. AtomVM's own `i2c` module is **not available**
in this firmware: ESP-IDF aborts at boot if its legacy I2C driver (which AtomVM uses) is linked
next to the new one (which ours uses), so `sdkconfig.babytalk` sets
`CONFIG_AVM_ENABLE_I2C_PORT_DRIVER=n` and `..._RESOURCE_NIFS=n`. Revisit when AtomVM moves to
the new driver.

### Other audio chips

For codecs the built-in driver doesn't know (a WM8960, an I2S MEMS mic, a MAX98357A...),
write a C component that implements `components/atomvm_babytalk/board_audio.h` (init,
capture start/read/stop, play, an underrun count) and build with
`CONFIG_BABYTALK_BOARD_CUSTOM=y`. Everything above that header (engines, NIFs, Erlang) is
board-independent; `board_audio.c` is a worked example. In a custom build, `audio_config/0,1`
return `{error, custom_board}` and `audio_presets/0` returns `[]`.

Two custom drivers run in Watchtower: the T-LoRa Pager's (ES8311 and an amp on an XL9555,
written before run-time configuration: the built-in driver could probably now do it with a
map) and the CrowPanel's PDM mic. What they taught us:

- `board_audio_rx_read` fills every frame and returns **0**, not the number of frames: any
  other value is taken as an error.
- The ESP32-S3's PDM receiver has no high-pass filter, and its output sits on a large DC
  offset. Run a one-pole DC blocker (~40 Hz) over it, seeded with the first sample so it
  doesn't start with a step.
- If the board's I2C bus is already open (a touch panel, a camera), use that bus's handle.

## OTP structure

The C side is deliberately small: NIFs start work and report back with **messages**, and the
supervision, state and policy are Erlang.

- **Nothing blocks a scheduler.** `transcribe/1` returns `{ok, Ref}` at once; inference runs
  on a FreeRTOS task and the result arrives as `{babytalk, Ref, {ok, Text, Info}}`. AtomVM has
  no dirty NIFs, and a scheduler must never block for a second.
- **One engine, one audio port.** Transcription and synthesis share one engine; the mic and
  the speaker share one I2S port. A second concurrent call gets `{error, busy}`, so put a
  single owner process in front of them, like the listener.
- `babytalk:listen/1` streams the mic as `{babytalk_mic, Ref, Pcm}` chunks, gapless, even
  while a transcription runs; `stop_listening/0` ends it with `{babytalk_mic, Ref, stopped}`.
- `babytalk_listener` is a **gen_server** that owns the mic *and* the speaker and runs the
  conversation, with chimes so a person can follow along:

  | Phase | Chime on entry | What it does |
  |---|---|---|
  | wake | *ready* (D5 -> A5) | scores the last 4 s against the wake phrase about once a second |
  | message | *wake* (A5 -> D6) | records until 700 ms of quiet after speech (or 6 s) |
  | idle | *decoding* (G5 -> C5) | transcribes, reports `{command, Text}`, speaks any reply, then *ready* again |

- `apps/listen_demo` shows the shape: a `rest_for_one` supervisor over a responder (the
  subscriber, which answers "You said: ...") and the listener.

```
listen_demo_sup (rest_for_one)
|-- listen_demo_responder    handle_info({babytalk_listener, {command, Text}}, ...) -> say/2
`-- babytalk_listener        mic + speaker: wake -> message -> transcribe -> reply
```

Put the listener under a supervisor: if the mic or the engine fails it crashes, and the
restart re-opens everything. It traps exits, so its `terminate/2` releases the mic even when
its supervisor shuts it down.

## API reference

Conventions for everything below:

- **Audio in** is 16 kHz mono signed 16-bit little-endian PCM in a binary. **Speech out**
  (`say`) is 24 kHz, same format. `play` takes any rate from 8 to 48 kHz.
- **Async calls** return `{ok, Ref}` and later send the caller one message tagged with that
  `Ref`. The `_sync` variants wait for it.
- **`{error, busy}`**: the engine or the audio port is taken (one user at a time).
  **`{error, no_memory}`**: PSRAM for a copy of the input couldn't be allocated.
- **Bad arguments** (wrong type, out of range) raise `badarg`, as BIFs do.
- `Code` in `{error, Code}` is a negative integer from C: see [Error reference](#error-reference).

### `babytalk` / `BabyTalk`

The Elixir module `BabyTalk` delegates every function 1:1 (`BabyTalk.transcribe_sync/2`...).

#### Speech to text

**`transcribe(Pcm) -> {ok, Ref} | {error, busy | no_memory}`**
Starts transcribing; later `{babytalk, Ref, {ok, Text, Info} | {error, Code}}`. `Pcm` is
copied, so it can be dropped at once. Up to 64 s of audio; about 0.2 s of processing per
second of audio on the S3 (0.15 on the P4), both cores busy meanwhile. `Text` is lower-case
English (a binary, `<<>>` if nothing was heard). `Info`:

| Key | Meaning |
|---|---|
| `score` | the wake phrase's score (see `phrase/1`): 0.0 = the phrase is the best reading of the audio, more negative = less likely; `undefined` when no phrase is set |
| `span` | `{First, Last}`: where the phrase matched, in 80 ms model frames |
| `frames` | 10 ms feature frames in the audio |
| `fe_ms`, `model_ms`, `dec_ms` | time in the front end (mel features), the model, the decoder |

**`transcribe_sync(Pcm, Timeout) -> {ok, Text, Info} | {error, busy | no_memory | timeout | Code}`**
`transcribe/1` and wait up to `Timeout` ms.

```erlang
{ok, Pcm} = babytalk:record(4),
{ok, Text, Info} = babytalk:transcribe_sync(Pcm, 10000),
io:format("~s (~p ms)~n", [Text, proplists:get_value(model_ms, Info)]).
```

**`phrase(Spellings) -> {ok, Sequences} | {error, busy | no_memory | Code}`**
Sets the wake phrase every later transcription's `score` is measured against: one or more
spellings (binaries or strings; only letters count), `[]` to clear. Several spellings of how
the model tends to hear the phrase work better than one (`../docs/MODELS.md` section 3,
`export/kws.py`). Returns the number of token sequences they expand to. Opens the engine
(and so maps the model) if needed.

**`cache(Bytes) -> {ok, Cached} | {error, busy | Code}`**
Keeps up to `Bytes` of model weights in PSRAM (faster than reading them from flash: ~20 ms
saved per MB per transcription). `0` releases them. Returns the bytes actually cached.

#### Microphone

**`listen(ChunkMs) -> {ok, Ref} | {error, busy}`**
Streams the mic to the caller in `ChunkMs` pieces (20..10000 ms), gapless, while
transcriptions keep running:

| Message | When |
|---|---|
| `{babytalk_mic, Ref, Pcm}` | every `ChunkMs` |
| `{babytalk_mic, Ref, stopped}` | after `stop_listening/0` (the last message) |
| `{babytalk_mic, Ref, {error, Code}}` | a capture failure (also the last message) |

One listener at a time, and not while the speaker plays. The first 200 ms after starting
are discarded (the ADC settling).

**`stop_listening() -> ok`**
Ends the stream after the current chunk. Wait for `stopped` before playing: the port is
free only then.

**`record(Secs) -> {ok, Pcm} | {error, busy | timeout | term()}`**
Records `Secs` seconds (a number) and returns the PCM; `listen` + `stop_listening`
underneath.

**`rms(Pcm) -> 0..32768`**
The root-mean-square level of 16-bit PCM, computed in C. Room noise is typically a few
hundred, speech at arm's length 1000 and up.

#### Text to speech

**`say(Text) -> {ok, Ref} | {error, busy}`**, **`say(Text, #{length_scale => S})`**
Starts synthesizing `Text` (English, iodata) with the sanoTTS `en_us_e12nano` voice; later
`{babytalk, Ref, {ok, Pcm24k, Info} | {error, Code}}`. `S` is the pace: 1.0 = the voice's
own, 1.1 = 10% slower (0.5..2.0). Shares the engine with transcription. Takes about a
quarter of the speech's duration (S3: 0.27x real time). `Info`: `rate` (24000), `phonemes`,
`g2p_ms` (pronunciation), `synth_ms`, `simd_pct` (the share of multiply-adds on the vector
unit).

**`say_sync(Text, Timeout)`, `say_sync(Text, Opts, Timeout) -> {ok, Pcm24k, Info} | {error, term()}`**
`say` and wait.

**`speak(Text) -> ok | {error, term()}`**, **`speak(Text, Volume)`**
Says `Text` out loud and returns when done: `say_sync` + `play` (volume default 76). The
simplest way to make the board talk. Long text is synthesized whole first, so it's silent for
a while; the listener speaks a sentence at a time instead.

#### Speaker

**`play(Pcm, Rate) -> {ok, Ref} | {error, busy | no_memory}`**, **`play(Pcm, Rate, Volume)`**
Plays mono PCM at `Rate` Hz (8000..48000); later `{babytalk_play, Ref, done | {error, Code}}`.
`Volume` 0..100 sets the codec's DAC: 75 = 0 dB, about 0.5 dB per step; default 76. Speech
from `say` already peaks at 77% of full scale, so much above ~80 it clips. The amp is on
only while playing. `{error, busy}` while listening.

**`tones(Notes, Volume) -> {ok, Ref} | {error, busy | no_memory}`**
Plays notes made on the board (chimes, cues); same messages and rules as `play/3`. Each note
is `{Hz, Ms}`, `{Hz, Ms, Level}` or `{Hz, Ms, Level, Shape}`: Hz 0..8000 (0 = a rest), Ms >= 1,
Level 0..100 (of the note's full loudness; default 100), Shape `flat` (steady, 8 ms edges;
the default) or `bloop` (a soft rise then a natural fall, the pitch settling slightly: for
ambient sounds). Up to 24 notes and 6 s per call.

```erlang
{ok, R} = babytalk:tones([{587, 90}, {0, 30}, {880, 140}], 70),    % the listener's "ready" chime
receive {babytalk_play, R, done} -> ok end.
```

#### Board audio

- **`audio_config() -> {ok, Map} | {error, custom_board}`**: the board in force.
- **`audio_config(Board | Changes) -> ok | {error, term()}`**: use another board.
- **`audio_preset(Name) -> {ok, Map} | {error, {unknown_board, Name}}`**: a preset's full map.
- **`audio_presets() -> [Name]`**: the presets compiled in.

Keys, examples and errors: [Boards and pins](#boards-and-pins).

#### Memory and model facts

**`info() -> [{Key, Integer}]`**
`model_bytes`, `cache_bytes`, `ops`, `int4_ops` (0 until the engine is first used),
`internal_free`, `psram_free`.

**`heap_info() -> [{Key, Integer}]`**
`internal_free`, `internal_largest`, `psram_free`, `psram_largest`, `pool_bytes`.
`pool_bytes` is the engines' 84.5 KB internal-RAM block: 0 if it wasn't reserved at boot,
and then transcription and speech can't run.

**`reserve_at_boot(Bool) -> ok | {error, term()}`**
Whether BabyTalk takes its 84.5 KB of internal RAM when the VM next starts (it must happen
early, before WiFi and the rest fragment the heap). Stored in NVS (namespace `babytalk`, key
`reserve`); takes effect after a restart; overrides the firmware's
`CONFIG_BABYTALK_RESERVE_AT_BOOT`. Boards short of internal RAM build with it off and let the
app opt in.

### `babytalk_listener` / `BabyTalk.Listener`

A `gen_server` that owns the mic and the speaker: listens for a wake phrase, records the
message after it until the speaker stops talking, transcribes it, reports it, and speaks
replies, with chimes between the phases.

**`start_link(Opts)`**, **`start_link(Name, Opts)`** (registered locally as `Name`),
**`stop(Server)`**. `Opts` is a map:

| Option | Default | Meaning |
|---|---|---|
| `notify` | (required) | pid or registered name that gets the events |
| `audio` | `none` | the board, as `babytalk:audio_config/1` takes it (a preset name or a map); `none` leaves the board in use alone |
| `spellings` | `[]` | the wake phrase's spellings; `[]` = no wake phrase: the listener chimes and waits for `ask/3` or `wake_on/2` |
| `threshold` | `-21.0` | wake when the phrase's score is at least this (from `export/kws.py`) |
| `window_ms` / `every_ms` | `4000` / `1000` | the audio scored for the phrase, and how often. A short phrase does better with a short window scored often: Watchtower uses `1500` / `500` (quicker to transcribe, and the phrase lands whole in one window) |
| `vad` | `true` | score only after something louder than the room was heard (a transcription takes both cores for 0.3-1.5 s) |
| `vad_factor` / `vad_min` | `2` / `200` | "louder": this many times the noise floor, and at least this RMS |
| `silence_ms` | `700` | quiet after speech that ends a message |
| `max_message_ms` | `6000` | the longest message |
| `reply_ms` | `1500` | how long to wait for a reply to a command before listening again |
| `chunk_ms` | `125` | mic chunk size |
| `cues` | `true` | play the chimes |
| `cue_volume` / `voice_volume` | `70` / `76` | chime and speech volume (0..100) |
| `length_scale` | `1.10` | speech pace: 10% slower than the voice's own, easier to follow |
| `greeting` | `none` | said after the wake chime, e.g. `<<"I'm here, how can I help?">>` |

Events arrive as `{babytalk_listener, Event}`:

| Event | When |
|---|---|
| `{heard, Text, Score}` | each scored wake window (`Score` is `undefined` with no phrase) |
| `{wake, Text, Score}` | the phrase was heard: recording the message |
| `{message_end, silence \| max_length \| fixed_length, #{ms, floor, peak, levels}}` | the message stopped (`levels`: the RMS of each chunk) |
| `{command, Text}` | the message after the wake phrase (`<<>>` if nothing was understood) |
| `{answer, Text}` | the answer to `ask/3` |
| `{said, Text}` | `say/2` finished speaking `Text` |
| `{error, Code}`, `{say_error, Code}`, `{play_error, Code}` | a wake-window transcription, a synthesis or a playback failed |

Calls (all casts, returning `ok`):

- **`say(Server, Text)`**: speak `Text` (iodata). Right after a `{command, _}` it counts as
  the reply: spoken, then the ready chime, then wake listening. Listening pauses while it
  speaks (so it doesn't hear itself). Long text is said about a sentence (at most 160 bytes)
  at a time: each piece is synthesized while the previous one plays.
- **`chime(Server, Notes)`**, **`chime(Server, Notes, Volume)`**: play notes (as
  `babytalk:tones/2`) in turn with speech, at the cue volume or `Volume`.
- **`ask(Server, Prompt, Opts)`**: say `Prompt`, chime, record the answer until silence (or
  exactly `#{ms => Ms}`), report `{answer, Text}`, then wait (mic off) for the next call.
- **`wake_on(Server, Spellings)`**: set a new wake phrase and start the wake loop.
- **`hold(Server, Ms)`**: no wake-phrase scoring for the next `Ms`, e.g. while someone is
  using the device's screen or keyboard, so the scoring doesn't slow it down. Each call
  sets the hold to `Ms` from now.

`sentences(Text)` is also exported: the pieces `say/2` would speak.

```erlang
{ok, L} = babytalk_listener:start_link(#{notify => self(), audio => waveshare_s3_cam,
                                         spellings => [<<"hey computer">>]}),
receive {babytalk_listener, {command, Text}} ->
    babytalk_listener:say(L, [<<"You said ">>, Text])
end.
```

A voice-enrolled wake phrase: `apps/wakeword_demo` (`ask/3` for "say your wake word", a
yes/no confirmation, then `wake_on/2` with the model's own spelling of what it heard).

### `mmrt` / `MMRT`

int8/int4 vector kernels on the chip's SIMD unit (the S3's PIE, the P4's vector
extension), over tagged binaries. Independent of the speech model: use them for your own
small networks, similarity search, filters.

| Data | Meaning |
|---|---|
| `{int8, Bin}` | one signed byte per element |
| `{int4, Bin}` | signed nibbles (-8..7), low nibble first, two per byte |
| `{int32, Bin}` | little-endian int32 (accumulators) |
| `{int8_m16 \| cb4_m16, Rows, Cols, MaxRowL1, Bin}` | a matrix packed by `pack/3` for the SIMD kernel, padded to multiples of 16. `Bin` is an ordinary binary: keep it, send it, or build it on the host |

| Call | Result |
|---|---|
| `pack({int8 \| int4, RowMajor}, Rows, Cols)` | the packed matrix (int4 -> `cb4_m16`, half the bytes). Pack once, reuse |
| `matvec(M, {int8, X})` | `{int32, _}`: Rows exact accumulators |
| `matvec(M, {int8, X}, Shift)` | `{int8, _}`: sat8(round_half_up(acc >> Shift)); a negative Shift multiplies by 2^-Shift |
| `matmul(M, {int8, Frames}, Shift)` | `Frames` holds T rows of Cols; the result T rows of Rows, each as `matvec/3`. Weights load once for all T rows: much faster than T `matvec` calls |
| `dot(A, B)` | the exact dot product of two vectors of the same length, any element types |
| `add(A, B)` | saturating elementwise sum, both int8 or both int32 |
| `relu(V)` | int8 or int32 |
| `requant({int32, _}, Shift)` | `{int8, _}`: sat8(round_half_up(V >> Shift)) |
| `argmax(V)` | the 0-based index of the first largest element |
| `top_k(V, K)` | the K largest as `[{Index, Value}]`, largest first, ties by lower index |
| `to_int4({int8, _})` / `from_int4({int4, _})` | conversions (clamped to -8..7; an odd-length int4 vector gains a trailing 0) |

`matvec`/`matmul` use the SIMD unit when the accumulators provably fit its 20-bit lanes
(`MaxRowL1 x max|x| < 2^19`; on the P4 also `Shift <= 13`) and an exact path otherwise (on
the P4, row dot products on its vector unit): the results are identical either way. Calls run
on the calling scheduler, so each is capped at 4M multiply-adds (a few ms): above that they
raise `too_big`; split the work.

```erlang
W = mmrt:pack({int8, Weights}, 64, 256),        % 64 outputs x 256 inputs
{int8, H} = mmrt:matvec(W, {int8, X}, 7),
[{Best, _} | _] = mmrt:top_k({int8, H}, 3).
```

### Error reference

`{error, Code}` from the C side:

| Where | Code | Meaning |
|---|---|---|
| transcription | -1 | no `model` partition (or it couldn't be mapped) |
| | -2 | the `model` partition doesn't hold a valid model image |
| | -3 | no memory for the engine's buffers (internal RAM: check `heap_info/0`'s `pool_bytes`) |
| | -4 | audio shorter than 10 ms or longer than 64 s |
| synthesis | -1 | nothing to say (no pronounceable words) |
| | -2 | no arena for the voice (the engines' pool isn't available) |
| | -3 | no memory for the resulting PCM |
| | -4 | this firmware was built without text to speech (no sanoTTS sources) |
| mic / speaker | -1 | the I2C bus couldn't be opened (see `audio_config/1`), or the port is capturing |
| | -2 | the amp or rail switch, or the DAC volume write, failed |
| | -3 / -4 | the mic's / speaker's codec didn't answer |
| | -5 / -6 / -7 | creating / starting / reading or writing the I2S channel failed (a pin conflict, or no DMA memory) |
| | -8 | this board has no mic (for `listen`) or no speaker (for `play`) |
| | -10 | no memory for a mic chunk |

## Build and flash

```bash
atomvm/build.sh [TARGET]          # or TARGET=... atomvm/build.sh
```

| `TARGET` | Chip | AtomVM tree / output (defaults) |
|---|---|---|
| `esp32s3` (default) | ESP32-S3, 16 MB flash, 8 MB octal PSRAM | `~/build/atomvm-<tag>`, `~/build/atomvm-out` |
| `esp32p4_pre_c6` | ESP32-P4 before silicon v3.0, ESP32-C6 for WiFi (the Waveshare ESP32-P4-WIFI6 we tested: v1.3) | `~/build/atomvm-<tag>-<TARGET>`, `~/build/atomvm-out-<TARGET>` |
| `esp32p4_c6` | ESP32-P4 v3.0 or later, ESP32-C6 for WiFi | same pattern |
| `esp32p4_pre`, `esp32p4` | ESP32-P4 without WiFi (before / from v3.0) | same pattern |

The P4 names are AtomVM's own presets. The boot log (or `esptool.py chip_id`) shows the
chip's revision. `<tag>` is the pinned AtomVM release (`v0.7.0-beta.0`; another with
`AVM_TAG`, if `patches/` still apply to it). ESP-IDF 5.5.1, which we build with, only builds for P4 silicon before v3.0:
for a v3 chip, use a later 5.5 release. `AVM_DIR`, `OUT_DIR` and `IDF_PATH` override the
locations; `PARTITIONS` the partition table (default `partitions-babytalk.csv`).

For every target, `build.sh` clones the pinned AtomVM tag (`AVM_DIR`), applies
`patches/` (fixes to AtomVM itself: see [AtomVM patches](#atomvm-patches)), links in `components/` (ours and the shared `mmrt`, `sram_pool`, `stt_engine`,
`sanotts` from the repo root), prepares the sanoTTS sources from a checkout (`SANOTTS_DIR`,
default `~/build/tts/sanoTTS`; without one the firmware builds without text to speech, and
`say` answers `{error, -4}`),
applies AtomVM's preset for the target and our settings (`sdkconfig.babytalk`, plus
`sdkconfig.babytalk.esp32s3` or `.esp32p4`, which ESP-IDF picks by chip), and builds. For a
P4 with a C6 it adds AtomVM's `esp_wifi_remote` dependency, so WiFi works through the C6
unchanged. The Erlang/Elixir standard libraries (`boot.avm`) come from AtomVM's release image
for the same target, so no host build of AtomVM is needed. The result is one image to flash
at 0x0 (the P4's bootloader sits at 0x2000 inside it).

| Partition | Offset | Size | |
|---|---|---|---|
| factory | 0x10000 | 4 MB | AtomVM + NIFs + sanoTTS (its dictionary and voice: ~1.9 MB) |
| boot.avm | 0x410000 | 512 KB | Erlang + Elixir libraries |
| model | 0x490000 | 6 MB | the int4 `.mmrt` model (`../models/`) |
| main.avm | 0xA90000 | 5.4 MB | your app: last, so it takes whatever flash is left |

The int8 model needs a 10 MB model partition: `../docs/MODELS.md` section 1.

**Text to speech and its voice.** Three things work together, and `build.sh` drives them:

- `../components/sanotts/prepare.sh` copies sanoTTS's C sources from your checkout
  (`SANOTTS_DIR`) into `$OUT/sanotts_src` and patches them. It refuses a checkout at any commit
  but the one whose licences we checked. With no checkout, the firmware builds without speech.
- The `sanotts` ESP-IDF component compiles those sources into the firmware.
- `SANOTTS_VOICE` picks the voice. `nano` (the default) is small enough to be compiled into the
  firmware. `heart` (ESP32-P4 only) sounds much better but is 2.4 MB: `prepare.sh` converts its
  weights into a separate file, `$OUT/voice.bin`, which you flash to a `voice` partition. The
  default partition table has none, so build with
  `PARTITIONS=atomvm/partitions-babytalk-voice.csv` (or your own table):

```bash
SANOTTS_VOICE=heart PARTITIONS=atomvm/partitions-babytalk-voice.csv atomvm/build.sh esp32p4_pre_c6
esptool.py --chip esp32p4 write_flash 0x0 ~/build/atomvm-out-esp32p4_pre_c6/atomvm-babytalk.img \
    0x490000 models/citrinet256_int4.mmrt 0xD00000 ~/build/atomvm-out-esp32p4_pre_c6/voice.bin
```

The firmware and `voice.bin` go together (the firmware is compiled for that voice's layer
sizes), so flash both after a rebuild. Voices compared, and how heart was made fast enough:
[../docs/TTS_VOICES.md](../docs/TTS_VOICES.md).

If boards share a serial port, flash with `../tools/flash.sh` instead of esptool: same
offset/file arguments, but it reads the chip's MAC first and refuses to write unless it
matches the `mac` in `board.conf`.

What the P4 build changes, and why: [../docs/BOARD_WAVESHARE_P4_WIFI6.md](../docs/BOARD_WAVESHARE_P4_WIFI6.md),
section 3.

Firmware options (`idf.py menuconfig`, or lines in `sdkconfig.babytalk*`):

| Option | Default | |
|---|---|---|
| `CONFIG_BABYTALK_BOARD_*` | S3-CAM on an S3, P4-WIFI6 on a P4 | the board assumed until `audio_config/1`; `CUSTOM` for your own C driver |
| `CONFIG_BABYTALK_RESERVE_AT_BOOT` | y | take the engines' 84.5 KB of internal RAM at boot (see `reserve_at_boot/1`) |
| `CONFIG_BABYTALK_STACKS_IN_PSRAM` | n | the worker and audio task stacks (16 KB) in PSRAM, for boards short of internal RAM |
| `CONFIG_SRAM_POOL_IN_PSRAM` | n (y on the P4) | the engines' pool in PSRAM (right on the P4, where only synthesis uses it) |
| `CONFIG_BABYTALK_YIELD` | n | speech work runs at the AtomVM schedulers' priority instead of above them, so the VM keeps running during a transcription (for a board with a screen) |

**Fitting next to WiFi, a camera or a screen.** Internal RAM is what runs out. What the
Watchtower boards use, roughly in order of payoff:

- `CONFIG_SRAM_POOL_IN_PSRAM=y` and `CONFIG_BABYTALK_STACKS_IN_PSRAM=y`: about 100 KB of
  internal RAM back, a little slower. On the S3-CAM with WiFi up: 97 KB free instead of 22.
- Fewer static WiFi buffers: the radio's DMA can't reach PSRAM, so WiFi stages frames in
  internal RAM. `CONFIG_ESP_WIFI_STATIC_TX_BUFFER_NUM=8` and `..._STATIC_RX_BUFFER_NUM=8`
  (from 16 and 10) gave the Pager 17 KB back, enough for the mic's DMA buffers.
- Software AES for TLS (`CONFIG_MBEDTLS_HARDWARE_AES` off): hardware AES wants an internal
  DMA buffer for every TLS record, and fails when RAM is tight.
- `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=256` (from 4096), so Erlang heaps and display widgets go
  to PSRAM; display frame buffers in PSRAM.
- `CONFIG_BABYTALK_RESERVE_AT_BOOT=n`, and let the app opt in with `reserve_at_boot/1` (the
  Pager has a setting for it).
- On a board with a screen, `CONFIG_BABYTALK_YIELD=y` and `babytalk_listener:hold/2` while
  the screen is in use: the CrowPanel's on-screen keyboard went from 0.5-1.7 s per key while
  the room was talking to ~90 ms.

`heap_info/0` shows where you stand: watch `internal_largest`, not just `internal_free`.

### AtomVM patches

`build.sh` applies these to the AtomVM checkout (all of them, for every target); an AtomVM
build of BabyTalk made any other way needs them too.

| Patch | Fixes |
|---|---|
| `0001-i2c-resource-guard-whole-file.patch` | with AtomVM's I2C disabled, its old I2C driver was still compiled in, and ESP-IDF aborts when the old and new I2C drivers are both linked |
| `0002-send-from-task-dead-pid-lock-leak.patch` | **a VM freeze.** A message from a native task (`globalcontext_send_message_from_task`) to a process that has just exited leaked the process table's read lock, and the next spawn or exit waited forever. BabyTalk's audio task hits this when a listener stops: its last `{babytalk_mic, Ref, stopped}` goes to the dead listener |
| `0003-gpio-deep-sleep-hold-without-pad-hold.patch` | the P4 build: AtomVM's GPIO deep-sleep hold NIFs call functions ESP-IDF only declares on chips with deep-sleep pad hold, which the P4 lacks (no change on the S3) |
| `0004-board-partition-table.patch` | since v0.7.0-beta.0 AtomVM's build writes its own partition table into `sdkconfig`; this lets a build bring its own (`build.sh` passes `-D AVM_PARTITION_TABLE_FILENAME=partitions-babytalk.csv`) |

## Example apps

| App | What |
|---|---|
| `apps/listen_demo` | the full conversation: wake phrase, message, "You said: ..." |
| `apps/voice_commands` | the same in Elixir, with a few commands ("count to three", "how much memory do you have") |
| `apps/wakeword_demo` | teach it your own wake phrase by voice ("Did you say ...?"), then talk to it |
| `apps/tts_demo` | speech out: pace, volume, PCM hashes |
| `apps/stt_server` + `tools/stt_client.py` | send WAVs over WiFi, get transcripts back (the board's address from `--host`, `$STT_HOST` or `board.conf`) |
| `apps/speed_bench` | transcription and synthesis timings |
| `apps/mmrt_test` | MMRT test vectors from `gen_vectors.py` + benchmarks |
| `apps/tcp_upload_repro` | a minimal repro of AtomVM's `gen_tcp` drops on long uploads |

Apps that use WiFi need `gen_wifi_creds.sh <app dir>`, which writes a git-ignored
`wifi_creds.erl`.

## Measured

**Waveshare ESP32-S3-CAM** (240 MHz, octal PSRAM):

- **Same transcripts as MicroPython**: 8 field clips x 4 runs, 32/32 identical to the
  MicroPython firmware on the same PCM; a 7.7 s clip takes 1.33 s (MicroPython: 1.41 s).
- **MMRT**: 256x256 int8 matvec 453 us (145 MMAC/s, limited by PSRAM bandwidth); matmul with
  16 rows 720 MMAC/s; int4 466 MMAC/s; the exact C fallback 22 MMAC/s. For scale: a dot
  product in plain Erlang over lists runs at 0.19 MMAC/s.
- **Text to speech**: RTF 0.27 (a 4.76 s sentence in 1.27 s; MicroPython 0.25-0.26) with
  sanoTTS's ~104 KB of static buffers moved to PSRAM (`../components/sanotts/linker.lf`); its
  84 KB arena shares the engine's internal SRAM pool with transcription. Playback: 0 DMA
  underruns.
- **Internal RAM**: 116 KB free at start (after the engines' 84.5 KB pool is reserved), 42 KB
  after WiFi joins, ~31 KB during inference.

**Waveshare ESP32-P4-WIFI6** (360 MHz, PSRAM at 200 MHz): MMRT 256x256 int8 matvec 246 us
(267 MMAC/s), matmul with 16 rows 1640 MMAC/s, int4 564 MMAC/s, 26/26 test vectors; 2 s of
speech transcribed in 0.30 s, 10 s in 1.35 s; speech synthesis 0.23x real time; 295 KB of
internal RAM free at boot, ~227 KB online. Details:
[../docs/BOARD_WAVESHARE_P4_WIFI6.md](../docs/BOARD_WAVESHARE_P4_WIFI6.md).

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

AtomVM v0.7 is still a pre-release (beta.0; the project recommends v0.6 for production). While a transcription
runs, MMRT keeps both cores busy (its helper task on core 0), so Erlang code and WiFi respond
more slowly for those ~0.2 s per second of audio (`CONFIG_BABYTALK_YIELD` trades speech speed
for a responsive VM). The mic and the speaker take turns (one
I2S port), so the board can't hear you while it talks. The end of a message is found by
loudness, so a noisy room may run to the 6 s cap. The built-in audio driver knows two codecs
(ES7210, ES8311); other chips need a C driver (`CONFIG_BABYTALK_BOARD_CUSTOM`).
