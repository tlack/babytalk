# Roadmap

What's next for BabyTalk, roughly in order of value, with an honest read of each: what exists
already, what it takes, and what could go wrong. Written 2026-09-25 from an outside review plus
our own notes; nothing here is scheduled.

## 1. Ship the fine-tuned speech model

**Why:** noise is BabyTalk's biggest weakness, and the fine-tuned model is the fix: on held-out
recordings with background noise, full-precision word error rate went from 46% to 24-25%
(root README, "Training for the real world"). Today it only exists on the PC.

**Status:** the recording, training and scoring tools work (`field/`, `train/`,
`export/field_eval.py`). Blocked on data: a few hundred more voice recordings in different
environments, so the model generalizes rather than learning one room.

**Then:** thread a `--ckpt` option through `export/export_onnx.py` and `export/mmrt_quant.py`
(they load NVIDIA's original weights today), build the int8 and int4 images, score them with
`field_eval.py --mmrt`, check them on the board, and publish them next to the originals in
`models/` (CC-BY-4.0 attribution carries over). No firmware change. See
[MODELS.md](MODELS.md#2-your-own-fine-tuned-variant).

## 2. Win back the int8/int4 accuracy loss (quantization-aware training)

**Why:** the board's models are worse than the full-precision one: on clean audiobooks 8.2%
(int4) and 6.3% (int8) against 3.8%. That gap applies to the fine-tuned model too.

**How:** fine-tune with the board's arithmetic simulated -- int8 activations with MMRT's
power-of-two scales and round-half-up, and for int4, the 16-level per-channel codebooks -- so
the weights learn to live with it. It's the standard remedy and usually recovers much of the
loss, but it is not a switch: the simulation has to match MMRT exactly (the bit-exact host
executor, `mmrt/` built for the PC, is the reference), and the codebooks need a
straight-through or re-clustering step during training.

**Depends on / pairs with:** 1 (do it on the fine-tuned model, not the original).

## 3. Live, partial transcripts ("streaming")

**Why:** today a message is transcribed after it ends; showing words as they come makes a
conversation feel responsive.

**The catch:** Citrinet isn't naturally streamable. Its squeeze-and-excitation layers average
over the whole utterance (MMRT's `MEAN` op), so transcribing chunks independently changes the
result and costs accuracy. NVIDIA's streaming models are a different architecture.

**Reachable version:** re-transcribe a growing or sliding window while you talk and show the
partial text, as the wake-phrase scorer already does every second; endpointing (the silence
detection in `babytalk_listener`) and gapless mic streaming (AtomVM) exist. Cost: both cores
are busy during each window's inference, so windows can't be too frequent. Call it "live
partial transcripts", not streaming.

## 4. A noise-reduction model in front of the recognizer

**Why:** noise again; and it would show MMRT hosting a second model.

**Caveats, in order of importance:**
- Enhancement in front of a recognizer often *doesn't* lower word error rate: its artifacts
  can hurt more than the noise, unless the recognizer is trained on enhanced audio. Measure
  after item 1 before building this.
- MMRT's op set is narrow (1-D depthwise and 1x1 convolutions, mean, lookup table, multiply,
  add). RNNoise-style models need recurrent layers and an FFT: new kernels. A convolution-only
  enhancer fits today.
- The board wires two of the ES7210's four inputs to microphones; two-mic beamforming gains a
  few dB at most -- cheap to try, small effect.

## 5. MMRT as a standalone component

**Why:** it's a good runtime (weights read in place from flash, both cores, fused layers,
hand-written ESP32-S3 SIMD, bit-exact tests) and useful beyond speech.

**Easy part:** package `components/mmrt` for the ESP-IDF component registry; the AtomVM
`mmrt` library (int8/int4 matvec, matmul, dot, top-k from Erlang) is already a standalone piece.

**Hard part:** a general ONNX-to-MMRT converter. MMRT supports Citrinet's ops only, and tiny
vision models need 3x3 (2-D) convolutions, pooling and more; its quantization is ESP-PPQ's
power-of-two scheme. Each new op family is kernel work plus bit-exact tests. Worth doing one
op family at a time, driven by a real model someone wants to run.

## 6. The LoRa bridge: speak a message, send it over the mesh

**Why:** the best demo story -- "say a message, it goes out as text over LoRa, no internet".

**What it takes:** a board with a radio (the Waveshare S3-CAM has none; the LilyGO T-Pager
and T-Watch S3 Plus have SX1262-class LoRa). The practical design is BabyTalk talking to a
Meshtastic node through Meshtastic's client API (serial, BLE or its TCP protobuf interface),
not merging the two firmwares. On AtomVM, the listener's `{command, Text}` event is the hook:
a small `gen_server` turns it into an outgoing message and speaks incoming ones.

## 7. Prebuilt firmware and a browser flasher

**Why:** the biggest adoption lever for people who won't set up ESP-IDF.

**What it takes:** release images (firmware + partition table + model) and an esp-web-tools
page. Licensing is now worked out: images with speech to text only are MIT/Apache/CC-BY
(attribution); images with text to speech should ship as GPL-3.0 with source until sanoTTS
clarifies three files ([components/sanotts/LICENSES.md](../components/sanotts/LICENSES.md)).
Limit to state up front: images serve one board (Waveshare ESP32-S3-CAM) until more boards
have audio drivers.

## 8. Other languages, and more boards

**Languages:** not "the same pipeline with a different checkpoint" -- NeMo's non-English
Citrinets are the 512/1024 sizes (~36M+ parameters, ~18 MB at 4 bits: too big for the flash).
The route is training a Citrinet-256 per language, plus firmware changes for a new tokenizer.
Speech output in another language needs a microcontroller build of a sanoTTS voice and a
pronunciation step. Full breakdown: [MODELS.md](MODELS.md#4-another-language).

**Boards:** the T-Watch S3 Plus is the next target in our notes. Everything above the audio
driver is board-independent; the watch has different audio hardware from the S3-CAM's
ES7210/ES8311 pair, so it needs its own `board_audio` driver (and pin map), plus a memory check.

## 9. Publish the field dataset and the hard-words workflow

**Why:** "record on the board, fine-tune on the laptop" is a workflow others will want to copy.

**Before publishing the recordings, decide on:**
- Your voice: it's your recording of your own voice -- your call, and worth a moment's thought
  given voice cloning.
- The sentence sources carry licences: Wikipedia text is CC BY-SA (share-alike would apply to
  the dataset), news text may be copyrighted, and each synthetic voice's training data has its
  own terms. That needs an audit, or a subset built only from permissive sources.

The tools themselves (`field/`, `export/hard_words.py`, `train/`) can be published anytime.

## Smaller items

- **Remember the enrolled wake phrase** across reboots (`wakeword_demo`): store it in NVS.
- **Trim silence before transcribing** in the listener (MicroPython's echo demo does; the
  model otherwise "hears" words in room noise). `babytalk:rms/1` makes it cheap.
- **Tune end-of-message detection** from real `message_end` level logs in noisy rooms.
- **Ask sanoTTS upstream** to add licence headers to its ESPHome component files, then re-audit
  and move the pinned commit.
- **AtomVM `gen_tcp` drops** ~6% of long uploads mid-transfer (`{error, closed}`): find out
  why, or report it upstream.
- **AtomVM's own I2C** is disabled in this firmware (it uses ESP-IDF's legacy driver, which
  can't coexist with the new one); revisit when AtomVM moves to the new driver.
- **A better voice**: sanoTTS's larger voices sound more natural but need more memory; test
  what fits next to the speech model and WiFi.
- **Bit-exact TTS check** against the MicroPython build (the AtomVM build logs a SHA-256 of
  each utterance for this).
