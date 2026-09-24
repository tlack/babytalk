# Ideas parking lot

Ideas that aren't in PLAN.md yet. Each entry: the idea as stated, then notes.

## Binary connectivity weights stored as Bloom filters (2026-09-23)

**Idea:** instead of float (or int8) weights, the network is binary: a node is
connected to an input or it isn't. Optimize for *sparse* connectivity, and store
each node's connection set as an integer that is a **Bloom filter** over the
indices of its connected inputs.

**Notes / why it might fit this hardware:**
- Storage per node is one fixed-width filter instead of a row of weights. That
  could shrink SD/flash bytes per expert by a large factor, which matters most for
  expert swap time (PLAN §2, SD → PSRAM wall).
- A node's output is (roughly) a popcount / sum over the inputs whose index tests
  positive in its filter. Membership tests are hash + bit tests: cheap scalar
  integer ops, with no MACs at all.
- **False positives are extra random connections.** They're deterministic given the
  hash, so training can see them (train through the filter, not the ideal set) and
  the net learns around them. Filter size / hash count trades bytes for noise.
- Evaluation direction matters: testing every input against every node's filter
  is O(inputs × nodes) hashes. Precomputing input-index hashes once per layer, then
  AND-ing bit masks, could make it vectorizable with PIE (128-bit AND + popcount).
- Training is the hard part: connectivity is discrete. Candidates: straight-through
  estimators (as in binary NNs), learned-mask / lottery-ticket-style pruning of a
  dense net then binarizing, or evolutionary search per block.
- Could combine with the MoE: experts as Bloom-filter connectivity *deltas* over a
  resident base (fits the LoRA-style-expert idea).

**Prior art to read first:** binary/XNOR nets (BinaryConnect, XNOR-Net), HashedNets
(hash-based weight sharing), lottery-ticket / supermask papers (fixed random
weights + learned binary masks), Bloom-filter embeddings (e.g. Bloom embeddings
for sparse inputs).

**Cheap first test:** on desktop, take one trained layer, keep its top-k
connections per node, encode as Bloom filters at a few sizes, and measure accuracy
vs bytes compared with int8 and int4.

## On-device TTS (study, 2026-09-24)

- Citrinet can't be inverted (CTC is many-to-one: prosody, pitch, speaker are discarded).
  Reusable: mmrt kernels (conv1d/dw/1x1), CB4 int4, esp-dsp FFT (iSTFT), and the ASR as a
  round-trip intelligibility judge for TTS training.
- Measured on the PC (1 thread, ORT): Piper medium (VITS) 15.7M params, 71 ms CPU per s of
  audio = 13x Citrinet (5.5 ms). On the S3 that is RTF >= ~1.3 even at Citrinet's
  efficiency; TinyTTS (VITS-style) reports 19-32x slower than real time on ESP32.
  Kokoro-82M: ~41 MB even at int4, doesn't fit 16 MB flash.
- sanoTTS (arXiv 2608.21378, CC BY 4.0): eSpeak NG G2P -> duration (36k) -> acoustic
  (200k) -> iSTFT decoder (331k); 567k params, 680 KB int8, ESP32-S3 RTF 0.22, ~289 KB
  SRAM, ~45 MMAC per s of audio. Quality gap vs teacher (UTMOS 2.8 vs 4.4); 1.45M
  variant UTMOS 4.1.
- Opportunity: our 1x1 kernels sustain ~3 GMAC/s, so a student ~10-20x sanoTTS's compute
  budget could still run faster than real time. Distill from Piper/Kokoro (datagen/),
  int4 via int4_gptq.py-style GPTQ, judge with UTMOS + Citrinet round-trip WER.
- Budget with STT: flash app 2 MB + STT int4 6 MB + TTS 1-4 MB + eSpeak data ~1 MB fits
  16 MB; RAM/compute time-share (the board's audio is half-duplex anyway).
