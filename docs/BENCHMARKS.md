# ESP32-S3 benchmarks (Waveshare ESP32-S3-CAM)

Measured with the `bench/` app on an ESP32-S3 (rev v0.2, 240 MHz) with 16 MB QIO flash at
80 MHz, 8 MB octal PSRAM at 80 MHz, and a 64 KB data cache. These are properties of the
chip and memory, so they may help with other projects. Terms are explained in the
glossary below.

## Memory speed

| memory | read | write |
|---|---|---|
| Internal SRAM | 425 MB/s (simple C loop; vector loads go faster) | 760 MB/s |
| PSRAM, streaming | 88 MB/s | 49 MB/s |
| Flash, memory-mapped | 32 MB/s | |
| Flash, `esp_partition_read()` | 12.7 MB/s | |
| SD card | not measured | |

Flash and PSRAM share one bus, so their traffic takes turns: reading from both at once is
no faster than reading from one. A second CPU core does not add memory bandwidth.

**Free memory with bare ESP-IDF** (no WiFi): internal RAM 286 KB of 316 KB free (largest
block 270 KB), PSRAM 8.0 MB free.

## Arithmetic (8-bit multiply-adds, in GMAC/s)

| setup | GMAC/s |
|---|---|
| Theoretical vector peak, one core (16 per cycle at 240 MHz) | 3.84 |
| MMRT 1x1 kernel, one core, weights in SRAM or cached | ~3.2 |
| Generic dot-product kernel called per row (ESP-NN), one core, SRAM | 0.68 |
| Same, two cores, SRAM | 1.35 |
| Weights streamed from PSRAM, each used once | 0.085 |
| Weights in PSRAM, each reused for 64 frames, two cores | 1.18 |
| Weights read in place from flash, reused for 64 frames, two cores | 0.95 |

The lesson: on this chip, speed comes from **reusing each weight many times once it is
in fast memory**. Streaming weights once per use caps you at the memory speed (88 MB/s of
8-bit weights is 0.088 GMAC/s), no matter how fast the math units are.

**MMRT kernels:** the 1x1 convolution runs at 1.19 CPU cycles per 16-wide vector
multiply-add, with operands in SRAM or cached PSRAM. The 4-bit weight unpacking runs at
5.2 cycles per weight (portable C: 9.3).

## Glossary

- **MAC / GMAC/s**: a multiply-accumulate is one multiplication added to a running sum, the
  basic step of a neural network. GMAC/s is billions of them per second.
- **SRAM** (internal RAM): the ~512 KB of fast memory inside the chip, shared with
  ESP-IDF, WiFi and MicroPython.
- **PSRAM**: the external 8 MB RAM chip; bigger and slower than SRAM.
- **Memory-mapped flash**: flash the CPU reads like RAM, through a cache, without copying.
- **PIE / SIMD**: the ESP32-S3's vector instructions, which do 16 8-bit operations at once.
- **int8 / int4 / quantization**: storing a network's numbers as 8- or 4-bit integers
  instead of 32-bit floats: smaller and faster, slightly less accurate.
- **GPTQ**: a quantization method that corrects each rounding error using the remaining
  weights, which keeps accuracy at 4 bits.
- **CTC**: the output style of the speech model: one letter or word piece (or "nothing")
  per 80 ms of audio, merged into text.
- **Word error rate**: the fraction of words a transcript gets wrong (substituted, missing
  or extra).
- **LoRa / LoRA**: LoRa is a long-range, very low-bandwidth radio. LoRA (Low-Rank
  Adaptation) is a way to adapt a neural network by adding small trainable matrices.
