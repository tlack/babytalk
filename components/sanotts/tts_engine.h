// On-device text to speech with sanoTTS's en_us_e12nano voice (294k params, int8, 24 kHz):
// espeak-free lexicon G2P -> duration + acoustic -> TinyVocos (iSTFT) decoder.
// Sources come from the sanoTTS repo's ESPHome component at build time (mpy/build.sh,
// SANOTTS_DIR); runtime MIT, dictionary misaki us_gold (Apache-2.0).
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TTS_SAMPLE_RATE 24000

typedef struct {
    int phonemes, frames, samples;
    float g2p_ms, synth_ms;   // synth_ms / (samples / 24000) = RTF
    size_t arena_bytes, arena_peak;
    int64_t macs_simd, macs_scalar;
} tts_stats_t;

// The arena is the shared internal SRAM pool (components/sram_pool), which stt also uses:
// tts_reserve() reserves the pool now (call early, before internal RAM fragments; -2 if no
// block), tts_release() drops tts's claim (tts_say() already does after each utterance).
int tts_reserve(void);
void tts_release(void);

// text -> int16 PCM at 24 kHz in PSRAM (*pcm, free with heap_caps_free), peak-normalized
// to `volume` (0..1 of full scale).
// 0 on success; -1 G2P failed, -2 no internal block for the arena, -3 out of PSRAM,
// else the synthesizer's error code.
int tts_say(const char *text, float volume, int16_t **pcm, int *n, tts_stats_t *st);

// Speaking pace for the next tts_say calls: every phoneme's duration is multiplied by
// `scale` (1.0 = the voice's own pace, 1.1 = 10% slower; clamped to 0.5..2.0).
void tts_set_length_scale(float scale);

#ifdef __cplusplus
}
#endif
