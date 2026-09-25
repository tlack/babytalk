// On-device speech to text for ESP-IDF apps (and the MicroPython `stt` module): 16 kHz
// PCM -> log-mel front end -> Citrinet-256 on mmrt -> greedy CTC text, plus optional
// wake-phrase scoring (CTC keyword spotting, any phrase) on the same model output.
//
// The model is read in place from the flash partition labelled "model" (an .mmrt
// image: int8 or 4-bit CB4 weights). Inference runs on both cores: the calling task
// (or the engine's background task, core 1) plus mmrt's worker on core 0.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STT_TEXT_MAX 1024

typedef struct {
    char text[STT_TEXT_MAX];
    float score;               // wake-phrase score (0 = the phrase is the best reading); -1e30 if no phrase
    int kw_start, kw_end;      // matched span, output frames (80 ms each)
    int frames;                // model input frames (10 ms)
    float fe_ms, model_ms, dec_ms;
} stt_result_t;

typedef struct {
    size_t model_bytes, cache_bytes;
    int version, ops, int4_ops;
    size_t internal_free, psram_free;
} stt_info_t;

// Map the model partition, open it, start the workers. 0 on success; negative: no
// partition / mmap failed (-1), not an mmrt image (-2), out of memory (-3).
int stt_engine_open(void);
void stt_engine_info(stt_info_t *info);

// Release everything open() took (weight cache, activations, internal staging buffers,
// phrase): e.g. before running another engine that needs internal RAM. Reopens on demand.
void stt_engine_close(void);

// Keep up to `bytes` of weights in PSRAM (faster than flash); 0 releases. Returns bytes cached.
size_t stt_engine_cache(size_t bytes);

// Activations up to max_bytes in internal SRAM while reserve_bytes stay free (0: all PSRAM).
void stt_engine_act_sram(size_t max_bytes, size_t reserve_bytes);

// Wake phrase: spellings (letters only matter; "tomato face" == "tomatoface"). n = 0 clears.
// Returns the number of piece sequences, or -1 (out of memory).
int stt_engine_set_phrase(const char *const *spellings, int n);

// Blocking: transcribe n samples of 16 kHz mono PCM. 0 on success.
int stt_engine_run(const int16_t *pcm, int n, stt_result_t *r);

// Same for interleaved PCM of `channels` channels (n frames); uses channel 0 (e.g. the
// left slot of a stereo I2S capture). Needs a mono copy in PSRAM when channels > 1.
int stt_engine_run_ch(const int16_t *pcm, int n, int channels, stt_result_t *r);

// Background: copy channel 0 of the PCM and transcribe it on the engine task. 0 started,
// -1 busy, -3 out of memory. Poll stt_engine_busy(), then stt_engine_take().
int stt_engine_start(const int16_t *pcm, int n, int channels);
int stt_engine_busy(void);
int stt_engine_take(stt_result_t *r);  // the run's return code; -2 if there is no result

#ifdef __cplusplus
}
#endif
