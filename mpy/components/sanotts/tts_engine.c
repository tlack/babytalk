#include "tts_engine.h"

#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "nano_lex_g2p.h"
#include "sanotts_model_data.h"
#include "snt_arch.h"
#include "snt_nano.h"
#include "snt_port.h"
#include "sram_pool.h"

#define POOL_OWNER_TTS 2

#define MAX_IDS 1024
#define ARENA_BYTES (84208 + 256)  // the voice's arena peak: constant in utterance length (sanoTTS)
#define SEED 2236265385529901705ULL  // the component's fixed noise seed

#if SANOTTS_S3_SIMD
extern int64_t g_snt_macs_simd, g_snt_macs_scalar;
void snt_res_reset(void);
#endif

typedef struct {
    int16_t *buf;
    int n, cap;
    float gain;
    int oom;
} sink_t;

static int on_pcm(const float *pcm, int n, void *user)
{
    sink_t *s = (sink_t *)user;
    if (s->n + n > s->cap) {
        int cap = s->cap ? s->cap * 2 : 48000;
        while (cap < s->n + n) cap *= 2;
        int16_t *b = (int16_t *)heap_caps_realloc(s->buf, (size_t)cap * 2, MALLOC_CAP_SPIRAM);
        if (!b) {
            s->oom = 1;
            return 1;  // abort synthesis
        }
        s->buf = b;
        s->cap = cap;
    }
    for (int i = 0; i < n; i++) {
        float v = pcm[i] * s->gain;
        s->buf[s->n++] = v > 32767.f ? 32767 : (v < -32768.f ? -32768 : (int16_t)v);
    }
    return 0;
}

_Static_assert(ARENA_BYTES <= SRAM_POOL_BYTES, "arena must fit the shared SRAM pool");

// The arena is the shared internal SRAM pool (sram_pool.h), held only during tts_say().
int tts_reserve(void) { return sram_pool_reserve() ? -2 : 0; }

void tts_release(void) { sram_pool_release(POOL_OWNER_TTS); }

int tts_say(const char *text, float volume, int16_t **pcm, int *n, tts_stats_t *st)
{
    static int32_t ids[MAX_IDS];
    memset(st, 0, sizeof(*st));
    *pcm = NULL;
    *n = 0;
    int64_t t0 = esp_timer_get_time();
    int n_ids = nano_lex_g2p_text_to_ids(text, ids, MAX_IDS);
    st->g2p_ms = (esp_timer_get_time() - t0) / 1000.0f;
    if (n_ids <= 0) return -1;
    st->phonemes = n_ids;
    // The arena must be internal SRAM (the PIE kernels only read staged operands) and one
    // contiguous block: the shared pool, taken from stt if it holds it (stt re-acquires it
    // on its next run).
    void *arena = sram_pool_acquire(POOL_OWNER_TTS, NULL);
    if (!arena) return -2;
    st->arena_bytes = ARENA_BYTES;
    sink_t s = {NULL, 0, 0, 16384.f, 0};  // headroom; normalized to `volume` at the end
    snt_nano_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.front_blob = SANOTTS_FRONT_BLOB;
    cfg.dec_blob = SANOTTS_DEC_BLOB;
    cfg.arena = arena;
    cfg.arena_size = ARENA_BYTES;
    cfg.noise_seed = SEED;
#if SANOTTS_S3_SIMD
    snt_res_reset();
#endif
    snt_nano_stats ns;
    memset(&ns, 0, sizeof(ns));
    int64_t t1 = esp_timer_get_time();
    int rc = snt_nano_synthesize(&cfg, ids, n_ids, on_pcm, &s, &ns);
    sram_pool_release(POOL_OWNER_TTS);
    st->synth_ms = (esp_timer_get_time() - t1) / 1000.0f;
    st->frames = ns.frames;
    st->samples = s.n;
    st->arena_peak = ns.arena_peak;
#if SANOTTS_S3_SIMD
    st->macs_simd = g_snt_macs_simd;
    st->macs_scalar = g_snt_macs_scalar;
#endif
    if (s.oom) rc = -3;
    if (rc) {
        heap_caps_free(s.buf);
        return rc;
    }
    // peak-normalize: `volume` is the output peak as a fraction of full scale
    int peak = 1;
    for (int i = 0; i < s.n; i++) {
        int v = s.buf[i] < 0 ? -s.buf[i] : s.buf[i];
        if (v > peak) peak = v;
    }
    const float k = 32767.f * volume / (float)peak;
    for (int i = 0; i < s.n; i++) {
        float v = s.buf[i] * k;
        s.buf[i] = v > 32767.f ? 32767 : (v < -32768.f ? -32768 : (int16_t)v);
    }
    *pcm = s.buf;
    *n = s.n;
    return 0;
}
