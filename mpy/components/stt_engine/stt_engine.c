#include "stt_engine.h"

#include <string.h>

#include "citrinet_tables.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "kws.h"
#include "mmrt.h"
#include "mmrt_s3.h"
#include "stt_core.h"

static mmrt_model_t s_model;
static int s_open;
static size_t s_model_bytes;
static kws_phrase_t *s_phrase;  // PSRAM (~41KB)
static size_t s_act_max, s_act_reserve = 40 * 1024;

static void *psram_alloc(size_t n) { return heap_caps_aligned_alloc(16, n ? n : 1, MALLOC_CAP_SPIRAM); }
static void mem_free(void *p) { heap_caps_free(p); }

static void *act_alloc(size_t n)
{
    if (n && n <= s_act_max && heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >= n + s_act_reserve) {
        void *p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL);
        if (p) return p;
    }
    return psram_alloc(n);
}

int stt_engine_open(void)
{
    if (s_open) return 0;
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const void *img;
    esp_partition_mmap_handle_t h;
    if (!p || esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &img, &h) != ESP_OK) return -1;
    if (mmrt_open(&s_model, img, p->size)) return -2;
    const mmrt_header_t *hd = s_model.hdr;
    s_model_bytes = hd->blob_off + hd->blob_size;
    if (mmrt_s3_init()) return -3;
    kws_init(VOCAB, VOCAB_N);
    s_open = 1;
    return 0;
}

void stt_engine_info(stt_info_t *info)
{
    memset(info, 0, sizeof(*info));
    if (s_open) {
        info->model_bytes = s_model_bytes;
        info->cache_bytes = s_model.wcache_bytes;
        info->version = (int)s_model.hdr->version;
        info->ops = (int)s_model.hdr->n_ops;
        for (uint32_t i = 0; i < s_model.hdr->n_ops; i++) info->int4_ops += s_model.ops[i].wfmt == MMRT_W_CB4;
    }
    info->internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    info->psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

size_t stt_engine_cache(size_t bytes)
{
    return s_open ? mmrt_cache_weights(&s_model, bytes, psram_alloc, mem_free) : 0;
}

void stt_engine_act_sram(size_t max_bytes, size_t reserve_bytes)
{
    s_act_max = max_bytes;
    s_act_reserve = reserve_bytes;
}

int stt_engine_set_phrase(const char *const *spellings, int n)
{
    if (n <= 0) {
        mem_free(s_phrase);
        s_phrase = NULL;
        return 0;
    }
    if (!s_phrase && !(s_phrase = (kws_phrase_t *)psram_alloc(sizeof(kws_phrase_t)))) return -1;
    s_phrase->n_seqs = 0;
    for (int i = 0; i < n; i++) kws_add_spelling(s_phrase, spellings[i], 64);
    return s_phrase->n_seqs;
}

int stt_engine_run(const int16_t *pcm, int n, stt_result_t *r)
{
    memset(r, 0, sizeof(*r));
    r->score = -1e30f;
    if (!s_open) return -1;
    const int T = stt_num_frames(n);
    if (n < MEL_HOP || T > 4 * STT_WIN_FRAMES) return -4;
    r->frames = T;
    int64_t t0 = esp_timer_get_time();
    float *feats = (float *)psram_alloc(sizeof(float) * T * MEL_N);
    float *scratch = (float *)heap_caps_malloc(sizeof(float) * stt_scratch_floats(), MALLOC_CAP_INTERNAL);
    int16_t *q16 = (int16_t *)psram_alloc((size_t)T * MEL_N * 2);
    int8_t *q8 = (int8_t *)psram_alloc((size_t)T * MEL_N);
    int rc = -3;
    if (feats && scratch && q16 && q8) {
        stt_features(pcm, n, feats, scratch);
        stt_fill_quant(feats, T, q16, T, s_model.tensors[s_model.hdr->input].exp);
        for (int i = 0; i < T * MEL_N; i++) q8[i] = q16[i] > 127 ? 127 : (q16[i] < -128 ? -128 : q16[i]);
        int64_t t1 = esp_timer_get_time();
        r->fe_ms = (t1 - t0) / 1000.0f;
        int T_out = 0;
        const int8_t *y = mmrt_run(&s_model, q8, T, &T_out, act_alloc, mem_free);
        int64_t t2 = esp_timer_get_time();
        r->model_ms = (t2 - t1) / 1000.0f;
        if (y) {
            const int C = s_model.tensors[s_model.hdr->output].channels, V = (int)s_model.hdr->out_valid;
            int16_t *wide = (int16_t *)psram_alloc((size_t)T_out * V * 2);
            if (wide) {
                for (int t = 0; t < T_out; t++)
                    for (int v = 0; v < V; v++) wide[t * V + v] = y[t * C + v];
                stt_ctc_greedy(wide, T_out, VOCAB_N, VOCAB, r->text, sizeof(r->text));
                mem_free(wide);
                if (s_phrase && s_phrase->n_seqs)
                    r->score = kws_score(s_phrase, y, T_out, C, s_model.tensors[s_model.hdr->output].exp,
                                         &r->kw_start, &r->kw_end);
                rc = 0;
            }
        }
        r->dec_ms = (esp_timer_get_time() - t2) / 1000.0f;
    }
    mem_free(feats);
    heap_caps_free(scratch);
    mem_free(q16);
    mem_free(q8);
    return rc;
}

static int16_t *mono_copy(const int16_t *pcm, int n, int channels)
{
    int16_t *m = (int16_t *)psram_alloc((size_t)n * 2);
    if (!m) return NULL;
    if (channels <= 1) {
        memcpy(m, pcm, (size_t)n * 2);
    } else {
        for (int i = 0; i < n; i++) m[i] = pcm[(size_t)i * channels];
    }
    return m;
}

int stt_engine_run_ch(const int16_t *pcm, int n, int channels, stt_result_t *r)
{
    if (channels <= 1) return stt_engine_run(pcm, n, r);
    int16_t *m = mono_copy(pcm, n, channels);
    if (!m) return -3;
    int rc = stt_engine_run(m, n, r);
    mem_free(m);
    return rc;
}

// ---------------------------------------------------------------- background task
// Same priority as MicroPython's task and on its core (1): while Python runs, the two
// time-slice; while Python sleeps (asyncio), inference has the core.
static TaskHandle_t s_task;
static SemaphoreHandle_t s_go;
static int16_t *s_pcm;
static int s_n;
static volatile int s_busy, s_has_result, s_rc;
static stt_result_t s_res;

static void engine_task(void *arg)
{
    (void)arg;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        s_rc = stt_engine_run(s_pcm, s_n, &s_res);
        mem_free(s_pcm);
        s_pcm = NULL;
        s_has_result = 1;
        s_busy = 0;
    }
}

int stt_engine_start(const int16_t *pcm, int n, int channels)
{
    if (s_busy) return -1;
    if (!s_task) {
        s_go = xSemaphoreCreateBinary();
        if (!s_go || xTaskCreatePinnedToCore(engine_task, "stt", 12 * 1024, NULL, 1, &s_task, 1) != pdPASS) return -3;
    }
    if (!(s_pcm = mono_copy(pcm, n, channels))) return -3;
    s_n = n;
    s_has_result = 0;
    s_busy = 1;
    xSemaphoreGive(s_go);
    return 0;
}

int stt_engine_busy(void) { return s_busy; }

int stt_engine_take(stt_result_t *r)
{
    if (s_busy || !s_has_result) return -2;
    *r = s_res;
    s_has_result = 0;
    return s_rc;
}
