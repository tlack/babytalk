// On-device speech-to-text test app: Citrinet-256 int8 on the mmrt runtime (../../mmrt),
// weights read in place from the `model` flash partition (export/mmrt_export.py image).
//
// WiFi TCP server on port 5555 (tools/stt.py, tools/listen.py). Each request is one text
// line, then a raw binary payload; replies are JSON lines ending in "DONE <cmd> rc=<n>":
//   pcm <n_samples>          + n_samples*2 bytes s16le mono 16 kHz -> text + timing
//   feats <T> [logits]       + T*80 int8 model input (exponent from the model): the
//                            bit-exact check against the host run of the same C runtime;
//                            logits=1 also returns "LOGITS <bytes>" + int8 [T_out][272]
//   listen <secs> [mode] [trim] [send_audio]
//                            record from the board's mic now: streams "LEVEL <dBFS> <ms>"
//                            lines while recording, then trims silence (trim=1) and
//                            transcribes; send_audio=1 also returns "AUDIO <bytes>" + PCM
//   info                     model summary + memory
// Every inference reply includes per-op-kind timings ("ops_ms").
//
// The USB console stays for status: `ip`.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_console.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "lwip/sockets.h"
#include "mic.h"
#include "mmrt.h"
#include "mmrt_ref.h"
#include "mmrt_s3.h"
#include "nvs_flash.h"
#include "stt_core.h"
#include "wifi_secrets.h"

extern "C" {
#include "citrinet_tables.h"
}

#define PORT 5555

static mmrt_model_t g_model;
static bool g_loaded;
static SemaphoreHandle_t g_lock;  // model is shared by the TCP task and the console
static char g_ip[16] = "0.0.0.0";

// ---------------------------------------------------------------- output sink
// Replies go to the console (fd < 0) or a TCP socket.
struct Out {
    int fd;
    void printf(const char *fmt, ...) __attribute__((format(printf, 2, 3)))
    {
        char buf[768];
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (n > (int)sizeof(buf) - 1) n = sizeof(buf) - 1;
        write(buf, n);
    }
    void write(const void *p, size_t n)
    {
        if (fd < 0) {
            fwrite(p, 1, n, stdout);
            fflush(stdout);
            return;
        }
        const char *c = (const char *)p;
        while (n) {
            int k = send(fd, c, n, 0);
            if (k <= 0) return;
            c += k;
            n -= k;
        }
    }
};

// ---------------------------------------------------------------- model
static void *psram_alloc(size_t n) { return heap_caps_aligned_alloc(16, n ? n : 1, MALLOC_CAP_SPIRAM); }
static void psram_free(void *p) { heap_caps_free(p); }

static void heap_line(Out &o, const char *when)
{
    o.printf("{\"when\":\"%s\",\"internal_free\":%u,\"psram_free\":%u,\"internal_largest\":%u}\n", when,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}

static int load_model(Out &o)
{
    if (g_loaded) return 0;
    const esp_partition_t *p =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const void *img;
    esp_partition_mmap_handle_t h;
    int64_t t0 = esp_timer_get_time();
    if (!p || esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &img, &h) != ESP_OK) {
        o.printf("{\"error\":\"model partition mmap failed\"}\n");
        return 1;
    }
    int rc = mmrt_open(&g_model, img, p->size);
    if (rc) {
        o.printf("{\"error\":\"mmrt_open %d (flash the .mmrt image to the model partition)\"}\n", rc);
        return 1;
    }
    g_loaded = true;
    const mmrt_header_t *hd = g_model.hdr;
    o.printf("{\"load_ms\":%.1f,\"ops\":%u,\"tensors\":%u,\"blob_mb\":%.2f,\"in_ch\":%u,\"in_exp\":%d,"
             "\"out_ch\":%u,\"out_valid\":%u}\n",
             (esp_timer_get_time() - t0) / 1000.0, (unsigned)hd->n_ops, (unsigned)hd->n_tensors,
             hd->blob_size / 1e6, g_model.tensors[hd->input].channels, g_model.tensors[hd->input].exp,
             g_model.tensors[hd->output].channels, (unsigned)hd->out_valid);
    return 0;
}

// Per-op-kind wall time, via the runtime's trace hook (called after every op).
static int64_t g_kind_us[8], g_last_us;
static void on_op(int, const mmrt_op_t *op, const int8_t *, int, int)
{
    int64_t now = esp_timer_get_time();
    g_kind_us[op->kind & 7] += now - g_last_us;
    g_last_us = now;
}

// Run the model on int8 input [T][80]; decode; report text, FNV of the logits, timing.
static int infer(Out &o, const int8_t *x, int T, double fe_ms, int n_samples, bool send_logits)
{
    memset(g_kind_us, 0, sizeof(g_kind_us));
    mmrt_trace = on_op;
    int T_out = 0;
    int64_t t0 = esp_timer_get_time();
    g_last_us = t0;
    const int8_t *y = mmrt_run(&g_model, x, T, &T_out, psram_alloc, psram_free);
    int64_t run_us = esp_timer_get_time() - t0;
    mmrt_trace = NULL;
    if (!y) {
        o.printf("{\"error\":\"mmrt_run failed (out of memory?)\"}\n");
        return 1;
    }
    const int C = g_model.tensors[g_model.hdr->output].channels, V = (int)g_model.hdr->out_valid;

    // FNV-1a over the valid logits [T_out][V]: the bit-exactness check.
    uint32_t fnv = 2166136261u;
    for (int t = 0; t < T_out; t++)
        for (int v = 0; v < V; v++) fnv = (fnv ^ (uint8_t)y[t * C + v]) * 16777619u;

    t0 = esp_timer_get_time();
    int16_t *wide = (int16_t *)psram_alloc((size_t)T_out * V * 2);
    for (int t = 0; t < T_out; t++)
        for (int v = 0; v < V; v++) wide[t * V + v] = y[t * C + v];
    static char text[1024];
    stt_ctc_greedy(wide, T_out, VOCAB_N, VOCAB, text, sizeof(text));
    psram_free(wide);
    double dec_ms = (esp_timer_get_time() - t0) / 1000.0;

    o.printf("{\"text\":\"%s\",\"logits_fnv\":\"%08lx\"}\n", text, (unsigned long)fnv);
    static const char *const K[8] = {"?", "dwconv", "conv1x1", "mean", "lut", "mul", "add", "?"};
    char ops[256];
    int k = 0;
    for (int i = 1; i <= 6; i++)
        k += snprintf(ops + k, sizeof(ops) - k, "%s\"%s\":%.1f", i > 1 ? "," : "", K[i], g_kind_us[i] / 1000.0);
    double audio_ms = n_samples / 16.0, total_ms = fe_ms + run_us / 1000.0 + dec_ms;
    o.printf("{\"T\":%d,\"out_frames\":%d,\"audio_ms\":%.0f,\"fe_ms\":%.1f,\"model_ms\":%.1f,\"decode_ms\":%.2f,"
             "\"total_ms\":%.1f,\"rtf\":%.3f,\"ops_ms\":{%s}}\n",
             T, T_out, audio_ms, fe_ms, run_us / 1000.0, dec_ms, total_ms, audio_ms > 0 ? total_ms / audio_ms : 0.0,
             ops);
    if (send_logits) {
        o.printf("LOGITS %d\n", T_out * C);
        o.write(y, (size_t)T_out * C);
    }
    return 0;
}

// PCM -> features -> int8 model input -> infer.
static int transcribe(Out &o, const int16_t *pcm, int n)
{
    int T = stt_num_frames(n);
    if (n < MEL_HOP || T > STT_WIN_FRAMES) {
        o.printf("{\"error\":\"length: max %d samples\"}\n", (STT_WIN_FRAMES - 1) * MEL_HOP);
        return 1;
    }
    float *feats = (float *)psram_alloc(sizeof(float) * T * MEL_N);
    float *scratch = (float *)heap_caps_malloc(sizeof(float) * stt_scratch_floats(), MALLOC_CAP_INTERNAL);
    int16_t *q16 = (int16_t *)psram_alloc((size_t)T * MEL_N * 2);
    int8_t *q8 = (int8_t *)psram_alloc((size_t)T * MEL_N);
    int rc = 1;
    if (feats && scratch && q16 && q8) {
        int64_t t0 = esp_timer_get_time();
        stt_features(pcm, n, feats, scratch);
        stt_fill_quant(feats, T, q16, T, g_model.tensors[g_model.hdr->input].exp);
        for (int i = 0; i < T * MEL_N; i++) q8[i] = q16[i] > 127 ? 127 : (q16[i] < -128 ? -128 : q16[i]);
        rc = infer(o, q8, T, (esp_timer_get_time() - t0) / 1000.0, n, false);
    }
    psram_free(feats);
    heap_caps_free(scratch);
    psram_free(q16);
    psram_free(q8);
    return rc;
}

// ---------------------------------------------------------------- TCP server
static int recv_all(int fd, void *dst, size_t n)
{
    char *p = (char *)dst;
    while (n) {
        int k = recv(fd, p, n, 0);
        if (k <= 0) return -1;
        p += k;
        n -= k;
    }
    return 0;
}

static int recv_line(int fd, char *buf, size_t cap)
{
    size_t i = 0;
    while (i + 1 < cap) {
        char c;
        if (recv(fd, &c, 1, 0) != 1) return -1;
        if (c == '\n') break;
        if (c != '\r') buf[i++] = c;
    }
    buf[i] = 0;
    return (int)i;
}

// Speech bounds by frame energy: 20 ms frames, threshold = noise floor (10th
// percentile) + 12 dB, padded 250 ms each side. Returns false if nothing crosses it.
static bool speech_bounds(const int16_t *pcm, int n, int *b, int *e)
{
    const int F = 320, nf = n / F;
    if (nf < 3) return false;
    float *db = (float *)psram_alloc(sizeof(float) * nf * 2), *srt = db + nf;
    for (int f = 0; f < nf; f++) {
        double sq = 0;
        for (int i = 0; i < F; i++) sq += (double)pcm[f * F + i] * pcm[f * F + i];
        db[f] = srt[f] = 10.0f * log10f((float)(sq / F) + 1.0f) - 90.3f;  // dBFS
    }
    for (int i = 1; i < nf; i++)  // insertion sort: nf <= 800
        for (int j = i; j > 0 && srt[j - 1] > srt[j]; j--) { float t = srt[j]; srt[j] = srt[j - 1]; srt[j - 1] = t; }
    float thr = srt[nf / 10] + 12.0f;
    int first = -1, last = -1;
    for (int f = 0; f < nf; f++)
        if (db[f] > thr) { if (first < 0) first = f; last = f; }
    psram_free(db);
    if (first < 0) return false;
    const int pad = 16000 / 4;
    *b = first * F - pad < 0 ? 0 : first * F - pad;
    *e = (last + 1) * F + pad > n ? n : (last + 1) * F + pad;
    return true;
}

static int cmd_listen(Out &o, int secs, bool trim, bool send_audio)
{
    if (secs < 1 || secs > 15) {
        o.printf("{\"error\":\"secs must be 1..15\"}\n");
        return 1;
    }
    int err = mic_open(14);
    if (err) {
        o.printf("{\"error\":\"mic_open %d\"}\n", err);
        return 1;
    }
    const int n = secs * 16000;
    int16_t *pcm = (int16_t *)psram_alloc(n * 2);
    if (!pcm || mic_start()) {
        psram_free(pcm);
        return 1;
    }
    o.printf("REC start %d\n", secs);
    const int chunk = 1600;  // 100 ms
    for (int done = 0; done < n; done += chunk) {
        if (mic_read_mono(pcm + done, chunk)) break;
        double sq = 0;
        for (int i = 0; i < chunk; i++) sq += (double)pcm[done + i] * pcm[done + i];
        o.printf("LEVEL %.1f %d\n", 10.0 * log10(sq / chunk + 1.0) - 90.3, (done + chunk) / 16);
    }
    mic_stop();
    int64_t t_end = esp_timer_get_time();
    o.printf("REC done\n");
    if (send_audio) {
        o.printf("AUDIO %d\n", n * 2);
        o.write(pcm, n * 2);
    }
    int b = 0, e = n, rc = 0;
    bool speech = !trim || speech_bounds(pcm, n, &b, &e);
    o.printf("{\"speech_ms\":%d,\"recorded_ms\":%d}\n", speech ? (e - b) / 16 : 0, n / 16);
    if (!speech) {
        o.printf("{\"text\":\"\"}\n");
    } else {
        rc = transcribe(o, pcm + b, e - b);
    }
    o.printf("{\"wait_ms\":%.0f}\n", (esp_timer_get_time() - t_end) / 1000.0);
    psram_free(pcm);
    return rc;
}

// ---------------------------------------------------------------- kernel tests
static uint32_t s_rng = 12345;
static int8_t rnd8(void)
{
    s_rng = s_rng * 1664525u + 1013904223u;
    return (int8_t)(s_rng >> 24);
}

// Optimized kernel vs reference on random data, across the model's shapes.
static int ktest(Out &o)
{
    struct { int C, N, T, stride, shift, relu, bias; } cases[] = {
        {256, 256, 64, 1, 8, 1, 1}, {256, 256, 64, 1, 6, 0, 1}, {256, 256, 32, 2, 7, 0, 1},
        {80, 256, 64, 1, 9, 1, 1},  {256, 32, 1, 1, 7, 1, 0},  {32, 256, 1, 1, 5, 0, 0},
        {256, 640, 16, 1, 8, 1, 1}, {640, 272, 16, 1, 9, 0, 1}, {256, 256, 64, 1, 0, 0, 1},
        {256, 256, 200, 1, 8, 1, 1},
    };
    int fails = 0;
    for (auto &c : cases) {
        size_t xn = (size_t)c.T * c.stride * c.C, wn = (size_t)c.C * c.N, yn = (size_t)c.T * c.N;
        int8_t *x = (int8_t *)psram_alloc(xn), *w = (int8_t *)psram_alloc(wn);
        int8_t *y0 = (int8_t *)psram_alloc(yn), *y1 = (int8_t *)psram_alloc(yn);
        int32_t *b = (int32_t *)psram_alloc(c.N * 4);
        for (size_t i = 0; i < xn; i++) x[i] = rnd8() >> 1;  // keep sums well inside 20 bits
        for (size_t i = 0; i < wn; i++) w[i] = rnd8() >> 1;
        for (int i = 0; i < c.N; i++) b[i] = (int32_t)(s_rng = s_rng * 1664525u + 1013904223u) >> 18;
        int64_t t0 = esp_timer_get_time();
        mmrt_conv1x1_ref(x, c.T * c.stride, c.C, w, c.bias ? b : NULL, c.N, c.stride, c.shift, c.relu, y0, c.T);
        int64_t t_ref = esp_timer_get_time() - t0;
        t0 = esp_timer_get_time();
        mmrt_s3_conv1x1(x, c.T * c.stride, c.C, w, c.bias ? b : NULL, c.N, c.stride, c.shift, c.relu, y1, c.T);
        int64_t t_s3 = esp_timer_get_time() - t0;
        int bad = 0, first = -1;
        for (size_t i = 0; i < yn; i++)
            if (y0[i] != y1[i]) { if (first < 0) first = (int)i; bad++; }
        fails += bad > 0;
        extern int64_t mmrt_s3_part_us[2];
        o.printf("{\"parts_us\":[%lld,%lld]}\n", (long long)mmrt_s3_part_us[0], (long long)mmrt_s3_part_us[1]);
        double gmac = (double)c.T * c.C * c.N / (t_s3 > 0 ? t_s3 : 1) / 1000.0;
        o.printf("{\"kernel\":\"conv1x1\",\"C\":%d,\"N\":%d,\"T\":%d,\"stride\":%d,\"shift\":%d,\"relu\":%d,"
                 "\"bias\":%d,\"bad\":%d,\"first_bad\":%d,\"ref_us\":%lld,\"s3_us\":%lld,\"gmacs\":%.3f,"
                 "\"speedup\":%.1f%s}\n",
                 c.C, c.N, c.T, c.stride, c.shift, c.relu, c.bias, bad, first, (long long)t_ref, (long long)t_s3,
                 gmac, (double)t_ref / (t_s3 > 0 ? t_s3 : 1),
                 first >= 0 ? "" : "");
        if (first >= 0)
            o.printf("{\"at\":%d,\"ref\":%d,\"s3\":%d}\n", first, y0[first], y1[first]);
        psram_free(x); psram_free(w); psram_free(y0); psram_free(y1); psram_free(b);
    }
    struct { int C, K, T_in, stride, pad, shift, relu; } dcases[] = {
        {80, 5, 64, 1, 2, 5, 0},    {256, 11, 64, 1, 5, 6, 0},  {256, 11, 64, 2, 5, 7, 0},
        {256, 41, 64, 1, 20, 7, 0}, {256, 41, 8, 1, 20, 6, 0},  {256, 25, 200, 1, 12, 7, 1},
        {256, 13, 1, 1, 6, 5, 0},   {256, 11, 7, 2, 5, 0, 0},
    };
    for (auto &c : dcases) {
        int T_out = (c.T_in + 2 * c.pad - c.K) / c.stride + 1;
        size_t xn = (size_t)c.T_in * c.C, wn = (size_t)c.C * c.K, yn = (size_t)T_out * c.C;
        int8_t *x = (int8_t *)psram_alloc(xn), *w = (int8_t *)psram_alloc(wn);
        int8_t *y0 = (int8_t *)psram_alloc(yn), *y1 = (int8_t *)psram_alloc(yn);
        for (size_t i = 0; i < xn; i++) x[i] = rnd8();
        for (size_t i = 0; i < wn; i++) w[i] = rnd8();
        int64_t t0 = esp_timer_get_time();
        mmrt_dwconv_ref(x, c.T_in, c.C, w, c.K, c.stride, c.pad, c.shift, c.relu, y0, T_out);
        int64_t t_ref = esp_timer_get_time() - t0;
        t0 = esp_timer_get_time();
        mmrt_s3_dwconv(x, c.T_in, c.C, w, c.K, c.stride, c.pad, c.shift, c.relu, y1, T_out);
        int64_t t_s3 = esp_timer_get_time() - t0;
        int bad = 0, first = -1;
        for (size_t i = 0; i < yn; i++)
            if (y0[i] != y1[i]) { if (first < 0) first = (int)i; bad++; }
        fails += bad > 0;
        o.printf("{\"kernel\":\"dwconv\",\"C\":%d,\"K\":%d,\"T_in\":%d,\"stride\":%d,\"shift\":%d,\"relu\":%d,"
                 "\"bad\":%d,\"first_bad\":%d,\"ref_us\":%lld,\"s3_us\":%lld,\"gmacs\":%.3f,\"speedup\":%.1f}\n",
                 c.C, c.K, c.T_in, c.stride, c.shift, c.relu, bad, first, (long long)t_ref, (long long)t_s3,
                 (double)T_out * c.C * c.K / (t_s3 > 0 ? t_s3 : 1) / 1000.0, (double)t_ref / (t_s3 > 0 ? t_s3 : 1));
        if (first >= 0) o.printf("{\"at\":%d,\"ref\":%d,\"s3\":%d}\n", first, y0[first], y1[first]);
        psram_free(x); psram_free(w); psram_free(y0); psram_free(y1);
    }
    // mean over time (column sums in QACC)
    struct { int T, k; } mcases[] = {{1, 0}, {50, 1}, {800, 2}, {401, 0}, {7, 2}};
    for (auto &c : mcases) {
        const int C = 256;
        int8_t *x = (int8_t *)psram_alloc((size_t)c.T * C), y0[256], y1[256];
        for (int i = 0; i < c.T * C; i++) x[i] = rnd8();
        int64_t t0 = esp_timer_get_time();
        mmrt_mean_ref(x, c.T, C, c.k, 0, y0);
        int64_t t_ref = esp_timer_get_time() - t0;
        t0 = esp_timer_get_time();
        mmrt_s3_mean(x, c.T, C, c.k, 0, y1);
        int64_t t_s3 = esp_timer_get_time() - t0;
        int bad = 0;
        for (int i = 0; i < C; i++) bad += y0[i] != y1[i];
        fails += bad > 0;
        o.printf("{\"kernel\":\"mean\",\"T\":%d,\"k\":%d,\"bad\":%d,\"ref_us\":%lld,\"s3_us\":%lld}\n", c.T, c.k, bad,
                 (long long)t_ref, (long long)t_s3);
        psram_free(x);
    }
    // fused tail vs mul -> add -> relu reference chain
    struct { int T, shift, mul, add; } tcases[] = {{64, 7, 1, 1}, {64, 5, 1, 0}, {64, 0, 0, 1}, {800, 8, 1, 1}, {1, 6, 1, 1}};
    for (auto &c : tcases) {
        const int C = 256;
        size_t n = (size_t)c.T * C;
        int8_t *a = (int8_t *)psram_alloc(n), *r = (int8_t *)psram_alloc(n), *m = (int8_t *)psram_alloc(n);
        int8_t *y0 = (int8_t *)psram_alloc(n), *y1 = (int8_t *)psram_alloc(n);
        int8_t *sv = (int8_t *)psram_alloc(C);
        for (size_t i = 0; i < n; i++) { a[i] = rnd8(); r[i] = rnd8(); }
        for (int i = 0; i < C; i++) sv[i] = rnd8();
        int64_t t0 = esp_timer_get_time();
        if (c.mul) mmrt_mul_bcast_ref(a, c.T, C, sv, 0, 0, c.shift, m);
        else memcpy(m, a, n);
        if (c.add) mmrt_add_ref(m, r, (int)n, 0, 0, 0, y0);
        else memcpy(y0, m, n);
        for (size_t i = 0; i < n; i++) if (y0[i] < 0) y0[i] = 0;
        int64_t t_ref = esp_timer_get_time() - t0;
        t0 = esp_timer_get_time();
        mmrt_s3_tail(a, c.mul ? sv : NULL, c.add ? r : NULL, c.T, C, c.shift, 1, y1);
        int64_t t_s3 = esp_timer_get_time() - t0;
        int bad = 0, first = -1;
        for (size_t i = 0; i < n; i++)
            if (y0[i] != y1[i]) { if (first < 0) first = (int)i; bad++; }
        fails += bad > 0;
        o.printf("{\"kernel\":\"tail\",\"T\":%d,\"shift\":%d,\"mul\":%d,\"add\":%d,\"bad\":%d,\"ref_us\":%lld,\"s3_us\":%lld}\n",
                 c.T, c.shift, c.mul, c.add, bad, (long long)t_ref, (long long)t_s3);
        if (first >= 0) o.printf("{\"at\":%d,\"ref\":%d,\"s3\":%d}\n", first, y0[first], y1[first]);
        psram_free(a); psram_free(r); psram_free(m); psram_free(y0); psram_free(y1); psram_free(sv);
    }
    return fails;
}

static int handle(Out &o, char *line)
{
    char *argv[6] = {};
    int argc = 0;
    for (char *t = strtok(line, " "); t && argc < 6; t = strtok(NULL, " ")) argv[argc++] = t;
    if (!argc) return 1;
    int a1 = argc > 1 ? atoi(argv[1]) : 0;
    if (load_model(o)) return 1;

    if (!strcmp(argv[0], "ktest")) return ktest(o);
    if (!strcmp(argv[0], "ref")) {  // ref 1: portable C ops everywhere (A/B)
        mmrt_use_ref = a1;
        o.printf("{\"use_ref\":%d}\n", a1);
        return 0;
    }
    if (!strcmp(argv[0], "cores")) {  // cores 1|2: split conv kernels across both cores
        mmrt_s3_cores = a1;
        o.printf("{\"cores\":%d}\n", a1);
        return 0;
    }
    if (!strcmp(argv[0], "stage")) {  // stage 0|1: SRAM-stage 1x1 weights
        mmrt_s3_stage = a1;
        o.printf("{\"stage\":%d}\n", a1);
        return 0;
    }
    if (!strcmp(argv[0], "info")) {
        heap_line(o, "now");
        return 0;
    }
    if (!strcmp(argv[0], "listen"))  // argv[2] (mode) is accepted for compatibility
        return cmd_listen(o, a1, argc > 3 ? atoi(argv[3]) : 1, argc > 4 && atoi(argv[4]));
    if (!strcmp(argv[0], "pcm")) {
        int n = a1;
        int16_t *pcm = (int16_t *)psram_alloc((size_t)n * 2);
        int rc = 1;
        int64_t t_rx = esp_timer_get_time();
        if (pcm && recv_all(o.fd, pcm, (size_t)n * 2) == 0) {
            o.printf("{\"rx_ms\":%.1f,\"rx_bytes\":%d}\n", (esp_timer_get_time() - t_rx) / 1000.0, n * 2);
            rc = transcribe(o, pcm, n);
        }
        psram_free(pcm);
        return rc;
    }
    if (!strcmp(argv[0], "feats")) {
        int T = a1;
        if (T < 1 || T > 4 * STT_WIN_FRAMES) return 1;
        int8_t *x = (int8_t *)psram_alloc((size_t)T * MEL_N);
        int rc = 1;
        if (x && recv_all(o.fd, x, (size_t)T * MEL_N) == 0) rc = infer(o, x, T, 0.0, T * MEL_HOP, argc > 2 && atoi(argv[2]));
        psram_free(x);
        return rc;
    }
    o.printf("{\"error\":\"unknown command\"}\n");
    return 1;
}

static void tcp_task(void *)
{
    int ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    bind(ls, (sockaddr *)&addr, sizeof(addr));
    listen(ls, 1);
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        Out o = {fd};
        static char line[256];
        while (recv_line(fd, line, sizeof(line)) >= 0) {
            if (!line[0]) continue;
            char cmd[16] = {};
            strncpy(cmd, line, sizeof(cmd) - 1);  // command word, for the DONE line
            strtok(cmd, " ");
            xSemaphoreTake(g_lock, portMAX_DELAY);
            int rc = handle(o, line);
            xSemaphoreGive(g_lock);
            o.printf("DONE %s rc=%d\n", cmd, rc);
        }
        close(fd);
    }
}

// ---------------------------------------------------------------- WiFi
static void on_wifi(void *, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && (id == WIFI_EVENT_STA_START || id == WIFI_EVENT_STA_DISCONNECTED)) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto *e = (ip_event_got_ip_t *)data;
        snprintf(g_ip, sizeof(g_ip), IPSTR, IP2STR(&e->ip_info.ip));
        printf("@@IP %s:%d\n", g_ip, PORT);
    }
}

static void wifi_start(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_err_t e = esp_wifi_init(&cfg);
    if (e != ESP_OK) {
        printf("@@ERR esp_wifi_init: %s (internal free %u)\n", esp_err_to_name(e),
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        return;
    }
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_wifi, NULL);
    wifi_config_t wc = {};
    strncpy((char *)wc.sta.ssid, WIFI_SSID, sizeof(wc.sta.ssid));
    strncpy((char *)wc.sta.password, WIFI_PASS, sizeof(wc.sta.password));
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wc);
    esp_wifi_set_ps(WIFI_PS_NONE);  // latency over power for a bench app
    e = esp_wifi_start();
    if (e != ESP_OK) printf("@@ERR esp_wifi_start: %s\n", esp_err_to_name(e));
}

// ---------------------------------------------------------------- console
static int cmd_ip(int, char **)
{
    printf("@@IP %s:%d\n@@DONE ip rc=0\n", g_ip, PORT);
    return 0;
}

extern "C" void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_WARN);
    g_lock = xSemaphoreCreateMutex();
    wifi_start();
    // Inference runs in this task, pinned to core 1 (WiFi lives on core 0).
    xTaskCreatePinnedToCore(tcp_task, "stt_tcp", 16 * 1024, NULL, 5, NULL, 1);

    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_cfg = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_cfg.prompt = "stt>";
    repl_cfg.task_stack_size = 8 * 1024;
    esp_console_dev_usb_serial_jtag_config_t hw_cfg = ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&hw_cfg, &repl_cfg, &repl));
    esp_console_register_help_command();
    esp_console_cmd_t cmd = {};
    cmd.command = "ip";
    cmd.help = "print the TCP address";
    cmd.func = cmd_ip;
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd));
    printf("@@BOOT micromodels_stt (mmrt)\n");
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
