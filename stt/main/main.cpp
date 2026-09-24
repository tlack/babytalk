// On-device Citrinet-256 speech-to-text test app (ESP-DL, weights read in place from
// the `model` flash partition).
//
// Audio goes in over WiFi: a TCP server on port 5555 (tools/stt.py). Each request is
// one text line, then a raw binary payload; replies are JSON lines ending in "DONE rc":
//   pcm <n_samples> [mode] [exact]  + n_samples*2 bytes s16le mono 16 kHz; exact=1 (default)
//                            rebuilds the graph for the clip length, 0 uses the 1600 window
//                            -> {"text":...} + timing
//   win <T> [mode]           + the 1600x80 int16 input window (bit-exact check vs the
//                            ESP-PPQ host sim) -> timing + "LOGITS <bytes>" + raw int16
//   load [internal_kb] [param_copy]
//   listen <secs> [mode] [trim] [send_audio]
//                            record from the board's mic now: streams "LEVEL <dBFS> <ms>"
//                            lines while recording, then trims silence (trim=1) and
//                            transcribes; send_audio=1 also returns "AUDIO <bytes>" + PCM
// mode: 0 = auto, 1 = single core, 2 = multi core (ESP-DL runtime_mode_t).
//
// The USB console stays for status: `ip`, `load`, `prof`.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dl_model_base.hpp"
#include "esp_console.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include "mic.h"
#include "stt_core.h"
#include "wifi_secrets.h"
#include <math.h>

extern "C" {
#include "citrinet_tables.h"
}

#define PORT 5555

extern "C" void dl_conv_sram_enable(bool on);  // patched ESP-DL (espdl/patches)

static dl::Model *g_model;
static int g_frames;  // input length the model is built for (frames)
static SemaphoreHandle_t g_lock;  // model is shared by the TCP task and the console
static char g_ip[16] = "0.0.0.0";

// ---------------------------------------------------------------- output sink
// Replies go to the console (fd < 0) or a TCP socket.
struct Out {
    int fd;
    void printf(const char *fmt, ...) __attribute__((format(printf, 2, 3)))
    {
        char buf[512];
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
static dl::TensorBase *io_tensor(bool input)
{
    auto &m = input ? g_model->get_inputs() : g_model->get_outputs();
    return m.begin()->second;
}

static void heap_line(Out &o, const char *when)
{
    o.printf("{\"when\":\"%s\",\"internal_free\":%u,\"psram_free\":%u,\"internal_largest\":%u}\n", when,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}

// frames > 0: build the graph for that input length (ESP-DL input_shapes override);
// 0: the static length stored in the model (1600).
static int load_model(Out &o, int internal_kb, bool param_copy, int frames = 0)
{
    delete g_model;
    g_model = nullptr;
    heap_line(o, "before_load");
    int64_t t0 = esp_timer_get_time();
    std::map<std::string, std::vector<int>> shapes;
    if (frames > 0) shapes["feats"] = {1, frames, MEL_N};
    g_model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, internal_kb * 1024,
                            dl::MEMORY_MANAGER_GREEDY, nullptr, param_copy, shapes);
    int64_t us = esp_timer_get_time() - t0;
    if (!g_model || g_model->get_inputs().empty()) {
        o.printf("{\"error\":\"model load failed\"}\n");
        return 1;
    }
    dl::TensorBase *in = io_tensor(true), *out = io_tensor(false);
    g_frames = in->shape[1];
    o.printf("{\"load_ms\":%.1f,\"internal_kb\":%d,\"param_copy\":%d,"
             "\"in_shape\":[%d,%d,%d],\"in_dtype\":\"%s\",\"in_exp\":%d,"
             "\"out_shape\":[%d,%d,%d],\"out_dtype\":\"%s\",\"out_exp\":%d}\n",
             us / 1000.0, internal_kb, param_copy, in->shape[0], in->shape[1], in->shape[2],
             dl::dtype_to_string(in->dtype), (int)in->exponent, out->shape[0], out->shape[1],
             out->shape[2], dl::dtype_to_string(out->dtype), (int)out->exponent);
    heap_line(o, "after_load");
    return 0;
}

static int ensure_model(Out &o) { return g_model ? 0 : load_model(o, 0, false); }

static dl::runtime_mode_t mode_of(int m)
{
    return m == 2 ? dl::RUNTIME_MODE_MULTI_CORE : (m == 0 ? dl::RUNTIME_MODE_AUTO : dl::RUNTIME_MODE_SINGLE_CORE);
}

// Model on the already-filled input; decode the first out_frames; report.
static int run_and_decode(Out &o, int T, int mode, double fe_ms, int n_samples, bool send_logits)
{
    dl::TensorBase *out = io_tensor(false);
    int64_t t0 = esp_timer_get_time();
    g_model->run(mode_of(mode));
    int64_t run_us = esp_timer_get_time() - t0;

    int frames = stt_out_frames(T);
    static char text[1024];
    t0 = esp_timer_get_time();
    const int16_t *logits = (const int16_t *)out->data;
    int16_t *wide = NULL;
    if (out->dtype == dl::DATA_TYPE_INT8) {  // int8 model: widen for the decoder
        size_t n = (size_t)frames * (VOCAB_N + 1);
        wide = (int16_t *)heap_caps_malloc(n * 2, MALLOC_CAP_SPIRAM);
        for (size_t i = 0; i < n; i++) wide[i] = ((const int8_t *)out->data)[i];
        logits = wide;
    }
    stt_ctc_greedy(logits, frames, VOCAB_N, VOCAB, text, sizeof(text));
    heap_caps_free(wide);
    int64_t dec_us = esp_timer_get_time() - t0;

    // FNV-1a over the decoded logits: bit-exactness check across kernel variants.
    uint32_t fnv = 2166136261u;
    const uint8_t *lp = (const uint8_t *)out->data;
    for (size_t i = 0; i < (size_t)frames * (VOCAB_N + 1) * (out->dtype == dl::DATA_TYPE_INT8 ? 1 : 2); i++)
        fnv = (fnv ^ lp[i]) * 16777619u;
    // text is lowercase letters, spaces and apostrophes: safe inside a JSON string.
    o.printf("{\"text\":\"%s\",\"logits_fnv\":\"%08lx\"}\n", text, (unsigned long)fnv);
    double audio_ms = n_samples / 16.0;
    double total_ms = fe_ms + run_us / 1000.0 + dec_us / 1000.0;
    o.printf("{\"T\":%d,\"out_frames\":%d,\"mode\":%d,\"audio_ms\":%.0f,\"fe_ms\":%.1f,\"model_ms\":%.1f,"
             "\"decode_ms\":%.2f,\"total_ms\":%.1f,\"rtf\":%.3f}\n",
             T, frames, mode, audio_ms, fe_ms, run_us / 1000.0, dec_us / 1000.0, total_ms,
             audio_ms > 0 ? total_ms / audio_ms : 0.0);
    if (send_logits) {
        size_t n = (size_t)out->shape[1] * out->shape[2] * 2;
        o.printf("LOGITS %u\n", (unsigned)n);
        o.write(out->data, n);
    }
    return 0;
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
    float *db = (float *)heap_caps_malloc(sizeof(float) * nf * 2, MALLOC_CAP_SPIRAM), *srt = db + nf;
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
    heap_caps_free(db);
    if (first < 0) return false;
    const int pad = 16000 / 4;
    *b = first * F - pad < 0 ? 0 : first * F - pad;
    *e = (last + 1) * F + pad > n ? n : (last + 1) * F + pad;
    return true;
}

// Features -> model -> CTC on pcm[0:n], rebuilding the graph if its length changed.
static int transcribe(Out &o, const int16_t *pcm, int n, int mode)
{
    int T = stt_num_frames(n);
    float *feats = (float *)heap_caps_malloc(sizeof(float) * T * MEL_N, MALLOC_CAP_SPIRAM);
    float *scratch = (float *)heap_caps_malloc(sizeof(float) * stt_scratch_floats(), MALLOC_CAP_INTERNAL);
    int rc = 1;
    if (feats && scratch && (g_frames == T || load_model(o, 0, false, T) == 0)) {
        dl::TensorBase *in = io_tensor(true);
        int64_t t0 = esp_timer_get_time();
        stt_features(pcm, n, feats, scratch);
        stt_fill_quant(feats, T, (int16_t *)in->data, T, (int)in->exponent);
        rc = run_and_decode(o, T, mode, (esp_timer_get_time() - t0) / 1000.0, n, false);
    }
    heap_caps_free(feats);
    heap_caps_free(scratch);
    return rc;
}

static int cmd_listen(Out &o, int secs, int mode, bool trim, bool send_audio)
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
    // Build the graph for the full recording now, so an untrimmed clip starts
    // inference the moment recording ends.
    if (g_frames != stt_num_frames(n) && load_model(o, 0, false, stt_num_frames(n))) return 1;
    int16_t *pcm = (int16_t *)heap_caps_malloc(n * 2, MALLOC_CAP_SPIRAM);
    if (!pcm || mic_start()) {
        heap_caps_free(pcm);
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
    // A shorter graph costs a ~0.5 s rebuild; only worth it if it drops > 0.3 s of audio.
    if (speech && trim && n - (e - b) < 16000 * 3 / 10) b = 0, e = n;
    o.printf("{\"speech_ms\":%d,\"recorded_ms\":%d,\"trimmed\":%d}\n", (e - b) / 16, n / 16, speech ? n - (e - b) > 0 : 1);
    if (!speech) {
        o.printf("{\"text\":\"\"}\n");
    } else {
        rc = transcribe(o, pcm + b, e - b, mode);
    }
    o.printf("{\"wait_ms\":%.0f}\n", (esp_timer_get_time() - t_end) / 1000.0);
    heap_caps_free(pcm);
    return rc;
}

static int handle(Out &o, char *line)
{
    char *argv[6] = {};
    int argc = 0;
    for (char *t = strtok(line, " "); t && argc < 6; t = strtok(NULL, " ")) argv[argc++] = t;
    if (!argc) return 1;
    int a1 = argc > 1 ? atoi(argv[1]) : 0;
    int a2 = argc > 2 ? atoi(argv[2]) : 1;

    if (!strcmp(argv[0], "load")) return load_model(o, a1, argc > 2 && a2);
    if (!strcmp(argv[0], "sram")) {  // sram 0|1: stage 1x1 conv filters in SRAM (patched ESP-DL)
        dl_conv_sram_enable(a1);
        o.printf("{\"sram\":%d}\n", a1);
        return 0;
    }
    if (!strcmp(argv[0], "listen"))
        return cmd_listen(o, a1, a2, argc > 3 ? atoi(argv[3]) : 1, argc > 4 && atoi(argv[4]));
    if (ensure_model(o)) return 1;
    int a3 = argc > 3 ? atoi(argv[3]) : 1;  // pcm: 1 = exact-length graph, 0 = static 1600 window

    if (!strcmp(argv[0], "pcm")) {
        int n = a1, T = stt_num_frames(n);
        if (n < MEL_HOP || T > STT_WIN_FRAMES) {
            o.printf("{\"error\":\"length: max %d samples\"}\n", (STT_WIN_FRAMES - 1) * MEL_HOP);
            return 1;
        }
        int16_t *pcm = (int16_t *)heap_caps_malloc(n * 2, MALLOC_CAP_SPIRAM);
        float *feats = (float *)heap_caps_malloc(sizeof(float) * T * MEL_N, MALLOC_CAP_SPIRAM);
        float *scratch = (float *)heap_caps_malloc(sizeof(float) * stt_scratch_floats(), MALLOC_CAP_INTERNAL);
        int rc = 1;
        int64_t t_rx = esp_timer_get_time();
        if (pcm && feats && scratch && recv_all(o.fd, pcm, n * 2) == 0) {
            double rx_ms = (esp_timer_get_time() - t_rx) / 1000.0;
            o.printf("{\"rx_ms\":%.1f,\"rx_bytes\":%d}\n", rx_ms, n * 2);
            int want = a3 ? T : STT_WIN_FRAMES;
            if (g_frames != want && load_model(o, 0, false, want)) {
                rc = 1;
                goto done;
            }
            dl::TensorBase *in = io_tensor(true);
            int64_t t0 = esp_timer_get_time();
            stt_features(pcm, n, feats, scratch);
            if (in->dtype == dl::DATA_TYPE_INT8) {  // int8 model: saturate to int8
                int16_t *tmp = (int16_t *)heap_caps_malloc((size_t)want * MEL_N * 2, MALLOC_CAP_SPIRAM);
                stt_fill_quant(feats, T, tmp, want, (int)in->exponent);
                for (size_t i = 0; i < (size_t)want * MEL_N; i++)
                    ((int8_t *)in->data)[i] = tmp[i] > 127 ? 127 : (tmp[i] < -128 ? -128 : tmp[i]);
                heap_caps_free(tmp);
            } else {
                stt_fill_quant(feats, T, (int16_t *)in->data, want, (int)in->exponent);
            }
            double fe_ms = (esp_timer_get_time() - t0) / 1000.0;
            rc = run_and_decode(o, T, a2, fe_ms, n, false);
        }
    done:
        heap_caps_free(pcm);
        heap_caps_free(feats);
        heap_caps_free(scratch);
        return rc;
    }
    if (!strcmp(argv[0], "win")) {
        int T = a1;
        if (T <= 0 || T > STT_WIN_FRAMES) return 1;
        if (g_frames != STT_WIN_FRAMES && load_model(o, 0, false, STT_WIN_FRAMES)) return 1;
        dl::TensorBase *in = io_tensor(true);
        if (recv_all(o.fd, in->data, (size_t)STT_WIN_FRAMES * MEL_N * 2)) return 1;
        return run_and_decode(o, T, a2, 0.0, 0, true);
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

static int cmd_load(int argc, char **argv)
{
    Out o = {-1};
    xSemaphoreTake(g_lock, portMAX_DELAY);
    int rc = load_model(o, argc > 1 ? atoi(argv[1]) : 0, argc > 2 && atoi(argv[2]));
    xSemaphoreGive(g_lock);
    printf("@@DONE load rc=%d\n", rc);
    return rc;
}

static int cmd_prof(int argc, char **argv)
{
    Out o = {-1};
    xSemaphoreTake(g_lock, portMAX_DELAY);
    int rc = ensure_model(o);
    if (rc == 0) {
        esp_log_level_set("*", ESP_LOG_INFO);  // ESP-DL prints its tables with ESP_LOGI
        g_model->profile(argc > 1 && atoi(argv[1]));
        esp_log_level_set("*", ESP_LOG_WARN);
    }
    xSemaphoreGive(g_lock);
    printf("@@DONE prof rc=%d\n", rc);
    return rc;
}

extern "C" void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_WARN);
    g_lock = xSemaphoreCreateMutex();
    wifi_start();
    // Inference runs in this task: big stack, pinned to core 1 (WiFi lives on core 0).
    xTaskCreatePinnedToCore(tcp_task, "stt_tcp", 16 * 1024, NULL, 5, NULL, 1);

    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_cfg = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_cfg.prompt = "stt>";
    repl_cfg.task_stack_size = 16 * 1024;
    esp_console_dev_usb_serial_jtag_config_t hw_cfg = ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&hw_cfg, &repl_cfg, &repl));
    esp_console_register_help_command();
    esp_console_cmd_t cmds[3] = {};
    cmds[0].command = "ip"; cmds[0].help = "print the TCP address"; cmds[0].func = cmd_ip;
    cmds[1].command = "load"; cmds[1].help = "load model [internal_kb] [param_copy]"; cmds[1].func = cmd_load;
    cmds[2].command = "prof"; cmds[2].help = "ESP-DL profile of the model [sort]"; cmds[2].func = cmd_prof;
    for (const auto &c : cmds) ESP_ERROR_CHECK(esp_console_cmd_register(&c));
    printf("@@BOOT micromodels_stt\n");
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
