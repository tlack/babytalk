// Record from the Waveshare S3 cam's ES7210 dual-mic ADC and dump it to the host.
//
//   rec [secs] [gain]      defaults 5 s, PGA gain 14 (0..14, ~3dB/step, 14 = max)
//
// 16 kHz, 16-bit, stereo (mic 1 = left, mic 2 = right) -- the rate Citrinet wants.
// Sequence and register values are the same as our MicroPython driver
// (mpy/drivers/es7210.py): expander P6 raises the audio rail (PA on P4 stays off), then the ES7210
// is configured over I2C, then I2S RX with MCLK = 256*Fs. STEREO only: MONO is
// known to garble this codec pair.
//
// Output protocol (for tools/rec.py):
//   @@REC start ...          capture has begun (host may start playback now)
//   @@R {...}                per-channel stats
//   @@A <base64>             raw interleaved s16le PCM, 3072 bytes per line
// Base64 because the console's newline translation makes raw binary unsafe.
//
// ES7210 register sequence derived from Espressif's es7210 driver (github.com/espressif/
// esp-bsp, components/es7210), Copyright Espressif Systems (Shanghai) CO LTD, Apache-2.0
// (licenses/Apache-2.0.txt); rewritten in C for one fixed configuration.
#include <math.h>
#include <string.h>
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "bench.h"

#define PIN_SCL   7
#define PIN_SDA   8
#define PIN_MCLK  10
#define PIN_BCLK  11
#define PIN_WS    12
#define PIN_DIN   13

#define EXPANDER_ADDR 0x24
#define EXP_REG_MODE  0x02
#define EXP_REG_OUT   0x03
#define EXP_RAIL_PIN  6

#define ES7210_ADDR   0x40
#define RATE          16000
#define WARMUP_MS     200  // ADC/HPF settling, discarded

static int write_reg(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t val)
{
    uint8_t b[2] = {reg, val};
    return i2c_master_transmit(dev, b, 2, 100) == ESP_OK ? 0 : -1;
}

static int es7210_init(i2c_master_dev_handle_t d, int gain)
{
    if (gain < 0) gain = 0;
    if (gain > 14) gain = 14;
    static const uint8_t seq1[][2] = {
        {0x00, 0xFF}, {0x00, 0x32},              // soft reset
        {0x09, 0x30}, {0x0A, 0x30},              // power-up timing
        {0x23, 0x2A}, {0x22, 0x0A},              // ADC12 HPF
        {0x21, 0x2A}, {0x20, 0x0A},              // ADC34 HPF
        {0x11, 0x60}, {0x12, 0x00},              // I2S, 16-bit, no TDM
        {0x40, 0xC3},                            // analog power + VMID
        {0x41, 0x70}, {0x42, 0x70},              // mic bias 2.87V
    };
    static const uint8_t seq2[][2] = {
        {0x47, 0x08}, {0x48, 0x08}, {0x49, 0x08}, {0x4A, 0x08},   // mic front-ends on
        {0x07, 0x20},                            // OSR
        {0x02, 0x01 | (1 << 6) | (1 << 7)},      // adc_div=1, doubler, dll
        {0x04, 0x01}, {0x05, 0x00},              // LRCK div = 256 (MCLK/Fs)
        {0x06, 0x04},                            // DLL down, rest up
        {0x4B, 0x0F}, {0x4C, 0x0F},
        {0x00, 0x71}, {0x00, 0x41},              // enable
    };
    for (size_t i = 0; i < sizeof(seq1) / 2; i++) {
        if (write_reg(d, seq1[i][0], seq1[i][1])) return -1;
    }
    for (uint8_t r = 0x43; r <= 0x46; r++) {     // PGA gain, all four channels
        if (write_reg(d, r, (uint8_t)gain | 0x10)) return -1;
    }
    for (size_t i = 0; i < sizeof(seq2) / 2; i++) {
        if (write_reg(d, seq2[i][0], seq2[i][1])) return -1;
    }
    return 0;
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void b64_line(const uint8_t *p, size_t n, char *out)
{
    char *o = out;
    size_t i = 0;
    for (; i + 2 < n; i += 3) {
        uint32_t v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
        *o++ = B64[v >> 18]; *o++ = B64[(v >> 12) & 63];
        *o++ = B64[(v >> 6) & 63]; *o++ = B64[v & 63];
    }
    if (i < n) {
        uint32_t v = p[i] << 16 | (i + 1 < n ? p[i + 1] << 8 : 0);
        *o++ = B64[v >> 18]; *o++ = B64[(v >> 12) & 63];
        *o++ = i + 1 < n ? B64[(v >> 6) & 63] : '=';
        *o++ = '=';
    }
    *o = 0;
}

int bench_rec(int argc, char **argv)
{
    int secs = bench_arg_int(argc, argv, 1, 5);
    int gain = bench_arg_int(argc, argv, 2, 14);
    if (secs < 1 || secs > 60) return 1;

    size_t bytes = (size_t)secs * RATE * 2 * sizeof(int16_t);
    int16_t *pcm = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
    if (!pcm) return 1;

    int rc = 1;
    i2c_master_bus_handle_t bus = NULL;
    i2c_master_dev_handle_t exp = NULL, adc = NULL;
    i2s_chan_handle_t rx = NULL;

    const i2c_master_bus_config_t bus_cfg = {
        .i2c_port = 0, .sda_io_num = PIN_SDA, .scl_io_num = PIN_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    if (i2c_new_master_bus(&bus_cfg, &bus) != ESP_OK) goto out;
    i2c_device_config_t dev_cfg = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .scl_speed_hz = 100000};
    dev_cfg.device_address = EXPANDER_ADDR;
    i2c_master_bus_add_device(bus, &dev_cfg, &exp);
    dev_cfg.device_address = ES7210_ADDR;
    i2c_master_bus_add_device(bus, &dev_cfg, &adc);

    // Audio rail up, PA (P4) low. The expander takes register-then-value writes.
    if (write_reg(exp, EXP_REG_MODE, 0xFF) || write_reg(exp, EXP_REG_OUT, 1 << EXP_RAIL_PIN)) {
        printf("expander write failed\n");
        goto out;
    }
    vTaskDelay(pdMS_TO_TICKS(50));
    if (es7210_init(adc, gain)) {
        printf("es7210 init failed\n");
        goto out;
    }

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_frame_num = 512;
    if (i2s_new_channel(&chan_cfg, NULL, &rx) != ESP_OK) goto out;
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(RATE),  // MCLK = 256 * Fs
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = PIN_MCLK, .bclk = PIN_BCLK, .ws = PIN_WS,
            .dout = I2S_GPIO_UNUSED, .din = PIN_DIN,
        },
    };
    if (i2s_channel_init_std_mode(rx, &std_cfg) != ESP_OK || i2s_channel_enable(rx) != ESP_OK) {
        printf("i2s init failed\n");
        goto out;
    }

    // Settle, discarding.
    size_t got;
    int16_t *scratch = pcm;
    int64_t warm_end = bench_now_us() + WARMUP_MS * 1000;
    while (bench_now_us() < warm_end) {
        i2s_channel_read(rx, scratch, 4096, &got, 1000);
    }

    printf("@@REC start rate=%d ch=2 bits=16 secs=%d gain=%d\n", RATE, secs, gain);
    fflush(stdout);
    size_t done = 0;
    while (done < bytes) {
        size_t want = bytes - done > 8192 ? 8192 : bytes - done;
        if (i2s_channel_read(rx, (uint8_t *)pcm + done, want, &got, 1000) != ESP_OK) {
            printf("i2s read failed at %u\n", (unsigned)done);
            goto out;
        }
        done += got;
    }

    // Per-channel peak / RMS / DC: catches a dead mic or a clipped one at a glance.
    size_t frames = bytes / 4;
    for (int ch = 0; ch < 2; ch++) {
        int64_t sum = 0;
        double sq = 0;
        int peak = 0;
        for (size_t i = 0; i < frames; i++) {
            int v = pcm[i * 2 + ch];
            sum += v;
            sq += (double)v * v;
            if (v < 0 ? -v > peak : v > peak) peak = v < 0 ? -v : v;
        }
        double mean = (double)sum / frames;
        double rms = sqrt(sq / frames - mean * mean);
        BENCH_RESULT("rec", "\"ch\":%d,\"peak\":%d,\"rms\":%.1f,\"dc\":%.1f,\"frames\":%u",
                     ch, peak, rms, mean, (unsigned)frames);
    }

    static char line[4200];
    for (size_t off = 0; off < bytes; off += 3072) {
        size_t n = bytes - off > 3072 ? 3072 : bytes - off;
        b64_line((const uint8_t *)pcm + off, n, line);
        printf("@@A %s\n", line);
    }
    rc = 0;

out:
    if (rx) {
        i2s_channel_disable(rx);
        i2s_del_channel(rx);
    }
    if (exp) write_reg(exp, EXP_REG_OUT, 0);  // rail down
    if (adc) i2c_master_bus_rm_device(adc);
    if (exp) i2c_master_bus_rm_device(exp);
    if (bus) i2c_del_master_bus(bus);
    heap_caps_free(pcm);
    return rc;
}
