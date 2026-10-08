// BabyTalk's built-in audio driver (board_audio.h), configured at run time
// (board_audio_config.h): any board made of these parts, on any pins --
//   ES7210   4-channel mic ADC                            I2C + I2S in
//   ES8311   mono codec: DAC for the speaker, and/or ADC for a mic
//   a speaker amp enable (NS4150B and the like) on a GPIO or an IO expander (CH32V003,
//   PCA9555/TCA9555/XL9555), and optionally an audio power rail switched the same way
// Presets: the Waveshare ESP32-S3-CAM (ES7210 mics, ES8311 speaker, a CH32V003 switches the
// rail and the amp; docs/BOARD_WAVESHARE_S3_CAM.md), the Waveshare ESP32-P4-WIFI6 (one
// ES8311 for mic and speaker, amp on GPIO 53; docs/BOARD_WAVESHARE_P4_WIFI6.md) and the
// LilyGO T-LoRa Pager (one ES8311 for mic and speaker, NS4150B amp on pin 1 of the XL9555
// that also switches the board's other rails; pins from LilyGO's pins_arduino.h).
//
// The ESP is always the I2S master at MCLK = 256 x Fs, so the codecs' dividers are one fixed
// row of Espressif's tables for every rate; the mic and the speaker take turns on one I2S port.
// I2S runs STEREO both ways (MONO garbles these codecs); the mic is one slot of it.
//
// ES7210 and ES8311 register sequences derived from Espressif's es7210 and es8311 drivers
// (github.com/espressif/esp-bsp, components/es7210, es8311) and esp_codec_dev's es8311 (open +
// start in BOTH mode), Copyright Espressif Systems (Shanghai) CO LTD, Apache-2.0
// (licenses/Apache-2.0.txt); rewritten in C for one fixed configuration each.
#include <sdkconfig.h>
#ifndef CONFIG_BABYTALK_BOARD_CUSTOM

#include "board_audio.h"
#include "board_audio_config.h"

#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "esp_attr.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TAG "board_audio"

#define MIC_RATE 16000
#define WARMUP_FRAMES (MIC_RATE / 5)  // 200 ms

// ------------------------------------------------------------------------------ presets
#define WAVESHARE_S3_CAM {                                                          \
    .i2c_port = 0, .sda = 8, .scl = 7, .i2c_hz = 100000,                            \
    .i2s_port = 0, .mclk = 10, .bclk = 11, .ws = 12, .dout = 14, .din = 13,         \
    .mic = BA_CODEC_ES7210, .mic_addr = 0x40, .mic_gain = -1, .mic_slot = 0,        \
    .spk = BA_CODEC_ES8311, .spk_addr = 0x18,                                       \
    .amp = {BA_CTL_CH32V003, 0x24, 4, 0}, .power = {BA_CTL_CH32V003, 0x24, 6, 0}}
#define WAVESHARE_P4_WIFI6 {                                                        \
    .i2c_port = 0, .sda = 7, .scl = 8, .i2c_hz = 100000,                            \
    .i2s_port = 0, .mclk = 13, .bclk = 12, .ws = 10, .dout = 9, .din = 11,          \
    .mic = BA_CODEC_ES8311, .mic_addr = 0x18, .mic_gain = -1, .mic_slot = 0,        \
    .spk = BA_CODEC_ES8311, .spk_addr = 0x18,                                       \
    .amp = {BA_CTL_GPIO, 0, 53, 0}, .power = {BA_CTL_NONE, 0, -1, 0}}
#define LILYGO_T_LORA_PAGER {                                                       \
    .i2c_port = 0, .sda = 3, .scl = 2, .i2c_hz = 100000,                            \
    .i2s_port = 0, .mclk = 10, .bclk = 11, .ws = 18, .dout = 45, .din = 17,         \
    .mic = BA_CODEC_ES8311, .mic_addr = 0x18, .mic_gain = -1, .mic_slot = 0,        \
    .spk = BA_CODEC_ES8311, .spk_addr = 0x18,                                       \
    .amp = {BA_CTL_PCA9555, 0x20, 1, 0}, .power = {BA_CTL_NONE, 0, -1, 0}}

const board_audio_preset_t board_audio_presets[] = {
    {"\x10" "waveshare_s3_cam", WAVESHARE_S3_CAM},
    {"\x12" "waveshare_p4_wifi6", WAVESHARE_P4_WIFI6},
    {"\x13" "lilygo_t_lora_pager", LILYGO_T_LORA_PAGER},
};
const int board_audio_n_presets = sizeof(board_audio_presets) / sizeof(board_audio_presets[0]);

// until the app configures another: the firmware's default board (Kconfig)
#if defined(CONFIG_BABYTALK_BOARD_WAVESHARE_P4_WIFI6)
static board_audio_config_t s_cfg = WAVESHARE_P4_WIFI6;
#elif defined(CONFIG_BABYTALK_BOARD_LILYGO_T_LORA_PAGER)
static board_audio_config_t s_cfg = LILYGO_T_LORA_PAGER;
#else
static board_audio_config_t s_cfg = WAVESHARE_S3_CAM;
#endif

const board_audio_config_t *board_audio_config(void) { return &s_cfg; }

// ------------------------------------------------------------------------------ checking
static bool pin_ok(int p, bool out) { return out ? GPIO_IS_VALID_OUTPUT_GPIO(p) : GPIO_IS_VALID_GPIO(p); }
static bool pin_opt(int p, bool out) { return p == -1 || pin_ok(p, out); }
static bool addr_ok(int a) { return a >= 0x08 && a <= 0x77; }

static bool ctl_ok(const board_audio_ctl_t *c)
{
    switch (c->kind) {
    case BA_CTL_NONE: return true;
    case BA_CTL_GPIO: return pin_ok(c->pin, true);
    case BA_CTL_CH32V003: return addr_ok(c->addr) && c->pin >= 0 && c->pin < 8;
    case BA_CTL_PCA9555: return addr_ok(c->addr) && c->pin >= 0 && c->pin < 16;
    default: return false;
    }
}

// field numbers: positions in babytalk.erl's tuple (1-based)
int board_audio_check(const board_audio_config_t *c)
{
    const bool mic = c->mic != BA_CODEC_NONE, spk = c->spk != BA_CODEC_NONE;
    if (c->i2c_port < 0 || c->i2c_port >= SOC_I2C_NUM) return 1;
    if (!pin_ok(c->sda, true)) return 2;
    if (!pin_ok(c->scl, true)) return 3;
    if (c->i2c_hz < 10000 || c->i2c_hz > 1000000) return 4;
    if (c->i2s_port < 0 || c->i2s_port >= SOC_I2S_NUM) return 5;
    if (!pin_opt(c->mclk, true)) return 6;
    if (!pin_ok(c->bclk, true)) return 7;
    if (!pin_ok(c->ws, true)) return 8;
    if (spk ? !pin_ok(c->dout, true) : !pin_opt(c->dout, true)) return 9;
    if (mic ? !pin_ok(c->din, false) : !pin_opt(c->din, false)) return 10;
    if (c->mic < BA_CODEC_NONE || c->mic > BA_CODEC_ES8311) return 11;
    if (mic && !addr_ok(c->mic_addr)) return 12;
    if (c->mic_gain < -1 || c->mic_gain > (c->mic == BA_CODEC_ES7210 ? 14 : 7)) return 13;
    if (c->mic_slot != 0 && c->mic_slot != 1) return 14;
    if (c->spk != BA_CODEC_NONE && c->spk != BA_CODEC_ES8311) return 15;
    if (spk && !addr_ok(c->spk_addr)) return 16;
    if (!ctl_ok(&c->amp)) return 17;
    if (!ctl_ok(&c->power)) return 21;
    return 0;
}

// ------------------------------------------------------------------------------ I2C
static i2c_master_bus_handle_t s_bus;
static bool s_own_bus;  // we created it (else someone else's, shared: never deleted)
static struct {
    uint8_t addr;
    i2c_master_dev_handle_t h;
} s_dev[4];
static int s_ndev;
static int s_ready;
static i2s_chan_handle_t s_rx;

static i2c_master_dev_handle_t dev(uint8_t addr)
{
    for (int i = 0; i < s_ndev; i++)
        if (s_dev[i].addr == addr) return s_dev[i].h;
    if (s_ndev == 4) return NULL;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = addr,
                              .scl_speed_hz = (uint32_t) s_cfg.i2c_hz};
    i2c_master_dev_handle_t h;
    if (i2c_master_bus_add_device(s_bus, &dc, &h) != ESP_OK) return NULL;
    s_dev[s_ndev].addr = addr;
    s_dev[s_ndev++].h = h;
    return h;
}

static int wr(i2c_master_dev_handle_t d, uint8_t reg, uint8_t val)
{
    uint8_t b[2] = {reg, val};
    return d && i2c_master_transmit(d, b, 2, 100) == ESP_OK ? 0 : -1;
}

static int rd(i2c_master_dev_handle_t d, uint8_t reg, uint8_t *val)
{
    return d && i2c_master_transmit_receive(d, &reg, 1, val, 1, 100) == ESP_OK ? 0 : -1;
}

// ------------------------------------------------------------------------------ switches
// The CH32V003's output register (0x03) is write-only and takes all eight pins at once: keep
// a shadow per expander.
static struct {
    uint8_t addr, out;
    bool used;
} s_ch[2];

static int ch_slot(uint8_t addr)
{
    for (int i = 0; i < 2; i++)
        if (s_ch[i].used && s_ch[i].addr == addr) return i;
    return -1;
}

// Set a switch on or off; `setup` the first time (the pin's direction)
static int ctl_set(const board_audio_ctl_t *c, bool on, bool setup)
{
    const bool level = on != (c->active_low != 0);
    switch (c->kind) {
    case BA_CTL_GPIO:
        if (setup) {
            const gpio_config_t g = {.pin_bit_mask = 1ULL << c->pin, .mode = GPIO_MODE_OUTPUT};
            if (gpio_config(&g) != ESP_OK) return -1;
        }
        return gpio_set_level(c->pin, level) == ESP_OK ? 0 : -1;
    case BA_CTL_CH32V003: {
        i2c_master_dev_handle_t d = dev(c->addr);
        int i = ch_slot(c->addr);
        if (i < 0) {  // first use: all eight pins outputs, all low
            i = s_ch[0].used ? 1 : 0;
            if (s_ch[i].used || wr(d, 0x02, 0xFF)) return -1;
            s_ch[i].used = true;
            s_ch[i].addr = c->addr;
            s_ch[i].out = 0;
        }
        uint8_t v = level ? (s_ch[i].out | (1 << c->pin)) : (s_ch[i].out & ~(1 << c->pin));
        if (wr(d, 0x03, v)) return -1;
        s_ch[i].out = v;
        return 0;
    }
    case BA_CTL_PCA9555: {  // output registers 0x02/0x03, direction 0x06/0x07 (0 = output)
        i2c_master_dev_handle_t d = dev(c->addr);
        const uint8_t port = c->pin >> 3, bit = 1 << (c->pin & 7);
        uint8_t v;
        if (rd(d, 0x02 + port, &v) || wr(d, 0x02 + port, level ? (v | bit) : (v & ~bit))) return -1;
        if (setup && (rd(d, 0x06 + port, &v) || wr(d, 0x06 + port, v & ~bit))) return -1;
        return 0;
    }
    default:
        return 0;
    }
}

// ------------------------------------------------------------------------------ codecs
static int es7210_init(i2c_master_dev_handle_t d, int gain)
{
    if (gain < 0) gain = 14;  // the Waveshare S3-CAM's default: as the field recordings
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
    for (size_t i = 0; i < sizeof(seq1) / 2; i++)
        if (wr(d, seq1[i][0], seq1[i][1])) return -1;
    for (uint8_t r = 0x43; r <= 0x46; r++)       // PGA gain, all four channels
        if (wr(d, r, (uint8_t) gain | 0x10)) return -1;
    for (size_t i = 0; i < sizeof(seq2) / 2; i++)
        if (wr(d, seq2[i][0], seq2[i][1])) return -1;
    return 0;
}

// ES8311 as the speaker's DAC only, 16-bit I2S slave
static int es8311_dac_init(i2c_master_dev_handle_t d)
{
    uint8_t v;
    if (wr(d, 0x00, 0x1F)) return -1;            // reset
    vTaskDelay(pdMS_TO_TICKS(20));
    if (wr(d, 0x00, 0x00) || wr(d, 0x00, 0x80)) return -1;   // power on
    if (wr(d, 0x01, 0x3F)) return -1;            // clocks from MCLK, all on
    if (rd(d, 0x02, &v) || wr(d, 0x02, v & 0x07)) return -1;  // pre_div 1, pre_multi 0
    if (wr(d, 0x03, 0x10) || wr(d, 0x04, 0x10) || wr(d, 0x05, 0x00)) return -1;  // OSR, dividers
    if (rd(d, 0x06, &v) || wr(d, 0x06, (v & 0xE0) | 0x03)) return -1;  // bclk_div 4
    if (rd(d, 0x07, &v) || wr(d, 0x07, v & 0xC0) || wr(d, 0x08, 0xFF)) return -1;  // LRCK 256
    if (rd(d, 0x00, &v) || wr(d, 0x00, v & 0xBF)) return -1;  // slave
    if (wr(d, 0x09, 0x0C) || wr(d, 0x0A, 0x0C)) return -1;    // SDP in/out: I2S 16-bit
    static const uint8_t up[][2] = {{0x0D, 0x01}, {0x0E, 0x02}, {0x12, 0x00}, {0x13, 0x10},
                                    {0x1C, 0x6A}, {0x37, 0x08}};  // analog, DAC, output driver
    for (size_t i = 0; i < sizeof(up) / 2; i++)
        if (wr(d, up[i][0], up[i][1])) return -1;
    return 0;
}

// ES8311 for both directions: ADC (analog mic on MIC1P/N) and DAC, 16-bit I2S slave
static int es8311_both_init(i2c_master_dev_handle_t d, int gain)
{
    if (gain < 0) gain = 4;  // ADC digital gain step (6 dB each): 24 dB
    uint8_t v;
    // the first write after power-up sometimes fails (Espressif's note): write twice
    wr(d, 0x44, 0x08);
    if (wr(d, 0x44, 0x08)) return -1;                            // I2C noise immunity; ADC on both slots
    if (wr(d, 0x01, 0x30) || wr(d, 0x02, 0x00) || wr(d, 0x03, 0x10) || wr(d, 0x16, 0x20 | gain)
        || wr(d, 0x04, 0x10) || wr(d, 0x05, 0x00) || wr(d, 0x0B, 0x00) || wr(d, 0x0C, 0x00)
        || wr(d, 0x10, 0x1F) || wr(d, 0x11, 0x7F) || wr(d, 0x00, 0x80)) return -1;
    if (rd(d, 0x00, &v) || wr(d, 0x00, v & 0xBF)) return -1;     // slave
    if (wr(d, 0x01, 0x3F)) return -1;                            // clocks from MCLK, all on
    // MCLK = 256 x Fs: pre_div 1, pre_multi 1, adc/dac_div 1, OSR 0x10, LRCK 256, BCLK / 4
    if (rd(d, 0x02, &v) || wr(d, 0x02, v & 0x07)) return -1;
    if (wr(d, 0x05, 0x00)) return -1;
    if (rd(d, 0x03, &v) || wr(d, 0x03, (v & 0x80) | 0x10)) return -1;
    if (rd(d, 0x04, &v) || wr(d, 0x04, (v & 0x80) | 0x10)) return -1;
    if (rd(d, 0x07, &v) || wr(d, 0x07, v & 0xC0) || wr(d, 0x08, 0xFF)) return -1;
    if (rd(d, 0x06, &v) || wr(d, 0x06, (v & 0xE0) | 0x03)) return -1;
    if (wr(d, 0x13, 0x10) || wr(d, 0x1B, 0x0A) || wr(d, 0x1C, 0x6A)) return -1;  // ADC HPF / EQ bypass
    if (wr(d, 0x09, 0x0C) || wr(d, 0x0A, 0x0C)) return -1;       // SDP in/out: I2S 16-bit, on
    if (wr(d, 0x17, 0xBF) || wr(d, 0x0E, 0x02) || wr(d, 0x12, 0x00) || wr(d, 0x14, 0x1A)  // ADC vol, PGA, DAC up, mic in
        || wr(d, 0x0D, 0x01) || wr(d, 0x15, 0x40) || wr(d, 0x37, 0x08) || wr(d, 0x45, 0x00)) return -1;
    if (rd(d, 0xFD, &v) || v != 0x83) ESP_LOGW(TAG, "ES8311 id 0x%02x (expected 0x83)", v);
    return 0;
}

// ------------------------------------------------------------------------------ bring-up
// Negative on failure: -1 the I2C bus, -2 the power rail or amp switch, -3 the mic's codec,
// -4 the speaker's codec
int board_audio_init(int mic_gain)
{
    if (s_ready) return 0;
    const board_audio_config_t *c = &s_cfg;
    if (mic_gain < 0) mic_gain = c->mic_gain;
    if (!s_bus) {
        // I2C may be up already (a camera's SCCB, a display, the app): share that bus
        if (i2c_master_get_bus_handle(c->i2c_port, &s_bus) == ESP_OK) {
            s_own_bus = false;
        } else {
            const i2c_master_bus_config_t bc = {
                .i2c_port = c->i2c_port, .sda_io_num = c->sda, .scl_io_num = c->scl,
                .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
                .flags = {.enable_internal_pullup = true},
            };
            if (i2c_new_master_bus(&bc, &s_bus) != ESP_OK) {
                s_bus = NULL;
                return -1;
            }
            s_own_bus = true;
        }
    }
    // the rail up and the amp off (the amp only once I2S is streaming: no turn-on pop)
    const bool rail = c->power.kind != BA_CTL_NONE;
    if (rail && ctl_set(&c->power, true, true)) return -2;
    if (c->amp.kind != BA_CTL_NONE && ctl_set(&c->amp, false, true)) return -2;
    if (rail) vTaskDelay(pdMS_TO_TICKS(50));

    bool spk_done = false;
    if (c->mic == BA_CODEC_ES7210) {
        if (es7210_init(dev(c->mic_addr), mic_gain)) return -3;
    } else if (c->mic == BA_CODEC_ES8311) {
        if (es8311_both_init(dev(c->mic_addr), mic_gain)) return -3;
        spk_done = c->spk == BA_CODEC_ES8311 && c->spk_addr == c->mic_addr;
    }
    if (c->spk == BA_CODEC_ES8311 && !spk_done && es8311_dac_init(dev(c->spk_addr))) return -4;
    s_ready = 1;
    ESP_LOGI(TAG, "audio up: I2C %d (SDA %d, SCL %d), I2S %d (MCLK %d, BCLK %d, WS %d, out %d, in %d)",
             c->i2c_port, c->sda, c->scl, c->i2s_port, c->mclk, c->bclk, c->ws, c->dout, c->din);
    return 0;
}

void board_audio_configure(const board_audio_config_t *c)
{
    board_audio_rx_stop();
    if (s_bus) {  // (only if we brought it up) amp and rail off, the pins and the bus given back
        if (s_cfg.amp.kind != BA_CTL_NONE) ctl_set(&s_cfg.amp, false, false);
        if (s_cfg.power.kind != BA_CTL_NONE) ctl_set(&s_cfg.power, false, false);
        if (s_cfg.amp.kind == BA_CTL_GPIO) gpio_reset_pin(s_cfg.amp.pin);
        if (s_cfg.power.kind == BA_CTL_GPIO) gpio_reset_pin(s_cfg.power.pin);
    }
    for (int i = 0; i < s_ndev; i++) i2c_master_bus_rm_device(s_dev[i].h);
    s_ndev = 0;
    memset(s_ch, 0, sizeof(s_ch));
    if (s_bus && s_own_bus) i2c_del_master_bus(s_bus);
    s_bus = NULL;
    s_ready = 0;
    s_cfg = *c;
}

// ------------------------------------------------------------------------------ I2S
static i2s_std_config_t std_cfg(int rate, int dout, int din)
{
    const board_audio_config_t *c = &s_cfg;
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(rate),  // MCLK = 256 x Fs
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {.mclk = c->mclk, .bclk = c->bclk, .ws = c->ws, .dout = dout, .din = din},
    };
    return sc;
}

static int16_t s_st[1024];  // 512 stereo frames of staging (reads/writes are chunked to fit)

int board_audio_rx_start(void)
{
    if (!s_ready) return -1;
    if (s_cfg.mic == BA_CODEC_NONE) return -8;
    if (!s_rx) {
        i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(s_cfg.i2s_port, I2S_ROLE_MASTER);
        cc.dma_frame_num = 320;
        cc.dma_desc_num = 6;  // 120 ms of slack in 7.5 KB of internal DMA RAM (a camera, WiFi and TLS want theirs)
        if (i2s_new_channel(&cc, NULL, &s_rx) != ESP_OK) return -5;
        i2s_std_config_t sc = std_cfg(MIC_RATE, I2S_GPIO_UNUSED, s_cfg.din);
        if (i2s_channel_init_std_mode(s_rx, &sc) != ESP_OK) {
            i2s_del_channel(s_rx);
            s_rx = NULL;
            return -5;
        }
    }
    if (i2s_channel_enable(s_rx) != ESP_OK) return -6;
    for (int left = WARMUP_FRAMES; left > 0;) {
        size_t got;
        int want = left > 512 ? 512 : left;
        if (i2s_channel_read(s_rx, s_st, want * 4, &got, 1000) != ESP_OK) return -7;
        left -= got / 4;
    }
    return 0;
}

int board_audio_rx_read(int16_t *mono, int frames)
{
    const int slot = s_cfg.mic_slot;
    for (int done = 0; done < frames;) {
        int want = frames - done > 512 ? 512 : frames - done;
        size_t got;
        if (i2s_channel_read(s_rx, s_st, want * 4, &got, 1000) != ESP_OK) return -7;
        for (size_t i = 0; i < got / 4; i++) mono[done + i] = s_st[2 * i + slot];
        done += got / 4;
    }
    return 0;
}

void board_audio_rx_stop(void)
{
    if (!s_rx) return;
    i2s_channel_disable(s_rx);
    i2s_del_channel(s_rx);
    s_rx = NULL;
}

// n frames of mono (NULL: silence) as stereo
static int tx_write(i2s_chan_handle_t tx, const int16_t *mono, int n)
{
    for (int done = 0; done < n;) {
        int k = n - done > 512 ? 512 : n - done;
        for (int i = 0; i < k; i++) s_st[2 * i] = s_st[2 * i + 1] = mono ? mono[done + i] : 0;
        size_t put;
        if (i2s_channel_write(tx, s_st, (size_t) k * 4, &put, 1000) != ESP_OK) return -7;
        done += k;
    }
    return 0;
}

static volatile int s_underruns;

static bool IRAM_ATTR on_underrun(i2s_chan_handle_t h, i2s_event_data_t *e, void *u)
{
    (void) h;
    (void) e;
    (void) u;
    s_underruns++;
    return false;
}

int board_audio_underruns(void) { return s_underruns; }

int board_audio_play(const int16_t *mono, int n, int rate, int volume)
{
    if (!s_ready || s_rx) return -1;  // the port is capturing
    if (s_cfg.spk == BA_CODEC_NONE) return -8;
    volume = volume < 0 ? 0 : (volume > 100 ? 100 : volume);
    if (wr(dev(s_cfg.spk_addr), 0x32, volume ? (uint8_t) (volume * 256 / 100 - 1) : 0)) return -2;
    i2s_chan_handle_t tx;
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(s_cfg.i2s_port, I2S_ROLE_MASTER);
    cc.dma_frame_num = 480;
    cc.dma_desc_num = 4;  // 80 ms at 24 kHz in 7.5 KB (the audio task runs at priority 6: enough)
    cc.auto_clear = true;  // underrun -> silence, not a repeated buffer
    if (i2s_new_channel(&cc, &tx, NULL) != ESP_OK) return -5;
    i2s_std_config_t sc = std_cfg(rate, s_cfg.dout, I2S_GPIO_UNUSED);
    const i2s_event_callbacks_t cb = {.on_send_q_ovf = on_underrun};
    s_underruns = 0;
    int rc = i2s_channel_init_std_mode(tx, &sc) == ESP_OK && i2s_channel_register_event_callback(tx, &cb, NULL) == ESP_OK
                     && i2s_channel_enable(tx) == ESP_OK ? 0 : -6;
    const bool amp = s_cfg.amp.kind != BA_CTL_NONE;
    if (!rc) {
        rc = tx_write(tx, NULL, rate / 20);                               // 50 ms of silence: clocks settle
        if (!rc && amp) rc = ctl_set(&s_cfg.amp, true, false) ? -2 : 0;   // then the amp
        if (!rc) rc = tx_write(tx, mono, n);
        if (!rc) rc = tx_write(tx, NULL, rate * 3 / 10);                  // drain the DMA ring before amp off
        if (amp) ctl_set(&s_cfg.amp, false, false);
        i2s_channel_disable(tx);
    }
    i2s_del_channel(tx);
    return rc;
}

#endif  // !CONFIG_BABYTALK_BOARD_CUSTOM
