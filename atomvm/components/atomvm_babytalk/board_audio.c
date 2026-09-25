// Waveshare ESP32-S3-CAM audio drivers (see board_audio.h). Register sequences: ES7210 from
// stt/main/mic.c (a port of Watchtower's proven es7210.py), ES8311 from mpy/drivers/es8311.py
// (Espressif's driver with the clock table collapsed to its MCLK = 256 x Fs row). Pin roles:
// BOARD_WAVESHARE_S3_CAM.md (P6 rail / P4 amp confirmed by ear, 2026-09-21).
#include "board_audio.h"

#include <string.h>

#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "esp_attr.h"
#include "freertos/task.h"

#define PIN_SCL 7
#define PIN_SDA 8
#define PIN_MCLK 10
#define PIN_BCLK 11
#define PIN_WS 12
#define PIN_DIN 13
#define PIN_DOUT 14

#define EXPANDER_ADDR 0x24
#define EXP_REG_MODE 0x02
#define EXP_REG_OUT 0x03  // write-only, all 8 pins at once: we keep a shadow
#define EXP_RAIL 6
#define EXP_AMP 4

#define ES7210_ADDR 0x40
#define ES8311_ADDR 0x18
#define MIC_RATE 16000
#define WARMUP_FRAMES (MIC_RATE / 5)  // 200 ms

static i2c_master_dev_handle_t s_exp, s_adc, s_dac;
static uint8_t s_exp_out;
static i2s_chan_handle_t s_rx;
static int s_ready;

static int wr(i2c_master_dev_handle_t d, uint8_t reg, uint8_t val)
{
    uint8_t b[2] = {reg, val};
    return i2c_master_transmit(d, b, 2, 100) == ESP_OK ? 0 : -1;
}

static int rd(i2c_master_dev_handle_t d, uint8_t reg, uint8_t *val)
{
    return i2c_master_transmit_receive(d, &reg, 1, val, 1, 100) == ESP_OK ? 0 : -1;
}

static int exp_pin(int pin, int on)
{
    uint8_t v = on ? (s_exp_out | (1 << pin)) : (s_exp_out & ~(1 << pin));
    if (wr(s_exp, EXP_REG_OUT, v)) return -1;
    s_exp_out = v;
    return 0;
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
    for (size_t i = 0; i < sizeof(seq1) / 2; i++)
        if (wr(d, seq1[i][0], seq1[i][1])) return -1;
    for (uint8_t r = 0x43; r <= 0x46; r++)       // PGA gain, all four channels
        if (wr(d, r, (uint8_t) gain | 0x10)) return -1;
    for (size_t i = 0; i < sizeof(seq2) / 2; i++)
        if (wr(d, seq2[i][0], seq2[i][1])) return -1;
    return 0;
}

// DAC path, 16-bit I2S slave; the dividers are the MCLK = 256 x Fs row of Espressif's
// coefficient table, identical for every rate from 8 to 64 kHz, so the rate is set by I2S alone.
static int es8311_init(i2c_master_dev_handle_t d)
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

int board_audio_init(int mic_gain)
{
    if (s_ready) return 0;
    i2c_master_bus_handle_t bus;
    const i2c_master_bus_config_t cfg = {
        .i2c_port = 0, .sda_io_num = PIN_SDA, .scl_io_num = PIN_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
        .flags = {.enable_internal_pullup = true},
    };
    if (i2c_new_master_bus(&cfg, &bus) != ESP_OK) return -1;
    i2c_device_config_t dev = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .scl_speed_hz = 100000};
    dev.device_address = EXPANDER_ADDR;
    i2c_master_bus_add_device(bus, &dev, &s_exp);
    dev.device_address = ES7210_ADDR;
    i2c_master_bus_add_device(bus, &dev, &s_adc);
    dev.device_address = ES8311_ADDR;
    i2c_master_bus_add_device(bus, &dev, &s_dac);
    s_exp_out = 1 << EXP_RAIL;  // rail up, amp (and the rest) off
    if (wr(s_exp, EXP_REG_MODE, 0xFF) || wr(s_exp, EXP_REG_OUT, s_exp_out)) return -2;
    vTaskDelay(pdMS_TO_TICKS(50));
    if (es7210_init(s_adc, mic_gain)) return -3;
    if (es8311_init(s_dac)) return -4;
    s_ready = 1;
    return 0;
}

static i2s_std_config_t std_cfg(int rate, int dout, int din)
{
    i2s_std_config_t c = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(rate),  // MCLK = 256 x Fs
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {.mclk = PIN_MCLK, .bclk = PIN_BCLK, .ws = PIN_WS, .dout = dout, .din = din},
    };
    return c;
}

static int16_t s_st[1024];  // 512 stereo frames of staging (internal, DMA-adjacent)

int board_audio_rx_start(void)
{
    if (!s_ready) return -1;
    if (!s_rx) {
        i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
        cc.dma_frame_num = 512;
        cc.dma_desc_num = 8;  // 256 ms of slack
        if (i2s_new_channel(&cc, NULL, &s_rx) != ESP_OK) return -5;
        i2s_std_config_t sc = std_cfg(MIC_RATE, I2S_GPIO_UNUSED, PIN_DIN);
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
    for (int done = 0; done < frames;) {
        int want = frames - done > 512 ? 512 : frames - done;
        size_t got;
        if (i2s_channel_read(s_rx, s_st, want * 4, &got, 1000) != ESP_OK) return -7;
        for (size_t i = 0; i < got / 4; i++) mono[done + i] = s_st[2 * i];  // mic 1 = left
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
    volume = volume < 0 ? 0 : (volume > 100 ? 100 : volume);
    if (wr(s_dac, 0x32, volume ? (uint8_t) (volume * 256 / 100 - 1) : 0)) return -2;
    i2s_chan_handle_t tx;
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_frame_num = 480;
    cc.dma_desc_num = 8;  // 160 ms at 24 kHz
    cc.auto_clear = true;  // underrun -> silence, not a repeated buffer
    if (i2s_new_channel(&cc, &tx, NULL) != ESP_OK) return -5;
    i2s_std_config_t sc = std_cfg(rate, PIN_DOUT, I2S_GPIO_UNUSED);
    const i2s_event_callbacks_t cb = {.on_send_q_ovf = on_underrun};
    s_underruns = 0;
    int rc = i2s_channel_init_std_mode(tx, &sc) == ESP_OK && i2s_channel_register_event_callback(tx, &cb, NULL) == ESP_OK
                     && i2s_channel_enable(tx) == ESP_OK ? 0 : -6;
    if (!rc) {
        rc = tx_write(tx, NULL, rate / 20);               // 50 ms of silence: clocks settle
        if (!rc) rc = exp_pin(EXP_AMP, 1) ? -2 : 0;       // then the amp
        if (!rc) rc = tx_write(tx, mono, n);
        if (!rc) rc = tx_write(tx, NULL, rate * 3 / 10);  // drain the DMA ring before amp off
        exp_pin(EXP_AMP, 0);
        i2s_channel_disable(tx);
    }
    i2s_del_channel(tx);
    return rc;
}
