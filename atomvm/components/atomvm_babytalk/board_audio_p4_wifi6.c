// BabyTalk's board_audio.h for the Waveshare ESP32-P4-WIFI6 (CONFIG_BABYTALK_BOARD_WAVESHARE_P4_WIFI6):
//   ES8311   mono codec, ADC (the board's microphone) and DAC   I2C 0x18 on I2C 0 (SDA 7, SCL 8) + I2S 0
//   NS4150B  class-D speaker amp (the MX1.25 speaker header)    enable = GPIO 53, high = on
// I2S pins (Waveshare's wiki): MCLK 13, BCLK 12, WS 10, ESP out (codec DSDIN) 9, ESP in (codec
// ASDOUT) 11. The ESP is the I2S master at MCLK = 256 x Fs, so the codec's dividers are one
// fixed row of Espressif's table for every rate. The mic and speaker take turns on the port,
// as BabyTalk's NIFs expect. Same codec setup as a board with one ES8311 for both directions
// (the LilyGO T-LoRa Pager's, in the Watchtower firmware).
//
// ES8311 register sequence from Espressif's esp_codec_dev es8311 driver (open + start in
// BOTH mode), Copyright Espressif Systems (Shanghai) CO LTD, Apache-2.0; rewritten in C for
// one fixed configuration.
#include "board_audio.h"

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "esp_attr.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TAG "board_audio"

#define PIN_SDA 7
#define PIN_SCL 8
#define PIN_MCLK 13
#define PIN_BCLK 12
#define PIN_WS 10
#define PIN_DOUT 9
#define PIN_DIN 11
#define PIN_AMP 53
#define ES8311_ADDR 0x18
#define MIC_RATE 16000
#define WARMUP_FRAMES (MIC_RATE / 5)  // 200 ms
#define MIC_GAIN_DEFAULT 4            // ADC digital gain step (6 dB each): 24 dB

static i2c_master_dev_handle_t s_dac;
static i2s_chan_handle_t s_rx;
static int s_ready;

static int wr(uint8_t reg, uint8_t val)
{
    uint8_t b[2] = {reg, val};
    return i2c_master_transmit(s_dac, b, 2, 100) == ESP_OK ? 0 : -1;
}

static int rd(uint8_t reg, uint8_t *val)
{
    return i2c_master_transmit_receive(s_dac, &reg, 1, val, 1, 100) == ESP_OK ? 0 : -1;
}

// ADC + DAC, 16-bit I2S slave, analog mic on MIC1P/N
static int es8311_init(int gain)
{
    uint8_t v;
    // the first write after power-up sometimes fails (Espressif's note): write twice
    wr(0x44, 0x08);
    if (wr(0x44, 0x08)) return -1;                               // I2C noise immunity; ADC on both slots
    if (wr(0x01, 0x30) || wr(0x02, 0x00) || wr(0x03, 0x10) || wr(0x16, 0x20 | gain)
        || wr(0x04, 0x10) || wr(0x05, 0x00) || wr(0x0B, 0x00) || wr(0x0C, 0x00)
        || wr(0x10, 0x1F) || wr(0x11, 0x7F) || wr(0x00, 0x80)) return -1;
    if (rd(0x00, &v) || wr(0x00, v & 0xBF)) return -1;           // slave
    if (wr(0x01, 0x3F)) return -1;                               // clocks from MCLK, all on
    // MCLK = 256 x Fs: pre_div 1, pre_multi 1, adc/dac_div 1, OSR 0x10, LRCK 256, BCLK / 4
    if (rd(0x02, &v) || wr(0x02, v & 0x07)) return -1;
    if (wr(0x05, 0x00)) return -1;
    if (rd(0x03, &v) || wr(0x03, (v & 0x80) | 0x10)) return -1;
    if (rd(0x04, &v) || wr(0x04, (v & 0x80) | 0x10)) return -1;
    if (rd(0x07, &v) || wr(0x07, v & 0xC0) || wr(0x08, 0xFF)) return -1;
    if (rd(0x06, &v) || wr(0x06, (v & 0xE0) | 0x03)) return -1;
    if (wr(0x13, 0x10) || wr(0x1B, 0x0A) || wr(0x1C, 0x6A)) return -1;  // ADC HPF / EQ bypass
    if (wr(0x09, 0x0C) || wr(0x0A, 0x0C)) return -1;             // SDP in/out: I2S 16-bit, on
    if (wr(0x17, 0xBF) || wr(0x0E, 0x02) || wr(0x12, 0x00) || wr(0x14, 0x1A)  // ADC vol, PGA, DAC up, mic in
        || wr(0x0D, 0x01) || wr(0x15, 0x40) || wr(0x37, 0x08) || wr(0x45, 0x00)) return -1;
    if (rd(0xFD, &v) || v != 0x83) ESP_LOGW(TAG, "ES8311 id 0x%02x (expected 0x83)", v);
    return 0;
}

static void amp(int on) { gpio_set_level(PIN_AMP, on); }

int board_audio_init(int mic_gain)
{
    if (s_ready) return 0;
    if (!s_dac) {
        // I2C 0 may be up already (another driver on the header's bus): share it
        i2c_master_bus_handle_t bus;
        if (i2c_master_get_bus_handle(0, &bus) != ESP_OK) {
            const i2c_master_bus_config_t cfg = {
                .i2c_port = 0, .sda_io_num = PIN_SDA, .scl_io_num = PIN_SCL,
                .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
                .flags = {.enable_internal_pullup = true},
            };
            if (i2c_new_master_bus(&cfg, &bus) != ESP_OK) return -1;
        }
        i2c_device_config_t dev = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = ES8311_ADDR,
                                   .scl_speed_hz = 100000};
        if (i2c_master_bus_add_device(bus, &dev, &s_dac) != ESP_OK) return -1;
    }
    const gpio_config_t pa = {.pin_bit_mask = 1ULL << PIN_AMP, .mode = GPIO_MODE_OUTPUT};
    gpio_config(&pa);
    amp(0);
    if (es8311_init(mic_gain < 0 ? MIC_GAIN_DEFAULT : (mic_gain > 7 ? 7 : mic_gain))) return -4;
    s_ready = 1;
    ESP_LOGI(TAG, "ES8311 up (mic and speaker)");
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

static int16_t s_st[1024];  // 512 stereo frames of staging (reads/writes are chunked to fit)

int board_audio_rx_start(void)
{
    if (!s_ready) return -1;
    if (!s_rx) {
        i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
        cc.dma_frame_num = 320;
        cc.dma_desc_num = 6;  // 120 ms of slack in 7.5 KB
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
        for (size_t i = 0; i < got / 4; i++) mono[done + i] = s_st[2 * i];  // the ADC: left slot
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
    if (wr(0x32, volume ? (uint8_t) (volume * 256 / 100 - 1) : 0)) return -2;
    i2s_chan_handle_t tx;
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_frame_num = 480;
    cc.dma_desc_num = 4;   // 80 ms at 24 kHz, 7.5 KB
    cc.auto_clear = true;  // underrun -> silence, not a repeated buffer
    if (i2s_new_channel(&cc, &tx, NULL) != ESP_OK) return -5;
    i2s_std_config_t sc = std_cfg(rate, PIN_DOUT, I2S_GPIO_UNUSED);
    const i2s_event_callbacks_t cb = {.on_send_q_ovf = on_underrun};
    s_underruns = 0;
    int rc = i2s_channel_init_std_mode(tx, &sc) == ESP_OK && i2s_channel_register_event_callback(tx, &cb, NULL) == ESP_OK
                     && i2s_channel_enable(tx) == ESP_OK ? 0 : -6;
    if (!rc) {
        rc = tx_write(tx, NULL, rate / 20);               // 50 ms of silence: clocks settle
        if (!rc) amp(1);                                  // then the amp
        if (!rc) rc = tx_write(tx, mono, n);
        if (!rc) rc = tx_write(tx, NULL, rate * 3 / 10);  // drain the DMA ring before amp off
        amp(0);
        i2s_channel_disable(tx);
    }
    i2s_del_channel(tx);
    return rc;
}
