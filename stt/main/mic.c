// Waveshare S3 cam ES7210 dual-mic capture, kept open between recordings.
// Register sequence and pins ported verbatim from bench/main/bench_rec.c (the same
// sequence as mpy/drivers/es7210.py): expander P6 raises the audio rail, ES7210 over
// I2C, I2S RX STEREO (MONO garbles this codec pair), MCLK = 256*Fs.
//
// ES7210 register sequence derived from Espressif's es7210 driver (github.com/espressif/
// esp-bsp, components/es7210), Copyright Espressif Systems (Shanghai) CO LTD, Apache-2.0
// (licenses/Apache-2.0.txt); rewritten in C for one fixed configuration.
#include "mic.h"

#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

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
#define WARMUP_BYTES (RATE * 4 / 5)  // 200 ms of stereo s16: ADC/HPF settling

static i2s_chan_handle_t s_rx;

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

int mic_open(int gain)
{
    if (s_rx) return 0;
    i2c_master_bus_handle_t bus = NULL;
    i2c_master_dev_handle_t exp = NULL, adc = NULL;
    const i2c_master_bus_config_t bus_cfg = {
        .i2c_port = 0, .sda_io_num = PIN_SDA, .scl_io_num = PIN_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
        .flags = {.enable_internal_pullup = true},
    };
    if (i2c_new_master_bus(&bus_cfg, &bus) != ESP_OK) return -1;
    i2c_device_config_t dev_cfg = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .scl_speed_hz = 100000};
    dev_cfg.device_address = EXPANDER_ADDR;
    i2c_master_bus_add_device(bus, &dev_cfg, &exp);
    dev_cfg.device_address = ES7210_ADDR;
    i2c_master_bus_add_device(bus, &dev_cfg, &adc);
    if (write_reg(exp, EXP_REG_MODE, 0xFF) || write_reg(exp, EXP_REG_OUT, 1 << EXP_RAIL_PIN)) return -2;
    vTaskDelay(pdMS_TO_TICKS(50));
    if (es7210_init(adc, gain)) return -3;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_frame_num = 512;
    chan_cfg.dma_desc_num = 8;
    if (i2s_new_channel(&chan_cfg, NULL, &s_rx) != ESP_OK) return -4;
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {.mclk = PIN_MCLK, .bclk = PIN_BCLK, .ws = PIN_WS,
                     .dout = I2S_GPIO_UNUSED, .din = PIN_DIN},
    };
    if (i2s_channel_init_std_mode(s_rx, &std_cfg) != ESP_OK) return -5;
    return 0;
}

int mic_start(void)
{
    if (!s_rx || i2s_channel_enable(s_rx) != ESP_OK) return -1;
    static int16_t junk[1024];
    size_t got, left = WARMUP_BYTES;
    while (left) {
        size_t want = left > sizeof(junk) ? sizeof(junk) : left;
        if (i2s_channel_read(s_rx, junk, want, &got, 1000) != ESP_OK) return -1;
        left -= got;
    }
    return 0;
}

int mic_read_mono(int16_t *dst, int frames)
{
    static int16_t st[1024];  // 512 stereo frames
    int done = 0;
    while (done < frames) {
        int want = frames - done > 512 ? 512 : frames - done;
        size_t got;
        if (i2s_channel_read(s_rx, st, want * 4, &got, 1000) != ESP_OK) return -1;
        for (size_t i = 0; i < got / 4; i++) dst[done + i] = st[2 * i];  // mic1 = left
        done += got / 4;
    }
    return 0;
}

void mic_stop(void)
{
    if (s_rx) i2s_channel_disable(s_rx);
}
