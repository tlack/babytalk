// Run-time configuration of BabyTalk's built-in audio driver (board_audio.c): which pins,
// which codec chips at which I2C addresses, and how the speaker amp and the audio power rail
// are switched. One firmware per chip (ESP32-S3, ESP32-P4) then serves every board built
// from these parts; an app picks a preset and/or overrides fields from Erlang
// (babytalk:audio_config/1). Not present with CONFIG_BABYTALK_BOARD_CUSTOM, where another
// component implements board_audio.h itself.
//
// Nothing touches a pin until the first listen/play or an audio_config call: a firmware
// whose default preset doesn't match the board is harmless until then.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { BA_CODEC_NONE = 0, BA_CODEC_ES7210 = 1, BA_CODEC_ES8311 = 2 };
// How a switch (the amp enable, the audio rail) is driven:
//   GPIO      a pin of the chip                                    pin
//   CH32V003  Waveshare's CH32V003 IO expander (write-only latch)  addr, pin 0..7
//   PCA9555   PCA9555 / TCA9555 / XL9555 16-bit IO expander        addr, pin 0..15
enum { BA_CTL_NONE = 0, BA_CTL_GPIO = 1, BA_CTL_CH32V003 = 2, BA_CTL_PCA9555 = 3 };

typedef struct {
    int8_t kind;        // BA_CTL_*
    uint8_t addr;       // I2C address (expanders)
    int8_t pin;
    int8_t active_low;  // 1: on = low
} board_audio_ctl_t;

typedef struct {
    int8_t i2c_port, sda, scl;  // an I2C bus already up on i2c_port is shared, not re-created
    int32_t i2c_hz;
    int8_t i2s_port, mclk, bclk, ws, dout, din;  // -1: unused (dout without a speaker, din without a mic)
    int8_t mic, mic_addr, mic_gain, mic_slot;    // BA_CODEC_*; gain: the codec's own step, -1 = its default; slot 0 left, 1 right
    int8_t spk, spk_addr;                        // BA_CODEC_ES8311 or NONE
    board_audio_ctl_t amp, power;                // amp: on only while playing; power: on from init
} board_audio_config_t;

typedef struct {
    const char *name;  // an AtomVM atom string: the length byte, then the name
    board_audio_config_t cfg;
} board_audio_preset_t;

extern const board_audio_preset_t board_audio_presets[];
extern const int board_audio_n_presets;

// The configuration in force (the firmware's default preset until board_audio_configure).
const board_audio_config_t *board_audio_config(void);

// 0 if `c` is usable on this chip, else the 1-based index of the first bad field in the
// order of the Erlang tuple (babytalk.erl, audio_to_tuple/1).
int board_audio_check(const board_audio_config_t *c);

// Release the pins and chips (if up) and adopt `c`; the next board_audio_init() brings the
// new board up. Same thread rules as board_audio.h (BabyTalk's audio task only).
void board_audio_configure(const board_audio_config_t *c);

#ifdef __cplusplus
}
#endif
