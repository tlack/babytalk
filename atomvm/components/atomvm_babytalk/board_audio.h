// Audio on the Waveshare ESP32-S3-CAM, for BabyTalk's AtomVM NIFs: our own drivers for
//   ES7210   4-mic ADC (mic 1 used)          I2C 0x40 + I2S in
//   ES8311   mono codec, DAC side            I2C 0x18 + I2S out
//   NS4150B  3 W class-D speaker amp         enable = expander P4
//   CH32V003 MCU as 8-bit IO expander        I2C 0x24: P6 = audio rail, P4 = amp enable
// Both codecs share I2S port 0's clock pins (MCLK 10, BCLK 11, WS 12; in 13, out 14), so the
// port is either capturing or playing: rx_start/rx_stop and play take turns (the caller
// serializes them). MCLK = 256 x Fs, ESP32 is the I2S master, STEREO frames (MONO garbles
// this codec pair). Not thread-safe: call from one task.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// I2C bus, expander (rail on, amp off), both codecs. Once; later calls return 0 at once.
// Negative: -1 bus, -2 expander, -3 ES7210, -4 ES8311.
int board_audio_init(int mic_gain);  // gain 0..14, ~3 dB per step

// Capture at 16 kHz; discards the first 200 ms (ADC/filter settling).
int board_audio_rx_start(void);
int board_audio_rx_read(int16_t *mono, int frames);  // blocking; mic 1
void board_audio_rx_stop(void);                      // frees the port

// Play mono PCM at `rate` Hz through the speaker, blocking. volume 0..100 (DAC; 75 = 0 dB).
// The amp is on only while playing (after the stream starts: no turn-on pop).
int board_audio_play(const int16_t *mono, int n, int rate, int volume);

#ifdef __cplusplus
}
#endif
