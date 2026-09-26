// The board's microphone and speaker, for BabyTalk's AtomVM NIFs: the only hardware-specific
// part of BabyTalk. board_audio.c implements it for the Waveshare ESP32-S3-CAM:
//   ES7210   4-mic ADC (mic 1 used)          I2C 0x40 + I2S in
//   ES8311   mono codec, DAC side            I2C 0x18 + I2S out
//   NS4150B  3 W class-D speaker amp         enable = expander P4
//   CH32V003 MCU as 8-bit IO expander        I2C 0x24: P6 = audio rail, P4 = amp enable
// Another board implements these functions in its own component and selects
// CONFIG_BABYTALK_BOARD_CUSTOM (see Kconfig).
//
// The contract: 16 kHz mono signed 16-bit capture; mono playback at any rate the board can
// do (8..48 kHz). The mic and the speaker may share one I2S port, so the NIFs never overlap
// them: rx_start .. rx_stop and play take turns. All calls come from one FreeRTOS task
// (BabyTalk's audio task), so implementations need not be thread-safe -- but if the board's
// I2C bus is shared with other drivers, use that bus's handle rather than creating one.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bring the audio chips up (rails on, amp off). Once; later calls return 0 at once.
// mic_gain: the board's own PGA step, or -1 for the board's default. Negative on failure
// (the Waveshare's: -1 bus, -2 expander, -3 ES7210, -4 ES8311).
int board_audio_init(int mic_gain);

// Capture at 16 kHz; discards the first 200 ms (ADC/filter settling).
int board_audio_rx_start(void);
int board_audio_rx_read(int16_t *mono, int frames);  // blocking; mic 1
void board_audio_rx_stop(void);                      // frees the port

// Play mono PCM at `rate` Hz through the speaker, blocking. volume 0..100 (DAC; 75 = 0 dB).
// The amp is on only while playing (after the stream starts: no turn-on pop).
int board_audio_play(const int16_t *mono, int n, int rate, int volume);
// DMA underruns during the last play (each one a short gap of silence).
int board_audio_underruns(void);

#ifdef __cplusplus
}
#endif
