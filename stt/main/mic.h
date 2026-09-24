// ES7210 mic capture (Waveshare S3 cam): 16 kHz, mic 1 as mono s16.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int mic_open(int gain);                       // once; gain 0..14 (~3dB/step). 0 on success
int mic_start(void);                          // enable capture, discard 200 ms settling
int mic_read_mono(int16_t *dst, int frames);  // blocking; 16 kHz frames of mic 1
void mic_stop(void);

#ifdef __cplusplus
}
#endif
