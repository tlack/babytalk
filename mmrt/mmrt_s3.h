// mmrt ESP32-S3 PIE kernels, drop-in replacements for the mmrt_ref.h ops (same
// arguments, bit-exact results). Buffers must be 16-byte aligned; channel counts
// multiples of 16.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 0 = read 1x1 weights in place (e.g. from flash); 1 = stage them in internal SRAM (default).
extern int mmrt_s3_stage;

void mmrt_s3_conv1x1(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out);

#ifdef __cplusplus
}
#endif
