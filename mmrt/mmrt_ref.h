// mmrt reference ops: portable C, the arithmetic spec of the runtime.
//
// Activations are int8, time-major [T][C]. Every op computes the exact result and rounds
// it half up at the output exponent, then saturates to int8 -- the rational form of what
// ESP-PPQ simulates. The PIE kernels (mmrt_s3.S) must match these bit for bit.
//
// Weight layouts are ESP-DL's (verified against the ONNX floats by export/mmrt_info.py):
//   depthwise: [C/16][K][16]       1x1 conv: [N/16][C][16]
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Largest |accumulator| seen by the conv refs since the last reset (the S3's QACC lanes
// are 20-bit for int8: |acc| >= 2^19 would wrap in hardware).
extern int32_t mmrt_ref_max_acc;

// y[t][c] = sat8(rnd(sum_k x[t*stride + k - pad][c] * w[c][k] >> shift)), zero padding.
void mmrt_dwconv_ref(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                     int shift, int relu, int8_t *y, int T_out);

// y[t][n] = act(sat8(rnd((sum_c x[t*stride][c] * w[n][c] + bias[n]) >> shift))).
void mmrt_conv1x1_ref(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                      int stride, int shift, int relu, int8_t *y, int T_out);

// y[c] = sat8(rnd(sum_t x[t][c] * 2^(e_in - e_out) / T)).
void mmrt_mean_ref(const int8_t *x, int T, int C, int e_in, int e_out, int8_t *y);

// y[i] = lut[x[i] + 128].
void mmrt_lut_ref(const int8_t *x, int n, const int8_t *lut, int8_t *y);

// y[t][c] = sat8(rnd(x[t][c] * s[c] * 2^(e_x + e_s - e_out))).
void mmrt_mul_bcast_ref(const int8_t *x, int T, int C, const int8_t *s, int e_x, int e_s, int e_out, int8_t *y);

// y[i] = sat8(rnd(a[i] * 2^e_a + b[i] * 2^e_b) / 2^e_out).
void mmrt_add_ref(const int8_t *a, const int8_t *b, int n, int e_a, int e_b, int e_out, int8_t *y);

#ifdef __cplusplus
}
#endif
