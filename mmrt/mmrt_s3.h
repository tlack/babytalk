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
// 1 or 2: split conv kernels across both cores (worker task pinned to core 0).
extern int mmrt_s3_cores;

void mmrt_s3_conv1x1(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out);

void mmrt_s3_dwconv(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                    int shift, int relu, int8_t *y, int T_out);

// Fused block tail: y = relu?(sat8(round(a*s >> shift) + r)); s (a [C] vector, shift >= 1)
// and r ([T][C]) may be NULL. Bit-exact with mmrt_mul_bcast_ref -> mmrt_add_ref (equal
// exponents) -> ReLU.
void mmrt_s3_tail(const int8_t *a, const int8_t *s, const int8_t *r, int T, int C, int shift, int relu, int8_t *y);

// Same as mmrt_mean_ref; column sums in QACC (T * 128 must stay below 2^19).
void mmrt_s3_mean(const int8_t *x, int T, int C, int e_in, int e_out, int8_t *y);

// Weight streaming for short inputs (see mmrt_s3.c): the executor lists its 1x1 ops in
// execution order and brackets the run with begin/end; begin returns 0 when streaming
// isn't used (long input, one core, ...), and the kernels then run as usual.
typedef struct {
    const int8_t *w;
    int C, N;
} mmrt_s3_stream_op_t;
extern int mmrt_s3_stream_max_T;  // input frames; 0 disables
int mmrt_s3_stream_begin(const mmrt_s3_stream_op_t *ops, int n_ops, int T_in);
void mmrt_s3_stream_end(void);

#ifdef __cplusplus
}
#endif
