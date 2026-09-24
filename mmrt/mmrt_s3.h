// mmrt ESP32-S3 PIE kernels, drop-in replacements for the mmrt_ref.h ops (same
// arguments, bit-exact results). Buffers must be 16-byte aligned; channel counts
// multiples of 16.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 0 = read 1x1 weights in place (e.g. from flash); 1 = stage them in internal SRAM (default).
extern int mmrt_s3_stage;
// 1 or 2: split conv kernels across both cores (worker task pinned to core 0).
extern int mmrt_s3_cores;

// Allocate the second SRAM staging buffer (32KB internal) and start the core-0 worker.
// Call once at startup, before the app takes internal RAM for other things: done lazily
// otherwise, and if that allocation fails later, 1x1 convs quietly run on one core.
// Returns 0 when both are ready.
int mmrt_s3_init(void);
// Free the heap staging buffers (all of them with MMRT_S3_HEAP_BUFFERS, which builds that
// share internal RAM with other engines define). mmrt_s3_init() brings them back.
void mmrt_s3_deinit(void);
// MMRT_S3_HEAP_BUFFERS builds: take the staging buffers (MMRT_S3_BUFFER_BYTES, one block,
// 16-aligned) from `get` and hand them back through `put` in mmrt_s3_deinit().
#define MMRT_S3_BUFFER_BYTES (2 * 32 * 1024 + 2 * 16 * 256)
void mmrt_s3_set_buffer_provider(void *(*get)(size_t bytes), void (*put)(void));

// Diagnostic: cycles per VSMULAS (16 MACs) of the 1x1 row kernel on the calling core.
float mmrt_s3_c1_bench(int groups, int C, int frames, int where);
// One CB4 weight group (mmrt.h) -> [C][16] int8; mmrt_wload uses it on the S3.
void mmrt_s3_cb4_group(int8_t *dst, const uint8_t *src, int C);
// Diagnostic: CB4 decode, asm vs portable C: differing bytes, cycles per weight of each.
int mmrt_s3_cb4_bench(int C, float *asm_cyc, float *ref_cyc);

// w: in format wfmt (mmrt_wfmt_t); staged/decoded into SRAM as int8.
void mmrt_s3_conv1x1(const int8_t *x, int T_in, int C, const int8_t *w, int wfmt, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out);

void mmrt_s3_dwconv(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                    int shift, int relu, int8_t *y, int T_out);

// Fused block tail: y = relu?(sat8(round(a*s >> shift) + r)); s (a [C] vector, shift >= 1)
// and r ([T][C]) may be NULL. Bit-exact with mmrt_mul_bcast_ref -> mmrt_add_ref (equal
// exponents) -> ReLU.
void mmrt_s3_tail(const int8_t *a, const int8_t *s, const int8_t *r, int T, int C, int shift, int relu, int8_t *y);

// Same as mmrt_mean_ref; column sums in QACC (T * 128 must stay below 2^19).
void mmrt_s3_mean(const int8_t *x, int T, int C, int e_in, int e_out, int8_t *y);

// Fused depthwise -> 1x1 (the depthwise output stays in SRAM tiles). Returns 0 if the
// shapes don't fit (then run the two ops separately). Same results as the two kernels.
extern int mmrt_s3_fuse;
int mmrt_s3_dw_pw(const int8_t *x, int T_in, int C, const int8_t *dw_w, int K, int stride, int pad, int dw_shift,
                  int dw_relu, const int8_t *pw_w, int pw_wfmt, const int32_t *bias, int N, int pw_shift, int pw_relu, int8_t *y,
                  int T);

// Weight streaming for short inputs (see mmrt_s3.c): the executor lists its 1x1 ops in
// execution order and brackets the run with begin/end; begin returns 0 when streaming
// isn't used (long input, one core, ...), and the kernels then run as usual.
typedef struct {
    const int8_t *w;
    int wfmt;
    int C, N;
} mmrt_s3_stream_op_t;
extern int mmrt_s3_stream_max_T;  // input frames; 0 disables
int mmrt_s3_stream_begin(const mmrt_s3_stream_op_t *ops, int n_ops, int T_in);
void mmrt_s3_stream_end(void);

#ifdef __cplusplus
}
#endif
