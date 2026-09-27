// mmrt portable kernels: plain C, bit-exact with the reference ops (mmrt_ref.c), for chips
// without the S3's PIE unit (the ESP32-P4, for one). The reference ops are written to be read:
// the 1x1 conv walks each output's weights with a 16-byte stride for every frame. These use the
// same packed layouts ([C/16][K][16] depthwise, [N/16][C][16] 1x1) the way they're laid out:
// sixteen outputs at a time from contiguous weights, a group's weights reused across every
// frame while they're in cache, and zero inputs (after a ReLU, most of them) skipped.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// as mmrt_dwconv_ref / mmrt_conv1x1_ref; C (depthwise) and N (1x1) multiples of 16
void mmrt_dwconv_port(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                      int shift, int relu, int8_t *y, int T_out);
void mmrt_conv1x1_port(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                       int stride, int shift, int relu, int8_t *y, int T_out);

// One CB4 weight group (mmrt.h: 256-byte table, then C x 8 index bytes) -> [C][16] int8. With
// scratch (MMRT_CB4_SCRATCH bytes, 2-byte aligned) and C >= 1024 it builds per-lane-pair tables
// of every index byte's two weights first, then decodes a pair of weights per lookup.
#define MMRT_CB4_SCRATCH (8 * 256 * 2)
void mmrt_cb4_group_port(int8_t *dst, const uint8_t *src, int C, uint16_t *scratch);

// The 1x1 conv as dot products over rows: the weights repacked [N][C] (each output's C weights
// contiguous), so each output is one dot product of C int8s -- which a vector unit does 16 at a
// time (the ESP32-P4's esp.vmulas.s8.xacc: mmrt_p4.S; plain C elsewhere). x's rows and w_nc must
// be 16-byte aligned; C a multiple of 16.
void mmrt_repack_nc(int8_t *w_nc, const int8_t *w, int C, int N);
void mmrt_conv1x1_nc(const int8_t *x, int C, const int8_t *w_nc, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out);
// sum of a[i] * b[i] for i < 16 * n16 (n16 >= 1; a, b 16-byte aligned)
int32_t mmrt_dot_s8(const int8_t *a, const int8_t *b, int n16);

#ifdef MMRT_P4
// the depthwise conv on the P4's vector unit (bit-exact with mmrt_dwconv_port, which it checks
// itself against before first use): 0, or -1 if it can't take this call (misaligned, shift < 0,
// or it failed its check) -- then use the portable kernel
int mmrt_dwconv_p4(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad, int shift,
                   int relu, int8_t *y, int T_out);
// 1 if the vector dot product matches C's on a range of lengths (checked once, before use)
int mmrt_dot_p4_check(void);
// 1 if the P4's 1x1 group kernel matches the same arithmetic in C (checked once, before use)
int mmrt_p4_matmul_ok(void);
#endif

#if defined(MMRT_P4) || defined(MMRT_C1)
// 1x1 convs / matmuls a 16-output group at a time from the packed [C][16] layout (the P4's
// vector unit; MMRT_C1 on the PC: the same arithmetic in C). x, w, y 16-byte aligned, C a
// multiple of 16, 0 <= shift <= 13, and every sum under 2^19 (20-bit accumulator lanes).
// A bias goes in as a block of 16 x 16 weights per group (mmrt_c1_bias_block: -1 if a bias is
// too large for it, |bias| over ~122,000).
int mmrt_c1_bias_block(int8_t blk[256], const int32_t *bias16);
void mmrt_conv1x1_c1(const int8_t *x, int C, const int8_t *w, const int8_t *bias_blks, int N, int stride,
                     int shift, int relu, int8_t *y, int T_out);
void mmrt_c1_matmul_group(const int8_t *x, int T, int C, const int8_t *w, int shift, int relu, int8_t *y, int y_stride);
#endif

#ifdef __cplusplus
}
#endif
