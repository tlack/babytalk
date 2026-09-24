// mmrt ESP32-S3 kernels: C drivers around the PIE assembly in mmrt_s3.S.
#include "mmrt_s3.h"

#include <string.h>

#include "esp_attr.h"
#include "esp_memory_utils.h"

typedef struct {
    const uint8_t *bias_q;
    int c16m1;
    int groups;
    int shift;
    int relu;
} mmrt_s3_c1_t;

void mmrt_s3_conv1x1_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_s3_c1_t *a);

typedef struct {
    int nk;
    int C;
    int groups;
    int K16;
    int shift;
    int relu;
} mmrt_s3_dw_t;

void mmrt_s3_dw_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_s3_dw_t *a);

typedef struct {
    int groups;
    int shift;
    int relu;
} mmrt_s3_tail_t;

void mmrt_s3_tail_row(int8_t *y, const int8_t *a, const int8_t *s, const int8_t *r, const mmrt_s3_tail_t *t);
void mmrt_s3_colsum16(const int8_t *x, int T, int C, uint8_t *qacc_out);

// Internal-SRAM staging: 1x1 weights are re-read for every output frame, so they are
// copied here (from flash) once per op instead of streaming through the cache per frame.
#define W_STAGE_BYTES (64 * 1024)
static DRAM_ATTR int8_t s_wstage[W_STAGE_BYTES] __attribute__((aligned(16)));
static DRAM_ATTR uint8_t s_bias_q[64 * 48] __attribute__((aligned(16)));  // up to 768 outputs

int mmrt_s3_stage = 1;

// int32 bias -> QACC load format: per 16-output group, two halves of 8 lanes x 20 bits
// packed little-endian (20 bytes), each half padded to 32 bytes.
static void pack_bias(const int32_t *bias, int N, uint8_t *dst)
{
    memset(dst, 0, (size_t)N / 16 * 64);
    for (int n = 0; n < N; n++) {
        uint8_t *half = dst + (n / 16) * 64 + ((n % 16) / 8) * 32;
        uint32_t v = (uint32_t)bias[n] & 0xfffff;
        int bit = (n % 8) * 20;
        for (int b = 0; b < 20; b++, bit++)
            if (v >> b & 1) half[bit / 8] |= (uint8_t)(1u << (bit % 8));
    }
}

void mmrt_s3_conv1x1(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out)
{
    (void)T_in;
    const int groups = N / 16;
    const size_t group_bytes = (size_t)C * 16;
    int chunk = groups;  // groups per SRAM-staged chunk
    if (mmrt_s3_stage && !esp_ptr_internal(w)) {
        chunk = (int)(W_STAGE_BYTES / group_bytes);
        if (chunk > groups) chunk = groups;
    }
    if (bias) pack_bias(bias, N, s_bias_q);
    for (int g0 = 0; g0 < groups; g0 += chunk) {
        int ng = groups - g0 < chunk ? groups - g0 : chunk;
        const int8_t *wg = w + (size_t)g0 * group_bytes;
        if (mmrt_s3_stage && !esp_ptr_internal(w)) {
            memcpy(s_wstage, wg, (size_t)ng * group_bytes);
            wg = s_wstage;
        }
        mmrt_s3_c1_t a = {bias ? s_bias_q + (size_t)g0 * 64 : NULL, C / 16 - 1, ng, shift, relu};
        for (int t = 0; t < T_out; t++)
            mmrt_s3_conv1x1_row(y + (size_t)t * N + g0 * 16, x + (size_t)t * stride * C, wg, &a);
    }
}

void mmrt_s3_dwconv(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                    int shift, int relu, int8_t *y, int T_out)
{
    mmrt_s3_dw_t a = {0, C, C / 16, K * 16, shift, relu};
    for (int t = 0; t < T_out; t++) {
        int start = t * stride - pad;  // input frame of tap 0
        int k0 = start < 0 ? -start : 0;
        int k1 = start + K > T_in ? T_in - start : K;
        a.nk = k1 - k0;
        mmrt_s3_dw_row(y + (size_t)t * C, x + (size_t)(start + k0) * C, w + k0 * 16, &a);
    }
}

void mmrt_s3_tail(const int8_t *a, const int8_t *s, const int8_t *r, int T, int C, int shift, int relu, int8_t *y)
{
    mmrt_s3_tail_t t = {C / 16, shift, relu};
    for (int i = 0; i < T; i++) {
        size_t off = (size_t)i * C;
        mmrt_s3_tail_row(y + off, a + off, s, r ? r + off : NULL, &t);
    }
}

// Unpack one 64-byte QACC dump (see pack_bias) into 16 sign-extended int32 lanes.
static void unpack_qacc(const uint8_t *q, int32_t *out)
{
    for (int n = 0; n < 16; n++) {
        const uint8_t *half = q + (n / 8) * 32;
        int bit = (n % 8) * 20;
        uint32_t v = 0;
        for (int b = 0; b < 20; b++, bit++) v |= (uint32_t)(half[bit / 8] >> (bit % 8) & 1) << b;
        out[n] = (int32_t)(v << 12) >> 12;
    }
}

static inline int64_t floordiv64(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

void mmrt_s3_mean(const int8_t *x, int T, int C, int e_in, int e_out, int8_t *y)
{
    static DRAM_ATTR uint8_t q[64] __attribute__((aligned(16)));
    int32_t sums[16];
    int k = e_in - e_out;  // value = sum * 2^k / T, rounded half up (as mmrt_mean_ref)
    for (int g = 0; g < C / 16; g++) {
        mmrt_s3_colsum16(x + g * 16, T, C, q);
        unpack_qacc(q, sums);
        for (int i = 0; i < 16; i++) {
            int64_t num = k >= 0 ? (int64_t)sums[i] << k : sums[i], den = k >= 0 ? T : (int64_t)T << -k;
            int64_t v = floordiv64(2 * num + den, 2 * den);
            y[g * 16 + i] = v > 127 ? 127 : (v < -128 ? -128 : (int8_t)v);
        }
    }
}
