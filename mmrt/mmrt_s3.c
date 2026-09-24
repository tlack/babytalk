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
