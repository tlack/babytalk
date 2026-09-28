// snt_matvec_q4 (snt_q4.h): the portable loop, and on the ESP32-P4 the vector unit
// (snt_q4_esp32p4.S) for aligned rows. Same float result either way: both sum each group's
// products as an exact integer before scaling it.
// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <string.h>

#include "snt_q4.h"

#define Q4_MAX_GROUPS 48                     // n16 up to 768 (heart: 704)
#define Q4_BATCH 8                           // rows per vector call (the dots live on the stack)

static void q4_rows_ref(const signed char *act, const unsigned char *w, float *out, int rows, int n16)
{
    int blocks = n16 >> 5, groups = n16 >> 4;
    for (int r = 0; r < rows; r++, w += n16) {
        const unsigned char *sb = w + snt_q4_scales_at(n16);
        float acc = 0.0f;
        for (int g = 0; g < groups; g++) {
            int32_t d = 0;
            for (int i = 0; i < 16; i++) {
                int x = g * 16 + i, u;
                if (g < 2 * blocks)
                    u = (g & 1) ? w[(g >> 1) * 16 + i] >> 4 : w[(g >> 1) * 16 + i] & 15;
                else
                    u = w[blocks * 16 + i] & 15;
                d += act[x] * (u - 8);
            }
            float sc;
            memcpy(&sc, sb + 4 * g, 4);
            acc += sc * (float)d;
        }
        out[r] = acc;
    }
}

#if SANOTTS_P4_SIMD
#include "esp_log.h"
#include "snt_port.h"

// dots[r * groups + g]: XACC after group g of row r (a running int32 sum of act . (weight + 8))
void snt_p4_q4_dots(const int8_t *act, const uint8_t *w, int32_t *dots, int rows, int n16, const uint8_t *mask);
extern int64_t g_snt_macs_simd, g_snt_macs_scalar, g_snt_calls_simd, g_snt_calls_scalar;
static const uint8_t k_nibble_mask[16] __attribute__((aligned(16))) = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};

static void q4_rows_p4(const signed char *act, const unsigned char *w, float *out, int rows, int n16);

// Before the first use: the vector path against the portable one on random rows (with and
// without the odd last group), exactly. On a mismatch everything takes the portable loop.
static int q4_p4_ok(void)
{
    static int ok = -1;
    if (ok >= 0) return ok;
    static signed char act[80] __attribute__((aligned(16)));
    static unsigned char w[5 * 80] __attribute__((aligned(16)));
    uint32_t x = 12345;
    ok = 1;
    for (int n16 = 32; n16 <= 80 && ok; n16 += 16) {
        for (int i = 0; i < n16; i++) act[i] = (signed char)((x = x * 1103515245u + 12345u) >> 24);
        for (int r = 0; r < 5; r++) {
            unsigned char *row = w + r * n16;
            for (int i = 0; i < snt_q4_scales_at(n16); i++) row[i] = (unsigned char)((x = x * 1103515245u + 12345u) >> 24);
            if (n16 & 16)                    // the last group's high nibbles are unused
                for (int i = 0; i < 16; i++) row[16 * (n16 >> 5) + i] &= 15;
            for (int g = 0; g < n16 / 16; g++) {
                float sc = (float)((x = x * 1103515245u + 12345u) >> 20) / 4096.0f - 128.0f;
                memcpy(row + snt_q4_scales_at(n16) + 4 * g, &sc, 4);
            }
        }
        float a[5], b[5];
        q4_rows_ref(act, w, a, 5, n16);
        q4_rows_p4(act, w, b, 5, n16);
        ok = memcmp(a, b, sizeof a) == 0;
    }
    if (!ok) ESP_LOGE("snt_q4", "vector 4-bit kernel failed its self-check: using the portable one");
    return ok;
}

void snt_matvec_q4(const signed char *act, const signed char *w, float *out, int rows, int n16)
{
    int groups = n16 >> 4;
    if (rows <= 0) return;
    if ((n16 & 15) || n16 < 32 || groups > Q4_MAX_GROUPS || (((uintptr_t)act | (uintptr_t)w) & 15)
        || !snt_weights_resident(act) || !snt_weights_resident(w) || !q4_p4_ok()) {
        g_snt_calls_scalar++;
        g_snt_macs_scalar += (int64_t)rows * n16;
        q4_rows_ref(act, (const unsigned char *)w, out, rows, n16);
        return;
    }
    g_snt_calls_simd++;
    g_snt_macs_simd += (int64_t)rows * n16;
    q4_rows_p4(act, (const unsigned char *)w, out, rows, n16);
}

static void q4_rows_p4(const signed char *act, const unsigned char *w, float *out, int rows, int n16)
{
    int groups = n16 >> 4;
    int32_t off[Q4_MAX_GROUPS];              // 8 * each group's activation sum: the nibbles' +8
    for (int g = 0; g < groups; g++) {
        int32_t s = 0;
        for (int i = 0; i < 16; i++) s += act[g * 16 + i];
        off[g] = 8 * s;
    }
    int32_t dots[Q4_BATCH * Q4_MAX_GROUPS];
    const unsigned char *wr = w;
    int sat = snt_q4_scales_at(n16);
    for (int r0 = 0; r0 < rows; r0 += Q4_BATCH) {
        int n = rows - r0 < Q4_BATCH ? rows - r0 : Q4_BATCH;
        snt_p4_q4_dots(act, wr, dots, n, n16, k_nibble_mask);
        for (int r = 0; r < n; r++, wr += n16) {
            const float *sc = (const float *)(wr + sat);
            const int32_t *d = dots + r * groups;
            int32_t prev = 0;
            float acc = 0.0f;
            for (int g = 0; g < groups; g++) {
                acc += sc[g] * (float)(d[g] - prev - off[g]);
                prev = d[g];
            }
            out[r0 + r] = acc;
        }
    }
}
#else
void snt_matvec_q4(const signed char *act, const signed char *w, float *out, int rows, int n16)
{
    q4_rows_ref(act, (const unsigned char *)w, out, rows, n16);
}
#endif
