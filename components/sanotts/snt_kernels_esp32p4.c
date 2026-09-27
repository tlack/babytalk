// BabyTalk's ESP32-P4 kernels for sanoTTS: the three functions sanoTTS's snt_port.h asks a
// port for (snt_dot_s8, snt_matvec_s8, snt_weights_resident), on the P4's vector unit
// (snt_matvec_esp32p4.S). They take the place of sanoTTS's scalar ones in snt_kernels_ref.c
// (prepare.sh guards those with !SANOTTS_P4_SIMD), as its own snt_kernels_esp32s3.c does on the
// S3. Same exact integer results as the reference kernels.
//
// Unlike the S3's, the P4's vector unit reads flash and PSRAM correctly, so every address is
// "resident" except the P4's low-power RTC RAM (which it can't read at all). The one contract
// left is 16-byte alignment and a length that's a multiple of 16; anything else takes the scalar
// loop, and the counters (g_snt_macs_simd / _scalar) say how much did.
//
// SPDX-License-Identifier: MIT
#include "sdkconfig.h"

#if SANOTTS_P4_SIMD

#include <stdint.h>

#include "esp_memory_utils.h"
#include "snt_port.h"

void snt_p4_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out, int rows, int chunks);

int64_t g_snt_macs_simd, g_snt_macs_scalar;
int64_t g_snt_calls_simd, g_snt_calls_scalar;

void snt_res_reset(void)
{
    g_snt_macs_simd = g_snt_macs_scalar = 0;
    g_snt_calls_simd = g_snt_calls_scalar = 0;
}

int snt_weights_resident(const void *p)
{
    return !esp_ptr_in_rtc_dram_fast(p) && !esp_ptr_in_rtc_slow(p);
}

static inline int simd_ok(const void *act, const void *w, int len)
{
    return len > 0 && (len & 15) == 0 && !(((uintptr_t)act | (uintptr_t)w) & 15) && snt_weights_resident(act)
           && snt_weights_resident(w);
}

void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out, int rows, int len)
{
    if (rows > 0 && simd_ok(act, w, len)) {
        g_snt_calls_simd++;
        g_snt_macs_simd += (int64_t)rows * len;
        snt_p4_matvec_s8(act, w, out, rows, len >> 4);
        return;
    }
    g_snt_calls_scalar++;
    g_snt_macs_scalar += (int64_t)rows * len;
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int i = 0; i < len; i++) acc += (int32_t)act[i] * (int32_t)w[(long)r * len + i];
        out[r] = acc;
    }
}

int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len)
{
    int32_t acc = 0;
    if (simd_ok(a, b, len)) {
        g_snt_calls_simd++;
        g_snt_macs_simd += len;
        snt_p4_matvec_s8(a, b, &acc, 1, len >> 4);
        return acc;
    }
    g_snt_calls_scalar++;
    g_snt_macs_scalar += len;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

#endif
