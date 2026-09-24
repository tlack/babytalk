#include "mmrt_ref.h"

int32_t mmrt_ref_max_acc;

static inline int8_t sat8(int64_t v) { return v > 127 ? 127 : (v < -128 ? -128 : (int8_t)v); }

// v * 2^-s, rounded half up (s > 0), or v * 2^-s exactly (s <= 0).
static inline int64_t rshift_rnd(int64_t v, int s)
{
    return s > 0 ? (v + ((int64_t)1 << (s - 1))) >> s : v * ((int64_t)1 << -s);
}

static inline void track(int32_t acc)
{
    int32_t a = acc < 0 ? -acc : acc;
    if (a > mmrt_ref_max_acc) mmrt_ref_max_acc = a;
}

void mmrt_dwconv_ref(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                     int shift, int relu, int8_t *y, int T_out)
{
    for (int t = 0; t < T_out; t++) {
        for (int c = 0; c < C; c++) {
            const int8_t *wc = w + (c / 16) * K * 16 + (c % 16);
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int ti = t * stride + k - pad;
                if (ti >= 0 && ti < T_in) acc += x[ti * C + c] * wc[k * 16];
            }
            track(acc);
            int64_t v = rshift_rnd(acc, shift);
            y[t * C + c] = sat8(relu && v < 0 ? 0 : v);
        }
    }
}

void mmrt_conv1x1_ref(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                      int stride, int shift, int relu, int8_t *y, int T_out)
{
    (void)T_in;
    for (int t = 0; t < T_out; t++) {
        const int8_t *xt = x + (int64_t)t * stride * C;
        for (int n = 0; n < N; n++) {
            const int8_t *wn = w + (n / 16) * C * 16 + (n % 16);
            int32_t acc = bias ? bias[n] : 0;
            for (int c = 0; c < C; c++) acc += xt[c] * wn[c * 16];
            track(acc);
            int64_t v = rshift_rnd(acc, shift);
            y[t * N + n] = sat8(relu && v < 0 ? 0 : v);
        }
    }
}

// floor(a / b) for b > 0
static inline int64_t floordiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

void mmrt_mean_ref(const int8_t *x, int T, int C, int e_in, int e_out, int8_t *y)
{
    int k = e_in - e_out;  // value = sum * 2^k / T
    for (int c = 0; c < C; c++) {
        int64_t sum = 0;
        for (int t = 0; t < T; t++) sum += x[t * C + c];
        int64_t num = k >= 0 ? sum << k : sum, den = k >= 0 ? T : (int64_t)T << -k;
        y[c] = sat8(floordiv(2 * num + den, 2 * den));  // round half up
    }
}

void mmrt_lut_ref(const int8_t *x, int n, const int8_t *lut, int8_t *y)
{
    for (int i = 0; i < n; i++) y[i] = lut[x[i] + 128];
}

void mmrt_mul_bcast_ref(const int8_t *x, int T, int C, const int8_t *s, int e_x, int e_s, int e_out, int8_t *y)
{
    int sh = e_out - e_x - e_s;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) y[t * C + c] = sat8(rshift_rnd((int32_t)x[t * C + c] * s[c], sh));
}

void mmrt_add_ref(const int8_t *a, const int8_t *b, int n, int e_a, int e_b, int e_out, int8_t *y)
{
    int e0 = e_a < e_b ? e_a : e_b;
    for (int i = 0; i < n; i++) {
        int64_t v = ((int64_t)a[i] << (e_a - e0)) + ((int64_t)b[i] << (e_b - e0));
        y[i] = sat8(rshift_rnd(v, e_out - e0));
    }
}
