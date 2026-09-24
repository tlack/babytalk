#include "stt_core.h"

#include <math.h>
#include <string.h>

#include "citrinet_tables.h"

#ifdef STT_USE_DSP  // ESP32-S3: esp-dsp's hand-optimized FFT (device build only)
#include "dsps_fft2r.h"
#endif

#define PREEMPH 0.97f
#define LOG_GUARD 5.9604644775390625e-08f  // 2^-24
#define HALF_FFT (MEL_N_FFT / 2)
#define WIN_OFF ((MEL_N_FFT - MEL_WIN) / 2)  // torch.stft centres the window in n_fft

int stt_num_frames(int n_samples) { return n_samples / MEL_HOP + 1; }

size_t stt_scratch_floats(void) { return 2 * HALF_FFT + HALF_FFT + 1; }

#ifndef STT_USE_DSP
// In-place iterative radix-2 complex FFT of size n (a power of two); twiddles
// tw[k * tw_step] = exp(-2 pi i k / n).
static void fft(float *re, float *im, int n, const float *tw_re, const float *tw_im, int tw_step)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int step = (n / len) * tw_step;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                float wr = tw_re[k * step], wi = tw_im[k * step];
                int a = i + k, b = a + len / 2;
                float xr = re[b] * wr - im[b] * wi;
                float xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
        }
    }
}
#endif

void stt_features(const int16_t *pcm, int n, float *out, float *scratch)
{
    // 512-point real FFT as a 256-point complex FFT: z[k] = x[2k] + i x[2k+1], then split
    // the even/odd halves apart (standard real-FFT post-processing).
    float *zr = scratch, *zi = scratch + HALF_FFT, *pw = scratch + 2 * HALF_FFT;
    static float tw_re[HALF_FFT], tw_im[HALF_FFT];  // exp(-2 pi i k / 512), k < 256
    static int tw_ready;
    if (!tw_ready) {
        for (int k = 0; k < HALF_FFT; k++) {
            double a = -2.0 * M_PI * k / MEL_N_FFT;
            tw_re[k] = (float)cos(a);
            tw_im[k] = (float)sin(a);
        }
        tw_ready = 1;
    }
    double sum[MEL_N] = {0}, sumsq[MEL_N] = {0};

    const int T = stt_num_frames(n);
    const float s = 1.0f / 32768.0f;
    float frame[MEL_N_FFT] __attribute__((aligned(16)));
#ifdef STT_USE_DSP
    static int dsp_ready;
    if (!dsp_ready) dsp_ready = dsps_fft2r_init_fc32(NULL, HALF_FFT) == 0;
#endif
    for (int t = 0; t < T; t++) {
        // Frame t spans padded positions [t*hop, t*hop + n_fft) = samples
        // [t*hop - n_fft/2, ...): zero outside the clip (center=True, constant pad).
        // Only the MEL_WIN window samples in the middle are nonzero.
        int base = t * MEL_HOP - HALF_FFT + WIN_OFF;
        memset(frame, 0, sizeof(frame));
        for (int k = 0; k < MEL_WIN; k++) {
            int i = base + k;
            if (i < 0 || i >= n) continue;
            float x = pcm[i] * s;
            if (i > 0) x -= PREEMPH * (pcm[i - 1] * s);  // pre-emphasis (y[0] = x[0])
            frame[WIN_OFF + k] = x * MEL_WINDOW[k];
        }
#ifdef STT_USE_DSP
        // frame[] already is the interleaved complex z[k] = x[2k] + i x[2k+1]
        dsps_fft2r_fc32(frame, HALF_FFT);
        dsps_bit_rev_fc32(frame, HALF_FFT);
        for (int k = 0; k < HALF_FFT; k++) {
            zr[k] = frame[2 * k];
            zi[k] = frame[2 * k + 1];
        }
#else
        for (int k = 0; k < HALF_FFT; k++) {
            zr[k] = frame[2 * k];
            zi[k] = frame[2 * k + 1];
        }
        fft(zr, zi, HALF_FFT, tw_re, tw_im, 2);
#endif
        for (int k = 0; k <= HALF_FFT; k++) {
            int a = k & (HALF_FFT - 1), b = (HALF_FFT - k) & (HALF_FFT - 1);
            float er = 0.5f * (zr[a] + zr[b]), ei = 0.5f * (zi[a] - zi[b]);   // even part
            float orr = 0.5f * (zi[a] + zi[b]), oi = 0.5f * (zr[b] - zr[a]);  // odd part
            float wr = k < HALF_FFT ? tw_re[k] : -1.0f, wi = k < HALF_FFT ? tw_im[k] : 0.0f;
            float xr = er + wr * orr - wi * oi, xi = ei + wr * oi + wi * orr;
            pw[k] = xr * xr + xi * xi;
        }
        float *o = out + (size_t)t * MEL_N;
        for (int m = 0; m < MEL_N; m++) {
            const mel_row_t *r = &MEL_ROWS[m];
            float acc = 0.0f;
            for (int j = 0; j < r->len; j++) acc += MEL_WEIGHTS[r->off + j] * pw[r->start + j];
            float v = logf(acc + LOG_GUARD);
            o[m] = v;
            sum[m] += v;
            sumsq[m] += (double)v * v;
        }
    }

    // Per-feature normalization over time, unbiased std + 1e-5 (one row-wise pass).
    float mean[MEL_N], std[MEL_N];
    for (int m = 0; m < MEL_N; m++) {
        double mu = sum[m] / T, var = (sumsq[m] - sum[m] * mu) / (T > 1 ? T - 1 : 1);
        mean[m] = (float)mu;
        std[m] = (float)sqrt(var > 0 ? var : 0) + 1e-5f;
    }
    for (int t = 0; t < T; t++) {
        float *o = out + (size_t)t * MEL_N;
        for (int m = 0; m < MEL_N; m++) o[m] = (o[m] - mean[m]) / std[m];
    }
}

void stt_fill_quant(const float *feats, int T, int16_t *dst, int W, int exponent)
{
    const float scale = ldexpf(1.0f, -exponent);
    const int unit = W == T ? T : T + STT_FILL_GAP;
    for (int t = 0; t < W; t++) {
        int u = t % unit;
        int16_t *d = dst + (size_t)t * MEL_N;
        if (u >= T) {
            memset(d, 0, MEL_N * sizeof(int16_t));
            continue;
        }
        const float *f = feats + (size_t)u * MEL_N;
        for (int m = 0; m < MEL_N; m++) {
            float q = floorf(f[m] * scale + 0.5f);
            d[m] = q > 32767.0f ? 32767 : (q < -32768.0f ? -32768 : (int16_t)q);
        }
    }
}

int stt_out_frames(int T)
{
    for (int i = 0; i < 3; i++) T = (T - 1) / 2 + 1;
    return T;
}

int stt_ctc_greedy(const int16_t *logits, int frames, int vocab_n, const char *const *vocab,
                   char *out, size_t cap)
{
    size_t len = 0;
    int prev = -1, glue = 0;
    out[0] = 0;
    for (int t = 0; t < frames; t++) {
        const int16_t *row = logits + (size_t)t * (vocab_n + 1);
        int best = 0;
        for (int c = 1; c <= vocab_n; c++) {
            if (row[c] > row[best]) best = c;
        }
        int emit = best != prev && best != vocab_n;
        prev = best;
        if (!emit) continue;
        const char *tok = vocab[best];
        size_t tl = strlen(tok);
        if (tok[0] == '[' && tok[tl - 1] == ']') continue;  // [PAD], [UNK], ...
        const char *piece = tok;
        int space = len > 0 && !glue;
        if (tl > 2 && tok[0] == '#' && tok[1] == '#') {
            piece = tok + 2;
            space = 0;
        }
        glue = strcmp(tok, "'") == 0;
        if (glue) space = 0;
        size_t pl = strlen(piece);
        if (len + space + pl + 1 > cap) break;
        if (space) out[len++] = ' ';
        memcpy(out + len, piece, pl);
        len += pl;
        out[len] = 0;
    }
    return (int)len;
}
