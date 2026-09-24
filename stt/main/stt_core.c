#include "stt_core.h"

#include <math.h>
#include <string.h>

#include "citrinet_tables.h"

#define PREEMPH 0.97f
#define LOG_GUARD 5.9604644775390625e-08f  // 2^-24
#define HALF_FFT (MEL_N_FFT / 2)
#define WIN_OFF ((MEL_N_FFT - MEL_WIN) / 2)  // torch.stft centres the window in n_fft

int stt_num_frames(int n_samples) { return n_samples / MEL_HOP + 1; }

size_t stt_scratch_floats(void) { return 2 * MEL_N_FFT + MEL_N_FFT; }

// In-place iterative radix-2 complex FFT, n a power of two. re/im separate.
static void fft(float *re, float *im, int n, const float *tw_re, const float *tw_im)
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
        int step = n / len;
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

void stt_features(const int16_t *pcm, int n, float *out, float *scratch)
{
    float *re = scratch, *im = scratch + MEL_N_FFT, *pw = scratch + 2 * MEL_N_FFT;
    static float tw_re[HALF_FFT], tw_im[HALF_FFT];
    static int tw_ready;
    if (!tw_ready) {
        for (int k = 0; k < HALF_FFT; k++) {
            double a = -2.0 * M_PI * k / MEL_N_FFT;
            tw_re[k] = (float)cos(a);
            tw_im[k] = (float)sin(a);
        }
        tw_ready = 1;
    }

    const int T = stt_num_frames(n);
    const float s = 1.0f / 32768.0f;
    for (int t = 0; t < T; t++) {
        // Frame t spans padded positions [t*hop, t*hop + n_fft) = samples
        // [t*hop - n_fft/2, ...): zero outside the clip (center=True, constant pad).
        int base = t * MEL_HOP - HALF_FFT;
        for (int k = 0; k < MEL_N_FFT; k++) {
            float v = 0.0f;
            int i = base + k;
            if (k >= WIN_OFF && k < WIN_OFF + MEL_WIN && i >= 0 && i < n) {
                float x = pcm[i] * s;
                if (i > 0) x -= PREEMPH * (pcm[i - 1] * s);  // pre-emphasis (y[0] = x[0])
                v = x * MEL_WINDOW[k - WIN_OFF];
            }
            re[k] = v;
            im[k] = 0.0f;
        }
        fft(re, im, MEL_N_FFT, tw_re, tw_im);
        for (int b = 0; b < MEL_BINS; b++) pw[b] = re[b] * re[b] + im[b] * im[b];
        float *o = out + (size_t)t * MEL_N;
        for (int m = 0; m < MEL_N; m++) {
            const mel_row_t *r = &MEL_ROWS[m];
            float acc = 0.0f;
            for (int j = 0; j < r->len; j++) acc += MEL_WEIGHTS[r->off + j] * pw[r->start + j];
            o[m] = logf(acc + LOG_GUARD);
        }
    }

    // Per-feature normalization over time, unbiased std + 1e-5.
    for (int m = 0; m < MEL_N; m++) {
        double sum = 0.0;
        for (int t = 0; t < T; t++) sum += out[(size_t)t * MEL_N + m];
        double mean = sum / T, sq = 0.0;
        for (int t = 0; t < T; t++) {
            double d = out[(size_t)t * MEL_N + m] - mean;
            sq += d * d;
        }
        float std = (float)sqrt(sq / (T > 1 ? T - 1 : 1)) + 1e-5f;
        for (int t = 0; t < T; t++) {
            float *p = &out[(size_t)t * MEL_N + m];
            *p = (float)((*p - mean) / std);
        }
    }
}

void stt_fill_quant(const float *feats, int T, int16_t *dst, int exponent)
{
    const float scale = ldexpf(1.0f, -exponent);
    const int unit = T + STT_FILL_GAP;
    for (int t = 0; t < STT_WIN_FRAMES; t++) {
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
