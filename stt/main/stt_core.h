// Portable Citrinet front end + CTC decoder. Plain C, no ESP-IDF: the same file
// builds on the host (stt/host_test) to be checked against the Python pipeline.
#pragma once

#include <stddef.h>
#include <stdint.h>

#define STT_WIN_FRAMES 1600  // static model input: 16 s of 10 ms frames
#define STT_SUB 8            // encoder time downsampling
#define STT_FILL_GAP 16      // zero frames between tiled copies ("tile16")

#ifdef __cplusplus
extern "C" {
#endif

// Frames produced for n samples (centered STFT): n / hop + 1.
int stt_num_frames(int n_samples);

// 16 kHz mono PCM -> per-feature-normalized log-mel, time-major [T][80], T =
// stt_num_frames(n). Matches NeMo's inference preprocessor (preemph 0.97, centered
// zero-padded STFT, |X|^2, slaney mel, log(x + 2^-24), mean / (unbiased std + 1e-5)).
// scratch: >= stt_scratch_floats() floats.
void stt_features(const int16_t *pcm, int n, float *out, float *scratch);
size_t stt_scratch_floats(void);

// Fill the static [1600][80] input: the clip repeated with STT_FILL_GAP zero frames
// between copies (keeps squeeze-excite's window mean close to the clip's mean), then
// quantized to int16 with scale 2^-exponent (round half up, saturating). T <= 1600.
void stt_fill_quant(const float *feats, int T, int16_t *dst, int exponent);

// Output frames that belong to T valid input frames (three stride-2 convs).
int stt_out_frames(int T);

// Greedy CTC over int16 logits [frames][vocab_n + 1] (blank = vocab_n), WordPiece
// detokenization ("##" continues a word, apostrophes glue to both neighbours).
// Returns the text length written to out (NUL-terminated, truncated to cap).
int stt_ctc_greedy(const int16_t *logits, int frames, int vocab_n, const char *const *vocab,
                   char *out, size_t cap);

#ifdef __cplusplus
}
#endif
