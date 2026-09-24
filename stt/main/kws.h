// CTC keyword spotting on the STT model's logits: any wake phrase, no training.
// Portable C (also built on the host and checked against export/kws.py).
//
// A phrase is given as one or more spellings (letters only; spaces/punctuation are
// ignored, so "tomato face" == "tomatoface"). Each spelling is split into every
// sequence of vocab pieces (longest first, capped); word boundaries don't matter
// because a piece scores as the better of its word-initial and "##" forms. The score is
// a "keyword anywhere" CTC Viterbi over emissions logit - max logit per frame (the
// softmax normalizer cancels): a log-likelihood ratio in logit units, 0 = the phrase is
// exactly the model's best reading of those frames.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_MAX_SEQS 512
#define KWS_MAX_LEN 40  // pieces per sequence

typedef struct {
    int n_seqs;
    uint8_t len[KWS_MAX_SEQS];
    uint16_t piece[KWS_MAX_SEQS][KWS_MAX_LEN];  // index into the piece table
} kws_phrase_t;

// Build the piece table from the model vocabulary (once).
void kws_init(const char *const *vocab, int vocab_n);

// Add every segmentation of `spelling` (at most `cap`) to `p`. Returns how many.
int kws_add_spelling(kws_phrase_t *p, const char *spelling, int cap);

// Best score over all sequences of `p` on int8 logits [T][stride] (first vocab_n+1
// columns; blank = vocab_n), scaled by 2^out_exp to logit units. *start/*end: the
// matched frame span of the best sequence (may be NULL).
float kws_score(const kws_phrase_t *p, const int8_t *logits, int T, int stride, int out_exp, int *start, int *end);

#ifdef __cplusplus
}
#endif
