#include "kws.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Piece table: distinct piece strings (without "##"), each with up to 2 token ids.
#define MAX_PIECES 256
static char s_piece[MAX_PIECES][8];
static int16_t s_ids[MAX_PIECES][2];
static int s_npieces, s_blank;

void kws_init(const char *const *vocab, int vocab_n)
{
    s_npieces = 0;
    s_blank = vocab_n;
    for (int i = 0; i < vocab_n; i++) {
        const char *t = vocab[i];
        if (t[0] == '[' || !strcmp(t, "'")) continue;
        const char *core = (t[0] == '#' && t[1] == '#') ? t + 2 : t;
        int k = 0;
        while (k < s_npieces && strcmp(s_piece[k], core)) k++;
        if (k == s_npieces) {
            if (s_npieces == MAX_PIECES || strlen(core) >= sizeof(s_piece[0])) continue;
            strcpy(s_piece[k], core);
            s_ids[k][0] = s_ids[k][1] = -1;
            s_npieces++;
        }
        s_ids[k][s_ids[k][0] < 0 ? 0 : 1] = (int16_t)i;
    }
}

static int find_piece(const char *s, int n)
{
    for (int k = 0; k < s_npieces; k++)
        if ((int)strlen(s_piece[k]) == n && !memcmp(s_piece[k], s, n)) return k;
    return -1;
}

// Depth-first segmentation, longest pieces first (same order as kws.py).
typedef struct {
    kws_phrase_t *p;
    const char *letters;
    int n, cap, added;
    uint16_t acc[KWS_MAX_LEN];
} seg_ctx_t;

static void seg(seg_ctx_t *c, int pos, int depth)
{
    if (c->added >= c->cap || c->p->n_seqs >= KWS_MAX_SEQS || depth > KWS_MAX_LEN) return;
    if (pos == c->n) {
        int i = c->p->n_seqs;
        for (int j = 0; j < i; j++)  // skip duplicates
            if (c->p->len[j] == depth && !memcmp(c->p->piece[j], c->acc, depth * sizeof(uint16_t))) return;
        c->p->len[i] = (uint8_t)depth;
        memcpy(c->p->piece[i], c->acc, depth * sizeof(uint16_t));
        c->p->n_seqs++;
        c->added++;
        return;
    }
    if (depth == KWS_MAX_LEN) return;
    for (int end = c->n; end > pos; end--) {
        int k = find_piece(c->letters + pos, end - pos);
        if (k < 0) continue;
        c->acc[depth] = (uint16_t)k;
        seg(c, end, depth + 1);
    }
}

int kws_add_spelling(kws_phrase_t *p, const char *spelling, int cap)
{
    char letters[128];
    int n = 0;
    for (const char *s = spelling; *s && n < (int)sizeof(letters) - 1; s++) {
        char ch = *s >= 'A' && *s <= 'Z' ? *s + 32 : *s;
        if (ch >= 'a' && ch <= 'z') letters[n++] = ch;
    }
    letters[n] = 0;
    seg_ctx_t c = {p, letters, n, cap, 0, {0}};
    seg(&c, 0, 0);
    return c.added;
}

#define NEG (-1e30f)

static float score_seq(const float *em, int T, int V, const uint16_t *seq, int L, int *st, int *en)
{
    // states: 0 blank, 1 piece0, 2 blank, ..., 2L-1 piece L-1, 2L blank
    const int S = 2 * L + 1;
    float alpha[2 * KWS_MAX_LEN + 1], nxt[2 * KWS_MAX_LEN + 1];
    int start[2 * KWS_MAX_LEN + 1], nstart[2 * KWS_MAX_LEN + 1];
    for (int s = 0; s < S; s++) { alpha[s] = NEG; start[s] = 0; }
    float best = NEG;
    for (int t = 0; t < T; t++) {
        const float *e = em + (size_t)t * V;  // V: pieces (by table index) then blank
        for (int s = 0; s < S; s++) {
            float v = alpha[s];
            int from = start[s];
            if (s >= 1 && alpha[s - 1] > v) { v = alpha[s - 1]; from = start[s - 1]; }
            if (s >= 2 && (s & 1) && seq[(s - 1) / 2] != seq[(s - 3) / 2] && alpha[s - 2] > v) {
                v = alpha[s - 2];
                from = start[s - 2];
            }
            if (s <= 1 && v < 0.0f) { v = 0.0f; from = t; }  // free start
            float emit = (s & 1) ? e[seq[(s - 1) / 2]] : e[V - 1];
            nxt[s] = v + emit;
            nstart[s] = from;
        }
        memcpy(alpha, nxt, sizeof(float) * S);
        memcpy(start, nstart, sizeof(int) * S);
        for (int s = S - 2; s < S; s++)
            if (alpha[s] > best) { best = alpha[s]; *st = start[s]; *en = t; }
    }
    return best;
}

float kws_score(const kws_phrase_t *p, const int8_t *logits, int T, int stride, int out_exp, int *start, int *end)
{
    // Per-frame emissions for every piece (better of its two token forms) and blank,
    // relative to the frame's best token: (q - q_max) * 2^out_exp.
    const int V = s_npieces + 1;
    float *em = (float *)malloc(sizeof(float) * (size_t)T * V);
    if (!em) return NEG;
    const float scale = ldexpf(1.0f, out_exp);
    for (int t = 0; t < T; t++) {
        const int8_t *q = logits + (size_t)t * stride;
        int qmax = -128;
        for (int k = 0; k <= s_blank; k++)
            if (q[k] > qmax) qmax = q[k];
        float *e = em + (size_t)t * V;
        for (int k = 0; k < s_npieces; k++) {
            int v = q[s_ids[k][0]];
            if (s_ids[k][1] >= 0 && q[s_ids[k][1]] > v) v = q[s_ids[k][1]];
            e[k] = (v - qmax) * scale;
        }
        e[V - 1] = (q[s_blank] - qmax) * scale;
    }
    float best = NEG;
    int bs = 0, be = 0;
    for (int i = 0; i < p->n_seqs; i++) {
        int st = 0, en = 0;
        float s = score_seq(em, T, V, p->piece[i], p->len[i], &st, &en);
        if (s > best) { best = s; bs = st; be = en; }
    }
    free(em);
    if (start) *start = bs;
    if (end) *end = be;
    return best;
}
