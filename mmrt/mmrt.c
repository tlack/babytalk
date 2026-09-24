// mmrt executor: walks the op list, sizes each tensor for the input length, frees
// intermediates at their last use. Kernels: the reference ops (mmrt_ref.c); the S3
// build swaps in PIE kernels op by op, each checked against these.
#include "mmrt.h"

#include <stdlib.h>
#include <string.h>

#include "mmrt_ref.h"
#ifdef MMRT_S3
#include "mmrt_s3.h"
#endif

mmrt_trace_fn mmrt_trace;
int mmrt_use_ref;

#define NONE16 0xffff

int mmrt_open(mmrt_model_t *m, const void *image, size_t size)
{
    memset(m, 0, sizeof(*m));
    const mmrt_header_t *h = (const mmrt_header_t *)image;
    if (size < sizeof(*h) || h->magic != MMRT_MAGIC || h->version != MMRT_VERSION) return -1;
    if (h->blob_off + h->blob_size > size) return -2;
    m->hdr = h;
    m->tensors = (const mmrt_tensor_t *)((const uint8_t *)image + h->tensors_off);
    m->ops = (const mmrt_op_t *)((const uint8_t *)image + h->ops_off);
    m->blob = (const uint8_t *)image + h->blob_off;
    m->T = (int *)calloc(h->n_tensors, sizeof(int));
    m->buf = (int8_t **)calloc(h->n_tensors, sizeof(int8_t *));
    return m->T && m->buf ? 0 : -3;
}

void mmrt_close(mmrt_model_t *m, mmrt_free_fn release)
{
    if (m->buf) {
        for (uint32_t i = 0; i < m->hdr->n_tensors; i++)
            if (m->buf[i] && i != m->hdr->input) release(m->buf[i]);
    }
    free(m->T);
    free(m->buf);
    memset(m, 0, sizeof(*m));
}

static int conv_out_len(int T, int K, int stride, int pad) { return (T + 2 * pad - K) / stride + 1; }

#ifdef MMRT_S3
// A LUT op that is exactly ReLU at an unchanged exponent (vector max instead of a lookup).
static int is_relu_lut(const mmrt_model_t *m, const mmrt_op_t *op)
{
    if (op->kind != MMRT_LUT || m->tensors[op->in0].exp != op->e_out) return 0;
    const int8_t *lut = (const int8_t *)(m->blob + op->w_off);
    for (int v = -128; v < 128; v++)
        if (lut[v + 128] != (v > 0 ? v : 0)) return 0;
    return 1;
}

// Fused block tail starting at op i: [MUL] -> [ADD] -> ReLU LUT (at least two of them),
// each intermediate used only by the next op, ADD at equal exponents. Returns the number
// of ops covered (0 = not fusable) and the kernel's operands.
static int match_tail(const mmrt_model_t *m, const int *last, int i, int n_ops, const int8_t **a,
                      const int8_t **s, const int8_t **r, int *shift, uint16_t *out)
{
    const mmrt_op_t *op = &m->ops[i];
    int j = i;
    *s = NULL;
    *r = NULL;
    *shift = 0;
    *a = m->buf[op->in0];
    if (op->kind == MMRT_MUL) {
        *s = m->buf[op->in1];
        *shift = op->e_out - op->e_a - op->e_b;
        if (*shift < 1) return 0;
        j++;
    }
    if (j < n_ops && m->ops[j].kind == MMRT_ADD) {
        const mmrt_op_t *ad = &m->ops[j];
        uint16_t prev = j > i ? m->ops[i].out : NONE16;
        if (prev != NONE16) {
            if (ad->in0 != prev && ad->in1 != prev) return 0;
            if (last[prev] != j) return 0;
            *r = m->buf[ad->in0 == prev ? ad->in1 : ad->in0];
        } else {
            *r = m->buf[ad->in1];
        }
        if (ad->e_a != ad->e_b || ad->e_a != ad->e_out) return 0;
        j++;
    }
    if (j == i || j >= n_ops) return 0;
    const mmrt_op_t *lu = &m->ops[j];
    if (!is_relu_lut(m, lu) || lu->in0 != m->ops[j - 1].out || last[m->ops[j - 1].out] != j) return 0;
    *out = lu->out;
    return j - i + 1;
}
#endif

const int8_t *mmrt_run(mmrt_model_t *m, const int8_t *input, int T_in, int *T_out,
                       mmrt_alloc_fn alloc, mmrt_free_fn release)
{
    const mmrt_header_t *h = m->hdr;
    const int n_ops = (int)h->n_ops;
    // last use of every tensor, so intermediates can be freed early
    int *last = (int *)calloc(h->n_tensors, sizeof(int));
    for (int i = 0; i < n_ops; i++) {
        last[m->ops[i].in0] = i;
        if (m->ops[i].in1 != NONE16) last[m->ops[i].in1] = i;
    }
    last[h->output] = n_ops;  // survives the run
    for (uint32_t i = 0; i < h->n_tensors; i++) {  // drop anything left from a previous run
        if (m->buf[i] && i != h->input) release(m->buf[i]);
        m->buf[i] = NULL;
    }
    m->buf[h->input] = (int8_t *)input;
    m->T[h->input] = T_in;
#ifdef MMRT_S3
    // short inputs: stream 1x1 weights out of flash on core 0 while core 1 computes
    mmrt_s3_stream_op_t *sops = NULL;
    int streaming = 0;
    if (!mmrt_use_ref) {
        sops = (mmrt_s3_stream_op_t *)malloc(sizeof(*sops) * n_ops);
        int ns = 0;
        for (int i = 0; sops && i < n_ops; i++) {
            const mmrt_op_t *op = &m->ops[i];
            if (op->kind != MMRT_CONV1X1) continue;
            sops[ns].w = (const int8_t *)(m->blob + op->w_off);
            sops[ns].C = m->tensors[op->in0].channels;
            sops[ns].N = m->tensors[op->out].channels;
            ns++;
        }
        streaming = sops && mmrt_s3_stream_begin(sops, ns, T_in);
    }
#define STREAM_END() do { if (streaming) mmrt_s3_stream_end(); free(sops); } while (0)
#else
#define STREAM_END() do { } while (0)
#endif

    for (int i = 0; i < n_ops; i++) {
        const mmrt_op_t *op = &m->ops[i];
#ifdef MMRT_S3
        if (!mmrt_use_ref && (op->kind == MMRT_MUL || op->kind == MMRT_ADD)) {
            const int8_t *a, *s, *r;
            int shift;
            uint16_t out;
            int span = match_tail(m, last, i, n_ops, &a, &s, &r, &shift, &out);
            if (span) {
                const int T = m->T[op->in0], C = m->tensors[op->in0].channels;
                int8_t *y = (int8_t *)alloc((size_t)T * C);
                if (!y) {
                    STREAM_END();
                    free(last);
                    return NULL;
                }
                mmrt_s3_tail(a, s, r, T, C, shift, 1, y);
                m->buf[out] = y;
                m->T[out] = T;
                if (mmrt_trace) mmrt_trace(i + span - 1, &m->ops[i + span - 1], y, T, C);
                for (int j = i; j < i + span; j++) {  // free inputs whose last use was in the span
                    uint16_t ins[2] = {m->ops[j].in0, m->ops[j].in1};
                    for (int k = 0; k < 2; k++) {
                        if (ins[k] != NONE16 && last[ins[k]] >= i && last[ins[k]] < i + span &&
                            ins[k] != h->input && m->buf[ins[k]]) {
                            release(m->buf[ins[k]]);
                            m->buf[ins[k]] = NULL;
                        }
                    }
                }
                i += span - 1;
                continue;
            }
        }
#endif
        const int8_t *x = m->buf[op->in0];
        const int Tx = m->T[op->in0], Cx = m->tensors[op->in0].channels, Cy = m->tensors[op->out].channels;
        int Ty = Tx;
        if (op->kind == MMRT_DWCONV || op->kind == MMRT_CONV1X1) Ty = conv_out_len(Tx, op->K, op->stride, op->pad);
        if (op->kind == MMRT_MEAN) Ty = 1;
        int8_t *y = (int8_t *)alloc((size_t)Ty * Cy);
        if (!y) {
            STREAM_END();
            free(last);
            return NULL;
        }
        const int8_t *w = (const int8_t *)(m->blob + op->w_off);
        switch (op->kind) {
        case MMRT_DWCONV:
#ifdef MMRT_S3
            if (!mmrt_use_ref)
                mmrt_s3_dwconv(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
            else
#endif
                mmrt_dwconv_ref(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
            break;
        case MMRT_CONV1X1:
        {
            const int32_t *b = op->b_off != 0xffffffffu ? (const int32_t *)(m->blob + op->b_off) : NULL;
#ifdef MMRT_S3
            if (!mmrt_use_ref)
                mmrt_s3_conv1x1(x, Tx, Cx, w, b, Cy, op->stride, op->shift, op->relu, y, Ty);
            else
#endif
                mmrt_conv1x1_ref(x, Tx, Cx, w, b, Cy, op->stride, op->shift, op->relu, y, Ty);
        }
            break;
        case MMRT_MEAN:
#ifdef MMRT_S3
            if (!mmrt_use_ref && Tx * 128 < (1 << 19))
                mmrt_s3_mean(x, Tx, Cx, op->e_a, op->e_out, y);
            else
#endif
                mmrt_mean_ref(x, Tx, Cx, op->e_a, op->e_out, y);
            break;
        case MMRT_LUT:
            mmrt_lut_ref(x, Tx * Cx, w, y);
            break;
        case MMRT_MUL:
            mmrt_mul_bcast_ref(x, Tx, Cx, m->buf[op->in1], op->e_a, op->e_b, op->e_out, y);
            break;
        case MMRT_ADD:
            mmrt_add_ref(x, m->buf[op->in1], Tx * Cx, op->e_a, op->e_b, op->e_out, y);
            break;
        default:
            STREAM_END();
            free(last);
            return NULL;
        }
        m->buf[op->out] = y;
        m->T[op->out] = Ty;
        if (mmrt_trace) mmrt_trace(i, op, y, Ty, Cy);
        // free inputs whose last use was this op
        uint16_t ins[2] = {op->in0, op->in1};
        for (int k = 0; k < 2; k++) {
            if (ins[k] != NONE16 && last[ins[k]] == i && ins[k] != h->input && m->buf[ins[k]]) {
                release(m->buf[ins[k]]);
                m->buf[ins[k]] = NULL;
            }
        }
    }
    STREAM_END();
    free(last);
    *T_out = m->T[h->output];
    return m->buf[h->output];
}
