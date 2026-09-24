// mmrt executor: walks the op list, sizes each tensor for the input length, frees
// intermediates at their last use. Kernels: the reference ops (mmrt_ref.c); the S3
// build swaps in PIE kernels op by op, each checked against these.
#include "mmrt.h"

#include <stdlib.h>
#include <string.h>

#include "mmrt_ref.h"

mmrt_trace_fn mmrt_trace;

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

    for (int i = 0; i < n_ops; i++) {
        const mmrt_op_t *op = &m->ops[i];
        const int8_t *x = m->buf[op->in0];
        const int Tx = m->T[op->in0], Cx = m->tensors[op->in0].channels, Cy = m->tensors[op->out].channels;
        int Ty = Tx;
        if (op->kind == MMRT_DWCONV || op->kind == MMRT_CONV1X1) Ty = conv_out_len(Tx, op->K, op->stride, op->pad);
        if (op->kind == MMRT_MEAN) Ty = 1;
        int8_t *y = (int8_t *)alloc((size_t)Ty * Cy);
        if (!y) {
            free(last);
            return NULL;
        }
        const int8_t *w = (const int8_t *)(m->blob + op->w_off);
        switch (op->kind) {
        case MMRT_DWCONV:
            mmrt_dwconv_ref(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
            break;
        case MMRT_CONV1X1:
            mmrt_conv1x1_ref(x, Tx, Cx, w, op->b_off != 0xffffffffu ? (const int32_t *)(m->blob + op->b_off) : NULL,
                             Cy, op->stride, op->shift, op->relu, y, Ty);
            break;
        case MMRT_MEAN:
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
    free(last);
    *T_out = m->T[h->output];
    return m->buf[h->output];
}
