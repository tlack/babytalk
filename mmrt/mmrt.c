// mmrt executor: walks the op list, sizes each tensor for the input length, frees
// intermediates at their last use. Kernels: the reference ops (mmrt_ref.c); the S3
// build swaps in PIE kernels op by op, each checked against these.
#include "mmrt.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mmrt_ref.h"
#ifdef MMRT_S3
#include "mmrt_s3.h"
#else
#include "mmrt_port.h"
#endif
#ifdef MMRT_P4
// The P4's vector unit can't load from its low-power RTC RAM, which ESP-IDF counts as internal
// heap: buffers there take the portable path, and the kept weight rows go to PSRAM.
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#define SIMD_REACH(p) (!esp_ptr_in_rtc_dram_fast(p) && !esp_ptr_in_rtc_slow(p))
#define ROWS_ALLOC(n) heap_caps_aligned_alloc(16, (n), MALLOC_CAP_SPIRAM)
#else
#define SIMD_REACH(p) 1
#define ROWS_ALLOC(n) aligned_alloc(16, (n))
#endif

mmrt_trace_fn mmrt_trace;
int mmrt_use_ref;

#define NONE16 0xffff

// the 1x1 conv without the S3's kernels: the portable one, or the reference when asked for
#ifdef MMRT_S3
#define CONV1X1 mmrt_conv1x1_ref
#else
#define CONV1X1(...) (mmrt_use_ref ? mmrt_conv1x1_ref(__VA_ARGS__) : mmrt_conv1x1_port(__VA_ARGS__))
#endif

int mmrt_open(mmrt_model_t *m, const void *image, size_t size)
{
    memset(m, 0, sizeof(*m));
    const mmrt_header_t *h = (const mmrt_header_t *)image;
    if (size < sizeof(*h) || h->magic != MMRT_MAGIC || h->version < 1 || h->version > MMRT_VERSION) return -1;
    if (h->blob_off + h->blob_size > size) return -2;
    m->hdr = h;
    m->tensors = (const mmrt_tensor_t *)((const uint8_t *)image + h->tensors_off);
    m->ops = (const mmrt_op_t *)((const uint8_t *)image + h->ops_off);
    m->blob = (const uint8_t *)image + h->blob_off;
    m->T = (int *)calloc(h->n_tensors, sizeof(int));
    m->buf = (int8_t **)calloc(h->n_tensors, sizeof(int8_t *));
    return m->T && m->buf ? 0 : -3;
}

size_t mmrt_wgroup_bytes(int C, int wfmt)
{
    return wfmt == MMRT_W_CB4 ? 256 + (size_t)C * 8 : (size_t)C * 16;
}

void mmrt_wload(int8_t *dst, const int8_t *w, int wfmt, int C, int g0, int ng)
{
    const size_t gb = mmrt_wgroup_bytes(C, wfmt);
    const uint8_t *src = (const uint8_t *)w + (size_t)g0 * gb;
    if (wfmt != MMRT_W_CB4) {
        memcpy(dst, src, (size_t)ng * gb);
        return;
    }
#ifdef MMRT_S3
    if (!mmrt_use_ref) {
        for (int g = 0; g < ng; g++, src += gb, dst += (size_t)C * 16) mmrt_s3_cb4_group(dst, src, C);
        return;
    }
#endif
    for (int g = 0; g < ng; g++, src += gb) {
        int8_t tab[256];  // [lane][level], local: the source may be slow memory
        memcpy(tab, src, 256);
        const uint8_t *idx = src + 256;
        for (int c = 0; c < C; c++, idx += 8, dst += 16)
            for (int j = 0; j < 8; j++) {
                const uint8_t b = idx[j];
                dst[2 * j] = tab[(2 * j) * 16 + (b & 15)];
                dst[2 * j + 1] = tab[(2 * j + 1) * 16 + (b >> 4)];
            }
    }
}

static const int8_t *op_weights(const mmrt_model_t *m, int i)
{
    if (m->wcache && m->wcache[i]) return m->wcache[i];
    return (const int8_t *)(m->blob + m->ops[i].w_off);
}

size_t mmrt_cache_weights(mmrt_model_t *m, size_t budget, mmrt_alloc_fn alloc, mmrt_free_fn release)
{
    const uint32_t n = m->hdr->n_ops;
    if (m->wcache) {
        for (uint32_t i = 0; i < n; i++)
            if (m->wcache[i]) release((void *)m->wcache[i]);
        free(m->wcache);
        m->wcache = NULL;
        m->wcache_bytes = 0;
    }
    if (!budget) return 0;
    m->wcache = (const int8_t **)calloc(n, sizeof(*m->wcache));
    if (!m->wcache) return 0;
    // depthwise weights first: small (C*K per op, ~120KB in all for Citrinet-256) and read
    // straight from the image for every frame, so while 1x1 weights stream out of flash
    // they compete for the same bus; then 1x1 weights in op order
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t i = 0; i < n; i++) {
            const mmrt_op_t *op = &m->ops[i];
            if (op->kind != (pass ? MMRT_CONV1X1 : MMRT_DWCONV)) continue;
            const int C = m->tensors[op->in0].channels;
            size_t bytes = pass ? mmrt_wgroup_bytes(C, op->wfmt) * (m->tensors[op->out].channels / 16) : (size_t)C * op->K;
            if (m->wcache_bytes + bytes > budget) goto done;
            int8_t *c = (int8_t *)alloc(bytes);
            if (!c) goto done;
            memcpy(c, m->blob + op->w_off, bytes);
            m->wcache[i] = c;
            m->wcache_bytes += bytes;
        }
done:
    return m->wcache_bytes;
}

#ifdef MMRT_ROWDOT
// op i's 1x1 weights as rows ([N][C], 16-byte aligned), decoded and repacked on first use and
// kept (C x N bytes each: ~9.8 MB for all of Citrinet-256 -- a board with PSRAM to spare); NULL
// if there's no memory for them (the op then runs on the portable kernel)
static const int8_t *op_rows(mmrt_model_t *m, int i, const int8_t *w, int wfmt, int C, int N,
                             mmrt_alloc_fn alloc, mmrt_free_fn release)
{
    if (!m->wrows) m->wrows = (int8_t **)calloc(m->hdr->n_ops, sizeof(*m->wrows));
    if (!m->wrows) return NULL;
    if (m->wrows[i]) return m->wrows[i];
    int8_t *r = (int8_t *)ROWS_ALLOC((size_t)C * N), *wd = NULL;
    if (!r) return NULL;
    if (wfmt != MMRT_W_INT8) {
        if (!(wd = (int8_t *)alloc((size_t)C * N))) {
            free(r);
            return NULL;
        }
        mmrt_wload(wd, w, wfmt, C, 0, N / 16);
    }
    mmrt_repack_nc(r, wd ? wd : w, C, N);
    if (wd) release(wd);
    m->wrows[i] = r;
    m->wrows_bytes += (size_t)C * N;
    return r;
}
#endif

void mmrt_close(mmrt_model_t *m, mmrt_free_fn release)
{
    if (m->wrows) {
        for (uint32_t i = 0; i < m->hdr->n_ops; i++) free(m->wrows[i]);
        free(m->wrows);
        m->wrows = NULL;
        m->wrows_bytes = 0;
    }
    mmrt_cache_weights(m, 0, NULL, release);
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
            sops[ns].w = op_weights(m, i);
            sops[ns].wfmt = op->wfmt;
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
#ifdef MMRT_S3
        // depthwise -> 1x1 whose only consumer it is: fused, the intermediate stays in SRAM
        if (!mmrt_use_ref && op->kind == MMRT_DWCONV && i + 1 < n_ops && m->ops[i + 1].kind == MMRT_CONV1X1 &&
            m->ops[i + 1].in0 == op->out && last[op->out] == i + 1 && m->ops[i + 1].stride == 1) {
            const mmrt_op_t *pw = &m->ops[i + 1];
            const int Tx = m->T[op->in0], Cx = m->tensors[op->in0].channels, N = m->tensors[pw->out].channels;
            const int T = conv_out_len(Tx, op->K, op->stride, op->pad);
            int8_t *y = (int8_t *)alloc((size_t)T * N);
            if (!y) {
                STREAM_END();
                free(last);
                return NULL;
            }
            const int32_t *b = pw->b_off != 0xffffffffu ? (const int32_t *)(m->blob + pw->b_off) : NULL;
            if (mmrt_s3_dw_pw(m->buf[op->in0], Tx, Cx, op_weights(m, i), op->K, op->stride,
                              op->pad, op->shift, op->relu, op_weights(m, i + 1), pw->wfmt, b, N, pw->shift,
                              pw->relu, y, T)) {
                m->buf[pw->out] = y;
                m->T[pw->out] = T;
                if (mmrt_trace) mmrt_trace(i + 1, pw, y, T, N);
                if (last[op->in0] <= i + 1 && op->in0 != h->input && m->buf[op->in0]) {
                    release(m->buf[op->in0]);
                    m->buf[op->in0] = NULL;
                }
                i += 1;
                continue;
            }
            release(y);
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
        const int8_t *w = op->kind == MMRT_CONV1X1 || op->kind == MMRT_DWCONV ? op_weights(m, i)
                                                                              : (const int8_t *)(m->blob + op->w_off);
        switch (op->kind) {
        case MMRT_DWCONV:
#ifdef MMRT_S3
            if (!mmrt_use_ref)
                mmrt_s3_dwconv(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
            else
                mmrt_dwconv_ref(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
#else
            if (mmrt_use_ref || Cx % 16)
                mmrt_dwconv_ref(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
#ifdef MMRT_P4  // the vector kernel, unless it declines (then the portable one)
            else if (!SIMD_REACH(x) || !SIMD_REACH(y)
                     || mmrt_dwconv_p4(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty))
                mmrt_dwconv_port(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
#else
            else
                mmrt_dwconv_port(x, Tx, Cx, w, op->K, op->stride, op->pad, op->shift, op->relu, y, Ty);
#endif
#endif
            break;
        case MMRT_CONV1X1:
        {
            const int32_t *b = op->b_off != 0xffffffffu ? (const int32_t *)(m->blob + op->b_off) : NULL;
#ifdef MMRT_S3
            if (!mmrt_use_ref)
                mmrt_s3_conv1x1(x, Tx, Cx, w, op->wfmt, b, Cy, op->stride, op->shift, op->relu, y, Ty);
            else
#endif
#ifdef MMRT_ROWDOT
            // a vector dot product (the P4's): each output's weights as one contiguous row
#ifdef MMRT_P4
            static int dot_ok = -1;
            if (dot_ok < 0) dot_ok = mmrt_dot_p4_check();
#else
            const int dot_ok = 1;
#endif
            const int8_t *wr = !mmrt_use_ref && dot_ok && Cx % 16 == 0 && !((uintptr_t)x & 15) && Cy % 16 == 0 && SIMD_REACH(x)
                                   ? op_rows(m, i, w, op->wfmt, Cx, Cy, alloc, release) : NULL;
            if (wr)
                mmrt_conv1x1_nc(x, Cx, wr, b, Cy, op->stride, op->shift, op->relu, y, Ty);
            else
#endif
            if (op->wfmt == MMRT_W_INT8) {
                CONV1X1(x, Tx, Cx, w, b, Cy, op->stride, op->shift, op->relu, y, Ty);
            } else {  // reference and portable paths: decode the whole op first
                int8_t *wd = (int8_t *)alloc((size_t)Cx * Cy);
                if (!wd) {
                    release(y);
                    STREAM_END();
                    free(last);
                    return NULL;
                }
                mmrt_wload(wd, w, op->wfmt, Cx, 0, Cy / 16);
                CONV1X1(x, Tx, Cx, wd, b, Cy, op->stride, op->shift, op->relu, y, Ty);
                release(wd);
            }
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
