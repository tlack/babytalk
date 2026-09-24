// mmrt: a small int8 inference runtime for Citrinet-style CTC speech models on the
// ESP32-S3. The model is a straight-line list of ops over time-major int8 tensors,
// exported from ESP-PPQ's deployed graph by export/mmrt_export.py.
//
// File layout (little endian, every section 16-byte aligned):
//   mmrt_header_t
//   mmrt_tensor_t[n_tensors]
//   mmrt_op_t[n_ops]
//   blob: weights / biases / LUTs, referenced by byte offsets from blob start
// Weights are read in place (e.g. from memory-mapped flash); only activations are
// allocated.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MMRT_MAGIC 0x54524d4du  // "MMRT"
#define MMRT_VERSION 1

typedef enum {
    MMRT_DWCONV = 1,   // depthwise conv, weights [C/16][K][16], no bias
    MMRT_CONV1X1 = 2,  // 1x1 conv, weights [N/16][C][16], int32 bias[N]
    MMRT_MEAN = 3,     // mean over time -> [1][C]
    MMRT_LUT = 4,      // y = lut[x + 128]
    MMRT_MUL = 5,      // y[t][c] = x[t][c] * s[c]   (in1 is the [1][C] vector)
    MMRT_ADD = 6,      // y = a + b
} mmrt_kind_t;

typedef struct {
    uint32_t magic, version;
    uint32_t n_tensors, n_ops;
    uint32_t input, output;   // tensor ids
    uint32_t tensors_off, ops_off, blob_off, blob_size;  // byte offsets from file start
    uint32_t out_valid;       // real output channels (e.g. 257; the tensor may be padded)
    uint32_t reserved[5];
} mmrt_header_t;

typedef struct {
    uint16_t channels;
    int8_t exp;
    uint8_t pad_;
} mmrt_tensor_t;

typedef struct {
    uint8_t kind;             // mmrt_kind_t
    uint8_t relu;
    uint8_t K, stride;
    int8_t pad, shift;        // conv: zero padding each side, requant right shift
    int8_t e_a, e_b;          // input exponents (mean/mul/add)
    int8_t e_out;
    uint8_t pad_[3];
    uint16_t in0, in1, out;   // tensor ids (in1 = 0xffff if unused)
    uint16_t pad2_;
    uint32_t w_off, b_off;    // blob offsets (0xffffffff if none); LUT uses w_off
} mmrt_op_t;

#ifdef __cplusplus
static_assert(sizeof(mmrt_header_t) == 64 && sizeof(mmrt_tensor_t) == 4 && sizeof(mmrt_op_t) == 28, "file layout");
#else
_Static_assert(sizeof(mmrt_header_t) == 64 && sizeof(mmrt_tensor_t) == 4 && sizeof(mmrt_op_t) == 28, "file layout");
#endif

typedef struct {
    const mmrt_header_t *hdr;
    const mmrt_tensor_t *tensors;
    const mmrt_op_t *ops;
    const uint8_t *blob;
    int *T;                   // frames per tensor for the planned input length
    int8_t **buf;             // activation buffers (NULL when not live)
    const int8_t **wcache;    // per op: faster copy of its weights (e.g. PSRAM), or NULL
    size_t wcache_bytes;
} mmrt_model_t;

// Allocator hooks: activations can be large (PSRAM on the S3).
typedef void *(*mmrt_alloc_fn)(size_t bytes);
typedef void (*mmrt_free_fn)(void *p);

// Bind to a model image (not copied). Returns 0 on success.
int mmrt_open(mmrt_model_t *m, const void *image, size_t size);

// Run on input [T_in][channels(input)] int8. On success returns the output tensor
// ([*T_out][channels(output)], owned by the model until the next run / mmrt_close) and
// frees every intermediate as soon as it is dead.
const int8_t *mmrt_run(mmrt_model_t *m, const int8_t *input, int T_in, int *T_out,
                       mmrt_alloc_fn alloc, mmrt_free_fn release);

void mmrt_close(mmrt_model_t *m, mmrt_free_fn release);

// Optional: copy depthwise, then 1x1-conv weights (in op order) into memory from `alloc` until `budget`
// bytes are used, e.g. PSRAM (~88 MB/s) when the image is in flash (~32 MB/s): every
// inference reads all weights once, so each cached MB saves ~20 ms per run on the S3 --
// at the cost of that much PSRAM. budget 0 releases the cache. Returns bytes cached.
size_t mmrt_cache_weights(mmrt_model_t *m, size_t budget, mmrt_alloc_fn alloc, mmrt_free_fn release);

// 1 = use the portable reference ops even where an optimized kernel exists (A/B checks).
extern int mmrt_use_ref;

// Per-op hook, called after each op (NULL = none): lets a test compare every tensor.
typedef void (*mmrt_trace_fn)(int op_index, const mmrt_op_t *op, const int8_t *out, int T, int C);
extern mmrt_trace_fn mmrt_trace;

#ifdef __cplusplus
}
#endif
