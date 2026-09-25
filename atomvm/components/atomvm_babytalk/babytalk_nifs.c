// BabyTalk NIFs for AtomVM: on-device speech to text, module `babytalk` (Erlang) /
// `BabyTalk` (Elixir, delegating). The Erlang docs are in atomvm/lib/babytalk/src/babytalk.erl.
//
//   babytalk:transcribe_nif(Pcm) -> {ok, Ref} | {error, busy | no_memory}
//       then the caller gets {babytalk, Ref, {ok, Text, Info} | {error, Code}}
//   babytalk:listen_nif(ChunkMs) -> {ok, Ref} | {error, busy}
//       then {babytalk_mic, Ref, Pcm16Mono} every ChunkMs (ES7210 mic 1, 16 kHz, gapless)
//       until stop_listening(), then {babytalk_mic, Ref, stopped}; or {babytalk_mic, Ref, {error, Code}}
//   babytalk:stop_listening() -> ok
//   babytalk:phrase(Spellings) -> {ok, Sequences} | {error, busy | no_memory}
//   babytalk:cache(Bytes) -> {ok, Cached} | {error, busy | Code}
//   babytalk:info() -> [{Key, Value}]
//   babytalk:heap_info() -> [{internal_free, B}, {internal_largest, B}, {psram_free, B}, {psram_largest, B}]
//
// Inference (0.3-2 s) runs on our own FreeRTOS task, never on a scheduler: AtomVM has no
// dirty NIFs. The model is the "model" flash partition (components/stt_engine).
#include <sdkconfig.h>
#ifdef CONFIG_AVM_ENABLE_BABYTALK_NIFS

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <atom.h>
#include <defaultatoms.h>
#include <globalcontext.h>
#include <interop.h>
#include <mailbox.h>
#include <memory.h>
#include <nifs.h>
#include <portnifloader.h>
#include <term.h>
#include <utils.h>

#include <esp_heap_caps.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include "mic.h"
#include "sram_pool.h"
#include "stt_engine.h"

#define TAG "babytalk"

// AtomVM atom strings: length byte, then the name
static const char *const A_BABYTALK = "\x08" "babytalk";
static const char *const A_BABYTALK_MIC = "\x0C" "babytalk_mic";
static const char *const A_STOPPED = "\x07" "stopped";
static const char *const A_BUSY = "\x04" "busy";
static const char *const A_NO_MEMORY = "\x09" "no_memory";
static const char *const A_SCORE = "\x05" "score";
static const char *const A_SPAN = "\x04" "span";
static const char *const A_FRAMES = "\x06" "frames";
static const char *const A_FE_MS = "\x05" "fe_ms";
static const char *const A_MODEL_MS = "\x08" "model_ms";
static const char *const A_DEC_MS = "\x06" "dec_ms";
static const char *const A_MODEL_BYTES = "\x0B" "model_bytes";
static const char *const A_CACHE_BYTES = "\x0B" "cache_bytes";
static const char *const A_OPS = "\x03" "ops";
static const char *const A_INT4_OPS = "\x08" "int4_ops";
static const char *const A_INTERNAL_FREE = "\x0D" "internal_free";
static const char *const A_INTERNAL_LARGEST = "\x10" "internal_largest";
static const char *const A_PSRAM_FREE = "\x0A" "psram_free";
static const char *const A_PSRAM_LARGEST = "\x0D" "psram_largest";

static term atom(GlobalContext *g, const char *a) { return globalcontext_make_atom(g, a); }

// ------------------------------------------------------------------------------ engine lock
// One owner of the engine at a time: a transcription job (from submit until the worker is
// done with the engine), or a phrase/cache call. Others get {error, busy}.
static atomic_int s_busy;

static bool lock_engine(void)
{
    int expected = 0;
    return atomic_compare_exchange_strong(&s_busy, &expected, 1);
}
static void unlock_engine(void) { atomic_store(&s_busy, 0); }

// ------------------------------------------------------------------------------ worker task
typedef struct {
    int16_t *pcm;  // PSRAM copy, freed by the worker
    int n;
    int32_t pid;
    uint64_t ref;
} job_t;

#define MIC_GAIN 14  // ES7210 PGA step (~3 dB each), as the field recordings

static GlobalContext *s_glb;
static QueueHandle_t s_jobs;

static term error_tuple(Heap *h, term reason)
{
    term t = term_alloc_tuple(2, h);
    term_put_tuple_element(t, 0, ERROR_ATOM);
    term_put_tuple_element(t, 1, reason);
    return t;
}

static term prop(Heap *h, term key, term val, term tail)
{
    term t = term_alloc_tuple(2, h);
    term_put_tuple_element(t, 0, key);
    term_put_tuple_element(t, 1, val);
    return term_list_prepend(t, tail, h);
}

// {babytalk, Ref, Result} to the job's owner; the message is copied, then the heap freed
static void send_msg(const job_t *j, Heap *heap, term result)
{
    GlobalContext *g = s_glb;
    term msg = term_alloc_tuple(3, heap);
    term_put_tuple_element(msg, 0, atom(g, A_BABYTALK));
    term_put_tuple_element(msg, 1, term_from_ref_ticks(j->ref, heap));
    term_put_tuple_element(msg, 2, result);
    globalcontext_send_message_from_task(g, j->pid, NormalMessage, msg);
    memory_destroy_heap(heap, g);
}
#define MSG_WORDS (TUPLE_SIZE(3) + REF_SIZE)

// Result: {ok, Text, Info} | {error, Code}
static void send_transcript(const job_t *j, int rc, const stt_result_t *r)
{
    GlobalContext *g = s_glb;
    const size_t len = rc ? 0 : strlen(r->text);
    const size_t words = MSG_WORDS + TUPLE_SIZE(3) + term_binary_heap_size(len)
        + 6 * (CONS_SIZE + TUPLE_SIZE(2)) + 4 * FLOAT_SIZE + TUPLE_SIZE(2) + TUPLE_SIZE(2);
    Heap heap;
    if (UNLIKELY(memory_init_heap(&heap, words) != MEMORY_GC_OK)) {
        ESP_LOGE(TAG, "no memory for the result message");
        return;
    }
    term result;
    if (rc) {
        result = error_tuple(&heap, term_from_int(rc));
    } else {
        term text = term_create_uninitialized_binary(len, &heap, g);
        memcpy((void *) term_binary_data(text), r->text, len);
        term span = term_alloc_tuple(2, &heap);
        term_put_tuple_element(span, 0, term_from_int(r->kw_start));
        term_put_tuple_element(span, 1, term_from_int(r->kw_end));
        term info = term_nil();
        info = prop(&heap, atom(g, A_DEC_MS), term_from_float(r->dec_ms, &heap), info);
        info = prop(&heap, atom(g, A_MODEL_MS), term_from_float(r->model_ms, &heap), info);
        info = prop(&heap, atom(g, A_FE_MS), term_from_float(r->fe_ms, &heap), info);
        info = prop(&heap, atom(g, A_FRAMES), term_from_int(r->frames), info);
        info = prop(&heap, atom(g, A_SPAN), span, info);
        info = prop(&heap, atom(g, A_SCORE), r->score < -1e29f ? UNDEFINED_ATOM : term_from_float(r->score, &heap), info);
        result = term_alloc_tuple(3, &heap);
        term_put_tuple_element(result, 0, OK_ATOM);
        term_put_tuple_element(result, 1, text);
        term_put_tuple_element(result, 2, info);
    }
    send_msg(j, &heap, result);
}

static void do_transcribe(const job_t *j)
{
    static stt_result_t r, out;  // ~1.1 KB each: off the task stack
    int rc = stt_engine_open();
    if (!rc) rc = stt_engine_run(j->pcm, j->n, &r);
    if (rc) ESP_LOGW(TAG, "transcription failed: %d", rc);
    heap_caps_free(j->pcm);
    out = r;  // copy out, then let the next job in while we reply
    unlock_engine();
    send_transcript(j, rc, &out);
}

static void worker(void *arg)
{
    (void) arg;
    job_t j;
    for (;;) {
        xQueueReceive(s_jobs, &j, portMAX_DELAY);
        do_transcribe(&j);
    }
}

// ------------------------------------------------------------------------------ mic task
// Streams the microphone to one listener in ChunkMs pieces, independently of the engine
// (so audio keeps flowing while a transcription runs). Core 1, priority 6: above the
// inference worker, so the I2S DMA ring (256 ms) is drained on time.
static SemaphoreHandle_t s_mic_go;
static atomic_int s_mic_on;       // 1 while a listener wants chunks
static atomic_int s_mic_claimed;  // listen() .. the task's `stopped` message
static int32_t s_mic_pid;
static uint64_t s_mic_ref;
static int s_mic_frames;

// {babytalk_mic, Ref, Payload}; Payload is built by `fill` on the message heap (an invalid
// term: send nothing)
static void send_mic(size_t extra_words, term (*fill)(Heap *, void *), void *arg)
{
    GlobalContext *g = s_glb;
    Heap heap;
    if (UNLIKELY(memory_init_heap(&heap, TUPLE_SIZE(3) + REF_SIZE + extra_words) != MEMORY_GC_OK)) {
        ESP_LOGE(TAG, "no memory for a mic message");
        return;
    }
    term payload = fill(&heap, arg);
    if (term_is_invalid_term(payload)) {
        memory_destroy_heap(&heap, g);
        return;
    }
    term msg = term_alloc_tuple(3, &heap);
    term_put_tuple_element(msg, 0, atom(g, A_BABYTALK_MIC));
    term_put_tuple_element(msg, 1, term_from_ref_ticks(s_mic_ref, &heap));
    term_put_tuple_element(msg, 2, payload);
    globalcontext_send_message_from_task(g, s_mic_pid, NormalMessage, msg);
    memory_destroy_heap(&heap, g);
}

static term fill_stopped(Heap *h, void *arg)
{
    (void) h;
    (void) arg;
    return atom(s_glb, A_STOPPED);
}

static term fill_error(Heap *h, void *arg) { return error_tuple(h, term_from_int(*(int *) arg)); }

static term fill_chunk(Heap *h, void *arg)
{
    int *rc = (int *) arg;
    term pcm = term_create_uninitialized_binary((size_t) s_mic_frames * 2, h, s_glb);
    if (term_is_invalid_term(pcm)) {
        *rc = -10;
        return pcm;
    }
    *rc = mic_read_mono((int16_t *) term_binary_data(pcm), s_mic_frames);
    return *rc ? term_invalid_term() : pcm;
}

static void mic_task(void *arg)
{
    (void) arg;
    for (;;) {
        xSemaphoreTake(s_mic_go, portMAX_DELAY);
        int rc = mic_open(MIC_GAIN);  // once; later calls are no-ops
        if (!rc) rc = mic_start();
        while (!rc && atomic_load(&s_mic_on)) {
            send_mic(term_binary_heap_size((size_t) s_mic_frames * 2), fill_chunk, &rc);
        }
        mic_stop();
        if (rc) {
            ESP_LOGW(TAG, "mic failed: %d", rc);
            send_mic(TUPLE_SIZE(2), fill_error, &rc);
        } else {
            send_mic(0, fill_stopped, NULL);
        }
        atomic_store(&s_mic_on, 0);
        atomic_store(&s_mic_claimed, 0);
    }
}

// ------------------------------------------------------------------------------ NIFs
// NIF arguments are not GC roots unless passed as such: read binaries after a GC only
// through a rooted argv.

// Queue a job (the caller holds the engine lock) and return {ok, Ref}
static term submit(Context *ctx, const job_t *j)
{
    xQueueSend(s_jobs, j, portMAX_DELAY);  // depth 1 and we hold the lock: never blocks
    term t = term_alloc_tuple(2, &ctx->heap);
    term_put_tuple_element(t, 0, OK_ATOM);
    term_put_tuple_element(t, 1, term_from_ref_ticks(j->ref, &ctx->heap));
    return t;
}

// transcribe_nif(Pcm16Mono) -> {ok, Ref}: the PCM is copied, so the binary may be dropped
static term nif_transcribe(Context *ctx, int argc, term argv[])
{
    VALIDATE_VALUE(argv[0], term_is_binary);
    const size_t bytes = term_binary_size(argv[0]);
    if (bytes < 2 || bytes % 2) RAISE_ERROR(BADARG_ATOM);
    if (UNLIKELY(memory_ensure_free_with_roots(ctx, TUPLE_SIZE(2) + REF_SIZE, argc, argv, MEMORY_CAN_SHRINK) != MEMORY_GC_OK)) {
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    if (!lock_engine()) return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    job_t j = {heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM), (int) (bytes / 2), ctx->process_id,
               globalcontext_get_ref_ticks(ctx->global)};
    if (!j.pcm) {
        unlock_engine();
        return error_tuple(&ctx->heap, atom(ctx->global, A_NO_MEMORY));
    }
    memcpy(j.pcm, term_binary_data(argv[0]), bytes);
    return submit(ctx, &j);
}

// listen_nif(ChunkMs) -> {ok, Ref}
static term nif_listen(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    VALIDATE_VALUE(argv[0], term_is_integer);
    const avm_int_t ms = term_to_int(argv[0]);
    if (ms < 20 || ms > 10000) RAISE_ERROR(BADARG_ATOM);
    if (UNLIKELY(memory_ensure_free(ctx, TUPLE_SIZE(2) + REF_SIZE) != MEMORY_GC_OK)) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    int expected = 0;
    if (!atomic_compare_exchange_strong(&s_mic_claimed, &expected, 1)) {
        return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    }
    s_mic_pid = ctx->process_id;
    s_mic_ref = globalcontext_get_ref_ticks(ctx->global);
    s_mic_frames = (int) ms * 16;
    atomic_store(&s_mic_on, 1);
    xSemaphoreGive(s_mic_go);
    term t = term_alloc_tuple(2, &ctx->heap);
    term_put_tuple_element(t, 0, OK_ATOM);
    term_put_tuple_element(t, 1, term_from_ref_ticks(s_mic_ref, &ctx->heap));
    return t;
}

// stop_listening() -> ok: the current chunk finishes, then {babytalk_mic, Ref, stopped}
static term nif_stop_listening(Context *ctx, int argc, term argv[])
{
    UNUSED(ctx);
    UNUSED(argc);
    UNUSED(argv);
    atomic_store(&s_mic_on, 0);
    return OK_ATOM;
}

static term ok_int(Context *ctx, avm_int_t v)
{
    term t = term_alloc_tuple(2, &ctx->heap);
    term_put_tuple_element(t, 0, OK_ATOM);
    term_put_tuple_element(t, 1, term_from_int(v));
    return t;
}

#define MAX_SPELLINGS 32

static void free_all(char **s, int n)
{
    for (int i = 0; i < n; i++) free(s[i]);
}

// phrase([Spelling :: binary() | string()]) -> {ok, Sequences}; [] clears the phrase
static term nif_phrase(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    VALIDATE_VALUE(argv[0], term_is_list);
    char *s[MAX_SPELLINGS];
    int n = 0;
    for (term l = argv[0]; term_is_nonempty_list(l); l = term_get_list_tail(l)) {
        int ok = 0;
        char *str = n < MAX_SPELLINGS ? interop_term_to_string(term_get_list_head(l), &ok) : NULL;
        if (!ok || !str) {
            free(str);
            free_all(s, n);
            RAISE_ERROR(BADARG_ATOM);
        }
        s[n++] = str;
    }
    if (UNLIKELY(memory_ensure_free(ctx, TUPLE_SIZE(2)) != MEMORY_GC_OK)) {
        free_all(s, n);
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    if (!lock_engine()) {
        free_all(s, n);
        return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    }
    int rc = stt_engine_open();  // the phrase tokenizer (kws_init) comes up with the engine
    int seqs = rc ? rc : stt_engine_set_phrase((const char *const *) s, n);
    unlock_engine();
    free_all(s, n);
    if (rc) return error_tuple(&ctx->heap, term_from_int(rc));
    if (seqs < 0) return error_tuple(&ctx->heap, atom(ctx->global, A_NO_MEMORY));
    return ok_int(ctx, seqs);
}

// cache(Bytes) -> {ok, Cached}: keep up to Bytes of weights in PSRAM (faster than flash)
static term nif_cache(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    VALIDATE_VALUE(argv[0], term_is_integer);
    avm_int_t b = term_to_int(argv[0]);
    if (b < 0) RAISE_ERROR(BADARG_ATOM);
    if (UNLIKELY(memory_ensure_free(ctx, TUPLE_SIZE(2)) != MEMORY_GC_OK)) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    if (!lock_engine()) return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    int rc = stt_engine_open();
    size_t got = rc ? 0 : stt_engine_cache((size_t) b);
    unlock_engine();
    if (rc) return error_tuple(&ctx->heap, term_from_int(rc));
    return ok_int(ctx, (avm_int_t) got);
}

static term proplist(Context *ctx, const char *const *names, const avm_int_t *vals, int n)
{
    if (UNLIKELY(memory_ensure_free(ctx, n * (TUPLE_SIZE(2) + CONS_SIZE)) != MEMORY_GC_OK)) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    term list = term_nil();
    for (int i = n - 1; i >= 0; i--) list = prop(&ctx->heap, atom(ctx->global, names[i]), term_from_int(vals[i]), list);
    return list;
}

// info() -> [{model_bytes, B}, {cache_bytes, B}, {ops, N}, {int4_ops, N}, {internal_free, B}, {psram_free, B}]
// (model fields are 0 until the engine is first used)
static term nif_info(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    UNUSED(argv);
    stt_info_t in;
    stt_engine_info(&in);
    const char *names[6] = {A_MODEL_BYTES, A_CACHE_BYTES, A_OPS, A_INT4_OPS, A_INTERNAL_FREE, A_PSRAM_FREE};
    const avm_int_t vals[6] = {(avm_int_t) in.model_bytes, (avm_int_t) in.cache_bytes, in.ops, in.int4_ops,
                               (avm_int_t) in.internal_free, (avm_int_t) in.psram_free};
    return proplist(ctx, names, vals, 6);
}

static term nif_heap_info(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    UNUSED(argv);
    const char *names[4] = {A_INTERNAL_FREE, A_INTERNAL_LARGEST, A_PSRAM_FREE, A_PSRAM_LARGEST};
    const avm_int_t vals[4] = {
        (avm_int_t) heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        (avm_int_t) heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
        (avm_int_t) heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        (avm_int_t) heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM),
    };
    return proplist(ctx, names, vals, 4);
}

#define NIF(name, fn) static const struct Nif name##_nif = {.base.type = NIFFunctionType, .nif_ptr = fn}
NIF(transcribe, nif_transcribe);
NIF(listen, nif_listen);
NIF(stop_listening, nif_stop_listening);
NIF(phrase, nif_phrase);
NIF(cache, nif_cache);
NIF(info, nif_info);
NIF(heap_info, nif_heap_info);

static const struct {
    const char *name;  // "fun/arity"
    const struct Nif *nif;
} NIFS[] = {
    {"transcribe_nif/1", &transcribe_nif}, {"listen_nif/1", &listen_nif},
    {"stop_listening/0", &stop_listening_nif}, {"phrase/1", &phrase_nif}, {"cache/1", &cache_nif},
    {"info/0", &info_nif}, {"heap_info/0", &heap_info_nif},
};

// The Elixir module BabyTalk delegates to :babytalk, so only "babytalk:..." names resolve here.
static const struct Nif *babytalk_get_nif(const char *name)
{
    if (strncmp(name, "babytalk:", 9)) return NULL;
    for (size_t i = 0; i < sizeof(NIFS) / sizeof(NIFS[0]); i++) {
        if (!strcmp(name + 9, NIFS[i].name)) return NIFS[i].nif;
    }
    return NULL;
}

// At VM start, before any Erlang code (and so before WiFi) takes internal RAM: reserve the
// engine's 84.5 KB contiguous SRAM block and start the worker. Inference runs on core 1 at
// the schedulers' priority (5), so it time-slices with the core-1 scheduler; mmrt's own
// helper task takes core 0 during the big layers. The mic task (see above) sits at 6.
static void babytalk_init(GlobalContext *global)
{
    s_glb = global;
    if (sram_pool_reserve()) ESP_LOGE(TAG, "could not reserve the SRAM pool: transcription will fail");
    s_jobs = xQueueCreate(1, sizeof(job_t));
    s_mic_go = xSemaphoreCreateBinary();
    if (!s_jobs || !s_mic_go || xTaskCreatePinnedToCore(worker, "babytalk", 12 * 1024, NULL, 5, NULL, 1) != pdPASS
        || xTaskCreatePinnedToCore(mic_task, "babytalk_mic", 4 * 1024, NULL, 6, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "could not start the babytalk tasks");
    }
}

REGISTER_NIF_COLLECTION(babytalk, babytalk_init, NULL, babytalk_get_nif)

#endif
