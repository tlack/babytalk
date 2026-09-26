// BabyTalk NIFs for AtomVM: on-device speech to text, module `babytalk` (Erlang) /
// `BabyTalk` (Elixir, delegating). The Erlang docs are in atomvm/lib/babytalk/src/babytalk.erl.
//
//   babytalk:transcribe_nif(Pcm) -> {ok, Ref} | {error, busy | no_memory}
//       then the caller gets {babytalk, Ref, {ok, Text, Info} | {error, Code}}
//   babytalk:listen_nif(ChunkMs) -> {ok, Ref} | {error, busy}
//       then {babytalk_mic, Ref, Pcm16Mono} every ChunkMs (the board's mic, 16 kHz, gapless)
//       until stop_listening(), then {babytalk_mic, Ref, stopped}; or {babytalk_mic, Ref, {error, Code}}
//   babytalk:stop_listening() -> ok
//   babytalk:play_nif(Pcm16Mono, Rate, Volume) -> {ok, Ref} | {error, busy}
//       then {babytalk_play, Ref, done | {error, Code}} (the board's speaker)
//   babytalk:tones_nif([{Hz, Ms [, Level [, flat | bloop]]}], Volume) -> {ok, Ref} | {error, busy}   (same messages as play)
//   babytalk:rms(Pcm16Mono) -> Integer
//   babytalk:say_nif(Text, LengthPermille) -> {ok, Ref} | {error, busy}
//       then {babytalk, Ref, {ok, Pcm24kMono, Info} | {error, Code}} (sanoTTS)
//   babytalk:phrase(Spellings) -> {ok, Sequences} | {error, busy | no_memory}
//   babytalk:cache(Bytes) -> {ok, Cached} | {error, busy | Code}
//   babytalk:info() -> [{Key, Value}]
//   babytalk:heap_info() -> [{internal_free, B}, {internal_largest, B}, {psram_free, B}, {psram_largest, B},
//                            {pool_bytes, B}]   (pool_bytes: 0 when the engines' block wasn't reserved at boot)
//
// Inference (0.3-2 s) runs on our own FreeRTOS task, never on a scheduler: AtomVM has no
// dirty NIFs. The model is the "model" flash partition (components/stt_engine).
#include <sdkconfig.h>
#ifdef CONFIG_AVM_ENABLE_BABYTALK_NIFS

#include <stdatomic.h>
#include <math.h>
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
#include <nvs.h>
#include <nvs_flash.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include "board_audio.h"
#include "tts_engine.h"
#include "sram_pool.h"
#include "stt_engine.h"

#define TAG "babytalk"

// AtomVM atom strings: length byte, then the name
static const char *const A_BABYTALK = "\x08" "babytalk";
static const char *const A_BABYTALK_MIC = "\x0C" "babytalk_mic";
static const char *const A_BABYTALK_PLAY = "\x0D" "babytalk_play";
static const char *const A_PHONEMES = "\x08" "phonemes";
static const char *const A_G2P_MS = "\x06" "g2p_ms";
static const char *const A_SYNTH_MS = "\x08" "synth_ms";
static const char *const A_RATE = "\x04" "rate";
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
static const char *const A_POOL_BYTES = "\x0A" "pool_bytes";

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
enum { JOB_STT, JOB_TTS };

typedef struct {
    int kind;
    int16_t *pcm;  // JOB_STT: PSRAM copy of the audio, freed by the worker
    int n;
    char *text;    // JOB_TTS: copy of the text, freed by the worker
    int length_permille;  // JOB_TTS: pace (1000 = the voice's own)
    int32_t pid;
    uint64_t ref;
} job_t;

#define MIC_GAIN (-1)  // the board's default (board_audio.h)

static GlobalContext *s_glb;
static bool s_reserved;  // the SRAM pool was taken at boot
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

// Result: {ok, Pcm24k, [{rate, 24000}, {phonemes, N}, {g2p_ms, F}, {synth_ms, F}]} | {error, Code}
static void do_say(const job_t *j)
{
    GlobalContext *g = s_glb;
    int16_t *pcm;
    int n;
    tts_stats_t st;
    tts_set_length_scale(j->length_permille / 1000.0f);
    int rc = tts_say(j->text, 0.77f, &pcm, &n, &st);  // peak at 77% of full scale, as mpy's echo
    free(j->text);
    unlock_engine();
    if (rc) ESP_LOGW(TAG, "speech synthesis failed: %d", rc);
    const size_t bytes = rc ? 0 : (size_t) n * 2;
    Heap heap;
    if (UNLIKELY(memory_init_heap(&heap, MSG_WORDS + TUPLE_SIZE(3) + term_binary_heap_size(bytes)
                                      + 4 * (CONS_SIZE + TUPLE_SIZE(2)) + 2 * FLOAT_SIZE + TUPLE_SIZE(2)) != MEMORY_GC_OK)) {
        ESP_LOGE(TAG, "no memory for the speech message");
        heap_caps_free(pcm);  // NULL when tts_say failed
        return;
    }
    term result;
    term bin = rc ? term_invalid_term() : term_create_uninitialized_binary(bytes, &heap, g);
    if (!rc && term_is_invalid_term(bin)) rc = -3;
    if (rc) {
        result = error_tuple(&heap, term_from_int(rc));
    } else {
        memcpy((void *) term_binary_data(bin), pcm, bytes);
        term info = term_nil();
        info = prop(&heap, atom(g, A_SYNTH_MS), term_from_float(st.synth_ms, &heap), info);
        info = prop(&heap, atom(g, A_G2P_MS), term_from_float(st.g2p_ms, &heap), info);
        info = prop(&heap, atom(g, A_PHONEMES), term_from_int(st.phonemes), info);
        info = prop(&heap, atom(g, A_RATE), term_from_int(TTS_SAMPLE_RATE), info);
        result = term_alloc_tuple(3, &heap);
        term_put_tuple_element(result, 0, OK_ATOM);
        term_put_tuple_element(result, 1, bin);
        term_put_tuple_element(result, 2, info);
    }
    heap_caps_free(pcm);
    send_msg(j, &heap, result);
}

static void worker(void *arg)
{
    (void) arg;
    job_t j;
    for (;;) {
        xQueueReceive(s_jobs, &j, portMAX_DELAY);
        if (j.kind == JOB_TTS) do_say(&j);
        else do_transcribe(&j);
    }
}

// ------------------------------------------------------------------------------ audio task
// Owns the board's audio (board_audio.c): streams the microphone to one listener in
// ChunkMs pieces, independently of the engine (so audio keeps flowing while a transcription
// runs), or plays PCM through the speaker. The two share one I2S port, so they take turns:
// one claim at a time (listen() .. `stopped`, play() .. `done`). Core 1, priority 6: above
// the inference worker, so the I2S DMA ring (256 ms) is served on time.
enum { AUDIO_LISTEN, AUDIO_PLAY };
static SemaphoreHandle_t s_audio_go;
static atomic_int s_audio_claimed;
static atomic_int s_mic_on;  // 1 while a listener wants chunks
static struct {
    int kind;
    int32_t pid;
    uint64_t ref;
    int frames;         // listen: per chunk
    int16_t *pcm;       // play: PSRAM copy, freed by the task
    int n, rate, volume;
} s_aj;

// {Tag, Ref, Payload}; Payload is built by `fill` on the message heap (an invalid term: send nothing)
static void send_audio(const char *tag, size_t extra_words, term (*fill)(Heap *, void *), void *arg)
{
    GlobalContext *g = s_glb;
    Heap heap;
    if (UNLIKELY(memory_init_heap(&heap, TUPLE_SIZE(3) + REF_SIZE + extra_words) != MEMORY_GC_OK)) {
        ESP_LOGE(TAG, "no memory for an audio message");
        return;
    }
    term payload = fill(&heap, arg);
    if (term_is_invalid_term(payload)) {
        memory_destroy_heap(&heap, g);
        return;
    }
    term msg = term_alloc_tuple(3, &heap);
    term_put_tuple_element(msg, 0, atom(g, tag));
    term_put_tuple_element(msg, 1, term_from_ref_ticks(s_aj.ref, &heap));
    term_put_tuple_element(msg, 2, payload);
    globalcontext_send_message_from_task(g, s_aj.pid, NormalMessage, msg);
    memory_destroy_heap(&heap, g);
}

static term fill_atom(Heap *h, void *arg)
{
    (void) h;
    return atom(s_glb, (const char *) arg);
}

static term fill_error(Heap *h, void *arg) { return error_tuple(h, term_from_int(*(int *) arg)); }

static term fill_chunk(Heap *h, void *arg)
{
    int *rc = (int *) arg;
    term pcm = term_create_uninitialized_binary((size_t) s_aj.frames * 2, h, s_glb);
    if (term_is_invalid_term(pcm)) {
        *rc = -10;
        return pcm;
    }
    *rc = board_audio_rx_read((int16_t *) term_binary_data(pcm), s_aj.frames);
    return *rc ? term_invalid_term() : pcm;
}

static const char *const A_DONE = "\x04" "done";

static void audio_task(void *arg)
{
    (void) arg;
    for (;;) {
        xSemaphoreTake(s_audio_go, portMAX_DELAY);
        int rc = board_audio_init(MIC_GAIN);  // once; later calls return at once
        if (s_aj.kind == AUDIO_PLAY) {
            if (!rc) rc = board_audio_play(s_aj.pcm, s_aj.n, s_aj.rate, s_aj.volume);
            if (!rc) ESP_LOGI(TAG, "played %d samples, %d DMA underruns", s_aj.n, board_audio_underruns());
            heap_caps_free(s_aj.pcm);
            if (rc) ESP_LOGW(TAG, "playback failed: %d", rc);
            atomic_store(&s_audio_claimed, 0);
            if (rc) send_audio(A_BABYTALK_PLAY, TUPLE_SIZE(2), fill_error, &rc);
            else send_audio(A_BABYTALK_PLAY, 0, fill_atom, (void *) A_DONE);
            continue;
        }
        if (!rc) rc = board_audio_rx_start();
        while (!rc && atomic_load(&s_mic_on)) {
            send_audio(A_BABYTALK_MIC, term_binary_heap_size((size_t) s_aj.frames * 2), fill_chunk, &rc);
        }
        board_audio_rx_stop();
        if (rc) ESP_LOGW(TAG, "mic failed: %d", rc);
        atomic_store(&s_mic_on, 0);
        atomic_store(&s_audio_claimed, 0);
        if (rc) send_audio(A_BABYTALK_MIC, TUPLE_SIZE(2), fill_error, &rc);
        else send_audio(A_BABYTALK_MIC, 0, fill_atom, (void *) A_STOPPED);
    }
}

static bool claim_audio(void)
{
    int expected = 0;
    return atomic_compare_exchange_strong(&s_audio_claimed, &expected, 1);
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
    job_t j = {JOB_STT, heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM), (int) (bytes / 2), NULL, 1000, ctx->process_id,
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
    if (!claim_audio()) return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    s_aj.kind = AUDIO_LISTEN;
    s_aj.pid = ctx->process_id;
    s_aj.ref = globalcontext_get_ref_ticks(ctx->global);
    s_aj.frames = (int) ms * 16;
    atomic_store(&s_mic_on, 1);
    xSemaphoreGive(s_audio_go);
    term t = term_alloc_tuple(2, &ctx->heap);
    term_put_tuple_element(t, 0, OK_ATOM);
    term_put_tuple_element(t, 1, term_from_ref_ticks(s_aj.ref, &ctx->heap));
    return t;
}

// Hand PCM (PSRAM, ours to free) to the audio task, whose claim the caller holds -> {ok, Ref}
static term start_play(Context *ctx, int16_t *pcm, int n, int rate, int volume)
{
    s_aj.kind = AUDIO_PLAY;
    s_aj.pid = ctx->process_id;
    s_aj.ref = globalcontext_get_ref_ticks(ctx->global);
    s_aj.pcm = pcm;
    s_aj.n = n;
    s_aj.rate = rate;
    s_aj.volume = volume;
    xSemaphoreGive(s_audio_go);
    term t = term_alloc_tuple(2, &ctx->heap);
    term_put_tuple_element(t, 0, OK_ATOM);
    term_put_tuple_element(t, 1, term_from_ref_ticks(s_aj.ref, &ctx->heap));
    return t;
}

// play_nif(Pcm16Mono, Rate, Volume) -> {ok, Ref}: then {babytalk_play, Ref, done | {error, Code}}
static term nif_play(Context *ctx, int argc, term argv[])
{
    VALIDATE_VALUE(argv[0], term_is_binary);
    VALIDATE_VALUE(argv[1], term_is_integer);
    VALIDATE_VALUE(argv[2], term_is_integer);
    const size_t bytes = term_binary_size(argv[0]);
    const avm_int_t rate = term_to_int(argv[1]), vol = term_to_int(argv[2]);
    if (bytes < 2 || bytes % 2 || rate < 8000 || rate > 48000 || vol < 0 || vol > 100) RAISE_ERROR(BADARG_ATOM);
    if (UNLIKELY(memory_ensure_free_with_roots(ctx, TUPLE_SIZE(2) + REF_SIZE, argc, argv, MEMORY_CAN_SHRINK) != MEMORY_GC_OK)) {
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    if (!claim_audio()) return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    int16_t *pcm = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
    if (!pcm) {
        atomic_store(&s_audio_claimed, 0);
        return error_tuple(&ctx->heap, atom(ctx->global, A_NO_MEMORY));
    }
    memcpy(pcm, term_binary_data(argv[0]), bytes);
    return start_play(ctx, pcm, (int) (bytes / 2), (int) rate, (int) vol);
}

#define TONE_RATE 24000
#define TONE_MAX_NOTES 24
#define TONE_MAX_MS 6000

// tones_nif(Notes, Volume) -> {ok, Ref}: then {babytalk_play, Ref, done | {error, Code}}.
// Notes: [{Hz, Ms} | {Hz, Ms, Level} | {Hz, Ms, Level, Shape}], Hz 0 = a rest, Level 0..100
// (of the note's full loudness, a third of full scale; default 100), Shape:
//   flat   a steady sine with 8 ms raised-cosine edges (the default: chimes, cues)
//   bloop  soft: a 15 ms rise, then an exponential fall to near silence by the end, the
//          pitch settling 4% as it goes -- a droplet, never a beep
// Made here: building PCM element by element in Erlang is slow on AtomVM.
static term nif_tones(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    VALIDATE_VALUE(argv[0], term_is_list);
    VALIDATE_VALUE(argv[1], term_is_integer);
    const avm_int_t vol = term_to_int(argv[1]);
    const term BLOOP = globalcontext_make_atom(ctx->global, "\x05" "bloop");
    const term FLAT = globalcontext_make_atom(ctx->global, "\x04" "flat");
    int hz[TONE_MAX_NOTES], ms[TONE_MAX_NOTES], level[TONE_MAX_NOTES], bloop[TONE_MAX_NOTES], n = 0, total_ms = 0;
    for (term l = argv[0]; term_is_nonempty_list(l); l = term_get_list_tail(l)) {
        term t = term_get_list_head(l);
        const int ar = term_is_tuple(t) ? term_get_tuple_arity(t) : 0;
        if (n == TONE_MAX_NOTES || ar < 2 || ar > 4 || !term_is_integer(term_get_tuple_element(t, 0))
            || !term_is_integer(term_get_tuple_element(t, 1))
            || (ar >= 3 && !term_is_integer(term_get_tuple_element(t, 2)))) {
            RAISE_ERROR(BADARG_ATOM);
        }
        hz[n] = (int) term_to_int(term_get_tuple_element(t, 0));
        ms[n] = (int) term_to_int(term_get_tuple_element(t, 1));
        level[n] = ar >= 3 ? (int) term_to_int(term_get_tuple_element(t, 2)) : 100;
        const term shape = ar == 4 ? term_get_tuple_element(t, 3) : FLAT;
        if (shape != BLOOP && shape != FLAT) RAISE_ERROR(BADARG_ATOM);
        bloop[n] = shape == BLOOP;
        if (hz[n] < 0 || hz[n] > 8000 || ms[n] < 1 || level[n] < 0 || level[n] > 100) RAISE_ERROR(BADARG_ATOM);
        total_ms += ms[n++];
    }
    if (n == 0 || total_ms > TONE_MAX_MS || vol < 0 || vol > 100) RAISE_ERROR(BADARG_ATOM);
    if (UNLIKELY(memory_ensure_free(ctx, TUPLE_SIZE(2) + REF_SIZE) != MEMORY_GC_OK)) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    if (!claim_audio()) return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    const int samples = total_ms * TONE_RATE / 1000;
    int16_t *pcm = heap_caps_malloc((size_t) samples * 2, MALLOC_CAP_SPIRAM);
    if (!pcm) {
        atomic_store(&s_audio_claimed, 0);
        return error_tuple(&ctx->heap, atom(ctx->global, A_NO_MEMORY));
    }
    const int fade = TONE_RATE * 8 / 1000, rise = TONE_RATE * 15 / 1000;
    int16_t *p = pcm;
    for (int k = 0; k < n; k++) {
        const int len = ms[k] * TONE_RATE / 1000;
        const float amp = 10900.0f * (float) level[k] / 100.0f;
        const float w0 = 2.0f * (float) M_PI * (float) hz[k] / TONE_RATE;
        const float tau = (float) len / 5.0f;  // bloop: ~-43 dB by the end
        float phase = 0.0f;
        for (int i = 0; i < len; i++) {
            float g;
            float w = w0;
            if (bloop[k]) {
                g = i < rise ? 0.5f - 0.5f * cosf((float) M_PI * (float) i / (float) rise) : expf(-(float) (i - rise) / tau);
                w = w0 * (1.0f - 0.04f * (float) i / (float) len);  // settling a little in pitch
            } else {
                const int edge = i < len - 1 - i ? i : len - 1 - i;
                g = edge < fade ? 0.5f - 0.5f * cosf((float) M_PI * (float) edge / (float) fade) : 1.0f;
            }
            phase += w;
            *p++ = hz[k] ? (int16_t) (amp * g * sinf(phase)) : 0;
        }
    }
    return start_play(ctx, pcm, (int) (p - pcm), TONE_RATE, (int) vol);
}

// rms(Pcm16Mono) -> Integer: root mean square level (0..32768), e.g. to tell speech from silence
static term nif_rms(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    VALIDATE_VALUE(argv[0], term_is_binary);
    const size_t n = term_binary_size(argv[0]) / 2;
    const uint8_t *b = (const uint8_t *) term_binary_data(argv[0]);
    int64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t v = (int16_t) (b[2 * i] | (b[2 * i + 1] << 8));
        sum += v * v;
    }
    return term_from_int(n ? (avm_int_t) sqrtf((float) sum / (float) n) : 0);
}

// say_nif(Text, LengthPermille) -> {ok, Ref}: then {babytalk, Ref, {ok, Pcm24k, Info} | {error, Code}}
static term nif_say(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    VALIDATE_VALUE(argv[1], term_is_integer);
    const avm_int_t pace = term_to_int(argv[1]);
    if (pace < 500 || pace > 2000) RAISE_ERROR(BADARG_ATOM);
    int ok = 0;
    char *text = interop_term_to_string(argv[0], &ok);  // binary, string or iolist
    if (!ok || !text) {
        free(text);
        RAISE_ERROR(BADARG_ATOM);
    }
    if (UNLIKELY(memory_ensure_free(ctx, TUPLE_SIZE(2) + REF_SIZE) != MEMORY_GC_OK)) {
        free(text);
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    if (!lock_engine()) {
        free(text);
        return error_tuple(&ctx->heap, atom(ctx->global, A_BUSY));
    }
    job_t j = {JOB_TTS, NULL, 0, text, (int) pace, ctx->process_id, globalcontext_get_ref_ticks(ctx->global)};
    return submit(ctx, &j);
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
    const char *names[5] = {A_INTERNAL_FREE, A_INTERNAL_LARGEST, A_PSRAM_FREE, A_PSRAM_LARGEST, A_POOL_BYTES};
    const avm_int_t vals[5] = {
        (avm_int_t) heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        (avm_int_t) heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
        (avm_int_t) heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        (avm_int_t) heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM),
        s_reserved ? SRAM_POOL_BYTES : 0,
    };
    return proplist(ctx, names, vals, 5);
}

#define NIF(name, fn) static const struct Nif name##_nif = {.base.type = NIFFunctionType, .nif_ptr = fn}
NIF(transcribe, nif_transcribe);
NIF(listen, nif_listen);
NIF(play, nif_play);
NIF(tones, nif_tones);
NIF(rms, nif_rms);
NIF(say, nif_say);
NIF(stop_listening, nif_stop_listening);
NIF(phrase, nif_phrase);
NIF(cache, nif_cache);
NIF(info, nif_info);
NIF(heap_info, nif_heap_info);

static const struct {
    const char *name;  // "fun/arity"
    const struct Nif *nif;
} NIFS[] = {
    {"transcribe_nif/1", &transcribe_nif}, {"listen_nif/1", &listen_nif}, {"play_nif/3", &play_nif},
    {"tones_nif/2", &tones_nif}, {"rms/1", &rms_nif}, {"say_nif/2", &say_nif},
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

// Whether to take the pool at boot: NVS babytalk/reserve (a binary, <<0>> or <<1>>, written by
// babytalk:reserve_at_boot/1) if present, else CONFIG_BABYTALK_RESERVE_AT_BOOT
static bool reserve_at_boot(void)
{
#ifdef CONFIG_BABYTALK_RESERVE_AT_BOOT
    bool want = true;
#else
    bool want = false;
#endif
    nvs_flash_init();  // (AtomVM's own NVS init may not have run yet; a second init is harmless)
    nvs_handle_t h;
    if (nvs_open("babytalk", NVS_READONLY, &h) == ESP_OK) {
        uint8_t v;
        size_t n = 1;
        if (nvs_get_blob(h, "reserve", &v, &n) == ESP_OK && n == 1) want = v != 0;
        nvs_close(h);
    }
    return want;
}

// At VM start, before any Erlang code (and so before WiFi) takes internal RAM: reserve the
// engine's 84.5 KB contiguous SRAM block (unless told not to: reserve_at_boot) and start the
// worker. Without the block, transcription and speech return errors; mic, speaker and tones work. Inference runs on core 1 at
// the schedulers' priority (5), so it time-slices with the core-1 scheduler; mmrt's own
// helper task takes core 0 during the big layers. The audio task (see above) sits at 6.
static void babytalk_init(GlobalContext *global)
{
    s_glb = global;
    if (!reserve_at_boot()) {
        ESP_LOGI(TAG, "SRAM pool not reserved (babytalk:reserve_at_boot/1): no transcription or speech");
    } else if (sram_pool_reserve()) {
        ESP_LOGE(TAG, "could not reserve the SRAM pool: transcription will fail");
    } else {
        s_reserved = true;
    }
    s_jobs = xQueueCreate(1, sizeof(job_t));
    s_audio_go = xSemaphoreCreateBinary();
#ifdef CONFIG_BABYTALK_STACKS_IN_PSRAM
    // (16 KB of internal RAM back: see Kconfig)
    const UBaseType_t caps = MALLOC_CAP_SPIRAM;
#else
    const UBaseType_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
#endif
    if (!s_jobs || !s_audio_go
        || xTaskCreatePinnedToCoreWithCaps(worker, "babytalk", 12 * 1024, NULL, 5, NULL, 1, caps) != pdPASS
        || xTaskCreatePinnedToCoreWithCaps(audio_task, "babytalk_audio", 4 * 1024, NULL, 6, NULL, 1, caps) != pdPASS) {
        ESP_LOGE(TAG, "could not start the babytalk tasks");
    }
}

REGISTER_NIF_COLLECTION(babytalk, babytalk_init, NULL, babytalk_get_nif)

#endif
