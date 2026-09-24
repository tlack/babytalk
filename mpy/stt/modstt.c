// MicroPython module `stt`: on-device speech to text (see stt_engine.h).
//
//   import stt
//   stt.transcribe(pcm[, channels])  # 16 kHz int16 bytes -> str (blocks; other threads run)
//   stt.start(pcm[, channels]); stt.busy(); stt.result()   # background: poll from asyncio
//   stt.phrase(["wake up tomato face"])        # wake-phrase scoring on every run
//   stt.last()                   # dict: text, score, span, timings
//   stt.cache(kb) / stt.act_sram(kb, reserve_kb) / stt.info() / stt.rms(pcm[, channels])
#include <string.h>

#include "py/mperrno.h"
#include "py/mphal.h"
#include "py/runtime.h"
#include "stt_engine.h"

static stt_result_t s_last;
static int s_has_last;

static void ensure_open(void)
{
    int rc = stt_engine_open();
    if (rc == -1) mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("stt: no 'model' partition"));
    if (rc == -2) mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("stt: 'model' partition holds no .mmrt image"));
    if (rc) mp_raise_OSError(MP_ENOMEM);
}

// PCM argument: int16 samples, `channels` interleaved (2 for a stereo I2S capture: the
// left slot is used). -> frames per channel.
static int pcm_arg(mp_obj_t o, mp_buffer_info_t *b, int channels)
{
    mp_get_buffer_raise(o, b, MP_BUFFER_READ);
    if (channels < 1 || channels > 8) mp_raise_ValueError(MP_ERROR_TEXT("stt: channels 1..8"));
    int n = (int)(b->len / (2 * channels));
    if (n < 160) mp_raise_ValueError(MP_ERROR_TEXT("stt: need >= 160 samples of int16 PCM"));
    return n;
}

static mp_obj_t stt_open(void)
{
    ensure_open();
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(stt_open_obj, stt_open);

// transcribe(pcm, channels=1) -> str
static mp_obj_t stt_transcribe(size_t n_args, const mp_obj_t *args)
{
    ensure_open();
    mp_buffer_info_t b;
    int ch = n_args > 1 ? mp_obj_get_int(args[1]) : 1;
    int n = pcm_arg(args[0], &b, ch);
    int rc;
    MP_THREAD_GIL_EXIT();
    rc = stt_engine_run_ch((const int16_t *)b.buf, n, ch, &s_last);
    MP_THREAD_GIL_ENTER();
    if (rc) mp_raise_OSError(rc == -3 ? MP_ENOMEM : MP_EINVAL);
    s_has_last = 1;
    return mp_obj_new_str(s_last.text, strlen(s_last.text));
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(stt_transcribe_obj, 1, 2, stt_transcribe);

// start(pcm, channels=1): transcribe in the background (copies the PCM).
static mp_obj_t stt_start(size_t n_args, const mp_obj_t *args)
{
    ensure_open();
    mp_buffer_info_t b;
    int ch = n_args > 1 ? mp_obj_get_int(args[1]) : 1;
    int n = pcm_arg(args[0], &b, ch);
    int rc = stt_engine_start((const int16_t *)b.buf, n, ch);
    if (rc == -1) mp_raise_OSError(MP_EBUSY);
    if (rc) mp_raise_OSError(MP_ENOMEM);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(stt_start_obj, 1, 2, stt_start);

static mp_obj_t stt_busy(void)
{
    return mp_obj_new_bool(stt_engine_busy());
}
static MP_DEFINE_CONST_FUN_OBJ_0(stt_busy_obj, stt_busy);

// The background run's text (also becomes last()); None while busy or if nothing ran.
static mp_obj_t stt_result(void)
{
    int rc = stt_engine_take(&s_last);
    if (rc == -2) return mp_const_none;
    if (rc) mp_raise_OSError(rc == -3 ? MP_ENOMEM : MP_EINVAL);
    s_has_last = 1;
    return mp_obj_new_str(s_last.text, strlen(s_last.text));
}
static MP_DEFINE_CONST_FUN_OBJ_0(stt_result_obj, stt_result);

static void store(mp_obj_t d, const char *k, mp_obj_t v)
{
    mp_obj_dict_store(d, mp_obj_new_str(k, strlen(k)), v);
}

static mp_obj_t stt_last(void)
{
    if (!s_has_last) return mp_const_none;
    mp_obj_t d = mp_obj_new_dict(8);
    store(d, "text", mp_obj_new_str(s_last.text, strlen(s_last.text)));
    store(d, "score", s_last.score > -1e29f ? mp_obj_new_float(s_last.score) : mp_const_none);
    store(d, "span", mp_obj_new_tuple(2, (mp_obj_t[]){mp_obj_new_int(s_last.kw_start), mp_obj_new_int(s_last.kw_end)}));
    store(d, "frames", mp_obj_new_int(s_last.frames));
    store(d, "fe_ms", mp_obj_new_float(s_last.fe_ms));
    store(d, "model_ms", mp_obj_new_float(s_last.model_ms));
    store(d, "dec_ms", mp_obj_new_float(s_last.dec_ms));
    return d;
}
static MP_DEFINE_CONST_FUN_OBJ_0(stt_last_obj, stt_last);

// phrase(list of spellings) or phrase(None): returns the number of piece sequences.
static mp_obj_t stt_phrase(mp_obj_t spellings)
{
    ensure_open();
    if (spellings == mp_const_none) {
        stt_engine_set_phrase(NULL, 0);
        return MP_OBJ_NEW_SMALL_INT(0);
    }
    size_t n;
    mp_obj_t *items;
    mp_obj_get_array(spellings, &n, &items);
    if (n == 0 || n > 64) mp_raise_ValueError(MP_ERROR_TEXT("stt: 1..64 spellings"));
    const char *sp[64];
    for (size_t i = 0; i < n; i++) sp[i] = mp_obj_str_get_str(items[i]);
    int k = stt_engine_set_phrase(sp, (int)n);
    if (k < 0) mp_raise_OSError(MP_ENOMEM);
    return MP_OBJ_NEW_SMALL_INT(k);
}
static MP_DEFINE_CONST_FUN_OBJ_1(stt_phrase_obj, stt_phrase);

// cache(kb): keep up to kb KB of weights in PSRAM (0 releases). Returns bytes cached.
static mp_obj_t stt_cache(mp_obj_t kb)
{
    ensure_open();
    if (stt_engine_busy()) mp_raise_OSError(MP_EBUSY);
    return mp_obj_new_int_from_uint(stt_engine_cache((size_t)mp_obj_get_int(kb) * 1024));
}
static MP_DEFINE_CONST_FUN_OBJ_1(stt_cache_obj, stt_cache);

// act_sram(max_kb, reserve_kb=40): model activations up to max_kb in internal SRAM.
static mp_obj_t stt_act_sram(size_t n_args, const mp_obj_t *args)
{
    size_t reserve = n_args > 1 ? (size_t)mp_obj_get_int(args[1]) : 40;
    stt_engine_act_sram((size_t)mp_obj_get_int(args[0]) * 1024, reserve * 1024);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(stt_act_sram_obj, 1, 2, stt_act_sram);

// rms(pcm, channels=1) -> RMS of channel 0 (int16 samples): cheap level/VAD checks.
static mp_obj_t stt_rms(size_t n_args, const mp_obj_t *args)
{
    mp_buffer_info_t b;
    mp_get_buffer_raise(args[0], &b, MP_BUFFER_READ);
    int ch = n_args > 1 ? mp_obj_get_int(args[1]) : 1;
    if (ch < 1) ch = 1;
    const int16_t *p = (const int16_t *)b.buf;
    size_t n = b.len / (2 * (size_t)ch);
    uint64_t acc = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t v = p[i * ch];
        acc += (uint64_t)(v * v);
    }
    uint32_t m = n ? (uint32_t)(acc / n) : 0, r = 0;
    for (uint32_t bit = 1u << 30; bit; bit >>= 2) {  // integer sqrt
        if (m >= r + bit) {
            m -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
    }
    return MP_OBJ_NEW_SMALL_INT(r);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(stt_rms_obj, 1, 2, stt_rms);

static mp_obj_t stt_info(void)
{
    stt_info_t i;
    stt_engine_info(&i);
    mp_obj_t d = mp_obj_new_dict(8);
    store(d, "model_bytes", mp_obj_new_int_from_uint(i.model_bytes));
    store(d, "cache_bytes", mp_obj_new_int_from_uint(i.cache_bytes));
    store(d, "version", mp_obj_new_int(i.version));
    store(d, "ops", mp_obj_new_int(i.ops));
    store(d, "int4_ops", mp_obj_new_int(i.int4_ops));
    store(d, "internal_free", mp_obj_new_int_from_uint(i.internal_free));
    store(d, "psram_free", mp_obj_new_int_from_uint(i.psram_free));
    return d;
}
static MP_DEFINE_CONST_FUN_OBJ_0(stt_info_obj, stt_info);

static const mp_rom_map_elem_t stt_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_stt)},
    {MP_ROM_QSTR(MP_QSTR_open), MP_ROM_PTR(&stt_open_obj)},
    {MP_ROM_QSTR(MP_QSTR_transcribe), MP_ROM_PTR(&stt_transcribe_obj)},
    {MP_ROM_QSTR(MP_QSTR_start), MP_ROM_PTR(&stt_start_obj)},
    {MP_ROM_QSTR(MP_QSTR_busy), MP_ROM_PTR(&stt_busy_obj)},
    {MP_ROM_QSTR(MP_QSTR_result), MP_ROM_PTR(&stt_result_obj)},
    {MP_ROM_QSTR(MP_QSTR_last), MP_ROM_PTR(&stt_last_obj)},
    {MP_ROM_QSTR(MP_QSTR_phrase), MP_ROM_PTR(&stt_phrase_obj)},
    {MP_ROM_QSTR(MP_QSTR_cache), MP_ROM_PTR(&stt_cache_obj)},
    {MP_ROM_QSTR(MP_QSTR_act_sram), MP_ROM_PTR(&stt_act_sram_obj)},
    {MP_ROM_QSTR(MP_QSTR_info), MP_ROM_PTR(&stt_info_obj)},
    {MP_ROM_QSTR(MP_QSTR_rms), MP_ROM_PTR(&stt_rms_obj)},
};
static MP_DEFINE_CONST_DICT(stt_globals, stt_globals_table);

const mp_obj_module_t stt_module = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t *)&stt_globals,
};

MP_REGISTER_MODULE(MP_QSTR_stt, stt_module);
