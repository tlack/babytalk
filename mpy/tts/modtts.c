// MicroPython module `tts`: on-device text to speech (sanoTTS nano voice, 24 kHz).
//
//   import tts
//   pcm = tts.say("Hello from a three dollar chip.")   # bytearray: int16 mono @ 24 kHz
//   tts.last()          # {'phonemes', 'frames', 'samples', 'g2p_ms', 'synth_ms', 'rtf', ...}
//   tts.RATE            # 24000
#include <string.h>

#include "esp_heap_caps.h"
#include "py/mperrno.h"
#include "py/runtime.h"
#include "tts_engine.h"

static tts_stats_t s_st;
static int s_has;

// say(text, volume=0.8, stereo=False) -> bytearray of int16 PCM at 24 kHz, peak-normalized
// to volume (fraction of full scale); stereo=True
// duplicates each sample into L+R (I2S codecs that garble MONO, e.g. the ES8311).
// Releases the GIL while it runs.
static mp_obj_t tts_say_(size_t n_args, const mp_obj_t *args)
{
    const char *text = mp_obj_str_get_str(args[0]);
    float vol = n_args > 1 ? mp_obj_get_float(args[1]) : 0.8f;
    int stereo = n_args > 2 && mp_obj_is_true(args[2]);
    int16_t *pcm = NULL;
    int n = 0, rc;
    MP_THREAD_GIL_EXIT();
    rc = tts_say(text, vol, &pcm, &n, &s_st);
    MP_THREAD_GIL_ENTER();
    s_has = 1;
    if (rc == -1) mp_raise_ValueError(MP_ERROR_TEXT("tts: no pronounceable text"));
    if (rc == -2) mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("tts: no 84KB internal block for the arena"));
    if (rc) mp_raise_OSError(rc == -3 ? MP_ENOMEM : MP_EIO);
    mp_obj_t out;
    if (stereo) {
        int16_t *st = m_new(int16_t, (size_t)n * 2);
        for (int i = 0; i < n; i++) st[2 * i] = st[2 * i + 1] = pcm[i];
        out = mp_obj_new_bytearray_by_ref((size_t)n * 4, st);
    } else {
        out = mp_obj_new_bytearray((size_t)n * 2, pcm);
    }
    heap_caps_free(pcm);
    return out;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(tts_say_obj, 1, 3, tts_say_);

// reserve(): reserve the shared internal SRAM pool (84KB, also used by stt) now -- call early,
// e.g. in boot.py, before internal RAM fragments. release(): drop tts's claim on it.
static mp_obj_t tts_reserve_(void)
{
    if (tts_reserve()) mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("tts: no 84KB internal block for the arena"));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(tts_reserve_obj, tts_reserve_);

static mp_obj_t tts_release_(void)
{
    tts_release();
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(tts_release_obj, tts_release_);

static void store(mp_obj_t d, const char *k, mp_obj_t v)
{
    mp_obj_dict_store(d, mp_obj_new_str(k, strlen(k)), v);
}

static mp_obj_t tts_last(void)
{
    if (!s_has) return mp_const_none;
    mp_obj_t d = mp_obj_new_dict(10);
    float secs = s_st.samples / (float)TTS_SAMPLE_RATE;
    store(d, "phonemes", mp_obj_new_int(s_st.phonemes));
    store(d, "frames", mp_obj_new_int(s_st.frames));
    store(d, "samples", mp_obj_new_int(s_st.samples));
    store(d, "g2p_ms", mp_obj_new_float(s_st.g2p_ms));
    store(d, "synth_ms", mp_obj_new_float(s_st.synth_ms));
    store(d, "rtf", mp_obj_new_float(secs > 0 ? s_st.synth_ms / 1000.0f / secs : 0));
    store(d, "arena_peak", mp_obj_new_int_from_uint(s_st.arena_peak));
    store(d, "macs_simd", mp_obj_new_int_from_ll(s_st.macs_simd));
    store(d, "macs_scalar", mp_obj_new_int_from_ll(s_st.macs_scalar));
    return d;
}
static MP_DEFINE_CONST_FUN_OBJ_0(tts_last_obj, tts_last);

static const mp_rom_map_elem_t tts_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_tts)},
    {MP_ROM_QSTR(MP_QSTR_say), MP_ROM_PTR(&tts_say_obj)},
    {MP_ROM_QSTR(MP_QSTR_last), MP_ROM_PTR(&tts_last_obj)},
    {MP_ROM_QSTR(MP_QSTR_reserve), MP_ROM_PTR(&tts_reserve_obj)},
    {MP_ROM_QSTR(MP_QSTR_release), MP_ROM_PTR(&tts_release_obj)},
    {MP_ROM_QSTR(MP_QSTR_RATE), MP_ROM_INT(TTS_SAMPLE_RATE)},
};
static MP_DEFINE_CONST_DICT(tts_globals, tts_globals_table);

const mp_obj_module_t tts_module = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t *)&tts_globals,
};

MP_REGISTER_MODULE(MP_QSTR_tts, tts_module);
