// BabyTalk NIFs for AtomVM: module `babytalk` (Erlang) / `BabyTalk` (Elixir).
//
//   babytalk:heap_info() -> [{internal_free, B}, {internal_largest, B}, {psram_free, B}, {psram_largest, B}]
//
// More (transcribe, phrase, record) arrive milestone by milestone; see atomvm/README.md.
#include <sdkconfig.h>
#ifdef CONFIG_AVM_ENABLE_BABYTALK_NIFS

#include <string.h>

#include <atom.h>
#include <defaultatoms.h>
#include <globalcontext.h>
#include <interop.h>
#include <memory.h>
#include <nifs.h>
#include <term.h>
#include <utils.h>

#include <esp_heap_caps.h>

#include "esp32_sys.h"

// AtomVM atom strings: length byte, then the name
static const char *const A_INTERNAL_FREE = "\x0D" "internal_free";
static const char *const A_INTERNAL_LARGEST = "\x10" "internal_largest";
static const char *const A_PSRAM_FREE = "\x0A" "psram_free";
static const char *const A_PSRAM_LARGEST = "\x0D" "psram_largest";

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
    if (UNLIKELY(memory_ensure_free(ctx, 4 * (TUPLE_SIZE(2) + CONS_SIZE)) != MEMORY_GC_OK)) {
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    term list = term_nil();
    for (int i = 3; i >= 0; i--) {
        term t = term_alloc_tuple(2, &ctx->heap);
        term_put_tuple_element(t, 0, globalcontext_make_atom(ctx->global, names[i]));
        term_put_tuple_element(t, 1, term_from_int(vals[i]));
        list = term_list_prepend(t, list, &ctx->heap);
    }
    return list;
}

static const struct Nif heap_info_nif = {.base.type = NIFFunctionType, .nif_ptr = nif_heap_info};

static const struct Nif *babytalk_get_nif(const char *name)
{
    if (!strcmp(name, "babytalk:heap_info/0") || !strcmp(name, "Elixir.BabyTalk:heap_info/0")) {
        return &heap_info_nif;
    }
    return NULL;
}

REGISTER_NIF_COLLECTION(babytalk, NULL, NULL, babytalk_get_nif)

#endif
