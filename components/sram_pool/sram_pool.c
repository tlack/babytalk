#include "sram_pool.h"

#include "esp_heap_caps.h"
#include "sdkconfig.h"

#ifdef CONFIG_SRAM_POOL_IN_PSRAM
#define POOL_CAPS (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#else
#define POOL_CAPS (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#endif

static void *s_mem;
static int s_owner;
static sram_pool_evict_fn s_evict;

int sram_pool_reserve(void)
{
    if (!s_mem) s_mem = heap_caps_aligned_alloc(16, SRAM_POOL_BYTES, POOL_CAPS);
    return s_mem ? 0 : -1;
}

void *sram_pool_acquire(int owner, sram_pool_evict_fn evict)
{
    if (sram_pool_reserve()) return NULL;
    if (s_owner && s_owner != owner) {
        if (s_evict && s_evict()) return NULL;  // holder is busy
    }
    s_owner = owner;
    s_evict = evict;
    return s_mem;
}

void sram_pool_release(int owner)
{
    if (s_owner == owner) {
        s_owner = 0;
        s_evict = NULL;
    }
}
