// One contiguous block of internal SRAM, reserved once and shared by engines that never
// run at the same time (on the Waveshare S3-CAM the mic and speaker share one I2S port,
// so listening and speaking alternate anyway). Two engines that each need a large
// contiguous internal block could not otherwise both live next to MicroPython: once
// something small lands in the middle of the free region, no big block comes back.
//
// Owners hold the pool until another owner asks for it; the holder's evict callback is
// then called and must stop using it (return 0), or refuse (nonzero: still busy).
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SRAM_POOL_BYTES (84 * 1024 + 512)  // sanoTTS nano arena (84464) > mmrt staging (73728)

typedef int (*sram_pool_evict_fn)(void);

// Reserve the block now (0) -- call early, before internal RAM fragments.
int sram_pool_reserve(void);

// The block for `owner` (any unique non-zero id), evicting the current holder if needed.
// NULL: no block (reserve failed) or the holder refused.
void *sram_pool_acquire(int owner, sram_pool_evict_fn evict);

// Give it up (no-op unless `owner` holds it).
void sram_pool_release(int owner);

#ifdef __cplusplus
}
#endif
