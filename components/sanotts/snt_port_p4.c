// BabyTalk's ESP32-P4 port of sanoTTS's snt_par_run, snt_now_us and snt_scratch_id, in place of
// snt_port_esphome.c (whose snt_par_run is serial) for the larger voices (prepare.sh ... heart,
// heart4): each range is split between the caller and a helper task on the other core. The
// runtime was written for this: every stage it hands to snt_par_run writes disjoint outputs
// and keeps its temporaries in scratch bank snt_scratch_id() (the helper's is bank 1, anyone
// else's bank 0; prepare.sh keeps both banks for these voices, in PSRAM with the rest of
// sanoTTS's static buffers). The same PCM as serial.
// SPDX-License-Identifier: MIT
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "snt_port.h"

#define HELPER_STACK (8 * 1024)
#define MIN_SPLIT 2  // ranges shorter than this run on the caller alone

static TaskHandle_t s_helper;
static SemaphoreHandle_t s_go, s_done;
static snt_par_fn s_f;
static void *s_ctx;
static int s_lo, s_hi;

static void helper(void *arg)
{
    (void)arg;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        s_f(s_lo, s_hi, s_ctx);
        xSemaphoreGive(s_done);
    }
}

// the helper on whichever core the caller isn't on; 0 if it can't be had (then all serial)
static int helper_ready(void)
{
    if (s_helper) return 1;
    if (!s_go) s_go = xSemaphoreCreateBinary();
    if (!s_done) s_done = xSemaphoreCreateBinary();
    if (!s_go || !s_done) return 0;
    int core = xPortGetCoreID() ? 0 : 1;
    if (xTaskCreatePinnedToCoreWithCaps(helper, "snt_par", HELPER_STACK, NULL, uxTaskPriorityGet(NULL), &s_helper,
                                        core, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) != pdPASS) {
        s_helper = NULL;
        return 0;
    }
    return 1;
}

void snt_par_run(snt_par_fn f, int n, void *ctx)
{
    if (n < MIN_SPLIT || !helper_ready()) {
        f(0, n, ctx);
        return;
    }
    int h = n / 2;
    s_f = f;
    s_ctx = ctx;
    s_lo = 0;
    s_hi = h;
    xSemaphoreGive(s_go);
    f(h, n, ctx);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

int64_t snt_now_us(void) { return esp_timer_get_time(); }

int snt_scratch_id(void) { return s_helper && xTaskGetCurrentTaskHandle() == s_helper; }
