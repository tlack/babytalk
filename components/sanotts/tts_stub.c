// No sanoTTS sources (SANOTTS_SRC unset: a speech-to-text-only firmware): the same API, every
// call answering that there's no voice -- so BabyTalk's NIFs build unchanged, and `say` fails
// cleanly (-4) instead of the link.
#include "tts_engine.h"

int tts_reserve(void) { return 0; }
void tts_release(void) {}
int tts_say(const char *text, float volume, int16_t **pcm, int *n, tts_stats_t *st)
{
    (void)text; (void)volume; (void)st;
    *pcm = NULL;
    *n = 0;
    return -4;
}
void tts_set_length_scale(float scale) { (void)scale; }
