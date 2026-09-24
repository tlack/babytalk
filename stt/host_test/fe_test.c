// Host build of stt_core.c: raw s16le mono PCM -> normalized features (float32 [T][80])
// and the quantized static window (int16 [1600][80]), for export/check_frontend.py.
//   fe_test in.raw feats.f32 window.s16 exponent
#include <stdio.h>
#include <stdlib.h>
#include "../main/stt_core.h"

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    long n = ftell(f) / 2;
    fseek(f, 0, SEEK_SET);
    int16_t *pcm = malloc(n * 2);
    fread(pcm, 2, n, f);
    fclose(f);
    int T = stt_num_frames((int)n);
    float *feats = malloc(sizeof(float) * T * 80);
    float *scratch = malloc(sizeof(float) * stt_scratch_floats());
    stt_features(pcm, (int)n, feats, scratch);
    f = fopen(argv[2], "wb"); fwrite(feats, sizeof(float), (size_t)T * 80, f); fclose(f);
    if (T <= STT_WIN_FRAMES) {
        int16_t *win = malloc(2 * STT_WIN_FRAMES * 80);
        stt_fill_quant(feats, T, win, atoi(argv[4]));
        f = fopen(argv[3], "wb"); fwrite(win, 2, STT_WIN_FRAMES * 80, f); fclose(f);
    }
    printf("%d\n", T);
    return 0;
}
