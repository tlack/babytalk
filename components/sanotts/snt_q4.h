// BabyTalk's 4-bit weight rows for sanoTTS's nano runtime (NANO_WEIGHT_FORMAT 2, built with
// SNT_NANO_W_Q4; tools/sanotts_voice.py writes the blobs; the heart4 voice). A row of n16 inputs
// (a multiple of 16, at least 32) takes n16 bytes, like an int8 row, so the runtime's offsets,
// staging and row indexing are the int8 ones:
//   blocks of 32 inputs, 16 bytes each: byte i's low nibble is input 32b+i, its high nibble
//     input 32b+16+i;
//   when n16 is an odd number of 16s, the last 16 inputs: 16 bytes, low nibbles only;
//   then, 16-byte aligned (at 16 * ceil(n16 / 32)), n16/16 float32 scales, one per 16 inputs;
//   the rest zero.
// Nibbles are unsigned, weight + 8 (weights -7..7). out[r] = sum over the groups g of
// scale[g] * sum_i act[i] * weight[i]: int8 activations, float result.
// SPDX-License-Identifier: MIT
#pragma once

void snt_matvec_q4(const signed char *act, const signed char *w, float *out, int rows, int n16);

// where a row's scales start
static inline int snt_q4_scales_at(int n16) { return 16 * ((n16 + 31) >> 5); }
