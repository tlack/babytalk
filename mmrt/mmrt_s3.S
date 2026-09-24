// mmrt PIE (ESP32-S3 SIMD) kernels. Each must match its mmrt_ref.c counterpart bit for
// bit; `ktest` in the stt firmware checks that on random data.
//
// The 1x1-conv inner loop and the QACC rounding sequence are adapted from ESP-DL's
// tie728_s8_conv2d_11c16 / tie728_s8_vector_round_result
// (https://github.com/espressif/esp-dl, MIT License, Copyright (c) 2021 Espressif Systems).

    .text
    .align 4

// void mmrt_s3_conv1x1_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_s3_c1_t *a)
//
// One output frame of a 1x1 conv: y[0 .. 16*groups) from input row x[0 .. C).
//   x: C int8, 16-byte aligned, C a multiple of 16
//   w: [groups][C][16] int8, 16-byte aligned
//   a: bias_q (QACC-packed biases, 64 bytes per group, or NULL), c16m1 = C/16 - 1,
//      groups, shift (>= 0), relu
// Per group: QACC = bias; QACC[i] += x[c] * w[c][i] (one VSMULAS = 16 MACs);
// y = sat8(round_half_up(QACC >> shift)); relu -> max(y, 0).
    .global mmrt_s3_conv1x1_row
    .type   mmrt_s3_conv1x1_row, @function
mmrt_s3_conv1x1_row:
    entry   a1, 32
    l32i    a6, a5, 0           // bias_q
    l32i    a7, a5, 4           // c16m1
    l32i    a8, a5, 8           // groups
    l32i    a10, a5, 12         // shift
    l32i    a11, a5, 16         // relu
    movi    a12, 0x101          // q7 = 0x01 in every byte (the +1 of the rounding step)
    slli    a13, a12, 16
    or      a12, a12, a13
    ee.movi.32.q q7, a12, 0
    ee.movi.32.q q7, a12, 1
    ee.movi.32.q q7, a12, 2
    ee.movi.32.q q7, a12, 3
    ee.zero.q q6                // relu floor

.Lc1_group:
    mov     a9, a3              // x, again for every group
    ee.zero.qacc
    beqz    a6, .Lc1_nobias
    ee.ld.qacc_l.l.128.ip a6, 16
    ee.ld.qacc_l.h.32.ip  a6, 16
    ee.ld.qacc_h.l.128.ip a6, 16
    ee.ld.qacc_h.h.32.ip  a6, 16
.Lc1_nobias:
    ee.vld.128.ip q0, a9, 16    // x[0..16)
    ee.vld.128.ip q1, a4, 16    // w[0]
    ee.vld.128.ip q2, a4, 16    // w[1]
    loopgtz a7, .Lc1_last
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 0
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 1
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 2
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 3
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 4
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 5
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 6
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 7
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 8
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 9
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 10
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 11
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 12
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 13
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 14
    ee.vsmulas.s8.qacc.ld.incp q0, a9, q2, q0, 15   // next x chunk
    ee.vld.128.ip q2, a4, 16
.Lc1_last:                      // last 16 input channels: no loads past this group's w
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 0
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 1
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 2
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 3
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 4
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 5
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 6
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 7
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 8
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 9
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 10
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 11
    ee.vsmulas.s8.qacc.ld.incp q1, a4, q1, q0, 12
    ee.vsmulas.s8.qacc.ld.incp q2, a4, q2, q0, 13
    ee.vsmulas.s8.qacc q1, q0, 14
    ee.vsmulas.s8.qacc q2, q0, 15

    // round half up: shift by (s-1), add 1 to every lane, shift by 1 (saturating to int8)
    beqz    a10, .Lc1_shift0
    addi    a12, a10, -1
    ee.srcmb.s8.qacc q3, a12, 0
    movi    a12, 1
    ee.movi.32.q q3, a12, 0     // q3[0] = 1
    ee.vsmulas.s8.qacc q7, q3, 0 // QACC[i] += 1 * 1
    ee.srcmb.s8.qacc q3, a12, 0
    j       .Lc1_rounded
.Lc1_shift0:
    ee.srcmb.s8.qacc q3, a10, 0
.Lc1_rounded:
    beqz    a11, .Lc1_store
    ee.vmax.s8 q3, q3, q6
.Lc1_store:
    ee.vst.128.ip q3, a2, 16
    addi    a8, a8, -1
    bnez    a8, .Lc1_group
    retw
    .size   mmrt_s3_conv1x1_row, .-mmrt_s3_conv1x1_row
