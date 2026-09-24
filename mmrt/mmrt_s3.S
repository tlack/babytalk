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

// void mmrt_s3_dw_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_s3_dw_t *a)
//
// One output frame of a depthwise conv, all channel groups. The C driver clips the
// taps to the clip (zero padding contributes nothing) and passes the first valid ones:
//   x: input row of the first valid tap (+ g*16 per group), rows C bytes apart
//   w: [groups][K][16] weights, pointing at the first valid tap of group 0
//   a: nk (valid taps >= 1), C (row stride), groups, K16 (group stride of w = K*16),
//      shift (>= 0), relu
// Per group: QACC[i] = sum_k x[k][i] * w[k][i] (VMULAS, 16 lanes), rounded as for 1x1.
    .align 4
    .global mmrt_s3_dw_row
    .type   mmrt_s3_dw_row, @function
mmrt_s3_dw_row:
    entry   a1, 32
    l32i    a6, a5, 0           // nk
    l32i    a7, a5, 4           // C (bytes between input rows)
    l32i    a8, a5, 8           // groups
    l32i    a14, a5, 12         // K16
    l32i    a10, a5, 16         // shift
    l32i    a11, a5, 20         // relu
    movi    a12, 0x101
    slli    a13, a12, 16
    or      a12, a12, a13
    ee.movi.32.q q7, a12, 0
    ee.movi.32.q q7, a12, 1
    ee.movi.32.q q7, a12, 2
    ee.movi.32.q q7, a12, 3
    ee.zero.q q6
    addi    a15, a6, -1         // loop count: nk - 1 (the last tap is peeled)

.Ldw_group:
    mov     a9, a3              // x (this group's 16 channels)
    mov     a13, a4             // w (this group)
    ee.zero.qacc
    ee.vld.128.xp q0, a9, a7    // x tap 0
    ee.vld.128.ip q1, a13, 16   // w tap 0
    loopgtz a15, .Ldw_last
    ee.vmulas.s8.qacc.ld.ip q1, a13, 16, q0, q1     // acc += x*w; next w
    ee.vld.128.xp q0, a9, a7                        // next x
.Ldw_last:
    ee.vmulas.s8.qacc q0, q1

    beqz    a10, .Ldw_shift0
    addi    a12, a10, -1
    ee.srcmb.s8.qacc q3, a12, 0
    movi    a12, 1
    ee.movi.32.q q3, a12, 0
    ee.vsmulas.s8.qacc q7, q3, 0
    ee.srcmb.s8.qacc q3, a12, 0
    j       .Ldw_rounded
.Ldw_shift0:
    ee.srcmb.s8.qacc q3, a10, 0
.Ldw_rounded:
    beqz    a11, .Ldw_store
    ee.vmax.s8 q3, q3, q6
.Ldw_store:
    ee.vst.128.ip q3, a2, 16
    addi    a3, a3, 16          // next channel group
    add     a4, a4, a14
    addi    a8, a8, -1
    bnez    a8, .Ldw_group
    retw
    .size   mmrt_s3_dw_row, .-mmrt_s3_dw_row

// void mmrt_s3_tail_row(int8_t *y, const int8_t *a, const int8_t *s, const int8_t *r,
//                       const mmrt_s3_tail_t *t)
//
// Fused Citrinet block tail for one time row (all channel groups):
//   m = s ? sat8(round_half_up(a[i] * s[i] >> shift)) : a[i]     (SE scale, shift >= 1)
//   v = r ? sat8(m + r[i]) : m                                    (residual add)
//   y = relu ? max(v, 0) : v
// t: groups, shift, relu. s/r may be NULL. All pointers 16-byte aligned.
    .align 4
    .global mmrt_s3_tail_row
    .type   mmrt_s3_tail_row, @function
mmrt_s3_tail_row:
    entry   a1, 32
    l32i    a8, a6, 0           // groups   (t: 5th argument, a6)
    l32i    a10, a6, 4          // shift
    l32i    a11, a6, 8          // relu
    movi    a12, 0x101
    slli    a13, a12, 16
    or      a12, a12, a13
    ee.movi.32.q q7, a12, 0
    ee.movi.32.q q7, a12, 1
    ee.movi.32.q q7, a12, 2
    ee.movi.32.q q7, a12, 3
    ee.zero.q q6
    addi    a14, a10, -1        // shift - 1
    movi    a15, 1
.Ltail_group:
    ee.vld.128.ip q3, a3, 16    // a
    beqz    a4, .Ltail_nomul
    ee.vld.128.ip q1, a4, 16    // s
    ee.zero.qacc
    ee.vmulas.s8.qacc q3, q1    // QACC[i] = a[i] * s[i]
    ee.srcmb.s8.qacc q3, a14, 0
    ee.movi.32.q q3, a15, 0
    ee.vsmulas.s8.qacc q7, q3, 0
    ee.srcmb.s8.qacc q3, a15, 0
.Ltail_nomul:
    beqz    a5, .Ltail_noadd
    ee.vld.128.ip q2, a5, 16    // r
    ee.vadds.s8 q3, q3, q2
.Ltail_noadd:
    beqz    a11, .Ltail_store
    ee.vmax.s8 q3, q3, q6
.Ltail_store:
    ee.vst.128.ip q3, a2, 16
    addi    a8, a8, -1
    bnez    a8, .Ltail_group
    retw
    .size   mmrt_s3_tail_row, .-mmrt_s3_tail_row

// void mmrt_s3_colsum16(const int8_t *x, int T, int C, uint8_t *qacc_out)
//
// Column sums of 16 channels over T rows (rows C bytes apart): QACC[i] += x[t][i] * 1.
// Stores QACC as the 64-byte packed form (two halves of 8 x 20-bit lanes, each half
// padded to 32 bytes) for C to unpack. |sum| must stay below 2^19.
    .align 4
    .global mmrt_s3_colsum16
    .type   mmrt_s3_colsum16, @function
mmrt_s3_colsum16:
    entry   a1, 32
    movi    a12, 0x101
    slli    a13, a12, 16
    or      a12, a12, a13
    ee.movi.32.q q5, a12, 0     // q5 = 1 in every byte
    ee.movi.32.q q5, a12, 1
    ee.movi.32.q q5, a12, 2
    ee.movi.32.q q5, a12, 3
    ee.zero.qacc
    loopgtz a3, .Lcs_done
    ee.vld.128.xp q0, a2, a4
    ee.vmulas.s8.qacc q0, q5
.Lcs_done:
    ee.st.qacc_l.l.128.ip a5, 16
    ee.st.qacc_l.h.32.ip  a5, 16
    ee.st.qacc_h.l.128.ip a5, 16
    ee.st.qacc_h.h.32.ip  a5, 16
    retw
    .size   mmrt_s3_colsum16, .-mmrt_s3_colsum16
