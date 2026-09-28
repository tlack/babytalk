// The ESP32-P4 vector part of snt_matvec_q4 (snt_q4.c, layout in snt_q4.h).
//
// void snt_p4_q4_dots(const int8_t *act, const uint8_t *w, int32_t *dots, int rows, int n16,
//                     const uint8_t mask[16])
//   For each row (stride n16 bytes) and each group of 16 inputs, XACC after that group: a
//   running int32 sum of act . (weight + 8). Per 32 inputs: one 16-byte load, the low nibbles
//   by an AND with the mask and the high ones by a 4-bit shift right (32-bit lanes, SAR = 4;
//   the bits shifted in from the next byte are the AND's to drop), two esp.vmulas.s8.xacc.
//   act, w 16-byte aligned; n16 a multiple of 16, at least 32; rows >= 1.
// SPDX-License-Identifier: MIT
    .text
    .align  4
    .global snt_p4_q4_dots
    .type   snt_p4_q4_dots, @function
snt_p4_q4_dots:
    li      t5, 4
    esp.movx.w.sar t5
    esp.vld.128.ip  q6, a5, 0
    srli    t1, a4, 5                   // t1: blocks of 32
    srli    t2, a4, 4
    andi    t2, t2, 1                   // t2: a last group of 16
.Lq_row:
    mv      t3, a0                      // activations
    mv      t4, a1                      // this row's nibbles
    esp.zero.xacc
    mv      t6, t1
.Lq_block:
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q3, t3, 16
    esp.andq        q1, q0, q6
    esp.vsr.u32     q2, q0
    esp.vmulas.s8.xacc q3, q1
    esp.movx.r.xacc.l t5
    esp.andq        q2, q2, q6
    esp.vld.128.ip  q3, t3, 16
    sw      t5, 0(a2)
    esp.vmulas.s8.xacc q3, q2
    esp.movx.r.xacc.l t5
    sw      t5, 4(a2)
    addi    a2, a2, 8
    addi    t6, t6, -1
    bnez    t6, .Lq_block
    beqz    t2, .Lq_next
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q3, t3, 16
    esp.andq        q1, q0, q6
    esp.vmulas.s8.xacc q3, q1
    esp.movx.r.xacc.l t5
    sw      t5, 0(a2)
    addi    a2, a2, 4
.Lq_next:
    add     a1, a1, a4
    addi    a3, a3, -1
    bnez    a3, .Lq_row
    ret
    .size   snt_p4_q4_dots, . - snt_p4_q4_dots
