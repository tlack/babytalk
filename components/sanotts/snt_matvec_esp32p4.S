// BabyTalk's ESP32-P4 int8 kernels for sanoTTS's snt_port.h contract (snt_kernels_esp32p4.c):
// matrix x vector on the P4's vector unit (the Xesppie extension), one output row per pass, 16
// int8 multiply-adds per esp.vmulas.s8.xacc into the 40-bit XACC (read back as int32: the
// reference's plain int32 accumulation, no saturation).
//
// void snt_p4_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out, int rows, int chunks)
//   out[r] = act . w[r], w rows contiguous (stride 16 * chunks); act and w 16-byte aligned,
//   chunks >= 1, rows >= 1. Reads exactly rows * 16 * chunks bytes of w.
//   Up to 6 chunks (96 inputs: sanoTTS's 48 and 80) the activations stay in q0..q5 for every
//   row, as sanoTTS's S3 kernels do; longer ones are read again for each row.
// SPDX-License-Identifier: MIT
    .text
    .align  4
    .global snt_p4_matvec_s8
    .type   snt_p4_matvec_s8, @function
snt_p4_matvec_s8:
    li      t0, 6
    bgt     a4, t0, .Lg_row
    // ---- short rows: act in q0..q5
    mv      t3, a0
    esp.vld.128.ip  q0, t3, 16
    li      t0, 2
    blt     a4, t0, .Ls_loaded
    esp.vld.128.ip  q1, t3, 16
    li      t0, 3
    blt     a4, t0, .Ls_loaded
    esp.vld.128.ip  q2, t3, 16
    li      t0, 4
    blt     a4, t0, .Ls_loaded
    esp.vld.128.ip  q3, t3, 16
    li      t0, 5
    blt     a4, t0, .Ls_loaded
    esp.vld.128.ip  q4, t3, 16
    li      t0, 6
    blt     a4, t0, .Ls_loaded
    esp.vld.128.ip  q5, t3, 16
.Ls_loaded:
.Ls_row:
    esp.zero.xacc
    esp.vld.128.ip  q7, a1, 16
    esp.vmulas.s8.xacc q0, q7
    li      t0, 2
    blt     a4, t0, .Ls_done
    esp.vld.128.ip  q7, a1, 16
    esp.vmulas.s8.xacc q1, q7
    li      t0, 3
    blt     a4, t0, .Ls_done
    esp.vld.128.ip  q7, a1, 16
    esp.vmulas.s8.xacc q2, q7
    li      t0, 4
    blt     a4, t0, .Ls_done
    esp.vld.128.ip  q7, a1, 16
    esp.vmulas.s8.xacc q3, q7
    li      t0, 5
    blt     a4, t0, .Ls_done
    esp.vld.128.ip  q7, a1, 16
    esp.vmulas.s8.xacc q4, q7
    li      t0, 6
    blt     a4, t0, .Ls_done
    esp.vld.128.ip  q7, a1, 16
    esp.vmulas.s8.xacc q5, q7
.Ls_done:
    esp.movx.r.xacc.l t5
    sw      t5, 0(a2)
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, .Ls_row
    ret
    // ---- long rows: act streamed again for each row
.Lg_row:
    mv      t3, a0
    mv      t4, a4
    esp.zero.xacc
.Lg_chunk:
    esp.vld.128.ip  q0, t3, 16
    esp.vld.128.ip  q7, a1, 16
    esp.vmulas.s8.xacc q0, q7
    addi    t4, t4, -1
    bnez    t4, .Lg_chunk
    esp.movx.r.xacc.l t5
    sw      t5, 0(a2)
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, .Lg_row
    ret
    .size   snt_p4_matvec_s8, . - snt_p4_matvec_s8
