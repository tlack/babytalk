// MMRT's dot product on the ESP32-P4's vector unit (PIE, the Xesppie extension): 16 int8
// multiply-accumulates per esp.vmulas.s8.xacc into the 40-bit XACC, loading the next 16 of `a`
// as it multiplies. Plain branch loops: the zero-overhead loop (esp.lp.setup) gave wrong sums
// here when re-armed with a different count (a depthwise frame after a clipped one), so none
// of these kernels use it.
//
// int32_t mmrt_dot_s8(const int8_t *a, const int8_t *b, int n16)
//   sum of a[i] * b[i], i < 16 * n16; n16 >= 1, a and b 16-byte aligned. Reads exactly
//   16 * n16 bytes of each. (Every sum MMRT asks for fits 32 bits: its reference ops add in int32.)
    .text
    .align  4
    .global mmrt_dot_s8
    .type   mmrt_dot_s8, @function
mmrt_dot_s8:
    esp.zero.xacc
    esp.vld.128.ip      q0, a0, 16              // a[0]
    addi                a2, a2, -1
    beqz                a2, 2f
1:  esp.vld.128.ip      q1, a1, 16              // b[i]
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1 // xacc += a[i] . b[i]; q0 = a[i + 1]
    addi                a2, a2, -1
    bnez                a2, 1b
2:  esp.vld.128.ip      q1, a1, 16              // b[n16 - 1]
    esp.vmulas.s8.xacc  q0, q1
    esp.movx.r.xacc.l   a0
    ret
    .size   mmrt_dot_s8, . - mmrt_dot_s8

// void mmrt_p4_dw_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_p4_dw_t *a)
//
// One output frame of a depthwise conv, all channel groups: the S3's mmrt_s3_dw_row on the
// P4's PIE. The C driver clips the taps to the clip and passes the first valid ones:
//   x: input row of the first valid tap (+ g*16 per group), rows C bytes apart
//   w: [groups][K][16] weights, pointing at the first valid tap of group 0
//   a: nk (valid taps >= 1), C (row stride), groups, K16 (group stride of w = K*16),
//      shift (0..13), relu, then at +32 and +48 two vectors whose product rnd_a * rnd_b[0]
//      is 2^(shift - 1) (the rounding term)
// Per group: QACC[i] = sum_k x[k][i] * w[k][i] (16 lanes); + 2^(shift - 1); then
// esp.srcmb.s8.qacc with its last operand 1, which shifts and saturates to int8 (with 0 it
// keeps the low byte: the S3's ee.srcmb clamps, the P4's needs asking). Rounding half up is
// then (acc + 2^(shift - 1)) >> shift, as the reference (the S3 kernel's in-place-shift trick
// doesn't carry over). All pointers 16-byte aligned.
    .text
    .align  4
    .global mmrt_p4_dw_row
    .type   mmrt_p4_dw_row, @function
mmrt_p4_dw_row:
    // (the PIE instructions take only some registers: t0-t2 aren't among them)
    lw      a4, 0(a3)               // nk
    addi    t3, a4, -1              // taps looped: nk - 1 (the last is peeled)
    lw      a4, 16(a3)              // shift
    lw      a5, 4(a3)               // C
    lw      a6, 8(a3)               // groups
    lw      a7, 12(a3)              // K16
    lw      t1, 20(a3)              // relu
    addi    t6, a3, 32
    esp.vld.128.ip  q5, t6, 16      // rnd_a
    esp.vld.128.ip  q4, t6, 16      // rnd_b (lane 0)
    esp.zero.q      q6              // zero (for the ReLU)
.Ldw_group:
    mv      t4, a1                  // x (this group's 16 channels)
    mv      t5, a2                  // w (this group)
    esp.zero.qacc
    esp.vld.128.xp  q0, t4, a5      // x tap 0
    esp.vld.128.ip  q1, t5, 16      // w tap 0
    beqz    t3, .Ldw_last
    mv      t0, t3
.Ldw_tap:
    esp.vmulas.s8.qacc.ld.ip q1, t5, 16, q0, q1     // acc += x * w; next w
    esp.vld.128.xp  q0, t4, a5                      // next x
    addi    t0, t0, -1
    bnez    t0, .Ldw_tap
.Ldw_last:
    esp.vmulas.s8.qacc q0, q1
    beqz    a4, .Ldw_shifted
    esp.vsmulas.s8.qacc q5, q4, 0   // + 2^(shift - 1)
.Ldw_shifted:
    esp.srcmb.s8.qacc q3, a4, 1     // q3 = sat8(QACC >> shift)
    beqz    t1, .Ldw_store
    esp.vmax.s8     q3, q3, q6
.Ldw_store:
    esp.vst.128.ip  q3, a0, 16
    addi    a1, a1, 16              // next channel group
    add     a2, a2, a7
    addi    a6, a6, -1
    bnez    a6, .Ldw_group
    ret
    .size   mmrt_p4_dw_row, . - mmrt_p4_dw_row
