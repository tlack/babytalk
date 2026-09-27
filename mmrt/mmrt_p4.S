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

// void mmrt_p4_c1_group(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_p4_c1_t *a)
//
// One 16-output group of a 1x1 conv / matmul over T frames, straight from MMRT's packed layout
// (the S3's mmrt_s3_conv1x1_row, one group): per frame, 16 inputs at a time, each one's 16
// weights times it in one esp.vsmulas.s8.qacc (vector x one lane of q0), which also loads the
// weights two steps ahead -- one instruction per 16 multiply-adds, no repacking.
//   x: T rows of C int8, 16-byte aligned, C a multiple of 16; row t at x + t * (C + x_skip)
//   w: [C][16] int8, 16-byte aligned
//   y: T rows of 16, y_stride bytes apart, 16-byte aligned
//   a: +0 T, +4 C, +8 y_stride, +12 shift (0..13), +16 relu, +20 x_skip (bytes between one
//      frame's inputs and the next's: (stride - 1) * C), +24 bias (NULL, or 16 x 16 int8 that
//      the constants x_bias turn into each lane's bias: one more 16-input chunk), then at +32
//      and +48 the rounding term's factors (as mmrt_p4_dw_row), at +64 x_bias
// The accumulator lanes are 20 bits: the caller makes sure every sum stays under 2^19.
    .text
    .align  4
    .global mmrt_p4_c1_group
    .type   mmrt_p4_c1_group, @function
mmrt_p4_c1_group:
    lw      a4, 0(a3)               // T
    lw      a5, 4(a3)               // C
    lw      a6, 8(a3)               // y_stride
    lw      t4, 12(a3)              // shift (the PIE operands take only some registers: a7 isn't one)
    lw      t1, 16(a3)              // relu
    lw      a7, 20(a3)              // x_skip
    lw      t2, 24(a3)              // bias block, or NULL
    addi    t6, a3, 32
    esp.vld.128.ip  q5, t6, 16      // rnd_a
    esp.vld.128.ip  q4, t6, 16      // rnd_b (lane 0)
    esp.zero.q      q6
    srli    t3, a5, 4               // 16-input chunks per frame
    addi    a6, a6, -16             // (the store advances y by 16 itself)
.Lc1_frame:
    mv      t5, a2                  // w, from the start for every frame
    esp.zero.qacc
    esp.vld.128.ip  q0, a1, 16      // x[0 .. 16)
    esp.vld.128.ip  q1, t5, 16      // w[0]
    esp.vld.128.ip  q2, t5, 16      // w[1]
    // (the S3's loop: each multiply loads the weights two steps ahead, alternating q1 and q2;
    // the last chunk is peeled so nothing is read past the group's weights)
    addi    t0, t3, -1              // chunks before the last
    beqz    t0, .Lc1_last
.Lc1_chunk:
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 0
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 1
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 2
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 3
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 4
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 5
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 6
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 7
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 8
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 9
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 10
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 11
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 12
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 13
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 14  // (loads the next chunk's w[0])
    esp.vsmulas.s8.qacc.ld.incp q0, a1, q2, q0, 15  // (loads the next chunk's x)
    esp.vld.128.ip  q2, t5, 16                      // the next chunk's w[1]
    addi    t0, t0, -1
    bnez    t0, .Lc1_chunk
.Lc1_last:
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 0
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 1
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 2
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 3
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 4
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 5
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 6
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 7
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 8
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 9
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 10
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 11
    esp.vsmulas.s8.qacc.ld.incp q1, t5, q1, q0, 12
    esp.vsmulas.s8.qacc.ld.incp q2, t5, q2, q0, 13
    esp.vsmulas.s8.qacc q1, q0, 14
    esp.vsmulas.s8.qacc q2, q0, 15
    add     a1, a1, a7              // (a strided conv skips frames)
    beqz    t2, .Lc1_biased
    mv      t5, t2                  // the bias: x_bias[k] * block[k][i], k < 16
    addi    t6, a3, 64
    esp.vld.128.ip  q0, t6, 16      // q0 = x_bias
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 0
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 1
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 2
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 3
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 4
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 5
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 6
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 7
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 8
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 9
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 10
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 11
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 12
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 13
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 14
    esp.vld.128.ip      q1, t5, 16
    esp.vsmulas.s8.qacc q1, q0, 15
.Lc1_biased:
    beqz    t4, .Lc1_shifted
    esp.vsmulas.s8.qacc q5, q4, 0   // + 2^(shift - 1)
.Lc1_shifted:
    esp.srcmb.s8.qacc q3, t4, 1     // sat8(QACC >> shift)
    beqz    t1, .Lc1_store
    esp.vmax.s8     q3, q3, q6
.Lc1_store:
    esp.vst.128.ip  q3, a0, 16
    add     a0, a0, a6              // next output row
    addi    a4, a4, -1
    bnez    a4, .Lc1_frame
    ret
    .size   mmrt_p4_c1_group, . - mmrt_p4_c1_group

// void mmrt_p4_cb4_rows(int8_t *dst, const uint8_t *idx, const int8_t *levels, int C)
//
// A CB4 weight group's index rows -> [C][16] int8 (mmrt.h: row c is 8 index bytes, lane 2j the
// low nibble of byte j and lane 2j + 1 the high one; each lane has its own 16-entry table).
// There's no gather, so per row: the 16 lane indices unpacked (a nibble mask, an arithmetic
// shift by 4, then esp.vzip.8 interleaves low and high), and for each of the 16 levels the
// lanes holding it take that level's values: compare, AND, OR.
//   levels: [16 levels][16 lanes] int8 (the group's table transposed), 16-byte aligned
//   dst 16-byte aligned; idx any alignment
    .section .rodata
    .align  4
mmrt_p4_nibble_mask:
    .byte 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
    .text
    .align  4
    .global mmrt_p4_cb4_rows
    .type   mmrt_p4_cb4_rows, @function
mmrt_p4_cb4_rows:
    la      t6, mmrt_p4_nibble_mask
    esp.vld.128.ip  q6, t6, 16      // 0x0F in every lane
    li      a5, 4
    esp.movx.w.sar  a5              // shifts by 4
    li      a4, 1
.Lcb4_row:
    esp.zero.q      q0
    esp.vld.l.64.ip q0, a1, 8       // this row's 8 index bytes (the high half stays 0)
    esp.andq        q1, q0, q6      // low nibbles
    esp.vsr.s32     q2, q0          // >> 4 (the byte above's bits land in the masked-off half)
    esp.andq        q2, q2, q6      // high nibbles
    esp.vzip.8      q1, q2          // q1: lo0 hi0 lo1 hi1 ... lo7 hi7 -- lane i's index
    esp.zero.q      q3              // the row
    esp.zero.q      q4              // the level compared against
    mv      t3, a2
    esp.vld.128.ip      q5, t3, 16      // level 0's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 0
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 1's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 1
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 2's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 2
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 3's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 3
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 4's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 4
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 5's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 5
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 6's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 6
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 7's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 7
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 8's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 8
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 9's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 9
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 10's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 10
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 11's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 11
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 12's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 12
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 13's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 13
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 14's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 14
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vld.128.ip      q5, t3, 16      // level 15's value for each lane
    esp.vcmp.eq.u8      q2, q1, q4      // lanes whose index is 15
    esp.andq            q2, q2, q5
    esp.orq             q3, q3, q2
    esp.vsadds.s8       q4, q4, a4      // next level
    esp.vst.128.ip  q3, a0, 16
    addi    a3, a3, -1
    bnez    a3, .Lcb4_row
    ret
    .size   mmrt_p4_cb4_rows, . - mmrt_p4_cb4_rows
