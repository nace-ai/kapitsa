#include "common.h"
#include "iq.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- Nemotron 3.5 Bolt's 8-row IQ2_XXS products on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5) ----
//
// nh_x2_v8 (nemotron/nh_x_iq2.metal) runs a verify's IQ2_XXS shared expert (5376 x 4096 up, 4096 x 5376 down) as
// 8x8x8 simdgroup MMAs, one simdgroup per 8 weight rows: on the M5 33-38 us per call with the weights in the cache
// (tools/nh_x2t_bench --same), bound by the decode and the MMA chain. Here nh_x4t_v8_l0's design (nh_v8t.metal: 16
// weight rows per simdgroup, one matmul2d of M = 16 weight rows, N = 8 token rows, K = 64 per 64-input group, KS
// simdgroups taking K's 256-input blocks in turn, their sums added in a fixed order) with the IQ2_XXS decode: a lane
// holds 4 consecutive inputs of a 16-input block (nh_v8t.metal's operand layout), half a grid point. Its 4 magnitude
// bytes (one 4-byte word of the grid) become two half2 (elements 0, 2 and 1, 3) as fp16 subnormals b 2^-24 with the
// signs in bit 15, times the sub-block scale 2 s4 + 1 (exact: b (2 s4 + 1) <= 1333; the matrix unit takes subnormal
// halves as they are, which tools/nh_x2t_bench's check against an fp64 product confirms). The matrix unit sums a
// 256-input block's 4 groups in fp32 into an accumulator of its own, scaled once by the block's half scale d
// (w = d (2 s4 + 1) b / 8: the post-scale 2^24 / 8 = 2^21). 22.5-23 us per call there (variants measured within 2%
// of it: the 8-byte grid gather, the grid in threadgroup memory, normal halves (x 2^10); 32 weight rows per
// simdgroup 29 us; the decode without the signs 19 us, without the decode at all 15 us). Row-independent (a token
// row's outputs depend on its inputs only); bitwise not the same as nh_x2_v8 (summation order).
//
// Kernel (NhV8tP [[4]], X half [R][x_ld], Y per p.epi as nh_x4t_v8_l0; codes [[1]] (K / 4 bytes per row), block
// scales [[2]] (half, K / 256 per row)):
//   nh_x2t_v8   grid ceil(N / 16) x (32 x KS): KS simdgroups (1..8) split K's 256-input blocks
// Requirements: K % 256 == 0.

[[kernel, max_total_threads_per_threadgroup(256)]] void nh_x2t_v8(
    const device half* X [[buffer(0)]], const device uchar* q [[buffer(1)]], const device uchar* s [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8tP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 16, NJ = NT / 8;
  threadgroup float red[7 * (NT / 2) * 32];
  const uint N = p.N, K = p.K, nb = K / 256, n0 = tg * NT;
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc =
      matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.get_left_input_cooperative_tensor<half, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
      (device half*)X, dextents<int32_t, 2>(int32_t(K), int32_t(p.R)), array<int32_t, 2>{1, int32_t(p.x_ld)});
  auto Rx0 = tX.slice<64, 8>(0, 0);
  auto acc = op.get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  auto part = op.get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  constexpr uint cap = NT / 2;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < cap; ++i) acc[i] = 0.0f;
  const device uchar* qr[NJ];
  const device half* sr[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    qr[j] = q + size_t(row) * (K / 4);
    sr[j] = (const device half*)s + size_t(row) * (K / 256);
  }
  constant uint* gt = (constant uint*)nh_iq2xxs_grid;  // a grid point's elements 0-3 at 2 i, 4-7 at 2 i + 1
  // the lane's inputs 4 c .. 4 c + 3 of a 16-input block: grid point (c >> 1) of the block's pair, its half c & 1
  const uint gh = c >> 1, hh = c & 1;
  for (uint b = sg; b < nb; b += nsg) {
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < cap; ++i) part[i] = 0.0f;
    for (uint gg = 0; gg < 4; ++gg) {
      const uint k64 = b * 256 + gg * 64;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) {
        const uint4 a = *(const device uint4*)(qr[j] + k64 / 4);  // sub-blocks 2 gg, 2 gg + 1
        _Pragma("clang loop unroll(full)")
        for (int jb = 0; jb < 4; ++jb) {
          const uint ax = jb < 2 ? a.x : a.z, ay = jb < 2 ? a.y : a.w;
          const uint gi = 2 * uint(jb & 1) + gh;
          const uint w = gt[2 * extract_bits(ax, 8 * gi, 8) + hh];
          // the half's 4 sign bits (elements 4..7: bits 4-6 and the parity bit), spread to bits 15 / 31 of the two
          // half2 words by one multiply each (the shifted copies do not overlap: no carries)
          const uint s7 = extract_bits(ay, 7 * gi, 7);
          const uint s4 = extract_bits(s7 | (popcount(s7) << 7), 4 * hh, 4);
          const uint m02 = (s4 * 0x20008000u) & 0x80008000u, m13 = (s4 * 0x10004000u) & 0x80008000u;
          const half sc = half(ushort(2u * (ay >> 28) + 1u));
          const half2 h02 = as_type<half2>((w & 0x00FF00FFu) | m02) * sc;
          const half2 h13 = as_type<half2>(((w >> 8) & 0x00FF00FFu) | m13) * sc;
          const int i = jb * 8 + j * 4;
          Lw[i] = h02.x;
          Lw[i + 1] = h13.x;
          Lw[i + 2] = h02.y;
          Lw[i + 3] = h13.y;
        }
      }
      auto Rx = tX.slice<64, 8>(int32_t(k64), 0);
      op.run(Lw, Rx, part);
    }
    // destination element j * 4 + kk: weight row r + 8 j
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const float d = float(sr[j][b]);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(d, part[j * 4 + kk], acc[j * 4 + kk]);
    }
  }
  // fixed-order sum of the k-slices (simdgroup 0 adds 1, 2, ... in turn)
  if (nsg > 1) {
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < nsg; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;  // padding token rows
  const float post = p.gs * 2097152.0f;
  const uint t0 = 4 * (lane & 1);
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk)
      if (t0 + kk < p.R) nh_v8t_store(Y, p, n, t0 + kk, acc[j * 4 + kk] * post);
  }
}
