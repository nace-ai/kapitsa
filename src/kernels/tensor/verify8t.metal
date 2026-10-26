#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- 8-row verify GEMM on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5) ----
//
// qmm8 (verify8.metal) runs the 8-row verify's projections as 8x8x8 simdgroup MMAs, which execute on the
// shader ALUs: bench_mma measures 15 TFLOP/s on the 40-core M5 Max, the FMA peak, so the 35B's 8-row
// GEMMs are bound by the MMAs (GDN in_proj 12352x2048: 27 us of MMAs alone at that rate, 25-27 us of weight
// reads). The M5's per-core matrix units (tensor_ops::matmul2d) run a 16 x 8 x 64 product at 32 TFLOP/s
// (8 rows fill half of their 16-row tile: 64 TFLOP/s at 16), which leaves these GEMMs bound by DRAM.
//
// qmm8t: per simdgroup NT weight rows x a k-slice; per 64-group one matmul2d (M = NT weight rows, N = 8
// token rows, K = 64) of the integer codes (uint8, the left operand, a cooperative tensor filled from
// registers) by the half inputs (right operand, a device tensor), then the qmm8 epilogue in fp32
// (y += s * dot + b * sum(x) per group): the same numerics as qmm8 (exact codes, half inputs, fp32 group
// dots), with the matrix unit's summation order inside a group.
//
// Operand layouts (probed with get_multidimensional_index; bench_tq probe): lane role c = (lane & 1) |
// ((lane >> 3) & 1) << 1 holds k-slots 4c + 16 jb + kk (jb, kk < 4) of weight rows r + 8 j, r = ((lane >> 1)
// & 3) | ((lane >> 4) & 1) << 2, as left elements (j >> 1) * 32 + jb * 8 + (j & 1) * 4 + kk; the destination
// element j * 4 + kk is token row 4 (lane & 1) + kk of weight row r + 8 j (lanes with bit 3 set hold the
// padding rows 8-15). The k-slots are a permutation of the group's inputs, applied to the input table (the
// "T table", written by x8_prep2 / gdn_gnorm_x8t): slot 4c + 16 jb + kk holds input 16 c + 8 (jb >> 1) +
// 2 kk + (jb & 1), so a lane's codes of a row are 8 contiguous bytes (one uint2 load) and the 4 slots of a
// jb are one masked word (codes 2 kk + (jb & 1) of word jb >> 1: (w >> 4 (jb & 1)) & 0x0F0F0F0F).

struct Qmm8tParams {  // = Qmm8Params (verify8.metal)
  uint K;
  uint R;
  uint nseg;
  uint ks;
  uint N[4];
  uint y_ld[4];
  uint add[4];
  uint tg_begin[5];
};

// PF: prefetch the next group's codes and coefficients before this group's product.
// Row tiles as qmm8 (verify8.metal v8_tiles: KFlash tree verifies of 16 / 32 rows): tg_begin[nseg] threadgroups per
// 8-row tile, tiles fastest; tile t reads T table rows 8 t .. 8 t + 7 and their sums, and writes those rows.
template <int NT, int PF, int MAXKS>
NI_INLINE void qmm8t_impl(const device uint* const thread* Wv, const device ushort* const thread* Sv,
                          const device ushort* const thread* Bv, device float* const thread* Yv,
                          const device half* T, const device float* sums, constant Qmm8tParams& p,
                          threadgroup float* red, uint tg, uint sg, uint lane, uint ntgs) {
  const uint per = p.tg_begin[p.nseg], nt = ntgs > per ? ntgs / per : 1u;
  const uint tile = nt > 1 ? tg % nt : 0u;
  if (nt > 1) {
    tg /= nt;
    T += tile * 8 * p.K;
    sums += tile * (p.K / 64) * 8;
  }
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint N = p.N[si], n0 = (tg - p.tg_begin[si]) * NT, gpr = p.K / 64, wpr = p.K / 8;
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(p.K, 8));
  auto Rx0 = tX.template slice<64, 8>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* wp[NJ];
  const device ushort* sp[NJ];
  const device ushort* bp[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    wp[j] = Wv[si] + row * wpr + 2 * c;
    sp[j] = Sv[si] + row * gpr;
    bp[j] = Bv[si] + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  uint2 wn[NJ];
  ushort sn[NJ], bn[NJ];
  auto load = [&](uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      wn[j] = *(const device uint2*)(wp[j] + g * 8);
      sn[j] = sp[j][g];
      bn[j] = bp[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint2 w[NJ];
    ushort sc[NJ], bc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) w[j] = wn[j], sc[j] = sn[j], bc[j] = bn[j];
    if (PF && g + 1 < g1) load(g + 1);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        const uint u = ((jb >> 1 ? w[j].y : w[j].x) >> (4 * (jb & 1))) & 0x0F0F0F0Fu;
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        Lw[i] = uchar(u);
        Lw[i + 1] = uchar(u >> 8);
        Lw[i + 2] = uchar(u >> 16);
        Lw[i + 3] = uchar(u >> 24);
      }
    auto Rx = tX.template slice<64, 8>(g * 64, 0);
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
    op.run(Lw, Rx, d);
    const float4 sx = *(const device float4*)(sums + g * 8 + t0);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx[kk], acc[j * 4 + kk]));
    }
  }
  // fixed-order sum of the k-slices (simdgroup 0 adds 1, 2, ... in turn)
  if (p.ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < p.ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;  // padding rows
  device float* Y = Yv[si];
  const uint y_ld = p.y_ld[si];
  const bool add = p.add[si] != 0;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = 8 * tile + t0 + kk;
      if (t < p.R) {
        device float* dst = Y + t * y_ld + n;
        *dst = add ? *dst + acc[j * 4 + kk] : acc[j * 4 + kk];
      }
    }
  }
}

// ---- Tile-pair variant for KFlash wide verifies (16 / 32 rows): one matmul2d of 16 token rows (two 8-row tiles)
// per weight load, instead of one per tile with every tile re-reading the weights. Pair q = tiles 2q, 2q + 1: T
// table rows 16 q .. 16 q + 15, destination element j * 4 + kk = token 4 (lane & 1) + 8 ((lane >> 3) & 1) + kk of the
// pair (lanes with bit 3 set hold the second tile; the sums of that tile are read at its own offset). tg_begin[nseg]
// threadgroups per pair, pairs fastest. A pair whose second tile is not live computes it from stale table rows and
// writes rows no later kernel reads.
template <int NT, int PF, int MAXKS>
NI_INLINE void qmm8t2_impl(const device uint* const thread* Wv, const device ushort* const thread* Sv,
                           const device ushort* const thread* Bv, device float* const thread* Yv,
                           const device half* T, const device float* sums, constant Qmm8tParams& p,
                           threadgroup float* red, uint tg, uint sg, uint lane, uint ntgs) {
  const uint per = p.tg_begin[p.nseg], np = ntgs > per ? ntgs / per : 1u;
  const uint pair = tg % np;
  tg /= np;
  T += pair * 16 * p.K;
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint N = p.N[si], n0 = (tg - p.tg_begin[si]) * NT, gpr = p.K / 64, wpr = p.K / 8;
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 16, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(p.K, 16));
  auto Rx0 = tX.template slice<64, 16>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* wp[NJ];
  const device ushort* sp[NJ];
  const device ushort* bp[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    wp[j] = Wv[si] + row * wpr + 2 * c;
    sp[j] = Sv[si] + row * gpr;
    bp[j] = Bv[si] + row * gpr;
  }
  const uint t0 = 4 * (lane & 1) + 8 * ((lane >> 3) & 1);
  sums += (2 * pair + ((lane >> 3) & 1)) * gpr * 8 + 4 * (lane & 1);
  uint2 wn[NJ];
  ushort sn[NJ], bn[NJ];
  auto load = [&](uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      wn[j] = *(const device uint2*)(wp[j] + g * 8);
      sn[j] = sp[j][g];
      bn[j] = bp[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint2 w[NJ];
    ushort sc[NJ], bc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) w[j] = wn[j], sc[j] = sn[j], bc[j] = bn[j];
    if (PF && g + 1 < g1) load(g + 1);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        const uint u = ((jb >> 1 ? w[j].y : w[j].x) >> (4 * (jb & 1))) & 0x0F0F0F0Fu;
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        Lw[i] = uchar(u);
        Lw[i + 1] = uchar(u >> 8);
        Lw[i + 2] = uchar(u >> 16);
        Lw[i + 3] = uchar(u >> 24);
      }
    auto Rx = tX.template slice<64, 16>(g * 64, 0);
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
    op.run(Lw, Rx, d);
    const float4 sx = *(const device float4*)(sums + g * 8);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx[kk], acc[j * 4 + kk]));
    }
  }
  if (p.ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < p.ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  device float* Y = Yv[si];
  const uint y_ld = p.y_ld[si];
  const bool add = p.add[si] != 0;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = 16 * pair + t0 + kk;
      if (t < p.R) {
        device float* dst = Y + t * y_ld + n;
        *dst = add ? *dst + acc[j * 4 + kk] : acc[j * 4 + kk];
      }
    }
  }
}

// ---- Tile-pair variant for the drafter's 16-row drafting block (DFlash2 B=16) ----
//
// The drafter's X8 tables of two 8-row tiles are 16 rows apart (values rows 0-7, T table rows 8-15 of each tile; see
// dfl_put2), so unlike qmm8t2 a pair cannot be one 16-row matmul: this variant runs the two 8-row products of a pair
// from one weight load (Lw) and two separate T tables / sums blocks, which halves the drafter's weight traffic and its
// dispatch count for the 16-row block. Destinations: rows t0 + kk of the pair's first tile and 8 + t0 + kk of the
// second, in one contiguous 16-row Y (the drafter's y rows are contiguous over a block).
template <int NT, int PF, int MAXKS>
NI_INLINE void qmm8t2d_impl(const device uint* const thread* Wv, const device ushort* const thread* Sv,
                            const device ushort* const thread* Bv, device float* const thread* Yv,
                            const device half* T0, const device float* sums0, const device half* T1,
                            const device float* sums1, constant Qmm8tParams& p, threadgroup float* red, uint tg,
                            uint sg, uint lane) {
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint N = p.N[si], n0 = (tg - p.tg_begin[si]) * NT, gpr = p.K / 64, wpr = p.K / 8;
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX0 = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T0, dextents<int32_t, 2>(p.K, 8));
  auto tX1 = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T1, dextents<int32_t, 2>(p.K, 8));
  auto Rx0 = tX0.template slice<64, 8>(0, 0);
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc0.get_capacity(); ++i) acc0[i] = 0.0f, acc1[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* wp[NJ];
  const device ushort* sp[NJ];
  const device ushort* bp[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    wp[j] = Wv[si] + row * wpr + 2 * c;
    sp[j] = Sv[si] + row * gpr;
    bp[j] = Bv[si] + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  uint2 wn[NJ];
  ushort sn[NJ], bn[NJ];
  auto load = [&](uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      wn[j] = *(const device uint2*)(wp[j] + g * 8);
      sn[j] = sp[j][g];
      bn[j] = bp[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint2 w[NJ];
    ushort sc[NJ], bc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) w[j] = wn[j], sc[j] = sn[j], bc[j] = bn[j];
    if (PF && g + 1 < g1) load(g + 1);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        const uint u = ((jb >> 1 ? w[j].y : w[j].x) >> (4 * (jb & 1))) & 0x0F0F0F0Fu;
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        Lw[i] = uchar(u);
        Lw[i + 1] = uchar(u >> 8);
        Lw[i + 2] = uchar(u >> 16);
        Lw[i + 3] = uchar(u >> 24);
      }
    const float4 sx0 = *(const device float4*)(sums0 + g * 8 + t0);
    const float4 sx1 = *(const device float4*)(sums1 + g * 8 + t0);
    {
      auto Rxa = tX0.template slice<64, 8>(g * 64, 0);
      auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rxa), float>();
      op.run(Lw, Rxa, d);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) {
        const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) acc0[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx0[kk], acc0[j * 4 + kk]));
      }
    }
    {
      auto Rxb = tX1.template slice<64, 8>(g * 64, 0);
      auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rxb), float>();
      op.run(Lw, Rxb, d);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) {
        const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) acc1[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx1[kk], acc1[j * 4 + kk]));
      }
    }
  }
  // fixed-order sum of the k-slices (simdgroup 0 adds 1, 2, ... in turn), the pair's two tiles in turn
  if (p.ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc0[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg)
      for (uint j = 1; j < p.ks; ++j)
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < cap; ++i) acc0[i] += red[((j - 1) * cap + i) * 32 + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc1[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg)
      for (uint j = 1; j < p.ks; ++j)
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < cap; ++i) acc1[i] += red[((j - 1) * cap + i) * 32 + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;  // the k-slices are summed in simdgroup 0 (as qmm8t)
  }
  if ((lane >> 3) & 1) return;  // padding rows
  device float* Y = Yv[si];
  const uint y_ld = p.y_ld[si];
  const bool add = p.add[si] != 0;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = t0 + kk;
      if (t < p.R) {
        device float* dst = Y + t * y_ld + n;
        *dst = add ? *dst + acc0[j * 4 + kk] : acc0[j * 4 + kk];
      }
      const uint t2 = 8 + t;
      device float* dst2 = Y + t2 * y_ld + n;
      if (t2 < p.R) *dst2 = add ? *dst2 + acc1[j * 4 + kk] : acc1[j * 4 + kk];
    }
  }
}

#define NI_QMM8T(NAME, NT, PF, MAXKS)                                                                         \
  [[kernel]] void NAME(const device uint* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],           \
                       const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                \
                       const device uint* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],           \
                       const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                \
                       const device uint* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],           \
                       const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],              \
                       const device uint* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],         \
                       const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],              \
                       const device half* T [[buffer(16)]], const device float* sums [[buffer(17)]],         \
                       constant Qmm8tParams& p [[buffer(18)]], uint tg [[threadgroup_position_in_grid]],     \
                       uint ntgs [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint lane [[thread_index_in_simdgroup]]) {                                            \
    threadgroup float red[(MAXKS - 1) * (NT / 2) * 32];                                                      \
    const device uint* Wv[4] = {W0, W1, W2, W3};                                                              \
    const device ushort* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device ushort* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmm8t_impl<NT, PF, MAXKS>(Wv, Sv, Bv, Yv, T, sums, p, red, tg, sg, lane, ntgs);                           \
  }

NI_QMM8T(qmm8t_b4_nt16, 16, 0, 8)
NI_QMM8T(qmm8t_b4_nt32, 32, 0, 8)
NI_QMM8T(qmm8tp_b4_nt16, 16, 1, 8)
NI_QMM8T(qmm8tp_b4_nt32, 32, 1, 8)

#define NI_QMM8T2(NAME, NT, PF, MAXKS)                                                                         \
  [[kernel]] void NAME(const device uint* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],           \
                       const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                \
                       const device uint* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],           \
                       const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                \
                       const device uint* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],           \
                       const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],              \
                       const device uint* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],         \
                       const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],              \
                       const device half* T [[buffer(16)]], const device float* sums [[buffer(17)]],         \
                       constant Qmm8tParams& p [[buffer(18)]], uint tg [[threadgroup_position_in_grid]],     \
                       uint ntgs [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint lane [[thread_index_in_simdgroup]]) {                                            \
    threadgroup float red[(MAXKS - 1) * (NT / 2) * 32];                                                      \
    const device uint* Wv[4] = {W0, W1, W2, W3};                                                              \
    const device ushort* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device ushort* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmm8t2_impl<NT, PF, MAXKS>(Wv, Sv, Bv, Yv, T, sums, p, red, tg, sg, lane, ntgs);                           \
  }

NI_QMM8T2(qmm8t2_b4_nt16, 16, 0, 8)
NI_QMM8T2(qmm8t2_b4_nt32, 32, 0, 8)
NI_QMM8T2(qmm8t2p_b4_nt16, 16, 1, 8)
NI_QMM8T2(qmm8t2p_b4_nt32, 32, 1, 8)

#define NI_QMM8T2D(NAME, NT, PF, MAXKS)                                                                        \
  [[kernel]] void NAME(const device uint* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],           \
                       const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                \
                       const device uint* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],           \
                       const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                \
                       const device uint* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],           \
                       const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],              \
                       const device uint* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],         \
                       const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],              \
                       const device half* T0 [[buffer(16)]], const device float* sums0 [[buffer(17)]],       \
                       const device half* T1 [[buffer(18)]], const device float* sums1 [[buffer(19)]],       \
                       constant Qmm8tParams& p [[buffer(20)]], uint tg [[threadgroup_position_in_grid]],     \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float red[(MAXKS - 1) * (NT / 2) * 32];                                                      \
    const device uint* Wv[4] = {W0, W1, W2, W3};                                                              \
    const device ushort* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device ushort* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmm8t2d_impl<NT, PF, MAXKS>(Wv, Sv, Bv, Yv, T0, sums0, T1, sums1, p, red, tg, sg, lane);                  \
  }

NI_QMM8T2D(qmm8t2d_b4_nt16, 16, 0, 8)
NI_QMM8T2D(qmm8t2d_b4_nt32, 32, 0, 8)
NI_QMM8T2D(qmm8t2dp_b4_nt16, 16, 1, 8)
NI_QMM8T2D(qmm8t2dp_b4_nt32, 32, 1, 8)

// ---- qmm6t: the f6g64 (6-bit group-64) 8-row verify GEMM, qmm8t's structure (Flash-Next's dense weights) ----
//
// Same matmul2d (NT weight rows, 8 token rows, 64), operand layouts, T tables and epilogue as qmm8t; two
// differences:
// - the codes are 6 bits: a row's 16 codes of a group are a 96-bit run (3 words) at word 12 g + 3 c (wfmt.metal's
//   wf_codes16<6>: little-endian bitstream, 16 codes in 3 words), so lane c loads 3 words per row and group and
//   extracts with qmt6_dec's shift-straddle (only codes 5 and 10 of the 16 cross a word boundary; the jb / kk
//   loops are fully unrolled, so both the branch and the shifts fold to constants).
// - LO: two-term inputs. The matrix unit's right operand is half, so the T table rounds x to 11 bits; with the
//   lo T table (x8_prep2's write_t_lo) the same codes run a second product against lo(x) and the epilogue adds
//   s * dot_lo + b * sum(lo), which keeps the drift at the ~1e-6 two-term scale instead of half-rounded x.
//   Every product is exact in fp32 (a 6-bit integer times a half is <= 17 significant bits), so this is the
//   qmm8 numerics with the input split, not a precision change. LO = 0 runs the single hi term (qmm6t1*).
//
// The Qmm8tParams block and the segment / row-tile handling are qmm8t's verbatim. Static threadgroup memory is
// red = (MAXKS - 1) * (NT / 2) * 32 floats: nt32 with MAXKS 8 is 7 * 16 * 32 * 4 = 14,336 B, nt16 7,168 B, both
// within the 16,384 B this macOS's layer doubles against the 32,768 B cap (AGENTS.md).
template <int NT, int PF, int MAXKS, int LO>
NI_INLINE void qmm6t_impl(const device uint* const thread* Wv, const device ushort* const thread* Sv,
                          const device ushort* const thread* Bv, device float* const thread* Yv,
                          const device half* T, const device float* sums, const device half* Tlo,
                          const device float* sums_lo, constant Qmm8tParams& p, threadgroup float* red, uint tg,
                          uint sg, uint lane, uint ntgs) {
  const uint per = p.tg_begin[p.nseg], nt = ntgs > per ? ntgs / per : 1u;
  const uint tile = nt > 1 ? tg % nt : 0u;
  if (nt > 1) {
    tg /= nt;
    T += tile * 8 * p.K;
    sums += tile * (p.K / 64) * 8;
    Tlo += tile * 8 * p.K;
    sums_lo += tile * (p.K / 64) * 8;
  }
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint N = p.N[si], n0 = (tg - p.tg_begin[si]) * NT, gpr = p.K / 64;
  const uint wpr = p.K * 3 / 16;  // 6-bit codes: a row is K * 6 / 32 words (K a multiple of 64)
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(p.K, 8));
  auto Rx0 = tX.template slice<64, 8>(0, 0);
  // the lo table as a second right operand (LO = 0 never runs it; the host binds the buffers either way)
  auto tL = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)Tlo, dextents<int32_t, 2>(p.K, 8));
  auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* wp[NJ];
  const device ushort* sp[NJ];
  const device ushort* bp[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    wp[j] = Wv[si] + row * wpr + 3 * c;  // this lane's 3 words of every group's 96-bit run
    sp[j] = Sv[si] + row * gpr;
    bp[j] = Bv[si] + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  uint wn[NJ][3];
  ushort sn[NJ], bn[NJ];
  auto load = [&](uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const device uint* wg = wp[j] + g * 12;
      wn[j][0] = wg[0];
      wn[j][1] = wg[1];
      wn[j][2] = wg[2];
      sn[j] = sp[j][g];
      bn[j] = bp[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint w[NJ][3];
    ushort sc[NJ], bc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      w[j][0] = wn[j][0];
      w[j][1] = wn[j][1];
      w[j][2] = wn[j][2];
      sc[j] = sn[j];
      bc[j] = bn[j];
    }
    if (PF && g + 1 < g1) load(g + 1);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        // the run's codes 8 (jb >> 1) + 2 kk + (jb & 1) at bit 6 * code of the 96-bit window (qmm8t's slot order)
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) {
          const uint b = 6u * uint(8 * (jb >> 1) + 2 * kk + (jb & 1)), wi = b >> 5, sh = b & 31;
          uint v = w[j][wi] >> sh;
          if (sh + 6 > 32) v |= w[j][wi + 1] << (32 - sh);  // codes 5 and 10 cross the word boundary
          Lw[i + kk] = uchar(v & 63u);
        }
      }
    auto Rx = tX.template slice<64, 8>(g * 64, 0);
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
    op.run(Lw, Rx, d);
    const float4 sx = *(const device float4*)(sums + g * 8 + t0);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx[kk], acc[j * 4 + kk]));
    }
    if (LO) {  // the lo term: the same codes against lo(x), and b * sum(lo)
      auto Rxl = tL.template slice<64, 8>(g * 64, 0);
      auto dl = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rxl), float>();
      op.run(Lw, Rxl, dl);
      const float4 sxl = *(const device float4*)(sums_lo + g * 8 + t0);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) {
        const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(s, dl[j * 4 + kk], fma(b, sxl[kk], acc[j * 4 + kk]));
      }
    }
  }
  // fixed-order sum of the k-slices (simdgroup 0 adds 1, 2, ... in turn)
  if (p.ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < p.ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;  // padding rows
  device float* Y = Yv[si];
  const uint y_ld = p.y_ld[si];
  const bool add = p.add[si] != 0;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = 8 * tile + t0 + kk;
      if (t < p.R) {
        device float* dst = Y + t * y_ld + n;
        *dst = add ? *dst + acc[j * 4 + kk] : acc[j * 4 + kk];
      }
    }
  }
}

// Buffers 18 / 19 carry the lo T table and the lo group sums (params move to 20); the LO = 0 instantiations take
// them too and never read them, so every dispatch binds all four table buffers.
#define NI_QMM6T(NAME, NT, PF, MAXKS, LO)                                                                          \
  [[kernel]] void NAME(const device uint* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],               \
                       const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                  \
                       const device uint* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],             \
                       const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                  \
                       const device uint* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],             \
                       const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],                \
                       const device uint* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],           \
                       const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],                \
                       const device half* T [[buffer(16)]], const device float* sums [[buffer(17)]],           \
                       const device half* Tlo [[buffer(18)]], const device float* sums_lo [[buffer(19)]],      \
                       constant Qmm8tParams& p [[buffer(20)]], uint tg [[threadgroup_position_in_grid]],       \
                       uint ntgs [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup float red[(MAXKS - 1) * (NT / 2) * 32];                                                          \
    const device uint* Wv[4] = {W0, W1, W2, W3};                                                                 \
    const device ushort* Sv[4] = {S0, S1, S2, S3};                                                               \
    const device ushort* Bv[4] = {B0, B1, B2, B3};                                                               \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                      \
    qmm6t_impl<NT, PF, MAXKS, LO>(Wv, Sv, Bv, Yv, T, sums, Tlo, sums_lo, p, red, tg, sg, lane, ntgs);            \
  }

// static threadgroup memory (red): nt16 (8 - 1) * 8 * 32 * 4 = 7,168 B; nt32 (8 - 1) * 16 * 32 * 4 = 14,336 B;
// both <= 16,384 B (the layer doubles them against the 32,768 B cap)
NI_QMM6T(qmm6t_b6_nt16, 16, 0, 8, 1)
NI_QMM6T(qmm6t_b6_nt32, 32, 0, 8, 1)
NI_QMM6T(qmm6tp_b6_nt16, 16, 1, 8, 1)
NI_QMM6T(qmm6tp_b6_nt32, 32, 1, 8, 1)
NI_QMM6T(qmm6t1_b6_nt16, 16, 0, 8, 0)
NI_QMM6T(qmm6t1_b6_nt32, 32, 0, 8, 0)
NI_QMM6T(qmm6t1p_b6_nt16, 16, 1, 8, 0)
NI_QMM6T(qmm6t1p_b6_nt32, 32, 1, 8, 0)
