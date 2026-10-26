// 8-row 4-bit GEMM on Metal 4 tensor_ops (the M5's GPU matrix units): the experiments behind qmm8t
// (src/kernels/tensor/verify8t.metal), against qmm8snb on the same rotating weights (interleaved runs).
//   bench_tq <N> <K> [reps=10] [variant prefix]   PRE=tw,twp,... (prefixes) KS=<k-slices> ACC=1 (all outputs vs fp64)
//   bench_tq probe <tq_probe_*>     operand / destination layouts of matmul2d's cooperative tensors, per lane
//   bench_tq gnorm [Hv] [R]         gdn_gnorm_x8t's T table against its X8 table
//   bench_tq stats <kernel> ...     code size, stack and GPRs (as ni_compile)
//   bench_tq router [N] [K] [reps]  the MoE router's precise 8-bit GEMM (qmm8x2_b8_nf1) vs tw8x2 on the matrix units
// Variants: tq = per-group matmul2d of device tensors (half inputs x uint4b codes); tw = codes as a uint8 cooperative
// left operand, inputs a device tensor (= qmm8t), twp prefetching, tw{a1,a2,a3,a4} ablations (no matrix op;
// cache-resident weights; loads only; weight loads only); twl = 32-bit offsets instead of row pointers (69 -> 59 GPRs);
// tc / tcp{,2,3,4} = both operands cooperative (16 x 16 x 32 ops), prefetch depth (CS=<column blocks per threadgroup>).
// GDN in_proj 12352x2048 (us): qmm8snb nf1 ks8 34.9, tq 49.5, tw 31.0, tc 30.8, twl 30.9, twa2 18.5 (the compute side),
// twa3 27.4 (the loads: 518 GB/s, bench_bw's ceiling at this size). Deeper prefetch (tcp2-4: 120+ GPRs) and column
// blocks instead of k-slices (cs 2-8) were slower on every 35B shape; router 257x2048: qmm8x2 5.4 us, tw8x2 4.6 (ks 16).
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace ni;

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct TqParams {
  uint K, N, ks, R, cs;
};

NI_TQ_KERNELS
)MSL";

// kernel bodies (macro-instantiated per NT)
static const char* kKern = R"MSL(
template <int NT>
inline void tq_body(const device uchar* W, const device ushort* S, const device ushort* Bq, const device half* X,
                    const device float* sums, device float* Y, constant TqParams& p, threadgroup float* red, uint tg,
                    uint sg, uint lane) {
  const uint n0 = tg * NT, gpr = p.K / 64, gper = gpr / p.ks, g0 = sg * gper, g1 = g0 + gper;
  auto tW = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>((device uchar*)W,
                                                                               dextents<int32_t, 2>(p.K, p.N));
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)X, dextents<int32_t, 2>(p.K, 8));
  constexpr auto desc = matmul2d_descriptor(8, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto sX0 = tX.template slice<64, 8>(0, 0);
  auto sW0 = tW.template slice<64, NT>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(sX0), decltype(sW0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  for (uint g = g0; g < g1; ++g) {
    auto sX = tX.template slice<64, 8>(g * 64, 0);
    auto sW = tW.template slice<64, NT>(g * 64, n0);
    auto d = op.template get_destination_cooperative_tensor<decltype(sX), decltype(sW), float>();
    op.run(sX, sW, d);
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < d.get_capacity(); ++i) {
      if (!d.is_valid_element(i)) continue;
      const auto ix = d.get_multidimensional_index(i);
      const uint n = n0 + ix[0], t = ix[1];
      const float s = as_type<float>(uint(S[n * gpr + g]) << 16), b = as_type<float>(uint(Bq[n * gpr + g]) << 16);
      acc[i] = fma(s, d[i], fma(b, sums[g * 8 + t], acc[i]));
    }
  }
  // fixed-order sum of the k-slices
  if (p.ks > 1) {
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) red[((sg - 1) * acc.get_capacity() + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < p.ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] += red[((j - 1) * acc.get_capacity() + i) * 32 + lane];
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    const uint n = n0 + ix[0], t = ix[1];
    if (t < p.R && n < p.N) Y[t * p.N + n] = acc[i];
  }
}
#define TQ(NAME, NT)                                                                                              \
  [[kernel]] void NAME(const device uchar* W [[buffer(0)]], const device ushort* S [[buffer(1)]],                  \
                       const device ushort* Bq [[buffer(2)]], const device half* X [[buffer(3)]],                  \
                       const device float* sums [[buffer(4)]], device float* Y [[buffer(5)]],                      \
                       constant TqParams& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],               \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {      \
    threadgroup float red[(NT > 32 ? 4 : 8) * (NT / 2) * 32];                                                    \
    tq_body<NT>(W, S, Bq, X, sums, Y, p, red, tg, sg, lane);                                                       \
  }

// layout probe: per lane and element, (valid, idx0, idx1) of the left / right input and destination cooperative tensors
template <int NT, bool TR, typename RT, int MM = 8, typename LT = half, int KK = 64>
inline void tq_probe_body(device int* out, uint lane) {
  constexpr auto desc = matmul2d_descriptor(MM, NT, KK, false, TR, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto L = op.template get_left_input_cooperative_tensor<LT, RT, float>();
  auto R = op.template get_right_input_cooperative_tensor<LT, RT, float>();
  auto D = op.template get_destination_cooperative_tensor<decltype(L), decltype(R), float>();
  device int* o = out + lane * 1024;
  o[0] = L.get_capacity(); o[1] = R.get_capacity(); o[2] = D.get_capacity();
  for (uint16_t i = 0; i < L.get_capacity(); ++i) { auto ix = L.get_multidimensional_index(i); o[8 + i * 3] = L.is_valid_element(i); o[9 + i * 3] = ix[0]; o[10 + i * 3] = ix[1]; }
  for (uint16_t i = 0; i < R.get_capacity(); ++i) { auto ix = R.get_multidimensional_index(i); o[400 + i * 3] = R.is_valid_element(i); o[401 + i * 3] = ix[0]; o[402 + i * 3] = ix[1]; }
  for (uint16_t i = 0; i < D.get_capacity(); ++i) { auto ix = D.get_multidimensional_index(i); o[800 + i * 3] = D.is_valid_element(i); o[801 + i * 3] = ix[0]; o[802 + i * 3] = ix[1]; }
}
[[kernel]] void tq_probe_32t_h(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<32, true, half>(out, lane); }
[[kernel]] void tq_probe_32t_u8(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<32, true, uint8_t>(out, lane); }
[[kernel]] void tq_probe_32n_h(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<32, false, half>(out, lane); }
[[kernel]] void tq_probe_16t_h(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<16, true, half>(out, lane); }
[[kernel]] void tq_probe_w32(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<8, true, half, 32, uint8_t>(out, lane); }
[[kernel]] void tq_probe_w64(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<8, true, half, 64, uint8_t>(out, lane); }
[[kernel]] void tq_probe_w16(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<8, true, half, 16, uint8_t>(out, lane); }
[[kernel]] void tq_probe_c16(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<16, true, half, 16, uint8_t, 32>(out, lane); }
[[kernel]] void tq_probe_c32(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<16, true, half, 32, uint8_t, 32>(out, lane); }
template <int NT, typename LT>
inline void tq_probe_wl(device int* out, uint lane) {
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto L = op.template get_left_input_cooperative_tensor<LT, half, float>();
  auto R = op.template get_right_input_cooperative_tensor<LT, half, float>();
  auto D = op.template get_destination_cooperative_tensor<decltype(L), decltype(R), float>();
  device int* o = out + lane * 1024;
  o[0] = L.get_capacity(); o[1] = R.get_capacity(); o[2] = D.get_capacity();
  for (uint16_t i = 0; i < L.get_capacity(); ++i) { auto ix = L.get_multidimensional_index(i); o[8 + i * 3] = L.is_valid_element(i); o[9 + i * 3] = ix[0]; o[10 + i * 3] = ix[1]; }
  for (uint16_t i = 0; i < R.get_capacity(); ++i) { auto ix = R.get_multidimensional_index(i); o[400 + i * 3] = R.is_valid_element(i); o[401 + i * 3] = ix[0]; o[402 + i * 3] = ix[1]; }
  for (uint16_t i = 0; i < D.get_capacity(); ++i) { auto ix = D.get_multidimensional_index(i); o[800 + i * 3] = D.is_valid_element(i); o[801 + i * 3] = ix[0]; o[802 + i * 3] = ix[1]; }
}
[[kernel]] void tq_probe_wl16_h(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_wl<16, half>(out, lane); }
[[kernel]] void tq_probe_wl16_u8(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_wl<16, uint8_t>(out, lane); }
[[kernel]] void tq_probe_wl32_h(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_wl<32, half>(out, lane); }
[[kernel]] void tq_probe_64t_h(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { tq_probe_body<64, true, half>(out, lane); }

// tw: weights as the LEFT cooperative operand (uint8 codes, M = NT weight rows), the 8 rows as the right operand
// (half, from a table in the cooperative layout), one matmul2d per 64-group, fp32 epilogue per group.
// Lane roles (probed): c = k chunk (lane & 1) | ((lane >> 3) & 1) << 1, r = (lane >> 1 & 3) | (lane >> 4 & 1) << 2.
// k-slot (c, jblk, kk) <-> physical input 16 c + 8 (jblk >> 1) + 2 kk + (jblk & 1) of the group, so a lane's codes of a
// weight row are 8 contiguous bytes and 4 slots of a jblk are one masked word.
template <int NT, int PF, int ABL, int ST = 0, int SP = 0, int LN = 0>
inline void tw_body(const device uint* W, const device ushort* S, const device ushort* Bq, const device half* T,
                    const device float* sums, device float* Y, constant TqParams& p, threadgroup float* red, uint tg,
                    uint sg, uint lane) {
  const uint n0 = tg * NT, gpr = p.K / 64, wpr = p.K / 8, gper = gpr / p.ks, g0 = sg * gper, g1 = g0 + gper;
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
    const uint row = n0 + r + 8 * j;
    wp[j] = W + row * wpr + 2 * c;
    sp[j] = S + row * gpr;
    bp[j] = Bq + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  const bool live = ((lane >> 3) & 1) == 0;
  uint2 wn[NJ];
  ushort sn[NJ], bn[NJ];
  uint2 s4[NJ], b4[NJ];
  const uint wo = (n0 + r) * wpr + 2 * c, so = (n0 + r) * gpr;  // LN: 32-bit offsets instead of NJ x 3 pointers
  auto load = [&](uint g) {
    const uint gw = ABL == 2 ? 0 : g;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      if (LN) {
        wn[j] = *(const device uint2*)(W + (wo + j * 8 * wpr + gw * 8));
        sn[j] = S[so + j * 8 * gpr + gw];
        bn[j] = Bq[so + j * 8 * gpr + gw];
        continue;
      }
      wn[j] = *(const device uint2*)(wp[j] + gw * 8);
      if (ABL == 4) { sn[j] = bn[j] = 0; continue; }
      if (SP) {
        if ((g & 3) == 0) {
          s4[j] = *(const device uint2*)(sp[j] + gw);
          b4[j] = *(const device uint2*)(bp[j] + gw);
        }
        const uint sw = (g & 2) ? s4[j].y : s4[j].x, bw = (g & 2) ? b4[j].y : b4[j].x;
        sn[j] = ushort((g & 1) ? sw >> 16 : sw);
        bn[j] = ushort((g & 1) ? bw >> 16 : bw);
      } else {
        sn[j] = sp[j][gw];
        bn[j] = bp[j][gw];
      }
    }
  };
  uint2 w2[NJ];
  ushort s2[NJ], b2[NJ];
  if (PF && g0 < g1) load(g0);
  if (PF == 2 && g0 + 1 < g1) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) w2[j] = wn[j], s2[j] = sn[j], b2[j] = bn[j];
    load(g0 + 1);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) { uint2 tw = wn[j]; wn[j] = w2[j]; w2[j] = tw; ushort ts = sn[j]; sn[j] = s2[j]; s2[j] = ts; ushort tb = bn[j]; bn[j] = b2[j]; b2[j] = tb; }
  }
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint2 w[NJ];
    ushort sc[NJ], bc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) w[j] = wn[j], sc[j] = sn[j], bc[j] = bn[j];
    if (ABL == 3 || ABL == 4) {  // loads only (4: weights only)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j)
        acc[j * 4] += float(as_type<half2>(w[j].x ^ w[j].y ^ (ABL == 3 ? uint(sc[j]) ^ (uint(bc[j]) << 16) : 0u)).x);
      continue;
    }
    if (PF == 1 && g + 1 < g1) load(g + 1);
    if (PF == 2) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) wn[j] = w2[j], sn[j] = s2[j], bn[j] = b2[j];
      if (g + 2 < g1) {
        load(g + 2);
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < NJ; ++j) { uint2 tw = wn[j]; wn[j] = w2[j]; w2[j] = tw; ushort ts = sn[j]; sn[j] = s2[j]; s2[j] = ts; ushort tb = bn[j]; bn[j] = b2[j]; b2[j] = tb; }
      }
    }
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        const uint u = ((jb >> 1 ? w[j].y : w[j].x) >> (4 * (jb & 1))) & 0x0F0F0F0Fu;
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        if (ST) ((thread uint*)&Lw[0])[i / 4] = u;
        else { Lw[i] = uchar(u); Lw[i + 1] = uchar(u >> 8); Lw[i + 2] = uchar(u >> 16); Lw[i + 3] = uchar(u >> 24); }
      }
    }
    auto Rx = tX.template slice<64, 8>(g * 64, 0);
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
    if (ABL == 1) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j)
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) {
          const int i = (j >> 1) * 32 + (j & 1) * 4 + kk;
          d[j * 4 + kk] = float(Lw[i] + Lw[i + 8] + Lw[i + 16] + Lw[i + 24]);
        }
    } else {
      op.run(Lw, Rx, d);
    }
    const float4 sx = *(const device float4*)(sums + g * 8 + t0);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx[kk], acc[j * 4 + kk]));
    }
  }
  if (p.ks > 1) {
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) red[((sg - 1) * acc.get_capacity() + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint jj = 1; jj < p.ks; ++jj)
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] += red[((jj - 1) * acc.get_capacity() + i) * 32 + lane];
  }
  if (!live) return;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j)
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = t0 + kk, n = n0 + r + 8 * j;
      if (t < p.R) Y[t * p.N + n] = acc[j * 4 + kk];
    }
}
#define TW(NAME, NT, PF, ABL, ...) TWX(NAME, NT, (NT > 32 ? 4 : NT > 16 ? 8 : 16), PF, ABL, ##__VA_ARGS__)
#define TWX(NAME, NT, MAXKS, PF, ABL, ...)                                                                                   \
  [[kernel]] void NAME(const device uint* W [[buffer(0)]], const device ushort* S [[buffer(1)]],                   \
                       const device ushort* Bq [[buffer(2)]], const device half* T [[buffer(3)]],                  \
                       const device float* sums [[buffer(4)]], device float* Y [[buffer(5)]],                      \
                       constant TqParams& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],               \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {      \
    threadgroup float red[(MAXKS - 1) * (NT / 2) * 32];                                                          \
    tw_body<NT, PF, ABL, ##__VA_ARGS__>(W, S, Bq, T, sums, Y, p, red, tg, sg, lane);                               \
  }

// tw8x2: the router's precise GEMM (8-bit codes, two-term inputs hi + lo) on the matrix units: codes as the uint8 left
// operand (16 bytes per row and lane), two matmul2d per group (hi, lo T tables) into one destination.
template <int NT>
inline void tw8x2_body(const device uint* W, const device ushort* S, const device ushort* Bq, const device half* Th,
                       const device half* Tl, const device float* sums, const device float* sums_lo, device float* Y,
                       constant TqParams& p, threadgroup float* red, uint tg, uint sg, uint lane) {
  const uint n0 = tg * NT, gpr = p.K / 64, wpr = p.K / 4, gper = (gpr + p.ks - 1) / p.ks;
  const uint g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tH = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)Th, dextents<int32_t, 2>(p.K, 8));
  auto tL = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)Tl, dextents<int32_t, 2>(p.K, 8));
  auto Rx0 = tH.template slice<64, 8>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int NJ = NT / 8;
  uint rows[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) rows[j] = min(n0 + r + 8 * j, p.N - 1);
  const uint t0 = 4 * (lane & 1);
  for (uint g = g0; g < g1; ++g) {
    uint4 w[NJ];
    float sc[NJ], bc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      w[j] = *(const device uint4*)(W + rows[j] * wpr + g * 16 + 4 * c);
      sc[j] = as_type<float>(uint(S[rows[j] * gpr + g]) << 16);
      bc[j] = as_type<float>(uint(Bq[rows[j] * gpr + g]) << 16);
    }
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        // slot (c, jb, kk) = input 16 c + 8 (jb >> 1) + 2 kk + (jb & 1): byte 8 (jb >> 1) + 2 kk + (jb & 1) of the 16
        const uint wa = w[j][2 * (jb >> 1)], wb = w[j][2 * (jb >> 1) + 1], sh = 8 * (jb & 1);
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        Lw[i] = uchar(wa >> sh);
        Lw[i + 1] = uchar(wa >> (sh + 16));
        Lw[i + 2] = uchar(wb >> sh);
        Lw[i + 3] = uchar(wb >> (sh + 16));
      }
    auto sH = tH.template slice<64, 8>(g * 64, 0);
    auto sL = tL.template slice<64, 8>(g * 64, 0);
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(sH), float>();
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < d.get_capacity(); ++i) d[i] = 0.0f;
    op.run(Lw, sH, d);
    op.run(Lw, sL, d);
    const float4 sx = *(const device float4*)(sums + g * 8 + t0) + *(const device float4*)(sums_lo + g * 8 + t0);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(sc[j], d[j * 4 + kk], fma(bc[j], sx[kk], acc[j * 4 + kk]));
  }
  if (p.ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint jj = 1; jj < p.ks; ++jj)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((jj - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j)
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = t0 + kk, n = n0 + r + 8 * j;
      if (t < p.R && n < p.N) Y[t * p.N + n] = acc[j * 4 + kk];
    }
}
[[kernel]] void tw8x2_nt16(const device uint* W [[buffer(0)]], const device ushort* S [[buffer(1)]],
                           const device ushort* Bq [[buffer(2)]], const device half* Th [[buffer(3)]],
                           const device half* Tl [[buffer(4)]], const device float* sums [[buffer(5)]],
                           const device float* sums_lo [[buffer(6)]], device float* Y [[buffer(7)]],
                           constant TqParams& p [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[15 * 8 * 32];
  tw8x2_body<16>(W, S, Bq, Th, Tl, sums, sums_lo, Y, p, red, tg, sg, lane);
}
TW(tw_nt16, 16, 0, 0)
TW(tw_nt32, 32, 0, 0)
TW(tw_nt64, 64, 0, 0)
TW(twp_nt16, 16, 1, 0)
TW(twp_nt32, 32, 1, 0)
TW(twa1_nt16, 16, 0, 1)
TW(twa2_nt16, 16, 0, 2)
TW(twpa1_nt16, 16, 1, 1)
TW(twpa2_nt16, 16, 1, 2)
TW(tws_nt16, 16, 0, 0, 1)
TW(tws_nt32, 32, 0, 0, 1)
TW(twss_nt16, 16, 0, 0, 1, 1)
TW(twss_nt32, 32, 0, 0, 1, 1)
TW(twpss_nt16, 16, 1, 0, 1, 1)
TW(twpss_nt32, 32, 1, 0, 1, 1)
TW(twssa1_nt16, 16, 0, 1, 1, 1)
TW(twq_nt16, 16, 2, 0, 1)
TW(twl_nt16, 16, 0, 0, 0, 0, 1)
TW(twlp_nt16, 16, 1, 0, 0, 0, 1)
TW(twl_nt32, 32, 0, 0, 0, 0, 1)
TW(twlp_nt32, 32, 1, 0, 0, 0, 1)
TWX(tw8_nt16, 16, 8, 0, 0)
TWX(tw4_nt16, 16, 4, 0, 0)
TW(twa3_nt16, 16, 0, 3, 1)
TW(twa4_nt16, 16, 0, 4, 1)
TW(twa3_nt32, 32, 0, 3, 1)
TW(twa4_nt32, 32, 0, 4, 1)
TW(twq_nt32, 32, 2, 0, 1)

// tc: both operands cooperative (registers only, no device reads inside the matrix op), 16 x 16 x 32 per op (the
// 8 rows padded to 16), two ops per 64-group into one destination. T table "C layout": per group, token t and
// k-chunk c, the 16 slots 4c + 16 jb + kk contiguous: tab[((g * 8 + t) * 4 + c) * 16 + jb * 4 + kk].
template <int NT, int PF, int MAXKS>
inline void tc_body(const device uint* W, const device ushort* S, const device ushort* Bq, const device half* T,
                    const device float* sums, device float* Y, constant TqParams& p, threadgroup float* red, uint tg,
                    uint sg, uint lane) {
  const uint cb = sg / p.ks, ksl = sg % p.ks;
  const uint n0 = (tg * p.cs + cb) * NT, gpr = p.K / 64, wpr = p.K / 8, gper = gpr / p.ks, g0 = ksl * gper, g1 = g0 + gper;
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 16, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto Rx = op.template get_right_input_cooperative_tensor<uint8_t, half, float>();
  auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < Rx.get_capacity(); ++i) Rx[i] = 0.0h;
  constexpr int NJ = NT / 8;
  const device uint* wp[NJ];
  const device ushort* sp[NJ];
  const device ushort* bp[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = n0 + r + 8 * j;
    wp[j] = W + row * wpr + 2 * c;
    sp[j] = S + row * gpr;
    bp[j] = Bq + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  // this lane's input token r, chunk c
  const device uint4* tp = (const device uint4*)(T + (r * 4 + c) * 16);
  struct Stage {
    uint2 w[NJ];
    ushort s[NJ], b[NJ];
    uint4 xa, xb;
    float4 sx;
  };
  auto load = [&](thread Stage& o, uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      o.w[j] = *(const device uint2*)(wp[j] + g * 8);
      o.s[j] = sp[j][g];
      o.b[j] = bp[j][g];
    }
    o.xa = tp[g * 64];
    o.xb = tp[g * 64 + 1];
    o.sx = *(const device float4*)(sums + g * 8 + t0);
  };
  auto compute = [&](thread const Stage& cur) {
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < d.get_capacity(); ++i) d[i] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int h = 0; h < 2; ++h) {  // k-half: slots 32 h + 4 c + 16 j + kk = jb 2 h + j
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j)
        _Pragma("clang loop unroll(full)")
        for (int jj = 0; jj < 2; ++jj) {
          const uint u = ((h ? cur.w[j].y : cur.w[j].x) >> (4 * jj)) & 0x0F0F0F0Fu;
          const int i = (j >> 1) * 16 + jj * 8 + (j & 1) * 4;
          Lw[i] = uchar(u); Lw[i + 1] = uchar(u >> 8); Lw[i + 2] = uchar(u >> 16); Lw[i + 3] = uchar(u >> 24);
        }
      const uint4 xv = h ? cur.xb : cur.xa;
      _Pragma("clang loop unroll(full)")
      for (int jj = 0; jj < 2; ++jj) {
        const half2 a = as_type<half2>(xv[2 * jj]), b = as_type<half2>(xv[2 * jj + 1]);
        Rx[jj * 8] = a.x; Rx[jj * 8 + 1] = a.y; Rx[jj * 8 + 2] = b.x; Rx[jj * 8 + 3] = b.y;
      }
      op.run(Lw, Rx, d);
    }
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const float s = as_type<float>(uint(cur.s[j]) << 16), b = as_type<float>(uint(cur.b[j]) << 16);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, cur.sx[kk], acc[j * 4 + kk]));
    }
  };
  if (PF == 0) {
    for (uint g = g0; g < g1; ++g) {
      Stage cur;
      load(cur, g);
      compute(cur);
    }
  } else {
    Stage st[PF > 0 ? PF : 1];
    _Pragma("clang loop unroll(full)")
    for (int q = 0; q < PF; ++q)
      if (g0 + q < g1) load(st[q], g0 + q);
    for (uint g = g0; g < g1; g += PF) {
      _Pragma("clang loop unroll(full)")
      for (int q = 0; q < PF; ++q) {
        const uint gg = g + q;
        if (gg < g1) {
          const Stage cur = st[q];
          if (gg + PF < g1) load(st[q], gg + PF);
          compute(cur);
        }
      }
    }
  }
  if (p.ks > 1) {
    constexpr uint cap = NT / 2;
    if (ksl)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((cb * (p.ks - 1) + ksl - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ksl) return;
    for (uint jj = 1; jj < p.ks; ++jj)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((cb * (p.ks - 1) + jj - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j)
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = t0 + kk, n = n0 + r + 8 * j;
      if (t < p.R) Y[t * p.N + n] = acc[j * 4 + kk];
    }
}
#define TC(NAME, NT, PF, MAXKS)                                                                                   \
  [[kernel]] void NAME(const device uint* W [[buffer(0)]], const device ushort* S [[buffer(1)]],                   \
                       const device ushort* Bq [[buffer(2)]], const device half* T [[buffer(3)]],                  \
                       const device float* sums [[buffer(4)]], device float* Y [[buffer(5)]],                      \
                       constant TqParams& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],               \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {      \
    threadgroup float red[(MAXKS - 1) * (NT / 2) * 32];                                                          \
    tc_body<NT, PF, MAXKS>(W, S, Bq, T, sums, Y, p, red, tg, sg, lane);                                            \
  }
TC(tc_nt16, 16, 0, 16)
TC(tcp_nt16, 16, 1, 16)
TC(tcp2_nt16, 16, 2, 16)
TC(tcp3_nt16, 16, 3, 16)
TC(tcp4_nt16, 16, 4, 16)
TC(tcp2_nt32, 32, 2, 8)
TC(tc_nt32, 32, 0, 8)
TC(tcp_nt32, 32, 1, 8)
TQ(tq_nt16, 16)
TQ(tq_nt32, 32)
TQ(tq_nt64, 64)
)MSL";

namespace {
using Bytes = std::vector<uint8_t>;
uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint32_t le32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
uint64_t le64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

// Section (seg, sect) of the 64-bit Mach-O image at m[0..n).
bool macho_section(const uint8_t* m, size_t n, const char* seg, const char* sect, const uint8_t** out, size_t* len) {
  if (n < 32) return false;
  uint32_t ncmds = le32(m + 16);
  size_t off = 32;
  for (uint32_t i = 0; i < ncmds && off + 8 <= n; ++i) {
    uint32_t cmd = le32(m + off), csz = le32(m + off + 4);
    if (cmd == 0x19 && off + 72 <= n) {  // LC_SEGMENT_64
      uint32_t nsect = le32(m + off + 64);
      for (uint32_t s = 0; s < nsect; ++s) {
        const uint8_t* sd = m + off + 72 + 80 * s;
        if (sd + 80 > m + n) return false;
        if (strncmp((const char*)sd, sect, 16) == 0 && strncmp((const char*)sd + 16, seg, 16) == 0) {
          uint64_t size = le64(sd + 40);
          uint32_t soff = le32(sd + 48);
          if (soff + size > n) return false;
          *out = m + soff;
          *len = size;
          return true;
        }
      }
    }
    off += csz;
  }
  return false;
}

// -> code bytes, stack bytes, GPRs (-1 when not found)
void kernel_stats(const Bytes& b, long* text, long* stack, long* gprs) {
  *text = *stack = *gprs = -1;
  if (b.size() < 8) return;
  uint32_t nfat = be32(b.data() + 4);
  for (uint32_t i = 0; i < nfat && 8 + 20 * (i + 1) <= b.size(); ++i) {
    const uint8_t* e = b.data() + 8 + 20 * i;
    if (be32(e) != 0x01000013) continue;  // applegpu slice
    uint32_t off = be32(e + 8), size = be32(e + 12);
    if (off + size > b.size()) return;
    const uint8_t *inner, *txt, *md;
    size_t ninner, ntxt, nmd;
    if (!macho_section(b.data() + off, size, "__TEXT", "__compute", &inner, &ninner)) return;
    if (macho_section(inner, ninner, "__TEXT", "__text", &txt, &ntxt)) *text = long(ntxt);
    if (!macho_section(inner, ninner, "__GPU_METADATA", "__compute", &md, &nmd)) return;
    std::vector<uint32_t> w(nmd / 4);
    for (size_t k = 0; k < w.size(); ++k) w[k] = le32(md + 4 * k);
    // A flags word 0x01{01|03}xxxx is followed by the GPR count, or by the stack size (which also
    // appears in the few words before the flags) and then the GPR count. Without a stack the word
    // after the GPR count is 0 or 1, so a stack needs a plausible GPR count (>= 4) after it: a
    // 32-GPR kernel whose count equals the 0x20 word before the flags is not "32 B stack, 1 GPR".
    for (size_t k = 4; k + 2 < w.size(); ++k) {
      uint32_t f = w[k];
      if (f >> 24 != 1 || (((f >> 16) & 0xff) != 1 && ((f >> 16) & 0xff) != 3) || f == 0x01000000 || (f & 0xff) > 3)
        continue;
      uint32_t c = w[k + 1];
      bool seen = c > 0 && (w[k - 1] == c || w[k - 2] == c || w[k - 3] == c || w[k - 4] == c);
      if (seen && w[k + 2] >= 4 && w[k + 2] <= 256) { *stack = c; *gprs = w[k + 2]; return; }
      if (c <= 256) { *stack = 0; *gprs = c; return; }
    }
    return;
  }
}
}  // namespace

struct QMat {
  Buffer w, s, b;
};
static int env_ks() { return getenv("KS") ? atoi(getenv("KS")) : 8; }

static QMat make_mat(Gpu& g, uint32_t N, uint32_t K, std::mt19937& rng) {
  QMat m;
  size_t words = size_t(N) * K / 8, groups = size_t(N) * K / 64;
  m.w = g.alloc(words * 4, "w", false);
  m.s = g.alloc(groups * 2, "s", false);
  m.b = g.alloc(groups * 2, "b", false);
  auto* w = m.w.as<uint32_t>();
  for (size_t i = 0; i < words; ++i) w[i] = rng();
  std::uniform_real_distribution<float> d(-0.02f, 0.02f);
  auto* s = m.s.as<uint16_t>();
  auto* b = m.b.as<uint16_t>();
  for (size_t i = 0; i < groups; ++i) {
    s[i] = f32_to_bf16(std::fabs(d(rng)) * 0.2f + 1e-4f);
    b[i] = f32_to_bf16(d(rng));
  }
  return m;
}

int main(int argc, char** argv) {
  if (!strcmp(argv[1], "router")) {  // bench_tq router [N=257] [K=2048] [reps=20]: qmm8x2_b8_nf1 vs tw8x2
    const uint32_t N = argc > 2 ? atoi(argv[2]) : 257, K = argc > 3 ? atoi(argv[3]) : 2048;
    const int reps = argc > 4 ? atoi(argv[4]) : 20;
    Gpu& g = Gpu::get();
    std::string src = kSrc;
    src.replace(src.find("NI_TQ_KERNELS"), 13, kKern);
    g.compile_extra("tq", strdup(src.c_str()), true);
    std::mt19937 rng(5);
    const size_t mbytes = size_t(N) * K + size_t(N) * K / 64 * 4;
    const int copies = std::clamp(int(1.0e9 / mbytes), 1, 400);
    std::vector<QMat> mats;
    for (int i = 0; i < copies; ++i) {
      QMat m;
      m.w = g.alloc(size_t(N) * K, "w", false);
      m.s = g.alloc(size_t(N) * K / 64 * 2, "s", false);
      m.b = g.alloc(size_t(N) * K / 64 * 2, "b", false);
      for (size_t q = 0; q < size_t(N) * K / 4; ++q) m.w.as<uint32_t>()[q] = rng();
      std::uniform_real_distribution<float> d(-0.02f, 0.02f);
      for (size_t q = 0; q < size_t(N) * K / 64; ++q) {
        m.s.as<uint16_t>()[q] = f32_to_bf16(std::fabs(d(rng)) * 0.02f + 1e-5f);
        m.b.as<uint16_t>()[q] = f32_to_bf16(d(rng));
      }
      mats.push_back(m);
    }
    Buffer x = g.alloc(size_t(8) * K * 4), tab = g.alloc(size_t(K) * 16), sm = g.alloc(size_t(K) / 64 * 32);
    Buffer tlo = g.alloc(size_t(K) * 16), slo = g.alloc(size_t(K) / 64 * 32), th = g.alloc(size_t(K) * 16), tl = g.alloc(size_t(K) * 16);
    Buffer y = g.alloc(size_t(8) * N * 4), y2 = g.alloc(size_t(8) * N * 4);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < size_t(8) * K; ++i) x.as<float>()[i] = nd(rng);
    {
      auto e = g.begin();
      e->use(g.pipeline("x8_prep2"));
      kp::X8Prep pp{K, 8, K, 0, 0, 1e-6f, K, 1u, 0u, 1u};
      e->buf(0, x); e->buf(1, x); e->buf(2, tab); e->buf(3, sm); e->buf(4, x);
      e->val(5, pp);
      e->buf(6, tlo); e->buf(7, slo); e->buf(8, th);
      e->buf(9, tlo);  // (unused unless write_t_lo)
      e->dispatch(K / 256, 1, 1, 256);
      e->commit_and_wait();
    }
    // the lo T table (the same permutation)
    for (uint32_t t = 0; t < 8; ++t)
      for (uint32_t gi = 0; gi < K / 64; ++gi)
        for (uint32_t sl = 0; sl < 64; ++sl) {
          const uint32_t c = (sl >> 2) & 3, jb = (sl >> 4) & 3, kk = sl & 3, k = 16 * c + 8 * (jb >> 1) + 2 * kk + (jb & 1);
          const float v = x.as<float>()[size_t(t) * K + gi * 64 + k], h = float(__fp16(v));
          tl.as<__fp16>()[size_t(t) * K + gi * 64 + sl] = __fp16(v - h);
        }
    Pipeline pq = g.pipeline("qmm8x2_b8_nf1"), pt = g.pipeline("tw8x2_nt16");
    const int ksq = 8, kst = env_ks();
    auto enc_q = [&](Encoder& e, int i) {
      kp::Qmm8 prm{};
      prm.K = K; prm.R = 8; prm.nseg = 1; prm.ks = ksq; prm.N[0] = N; prm.y_ld[0] = N;
      const uint32_t ntg = (N + 7) / 8;
      prm.tg_begin[1] = ntg;
      e.use(pq);
      for (int sgi = 0; sgi < 4; ++sgi) {
        e.buf(sgi * 4 + 0, mats[i].w); e.buf(sgi * 4 + 1, mats[i].s); e.buf(sgi * 4 + 2, mats[i].b); e.buf(sgi * 4 + 3, y);
      }
      e.buf(16, tab); e.buf(17, sm); e.val(18, prm); e.buf(19, tlo); e.buf(20, slo);
      e.dispatch(ntg, 1, 1, 32 * ksq);
    };
    auto enc_t = [&](Encoder& e, int i) {
      struct { uint32_t K, N, ks, R, cs; } prm{K, N, uint32_t(kst), 8, 1};
      e.use(pt);
      e.buf(0, mats[i].w); e.buf(1, mats[i].s); e.buf(2, mats[i].b); e.buf(3, th); e.buf(4, tl); e.buf(5, sm);
      e.buf(6, slo); e.buf(7, y2); e.val(8, prm);
      e.dispatch((N + 15) / 16, 1, 1, 32 * kst);
    };
    {
      auto e = g.begin();
      enc_q(*e, 0);
      enc_t(*e, 0);
      e->commit_and_wait();
    }
    // fp64 reference from x (hi + lo = x to ~22 bits)
    double eq = 0, et = 0, rr = 0, mq = 0, mt = 0;
    for (uint32_t t = 0; t < 8; ++t)
      for (uint32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (uint32_t gi = 0; gi < K / 64; ++gi) {
          const double s_ = bf16_to_f32(mats[0].s.as<uint16_t>()[size_t(n) * (K / 64) + gi]);
          const double b_ = bf16_to_f32(mats[0].b.as<uint16_t>()[size_t(n) * (K / 64) + gi]);
          for (uint32_t k = 0; k < 64; ++k) {
            const uint32_t kk = gi * 64 + k;
            const float v = x.as<float>()[size_t(t) * K + kk], h = float(__fp16(v));
            const double xv = double(h) + double(float(__fp16(v - h)));
            acc += (s_ * mats[0].w.as<uint8_t>()[size_t(n) * K + kk] + b_) * xv;
          }
        }
        rr += acc * acc;
        const double dq = y.as<float>()[size_t(t) * N + n] - acc, dt = y2.as<float>()[size_t(t) * N + n] - acc;
        eq += dq * dq; et += dt * dt; mq = std::max(mq, std::fabs(dq)); mt = std::max(mt, std::fabs(dt));
      }
    rr = std::sqrt(rr / (8.0 * N));
    printf("router %ux%u: vs fp64 rms err / rms: qmm8x2 %.3e (max %.3e)  tw8x2 %.3e (max %.3e)\n", N, K,
           std::sqrt(eq / (8.0 * N)) / rr, mq / rr, std::sqrt(et / (8.0 * N)) / rr, mt / rr);
    std::vector<double> tq, tt;
    auto run = [&](bool tens) {
      auto e = g.begin();
      for (int i = 0; i < copies; ++i) { if (tens) enc_t(*e, i); else enc_q(*e, i); }
      e->commit_and_wait();
      return e->gpu_ms() * 1000.0 / copies;
    };
    run(false); run(true);
    for (int r = 0; r < reps; ++r) { tq.push_back(run(false)); tt.push_back(run(true)); }
    std::sort(tq.begin(), tq.end()); std::sort(tt.begin(), tt.end());
    printf("qmm8x2_b8_nf1 ks%d median %.2f us | tw8x2_nt16 ks%d median %.2f us\n", ksq, tq[tq.size() / 2], kst, tt[tt.size() / 2]);
    return 0;
  }
  if (argc < 3) {
    fprintf(stderr, "usage: bench_tq <N> <K> [reps] [variant]\n");
    return 1;
  }
  if (!strcmp(argv[1], "gnorm")) {  // gdn_gnorm_x8t's T table vs its X8 table (same values, two layouts)
    Gpu& g = Gpu::get();
    const uint32_t Hv = argc > 2 ? atoi(argv[2]) : 4, R = argc > 3 ? atoi(argv[3]) : 8, D = 128, VD = Hv * D;
    Buffer o = g.alloc(size_t(8) * VD * 4), z = g.alloc(size_t(8) * VD * 4), w = g.alloc(D * 2);
    Buffer tab = g.alloc(size_t(VD) * 8 * 2), sm = g.alloc(size_t(VD) / 64 * 8 * 4), tt = g.alloc(size_t(VD) * 8 * 2);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < size_t(8) * VD; ++i) o.as<float>()[i] = nd(rng), z.as<float>()[i] = nd(rng);
    for (uint32_t i = 0; i < D; ++i) w.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.1f * nd(rng));
    memset(tt.ptr(), 0x7f, tt.size);
    struct { uint32_t R, n_pend, Hk, Hv, C; float eps; } gp{R, 0, Hv, Hv, 3 * VD, 1e-6f};
    auto e = g.begin();
    e->use(g.pipeline("gdn_gnorm_x8t"));
    e->buf(0, o); e->buf(1, z); e->buf(2, w); e->buf(3, tab); e->buf(4, sm);
    e->val(5, gp);
    e->buf(6, tt);
    e->dispatch(Hv, 1, 1, 256);
    e->commit_and_wait();
    int bad = 0;
    for (uint32_t t = 0; t < 8; ++t)
      for (uint32_t k = 0; k < VD; ++k) {
        const uint32_t gi = k / 64, kk = k % 64;
        // X8 table offset of input kk of group gi, row t (verify8.metal v8_tab)
        const uint32_t f = 4 * ((kk >> 3) & 1) + (kk & 3), j = 2 * (kk >> 4) + ((kk >> 2) & 1);
        const uint32_t fm = j, fn = t & 6u;
        const uint32_t ln = ((fm >> 2) << 4) | ((fn >> 2) << 3) | ((fm & 3) << 1) | ((fn >> 1) & 1);
        const float a = float(tab.as<__fp16>()[(gi * 32 + ln) * 16 + 2 * f + (t & 1)]);
        // T table: slot s of input kk
        uint32_t sl = 0;
        for (uint32_t q = 0; q < 64; ++q) {
          const uint32_t c = (q >> 2) & 3, jb = (q >> 4) & 3, k4 = q & 3;
          if (16 * c + 8 * (jb >> 1) + 2 * k4 + (jb & 1) == kk) sl = q;
        }
        const float b = float(tt.as<__fp16>()[size_t(t) * VD + gi * 64 + sl]);
        if (a != b && bad++ < 5) printf("  t %u k %u: x8 %g T %g\n", t, k, a, b);
      }
    printf("gdn_gnorm_x8t Hv=%u R=%u: %d mismatches of %u\n", Hv, R, bad, 8 * VD);
    return 0;
  }
  if (!strcmp(argv[1], "stats")) {  // bench_tq stats <kernel> ...: code size, stack, GPRs
    Gpu& g = Gpu::get();
    std::string src = kSrc;
    src.replace(src.find("NI_TQ_KERNELS"), 13, kKern);
    g.compile_extra("tq", strdup(src.c_str()), true);
    for (int i = 2; i < argc; ++i) {
      const char* td = getenv("TMPDIR");
      const std::string path = std::string(td && *td ? td : "/tmp") + "/ni_tq_" + std::string(argv[i]) + ".bin";
      std::ifstream f;
      if (g.archive(argv[i], path) && (f.open(path, std::ios::binary), f)) {
        Bytes b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        long text, stack, gprs;
        kernel_stats(b, &text, &stack, &gprs);
        printf("%-20s code %6ld B  stack %5ld B  gprs %3ld\n", argv[i], text, stack, gprs);
      }
    }
    return 0;
  }
  if (!strcmp(argv[1], "probe")) {
    Gpu& g = Gpu::get();
    std::string src = kSrc;
    src.replace(src.find("NI_TQ_KERNELS"), 13, kKern);
    g.compile_extra("tq", strdup(src.c_str()), true);
    Buffer out = g.alloc(32 * 1024 * 4);
    Pipeline p = g.pipeline(argv[2]);
    memset(out.ptr(), 0xff, out.size);
    auto e = g.begin();
    e->use(p);
    e->buf(0, out);
    e->dispatch(1, 1, 1, 32);
    e->commit_and_wait();
    const int* o = out.as<int>();
    printf("%s: capacity left %d right %d dest %d\n", argv[2], o[0], o[1], o[2]);
    const char* nm[3] = {"left", "right", "dest"};
    const int base[3] = {8, 400, 800};
    for (int which = 0; which < 3; ++which)
      for (int lane = 0; lane < 32; ++lane) {
        const int* q = out.as<int>() + lane * 1024;
        printf("%s lane %2d:", nm[which], lane);
        for (int i = 0; i < q[which]; ++i)
          printf(" %s(%d,%d)", q[base[which] + i * 3] ? "" : "!", q[base[which] + i * 3 + 1], q[base[which] + i * 3 + 2]);
        printf("\n");
      }
    return 0;
  }
  const uint32_t N = atoi(argv[1]), K = atoi(argv[2]);
  const int reps = argc > 3 ? atoi(argv[3]) : 10;
  const std::string only = argc > 4 ? argv[4] : "";
  Gpu& g = Gpu::get();
  std::string src = kSrc;
  src.replace(src.find("NI_TQ_KERNELS"), 13, kKern);
  g.compile_extra("tq", strdup(src.c_str()), true);
  std::mt19937 rng(7);
  const size_t mbytes = size_t(N) * K / 2 + size_t(N) * K / 64 * 4;
  const int copies = std::clamp(int(1.0e9 / mbytes), 1, 400);
  std::vector<QMat> mats;
  for (int i = 0; i < copies; ++i) mats.push_back(make_mat(g, N, K, rng));
  Buffer x = g.alloc(size_t(8) * K * 4), table = g.alloc(size_t(K) * 8 * 2), sums = g.alloc(size_t(K) / 64 * 8 * 4);
  Buffer xh = g.alloc(size_t(8) * K * 2), y = g.alloc(size_t(8) * N * 4), y2 = g.alloc(size_t(8) * N * 4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(8) * K; ++i) x.as<float>()[i] = nd(rng);
  for (size_t i = 0; i < size_t(8) * K; ++i) xh.as<__fp16>()[i] = __fp16(x.as<float>()[i]);
  {
    auto e = g.begin();
    e->use(g.pipeline("x8_prep2"));
    kp::X8Prep pp{K, 8, K, 0, 0, 1e-6f, K, 0u};
    e->buf(0, x); e->buf(1, x); e->buf(2, table); e->buf(3, sums); e->buf(4, x);
    e->val(5, pp);
    e->buf(6, table); e->buf(7, sums); e->buf(8, table);
    e->buf(9, table);  // (unused unless write_t_lo)
    e->dispatch(K / 256, 1, 1, 256);
    e->commit_and_wait();
  }
  // tw table: per group and lane the 16 inputs of its k-slots in element order (see tw_body)
  Buffer ttab = g.alloc(size_t(K) * 8 * 2);
  for (uint32_t t = 0; t < 8; ++t)
    for (uint32_t gi = 0; gi < K / 64; ++gi)
      for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t jb = 0; jb < 4; ++jb)
          for (uint32_t kk = 0; kk < 4; ++kk) {
            const uint32_t k = 16 * c + 8 * (jb >> 1) + 2 * kk + (jb & 1), sl = 4 * c + 16 * jb + kk;
            ttab.as<__fp16>()[size_t(t) * K + gi * 64 + sl] = xh.as<__fp16>()[size_t(t) * K + gi * 64 + k];
          }
  // tc's T table (C layout)
  Buffer ctab = g.alloc(size_t(K) * 8 * 2);
  for (uint32_t t = 0; t < 8; ++t)
    for (uint32_t gi = 0; gi < K / 64; ++gi)
      for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t jb = 0; jb < 4; ++jb)
          for (uint32_t kk = 0; kk < 4; ++kk) {
            const uint32_t k = 16 * c + 8 * (jb >> 1) + 2 * kk + (jb & 1);
            ctab.as<__fp16>()[((size_t(gi) * 8 + t) * 4 + c) * 16 + jb * 4 + kk] = xh.as<__fp16>()[size_t(t) * K + gi * 64 + k];
          }
  struct V {
    std::string name;
    Pipeline p;
    int nt, ks;  // tq: rows per simdgroup tile, k-slices; qmm8: nf, ks (cs 1)
    bool tq;
    std::vector<double> t;
  };
  std::vector<V> vs;
  auto add_q = [&](int nf, int ks) {
    vs.push_back({"qmm8snb_b4_nf" + std::to_string(nf) + " ks" + std::to_string(ks), g.pipeline("qmm8snb_b4_nf" + std::to_string(nf)), nf, ks, false, {}});
  };
  auto add_t = [&](int nt, int ks, const char* pre = "tq") {
    const std::string n = std::string(pre) + "_nt" + std::to_string(nt);
    if (!only.empty() && only != n && only != pre) return;
    Pipeline p = g.pipeline(n, false);
    if (!p.pso) return;
    vs.push_back({n + " ks" + std::to_string(ks), p, nt, ks, true, {}});
  };
  add_q(1, 8);
  if (only.empty() || only == "q") add_q(2, 4);
  for (int nt : {16, 32, 64})
    for (int ks : {1, 2, 4, 8})
      if ((K / 64) % ks == 0 && N % nt == 0 && (nt < 64 || ks <= 4)) add_t(nt, ks);
  std::vector<std::string> pres = {"tw", "twp", "twl", "twlp", "tc", "tcp"};
  if (getenv("PRE")) {
    pres.clear();
    std::string e = getenv("PRE");
    size_t a = 0;
    while (a <= e.size()) { size_t b = e.find(',', a); if (b == std::string::npos) b = e.size(); pres.push_back(e.substr(a, b - a)); a = b + 1; }
  }
  for (const std::string& pre_s : pres)
    for (int nt : {16, 32, 64})
      for (int ks : {1, 2, 4, 8, 16})
        if ((K / 64) % ks == 0 && N % nt == 0 && (nt < 64 || ks <= 4) && (nt < 32 || ks <= 8) &&
            (pre_s.rfind("tw", 0) != 0 || nt < 16 || ks <= 8) && (getenv("KS") == nullptr || atoi(getenv("KS")) == ks))
          add_t(nt, ks, pre_s.c_str());
  auto enc = [&](Encoder& e, V& v, int i) {
    if (v.tq) {
      const uint32_t cs = v.name.rfind("tc", 0) == 0 && getenv("CS") ? uint32_t(atoi(getenv("CS"))) : 1u;
      struct { uint32_t K, N, ks, R, cs; } prm{K, N, uint32_t(v.ks), 8, cs};
      e.use(v.p);
      e.buf(0, mats[i].w); e.buf(1, mats[i].s); e.buf(2, mats[i].b); e.buf(3, v.name.rfind("tw", 0) == 0 ? ttab : v.name.rfind("tc", 0) == 0 ? ctab : xh);
      e.buf(4, sums); e.buf(5, y2);
      e.val(6, prm);
      e.dispatch(N / (v.nt * cs), 1, 1, 32 * v.ks * cs);
    } else {
      kp::Qmm8 prm{};
      prm.K = K; prm.R = 8; prm.nseg = 1; prm.ks = uint32_t(v.ks);
      prm.N[0] = N; prm.y_ld[0] = N;
      const uint32_t ntg = (N + 8 * v.nt - 1) / (8 * v.nt);
      prm.tg_begin[1] = ntg;
      e.use(v.p);
      for (int sgi = 0; sgi < 4; ++sgi) {
        e.buf(sgi * 4 + 0, mats[i].w); e.buf(sgi * 4 + 1, mats[i].s); e.buf(sgi * 4 + 2, mats[i].b); e.buf(sgi * 4 + 3, y);
      }
      e.buf(16, table); e.buf(17, sums);
      e.val(18, prm);
      e.buf(19, table); e.buf(20, sums);
      e.dispatch(ntg, 1, 1, 32 * v.ks);
    }
  };
  // ACC=1: every output of matrix 0 against a double reference from the same half inputs
  std::vector<double> ref;
  if (getenv("ACC")) {
    ref.assign(size_t(8) * N, 0.0);
    const uint32_t* w = mats[0].w.as<uint32_t>();
    const uint16_t* sc = mats[0].s.as<uint16_t>();
    const uint16_t* bi = mats[0].b.as<uint16_t>();
    std::vector<double> xd(size_t(8) * K);
    for (size_t i = 0; i < xd.size(); ++i) xd[i] = double(float(xh.as<__fp16>()[i]));
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t gi = 0; gi < K / 64; ++gi) {
        const double s_ = bf16_to_f32(sc[size_t(n) * (K / 64) + gi]), b_ = bf16_to_f32(bi[size_t(n) * (K / 64) + gi]);
        for (uint32_t t = 0; t < 8; ++t) {
          double d = 0, sx = 0;
          for (uint32_t k = 0; k < 64; ++k) {
            const uint32_t kk = gi * 64 + k;
            const double q = double((w[size_t(n) * K / 8 + kk / 8] >> (4 * (kk % 8))) & 15u);
            d += q * xd[size_t(t) * K + kk];
            sx += xd[size_t(t) * K + kk];
          }
          ref[size_t(t) * N + n] += s_ * d + b_ * sx;
        }
      }
  }
  // correctness: each variant once on matrix 0 vs the first qmm8 variant
  for (size_t vi = 0; vi < vs.size(); ++vi) {
    auto& v = vs[vi];
    memset((v.tq ? y2 : y).ptr(), 0, size_t(8) * N * 4);
    auto e = g.begin();
    enc(*e, v, 0);
    e->commit_and_wait();
    if (!ref.empty()) {
      const float* yy = (v.tq ? y2 : y).as<float>();
      double rr = 0, ee = 0, mx = 0;
      for (size_t i = 0; i < ref.size(); ++i) {
        rr += ref[i] * ref[i];
        const double d = double(yy[i]) - ref[i];
        ee += d * d;
        mx = std::max(mx, std::fabs(d));
      }
      rr = std::sqrt(rr / ref.size());
      printf("%-22s vs fp64: rms err / rms %.3e  max err / rms %.3e\n", v.name.c_str(), std::sqrt(ee / ref.size()) / rr,
             mx / rr);
    }
    if (vi == 0) continue;
    if (!v.tq) continue;
    double maxd = 0, rms = 0;
    for (size_t i = 0; i < size_t(8) * N; ++i) rms += double(y.as<float>()[i]) * y.as<float>()[i];
    rms = std::sqrt(rms / (8.0 * N));
    size_t bad = 0;
    for (size_t i = 0; i < size_t(8) * N; ++i) {
      const double dd = std::fabs(double(y.as<float>()[i]) - y2.as<float>()[i]) / (std::fabs(y.as<float>()[i]) + rms);
      maxd = std::max(maxd, dd);
      if (dd > 1e-3 && bad++ < 4)
        printf("  %s mismatch t=%zu n=%zu: qmm8 %g tq %g\n", v.name.c_str(), i / N, i % N, y.as<float>()[i], y2.as<float>()[i]);
    }
    printf("%-22s vs qmm8: max rel diff %.2e (%zu bad)\n", v.name.c_str(), maxd, bad);
  }
  auto run = [&](V& v) {
    auto e = g.begin();
    for (int i = 0; i < copies; ++i) enc(*e, v, i);
    e->commit_and_wait();
    return e->gpu_ms() * 1000.0 / copies;
  };
  for (auto& v : vs) run(v);
  for (int r = 0; r < reps; ++r)
    for (auto& v : vs) v.t.push_back(run(v));
  for (auto& v : vs) {
    std::sort(v.t.begin(), v.t.end());
    const double mn = v.t.front(), med = v.t[v.t.size() / 2];
    printf("%-22s mt%4u | min %8.1f us %6.1f GB/s | median %8.1f us %6.1f GB/s\n", v.name.c_str(), v.p.max_threads, mn,
           mbytes / (mn * 1e3), med, mbytes / (med * 1e3));
  }
  return 0;
}
