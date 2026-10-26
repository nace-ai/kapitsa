#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- the Q2_0 experts of an 8-row verify on the matrix units (NI_FN8T_E; Metal 4 tensor_ops, Apple10) ----
//
// moe_q20v_gate_up / moe_q20_down (zq2moe.metal) run the verify's experts as simdgroup MMA row kernels: per
// unique expert, per token pair, the weights are re-read and the dots run on the ALUs. The M5's matrix units run a
// 16 x 8 x 64 product per simdgroup (verify8t.metal's qmm8t class), and an 8-row verify has R * k = 64 (expert,
// token) pairs over <= 64 unique experts, so the tensor path runs each unique expert once for all 8 tokens:
//
//   qmt8v_gu20  per unique routed expert u (grid z): gate and up as 16-row weight tiles x the 8 tokens' T table
//               (x8_prep2's, the same tables the dense qmm6t reads), h = silu(gate . x) * (up . x) in fp32 into
//               he8_'s [8][Imax] rows of plane u. Q2_0: weight = (code - 1) * d, so per 64-group the epilogue is
//               acc += d * (dot - sum(x)) with d the fp16 block scale -- the q20_rows arithmetic, with the matrix
//               unit's summation order inside a group (uint8 codes x half inputs, exact products in fp32).
//   qmt8v_gu4   the shared expert (4-bit group-64 affine, the qmvs/qmm8 arithmetic: acc += s * dot + b * sum(x),
//               s / b the bf16 bit patterns) into he8_'s plane 8k.
//   qmt8v_hsum  splits each he8_ plane into the permuted half T tables + group sums that qmt8v_dn* reads as the
//               right operand (x8_prep2's staging / sums / write_t arithmetic; I = 640 is no multiple of 256, so
//               the 256-input blocks are masked, and the tables' row stride is exactly K).
//   qmt8v_dn20  per unique routed expert u: down = h-table x the expert's [D][I] codes, scattered into the old
//               out planes out8_[(t * (k + 1) + uslot[u * 8 + t]) * D] (rows the umask bit t selects only).
//   qmt8v_dn4   the shared expert's down into plane k of every token's row.
//   qmt8v_combine  y[t][d] += sum_j wts[t * (k + 1) + j] * out8_[(t * (k + 1) + j) * D + d], j = 0 .. k in the
//               old moe_q20_down's slot order (one thread per (t, d), so one writer per output).
//
// Two-term inputs (LO = 1, the default; NI_FN8T_LO = 0 is the single-term diagnostic): each product also runs
// against the lo T table with the lo group sums, exactly qmm6t's second term. The k-slices reduce in a fixed
// order through threadgroup memory (qmm8t's red pattern).
//
// The left operand's lane layout is verify8t.metal's (probed there): lane role c = (lane & 1) | ((lane >> 3) & 1)
// << 1 holds k-slots 4c + 16 jb + kk of weight rows r + 8 j, r = ((lane >> 1) & 3) | ((lane >> 4) & 1) << 2, as
// left elements (j >> 1) * 32 + jb * 8 + (j & 1) * 4 + kk; slot 4c + 16 jb + kk holds input 16 c + 8 (jb >> 1) +
// 2 kk + (jb & 1). Q2_0 packs 16 codes a word (MLX order, code i at bit 2i), so a lane's word c of a group holds
// the inputs 16c + e, e < 16, and the slot's code is at bit 2 * (8 (jb >> 1) + 2 kk + (jb & 1)) = 16 (jb >> 1) +
// 4 kk + 2 (jb & 1) of that word; the 4-bit shared expert is qmm8t's uint2 load and 0x0F0F0F0F mask.
//
// Static threadgroup memory: gu 14,336 B (two accumulator sets' k-slices), dn 7,168 B, hsum 8,192 B (staging),
// combine 0 -- each within the 16,384 B this macOS's validation layer doubles against the 32,768 B cap.

struct Qmt8vParams {  // = kp::Qmt8v (model/kparams.h)
  uint D;
  uint I;
  uint SI;
  uint k;
  uint R;
  uint ks_gu;   // k-slices of the gate/up GEMMs (gpr = D / 64)
  uint ks_d20;  // ... of the routed down (gpr = I / 64)
  uint ks_d4;   // ... of the shared down (gpr = SI / 64)
  uint arena;   // 1: the weights are expert-store chunks resolved through the slot table
  uint Imax;    // max(I, SI): he8_'s row stride
  uint lo;      // informational: the LO instantiations encode it
  uint lgc, lgd, luc, lud, ldc, ldd;  // Q20Slot: the planes' byte offsets inside an arena slot
};

// the T table's slot permutation (verify8.metal's v8t_input_of_slot; this library compiles on its own)
NI_INLINE uint v8t_input_of_slot(uint s) {
  const uint c = (s >> 2) & 3, jb = (s >> 4) & 3, kk = s & 3;
  return 16 * c + 8 * (jb >> 1) + 2 * kk + (jb & 1);
}

// ---- the routed experts' gate/up: plane u0 = grid z, all 8 tokens, h = silu(gate . x) * (up . x) ----
// One cooperative left tensor, refilled per matrix (gate, up): qmm6t runs one Lw against two right operands, and
// refilling between runs keeps this correct whatever the register file does with two live left tensors.
template <int NT, int PF, int MAXKS, int LO>
NI_INLINE void qmt8v_gu20_impl(const device uint* gcod, const device half* gdd, const device uint* ucod,
                               const device half* udd, const device half* T, const device float* sums,
                               const device half* Tlo, const device float* sums_lo, device float* he, uint u0,
                               constant Qmt8vParams& p, uint ks, threadgroup float* red, uint tg, uint sg,
                               uint lane) {
  const uint N = p.I, n0 = tg * NT, gpr = p.D / 64, wpr = p.D / 16;
  const uint gper = (gpr + ks - 1) / ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(p.D, 8));
  auto Rx0 = tX.template slice<64, 8>(0, 0);
  auto tL =
      tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)Tlo, dextents<int32_t, 2>(p.D, 8));
  auto accG = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  auto accU = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < accG.get_capacity(); ++i) accG[i] = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < accU.get_capacity(); ++i) accU[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* gp[NJ];
  const device uint* up[NJ];
  const device half* gd[NJ];
  const device half* ud[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    gp[j] = gcod + row * wpr + c;  // this lane's word (16 codes) of every group
    up[j] = ucod + row * wpr + c;
    gd[j] = gdd + row * gpr;
    ud[j] = udd + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  uint gwn[NJ], uwn[NJ];
  half gdn[NJ], udn[NJ];
  auto load = [&](uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      gwn[j] = gp[j][4 * g];  // a Q2_0 group is 4 words wide; this lane takes word c of it
      uwn[j] = up[j][4 * g];
      gdn[j] = gd[j][g];
      udn[j] = ud[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint gw[NJ], uw[NJ];
    half gdc[NJ], udc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      gw[j] = gwn[j];
      uw[j] = uwn[j];
      gdc[j] = gdn[j];
      udc[j] = udn[j];
    }
    if (PF && g + 1 < g1) load(g + 1);
    auto Rx = tX.template slice<64, 8>(g * 64, 0);
    auto Rxl = tL.template slice<64, 8>(g * 64, 0);
    const float4 sx = *(const device float4*)(sums + g * 8 + t0);
    const float4 sxl = LO ? *(const device float4*)(sums_lo + g * 8 + t0) : float4(0.0f);
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < 2; ++m) {  // 0 = gate, 1 = up: fill, the products, the group's affine terms
      const thread uint* w = m ? uw : gw;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j)
        _Pragma("clang loop unroll(full)")
        for (int jb = 0; jb < 4; ++jb) {
          const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
          _Pragma("clang loop unroll(full)")
          for (int kk = 0; kk < 4; ++kk)
            Lw[i + kk] = uchar((w[j] >> (16u * uint(jb >> 1) + 4u * uint(kk) + 2u * uint(jb & 1))) & 3u);
        }
      auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
      op.run(Lw, Rx, d);
      auto dl = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
      if (LO) op.run(Lw, Rxl, dl);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) {
        const float dd = float(m ? udc[j] : gdc[j]);
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) {
          if (m) {  // the m loop is fully unrolled: this folds into the up tensor's arm
            accU[j * 4 + kk] = fma(dd, d[j * 4 + kk] - sx[kk], accU[j * 4 + kk]);
            if (LO) accU[j * 4 + kk] = fma(dd, dl[j * 4 + kk] - sxl[kk], accU[j * 4 + kk]);
          } else {
            accG[j * 4 + kk] = fma(dd, d[j * 4 + kk] - sx[kk], accG[j * 4 + kk]);
            if (LO) accG[j * 4 + kk] = fma(dd, dl[j * 4 + kk] - sxl[kk], accG[j * 4 + kk]);
          }
        }
      }
    }
  }
  // fixed-order sum of the k-slices (simdgroup 0 adds 1, 2, ... in turn): the gate slices, then the up's
  if (ks > 1) {
    constexpr uint cap = NT / 2;
    threadgroup float* redU = red + (MAXKS - 1) * cap * 32;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) {
        red[((sg - 1) * cap + i) * 32 + lane] = accG[i];
        redU[((sg - 1) * cap + i) * 32 + lane] = accU[i];
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) {
        accG[i] += red[((j - 1) * cap + i) * 32 + lane];
        accU[i] += redU[((j - 1) * cap + i) * 32 + lane];
      }
  }
  if ((lane >> 3) & 1) return;  // padding rows
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk)
      he[(u0 * 8 + t0 + kk) * p.Imax + n] = silu(accG[j * 4 + kk]) * accU[j * 4 + kk];
  }
}

// ---- the shared expert's gate/up (4-bit group-64 affine: qmm8t's loads, masks and epilogue) ----
template <int NT, int PF, int MAXKS, int LO>
NI_INLINE void qmt8v_gu4_impl(const device uint* sgw, const device ushort* sgs, const device ushort* sgb,
                              const device uint* suw, const device ushort* sus, const device ushort* sub,
                              const device half* T, const device float* sums, const device half* Tlo,
                              const device float* sums_lo, device float* he, uint u0, constant Qmt8vParams& p,
                              uint ks, threadgroup float* red, uint tg, uint sg, uint lane) {
  const uint N = p.SI, n0 = tg * NT, gpr = p.D / 64, wpr = p.D / 8;
  const uint gper = (gpr + ks - 1) / ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(p.D, 8));
  auto Rx0 = tX.template slice<64, 8>(0, 0);
  auto tL =
      tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)Tlo, dextents<int32_t, 2>(p.D, 8));
  auto accG = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  auto accU = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < accG.get_capacity(); ++i) accG[i] = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < accU.get_capacity(); ++i) accU[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* gp[NJ];
  const device uint* up[NJ];
  const device ushort* gs[NJ];
  const device ushort* gb[NJ];
  const device ushort* us[NJ];
  const device ushort* ub[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    gp[j] = sgw + row * wpr + 2 * c;  // this lane's 2 words of every group's 128-bit run
    up[j] = suw + row * wpr + 2 * c;
    gs[j] = sgs + row * gpr;
    gb[j] = sgb + row * gpr;
    us[j] = sus + row * gpr;
    ub[j] = sub + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  uint2 gwn[NJ], uwn[NJ];
  ushort gsn[NJ], gbn[NJ], usn[NJ], ubn[NJ];
  auto load = [&](uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      gwn[j] = *(const device uint2*)(gp[j] + g * 8);
      uwn[j] = *(const device uint2*)(up[j] + g * 8);
      gsn[j] = gs[j][g];
      gbn[j] = gb[j][g];
      usn[j] = us[j][g];
      ubn[j] = ub[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint2 gw[NJ], uw[NJ];
    ushort gsc[NJ], gbc[NJ], usc[NJ], ubc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      gw[j] = gwn[j];
      uw[j] = uwn[j];
      gsc[j] = gsn[j];
      gbc[j] = gbn[j];
      usc[j] = usn[j];
      ubc[j] = ubn[j];
    }
    if (PF && g + 1 < g1) load(g + 1);
    auto Rx = tX.template slice<64, 8>(g * 64, 0);
    auto Rxl = tL.template slice<64, 8>(g * 64, 0);
    const float4 sx = *(const device float4*)(sums + g * 8 + t0);
    const float4 sxl = LO ? *(const device float4*)(sums_lo + g * 8 + t0) : float4(0.0f);
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < 2; ++m) {  // 0 = gate, 1 = up
      const thread uint2* w = m ? uw : gw;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j)
        _Pragma("clang loop unroll(full)")
        for (int jb = 0; jb < 4; ++jb) {
          const uint u = ((jb >> 1) ? w[j].y : w[j].x) >> (4 * (jb & 1));
          const uint v = u & 0x0F0F0F0F;
          const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
          Lw[i] = uchar(v);
          Lw[i + 1] = uchar(v >> 8);
          Lw[i + 2] = uchar(v >> 16);
          Lw[i + 3] = uchar(v >> 24);
        }
      auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
      op.run(Lw, Rx, d);
      auto dl = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
      if (LO) op.run(Lw, Rxl, dl);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) {
        const float s = as_type<float>(uint(m ? usc[j] : gsc[j]) << 16);
        const float b = as_type<float>(uint(m ? ubc[j] : gbc[j]) << 16);
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) {
          if (m) {  // the m loop is fully unrolled: this folds into the up tensor's arm
            accU[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx[kk], accU[j * 4 + kk]));
            if (LO) accU[j * 4 + kk] = fma(s, dl[j * 4 + kk], fma(b, sxl[kk], accU[j * 4 + kk]));
          } else {
            accG[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx[kk], accG[j * 4 + kk]));
            if (LO) accG[j * 4 + kk] = fma(s, dl[j * 4 + kk], fma(b, sxl[kk], accG[j * 4 + kk]));
          }
        }
      }
    }
  }
  if (ks > 1) {
    constexpr uint cap = NT / 2;
    threadgroup float* redU = red + (MAXKS - 1) * cap * 32;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) {
        red[((sg - 1) * cap + i) * 32 + lane] = accG[i];
        redU[((sg - 1) * cap + i) * 32 + lane] = accU[i];
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) {
        accG[i] += red[((j - 1) * cap + i) * 32 + lane];
        accU[i] += redU[((j - 1) * cap + i) * 32 + lane];
      }
  }
  if ((lane >> 3) & 1) return;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk)
      he[(u0 * 8 + t0 + kk) * p.Imax + n] = silu(accG[j * 4 + kk]) * accU[j * 4 + kk];
  }
}

// ---- the routed experts' down: plane u0's h table x the expert's [D][I] codes, scattered into out8_ ----
// out8_[(t * (k + 1) + slot) * D + n] = acc, slot = uslot[u0 * 8 + t], for the rows t the umask bit selects: the
// old moe_q20_down's out planes, which qmt8v_combine sums in the slot order.
template <int NT, int PF, int MAXKS, int LO>
NI_INLINE void qmt8v_dn20_impl(const device uint* dcod, const device half* ddd, const device half* T,
                               const device float* sums, const device half* Tlo, const device float* sums_lo,
                               device float* out, const device uint* umask, const device uchar* uslot, uint u0,
                               constant Qmt8vParams& p, uint ks, threadgroup float* red, uint tg, uint sg,
                               uint lane) {
  const uint N = p.D, K = p.I, n0 = tg * NT, gpr = K / 64, wpr = K / 16;
  const uint gper = (gpr + ks - 1) / ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(K, 8));
  auto Rx0 = tX.template slice<64, 8>(0, 0);
  auto tL = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)Tlo, dextents<int32_t, 2>(K, 8));
  auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* dp[NJ];
  const device half* dd[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    dp[j] = dcod + row * wpr + c;
    dd[j] = ddd + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  uint dwn[NJ];
  half ddn[NJ];
  auto load = [&](uint g) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      dwn[j] = dp[j][4 * g];  // a Q2_0 group is 4 words wide; this lane takes word c of it
      ddn[j] = dd[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint dw[NJ];
    half dsc[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      dw[j] = dwn[j];
      dsc[j] = ddn[j];
    }
    if (PF && g + 1 < g1) load(g + 1);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk)
          Lw[i + kk] = uchar((dw[j] >> (16u * uint(jb >> 1) + 4u * uint(kk) + 2u * uint(jb & 1))) & 3u);
      }
    auto Rx = tX.template slice<64, 8>(g * 64, 0);
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
    op.run(Lw, Rx, d);
    const float4 sx = *(const device float4*)(sums + g * 8 + t0);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      const float ds = float(dsc[j]);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(ds, d[j * 4 + kk] - sx[kk], acc[j * 4 + kk]);
    }
    if (LO) {  // the lo term: the same codes against lo(h), and d * sum(lo)
      auto Rxl = tL.template slice<64, 8>(g * 64, 0);
      auto dl = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rxl), float>();
      op.run(Lw, Rxl, dl);
      const float4 sxl = *(const device float4*)(sums_lo + g * 8 + t0);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NJ; ++j) {
        const float ds = float(dsc[j]);
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(ds, dl[j * 4 + kk] - sxl[kk], acc[j * 4 + kk]);
      }
    }
  }
  if (ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;  // padding rows
  const uint mask = umask[u0];
  uint slotv[8];
  _Pragma("clang loop unroll(full)")
  for (uint t = 0; t < 8; ++t) slotv[t] = (mask >> t) & 1u ? uslot[u0 * 8 + t] : 0u;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) {
      const uint t = t0 + kk;
      if (!((mask >> t) & 1u)) continue;  // this token did not pick expert u0: its plane is another threadgroup's
      out[(t * (p.k + 1) + slotv[t]) * p.D + n] = acc[j * 4 + kk];
    }
  }
}

// ---- the shared expert's down: the SI-stride h table x [D][SI] affine weights, into every token's plane k ----
template <int NT, int PF, int MAXKS, int LO>
NI_INLINE void qmt8v_dn4_impl(const device uint* dw, const device ushort* ds, const device ushort* db,
                              const device half* T, const device float* sums, const device half* Tlo,
                              const device float* sums_lo, device float* out, constant Qmt8vParams& p, uint ks,
                              threadgroup float* red, uint tg, uint sg, uint lane) {
  const uint N = p.D, K = p.SI, n0 = tg * NT, gpr = K / 64, wpr = K / 8;
  const uint gper = (gpr + ks - 1) / ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(K, 8));
  auto Rx0 = tX.template slice<64, 8>(0, 0);
  auto tL = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)Tlo, dextents<int32_t, 2>(K, 8));
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
    wp[j] = dw + row * wpr + 2 * c;
    sp[j] = ds + row * gpr;
    bp[j] = db + row * gpr;
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
    for (int j = 0; j < NJ; ++j) {
      w[j] = wn[j];
      sc[j] = sn[j];
      bc[j] = bn[j];
    }
    if (PF && g + 1 < g1) load(g + 1);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j)
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        const uint u = ((jb >> 1) ? w[j].y : w[j].x) >> (4 * (jb & 1));
        const uint v = u & 0x0F0F0F0F;
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        Lw[i] = uchar(v);
        Lw[i + 1] = uchar(v >> 8);
        Lw[i + 2] = uchar(v >> 16);
        Lw[i + 3] = uchar(v >> 24);
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
    if (LO) {
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
  if (ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk) out[((t0 + kk) * (p.k + 1) + p.k) * p.D + n] = acc[j * 4 + kk];
  }
}

// qmt8v_hsum: one he8_ plane's [8][Imax] fp32 rows into the permuted half T tables [8][K] + lo + group sums the
// down GEMMs read as their right operand -- x8_prep2's staging, sums and write_t arithmetic with the norm / xn
// stripped and the 256-input blocks masked (I = 640 is no multiple of 256). Both lo tables are written always
// (the LO = 0 instantiations never read them).
// grid: (ceil(Imax / 256), 1, 8 k + 1) x 256 threads; plane 8 k is the shared expert. Buffers: 0 he8_, 1-4 the
// routed tables / sums (plane u's at u * 8 * I halfs / u * (I / 64) * 8 floats), 5-8 the shared's (stride SI),
// 9 un (moe_q20_unique's count: planes >= un[0] are stale and skipped, nothing reads them), 10 params.
[[kernel]] void qmt8v_hsum(const device float* he [[buffer(0)]], device half* tab [[buffer(1)]],
                           device half* tabl [[buffer(2)]], device float* sums [[buffer(3)]],
                           device float* sums_lo [[buffer(4)]], device half* stab [[buffer(5)]],
                           device half* stabl [[buffer(6)]], device float* ssums [[buffer(7)]],
                           device float* ssums_lo [[buffer(8)]], const device uint* un [[buffer(9)]],
                           constant Qmt8vParams& p [[buffer(10)]],
                           uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half stg[2][8][256];
  const uint pln = tgp.z, tg = tgp.x, Rk = 8 * p.k;
  const bool shared = pln == Rk;
  if (pln < Rk && pln >= un[0]) return;
  const uint K = shared ? p.SI : p.I, G = K / 64;
  device half* tt = shared ? stab : tab + ulong(pln) * 8 * p.I;
  device half* ttl = shared ? stabl : tabl + ulong(pln) * 8 * p.I;
  device float* sm = shared ? ssums : sums + ulong(pln) * G * 8;
  device float* sml = shared ? ssums_lo : sums_lo + ulong(pln) * G * 8;
  const uint t = sg;  // simdgroup = token row
  const device float* hr = he + (pln * 8 + t) * p.Imax;
  const uint k0 = tg * 256 + lane * 8;
  float v[8];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; ++i) v[i] = k0 + uint(i) < K ? hr[k0 + i] : 0.0f;
  float sh = 0.0f, sl = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const half h = half(v[i]);
    stg[0][t][lane * 8 + i] = h;
    sh += float(h);
    const half l = half(v[i] - float(h));
    stg[1][t][lane * 8 + i] = l;
    sl += float(l);
  }
  // group sums: the 8 lanes of a group (lane / 8) in a fixed order
  sh += simd_shuffle_xor(sh, 1u);
  sh += simd_shuffle_xor(sh, 2u);
  sh += simd_shuffle_xor(sh, 4u);
  sl += simd_shuffle_xor(sl, 1u);
  sl += simd_shuffle_xor(sl, 2u);
  sl += simd_shuffle_xor(sl, 4u);
  if ((lane & 7) == 0 && tg * 4 + lane / 8 < G) {
    sm[(tg * 4 + lane / 8) * 8 + t] = sh;
    sml[(tg * 4 + lane / 8) * 8 + t] = sl;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // T table: thread = (row tid / 32, 8 slots of this threadgroup's 256), the tail block scalar-masked
  const uint tr = tid / 32, s0 = (tid % 32) * 8, gb = s0 & ~63u, base = tg * 256 + s0;
  if (base >= K) return;
  half o[8], ol[8];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; ++i) {
    o[i] = stg[0][tr][gb + v8t_input_of_slot((s0 + i) & 63)];
    ol[i] = stg[1][tr][gb + v8t_input_of_slot((s0 + i) & 63)];
  }
  if (base + 8 <= K) {
    *(device uint4*)(tt + tr * K + base) = uint4(as_type<uint>(half2(o[0], o[1])), as_type<uint>(half2(o[2], o[3])),
                                                 as_type<uint>(half2(o[4], o[5])), as_type<uint>(half2(o[6], o[7])));
    *(device uint4*)(ttl + tr * K + base) =
        uint4(as_type<uint>(half2(ol[0], ol[1])), as_type<uint>(half2(ol[2], ol[3])), as_type<uint>(half2(ol[4], ol[5])),
              as_type<uint>(half2(ol[6], ol[7])));
  } else {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 8; ++i)
      if (base + uint(i) < K) {
        tt[tr * K + base + i] = o[i];
        ttl[tr * K + base + i] = ol[i];
      }
  }
}

// qmt8v_combine: y[t][d] += sum_j wts[t * (k + 1) + j] * out8_[(t * (k + 1) + j) * D + d], j = 0 .. k in the old
// moe_q20_down's slot order; one thread per (t, d), so one writer per output element. Buffers: 0 out8_, 1 wts,
// 2 y, 3 params. grid: ceil(R * D / 256) x 256 threads.
[[kernel]] void qmt8v_combine(const device float* out [[buffer(0)]], const device float* wts [[buffer(1)]],
                              device float* y [[buffer(2)]], constant Qmt8vParams& p [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint o = tg * 256 + tid;
  if (o >= p.R * p.D) return;
  const uint t = o / p.D, d = o % p.D;
  float tot = 0.0f;
  for (uint j = 0; j <= p.k; ++j) tot += wts[t * (p.k + 1) + j] * out[(t * (p.k + 1) + j) * p.D + d];
  y[t * p.D + d] += tot;
}

// ---- the entries: NT = 16 (one 16-row weight tile), MAXKS = 8; 'p' = prefetch, '1' = single-term (LO = 0) ----

// gu20 buffers: 0-3 the routed gate/up weights (stacked: gate codes, gate d, up codes, up d; arena: chunk 0,
// chunk 1, the slot table, the slot table), 4 T, 5 sums, 6 Tlo, 7 sums_lo (x8_prep2's, K = D), 8 he8_, 9 ue,
// 10 un, 11 params. grid: (I / 16, 1, 8 k) x (32 * ks_gu) threads.
#define NI_QMT8V_GU20(NAME, PF, LO)                                                                              \
  [[kernel]] void NAME(const device uchar* w0 [[buffer(0)]], const device uchar* w1 [[buffer(1)]],               \
                       const device uchar* w2 [[buffer(2)]], const device uchar* w3 [[buffer(3)]],               \
                       const device half* T [[buffer(4)]], const device float* sums [[buffer(5)]],               \
                       const device half* Tlo [[buffer(6)]], const device float* sums_lo [[buffer(7)]],          \
                       device float* he [[buffer(8)]], const device uint* ue [[buffer(9)]],                      \
                       const device uint* un [[buffer(10)]], constant Qmt8vParams& p [[buffer(11)]],             \
                       uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                                \
    threadgroup float red[2 * (8 - 1) * (16 / 2) * 32]; /* the gate and the up accumulator's k-slices */         \
    const uint u0 = tgp.z;                                                                                       \
    if (u0 >= un[0]) return;                                                                                     \
    const device uint* gcod;                                                                                     \
    const device half* gdd;                                                                                      \
    const device uint* ucod;                                                                                     \
    const device half* udd;                                                                                      \
    if (p.arena) {                                                                                               \
      const uint sw = ((const device uint*)w2)[ue[u0]];                                                          \
      const device uchar* b = ((sw >> 28) != 0u ? w1 : w0) + ulong(sw & 0x0FFFFFFFu) * 16384ul;                  \
      gcod = (const device uint*)(b + p.lgc);                                                                    \
      gdd = (const device half*)(b + p.lgd);                                                                     \
      ucod = (const device uint*)(b + p.luc);                                                                    \
      udd = (const device half*)(b + p.lud);                                                                     \
    } else {                                                                                                     \
      const ulong cs = ulong(p.I) * (p.D / 16), ds = ulong(p.I) * (p.D / 64), e = ue[u0];                        \
      gcod = (const device uint*)w0 + e * cs;                                                                    \
      gdd = (const device half*)w1 + e * ds;                                                                     \
      ucod = (const device uint*)w2 + e * cs;                                                                    \
      udd = (const device half*)w3 + e * ds;                                                                     \
    }                                                                                                            \
    qmt8v_gu20_impl<16, PF, 8, LO>(gcod, gdd, ucod, udd, T, sums, Tlo, sums_lo, he, u0, p, p.ks_gu, red, tgp.x,  \
                                   sg, lane);                                                                    \
  }

// gu4 buffers: 0-5 the shared gate/up (codes, scales, biases; the scales / biases are bf16 read as ushort), 6 T,
// 7 sums, 8 Tlo, 9 sums_lo, 10 he8_, 11 params. grid: (SI / 16, 1, 1) x (32 * ks_gu) threads.
#define NI_QMT8V_GU4(NAME, PF, LO)                                                                               \
  [[kernel]] void NAME(const device uint* sgw [[buffer(0)]], const device ushort* sgs [[buffer(1)]],             \
                       const device ushort* sgb [[buffer(2)]], const device uint* suw [[buffer(3)]],             \
                       const device ushort* sus [[buffer(4)]], const device ushort* sub [[buffer(5)]],           \
                       const device half* T [[buffer(6)]], const device float* sums [[buffer(7)]],               \
                       const device half* Tlo [[buffer(8)]], const device float* sums_lo [[buffer(9)]],          \
                       device float* he [[buffer(10)]], constant Qmt8vParams& p [[buffer(11)]],                  \
                       uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                                \
    threadgroup float red[2 * (8 - 1) * (16 / 2) * 32];                                                          \
    qmt8v_gu4_impl<16, PF, 8, LO>(sgw, sgs, sgb, suw, sus, sub, T, sums, Tlo, sums_lo, he, 8 * p.k, p, p.ks_gu,  \
                                  red, tgp.x, sg, lane);                                                         \
  }

// dn20 buffers: 0-2 the routed down weights (stacked: down codes, down d, down codes; arena: chunk 0, chunk 1,
// the slot table), 3-6 the routed h tables / sums, 7 out8_, 8 umask, 9 uslot, 10 ue, 11 un, 12 params.
// grid: (D / 16, 1, 8 k) x (32 * ks_d20) threads.
#define NI_QMT8V_DN20(NAME, PF, LO)                                                                              \
  [[kernel]] void NAME(const device uchar* w0 [[buffer(0)]], const device uchar* w1 [[buffer(1)]],               \
                       const device uchar* w2 [[buffer(2)]], const device half* T [[buffer(3)]],                 \
                       const device float* sums [[buffer(4)]], const device half* Tlo [[buffer(5)]],             \
                       const device float* sums_lo [[buffer(6)]], device float* out [[buffer(7)]],               \
                       const device uint* umask [[buffer(8)]], const device uchar* uslot [[buffer(9)]],          \
                       const device uint* ue [[buffer(10)]], const device uint* un [[buffer(11)]],               \
                       constant Qmt8vParams& p [[buffer(12)]], uint3 tgp [[threadgroup_position_in_grid]],       \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {    \
    threadgroup float red[(8 - 1) * (16 / 2) * 32];                                                              \
    const uint u0 = tgp.z;                                                                                       \
    if (u0 >= un[0]) return;                                                                                     \
    const device uint* dcod;                                                                                     \
    const device half* ddd;                                                                                      \
    if (p.arena) {                                                                                               \
      const uint sw = ((const device uint*)w2)[ue[u0]];                                                          \
      const device uchar* b = ((sw >> 28) != 0u ? w1 : w0) + ulong(sw & 0x0FFFFFFFu) * 16384ul;                  \
      dcod = (const device uint*)(b + p.ldc);                                                                    \
      ddd = (const device half*)(b + p.ldd);                                                                     \
    } else {                                                                                                     \
      const ulong cs = ulong(p.D) * (p.I / 16), ds = ulong(p.D) * (p.I / 64), e = ue[u0];                        \
      dcod = (const device uint*)w0 + e * cs;                                                                    \
      ddd = (const device half*)w1 + e * ds;                                                                     \
    }                                                                                                            \
    qmt8v_dn20_impl<16, PF, 8, LO>(dcod, ddd, T + ulong(u0) * 8 * p.I, sums + ulong(u0) * (p.I / 64) * 8,        \
                                   Tlo + ulong(u0) * 8 * p.I, sums_lo + ulong(u0) * (p.I / 64) * 8, out, umask,  \
                                   uslot, u0, p, p.ks_d20, red, tgp.x, sg, lane);                                \
  }

// dn4 buffers: 0-2 the shared down (codes, scales, biases), 3-6 the shared h tables / sums, 7 out8_, 8 params.
// grid: (D / 16, 1, 1) x (32 * ks_d4) threads.
#define NI_QMT8V_DN4(NAME, PF, LO)                                                                               \
  [[kernel]] void NAME(const device uint* dw [[buffer(0)]], const device ushort* ds [[buffer(1)]],               \
                       const device ushort* db [[buffer(2)]], const device half* T [[buffer(3)]],                \
                       const device float* sums [[buffer(4)]], const device half* Tlo [[buffer(5)]],             \
                       const device float* sums_lo [[buffer(6)]], device float* out [[buffer(7)]],               \
                       constant Qmt8vParams& p [[buffer(8)]], uint3 tgp [[threadgroup_position_in_grid]],        \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {    \
    threadgroup float red[(8 - 1) * (16 / 2) * 32];                                                              \
    qmt8v_dn4_impl<16, PF, 8, LO>(dw, ds, db, T, sums, Tlo, sums_lo, out, p, p.ks_d4, red, tgp.x, sg, lane);     \
  }

NI_QMT8V_GU20(qmt8v_gu20, 0, 1)
NI_QMT8V_GU20(qmt8v_gu20p, 1, 1)
NI_QMT8V_GU20(qmt8v_gu201, 0, 0)
NI_QMT8V_GU20(qmt8v_gu201p, 1, 0)
NI_QMT8V_GU4(qmt8v_gu4, 0, 1)
NI_QMT8V_GU4(qmt8v_gu4p, 1, 1)
NI_QMT8V_GU4(qmt8v_gu41, 0, 0)
NI_QMT8V_GU4(qmt8v_gu41p, 1, 0)
NI_QMT8V_DN20(qmt8v_dn20, 0, 1)
NI_QMT8V_DN20(qmt8v_dn20p, 1, 1)
NI_QMT8V_DN20(qmt8v_dn201, 0, 0)
NI_QMT8V_DN20(qmt8v_dn201p, 1, 0)
NI_QMT8V_DN4(qmt8v_dn4, 0, 1)
NI_QMT8V_DN4(qmt8v_dn4p, 1, 1)
NI_QMT8V_DN4(qmt8v_dn41, 0, 0)
NI_QMT8V_DN4(qmt8v_dn41p, 1, 0)
