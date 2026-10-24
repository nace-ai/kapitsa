#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qtm_q20: the qmt tile (qmt_moe.metal's 64-row design) on the Q2_0 experts' grouped GEMMs ----
// The Flash-Next Q2_0 expert store (src/model/moe_q20.cpp) runs its routed prefill GEMMs on the qrm-design kernels
// (qrm_*_q20_m32_*, SIMD dequantization into MMA fragments, ~7 TFLOP/s class) while the M5's matrix units reach
// 21-23 on the dense qmt tile. Here the same tile as qmt_moe.metal, driven by the same moe_bucket tiles, with Q2_0
// weight rows: one 32-bit code word (16 codes, 2 bits) and one half scale a row, w = (code - 1) * d. The decode is
// exact in one half FMA a weight: the four products -d, 0, d and 2d are all exactly representable (2d is a shift),
// as zq2moe.metal's decode kernels already rely on. The tile, the double-buffered k-loop and both store paths are
// qmt_moe.metal's qtmm_tile unchanged except for the weight read; the per-element arithmetic is the qrm_q20
// kernels' (same exact halfs, fp32 MMA accumulation over K).
//  * gate/up (NP = 2): BN / 2 gate weight rows + BN / 2 up rows of the same columns; H = silu(g) * u as half in
//    sorted-row order (qrm_gu_q20's interface and output).
//  * down (NP = 1): Y[yrows[m]] (fp32 rows by pair id) = H[m] W_e^T (qrm_dn_q20's interface and output).
// Buffers and params are the qrm_*_q20 kernels' (bind_gu / bind_dn, Qmm2 + Q20Slot): stacked planes address the
// expert through w_expert_stride / s_expert_stride, the arena through the slot word (chunk << 28 | 16 KiB page)
// plus the plane offsets, exactly as zq2moe.metal's q20_mat does. No shared-expert tiles (shared_e = E + 1).
// Requirements (host): K % 64 == 0, N % (BN / 2) == 0 (gu) / N % BN == 0 (dn), x_ld % 8 == 0, y_ld % 8 == 0.

// Qmm2 with the Q2_0 plane strides: N, K, x_ld, y_ld, gs, the codes' / scales' per-expert strides, shared_e
struct Qmm2Params {
  uint N, K, x_ld, y_ld, gs, w_expert_stride, s_expert_stride, shared_e;
};

// The expert's Q2_0 planes (zq2moe.metal's Q20Mat)
struct Q20MatT {
  const device uint32_t* c;  // codes, K/16 words a row
  const device half* d;      // scales, K/64 halfs a row
};

// Byte offsets of the planes inside an arena slot (zq2moe.metal's Q20Slot; the host's kp::Q20Slot)
struct Q20Slot {
  uint gc, gd, uc, ud, dc, dd;
};

// Expert e's matrix. Stacked: base0 / base1 = its codes / d plane, cs / ds = the planes' per-expert strides (words /
// halfs). SLOT: base0 / base1 = arena chunks, coff / doff = the matrix's offsets inside a slot (zq2moe.metal's
// q20_mat exactly: slot word = chunk << 28 | a 16 KiB page index).
template <bool SLOT>
NI_INLINE Q20MatT q20_mat20(const device uchar* base0, const device uchar* base1, const device uint* slots, uint e,
                            uint cs, uint ds, uint coff, uint doff) {
  if (SLOT) {
    const uint sw = slots[e];
    const device uchar* b = ((sw >> 28) != 0u ? base1 : base0) + ulong(sw & 0x0FFFFFFFu) * 16384ul;
    return Q20MatT{(const device uint32_t*)(b + coff), (const device half*)(b + doff)};
  }
  return Q20MatT{(const device uint32_t*)base0 + ulong(e) * cs, (const device half*)base1 + ulong(e) * ds};
}

// One weight row's 16 codes and the 64-block scale (k % 16 == 0)
struct Qtm20Raw {
  uint w;
  half d;
};
NI_INLINE Qtm20Raw qtm20_raw(Q20MatT m, uint K, uint n, uint k) {
  Qtm20Raw r;
  r.w = m.c[size_t(n) * (K / 16) + k / 16];
  r.d = m.d[size_t(n) * (K / 64) + k / 64];
  return r;
}

// w = (code - 1) * d, exact in one half FMA: -d, 0, d and 2d are exactly representable halfs
NI_INLINE void qtm20_dec(const Qtm20Raw r, threadgroup half* dst) {
  half w[16];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 16; ++j) w[j] = fma(half(float((r.w >> (2 * j)) & 3u)), r.d, -r.d);
  *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
  *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
  *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
  *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
}

// One bucket tile (rows = the tile's live rows, m0 = its first sorted row) at output columns n0:
// NP = 2 writes half(silu(g) * u) to H row m0 + m; NP = 1 writes fp32 Y[yrows[m0 + m]]. xrows == nullptr: the tile's
// X rows are contiguous (down); otherwise xrows maps them to token rows (gate/up). qmt_moe.metal's qtmm_tile with
// the Q2_0 weight read.
template <int NP, int BM, int BN, int BK, bool STG = true>
NI_INLINE void qtmm20_tile(const device half* X, thread const Q20MatT* M, device void* Y, constant Qmm2Params& p,
                           uint m0, uint rows, uint n0, const device uint* xrows, const device uint* yrows, uint tid,
                           uint sg, threadgroup half* xs, threadgroup half* ws) {
  constexpr int SM = BM / 2, SN = BN / 2;
  const uint sm = sg / 2, sn = sg % 2;
  constexpr auto desc =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tg_t;
  auto tX0 = tg_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tX1 = tg_t(xs + BM * BK + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  constexpr int WBUF = BN * BK;
  auto tW0 = tg_t(ws + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto tW1 = tg_t(ws + WBUF + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tX0), decltype(tW0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int XN = BM * BK / 8, XP = (XN + 127) / 128;
  uint xrow[XP], xk[XP], xr[XP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < XP; ++i) {
    const uint j = tid + 128 * i;
    xr[i] = j / (BK / 8);
    xk[i] = (j % (BK / 8)) * 8;
    const uint r = xr[i];
    xrow[i] = j < XN && r < rows ? (xrows ? xrows[m0 + r] : m0 + r) : 0xffffffffu;
  }
  constexpr int WP = BN * BK / 16 / 128;
  uint wr[WP], wk[WP], wn[WP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < WP; ++i) {
    const uint j = tid + 128 * i;
    wr[i] = j / (BK / 16);
    wk[i] = (j % (BK / 16)) * 16;
    wn[i] = NP == 2 ? min(n0 + (wr[i] % (BN / 2)), p.N - 1) : min(n0 + wr[i], p.N - 1);
  }
  uint4 xv[XP];
  Qtm20Raw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(X + size_t(xrow[i]) * p.x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) {
      const int q = NP == 2 && wr[i] >= BN / 2 ? 1 : 0;
      wv[i] = qtm20_raw(M[q], p.K, wn[i], k0 + wk[i]);
    }
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) *(threadgroup uint4*)(xs + b * BM * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) qtm20_dec(wv[i], ws + b * WBUF + wr[i] * BK + wk[i]);
  };
  load(0);
  store(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint steps = p.K / BK;
  // a simdgroup whose SM rows are all past the tile's live rows skips its products (its outputs are never stored);
  // part of the staged kernels only, so the *_o ones (NI_MOE_STAGE=0) are main's kernel exactly
  const bool live = !STG || sm * SM < rows;
  for (uint st = 0; st < steps; ++st) {
    const bool more = st + 1 < steps;
    if (more) load((st + 1) * BK);
    if (live) {
      if (st & 1) op.run(tX1, tW1, acc);
      else op.run(tX0, tW0, acc);
    }
    if (more) store((st + 1) & 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (NP == 2) {
    // up products to threadgroup memory (ws is free after the last barrier), then the gate column combines them
    threadgroup float* ex = (threadgroup float*)ws + (sm * 32 + (tid & 31)) * acc.get_capacity();
    if (sn == 1)
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) ex[i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!STG) {  // main's element stores (NI_MOE_STAGE=0)
      if (sn == 0)
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
          if (!acc.is_valid_element(i)) continue;
          const auto ix = acc.get_multidimensional_index(i);
          const uint n = n0 + ix[0], r = sm * SM + ix[1];
          if (r >= rows || n >= p.N) continue;
          ((device half*)Y)[size_t(m0 + r) * p.y_ld + n] = half(silu(acc[i]) * ex[i]);
        }
      return;
    }
    // H goes out as whole rows (SN halfs each) through xs, as the down tile's fp32 rows do
    static_assert(BM * SN * 2 <= 2 * BM * BK * 2 && (SN * 2) % 16 == 0, "qtmm20_tile: H staging");
    threadgroup half* hs = xs;
    if (sn == 0)
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto ix = acc.get_multidimensional_index(i);
        hs[(sm * SM + ix[1]) * SN + ix[0]] = half(silu(acc[i]) * ex[i]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int Q = SN / 8;  // uint4 (8 halfs) per row
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < (BM * Q + 127) / 128; ++i) {
      const uint j = tid + 128 * i, r = j / Q, c = (j % Q) * 8;
      if (j >= BM * Q || r >= rows) continue;
      *(device uint4*)((device half*)Y + size_t(m0 + r) * p.y_ld + n0 + c) = *(const threadgroup uint4*)(hs + r * SN + c);
    }
    return;
  }
  // the fp32 tile goes out through threadgroup memory as whole rows of float4 (as qmt_moe.metal's down tile does)
  static_assert(SM * BN * 4 <= 2 * BM * BK * 2 && SM * BN * 4 <= 2 * BN * BK * 2, "qtmm20_tile: staging buffer");
  if (!STG) {  // main's element stores (NI_MOE_STAGE=0)
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto ix = acc.get_multidimensional_index(i);
      const uint n = n0 + sn * SN + ix[0], r = sm * SM + ix[1];
      if (r >= rows || n >= p.N) continue;
      ((device float*)Y)[size_t(yrows[m0 + r]) * p.y_ld + n] = acc[i];
    }
    return;
  }
  threadgroup float* st = (threadgroup float*)(sm ? xs : ws);
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    st[ix[1] * BN + sn * SN + ix[0]] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int Q = BN / 4;  // float4 per row
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < BM * Q / 128; ++i) {
    const uint j = tid + 128 * i, r = j / Q, c = (j % Q) * 4;
    if (r >= rows) continue;
    const threadgroup float* src = (const threadgroup float*)(r < SM ? ws : xs) + (r % SM) * BN + c;
    *(device float4*)((device float*)Y + size_t(yrows[m0 + r]) * p.y_ld + n0 + c) = *(const threadgroup float4*)src;
  }
}

// Fused expert gate/up (+ SwiGLU) on moe_bucket tiles: buffers as qrm_gu_q20_* (bind_gu; shared_e is E + 1 — the
// shared expert runs on the dense path, no shared buffers in use). grid (I / (BN / 2), max tiles), 128 threads.
#define NI_QTM20_GU(NAME, BM, BN, BK, STG, SLOT)                                                                       \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                        \
      const device half* X [[buffer(0)]], const device uchar* w0 [[buffer(1)]], const device uchar* w1 [[buffer(2)]],  \
      const device uchar* w2 [[buffer(3)]], const device uchar* w3 [[buffer(4)]], device half* H [[buffer(5)]],        \
      constant Qmm2Params& p [[buffer(6)]], const device uint* tiles [[buffer(7)]],                                    \
      const device uint* ntiles [[buffer(8)]], const device uint* xrows [[buffer(9)]],                                 \
      constant Q20Slot& L [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],                                   \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                          \
    if (tg.y >= ntiles[0]) return;                                                                                     \
    threadgroup half xs[2 * BM * BK];                                                                                  \
    threadgroup half ws[2 * BN * BK];                                                                                  \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;                     \
    const device uint* slots = (const device uint*)w2;                                                                 \
    const Q20MatT M[2] = {q20_mat20<SLOT>(w0, w1, slots, e, p.w_expert_stride, p.s_expert_stride, L.gc, L.gd),         \
                          q20_mat20<SLOT>(SLOT ? w0 : w2, SLOT ? w1 : w3, slots, e, p.w_expert_stride,                 \
                                          p.s_expert_stride, L.uc, L.ud)};                                             \
    qtmm20_tile<2, BM, BN, BK, STG>(X, M, H, p, m0, rows, tg.x * (BN / 2), xrows, nullptr, tid, sg, xs, ws);           \
  }

// Grouped down GEMM to fp32 rows by pair: buffers as qrm_dn_q20_* (bind_dn). grid (D / BN, max tiles), 128 threads.
#define NI_QTM20_DN(NAME, BM, BN, BK, STG, SLOT)                                                                       \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                        \
      const device half* X [[buffer(0)]], const device uchar* w0 [[buffer(1)]], const device uchar* w1 [[buffer(2)]],  \
      const device uchar* w2 [[buffer(3)]], device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]],       \
      const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],                                 \
      const device uint* yrows [[buffer(8)]], constant Q20Slot& L [[buffer(9)]],                                       \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                             \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                    \
    if (tg.y >= ntiles[0]) return;                                                                                     \
    threadgroup half xs[2 * BM * BK];                                                                                  \
    threadgroup half ws[2 * BN * BK];                                                                                  \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;                     \
    const Q20MatT M[1] = {                                                                                             \
        q20_mat20<SLOT>(w0, w1, (const device uint*)w2, e, p.w_expert_stride, p.s_expert_stride, L.dc, L.dd)};         \
    qtmm20_tile<1, BM, BN, BK, STG>(X, M, Y, p, m0, rows, tg.x * BN, nullptr, yrows, tid, sg, xs, ws);                 \
  }

// naming as qrm_q20 / qmt_moe: qmtm_{gu,dn}_q20_m64_n64, "_o" = main's element stores (NI_MOE_STAGE=0), "_slot" = the
// arena's slot table (bind_gu / bind_dn pick the suffixes through moe_q20.cpp's pipe())
NI_QTM20_GU(qmtm_gu_q20_m64_n64, 64, 64, 32, true, false)
NI_QTM20_GU(qmtm_gu_q20_m64_n64_slot, 64, 64, 32, true, true)
NI_QTM20_DN(qmtm_dn_q20_m64_n64, 64, 64, 32, true, false)
NI_QTM20_DN(qmtm_dn_q20_m64_n64_slot, 64, 64, 32, true, true)
NI_QTM20_GU(qmtm_gu_q20_m64_n64_o, 64, 64, 32, false, false)
NI_QTM20_GU(qmtm_gu_q20_m64_n64_o_slot, 64, 64, 32, false, true)
NI_QTM20_DN(qmtm_dn_q20_m64_n64_o, 64, 64, 32, false, false)
NI_QTM20_DN(qmtm_dn_q20_m64_n64_o_slot, 64, 64, 32, false, true)
