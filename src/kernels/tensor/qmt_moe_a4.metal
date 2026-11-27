#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qmtm_a4: the qmt tile (qmt_moe.metal's 64-row design) on Hy4-flash's routed experts' grouped GEMMs ----
// Hy4-flash's 4-bit affine experts (src/model/moe_a4.h) run their prefill GEMMs on a4_gu_mm / a4_dn_mm
// (hy4/a4moe.metal: SIMD dequantization into 8x8 simdgroup MMA fragments, ~12 TFLOP/s on the M5 Max at 2048-row chunks)
// while the M5's matrix units run the trunk's 8-bit GEMMs at ~50. Here qmt_moe_q20.metal's tile (the same tile,
// moe_bucket3's tiles of 64 rows, the double-buffered k-loop and both store paths) with the a4 weight read: a row's
// 4-bit codes (MLX's packing, 8 a word), its group's bf16 scale and bias, w = half(scale * code + bias) in fp32 as
// a4_tile writes it, so the weights are a4_gu_mm's bit for bit and only the order of the fp32 MMA sums differs.
//  * gate/up (NP = 2): BN / 2 gate rows + BN / 2 up rows of the same columns; H = a4_swiglu(g, u) (silu of the gate
//    clamped above at limit, the up clamped to [-limit, limit]; limit <= 0: no clamp) as half in sorted-row order,
//    a4_gu_mm's interface and output.
//  * down (NP = 1): Y[yrows[m]] (fp32 rows by pair) = H[m] W_e^T, a4_dn_mm's interface and output.
// Buffers and params are a4_gu_mm's / a4_dn_mm's (moe_a4.cpp's bind_mat): stacked planes [E][N][...], or the arena
// through the slot table (chunk << 28 | 16 KiB page) plus A4Slot's plane offsets. 4-bit only, group 32, 64 or 128.
// Requirements (host): K % 32 == 0, N % 32 == 0 (gu) / N % 64 == 0 (dn), D % 8 == 0, I % 8 == 0.

struct A4tParams {  // hy4/a4moe.metal's A4Params
  uint D, I, k, E;
  float limit;
  uint R;
};
struct A4tSlot {  // A4Slot: plane byte offsets inside a slot, gate w s b, up w s b, down w s b
  uint o[9];
};
struct A4tMat {
  const device uint32_t* w;
  const device bfloat* s;
  const device bfloat* b;
};

// Expert e's matrix m (0 gate, 1 up, 2 down) of N rows and K columns (a4moe.metal's a4_mat at 4 bits)
template <bool SLOT, int GS>
NI_INLINE A4tMat a4t_mat(const device uchar* w0, const device uchar* w1, const device uchar* w2, uint e, uint N,
                         uint K, constant A4tSlot& L, uint m) {
  if (SLOT) {
    const uint sw = ((const device uint*)w2)[e];
    const device uchar* b = ((sw >> 28) != 0u ? w1 : w0) + ulong(sw & 0x0FFFFFFFu) * 16384ul;
    return A4tMat{(const device uint32_t*)(b + L.o[3 * m]), (const device bfloat*)(b + L.o[3 * m + 1]),
                  (const device bfloat*)(b + L.o[3 * m + 2])};
  }
  const ulong r = ulong(e) * N;
  return A4tMat{(const device uint32_t*)w0 + r * (K / 8), (const device bfloat*)w1 + r * (K / GS),
                (const device bfloat*)w2 + r * (K / GS)};
}

// One weight row's 16 codes from k (k % 16 == 0: two words) and their group's scale and bias
struct A4tRaw {
  uint w0, w1;
  float s, b;
};
template <int GS>
NI_INLINE A4tRaw a4t_raw(A4tMat m, uint K, uint n, uint k) {
  A4tRaw r;
  const device uint32_t* row = m.w + size_t(n) * (K / 8) + k / 8;
  r.w0 = row[0];
  r.w1 = row[1];
  r.s = float(m.s[size_t(n) * (K / GS) + k / GS]);
  r.b = float(m.b[size_t(n) * (K / GS) + k / GS]);
  return r;
}

// w = half(scale * code + bias), a4_tile's expression
NI_INLINE void a4t_dec(const A4tRaw r, threadgroup half* dst) {
  half w[16];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; ++j) {
    w[j] = half(r.s * float((r.w0 >> (4 * j)) & 15u) + r.b);
    w[8 + j] = half(r.s * float((r.w1 >> (4 * j)) & 15u) + r.b);
  }
  *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
  *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
  *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
  *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
}

// a4moe.metal's a4_swiglu
NI_INLINE float a4t_swiglu(float g, float u, float lim) {
  if (lim > 0.0f) {
    g = min(g, lim);
    u = clamp(u, -lim, lim);
  }
  return silu(g) * u;
}

// One bucket tile (rows = the tile's live rows, m0 = its first sorted row) at output columns n0 of N, K deep:
// NP = 2 writes half(a4t_swiglu(g, u)) to H row m0 + m (ld y_ld); NP = 1 writes fp32 Y[yrows[m0 + m]] (ld y_ld).
// xrows == nullptr: the tile's X rows are contiguous (down); otherwise xrows maps them to token rows (gate/up).
// qmt_moe_q20.metal's qtmm20_tile (staged stores) with the a4 weight read.
template <int NP, int BM, int BN, int BK, int GS>
NI_INLINE void qtmma4_tile(const device half* X, thread const A4tMat* M, device void* Y, uint N, uint K, uint x_ld,
                           uint y_ld, float limit, uint m0, uint rows, uint n0, const device uint* xrows,
                           const device uint* yrows, uint tid, uint sg, threadgroup half* xs, threadgroup half* ws) {
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
    wn[i] = NP == 2 ? min(n0 + (wr[i] % (BN / 2)), N - 1) : min(n0 + wr[i], N - 1);
  }
  uint4 xv[XP];
  A4tRaw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(X + size_t(xrow[i]) * x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) {
      const int q = NP == 2 && wr[i] >= BN / 2 ? 1 : 0;
      wv[i] = a4t_raw<GS>(M[q], K, wn[i], k0 + wk[i]);
    }
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) *(threadgroup uint4*)(xs + b * BM * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) a4t_dec(wv[i], ws + b * WBUF + wr[i] * BK + wk[i]);
  };
  load(0);
  store(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint steps = K / BK;
  // a simdgroup whose SM rows are all past the tile's live rows skips its products (its outputs are never stored)
  const bool live = sm * SM < rows;
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
    // H goes out as whole rows (SN halfs each) through xs, as the down tile's fp32 rows do
    static_assert(BM * SN * 2 <= 2 * BM * BK * 2 && (SN * 2) % 16 == 0, "qtmma4_tile: H staging");
    threadgroup half* hs = xs;
    if (sn == 0)
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto ix = acc.get_multidimensional_index(i);
        hs[(sm * SM + ix[1]) * SN + ix[0]] = half(a4t_swiglu(acc[i], ex[i], limit));
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int Q = SN / 8;  // uint4 (8 halfs) per row
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < (BM * Q + 127) / 128; ++i) {
      const uint j = tid + 128 * i, r = j / Q, c = (j % Q) * 8;
      if (j >= BM * Q || r >= rows) continue;
      *(device uint4*)((device half*)Y + size_t(m0 + r) * y_ld + n0 + c) = *(const threadgroup uint4*)(hs + r * SN + c);
    }
    return;
  }
  // the fp32 tile goes out through threadgroup memory as whole rows of float4 (as qmt_moe.metal's down tile does)
  static_assert(SM * BN * 4 <= 2 * BM * BK * 2 && SM * BN * 4 <= 2 * BN * BK * 2, "qtmma4_tile: staging buffer");
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
    *(device float4*)((device float*)Y + size_t(yrows[m0 + r]) * y_ld + n0 + c) = *(const threadgroup float4*)src;
  }
}

// Fused expert gate/up (+ clamped SwiGLU) on moe_bucket3's 64-row tiles: a4_gu_mm's buffers. grid (I / 32, max tiles),
// 128 threads.
#define NI_QTMA4_GU(NAME, GS, SLOT)                                                                                    \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                        \
      const device half* X [[buffer(0)]], const device uchar* g0 [[buffer(1)]], const device uchar* g1 [[buffer(2)]],  \
      const device uchar* g2 [[buffer(3)]], const device uchar* u0 [[buffer(4)]], const device uchar* u1 [[buffer(5)]], \
      const device uchar* u2 [[buffer(6)]], device half* H [[buffer(7)]], constant A4tParams& p [[buffer(8)]],         \
      constant A4tSlot& L [[buffer(9)]], const device uint* tiles [[buffer(10)]],                                      \
      const device uint* ntiles [[buffer(11)]], const device uint* xrows [[buffer(12)]],                               \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                             \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                    \
    if (tg.y >= ntiles[0]) return;                                                                                     \
    threadgroup half xs[2 * 64 * 32];                                                                                  \
    threadgroup half ws[2 * 64 * 32];                                                                                  \
    const uint e = min(tiles[tg.y * 3 + 0], p.E - 1), m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;      \
    const A4tMat M[2] = {a4t_mat<SLOT, GS>(g0, g1, g2, e, p.I, p.D, L, 0),                                             \
                         a4t_mat<SLOT, GS>(u0, u1, u2, e, p.I, p.D, L, 1)};                                            \
    qtmma4_tile<2, 64, 64, 32, GS>(X, M, H, p.I, p.D, p.D, p.I, p.limit, m0, rows, tg.x * 32, xrows, nullptr, tid, sg, \
                                   xs, ws);                                                                            \
  }

// Grouped down GEMM to fp32 rows by pair: a4_dn_mm's buffers. grid (D / 64, max tiles), 128 threads.
#define NI_QTMA4_DN(NAME, GS, SLOT)                                                                                    \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                        \
      const device half* Hin [[buffer(0)]], const device uchar* d0 [[buffer(1)]], const device uchar* d1 [[buffer(2)]], \
      const device uchar* d2 [[buffer(3)]], device float* Y [[buffer(4)]], constant A4tParams& p [[buffer(5)]],        \
      constant A4tSlot& L [[buffer(6)]], const device uint* tiles [[buffer(7)]],                                       \
      const device uint* ntiles [[buffer(8)]], const device uint* yrows [[buffer(9)]],                                 \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                             \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                    \
    if (tg.y >= ntiles[0]) return;                                                                                     \
    threadgroup half xs[2 * 64 * 32];                                                                                  \
    threadgroup half ws[2 * 64 * 32];                                                                                  \
    const uint e = min(tiles[tg.y * 3 + 0], p.E - 1), m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;      \
    const A4tMat M[1] = {a4t_mat<SLOT, GS>(d0, d1, d2, e, p.D, p.I, L, 2)};                                            \
    qtmma4_tile<1, 64, 64, 32, GS>(Hin, M, Y, p.D, p.I, p.I, p.D, 0.0f, m0, rows, tg.x * 64, nullptr, yrows, tid, sg,  \
                                   xs, ws);                                                                            \
  }

// names: qmtm_{gu,dn}_a4_b4g{32,64,128}[_slot] (moe_a4.cpp's qtm_name)
NI_QTMA4_GU(qmtm_gu_a4_b4g32, 32, false)
NI_QTMA4_GU(qmtm_gu_a4_b4g32_slot, 32, true)
NI_QTMA4_GU(qmtm_gu_a4_b4g64, 64, false)
NI_QTMA4_GU(qmtm_gu_a4_b4g64_slot, 64, true)
NI_QTMA4_GU(qmtm_gu_a4_b4g128, 128, false)
NI_QTMA4_GU(qmtm_gu_a4_b4g128_slot, 128, true)
NI_QTMA4_DN(qmtm_dn_a4_b4g32, 32, false)
NI_QTMA4_DN(qmtm_dn_a4_b4g32_slot, 32, true)
NI_QTMA4_DN(qmtm_dn_a4_b4g64, 64, false)
NI_QTMA4_DN(qmtm_dn_a4_b4g64_slot, 64, true)
NI_QTMA4_DN(qmtm_dn_a4_b4g128, 128, false)
NI_QTMA4_DN(qmtm_dn_a4_b4g128_slot, 128, true)
