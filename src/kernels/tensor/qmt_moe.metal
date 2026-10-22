#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qtm: the qmt tile (MLX affine 4-bit, group 64) on the routed experts' grouped GEMMs (M5's matrix units) ----
// qmt_m64_n64 runs the dense projections at 21-23 TFLOP/s on the M5 Pro while the MoE expert GEMMs (qrm_gu/qrm_dn,
// SIMD dequantization into MMA fragments) reach only ~7: the routed half of a 4K 35B prefill is 53% of it. Here the
// same tile as qmt, driven by moe_bucket2's tiles instead of a dense row range: every tile holds the rows of one
// expert (at most BM), so the weights are indexed once per tile and the tile's rows are staged from their sorted
// positions (gate/up: xrows maps a sorted pair row to its token row; down: contiguous sorted rows).
//  * gate/up (NP = 2): the tile's first BN / 2 weight rows are gate rows, the next BN / 2 the up rows of the same
//    columns; H = silu(g) * u as half in sorted-row order (qrm_gu's interface and output).
//  * down (NP = 1): Y[yrows[m]] (fp32 rows by pair id) = H[m] W_e^T (qrm_dn's interface and output).
//  * the shared expert can ride along as expert index p.shared_e with its own weight buffers, as in qmm_moe_gu_*.
// Per output element the arithmetic is qrm's (same dequantized halfs, fp32 MMA accumulation over K); the halfs are
// rounded once as qmt does (fma in fp32, one rounding to half).
// Requirements (host): 4-bit weights, gs 64 (checked by the caller), K % 64 == 0, N % BN == 0, x_ld % 8 == 0.

struct QtmParams {
  uint N, K, x_ld, y_ld, gs, w_es, s_es, shared_e;
};

// 16 weights of row n at k (k % 16 == 0): 2 packed words, one scale / bias
struct QtmRaw {
  uint2 w;
  float s, b;
};
NI_INLINE QtmRaw qtm_raw(const device uint* W, const device bfloat* S, const device bfloat* B, uint K, uint n, uint k) {
  QtmRaw r;
  r.w = *(const device uint2*)(W + size_t(n) * (K / 8) + k / 8);
  const size_t g = size_t(n) * (K / 64) + k / 64;
  r.s = float(S[g]);
  r.b = float(B[g]);
  return r;
}
NI_INLINE void qtm_dec(const QtmRaw r, threadgroup half* dst) {
  half w[16];
  for (int j = 0; j < 8; ++j) {
    w[j] = half(fma(float((r.w.x >> (4 * j)) & 15u), r.s, r.b));
    w[8 + j] = half(fma(float((r.w.y >> (4 * j)) & 15u), r.s, r.b));
  }
  *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
  *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
  *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
  *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
}

// One bucket tile (rows = the tile's live rows, m0 = its first sorted row) at output columns n0:
// NP = 2 writes half(silu(g) * u) to H row m0 + m; NP = 1 writes fp32 Y[yrows[m0 + m]]. xrows == nullptr: the tile's
// X rows are contiguous (down); otherwise xrows maps them to token rows (gate/up).
template <int NP, int BM, int BN, int BK>
NI_INLINE void qtmm_tile(const device half* X, const device uint* const thread* W, const device bfloat* const thread* S,
                        const device bfloat* const thread* B, device void* Y, constant QtmParams& p, uint m0,
                        uint rows, uint n0, const device uint* xrows, const device uint* yrows, uint tid, uint sg,
                        threadgroup half* xs, threadgroup half* ws) {
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
  QtmRaw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(X + size_t(xrow[i]) * p.x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) {
      const int q = NP == 2 && wr[i] >= BN / 2 ? 1 : 0;
      wv[i] = qtm_raw(W[q], S[q], B[q], p.K, wn[i], k0 + wk[i]);
    }
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) *(threadgroup uint4*)(xs + b * BM * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) qtm_dec(wv[i], ws + b * WBUF + wr[i] * BK + wk[i]);
  };
  load(0);
  store(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint steps = p.K / BK;
  for (uint st = 0; st < steps; ++st) {
    const bool more = st + 1 < steps;
    if (more) load((st + 1) * BK);
    if (st & 1) op.run(tX1, tW1, acc);
    else op.run(tX0, tW0, acc);
    if (more) store((st + 1) & 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (NP == 2) {
    // up products to threadgroup memory (ws is free after the last barrier), then the gate column combines them
    threadgroup float* ex = (threadgroup float*)ws + (sm * 32 + (tid & 31)) * acc.get_capacity();
    if (sn == 1)
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) ex[i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sn == 0) {
      const uint nh = n0;  // already tg.x * (BN / 2)
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto ix = acc.get_multidimensional_index(i);
        const uint n = nh + ix[0], r = sm * SM + ix[1];
        if (r >= rows || n >= p.N) continue;
        ((device half*)Y)[size_t(m0 + r) * p.y_ld + n] = half(silu(acc[i]) * ex[i]);
      }
    }
    return;
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    const uint n = n0 + sn * SN + ix[0], r = sm * SM + ix[1];
    if (r >= rows || n >= p.N) continue;
    ((device float*)Y)[size_t(yrows[m0 + r]) * p.y_ld + n] = acc[i];
  }
}

// Fused expert gate/up (+ SwiGLU) on moe_bucket2 tiles: buffers as qrm_gu_* (the shared expert rides along as
// expert index p.shared_e with its own weights). grid (N / (BN / 2), max tiles), 128 threads.
#define NI_QTM_GU(NAME, BM, BN, BK)                                                                                \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uint* Wg [[buffer(1)]], const device bfloat* Sg [[buffer(2)]], \
      const device bfloat* Bg [[buffer(3)]], const device uint* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]], \
      const device bfloat* Bu [[buffer(6)]], device half* H [[buffer(7)]], constant QtmParams& p [[buffer(8)]],       \
      const device uint* tiles [[buffer(9)]], const device uint* ntiles [[buffer(10)]],                            \
      const device uint* xrows [[buffer(11)]], const device uint* SWg [[buffer(12)]],                              \
      const device bfloat* SSg [[buffer(13)]], const device bfloat* SBg [[buffer(14)]],                            \
      const device uint* SWu [[buffer(15)]], const device bfloat* SSu [[buffer(16)]],                              \
      const device bfloat* SBu [[buffer(17)]], uint2 tg [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                      \
    if (tg.y >= ntiles[0]) return;                                                                                 \
    threadgroup half xs[2 * BM * BK];                                                                              \
    threadgroup half ws[2 * BN * BK];                                                                              \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;                 \
    const bool sh = e == p.shared_e;                                                                               \
    const uint wo = sh ? 0 : e * p.w_es, so = sh ? 0 : e * p.s_es;                                                 \
    const device uint* W[2] = {(sh ? SWg : Wg) + wo, (sh ? SWu : Wu) + wo};                                        \
    const device bfloat* S[2] = {(sh ? SSg : Sg) + so, (sh ? SSu : Su) + so};                                      \
    const device bfloat* B[2] = {(sh ? SBg : Bg) + so, (sh ? SBu : Bu) + so};                                      \
    qtmm_tile<2, BM, BN, BK>(X, W, S, B, H, p, m0, rows, tg.x * (BN / 2), xrows, nullptr, tid, sg, xs, ws);         \
  }

// Grouped down GEMM to fp32 rows by pair: buffers as qrm_dn_*. grid (N / BN, max tiles), 128 threads.
#define NI_QTM_DN(NAME, BM, BN, BK)                                                                                \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uint* Wd [[buffer(1)]], const device bfloat* Sd [[buffer(2)]], \
      const device bfloat* Bd [[buffer(3)]], device float* Y [[buffer(4)]], constant QtmParams& p [[buffer(5)]],     \
      const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],                              \
      const device uint* yrows [[buffer(8)]], const device uint* SWd [[buffer(9)]],                                 \
      const device bfloat* SSd [[buffer(10)]], const device bfloat* SBd [[buffer(11)]],                             \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                          \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                 \
    if (tg.y >= ntiles[0]) return;                                                                                 \
    threadgroup half xs[2 * BM * BK];                                                                              \
    threadgroup half ws[2 * BN * BK];                                                                              \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;                 \
    const bool sh = e == p.shared_e;                                                                               \
    const uint wo = sh ? 0 : e * p.w_es, so = sh ? 0 : e * p.s_es;                                                 \
    const device uint* W[1] = {(sh ? SWd : Wd) + wo};                                                              \
    const device bfloat* S[1] = {(sh ? SSd : Sd) + so};                                                            \
    const device bfloat* B[1] = {(sh ? SBd : Bd) + so};                                                            \
    qtmm_tile<1, BM, BN, BK>(X, W, S, B, Y, p, m0, rows, tg.x * BN, nullptr, yrows, tid, sg, xs, ws);               \
  }

// BN 64 / BK 32 is the measured best; the BN 128 and BK 64 variants of the same tile ran slower on the M5 Pro
// (dn BN 128 566 -> 221 ms for the 4K layer, BK 64 422 -> 492 ms at 1024 rows), so they are not built.
NI_QTM_GU(qmtm_gu_m64_n64, 64, 64, 32)
NI_QTM_DN(qmtm_dn_m64_n64, 64, 64, 32)
