#include "common.h"

// ---- qmr MoE: grouped expert GEMMs with the weights dequantized straight into MMA registers ----
// The qmr design (qmm3.metal) applied to the routed experts' sorted pair rows (moe_bucket2 tiles,
// Qmm2Params as qmm_moe.metal): C = W_tile * X_tile^T with the 4-bit g64 weights as the A operand,
// dequantized from the packed words straight into fragment registers (qr_pair: ~2.5 instructions
// per weight, each weight dequantized once per threadgroup: the 4 simdgroups split N); the tile's
// X rows (gathered through xrows for gate/up, contiguous sorted rows for down) are staged once per
// K step in threadgroup memory in qmr's permuted K order and read as the transposed B operand.
// W never touches threadgroup memory (qmm_moe_gu/_dn dequantize every W tile into threadgroup
// memory behind the barriers).
//  * a tile holds at most BM = 8*FB rows of one expert; the kernel branches once (uniform per
//    threadgroup) on the tile's live 8-row fragments, so partial tiles pay MMAs and X staging only
//    for the rows they hold. The whole K loop is specialized per live count.
//  * gate/up: NP = 2 matrices share the X tile (simdgroup tile 8*FB rows m x (TN gate + TN up) rows
//    n); the epilogue writes half(silu(g) * u) into H in sorted-row order (row stride y_ld).
//  * down: Y[yrows[m]] (fp32 rows by pair id) = H[m] W_e^T.
// Per output element the arithmetic is qmr's: dequantized halfs round once (fma), fp32 MMA
// accumulation over K in the permuted order. Against the qmm_moe path (float dequant then a half
// rounding, natural K order) outputs agree to rounding.
// Requirements (host): 4-bit weights, gs 64, K % 64 == 0, N % BN == 0, x_ld % 8 == 0.

// X staging with per-thread row offsets (uint4 units, ~0u: zero row), resolved once per tile.
template <int BMl, int BK, int NT>
NI_INLINE void qrm_xoff(const device uint* xrows, bool gather, uint x_ld8, uint m0, uint rows, uint tid,
                        thread uint* xo) {
  constexpr int CPR = BK / 8, XL = (BMl * CPR + NT - 1) / NT;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < XL; ++j) {
    const uint i = tid + j * NT, r = i / CPR, q = i % CPR;
    xo[j] = (i < uint(BMl * CPR) && r < rows) ? (gather ? xrows[m0 + r] : m0 + r) * x_ld8 + q : ~0u;
  }
}

template <int BMl, int BK, int NT>
NI_INLINE void qrm_load_x(const device uint4* X4, thread const uint* xo, uint k0, thread uint4* xr) {
  constexpr int CPR = BK / 8, XL = (BMl * CPR + NT - 1) / NT;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < XL; ++j) xr[j] = xo[j] != ~0u ? X4[xo[j] + k0 / 8] : uint4(0);
}

// acc[p*FA + a][b] = C_p[n = n0 + sgn*TN + 8a + rc.x][m = 8b + rc.y + e] over the whole K for the
// FL live row fragments of the tile (rows m0 .. m0 + rows of the sorted order).
template <int FL, int BN, int WN, int BK, int NP>
NI_INLINE void qrm_acc(const device half* X, const device uint* xrows, bool gather, uint x_ld, uint m0, uint rows,
                       const device uint32_t* const thread* W, const device bfloat* const thread* S,
                       const device bfloat* const thread* B, uint N, uint K, uint n0, threadgroup half* xs,
                       uint tid, uint sgn, uint2 rc, thread simdgroup_float8x8 (&acc)[NP * (BN / WN / 8)][FL]) {
  constexpr int NT = WN * 32, TN = BN / WN, FA = TN / 8, BMl = FL * 8;
  constexpr int CPR = BK / 8, XL = (BMl * CPR + NT - 1) / NT, WPL = BK / 32;
  const uint c = rc.y / 2;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < NP * FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FL; ++b) acc[a][b] = simdgroup_float8x8(0);
  const uint wpr = K / 8, gpr = K / 64;
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, N - 1);
  uint xo[XL];
  qrm_xoff<BMl, BK, NT>(xrows, gather, x_ld / 8, m0, rows, tid, xo);
  const device uint4* X4 = (const device uint4*)X;
  uint4 xr[XL];
  uint wr[NP * FA * WPL];
  half2 sb[NP * FA];
  for (uint k0 = 0; k0 < K; k0 += BK) {
    qrm_load_x<BMl, BK, NT>(X4, xo, k0, xr);
    qr_load_w<FA, BK, NP>(W, S, B, wrow, wpr, gpr, k0, c, wr, sb);
    qr_store_x<BMl, BK, NT>(xs, tid, xr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qr_mma<NP * FA, FL, BK, BMl, 0>(acc, xs, 0, wr, sb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// gate/up body for FL live fragments: H[m0 + m, n] = half(silu(g) * u)
template <int FL, int BN, int WN, int BK>
NI_INLINE void qrm_gu_body(const device half* X, const device uint* xrows, uint x_ld, uint m0, uint rows,
                           const device uint32_t* const thread* W, const device bfloat* const thread* S,
                           const device bfloat* const thread* B, device half* H, uint y_ld, uint N, uint K,
                           uint n0, threadgroup half* xs, uint tid, uint sgn, uint2 rc) {
  constexpr int TN = BN / WN, FA = TN / 8;
  simdgroup_float8x8 acc[2 * FA][FL];
  qrm_acc<FL, BN, WN, BK, 2>(X, xrows, true, x_ld, m0, rows, W, S, B, N, K, n0, xs, tid, sgn, rc, acc);
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) {
    const uint n = n0 + sgn * TN + a * 8 + rc.x;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FL; ++b)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint m = b * 8 + rc.y + e;
        const float g = acc[a][b].thread_elements()[e], u = acc[FA + a][b].thread_elements()[e];
        if (m < rows) H[(m0 + m) * y_ld + n] = half(silu(g) * u);
      }
  }
}

// down body for FL live fragments: Y[yrows[m0 + m], n] = acc
template <int FL, int BN, int WN, int BK>
NI_INLINE void qrm_dn_body(const device half* X, uint x_ld, uint m0, uint rows, const device uint32_t* const thread* W,
                           const device bfloat* const thread* S, const device bfloat* const thread* B,
                           device float* Y, const device uint* yrows, uint y_ld, uint N, uint K, uint n0,
                           threadgroup half* xs, uint tid, uint sgn, uint2 rc) {
  constexpr int TN = BN / WN, FA = TN / 8;
  simdgroup_float8x8 acc[FA][FL];
  qrm_acc<FL, BN, WN, BK, 1>(X, nullptr, false, x_ld, m0, rows, W, S, B, N, K, n0, xs, tid, sgn, rc, acc);
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < FL; ++b)
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint m = b * 8 + rc.y + e;
      if (m < rows) {
        device float* yr = Y + yrows[m0 + m] * y_ld + n0 + sgn * TN + rc.x;
        _Pragma("clang loop unroll(full)")
        for (int a = 0; a < FA; ++a) yr[a * 8] = acc[a][b].thread_elements()[e];
      }
    }
}

// Fused expert gate/up (+ SwiGLU) on moe_bucket2 tiles of at most 8*FB rows; buffers as
// qmm_moe_gu_* (the shared expert rides along as expert index shared_e with its own weights).
// grid (N / BN, max tiles), 32*WN threads.
#define NI_QRM_GU(NAME, FB, BN, WN, BK)                                                                    \
  [[kernel, max_total_threads_per_threadgroup(WN * 32)]] void NAME(                                         \
      const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],                         \
      const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],                        \
      const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],                      \
      const device bfloat* Bu [[buffer(6)]], device half* H [[buffer(7)]], constant Qmm2Params& p [[buffer(8)]], \
      const device uint* tiles [[buffer(9)]], const device uint* ntiles [[buffer(10)]],                    \
      const device uint* xrows [[buffer(11)]], const device uint32_t* SWg [[buffer(12)]],                  \
      const device bfloat* SSg [[buffer(13)]], const device bfloat* SBg [[buffer(14)]],                    \
      const device uint32_t* SWu [[buffer(15)]], const device bfloat* SSu [[buffer(16)]],                  \
      const device bfloat* SBu [[buffer(17)]], uint2 tg [[threadgroup_position_in_grid]],                  \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                \
      uint lane [[thread_index_in_simdgroup]]) {                                                           \
    if (tg.y >= ntiles[0]) return;                                                                         \
    threadgroup uint4 xs4[8 * FB * (BK + 8) / 8];                                                          \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;         \
    const bool sh = e == p.shared_e;                                                                       \
    const uint wo = sh ? 0 : e * p.w_expert_stride, so = sh ? 0 : e * p.s_expert_stride;                   \
    const device uint32_t* W[2] = {(sh ? SWg : Wg) + wo, (sh ? SWu : Wu) + wo};                            \
    const device bfloat* S[2] = {(sh ? SSg : Sg) + so, (sh ? SSu : Su) + so};                              \
    const device bfloat* B[2] = {(sh ? SBg : Bg) + so, (sh ? SBu : Bu) + so};                              \
    const uint n0 = tg.x * BN;                                                                             \
    const uint2 rc = qr_rc(lane);                                                                          \
    threadgroup half* xs = (threadgroup half*)xs4;                                                         \
    const uint live = (rows + 7) / 8;                                                                      \
    if (FB >= 4 && live >= 4)                                                                              \
      qrm_gu_body<(FB >= 4 ? 4 : 1), BN, WN, BK>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K,  \
                                                 n0, xs, tid, sg, rc);                                     \
    else if (FB >= 3 && live == 3)                                                                         \
      qrm_gu_body<(FB >= 3 ? 3 : 1), BN, WN, BK>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K,  \
                                                 n0, xs, tid, sg, rc);                                     \
    else if (FB >= 2 && live == 2)                                                                         \
      qrm_gu_body<(FB >= 2 ? 2 : 1), BN, WN, BK>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K,  \
                                                 n0, xs, tid, sg, rc);                                     \
    else                                                                                                   \
      qrm_gu_body<1, BN, WN, BK>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                 rc);                                                                      \
  }

// Grouped down GEMM to fp32 rows by pair: buffers as qmm_moe_dn_*. grid (N / BN, max tiles).
#define NI_QRM_DN(NAME, FB, BN, WN, BK)                                                                    \
  [[kernel, max_total_threads_per_threadgroup(WN * 32)]] void NAME(                                         \
      const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                         \
      const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],                        \
      device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]], const device uint* tiles [[buffer(6)]], \
      const device uint* ntiles [[buffer(7)]], const device uint* yrows [[buffer(8)]],                     \
      const device uint32_t* SWd [[buffer(9)]], const device bfloat* SSd [[buffer(10)]],                   \
      const device bfloat* SBd [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],                  \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                \
      uint lane [[thread_index_in_simdgroup]]) {                                                           \
    if (tg.y >= ntiles[0]) return;                                                                         \
    threadgroup uint4 xs4[8 * FB * (BK + 8) / 8];                                                          \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;         \
    const bool sh = e == p.shared_e;                                                                       \
    const device uint32_t* W[1] = {sh ? SWd : Wd + e * p.w_expert_stride};                                 \
    const device bfloat* S[1] = {sh ? SSd : Sd + e * p.s_expert_stride};                                   \
    const device bfloat* B[1] = {sh ? SBd : Bd + e * p.s_expert_stride};                                   \
    const uint n0 = tg.x * BN;                                                                             \
    const uint2 rc = qr_rc(lane);                                                                          \
    threadgroup half* xs = (threadgroup half*)xs4;                                                         \
    const uint live = (rows + 7) / 8;                                                                      \
    if (FB >= 4 && live >= 4)                                                                              \
      qrm_dn_body<(FB >= 4 ? 4 : 1), BN, WN, BK>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K,  \
                                                 n0, xs, tid, sg, rc);                                     \
    else if (FB >= 3 && live == 3)                                                                         \
      qrm_dn_body<(FB >= 3 ? 3 : 1), BN, WN, BK>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K,  \
                                                 n0, xs, tid, sg, rc);                                     \
    else if (FB >= 2 && live == 2)                                                                         \
      qrm_dn_body<(FB >= 2 ? 2 : 1), BN, WN, BK>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K,  \
                                                 n0, xs, tid, sg, rc);                                     \
    else                                                                                                   \
      qrm_dn_body<1, BN, WN, BK>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K, n0, xs, tid, sg,  \
                                 rc);                                                                      \
  }

// naming: qrm_{gu,dn}_k{BK}_m{8*FB}_n{BN}; 4 simdgroups along N (128 threads)
NI_QRM_GU(qrm_gu_k32_m32_n64, 4, 64, 4, 32)    // 88 GPRs, no stack
NI_QRM_GU(qrm_gu_k32_m16_n64, 2, 64, 4, 32)    // NI_MOE_QMR=16
NI_QRM_DN(qrm_dn_k32_m32_n128, 4, 128, 4, 32)  // 85 GPRs, no stack
NI_QRM_DN(qrm_dn_k32_m16_n128, 2, 128, 4, 32)
