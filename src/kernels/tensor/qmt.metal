#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qmt: Qwen dense prefill GEMM (MLX affine 4-bit, group 64) on the GPU's matrix units (Metal 4 tensor_ops; M5) ----
// Y[M, N] (+)= X[M, K] * W[N, K]^T, X half, W 4-bit (bf16 scales / biases), fp32 Y: qmr's interface (QmmParams,
// buffers 0-5; the gate/up form qmtg_*: qmr_gu's, buffers 0-8). Tiles of BM token rows x BN weight rows x 32 K per
// step: the dequantized half weights (w = q * scale + bias, in fp32, rounded once to half) and the half inputs go
// to threadgroup memory, 4 simdgroups of (BM/2) x (BN/2) with one matmul2d each per step, double-buffered (a step's
// global loads are issued before the previous step's product). K % 64 == 0, X rows 16-byte aligned.

struct QmtParams {
  uint M, N, K, x_ld, y_ld, add, gs, w_es, s_es;
};

// 16 weights of row n at k (k % 16 == 0): 2 packed words, one scale / bias
struct QmtRaw {
  uint2 w;
  float s, b;
};
// (rows of wld words / sld scale groups: K / 8 and K / 64 but for a view of the leading K columns, qmt_m64_n64_ld)
NI_INLINE QmtRaw qmt_raw(const device uint* W, const device bfloat* S, const device bfloat* B, uint wld, uint sld,
                         uint n, uint k) {
  QmtRaw r;
  r.w = *(const device uint2*)(W + size_t(n) * wld + k / 8);
  const size_t g = size_t(n) * sld + k / 64;
  r.s = float(S[g]);
  r.b = float(B[g]);
  return r;
}
NI_INLINE void qmt_dec(const QmtRaw r, threadgroup half* dst) {
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

// NP = 1: Y; NP = 2: gate / up fused: the tile's first BN / 2 weight rows are gate rows n0h.., the next BN / 2 the up
// rows of the same columns; simdgroup column 0 holds the gate products, column 1 the up products of the same
// (row, column) elements (same op shape, same element index), and H = silu(g) * u as half comes out of column 0.
template <int NP, int BM, int BN, int BK>
NI_INLINE void qmt_tile(const device half* X, const device uint* const thread* W, const device bfloat* const thread* S,
                        const device bfloat* const thread* B, device void* Y, constant QmtParams& p, uint m0, uint n0,
                        uint tid, uint sg, threadgroup half* xs, threadgroup half* ws, uint wld, uint sld) {
  constexpr int SM = BM / 2, SN = BN / 2;
  const uint sm = sg / 2, sn = sg % 2;
  constexpr auto desc =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tg_t;
  auto tX0 = tg_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tX1 = tg_t(xs + BM * BK + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  // weights: NP matrices, each BN rows
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
    const uint m = m0 + xr[i];
    xrow[i] = j < XN && m < p.M ? m : 0xffffffffu;
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
  QmtRaw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(X + size_t(xrow[i]) * p.x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) {
      const int q = NP == 2 && wr[i] >= BN / 2 ? 1 : 0;
      wv[i] = qmt_raw(W[q], S[q], B[q], wld, sld, wn[i], k0 + wk[i]);
    }
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) *(threadgroup uint4*)(xs + b * BM * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) qmt_dec(wv[i], ws + b * WBUF + wr[i] * BK + wk[i]);
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
        const uint n = nh + ix[0], m = m0 + sm * SM + ix[1];
        if (m >= p.M || n >= p.N) continue;
        ((device half*)Y)[size_t(m) * p.y_ld + n] = half(silu(acc[i]) * ex[i]);
      }
    }
    return;
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    const uint n = n0 + sn * SN + ix[0], m = m0 + sm * SM + ix[1];
    if (m >= p.M || n >= p.N) continue;
    const size_t o = size_t(m) * p.y_ld + n;
    const float v = acc[i];
    ((device float*)Y)[o] = p.add ? ((device float*)Y)[o] + v : v;
  }
}

#define NI_QMT(NAME, BM, BN, BK)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uint* Wd [[buffer(1)]], const device bfloat* Sd [[buffer(2)]], \
      const device bfloat* Bd [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtParams& p [[buffer(5)]], \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                         \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                \
    threadgroup half xs[2 * BM * BK];                                                                              \
    threadgroup half ws[2 * BN * BK];                                                                              \
    const device uint* W[1] = {Wd};                                                                                \
    const device bfloat* S[1] = {Sd};                                                                              \
    const device bfloat* B[1] = {Bd};                                                                              \
    qmt_tile<1, BM, BN, BK>(X, W, S, B, Y, p, tg.y * BM, tg.x * BN,                                                \
        tid, sg, xs, ws, p.K / 8, p.K / 64);                                                                       \
  }

#define NI_QMTG(NAME, BM, BN, BK)                                                                                  \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uint* Wg [[buffer(1)]], const device bfloat* Sg [[buffer(2)]], \
      const device bfloat* Bg [[buffer(3)]], device half* H [[buffer(4)]], constant QmtParams& p [[buffer(5)]],   \
      const device uint* Wu [[buffer(6)]], const device bfloat* Su [[buffer(7)]],                                  \
      const device bfloat* Bu [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                      \
    threadgroup half xs[2 * BM * BK];                                                                              \
    threadgroup half ws[2 * BN * BK];                                                                              \
    const device uint* W[2] = {Wg, Wu};                                                                            \
    const device bfloat* S[2] = {Sg, Su};                                                                          \
    const device bfloat* B[2] = {Bg, Bu};                                                                          \
    qmt_tile<2, BM, BN, BK>(X, W, S, B, H, p, tg.y * BM, tg.x * (BN / 2),                                          \
        tid, sg, xs, ws, p.K / 8, p.K / 64);                                                                       \
  }

// Only the 64x64 tile is dispatched (src/model/qwen35.cpp); the wider and M-split shapes the tile experiments tried
// are not instantiated. To sweep one of them again, add its line here: tools/bench_qmm.cpp finds kernels by name in
// this source, so an absent line is an absent kernel ("kernel %s not found"), not a slower path.
NI_QMT(qmt_m64_n64, 64, 64, 32)
NI_QMTG(qmtg_m64_n64, 64, 64, 32)

// qmt_m64_n64 over a view: the leading K columns of rows w_ld words / s_ld scale groups long (QmmLdParams, as
// qmm_b4_m64_n64_ld): the GPU's share of the down projection in the Neural Engine split (NI_ANE).
struct QmtLdParams {
  QmtParams q;
  uint w_ld, s_ld;
};
[[kernel, max_total_threads_per_threadgroup(128)]] void qmt_m64_n64_ld(
    const device half* X [[buffer(0)]], const device uint* Wd [[buffer(1)]], const device bfloat* Sd [[buffer(2)]],
    const device bfloat* Bd [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtLdParams& pl [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half xs[2 * 64 * 32];
  threadgroup half ws[2 * 64 * 32];
  const device uint* W[1] = {Wd};
  const device bfloat* S[1] = {Sd};
  const device bfloat* B[1] = {Bd};
  qmt_tile<1, 64, 64, 32>(X, W, S, B, Y, pl.q, tg.y * 64, tg.x * 64, tid, sg, xs, ws, pl.w_ld, pl.s_ld);
}
