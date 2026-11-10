#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qmtf: qmt.metal's 128-row tile (qmt128_tile) on the other MLX affine formats (wfmt.metal's f{F}g{GS}) ----
// Y[M, N] (+)= X[M, K] * W[N, K]^T (qmtf_f{F}g{GS}_m128_n64) or H = silu(X Wg^T) * (X Wu^T) as half (qmtfg_*):
// qmt_m128_n64 / qmtg_m128_n64's interface (QmtParams, buffers 0-5, gate/up 0-8), grid and threads. The weight
// decode is qrf_load_w + qrf_mma's (wfmt.metal) element for element, as qmt6.metal's: the code q enters as the half
// 0x3400|q (= 0.25 + q/4096, exact for q < 1024), the scale is pre-multiplied by 4096 and kept as a half next to the
// half bias, and w = fma(qh - 0.25h, s4096, b), one half rounding. So the dequantized halfs are the bits qrf feeds its
// MMA fragments; only the product differs: two 32 x 32 x 32 matmul2d per step on the matrix units instead of 8-K
// simdgroup MMAs (fp32 accumulation over K, another association order). Requirements (host): K % 64 == 0, gs == GS,
// M >= 32, X rows 16-byte aligned. Weight rows are K * F / 32 words; a 16-run of codes starts at bit 16F*m, word
// aligned for even F and 16-bit aligned for odd F (wf_codes16's window).

struct QmtfParams {
  uint M, N, K, x_ld, y_ld, add, gs, w_es, s_es;
};

// words a 16-run of F-bit codes spans from its (16-bit aligned) start: 16F bits, plus 16 for odd F's half-word shift
#define QMTF_RW(F) ((F) % 2 ? ((F) + 1) / 2 : (F) / 2)
template <int F>
struct QmtfRaw {
  uint w[QMTF_RW(F)];
  half s4, bh;  // half(float(S[g]) * 4096), half(float(B[g])): qrf_load_w's sb pair (8-bit: qmr8's, scale * 1024)
};

template <int F, int GS>
NI_INLINE void qmtf_raw(const device uint* W, const device bfloat* S, const device bfloat* B, uint wld, uint sld,
                        uint n, uint k, bool refresh, thread QmtfRaw<F>& r) {
  const device uint* p = W + size_t(n) * wld + (uint(F) * k) / 32;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < QMTF_RW(F); ++i) r.w[i] = p[i];
  if (refresh) {
    const size_t g = size_t(n) * sld + k / GS;
    r.s4 = half(float(S[g]) * (F == 8 ? 1024.0f : 4096.0f));
    r.bh = half(float(B[g]));
  }
}

// 16 codes from the run (wf_codes16<F>'s shift and wf_extract<F, 16>), each made into qrf_mma's lane value; 8-bit
// codes as qmm3.metal's qmr8 makes them (the half 0x3400|q<<2, the scale times 1024: 8-bit has no qrf path)
template <int F>
NI_INLINE void qmtf_dec(const thread QmtfRaw<F>& r, uint k, threadgroup half* dst) {
  constexpr int RW = QMTF_RW(F);
  constexpr uint M = (1u << F) - 1u;
  uint w[RW + 1];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < RW; ++i) w[i] = r.w[i];
  w[RW] = 0u;
  if (F % 2 && ((uint(F) * k) & 16u)) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < RW; ++i) w[i] = (w[i] >> 16) | (w[i + 1] << 16);
  }
  half v[16];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 16; ++j) {
    const int b = F * j, wi = b / 32, sh = b % 32;
    uint q = w[wi] >> sh;
    if (sh + F > 32) q |= w[wi + 1] << (32 - sh);
    const half qh = as_type<half>(uint16_t(0x3400u | (F == 8 ? (q & M) << 2 : q & M)));
    v[j] = fma(qh - 0.25h, r.s4, r.bh);
  }
  *(threadgroup half4*)(dst) = half4(v[0], v[1], v[2], v[3]);
  *(threadgroup half4*)(dst + 4) = half4(v[4], v[5], v[6], v[7]);
  *(threadgroup half4*)(dst + 8) = half4(v[8], v[9], v[10], v[11]);
  *(threadgroup half4*)(dst + 12) = half4(v[12], v[13], v[14], v[15]);
}

// qmt.metal's qmt128_tile with the weight read / decode above; the X tensor, the k-loop, the matmuls, the barriers and
// the stores are qmt128_tile's unchanged (see its comment for the layout and the last-rows rule).
template <int F, int GS, int NP, typename YT>
NI_INLINE void qmtf128_tile(const device half* X, const device uint* W, const device bfloat* S, const device bfloat* B,
                            device YT* Y, constant QmtfParams& p, uint m0, uint n0, uint tid, uint sg,
                            threadgroup half* ws, uint wld, uint sld, const device uint* Wu, const device bfloat* Su,
                            const device bfloat* Bu) {
  constexpr int BK = 32, WBUF = 64 * BK;
  const uint r0 = m0 + sg * 32;
  const bool mine = r0 < p.M;
  const uint rb = mine && r0 + 32 > p.M ? p.M - 32 : r0;
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
      (device half*)(X + size_t(rb) * p.x_ld), dextents<int32_t, 2>(p.K, 32), array<int32_t, 2>{1, int32_t(p.x_ld)});
  typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tg_t;
  constexpr auto desc =
      matmul2d_descriptor(32, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto tW0a = tg_t(ws, dextents<int32_t, 2>(BK, 32));
  auto tW1a = tg_t(ws + WBUF, dextents<int32_t, 2>(BK, 32));
  auto tW0b = tg_t(ws + 32 * BK, dextents<int32_t, 2>(BK, 32));
  auto tW1b = tg_t(ws + WBUF + 32 * BK, dextents<int32_t, 2>(BK, 32));
  auto x0 = tX.template slice<BK, 32>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(x0), decltype(tW0a), float>();
  auto acc2 = op.template get_destination_cooperative_tensor<decltype(x0), decltype(tW0a), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f, acc2[i] = 0.0f;
  const uint wr = tid / 2, wk = (tid % 2) * 16;
  const uint wn = NP == 2 ? min(n0 + (wr % 32), p.N - 1) : min(n0 + wr, p.N - 1);
  if (NP == 2 && wr >= 32) W = Wu, S = Su, B = Bu;
  QmtfRaw<F> wv;
  qmtf_raw<F, GS>(W, S, B, wld, sld, wn, wk, true, wv);
  qmtf_dec<F>(wv, wk, ws + wr * BK + wk);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint steps = p.K / BK;
  for (uint st = 0; st < steps; ++st) {
    const bool more = st + 1 < steps;
    const uint kn = (st + 1) * BK + wk;
    if (more) qmtf_raw<F, GS>(W, S, B, wld, sld, wn, kn, (((st + 1) * BK) % GS) == 0, wv);  // (a run of 16 never crosses a group)
    if (mine) {
      auto xs = tX.template slice<BK, 32>(int32_t(st * BK), 0);
      if (st & 1) {
        op.run(xs, tW1a, acc);
        op.run(xs, tW1b, acc2);
      } else {
        op.run(xs, tW0a, acc);
        op.run(xs, tW0b, acc2);
      }
    }
    if (more) qmtf_dec<F>(wv, kn, ws + ((st + 1) & 1) * WBUF + wr * BK + wk);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!mine) return;
  auto put = [&](thread decltype(acc)& a, uint nb) {
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < a.get_capacity(); ++i) {
      if (!a.is_valid_element(i)) continue;
      const auto ix = a.get_multidimensional_index(i);
      const uint n = n0 + nb + ix[0], m = rb + ix[1];
      if (m < r0 || m >= p.M || n >= p.N) continue;
      const size_t o = size_t(m) * p.y_ld + n;
      const float v = a[i];
      if (NP == 2) Y[o] = YT(v);
      else Y[o] = p.add ? Y[o] + v : v;
    }
  };
  if (NP == 2) {
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc2[i] = silu(acc[i]) * acc2[i];
    put(acc2, 0);
    return;
  }
  if (rb == r0 && n0 + 64 <= p.N) {
    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        (device float*)Y + size_t(r0) * p.y_ld + n0, dextents<int32_t, 2>(64, 32), array<int32_t, 2>{1, int32_t(p.y_ld)});
    auto y0 = tY.template slice<32, 32>(0, 0);
    auto y1 = tY.template slice<32, 32>(32, 0);
    if (p.add) {
      auto c = op.template get_destination_cooperative_tensor<decltype(x0), decltype(tW0a), float>();
      c.load(y0);
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < c.get_capacity(); ++i) c[i] = c[i] + acc[i];
      c.store(y0);
      c.load(y1);
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < c.get_capacity(); ++i) c[i] = c[i] + acc2[i];
      c.store(y1);
      return;
    }
    acc.store(y0);
    acc2.store(y1);
    return;
  }
  put(acc, 0);
  put(acc2, 32);
}

// qmtf<F>_m128_n64_ld: over a view, the leading K columns of rows w_ld words / s_ld scale groups long (qmt_m128_n64_ld's
// QmtLdParams): the GPU's share of the down projection in the Neural Engine split (NI_ANE)
struct QmtfLdParams {
  QmtfParams q;
  uint w_ld, s_ld;
};

#define NI_QMTF(F, GS)                                                                                                 \
  [[kernel, max_total_threads_per_threadgroup(128)]] void qmtf_f##F##g##GS##_m128_n64(                                      \
      const device half* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]], \
      const device bfloat* B [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtfParams& p [[buffer(5)]],  \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                         \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                \
    threadgroup half ws[2 * 64 * 32];                                                                              \
    qmtf128_tile<F, GS, 1>(X, W, S, B, Y, p, tg.y * 128, tg.x * 64, tid, sg, ws, p.K * F / 32, p.K / GS, W, S, B);     \
  }                                                                                                                \
  [[kernel, max_total_threads_per_threadgroup(128)]] void qmtfg_f##F##g##GS##_m128_n64(                                     \
      const device half* X [[buffer(0)]], const device uint* Wg [[buffer(1)]], const device bfloat* Sg [[buffer(2)]], \
      const device bfloat* Bg [[buffer(3)]], device half* H [[buffer(4)]], constant QmtfParams& p [[buffer(5)]],  \
      const device uint* Wu [[buffer(6)]], const device bfloat* Su [[buffer(7)]], const device bfloat* Bu [[buffer(8)]], \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                         \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                \
    threadgroup half ws[2 * 64 * 32];                                                                              \
    qmtf128_tile<F, GS, 2>(X, Wg, Sg, Bg, H, p, tg.y * 128, tg.x * 32, tid, sg, ws, p.K * F / 32, p.K / GS, Wu, Su, Bu); \
  }                                                                                                                \
  [[kernel, max_total_threads_per_threadgroup(128)]] void qmtf_f##F##g##GS##_m128_n64_ld(                                   \
      const device half* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]], \
      const device bfloat* B [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtfLdParams& pl [[buffer(5)]], \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                         \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                \
    threadgroup half ws[2 * 64 * 32];                                                                              \
    qmtf128_tile<F, GS, 1>(X, W, S, B, Y, pl.q, tg.y * 128, tg.x * 64, tid, sg, ws, pl.w_ld, pl.s_ld, W, S, B);        \
  }

// The formats that dispatch them (src/model/qwen35.cpp's enc_qmm_large and enc_qmm_gu, qmtf_for); an absent line is
// an absent kernel.
NI_QMTF(3, 64)
NI_QMTF(8, 64)
NI_QMTF(4, 32)
