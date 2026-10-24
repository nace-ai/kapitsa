#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qmt6: qmt.metal's 64x64x32 tile on the 6-bit group-64 dense weights (wfmt.metal's f6g64) ----
// Y[M, N] (+)= X[M, K] * W[N, K]^T, X half, W 6-bit (bf16 scales / biases, group 64), fp32 Y: qmt_m64_n64's
// interface (QmmParams, buffers 0-5), grid and threads. The weight decode is a bitwise mirror of qrf_f6g64's
// (wfmt.metal's qrf_load_w + qrf_mma): the code q enters as the half 0x3400|q (= 0.25 + q/4096, exact: q <= 63
// fits the mantissa), the scale is pre-multiplied by 4096 and kept as a half next to the half bias, and the
// value is one half fma, w = fma(qh - 0.25h, s4096, b) — the same inputs and the same single rounding, so the
// dequantized halfs in the tile's threadgroup memory are the bits qrf feeds its MMA fragments. The only change
// from qrf is the product: one matmul2d over the whole BK = 32 step with 4 simdgroups instead of four 8-K
// simdgroup MMAs (fp32 accumulation over K) — the same exchange qmt_m64_n64 made for the 4-bit weights.
// Requirements (host): K % 64 == 0, gs == 64, X rows 16-byte aligned (x_ld % 8 == 0).

struct Qmt6Params {
  uint M, N, K, x_ld, y_ld, add, gs, w_es, s_es;
};

// 16 weights of row n at k (k % 16 == 0): 3 words (96 bits, wf_codes16<6>'s window) plus the group's scale /
// bias as qrf_load_w's pre-scaled half pair (a 16-run never crosses a group of 64)
struct Qmt6Raw {
  uint w[3];
  half s4, bh;  // half(float(S[g]) * 4096), half(float(B[g]))
};
// (rows of wld words / sld scale groups: K * 3 / 16 and K / 64 for the full-K rows)
NI_INLINE Qmt6Raw qmt6_raw(const device uint* W, const device bfloat* S, const device bfloat* B, uint wld, uint sld,
                           uint n, uint k) {
  Qmt6Raw r;
  const device uint* p = W + size_t(n) * wld + (k / 16) * 3;
  r.w[0] = p[0];
  r.w[1] = p[1];
  r.w[2] = p[2];
  const size_t g = size_t(n) * sld + k / 64;
  r.s4 = half(float(S[g]) * 4096.0f);
  r.bh = half(float(B[g]));
  return r;
}
// The code extraction is wf_extract<6, 16, 4>'s (code j = bits [6j, 6j+6) of the run; only j = 5 and j = 10
// cross a word); the value is qrf_mma's lane op: fma(half2(v) - 0.25h, sb.x, sb.y) element for element.
NI_INLINE void qmt6_dec(const Qmt6Raw r, threadgroup half* dst) {
  half w[16];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 16; ++j) {
    const int b = 6 * j, wi = b / 32, sh = b % 32;
    uint v = r.w[wi] >> sh;
    if (sh + 6 > 32) v |= r.w[wi + 1] << (32 - sh);
    const half qh = as_type<half>(uint16_t(0x3400u | (v & 63u)));
    w[j] = fma(qh - 0.25h, r.s4, r.bh);
  }
  *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
  *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
  *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
  *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
}

// qmt.metal's qmt_tile with NP = 1 (no gate/up form: the f6g64 tensors are plain projections) and the weight
// read / decode above. The X path, the double-buffered k-loop, the descriptors and both store paths are qmt's
// unchanged.
template <int BM, int BN, int BK>
NI_INLINE void qmt6_tile(const device half* X, const device uint* W, const device bfloat* S, const device bfloat* B,
                         device float* Y, constant Qmt6Params& p, uint m0, uint n0, uint tid, uint sg,
                         threadgroup half* xs, threadgroup half* ws, uint wld, uint sld) {
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
    wn[i] = min(n0 + wr[i], p.N - 1);
  }
  uint4 xv[XP];
  Qmt6Raw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(X + size_t(xrow[i]) * p.x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) wv[i] = qmt6_raw(W, S, B, wld, sld, wn[i], k0 + wk[i]);
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) *(threadgroup uint4*)(xs + b * BM * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) qmt6_dec(wv[i], ws + b * WBUF + wr[i] * BK + wk[i]);
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
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    const uint n = n0 + sn * SN + ix[0], m = m0 + sm * SM + ix[1];
    if (m >= p.M || n >= p.N) continue;
    const size_t o = size_t(m) * p.y_ld + n;
    const float v = acc[i];
    Y[o] = p.add ? Y[o] + v : v;
  }
}

// Only the 64x64 tile is dispatched (src/model/qwen35.cpp's enc_qmm_large); as in qmt.metal, an absent shape
// line is an absent kernel, not a slower path.
[[kernel, max_total_threads_per_threadgroup(128)]] void qmt6_m64_n64(
    const device half* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]],
    const device bfloat* B [[buffer(3)]], device float* Y [[buffer(4)]], constant Qmt6Params& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half xs[2 * 64 * 32];
  threadgroup half ws[2 * 64 * 32];
  qmt6_tile<64, 64, 32>(X, W, S, B, Y, p, tg.y * 64, tg.x * 64, tid, sg, xs, ws, p.K * 3 / 16, p.K / 64);
}
