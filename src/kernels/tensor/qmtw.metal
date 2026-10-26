#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qmtw: qmt.metal's 64x64x32 tile on the other dense weight formats, MLX affine 8-bit group 64 (qmt8_*) and
// bf16 (qmtf16_*), and in a swizzled tile order (qmt4s8_*, qmt8ds8_*: below) ----
// Y[M, N] (+)= X[M, K] * W[N, K]^T, X half, fp32 Y: qmt_m64_n64's interface (QmmParams, buffers 0-5; a bf16 matrix
// binds its weights to buffers 1-3 alike), grid and threads. Each format's weight decode is a bitwise mirror of the
// simdgroup kernel it replaces, so the half operands in threadgroup memory are that kernel's and only each output's
// K reduction is reassociated (one matmul2d over a 32-K step instead of four 8-K simdgroup MMAs):
//  * 8-bit (qmm3.metal's qr_pair8, qmr8_*): the code q enters as the half 0x3400 | q << 2 (= 0.25 + q/1024, exact
//    for q < 256), the scale is pre-multiplied by 1024 and kept as a half next to the half bias, and the value is
//    one half fma, w = fma(qh - 0.25h, s1024, b);
//  * bf16 (wfmt.metal's qrf_load_w for F = 16, qrf_f16_*): half(float(w)), round to nearest.
// The X path, the double-buffered k-loop, the descriptors and the store are qmt_tile's (NP = 1). Requirements
// (host): K % 64 == 0 (8-bit: gs == 64), X rows 16-byte aligned (x_ld % 8 == 0).

struct QmtwParams {
  uint M, N, K, x_ld, y_ld, add, gs, w_es, s_es;
};

// 16 weights of row n at k (k % 16 == 0) as read from memory, and their decode into 16 halfs at dst
struct Qmt8Dec {
  struct Raw {
    uint4 w;      // codes k .. k + 15, byte j of word i is code 4i + j
    half s, b;    // half(float(S[g]) * 1024), half(float(B[g]))
  };
  static Raw load(const device uint* W, const device bfloat* S, const device bfloat* B, uint K, uint n, uint k) {
    Raw r;
    r.w = *(const device uint4*)(W + size_t(n) * (K / 4) + k / 4);
    const size_t g = size_t(n) * (K / 64) + k / 64;
    r.s = half(float(S[g]) * 1024.0f);
    r.b = half(float(B[g]));
    return r;
  }
  static void dec(const Raw r, threadgroup half* dst) {
    half w[16];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 4; ++j) {
        const half qh = as_type<half>(uint16_t(0x3400u | (((r.w[i] >> (8 * j)) & 0xFFu) << 2)));
        w[4 * i + j] = fma(qh - 0.25h, r.s, r.b);
      }
    *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
    *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
    *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
    *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
  }
};

struct QmtF16Dec {
  struct Raw {
    uint4 lo, hi;  // bf16 weights k .. k + 7 and k + 8 .. k + 15, two to a word (the lower half first)
  };
  static Raw load(const device uint* W, const device bfloat*, const device bfloat*, uint K, uint n, uint k) {
    const device uint4* p = (const device uint4*)(W + size_t(n) * (K / 2) + k / 2);
    Raw r;
    r.lo = p[0];
    r.hi = p[1];
    return r;
  }
  static half2 pair(uint w) {
    return half2(half(as_type<float>(w << 16)), half(as_type<float>(w & 0xFFFF0000u)));
  }
  static void dec(const Raw r, threadgroup half* dst) {
    *(threadgroup half4*)(dst) = half4(pair(r.lo.x), pair(r.lo.y));
    *(threadgroup half4*)(dst + 4) = half4(pair(r.lo.z), pair(r.lo.w));
    *(threadgroup half4*)(dst + 8) = half4(pair(r.hi.x), pair(r.hi.y));
    *(threadgroup half4*)(dst + 12) = half4(pair(r.hi.z), pair(r.hi.w));
  }
};


// 4-bit as qmt_m64_n64 decodes it (qmt.metal's qmt_dec: q * s + b in fp32, rounded once to half), for the swizzled
// 4-bit tile below
struct Qmt4Dec {
  struct Raw {
    uint2 w;
    float s, b;
  };
  static Raw load(const device uint* W, const device bfloat* S, const device bfloat* B, uint K, uint n, uint k) {
    Raw r;
    r.w = *(const device uint2*)(W + size_t(n) * (K / 8) + k / 8);
    const size_t g = size_t(n) * (K / 64) + k / 64;
    r.s = float(S[g]);
    r.b = float(B[g]);
    return r;
  }
  static void dec(const Raw r, threadgroup half* dst) {
    half w[16];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 8; ++j) {
      w[j] = half(fma(float((r.w.x >> (4 * j)) & 15u), r.s, r.b));
      w[8 + j] = half(fma(float((r.w.y >> (4 * j)) & 15u), r.s, r.b));
    }
    *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
    *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
    *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
    *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
  }
};

template <class D, int BM, int BN, int BK>
NI_INLINE void qmtw_tile(const device half* X, const device uint* W, const device bfloat* S, const device bfloat* B,
                         device float* Y, constant QmtwParams& p, uint m0, uint n0, uint tid, uint sg,
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
  typename D::Raw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(X + size_t(xrow[i]) * p.x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) wv[i] = D::load(W, S, B, p.K, wn[i], k0 + wk[i]);
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) *(threadgroup uint4*)(xs + b * BM * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) D::dec(wv[i], ws + b * WBUF + wr[i] * BK + wk[i]);
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

#define NI_QMTW(NAME, D, BM, BN, BK)                                                                               \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]], \
      const device bfloat* B [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtwParams& p [[buffer(5)]],   \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                         \
      uint sg [[simdgroup_index_in_threadgroup]]) {                                                                \
    threadgroup half xs[2 * BM * BK];                                                                              \
    threadgroup half ws[2 * BN * BK];                                                                              \
    qmtw_tile<D, BM, BN, BK>(X, W, S, B, Y, p, tg.y * BM, tg.x * BN, tid, sg, xs, ws);                            \
  }


// The device-X tile (qmt*d_*): X is not staged; each simdgroup's matmul2d reads its rows of X as a device tensor and only
// the decoded weights go through threadgroup memory (double-buffered, BN x BK halfs per
// buffer), so BM = 128 fits: 4 simdgroups of (BM / 2) x (BN / 2), the weight tile decoded once for 128 token rows. A
// simdgroup whose first row is at or past M skips its products, but one that starts below M reads all BM / 2 of its rows:
// the host takes this tile only where M is a multiple of BM / 2 (src/model/flint.cpp's enc_gemm), so no read passes M.
template <class D, int BM, int BN, int BK>
NI_INLINE void qmtw_tile_dx(const device half* X, const device uint* W, const device bfloat* S, const device bfloat* B,
                            device float* Y, constant QmtwParams& p, uint m0, uint n0, uint tid, uint sg,
                            threadgroup half* ws) {
  constexpr int SM = BM / 2, SN = BN / 2;
  const uint sm = sg / 2, sn = sg % 2;
  constexpr auto desc =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tg_t;
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
      (device half*)X, dextents<int32_t, 2>(int32_t(p.K), int32_t(p.M)), array<int32_t, 2>{1, int32_t(p.x_ld)});
  const uint mr = m0 + sm * SM;  // this simdgroup's first row
  const bool live = mr < p.M;
  constexpr int WBUF = BN * BK;
  auto tW0 = tg_t(ws + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto tW1 = tg_t(ws + WBUF + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto tA0 = tX.slice<BK, SM>(0, int32_t(live ? mr : 0));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tW0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int WP = BN * BK / 16 / 128;
  uint wr[WP], wk[WP], wn[WP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < WP; ++i) {
    const uint j = tid + 128 * i;
    wr[i] = j / (BK / 16);
    wk[i] = (j % (BK / 16)) * 16;
    wn[i] = min(n0 + wr[i], p.N - 1);
  }
  typename D::Raw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) wv[i] = D::load(W, S, B, p.K, wn[i], k0 + wk[i]);
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) D::dec(wv[i], ws + b * WBUF + wr[i] * BK + wk[i]);
  };
  load(0);
  store(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint steps = p.K / BK;
  for (uint st = 0; st < steps; ++st) {
    const bool more = st + 1 < steps;
    if (more) load((st + 1) * BK);
    if (live) {
      auto tA = tX.slice<BK, SM>(int32_t(st * BK), int32_t(mr));
      if (st & 1) op.run(tA, tW1, acc);
      else op.run(tA, tW0, acc);
    }
    if (more) store((st + 1) & 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!live) return;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    const uint n = n0 + sn * SN + ix[0], m = mr + ix[1];
    if (m >= p.M || n >= p.N) continue;
    const size_t o = size_t(m) * p.y_ld + n;
    const float v = acc[i];
    Y[o] = p.add ? Y[o] + v : v;
  }
}


// Swizzled tile order (qmt*s<SW>_*): the grid's threadgroups in groups of SW row tiles, the N tiles walked inside a
// group with its SW row tiles adjacent, so the threadgroups running together share each weight tile through the cache
// (the weights are read from DRAM about M / (64 SW) times instead of once per row tile) and the group's SW x 64 rows of
// X stay cached. Only which threadgroup computes which tile changes: every output is computed exactly as unswizzled.
NI_INLINE uint2 qmtw_swizzle(uint2 tg, uint2 ng, uint SW) {
  const uint l = tg.y * ng.x + tg.x, per = SW * ng.x, grp = l / per, in = l % per;
  const uint rows = min(SW, ng.y - grp * SW);
  return uint2(in / rows, grp * SW + in % rows);  // (N tile, M tile)
}

#define NI_QMTWS(NAME, D, BM, BN, BK, SW)                                                                          \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]], \
      const device bfloat* B [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtwParams& p [[buffer(5)]],   \
      uint2 tg0 [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]],                              \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                      \
    threadgroup half xs[2 * BM * BK];                                                                              \
    threadgroup half ws[2 * BN * BK];                                                                              \
    const uint2 tg = qmtw_swizzle(tg0, ng, SW);                                                                    \
    qmtw_tile<D, BM, BN, BK>(X, W, S, B, Y, p, tg.y * BM, tg.x * BN, tid, sg, xs, ws);                            \
  }
#define NI_QMTWDS(NAME, D, BM, BN, BK, SW)                                                                         \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]], \
      const device bfloat* B [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtwParams& p [[buffer(5)]],   \
      uint2 tg0 [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]],                              \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                      \
    threadgroup half ws[2 * BN * BK];                                                                              \
    const uint2 tg = qmtw_swizzle(tg0, ng, SW);                                                                    \
    qmtw_tile_dx<D, BM, BN, BK>(X, W, S, B, Y, p, tg.y * BM, tg.x * BN, tid, sg, ws);                             \
  }

// Dispatched (src/model/flint.cpp's enc_gemm, a dense Llama's prefill): qmt4s8 / qmt8ds8 / qmtf16, and with
// NI_FLINT_QMTS=0 qmt_m64_n64 / qmt8 / qmtf16. Tried and dropped (tools/bench_qmm, Nanbeige4.3's shapes at 2048 rows,
// M5 Max, sustained 7-9 rounds): the 4-bit decode as qmr's half2 fma (5-10% slower than the fp32 one), X read in place
// for 4-bit and bf16 (no gain / 5% slower), 64-deep K steps (staged: 32 KB, 30% slower; X in place: 3-6% slower), the
// swizzle groups of 4 and 16 (as 8, within noise), matmul2d's relaxed precision (identical), and both operands in
// registers with 16 x 32 x 16 products as MLX's NAX kernels do (10-20% slower here). As in qmt.metal, an absent shape
// line is an absent kernel, not a slower path.
NI_QMTW(qmt8_m64_n64, Qmt8Dec, 64, 64, 32)
NI_QMTW(qmtf16_m64_n64, QmtF16Dec, 64, 64, 32)
NI_QMTWS(qmt4s8_m64_n64, Qmt4Dec, 64, 64, 32, 8)
NI_QMTWDS(qmt8ds8_m128_n64, Qmt8Dec, 128, 64, 32, 8)
