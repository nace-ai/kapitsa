#include "common.h"

// ---- Short prefill (NI_SHORTPF): quantized GEMMs for T <= 64 token rows ----
// Y[T, N] (+)= X[T, K] * W[N, K]^T with MLX affine 4/8-bit weights (group 64), half X rows.
// The prefill GEMM (qmm_tile) stages dequantized weight tiles in threadgroup memory behind two
// barriers per 32-wide K step and pays 32/64-row tiles for any T; at 4-32 rows it streams the
// weights at ~40-70 GB/s. Here the weights go straight from device memory into registers:
//  * a simdgroup owns 8 weight rows; per 64-group each lane loads 16 contiguous codes of its row
//    (8 bytes at 4 bits) and dequantizes them as half2 (exactly qmm's half(s*q + b) up to the
//    fma's single rounding) into the A operands of 8 simdgroup 8x8 fragments;
//  * the lane's codes cover a permuted set of k (any bijection works for a dot product):
//      lane row fm, slot q = fn/2, fragment f, element e  <->  k = 64g + 16q + 8(f/4) + f%4 + 4e
//    so the B operand (8 k x 8 tokens: lane holds kk = fm, tokens fn, fn+1) is gathered with four
//    8-byte loads per lane per group from the row-major X (k = 64g + 16(fm/2) + 4(fm%2) + {0..3}
//    and +8 of two token rows), which every simdgroup reads from cache;
//  * no threadgroup memory and no barriers on the main path; K may be split over the
//    simdgroups of a threadgroup (ks > 1) with a fixed-order sum in threadgroup memory.
// Simdgroup 8x8 lane layout: lane holds row (lane/4 & 4) + (lane/2)%4, columns
// 2*(lane/4 & 2) + 2*(lane%2) and +1.

struct QmmsParams {
  uint K;          // shared input width (multiple of 64)
  uint T;          // token rows (<= 8 * TB)
  uint x_ld;       // halfs per X row
  uint nseg;       // segments (projections sharing X), <= 4
  uint ks;         // simdgroups splitting K per row block (divides the threadgroup's simdgroups)
  uint N[4];
  uint y_ld[4];
  uint add[4];     // 1: Y += result
  uint tg_begin[5];
};

NI_INLINE uint2 qs_lane_rc(uint lane) {
  uint q = lane / 4;
  return uint2((q & 4) + (lane / 2) % 4, (q & 2) * 2 + (lane % 2) * 2);
}

// This lane's A elements of the 8 fragments of group g of weight row `wrow` (dequantized).
template <int BITS>
NI_INLINE void qs_dequant(const device uint32_t* wrow, half s, half b, uint g, uint q, thread half2* a) {
  const half2 s2 = half2(s), b2 = half2(b);
  if (BITS == 4) {
    const uint2 w = *(const device uint2*)(wrow + g * 8 + 2 * q);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; ++f) {
      const uint word = f < 4 ? w.x : w.y;
      const uint v = ((word >> (4 * (f % 4))) & 0x000F000Fu) | 0x64006400u;  // 1024 + code, exact
      a[f] = fma(as_type<half2>(v) - half2(1024.0h), s2, b2);
    }
  } else {
    const uint4 w = *(const device uint4*)(wrow + g * 16 + 4 * q);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; ++f) {
      const uint wa = w[2 * (f / 4)], wb = w[2 * (f / 4) + 1], sh = 8 * (f % 4);
      const uint v = ((wa >> sh) & 0xFFu) | (((wb >> sh) & 0xFFu) << 16) | 0x64006400u;
      a[f] = fma(as_type<half2>(v) - half2(1024.0h), s2, b2);
    }
  }
}

// This lane's B elements of the 8 fragments of group g for token rows r0 (element 0) and r1
// (element 1); null rows read as zero.
NI_INLINE void qs_gather(const device half* x0, const device half* x1, uint g, uint fm, thread half2* b) {
  const uint o = g * 64 + 16 * (fm / 2) + 4 * (fm % 2);
  const half4 z = half4(0.0h);
  const half4 c00 = x0 ? *(const device half4*)(x0 + o) : z, c01 = x0 ? *(const device half4*)(x0 + o + 8) : z;
  const half4 c10 = x1 ? *(const device half4*)(x1 + o) : z, c11 = x1 ? *(const device half4*)(x1 + o + 8) : z;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 4; ++f) {
    b[f] = half2(c00[f], c10[f]);
    b[f + 4] = half2(c01[f], c11[f]);
  }
}

NI_INLINE void qs_mma8(thread simdgroup_float8x8& acc, thread const half2* a, thread const half2* b) {
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) {
    simdgroup_half8x8 A, B;
    A.thread_elements()[0] = a[f].x;
    A.thread_elements()[1] = a[f].y;
    B.thread_elements()[0] = b[f].x;
    B.thread_elements()[1] = b[f].y;
    simdgroup_multiply_accumulate(acc, A, B, acc);
  }
}

// One simdgroup: NM weight matrices (same rows n0..n0+7, e.g. gate and up) x TB token blocks over
// groups [g0, g1). acc[m][tb] gets the partial products.
template <int BITS, int NM, int TB>
NI_INLINE void qs_run(const device uint32_t* const* W, const device bfloat* const* S, const device bfloat* const* B,
                      uint row, uint wpr, uint gpr, const device half* X, uint x_ld, uint T, uint g0, uint g1,
                      uint lane, thread simdgroup_float8x8 (&acc)[NM][TB]) {
  const uint2 rc = qs_lane_rc(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const device half* xr[TB][2];
  _Pragma("clang loop unroll(full)")
  for (int tb = 0; tb < TB; ++tb) {
    const uint t0 = tb * 8 + fn;
    xr[tb][0] = t0 < T ? X + t0 * x_ld : nullptr;
    xr[tb][1] = t0 + 1 < T ? X + (t0 + 1) * x_ld : nullptr;
  }
  _Pragma("clang loop unroll(full)")
  for (int m = 0; m < NM; ++m)
    _Pragma("clang loop unroll(full)")
    for (int tb = 0; tb < TB; ++tb) acc[m][tb] = simdgroup_float8x8(0);
  for (uint g = g0; g < g1; ++g) {
    half2 a[NM][8];
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < NM; ++m) {
      const half s = half(float(S[m][row * gpr + g])), bi = half(float(B[m][row * gpr + g]));
      qs_dequant<BITS>(W[m] + row * wpr, s, bi, g, q, a[m]);
    }
    _Pragma("clang loop unroll(full)")
    for (int tb = 0; tb < TB; ++tb) {
      half2 b[8];
      qs_gather(xr[tb][0], xr[tb][1], g, fm, b);
      _Pragma("clang loop unroll(full)")
      for (int m = 0; m < NM; ++m) qs_mma8(acc[m][tb], a[m], b);
    }
  }
}

// Sums the ks partial accumulators of a row block in order (ks > 1); every simdgroup of the
// threadgroup must call it. Afterwards acc holds the total in the simdgroup with kslice 0.
template <int NM, int TB>
NI_INLINE void qs_reduce(thread simdgroup_float8x8 (&acc)[NM][TB], threadgroup float2* red, uint ks, uint sg,
                         uint lane) {
  const uint kslice = sg % ks;
  if (kslice)
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < NM; ++m)
      _Pragma("clang loop unroll(full)")
      for (int tb = 0; tb < TB; ++tb)
        red[((sg * NM + m) * TB + tb) * 32 + lane] =
            float2(acc[m][tb].thread_elements()[0], acc[m][tb].thread_elements()[1]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (kslice == 0)
    for (uint j = 1; j < ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (int m = 0; m < NM; ++m)
        _Pragma("clang loop unroll(full)")
        for (int tb = 0; tb < TB; ++tb) {
          float2 v = red[(((sg + j) * NM + m) * TB + tb) * 32 + lane];
          acc[m][tb].thread_elements()[0] += v.x;
          acc[m][tb].thread_elements()[1] += v.y;
        }
}

// Dense, up to 4 segments sharing X (e.g. GDN qkv/z/b/a): grid = tg_begin[nseg] threadgroups of
// nsg simdgroups; a threadgroup covers nsg/ks row blocks of 8 rows of one segment.
// tgy (grid.y): token-block group; each covers TB blocks of 8 rows (rows tgy*8*TB ..). Splitting the
// rows over grid.y gives small-N projections (the 35B router: 17 threadgroups) enough threadgroups;
// per output the arithmetic (K split, order) is unchanged.
template <int BITS, int TB>
NI_INLINE void qmms_impl(const device uint32_t* const* Wv, const device bfloat* const* Sv,
                         const device bfloat* const* Bv, device float* const* Yv, const device half* X,
                         constant QmmsParams& p, threadgroup float2* red, uint tg, uint sg, uint nsg, uint lane,
                         uint tgy = 0) {
  const uint r0 = tgy * 8 * TB;
  if (r0 >= p.T) return;
  const uint T = min(p.T - r0, uint(8 * TB));
  X += r0 * p.x_ld;
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint rbs = nsg / p.ks;
  const uint rb = (tg - p.tg_begin[si]) * rbs + sg / p.ks, kslice = sg % p.ks;
  const uint N = p.N[si], gpr = p.K / 64, wpr = p.K * BITS / 32;
  const uint2 rc = qs_lane_rc(lane);
  const uint n = rb * 8 + rc.x;
  const uint row = min(n, N - 1);
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* W[1] = {Wv[si]};
  const device bfloat* S[1] = {Sv[si]};
  const device bfloat* B[1] = {Bv[si]};
  simdgroup_float8x8 acc[1][TB];
  qs_run<BITS, 1, TB>(W, S, B, row, wpr, gpr, X, p.x_ld, T, g0, g1, lane, acc);
  if (p.ks > 1) {
    qs_reduce<1, TB>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  if (n >= N) return;
  const uint y_ld = p.y_ld[si];
  device float* Y = Yv[si] + r0 * y_ld;
  const bool add = p.add[si];
  _Pragma("clang loop unroll(full)")
  for (int tb = 0; tb < TB; ++tb)
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint t = tb * 8 + rc.y + e;
      if (t < T) {
        device float* d = Y + t * y_ld + n;
        const float v = acc[0][tb].thread_elements()[e];
        *d = add ? *d + v : v;
      }
    }
}

#define NI_QMMS(NAME, BITS, TB)                                                                              \
  [[kernel]] void NAME(const device uint32_t* W0 [[buffer(0)]], const device bfloat* S0 [[buffer(1)]],         \
                       const device bfloat* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                 \
                       const device uint32_t* W1 [[buffer(4)]], const device bfloat* S1 [[buffer(5)]],         \
                       const device bfloat* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                 \
                       const device uint32_t* W2 [[buffer(8)]], const device bfloat* S2 [[buffer(9)]],         \
                       const device bfloat* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],               \
                       const device uint32_t* W3 [[buffer(12)]], const device bfloat* S3 [[buffer(13)]],       \
                       const device bfloat* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],               \
                       const device half* X [[buffer(16)]], constant QmmsParams& p [[buffer(17)]],             \
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {     \
    threadgroup float2 red[8 * TB * 32];                                                                     \
    const device uint32_t* Wv[4] = {W0, W1, W2, W3};                                                          \
    const device bfloat* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device bfloat* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmms_impl<BITS, TB>(Wv, Sv, Bv, Yv, X, p, red, tg.x, sg, nsg, lane, tg.y);                               \
  }

NI_QMMS(qmms_b4_t1, 4, 1)
NI_QMMS(qmms_b4_t2, 4, 2)
NI_QMMS(qmms_b4_t3, 4, 3)
NI_QMMS(qmms_b4_t4, 4, 4)
NI_QMMS(qmms_b4_t5, 4, 5)
NI_QMMS(qmms_b4_t6, 4, 6)
NI_QMMS(qmms_b4_t7, 4, 7)
NI_QMMS(qmms_b4_t8, 4, 8)
NI_QMMS(qmms_b8_t1, 8, 1)
NI_QMMS(qmms_b8_t2, 8, 2)
NI_QMMS(qmms_b8_t3, 8, 3)
NI_QMMS(qmms_b8_t4, 8, 4)
NI_QMMS(qmms_b8_t5, 8, 5)
NI_QMMS(qmms_b8_t6, 8, 6)
NI_QMMS(qmms_b8_t7, 8, 7)
NI_QMMS(qmms_b8_t8, 8, 8)

// Dense gate/up + SwiGLU: H[t, n] = half(silu(x W_gate^T) * (x W_up^T)), half rows of N (= y_ld[0]).
// Segment 0 = gate, segment 1 = up (same N, K, bits); grid = ceil(N/8 / (nsg/ks)) threadgroups.
template <int BITS, int TB>
NI_INLINE void qmms_gu_impl(const device uint32_t* Wg, const device bfloat* Sg, const device bfloat* Bg,
                            const device uint32_t* Wu, const device bfloat* Su, const device bfloat* Bu,
                            device half* H, const device half* X, constant QmmsParams& p, threadgroup float2* red,
                            uint tg, uint sg, uint nsg, uint lane, uint tgy = 0) {
  const uint r0 = tgy * 8 * TB;
  if (r0 >= p.T) return;
  const uint T = min(p.T - r0, uint(8 * TB));
  X += r0 * p.x_ld;
  H += r0 * p.y_ld[0];
  const uint rbs = nsg / p.ks;
  const uint rb = tg * rbs + sg / p.ks, kslice = sg % p.ks;
  const uint N = p.N[0], gpr = p.K / 64, wpr = p.K * BITS / 32;
  const uint2 rc = qs_lane_rc(lane);
  const uint n = rb * 8 + rc.x;
  const uint row = min(n, N - 1);
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* W[2] = {Wg, Wu};
  const device bfloat* S[2] = {Sg, Su};
  const device bfloat* B[2] = {Bg, Bu};
  simdgroup_float8x8 acc[2][TB];
  qs_run<BITS, 2, TB>(W, S, B, row, wpr, gpr, X, p.x_ld, T, g0, g1, lane, acc);
  if (p.ks > 1) {
    qs_reduce<2, TB>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  if (n >= N) return;
  const uint ld = p.y_ld[0];
  _Pragma("clang loop unroll(full)")
  for (int tb = 0; tb < TB; ++tb)
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint t = tb * 8 + rc.y + e;
      if (t < T) {
        const float gv = acc[0][tb].thread_elements()[e], uv = acc[1][tb].thread_elements()[e];
        H[t * ld + n] = half(silu(gv) * uv);
      }
    }
}

#define NI_QMMS_GU(NAME, BITS, TB)                                                                           \
  [[kernel]] void NAME(const device uint32_t* Wg [[buffer(0)]], const device bfloat* Sg [[buffer(1)]],         \
                       const device bfloat* Bg [[buffer(2)]], device half* H [[buffer(3)]],                   \
                       const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],         \
                       const device bfloat* Bu [[buffer(6)]], const device half* X [[buffer(16)]],             \
                       constant QmmsParams& p [[buffer(17)]], uint2 tg [[threadgroup_position_in_grid]],       \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],    \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup float2 red[8 * 2 * TB * 32];                                                                 \
    qmms_gu_impl<BITS, TB>(Wg, Sg, Bg, Wu, Su, Bu, H, X, p, red, tg.x, sg, nsg, lane, tg.y);                 \
  }

NI_QMMS_GU(qmms_gu_b4_t1, 4, 1)
NI_QMMS_GU(qmms_gu_b4_t2, 4, 2)
NI_QMMS_GU(qmms_gu_b4_t3, 4, 3)
NI_QMMS_GU(qmms_gu_b4_t4, 4, 4)
NI_QMMS_GU(qmms_gu_b4_t5, 4, 5)
NI_QMMS_GU(qmms_gu_b4_t6, 4, 6)
NI_QMMS_GU(qmms_gu_b4_t7, 4, 7)
NI_QMMS_GU(qmms_gu_b4_t8, 4, 8)
NI_QMMS_GU(qmms_gu_b8_t1, 8, 1)
NI_QMMS_GU(qmms_gu_b8_t2, 8, 2)
NI_QMMS_GU(qmms_gu_b8_t3, 8, 3)
NI_QMMS_GU(qmms_gu_b8_t4, 8, 4)
NI_QMMS_GU(qmms_gu_b8_t5, 8, 5)
NI_QMMS_GU(qmms_gu_b8_t6, 8, 6)
NI_QMMS_GU(qmms_gu_b8_t7, 8, 7)
NI_QMMS_GU(qmms_gu_b8_t8, 8, 8)

// ---- MoE experts at small T: tiles of <= 8 sorted pair rows (moe_bucket2 with BM = 8) ----
// Same tile lists and row maps as qmm_moe_gu/_dn (Qmm2Params): the gate/up kernel gathers X rows
// through xrows and writes half h in sorted-row order; the down kernel reads those rows and writes
// fp32 rows by pair (yrows). A simdgroup owns 8 weight rows of one tile; grid: (N/8/nsg, tiles).
NI_INLINE void qs_tile(constant Qmm2Params& p, const device uint* tiles, uint t, thread uint& e, thread uint& m0,
                       thread uint& rows) {
  e = tiles[t * 3 + 0];
  m0 = tiles[t * 3 + 1];
  rows = tiles[t * 3 + 2] - m0;
}

[[kernel]] void qmms_moe_gu_b4(const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],
                               const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],
                               const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],
                               const device bfloat* Bu [[buffer(6)]], device half* H [[buffer(7)]],
                               constant Qmm2Params& p [[buffer(8)]], const device uint* tiles [[buffer(9)]],
                               const device uint* ntiles [[buffer(10)]], const device uint* xrows [[buffer(11)]],
                               const device uint32_t* SWg [[buffer(12)]], const device bfloat* SSg [[buffer(13)]],
                               const device bfloat* SBg [[buffer(14)]], const device uint32_t* SWu [[buffer(15)]],
                               const device bfloat* SSu [[buffer(16)]], const device bfloat* SBu [[buffer(17)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const bool sh = e == p.shared_e;
  const uint wo = sh ? 0 : e * p.w_expert_stride, so = sh ? 0 : e * p.s_expert_stride;
  const device uint32_t* W[2] = {(sh ? SWg : Wg) + wo, (sh ? SWu : Wu) + wo};
  const device bfloat* S[2] = {(sh ? SSg : Sg) + so, (sh ? SSu : Su) + so};
  const device bfloat* B[2] = {(sh ? SBg : Bg) + so, (sh ? SBu : Bu) + so};
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;  // N % 8 == 0 (checked on the host)
  const uint gpr = p.K / 64, wpr = p.K / 8;
  // the tile's rows gathered: B rows r = fn, fn+1 of the tile (null past its end)
  const device half* x0 = rc.y < rows ? X + xrows[m0 + rc.y] * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + xrows[m0 + rc.y + 1] * p.x_ld : nullptr;
  simdgroup_float8x8 ag = simdgroup_float8x8(0), au = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a0[8], a1[8], b[8];
    qs_dequant<4>(W[0] + n * wpr, half(float(S[0][n * gpr + g])), half(float(B[0][n * gpr + g])), g, q, a0);
    qs_dequant<4>(W[1] + n * wpr, half(float(S[1][n * gpr + g])), half(float(B[1][n * gpr + g])), g, q, a1);
    qs_gather(x0, x1, g, fm, b);
    qs_mma8(ag, a0, b);
    qs_mma8(au, a1, b);
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 2; ++j) {
    const uint r = rc.y + j;
    if (r < rows) H[(m0 + r) * p.y_ld + n] = half(silu(ag.thread_elements()[j]) * au.thread_elements()[j]);
  }
}

[[kernel]] void qmms_moe_dn_b4(const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],
                               const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],
                               device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]],
                               const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],
                               const device uint* yrows [[buffer(8)]], const device uint32_t* SWd [[buffer(9)]],
                               const device bfloat* SSd [[buffer(10)]], const device bfloat* SBd [[buffer(11)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const bool sh = e == p.shared_e;
  const device uint32_t* W = sh ? SWd : Wd + e * p.w_expert_stride;
  const device bfloat* S = sh ? SSd : Sd + e * p.s_expert_stride;
  const device bfloat* B = sh ? SBd : Bd + e * p.s_expert_stride;
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;
  const uint gpr = p.K / 64, wpr = p.K / 8;
  const device half* x0 = rc.y < rows ? X + (m0 + rc.y) * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + (m0 + rc.y + 1) * p.x_ld : nullptr;
  simdgroup_float8x8 acc = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a[8], b[8];
    qs_dequant<4>(W + n * wpr, half(float(S[n * gpr + g])), half(float(B[n * gpr + g])), g, q, a);
    qs_gather(x0, x1, g, fm, b);
    qs_mma8(acc, a, b);
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 2; ++j) {
    const uint r = rc.y + j;
    if (r < rows) Y[yrows[m0 + r] * p.y_ld + n] = acc.thread_elements()[j];
  }
}

// ---- MoE experts at small T, hybrid (qmms_moe2_*, NI_SHORTPF_MOE_FMA): experts of <= 2 rows by FMAs ----
// At 37-65-row suffixes the 35B routes 1-2 rows to most experts; qmms_moe_* pads every tile to an
// 8-row MMA, i.e. 8 multiply-adds per weight on the same FP ALUs (the M4 has no matrix units) plus
// ~2.4 instructions of dequantization. Experts of at most 2 rows (their first and only tile) take a
// matvec form instead, with qmms's lane layout (a simdgroup's lane = weight row fm of its 8 and the
// 16 codes q of each 64-group): the weights are dequantized exactly as qs_dequant does (the same half
// values as the MMA path) and multiplied with the tile's rows in fp32, the 4 lanes of a row summed by
// two xor shuffles; so the two paths differ only in the order of the fp32 sums (an earlier form that
// skipped the half rounding of the weights moved the cached-turn path further from the large-prefill
// one). Larger experts keep the 8-row MMA path. Same grid, tiles and buffers as qmms_moe_*; group
// size 32 or 64.
template <int R>
NI_INLINE void qv_load_x(thread const device half* const (&xr)[R], uint g, uint q, thread float (&xs)[R][16]) {
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) {
    const device half4* p = (const device half4*)(xr[t] + g * 64 + 16 * q);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i) {
      const half4 v = p[i];
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 4; ++e) xs[t][i * 4 + e] = float(v[e]);
    }
  }
}

// acc[t] += sum over the lane's 16 codes of w * x[t]: w the half dequantized pairs of qs_dequant (codes j
// and j + 4 of each word in one half2), products and sums in fp32
template <int R>
NI_INLINE void qv_dot(const device uint32_t* wrow, half s, half b, uint g, uint q, thread const float (&xs)[R][16],
                      thread float (&acc)[R]) {
  const uint2 w = *(const device uint2*)(wrow + g * 8 + 2 * q);
  const half2 s2 = half2(s), b2 = half2(b);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) {
    const uint word = f < 4 ? w.x : w.y;
    const uint v = ((word >> (4 * (f % 4))) & 0x000F000Fu) | 0x64006400u;  // 1024 + code, exact
    const half2 wv = fma(as_type<half2>(v) - half2(1024.0h), s2, b2);       // codes 8(f/4) + f%4 and + 4
    const float w0 = float(wv.x), w1 = float(wv.y);
    const int c = 8 * (f / 4) + f % 4;
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < R; ++t) acc[t] = fma(w1, xs[t][c + 4], fma(w0, xs[t][c], acc[t]));
  }
}

// gsh: log2 of the group size (5 or 6): the lane's 16 codes of 64-block g lie in scale group
// (64g + 16q) >> gsh of its row (spr groups per row)
template <int R>
NI_INLINE void qv_moe_gu(const device half* X, const device uint* xrows, uint x_ld, uint m0, uint rows,
                         const device uint32_t* const thread* W, const device bfloat* const thread* S,
                         const device bfloat* const thread* B, device half* H, uint y_ld, uint n, uint K,
                         uint gsh, uint lane) {
  const uint2 rc = qs_lane_rc(lane);
  const uint q = rc.y / 2, gpr = K / 64, wpr = K / 8, spr = K >> gsh;
  const device half* xr[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) xr[t] = X + xrows[m0 + min(uint(t), rows - 1)] * x_ld;
  float ag[R], au[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) ag[t] = au[t] = 0.0f;
  for (uint g = 0; g < gpr; ++g) {
    float xs[R][16];
    qv_load_x<R>(xr, g, q, xs);
    const uint si = n * spr + ((64 * g + 16 * q) >> gsh);
    qv_dot<R>(W[0] + n * wpr, half(float(S[0][si])), half(float(B[0][si])), g, q, xs, ag);
    qv_dot<R>(W[1] + n * wpr, half(float(S[1][si])), half(float(B[1][si])), g, q, xs, au);
  }
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) {
    ag[t] += simd_shuffle_xor(ag[t], 1);
    ag[t] += simd_shuffle_xor(ag[t], 8);
    au[t] += simd_shuffle_xor(au[t], 1);
    au[t] += simd_shuffle_xor(au[t], 8);
  }
  if ((lane & 9) == 0)
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < R; ++t)
      if (uint(t) < rows) H[(m0 + t) * y_ld + n] = half(silu(ag[t]) * au[t]);
}

template <int R>
NI_INLINE void qv_moe_dn(const device half* X, uint x_ld, uint m0, uint rows, const device uint32_t* W,
                         const device bfloat* S, const device bfloat* B, device float* Y, const device uint* yrows,
                         uint y_ld, uint n, uint K, uint gsh, uint lane) {
  const uint2 rc = qs_lane_rc(lane);
  const uint q = rc.y / 2, gpr = K / 64, wpr = K / 8, spr = K >> gsh;
  const device half* xr[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) xr[t] = X + (m0 + min(uint(t), rows - 1)) * x_ld;
  float acc[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) acc[t] = 0.0f;
  for (uint g = 0; g < gpr; ++g) {
    float xs[R][16];
    qv_load_x<R>(xr, g, q, xs);
    const uint si = n * spr + ((64 * g + 16 * q) >> gsh);
    qv_dot<R>(W + n * wpr, half(float(S[si])), half(float(B[si])), g, q, xs, acc);
  }
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) {
    acc[t] += simd_shuffle_xor(acc[t], 1);
    acc[t] += simd_shuffle_xor(acc[t], 8);
  }
  if ((lane & 9) == 0)
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < R; ++t)
      if (uint(t) < rows) Y[yrows[m0 + t] * y_ld + n] = acc[t];
}

[[kernel]] void qmms_moe2_gu_b4(const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],
                                const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],
                                const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],
                                const device bfloat* Bu [[buffer(6)]], device half* H [[buffer(7)]],
                                constant Qmm2Params& p [[buffer(8)]], const device uint* tiles [[buffer(9)]],
                                const device uint* ntiles [[buffer(10)]], const device uint* xrows [[buffer(11)]],
                                const device uint32_t* SWg [[buffer(12)]], const device bfloat* SSg [[buffer(13)]],
                                const device bfloat* SBg [[buffer(14)]], const device uint32_t* SWu [[buffer(15)]],
                                const device bfloat* SSu [[buffer(16)]], const device bfloat* SBu [[buffer(17)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const bool sh = e == p.shared_e;
  const uint wo = sh ? 0 : e * p.w_expert_stride, so = sh ? 0 : e * p.s_expert_stride;
  const device uint32_t* W[2] = {(sh ? SWg : Wg) + wo, (sh ? SWu : Wu) + wo};
  const device bfloat* S[2] = {(sh ? SSg : Sg) + so, (sh ? SSu : Su) + so};
  const device bfloat* B[2] = {(sh ? SBg : Bg) + so, (sh ? SBu : Bu) + so};
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;  // N % 8 == 0 (checked on the host)
  const uint gsh = p.gs == 32 ? 5u : 6u;           // group size 32 (the MTP head's experts) or 64
  // FMA only for experts of at most 2 rows (their first and only tile): which rows of a larger
  // expert land in its last tile depends on moe_bucket's atomic scatter order, so its tiles all
  // take the MMA path and every pair's arithmetic depends on the routing alone (deterministic)
  const bool first = tg.y == 0 || tiles[(tg.y - 1) * 3] != e;
  if (first && rows == 1) {
    qv_moe_gu<1>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, n, p.K, gsh, lane);
    return;
  }
  if (first && rows == 2) {
    qv_moe_gu<2>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, n, p.K, gsh, lane);
    return;
  }
  const uint gpr = p.K / 64, wpr = p.K / 8, spr = p.K >> gsh;
  const device half* x0 = rc.y < rows ? X + xrows[m0 + rc.y] * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + xrows[m0 + rc.y + 1] * p.x_ld : nullptr;
  simdgroup_float8x8 ag = simdgroup_float8x8(0), au = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a0[8], a1[8], b[8];
    const uint si = n * spr + ((64 * g + 16 * q) >> gsh);
    qs_dequant<4>(W[0] + n * wpr, half(float(S[0][si])), half(float(B[0][si])), g, q, a0);
    qs_dequant<4>(W[1] + n * wpr, half(float(S[1][si])), half(float(B[1][si])), g, q, a1);
    qs_gather(x0, x1, g, fm, b);
    qs_mma8(ag, a0, b);
    qs_mma8(au, a1, b);
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 2; ++j) {
    const uint r = rc.y + j;
    if (r < rows) H[(m0 + r) * p.y_ld + n] = half(silu(ag.thread_elements()[j]) * au.thread_elements()[j]);
  }
}

[[kernel]] void qmms_moe2_dn_b4(const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],
                                const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],
                                device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]],
                                const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],
                                const device uint* yrows [[buffer(8)]], const device uint32_t* SWd [[buffer(9)]],
                                const device bfloat* SSd [[buffer(10)]], const device bfloat* SBd [[buffer(11)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const bool sh = e == p.shared_e;
  const device uint32_t* W = sh ? SWd : Wd + e * p.w_expert_stride;
  const device bfloat* S = sh ? SSd : Sd + e * p.s_expert_stride;
  const device bfloat* B = sh ? SBd : Bd + e * p.s_expert_stride;
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;
  const uint gsh = p.gs == 32 ? 5u : 6u;
  const bool first = tg.y == 0 || tiles[(tg.y - 1) * 3] != e;  // see qmms_moe2_gu_b4
  if (first && rows == 1) {
    qv_moe_dn<1>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, n, p.K, gsh, lane);
    return;
  }
  if (first && rows == 2) {
    qv_moe_dn<2>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, n, p.K, gsh, lane);
    return;
  }
  const uint gpr = p.K / 64, wpr = p.K / 8, spr = p.K >> gsh;
  const device half* x0 = rc.y < rows ? X + (m0 + rc.y) * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + (m0 + rc.y + 1) * p.x_ld : nullptr;
  simdgroup_float8x8 acc = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a[8], b[8];
    const uint si = n * spr + ((64 * g + 16 * q) >> gsh);
    qs_dequant<4>(W + n * wpr, half(float(S[si])), half(float(B[si])), g, q, a);
    qs_gather(x0, x1, g, fm, b);
    qs_mma8(acc, a, b);
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 2; ++j) {
    const uint r = rc.y + j;
    if (r < rows) Y[yrows[m0 + r] * p.y_ld + n] = acc.thread_elements()[j];
  }
}
