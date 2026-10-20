#include "common.h"

// ---- qmr: dense prefill GEMM with the weights dequantized straight into MMA registers ----
// Y[M, N] (+)= X[M, K] * W[N, K]^T, X half, W MLX affine 4-bit (group 64, bf16 scales/biases), Y fp32.
// Computes the transposed tile C = W_tile * X_tile^T with W as the A operand (rows n, columns k):
// in an 8x8 A fragment a lane holds row (lane/4 & 4) + (lane/2) % 4 and two adjacent columns
// 2c, 2c+1 (c = (lane/4 & 2) + lane % 2). The K order inside each BK block is permuted so that a
// lane's columns over the BK/8 fragments are exactly the 8 (BK=32) or 16 (BK=64) nibbles of its own
// 4/8 bytes of the packed row: fragment f, column 2c+e <-> nibble (f%4) + 4e of word
// (BK/32)*c + f/4. A fragment element pair is then one shift/mask/or of the word (nibbles j and
// j+4 land in the two 16-bit halves) giving half2 0.25 + q/4096, minus 0.25, times 4096*scale plus
// the bias (half2 FMA; see qr_pair). X is staged in threadgroup memory in the same permuted K order
// and read as the transposed B operand. Each W element is dequantized once per threadgroup
// (simdgroups split N) when WM == 1, and W never touches threadgroup memory. No double buffering:
// several threadgroups per core hide the barriers (a two-buffer, one-barrier variant and a
// register-prefetch variant measured no faster; prefetch costs registers and spills).
// Requirements (host): bits 4, gs 64, K % 64 == 0, X rows 16-byte aligned (x_ld % 8 == 0).

// lane -> (row, column pair) of an 8x8 fragment (same layout as q2_lane_rc)
NI_INLINE uint2 qr_rc(uint lane) {
  uint q = lane / 4;
  return uint2((q & 4) + (lane / 2) % 4, (q & 2) * 2 + (lane % 2) * 2);
}

// One A-fragment element pair: nibbles j and j+4 of w (j = 0..3), dequantized as half2;
// sb = (scale * 4096, bias). 0x3400 | q is the half 0.25 + q/4096 (exact), so the pair costs a
// shift, mask, or, subtract and FMA. Both factors of q*s stay normal halfs for scales in
// [1.5e-8, 16] (a plain half scale goes subnormal below 6.1e-5: 2162 scales of the 27B, 14818 of
// the 35B's dense projections); the FMA rounds once.
NI_INLINE half2 qr_pair(uint w, uint j, half2 sb) {
  uint t = ((w >> (4 * j)) & 0x000F000Fu) | 0x34003400u;
  return fma(as_type<half2>(t) - half2(0.25h), half2(sb.x), half2(sb.y));
}

// X staging: chunk i (8 halfs, physical k0+8q..) of tile row r -> permuted threadgroup positions.
template <int BM, int BK, int NT>
NI_INLINE void qr_load_x(const device uint4* X4, uint x_ld8, uint M, uint m0, uint k0, uint tid, thread uint4* xr) {
  constexpr int CPR = BK / 8, XL = (BM * CPR + NT - 1) / NT;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < XL; ++j) {
    uint i = tid + j * NT, r = i / CPR, q = i % CPR;
    xr[j] = (i < uint(BM * CPR) && m0 + r < M) ? X4[(m0 + r) * x_ld8 + k0 / 8 + q] : uint4(0);
  }
}

template <int BM, int BK, int NT>
NI_INLINE void qr_store_x(threadgroup half* xs, uint tid, thread const uint4* xr) {
  constexpr int CPR = BK / 8, XL = (BM * CPR + NT - 1) / NT, LD = BK + 8, WPL = BK / 32;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < XL; ++j) {
    uint i = tid + j * NT, r = i / CPR, q = i % CPR;
    if (i < uint(BM * CPR)) {
      uint cc = q / WPL, h = q % WPL;
      // halfs x[0..7] of the chunk: half2(x[f4], x[f4+4]) goes to fragment 4h+f4, column pair cc
      uint4 v = xr[j];
      threadgroup uint* dst = (threadgroup uint*)(xs + r * LD + 32 * h + 2 * cc);
      dst[0] = (v.x & 0xFFFFu) | (v.z << 16);           // x0, x4
      dst[4] = (v.x >> 16) | (v.z & 0xFFFF0000u);        // x1, x5
      dst[8] = (v.y & 0xFFFFu) | (v.w << 16);           // x2, x6
      dst[12] = (v.y >> 16) | (v.w & 0xFFFF0000u);       // x3, x7
    }
  }
}

// This lane's packed words of its FA rows of each of the NP matrices for the K step at k0, and
// their (scale, bias): the lane's 8*WPL values lie in one 64-group, (k0 + 8*WPL*c) / 64.
// wr[(p*FA + a)*WPL + u], sb[p*FA + a].
template <int FA, int BK, int NP>
NI_INLINE void qr_load_w(const device uint32_t* const thread* W, const device bfloat* const thread* S,
                         const device bfloat* const thread* B, thread const uint* wrow, uint wpr, uint gpr,
                         uint k0, uint c, thread uint* wr, thread half2* sb) {
  constexpr int WPL = BK / 32;
  const uint g = (k0 + 8 * WPL * c) / 64;
  _Pragma("clang loop unroll(full)")
  for (int p = 0; p < NP; ++p)
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      const int i = p * FA + a;
      const device uint* wp = W[p] + wrow[a] * wpr + k0 / 8 + WPL * c;
      if (WPL == 4) {
        uint4 v = *(const device uint4*)wp;
        wr[i * 4 + 0] = v.x; wr[i * 4 + 1] = v.y; wr[i * 4 + 2] = v.z; wr[i * 4 + 3] = v.w;
      } else if (WPL == 2) {
        uint2 v = *(const device uint2*)wp;
        wr[i * 2 + 0] = v.x; wr[i * 2 + 1] = v.y;
      } else {
        wr[i] = wp[0];
      }
      if (BK >= 64 || (k0 & 63) == 0)
        sb[i] = half2(half(float(S[p][wrow[a] * gpr + g]) * 4096.0f), half(float(B[p][wrow[a] * gpr + g])));
    }
}

// acc[a][b] += A_a(f) * X_b(f)^T over the BK/8 fragments of one K step; A_a from this lane's words.
template <int FA, int FB, int BK, int TM, int ABL>
NI_INLINE void qr_mma(thread simdgroup_float8x8 (&acc)[FA][FB], threadgroup const half* xs, uint sgm,
                      thread const uint* wr, thread const half2* sb) {
  constexpr int LD = BK + 8, WPL = BK / 32, NF = BK / 8;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    simdgroup_half8x8 af[FA];
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      half2 v = (ABL & 1) ? as_type<half2>(wr[a * WPL + f / 4]) : qr_pair(wr[a * WPL + f / 4], f % 4, sb[a]);
      af[a].thread_elements()[0] = v.x;
      af[a].thread_elements()[1] = v.y;
    }
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) {
      simdgroup_half8x8 bf;
      simdgroup_load(bf, xs + (sgm * TM + b * 8) * LD + 8 * f, LD, ulong2(0, 0), true);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FA; ++a) simdgroup_multiply_accumulate(acc[a][b], af[a], bf, acc[a][b]);
    }
  }
}

// Accumulates the BM x BN tile (per simdgroup TM = BM/WM rows m, TN = BN/WN rows n of each of the NP
// matrices) into acc[p*FA + a][b] = C_p[n = n0 + sgn*TN + 8a + rc.x][m = m0 + sgm*TM + 8b + rc.y + e].
// One X buffer: load X/W, stage X, barrier, MMAs, barrier. ABL (bench-only ablations): 1 no dequant
// math, 2 no X staging, 4 no barriers, 8 no W loads.
template <int BM, int BN, int WM, int WN, int BK, int NP, int ABL>
NI_INLINE void qr_acc(const device half* X, const device uint32_t* const thread* W,
                      const device bfloat* const thread* S, const device bfloat* const thread* B, uint M, uint N,
                      uint K, uint x_ld, uint m0, uint n0, threadgroup half* xs, uint tid, uint sgm, uint sgn,
                      uint2 rc, thread simdgroup_float8x8 (&acc)[NP * (BN / WN / 8)][BM / WM / 8]) {
  constexpr int NT = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int FB = TM / 8, FA = TN / 8;
  constexpr int CPR = BK / 8, XL = (BM * CPR + NT - 1) / NT;
  constexpr int WPL = BK / 32;
  const uint c = rc.y / 2;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < NP * FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) acc[a][b] = simdgroup_float8x8(0);

  const uint wpr = K / 8, gpr = K / 64, x_ld8 = x_ld / 8;
  // this lane's W rows (clamped: rows past N compute garbage that is never stored)
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, N - 1);
  const device uint4* X4 = (const device uint4*)X;

  uint4 xr[XL];
  uint wr[NP * FA * WPL];
  half2 sb[NP * FA];
  for (uint k0 = 0; k0 < K; k0 += BK) {
    if (!(ABL & 2)) qr_load_x<BM, BK, NT>(X4, x_ld8, M, m0, k0, tid, xr);
    if (!(ABL & 8)) {
      qr_load_w<FA, BK, NP>(W, S, B, wrow, wpr, gpr, k0, c, wr, sb);
    } else {
      for (int u = 0; u < NP * FA * WPL; ++u) wr[u] = k0 + u;
      for (int a = 0; a < NP * FA; ++a) sb[a] = half2(0.01h, 0.1h);
    }
    if (!(ABL & 2)) qr_store_x<BM, BK, NT>(xs, tid, xr);
    if (!(ABL & 4)) threadgroup_barrier(mem_flags::mem_threadgroup);
    qr_mma<NP * FA, FB, BK, TM, ABL>(acc, xs, sgm, wr, sb);
    if (!(ABL & 4)) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Dense GEMM with qmm_b4_*'s interface (QmmParams, buffers 0-5): Y (+)= X W^T, fp32 Y.
#define NI_QMRA(NAME, BM, BN, WM, WN, BK, ABL)                                                            \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                         \
      const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],                        \
      device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                   \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    constexpr int TM = BM / WM, TN = BN / WN;                                                              \
    threadgroup uint4 xs4[BM * (BK + 8) / 8];                                                              \
    const device uint32_t* W[1] = {Wd};                                                                    \
    const device bfloat* S[1] = {Sd};                                                                      \
    const device bfloat* B[1] = {Bd};                                                                      \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = tg.y * BM, n0 = tg.x * BN;                               \
    const uint2 rc = qr_rc(lane);                                                                          \
    simdgroup_float8x8 acc[TN / 8][TM / 8];                                                                \
    qr_acc<BM, BN, WM, WN, BK, 1, ABL>(X, W, S, B, p.M, p.N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4,  \
                                       tid, sgm, sgn, rc, acc);                                            \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int a = 0; a < TN / 8; ++a) {                                                                     \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                         \
      if (n >= p.N) continue;                                                                              \
      _Pragma("clang loop unroll(full)")                                                                   \
      for (int b = 0; b < TM / 8; ++b)                                                                     \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int e = 0; e < 2; ++e) {                                                                      \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                 \
          if (m < p.M) {                                                                                   \
            device float* dst = Y + m * p.y_ld + n;                                                        \
            const float v = acc[a][b].thread_elements()[e];                                                \
            *dst = p.add ? *dst + v : v;                                                                   \
          }                                                                                                \
        }                                                                                                  \
    }                                                                                                      \
  }
#define NI_QMR(NAME, BM, BN, WM, WN, BK) NI_QMRA(NAME, BM, BN, WM, WN, BK, 0)

// Fused dense MLP gate/up: H[m, n] = half(silu(X Wg^T) * (X Wu^T)) (y_ld: halfs per H row); both
// projections share the X tile. Buffers 0-5 as the dense kernel (W/S/B = gate, Y = H), 6-8 up.
#define NI_QMR_GU(NAME, BM, BN, WM, WN, BK)                                                               \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],                         \
      const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],                        \
      device half* H [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                    \
      const device uint32_t* Wu [[buffer(6)]], const device bfloat* Su [[buffer(7)]],                      \
      const device bfloat* Bu [[buffer(8)]],                                                               \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    constexpr int TM = BM / WM, TN = BN / WN, FA = TN / 8;                                                 \
    threadgroup uint4 xs4[BM * (BK + 8) / 8];                                                              \
    const device uint32_t* W[2] = {Wg, Wu};                                                                \
    const device bfloat* S[2] = {Sg, Su};                                                                  \
    const device bfloat* B[2] = {Bg, Bu};                                                                  \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = tg.y * BM, n0 = tg.x * BN;                               \
    const uint2 rc = qr_rc(lane);                                                                          \
    simdgroup_float8x8 acc[2 * FA][TM / 8];                                                                \
    qr_acc<BM, BN, WM, WN, BK, 2, 0>(X, W, S, B, p.M, p.N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4,    \
                                     tid, sgm, sgn, rc, acc);                                              \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int a = 0; a < FA; ++a) {                                                                         \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                         \
      if (n >= p.N) continue;                                                                              \
      _Pragma("clang loop unroll(full)")                                                                   \
      for (int b = 0; b < TM / 8; ++b)                                                                     \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int e = 0; e < 2; ++e) {                                                                      \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                 \
          const float g = acc[a][b].thread_elements()[e], u = acc[FA + a][b].thread_elements()[e];        \
          if (m < p.M) H[m * p.y_ld + n] = half(silu(g) * u);                                              \
        }                                                                                                  \
    }                                                                                                      \
  }

// Multi-projection form (NI_SHORTPF): up to 4 projections sharing X (same K, 4-bit g64), e.g. the
// GDN in-proj (qkv, z, b, a) or attention q/k/v, in ONE dispatch: tg.x runs over the concatenated N
// tiles of the segments (tg_begin), tg.y over M tiles. Per element the arithmetic is the
// single-projection kernel's with the same tile. Buffers: 4 x (W, S, B, Y), X at 16, params at 17
// (the host's kp::Qmms layout; `ks` unused).
struct QmrSegParams {
  uint K, M, x_ld, nseg, unused;
  uint N[4], y_ld[4], add[4], tg_begin[5];
};

#define NI_QMR_SEG(NAME, BM, BN, WM, WN, BK)                                                              \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device uint32_t* W0 [[buffer(0)]], const device bfloat* S0 [[buffer(1)]],                      \
      const device bfloat* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                               \
      const device uint32_t* W1 [[buffer(4)]], const device bfloat* S1 [[buffer(5)]],                      \
      const device bfloat* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                               \
      const device uint32_t* W2 [[buffer(8)]], const device bfloat* S2 [[buffer(9)]],                      \
      const device bfloat* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],                             \
      const device uint32_t* W3 [[buffer(12)]], const device bfloat* S3 [[buffer(13)]],                    \
      const device bfloat* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],                             \
      const device half* X [[buffer(16)]], constant QmrSegParams& p [[buffer(17)]],                        \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    constexpr int TM = BM / WM, TN = BN / WN;                                                              \
    threadgroup uint4 xs4[BM * (BK + 8) / 8];                                                              \
    uint si = 0;                                                                                           \
    while (si + 1 < p.nseg && tg.x >= p.tg_begin[si + 1]) ++si;                                             \
    const device uint32_t* W[1] = {si == 0 ? W0 : si == 1 ? W1 : si == 2 ? W2 : W3};                       \
    const device bfloat* S[1] = {si == 0 ? S0 : si == 1 ? S1 : si == 2 ? S2 : S3};                         \
    const device bfloat* B[1] = {si == 0 ? B0 : si == 1 ? B1 : si == 2 ? B2 : B3};                         \
    device float* Y = si == 0 ? Y0 : si == 1 ? Y1 : si == 2 ? Y2 : Y3;                                     \
    const uint N = p.N[si], y_ld = p.y_ld[si], add = p.add[si];                                            \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = tg.y * BM, n0 = (tg.x - p.tg_begin[si]) * BN;            \
    const uint2 rc = qr_rc(lane);                                                                          \
    simdgroup_float8x8 acc[TN / 8][TM / 8];                                                                \
    qr_acc<BM, BN, WM, WN, BK, 1, 0>(X, W, S, B, p.M, N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4, tid, \
                                     sgm, sgn, rc, acc);                                                   \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int a = 0; a < TN / 8; ++a) {                                                                     \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                         \
      if (n >= N) continue;                                                                                \
      _Pragma("clang loop unroll(full)")                                                                   \
      for (int b = 0; b < TM / 8; ++b)                                                                     \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int e = 0; e < 2; ++e) {                                                                      \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                 \
          if (m < p.M) {                                                                                   \
            device float* dst = Y + m * y_ld + n;                                                          \
            const float v = acc[a][b].thread_elements()[e];                                                \
            *dst = add ? *dst + v : v;                                                                     \
          }                                                                                                \
        }                                                                                                  \
    }                                                                                                      \
  }

// ---- 8-bit form (qmr8_*): MLX affine 8-bit g64 weights (the 35B's router and shared-expert gate,
// the MTP head's projections, quantized to 8 bits at load). A lane's 8 codes of a 32-block are 2
// words, and byte j of word e is code 8c + 4e + j: the 4-bit kernel's k for fragment j, column e, so
// X is staged exactly as there. 0x3400 | q << 2 is the half 0.25 + q/1024 (exact for q < 256);
// minus 0.25, times 1024*scale (normal for scales >= 6e-8), plus the bias: one half2 FMA rounding.
NI_INLINE half2 qr_pair8(uint w0, uint w1, uint j, half2 sb) {
  const uint t = (((w0 >> (8 * j)) & 0xFFu) << 2) | (((w1 >> (8 * j)) & 0xFFu) << 18) | 0x34003400u;
  return fma(as_type<half2>(t) - half2(0.25h), half2(sb.x), half2(sb.y));
}

// 8-bit qr_load_w: WPL8 = BK/16 words per lane per K step (lane c: words (BK/8)*c/... of the step).
template <int FA, int BK, int NP>
NI_INLINE void qr_load_w8(const device uint32_t* const thread* W, const device bfloat* const thread* S,
                          const device bfloat* const thread* B, thread const uint* wrow, uint wpr, uint gpr,
                          uint k0, uint c, thread uint* wr, thread half2* sb) {
  constexpr int WPL = BK / 16;
  const uint g = (k0 + 8 * (BK / 32) * c) / 64;
  _Pragma("clang loop unroll(full)")
  for (int p = 0; p < NP; ++p)
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      const int i = p * FA + a;
      const device uint* wp = W[p] + wrow[a] * wpr + k0 / 4 + WPL * c;
      if (WPL == 4) {
        uint4 v = *(const device uint4*)wp;
        wr[i * 4 + 0] = v.x; wr[i * 4 + 1] = v.y; wr[i * 4 + 2] = v.z; wr[i * 4 + 3] = v.w;
      } else {
        uint2 v = *(const device uint2*)wp;
        wr[i * 2 + 0] = v.x; wr[i * 2 + 1] = v.y;
      }
      if (BK >= 64 || (k0 & 63) == 0)
        sb[i] = half2(half(float(S[p][wrow[a] * gpr + g]) * 1024.0f), half(float(B[p][wrow[a] * gpr + g])));
    }
}

template <int FA, int FB, int BK, int TM>
NI_INLINE void qr_mma8(thread simdgroup_float8x8 (&acc)[FA][FB], threadgroup const half* xs, uint sgm,
                       thread const uint* wr, thread const half2* sb) {
  constexpr int LD = BK + 8, WPL = BK / 16, NF = BK / 8;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    simdgroup_half8x8 af[FA];
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      const half2 v = qr_pair8(wr[a * WPL + 2 * (f / 4)], wr[a * WPL + 2 * (f / 4) + 1], f % 4, sb[a]);
      af[a].thread_elements()[0] = v.x;
      af[a].thread_elements()[1] = v.y;
    }
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) {
      simdgroup_half8x8 bf;
      simdgroup_load(bf, xs + (sgm * TM + b * 8) * LD + 8 * f, LD, ulong2(0, 0), true);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FA; ++a) simdgroup_multiply_accumulate(acc[a][b], af[a], bf, acc[a][b]);
    }
  }
}

template <int BM, int BN, int WM, int WN, int BK, int NP>
NI_INLINE void qr_acc8(const device half* X, const device uint32_t* const thread* W,
                       const device bfloat* const thread* S, const device bfloat* const thread* B, uint M, uint N,
                       uint K, uint x_ld, uint m0, uint n0, threadgroup half* xs, uint tid, uint sgm, uint sgn,
                       uint2 rc, thread simdgroup_float8x8 (&acc)[NP * (BN / WN / 8)][BM / WM / 8]) {
  constexpr int NT = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int FB = TM / 8, FA = TN / 8;
  constexpr int CPR = BK / 8, XL = (BM * CPR + NT - 1) / NT;
  constexpr int WPL = BK / 16;
  const uint c = rc.y / 2;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < NP * FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) acc[a][b] = simdgroup_float8x8(0);
  const uint wpr = K / 4, gpr = K / 64, x_ld8 = x_ld / 8;
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, N - 1);
  const device uint4* X4 = (const device uint4*)X;
  uint4 xr[XL];
  uint wr[NP * FA * WPL];
  half2 sb[NP * FA];
  for (uint k0 = 0; k0 < K; k0 += BK) {
    qr_load_x<BM, BK, NT>(X4, x_ld8, M, m0, k0, tid, xr);
    qr_load_w8<FA, BK, NP>(W, S, B, wrow, wpr, gpr, k0, c, wr, sb);
    qr_store_x<BM, BK, NT>(xs, tid, xr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qr_mma8<NP * FA, FB, BK, TM>(acc, xs, sgm, wr, sb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// MFAST: grid x runs over the M tiles (the threadgroups running together share the weight tile through the cache)
#define NI_QMR8(NAME, BM, BN, WM, WN, BK) NI_QMR8_G(NAME, BM, BN, WM, WN, BK, false)
#define NI_QMR8_G(NAME, BM, BN, WM, WN, BK, MFAST)                                                        \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                         \
      const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],                        \
      device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                   \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    constexpr int TM = BM / WM, TN = BN / WN;                                                              \
    threadgroup uint4 xs4[BM * (BK + 8) / 8];                                                              \
    const device uint32_t* W[1] = {Wd};                                                                    \
    const device bfloat* S[1] = {Sd};                                                                      \
    const device bfloat* B[1] = {Bd};                                                                      \
    const uint mt = MFAST ? tg.x : tg.y, nt = MFAST ? tg.y : tg.x;                                        \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = mt * BM, n0 = nt * BN;                                   \
    const uint2 rc = qr_rc(lane);                                                                          \
    simdgroup_float8x8 acc[TN / 8][TM / 8];                                                                \
    qr_acc8<BM, BN, WM, WN, BK, 1>(X, W, S, B, p.M, p.N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4, tid, \
                                   sgm, sgn, rc, acc);                                                     \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int a = 0; a < TN / 8; ++a) {                                                                     \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                         \
      if (n >= p.N) continue;                                                                              \
      _Pragma("clang loop unroll(full)")                                                                   \
      for (int b = 0; b < TM / 8; ++b)                                                                     \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int e = 0; e < 2; ++e) {                                                                      \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                 \
          if (m < p.M) {                                                                                   \
            device float* dst = Y + m * p.y_ld + n;                                                        \
            const float v = acc[a][b].thread_elements()[e];                                                \
            *dst = p.add ? *dst + v : v;                                                                   \
          }                                                                                                \
        }                                                                                                  \
    }                                                                                                      \
  }

#define NI_QMR8_GU(NAME, BM, BN, WM, WN, BK)                                                              \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],                         \
      const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],                        \
      device half* H [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                    \
      const device uint32_t* Wu [[buffer(6)]], const device bfloat* Su [[buffer(7)]],                      \
      const device bfloat* Bu [[buffer(8)]],                                                               \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    constexpr int TM = BM / WM, TN = BN / WN, FA = TN / 8;                                                 \
    threadgroup uint4 xs4[BM * (BK + 8) / 8];                                                              \
    const device uint32_t* W[2] = {Wg, Wu};                                                                \
    const device bfloat* S[2] = {Sg, Su};                                                                  \
    const device bfloat* B[2] = {Bg, Bu};                                                                  \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = tg.y * BM, n0 = tg.x * BN;                               \
    const uint2 rc = qr_rc(lane);                                                                          \
    simdgroup_float8x8 acc[2 * FA][TM / 8];                                                                \
    qr_acc8<BM, BN, WM, WN, BK, 2>(X, W, S, B, p.M, p.N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4, tid, \
                                   sgm, sgn, rc, acc);                                                     \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int a = 0; a < FA; ++a) {                                                                         \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                         \
      if (n >= p.N) continue;                                                                              \
      _Pragma("clang loop unroll(full)")                                                                   \
      for (int b = 0; b < TM / 8; ++b)                                                                     \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int e = 0; e < 2; ++e) {                                                                      \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                 \
          const float g = acc[a][b].thread_elements()[e], u = acc[FA + a][b].thread_elements()[e];        \
          if (m < p.M) H[m * p.y_ld + n] = half(silu(g) * u);                                              \
        }                                                                                                  \
    }                                                                                                      \
  }

NI_QMR8(qmr8_k32_w14_m32_n128, 32, 128, 1, 4, 32)   // 8-bit, M >= 64
NI_QMR8_G(qmr8_k32_w14_m32_n128_mf, 32, 128, 1, 4, 32, true)  // the M1's (enc_qmm_large)
NI_QMR8(qmr8_k32_w14_m16_n64, 16, 64, 1, 4, 32)     // 8-bit, M < 64 or N <= 64
NI_QMR8_GU(qmr8_gu_k32_w14_m32_n64, 32, 64, 1, 4, 32)
NI_QMR8_GU(qmr8_gu_k32_w14_m16_n64, 16, 64, 1, 4, 32)

// naming: qmr[_gu]_k{BK}_w{WM}{WN}_m{BM}_n{BN}; threads 32*WM*WN; host requires bits 4, gs 64,
// K % 64 == 0 (x_ld % 8 == 0). Rows/columns past M/N are computed on clamped rows and not stored.
NI_QMR(qmr_k32_w14_m32_n128, 32, 128, 1, 4, 32)   // default for M >= 64: 80 GPRs, no stack
NI_QMR(qmr_k64_w14_m32_n128, 32, 128, 1, 4, 64)
NI_QMR(qmr_k64_w12_m32_n64, 32, 64, 1, 2, 64)     // as fast, but 96 GPRs + 16 B stack
NI_QMR(qmr_k32_w14_m64_n64, 64, 64, 1, 4, 32)
NI_QMR(qmr_k64_w14_m64_n64, 64, 64, 1, 4, 64)
NI_QMR(qmr_k64_w14_m32_n64, 32, 64, 1, 4, 64)
NI_QMR(qmr_k64_w14_m16_n128, 16, 128, 1, 4, 64)
NI_QMR(qmr_k64_w14_m16_n64, 16, 64, 1, 4, 64)     // M < 64 or N <= 64
NI_QMR(qmr_k64_w12_m16_n64, 16, 64, 1, 2, 64)
NI_QMR(qmr_k64_w12_m8_n64, 8, 64, 1, 2, 64)       // M <= 8
// short prefills (NI_SHORTPF, Qwen35::short_tile): one row tile holding all rows up to 48 (8-row
// steps; two tiles up to 128 rows), so each weight is read and dequantized once (twice) per pass
// instead of once per 16-row tile; qmrs_* = the multi-projection form above
NI_QMR(qmr_k64_w14_m24_n64, 24, 64, 1, 4, 64)
NI_QMR(qmr_k64_w14_m40_n64, 40, 64, 1, 4, 64)
NI_QMR(qmr_k64_w14_m48_n64, 48, 64, 1, 4, 64)
NI_QMR(qmr_k64_w14_m56_n64, 56, 64, 1, 4, 64)
NI_QMR_GU(qmr_gu_k64_w14_m24_n64, 24, 64, 1, 4, 64)
NI_QMR_GU(qmr_gu_k32_w14_m32_n64, 32, 64, 1, 4, 32)
NI_QMR_SEG(qmrs_k64_w12_m8_n64, 8, 64, 1, 2, 64)
NI_QMR_SEG(qmrs_k64_w14_m16_n64, 16, 64, 1, 4, 64)
NI_QMR_SEG(qmrs_k64_w14_m24_n64, 24, 64, 1, 4, 64)
NI_QMR_SEG(qmrs_k64_w14_m32_n64, 32, 64, 1, 4, 64)
NI_QMR_SEG(qmrs_k64_w14_m40_n64, 40, 64, 1, 4, 64)
NI_QMR_SEG(qmrs_k64_w14_m48_n64, 48, 64, 1, 4, 64)
NI_QMR_SEG(qmrs_k64_w14_m56_n64, 56, 64, 1, 4, 64)
NI_QMR_SEG(qmrs_k64_w14_m64_n64, 64, 64, 1, 4, 64)

NI_QMR_GU(qmr_gu_k64_w14_m32_n64, 32, 64, 1, 4, 64)
NI_QMR_GU(qmr_gu_k32_w12_m32_n32, 32, 32, 1, 2, 32)
NI_QMR_GU(qmr_gu_k64_w14_m16_n64, 16, 64, 1, 4, 64)
NI_QMR_GU(qmr_gu_k64_w12_m8_n32, 8, 32, 1, 2, 64)
// ablations (bench only): 1 no dequant math, 2 no X staging, 4 no barriers, 8 no W loads
NI_QMRA(qmra1_k32_w14_m32_n128, 32, 128, 1, 4, 32, 1)
NI_QMRA(qmra2_k32_w14_m32_n128, 32, 128, 1, 4, 32, 2)
NI_QMRA(qmra6_k32_w14_m32_n128, 32, 128, 1, 4, 32, 6)
NI_QMRA(qmra8_k32_w14_m32_n128, 32, 128, 1, 4, 32, 8)
NI_QMRA(qmra9_k32_w14_m32_n128, 32, 128, 1, 4, 32, 9)
NI_QMRA(qmra15_k32_w14_m32_n128, 32, 128, 1, 4, 32, 15)
