#include "common.h"

// ---- qmr: dense prefill GEMM with the weights dequantized straight into MMA registers ----
// Y[M, N] (+)= X[M, K] * W[N, K]^T, X half, W MLX affine 4-bit (group 64, bf16 scales/biases), Y fp32.
// Computes the transposed tile C = W_tile * X_tile^T with W as the A operand (rows n, columns k):
// in an 8x8 A fragment a lane holds row (lane/4 & 4) + (lane/2) % 4 and two adjacent columns
// 2c, 2c+1 (c = (lane/4 & 2) + lane % 2). The K order inside each BK block is permuted so that a
// lane's columns over the BK/8 fragments are exactly the 8 (BK=32) or 16 (BK=64) nibbles of its own
// 4/8 bytes of the packed row: fragment f, column 2c+e <-> nibble (f%4) + 4e of word
// (BK/32)*c + f/4. A fragment element pair is then one shift/mask/or of the word (nibbles j and
// j+4 land in the two 16-bit halves), as half2 1024+q (0x6400 | q), minus 1024, times the scale plus
// the bias (half2 FMA). X is staged in threadgroup memory in the same permuted K order and read as
// the transposed B operand. Each W element is dequantized once per threadgroup (simdgroups split N)
// when WM == 1, and W never touches threadgroup memory.
// Requirements (host): bits 4, gs 64, K % BK == 0.

// lane -> (row, column pair) of an 8x8 fragment (same layout as q2_lane_rc)
NI_INLINE uint2 qr_rc(uint lane) {
  uint q = lane / 4;
  return uint2((q & 4) + (lane / 2) % 4, (q & 2) * 2 + (lane % 2) * 2);
}

// One A-fragment element pair: nibbles j and j+4 of w (j = 0..3), dequantized as half2;
// sb = (scale, bias).
NI_INLINE half2 qr_pair(uint w, uint j, half2 sb) {
  uint t = ((w >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
  return fma(as_type<half2>(t) - half2(1024.0h), half2(sb.x), half2(sb.y));
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

// This lane's packed words of its FA rows for the K step at k0, and their (scale, bias): the lane's
// 8*WPL values lie in one 64-group, (k0 + 8*WPL*c) / 64.
template <int FA, int BK>
NI_INLINE void qr_load_w(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                         thread const uint* wrow, uint wpr, uint gpr, uint k0, uint c, thread uint* wr,
                         thread half2* sb) {
  constexpr int WPL = BK / 32;
  const uint g = (k0 + 8 * WPL * c) / 64;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) {
    const device uint* wp = W + wrow[a] * wpr + k0 / 8 + WPL * c;
    if (WPL == 4) {
      uint4 v = *(const device uint4*)wp;
      wr[a * 4 + 0] = v.x; wr[a * 4 + 1] = v.y; wr[a * 4 + 2] = v.z; wr[a * 4 + 3] = v.w;
    } else if (WPL == 2) {
      uint2 v = *(const device uint2*)wp;
      wr[a * 2 + 0] = v.x; wr[a * 2 + 1] = v.y;
    } else {
      wr[a] = wp[0];
    }
    if (BK >= 64 || (k0 & 63) == 0)
      sb[a] = half2(half(float(S[wrow[a] * gpr + g])), half(float(B[wrow[a] * gpr + g])));
  }
}

template <int FA, int FB, int BK, int TM>
NI_INLINE void qr_mma(thread simdgroup_float8x8 (&acc)[FA][FB], threadgroup const half* xs, uint sgm,
                      thread const uint* wr, thread const half2* sb) {
  constexpr int LD = BK + 8, WPL = BK / 32, NF = BK / 8;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    simdgroup_half8x8 af[FA];
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      half2 v = qr_pair(wr[a * WPL + f / 4], f % 4, sb[a]);
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

// MODE 0: load X/W, stage X, barrier, MMAs, barrier (one X buffer).
// MODE 1: two X buffers, one barrier per step; the next X tile's device loads are in flight during
// the MMAs, W words for the next step are loaded after them.
template <int BM, int BN, int WM, int WN, int BK, int MODE>
NI_INLINE void qr_tile(const device half* X, const device uint32_t* W, const device bfloat* S,
                       const device bfloat* B, device float* Y, constant QmmParams& p, uint m0, uint n0,
                       threadgroup half* xs, uint tid, uint sg, uint lane) {
  constexpr int NT = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int FB = TM / 8, FA = TN / 8;
  constexpr int CPR = BK / 8, XL = (BM * CPR + NT - 1) / NT;
  constexpr int WPL = BK / 32;
  constexpr int XT = BM * (BK + 8);  // halfs per X buffer
  const uint sgm = sg / WN, sgn = sg % WN;
  const uint2 rc = qr_rc(lane);
  const uint c = rc.y / 2;

  simdgroup_float8x8 acc[FA][FB];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) acc[a][b] = simdgroup_float8x8(0);

  const uint K = p.K, wpr = K / 8, gpr = K / 64, x_ld8 = p.x_ld / 8;
  // this lane's W rows (clamped: rows past N compute garbage that is never stored)
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, p.N - 1);
  const device uint4* X4 = (const device uint4*)X;

  uint4 xr[XL];
  uint wr[FA * WPL];
  half2 sb[FA];
  if (MODE == 1) {
    qr_load_x<BM, BK, NT>(X4, x_ld8, p.M, m0, 0, tid, xr);
    qr_load_w<FA, BK>(W, S, B, wrow, wpr, gpr, 0, c, wr, sb);
    qr_store_x<BM, BK, NT>(xs, tid, xr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cur = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
      const bool more = k0 + BK < K;
      if (more) qr_load_x<BM, BK, NT>(X4, x_ld8, p.M, m0, k0 + BK, tid, xr);
      qr_mma<FA, FB, BK, TM>(acc, xs + cur * XT, sgm, wr, sb);
      if (more) {
        qr_store_x<BM, BK, NT>(xs + (cur ^ 1) * XT, tid, xr);
        qr_load_w<FA, BK>(W, S, B, wrow, wpr, gpr, k0 + BK, c, wr, sb);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      cur ^= 1;
    }
  } else {
    for (uint k0 = 0; k0 < K; k0 += BK) {
      qr_load_x<BM, BK, NT>(X4, x_ld8, p.M, m0, k0, tid, xr);
      qr_load_w<FA, BK>(W, S, B, wrow, wpr, gpr, k0, c, wr, sb);
      qr_store_x<BM, BK, NT>(xs, tid, xr);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      qr_mma<FA, FB, BK, TM>(acc, xs, sgm, wr, sb);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // epilogue: lane holds C[n = rc.x][m = rc.y + e] of each fragment
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) {
    const uint n = n0 + sgn * TN + a * 8 + rc.x;
    if (n >= p.N) continue;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) {
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint m = m0 + sgm * TM + b * 8 + rc.y + e;
        if (m < p.M) {
          device float* dst = Y + m * p.y_ld + n;
          float v = acc[a][b].thread_elements()[e];
          *dst = p.add ? *dst + v : v;
        }
      }
    }
  }
}

// SW: rasterization swizzle, log2 of the number of M tiles walked together (0: row-major, N fastest)
#define NI_QMR(NAME, BM, BN, WM, WN, BK, MODE, SW)                                                        \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device half* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],                          \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],                          \
      device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                   \
      uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],                      \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                 \
      uint lane [[thread_index_in_simdgroup]]) {                                                           \
    threadgroup uint4 xs4[(MODE == 1 ? 2 : 1) * BM * (BK + 8) / 8];                                        \
    uint tm = tg.y, tn = tg.x;                                                                             \
    if (SW > 0) {                                                                                          \
      const uint G = 1u << SW, t = tg.y * ntg.x + tg.x, grp = t / (G * ntg.x);                             \
      const uint rows = min(G, ntg.y - grp * G), r = t - grp * G * ntg.x;                                  \
      tm = grp * G + r % rows;                                                                             \
      tn = r / rows;                                                                                       \
    }                                                                                                      \
    qr_tile<BM, BN, WM, WN, BK, MODE>(X, W, S, B, Y, p, tm * BM, tn * BN, (threadgroup half*)xs4, tid, sg, \
                                      lane);                                                               \
  }

// naming: qmr[d][s<SW>]_k{BK}_w{WM}{WN}_m{BM}_n{BN}  (d: MODE 1)
NI_QMR(qmr_k32_w14_m64_n64, 64, 64, 1, 4, 32, 0, 0)
NI_QMR(qmr_k64_w14_m64_n64, 64, 64, 1, 4, 64, 0, 0)
NI_QMR(qmr_k128_w14_m64_n64, 64, 64, 1, 4, 128, 0, 0)
NI_QMR(qmr_k32_w14_m32_n128, 32, 128, 1, 4, 32, 0, 0)
NI_QMR(qmr_k64_w14_m32_n128, 32, 128, 1, 4, 64, 0, 0)
NI_QMR(qmr_k128_w14_m32_n128, 32, 128, 1, 4, 128, 0, 0)
NI_QMR(qmr_k64_w14_m32_n64, 32, 64, 1, 4, 64, 0, 0)
NI_QMR(qmr_k64_w14_m16_n128, 16, 128, 1, 4, 64, 0, 0)
NI_QMR(qmr_k64_w12_m32_n64, 32, 64, 1, 2, 64, 0, 0)
NI_QMR(qmr_k128_w12_m32_n64, 32, 64, 1, 2, 128, 0, 0)
NI_QMR(qmrd_k32_w14_m32_n128, 32, 128, 1, 4, 32, 1, 0)
NI_QMR(qmrd_k64_w14_m32_n128, 32, 128, 1, 4, 64, 1, 0)
NI_QMR(qmrd_k64_w14_m64_n64, 64, 64, 1, 4, 64, 1, 0)
NI_QMR(qmrd_k128_w14_m32_n128, 32, 128, 1, 4, 128, 1, 0)
NI_QMR(qmrs2_k64_w14_m32_n128, 32, 128, 1, 4, 64, 0, 2)
NI_QMR(qmr_k64_w12_m16_n64, 16, 64, 1, 2, 64, 0, 0)
NI_QMR(qmr_k64_w14_m16_n64, 16, 64, 1, 4, 64, 0, 0)
NI_QMR(qmr_k64_w14_m8_n128, 8, 128, 1, 4, 64, 0, 0)
NI_QMR(qmr_k64_w12_m8_n64, 8, 64, 1, 2, 64, 0, 0)
NI_QMR(qmr_k32_w12_m32_n64, 32, 64, 1, 2, 32, 0, 0)
