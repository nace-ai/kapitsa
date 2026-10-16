#include "common.h"

// ---- Prefill: quantized GEMM  Y[M, N] (+)= X[M, K] * W[N, K]^T ----
// X is half (already normalized where needed); W is MLX affine 4/8-bit; Y is fp32.
// Threadgroup tile BM x BN, K step BK=32 (half a 64-group or one 32-group), 4 simdgroups in a
// 2x2 layout, each computing (BM/2) x (BN/2) with 8x8 simdgroup matrices. W tiles are dequantized
// into threadgroup memory as half.
//
// Rows can be gathered (MoE): row r of the tile reads X row `xrow[r]` and writes Y row `yrow[r]`
// via index lists; the dense path uses identity indices.

struct QmmParams {
  uint M;        // rows of X / Y
  uint N;
  uint K;
  uint x_ld;     // halfs per X row
  uint y_ld;     // floats per Y row
  uint add;      // 1: Y += result
  uint gs;       // group size (32 or 64)
  // MoE grouping (gather mode): tiles are listed in tile_info as (expert, row_begin, row_end)
  uint w_expert_stride;   // words between experts' weight matrices
  uint s_expert_stride;   // scales/biases elements between experts
};

constant constexpr int BK = 32;
constant constexpr int PAD = 8;

template <int BITS, int BM, int BN>
NI_INLINE void qmm_tile(const device half* X, const device uint32_t* W, const device bfloat* S,
                        const device bfloat* B, device float* Y, constant QmmParams& p, uint m0, uint m_end,
                        uint n0, const device uint* xrows, const device uint* yrows, threadgroup half* xs,
                        threadgroup half* ws, uint tid, uint sg, uint lane) {
  constexpr int TM = BM / 2, TN = BN / 2;     // per-simdgroup tile
  constexpr int FM = TM / 8, FN = TN / 8;     // 8x8 fragments per simdgroup
  constexpr int PER_WORD = 32 / BITS;
  const uint sgm = sg / 2, sgn = sg % 2;

  simdgroup_float8x8 acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);

  const uint K = p.K;
  const uint wpr = K / PER_WORD;
  const uint gpr = K / p.gs;
  for (uint k0 = 0; k0 < K; k0 += BK) {
    // X tile: BM rows x 32 halfs; 128 threads, each loads (BM*32/128) halfs as half4s
    for (uint i = tid; i < BM * BK / 4; i += 128) {
      uint r = i / (BK / 4), c4 = i % (BK / 4);
      uint m = m0 + r;
      half4 v = half4(0);
      if (m < m_end) {
        uint xr = xrows ? xrows[m] : m;
        v = *(const device half4*)(X + xr * p.x_ld + k0 + c4 * 4);
      }
      *(threadgroup half4*)(xs + r * (BK + PAD) + c4 * 4) = v;
    }
    // W tile: BN rows x 32 values = BN * (32/PER_WORD) words
    for (uint i = tid; i < BN * (BK / PER_WORD); i += 128) {
      uint r = i / (BK / PER_WORD), wi = i % (BK / PER_WORD);
      uint n = min(n0 + r, p.N - 1);
      uint32_t word = W[n * wpr + k0 / PER_WORD + wi];
      uint g = (k0 + wi * PER_WORD) / p.gs;
      float sc = float(S[n * gpr + g]), bi = float(B[n * gpr + g]);
      threadgroup half* dst = ws + r * (BK + PAD) + wi * PER_WORD;
      for (int j = 0; j < PER_WORD; ++j) dst[j] = half(sc * float((word >> (BITS * j)) & ((1u << BITS) - 1u)) + bi);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_half8x8 a[FM], b[FN];
      for (int i = 0; i < FM; ++i) simdgroup_load(a[i], xs + (sgm * TM + i * 8) * (BK + PAD) + kk, BK + PAD);
      for (int j = 0; j < FN; ++j)
        simdgroup_load(b[j], ws + (sgn * TN + j * 8) * (BK + PAD) + kk, BK + PAD, ulong2(0, 0), true);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store via threadgroup scratch (reuse xs as float staging per simdgroup: 8x8 floats each)
  threadgroup float* stage = (threadgroup float*)ws + sg * 64;
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      simdgroup_store(acc[i][j], stage, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) {
        uint r = e / 8, c = e % 8;
        uint m = m0 + sgm * TM + i * 8 + r;
        uint n = n0 + sgn * TN + j * 8 + c;
        if (m < m_end && n < p.N) {
          uint yr = yrows ? yrows[m] : m;
          device float* dst = Y + yr * p.y_ld + n;
          float v = stage[e];
          *dst = p.add ? *dst + v : v;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define NI_QMM(NAME, BITS, BM, BN)                                                                        \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],         \
                       const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],         \
                       device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                 \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[BM * (BK + PAD)];                                                                 \
    threadgroup half ws[(BN > 64 ? BN : 64) * (BK + PAD)];                                                \
    uint m0 = tg.y * BM, n0 = tg.x * BN;                                                                  \
    qmm_tile<BITS, BM, BN>(X, W, S, B, Y, p, m0, p.M, n0, nullptr, nullptr, xs, ws, tid, sg, lane);       \
  }

NI_QMM(qmm_b4_m32_n32, 4, 32, 32)
NI_QMM(qmm_b4_m64_n64, 4, 64, 64)
NI_QMM(qmm_b4_m16_n64, 4, 16, 64)
NI_QMM(qmm_b4_m32_n64, 4, 32, 64)
NI_QMM(qmm_b4_m64_n32, 4, 64, 32)
NI_QMM(qmm_b4_m32_n128, 4, 32, 128)
NI_QMM(qmm_b8_m32_n32, 8, 32, 32)
NI_QMM(qmm_b8_m64_n64, 8, 64, 64)

// MoE grouped GEMM. tiles[i] = (expert, row_begin, row_end) over the expert-sorted pair list;
// xrows/yrows map sorted pair rows to X rows (token) and Y rows (pair). ntiles[0] = valid tiles.
#define NI_QMM_MOE(NAME, BITS, BM, BN)                                                                    \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],         \
                       const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],         \
                       device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                 \
                       const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],    \
                       const device uint* xrows [[buffer(8)]], const device uint* yrows [[buffer(9)]],     \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    if (tg.y >= ntiles[0]) return;                                                                        \
    threadgroup half xs[BM * (BK + PAD)];                                                                 \
    threadgroup half ws[(BN > 64 ? BN : 64) * (BK + PAD)];                                                \
    uint e = tiles[tg.y * 3 + 0], r0 = tiles[tg.y * 3 + 1], r1 = tiles[tg.y * 3 + 2];                     \
    qmm_tile<BITS, BM, BN>(X, W + e * p.w_expert_stride, S + e * p.s_expert_stride,                      \
                           B + e * p.s_expert_stride, Y, p, r0, r1, tg.x * BN, xrows, yrows, xs, ws, tid, \
                           sg, lane);                                                                     \
  }

NI_QMM_MOE(qmm_moe_b4_m16_n64, 4, 16, 64)
NI_QMM_MOE(qmm_moe_b4_m32_n64, 4, 32, 64)
NI_QMM_MOE(qmm_moe_b4_m64_n64, 4, 64, 64)

// ---- qmm2: dense prefill GEMM with BK = 64 (one quantization group per step) ----
// 128 threads (2x2 simdgroups), tile BM x 64 output columns. Each thread dequantizes one row's
// 64-value group (or half of it) with a single scale/bias load and vector stores, and X tiles are
// loaded as half4. Half the barriers of qmm_tile (BK = 32).
template <int BITS, int BM, int BN>
NI_INLINE void qmm2_tile(const device half* X, const device uint32_t* W, const device bfloat* S,
                         const device bfloat* B, device float* Y, constant QmmParams& p, uint m0, uint m_end,
                         uint n0, threadgroup half* xs, threadgroup half* ws, uint tid, uint sg, uint lane) {
  constexpr int BK2 = 64, LD = BK2 + 8;
  constexpr int TM = BM / 2, TN = BN / 2;
  constexpr int FM = TM / 8, FN = TN / 8;
  constexpr int PER_WORD = 32 / BITS;
  constexpr int WPG = BK2 / PER_WORD;        // words per row per group (8 for 4-bit)
  const uint sgm = sg / 2, sgn = sg % 2;
  simdgroup_float8x8 acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);
  const uint K = p.K, wpr = K / PER_WORD, gpr = K / 64;
  for (uint k0 = 0; k0 < K; k0 += BK2) {
    for (uint i = tid; i < BM * BK2 / 4; i += 128) {
      uint r = i / (BK2 / 4), c4 = i % (BK2 / 4);
      uint m = m0 + r;
      half4 v = m < m_end ? *(const device half4*)(X + m * p.x_ld + k0 + c4 * 4) : half4(0);
      *(threadgroup half4*)(xs + r * LD + c4 * 4) = v;
    }
    // W: BN rows x WPG words; each thread takes 4 consecutive words of one row (same group)
    for (uint i = tid; i < BN * WPG / 4; i += 128) {
      uint r = i / (WPG / 4), q = i % (WPG / 4);
      uint n = min(n0 + r, p.N - 1);
      const float sc = float(S[n * gpr + k0 / 64]), bi = float(B[n * gpr + k0 / 64]);
      uint4 w4 = *(const device uint4*)(W + n * wpr + k0 / PER_WORD + q * 4);
      threadgroup half* dst = ws + r * LD + q * 4 * PER_WORD;
      for (int wi = 0; wi < 4; ++wi) {
        uint word = w4[wi];
        for (int j = 0; j < PER_WORD; j += 4) {
          half4 h;
          for (int e = 0; e < 4; ++e) h[e] = half(sc * float((word >> (BITS * (j + e))) & ((1u << BITS) - 1u)) + bi);
          *(threadgroup half4*)(dst + wi * PER_WORD + j) = h;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BK2; kk += 8) {
      simdgroup_half8x8 a[FM], b[FN];
      for (int i = 0; i < FM; ++i) simdgroup_load(a[i], xs + (sgm * TM + i * 8) * LD + kk, LD);
      for (int j = 0; j < FN; ++j) simdgroup_load(b[j], ws + (sgn * TN + j * 8) * LD + kk, LD, ulong2(0, 0), true);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // epilogue: each lane writes its own fragment elements directly
  const uint qd = lane / 4;
  const uint fm = (qd & 4) + (lane / 2) % 4, fn = (qd & 2) * 2 + (lane % 2) * 2;
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      uint m = m0 + sgm * TM + i * 8 + fm;
      uint n = n0 + sgn * TN + j * 8 + fn;
      if (m >= m_end) continue;
      for (int e = 0; e < 2; ++e) {
        if (n + e >= p.N) continue;
        device float* dst = Y + m * p.y_ld + n + e;
        float v = acc[i][j].thread_elements()[e];
        *dst = p.add ? *dst + v : v;
      }
    }
}

#define NI_QMM2(NAME, BITS, BM, BN)                                                                       \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],         \
                       const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],         \
                       device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                 \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[BM * 72];                                                                         \
    threadgroup half ws[BN * 72];                                                                         \
    qmm2_tile<BITS, BM, BN>(X, W, S, B, Y, p, tg.y * BM, p.M, tg.x * BN, xs, ws, tid, sg, lane);         \
  }

NI_QMM2(qmm2_b4_m64_n64, 4, 64, 64)
NI_QMM2(qmm2_b4_m32_n64, 4, 32, 64)
NI_QMM2(qmm2_b4_m64_n32, 4, 64, 32)
NI_QMM2(qmm2_b4_m16_n64, 4, 16, 64)
NI_QMM2(qmm2_b8_m64_n64, 8, 64, 64)
