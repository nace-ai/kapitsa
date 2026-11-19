#include "common.h"

// ---- Routed experts in MLX affine 3- or 4-bit, or Q2_0, with a clamped SwiGLU (Hy4-flash;
// src/model/moe_a4.h) ----
// An expert is three matrices, gate and up [I, D] and down [D, I], each three planes: codes [N][K * BITS / 32] uint32
// (each row a little-endian bit stream, value j at bits [BITS j, BITS j + BITS): MLX's packing, for 3 bits eight
// values in three bytes), scales and biases [N][K/GS] bf16, weight = code * scale + bias. Gate and up share a format
// (BITS, GS); down has its own. BITS == 2 is Q2_0 only: GS 64, signed f16 d, (code - 1) * d, no bias.
// Kernel names carry the format: a4_gu_rows_b3g64_slot or a4_gu_rows_q20. The planes are either stacked
// over the experts ([E][N][...]: w0..w2 = the matrix's codes, scales, biases) or one slot of the store's arena read
// through the layer's slot table (SLOT: w0 = arena chunk 0, w1 = chunk 1, w2 = the slot table of E words
// chunk << 28 | page, the planes at A4Slot's byte offsets), as zq2moe.metal's q20_mat<SLOT> does for Q2_0.
// h = silu(min(g, limit)) * clamp(u, -limit, limit) (limit <= 0: no clamp), y[r] = sum_j wts[r, j] down_e(h).
// Rows (decode, verify): a4_gu_rows (h fp32, one threadgroup per (pair, row tile)), a4_dn_rows (y overwritten, the k
// experts of a row summed in order in one simdgroup). Prefill: moe_bucket3's tiles (qmm_moe.metal) through a4_gu_mm
// (H half in sorted order) and a4_dn_mm (Y fp32 by pair), then a4_combine. Static threadgroup memory <= 8 KB each.
#ifndef NI_WF_INST

struct A4Params {
  uint D, I, k, E;
  float limit;
  uint R;
};

struct A4Slot {
  uint o[9];  // plane byte offsets inside a slot: gate w s b, up w s b, down w s b
};

struct A4Mat {
  const device uint32_t* w;
  const device bfloat* s;
  const device bfloat* b;
};

// Expert e's matrix m (0 gate, 1 up, 2 down) of N rows and K columns
template <bool SLOT, int BITS, int GS>
NI_INLINE A4Mat a4_mat(const device uchar* w0, const device uchar* w1, const device uchar* w2, uint e, uint N, uint K,
                       constant A4Slot& L, uint m) {
  if (SLOT) {
    const uint sw = ((const device uint*)w2)[e];
    const device uchar* b = ((sw >> 28) != 0u ? w1 : w0) + ulong(sw & 0x0FFFFFFFu) * 16384ul;
    return A4Mat{(const device uint32_t*)(b + L.o[3 * m]), (const device bfloat*)(b + L.o[3 * m + 1]),
                 (const device bfloat*)(b + L.o[3 * m + 2])};
  }
  const ulong r = ulong(e) * N;
  return A4Mat{(const device uint32_t*)w0 + r * (K * BITS / 32), (const device bfloat*)w1 + r * (K / GS),
               (const device bfloat*)w2 + r * (K / GS)};
}

// 3-bit dot product of a row's values [k0, k0 + 16) (k0 % 16 == 0: 48 bits starting on a 16-bit boundary, inside two
// words of the row) with xv
NI_INLINE float a3_dot16(const device uint32_t* row, uint k0, thread const float* xv) {
  const uint bit0 = 3u * k0;
  const device uint32_t* p = row + bit0 / 32;
  uint lo = p[0], hi = p[1];
  if (bit0 & 16u) {
    lo = (lo >> 16) | (hi << 16);
    hi >>= 16;
  }
  float d = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 16; ++j) {
    const int b = 3 * j;
    const uint c = b + 3 <= 32 ? lo >> b : b >= 32 ? hi >> (b - 32) : (lo >> b) | (hi << (32 - b));
    d = fma(float(c & 7u), xv[j], d);
  }
  return d;
}

// acc[r] = row (row0 + r) . x for ROWS rows (simd-summed), 3-bit: lanes take 16 values at a time (qmv2_rows' layout)
#define NI_A3_ROWS(NAME, XSPACE)                                                                                   \
  template <int GS, int ROWS>                                                                                     \
  NI_INLINE void NAME(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,         \
                      XSPACE const float* xs, uint N, uint K, uint row0, uint lane, thread float* acc) {         \
    const uint wpr = K * 3 / 32, gpr = K / GS;                                                                    \
    uint rows[ROWS];                                                                                              \
    for (int r = 0; r < ROWS; ++r) {                                                                              \
      acc[r] = 0.0f;                                                                                              \
      rows[r] = min(row0 + r, N - 1);                                                                             \
    }                                                                                                             \
    for (uint k0 = lane * 16; k0 < K; k0 += 512) {                                                                \
      float xv[16];                                                                                               \
      float xsum = 0.0f;                                                                                          \
      for (int i = 0; i < 4; ++i) {                                                                               \
        const float4 v = ((XSPACE const float4*)(xs + k0))[i];                                                    \
        xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;                           \
        xsum += v.x + v.y + v.z + v.w;                                                                            \
      }                                                                                                           \
      const uint g = k0 / GS;                                                                                     \
      for (int r = 0; r < ROWS; ++r) {                                                                            \
        const float d = a3_dot16(w + ulong(rows[r]) * wpr, k0, xv);                                               \
        acc[r] += float(scales[ulong(rows[r]) * gpr + g]) * d + float(biases[ulong(rows[r]) * gpr + g]) * xsum;   \
      }                                                                                                           \
    }                                                                                                             \
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);                                                     \
  }
NI_A3_ROWS(a3_rows_tg, threadgroup)
NI_A3_ROWS(a3_rows_dx, device)

// Q2_0 rows: each lane owns 16 consecutive codes, all sharing one signed f16 scale.
#define NI_A4_Q20_ROWS(NAME, XSPACE)                                                                              \
  template <int ROWS>                                                                                            \
  NI_INLINE void NAME(const device uint32_t* w, const device half* scales, XSPACE const float* x,                 \
                      uint N, uint K, uint row0, uint lane, thread float* acc) {                                 \
    const uint wpr = K / 16, gpr = K / 64;                                                                       \
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;                                                                \
    for (uint k0 = lane * 16; k0 < K; k0 += 512) {                                                              \
      float xv[16];                                                                                              \
      for (int j = 0; j < 16; ++j) xv[j] = x[k0 + j];                                                           \
      for (int r = 0; r < ROWS; ++r) {                                                                          \
        const uint row = min(row0 + r, N - 1);                                                                   \
        uint codes = w[ulong(row) * wpr + k0 / 16];                                                              \
        float dot = 0.0f;                                                                                        \
        for (int j = 0; j < 16; ++j) {                                                                          \
          dot = fma(float(int(codes & 3u) - 1), xv[j], dot);                                                     \
          codes >>= 2;                                                                                           \
        }                                                                                                        \
        acc[r] += float(scales[ulong(row) * gpr + k0 / 64]) * dot;                                               \
      }                                                                                                          \
    }                                                                                                            \
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);                                                    \
  }
NI_A4_Q20_ROWS(a4_q20_rows_tg, threadgroup)
NI_A4_Q20_ROWS(a4_q20_rows_dx, device)

NI_INLINE float a4_swiglu(float g, float u, float lim) {
  if (lim > 0.0f) {
    g = min(g, lim);
    u = clamp(u, -lim, lim);
  }
  return silu(g) * u;
}

// ---- rows ----

// grid: x = row tiles of nsg * ROWS rows over I, y = pair (r * k + j); h[pair, I] fp32. D <= 2048.
template <bool SLOT, int BITS, int GS, int ROWS>
NI_INLINE void a4_gu_rows_impl(const device uchar* g0, const device uchar* g1, const device uchar* g2,
                               const device uchar* u0, const device uchar* u1, const device uchar* u2,
                               const device float* x, const device uint* idx, device float* h, constant A4Params& p,
                               constant A4Slot& L, uint2 tgp, uint tid, uint ntg, uint sg, uint nsg, uint lane,
                               threadgroup float* xs) {
  const uint pair = tgp.y, r = pair / p.k;
  for (uint i = tid; i < p.D / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)(x + ulong(r) * p.D))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= p.I) return;
  const uint e = min(idx[pair], p.E - 1);
  const A4Mat G = a4_mat<SLOT, BITS, GS>(g0, g1, g2, e, p.I, p.D, L, 0),
              U = a4_mat<SLOT, BITS, GS>(u0, u1, u2, e, p.I, p.D, L, 1);
  float ag[ROWS], au[ROWS];
  if (BITS == 2) {
    a4_q20_rows_tg<ROWS>(G.w, (const device half*)G.s, xs, p.I, p.D, row0, lane, ag);
    a4_q20_rows_tg<ROWS>(U.w, (const device half*)U.s, xs, p.I, p.D, row0, lane, au);
  } else if (BITS == 4) {
    qmv2_rows<4, 16, ROWS, GS>(G.w, G.s, G.b, xs, p.I, p.D, row0, lane, ag);
    qmv2_rows<4, 16, ROWS, GS>(U.w, U.s, U.b, xs, p.I, p.D, row0, lane, au);
  } else {
    a3_rows_tg<GS, ROWS>(G.w, G.s, G.b, xs, p.I, p.D, row0, lane, ag);
    a3_rows_tg<GS, ROWS>(U.w, U.s, U.b, xs, p.I, p.D, row0, lane, au);
  }
  if (lane == 0)
    for (int i = 0; i < ROWS; ++i)
      if (row0 + i < p.I) h[ulong(pair) * p.I + row0 + i] = a4_swiglu(ag[i], au[i], p.limit);
}

#define NI_A4_GU_ROWS(NAME, SLOT, BITS, GS)                                                                           \
  [[kernel]] void NAME(const device uchar* g0 [[buffer(0)]], const device uchar* g1 [[buffer(1)]],                 \
                       const device uchar* g2 [[buffer(2)]], const device uchar* u0 [[buffer(3)]],                 \
                       const device uchar* u1 [[buffer(4)]], const device uchar* u2 [[buffer(5)]],                 \
                       const device float* x [[buffer(6)]], const device uint* idx [[buffer(7)]],                  \
                       device float* h [[buffer(8)]], constant A4Params& p [[buffer(9)]],                          \
                       constant A4Slot& L [[buffer(10)]], uint2 tgp [[threadgroup_position_in_grid]],              \
                       uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]],           \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    threadgroup float xs[2048];                                                                                    \
    a4_gu_rows_impl<SLOT, BITS, GS, 2>(g0, g1, g2, u0, u1, u2, x, idx, h, p, L, tgp, tid, ntg2.x, sg, nsg, lane, xs); \
  }
#define NI_A4_ALL(X, BASE)                                                                                       \
  X(BASE##_q20, false, 2, 64) X(BASE##_q20_slot, true, 2, 64)                                                    \
  X(BASE##_b3g32, false, 3, 32) X(BASE##_b3g32_slot, true, 3, 32)                                                \
  X(BASE##_b3g64, false, 3, 64) X(BASE##_b3g64_slot, true, 3, 64)                                                \
  X(BASE##_b3g128, false, 3, 128) X(BASE##_b3g128_slot, true, 3, 128)                                            \
  X(BASE##_b4g32, false, 4, 32) X(BASE##_b4g32_slot, true, 4, 32)                                                \
  X(BASE##_b4g64, false, 4, 64) X(BASE##_b4g64_slot, true, 4, 64)                                                \
  X(BASE##_b4g128, false, 4, 128) X(BASE##_b4g128_slot, true, 4, 128)
NI_A4_ALL(NI_A4_GU_ROWS, a4_gu_rows)

// grid: x = row tiles of nsg * ROWS rows over D, y = row r; y[r, D] = sum_j wts[r, j] * down_e(h[r * k + j]), the
// experts in order j = 0 .. k - 1 (h read through the caches)
template <bool SLOT, int BITS, int GS, int ROWS>
NI_INLINE void a4_dn_rows_impl(const device uchar* d0, const device uchar* d1, const device uchar* d2,
                               const device float* h, const device uint* idx, const device float* wts, device float* y,
                               constant A4Params& p, constant A4Slot& L, uint2 tgp, uint sg, uint nsg, uint lane) {
  const uint r = tgp.y;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= p.D) return;
  float tot[ROWS];
  for (int i = 0; i < ROWS; ++i) tot[i] = 0.0f;
  for (uint j = 0; j < p.k; ++j) {
    const uint pair = r * p.k + j;
    const uint e = min(idx[pair], p.E - 1);
    const A4Mat W = a4_mat<SLOT, BITS, GS>(d0, d1, d2, e, p.D, p.I, L, 2);
    float a[ROWS];
    if (BITS == 2) a4_q20_rows_dx<ROWS>(W.w, (const device half*)W.s, h + ulong(pair) * p.I, p.D, p.I, row0, lane, a);
    else if (BITS == 4) qmv2_rows_dx<4, 16, ROWS, GS>(W.w, W.s, W.b, h + ulong(pair) * p.I, p.D, p.I, row0, lane, a);
    else a3_rows_dx<GS, ROWS>(W.w, W.s, W.b, h + ulong(pair) * p.I, p.D, p.I, row0, lane, a);
    const float ws = wts[pair];
    for (int i = 0; i < ROWS; ++i) tot[i] += ws * a[i];
  }
  if (lane == 0)
    for (int i = 0; i < ROWS; ++i)
      if (row0 + i < p.D) y[ulong(r) * p.D + row0 + i] = tot[i];
}

#define NI_A4_DN_ROWS(NAME, SLOT, BITS, GS)                                                                           \
  [[kernel]] void NAME(const device uchar* d0 [[buffer(0)]], const device uchar* d1 [[buffer(1)]],                 \
                       const device uchar* d2 [[buffer(2)]], const device float* h [[buffer(3)]],                  \
                       const device uint* idx [[buffer(4)]], const device float* wts [[buffer(5)]],                \
                       device float* y [[buffer(6)]], constant A4Params& p [[buffer(7)]],                          \
                       constant A4Slot& L [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],               \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    a4_dn_rows_impl<SLOT, BITS, GS, 2>(d0, d1, d2, h, idx, wts, y, p, L, tgp, sg, nsg, lane);                         \
  }
NI_A4_ALL(NI_A4_DN_ROWS, a4_dn_rows)

// ---- prefill: grouped GEMMs on moe_bucket3's tiles ----

// fp32 rows to half: n4 float4s
[[kernel]] void a4_x_half(const device float4* x [[buffer(0)]], device half4* xh [[buffer(1)]],
                          constant uint& n4 [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n4) xh[i] = half4(x[i]);
}

constant constexpr int A4_BK = 32;
constant constexpr int A4_LD = A4_BK + 8;  // threadgroup row stride in halfs (80 B: conflict-free 8x8 loads)

// acc[p][f] += A_f * B_p^T over one K step for the first FL row fragments (qmm_moe.metal's q2_mma)
template <int FL, int BN, int NP, int FM>
NI_INLINE void a4_mma(thread simdgroup_float8x8 (&acc)[NP][FM][BN / 16], threadgroup const half* xs,
                      threadgroup const half* ws, uint sgm, uint sgn) {
  constexpr int FN = BN / 16;
  for (int kk = 0; kk < A4_BK; kk += 8) {
    simdgroup_half8x8 a[FL], b[NP][FN];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FL; ++f) simdgroup_load(a[f], xs + (2 * f + sgm) * 8 * A4_LD + kk, A4_LD);
    _Pragma("clang loop unroll(full)")
    for (int q = 0; q < NP; ++q)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < FN; ++j)
        simdgroup_load(b[q][j], ws + (q * BN + sgn * (BN / 2) + j * 8) * A4_LD + kk, A4_LD, ulong2(0, 0), true);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FL; ++f)
      _Pragma("clang loop unroll(full)")
      for (int q = 0; q < NP; ++q)
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[q][f][j], a[f], b[q][j], acc[q][f][j]);
  }
}

// acc[q] += X[rows, K] * W_q[n0 .. n0 + BN, K]^T for NP matrices sharing X (4 simdgroups, 2x2), qmm_moe.metal's
// q2_tile: X row r of the tile is xrows[m0 + r] with GATHER, else m0 + r. GS >= A4_BK: a K step is in one group.
template <int BITS, int GS, int BM, int BN, int NP, bool GATHER>
NI_INLINE void a4_tile(const device half* X, const device uint* xrows, uint x_ld, uint m0, uint rows,
                       thread const A4Mat* M, uint n0, uint K, thread simdgroup_float8x8 (&acc)[NP][BM / 16][BN / 16],
                       threadgroup half* xs, threadgroup half* ws, uint tid, uint sgm, uint sgn, int fm_live) {
  constexpr int FM = BM / 16;
  constexpr int XL = (BM * 4 + 127) / 128;
  constexpr int WL = (BN * 4 + 127) / 128;
  _Pragma("clang loop unroll(full)")
  for (int q = 0; q < NP; ++q)
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FM; ++f)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < BN / 16; ++j) acc[q][f][j] = simdgroup_float8x8(0);
  const device uint4* X4 = (const device uint4*)X;
  uint xoff[XL];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < XL; ++j) {
    const uint i = tid + j * 128, r = i / 4;
    xoff[j] = i < BM * 4 && r < rows ? (GATHER ? xrows[m0 + r] : m0 + r) * (x_ld / 8) + i % 4 : 0;
  }
  const uint wpr = K * BITS / 32, gpr = K / GS;
  for (uint k0 = 0; k0 < K; k0 += A4_BK) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < XL; ++j) {
      const uint i = tid + j * 128, r = i / 4;
      if (i < BM * 4) *(threadgroup uint4*)(xs + r * A4_LD + (i % 4) * 8) = r < rows ? X4[xoff[j] + k0 / 8] : uint4(0);
    }
    const uint g = k0 / GS;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < WL; ++j) {
      const uint i = tid + j * 128, n = n0 + i / 4;
      if (i < BN * 4) {
        _Pragma("clang loop unroll(full)")
        for (int q = 0; q < NP; ++q) {
          const ulong group = ulong(n) * gpr + g;
          const float sc = BITS == 2 ? float(((const device half*)M[q].s)[group]) : float(M[q].s[group]);
          const float bi = BITS == 2 ? 0.0f : float(M[q].b[group]);
          half4 lo, hi;
          if (BITS == 2) {
            // Each lane decodes 8 values from one half of a 16-value word.
            const uint32_t word = M[q].w[ulong(n) * wpr + k0 / 16 + (i % 4) / 2] >> (16 * (i % 2));
            for (int c = 0; c < 4; ++c) {
              lo[c] = half(sc * float(int((word >> (2 * c)) & 3u) - 1));
              hi[c] = half(sc * float(int((word >> (2 * c + 8)) & 3u) - 1));
            }
          } else if (BITS == 4) {
            const uint32_t word = M[q].w[ulong(n) * wpr + k0 / 8 + i % 4];
            for (int c = 0; c < 4; ++c) {
              lo[c] = half(sc * float((word >> (4 * c)) & 15u) + bi);
              hi[c] = half(sc * float((word >> (4 * c + 16)) & 15u) + bi);
            }
          } else {  // 8 values = 3 bytes at byte 3 (k0 / 8 + i % 4) of the row
            const device uchar* bp = (const device uchar*)(M[q].w + ulong(n) * wpr) + (k0 / 8 + i % 4) * 3;
            const uint v = uint(bp[0]) | (uint(bp[1]) << 8) | (uint(bp[2]) << 16);
            for (int c = 0; c < 4; ++c) {
              lo[c] = half(sc * float((v >> (3 * c)) & 7u) + bi);
              hi[c] = half(sc * float((v >> (3 * c + 12)) & 7u) + bi);
            }
          }
          *(threadgroup uint4*)(ws + (q * BN + i / 4) * A4_LD + (i % 4) * 8) = uint4(as_type<uint2>(lo), as_type<uint2>(hi));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (FM >= 4 && fm_live == 4) a4_mma<(FM < 4 ? FM : 4), BN, NP>(acc, xs, ws, sgm, sgn);
    else if (FM >= 3 && fm_live == 3) a4_mma<(FM < 3 ? FM : 3), BN, NP>(acc, xs, ws, sgm, sgn);
    else if (FM >= 2 && fm_live == 2) a4_mma<(FM < 2 ? FM : 2), BN, NP>(acc, xs, ws, sgm, sgn);
    else if (fm_live == 1) a4_mma<1, BN, NP>(acc, xs, ws, sgm, sgn);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <int BM>
NI_INLINE int a4_live(uint rows, uint sgm) {
  const int nf = int(rows + 7) / 8;
  return clamp((nf - int(sgm) + 1) / 2, 0, BM / 16);
}

// an 8x8 accumulator's lane position: row, first of two adjacent columns
NI_INLINE uint2 a4_lane_rc(uint lane) {
  const uint q = lane / 4;
  return uint2((q & 4) + (lane / 2) % 4, (q & 2) * 2 + (lane % 2) * 2);
}

NI_INLINE float2 a4_elems(thread simdgroup_float8x8& m) {
  return float2(m.thread_elements()[0], m.thread_elements()[1]);
}

// H[m, n] (half, sorted rows) = swiglu(X W_gate^T, X W_up^T) for each tile's rows; X: half [R, D] gathered through
// xrows. grid (I / BN, max tiles), 128 threads; tiles[3 i] = (expert, begin, end) (moe_bucket3)
template <bool SLOT, int BITS, int GS, int BM, int BN>
NI_INLINE void a4_gu_mm_impl(const device half* X, const device uchar* g0, const device uchar* g1,
                             const device uchar* g2, const device uchar* u0, const device uchar* u1,
                             const device uchar* u2, device half* H, constant A4Params& p, constant A4Slot& L,
                             const device uint* tiles, const device uint* ntiles, const device uint* xrows, uint2 tg,
                             uint tid, uint sg, uint lane, threadgroup half* xs, threadgroup half* ws) {
  if (tg.y >= ntiles[0]) return;
  const uint e = min(tiles[tg.y * 3 + 0], p.E - 1), m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;
  const A4Mat M[2] = {a4_mat<SLOT, BITS, GS>(g0, g1, g2, e, p.I, p.D, L, 0),
                      a4_mat<SLOT, BITS, GS>(u0, u1, u2, e, p.I, p.D, L, 1)};
  const uint sgm = sg / 2, sgn = sg % 2, n0 = tg.x * BN;
  const int live = a4_live<BM>(rows, sgm);
  simdgroup_float8x8 acc[2][BM / 16][BN / 16];
  a4_tile<BITS, GS, BM, BN, 2, true>(X, xrows, p.D, m0, rows, M, n0, p.D, acc, xs, ws, tid, sgm, sgn, live);
  const uint2 rc = a4_lane_rc(lane);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < BM / 16; ++f) {
    const uint m = (2 * f + sgm) * 8 + rc.x;
    if (f < live && m < rows) {
      device half* hp = H + ulong(m0 + m) * p.I + n0 + sgn * (BN / 2) + rc.y;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < BN / 16; ++j) {
        const float2 gv = a4_elems(acc[0][f][j]), uv = a4_elems(acc[1][f][j]);
        *(device half2*)(hp + j * 8) = half2(half(a4_swiglu(gv.x, uv.x, p.limit)), half(a4_swiglu(gv.y, uv.y, p.limit)));
      }
    }
  }
}

#define NI_A4_GU_MM(NAME, SLOT, BITS, GS, BM, BN)                                                                     \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uchar* g0 [[buffer(1)]],                   \
                       const device uchar* g1 [[buffer(2)]], const device uchar* g2 [[buffer(3)]],                 \
                       const device uchar* u0 [[buffer(4)]], const device uchar* u1 [[buffer(5)]],                 \
                       const device uchar* u2 [[buffer(6)]], device half* H [[buffer(7)]],                         \
                       constant A4Params& p [[buffer(8)]], constant A4Slot& L [[buffer(9)]],                       \
                       const device uint* tiles [[buffer(10)]], const device uint* ntiles [[buffer(11)]],          \
                       const device uint* xrows [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],         \
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],       \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    threadgroup uint4 xs4[BM * A4_LD / 8];                                                                         \
    threadgroup uint4 ws4[2 * BN * A4_LD / 8];                                                                     \
    a4_gu_mm_impl<SLOT, BITS, GS, BM, BN>(X, g0, g1, g2, u0, u1, u2, H, p, L, tiles, ntiles, xrows, tg, tid, sg, lane, \
                                (threadgroup half*)xs4, (threadgroup half*)ws4);                                   \
  }
#define NI_A4_GU_MM32(NAME, SLOT, BITS, GS) NI_A4_GU_MM(NAME, SLOT, BITS, GS, 32, 32)
NI_A4_ALL(NI_A4_GU_MM32, a4_gu_mm)

// Y[yrows[m], n] (fp32, by pair) = H[m, :] W_down^T for each tile's sorted rows m. grid (D / BN, max tiles), 128 threads
template <bool SLOT, int BITS, int GS, int BM, int BN>
NI_INLINE void a4_dn_mm_impl(const device half* H, const device uchar* d0, const device uchar* d1,
                             const device uchar* d2, device float* Y, constant A4Params& p, constant A4Slot& L,
                             const device uint* tiles, const device uint* ntiles, const device uint* yrows, uint2 tg,
                             uint tid, uint sg, uint lane, threadgroup half* xs, threadgroup half* ws) {
  if (tg.y >= ntiles[0]) return;
  const uint e = min(tiles[tg.y * 3 + 0], p.E - 1), m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;
  const A4Mat M[1] = {a4_mat<SLOT, BITS, GS>(d0, d1, d2, e, p.D, p.I, L, 2)};
  const uint sgm = sg / 2, sgn = sg % 2, n0 = tg.x * BN;
  const int live = a4_live<BM>(rows, sgm);
  simdgroup_float8x8 acc[1][BM / 16][BN / 16];
  a4_tile<BITS, GS, BM, BN, 1, false>(H, nullptr, p.I, m0, rows, M, n0, p.I, acc, xs, ws, tid, sgm, sgn, live);
  const uint2 rc = a4_lane_rc(lane);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < BM / 16; ++f) {
    const uint m = (2 * f + sgm) * 8 + rc.x;
    if (f < live && m < rows) {
      device float* yp = Y + ulong(yrows[m0 + m]) * p.D + n0 + sgn * (BN / 2) + rc.y;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < BN / 16; ++j) *(device float2*)(yp + j * 8) = a4_elems(acc[0][f][j]);
    }
  }
}

#define NI_A4_DN_MM(NAME, SLOT, BITS, GS, BM, BN)                                                                     \
  [[kernel]] void NAME(const device half* H [[buffer(0)]], const device uchar* d0 [[buffer(1)]],                   \
                       const device uchar* d1 [[buffer(2)]], const device uchar* d2 [[buffer(3)]],                 \
                       device float* Y [[buffer(4)]], constant A4Params& p [[buffer(5)]],                          \
                       constant A4Slot& L [[buffer(6)]], const device uint* tiles [[buffer(7)]],                   \
                       const device uint* ntiles [[buffer(8)]], const device uint* yrows [[buffer(9)]],            \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],        \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {      \
    threadgroup uint4 xs4[BM * A4_LD / 8];                                                                         \
    threadgroup uint4 ws4[BN * A4_LD / 8];                                                                         \
    a4_dn_mm_impl<SLOT, BITS, GS, BM, BN>(H, d0, d1, d2, Y, p, L, tiles, ntiles, yrows, tg, tid, sg, lane,            \
                                (threadgroup half*)xs4, (threadgroup half*)ws4);                                   \
  }
#define NI_A4_DN_MM64(NAME, SLOT, BITS, GS) NI_A4_DN_MM(NAME, SLOT, BITS, GS, 32, 64)
NI_A4_ALL(NI_A4_DN_MM64, a4_dn_mm)

// y[r, :] = sum_j wts[r, j] * Y[r * k + j, :] in order j = 0 .. k - 1. grid (D / 4 / 64, R) x 64 threads
[[kernel]] void a4_combine(const device float4* Y [[buffer(0)]], const device float* wts [[buffer(1)]],
                           device float4* y [[buffer(2)]], constant A4Params& p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const uint d4 = p.D / 4, c = gid.x, r = gid.y;
  if (c >= d4) return;
  float4 acc = float4(0.0f);
  for (uint j = 0; j < p.k; ++j) acc += wts[r * p.k + j] * Y[ulong(r * p.k + j) * d4 + c];
  y[ulong(r) * d4 + c] = acc;
}

#endif  // NI_WF_INST
