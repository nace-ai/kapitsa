#include "common.h"

// ---- Hy4-flash's DFly2 drafter (src/model/hy4_dfly.h) --------------------------------------------------------------
// The drafter's dense bf16 linears over up to 6 rows, the target's affine-quantized embedding rows and LM head (MLX
// affine, group 64, 4 or 8 bits, codes LSB first), the head's argmax and two elementwise steps. Its attention is
// hy4_ops' enc_attn_latent; its norms, adds and SwiGLU are hy4_ops' too. Static threadgroup memory <= 4 KB each.
#ifndef NI_WF_INST

struct DflyMvP {
  uint N, K, R;           // y [R, N] = x [R, K] . W[N, K]^T (+ resid)
  uint x_ld, y_ld, r_ld;  // row strides in floats
  uint add;               // 1: y = W x + resid (resid may alias y)
};

NI_INLINE float dfly_bf_lo(uint v) { return as_type<float>(v << 16); }
NI_INLINE float dfly_bf_hi(uint v) { return as_type<float>(v & 0xFFFF0000u); }

// One simdgroup: ROWS output rows for NT input rows (rows past R re-read row R - 1 and are not written); lanes step K
// by 8 bf16 (one 16-byte load a row). W rows 16-byte aligned (K % 8 == 0), x rows 16-byte aligned (x_ld % 4 == 0).
template <int NT, int ROWS>
NI_INLINE void dfly_mv_impl(const device uint* W, const device float* x, const device float* resid, device float* y,
                            constant DflyMvP& p, uint tg, uint sg, uint nsg, uint lane) {
  const uint row0 = (tg * nsg + sg) * ROWS;
  if (row0 >= p.N) return;
  float acc[ROWS][NT];
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  const uint wpr = p.K / 2;  // uints a row
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, p.N - 1);
  for (uint k0 = lane * 8; k0 < p.K; k0 += 256) {
    float4 xa[NT], xb[NT];
    for (int t = 0; t < NT; ++t) {
      const uint tt = min(uint(t), p.R - 1);
      const device float4* xp = (const device float4*)(x + ulong(tt) * p.x_ld + k0);
      xa[t] = xp[0];
      xb[t] = xp[1];
    }
    for (int r = 0; r < ROWS; ++r) {
      const uint4 v = *(const device uint4*)(W + ulong(rows[r]) * wpr + k0 / 2);
      const float4 wa = float4(dfly_bf_lo(v.x), dfly_bf_hi(v.x), dfly_bf_lo(v.y), dfly_bf_hi(v.y));
      const float4 wb = float4(dfly_bf_lo(v.z), dfly_bf_hi(v.z), dfly_bf_lo(v.w), dfly_bf_hi(v.w));
      for (int t = 0; t < NT; ++t) acc[r][t] += dot(wa, xa[t]) + dot(wb, xb[t]);
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = simd_sum(acc[r][t]);
  if (lane < uint(NT)) {
    const uint t = lane;
    if (t < p.R)
      for (int r = 0; r < ROWS; ++r) {
        const uint n = row0 + r;
        if (n >= p.N) break;
        float v = 0.0f;
        for (int q = 0; q < NT; ++q) v = q == int(t) ? acc[r][q] : v;
        if (p.add) v += resid[ulong(t) * p.r_ld + n];
        y[ulong(t) * p.y_ld + n] = v;
      }
  }
}

#define NI_DFLY_MV(NAME, NT, ROWS)                                                                                 \
  [[kernel]] void NAME(const device uint* W [[buffer(0)]], const device float* x [[buffer(1)]],                    \
                       const device float* resid [[buffer(2)]], device float* y [[buffer(3)]],                     \
                       constant DflyMvP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],                \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    dfly_mv_impl<NT, ROWS>(W, x, resid, y, p, tg, sg, nsg, lane);                                                  \
  }
NI_DFLY_MV(dfly_mv_t1_r4, 1, 4)
NI_DFLY_MV(dfly_mv_t6_r2, 6, 2)
NI_DFLY_MV(dfly_mv_t6_r4, 6, 4)

// One row through an affine-quantized matrix (the target's LM head): y [N] = W x, x fp32 [K] (16-byte aligned).
// 8 simdgroups x 4 rows a threadgroup.
struct DflyQmvP {
  uint N, K;
};
#define NI_DFLY_QMV(NAME, BITS)                                                                                     \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],              \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],                 \
                       device float* y [[buffer(4)]], constant DflyQmvP& p [[buffer(5)]],                         \
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {        \
    const uint row0 = (tg * nsg + sg) * 4;                                                                          \
    if (row0 >= p.N) return;                                                                                        \
    float acc[4];                                                                                                   \
    qmv2_rows_dx<BITS, 16, 4, 64>(w, s, b, x, p.N, p.K, row0, lane, acc);                                           \
    if (lane == 0)                                                                                                  \
      for (int r = 0; r < 4; ++r)                                                                                   \
        if (row0 + r < p.N) y[row0 + r] = acc[r];                                                                   \
  }
NI_DFLY_QMV(dfly_qmv_b8, 8)
NI_DFLY_QMV(dfly_qmv_b4, 4)

// Embedding rows of an affine-quantized table: y[t, :] = dequant(row tok_t), tok_t = ids[t] for t < nids, else fill;
// ids past V are clamped. grid (K / (32 / BITS) / 64 words, R) x 64.
struct DflyEmbP {
  uint K, V, R, nids, fill;
};
template <int BITS>
NI_INLINE void dfly_embed_impl(const device uint32_t* w, const device bfloat* s, const device bfloat* b,
                               const device uint* ids, device float* y, constant DflyEmbP& p, uint2 gid) {
  constexpr uint PW = 32 / BITS, MASK = (1u << BITS) - 1u;
  const uint wi = gid.x, t = gid.y;
  if (wi >= p.K / PW || t >= p.R) return;
  const uint tok = min(t < p.nids ? ids[t] : p.fill, p.V - 1);
  const uint word = w[ulong(tok) * (p.K / PW) + wi];
  const uint g = wi * PW / 64;
  const float sc = float(s[ulong(tok) * (p.K / 64) + g]), bi = float(b[ulong(tok) * (p.K / 64) + g]);
  device float* yp = y + ulong(t) * p.K + wi * PW;
  for (uint j = 0; j < PW; ++j) yp[j] = sc * float((word >> (BITS * j)) & MASK) + bi;
}
#define NI_DFLY_EMBED(NAME, BITS)                                                                                   \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],              \
                       const device bfloat* b [[buffer(2)]], const device uint* ids [[buffer(3)]],                \
                       device float* y [[buffer(4)]], constant DflyEmbP& p [[buffer(5)]],                         \
                       uint2 gid [[thread_position_in_grid]]) {                                                   \
    dfly_embed_impl<BITS>(w, s, b, ids, y, p, gid);                                                                 \
  }
NI_DFLY_EMBED(dfly_embed_b8, 8)
NI_DFLY_EMBED(dfly_embed_b4, 4)

// out[0] = argmax over x[0 .. n) (the first index of the largest value; NaNs never win). One threadgroup of 1024.
[[kernel]] void dfly_argmax(const device float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  float v = -INFINITY;
  uint i = 0xFFFFFFFFu;
  for (uint j = tid; j < n; j += 1024) {
    const float xj = x[j];
    if (xj > v || (i == 0xFFFFFFFFu && !isnan(xj))) {  // strictly larger: j only grows, so ties keep the first
      v = xj;
      i = j;
    }
  }
  for (uint o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(v, o);
    const uint oi = simd_shuffle_xor(i, o);
    if (oi != 0xFFFFFFFFu && (i == 0xFFFFFFFFu || ov > v || (ov == v && oi < i))) {
      v = ov;
      i = oi;
    }
  }
  if (lane == 0) {
    bv[sg] = v;
    bi[sg] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    v = bv[lane];
    i = bi[lane];
    for (uint o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_xor(v, o);
      const uint oi = simd_shuffle_xor(i, o);
      if (oi != 0xFFFFFFFFu && (i == 0xFFFFFFFFu || ov > v || (ov == v && oi < i))) {
        v = ov;
        i = oi;
      }
    }
    if (lane == 0) out[0] = i == 0xFFFFFFFFu ? 0u : i;
  }
}

NI_INLINE float dfly_bf16_rne(float x) {
  uint b = as_type<uint>(x);
  if ((b & 0x7F800000u) == 0x7F800000u) return x;  // inf / nan
  b += 0x7FFFu + ((b >> 16) & 1u);
  return as_type<float>(b & 0xFFFF0000u);
}

// y[i] = bf16(x[i]) (nearest even), n elements
[[kernel]] void dfly_round_bf16(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = dfly_bf16_rne(x[i]);
}

// x[i] += g[i] * f[i] (g bf16: mtp_fusion.gate), n elements
[[kernel]] void dfly_gate_add(device float* x [[buffer(0)]], const device bfloat* g [[buffer(1)]],
                              const device float* f [[buffer(2)]], constant uint& n [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i < n) x[i] += float(g[i]) * f[i];
}

#endif  // NI_WF_INST
