#include "common.h"

// ---- qmvf: small-batch quantized matvec for decode and MTP verify (R <= 4 token rows) ----
// y_s[t, n] = inv_t * sum_k W_s[n, k] * x[t, k] * nw[k]  (NORM: RMSNorm of x folded in; else inv_t = 1, nw = 1)
//             (+ y_s[t, n] when add[s])
// for token rows t < R and up to 4 matrices s that share K and the input (GDN in-proj qkv|z|b|a, attention
// q|k|v): one dispatch for what took a norm pass and one matvec per matrix.
// * The norm is folded: inv_t = rsqrt(mean(x_t^2) + eps) scales the finished dot product, and each simdgroup
//   gets sum(x_t^2) from the x values it loads anyway (its lanes walk all of K), so x is read once and there is
//   no threadgroup staging or barrier. The sums are reassociated against the staged kernels (rounding only).
// * A thread holds NT token rows, so each weight word is dequantized once for all of them (the qmvrc1 kernels
//   dequantize it once per row). Grid (ceil(R / NT), blocks): the token-row groups of one weight block are
//   adjacent and share the block through the cache.
// * Weights: MLX affine, BITS = 4 or 8, group GS = 64 (or 32: the 4-bit group-32 LM heads of GGUF imports, on the M1);
//   each lane reads 16 values per row per step (MLX's qmv layout: x pre-scaled by 2^-BITS*j so a value costs AND +
//   convert + one FMA per token row).
// qmvf_gu_*: segments 0 / 1 are the gate / up of one SwiGLU MLP: y_0[t, n] = silu(gate) * up (one dispatch for
// norm, gate, up and the activation).
struct QmvfParams {
  uint K;
  uint R;
  uint x_ld;
  uint nseg;
  float eps;
  uint N[4];
  uint y_ld[4];
  uint add[4];
  uint tg_begin[5];
  uint write_xn;  // NORM: token-row group 0 of segment 0 also writes the normalized rows to xn [R, K]
};

// acc[m][r][t] = partial dot products of matrix m (NM = 1 or 2 matrices with the same rows), weight row
// row0 + r, token row t; ss[t] = this lane's part of sum(x_t^2) (NORM).
template <int BITS, int NT, int ROWS, int NM, bool NORM, int GS = 64>
NI_INLINE void qmvf_core(const device uint32_t* w0, const device bfloat* s0, const device bfloat* b0,
                         const device uint32_t* w1, const device bfloat* s1, const device bfloat* b1,
                         const device float* x, uint x_ld, uint t0, uint R, uint K, uint N, const device bfloat* nw,
                         uint row0, uint lane, thread float (&acc)[NM][ROWS][NT], thread float (&ss)[NT]) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr int VPL = 16;
  constexpr int WORDS = VPL / PER_WORD;
  constexpr uint MASK = (1u << BITS) - 1u;
  const uint wpr = K / PER_WORD, gpr = K / GS;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, N - 1);
  for (int m = 0; m < NM; ++m)
    for (int r = 0; r < ROWS; ++r)
      for (int t = 0; t < NT; ++t) acc[m][r][t] = 0.0f;
  for (int t = 0; t < NT; ++t) ss[t] = 0.0f;
  uint xo[NT];
  for (int t = 0; t < NT; ++t) xo[t] = min(t0 + uint(t), R - 1) * x_ld;
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL];
    float xs[NT];
    float4 nv[VPL / 4];
    if (NORM)
      for (int i = 0; i < VPL / 4; ++i) {
        const uint k = k0 + i * 4;
        nv[i] = float4(float(nw[k]), float(nw[k + 1]), float(nw[k + 2]), float(nw[k + 3]));
      }
    for (int t = 0; t < NT; ++t) {
      xs[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = ((const device float4*)(x + xo[t] + k0))[i];
        if (NORM) {
          ss[t] += dot(v, v);
          v *= nv[i];
        }
        xs[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e)
          xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
      }
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int m = 0; m < NM; ++m) {
      const device uint32_t* W = m == 0 ? w0 : w1;
      const device bfloat* S = m == 0 ? s0 : s1;
      const device bfloat* B = m == 0 ? b0 : b1;
      for (int r = 0; r < ROWS; ++r) {
        const device uint32_t* wp = W + rows[r] * wpr + wo;
        uint32_t wv[WORDS];
        if (WORDS == 4) {
          uint4 v = *(const device uint4*)wp;
          wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
        } else {
          uint2 v = *(const device uint2*)wp;
          wv[0] = v.x; wv[1] = v.y;
        }
        float d[NT];
        for (int t = 0; t < NT; ++t) d[t] = 0.0f;
        for (int i = 0; i < WORDS; ++i)
          for (int j = 0; j < PER_WORD; ++j) {
            const float wf = float(wv[i] & (MASK << (BITS * j)));
            for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
          }
        const float sc = float(S[rows[r] * gpr + g]), bi = float(B[rows[r] * gpr + g]);
        for (int t = 0; t < NT; ++t) acc[m][r][t] += sc * d[t] + bi * xs[t];
      }
    }
  }
}

#define NI_QMVF_ARGS                                                                                      \
  const device uint32_t *w0 [[buffer(0)]], const device bfloat *s0 [[buffer(1)]],                        \
      const device bfloat *b0 [[buffer(2)]], device float *y0 [[buffer(3)]],                              \
      const device uint32_t *w1 [[buffer(4)]], const device bfloat *s1 [[buffer(5)]],                    \
      const device bfloat *b1 [[buffer(6)]], device float *y1 [[buffer(7)]],                              \
      const device uint32_t *w2 [[buffer(8)]], const device bfloat *s2 [[buffer(9)]],                    \
      const device bfloat *b2 [[buffer(10)]], device float *y2 [[buffer(11)]],                            \
      const device uint32_t *w3 [[buffer(12)]], const device bfloat *s3 [[buffer(13)]],                  \
      const device bfloat *b3 [[buffer(14)]], device float *y3 [[buffer(15)]],                            \
      const device float *x [[buffer(16)]], constant QmvfParams &p [[buffer(17)]],                       \
      const device bfloat *nw [[buffer(18)]], device float *xn [[buffer(19)]],                           \
      uint2 tgp [[threadgroup_position_in_grid]],                                                         \
      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],               \
      uint lane [[thread_index_in_simdgroup]]

template <int BITS, int NT, int ROWS, bool NORM, int GS = 64>
NI_INLINE void qmvf_multi(const device uint32_t* w0, const device bfloat* s0, const device bfloat* b0, device float* y0,
                          const device uint32_t* w1, const device bfloat* s1, const device bfloat* b1, device float* y1,
                          const device uint32_t* w2, const device bfloat* s2, const device bfloat* b2, device float* y2,
                          const device uint32_t* w3, const device bfloat* s3, const device bfloat* b3, device float* y3,
                          const device float* x, constant QmvfParams& p, const device bfloat* nw, device float* xn,
                          uint2 tgp, uint sg, uint nsg, uint lane) {
  const uint t0 = tgp.x * NT;
  uint seg = 0;
  for (uint i = 1; i < p.nseg; ++i)
    if (tgp.y >= p.tg_begin[i]) seg = i;
  const uint N = p.N[seg];
  const uint row0 = ((tgp.y - p.tg_begin[seg]) * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;
  const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;
  const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;
  device float* Y = seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3;
  float acc[1][ROWS][NT], ss[NT];
  qmvf_core<BITS, NT, ROWS, 1, NORM, GS>(W, S, B, W, S, B, x, p.x_ld, t0, p.R, p.K, N, nw, row0, lane, acc, ss);
  const uint yld = p.y_ld[seg];
  const bool add = p.add[seg] != 0;
  for (int t = 0; t < NT; ++t) {
    float inv = 1.0f;
    if (NORM) inv = metal::precise::rsqrt(simd_sum(ss[t]) / float(p.K) + p.eps);
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[0][r][t]);
      if (NORM) v *= inv;
      const uint row = row0 + r;
      if (lane == 0 && row < N && t0 + uint(t) < p.R) {
        device float* dst = Y + (t0 + t) * yld + row;
        *dst = add ? *dst + v : v;
      }
    }
    if (NORM && p.write_xn && tgp.y == 0 && sg == 0 && t0 + uint(t) < p.R)
      for (uint k = lane; k < p.K; k += 32) xn[(t0 + t) * p.K + k] = x[(t0 + t) * p.x_ld + k] * inv * float(nw[k]);
  }
}

template <int BITS, int NT, int ROWS, bool NORM>
NI_INLINE void qmvf_gu(const device uint32_t* w0, const device bfloat* s0, const device bfloat* b0, device float* y0,
                       const device uint32_t* w1, const device bfloat* s1, const device bfloat* b1,
                       const device float* x, constant QmvfParams& p, const device bfloat* nw, uint2 tgp, uint sg,
                       uint nsg, uint lane) {
  const uint t0 = tgp.x * NT;
  const uint N = p.N[0];
  const uint row0 = (tgp.y * nsg + sg) * ROWS;
  if (row0 >= N) return;
  float acc[2][ROWS][NT], ss[NT];
  qmvf_core<BITS, NT, ROWS, 2, NORM>(w0, s0, b0, w1, s1, b1, x, p.x_ld, t0, p.R, p.K, N, nw, row0, lane, acc, ss);
  const uint yld = p.y_ld[0];
  for (int t = 0; t < NT; ++t) {
    float inv = 1.0f;
    if (NORM) inv = metal::precise::rsqrt(simd_sum(ss[t]) / float(p.K) + p.eps);
    for (int r = 0; r < ROWS; ++r) {
      float g = simd_sum(acc[0][r][t]), u = simd_sum(acc[1][r][t]);
      if (NORM) {
        g *= inv;
        u *= inv;
      }
      const uint row = row0 + r;
      if (lane == 0 && row < N && t0 + uint(t) < p.R) y0[(t0 + t) * yld + row] = silu(g) * u;
    }
  }
}

#define NI_QMVF(NAME, BITS, NT, ROWS, NORM)                                                                \
  [[kernel]] void NAME(NI_QMVF_ARGS) {                                                                    \
    qmvf_multi<BITS, NT, ROWS, NORM>(w0, s0, b0, y0, w1, s1, b1, y1, w2, s2, b2, y2, w3, s3, b3, y3, x, p, nw, xn, \
                                     tgp, sg, nsg, lane);                                                 \
  }
#define NI_QMVF_GU(NAME, BITS, NT, ROWS, NORM)                                                             \
  [[kernel]] void NAME(NI_QMVF_ARGS) {                                                                    \
    qmvf_gu<BITS, NT, ROWS, NORM>(w0, s0, b0, y0, w1, s1, b1, x, p, nw, tgp, sg, nsg, lane);              \
  }

// name: qmvf[_gu]_b{bits}_t{token rows per thread}_r{weight rows per simdgroup}[_norm]. Tried and dropped (bench_qmvf,
// 4B shapes, M4 Pro): token rows sharing a thread's dequantization (t2-t4: 1.06-1.12x the time of t1 grids at 2 rows),
// next-step prefetch of the weight words (spills at 96 GPRs), the loop interchanged to consume x one float4 at a time
// against all rows (fewer registers, 2-6% slower), 2 or 8 weight rows per simdgroup, 4-8 simdgroups.
#define NI_QMVF_SET(BITS, NT, ROWS)                                                                       \
  NI_QMVF(qmvf_b##BITS##_t##NT##_r##ROWS, BITS, NT, ROWS, false)                                          \
  NI_QMVF(qmvf_b##BITS##_t##NT##_r##ROWS##_norm, BITS, NT, ROWS, true)                                    \
  NI_QMVF_GU(qmvf_gu_b##BITS##_t##NT##_r##ROWS, BITS, NT, ROWS, false)                                    \
  NI_QMVF_GU(qmvf_gu_b##BITS##_t##NT##_r##ROWS##_norm, BITS, NT, ROWS, true)

NI_QMVF_SET(4, 1, 4)
NI_QMVF_SET(8, 1, 4)

#ifdef NI_WF_INST  // the weight-format library (Gpu::compile_formats)
// 4-bit group 32 (GGUF imports: Q4_K's codes as stored), on the M1 (Qwen35::enc_qmvf): Qwen3.6-35B-A3B's 248K-row LM
// head 2.6 instead of 3.1 ms (qmvg_f4g32_r4_norm) on the M1 Pro
#define NI_QMVF_G32(NAME, NT, ROWS, NORM)                                                                   \
  [[kernel]] void NAME(NI_QMVF_ARGS) {                                                                    \
    qmvf_multi<4, NT, ROWS, NORM, 32>(w0, s0, b0, y0, w1, s1, b1, y1, w2, s2, b2, y2, w3, s3, b3, y3, x, p, nw, xn, \
                                      tgp, sg, nsg, lane);                                                \
  }
NI_QMVF_G32(qmvf_b4g32_t1_r4, 1, 4, false)
NI_QMVF_G32(qmvf_b4g32_t1_r4_norm, 1, 4, true)

// 6-bit group 64 (GGUF imports: Q5_K's projections), in IQ models on the M1 (Qwen35::enc_qmvf). A lane's 16 values of a
// row are 3 words (value j at bits 6j of the row's stream): 14 of them lie inside one word and cost AND + convert + FMA
// against x pre-scaled by 2^-shift, as in qmvf_core; values 5 and 10 straddle two words and take one FMA per part
// (x scaled by 2^-30 and 4, 2^-28 and 16). Every product is exact as in qmvf_core (codes times powers of two).
template <int NT, int ROWS, bool NORM>
NI_INLINE void qmvf6_core(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                          const device float* x, uint x_ld, uint t0, uint R, uint K, uint N, const device bfloat* nw,
                          uint row0, uint lane, thread float (&acc)[ROWS][NT], thread float (&ss)[NT]) {
  constexpr int VPL = 16;
  // bit shift of value j inside its (first) word
  constexpr uint SH[16] = {0, 6, 12, 18, 24, 30, 4, 10, 16, 22, 28, 2, 8, 14, 20, 26};
  const uint wpr = K * 6 / 32, gpr = K / 64;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, N - 1);
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  for (int t = 0; t < NT; ++t) ss[t] = 0.0f;
  uint xo[NT];
  for (int t = 0; t < NT; ++t) xo[t] = min(t0 + uint(t), R - 1) * x_ld;
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL + 2];  // [16], [17]: the high parts of values 5 and 10
    float xs[NT];
    float4 nv[VPL / 4];
    if (NORM)
      for (int i = 0; i < VPL / 4; ++i) {
        const uint k = k0 + i * 4;
        nv[i] = float4(float(nw[k]), float(nw[k + 1]), float(nw[k + 2]), float(nw[k + 3]));
      }
    for (int t = 0; t < NT; ++t) {
      xs[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = ((const device float4*)(x + xo[t] + k0))[i];
        if (NORM) {
          ss[t] += dot(v, v);
          v *= nv[i];
        }
        xs[t] += v.x + v.y + v.z + v.w;
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << SH[i * 4 + e]));
      }
      xv[t][16] = xv[t][5] * 4294967296.0f;  // x5 * 2^-30 * 2^32: x5 * 4
      xv[t][17] = xv[t][10] * 4294967296.0f;  // x10 * 16
    }
    const uint g = k0 / 64, wo = k0 / 16 * 3;
    for (int r = 0; r < ROWS; ++r) {
      const uint3 w = uint3(*(const device packed_uint3*)(W + rows[r] * wpr + wo));
      float d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 16; ++j) {
        const uint wi = j < 6 ? w.x : j < 11 ? w.y : w.z;
        const uint m = j == 5 ? 0xC0000000u : j == 10 ? 0xF0000000u : 63u << SH[j];
        const float wf = float(wi & m);
        for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][j];
      }
      const float h5 = float(w.y & 0xFu), h10 = float(w.z & 0x3u);
      for (int t = 0; t < NT; ++t) d[t] += h5 * xv[t][16] + h10 * xv[t][17];
      const float sc = float(S[rows[r] * gpr + g]), bi = float(B[rows[r] * gpr + g]);
      for (int t = 0; t < NT; ++t) acc[r][t] += sc * d[t] + bi * xs[t];
    }
  }
}

template <int NT, int ROWS, bool NORM>
NI_INLINE void qmvf6_multi(const device uint32_t* w0, const device bfloat* s0, const device bfloat* b0, device float* y0,
                           const device uint32_t* w1, const device bfloat* s1, const device bfloat* b1, device float* y1,
                           const device uint32_t* w2, const device bfloat* s2, const device bfloat* b2, device float* y2,
                           const device uint32_t* w3, const device bfloat* s3, const device bfloat* b3, device float* y3,
                           const device float* x, constant QmvfParams& p, const device bfloat* nw, device float* xn,
                           uint2 tgp, uint sg, uint nsg, uint lane) {
  const uint t0 = tgp.x * NT;
  uint seg = 0;
  for (uint i = 1; i < p.nseg; ++i)
    if (tgp.y >= p.tg_begin[i]) seg = i;
  const uint N = p.N[seg];
  const uint row0 = ((tgp.y - p.tg_begin[seg]) * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;
  const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;
  const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;
  device float* Y = seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3;
  float acc[ROWS][NT], ss[NT];
  qmvf6_core<NT, ROWS, NORM>(W, S, B, x, p.x_ld, t0, p.R, p.K, N, nw, row0, lane, acc, ss);
  const uint yld = p.y_ld[seg];
  const bool add = p.add[seg] != 0;
  for (int t = 0; t < NT; ++t) {
    float inv = 1.0f;
    if (NORM) inv = metal::precise::rsqrt(simd_sum(ss[t]) / float(p.K) + p.eps);
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[r][t]);
      if (NORM) v *= inv;
      const uint row = row0 + r;
      if (lane == 0 && row < N && t0 + uint(t) < p.R) {
        device float* dst = Y + (t0 + t) * yld + row;
        *dst = add ? *dst + v : v;
      }
    }
    if (NORM && p.write_xn && tgp.y == 0 && sg == 0 && t0 + uint(t) < p.R)
      for (uint k = lane; k < p.K; k += 32) xn[(t0 + t) * p.K + k] = x[(t0 + t) * p.x_ld + k] * inv * float(nw[k]);
  }
}
#define NI_QMVF6(NAME, NT, ROWS, NORM)                                                                       \
  [[kernel]] void NAME(NI_QMVF_ARGS) {                                                                    \
    qmvf6_multi<NT, ROWS, NORM>(w0, s0, b0, y0, w1, s1, b1, y1, w2, s2, b2, y2, w3, s3, b3, y3, x, p, nw, xn, tgp, sg, \
                                nsg, lane);                                                               \
  }
NI_QMVF6(qmvf_b6_t1_r4, 1, 4, false)
NI_QMVF6(qmvf_b6_t1_r4_norm, 1, 4, true)
#endif
