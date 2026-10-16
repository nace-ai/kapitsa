#include "common.h"

// Quantized matrix x vector(s) for MLX affine weights.
//   W: [N, K] packed little-endian in uint32 words (value j of a word = (w >> BITS*j) & mask)
//   scales/biases: bf16 [N, K/GS];  w_real = scale * q + bias
// y[t, n] = sum_k W[n, k] * x[t, k]   for t < NT (NT > 1 = several tokens share one weight read,
// used to verify speculative drafts).
//
// Each simdgroup owns ROWS consecutive rows; each lane reads VPL values per row per step.
// Shift-free dequant: the value at slot j of a word is (w & (mask << BITS*j)) == q * 2^(BITS*j);
// x is pre-scaled by 2^-(BITS*j), so each value costs one AND + convert + FMA.
// NORM=true fuses an RMSNorm of x (weight `nw`, bf16) into the load: x' = x * rsqrt(mean(x^2)+eps) * nw.

struct QmvParams {
  uint N;
  uint K;
  uint x_stride;  // floats between consecutive input vectors (t)
  uint y_stride;  // floats between consecutive output vectors (t)
  float eps;      // NORM only
};

template <int BITS, int VPL, int ROWS, int GS, int NT, bool ADD, bool NORM>
NI_INLINE void qmv_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                        const device float* x, device float* y, constant QmvParams& p,
                        const device float* resid, const device bfloat* nw, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  static_assert(VPL % PER_WORD == 0, "VPL must cover whole words");
  static_assert(GS % VPL == 0, "a lane chunk must stay inside one group");
  const uint K = p.K;
  const uint words_per_row = K / PER_WORD;
  const uint groups_per_row = K / GS;

  NormCtx<NT> nc;
  if (NORM) nc = rms_ctx<NT>(x, p.x_stride, K, p.eps, lane);

  float acc[ROWS][NT];
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

  // Rows past N are clamped (computed redundantly, never stored) so lanes never diverge.
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, p.N - 1);

  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    XChunk<BITS, VPL, NT> c;
    load_x<BITS, VPL, NT, NORM>(c, x, p.x_stride, k0, nc, nw);
    const uint g = k0 / GS;
    const uint wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const float s = float(scales[rows[r] * groups_per_row + g]);
      const float b = float(biases[rows[r] * groups_per_row + g]);
      dot_row<BITS, VPL, NT>(c, w + rows[r] * words_per_row + wo, s, b, acc[r]);
    }
  }

  for (int t = 0; t < NT; ++t) {
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N) {
        if (ADD) v += resid[t * p.y_stride + row];
        y[t * p.y_stride + row] = v;
      }
    }
  }
}

#define NI_QMV(NAME, BITS, VPL, ROWS, GS, NT, ADD, NORM)                                                 \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],     \
                       device float* y [[buffer(4)]], constant QmvParams& p [[buffer(5)]],                 \
                       const device float* resid [[buffer(6)]], const device bfloat* nw [[buffer(7)]],     \
                       uint3 tgp [[threadgroup_position_in_grid]],                                         \
                       uint sg [[simdgroup_index_in_threadgroup]],                                         \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    uint row0 = (tgp.x * nsg + sg) * ROWS;                                                                \
    qmv_impl<BITS, VPL, ROWS, GS, NT, ADD, NORM>(w, scales, biases, x, y, p, resid, nw, row0, lane);     \
  }

// name: qmv_b{bits}_g{group}_v{values per lane}_r{rows per simdgroup}_t{vectors}[_add][_norm]
NI_QMV(qmv_b4_g64_v16_r4_t1, 4, 16, 4, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v32_r4_t1, 4, 32, 4, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v16_r8_t1, 4, 16, 8, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v32_r2_t1, 4, 32, 2, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v16_r2_t1, 4, 16, 2, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v16_r4_t1_add, 4, 16, 4, 64, 1, true, false)
NI_QMV(qmv_b4_g64_v16_r4_t1_norm, 4, 16, 4, 64, 1, false, true)
NI_QMV(qmv_b8_g64_v16_r4_t1, 8, 16, 4, 64, 1, false, false)
NI_QMV(qmv_b8_g64_v16_r4_t1_norm, 8, 16, 4, 64, 1, false, true)
NI_QMV(qmv_b4_g64_v16_r4_t4, 4, 16, 4, 64, 4, false, false)

// ---- Multi-segment: several matrices that share the same input (e.g. q/k/v or qkv/z/a/b).
// Threadgroups are partitioned across segments: segment i owns tg.x in [tg_begin[i], tg_begin[i+1]).
struct QmvMultiParams {
  uint K;
  uint nseg;
  float eps;
  uint N[4];
  uint tg_begin[5];
  uint bits[4]; // per segment: 4 or 8
};

template <int VPL, int ROWS, int GS, bool NORM>
NI_INLINE void qmv_seg(const device uint32_t* w, const device bfloat* s, const device bfloat* b,
                       const device float* x, device float* y, uint N, uint bits, uint K, float eps,
                       const device bfloat* nw, uint row0, uint lane) {
  NormCtx<1> nc;
  if (NORM) nc = rms_ctx<1>(x, K, K, eps, lane);
  float acc[ROWS][1];
  for (int r = 0; r < ROWS; ++r) acc[r][0] = 0.0f;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, N - 1);
  const uint groups_per_row = K / GS;
  if (bits == 4) {
    const uint wpr = K / 8;
    for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
      XChunk<4, VPL, 1> c;
      load_x<4, VPL, 1, NORM>(c, x, K, k0, nc, nw);
      for (int r = 0; r < ROWS; ++r)
        dot_row<4, VPL, 1>(c, w + rows[r] * wpr + k0 / 8, float(s[rows[r] * groups_per_row + k0 / GS]),
                           float(b[rows[r] * groups_per_row + k0 / GS]), acc[r]);
    }
  } else {
    const uint wpr = K / 4;
    for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
      XChunk<8, VPL, 1> c;
      load_x<8, VPL, 1, NORM>(c, x, K, k0, nc, nw);
      for (int r = 0; r < ROWS; ++r)
        dot_row<8, VPL, 1>(c, w + rows[r] * wpr + k0 / 4, float(s[rows[r] * groups_per_row + k0 / GS]),
                           float(b[rows[r] * groups_per_row + k0 / GS]), acc[r]);
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r][0]);
    if (lane == 0 && row0 + r < N) y[row0 + r] = v;
  }
}

#define NI_QMV_MULTI(NAME, VPL, ROWS, GS, NORM)                                                          \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvMultiParams& p [[buffer(17)]],                   \
      const device bfloat* nw [[buffer(18)]], uint3 tgp [[threadgroup_position_in_grid]],                \
      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],               \
      uint lane [[thread_index_in_simdgroup]]) {                                                         \
    uint seg = 0;                                                                                        \
    for (uint i = 1; i < p.nseg; ++i)                                                                    \
      if (tgp.x >= p.tg_begin[i]) seg = i;                                                               \
    uint row0 = ((tgp.x - p.tg_begin[seg]) * nsg + sg) * ROWS;                                           \
    switch (seg) {                                                                                       \
      case 0: qmv_seg<VPL, ROWS, GS, NORM>(w0, s0, b0, x, y0, p.N[0], p.bits[0], p.K, p.eps, nw, row0, lane); break; \
      case 1: qmv_seg<VPL, ROWS, GS, NORM>(w1, s1, b1, x, y1, p.N[1], p.bits[1], p.K, p.eps, nw, row0, lane); break; \
      case 2: qmv_seg<VPL, ROWS, GS, NORM>(w2, s2, b2, x, y2, p.N[2], p.bits[2], p.K, p.eps, nw, row0, lane); break; \
      default: qmv_seg<VPL, ROWS, GS, NORM>(w3, s3, b3, x, y3, p.N[3], p.bits[3], p.K, p.eps, nw, row0, lane); break; \
    }                                                                                                    \
  }

NI_QMV_MULTI(qmv_multi_g64_v16_r4_norm, 16, 4, 64, true)
NI_QMV_MULTI(qmv_multi_g64_v16_r4, 16, 4, 64, false)

// ---- v2: x staged once per threadgroup (fp32, optionally RMS-normalized) ----
// Threadgroup = NSG simdgroups; each simdgroup owns ROWS rows. The input vector is loaded (and
// normalized) once per threadgroup instead of once per simdgroup.
template <int BITS, int VPL, int ROWS, int GS, bool ADD, bool NORM>
NI_INLINE void qmv2_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         const device float* x, device float* y, uint N, uint K, float eps,
                         const device float* resid, const device bfloat* nw, threadgroup float* xs,
                         threadgroup float* red, uint row0, uint tid, uint ntg, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int WORDS = VPL / PER_WORD;
  // stage x
  float ss = 0.0f;
  for (uint i = tid; i < K / 4; i += ntg) {
    float4 v = ((const device float4*)x)[i];
    ((threadgroup float4*)xs)[i] = v;
    if (NORM) ss += dot(v, v);
  }
  if (NORM) {
    ss = tg_sum(ss, red, tid, ntg);
    float inv = metal::precise::rsqrt(ss / float(K) + eps);
    for (uint i = tid; i < K; i += ntg) xs[i] = xs[i] * inv * float(nw[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (row0 >= N) return;

  const uint words_per_row = K / PER_WORD;
  const uint groups_per_row = K / GS;
  float acc[ROWS];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[VPL];
    float xsum = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = ((threadgroup float4*)(xs + k0))[i];
      xsum += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) xv[i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * words_per_row + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else if (WORDS == 2) {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      } else {
        for (int i = 0; i < WORDS; ++i) wv[i] = wp[i];
      }
      float d = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * xv[i * PER_WORD + j];
      acc[r] += float(scales[rows[r] * groups_per_row + g]) * d + float(biases[rows[r] * groups_per_row + g]) * xsum;
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    uint row = row0 + r;
    if (lane == 0 && row < N) {
      if (ADD) v += resid[row];
      y[row] = v;
    }
  }
}

#define NI_QMV2(NAME, BITS, VPL, ROWS, GS, ADD, NORM)                                                     \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],     \
                       device float* y [[buffer(4)]], constant QmvParams& p [[buffer(5)]],                 \
                       const device float* resid [[buffer(6)]], const device bfloat* nw [[buffer(7)]],     \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    threadgroup float xs[4096];                                                                           \
    threadgroup float red[32];                                                                            \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                  \
    qmv2_impl<BITS, VPL, ROWS, GS, ADD, NORM>(w, scales, biases, x, y, p.N, p.K, p.eps, resid, nw, xs, red, \
                                              row0, tid, ntg, lane);                                      \
  }

NI_QMV2(qmv2_b4_g64_r4, 4, 16, 4, 64, false, false)
NI_QMV2(qmv2_b4_g64_r2, 4, 16, 2, 64, false, false)
NI_QMV2(qmv2_b4_g64_r1, 4, 16, 1, 64, false, false)
NI_QMV2(qmv2_b4_g64_r4_add, 4, 16, 4, 64, true, false)
NI_QMV2(qmv2_b4_g64_r2_add, 4, 16, 2, 64, true, false)
NI_QMV2(qmv2_b4_g64_r4_norm, 4, 16, 4, 64, false, true)
NI_QMV2(qmv2_b4_g64_r2_norm, 4, 16, 2, 64, false, true)
NI_QMV2(qmv2_b8_g64_r4_norm, 8, 16, 4, 64, false, true)
NI_QMV2(qmv2_b8_g64_r2_norm, 8, 16, 2, 64, false, true)
NI_QMV2(qmv2_b8_g64_r1_norm, 8, 16, 1, 64, false, true)

// Multi-segment v2 (shared input, up to 4 matrices, 4- or 8-bit each). With xn_out bound
// (write_xn=1), threadgroup 0 also stores the staged (normalized) x for later kernels.
struct QmvMulti2Params {
  uint K;
  uint nseg;
  float eps;
  uint norm;
  uint write_xn;
  uint N[4];
  uint tg_begin[5];
  uint bits[4];
};

#define NI_QMV2_MULTI(NAME, ROWS)                                                                         \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvMulti2Params& p [[buffer(17)]],                  \
      const device bfloat* nw [[buffer(18)]], device float* xn_out [[buffer(19)]],                       \
      uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],               \
      uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                  \
      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    threadgroup float xs[4096];                                                                           \
    threadgroup float red[32];                                                                            \
    stage_x(x, p.K, p.norm != 0, p.eps, nw, xs, red, tid, ntg);                                          \
    if (p.write_xn && tgp == 0)                                                                           \
      for (uint i = tid; i < p.K; i += ntg) xn_out[i] = xs[i];                                            \
    uint seg = 0;                                                                                        \
    for (uint i = 1; i < p.nseg; ++i)                                                                    \
      if (tgp >= p.tg_begin[i]) seg = i;                                                                  \
    uint row0 = ((tgp - p.tg_begin[seg]) * nsg + sg) * ROWS;                                             \
    const uint N = p.N[seg];                                                                              \
    if (row0 >= N) return;                                                                                \
    const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;                        \
    const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;                          \
    const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;                          \
    device float* Y = seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3;                                 \
    float acc[ROWS];                                                                                      \
    if (p.bits[seg] == 8) qmv2_rows<8, 16, ROWS, 64>(W, S, B, xs, N, p.K, row0, lane, acc);              \
    else qmv2_rows<4, 16, ROWS, 64>(W, S, B, xs, N, p.K, row0, lane, acc);                                \
    if (lane == 0)                                                                                        \
      for (int r = 0; r < ROWS; ++r)                                                                      \
        if (row0 + r < N) Y[row0 + r] = acc[r];                                                           \
  }

NI_QMV2_MULTI(qmv2_multi_r2, 2)
NI_QMV2_MULTI(qmv2_multi_r1, 1)

// ---- v3: any K (x staged in 4096-wide tiles). Used for the wide dense models (27B: hidden 5120,
// DeltaNet value dim 6144, MLP 17408). With NORM the RMS is computed over the whole x first.
NI_INLINE float rms_inv_device(const device float* x, uint K, float eps, threadgroup float* red, uint tid, uint ntg) {
  float ss = 0.0f;
  for (uint i = tid; i < K / 4; i += ntg) {
    float4 v = ((const device float4*)x)[i];
    ss += dot(v, v);
  }
  ss = tg_sum(ss, red, tid, ntg);
  return metal::precise::rsqrt(ss / float(K) + eps);
}

template <int BITS, int ROWS>
NI_INLINE void qmv3_tile_rows(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                              threadgroup const float* xs, uint K, uint kt, uint tile, uint N, thread const uint* rows,
                              uint lane, thread float* acc) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int VPL = 16;
  constexpr int WORDS = VPL / PER_WORD;
  const uint wpr = K / PER_WORD, gpr = K / 64;
  for (uint k0 = lane * VPL; k0 < tile; k0 += 32 * VPL) {
    float xv[VPL];
    float xsum = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = ((threadgroup const float4*)(xs + k0))[i];
      xsum += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) xv[i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
    }
    const uint kg = kt + k0;
    const uint g = kg / 64, wo = kg / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * wpr + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      }
      float d = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * xv[i * PER_WORD + j];
      acc[r] += float(scales[rows[r] * gpr + g]) * d + float(biases[rows[r] * gpr + g]) * xsum;
    }
  }
}

// Stages x[kt, kt+tile) (scaled by inv * nw when norm) into xs.
NI_INLINE void stage_tile(const device float* x, uint kt, uint tile, bool norm, float inv, const device bfloat* nw,
                          threadgroup float* xs, uint tid, uint ntg) {
  for (uint i = tid; i < tile / 4; i += ntg) {
    float4 v = ((const device float4*)(x + kt))[i];
    if (norm) {
      uint k = kt + i * 4;
      v *= inv * float4(nw[k], nw[k + 1], nw[k + 2], nw[k + 3]);
    }
    ((threadgroup float4*)xs)[i] = v;
  }
}

template <int BITS, int ROWS, bool ADD, bool NORM>
NI_INLINE void qmv3_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         const device float* x, device float* y, uint N, uint K, float eps, const device float* resid,
                         const device bfloat* nw, threadgroup float* xs, threadgroup float* red, uint row0, uint tid,
                         uint ntg, uint lane) {
  const float inv = NORM ? rms_inv_device(x, K, eps, red, tid, ntg) : 1.0f;
  uint rows[ROWS];
  float acc[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, N - 1);
    acc[r] = 0.0f;
  }
  for (uint kt = 0; kt < K; kt += 4096) {
    const uint tile = min(4096u, K - kt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_tile(x, kt, tile, NORM, inv, nw, xs, tid, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < N) qmv3_tile_rows<BITS, ROWS>(w, scales, biases, xs, K, kt, tile, N, rows, lane, acc);
  }
  if (row0 >= N) return;
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    uint row = row0 + r;
    if (lane == 0 && row < N) {
      if (ADD) v += resid[row];
      y[row] = v;
    }
  }
}

#define NI_QMV3(NAME, BITS, ROWS, ADD, NORM)                                                              \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],     \
                       device float* y [[buffer(4)]], constant QmvParams& p [[buffer(5)]],                 \
                       const device float* resid [[buffer(6)]], const device bfloat* nw [[buffer(7)]],     \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    threadgroup float xs[4096];                                                                           \
    threadgroup float red[32];                                                                            \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                  \
    qmv3_impl<BITS, ROWS, ADD, NORM>(w, scales, biases, x, y, p.N, p.K, p.eps, resid, nw, xs, red, row0, tid, \
                                     ntg, lane);                                                          \
  }

NI_QMV3(qmv3_b4_r2, 4, 2, false, false)
NI_QMV3(qmv3_b4_r4, 4, 4, false, false)
NI_QMV3(qmv3_b4_r2_add, 4, 2, true, false)
NI_QMV3(qmv3_b4_r2_norm, 4, 2, false, true)
NI_QMV3(qmv3_b4_r4_norm, 4, 4, false, true)
NI_QMV3(qmv3_b8_r2, 8, 2, false, false)
NI_QMV3(qmv3_b8_r2_add, 8, 2, true, false)
NI_QMV3(qmv3_b8_r2_norm, 8, 2, false, true)

// Multi-segment v3 (any K), optional norm, optional normalized-x output (full x, all threadgroups
// write disjoint slices of it).
#define NI_QMV3_MULTI(NAME, ROWS)                                                                         \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvMulti2Params& p [[buffer(17)]],                  \
      const device bfloat* nw [[buffer(18)]], device float* xn_out [[buffer(19)]],                       \
      uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],               \
      uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                  \
      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    threadgroup float xs[4096];                                                                           \
    threadgroup float red[32];                                                                            \
    const bool norm = p.norm != 0;                                                                        \
    const float inv = norm ? rms_inv_device(x, p.K, p.eps, red, tid, ntg) : 1.0f;                         \
    if (p.write_xn && tgp == 0)                                                                           \
      for (uint i = tid; i < p.K; i += ntg) xn_out[i] = norm ? x[i] * inv * float(nw[i]) : x[i];         \
    uint seg = 0;                                                                                        \
    for (uint i = 1; i < p.nseg; ++i)                                                                    \
      if (tgp >= p.tg_begin[i]) seg = i;                                                                  \
    const uint N = p.N[seg];                                                                              \
    uint row0 = ((tgp - p.tg_begin[seg]) * nsg + sg) * ROWS;                                             \
    const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;                        \
    const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;                          \
    const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;                          \
    device float* Y = seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3;                                 \
    uint rows[ROWS];                                                                                      \
    float acc[ROWS];                                                                                      \
    for (int r = 0; r < ROWS; ++r) { rows[r] = min(row0 + r, N - 1); acc[r] = 0.0f; }                     \
    for (uint kt = 0; kt < p.K; kt += 4096) {                                                             \
      const uint tile = min(4096u, p.K - kt);                                                             \
      threadgroup_barrier(mem_flags::mem_threadgroup);                                                    \
      stage_tile(x, kt, tile, norm, inv, nw, xs, tid, ntg);                                               \
      threadgroup_barrier(mem_flags::mem_threadgroup);                                                    \
      if (row0 < N) {                                                                                     \
        if (p.bits[seg] == 8) qmv3_tile_rows<8, ROWS>(W, S, B, xs, p.K, kt, tile, N, rows, lane, acc);    \
        else qmv3_tile_rows<4, ROWS>(W, S, B, xs, p.K, kt, tile, N, rows, lane, acc);                     \
      }                                                                                                   \
    }                                                                                                     \
    if (row0 >= N) return;                                                                                \
    for (int r = 0; r < ROWS; ++r) {                                                                      \
      float v = simd_sum(acc[r]);                                                                         \
      if (lane == 0 && row0 + r < N) Y[row0 + r] = v;                                                     \
    }                                                                                                     \
  }

NI_QMV3_MULTI(qmv3_multi_r2, 2)

// Dense SwiGLU MLP, gate+up fused (any K): h[i] = silu(gate_i . xn) * (up_i . xn), xn = rmsnorm(x).
[[kernel]] void mlp_gate_up(const device uint32_t* gw [[buffer(0)]], const device bfloat* gs [[buffer(1)]],
                            const device bfloat* gb [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],
                            const device bfloat* us [[buffer(4)]], const device bfloat* ub [[buffer(5)]],
                            const device float* x [[buffer(6)]], const device bfloat* nw [[buffer(7)]],
                            device float* h [[buffer(8)]], constant QmvParams& p [[buffer(9)]],
                            uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 2;
  threadgroup float xs[4096];
  threadgroup float red[32];
  const float inv = rms_inv_device(x, p.K, p.eps, red, tid, ntg);
  const uint row0 = (tgp * nsg + sg) * ROWS;
  uint rows[ROWS];
  float ag[ROWS], au[ROWS];
  for (int r = 0; r < ROWS; ++r) { rows[r] = min(row0 + r, p.N - 1); ag[r] = au[r] = 0.0f; }
  for (uint kt = 0; kt < p.K; kt += 4096) {
    const uint tile = min(4096u, p.K - kt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_tile(x, kt, tile, true, inv, nw, xs, tid, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < p.N) {
      qmv3_tile_rows<4, ROWS>(gw, gs, gb, xs, p.K, kt, tile, p.N, rows, lane, ag);
      qmv3_tile_rows<4, ROWS>(uw, us, ub, xs, p.K, kt, tile, p.N, rows, lane, au);
    }
  }
  if (row0 >= p.N) return;
  for (int r = 0; r < ROWS; ++r) {
    float g = simd_sum(ag[r]), u = simd_sum(au[r]);
    if (lane == 0 && row0 + r < p.N) h[row0 + r] = silu(g) * u;
  }
}
