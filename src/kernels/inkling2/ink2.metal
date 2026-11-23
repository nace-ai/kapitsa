#include "common.h"
#ifndef NI_UNROLL
#define NI_UNROLL _Pragma("clang loop unroll(full)")
#endif

// Inkling-2-Mini (thinkingmachines/Inkling-2-Mini, model_type inkling2_loop) Metal kernels.
// Decode, router GEMMs (exact_f32), norms, attention, and MoE kernels run in FP32 matching the PyTorch
// golden reference; non-router prefill linear projections with M > 32 use ink2_bf16_mm (FP16 simdgroup
// inputs with FP32 accumulation):
//  - AttRes over up to 6 states and 4 queries
//  - Low-rank MessageGate controllers (rank 64, tanh gates)
//  - Static (divisor sqrt(5)) and dynamic (divisor 3) causal short convolutions
//  - Q/K L2 clipping, unweighted V RMSNorm, global tau log-scaling, sink-logit attention with rel bias
//  - 128-expert top-8 sigmoid MoE with interleaved gate/up SiTU (4*tanh*sigmoid * 25*tanh):
//      * layers 0..19: NVFP4 (low/high nibble E2M1, F8_E4M3FN block-16 scales, outer scale amax/denom)
//      * layers 20..23: MXFP8 (F8_E4M3FN weights, UE8M0 block-32 scales)

NI_INLINE float4 ink2_bf16x4(uint2 r) {
  return float4(as_type<float>(r.x << 16), as_type<float>(r.x & 0xFFFF0000u),
                as_type<float>(r.y << 16), as_type<float>(r.y & 0xFFFF0000u));
}

constant float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

NI_INLINE float ink2_e4m3(uint b) {
  const ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return float(as_type<half>(h)) * 256.0f;
}

NI_INLINE float ink2_ue8m0(uint b) {
  return b == 0u ? 5.8774718e-39f : as_type<float>(b << 23);
}

NI_INLINE float ink2_nvfp4_dot(const device uchar* w, size_t wrow, uint half_k, uint blk, const thread float* xf) {
  const uint2 pw = *(const device uint2*)(w + wrow * half_k + blk * 8);
  const thread uchar* bw = (const thread uchar*)&pw;
  float d = 0.0f;
  for (uint b = 0; b < 8; ++b) d += kE2M1[bw[b] & 15u] * xf[2 * b] + kE2M1[bw[b] >> 4] * xf[2 * b + 1];
  return d;
}

// Same sum as ink2_nvfp4_dot with a per-threadgroup byte -> (lo, hi) E2M1 table (one 8-byte LUT read per byte instead
// of two constant-space lookups + nibble masks; lab2 probe: w13 6.30 -> 5.44 ms/token)
NI_INLINE float ink2_nvfp4_dot_l(const device uchar* w, size_t wrow, uint half_k, uint blk, const thread float* xf,
                                 threadgroup const float2* lut) {
  const uint2 pw = *(const device uint2*)(w + wrow * half_k + blk * 8);
  float d = 0.0f;
  for (uint b = 0; b < 8; ++b) {
    const float2 v = lut[((b < 4 ? pw.x : pw.y) >> (8u * (b & 3u))) & 0xFFu];
    d += v.x * xf[2 * b] + v.y * xf[2 * b + 1];
  }
  return d;
}
#define INK2_NVFP4_LUT_FILL(tid, nthreads)                                                                      \
  threadgroup float2 lut[256];                                                                                  \
  if (LUT) {                                                                                                    \
    for (uint i_ = tid; i_ < 256u; i_ += nthreads) lut[i_] = float2(kE2M1[i_ & 15u], kE2M1[i_ >> 4]);          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                            \
  }
#define INK2_NVFP4_DOT(w, row, half_k, blk, xf) \
  (LUT ? ink2_nvfp4_dot_l(w, row, half_k, blk, xf, lut) : ink2_nvfp4_dot(w, row, half_k, blk, xf))

// ---- 1. Embedding lookup + RMSNorm (BF16) ----
struct Ink2EmbedP {
  uint T, H, tok_off, V;
  float eps;
};

[[kernel]] void ink2_embed_norm(const device int* tokens [[buffer(0)]], const device bfloat* embed [[buffer(1)]],
                                const device bfloat* norm_w [[buffer(2)]], device float* out [[buffer(3)]],
                                constant Ink2EmbedP& p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int raw_id = tokens[p.tok_off + t];
  const uint id = (raw_id >= 0 && uint(raw_id) < p.V) ? uint(raw_id) : 0u;
  const device bfloat* row = embed + size_t(id) * p.H;
  float xs[6];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = float(row[c]);
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  device float* dst = out + size_t(t) * p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * float(norm_w[c]);
}

// ---- 2. Generic FP32 RMSNorm (optional BF16/F32 weight, alias-safe) ----
struct Ink2NormP {
  uint T, H, has_w, w_f32;
  float eps;
};

[[kernel]] void ink2_rmsnorm(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                             device float* out [[buffer(2)]], constant Ink2NormP& p [[buffer(3)]],
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* src = x + size_t(t) * p.H;
  device float* dst = out + size_t(t) * p.H;
  float xs[8];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = src[c];
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    float y = xs[k] * inv;
    if (p.has_w) y *= p.w_f32 ? ((const device float*)w)[c] : float(((const device bfloat*)w)[c]);
    dst[c] = y;
  }
}

// ---- 3. Block Attention Residual (M <= 6 states, Q <= 4 queries) ----
struct Ink2AttnResP {
  uint T, H, M, Q, q_f32;
  float eps, scale;
};

[[kernel]] void ink2_attn_res(const device float* s0 [[buffer(0)]], const device float* s1 [[buffer(1)]],
                              const device float* s2 [[buffer(2)]], const device float* s3 [[buffer(3)]],
                              const device float* s4 [[buffer(4)]], const device float* s5 [[buffer(5)]],
                              const device void* queries [[buffer(6)]], device float* out [[buffer(7)]],
                              constant Ink2AttnResP& p [[buffer(8)]], uint t [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  const device float* ptrs[6] = {s0, s1, s2, s3, s4, s5};
  threadgroup float scratch[32];
  threadgroup float tinv[6];
  threadgroup float tsc[6][4];
  threadgroup float tw[6][4];

  float qv[4][5];
  for (uint q = 0; q < p.Q; ++q) {
    uint k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      const uint idx = q * p.H + c;
      qv[q][k] = p.q_f32 ? ((const device float*)queries)[idx] : float(((const device bfloat*)queries)[idx]);
    }
  }

  for (uint m = 0; m < p.M; ++m) {
    const device float* sr = ptrs[m] + size_t(t) * p.H;
    float sv[5];
    float ss = 0.0f;
    uint k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      const float v = sr[c];
      sv[k] = v;
      ss += v * v;
    }
    ss = tg_sum(ss, scratch, tid, ntg);
    const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
    if (tid == 0) tinv[m] = inv;
    for (uint q = 0; q < p.Q; ++q) {
      float dot = 0.0f;
      for (uint kk = 0; kk < k; ++kk) dot += (sv[kk] * inv) * qv[q][kk];
      dot = tg_sum(dot, scratch, tid, ntg);
      if (tid == 0) tsc[m][q] = dot * p.scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < p.Q) {
    float mx = -INFINITY;
    for (uint m = 0; m < p.M; ++m) mx = max(mx, tsc[m][tid]);
    float exps[6];
    float sum = 0.0f;
    for (uint m = 0; m < p.M; ++m) {
      exps[m] = metal::precise::exp(tsc[m][tid] - mx);
      sum += exps[m];
    }
    const float isum = 1.0f / sum;
    for (uint m = 0; m < p.M; ++m) tw[m][tid] = exps[m] * isum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint c = tid; c < p.H; c += ntg) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint m = 0; m < p.M; ++m) {
      const float kv = ptrs[m][size_t(t) * p.H + c] * tinv[m];
      for (uint q = 0; q < p.Q; ++q) acc[q] += tw[m][q] * kv;
    }
    for (uint q = 0; q < p.Q; ++q) out[(size_t(q) * p.T + t) * p.H + c] = acc[q];
  }
}

// ink2_attn_res with one reduction round: each thread accumulates all M x (1 + Q) partial sums (sum of squares and the
// raw state . query dots, scaled by the state's rms afterwards), one simd_sum each, one threadgroup exchange; the
// original runs M x (1 + Q) serial tg_sum rounds (a single threadgroup at decode: ~56 us a call on the M1 Max).
[[kernel]] void ink2_attn_res1(const device float* s0 [[buffer(0)]], const device float* s1 [[buffer(1)]],
                               const device float* s2 [[buffer(2)]], const device float* s3 [[buffer(3)]],
                               const device float* s4 [[buffer(4)]], const device float* s5 [[buffer(5)]],
                               const device void* queries [[buffer(6)]], device float* out [[buffer(7)]],
                               constant Ink2AttnResP& p [[buffer(8)]], uint t [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const device float* ptrs[6] = {s0, s1, s2, s3, s4, s5};
  threadgroup float part[16][30];
  threadgroup float tot[30];
  threadgroup float tinv[6];
  threadgroup float tw[6][4];
  float acc[30];
  for (uint i = 0; i < 30; ++i) acc[i] = 0.0f;
  for (uint c = tid; c < p.H; c += ntg) {
    float qv[4];
    for (uint q = 0; q < p.Q; ++q) {
      const uint idx = q * p.H + c;
      qv[q] = p.q_f32 ? ((const device float*)queries)[idx] : float(((const device bfloat*)queries)[idx]);
    }
    for (uint m = 0; m < p.M; ++m) {
      const float v = ptrs[m][size_t(t) * p.H + c];
      acc[m * 5] += v * v;
      for (uint q = 0; q < p.Q; ++q) acc[m * 5 + 1 + q] += v * qv[q];
    }
  }
  const uint n = p.M * 5, nsg = (ntg + 31) / 32;
  for (uint i = 0; i < n; ++i) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) part[sg][i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < n) {
    float s = 0.0f;
    for (uint g = 0; g < nsg; ++g) s += part[g][tid];
    tot[tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.Q) {
    float sc[6], mx = -INFINITY;
    for (uint m = 0; m < p.M; ++m) {
      const float inv = metal::precise::rsqrt(tot[m * 5] / float(p.H) + p.eps);
      if (tid == 0) tinv[m] = inv;
      sc[m] = tot[m * 5 + 1 + tid] * inv * p.scale;
      mx = max(mx, sc[m]);
    }
    float exps[6], sum = 0.0f;
    for (uint m = 0; m < p.M; ++m) {
      exps[m] = metal::precise::exp(sc[m] - mx);
      sum += exps[m];
    }
    const float isum = 1.0f / sum;
    for (uint m = 0; m < p.M; ++m) tw[m][tid] = exps[m] * isum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = tid; c < p.H; c += ntg) {
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint m = 0; m < p.M; ++m) {
      const float kv = ptrs[m][size_t(t) * p.H + c] * tinv[m];
      for (uint q = 0; q < p.Q; ++q) o[q] += tw[m][q] * kv;
    }
    for (uint q = 0; q < p.Q; ++q) out[(size_t(q) * p.T + t) * p.H + c] = o[q];
  }
}


// ink2_attn_res1 with M states / Q queries compile-time (the runtime-indexed acc[30] of ink2_attn_res1 lives in memory:
// 42 us a call at T=1, 120 calls a token). Same math and reduction tree.
template <uint M, uint Q>
[[kernel]] void ink2_attn_res_mq(const device float* s0 [[buffer(0)]], const device float* s1 [[buffer(1)]],
                                 const device float* s2 [[buffer(2)]], const device float* s3 [[buffer(3)]],
                                 const device float* s4 [[buffer(4)]], const device float* s5 [[buffer(5)]],
                                 const device void* queries [[buffer(6)]], device float* out [[buffer(7)]],
                                 constant Ink2AttnResP& p [[buffer(8)]], uint t [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint S = Q + 1, N = M * S;
  const device float* ptrs[6] = {s0, s1, s2, s3, s4, s5};
  threadgroup float part[16][N];
  threadgroup float tot[N];
  threadgroup float tinv[M];
  threadgroup float tw[M][Q];
  float acc[N];
  NI_UNROLL for (uint i = 0; i < N; ++i) acc[i] = 0.0f;
  for (uint c = tid; c < p.H; c += ntg) {
    float qv[Q];
    NI_UNROLL for (uint q = 0; q < Q; ++q) {
      const uint idx = q * p.H + c;
      qv[q] = p.q_f32 ? ((const device float*)queries)[idx] : float(((const device bfloat*)queries)[idx]);
    }
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      const float v = ptrs[m][size_t(t) * p.H + c];
      acc[m * S] += v * v;
      NI_UNROLL for (uint q = 0; q < Q; ++q) acc[m * S + 1 + q] += v * qv[q];
    }
  }
  const uint nsg = (ntg + 31) / 32;
  NI_UNROLL for (uint i = 0; i < N; ++i) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) part[sg][i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < N) {
    float s = 0.0f;
    for (uint g = 0; g < nsg; ++g) s += part[g][tid];
    tot[tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < Q) {
    float sc[M], mx = -INFINITY;
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      const float inv = metal::precise::rsqrt(tot[m * S] / float(p.H) + p.eps);
      if (tid == 0) tinv[m] = inv;
      sc[m] = tot[m * S + 1 + tid] * inv * p.scale;
      mx = max(mx, sc[m]);
    }
    float exps[M], sum = 0.0f;
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      exps[m] = metal::precise::exp(sc[m] - mx);
      sum += exps[m];
    }
    const float isum = 1.0f / sum;
    NI_UNROLL for (uint m = 0; m < M; ++m) tw[m][tid] = exps[m] * isum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = tid; c < p.H; c += ntg) {
    float o[Q];
    NI_UNROLL for (uint q = 0; q < Q; ++q) o[q] = 0.0f;
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      const float kv = ptrs[m][size_t(t) * p.H + c] * tinv[m];
      NI_UNROLL for (uint q = 0; q < Q; ++q) o[q] += tw[m][q] * kv;
    }
    NI_UNROLL for (uint q = 0; q < Q; ++q) out[(size_t(q) * p.T + t) * p.H + c] = o[q];
  }
}
// ink2_attn_res_mq + the RMSNorm(s) of its outputs in the same threadgroup (lab7: one stage less per fold; ~48 a
// token): NN norms, norm j of output query src[j] with weight w_j into nout_j. Same per-thread channel walk
// (c = tid + k * ntg) and tg_sum as ink2_rmsnorm at 512 threads, so outputs are bit-identical. H <= 5 * ntg.
struct Ink2ResNormP {
  uint src[3], has_w[3], w_f32[3];
};
template <uint M, uint Q, uint NN>
[[kernel]] void ink2_attn_res_n(const device float* s0 [[buffer(0)]], const device float* s1 [[buffer(1)]],
                                const device float* s2 [[buffer(2)]], const device float* s3 [[buffer(3)]],
                                const device float* s4 [[buffer(4)]], const device float* s5 [[buffer(5)]],
                                const device void* queries [[buffer(6)]], device float* out [[buffer(7)]],
                                constant Ink2AttnResP& p [[buffer(8)]], const device void* nw0 [[buffer(9)]],
                                const device void* nw1 [[buffer(10)]], const device void* nw2 [[buffer(11)]],
                                device float* no0 [[buffer(12)]], device float* no1 [[buffer(13)]],
                                device float* no2 [[buffer(14)]], constant Ink2ResNormP& np [[buffer(15)]],
                                uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint S = Q + 1, N = M * S;
  const device float* ptrs[6] = {s0, s1, s2, s3, s4, s5};
  threadgroup float part[16][N];
  threadgroup float tot[N];
  threadgroup float tinv[M];
  threadgroup float tw[M][Q];
  threadgroup float scratch[32];
  float acc[N];
  NI_UNROLL for (uint i = 0; i < N; ++i) acc[i] = 0.0f;
  for (uint c = tid; c < p.H; c += ntg) {
    float qv[Q];
    NI_UNROLL for (uint q = 0; q < Q; ++q) {
      const uint idx = q * p.H + c;
      qv[q] = p.q_f32 ? ((const device float*)queries)[idx] : float(((const device bfloat*)queries)[idx]);
    }
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      const float v = ptrs[m][size_t(t) * p.H + c];
      acc[m * S] += v * v;
      NI_UNROLL for (uint q = 0; q < Q; ++q) acc[m * S + 1 + q] += v * qv[q];
    }
  }
  const uint nsg = (ntg + 31) / 32;
  NI_UNROLL for (uint i = 0; i < N; ++i) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) part[sg][i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < N) {
    float s = 0.0f;
    for (uint g = 0; g < nsg; ++g) s += part[g][tid];
    tot[tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < Q) {
    float sc[M], mx = -INFINITY;
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      const float inv = metal::precise::rsqrt(tot[m * S] / float(p.H) + p.eps);
      if (tid == 0) tinv[m] = inv;
      sc[m] = tot[m * S + 1 + tid] * inv * p.scale;
      mx = max(mx, sc[m]);
    }
    float exps[M], sum = 0.0f;
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      exps[m] = metal::precise::exp(sc[m] - mx);
      sum += exps[m];
    }
    const float isum = 1.0f / sum;
    NI_UNROLL for (uint m = 0; m < M; ++m) tw[m][tid] = exps[m] * isum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ov[Q][8];
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    float o[Q];
    NI_UNROLL for (uint q = 0; q < Q; ++q) o[q] = 0.0f;
    NI_UNROLL for (uint m = 0; m < M; ++m) {
      const float kv = ptrs[m][size_t(t) * p.H + c] * tinv[m];
      NI_UNROLL for (uint q = 0; q < Q; ++q) o[q] += tw[m][q] * kv;
    }
    NI_UNROLL for (uint q = 0; q < Q; ++q) out[(size_t(q) * p.T + t) * p.H + c] = o[q], ov[q][k] = o[q];
  }
  const device void* nws[3] = {nw0, nw1, nw2};
  device float* nos[3] = {no0, no1, no2};
  NI_UNROLL for (uint j = 0; j < NN; ++j) {
    const uint q = Q == 1u ? 0u : np.src[j];
    // (ink2_rmsnorm's loops verbatim on the register copy)
    float xs[8];
    float ss = 0.0f;
    k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      float v = ov[0][k];
      NI_UNROLL for (uint qq = 1; qq < Q; ++qq) v = qq == q ? ov[qq][k] : v;
      xs[k] = v;
      ss += v * v;
    }
    ss = tg_sum(ss, scratch, tid, ntg);
    const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
    device float* dst = nos[j] + size_t(t) * p.H;
    k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      float y = xs[k] * inv;
      if (np.has_w[j]) y *= np.w_f32[j] ? ((const device float*)nws[j])[c] : float(((const device bfloat*)nws[j])[c]);
      dst[c] = y;
    }
  }
}
#define INK2_ARESN(m, q, nn)                                                                                     \
  template [[host_name("ink2_attn_res_n_m" #m "q" #q "n" #nn)]] [[kernel]] decltype(ink2_attn_res_n<m, q, nn>) \
      ink2_attn_res_n<m, q, nn>;
#define INK2_ARESN_M(m) INK2_ARESN(m, 1, 1) INK2_ARESN(m, 1, 3) INK2_ARESN(m, 3, 1) INK2_ARESN(m, 3, 3)
INK2_ARESN_M(1) INK2_ARESN_M(2) INK2_ARESN_M(3) INK2_ARESN_M(4) INK2_ARESN_M(5) INK2_ARESN_M(6)
#undef INK2_ARESN_M
#undef INK2_ARESN
#define INK2_ARES(m, q) template [[host_name("ink2_attn_res_m" #m "q" #q)]] [[kernel]] decltype(ink2_attn_res_mq<m, q>) ink2_attn_res_mq<m, q>;
#define INK2_ARES_Q(m) INK2_ARES(m, 1) INK2_ARES(m, 2) INK2_ARES(m, 3) INK2_ARES(m, 4)
INK2_ARES_Q(1) INK2_ARES_Q(2) INK2_ARES_Q(3) INK2_ARES_Q(4) INK2_ARES_Q(5) INK2_ARES_Q(6)
#undef INK2_ARES_Q
#undef INK2_ARES

// ---- 4. Cross-Pass Controller MessageGate (rank 64) ----
struct Ink2MsgGateP {
  uint T, H, R, w_f32;
  float eps;
};

[[kernel]] void ink2_msg_gate(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                              const device void* down_w [[buffer(2)]], const device void* forget_w [[buffer(3)]],
                              const device void* gain_w [[buffer(4)]], device float* out [[buffer(5)]],
                              constant Ink2MsgGateP& p [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  // Single H-element shared buffer reused across phases (9,600 B total static tgmem, < 32 KB even under MTL_DEBUG_LAYER=1).
  threadgroup float scratch[32];
  threadgroup float th_buf[2304];
  threadgroup float th_g[64];

  const device float* hr = h + size_t(t) * p.H;
  const device float* mr = m + size_t(t) * p.H;
  float hs[5], ms[5];
  float ssh = 0.0f, ssm = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float hv = hr[c], mv = mr[c];
    hs[k] = hv;
    ms[k] = mv;
    ssh += hv * hv;
    ssm += mv * mv;
  }
  ssh = tg_sum(ssh, scratch, tid, ntg);
  ssm = tg_sum(ssm, scratch, tid, ntg);
  const float invh = metal::precise::rsqrt(ssh / float(p.H) + p.eps);
  const float invm = metal::precise::rsqrt(ssm / float(p.H) + p.eps);

  // Phase 1: norm(h) * down_w[:, 0:H] -> th_g
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) th_buf[c] = hs[k] * invh;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint nsg = (ntg + 31) / 32;
  for (uint r = sg; r < p.R; r += nsg) {
    float acc = 0.0f;
    const size_t row_off = size_t(r) * (2 * p.H);
    for (uint c = lane * 4; c < p.H; c += 128) {
      const float4 nh4 = *(const threadgroup float4*)(th_buf + c);
      float4 dw0;
      if (p.w_f32) {
        dw0 = *(const device float4*)((const device float*)down_w + row_off + c);
      } else {
        const device bfloat* b0 = (const device bfloat*)down_w + row_off + c;
        dw0 = float4(b0[0], b0[1], b0[2], b0[3]);
      }
      acc += dot(nh4, dw0);
    }
    acc = simd_sum(acc);
    if (lane == 0) th_g[r] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: norm(m) * down_w[:, H:2H] -> accumulate into th_g
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) th_buf[c] = ms[k] * invm;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint r = sg; r < p.R; r += nsg) {
    float acc = 0.0f;
    const size_t row_off = size_t(r) * (2 * p.H) + p.H;
    for (uint c = lane * 4; c < p.H; c += 128) {
      const float4 nm4 = *(const threadgroup float4*)(th_buf + c);
      float4 dw1;
      if (p.w_f32) {
        dw1 = *(const device float4*)((const device float*)down_w + row_off + c);
      } else {
        const device bfloat* b1 = (const device bfloat*)down_w + row_off + c;
        dw1 = float4(b1[0], b1[1], b1[2], b1[3]);
      }
      acc += dot(nm4, dw1);
    }
    acc = simd_sum(acc);
    if (lane == 0) th_g[r] += acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device float* dst = out + size_t(t) * p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    float df = 0.0f, dg = 0.0f;
    const size_t off = size_t(c) * p.R;
    for (uint r = 0; r < p.R; r += 4) {
      const float4 g4 = *(const threadgroup float4*)(th_g + r);
      float4 fw, gw;
      if (p.w_f32) {
        fw = *(const device float4*)((const device float*)forget_w + off + r);
        gw = *(const device float4*)((const device float*)gain_w + off + r);
      } else {
        const device bfloat* fb = (const device bfloat*)forget_w + off + r;
        const device bfloat* gb = (const device bfloat*)gain_w + off + r;
        fw = float4(fb[0], fb[1], fb[2], fb[3]);
        gw = float4(gb[0], gb[1], gb[2], gb[3]);
      }
      df += dot(g4, fw);
      dg += dot(g4, gw);
    }
    const float fg = metal::precise::tanh(df);
    const float gn = metal::precise::tanh(dg);
    dst[c] = (1.0f - fg) * hs[k] + gn * ms[k];
  }
}

// MessageGate tail of the GEMM path (T > 16): f forget logits, g gain logits -> (1 - tanh f) h + tanh g m (out may be h)
[[kernel]] void ink2_msg_gate_mix(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                                  const device float* f [[buffer(2)]], const device float* g [[buffer(3)]],
                                  device float* out [[buffer(4)]], constant uint& n [[buffer(5)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = (1.0f - metal::precise::tanh(f[i])) * h[i] + metal::precise::tanh(g[i]) * m[i];
}

[[kernel]] void ink2_msg_gate_prep_s(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                                     device float* cat [[buffer(2)]], constant Ink2MsgGateP& p [[buffer(3)]],
                                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* hr = h + size_t(t) * p.H;
  const device float* mr = m + size_t(t) * p.H;
  float hs[5], ms[5];
  float ssh = 0.0f, ssm = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float hv = hr[c], mv = mr[c];
    hs[k] = hv; ms[k] = mv;
    ssh += hv * hv; ssm += mv * mv;
  }
  ssh = tg_sum(ssh, scratch, tid, ntg);
  ssm = tg_sum(ssm, scratch, tid, ntg);
  const float invh = metal::precise::rsqrt(ssh / float(p.H) + p.eps);
  const float invm = metal::precise::rsqrt(ssm / float(p.H) + p.eps);
  device float* d0 = cat + size_t(t) * (2u * p.H);
  device float* d1 = d0 + p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    d0[c] = hs[k] * invh;
    d1[c] = ms[k] * invm;
  }
}

[[kernel]] void ink2_msg_gate_down_s(const device float* cat [[buffer(0)]], const device void* down_w [[buffer(1)]],
                                     device float* lat [[buffer(2)]], constant Ink2MsgGateP& p [[buffer(3)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x * 4u + sg;
  if (r >= p.R) return;
  const uint t0 = tg.y * 4u, mc = min(4u, p.T - t0);
  const size_t row_off0 = size_t(r) * (2u * p.H), row_off1 = row_off0 + p.H;
  float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
  for (uint c = lane * 4u; c < p.H; c += 128u) {
    float4 dw0, dw1;
    if (p.w_f32) {
      dw0 = *(const device float4*)((const device float*)down_w + row_off0 + c);
      dw1 = *(const device float4*)((const device float*)down_w + row_off1 + c);
    } else {
      dw0 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row_off0 + c));
      dw1 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row_off1 + c));
    }
    for (uint m = 0u; m < mc; ++m) {
      const device float* row = cat + size_t(t0 + m) * (2u * p.H) + c;
      a0[m] += dot(*(const device float4*)(row), dw0);
      a1[m] += dot(*(const device float4*)(row + p.H), dw1);
    }
  }
  for (uint m = 0u; m < mc; ++m) {
    const float s0 = simd_sum(a0[m]), s1 = simd_sum(a1[m]);
    if (lane == 0u) lat[size_t(t0 + m) * p.R + r] = s0 + s1;
  }
}

[[kernel]] void ink2_msg_gate_up_mix_s(const device float* h [[buffer(0)]], const device float* m_buf [[buffer(1)]],
                                       const device float* lat [[buffer(2)]], const device void* forget_w [[buffer(3)]],
                                       const device void* gain_w [[buffer(4)]], device float* out [[buffer(5)]],
                                       constant Ink2MsgGateP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sh_lat[8 * 64];
  const uint t0 = tg.y * 8u, mc = min(8u, p.T - t0);
  for (uint idx = tid; idx < mc * p.R; idx += 128u) sh_lat[idx] = lat[size_t(t0) * p.R + idx];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg.x * 128u + tid;
  if (c >= p.H) return;
  float df[8] = {0, 0, 0, 0, 0, 0, 0, 0}, dg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t off = size_t(c) * p.R;
  for (uint r = 0u; r < p.R; r += 4u) {
    float4 fw, gw;
    if (p.w_f32) {
      fw = *(const device float4*)((const device float*)forget_w + off + r);
      gw = *(const device float4*)((const device float*)gain_w + off + r);
    } else {
      fw = ink2_bf16x4(*(const device uint2*)((const device bfloat*)forget_w + off + r));
      gw = ink2_bf16x4(*(const device uint2*)((const device bfloat*)gain_w + off + r));
    }
    for (uint m = 0u; m < mc; ++m) {
      const float4 g4 = *(const threadgroup float4*)(sh_lat + m * p.R + r);
      df[m] += dot(g4, fw);
      dg[m] += dot(g4, gw);
    }
  }
  for (uint m = 0u; m < mc; ++m) {
    const size_t idx = size_t(t0 + m) * p.H + c;
    const float fg = metal::precise::tanh(df[m]), gn = metal::precise::tanh(dg[m]);
    out[idx] = (1.0f - fg) * h[idx] + gn * m_buf[idx];
  }
}

// Small-T MessageGate, fused (round 3): down reads h and m directly and folds the two RMSNorms in as scalars
// (lat = invh * h.w0 + invm * m.w1), one threadgroup of 256 per latent row (64 TGs, was 16 TGs of 4 rows plus a
// separate prep kernel); up_mix reads each 64-wide forget / gain row with 8 lanes (16 B each, coalesced).
[[kernel]] void ink2_msg_gate_down_f(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                                     const device void* down_w [[buffer(2)]], device float* lat [[buffer(3)]],
                                     constant Ink2MsgGateP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8][16];
  threadgroup float tot[16];
  const uint r = tg.x, t0 = tg.y * 4u, mc = min(4u, p.T - t0);
  const size_t row0 = size_t(r) * (2u * p.H), row1 = row0 + p.H;
  float ah[4] = {0, 0, 0, 0}, am[4] = {0, 0, 0, 0}, sh[4] = {0, 0, 0, 0}, sm[4] = {0, 0, 0, 0};
  for (uint c = tid * 4u; c < p.H; c += 1024u) {
    float4 w0, w1;
    if (p.w_f32) {
      w0 = *(const device float4*)((const device float*)down_w + row0 + c);
      w1 = *(const device float4*)((const device float*)down_w + row1 + c);
    } else {
      w0 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row0 + c));
      w1 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row1 + c));
    }
    for (uint j = 0u; j < mc; ++j) {
      const float4 hv = *(const device float4*)(h + size_t(t0 + j) * p.H + c);
      const float4 mv = *(const device float4*)(m + size_t(t0 + j) * p.H + c);
      ah[j] += dot(hv, w0); am[j] += dot(mv, w1);
      sh[j] += dot(hv, hv); sm[j] += dot(mv, mv);
    }
  }
  for (uint j = 0u; j < 4u; ++j) {
    const float a = simd_sum(ah[j]), b = simd_sum(am[j]), c2 = simd_sum(sh[j]), d = simd_sum(sm[j]);
    if (lane == 0u) { red[sg][j] = a; red[sg][4 + j] = b; red[sg][8 + j] = c2; red[sg][12 + j] = d; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 16u) {
    float v = 0.0f;
    for (uint g = 0u; g < 8u; ++g) v += red[g][tid];
    tot[tid] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < mc) {
    const float invh = metal::precise::rsqrt(tot[8 + tid] / float(p.H) + p.eps);
    const float invm = metal::precise::rsqrt(tot[12 + tid] / float(p.H) + p.eps);
    lat[size_t(t0 + tid) * p.R + r] = tot[tid] * invh + tot[4 + tid] * invm;
  }
}

// R == 64: 8 lanes per row (8 weights each), 4 rows per simdgroup, 32 rows per 256-thread TG; up to 8 tokens per TG.
[[kernel]] void ink2_msg_gate_up_mix_c(const device float* h [[buffer(0)]], const device float* m_buf [[buffer(1)]],
                                       const device float* lat [[buffer(2)]], const device void* forget_w [[buffer(3)]],
                                       const device void* gain_w [[buffer(4)]], device float* out [[buffer(5)]],
                                       constant Ink2MsgGateP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x * 32u + sg * 4u + (lane >> 3u), sub = lane & 7u;
  const uint t0 = tg.y * 8u, mc = min(8u, p.T - t0);
  if (c >= p.H) return;
  const size_t off = size_t(c) * 64u + sub * 8u;
  float4 f0, f1, g0, g1;
  if (p.w_f32) {
    f0 = *(const device float4*)((const device float*)forget_w + off);
    f1 = *(const device float4*)((const device float*)forget_w + off + 4u);
    g0 = *(const device float4*)((const device float*)gain_w + off);
    g1 = *(const device float4*)((const device float*)gain_w + off + 4u);
  } else {
    const uint4 fr = *(const device uint4*)((const device bfloat*)forget_w + off);
    const uint4 gr = *(const device uint4*)((const device bfloat*)gain_w + off);
    f0 = ink2_bf16x4(fr.xy); f1 = ink2_bf16x4(fr.zw);
    g0 = ink2_bf16x4(gr.xy); g1 = ink2_bf16x4(gr.zw);
  }
  for (uint j = 0u; j < mc; ++j) {
    const device float* lr = lat + size_t(t0 + j) * 64u + sub * 8u;
    const float4 l0 = *(const device float4*)lr, l1 = *(const device float4*)(lr + 4u);
    float df = dot(l0, f0) + dot(l1, f1), dg = dot(l0, g0) + dot(l1, g1);
    df += simd_shuffle_xor(df, 1u); dg += simd_shuffle_xor(dg, 1u);
    df += simd_shuffle_xor(df, 2u); dg += simd_shuffle_xor(dg, 2u);
    df += simd_shuffle_xor(df, 4u); dg += simd_shuffle_xor(dg, 4u);
    if (sub == (j & 7u)) {
      const size_t idx = size_t(t0 + j) * p.H + c;
      out[idx] = (1.0f - metal::precise::tanh(df)) * h[idx] + metal::precise::tanh(dg) * m_buf[idx];
    }
  }
}

// ---- 5. Linear Projections: BF16/F32 MatVec (M <= 8) and Tiled MMA GEMM (M > 8) ----
struct Ink2MmP {
  uint M, N, K, x_ld, y_ld, w_f32, add;
};

[[kernel]] void ink2_bf16_mv(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                             device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4 + sg;
  if (n >= p.N) return;
  const uint m0 = tg.y * 8;
  const uint mc = min(8u, p.M - m0);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff = size_t(n) * p.K;
  for (uint k = lane * 4; k < p.K; k += 128) {
    float4 wv;
    if (p.w_f32) {
      wv = *(const device float4*)((const device float*)w + woff + k);
    } else {
      const device bfloat* wb = (const device bfloat*)w + woff + k;
      wv = float4(wb[0], wb[1], wb[2], wb[3]);
    }
    for (uint m = 0; m < mc; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m0 + m) * p.x_ld + k);
      acc[m] += dot(xv, wv);
    }
  }
  for (uint m = 0; m < mc; ++m) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) {
      device float* dst = y + size_t(m0 + m) * p.y_ld + n;
      *dst = p.add ? (*dst + s) : s;
    }
  }
}

// One row (M = 1) per 128-thread TG, K split over the 4 simdgroups: for the small-N matvecs (router N = 128, sinks,
// gates) that left ink2_bf16_mv4 with 8 TGs (25 us a call, latency-bound)
[[kernel]] void ink2_mv_sk(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                           device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                           uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4];
  const size_t woff = size_t(n) * p.K;
  float a = 0.0f;
  for (uint k = tid * 4u; k < p.K; k += 512u) {
    const float4 xv = *(const device float4*)(x + k);
    a += dot(xv, p.w_f32 ? *(const device float4*)((const device float*)w + woff + k)
                         : ink2_bf16x4(*(const device uint2*)((const device bfloat*)w + woff + k)));
  }
  a = simd_sum(a);
  if (lane == 0u) part[sg] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float v = (part[0] + part[1]) + (part[2] + part[3]);
    y[n] = p.add ? y[n] + v : v;
  }
}

// ink2_mv_sk over M = 2..8 rows (x / y rows at x_ld / y_ld): the weight row read once, each row's sum in the order
// of ink2_mv_sk (same lane split, simd_sum, part order), so every row is bit-identical to its one-row result
template <uint M>
[[kernel]] void ink2_mv_sk_m(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                             device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                             uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][M];
  const size_t woff = size_t(n) * p.K;
  float a[M];
  for (uint m = 0u; m < M; ++m) a[m] = 0.0f;
  for (uint k = tid * 4u; k < p.K; k += 512u) {
    const float4 wv = p.w_f32 ? *(const device float4*)((const device float*)w + woff + k)
                              : ink2_bf16x4(*(const device uint2*)((const device bfloat*)w + woff + k));
    for (uint m = 0u; m < M; ++m) a[m] += dot(*(const device float4*)(x + size_t(m) * p.x_ld + k), wv);
  }
  for (uint m = 0u; m < M; ++m) {
    const float t = simd_sum(a[m]);
    if (lane == 0u) part[sg][m] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < M) {
    const float v = (part[0][tid] + part[1][tid]) + (part[2][tid] + part[3][tid]);
    device float* dst = y + size_t(tid) * p.y_ld + n;
    *dst = p.add ? *dst + v : v;
  }
}
#define INK2_SKM(m) template [[host_name("ink2_mv_sk_m" #m)]] [[kernel]] decltype(ink2_mv_sk_m<m>) ink2_mv_sk_m<m>;
INK2_SKM(2) INK2_SKM(3) INK2_SKM(4) INK2_SKM(5) INK2_SKM(6) INK2_SKM(7) INK2_SKM(8)

[[kernel]] void ink2_bf16_mv4(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N) return;
  const uint mc = min(4u, p.N - n0);
  const size_t woff0 = size_t(n0) * p.K;
  if (p.M == 1u && !p.w_f32 && mc == 4u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const device bfloat* wb = (const device bfloat*)w + woff0;
    for (uint k = lane * 4u; k < p.K; k += 128u) {
      const float4 xv = *(const device float4*)(x + k);
      a0 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 0u * size_t(p.K) + k)));
      a1 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 1u * size_t(p.K) + k)));
      a2 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 2u * size_t(p.K) + k)));
      a3 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 3u * size_t(p.K) + k)));
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2), s3 = simd_sum(a3);
    if (lane == 0u) {
      device float* dst = y + n0;
      if (p.add) { dst[0] += s0; dst[1] += s1; dst[2] += s2; dst[3] += s3; }
      else { dst[0] = s0; dst[1] = s1; dst[2] = s2; dst[3] = s3; }
    }
    return;
  }
  float acc[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  for (uint k = lane * 4u; k < p.K; k += 128u) {
    float4 wv[4];
    for (uint q = 0u; q < mc; ++q) {
      const size_t woff = woff0 + size_t(q) * p.K + k;
      wv[q] = p.w_f32 ? *(const device float4*)((const device float*)w + woff)
                      : ink2_bf16x4(*(const device uint2*)((const device bfloat*)w + woff));
    }
    for (uint m = 0u; m < p.M; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m) * p.x_ld + k);
      for (uint q = 0u; q < mc; ++q) acc[q][m] += dot(xv, wv[q]);
    }
  }
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.y_ld + n0;
      for (uint q = 0u; q < mc; ++q) dst[q] = p.add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}

struct Ink2MultiMmP {
  uint M, K, N_tot, n_segs;
  uint offs[7];
};

[[kernel]] void ink2_bf16_mv4_multi6(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                                     device float* y0 [[buffer(2)]], device float* y1 [[buffer(3)]],
                                     device float* y2 [[buffer(4)]], device float* y3 [[buffer(5)]],
                                     device float* y4 [[buffer(6)]], device float* y5 [[buffer(7)]],
                                     constant Ink2MultiMmP& p [[buffer(8)]],
                                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg + 1u < p.n_segs && n0 >= p.offs[seg + 1u]) ++seg;
  if (seg >= p.n_segs) return;
  const uint seg_off = p.offs[seg];
  const uint seg_len = p.offs[seg + 1u] - seg_off;
  const uint mc = min(4u, p.offs[seg + 1u] - n0);
  const size_t woff0 = size_t(n0) * p.K;
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  device float* dst_base = bases[seg] + (n0 - seg_off);
  if (p.M == 1u && mc == 4u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const device bfloat* wb = w + woff0;
    for (uint k = lane * 4u; k < p.K; k += 128u) {
      const float4 xv = *(const device float4*)(x + k);
      a0 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 0u * size_t(p.K) + k)));
      a1 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 1u * size_t(p.K) + k)));
      a2 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 2u * size_t(p.K) + k)));
      a3 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 3u * size_t(p.K) + k)));
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2), s3 = simd_sum(a3);
    if (lane == 0u) { dst_base[0] = s0; dst_base[1] = s1; dst_base[2] = s2; dst_base[3] = s3; }
    return;
  }
  float acc[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  for (uint k = lane * 4u; k < p.K; k += 128u) {
    float4 wv[4];
    for (uint q = 0u; q < mc; ++q)
      wv[q] = ink2_bf16x4(*(const device uint2*)(w + woff0 + size_t(q) * p.K + k));
    for (uint m = 0u; m < p.M; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m) * p.K + k);
      for (uint q = 0u; q < mc; ++q) acc[q][m] += dot(xv, wv[q]);
    }
  }
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      for (uint q = 0u; q < mc; ++q) dst[q] = sv[q];
    }
  }
}

// Verify-row matvec (2 <= M <= 8 rows, M compile-time): each simdgroup reads 4 weight rows once for all M input
// rows with the accumulators in registers (the runtime-M acc[4][4] above lives in memory: 218 vs 57 us at M=4).
// Per-row sum order equals the M=1 path, so verify rows match decode rows bit for bit.
template <uint M>
[[kernel]] void ink2_bf16_mvm(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N) return;
  const uint mc = min(4u, p.N - n0);
  const size_t woff0 = size_t(n0) * p.K;
  float acc[4][M];
  NI_UNROLL for (uint q = 0u; q < 4u; ++q) NI_UNROLL for (uint m = 0u; m < M; ++m) acc[q][m] = 0.0f;
  for (uint k = lane * 4u; k < p.K; k += 128u) {
    float4 wv[4];
    NI_UNROLL for (uint q = 0u; q < 4u; ++q) {
      const size_t woff = woff0 + size_t(min(q, mc - 1u)) * p.K + k;
      wv[q] = p.w_f32 ? *(const device float4*)((const device float*)w + woff)
                      : ink2_bf16x4(*(const device uint2*)((const device bfloat*)w + woff));
    }
    NI_UNROLL for (uint m = 0u; m < M; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m) * p.x_ld + k);
      NI_UNROLL for (uint q = 0u; q < 4u; ++q) acc[q][m] += dot(xv, wv[q]);
    }
  }
  NI_UNROLL for (uint m = 0u; m < M; ++m) {
    float sv[4];
    NI_UNROLL for (uint q = 0u; q < 4u; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.y_ld + n0;
      for (uint q = 0u; q < mc; ++q) dst[q] = p.add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}
#define INK2_MVM(m) template [[host_name("ink2_bf16_mvm" #m)]] [[kernel]] decltype(ink2_bf16_mvm<m>) ink2_bf16_mvm<m>;
INK2_MVM(2) INK2_MVM(3) INK2_MVM(4) INK2_MVM(5) INK2_MVM(6) INK2_MVM(7) INK2_MVM(8)
#undef INK2_MVM

// ink2_bf16_mv4_multi6 for 2 <= M <= 8 verify rows (compile-time M, as ink2_bf16_mvm).
template <uint M>
[[kernel]] void ink2_bf16_mvm_multi6(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                                     device float* y0 [[buffer(2)]], device float* y1 [[buffer(3)]],
                                     device float* y2 [[buffer(4)]], device float* y3 [[buffer(5)]],
                                     device float* y4 [[buffer(6)]], device float* y5 [[buffer(7)]],
                                     constant Ink2MultiMmP& p [[buffer(8)]],
                                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg + 1u < p.n_segs && n0 >= p.offs[seg + 1u]) ++seg;
  if (seg >= p.n_segs) return;
  const uint seg_off = p.offs[seg];
  const uint seg_len = p.offs[seg + 1u] - seg_off;
  const uint mc = min(4u, p.offs[seg + 1u] - n0);
  const size_t woff0 = size_t(n0) * p.K;
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  device float* dst_base = bases[seg] + (n0 - seg_off);
  float acc[4][M];
  NI_UNROLL for (uint q = 0u; q < 4u; ++q) NI_UNROLL for (uint m = 0u; m < M; ++m) acc[q][m] = 0.0f;
  for (uint k = lane * 4u; k < p.K; k += 128u) {
    float4 wv[4];
    NI_UNROLL for (uint q = 0u; q < 4u; ++q)
      wv[q] = ink2_bf16x4(*(const device uint2*)(w + woff0 + size_t(min(q, mc - 1u)) * p.K + k));
    NI_UNROLL for (uint m = 0u; m < M; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m) * p.K + k);
      NI_UNROLL for (uint q = 0u; q < 4u; ++q) acc[q][m] += dot(xv, wv[q]);
    }
  }
  NI_UNROLL for (uint m = 0u; m < M; ++m) {
    float sv[4];
    NI_UNROLL for (uint q = 0u; q < 4u; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      for (uint q = 0u; q < mc; ++q) dst[q] = sv[q];
    }
  }
}
#define INK2_MVM6(m) template [[host_name("ink2_bf16_mvm_multi6_" #m)]] [[kernel]] decltype(ink2_bf16_mvm_multi6<m>) ink2_bf16_mvm_multi6<m>;
INK2_MVM6(2) INK2_MVM6(3) INK2_MVM6(4) INK2_MVM6(5) INK2_MVM6(6) INK2_MVM6(7) INK2_MVM6(8)
#undef INK2_MVM6

// ---- 5b. Q8_0 dense matvec (load-time packed int8 [N, K] + half scales [N, K/32]; NI_INK2_Q8) ----
// M (1..8) compile-time rows, 4 weight rows per simdgroup, 8 int8 per lane per step (256 k per step; K % 256 == 0).
// MULTI: output rows split over up to 6 destinations (the fused [wq; wr; sink; gate; wk; wv] projection).
template <uint M, bool MULTI, uint R = 4>
NI_INLINE void ink2_q8_mvm_impl(const device float* x, const device uchar* wq, const device half* ws, uint N, uint K,
                                uint x_ld, uint add, device float* y, uint y_ld, device float* const thread* bases,
                                constant uint* offs, uint n_segs, uint tg, uint sg, uint lane) {
  const uint n0 = (tg * 4u + sg) * R;
  if (n0 >= N) return;
  uint mc = min(R, N - n0);
  device float* dst0 = y + n0;
  uint ld = y_ld;
  if (MULTI) {
    uint seg = 0u;
    while (seg + 1u < n_segs && n0 >= offs[seg + 1u]) ++seg;
    mc = min(R, offs[seg + 1u] - n0);
    dst0 = bases[seg] + (n0 - offs[seg]);
    ld = offs[seg + 1u] - offs[seg];
  }
  const uint nb = K / 32u;
  float acc[R][M];
  for (uint q = 0u; q < R; ++q) NI_UNROLL for (uint m = 0u; m < M; ++m) acc[q][m] = 0.0f;
  for (uint k = lane * 8u; k < K; k += 256u) {
    float4 wa[R], wb[R];
    NI_UNROLL for (uint q = 0u; q < R; ++q) {
      const uint r = n0 + min(q, mc - 1u);
      const uint2 v = *(const device uint2*)(wq + size_t(r) * K + k);
      const float s = float(ws[size_t(r) * nb + k / 32u]);
      wa[q] = float4(as_type<char4>(v.x)) * s;
      wb[q] = float4(as_type<char4>(v.y)) * s;
    }
    NI_UNROLL for (uint m = 0u; m < M; ++m) {
      const float4 xa = *(const device float4*)(x + size_t(m) * x_ld + k);
      const float4 xb = *(const device float4*)(x + size_t(m) * x_ld + k + 4u);
      NI_UNROLL for (uint q = 0u; q < R; ++q) acc[q][m] += dot(xa, wa[q]) + dot(xb, wb[q]);
    }
  }
  NI_UNROLL for (uint m = 0u; m < M; ++m) {
    float sv[R];
    NI_UNROLL for (uint q = 0u; q < R; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = dst0 + size_t(m) * ld;
      for (uint q = 0u; q < mc; ++q) dst[q] = add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}

struct Ink2Q8P {
  uint M, N, K, x_ld, y_ld, add, n_segs;
  uint offs[7];
};

template <uint M>
[[kernel]] void ink2_q8_mvm(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                            const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                            constant Ink2Q8P& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ink2_q8_mvm_impl<M, false>(x, wq, ws, p.N, p.K, p.x_ld, p.add, y, p.y_ld, nullptr, p.offs, 0u, tg, sg, lane);
}

template <uint M>
[[kernel]] void ink2_q8_mvm_multi6(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                   const device half* ws [[buffer(2)]], device float* y0 [[buffer(3)]],
                                   device float* y1 [[buffer(5)]], device float* y2 [[buffer(6)]],
                                   device float* y3 [[buffer(7)]], device float* y4 [[buffer(8)]],
                                   device float* y5 [[buffer(9)]], constant Ink2Q8P& p [[buffer(4)]],
                                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  ink2_q8_mvm_impl<M, true>(x, wq, ws, p.offs[p.n_segs], p.K, p.K, 0u, y0, 0u, bases, p.offs, p.n_segs, tg, sg, lane);
}
#define INK2_Q8(m)                                                                                         \
  template [[host_name("ink2_q8_mvm" #m)]] [[kernel]] decltype(ink2_q8_mvm<m>) ink2_q8_mvm<m>;              \
  template [[host_name("ink2_q8_mvm_multi6_" #m)]] [[kernel]] decltype(ink2_q8_mvm_multi6<m>) ink2_q8_mvm_multi6<m>;
INK2_Q8(1) INK2_Q8(2) INK2_Q8(3) INK2_Q8(4) INK2_Q8(5) INK2_Q8(6) INK2_Q8(7) INK2_Q8(8)

// One row (M = 1), R weight rows per simdgroup (R = 1, 2): more threadgroups for the small-N matvecs (N = 1024 / 2304:
// 64 / 144 threadgroups at R = 4 on 32 cores)
template <uint R>
[[kernel]] void ink2_q8_mv1r(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                             const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                             constant Ink2Q8P& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ink2_q8_mvm_impl<1, false, R>(x, wq, ws, p.N, p.K, p.x_ld, p.add, y, p.y_ld, nullptr, p.offs, 0u, tg, sg, lane);
}
template [[host_name("ink2_q8_mv1r1")]] [[kernel]] decltype(ink2_q8_mv1r<1>) ink2_q8_mv1r<1>;
template [[host_name("ink2_q8_mv1r2")]] [[kernel]] decltype(ink2_q8_mv1r<2>) ink2_q8_mv1r<2>;

#undef INK2_Q8

// lab8 (NI_INK2_CAPFOLD): the one-row Q8 matvec (R = 4, ink2_q8_mvm1's sums) with ink2_softcap on its outputs
// (unembed): one stage less per token
template <uint R>
NI_INLINE void ink2_q8_mv1_cap_impl(const device float* x, const device uchar* wq, const device half* ws, uint N,
                                    uint K, device float* y, float cap, float inv_cap, uint tg, uint sg, uint lane) {
  const uint n0 = (tg * 4u + sg) * R;
  if (n0 >= N) return;
  const uint mc = min(R, N - n0);
  const uint nb = K / 32u;
  float acc[R][1];
  for (uint q = 0u; q < R; ++q) acc[q][0] = 0.0f;
  for (uint k = lane * 8u; k < K; k += 256u) {
    float4 wa[R], wb[R];
    NI_UNROLL for (uint q = 0u; q < R; ++q) {
      const uint r = n0 + min(q, mc - 1u);
      const uint2 v = *(const device uint2*)(wq + size_t(r) * K + k);
      const float s = float(ws[size_t(r) * nb + k / 32u]);
      wa[q] = float4(as_type<char4>(v.x)) * s;
      wb[q] = float4(as_type<char4>(v.y)) * s;
    }
    const float4 xa = *(const device float4*)(x + k);
    const float4 xb = *(const device float4*)(x + k + 4u);
    NI_UNROLL for (uint q = 0u; q < R; ++q) acc[q][0] += dot(xa, wa[q]) + dot(xb, wb[q]);
  }
  float sv[R];
  NI_UNROLL for (uint q = 0u; q < R; ++q) sv[q] = simd_sum(acc[q][0]);
  if (lane == 0u) {
    device float* dst = y + n0;
    for (uint q = 0u; q < mc; ++q) {
      dst[q] = sv[q];
      dst[q] = cap * metal::precise::tanh(dst[q] * inv_cap);
    }
  }
}
struct Ink2CapP {
  uint N, K;
  float cap, inv_cap;
};
[[kernel]] void ink2_q8_mv1_cap(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                                constant Ink2CapP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ink2_q8_mv1_cap_impl<4>(x, wq, ws, p.N, p.K, y, p.cap, p.inv_cap, tg, sg, lane);
}

[[kernel]] void ink2_bf16_mv8(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][128];
  const uint m0 = tg.y * 8u;
  if (m0 >= p.M) return;
  const uint mc_m = min(8u, p.M - m0);
  const uint n0 = (tg.x * 4u + sg) * 4u;
  const bool valid = n0 < p.N;
  const uint mc = valid ? min(4u, p.N - n0) : 0u;
  float acc[4][8] = {};
  const size_t woff0 = size_t(n0) * p.K;
  const uint total_v4 = mc_m * 32u;
  for (uint kbase = 0u; kbase < p.K; kbase += 128u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 5u;
      const uint c4 = (idx & 31u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m0 + m) * p.x_ld + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 4u;
      float4 wv[4];
      for (uint q = 0u; q < mc; ++q)
        wv[q] = ink2_bf16x4(*(const device uint2*)(w + woff0 + size_t(q) * p.K + kbase + kl));
      for (uint m = 0u; m < mc_m; ++m) {
        const float4 xv = *(const threadgroup float4*)(xs[m] + kl);
        for (uint q = 0u; q < mc; ++q) acc[q][m] += dot(xv, wv[q]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  for (uint m = 0u; m < mc_m; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m0 + m) * p.y_ld + n0;
      for (uint q = 0u; q < mc; ++q) dst[q] = p.add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}

[[kernel]] void ink2_bf16_mm(const device float* X [[buffer(0)]], const device void* W [[buffer(1)]],
                             device float* Y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BM = 32, BN = 64, BK = 32, PAD = 8;
  constexpr uint TM = 16, TN = 32, FM = 2, FN = 4;
  threadgroup half xs[BM * (BK + PAD)];
  threadgroup half ws[BN * (BK + PAD)];
  const uint m0 = tg.y * BM, n0 = tg.x * BN;
  const uint sgm = sg / 2, sgn = sg % 2;

  simdgroup_float8x8 acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);

  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    // 128 threads load 32 * 32 = 1024 halfs of X (8 per thread)
    const uint xr = tid / 4, xc = (tid % 4) * 8;
    const uint xm = m0 + xr;
    threadgroup half* xdst = xs + xr * (BK + PAD) + xc;
    if (xm < p.M && k0 + xc + 7 < p.K) {
      const float4 v0 = *(const device float4*)(X + size_t(xm) * p.x_ld + k0 + xc);
      const float4 v1 = *(const device float4*)(X + size_t(xm) * p.x_ld + k0 + xc + 4);
      *(threadgroup half4*)(xdst) = half4(v0);
      *(threadgroup half4*)(xdst + 4) = half4(v1);
    } else {
      for (uint j = 0; j < 8; ++j)
        xdst[j] = (xm < p.M && k0 + xc + j < p.K) ? half(X[size_t(xm) * p.x_ld + k0 + xc + j]) : 0.0h;
    }

    // 128 threads load 64 * 32 = 2048 halfs of W (16 per thread)
    const uint wr = tid / 2, wc = (tid % 2) * 16;
    const uint wn = n0 + wr;
    threadgroup half* wdst = ws + wr * (BK + PAD) + wc;
    if (wn < p.N && k0 + wc + 15 < p.K) {
      if (p.w_f32) {
        const device float4* wp = (const device float4*)((const device float*)W + size_t(wn) * p.K + k0 + wc);
        for (int q = 0; q < 4; ++q) *(threadgroup half4*)(wdst + q * 4) = half4(wp[q]);
      } else {
        const device bfloat* wb = (const device bfloat*)W + size_t(wn) * p.K + k0 + wc;
        for (int q = 0; q < 16; ++q) wdst[q] = half(float(wb[q]));
      }
    } else {
      for (uint j = 0; j < 16; ++j) {
        float v = 0.0f;
        if (wn < p.N && k0 + wc + j < p.K) {
          const size_t idx = size_t(wn) * p.K + k0 + wc + j;
          v = p.w_f32 ? ((const device float*)W)[idx] : float(((const device bfloat*)W)[idx]);
        }
        wdst[j] = half(v);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_half8x8 a[FM], b[FN];
      for (int i = 0; i < FM; ++i) simdgroup_load(a[i], xs + (sgm * TM + i * 8) * (BK + PAD) + kk, BK + PAD);
      for (int j = 0; j < FN; ++j)
        simdgroup_load(b[j], ws + (sgn * TN + j * 8) * (BK + PAD) + kk, BK + PAD, ulong2(0, 0), true);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float stage_buf[4 * 64];
  threadgroup float* stage = stage_buf + sg * 64;
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      simdgroup_store(acc[i][j], stage, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) {
        const uint r = e / 8, c = e % 8;
        const uint m = m0 + sgm * TM + i * 8 + r;
        const uint n = n0 + sgn * TN + j * 8 + c;
        if (m < p.M && n < p.N) {
          device float* dst = Y + size_t(m) * p.y_ld + n;
          const float v = stage[e];
          *dst = p.add ? (*dst + v) : v;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- 6. Short Causal Convolutions (static & dynamic) ----
struct Ink2SconvStaticP {
  uint T, C, w_f32, add_res;
  float scale;
  uint snap_stride;  // 0 = no snapshots; else floats between per-row conv snapshots
};

[[kernel]] void ink2_sconv_static(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                  device float* hist [[buffer(2)]], device float* out [[buffer(3)]],
                                  constant Ink2SconvStaticP& p [[buffer(4)]], const device float* res [[buffer(5)]],
                                  device float* snap [[buffer(6)]],
                                  uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  float w0, w1, w2, w3;
  if (p.w_f32) {
    const float4 wf = *(const device float4*)((const device float*)w + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const device bfloat* wb = (const device bfloat*)w + size_t(c) * 4;
    w0 = float(wb[0]); w1 = float(wb[1]); w2 = float(wb[2]); w3 = float(wb[3]);
  }
  float h0 = hist[0 * p.C + c], h1 = hist[1 * p.C + c], h2 = hist[2 * p.C + c];
  for (uint t = 0; t < p.T; ++t) {
    const size_t idx = size_t(t) * p.C + c;
    const float xt = x[idx];
    // (explicit fmas, as ink2_sconv1: the fused epilogues and this kernel round identically)
    const float conv = metal::fma(w3, xt, metal::fma(w2, h2, metal::fma(w1, h1, w0 * h0)));
    const float y = p.add_res ? metal::fma(conv + xt, p.scale, res[idx]) : (conv + xt) * p.scale;
    out[idx] = y;
    h0 = h1; h1 = h2; h2 = xt;
    if (p.snap_stride) {
      device float* sh = snap + size_t(t) * p.snap_stride;
      sh[0 * p.C + c] = h0;
      sh[1 * p.C + c] = h1;
      sh[2 * p.C + c] = h2;
    }
  }
  hist[0 * p.C + c] = h0;
  hist[1 * p.C + c] = h1;
  hist[2 * p.C + c] = h2;
}

// ink2_sconv_static's body for one row (T = 1) of channel c with xt already in a register (the fused epilogues)
NI_INLINE void ink2_sconv1(float xt, const device void* w, device float* hist, device float* out,
                           constant Ink2SconvStaticP& p, const device float* res, device float* snap, uint c) {
  float w0, w1, w2, w3;
  if (p.w_f32) {
    const float4 wf = *(const device float4*)((const device float*)w + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const device bfloat* wb = (const device bfloat*)w + size_t(c) * 4;
    w0 = float(wb[0]); w1 = float(wb[1]); w2 = float(wb[2]); w3 = float(wb[3]);
  }
  float h0 = hist[0 * p.C + c], h1 = hist[1 * p.C + c], h2 = hist[2 * p.C + c];
  // (explicit fmas, shared with ink2_sconv_static: the plain expression contracted differently per kernel, ulp-off)
  const float conv = metal::fma(w3, xt, metal::fma(w2, h2, metal::fma(w1, h1, w0 * h0)));
  float y = p.add_res ? metal::fma(conv + xt, p.scale, res[c]) : (conv + xt) * p.scale;
  out[c] = y;
  h0 = h1; h1 = h2; h2 = xt;
  if (p.snap_stride) {
    snap[0 * p.C + c] = h0;
    snap[1 * p.C + c] = h1;
    snap[2 * p.C + c] = h2;
  }
  hist[0 * p.C + c] = h0;
  hist[1 * p.C + c] = h1;
  hist[2 * p.C + c] = h2;
}

// ink2_q8_mv1r (M = 1, no add) + the attention short conv + residual on its output channels (lab7: wo_ud's epilogue,
// one stage less per visit). Same per-row sums as ink2_q8_mvm_impl<1, false, R>.
template <uint R>
[[kernel]] void ink2_q8_mv1r_sc(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                                constant Ink2Q8P& p [[buffer(4)]], const device void* cw [[buffer(5)]],
                                device float* hist [[buffer(6)]], device float* cout [[buffer(7)]],
                                constant Ink2SconvStaticP& cp [[buffer(8)]], const device float* res [[buffer(9)]],
                                device float* snap [[buffer(10)]], uint tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint N = p.N, K = p.K;
  const uint n0 = (tg * 4u + sg) * R;
  if (n0 >= N) return;
  const uint mc = min(R, N - n0);
  const uint nb = K / 32u;
  float acc[R];
  for (uint q = 0u; q < R; ++q) acc[q] = 0.0f;
  for (uint k = lane * 8u; k < K; k += 256u) {
    float4 wa[R], wb[R];
    NI_UNROLL for (uint q = 0u; q < R; ++q) {
      const uint r = n0 + min(q, mc - 1u);
      const uint2 v = *(const device uint2*)(wq + size_t(r) * K + k);
      const float s = float(ws[size_t(r) * nb + k / 32u]);
      wa[q] = float4(as_type<char4>(v.x)) * s;
      wb[q] = float4(as_type<char4>(v.y)) * s;
    }
    const float4 xa = *(const device float4*)(x + k);
    const float4 xb = *(const device float4*)(x + k + 4u);
    NI_UNROLL for (uint q = 0u; q < R; ++q) acc[q] += dot(xa, wa[q]) + dot(xb, wb[q]);
  }
  NI_UNROLL for (uint q = 0u; q < R; ++q) {
    const float sv = simd_sum(acc[q]);
    if (lane == q && q < mc) {
      y[n0 + q] = sv;
      ink2_sconv1(sv, cw, hist, cout, cp, res, snap, n0 + q);
    }
  }
}
template [[host_name("ink2_q8_mv1r_sc1")]] [[kernel]] decltype(ink2_q8_mv1r_sc<1>) ink2_q8_mv1r_sc<1>;
template [[host_name("ink2_q8_mv1r_sc2")]] [[kernel]] decltype(ink2_q8_mv1r_sc<2>) ink2_q8_mv1r_sc<2>;
template [[host_name("ink2_q8_mv1r_sc4")]] [[kernel]] decltype(ink2_q8_mv1r_sc<4>) ink2_q8_mv1r_sc<4>;

// ink2_q8_mvm_multi6_1 (6 segments, one row) + the SWA k / v short convs (segments 4, 5: in place, no residual) by the
// lane owning the channel (lab7: one stage less per SWA visit without a message gate). Same sums as the impl at R = 4.
[[kernel]] void ink2_q8_multi6_sckv(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                    const device half* ws [[buffer(2)]], device float* y0 [[buffer(3)]],
                                    constant Ink2Q8P& p [[buffer(4)]], device float* y1 [[buffer(5)]],
                                    device float* y2 [[buffer(6)]], device float* y3 [[buffer(7)]],
                                    device float* y4 [[buffer(8)]], device float* y5 [[buffer(9)]],
                                    const device void* kw [[buffer(10)]], device float* khist [[buffer(11)]],
                                    constant Ink2SconvStaticP& kcp [[buffer(12)]], device float* ksnap [[buffer(13)]],
                                    const device void* vw [[buffer(14)]], device float* vhist [[buffer(15)]],
                                    constant Ink2SconvStaticP& vcp [[buffer(16)]], device float* vsnap [[buffer(17)]],
                                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  const uint N = p.offs[p.n_segs], K = p.K;
  const uint n0 = (tg * 4u + sg) * R;
  if (n0 >= N) return;
  uint seg = 0u;
  while (seg + 1u < p.n_segs && n0 >= p.offs[seg + 1u]) ++seg;
  const uint mc = min(R, p.offs[seg + 1u] - n0);
  const uint c0 = n0 - p.offs[seg];
  device float* dst0 = bases[seg] + c0;
  const uint nb = K / 32u;
  float acc[R];
  for (uint q = 0u; q < R; ++q) acc[q] = 0.0f;
  for (uint k = lane * 8u; k < K; k += 256u) {
    float4 wa[R], wb[R];
    NI_UNROLL for (uint q = 0u; q < R; ++q) {
      const uint r = n0 + min(q, mc - 1u);
      const uint2 v = *(const device uint2*)(wq + size_t(r) * K + k);
      const float s = float(ws[size_t(r) * nb + k / 32u]);
      wa[q] = float4(as_type<char4>(v.x)) * s;
      wb[q] = float4(as_type<char4>(v.y)) * s;
    }
    const float4 xa = *(const device float4*)(x + k);
    const float4 xb = *(const device float4*)(x + k + 4u);
    NI_UNROLL for (uint q = 0u; q < R; ++q) acc[q] += dot(xa, wa[q]) + dot(xb, wb[q]);
  }
  NI_UNROLL for (uint q = 0u; q < R; ++q) {
    const float sv = simd_sum(acc[q]);
    if (lane == q && q < mc) {
      dst0[q] = sv;
      if (seg == 4u) ink2_sconv1(sv, kw, khist, y4, kcp, y4, ksnap, c0 + q);
      else if (seg == 5u) ink2_sconv1(sv, vw, vhist, y5, vcp, y5, vsnap, c0 + q);
    }
  }
}

[[kernel]] void ink2_sconv_static_pf(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                     const device float* hist [[buffer(2)]], device float* out [[buffer(3)]],
                                     constant Ink2SconvStaticP& p [[buffer(4)]], const device float* res [[buffer(5)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint c = tg.x * 128u + tid;
  if (c >= p.C) return;
  float w0, w1, w2, w3;
  if (p.w_f32) {
    const float4 wf = *(const device float4*)((const device float*)w + size_t(c) * 4u);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const device bfloat* wb = (const device bfloat*)w + size_t(c) * 4u;
    w0 = float(wb[0]); w1 = float(wb[1]); w2 = float(wb[2]); w3 = float(wb[3]);
  }
  const uint t0 = tg.y * 16u;
  const uint t1 = min(p.T, t0 + 16u);
  float h0 = (t0 >= 3u) ? x[size_t(t0 - 3u) * p.C + c] : hist[t0 * p.C + c];
  float h1 = (t0 >= 2u) ? x[size_t(t0 - 2u) * p.C + c] : hist[(t0 + 1u) * p.C + c];
  float h2 = (t0 >= 1u) ? x[size_t(t0 - 1u) * p.C + c] : hist[(t0 + 2u) * p.C + c];
  for (uint t = t0; t < t1; ++t) {
    const size_t idx = size_t(t) * p.C + c;
    const float xt = x[idx];
    const float conv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
    float y = (conv + xt) * p.scale;
    if (p.add_res) y += res[idx];
    out[idx] = y;
    h0 = h1; h1 = h2; h2 = xt;
  }
}

struct Ink2SconvDynP {
  uint T, C, n_heads, sw_f32, snap_stride;
};

[[kernel]] void ink2_sconv_dynamic(const device float* x [[buffer(0)]], const device float* dyn [[buffer(1)]],
                                   const device void* static_w [[buffer(2)]], device float* hist [[buffer(3)]],
                                   device float* out [[buffer(4)]], constant Ink2SconvDynP& p [[buffer(5)]],
                                   device float* snap [[buffer(6)]], uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  const uint head_dim = p.C / p.n_heads;
  const uint h = c / head_dim;
  float sw[4];
  for (uint lag = 0; lag < 4; ++lag) {
    const size_t idx = size_t(lag) * p.C + c;
    sw[lag] = p.sw_f32 ? ((const device float*)static_w)[idx] : float(((const device bfloat*)static_w)[idx]);
  }
  float h0 = hist[0 * p.C + c], h1 = hist[1 * p.C + c], h2 = hist[2 * p.C + c];
  for (uint t = 0; t < p.T; ++t) {
    const float xt = x[size_t(t) * p.C + c];
    const float4 d = *(const device float4*)(dyn + (size_t(t) * p.n_heads + h) * 4);
    const float conv = (sw[0] + d.x) * xt + (sw[1] + d.y) * h2 + (sw[2] + d.z) * h1 + (sw[3] + d.w) * h0;
    out[size_t(t) * p.C + c] = (conv + xt) * (1.0f / 3.0f);
    h0 = h1; h1 = h2; h2 = xt;
    if (p.snap_stride) {
      device float* sh = snap + size_t(t) * p.snap_stride;
      sh[0 * p.C + c] = h0;
      sh[1 * p.C + c] = h1;
      sh[2 * p.C + c] = h2;
    }
  }
  hist[0 * p.C + c] = h0;
  hist[1 * p.C + c] = h1;
  hist[2 * p.C + c] = h2;
}

// ---- 7. Q/K/V/R Clip + V Norm + Tau + KV Cache Write ----
struct Ink2QkvrP {
  uint T, Hq, Hkv, D, d_rel, pos0, kv_cap, is_swa, scale_f32;
  float qk_clip, r_clip, v_eps, log_floor, log_alpha;
};

[[kernel]] void ink2_qkvr_prep(device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                               const device float* v [[buffer(2)]], device float* r [[buffer(3)]],
                               const device void* q_scale [[buffer(4)]], const device void* r_scale [[buffer(5)]],
                               device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                               constant Ink2QkvrP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const uint kind_h = tg.x, t = tg.y;
  const uint pos = p.pos0 + t;
  const uint slot = p.is_swa ? (pos % p.kv_cap) : pos;
  const float tau = p.is_swa ? 1.0f : (1.0f + p.log_alpha * metal::precise::log(max(1.0f, float(pos + 1) / p.log_floor)));

  if (kind_h < p.Hq) {
    const uint h = kind_h;
    const size_t off = (size_t(t) * p.Hq + h) * p.D + tid;
    const size_t soff = size_t(h) * p.D + tid;
    const float sc = p.scale_f32 ? ((const device float*)q_scale)[soff] : float(((const device bfloat*)q_scale)[soff]);
    const float val = q[off] * sc;
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float mult = min(1.0f, p.qk_clip / (metal::precise::sqrt(ss) + 1e-6f)) * tau;
    q[off] = val * mult;
    return;
  }
  uint rem = kind_h - p.Hq;
  if (rem < p.Hkv) {
    const uint h = rem;
    const float val = k[(size_t(t) * p.Hkv + h) * p.D + tid];
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float mult = min(1.0f, p.qk_clip / (metal::precise::sqrt(ss) + 1e-6f));
    kc[(size_t(slot) * p.Hkv + h) * p.D + tid] = half(val * mult);
    return;
  }
  rem -= p.Hkv;
  if (rem < p.Hkv) {
    const uint h = rem;
    const float val = v[(size_t(t) * p.Hkv + h) * p.D + tid];
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float inv = metal::precise::rsqrt(ss / float(p.D) + p.v_eps);
    vc[(size_t(slot) * p.Hkv + h) * p.D + tid] = half(val * inv);
    return;
  }
  const uint h = rem - p.Hkv;
  const size_t off = (size_t(t) * p.Hq + h) * p.d_rel + tid;
  const size_t soff = size_t(h) * p.d_rel + tid;
  float val = 0.0f;
  if (tid < p.d_rel) {
    const float sc = p.scale_f32 ? ((const device float*)r_scale)[soff] : float(((const device bfloat*)r_scale)[soff]);
    val = r[off] * sc;
  }
  const float ss = tg_sum(val * val, scratch, tid, p.D);
  if (tid < p.d_rel) {
    const float mult = min(1.0f, p.r_clip / (metal::precise::sqrt(ss) + 1e-6f)) * tau;
    r[off] = val * mult;
  }
}

// ---- 8. Relative Position Logit Bias: rb [T, Hq, rel_extent] = r @ rel_proj ----
struct Ink2RelP {
  uint T, Hq, d_rel, extent, proj_f32;
};

// lab8 (NI_INK2_RBFOLD): ink2_qkvr_prep whose r heads run as J = ceil(extent / D) threadgroups each: every one clips
// the head's r (the same sums) into threadgroup memory and computes D columns of its rel bias row (ink2_rel_bias's
// loop, one column per thread). r is not written back (rel bias is its only reader). One stage less per visit, same
// values as the two kernels. (One threadgroup per head doing all 1024 columns: +2.5 ms/token, latency-bound.)
[[kernel]] void ink2_qkvr_prep_rb(device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                                  const device float* v [[buffer(2)]], device float* r [[buffer(3)]],
                                  const device void* q_scale [[buffer(4)]], const device void* r_scale [[buffer(5)]],
                                  device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                                  constant Ink2QkvrP& p [[buffer(8)]], const device void* rel_proj [[buffer(9)]],
                                  device float* rb [[buffer(10)]], constant Ink2RelP& rp [[buffer(11)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float rs[128];
  const uint kind_h = tg.x, t = tg.y;
  const uint pos = p.pos0 + t;
  const uint slot = p.is_swa ? (pos % p.kv_cap) : pos;
  const float tau = p.is_swa ? 1.0f : (1.0f + p.log_alpha * metal::precise::log(max(1.0f, float(pos + 1) / p.log_floor)));

  if (kind_h < p.Hq) {
    const uint h = kind_h;
    const size_t off = (size_t(t) * p.Hq + h) * p.D + tid;
    const size_t soff = size_t(h) * p.D + tid;
    const float sc = p.scale_f32 ? ((const device float*)q_scale)[soff] : float(((const device bfloat*)q_scale)[soff]);
    const float val = q[off] * sc;
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float mult = min(1.0f, p.qk_clip / (metal::precise::sqrt(ss) + 1e-6f)) * tau;
    q[off] = val * mult;
    return;
  }
  uint rem = kind_h - p.Hq;
  if (rem < p.Hkv) {
    const uint h = rem;
    const float val = k[(size_t(t) * p.Hkv + h) * p.D + tid];
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float mult = min(1.0f, p.qk_clip / (metal::precise::sqrt(ss) + 1e-6f));
    kc[(size_t(slot) * p.Hkv + h) * p.D + tid] = half(val * mult);
    return;
  }
  rem -= p.Hkv;
  if (rem < p.Hkv) {
    const uint h = rem;
    const float val = v[(size_t(t) * p.Hkv + h) * p.D + tid];
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float inv = metal::precise::rsqrt(ss / float(p.D) + p.v_eps);
    vc[(size_t(slot) * p.Hkv + h) * p.D + tid] = half(val * inv);
    return;
  }
  const uint J = (rp.extent + p.D - 1u) / p.D;
  const uint h = (rem - p.Hkv) / J, col = ((rem - p.Hkv) % J) * p.D + tid;
  const size_t off = (size_t(t) * p.Hq + h) * p.d_rel + tid;
  const size_t soff = size_t(h) * p.d_rel + tid;
  float val = 0.0f;
  if (tid < p.d_rel) {
    const float sc = p.scale_f32 ? ((const device float*)r_scale)[soff] : float(((const device bfloat*)r_scale)[soff]);
    val = r[off] * sc;
  }
  const float ss = tg_sum(val * val, scratch, tid, p.D);
  if (tid < p.d_rel) {
    const float mult = min(1.0f, p.r_clip / (metal::precise::sqrt(ss) + 1e-6f)) * tau;
    rs[tid] = val * mult;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (col < rp.extent) {
    float acc = 0.0f;
    for (uint d = 0; d < rp.d_rel; ++d) {
      const size_t pidx = size_t(d) * rp.extent + col;
      const float pv = rp.proj_f32 ? ((const device float*)rel_proj)[pidx] : float(((const device bfloat*)rel_proj)[pidx]);
      acc += rs[d] * pv;
    }
    rb[(size_t(t) * rp.Hq + h) * rp.extent + col] = acc;
  }
}

[[kernel]] void ink2_rel_bias(const device float* r [[buffer(0)]], const device void* rel_proj [[buffer(1)]],
                              device float* rb [[buffer(2)]], constant Ink2RelP& p [[buffer(3)]],
                              uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = tg.x * 32 + tid;
  if (col >= p.extent) return;
  const uint h = tg.y, t = tg.z;
  const device float* rv = r + (size_t(t) * p.Hq + h) * p.d_rel;
  float acc = 0.0f;
  for (uint d = 0; d < p.d_rel; ++d) {
    const size_t pidx = size_t(d) * p.extent + col;
    const float pv = p.proj_f32 ? ((const device float*)rel_proj)[pidx] : float(((const device bfloat*)rel_proj)[pidx]);
    acc += rv[d] * pv;
  }
  rb[(size_t(t) * p.Hq + h) * p.extent + col] = acc;
}

[[kernel]] void ink2_rel_bias_pf(const device float* r [[buffer(0)]], const device void* rel_proj [[buffer(1)]],
                                  device float* rb [[buffer(2)]], constant Ink2RelP& p [[buffer(3)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 sr[64][4];
  const uint col = tg.x * 32u + lane;
  const bool col_ok = col < p.extent;
  const uint c_safe = col_ok ? col : (p.extent - 1u);
  float4 p0, p1, p2, p3;
  if (p.proj_f32) {
    const device float* rp = (const device float*)rel_proj + c_safe;
    p0 = float4(rp[0 * p.extent], rp[1 * p.extent], rp[2 * p.extent], rp[3 * p.extent]);
    p1 = float4(rp[4 * p.extent], rp[5 * p.extent], rp[6 * p.extent], rp[7 * p.extent]);
    p2 = float4(rp[8 * p.extent], rp[9 * p.extent], rp[10 * p.extent], rp[11 * p.extent]);
    p3 = float4(rp[12 * p.extent], rp[13 * p.extent], rp[14 * p.extent], rp[15 * p.extent]);
  } else {
    const device bfloat* rp = (const device bfloat*)rel_proj + c_safe;
    p0 = float4(float(rp[0 * p.extent]), float(rp[1 * p.extent]), float(rp[2 * p.extent]), float(rp[3 * p.extent]));
    p1 = float4(float(rp[4 * p.extent]), float(rp[5 * p.extent]), float(rp[6 * p.extent]), float(rp[7 * p.extent]));
    p2 = float4(float(rp[8 * p.extent]), float(rp[9 * p.extent]), float(rp[10 * p.extent]), float(rp[11 * p.extent]));
    p3 = float4(float(rp[12 * p.extent]), float(rp[13 * p.extent]), float(rp[14 * p.extent]), float(rp[15 * p.extent]));
  }
  const uint t0 = tg.y * 8u, h0 = tg.z * 8u;
  for (uint idx = tid; idx < 256u; idx += 128u) {
    const uint row = idx >> 2u, q = idx & 3u;
    const uint t = t0 + (row >> 3u), h = h0 + (row & 7u);
    sr[row][q] = (t < p.T && h < p.Hq) ? *(const device float4*)(r + (size_t(t) * p.Hq + h) * 16u + q * 4u) : float4(0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!col_ok) return;
  for (uint dt = 0u; dt < 2u; ++dt) {
    const uint t = t0 + sg * 2u + dt;
    if (t >= p.T) break;
    const uint row_base = (sg * 2u + dt) << 3u;
    device float* rbt = rb + size_t(t) * p.Hq * p.extent + col;
    for (uint hh = 0u; hh < 8u && (h0 + hh) < p.Hq; ++hh) {
      const uint row = row_base + hh;
      const float4 r0 = sr[row][0], r1 = sr[row][1], r2 = sr[row][2], r3 = sr[row][3];
      float acc = 0.0f;
      for (uint d = 0u; d < 4u; ++d) acc += r0[d] * p0[d];
      for (uint d = 0u; d < 4u; ++d) acc += r1[d] * p1[d];
      for (uint d = 0u; d < 4u; ++d) acc += r2[d] * p2[d];
      for (uint d = 0u; d < 4u; ++d) acc += r3[d] * p3[d];
      rbt[size_t(h0 + hh) * p.extent] = acc;
    }
  }
}

// ---- 9. Sink Attention + Rel Bias + Gate Multiply (Split-K Decode & Prefill) ----
struct Ink2AttnDecP {
  uint pos, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks;
  float scale;
};

[[kernel]] void ink2_attn_decode(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                 const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                 device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                 device float* po [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, cid = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4;
  const float4 qv = *(const device float4*)(q + size_t(h) * p.D + d0);

  const uint lk = p.is_swa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;
  const uint c0 = k_first + cid * p.chunk;
  const uint c1 = min(p.pos + 1u, c0 + p.chunk);

  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);
  for (uint kp = c0 + sg; kp < c1; kp += 4) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = float4(*(const device half4*)(kc + koff));
    float sc = simd_sum(dot(qv, kv)) * p.scale;
    const uint dist = p.pos - kp;
    if (dist < p.rel_extent) sc += rb[size_t(h) * p.rel_extent + dist];
    const float4 vv = float4(*(const device half4*)(vc + koff));
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = max(mw, sc);
      const float e1 = metal::precise::exp(mw - nm), e2 = metal::precise::exp(sc - nm);
      ow = ow * e1 + vv * e2;
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0) {
    tgm[sg] = mw;
    tgl[sg] = lw;
  }
  *(threadgroup float4*)(&tgo[sg][d0]) = ow;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0; s < 4; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *(const threadgroup float4*)(&tgo[s][d0]);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = max(mm, m2);
        const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
        oo = oo * e1 + o2 * e2;
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const size_t out_idx = size_t(h) * p.nchunks + cid;
    if (lane == 0) {
      pm[out_idx] = mm;
      pl[out_idx] = ll;
    }
    *(device float4*)(po + out_idx * p.D + d0) = oo;
  }
}

// GQA split-K decode (3 query heads per KV head, D = 128): one threadgroup per (KV head, key chunk) reads each K / V
// row once for its 3 heads; each simdgroup takes blocks of 8 keys (8 independent loads in flight, one softmax rescale
// per block). Partials in ink2_attn_decode's [h][nchunks] format, so ink2_attn_decode_reduce combines them.
template <uint KB, bool FX>
NI_INLINE void ink2_attn_decode_g3_impl(const device float* q, const device half* kc,
                                    const device half* vc, const device float* rb,
                                    device float* pm, device float* pl,
                                    device float* po, constant Ink2AttnDecP& p,
                                    uint2 tg, uint sg, uint lane, threadgroup float* tgb) {
  constexpr uint G = 3;
  const uint hkv = tg.x, cid = tg.y, h0 = hkv * G, d0 = lane * 4;
  float4 qv[G];
  for (uint g = 0; g < G; ++g) qv[g] = *(const device float4*)(q + size_t(h0 + g) * p.D + d0) * p.scale;
  const uint lk = p.is_swa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;
  const uint c0 = k_first + cid * p.chunk;
  const uint c1 = min(p.pos + 1u, c0 + p.chunk);
  float mw[G], lw[G];
  float4 ow[G];
  for (uint g = 0; g < G; ++g) { mw[g] = -INFINITY; lw[g] = 0.0f; ow[g] = float4(0.0f); }
  for (uint kb = c0 + sg * KB; kb < c1; kb += 4 * KB) {
    float4 kv[KB], vv[KB];
    for (uint j = 0; j < KB; ++j) {
      const uint kp = min(kb + j, c1 - 1u);
      const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
      const size_t off = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
      kv[j] = float4(*(const device half4*)(kc + off));
      vv[j] = float4(*(const device half4*)(vc + off));
    }
    for (uint g = 0; g < G; ++g) {
      float s[KB];
      float bm = -INFINITY;
      for (uint j = 0; j < KB; ++j) {
        const uint kp = kb + j;
        float sc = simd_sum(dot(qv[g], kv[j]));
        const uint dist = p.pos - kp;
        if (dist < p.rel_extent) sc += rb[size_t(h0 + g) * p.rel_extent + dist];
        s[j] = kp < c1 ? sc : -INFINITY;
        bm = max(bm, s[j]);
      }
      const float nm = max(mw[g], bm);
      const float corr = FX ? fast::exp(mw[g] - nm) : metal::precise::exp(mw[g] - nm);
      float4 o = ow[g] * corr;
      float l = lw[g] * corr;
      for (uint j = 0; j < KB; ++j) {
        const float e = FX ? fast::exp(s[j] - nm) : metal::precise::exp(s[j] - nm);
        o += vv[j] * e;
        l += e;
      }
      ow[g] = o; lw[g] = l; mw[g] = nm;
    }
  }
  for (uint g = 0; g < G; ++g) {
    if (lane == 0) { tgb[sg * G + g] = mw[g]; tgb[12 + sg * G + g] = lw[g]; }
    *(threadgroup float4*)(tgb + 24 + (sg * G + g) * 128 + d0) = ow[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < G) {
    const uint g = sg;
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0; s < 4; ++s) {
      const float l2 = tgb[12 + s * G + g];
      if (l2 == 0.0f) continue;
      const float m2 = tgb[s * G + g];
      const float4 o2 = *(const threadgroup float4*)(tgb + 24 + (s * G + g) * 128 + d0);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = max(mm, m2);
        const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
        oo = oo * e1 + o2 * e2;
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const size_t out_idx = size_t(h0 + g) * p.nchunks + cid;
    if (lane == 0) { pm[out_idx] = mm; pl[out_idx] = ll; }
    *(device float4*)(po + out_idx * p.D + d0) = oo;
  }
}

#define INK2_DEC_G3(NAME, KB, FX)                                                                                     \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],                        \
                       const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],                     \
                       device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], device float* po [[buffer(6)]], \
                       constant Ink2AttnDecP& p [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],             \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {          \
    threadgroup float tgb[24 + 12 * 128];                                                                              \
    ink2_attn_decode_g3_impl<KB, FX>(q, kc, vc, rb, pm, pl, po, p, tg, sg, lane, tgb);                                     \
  }
INK2_DEC_G3(ink2_attn_decode_g3, 8, false)
INK2_DEC_G3(ink2_attn_decode_g3_k16, 16, false)
INK2_DEC_G3(ink2_attn_decode_g3_fx, 8, true)
INK2_DEC_G3(ink2_attn_decode_g3_k4, 4, false)
INK2_DEC_G3(ink2_attn_decode_g3_k4fx, 4, true)
INK2_DEC_G3(ink2_attn_decode_g3_k6fx, 6, true)

// GQA split-K decode, keys across lanes (round 3): for QK each lane takes one key of a 32-key block and dots all 128
// dims for the 3 query heads (q in threadgroup memory, 16-byte K loads; no simd_sum per (key, head)); for PV lanes
// take 4 dims and the weights are shuffled from the key's lane. Same partials as ink2_attn_decode_g3.
// decbench 84K global: g3_fx 1.60 -> 1.20 ms a visit (raw K+V read 0.93); 40K 0.80 -> 0.59.
NI_INLINE void ink2_st4(device float* a, float4 v) { *(device float4*)a = v; }
#if __METAL_VERSION__ >= 320
NI_INLINE void ink2_st4(device coherent(device) float* a, float4 v) { *(device coherent(device) float4*)a = v; }
#endif
// (PT: the partials' pointer type; lab11 REDLAST 3 passes device coherent(device) float*)
template <uint VB, bool AT = false, typename PT = device float*>
NI_INLINE void ink2_attn_decode_g3kw_impl(const device float* q, const device half* kc, const device half* vc,
                                        const device float* rb, PT pm, PT pl, PT po,
                                        constant Ink2AttnDecP& p, uint2 tg, uint sg, uint lane, uint tid,
                                        threadgroup float* tgb, threadgroup float4* qs) {
  constexpr uint G = 3;
  const uint hkv = tg.x, cid = tg.y, h0 = hkv * G, d0 = lane * 4;
  for (uint i = tid; i < G * 32u; i += 128u) qs[i] = *(const device float4*)(q + size_t(h0) * p.D + i * 4u) * p.scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint lk = p.is_swa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;
  const uint c0 = k_first + cid * p.chunk;
  const uint c1 = min(p.pos + 1u, c0 + p.chunk);
  float mw[G], lw[G];
  float4 ow[G];
  for (uint g = 0; g < G; ++g) { mw[g] = -INFINITY; lw[g] = 0.0f; ow[g] = float4(0.0f); }
  for (uint kb = c0 + sg * 32u; kb < c1; kb += 128u) {
    const uint kp = kb + lane;
    const uint kpc = min(kp, c1 - 1u);
    const uint slot = p.is_swa ? (kpc % p.kv_cap) : kpc;
    const device uint4* kr = (const device uint4*)(kc + (size_t(slot) * p.Hkv + hkv) * p.D);
    float s[G] = {0.0f, 0.0f, 0.0f};
    for (uint c = 0; c < 32u; c += 2u) {
      const uint4 kk = kr[c >> 1];
      const float4 k0 = float4(as_type<half4>(kk.xy)), k1 = float4(as_type<half4>(kk.zw));
      for (uint g = 0; g < G; ++g) s[g] += dot(k0, qs[g * 32u + c]) + dot(k1, qs[g * 32u + c + 1u]);
    }
    const uint dist = p.pos - kp;
    float e[G];
    for (uint g = 0; g < G; ++g) {
      float sc = s[g];
      if (dist < p.rel_extent && kp < c1) sc += rb[size_t(h0 + g) * p.rel_extent + dist];
      sc = kp < c1 ? sc : -INFINITY;
      const float nm = max(mw[g], simd_max(sc));
      const float corr = fast::exp(mw[g] - nm);
      e[g] = fast::exp(sc - nm);
      ow[g] *= corr;
      lw[g] = lw[g] * corr + e[g];
      mw[g] = nm;
    }
    const uint nk = min(32u, c1 - kb);
    for (uint j0 = 0; j0 < nk; j0 += VB) {
      half4 vv[VB];
      for (uint j = 0; j < VB; ++j) {
        const uint vp = min(kb + j0 + j, c1 - 1u);
        const uint vs = p.is_swa ? (vp % p.kv_cap) : vp;
        vv[j] = *(const device half4*)(vc + (size_t(vs) * p.Hkv + hkv) * p.D + d0);
      }
      for (uint j = 0; j < VB; ++j) {
        const float4 v = float4(vv[j]);
        for (uint g = 0; g < G; ++g) ow[g] += v * simd_shuffle(e[g], ushort(j0 + j));  // (e = 0 past c1)
      }
    }
  }
  for (uint g = 0; g < G; ++g) {
    const float l = simd_sum(lw[g]);
    if (lane == 0) { tgb[sg * G + g] = mw[g]; tgb[12 + sg * G + g] = l; }
    *(threadgroup float4*)(tgb + 24 + (sg * G + g) * 128 + d0) = ow[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < G) {
    const uint g = sg;
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = float4(0.0f);
    for (uint s2 = 0; s2 < 4; ++s2) {
      const float l2 = tgb[12 + s2 * G + g];
      if (l2 == 0.0f) continue;
      const float m2 = tgb[s2 * G + g];
      const float4 o2 = *(const threadgroup float4*)(tgb + 24 + (s2 * G + g) * 128 + d0);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = max(mm, m2);
        const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
        oo = oo * e1 + o2 * e2;
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const size_t out_idx = size_t(h0 + g) * p.nchunks + cid;
    if constexpr (AT) {  // (g3kw_r: through device atomics, for the last threadgroup's reduce)
      device atomic_uint* am = (device atomic_uint*)pm;
      device atomic_uint* al = (device atomic_uint*)pl;
      device atomic_uint* ao = (device atomic_uint*)(po + out_idx * p.D + d0);
      if (lane == 0) {
        atomic_store_explicit(am + out_idx, as_type<uint>(mm), memory_order_relaxed);
        atomic_store_explicit(al + out_idx, as_type<uint>(ll), memory_order_relaxed);
      }
      atomic_store_explicit(ao + 0, as_type<uint>(oo.x), memory_order_relaxed);
      atomic_store_explicit(ao + 1, as_type<uint>(oo.y), memory_order_relaxed);
      atomic_store_explicit(ao + 2, as_type<uint>(oo.z), memory_order_relaxed);
      atomic_store_explicit(ao + 3, as_type<uint>(oo.w), memory_order_relaxed);
    } else {
      if (lane == 0) { pm[out_idx] = mm; pl[out_idx] = ll; }
      ink2_st4(po + out_idx * p.D + d0, oo);
    }
  }
}
#define INK2_DEC_G3KW(NAME, VB)                                                                                         \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],                        \
                       const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],                     \
                       device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], device float* po [[buffer(6)]], \
                       constant Ink2AttnDecP& p [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],             \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       uint tid [[thread_index_in_threadgroup]]) {                                                    \
    threadgroup float tgb[24 + 12 * 128];                                                                              \
    threadgroup float4 qs[3 * 32];                                                                                     \
    ink2_attn_decode_g3kw_impl<VB>(q, kc, vc, rb, pm, pl, po, p, tg, sg, lane, tid, tgb, qs);                          \
  }
INK2_DEC_G3KW(ink2_attn_decode_g3kw, 4)

#if __METAL_VERSION__ >= 320
// (the partials of the other threadgroups read through device atomics: plain loads in the last threadgroup read stale
// values on the M1 Max although behind the arrival fences)
NI_INLINE void ink2_attn_reduce_par(const device float* pm, const device float* pl, const device float* po,
                                    float sink, const device float* gate, device float* out, size_t base, uint nchunks,
                                    uint D, uint lane, threadgroup float* wsh);
NI_INLINE float ink2_ld_at(const device float* a) {
  return as_type<float>(atomic_load_explicit((device atomic_uint*)a, memory_order_relaxed));
}
NI_INLINE void ink2_attn_reduce_par_at(const device float* pm, const device float* pl, const device float* po,
                                       float sink, const device float* gate, device float* out, size_t base,
                                       uint nchunks, uint D, uint lane, threadgroup float* wsh) {
  const uint d0 = lane * 4;
  float mloc = -INFINITY;
  for (uint c = lane; c < nchunks; c += 32u)
    if (ink2_ld_at(pl + base + c) != 0.0f) mloc = max(mloc, ink2_ld_at(pm + base + c));
  const float mx = max(simd_max(mloc), sink);
  float lloc = 0.0f;
  for (uint c = lane; c < nchunks; c += 32u) {
    const float l2 = ink2_ld_at(pl + base + c);
    const float w = l2 != 0.0f ? metal::precise::exp(ink2_ld_at(pm + base + c) - mx) : 0.0f;
    wsh[c] = w;
    lloc += w * l2;
  }
  const float ll = simd_sum(lloc) + metal::precise::exp(sink - mx);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float4 oo = float4(0.0f);
  for (uint c = 0; c < nchunks; ++c) {
    const device float* pc = po + (base + c) * D + d0;
    oo += wsh[c] * float4(ink2_ld_at(pc), ink2_ld_at(pc + 1), ink2_ld_at(pc + 2), ink2_ld_at(pc + 3));
  }
  const float4 g4 = *(const device float4*)(gate + d0);
  const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                           metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
  *(device float4*)(out + d0) = (oo / ll) * sig;
}
// lab8 (NI_INK2_REDLAST): ink2_attn_decode_g3kw whose last threadgroup to finish a KV head (Nemotron-H's arrival
// order, flint_arrive) runs ink2_attn_decode_reduce_p for the head's 3 query heads, one simdgroup each (the reduce's
// lane mapping, so the same sums): one stage less per visit. cnt[hkv] returns to 0 for the next dispatch.
template <bool READ_AT>
[[kernel]] void ink2_attn_decode_g3kw_r(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                        const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                        device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                        device float* po [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                        const device float* sinks [[buffer(8)]], const device float* gate [[buffer(9)]],
                                        device float* out [[buffer(10)]], device atomic_uint* cnt [[buffer(11)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tgb[24 + 12 * 128];
  threadgroup float4 qs[3 * 32];
  threadgroup uint last;
  ink2_attn_decode_g3kw_impl<4, true>(q, kc, vc, rb, pm, pl, po, p, tg, sg, lane, tid, tgb, qs);
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    last = atomic_fetch_add_explicit(cnt + tg.x, 1u, memory_order_relaxed) == p.nchunks - 1u ? 1u : 0u;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (last == 0u) return;
  if (sg < 3u) {
    const uint h = tg.x * 3u + sg;
    if (READ_AT)
      ink2_attn_reduce_par_at(pm, pl, po, sinks[h], gate + size_t(h) * p.D, out + size_t(h) * p.D,
                              size_t(h) * p.nchunks, p.nchunks, p.D, lane, tgb + sg * 128u);
    else
      ink2_attn_reduce_par(pm, pl, po, sinks[h], gate + size_t(h) * p.D, out + size_t(h) * p.D, size_t(h) * p.nchunks,
                           p.nchunks, p.D, lane, tgb + sg * 128u);
  }
  if (tid == 0) atomic_store_explicit(cnt + tg.x, 0u, memory_order_relaxed);
}
template [[host_name("ink2_attn_decode_g3kw_r")]] [[kernel]] decltype(ink2_attn_decode_g3kw_r<true>) ink2_attn_decode_g3kw_r<true>;
template [[host_name("ink2_attn_decode_g3kw_r0")]] [[kernel]] decltype(ink2_attn_decode_g3kw_r<false>) ink2_attn_decode_g3kw_r<false>;
// lab11 (NI_INK2_REDLAST=3): the same last-threadgroup reduce with the partials as device coherent(device) memory and
// thread_scope_device fences on both sides of the arrival count (the consult's pattern; lab11/coh stress test: only
// coherent + fences passes, 0 bad of 10^5 runs also under GPU load; plain / coherent only / fence only fail): plain
// float4 stores and loads, ink2_attn_reduce_par's sums verbatim.
NI_INLINE void ink2_attn_reduce_par_c(device coherent(device) float* pm, device coherent(device) float* pl,
                                      device coherent(device) float* po, float sink, const device float* gate,
                                      device float* out, size_t base, uint nchunks, uint D, uint lane,
                                      threadgroup float* wsh) {
  const uint d0 = lane * 4;
  float mloc = -INFINITY;
  for (uint c = lane; c < nchunks; c += 32u)
    if (pl[base + c] != 0.0f) mloc = max(mloc, pm[base + c]);
  const float mx = max(simd_max(mloc), sink);
  float lloc = 0.0f;
  for (uint c = lane; c < nchunks; c += 32u) {
    const float l2 = pl[base + c];
    const float w = l2 != 0.0f ? metal::precise::exp(pm[base + c] - mx) : 0.0f;
    wsh[c] = w;
    lloc += w * l2;
  }
  const float ll = simd_sum(lloc) + metal::precise::exp(sink - mx);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float4 oo = float4(0.0f);
  for (uint c = 0; c < nchunks; ++c) oo += wsh[c] * *(device coherent(device) float4*)(po + (base + c) * D + d0);
  const float4 g4 = *(const device float4*)(gate + d0);
  const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                           metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
  *(device float4*)(out + d0) = (oo / ll) * sig;
}
[[kernel]] void ink2_attn_decode_g3kw_rc(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                         const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                         device coherent(device) float* pm [[buffer(4)]],
                                         device coherent(device) float* pl [[buffer(5)]],
                                         device coherent(device) float* po [[buffer(6)]],
                                         constant Ink2AttnDecP& p [[buffer(7)]], const device float* sinks [[buffer(8)]],
                                         const device float* gate [[buffer(9)]], device float* out [[buffer(10)]],
                                         device atomic_uint* cnt [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],
                                         uint sg [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tgb[24 + 12 * 128];
  threadgroup float4 qs[3 * 32];
  threadgroup uint last;
  ink2_attn_decode_g3kw_impl<4, false, device coherent(device) float*>(q, kc, vc, rb, pm, pl, po, p, tg, sg, lane, tid,
                                                                       tgb, qs);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);  // every writer
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0) {
    last = atomic_fetch_add_explicit(cnt + tg.x, 1u, memory_order_relaxed) == p.nchunks - 1u ? 1u : 0u;
    if (last) atomic_store_explicit(cnt + tg.x, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last == 0u) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (sg < 3u) {
    const uint h = tg.x * 3u + sg;
    ink2_attn_reduce_par_c(pm, pl, po, sinks[h], gate + size_t(h) * p.D, out + size_t(h) * p.D, size_t(h) * p.nchunks,
                           p.nchunks, p.D, lane, tgb + sg * 128u);
  }
}
#endif

// ink2_attn_decode_g3 with a transpose-reduce: per 8-key block a lane holds the 24 partial (head, key) dots of its 4
// dims; 5 halving shuffle steps (31 shuffles, not 24 simd_sums) leave lane L = 8 g + j with head g's score of key j,
// so the softmax runs once per (head, key) (fast::exp; the scores are fp32) and the 3 x 8 weights are broadcast for PV.
[[kernel]] void ink2_attn_decode_g3t(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                     const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                     device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                     device float* po [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint G = 3, KB = 8;
  threadgroup float tgm[4][G], tgl[4][G], tgo[4][G][128];
  const uint hkv = tg.x, cid = tg.y, h0 = hkv * G, d0 = lane * 4;
  const uint mg = min(lane >> 3, G - 1u), mj = lane & 7u;  // this lane's (head, key) after the reduce
  float4 qv[G];
  for (uint g = 0; g < G; ++g) qv[g] = *(const device float4*)(q + size_t(h0 + g) * p.D + d0) * p.scale;
  const uint lk = p.is_swa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;
  const uint c0 = k_first + cid * p.chunk;
  const uint c1 = min(p.pos + 1u, c0 + p.chunk);
  float m_run = -INFINITY, l_run = 0.0f;  // head mg's running max / sum (all 8 lanes of the group agree)
  float4 ow[G];
  for (uint g = 0; g < G; ++g) ow[g] = float4(0.0f);
  for (uint kb = c0 + sg * KB; kb < c1; kb += 4 * KB) {
    float4 vv[KB];
    float v[32];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < KB; ++j) {
      const uint kp = min(kb + j, c1 - 1u);
      const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
      const size_t off = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
      const float4 kv = float4(*(const device half4*)(kc + off));
      vv[j] = float4(*(const device half4*)(vc + off));
      _Pragma("clang loop unroll(full)")
      for (uint g = 0; g < G; ++g) v[g * KB + j] = dot(qv[g], kv);
    }
    _Pragma("clang loop unroll(full)")
    for (uint i = G * KB; i < 32; ++i) v[i] = 0.0f;
    // reduce-scatter: after mask n the lane keeps the half selected by its bit n
    _Pragma("clang loop unroll(full)")
    for (int sh = 4; sh >= 0; --sh) {
      const uint n = 1u << sh;
      const bool up = (lane & n) != 0;
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < n; ++i) {
        const float mine = up ? v[n + i] : v[i];
        const float send = up ? v[i] : v[n + i];
        v[i] = mine + simd_shuffle_xor(send, ushort(n));
      }
    }
    const uint kp = kb + mj;
    float sc = v[0];
    const uint dist = p.pos - kp;
    if (dist < p.rel_extent) sc += rb[size_t(h0 + mg) * p.rel_extent + dist];
    if (kp >= c1 || lane >= G * KB) sc = -INFINITY;
    float bm = sc;
    bm = max(bm, simd_shuffle_xor(bm, ushort(1)));
    bm = max(bm, simd_shuffle_xor(bm, ushort(2)));
    bm = max(bm, simd_shuffle_xor(bm, ushort(4)));
    const float nm = max(m_run, bm);
    const float corr = fast::exp(m_run - nm);
    const float e = sc == -INFINITY ? 0.0f : fast::exp(sc - nm);
    float es = e;
    es += simd_shuffle_xor(es, ushort(1));
    es += simd_shuffle_xor(es, ushort(2));
    es += simd_shuffle_xor(es, ushort(4));
    l_run = l_run * corr + es;
    m_run = nm;
    _Pragma("clang loop unroll(full)")
    for (uint g = 0; g < G; ++g) {
      float4 o = ow[g] * simd_shuffle(corr, ushort(g * KB));
      _Pragma("clang loop unroll(full)")
      for (uint j = 0; j < KB; ++j) o += vv[j] * simd_shuffle(e, ushort(g * KB + j));
      ow[g] = o;
    }
  }
  for (uint g = 0; g < G; ++g) {
    const float mg_ = simd_shuffle(m_run, ushort(g * KB)), lg_ = simd_shuffle(l_run, ushort(g * KB));
    if (lane == 0) { tgm[sg][g] = mg_; tgl[sg][g] = lg_; }
    *(threadgroup float4*)(&tgo[sg][g][d0]) = ow[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < G) {
    const uint g = sg;
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0; s < 4; ++s) {
      const float l2 = tgl[s][g];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s][g];
      const float4 o2 = *(const threadgroup float4*)(&tgo[s][g][d0]);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = max(mm, m2);
        const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
        oo = oo * e1 + o2 * e2;
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const size_t out_idx = size_t(h0 + g) * p.nchunks + cid;
    if (lane == 0) { pm[out_idx] = mm; pl[out_idx] = ll; }
    *(device float4*)(po + out_idx * p.D + d0) = oo;
  }
}

[[kernel]] void ink2_attn_decode_reduce(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                        const device float* po [[buffer(2)]], const device float* sinks [[buffer(3)]],
                                        const device float* gate [[buffer(4)]], device float* out [[buffer(5)]],
                                        constant Ink2AttnDecP& p [[buffer(6)]], uint h [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint d0 = lane * 4;
  float mm = sinks[h], ll = 1.0f;
  float4 oo = float4(0.0f);
  for (uint cid = 0; cid < p.nchunks; ++cid) {
    const size_t idx = size_t(h) * p.nchunks + cid;
    const float l2 = pl[idx];
    if (l2 == 0.0f) continue;
    const float m2 = pm[idx];
    const float4 o2 = *(const device float4*)(po + idx * p.D + d0);
    const float nm = max(mm, m2);
    const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
    oo = oo * e1 + o2 * e2;
    ll = ll * e1 + l2 * e2;
    mm = nm;
  }
  const float4 g4 = *(const device float4*)(gate + size_t(h) * p.D + d0);
  const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                           metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
  *(device float4*)(out + size_t(h) * p.D + d0) = (oo / ll) * sig;
}

// Split-K combine without the serial exp chain: one max over all chunks (simd_max), then per-chunk weights in
// parallel and one weighted sum (ink2_attn_decode_reduce runs nchunks dependent rescale steps: ~40 us a call).
NI_INLINE void ink2_attn_reduce_par(const device float* pm, const device float* pl, const device float* po,
                                    float sink, const device float* gate, device float* out, size_t base, uint nchunks,
                                    uint D, uint lane, threadgroup float* wsh) {
  const uint d0 = lane * 4;
  float mloc = -INFINITY;
  for (uint c = lane; c < nchunks; c += 32u)
    if (pl[base + c] != 0.0f) mloc = max(mloc, pm[base + c]);
  const float mx = max(simd_max(mloc), sink);
  float lloc = 0.0f;
  for (uint c = lane; c < nchunks; c += 32u) {
    const float l2 = pl[base + c];
    const float w = l2 != 0.0f ? metal::precise::exp(pm[base + c] - mx) : 0.0f;
    wsh[c] = w;
    lloc += w * l2;
  }
  const float ll = simd_sum(lloc) + metal::precise::exp(sink - mx);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float4 oo = float4(0.0f);
  for (uint c = 0; c < nchunks; ++c) oo += wsh[c] * *(const device float4*)(po + (base + c) * D + d0);
  const float4 g4 = *(const device float4*)(gate + d0);
  const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                           metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
  *(device float4*)(out + d0) = (oo / ll) * sig;
}

[[kernel]] void ink2_attn_decode_reduce_p(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                          const device float* po [[buffer(2)]], const device float* sinks [[buffer(3)]],
                                          const device float* gate [[buffer(4)]], device float* out [[buffer(5)]],
                                          constant Ink2AttnDecP& p [[buffer(6)]], uint h [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float wsh[128];
  ink2_attn_reduce_par(pm, pl, po, sinks[h], gate + size_t(h) * p.D, out + size_t(h) * p.D, size_t(h) * p.nchunks,
                       p.nchunks, p.D, lane, wsh);
}

// Verify rows (T = 2..8) as split-K decode: one threadgroup per (KV head, key chunk, group of TR rows) reads each K / V
// row once for its TR rows x 3 heads (ink2_attn_prefill re-reads the context per row and head: 389 us a call at T=2,
// 2k context). q in threadgroup memory, 4-key blocks per simdgroup, per-row causal / sliding-window masks.
// Partials [t][h][nchunks] for ink2_attn_decode_reduce_t.
struct Ink2AttnVerP {
  uint pos0, T, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, k_first, k_end, chunk, nchunks;
  float scale;
};

template <uint TR>
[[kernel]] void ink2_attn_verify_g3(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                    const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                    device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                    device float* po [[buffer(6)]], constant Ink2AttnVerP& p [[buffer(7)]],
                                    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint G = 3, KB = 4, NQ = TR * G;
  threadgroup float tq[NQ * 128];
  threadgroup float tml[4 * NQ * 2];
  threadgroup float to[4 * NQ * 128];
  const uint hkv = tg.x, cid = tg.y, t0 = tg.z * TR, h0 = hkv * G, d0 = lane * 4;
  const uint nr = min(TR, p.T - t0);
  for (uint i = tid; i < NQ * 32u; i += 128u) {
    const uint qi = i / 32u, c = (i % 32u) * 4u, t = qi / G, g = qi % G;
    const float4 v = t < nr ? *(const device float4*)(q + (size_t(t0 + t) * p.Hq + h0 + g) * p.D + c) * p.scale
                            : float4(0.0f);
    *(threadgroup float4*)(tq + qi * 128u + c) = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c0 = p.k_first + cid * p.chunk;
  const uint c1 = min(p.k_end, c0 + p.chunk);
  float mw[NQ], lw[NQ];
  float4 ow[NQ];
  NI_UNROLL for (uint i = 0; i < NQ; ++i) { mw[i] = -INFINITY; lw[i] = 0.0f; ow[i] = float4(0.0f); }
  for (uint kb = c0 + sg * KB; kb < c1; kb += 4u * KB) {
    float4 kv[KB], vv[KB];
    NI_UNROLL for (uint j = 0; j < KB; ++j) {
      const uint kp = min(kb + j, c1 - 1u);
      const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
      const size_t off = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
      kv[j] = float4(*(const device half4*)(kc + off));
      vv[j] = float4(*(const device half4*)(vc + off));
    }
    NI_UNROLL for (uint t = 0; t < TR; ++t) {
      const uint qpos = p.pos0 + t0 + t;
      const uint lo = p.is_swa ? (qpos + 1u - min(qpos + 1u, p.swa)) : 0u;
      NI_UNROLL for (uint g = 0; g < G; ++g) {
        const uint qi = t * G + g;
        const float4 qv = *(const threadgroup float4*)(tq + qi * 128u + d0);
        float s[KB];
        float bm = -INFINITY;
        NI_UNROLL for (uint j = 0; j < KB; ++j) {
          const uint kp = kb + j;
          float sc = simd_sum(dot(qv, kv[j]));
          const uint dist = qpos - kp;
          if (dist < p.rel_extent) sc += rb[(size_t(t0 + t) * p.Hq + h0 + g) * p.rel_extent + dist];
          s[j] = (t < nr && kp < c1 && kp <= qpos && kp >= lo) ? sc : -INFINITY;
          bm = max(bm, s[j]);
        }
        if (bm == -INFINITY) continue;
        const float nm = max(mw[qi], bm);
        const float corr = fast::exp(mw[qi] - nm);
        float4 o = ow[qi] * corr;
        float l = lw[qi] * corr;
        NI_UNROLL for (uint j = 0; j < KB; ++j) {
          const float e = fast::exp(s[j] - nm);
          o += vv[j] * e;
          l += e;
        }
        ow[qi] = o; lw[qi] = l; mw[qi] = nm;
      }
    }
  }
  NI_UNROLL for (uint i = 0; i < NQ; ++i) {
    if (lane == 0) { tml[(sg * NQ + i) * 2] = mw[i]; tml[(sg * NQ + i) * 2 + 1] = lw[i]; }
    *(threadgroup float4*)(to + (sg * NQ + i) * 128u + d0) = ow[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = sg; i < nr * G; i += 4u) {
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0; s < 4; ++s) {
      const float l2 = tml[(s * NQ + i) * 2 + 1];
      if (l2 == 0.0f) continue;
      const float m2 = tml[(s * NQ + i) * 2];
      const float4 o2 = *(const threadgroup float4*)(to + (s * NQ + i) * 128u + d0);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = max(mm, m2);
        const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
        oo = oo * e1 + o2 * e2;
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const uint t = t0 + i / G, h = h0 + i % G;
    const size_t out_idx = (size_t(t) * p.Hq + h) * p.nchunks + cid;
    if (lane == 0) { pm[out_idx] = mm; pl[out_idx] = ll; }
    *(device float4*)(po + out_idx * p.D + d0) = oo;
  }
}
#define INK2_AVER(r) template [[host_name("ink2_attn_verify_g3_" #r)]] [[kernel]] decltype(ink2_attn_verify_g3<r>) ink2_attn_verify_g3<r>;
INK2_AVER(2) INK2_AVER(3) INK2_AVER(4)
#undef INK2_AVER

// ink2_attn_decode_reduce for verify rows: grid (Hq, T), partials [t][h][nchunks].
[[kernel]] void ink2_attn_verify_reduce(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                        const device float* po [[buffer(2)]], const device float* sinks [[buffer(3)]],
                                        const device float* gate [[buffer(4)]], device float* out [[buffer(5)]],
                                        constant Ink2AttnVerP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, t = tg.y, d0 = lane * 4;
  const size_t th = size_t(t) * p.Hq + h;
  float mm = sinks[th], ll = 1.0f;
  float4 oo = float4(0.0f);
  for (uint cid = 0; cid < p.nchunks; ++cid) {
    const size_t idx = th * p.nchunks + cid;
    const float l2 = pl[idx];
    if (l2 == 0.0f) continue;
    const float m2 = pm[idx];
    const float4 o2 = *(const device float4*)(po + idx * p.D + d0);
    const float nm = max(mm, m2);
    const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
    oo = oo * e1 + o2 * e2;
    ll = ll * e1 + l2 * e2;
    mm = nm;
  }
  const float4 g4 = *(const device float4*)(gate + th * p.D + d0);
  const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                           metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
  *(device float4*)(out + th * p.D + d0) = (oo / ll) * sig;
}

[[kernel]] void ink2_attn_verify_reduce_p(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                          const device float* po [[buffer(2)]], const device float* sinks [[buffer(3)]],
                                          const device float* gate [[buffer(4)]], device float* out [[buffer(5)]],
                                          constant Ink2AttnVerP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float wsh[128];
  const size_t th = size_t(tg.y) * p.Hq + tg.x;
  ink2_attn_reduce_par(pm, pl, po, sinks[th], gate + th * p.D, out + th * p.D, th * p.nchunks, p.nchunks, p.D, lane, wsh);
}

struct Ink2AttnPfP {
  uint T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa;
  float scale;
};

[[kernel]] void ink2_attn_prefill(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                  const device half* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                  const device float* sinks [[buffer(4)]], const device float* gate [[buffer(5)]],
                                  device float* out [[buffer(6)]], constant Ink2AttnPfP& p [[buffer(7)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, t = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint qpos = p.pos0 + t;
  const uint d0 = lane * 4;
  const size_t qoff = (size_t(t) * p.Hq + h) * p.D + d0;
  const float4 qv = *(const device float4*)(q + qoff);
  const device float* rbh = rb + (size_t(t) * p.Hq + h) * p.rel_extent;

  const uint lk = p.is_swa ? min(qpos + 1u, p.swa) : (qpos + 1u);
  const uint c0 = qpos + 1u - lk;
  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);
  for (uint kp = c0 + sg; kp <= qpos; kp += 4) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = float4(*(const device half4*)(kc + koff));
    float sc = simd_sum(dot(qv, kv)) * p.scale;
    const uint dist = qpos - kp;
    if (dist < p.rel_extent) sc += rbh[dist];
    const float4 vv = float4(*(const device half4*)(vc + koff));
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = max(mw, sc);
      const float e1 = metal::precise::exp(mw - nm), e2 = metal::precise::exp(sc - nm);
      ow = ow * e1 + vv * e2;
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0) {
    tgm[sg] = mw;
    tgl[sg] = lw;
  }
  *(threadgroup float4*)(&tgo[sg][d0]) = ow;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float mm = sinks[size_t(t) * p.Hq + h], ll = 1.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0; s < 4; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *(const threadgroup float4*)(&tgo[s][d0]);
      const float nm = max(mm, m2);
      const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
      oo = oo * e1 + o2 * e2;
      ll = ll * e1 + l2 * e2;
      mm = nm;
    }
    const float4 g4 = *(const device float4*)(gate + qoff);
    const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                             metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
    *(device float4*)(out + qoff) = (oo / ll) * sig;
  }
}

// ---- 10. MoE Router (E <= 128, top-k = 8) ----
struct Ink2RouteP {
  uint T, E, K, bias_f32;
  float route_scale;
};

[[kernel]] void ink2_moe_route(const device float* logits [[buffer(0)]], const device void* bias [[buffer(1)]],
                               device uint* eid [[buffer(2)]], device float* ew [[buffer(3)]],
                               constant Ink2RouteP& p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  float ss[8], sb[8];
  uint cnt = 0u;
  for (uint e = lane; e < p.E && cnt < 8u; e += 32u, ++cnt) {
    const float l = logits[size_t(t) * p.E + e];
    const float s = 1.0f / (1.0f + metal::precise::exp(-l));
    const float b = p.bias_f32 ? ((const device float*)bias)[e] : float(((const device bfloat*)bias)[e]);
    ss[cnt] = s;
    sb[cnt] = s + b;
  }
  uint chosen[8];
  float c_ss[8];
  float wsum = 0.0f;
  for (uint slot = 0; slot < p.K; ++slot) {
    float lbest = -INFINITY;
    uint lidx = 0xFFFFFFFFu;
    for (uint i = 0; i < cnt; ++i) {
      if (sb[i] > lbest) {
        lbest = sb[i];
        lidx = lane + i * 32u;
      }
    }
    const float gbest = simd_max(lbest);
    const uint cand = (lbest == gbest && isfinite(gbest)) ? lidx : 0xFFFFFFFFu;
    const uint winner = simd_min(cand);
    const uint wlane = winner & 31u;
    const uint wi = (winner < p.E && wlane == lane) ? (winner >> 5) : 0u;
    const float my_s = (winner < p.E && wlane == lane) ? ss[wi] : 0.0f;
    if (winner < p.E && wlane == lane) sb[wi] = -INFINITY;
    const float ws = (winner < p.E) ? simd_broadcast(my_s, ushort(wlane)) : 0.0f;
    if (lane == 0u) {
      chosen[slot] = winner < p.E ? winner : 0u;
      c_ss[slot] = ws;
      wsum += ws;
    }
  }
  if (lane == 0u) {
    const float inv = p.route_scale / (wsum + 1e-20f);
    for (uint slot = 0; slot < p.K; ++slot) {
      eid[size_t(t) * p.K + slot] = chosen[slot];
      ew[size_t(t) * p.K + slot] = c_ss[slot] * inv;
    }
  }
}

#if __METAL_VERSION__ >= 320
// lab11 (NI_INK2_RTLAST=1): the router matvec (ink2_mv_sk, one logit per threadgroup, same sums) whose last
// threadgroup to finish runs ink2_moe_route (verbatim, simdgroup 0) on the logits: no route stage. The logits go
// through device coherent(device) memory with thread_scope_device fences around the arrival count (lab11/coh: the
// only variant that passes). cnt returns to 0 for the next dispatch. M = 1 row.
[[kernel]] void ink2_mv_sk_route(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                 device coherent(device) float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                                 const device void* bias [[buffer(4)]], device uint* eid [[buffer(5)]],
                                 device float* ew [[buffer(6)]], constant Ink2RouteP& rp [[buffer(7)]],
                                 device atomic_uint* cnt [[buffer(8)]], uint n [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4];
  threadgroup uint last;
  const size_t woff = size_t(n) * p.K;
  float a = 0.0f;
  for (uint k = tid * 4u; k < p.K; k += 512u) {
    const float4 xv = *(const device float4*)(x + k);
    a += dot(xv, p.w_f32 ? *(const device float4*)((const device float*)w + woff + k)
                         : ink2_bf16x4(*(const device uint2*)((const device bfloat*)w + woff + k)));
  }
  a = simd_sum(a);
  if (lane == 0u) part[sg] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float v = (part[0] + part[1]) + (part[2] + part[3]);
    y[n] = v;
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0u) {
    last = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == p.N - 1u ? 1u : 0u;
    if (last) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last == 0u || sg != 0u) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  // ---- ink2_moe_route, t = 0 ----
  const uint t = 0u;
  float ss[8], sb[8];
  uint cnt8 = 0u;
  for (uint e = lane; e < rp.E && cnt8 < 8u; e += 32u, ++cnt8) {
    const float l = y[size_t(t) * rp.E + e];
    const float s = 1.0f / (1.0f + metal::precise::exp(-l));
    const float b = rp.bias_f32 ? ((const device float*)bias)[e] : float(((const device bfloat*)bias)[e]);
    ss[cnt8] = s;
    sb[cnt8] = s + b;
  }
  uint chosen[8];
  float c_ss[8];
  float wsum = 0.0f;
  for (uint slot = 0; slot < rp.K; ++slot) {
    float lbest = -INFINITY;
    uint lidx = 0xFFFFFFFFu;
    for (uint i = 0; i < cnt8; ++i) {
      if (sb[i] > lbest) {
        lbest = sb[i];
        lidx = lane + i * 32u;
      }
    }
    const float gbest = simd_max(lbest);
    const uint cand = (lbest == gbest && isfinite(gbest)) ? lidx : 0xFFFFFFFFu;
    const uint winner = simd_min(cand);
    const uint wlane = winner & 31u;
    const uint wi = (winner < rp.E && wlane == lane) ? (winner >> 5) : 0u;
    const float my_s = (winner < rp.E && wlane == lane) ? ss[wi] : 0.0f;
    if (winner < rp.E && wlane == lane) sb[wi] = -INFINITY;
    const float ws = (winner < rp.E) ? simd_broadcast(my_s, ushort(wlane)) : 0.0f;
    if (lane == 0u) {
      chosen[slot] = winner < rp.E ? winner : 0u;
      c_ss[slot] = ws;
      wsum += ws;
    }
  }
  if (lane == 0u) {
    const float inv = rp.route_scale / (wsum + 1e-20f);
    for (uint slot = 0; slot < rp.K; ++slot) {
      eid[size_t(t) * rp.K + slot] = chosen[slot];
      ew[size_t(t) * rp.K + slot] = c_ss[slot] * inv;
    }
  }
}
#endif

// ---- 11. MoE NVFP4 (layers 0..19): vectorized loads with exact FP32 E2M1 accumulation ----
struct Ink2MoeP {
  uint T, H, I, K_top;
};

template <bool LUT>
[[kernel]] void ink2_moe_nvfp4_w13_situ_t(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                        const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                        const device float* outer [[buffer(4)]], device float* act [[buffer(5)]],
                                        constant Ink2MoeP& p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                        uint tid [[thread_index_in_threadgroup]]) {
  INK2_NVFP4_LUT_FILL(tid, 128u)
  const uint j0 = (tg.x * 4u + sg) * 4u;
  if (j0 >= p.I) return;
  const uint nr = min(4u, p.I - j0);
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint two_i = 2u * p.I, half_h = p.H / 2u, blocks = p.H / 16u;
  const size_t wg0_row = size_t(e) * two_i + 2u * j0;
  const device float* xp = x + size_t(t) * p.H;

  float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint blk = lane; blk < blocks; blk += 32u) {
    const uint kbase = blk * 16u;
    const float4 xv[4] = {
        *(const device float4*)(xp + kbase + 0u), *(const device float4*)(xp + kbase + 4u),
        *(const device float4*)(xp + kbase + 8u), *(const device float4*)(xp + kbase + 12u),
    };
    const thread float* xf = (const thread float*)xv;
    for (uint q = 0u; q < nr; ++q) {
      ag[q] += INK2_NVFP4_DOT(w13, wg0_row + 2u * q, half_h, blk, xf) * ink2_e4m3(sc[(wg0_row + 2u * q) * blocks + blk]);
      au[q] += INK2_NVFP4_DOT(w13, wg0_row + 2u * q + 1u, half_h, blk, xf) * ink2_e4m3(sc[(wg0_row + 2u * q + 1u) * blocks + blk]);
    }
  }
  const float os = outer[e];
  const size_t obase = (size_t(t) * p.K_top + slot) * p.I + j0;
  for (uint q = 0u; q < nr; ++q) {
    const float g = simd_sum(ag[q]) * os, u = simd_sum(au[q]) * os;
    if (lane == 0u) {
      act[obase + q] = (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
                       (25.0f * metal::precise::tanh(u * 0.04f));
    }
  }
}

template [[host_name("ink2_moe_nvfp4_w13_situ")]] [[kernel]] decltype(ink2_moe_nvfp4_w13_situ_t<false>) ink2_moe_nvfp4_w13_situ_t<false>;
template [[host_name("ink2_moe_nvfp4_w13_situ_l")]] [[kernel]] decltype(ink2_moe_nvfp4_w13_situ_t<true>) ink2_moe_nvfp4_w13_situ_t<true>;

template <bool LUT>
[[kernel]] void ink2_moe_nvfp4_w2_acc_t(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                      const device uchar* sc [[buffer(4)]], const device float* outer [[buffer(5)]],
                                      device float* out [[buffer(6)]], constant Ink2MoeP& p [[buffer(7)]],
                                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  INK2_NVFP4_LUT_FILL(tid, 128u)
  const uint n0 = (tg.x * 4u + sg) * 4u;
  if (n0 >= p.H) return;
  const uint nr = min(4u, p.H - n0);
  const uint t = tg.y, half_i = p.I / 2u, blocks = p.I / 16u;
  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint slot = 0; slot < p.K_top; ++slot) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = ew[size_t(t) * p.K_top + slot];
    const size_t wrow0 = size_t(e) * p.H + n0;
    const device float* ap = act + (size_t(t) * p.K_top + slot) * p.I;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 16u;
      const float4 av[4] = {
          *(const device float4*)(ap + kbase + 0u), *(const device float4*)(ap + kbase + 4u),
          *(const device float4*)(ap + kbase + 8u), *(const device float4*)(ap + kbase + 12u),
      };
      const thread float* af = (const thread float*)av;
      for (uint q = 0u; q < nr; ++q)
        sum[q] += INK2_NVFP4_DOT(w2, wrow0 + q, half_i, blk, af) * ink2_e4m3(sc[(wrow0 + q) * blocks + blk]);
    }
    for (uint q = 0u; q < nr; ++q) total[q] += wt * (simd_sum(sum[q]) * outer[e]);
  }
  if (lane == 0u) {
    device float* dst = out + size_t(t) * p.H + n0;
    for (uint q = 0u; q < nr; ++q) dst[q] = total[q];
  }
}

template [[host_name("ink2_moe_nvfp4_w2_acc")]] [[kernel]] decltype(ink2_moe_nvfp4_w2_acc_t<false>) ink2_moe_nvfp4_w2_acc_t<false>;
template [[host_name("ink2_moe_nvfp4_w2_acc_l")]] [[kernel]] decltype(ink2_moe_nvfp4_w2_acc_t<true>) ink2_moe_nvfp4_w2_acc_t<true>;

// ---- 11b. MoE NVFP4 decode, wide (lab4; NI_INK2_MOEBT=0 for the kernels above). E2M1 by bits, no table: a nibble
// eem|s as half bits (eem << 9 | s << 15) is the value / 2^14 exactly (e=0 -> half subnormals 0, 2^-15). Probe
// (lab2/moebench2, 44 visits, random weights): w13 5.50 -> 4.06 ms, w2 3.70 -> 2.74 ms.
NI_INLINE float ink2_bt_dot8(uint w, const thread float* xf) {  // nibble k of w (bits 4k..) times xf[k]
  float d = 0.0f;
  _Pragma("clang loop unroll(full)") for (uint s = 0; s < 4; ++s) {
    const uint v = w >> (4u * s);
    const half2 h = as_type<half2>(((v << 9) & 0x0E000E00u) | ((v << 12) & 0x80008000u));
    d += float(h.x) * xf[s] + float(h.y) * xf[s + 4];
  }
  return d;
}
NI_INLINE float ink2_bt_dot16(uint2 pw, const thread float* xf) {
  return (ink2_bt_dot8(pw.x, xf) + ink2_bt_dot8(pw.y, xf + 8)) * 16384.0f;
}

// lab10 DEC mask for the bt kernels (0 = the code above). Exact: 1 = nibbles repacked at load (Inkling2::repack_moe:
// per 16-bit half, nibble 0 sign 15 / eem 9-11, 1: 12 / 6-8, 2: 13 / 3-5, 3: 14 / 0-2): 5 left shifts per word instead
// of 7 (one right; lab10 M4: 7.3 -> 3.9 issue cycles / nibble). The half values are identical (lab10/repack_check.cpp,
// all 2^32 words) but fast math sums the 16 products in another order once the integer code around them changes (any
// change: DEC 2 alone too; an equal-depth layout did not restore it), so logits are ulp-level, not bit-identical. 2 = E4M3 block scales sign-0 (checked at load), decoded without the
// sign term (w13: two scales per shift), the x16384 (E2M1) and x256 (E4M3) folded into the outer scale as 2^22.
// Timing probes (wrong math): 4 = decode-lite (every half2 = w & mask), 8 = x / act read from a 4 KB window.
template <uint DEC>
NI_INLINE float ink2_dot8d(uint w, const thread float* xf) {
  if ((DEC & 5u) == 0u) return ink2_bt_dot8(w, xf);
  float d = 0.0f;  // one half2 per step, decoded where ink2_bt_dot8 decodes it
  NI_UNROLL for (uint s = 0; s < 4; ++s) {
    const half2 h = as_type<half2>((DEC & 4u) ? w & 0x8E008E00u
                                   : s == 0   ? w & 0x8E008E00u
                                   : s == 1   ? (w << 3) & 0x8E008E00u
                                   : s == 2   ? ((w << 6) & 0x0E000E00u) | ((w << 2) & 0x80008000u)
                                              : ((w << 9) & 0x0E000E00u) | ((w << 1) & 0x80008000u));
    d += float(h.x) * xf[s] + float(h.y) * xf[s + 4];
  }
  return d;
}
template <uint DEC>
NI_INLINE float ink2_dot16d(uint2 pw, const thread float* xf) {
  const float d = ink2_dot8d<DEC>(pw.x, xf) + ink2_dot8d<DEC>(pw.y, xf + 8);
  return (DEC & 2u) ? d : d * 16384.0f;
}
template <uint DEC>
NI_INLINE float ink2_e4m3d(uint b) {  // DEC & 2: sign-0 byte, value / 256
  return (DEC & 2u) ? float(as_type<half>(ushort(b << 7))) : ink2_e4m3(b);
}
constant constexpr float kInk2Fold = 4194304.0f;  // 2^22 = 16384 x 256

// w13 + situ: 4 lanes per weight row, 4 blocks (32 bytes) per lane step (H/64 steps, H % 256 == 0: 9 per lane at
// H = 2304), 8 rows = 4 j (gate, up) per simdgroup, 16 j per threadgroup (grid as ink2_moe_nvfp4_w13_situ)
template <uint DEC>
[[kernel]] void ink2_moe_nvfp4_w13_q4_t(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                      const device float* outer [[buffer(4)]], device float* act [[buffer(5)]],
                                      constant Ink2MoeP& p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint j0 = (tg.x * 4u + sg) * 4u;
  if (j0 >= p.I) return;
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint half_h = p.H / 2u, blocks = p.H / 16u, l4 = lane & 3u, rr = lane >> 2;
  const size_t r = size_t(e) * 2u * p.I + 2u * j0 + min(rr, 2u * (p.I - j0) - 1u);
  const device float* xp = x + size_t(t) * p.H;
  float a = 0.0f;
  for (uint bq = l4; bq < blocks / 4u; bq += 4u) {
    const device uint4* wp = (const device uint4*)(w13 + r * half_h + bq * 32);
    const uint4 w0 = wp[0], w1 = wp[1];
    const uint s4 = *(const device uint*)(sc + r * blocks + 4u * bq);
    const half2 se = as_type<half2>((s4 << 7) & 0x7F807F80u), so = as_type<half2>((s4 >> 1) & 0x7F807F80u);
    _Pragma("clang loop unroll(full)") for (uint h = 0; h < 4; ++h) {
      float4 xv[4];
      _Pragma("clang loop unroll(full)") for (uint i = 0; i < 4; ++i)
        xv[i] = *(const device float4*)(xp + ((DEC & 8u) ? (bq * 64u + 16u * h + 4u * i) & 1023u : bq * 64u + 16u * h + 4u * i));
      const uint2 w = h == 0 ? w0.xy : h == 1 ? w0.zw : h == 2 ? w1.xy : w1.zw;
      if (DEC & 2u)
        a += ink2_dot16d<DEC>(w, (const thread float*)xv) * float(h == 0 ? se.x : h == 1 ? so.x : h == 2 ? se.y : so.y);
      else
        a += ink2_dot16d<DEC>(w, (const thread float*)xv) * ink2_e4m3((s4 >> (8u * h)) & 0xFFu);
    }
  }
  a += simd_shuffle_xor(a, 2);
  a += simd_shuffle_xor(a, 1);
  const float os = (DEC & 2u) ? outer[e] * kInk2Fold : outer[e];
  const float g = simd_shuffle(a, ushort((lane & 3u) * 8u)) * os, u = simd_shuffle(a, ushort((lane & 3u) * 8u + 4u)) * os;
  if (lane < min(4u, p.I - j0))
    act[(size_t(t) * p.K_top + slot) * p.I + j0 + lane] =
        (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
        (25.0f * metal::precise::tanh(u * 0.04f));
}
#define INK2_W13Q4(name, dec) \
  template [[host_name(name)]] [[kernel]] decltype(ink2_moe_nvfp4_w13_q4_t<dec>) ink2_moe_nvfp4_w13_q4_t<dec>;
INK2_W13Q4("ink2_moe_nvfp4_w13_q4", 0) INK2_W13Q4("ink2_moe_nvfp4_w13_q4_d1", 1) INK2_W13Q4("ink2_moe_nvfp4_w13_q4_d2", 2)
INK2_W13Q4("ink2_moe_nvfp4_w13_q4_d3", 3) INK2_W13Q4("ink2_moe_nvfp4_w13_q4_d4", 4) INK2_W13Q4("ink2_moe_nvfp4_w13_q4_d8", 8)
INK2_W13Q4("ink2_moe_nvfp4_w13_q4_d7", 7) INK2_W13Q4("ink2_moe_nvfp4_w13_q4_d11", 11)
#undef INK2_W13Q4

// w2 + weighted sum over the K slots: the (slot, block) loop flattened, K * I/16 units split evenly over the lanes
// (I = 1152: 576 = 18 x 32; per slot 72 blocks gave lanes 3 or 2 steps), 2 rows per simdgroup (8 per threadgroup)
template <uint DEC>
[[kernel]] void ink2_moe_nvfp4_w2_f2_t(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                     const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                     const device uchar* sc [[buffer(4)]], const device float* outer [[buffer(5)]],
                                     device float* out [[buffer(6)]], constant Ink2MoeP& p [[buffer(7)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4u + sg) * 2u;
  if (n0 >= p.H) return;
  const uint n1 = min(n0 + 1u, p.H - 1u);
  const uint t = tg.y, half_i = p.I / 2u, blocks = p.I / 16u, units = p.K_top * blocks;
  float t0 = 0.0f, t1 = 0.0f;
  uint slot = lane / blocks, blk = lane % blocks;
  for (uint u = lane; u < units; u += 32u) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = (DEC & 2u) ? (ew[size_t(t) * p.K_top + slot] * outer[e]) * kInk2Fold
                                : ew[size_t(t) * p.K_top + slot] * outer[e];
    const size_t ra = size_t(e) * p.H + n0, rb = size_t(e) * p.H + n1;
    const device float* ap = act + (size_t(t) * p.K_top + slot) * p.I + blk * 16u;
    const float4 av[4] = {*(const device float4*)(ap + 0u), *(const device float4*)(ap + 4u),
                          *(const device float4*)(ap + 8u), *(const device float4*)(ap + 12u)};
    const thread float* af = (const thread float*)av;
    t0 += ink2_dot16d<DEC>(*(const device uint2*)(w2 + ra * half_i + blk * 8), af) * (ink2_e4m3d<DEC>(sc[ra * blocks + blk]) * wt);
    t1 += ink2_dot16d<DEC>(*(const device uint2*)(w2 + rb * half_i + blk * 8), af) * (ink2_e4m3d<DEC>(sc[rb * blocks + blk]) * wt);
    blk += 32u;
    while (blk >= blocks) {
      blk -= blocks;
      ++slot;
    }
  }
  t0 = simd_sum(t0);
  t1 = simd_sum(t1);
  if (lane == 0u) {
    out[size_t(t) * p.H + n0] = t0;
    if (n1 != n0) out[size_t(t) * p.H + n1] = t1;
  }
}
template [[host_name("ink2_moe_nvfp4_w2_f2")]] [[kernel]] decltype(ink2_moe_nvfp4_w2_f2_t<0>) ink2_moe_nvfp4_w2_f2_t<0>;
template [[host_name("ink2_moe_nvfp4_w2_f2_d1")]] [[kernel]] decltype(ink2_moe_nvfp4_w2_f2_t<1>) ink2_moe_nvfp4_w2_f2_t<1>;
template [[host_name("ink2_moe_nvfp4_w2_f2_d2")]] [[kernel]] decltype(ink2_moe_nvfp4_w2_f2_t<2>) ink2_moe_nvfp4_w2_f2_t<2>;
template [[host_name("ink2_moe_nvfp4_w2_f2_d3")]] [[kernel]] decltype(ink2_moe_nvfp4_w2_f2_t<3>) ink2_moe_nvfp4_w2_f2_t<3>;

// w2_f2 with R rows per simdgroup: act (fp32 [K_top, I], 36.9 KB > 8 KB L1) is re-read once per R rows instead of per
// 2. B = 1: the f2 lane walk, so each row's sum is bit-identical to f2; B = 2: 2 blocks (16-byte weight loads) per unit
// (sum order changes, ~ulp). Grid: (H / (4 R), T) x 128.
// SC: one row (T = 1) with the mlp short conv + residual (ink2_sconv_static's T = 1 body) applied by the lane that owns
// the output channel (lab7: one stage less per MoE visit).
template <uint R, uint B, bool SC, uint DEC = 0>
[[kernel]] void ink2_moe_nvfp4_w2_fr(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                     const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                     const device uchar* sc [[buffer(4)]], const device float* outer [[buffer(5)]],
                                     device float* out [[buffer(6)]], constant Ink2MoeP& p [[buffer(7)]],
                                     const device void* cw [[buffer(8)]], device float* hist [[buffer(9)]],
                                     device float* cout [[buffer(10)]], constant Ink2SconvStaticP& cp [[buffer(11)]],
                                     const device float* res [[buffer(12)]], device float* snap [[buffer(13)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4u + sg) * R;
  if (n0 >= p.H) return;
  const uint t = tg.y, half_i = p.I / 2u, blocks = p.I / 16u, ub = blocks / B, units = p.K_top * ub;
  float acc[R];
  uint nr[R];
  NI_UNROLL for (uint q = 0u; q < R; ++q) acc[q] = 0.0f, nr[q] = min(n0 + q, p.H - 1u);
  uint slot = lane / ub, blk = lane % ub;
  for (uint u = lane; u < units; u += 32u) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = (DEC & 2u) ? (ew[size_t(t) * p.K_top + slot] * outer[e]) * kInk2Fold
                                : ew[size_t(t) * p.K_top + slot] * outer[e];
    size_t aoff = (size_t(t) * p.K_top + slot) * p.I + blk * (16u * B);
    if (DEC & 8u) aoff &= 1023u;
    const device float* ap = act + aoff;
    float4 av[4 * B];
    NI_UNROLL for (uint i = 0u; i < 4u * B; ++i) av[i] = *(const device float4*)(ap + 4u * i);
    const thread float* af = (const thread float*)av;
    const size_t rbase = size_t(e) * p.H;
    NI_UNROLL for (uint q = 0u; q < R; ++q) {
      const size_t r = rbase + nr[q];
      if (B == 1u) {
        acc[q] += ink2_dot16d<DEC>(*(const device uint2*)(w2 + r * half_i + blk * 8u), af) *
                  (ink2_e4m3d<DEC>(sc[r * blocks + blk]) * wt);
      } else {
        const uint4 w = *(const device uint4*)(w2 + r * half_i + blk * 16u);
        const uint s2 = uint(*(const device ushort*)(sc + r * blocks + 2u * blk));
        acc[q] += ink2_bt_dot16(w.xy, af) * (ink2_e4m3(s2 & 0xFFu) * wt) +
                  ink2_bt_dot16(w.zw, af + 16) * (ink2_e4m3(s2 >> 8) * wt);
      }
    }
    blk += 32u;
    while (blk >= ub) {
      blk -= ub;
      ++slot;
    }
  }
  NI_UNROLL for (uint q = 0u; q < R; ++q) {
    const float s = simd_sum(acc[q]);
    if (lane == q && n0 + q < p.H) {
      out[size_t(t) * p.H + n0 + q] = s;
      if (SC) ink2_sconv1(s, cw, hist, cout, cp, res, snap, n0 + q);
    }
  }
}
#define INK2_W2R(r, b)                                                                                                 \
  template [[host_name("ink2_moe_nvfp4_w2_r" #r "b" #b)]] [[kernel]] decltype(ink2_moe_nvfp4_w2_fr<r, b, false>)     \
      ink2_moe_nvfp4_w2_fr<r, b, false>;                                                                              \
  template [[host_name("ink2_moe_nvfp4_w2_r" #r "b" #b "_sc")]] [[kernel]] decltype(ink2_moe_nvfp4_w2_fr<r, b, true>) \
      ink2_moe_nvfp4_w2_fr<r, b, true>;
INK2_W2R(2, 1) INK2_W2R(4, 1) INK2_W2R(8, 1) INK2_W2R(2, 2) INK2_W2R(4, 2) INK2_W2R(8, 2)
#undef INK2_W2R
#define INK2_W2D(dec, sc, suf)                                                                                       \
  template [[host_name("ink2_moe_nvfp4_w2_r2b1" suf "_d" #dec)]] [[kernel]]                                         \
      decltype(ink2_moe_nvfp4_w2_fr<2, 1, sc, dec>) ink2_moe_nvfp4_w2_fr<2, 1, sc, dec>;
INK2_W2D(1, false, "") INK2_W2D(2, false, "") INK2_W2D(3, false, "")
INK2_W2D(1, true, "_sc") INK2_W2D(2, true, "_sc") INK2_W2D(3, true, "_sc") INK2_W2D(4, true, "_sc") INK2_W2D(8, true, "_sc")
INK2_W2D(7, true, "_sc") INK2_W2D(11, true, "_sc")
#undef INK2_W2D

// ---- 12. MoE MXFP8 (layers 20..23): F8_E4M3 weights + UE8M0 block-32 scales ----
[[kernel]] void ink2_moe_mxfp8_w13_situ(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                        const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                        device float* act [[buffer(4)]], constant Ink2MoeP& p [[buffer(5)]],
                                        uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg.x * 4 + sg;
  if (j >= p.I) return;
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint two_i = 2 * p.I;
  const uint blocks = p.H / 32;
  const size_t wg_row = size_t(e) * two_i + 2 * j;
  const size_t wu_row = wg_row + 1;
  const device float* xp = x + size_t(t) * p.H;

  float ag = 0.0f, au = 0.0f;
  for (uint blk = lane; blk < blocks; blk += 32) {
    const float sg_b = ink2_ue8m0(sc[wg_row * blocks + blk]);
    const float su_b = ink2_ue8m0(sc[wu_row * blocks + blk]);
    const device uchar* pg = w13 + wg_row * p.H + blk * 32;
    const device uchar* pu = w13 + wu_row * p.H + blk * 32;
    const uint kbase = blk * 32;
    float dg = 0.0f, du = 0.0f;
    for (uint b = 0; b < 32; ++b) {
      const float xv = xp[kbase + b];
      dg += ink2_e4m3(pg[b]) * xv;
      du += ink2_e4m3(pu[b]) * xv;
    }
    ag += dg * sg_b;
    au += du * su_b;
  }
  const float g = simd_sum(ag);
  const float u = simd_sum(au);
  if (lane == 0) {
    const float out_v =
        (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
        (25.0f * metal::precise::tanh(u * 0.04f));
    act[(size_t(t) * p.K_top + slot) * p.I + j] = out_v;
  }
}

[[kernel]] void ink2_moe_mxfp8_w2_acc(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                      const device uchar* sc [[buffer(4)]], device float* out [[buffer(5)]],
                                      constant Ink2MoeP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4 + sg;
  if (n >= p.H) return;
  const uint t = tg.y;
  const uint blocks = p.I / 32;
  float total = 0.0f;
  for (uint slot = 0; slot < p.K_top; ++slot) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = ew[size_t(t) * p.K_top + slot];
    const size_t wrow = size_t(e) * p.H + n;
    const device float* ap = act + (size_t(t) * p.K_top + slot) * p.I;
    float sum = 0.0f;
    for (uint blk = lane; blk < blocks; blk += 32) {
      const float sb = ink2_ue8m0(sc[wrow * blocks + blk]);
      const device uchar* pw = w2 + wrow * p.I + blk * 32;
      const uint kbase = blk * 32;
      float d = 0.0f;
      for (uint b = 0; b < 32; ++b) d += ink2_e4m3(pw[b]) * ap[kbase + b];
      sum += d * sb;
    }
    total += wt * simd_sum(sum);
  }
  if (lane == 0) out[size_t(t) * p.H + n] = total;
}

// Vectorized MXFP8 decode (round 3): 16-byte weight loads, E4M3 through a threadgroup float LUT[256]; w13: 16 lanes
// per row (144 uint4 = 9 per lane), a simdgroup = one j (gate lanes 0-15, up lanes 16-31); w2: 8 lanes per row
// (72 uint4 = 9 per lane), 4 rows per simdgroup, 16 per TG.
NI_INLINE float ink2_e4m3_dot16(uint4 w, const thread float4* xv, threadgroup const float* lut) {
  float d = 0.0f;
  for (uint i = 0; i < 4; ++i) {
    const uint wd = w[i];
    const float4 wf = float4(lut[wd & 0xFFu], lut[(wd >> 8) & 0xFFu], lut[(wd >> 16) & 0xFFu], lut[wd >> 24]);
    d += dot(wf, xv[i]);
  }
  return d;
}

[[kernel]] void ink2_moe_mxfp8_w13_situ_v(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                          const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                          device float* act [[buffer(4)]], constant Ink2MoeP& p [[buffer(5)]],
                                          uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float lut[256];
  for (uint i = tid; i < 256u; i += 128u) lut[i] = ink2_e4m3(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint j = tg.x * 4 + sg;
  if (j >= p.I) return;
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint blocks = p.H / 32u, half_lane = lane & 15u;
  const size_t row = size_t(e) * 2u * p.I + 2u * j + (lane >> 4);
  const device uint4* wr = (const device uint4*)(w13 + row * p.H);
  const device uchar* sr = sc + row * blocks;
  const device float* xp = x + size_t(t) * p.H;
  float a = 0.0f;
  for (uint c = half_lane; c < p.H / 16u; c += 16u) {
    const device float4* xq = (const device float4*)(xp + c * 16u);
    const float4 xv[4] = {xq[0], xq[1], xq[2], xq[3]};
    a += ink2_e4m3_dot16(wr[c], xv, lut) * ink2_ue8m0(sr[c >> 1]);
  }
  a += simd_shuffle_xor(a, 8u);
  a += simd_shuffle_xor(a, 4u);
  a += simd_shuffle_xor(a, 2u);
  a += simd_shuffle_xor(a, 1u);
  const float g = simd_shuffle(a, ushort(0)), u = simd_shuffle(a, ushort(16));
  if (lane == 0) {
    act[(size_t(t) * p.K_top + slot) * p.I + j] =
        (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
        (25.0f * metal::precise::tanh(u * 0.04f));
  }
}

[[kernel]] void ink2_moe_mxfp8_w2_acc_v(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                        const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                        const device uchar* sc [[buffer(4)]], device float* out [[buffer(5)]],
                                        constant Ink2MoeP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                        uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float lut[256];
  for (uint i = tid; i < 256u; i += 128u) lut[i] = ink2_e4m3(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = (tg.x * 4u + sg) * 4u + (lane >> 3);
  if (n >= p.H) return;
  const uint t = tg.y, sub = lane & 7u, blocks = p.I / 32u;
  float total = 0.0f;
  for (uint slot = 0; slot < p.K_top; ++slot) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = ew[size_t(t) * p.K_top + slot];
    const size_t row = size_t(e) * p.H + n;
    const device uint4* wr = (const device uint4*)(w2 + row * p.I);
    const device uchar* sr = sc + row * blocks;
    const device float* ap = act + (size_t(t) * p.K_top + slot) * p.I;
    float sum = 0.0f;
    for (uint c = sub; c < p.I / 16u; c += 8u) {
      const device float4* aq = (const device float4*)(ap + c * 16u);
      const float4 av[4] = {aq[0], aq[1], aq[2], aq[3]};
      sum += ink2_e4m3_dot16(wr[c], av, lut) * ink2_ue8m0(sr[c >> 1]);
    }
    sum += simd_shuffle_xor(sum, 4u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 1u);
    total += wt * sum;
  }
  if (sub == 0u) out[size_t(t) * p.H + n] = total;
}

// ---- 13. Buffer Copy, Final Logit Softcap ----
[[kernel]] void ink2_copy_bytes(const device uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                                constant uint64_t& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint64_t off = uint64_t(i) * 16;
  if (off + 16 <= n) {
    *(device uint4*)(dst + off) = *(const device uint4*)(src + off);
  } else {
    for (uint64_t k = off; k < n; ++k) dst[k] = src[k];
  }
}

struct Ink2SoftcapP {
  uint n;
  float cap, inv_cap;
};

[[kernel]] void ink2_softcap(device float* x [[buffer(0)]], constant Ink2SoftcapP& p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < p.n) x[i] = p.cap * metal::precise::tanh(x[i] * p.inv_cap);
}

// ---- 14. Grouped MoE Prefill Helpers (bucket, SiTU -> half, weighted combine) ----
struct Ink2BucketP {
  uint P, E, BM;
};

[[kernel]] void ink2_moe_bucket(const device uint* eid [[buffer(0)]], device uint* sorted [[buffer(1)]],
                                device uint4* tiles [[buffer(2)]], device uint* ntiles [[buffer(3)]],
                                device uint* rank [[buffer(4)]], constant Ink2BucketP& p [[buffer(5)]],
                                uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[128];
  threadgroup uint off[128], toff[128];
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) rank[i] = atomic_fetch_add_explicit(&cnt[eid[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint o = 0, t = 0;
    for (uint e = 0; e < p.E; ++e) {
      const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      off[e] = o;
      toff[e] = t;
      o += c;
      t += (c + p.BM - 1) / p.BM;
    }
    ntiles[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.E; e += ntg) {
    const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
    for (uint j = 0; j * p.BM < c; ++j) tiles[toff[e] + j] = uint4(e, off[e] + j * p.BM, min(p.BM, c - j * p.BM), 0);
  }
  for (uint i = tid; i < p.P; i += ntg) sorted[off[eid[i]] + rank[i]] = i;
}

struct Ink2SituP {
  uint n;
};

[[kernel]] void ink2_moe_situ_half(const device float2* w13_out [[buffer(0)]], device half* act_h [[buffer(1)]],
                                   constant Ink2SituP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  const float2 gu = w13_out[i];
  const float g = gu.x, u = gu.y;
  const float v =
      (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
      (25.0f * metal::precise::tanh(u * 0.04f));
  act_h[i] = half(v);
}

struct Ink2CombineP {
  uint T, H, K_top;
};

[[kernel]] void ink2_moe_combine(const device float* pair_out [[buffer(0)]], const device float* ew [[buffer(1)]],
                                 device float* out [[buffer(2)]], constant Ink2CombineP& p [[buffer(3)]],
                                 uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint ntg [[threads_per_threadgroup]]) {
  const uint h4 = p.H / 4;
  for (uint c = tid; c < h4; c += ntg) {
    float4 acc = float4(0.0f);
    for (uint s = 0; s < p.K_top; ++s) {
      const float w = ew[size_t(t) * p.K_top + s];
      acc += w * *(const device float4*)(pair_out + (size_t(t) * p.K_top + s) * p.H + c * 4);
    }
    *(device float4*)(out + size_t(t) * p.H + c * 4) = acc;
  }
}
// ---- 15. Q8_0 Load-Packed Dense Projections & Unembed (int8 codes [N, K] + half scales [N, K/32]) ----
NI_INLINE float ink2_q8_dot16(uint4 v, const device float4* xp) {
  float d = 0.0f;
  for (uint m = 0; m < 4; ++m) {
    const uint w = v[m];
    const float4 xv = xp[m];
    d += float(as_type<char>(uchar(w & 0xFFu))) * xv.x +
         float(as_type<char>(uchar((w >> 8) & 0xFFu))) * xv.y +
         float(as_type<char>(uchar((w >> 16) & 0xFFu))) * xv.z +
         float(as_type<char>(uchar((w >> 24) & 0xFFu))) * xv.w;
  }
  return d;
}

[[kernel]] void ink2_q8_mv1(const device float* x [[buffer(0)]], const device uchar* q [[buffer(1)]],
                            const device half* s [[buffer(2)]], device float* y [[buffer(3)]],
                            constant Ink2MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4 + sg) * 2;
  if (n0 >= p.N) return;
  const bool has1 = (n0 + 1) < p.N;
  const uint blocks = p.K / 32;
  const size_t qoff0 = size_t(n0) * p.K;
  const size_t qoff1 = qoff0 + p.K;
  const size_t soff0 = size_t(n0) * blocks;
  const size_t soff1 = soff0 + blocks;
  float acc0 = 0.0f, acc1 = 0.0f;
  for (uint blk = lane; blk < blocks; blk += 32) {
    const device float4* xp = (const device float4*)(x + blk * 32);
    const device uint4* q0 = (const device uint4*)(q + qoff0 + blk * 32);
    const float d0 = ink2_q8_dot16(q0[0], xp) + ink2_q8_dot16(q0[1], xp + 4);
    acc0 += d0 * float(s[soff0 + blk]);
    if (has1) {
      const device uint4* q1 = (const device uint4*)(q + qoff1 + blk * 32);
      const float d1 = ink2_q8_dot16(q1[0], xp) + ink2_q8_dot16(q1[1], xp + 4);
      acc1 += d1 * float(s[soff1 + blk]);
    }
  }
  const float s0 = simd_sum(acc0);
  const float s1 = has1 ? simd_sum(acc1) : 0.0f;
  if (lane == 0) {
    y[n0] = p.add ? (y[n0] + s0) : s0;
    if (has1) y[n0 + 1] = p.add ? (y[n0 + 1] + s1) : s1;
  }
}

[[kernel]] void ink2_q8_mv(const device float* x [[buffer(0)]], const device uchar* q [[buffer(1)]],
                           const device half* s [[buffer(2)]], device float* y [[buffer(3)]],
                           constant Ink2MmP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4 + sg;
  if (n >= p.N) return;
  const uint m0 = tg.y * 8;
  const uint mc = min(8u, p.M - m0);
  const uint blocks = p.K / 32;
  const size_t qoff = size_t(n) * p.K;
  const size_t soff = size_t(n) * blocks;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint blk = lane; blk < blocks; blk += 32) {
    const float sc = float(s[soff + blk]);
    const device uint4* qp = (const device uint4*)(q + qoff + blk * 32);
    const uint4 v0 = qp[0], v1 = qp[1];
    for (uint m = 0; m < mc; ++m) {
      const device float4* xp = (const device float4*)(x + size_t(m0 + m) * p.x_ld + blk * 32);
      acc[m] += (ink2_q8_dot16(v0, xp) + ink2_q8_dot16(v1, xp + 4)) * sc;
    }
  }
  for (uint m = 0; m < mc; ++m) {
    const float sum = simd_sum(acc[m]);
    if (lane == 0) {
      device float* dst = y + size_t(m0 + m) * p.y_ld + n;
      *dst = p.add ? (*dst + sum) : sum;
    }
  }
}

[[kernel]] void ink2_q8_mm(const device float* X [[buffer(0)]], const device uchar* Q [[buffer(1)]],
                           const device half* S [[buffer(2)]], device float* Y [[buffer(3)]],
                           constant Ink2MmP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BM = 32, BN = 64, BK = 32, PAD = 8;
  constexpr uint TM = 16, TN = 32, FM = 2, FN = 4;
  threadgroup half xs[BM * (BK + PAD)];
  threadgroup half ws[BN * (BK + PAD)];
  const uint m0 = tg.y * BM, n0 = tg.x * BN;
  const uint sgm = sg / 2, sgn = sg % 2;
  const uint blocks = p.K / 32;

  simdgroup_float8x8 acc[FM][FN];
  for (uint i = 0; i < FM; ++i)
    for (uint j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);

  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    const uint xr = tid / 4, xc = (tid % 4) * 8;
    const uint xm = m0 + xr;
    threadgroup half* xdst = xs + xr * (BK + PAD) + xc;
    if (xm < p.M && k0 + xc + 7 < p.K) {
      const float4 v0 = *(const device float4*)(X + size_t(xm) * p.x_ld + k0 + xc);
      const float4 v1 = *(const device float4*)(X + size_t(xm) * p.x_ld + k0 + xc + 4);
      *(threadgroup half4*)(xdst) = half4(v0);
      *(threadgroup half4*)(xdst + 4) = half4(v1);
    } else {
      for (uint j = 0; j < 8; ++j)
        xdst[j] = (xm < p.M && k0 + xc + j < p.K) ? half(X[size_t(xm) * p.x_ld + k0 + xc + j]) : 0.0h;
    }

    const uint wr = tid / 2, wc = (tid % 2) * 16;
    const uint wn = n0 + wr;
    threadgroup half* wdst = ws + wr * (BK + PAD) + wc;
    if (wn < p.N) {
      const float sc = float(S[size_t(wn) * blocks + k0 / 32]);
      const uint4 v = *(const device uint4*)(Q + size_t(wn) * p.K + k0 + wc);
      for (uint m = 0; m < 4; ++m) {
        const uint w = v[m];
        *(threadgroup half4*)(wdst + m * 4) = half4(
            half(float(as_type<char>(uchar(w & 0xFFu))) * sc),
            half(float(as_type<char>(uchar((w >> 8) & 0xFFu))) * sc),
            half(float(as_type<char>(uchar((w >> 16) & 0xFFu))) * sc),
            half(float(as_type<char>(uchar((w >> 24) & 0xFFu))) * sc));
      }
    } else {
      for (uint j = 0; j < 16; j += 4) *(threadgroup half4*)(wdst + j) = half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_half8x8 a[FM], b[FN];
      for (uint i = 0; i < FM; ++i) simdgroup_load(a[i], xs + (sgm * TM + i * 8) * (BK + PAD) + kk, BK + PAD);
      for (uint j = 0; j < FN; ++j)
        simdgroup_load(b[j], ws + (sgn * TN + j * 8) * (BK + PAD) + kk, BK + PAD, ulong2(0, 0), true);
      for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float stage_buf[4 * 64];
  threadgroup float* stage = stage_buf + sg * 64;
  for (uint i = 0; i < FM; ++i)
    for (uint j = 0; j < FN; ++j) {
      simdgroup_store(acc[i][j], stage, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) {
        const uint r = e / 8, c = e % 8;
        const uint m = m0 + sgm * TM + i * 8 + r;
        const uint n = n0 + sgn * TN + j * 8 + c;
        if (m < p.M && n < p.N) {
          device float* dst = Y + size_t(m) * p.y_ld + n;
          const float v = stage[e];
          *dst = p.add ? (*dst + v) : v;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- Prefill attention on simdgroup_matrix 8x8 (lab5, NI_INK2_FA; MLX steel_attention-style, fp32 fragments) ----
// One threadgroup = NT tokens x the G = 3 query heads of one KV head (3 * NT / 8 simdgroups of 8 rows), K/V tiles of
// BK keys staged row-major in threadgroup memory and shared by all of them; Q (pre-scaled by scale * log2 e, rounded to
// half like the tensor kernels) stays in registers; S and P stay fp32 (the tensor kernel rounds P to half). Sinks,
// rel bias, SWA ring and sigmoid output gate as qattn_ink2. Heaviest (last) token blocks are scheduled first.
// attnbench 500 @ 40K global: 53.7 ms vs 81.0 (qattn_ink2_s6k32p_l1tn), 4096 @ 40K 431 vs 637, SWA 4096 12.3 vs 20.5.
// grid: (ceil(T / NT), Hkv) x 3 * NT * 4 threads.
template <int NT, int BK>
NI_INLINE void ink2_fa_impl(const device float* q, const device half* kc, const device half* vc,
                            const device float* rb, const device float* sinks, const device float* gate,
                            device float* out, constant Ink2AttnPfP& p, uint2 tg, uint sg, ushort lane,
                            threadgroup half* ks, threadgroup half* vs) {
  constexpr int D = 128, G = 3, SPH = NT / 8, NS = G * SPH, NTH = NS * 32, LD = D + 8;
  const uint ntg = (p.T + NT - 1) / NT;
  const uint kvh = tg.y, tgt0 = (ntg - 1 - tg.x) * NT;
  const uint hd = kvh * G + sg / SPH;
  const uint t0 = tgt0 + (sg % SPH) * 8;
  const bool live = t0 < p.T;
  const uint tid = sg * 32 + lane;
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);  // simdgroup_matrix 8x8 lane layout: row fm, cols fn, fn + 1
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  const uint tq = t0 + fm, tqc = min(tq, p.T - 1);
  const uint lim = p.pos0 + tqc;
  const uint lo = (p.is_swa && lim + 1 > p.swa) ? lim + 1 - p.swa : 0u;
  const device float* rbh = rb + (size_t(tqc) * p.Hq + hd) * p.rel_extent;
  const uint kend = p.pos0 + min(tgt0 + NT, p.T);
  const uint kfirst = (p.is_swa && p.pos0 + tgt0 + 1 > p.swa) ? p.pos0 + tgt0 + 1 - p.swa : 0u;
  const uint sfirst = p.pos0 + min(t0, p.T - 1), slast = p.pos0 + min(t0 + 7, p.T - 1);
  const uint slo = (p.is_swa && slast + 1 > p.swa) ? slast + 1 - p.swa : 0u;
  const uint sflo = (p.is_swa && sfirst + 1 > p.swa) ? sfirst + 1 - p.swa : 0u;

  const float qscale = p.scale * M_LOG2E_F;
  simdgroup_matrix<float, 8, 8> qm[D / 8], o[D / 8];
  {
    const device float* qr = q + (size_t(tqc) * p.Hq + hd) * D + fn;
    NI_UNROLL
    for (short dd = 0; dd < D / 8; ++dd) {
      const float2 x = *(const device float2*)(qr + dd * 8) * qscale;
      qm[dd].thread_elements()[0] = float(half(x.x));
      qm[dd].thread_elements()[1] = float(half(x.y));
      o[dd] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  float m = sinks[size_t(tqc) * p.Hq + hd] * M_LOG2E_F, l = 1.0f;

  for (uint k0 = kfirst / BK * BK; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * (D / 8); i += NTH) {
      const uint r = i / (D / 8), c = (i % (D / 8)) * 8, key = k0 + r;
      uint4 kv4 = uint4(0), vv4 = uint4(0);
      if (key >= kfirst && key < kend) {
        const uint slot = p.is_swa ? (key % p.kv_cap) : key;
        const size_t at = (size_t(slot) * p.Hkv + kvh) * D + c;
        kv4 = *(const device uint4*)(kc + at);
        vv4 = *(const device uint4*)(vc + at);
      }
      *(threadgroup uint4*)(ks + r * LD + c) = kv4;
      *(threadgroup uint4*)(vs + r * LD + c) = vv4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || k0 > slast || k0 + BK <= sflo) continue;

    simdgroup_matrix<float, 8, 8> s[BK / 8];
    NI_UNROLL
    for (short f = 0; f < BK / 8; ++f) s[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    NI_UNROLL
    for (short dd = 0; dd < D / 8; ++dd)
      NI_UNROLL
      for (short f = 0; f < BK / 8; ++f) {
        simdgroup_matrix<float, 8, 8> km;  // K^T fragment: rows = dims dd * 8 + fm, cols = keys f * 8 + fn + j
        km.thread_elements()[0] = float(ks[(f * 8 + fn) * LD + dd * 8 + fm]);
        km.thread_elements()[1] = float(ks[(f * 8 + fn + 1) * LD + dd * 8 + fm]);
        simdgroup_multiply_accumulate(s[f], qm[dd], km, s[f]);
      }

    const bool edge = k0 + BK - 1 > sfirst || k0 < slo;
    const bool near_rb = p.rel_extent > 0 && (k0 + BK + p.rel_extent > sfirst);
    float mx = -1e30f;
    NI_UNROLL
    for (short f = 0; f < BK / 8; ++f)
      NI_UNROLL
      for (short j = 0; j < 2; ++j) {
        float v = s[f].thread_elements()[j];
        const uint key = k0 + f * 8 + fn + j;
        if (edge && (key > lim || key < lo)) {
          v = -1e30f;
        } else if (near_rb) {
          const uint dist = lim - key;
          if (dist < p.rel_extent) v += rbh[dist] * M_LOG2E_F;
        }
        s[f].thread_elements()[j] = v;
        mx = max(mx, v);
      }
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float mn = max(m, mx);
    const float corr = exp2(m - mn);
    m = mn;
    float ps = 0.0f;
    NI_UNROLL
    for (short f = 0; f < BK / 8; ++f)
      NI_UNROLL
      for (short j = 0; j < 2; ++j) {
        const float v = s[f].thread_elements()[j];
        const float pv = v <= -1e29f ? 0.0f : exp2(v - mn);
        s[f].thread_elements()[j] = pv;
        ps += pv;
      }
    ps += simd_shuffle_xor(ps, ushort(1));
    ps += simd_shuffle_xor(ps, ushort(8));
    l = l * corr + ps;
    if (!simd_all(corr == 1.0f)) {
      NI_UNROLL
      for (short dd = 0; dd < D / 8; ++dd) {
        o[dd].thread_elements()[0] *= corr;
        o[dd].thread_elements()[1] *= corr;
      }
    }
    NI_UNROLL
    for (short f = 0; f < BK / 8; ++f)
      NI_UNROLL
      for (short dd = 0; dd < D / 8; ++dd) {
        simdgroup_matrix<float, 8, 8> vm;
        const half2 v2 = *(const threadgroup half2*)(vs + (f * 8 + fm) * LD + dd * 8 + fn);
        vm.thread_elements()[0] = float(v2.x);
        vm.thread_elements()[1] = float(v2.y);
        simdgroup_multiply_accumulate(o[dd], s[f], vm, o[dd]);
      }
  }
  if (!live || tq >= p.T) return;
  const float inv = 1.0f / l;
  const size_t base = (size_t(tq) * p.Hq + hd) * D + fn;
  NI_UNROLL
  for (short dd = 0; dd < D / 8; ++dd) {
    const float2 r = float2(o[dd].thread_elements()[0], o[dd].thread_elements()[1]) * inv;
    const float2 g2 = *(const device float2*)(gate + base + dd * 8);
    const float2 sig = float2(1.0f) / (float2(1.0f) + float2(metal::precise::exp(-g2.x), metal::precise::exp(-g2.y)));
    *(device float2*)(out + base + dd * 8) = r * sig;
  }
}

#define INK2_FA(NAME, NT, BK)                                                                                           \
  [[kernel, max_total_threads_per_threadgroup(12 * (NT))]] void NAME(                                                  \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],   \
      const device float* rb [[buffer(3)]], const device float* sinks [[buffer(4)]],                                   \
      const device float* gate [[buffer(5)]], device float* out [[buffer(6)]], constant Ink2AttnPfP& p [[buffer(7)]],  \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                           \
      uint lane [[thread_index_in_simdgroup]]) {                                                                       \
    threadgroup half ks[(BK) * 136], vs[(BK) * 136];                                                                   \
    ink2_fa_impl<NT, BK>(q, kc, vc, rb, sinks, gate, out, p, tg, sg, ushort(lane), ks, vs);                            \
  }
INK2_FA(ink2_fa_t16k32, 16, 32)
INK2_FA(ink2_fa_t16k16, 16, 16)

// y[i] (+)= sum over z < S of part[z * n + i], in order (split-K GEMM partials, ink2_t_bf16_sk_m64)
struct Ink2SumPartsP {
  uint n, S, add;
};
[[kernel]] void ink2_sum_parts(const device float* part [[buffer(0)]], device float* y [[buffer(1)]],
                               constant Ink2SumPartsP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  float a = part[i];
  for (uint z = 1; z < p.S; ++z) a += part[size_t(z) * p.n + i];
  y[i] = p.add ? y[i] + a : a;
}
