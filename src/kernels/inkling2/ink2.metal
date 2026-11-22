#include "common.h"

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
};

[[kernel]] void ink2_sconv_static(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                  device float* hist [[buffer(2)]], device float* out [[buffer(3)]],
                                  constant Ink2SconvStaticP& p [[buffer(4)]], const device float* res [[buffer(5)]],
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
    const float conv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
    float y = (conv + xt) * p.scale;
    if (p.add_res) y += res[idx];
    out[idx] = y;
    h0 = h1; h1 = h2; h2 = xt;
  }
  hist[0 * p.C + c] = h0;
  hist[1 * p.C + c] = h1;
  hist[2 * p.C + c] = h2;
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
  uint T, C, n_heads, sw_f32;
};

[[kernel]] void ink2_sconv_dynamic(const device float* x [[buffer(0)]], const device float* dyn [[buffer(1)]],
                                   const device void* static_w [[buffer(2)]], device float* hist [[buffer(3)]],
                                   device float* out [[buffer(4)]], constant Ink2SconvDynP& p [[buffer(5)]],
                                   uint c [[thread_position_in_grid]]) {
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

// ---- 11. MoE NVFP4 (layers 0..19): vectorized loads with exact FP32 E2M1 accumulation ----
struct Ink2MoeP {
  uint T, H, I, K_top;
};

[[kernel]] void ink2_moe_nvfp4_w13_situ(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                        const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                        const device float* outer [[buffer(4)]], device float* act [[buffer(5)]],
                                        constant Ink2MoeP& p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
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
      ag[q] += ink2_nvfp4_dot(w13, wg0_row + 2u * q, half_h, blk, xf) * ink2_e4m3(sc[(wg0_row + 2u * q) * blocks + blk]);
      au[q] += ink2_nvfp4_dot(w13, wg0_row + 2u * q + 1u, half_h, blk, xf) * ink2_e4m3(sc[(wg0_row + 2u * q + 1u) * blocks + blk]);
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

[[kernel]] void ink2_moe_nvfp4_w2_acc(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                      const device uchar* sc [[buffer(4)]], const device float* outer [[buffer(5)]],
                                      device float* out [[buffer(6)]], constant Ink2MoeP& p [[buffer(7)]],
                                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
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
        sum[q] += ink2_nvfp4_dot(w2, wrow0 + q, half_i, blk, af) * ink2_e4m3(sc[(wrow0 + q) * blocks + blk]);
    }
    for (uint q = 0u; q < nr; ++q) total[q] += wt * (simd_sum(sum[q]) * outer[e]);
  }
  if (lane == 0u) {
    device float* dst = out + size_t(t) * p.H + n0;
    for (uint q = 0u; q < nr; ++q) dst[q] = total[q];
  }
}

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
