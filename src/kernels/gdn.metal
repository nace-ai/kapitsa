#include "common.h"

// ---- Token embedding (MLX affine 4-bit, row per token) ----
struct EmbedParams {
  uint D;        // hidden size
  uint gs;       // group size
  uint tok_off;  // index into `tokens` of the first token
};

// grid: x = D/8 words, y = T tokens
[[kernel]] void embed_q4(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],
                         const device bfloat* b [[buffer(2)]], const device int* tokens [[buffer(3)]],
                         device float* x [[buffer(4)]], constant EmbedParams& p [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
  uint i = gid.x, t = gid.y;
  if (i >= p.D / 8) return;
  int tok = tokens[p.tok_off + t];
  uint32_t word = w[uint(tok) * (p.D / 8) + i];
  uint gpr = p.D / p.gs;
  for (uint j = 0; j < 8; ++j) {
    uint k = i * 8 + j;
    float sc = float(s[uint(tok) * gpr + k / p.gs]), bi = float(b[uint(tok) * gpr + k / p.gs]);
    x[t * p.D + k] = sc * float((word >> (4 * j)) & 0xF) + bi;
  }
}

// ---- Gated DeltaNet: conv1d + SiLU + q/k L2 norm + decay/beta ----
// Channels of the qkv projection: [q: Hk*Dk | k: Hk*Dk | v: Hv*Dv]; one threadgroup of 128 threads
// per 128-channel head chunk. conv_state holds the previous 3 raw inputs ([3, C]).
struct GdnPrepParams {
  uint T;
  uint Hk;
  uint Hv;
  uint C;  // conv channels = 2*Hk*Dk + Hv*Dv
};

[[kernel]] void gdn_prep(const device float* qkv [[buffer(0)]], const device float* a [[buffer(1)]],
                         const device float* bb [[buffer(2)]], device float* conv_state [[buffer(3)]],
                         const device bfloat* conv_w [[buffer(4)]], const device bfloat* A_log [[buffer(5)]],
                         const device bfloat* dt_bias [[buffer(6)]], device float* q_out [[buffer(7)]],
                         device float* k_out [[buffer(8)]], device float* v_out [[buffer(9)]],
                         device float* g_out [[buffer(10)]], device float* beta_out [[buffer(11)]],
                         constant GdnPrepParams& p [[buffer(12)]], uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[32];
  constexpr uint D = 128;
  const uint c = tg * D + tid;
  const uint C = p.C;
  const uint qk = p.Hk * D;
  const bool is_q = c < qk, is_k = c >= qk && c < 2 * qk;
  float w0 = conv_w[c * 4 + 0], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], w3 = conv_w[c * 4 + 3];

  for (uint t = 0; t < p.T; ++t) {
    // concat = [state0, state1, state2, qkv[0], qkv[1], ...]; output t uses concat[t..t+3]
    float in0 = t + 0 < 3 ? conv_state[(t + 0) * C + c] : qkv[(t + 0 - 3) * C + c];
    float in1 = t + 1 < 3 ? conv_state[(t + 1) * C + c] : qkv[(t + 1 - 3) * C + c];
    float in2 = t + 2 < 3 ? conv_state[(t + 2) * C + c] : qkv[(t + 2 - 3) * C + c];
    float in3 = qkv[t * C + c];
    float o = silu(w0 * in0 + w1 * in1 + w2 * in2 + w3 * in3);
    if (is_q || is_k) {
      float ss = tg_sum(o * o, scratch, tid, D);
      o *= metal::precise::rsqrt(ss + 1e-6f);
      if (is_q) {
        o *= metal::precise::rsqrt(float(D));
        q_out[t * qk + c] = o;
      } else {
        k_out[t * qk + (c - qk)] = o;
      }
    } else {
      v_out[t * (C - 2 * qk) + (c - 2 * qk)] = o;
    }
  }
  // Roll the conv window: new rows = concat[T..T+2]. Ascending r only reads rows > r.
  for (uint r = 0; r < 3; ++r) {
    uint pos = p.T + r;
    float v = pos < 3 ? conv_state[pos * C + c] : qkv[(pos - 3) * C + c];
    conv_state[r * C + c] = v;
  }
  if (tg == 0 && tid < p.Hv) {
    float A = metal::precise::exp(float(A_log[tid]));
    float dtb = float(dt_bias[tid]);
    for (uint t = 0; t < p.T; ++t) {
      float av = a[t * p.Hv + tid];
      g_out[t * p.Hv + tid] = metal::precise::exp(-A * softplus(av + dtb));
      beta_out[t * p.Hv + tid] = sigmoid(bb[t * p.Hv + tid]);
    }
  }
}

// ---- Gated DeltaNet recurrence + gated RMSNorm, one threadgroup (1024 threads) per value head ----
// state: [Hv, Dv=128, Dk=128] fp32. Simdgroup sg owns rows dv = sg*4..sg*4+3; lane owns dk = lane*4..+3.
//   S = g*S;  kv = S k;  delta = (v - kv) * beta;  S += delta k^T;  o = S q
//   out = rmsnorm(o) * w * silu(z)
struct GdnStepParams {
  uint T;
  uint Hk;
  uint Hv;
  float eps;
};

[[kernel]] void gdn_step(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                         const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],
                         const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                         const device float* z [[buffer(6)]], const device bfloat* norm_w [[buffer(7)]],
                         device float* out [[buffer(8)]], constant GdnStepParams& p [[buffer(9)]],
                         uint h [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  constexpr uint D = 128;
  const uint hk = h / (p.Hv / p.Hk);
  device float* S = state + h * D * D;
  float4 st[4];
  for (int r = 0; r < 4; ++r) st[r] = *(device float4*)(S + (sg * 4 + r) * D + lane * 4);
  float nw[4];
  for (int r = 0; r < 4; ++r) nw[r] = float(norm_w[sg * 4 + r]);

  for (uint t = 0; t < p.T; ++t) {
    float4 q4 = *(const device float4*)(q + (t * p.Hk + hk) * D + lane * 4);
    float4 k4 = *(const device float4*)(k + (t * p.Hk + hk) * D + lane * 4);
    float gg = g[t * p.Hv + h], be = beta[t * p.Hv + h];
    float o[4];
    float ss = 0.0f;
    for (int r = 0; r < 4; ++r) {
      st[r] *= gg;
      float kv = simd_sum(dot(st[r], k4));
      float delta = (v[(t * p.Hv + h) * D + sg * 4 + r] - kv) * be;
      st[r] += k4 * delta;
      o[r] = simd_sum(dot(st[r], q4));
      ss += o[r] * o[r];
    }
    if (lane == 0) scratch[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(scratch[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::precise::rsqrt(tot / float(D) + p.eps);
    if (lane < 4) {
      uint dv = sg * 4 + lane;
      float ov = lane == 0 ? o[0] : lane == 1 ? o[1] : lane == 2 ? o[2] : o[3];
      float nv = lane == 0 ? nw[0] : lane == 1 ? nw[1] : lane == 2 ? nw[2] : nw[3];
      uint idx = (t * p.Hv + h) * D + dv;
      out[idx] = ov * inv * nv * silu(z[idx]);
    }
  }
  for (int r = 0; r < 4; ++r) *(device float4*)(S + (sg * 4 + r) * D + lane * 4) = st[r];
}

// ---- Gated DeltaNet prefill path (NI_GDN_PAR=1): gdn_prep_par -> gdn_step_par_l* -> gdn_gnorm ----
// Same math as gdn_prep + gdn_step + rows_to_half, laid out for many tokens. Every sum follows the
// order of the kernel it replaces (see each kernel), so with simd_sum lowering to the ascending
// shuffle_xor butterfly (current Apple GPUs) the results are bit-identical.

// conv input row r of concat = [conv_state rows 0..2, qkv rows 0..T-1]
NI_INLINE float gdn_conv_in(const device float* qkv, const device float* conv_state, uint r, uint C, uint c) {
  return r < 3 ? conv_state[r * C + c] : qkv[(r - 3) * C + c];
}

// Token-parallel conv1d + SiLU + q/k L2 norm + decay/beta. One simdgroup per (block of 8 tokens,
// 128-channel chunk) slides the conv window (one new qkv row per token); lane owns channels
// lane + 32*i, so the q/k norm is gdn_prep's tg_sum without barriers: four simd_sums (one per
// simdgroup of gdn_prep) and a simd_sum over those four. The block holding tokens 0..2 is the only
// reader of conv_state for its channels and rolls it at the end (same lanes, ascending rows).
// grid: x = ceil(T/32) threadgroups x 128 threads (4 token blocks), y = C/128 + 1 (last row: g, beta).
[[kernel]] void gdn_prep_par(const device float* qkv [[buffer(0)]], const device float* a [[buffer(1)]],
                             const device float* bb [[buffer(2)]], device float* conv_state [[buffer(3)]],
                             const device bfloat* conv_w [[buffer(4)]], const device bfloat* A_log [[buffer(5)]],
                             const device bfloat* dt_bias [[buffer(6)]], device float* q_out [[buffer(7)]],
                             device float* k_out [[buffer(8)]], device float* v_out [[buffer(9)]],
                             device float* g_out [[buffer(10)]], device float* beta_out [[buffer(11)]],
                             constant GdnPrepParams& p [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 128, TB = 8;
  const uint t0 = (tg.x * nsg + sg) * TB;
  if (t0 >= p.T) return;
  const uint t1 = min(t0 + TB, p.T);
  const uint C = p.C;
  const uint qk = p.Hk * D;
  if (tg.y == C / D) {
    for (uint h = lane; h < p.Hv; h += 32) {
      float A = metal::precise::exp(float(A_log[h]));
      float dtb = float(dt_bias[h]);
      for (uint t = t0; t < t1; ++t) {
        float av = a[t * p.Hv + h];
        g_out[t * p.Hv + h] = metal::precise::exp(-A * softplus(av + dtb));
        beta_out[t * p.Hv + h] = sigmoid(bb[t * p.Hv + h]);
      }
    }
    return;
  }
  const uint c0 = tg.y * D + lane;
  const bool is_q = c0 < qk, is_k = c0 >= qk && c0 < 2 * qk;
  float w0[4], w1[4], w2[4], w3[4], h0[4], h1[4], h2[4];
  for (uint i = 0; i < 4; ++i) {
    uint c = c0 + 32 * i;
    w0[i] = conv_w[c * 4 + 0];
    w1[i] = conv_w[c * 4 + 1];
    w2[i] = conv_w[c * 4 + 2];
    w3[i] = conv_w[c * 4 + 3];
    h0[i] = gdn_conv_in(qkv, conv_state, t0 + 0, C, c);
    h1[i] = gdn_conv_in(qkv, conv_state, t0 + 1, C, c);
    h2[i] = gdn_conv_in(qkv, conv_state, t0 + 2, C, c);
  }
  for (uint t = t0; t < t1; ++t) {
    float o[4];
    for (uint i = 0; i < 4; ++i) {
      float in3 = qkv[t * C + c0 + 32 * i];
      o[i] = silu(w0[i] * h0[i] + w1[i] * h1[i] + w2[i] * h2[i] + w3[i] * in3);
      h0[i] = h1[i];
      h1[i] = h2[i];
      h2[i] = in3;
    }
    if (is_q || is_k) {
      float s0 = simd_sum(o[0] * o[0]), s1 = simd_sum(o[1] * o[1]);
      float s2 = simd_sum(o[2] * o[2]), s3 = simd_sum(o[3] * o[3]);
      float ss = simd_sum(lane == 0 ? s0 : lane == 1 ? s1 : lane == 2 ? s2 : lane == 3 ? s3 : 0.0f);
      float inv = metal::precise::rsqrt(ss + 1e-6f);
      for (uint i = 0; i < 4; ++i) {
        float x = o[i] * inv;
        if (is_q) q_out[t * qk + c0 + 32 * i] = x * metal::precise::rsqrt(float(D));
        else k_out[t * qk + (c0 + 32 * i - qk)] = x;
      }
    } else {
      for (uint i = 0; i < 4; ++i) v_out[t * (C - 2 * qk) + (c0 + 32 * i - 2 * qk)] = o[i];
    }
  }
  // Roll the conv window (as gdn_prep): new rows = concat[T..T+2]; ascending r only reads rows > r.
  if (t0 == 0) {
    for (uint r = 0; r < 3; ++r)
      for (uint i = 0; i < 4; ++i) {
        uint c = c0 + 32 * i;
        conv_state[r * C + c] = gdn_conv_in(qkv, conv_state, p.T + r, C, c);
      }
  }
}

// Recurrence over T tokens with the value rows of a head spread over simdgroups and threadgroups:
// row dv of S only ever sees v[dv], so rows run independently with no barriers, and the gated
// RMSNorm across the head follows in gdn_gnorm (o written raw to `out`). L lanes share one row and
// own N = 32/L float4 chunks of it (dk = li*4N .. li*4N + 4N-1); a simdgroup holds N rows. Each
// chunk is one gdn_step lane, so the dot products sum in gdn_step's simd_sum order: per-chunk dot,
// an ascending pairwise tree over the lane's chunks, then shuffle_xor 1, 2, .., L/2.
// grid: Hv * 128/(N*nsg) threadgroups x 32*nsg threads.
template <uint N, uint L>
NI_INLINE float gdn_row_sum(thread float* part) {
  for (uint w = 1; w < N; w *= 2)
    for (uint j = 0; j + w < N; j += 2 * w) part[j] += part[j + w];
  float s = part[0];
  if (L == 32) return simd_sum(s);
  for (uint m = 1; m < L; m *= 2) s += simd_shuffle_xor(s, ushort(m));
  return s;
}

template <uint L>
NI_INLINE void gdn_step_par_impl(const device float* q, const device float* k, const device float* v,
                                 const device float* g, const device float* beta, device float* state,
                                 device float* out, constant GdnStepParams& p, uint tg, uint sg, uint nsg,
                                 uint lane) {
  constexpr uint D = 128, N = 32 / L;
  const uint tph = D / (N * nsg);  // threadgroups per head
  const uint h = tg / tph;
  const uint dv = ((tg % tph) * nsg + sg) * N + lane / L;
  const uint li = lane % L;
  const uint hk = h / (p.Hv / p.Hk);
  device float* S = state + (h * D + dv) * D + li * 4 * N;
  float4 st[N];
  for (uint j = 0; j < N; ++j) st[j] = *(device float4*)(S + j * 4);

  for (uint t = 0; t < p.T; ++t) {
    const device float4* q4 = (const device float4*)(q + (t * p.Hk + hk) * D + li * 4 * N);
    const device float4* k4 = (const device float4*)(k + (t * p.Hk + hk) * D + li * 4 * N);
    float gg = g[t * p.Hv + h], be = beta[t * p.Hv + h];
    float4 kk[N];
    float part[N];
    for (uint j = 0; j < N; ++j) {
      kk[j] = k4[j];
      st[j] *= gg;
      part[j] = dot(st[j], kk[j]);
    }
    float kv = gdn_row_sum<N, L>(part);
    float delta = (v[(t * p.Hv + h) * D + dv] - kv) * be;
    for (uint j = 0; j < N; ++j) {
      st[j] += kk[j] * delta;
      part[j] = dot(st[j], q4[j]);
    }
    float o = gdn_row_sum<N, L>(part);
    if (li == 0) out[(t * p.Hv + h) * D + dv] = o;
  }
  for (uint j = 0; j < N; ++j) *(device float4*)(S + j * 4) = st[j];
}

#define NI_GDN_STEP_PAR(NAME, L)                                                                           \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],           \
                       const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],           \
                       const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],          \
                       device float* out [[buffer(6)]], constant GdnStepParams& p [[buffer(7)]],           \
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    gdn_step_par_impl<L>(q, k, v, g, beta, state, out, p, tg, sg, nsg, lane);                             \
  }

// name: gdn_step_par_l{lanes per row}; rows per simdgroup = 32/L, state floats per lane = 128/L
NI_GDN_STEP_PAR(gdn_step_par_l4, 4)
NI_GDN_STEP_PAR(gdn_step_par_l8, 8)
NI_GDN_STEP_PAR(gdn_step_par_l16, 16)
NI_GDN_STEP_PAR(gdn_step_par_l32, 32)

// gdn_step's epilogue as its own pass, fused with rows_to_half:
//   y[t, h*128 + dv] = half(rmsnorm(o[t, h, :])[dv] * w[dv] * silu(z[t, h, dv]))
// One simdgroup per (token, head); lane l holds rows 4l..4l+3 and sums them like gdn_step's
// simdgroup l. grid: ceil(T*Hv / 8) threadgroups x 256 threads.
[[kernel]] void gdn_gnorm(const device float* o [[buffer(0)]], const device float* z [[buffer(1)]],
                          const device bfloat* norm_w [[buffer(2)]], device half* y [[buffer(3)]],
                          constant GdnStepParams& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 128;
  const uint row = tg * nsg + sg;  // t * Hv + h
  if (row >= p.T * p.Hv) return;
  const uint idx = row * D + lane * 4;
  float4 ov = *(const device float4*)(o + idx);
  float ss = 0.0f;
  for (int r = 0; r < 4; ++r) ss += ov[r] * ov[r];
  float tot = simd_sum(ss);
  float inv = metal::precise::rsqrt(tot / float(D) + p.eps);
  float4 zz = *(const device float4*)(z + idx);
  half4 y4;
  for (int r = 0; r < 4; ++r) {
    float nv = float(norm_w[lane * 4 + r]);
    y4[r] = half(ov[r] * inv * nv * silu(zz[r]));
  }
  *(device half4*)(y + idx) = y4;
}

// ---- Prefill helpers ----
// Y[t, :] = half(rmsnorm(X[t, :]) * w) when w is given (norm=1), else half(X[t, :]).
// grid: T threadgroups x 256 threads.
struct RowsParams {
  uint D;
  uint norm;
  float eps;
};
[[kernel]] void rows_to_half(const device float* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                             device half* Y [[buffer(2)]], constant RowsParams& p [[buffer(3)]],
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* x = X + t * p.D;
  float inv = 1.0f;
  if (p.norm) {
    float ss = 0.0f;
    for (uint i = tid; i < p.D; i += ntg) ss += x[i] * x[i];
    ss = tg_sum(ss, scratch, tid, ntg);
    inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  }
  for (uint i = tid; i < p.D; i += ntg) Y[t * p.D + i] = half(p.norm ? x[i] * inv * float(w[i]) : x[i]);
}
