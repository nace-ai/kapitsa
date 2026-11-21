#include "common.h"

// Inkling-2-Mini bundled DFlare / DSpark block drafter (mtp.safetensors):
// 3 Qwen3-style layers (RoPE theta=1e6, QK-norm, SwiGLU MLP, sliding context window 1023)
// with per-layer softmax tap fusion (dflare_norm) + bidirectional block attention +
// RNN Markov head (rank 256, top-32 candidate rescoring against target unembed).

struct DflMixP {
  uint R, H, row_off, total_rows;
  float a0, a1, a2, a3, eps;
};

// Mixes 4 target taps [4, total_rows, H] for rows [row_off .. row_off + R) with alpha, RMSNorm -> out [R, H]
[[kernel]] void dflare_mix_norm(const device float* taps [[buffer(0)]], const device float* w [[buffer(1)]],
                                device float* out [[buffer(2)]], constant DflMixP& p [[buffer(3)]],
                                uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const size_t stride = size_t(p.total_rows) * p.H;
  const size_t roff = size_t(p.row_off + r) * p.H;
  const device float* t0 = taps + 0 * stride + roff;
  const device float* t1 = taps + 1 * stride + roff;
  const device float* t2 = taps + 2 * stride + roff;
  const device float* t3 = taps + 3 * stride + roff;
  float xs[8];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = p.a0 * t0[c] + p.a1 * t1[c] + p.a2 * t2[c] + p.a3 * t3[c];
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  device float* dst = out + size_t(r) * p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * w[c];
}

struct DflQ8P {
  uint M, N, K, add;
};

NI_INLINE float dfl_q8_dot16(uint4 v, const device float4* xp) {
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

// Q8_0 matmul for draft layers (2D grid: tg.x = n/4, tg.y = m/8, supports any M >= 1)
[[kernel]] void dflare_q8_mm(const device float* x [[buffer(0)]], const device uchar* q [[buffer(1)]],
                             const device half* s [[buffer(2)]], device float* y [[buffer(3)]],
                             constant DflQ8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
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
      const device float4* xp = (const device float4*)(x + size_t(m0 + m) * p.K + blk * 32);
      acc[m] += (dfl_q8_dot16(v0, xp) + dfl_q8_dot16(v1, xp + 4)) * sc;
    }
  }
  for (uint m = 0; m < mc; ++m) {
    const float sum = simd_sum(acc[m]);
    if (lane == 0) {
      device float* dst = y + size_t(m0 + m) * p.N + n;
      *dst = p.add ? (*dst + sum) : sum;
    }
  }
}

struct DflCtxKvP {
  uint R, Hkv, D, pos0, cap;
  float eps, theta;
};

// Apply k_norm_ctx + NeoX split-half RoPE to ctx_k [R, Hkv, D] and write half kc/vc into ring [cap, Hkv*D]
[[kernel]] void dflare_ctx_kv_write(const device float* kin [[buffer(0)]], const device float* vin [[buffer(1)]],
                                    const device float* knw [[buffer(2)]], device half* kc [[buffer(3)]],
                                    device half* vc [[buffer(4)]], constant DflCtxKvP& p [[buffer(5)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float kbuf[128];
  const uint h = tg.x, r = tg.y;
  const uint pos = p.pos0 + r;
  const uint slot = pos % p.cap;
  const size_t in_off = (size_t(r) * p.Hkv + h) * p.D;
  const size_t out_off = (size_t(slot) * p.Hkv + h) * p.D;

  const float kv = kin[in_off + tid];
  const float ss = tg_sum(kv * kv, scratch, tid, p.D);
  const float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  kbuf[tid] = kv * inv * knw[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_d = p.D / 2;
  if (tid < half_d) {
    const float freq = metal::precise::pow(p.theta, -float(2 * tid) / float(p.D));
    const float ang = float(pos) * freq;
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    const float x0 = kbuf[tid], x1 = kbuf[tid + half_d];
    kc[out_off + tid] = half(x0 * cs - x1 * sn);
    kc[out_off + tid + half_d] = half(x0 * sn + x1 * cs);
  }
  vc[out_off + tid] = half(vin[in_off + tid]);
}

struct DflBlockP {
  uint B, H, anchor_idx, V;
  float eps;
};

// Build draft block input [B, H]: row 0 = embed_norm(embed[tokens[anchor_idx]]), rows 1..B-1 = mask_embed
[[kernel]] void dflare_build_block(const device int* tokens [[buffer(0)]], const device bfloat* embed [[buffer(1)]],
                                   const device bfloat* embed_norm [[buffer(2)]], const device float* mask_emb [[buffer(3)]],
                                   device float* out [[buffer(4)]], constant DflBlockP& p [[buffer(5)]],
                                   uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
  device float* dst = out + size_t(b) * p.H;
  if (b > 0) {
    for (uint c = tid; c < p.H; c += ntg) dst[c] = mask_emb[c];
    return;
  }
  threadgroup float scratch[32];
  const int raw_id = tokens[p.anchor_idx];
  const uint id = (raw_id >= 0 && uint(raw_id) < p.V) ? uint(raw_id) : 0u;
  const device bfloat* row = embed + size_t(id) * p.H;
  float xs[8];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = float(row[c]);
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * float(embed_norm[c]);
}

struct DflQkRopeP {
  uint B, Hq, Hkv, D, pos0;
  float eps, theta;
};

// In-place q_norm + NeoX RoPE on q [B, Hq, D] and k_norm + NeoX RoPE on k [B, Hkv, D]
[[kernel]] void dflare_qk_rope(device float* q [[buffer(0)]], device float* k [[buffer(1)]],
                               const device float* qnw [[buffer(2)]], const device float* knw [[buffer(3)]],
                               constant DflQkRopeP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float buf[128];
  const uint kh = tg.x, b = tg.y;
  const bool is_q = kh < p.Hq;
  const uint h = is_q ? kh : (kh - p.Hq);
  device float* vec = (is_q ? q : k) + (size_t(b) * (is_q ? p.Hq : p.Hkv) + h) * p.D;
  const device float* nw = is_q ? qnw : knw;
  const float val = vec[tid];
  const float ss = tg_sum(val * val, scratch, tid, p.D);
  const float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  buf[tid] = val * inv * nw[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_d = p.D / 2;
  if (tid < half_d) {
    const float freq = metal::precise::pow(p.theta, -float(2 * tid) / float(p.D));
    const float ang = float(p.pos0 + b) * freq;
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    const float x0 = buf[tid], x1 = buf[tid + half_d];
    vec[tid] = x0 * cs - x1 * sn;
    vec[tid + half_d] = x0 * sn + x1 * cs;
  }
}

struct DflAttnP {
  uint B, Hq, Hkv, D, pos0, ctx_len, ctx_cap;
  float scale;
};

// Bidirectional block attention over [ctx_len context ring keys, B block keys] (no causal mask, no sinks)
[[kernel]] void dflare_attn(const device float* q [[buffer(0)]], const device half* ctx_k [[buffer(1)]],
                            const device half* ctx_v [[buffer(2)]], const device float* blk_k [[buffer(3)]],
                            const device float* blk_v [[buffer(4)]], device float* out [[buffer(5)]],
                            constant DflAttnP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, b = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4;
  const size_t qoff = (size_t(b) * p.Hq + h) * p.D + d0;
  const float4 qv = *(const device float4*)(q + qoff);

  const uint c_start = p.pos0 - p.ctx_len;
  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);

  // Context ring keys
  for (uint i = sg; i < p.ctx_len; i += 4) {
    const uint slot = (c_start + i) % p.ctx_cap;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = float4(*(const device half4*)(ctx_k + koff));
    const float sc = simd_sum(dot(qv, kv)) * p.scale;
    const float4 vv = float4(*(const device half4*)(ctx_v + koff));
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
  // Bidirectional block keys (all B positions)
  for (uint j = sg; j < p.B; j += 4) {
    const size_t bkoff = (size_t(j) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = *(const device float4*)(blk_k + bkoff);
    const float sc = simd_sum(dot(qv, kv)) * p.scale;
    const float4 vv = *(const device float4*)(blk_v + bkoff);
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
    *(device float4*)(out + qoff) = oo / ll;
  }
}

struct DflSwigluP {
  uint n;
};

[[kernel]] void dflare_swiglu(const device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
                              device float* out [[buffer(2)]], constant DflSwigluP& p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  const float g = gate[i];
  out[i] = (g / (1.0f + metal::precise::exp(-g))) * up[i];
}

// ---- RNN Markov Head (rank R=256, folded Q8_0 [V, 256] full-vocab rescoring) ----
struct DflMarkovP {
  uint R, H, K, step, tok_prev_idx, tok_out_idx;
  float softcap, inv_softcap;
};

// Step 1 of Markov head: build concat([state(R), w1[prev_tok](R), draft_h[step](H), rows_f(H)]) -> [2R+2H]
[[kernel]] void dflare_markov_concat(const device float* state [[buffer(0)]], const device bfloat* w1 [[buffer(1)]],
                                     const device float* draft_h [[buffer(2)]], const device float* rows_f [[buffer(3)]],
                                     const device int* tokens [[buffer(4)]], device float* rnn_in [[buffer(5)]],
                                     constant DflMarkovP& p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  const uint tot = 2u * p.R + 2u * p.H;
  if (i >= tot) return;
  if (i < p.R) {
    rnn_in[i] = p.step == 0u ? 0.0f : state[i];
  } else if (i < 2u * p.R) {
    const int tok = tokens[p.tok_prev_idx];
    const uint tid = tok >= 0 ? uint(tok) : 0u;
    rnn_in[i] = float(w1[size_t(tid) * p.R + (i - p.R)]);
  } else if (i < 2u * p.R + p.H) {
    rnn_in[i] = draft_h[size_t(p.step) * p.H + (i - 2u * p.R)];
  } else {
    rnn_in[i] = rows_f[i - (2u * p.R + p.H)];
  }
}

// Step 2: rnn_proj [3R, 2R+2H] BF16 + bias -> update state [R] & write latent [R]
[[kernel]] void dflare_markov_gru(const device float* rnn_in [[buffer(0)]], const device bfloat* proj_w [[buffer(1)]],
                                  const device float* proj_b [[buffer(2)]], device float* state [[buffer(3)]],
                                  device float* latent [[buffer(4)]], constant DflMarkovP& p [[buffer(5)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg * 4u + sg;
  if (r >= p.R) return;
  const uint K = 2u * p.R + 2u * p.H;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  const device bfloat* w0 = proj_w + size_t(r) * K;
  const device bfloat* w1 = proj_w + size_t(r + p.R) * K;
  const device bfloat* w2 = proj_w + size_t(r + 2u * p.R) * K;
  for (uint k = lane * 4u; k < K; k += 128u) {
    const float4 xv = *(const device float4*)(rnn_in + k);
    const uint2 q0 = *(const device uint2*)(w0 + k);
    const uint2 q1 = *(const device uint2*)(w1 + k);
    const uint2 q2 = *(const device uint2*)(w2 + k);
    const float4 f0 = float4(as_type<float>(q0.x << 16), as_type<float>(q0.x & 0xFFFF0000u),
                             as_type<float>(q0.y << 16), as_type<float>(q0.y & 0xFFFF0000u));
    const float4 f1 = float4(as_type<float>(q1.x << 16), as_type<float>(q1.x & 0xFFFF0000u),
                             as_type<float>(q1.y << 16), as_type<float>(q1.y & 0xFFFF0000u));
    const float4 f2 = float4(as_type<float>(q2.x << 16), as_type<float>(q2.x & 0xFFFF0000u),
                             as_type<float>(q2.y << 16), as_type<float>(q2.y & 0xFFFF0000u));
    a0 += dot(xv, f0);
    a1 += dot(xv, f1);
    a2 += dot(xv, f2);
  }
  const float p0 = simd_sum(a0) + proj_b[r];
  const float p1 = simd_sum(a1) + proj_b[r + p.R];
  const float p2 = simd_sum(a2) + proj_b[r + 2u * p.R];
  if (lane == 0) {
    const float st_prev = p.step == 0u ? 0.0f : state[r];
    const float gate = 1.0f / (1.0f + metal::precise::exp(-p0));
    state[r] = gate * st_prev + (1.0f - gate) * metal::precise::tanh(p1);
    latent[r] = metal::precise::tanh(p2);
  }
}

// One-time load kernel: fold unembed [V, H] BF16 @ hidden_out.weight [H, 256] BF16 -> Q8_0 [V, 256]
struct DflFoldP {
  uint V, H, R;
};

[[kernel]] void dflare_fold_mm_q8(const device bfloat* unembed [[buffer(0)]], const device bfloat* hout_w [[buffer(1)]],
                                  device uchar* fold_q [[buffer(2)]], device half* fold_s [[buffer(3)]],
                                  constant DflFoldP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[32][36];
  threadgroup float ws[32][36];
  const uint blk = tg.x;        // 0..7 (each block = 32 cols of R=256)
  const uint r0 = blk * 32u;
  const uint v0 = tg.y * 32u;
  const uint sgm = sg >> 1;     // 0..1 (16 rows)
  const uint sgn = sg & 1u;     // 0..1 (16 cols)

  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

  for (uint k0 = 0; k0 < p.H; k0 += 32u) {
    for (uint idx = tid; idx < 1024u; idx += 128u) {
      const uint r = idx >> 5, c = idx & 31u;
      const uint v = v0 + r;
      xs[r][c] = (v < p.V && k0 + c < p.H) ? float(unembed[size_t(v) * p.H + (k0 + c)]) : 0.0f;
      ws[r][c] = (k0 + r < p.H && r0 + c < p.R) ? float(hout_w[size_t(k0 + r) * p.R + (r0 + c)]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32u; kk += 8u) {
      simdgroup_float8x8 a[2], b[2];
      for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], &xs[sgm * 16 + i * 8][kk], 36);
      for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], &ws[kk][sgn * 16 + j * 8], 36);
      for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], &xs[sgm * 16 + i * 8][sgn * 16 + j * 8], 36);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint nblk = p.R / 32u;
  for (uint i = 0; i < 8u; ++i) {
    const uint r = sg * 8u + i;
    const uint v = v0 + r;
    if (v >= p.V) continue;
    const float val = xs[r][lane];
    const float amax = simd_max(fabs(val));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? (1.0f / d) : 0.0f;
    fold_q[size_t(v) * p.R + r0 + lane] = uchar(char(clamp(int(rint(val * id)), -127, 127)));
    if (lane == 0) fold_s[size_t(v) * nblk + blk] = half(d);
  }
}

struct DflFoldPickP {
  uint V, R, step, tok_out_idx;
};

// Fused full-vocab Markov rescore + stage-1 argmax:
// score(v) = base_logits[step, v] + fold_b[v] + dot(latent[0..255], fold_q8[v, 0..255])
// Launched with ceil(V / 1024) threadgroups of 128 threads (16 groups of 8 lanes; 1 lane = 1 Q8 block of 32).
[[kernel]] void dflare_fold_pick_s1(const device float* latent [[buffer(0)]], const device uchar* fold_q [[buffer(1)]],
                                    const device half* fold_s [[buffer(2)]], const device float* fold_b [[buffer(3)]],
                                    const device float* base_logits [[buffer(4)]], device float* out_v [[buffer(5)]],
                                    device uint* out_i [[buffer(6)]], constant DflFoldPickP& p [[buffer(7)]],
                                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float lat[256];
  threadgroup float sv[16];
  threadgroup uint si[16];
  for (uint i = tid; i < 256u; i += 128u) lat[i] = latent[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint grp = tid >> 3;   // 0..15 (which of the 16 concurrent vocab rows)
  const uint blk = lane & 7u;  // 0..7  (which 32-elem Q8 block of R=256)
  const uint v_start = tg * 1024u;
  const uint v_end = min(p.V, v_start + 1024u);
  const device float* br = base_logits + size_t(p.step) * p.V;

  // Load this lane's 32 floats of latent into registers (reused across all 64 vocab rows!)
  float4 lx[8];
  for (uint j = 0; j < 8u; ++j) lx[j] = *(const threadgroup float4*)(&lat[blk * 32u + j * 4u]);

  float best_v = -INFINITY;
  uint best_i = 0u;
  for (uint v = v_start + grp; v < v_end; v += 16u) {
    const float sc = float(fold_s[size_t(v) * 8u + blk]);
    const device uint4* qp = (const device uint4*)(fold_q + size_t(v) * 256u + blk * 32u);
    const uint4 q0 = qp[0], q1 = qp[1];
    float d = 0.0f;
    for (uint m = 0; m < 4u; ++m) {
      const uint w = q0[m];
      const float4 xv = lx[m];
      d += float(as_type<char>(uchar(w & 0xFFu))) * xv.x +
           float(as_type<char>(uchar((w >> 8) & 0xFFu))) * xv.y +
           float(as_type<char>(uchar((w >> 16) & 0xFFu))) * xv.z +
           float(as_type<char>(uchar((w >> 24) & 0xFFu))) * xv.w;
    }
    for (uint m = 0; m < 4u; ++m) {
      const uint w = q1[m];
      const float4 xv = lx[m + 4u];
      d += float(as_type<char>(uchar(w & 0xFFu))) * xv.x +
           float(as_type<char>(uchar((w >> 8) & 0xFFu))) * xv.y +
           float(as_type<char>(uchar((w >> 16) & 0xFFu))) * xv.z +
           float(as_type<char>(uchar((w >> 24) & 0xFFu))) * xv.w;
    }
    float s = d * sc;
    s += simd_shuffle_xor(s, 1);
    s += simd_shuffle_xor(s, 2);
    s += simd_shuffle_xor(s, 4);
    if (blk == 0u) {
      const float score = br[v] + fold_b[v] + s;
      if (score > best_v) {
        best_v = score;
        best_i = v;
      }
    }
  }
  if (blk == 0u) {
    sv[grp] = best_v;
    si[grp] = best_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float mv = -INFINITY;
    uint mi = 0u;
    for (uint g = 0; g < 16u; ++g) {
      if (sv[g] > mv) {
        mv = sv[g];
        mi = si[g];
      }
    }
    out_v[tg] = mv;
    out_i[tg] = mi;
  }
}

[[kernel]] void dflare_fold_pick_s2(const device float* in_v [[buffer(0)]], const device uint* in_i [[buffer(1)]],
                                    device int* tokens [[buffer(2)]], constant DflFoldPickP& p [[buffer(3)]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[256];
  threadgroup uint si[256];
  const uint n = (p.V + 1023u) / 1024u;
  sv[tid] = tid < n ? in_v[tid] : -INFINITY;
  si[tid] = tid < n ? in_i[tid] : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 128u; s > 0u; s >>= 1) {
    if (tid < s && sv[tid + s] > sv[tid]) {
      sv[tid] = sv[tid + s];
      si[tid] = si[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) tokens[p.tok_out_idx] = int(si[0]);
}
