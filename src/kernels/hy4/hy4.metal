#include "common.h"
#include "kv_int8.h"

// ---- Hy4-flash (hy_v4_flash) blocks: hyper-connections, RMSNorm rows, SwiGLU, sigmoid router, Engram mix, MLA
// attention with sinks and a sliding window. Host side and semantics: src/model/hy4_ops.{h,cpp}; reference:
// transformers' modeling_hy_v4_flash.py. Param structs mirror ni::kp::Hy4* in src/model/kparams.h.
// Weight-like inputs (norm weights, biases, hc params, sinks) come as stored: dtype code 0 fp32, 1 bf16, 2 fp16.

NI_INLINE float hy4_ld(const device void* p, uint i, uint t) {
  return t == 1 ? float(((const device bfloat*)p)[i])
       : t == 2 ? float(((const device half*)p)[i]) : ((const device float*)p)[i];
}
NI_INLINE float hy4_sigmoid(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

// ---- hyper-connection pre (HYV4FlashHyperConnection / HYV4FlashHyperHead) ----
// grid: R threadgroups x 1024 threads (any multiple of 32 up to 1024). S = 4 streams, nout 4 or 8.
struct Hy4HcP {
  uint D, nout, has_y, u_half, has_post, fn_t, base_t, scale_t;
  float rms_eps, hc_eps, mag;
};

template <typename FT>
NI_INLINE void hy4_hc_pre_impl(device float* X, const device float* y, const device float* post_prev,
                               const device FT* fn, const device void* base, const device void* scale, device void* u,
                               device float* post_out, constant Hy4HcP& p, uint r, uint tid, uint ntg, uint lane,
                               uint sg, threadgroup float (*red)[9], threadgroup float* coef) {
  constexpr uint S = 4;
  const uint D = p.D, SD = S * D;
  device float* xr = X + size_t(r) * SD;
  float pp[S];
  NI_UNROLL for (uint s = 0; s < S; ++s) pp[s] = p.has_y ? post_prev[r * S + s] : 0.0f;
  float acc[9];
  NI_UNROLL for (uint j = 0; j < 9; ++j) acc[j] = 0.0f;
  for (uint c = tid; c < D; c += ntg) {
    const float yv = p.has_y ? y[size_t(r) * D + c] : 0.0f;
    NI_UNROLL for (uint s = 0; s < S; ++s) {
      float x = xr[s * D + c];
      if (p.has_y) {
        x += pp[s] * yv;
        xr[s * D + c] = x;
      }
      acc[8] += x * x;
      NI_UNROLL for (uint j = 0; j < 8; ++j)
        if (j < p.nout) acc[j] += float(fn[j * SD + s * D + c]) * x;
    }
  }
  NI_UNROLL for (uint j = 0; j < 9; ++j) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) red[sg][j] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 9) {
    const uint nsg = (ntg + 31) / 32;
    float v = 0.0f;
    for (uint i = 0; i < nsg; ++i) v += red[i][tid];
    coef[8 + tid] = v;  // coef[8..16]: the 8 dots, then ss
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.nout) {
    const float inv = metal::precise::rsqrt(coef[16] / float(SD) + p.rms_eps);
    const float m = coef[8 + tid] * inv;
    if (tid < S) {
      coef[tid] = hy4_sigmoid(m * hy4_ld(scale, 0, p.scale_t) + hy4_ld(base, tid, p.base_t)) + p.hc_eps;
    } else if (p.has_post) {
      post_out[r * S + tid - S] =
          p.mag * hy4_sigmoid(m * hy4_ld(scale, 1, p.scale_t) + hy4_ld(base, tid, p.base_t)) + p.hc_eps;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float c0 = coef[0], c1 = coef[1], c2 = coef[2], c3 = coef[3];
  for (uint c = tid; c < D; c += ntg) {
    const float v = c0 * xr[c] + c1 * xr[D + c] + c2 * xr[2 * D + c] + c3 * xr[3 * D + c];
    if (p.u_half) ((device half*)u)[size_t(r) * D + c] = half(v);
    else ((device float*)u)[size_t(r) * D + c] = v;
  }
}

[[kernel]] void hy4_hc_pre(device float* X [[buffer(0)]], const device float* y [[buffer(1)]],
                           const device float* post_prev [[buffer(2)]], const device void* fn [[buffer(3)]],
                           const device void* base [[buffer(4)]], const device void* scale [[buffer(5)]],
                           device void* u [[buffer(6)]], device float* post_out [[buffer(7)]],
                           constant Hy4HcP& p [[buffer(8)]], uint r [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[32][9];
  threadgroup float coef[17];
  if (p.fn_t == 1)
    hy4_hc_pre_impl<bfloat>(X, y, post_prev, (const device bfloat*)fn, base, scale, u, post_out, p, r, tid, ntg, lane,
                            sg, red, coef);
  else if (p.fn_t == 2)
    hy4_hc_pre_impl<half>(X, y, post_prev, (const device half*)fn, base, scale, u, post_out, p, r, tid, ntg, lane, sg,
                          red, coef);
  else
    hy4_hc_pre_impl<float>(X, y, post_prev, (const device float*)fn, base, scale, u, post_out, p, r, tid, ntg, lane,
                           sg, red, coef);
}

// X[r][s][c] += post[r*4 + s] * y[r][c]. grid: (ceil(D/256), R) x 256.
[[kernel]] void hy4_hc_inject(device float* X [[buffer(0)]], const device float* y [[buffer(1)]],
                              const device float* post [[buffer(2)]], constant uint& D [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= D) return;
  const float yv = y[size_t(r) * D + c];
  device float* xr = X + size_t(r) * 4 * D;
  NI_UNROLL for (uint s = 0; s < 4; ++s) xr[s * D + c] += post[r * 4 + s] * yv;
}

// X[r][s][c] = emb[r][c]. grid: (ceil(D/256), R) x 256.
[[kernel]] void hy4_hc_expand(const device float* emb [[buffer(0)]], device float* X [[buffer(1)]],
                              constant uint& D [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= D) return;
  const float v = emb[size_t(r) * D + c];
  device float* xr = X + size_t(r) * 4 * D;
  NI_UNROLL for (uint s = 0; s < 4; ++s) xr[s * D + c] = v;
}

// ---- RMSNorm rows: y = w * (x * rsqrt(mean(x^2) + eps)); y may alias x (fp32). grid: R x 1024 ----
struct Hy4NormP {
  uint D, w_t, y_half;
  float eps;
};
[[kernel]] void hy4_rmsnorm(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                            device void* y [[buffer(2)]], constant Hy4NormP& p [[buffer(3)]],
                            uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* xr = x + size_t(r) * p.D;
  float ss = 0.0f;
  for (uint c = tid; c < p.D; c += ntg) ss += xr[c] * xr[c];
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  for (uint c = tid; c < p.D; c += ntg) {
    const float v = hy4_ld(w, c, p.w_t) * (xr[c] * inv);
    if (p.y_half) ((device half*)y)[size_t(r) * p.D + c] = half(v);
    else ((device float*)y)[size_t(r) * p.D + c] = v;
  }
}

// y[i] += x[i], i < n. grid: ceil(n/256) x 256.
[[kernel]] void hy4_add(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] += x[i];
}

// h = silu(min(g, limit)) * clamp(u, -limit, limit) (limit > 0), silu(g) * u (limit <= 0). grid: ceil(n/256) x 256.
struct Hy4SwigluP {
  uint n, h_half;
  float limit;
};
[[kernel]] void hy4_swiglu(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                           device void* h [[buffer(2)]], constant Hy4SwigluP& p [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  float gv = g[i], uv = u[i];
  if (p.limit > 0.0f) {
    gv = min(gv, p.limit);
    uv = clamp(uv, -p.limit, p.limit);
  }
  const float v = gv / (1.0f + metal::precise::exp(-gv)) * uv;
  if (p.h_half) ((device half*)h)[i] = half(v);
  else ((device float*)h)[i] = v;
}

// ---- sigmoid router (HYV4FlashTopkRouter, n_group 1): one simdgroup a row, 4 rows a threadgroup of 128 ----
// choose top k of sigmoid(l) + bias (ties: lower expert), weight sigmoid(l) / (sum + 1e-20) * scale. E <= 512, k <= 16.
struct Hy4RouterP {
  uint R, E, k, bias_t;
  float scale;
};
[[kernel]] void hy4_router(const device float* logits [[buffer(0)]], const device void* bias [[buffer(1)]],
                           device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                           constant Hy4RouterP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 4 + sg;
  if (row >= p.R) return;
  float sc[16], ch[16];
  NI_UNROLL for (uint i = 0; i < 16; ++i) {
    const uint e = lane + 32 * i;
    if (e < p.E) {
      sc[i] = hy4_sigmoid(logits[size_t(row) * p.E + e]);
      ch[i] = sc[i] + hy4_ld(bias, e, p.bias_t);
    } else {
      sc[i] = 0.0f;
      ch[i] = -INFINITY;
    }
  }
  float sum = 0.0f, my_w = 0.0f;
  uint my_i = 0;
  for (uint j = 0; j < p.k; ++j) {
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    NI_UNROLL for (uint i = 0; i < 16; ++i)
      if (ch[i] > bv) {
        bv = ch[i];
        bi = lane + 32 * i;
      }
    const float gv = simd_max(bv);
    const uint gi = simd_min(bv == gv ? bi : 0xffffffffu);
    float s_sel = 0.0f;
    if (gi % 32 == lane) {
      NI_UNROLL for (uint i = 0; i < 16; ++i)
        if (lane + 32 * i == gi) {
          s_sel = sc[i];
          ch[i] = -INFINITY;
        }
    }
    s_sel = simd_shuffle(s_sel, gi % 32);
    sum += s_sel;
    if (lane == j) {
      my_w = s_sel;
      my_i = gi;
    }
  }
  if (lane < p.k) {
    idx[row * p.k + lane] = my_i;
    wts[row * p.k + lane] = my_w / (sum + 1e-20f) * p.scale;
  }
}

// ---- Engram mix (HYV4FlashEngramLayer.forward after the lookup), in place on X. grid: R x 1024 ----
// masked (p.masked): a row whose mask[r] is 0 (its token is a pad, image or video id: the reference's engram_mask)
// gets no Engram output, as the reference's masked_fill
struct Hy4EngramP {
  uint D, R, t_vb, t_kb, t_nk, t_nq, t_pg, t_hs, masked;
  float eps, inv_sqrt_d;
};
[[kernel]] void hy4_engram(device float* X [[buffer(0)]], const device float* vraw [[buffer(1)]],
                           const device float* kraw [[buffer(2)]], const device void* vb [[buffer(3)]],
                           const device void* kb0 [[buffer(4)]], const device void* kb1 [[buffer(5)]],
                           const device void* kb2 [[buffer(6)]], const device void* kb3 [[buffer(7)]],
                           const device void* nk0 [[buffer(8)]], const device void* nk1 [[buffer(9)]],
                           const device void* nk2 [[buffer(10)]], const device void* nk3 [[buffer(11)]],
                           const device void* nq0 [[buffer(12)]], const device void* nq1 [[buffer(13)]],
                           const device void* nq2 [[buffer(14)]], const device void* nq3 [[buffer(15)]],
                           const device void* pg [[buffer(16)]], const device void* hs0 [[buffer(17)]],
                           const device void* hs1 [[buffer(18)]], const device void* hs2 [[buffer(19)]],
                           const device void* hs3 [[buffer(20)]], constant Hy4EngramP& p [[buffer(21)]],
                           const device uint* mask [[buffer(22)]], uint r [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  if (p.masked && mask[r] == 0) return;  // the whole threadgroup is row r: no barrier is left waiting
  threadgroup float red[32][13];
  threadgroup float tot[13];
  const device void* kb[4] = {kb0, kb1, kb2, kb3};
  const device void* nk[4] = {nk0, nk1, nk2, nk3};
  const device void* nq[4] = {nq0, nq1, nq2, nq3};
  const device void* hs[4] = {hs0, hs1, hs2, hs3};
  const uint D = p.D;
  device float* xr = X + size_t(r) * 4 * D;
  // acc: 0 ss(v), 1..4 ss(k_s), 5..8 ss(q_s), 9..12 sum(nk_s * nq_s * k_s * q_s)
  float acc[13];
  NI_UNROLL for (uint j = 0; j < 13; ++j) acc[j] = 0.0f;
  for (uint c = tid; c < D; c += ntg) {
    const float v = vraw[size_t(r) * D + c] + hy4_ld(vb, c, p.t_vb);
    acc[0] += v * v;
    NI_UNROLL for (uint s = 0; s < 4; ++s) {
      const float k = kraw[(size_t(s) * p.R + r) * D + c] + hy4_ld(kb[s], c, p.t_kb);
      const float q = xr[s * D + c];
      acc[1 + s] += k * k;
      acc[5 + s] += q * q;
      acc[9 + s] += hy4_ld(nk[s], c, p.t_nk) * hy4_ld(nq[s], c, p.t_nq) * k * q;
    }
  }
  NI_UNROLL for (uint j = 0; j < 13; ++j) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) red[sg][j] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 13) {
    const uint nsg = (ntg + 31) / 32;
    float v = 0.0f;
    for (uint i = 0; i < nsg; ++i) v += red[i][tid];
    tot[tid] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float fD = float(D);
  const float iv = metal::precise::rsqrt(tot[0] / fD + p.eps);
  float gs[4];
  NI_UNROLL for (uint s = 0; s < 4; ++s) {
    const float ik = metal::precise::rsqrt(tot[1 + s] / fD + p.eps);
    const float iq = metal::precise::rsqrt(tot[5 + s] / fD + p.eps);
    float g = tot[9 + s] * ik * iq * p.inv_sqrt_d;
    g = sign(g) * metal::precise::sqrt(max(fabs(g), 1e-6f));
    gs[s] = hy4_sigmoid(g);
  }
  for (uint c = tid; c < D; c += ntg) {
    const float v = hy4_ld(pg, c, p.t_pg) * ((vraw[size_t(r) * D + c] + hy4_ld(vb, c, p.t_vb)) * iv);
    NI_UNROLL for (uint s = 0; s < 4; ++s) xr[s * D + c] += gs[s] * v + hy4_ld(hs[s], c, p.t_hs) * v;
  }
}

// ---- attention ----
// K/V cache: [H][cap] rows of KVQ_ROW bytes (kv_int8.h), position pos in row (ring ? pos % cap : pos).
// Prep: q rope -> qo fp32 [T, H, 256]; K = [k_nope | rope(k_rope)], V -> cache. grid: (H, T) x 256.
struct Hy4AttnPrepP {
  uint T, H, pos0, cap, ring, kr_ld, kr_off, rope_bf16;
  float theta;
};
NI_INLINE float hy4_bf16_rne(float x) {
  uint b = as_type<uint>(x);
  b += 0x7fffu + ((b >> 16) & 1u);
  return as_type<float>(b & 0xffff0000u);
}
[[kernel]] void hy4_attn_prep(const device float* q [[buffer(0)]], const device float* kvb [[buffer(1)]],
                              const device float* kr [[buffer(2)]], device float* qo [[buffer(3)]],
                              device half* kc [[buffer(4)]], device half* vc [[buffer(5)]],
                              constant Hy4AttnPrepP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sk[32], sv[32];
  const uint h = tg.x, t = tg.y;
  const uint pos = p.pos0 + t;
  const uint row = p.ring ? pos % p.cap : pos;
  float cs = 1.0f, sn = 0.0f;
  const bool rot = tid >= 192;
  if (rot) {
    const uint i = (tid - 192) / 2;
    const float inv_freq = 1.0f / metal::precise::pow(p.theta, float(2 * i) / 64.0f);
    const float ang = float(pos) * inv_freq;
    cs = metal::precise::cos(ang);
    sn = metal::precise::sin(ang);
    if (p.rope_bf16) {
      cs = hy4_bf16_rne(cs);
      sn = hy4_bf16_rne(sn);
    }
  }
  const bool odd = tid & 1;
  // q: rotate the pair (2i, 2i+1) of the rope dims: out[2i] = x0 c - x1 s, out[2i+1] = x1 c + x0 s
  const size_t qi = (size_t(t) * p.H + h) * 256 + tid;
  float x = q[qi];
  float xp = simd_shuffle_xor(x, 1);
  if (rot) x = odd ? x * cs + xp * sn : x * cs - xp * sn;
  qo[qi] = x;
  // k
  const size_t kvi = (size_t(t) * p.H + h) * 448;
  float k = rot ? kr[size_t(t) * p.kr_ld + p.kr_off + tid - 192] : kvb[kvi + tid];
  float kp = simd_shuffle_xor(k, 1);
  if (rot) k = odd ? k * cs + kp * sn : k * cs - kp * sn;
  device half* kh = (device half*)((device char*)kc + size_t(h) * p.cap * KVQ_ROW);
  device half* vh = (device half*)((device char*)vc + size_t(h) * p.cap * KVQ_ROW);
  kvq_store1(kh, row * KVQ_ROW, sk, tid, k);
  kvq_store1(vh, row * KVQ_ROW, sv, tid, kvb[kvi + 192 + tid]);
}

// Decode, split-K: simdgroup = one chunk of one head's keys [lo + c*chunk, ...) up to L (positions); 4 chunks a
// threadgroup of 128. Scores in the log2 domain. grid: (ceil(nchunks/4), H). part_o [H][nchunks][256],
// part_ml [H][nchunks][2] (running max, sum).
struct Hy4AttnDecP {
  uint H, L, lo, chunk, nchunks, cap, ring, sink_t, out_half;
  float scale;
};
[[kernel]] void hy4_attn_dec(const device float* qo [[buffer(0)]], const device half* kc [[buffer(1)]],
                             const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                             device float* part_ml [[buffer(4)]], constant Hy4AttnDecP& p [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x * 4 + sg, h = tg.y;
  if (c >= p.nchunks) return;
  const uint start = p.lo + c * p.chunk;
  const uint end = min(start + p.chunk, p.L);
  float qv[8], acc[8];
  {
    const float sc = p.scale * M_LOG2E_F;
    const float4 a = *(const device float4*)(qo + h * 256 + lane * 8);
    const float4 b = *(const device float4*)(qo + h * 256 + lane * 8 + 4);
    qv[0] = a.x * sc; qv[1] = a.y * sc; qv[2] = a.z * sc; qv[3] = a.w * sc;
    qv[4] = b.x * sc; qv[5] = b.y * sc; qv[6] = b.z * sc; qv[7] = b.w * sc;
  }
  NI_UNROLL for (uint i = 0; i < 8; ++i) acc[i] = 0.0f;
  const device half* kh = (const device half*)((const device char*)kc + size_t(h) * p.cap * KVQ_ROW);
  const device half* vh = (const device half*)((const device char*)vc + size_t(h) * p.cap * KVQ_ROW);
  constexpr float kNeg = -1.0e30f;
  float m = kNeg, l = 0.0f;
  for (uint j = start; j < end; j += 4) {
    float s[4];
    uint rows[4];
    NI_UNROLL for (uint u = 0; u < 4; ++u) {
      const uint pos = min(j + u, end - 1);
      rows[u] = p.ring ? pos % p.cap : pos;
      const uint4 kw = kvq_load8(kh, rows[u] * KVQ_ROW, lane);
      float d = 0.0f;
      NI_UNROLL for (uint w = 0; w < 4; ++w) {
        const float2 k2 = float2(as_type<half2>(kw[w]));
        d += qv[2 * w] * k2.x + qv[2 * w + 1] * k2.y;
      }
      s[u] = d;
    }
    NI_UNROLL for (uint u = 0; u < 4; ++u) s[u] = j + u < end ? simd_sum(s[u]) : kNeg;
    const float mx = max(max(s[0], s[1]), max(s[2], s[3]));
    const float mn = max(m, mx);
    const float corr = metal::fast::exp2(m - mn);
    float e[4];
    NI_UNROLL for (uint u = 0; u < 4; ++u) e[u] = metal::fast::exp2(s[u] - mn);
    l = l * corr + (e[0] + e[1]) + (e[2] + e[3]);
    NI_UNROLL for (uint i = 0; i < 8; ++i) acc[i] *= corr;
    NI_UNROLL for (uint u = 0; u < 4; ++u) {
      const uint4 vw = kvq_load8(vh, rows[u] * KVQ_ROW, lane);
      NI_UNROLL for (uint w = 0; w < 4; ++w) {
        const float2 v2 = float2(as_type<half2>(vw[w]));
        acc[2 * w] += e[u] * v2.x;
        acc[2 * w + 1] += e[u] * v2.y;
      }
    }
    m = mn;
  }
  const size_t base = size_t(h) * p.nchunks + c;
  device float4* po = (device float4*)(part_o + base * 256 + lane * 8);
  po[0] = float4(acc[0], acc[1], acc[2], acc[3]);
  po[1] = float4(acc[4], acc[5], acc[6], acc[7]);
  if (lane == 0) {
    part_ml[base * 2] = m;
    part_ml[base * 2 + 1] = l;
  }
}

// out[h*256 + d] = (sum_c 2^(m_c - M) o_c) / (2^(sink - M) + sum_c 2^(m_c - M) l_c) * sigmoid(gate). grid: H x 256.
[[kernel]] void hy4_attn_dec_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                    const device float* gate [[buffer(2)]], const device void* sinks [[buffer(3)]],
                                    device void* out [[buffer(4)]], constant Hy4AttnDecP& p [[buffer(5)]],
                                    uint h [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const float sk = hy4_ld(sinks, h, p.sink_t) * M_LOG2E_F;
  float M = sk;
  for (uint c = 0; c < p.nchunks; ++c) M = max(M, part_ml[(h * p.nchunks + c) * 2]);
  float num = 0.0f, den = metal::fast::exp2(sk - M);
  for (uint c = 0; c < p.nchunks; ++c) {
    const uint b = h * p.nchunks + c;
    const float w = metal::fast::exp2(part_ml[b * 2] - M);
    num += w * part_o[size_t(b) * 256 + d];
    den += w * part_ml[b * 2 + 1];
  }
  const float y = num / den * sigmoid(gate[h * 256 + d]);
  if (p.out_half) ((device half*)out)[h * 256 + d] = half(y);
  else ((device float*)out)[h * 256 + d] = y;
}

// Prefill: attn.metal's v11 kernel with the HY flag (sinks, window, ring, gate [T, H, 256], half out).
// grid: (ceil(T/32), H) x 256. kc/vc: [H][cap] rows; p.Hq = p.Hkv = H, p.max_seq = cap.
// hy4_attn_prefill: 32-key tiles, 20,480 B of static threadgroup memory (the Metal validation layer counts it twice
// and aborts: run hy4_attn_prefill_k16 under the layer); hy4_attn_prefill_k16: 16-key tiles, 16,384 B.
[[kernel, max_total_threads_per_threadgroup(256)]] void hy4_attn_prefill(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* gate [[buffer(3)]], device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
    const device void* sinks [[buffer(6)]], constant AttnHyParams& hy [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  threadgroup float2 xs[1];
  attn_prefill11_impl<32, false, false, false, true, false, false, true>(
      q, kc, vc, gate, out, p, tg, tid, sg, lane, (threadgroup half*)kvm, xs, 0, 1, nullptr, nullptr, nullptr, 0,
      nullptr, 0, sinks, &hy);
}
[[kernel, max_total_threads_per_threadgroup(256)]] void hy4_attn_prefill_k16(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* gate [[buffer(3)]], device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
    const device void* sinks [[buffer(6)]], constant AttnHyParams& hy [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<16>() / 8];
  threadgroup float2 xs[attn11_xs<16>()];
  attn_prefill11_impl<16, true, false, false, true, false, false, true>(
      q, kc, vc, gate, out, p, tg, tid, sg, lane, (threadgroup half*)kvm, xs, 0, 1, nullptr, nullptr, nullptr, 0,
      nullptr, 0, sinks, &hy);
}

// ---- latent MLA (hy4_ops.h's LatentCache / enc_attn_latent) ----
// Latent rows: half [cap][576] = kv_a_norm latent 512 | roped k_rope 64, every value rounded to bf16 first.
// Position pos sits in row (ring ? pos % cap : pos); cap is a multiple of 16.
constant constexpr uint kMlaW = 576;   // latent row width
constant constexpr uint kMlaL = 512;   // kv_lora_rank
constant constexpr uint kIxSel = 512;  // the compressed indexer's groups per query (group_topk)

NI_INLINE void hy4_rope_cs(uint pos, uint i, float theta, uint rbf, thread float& cs, thread float& sn) {
  const float inv_freq = 1.0f / metal::precise::pow(theta, float(2 * i) / 64.0f);
  const float ang = float(pos) * inv_freq;
  cs = metal::precise::cos(ang);
  sn = metal::precise::sin(ang);
  if (rbf) {
    cs = hy4_bf16_rne(cs);
    sn = hy4_bf16_rne(sn);
  }
}

// T rows into the cache. grid: T x 64.
struct Hy4MlaWriteP {
  uint pos0, cap, ring, lat_ld, kr_ld, kr_off, rope_bf16;
  float theta;
};
[[kernel]] void hy4_mla_write(const device float* lat [[buffer(0)]], const device float* kr [[buffer(1)]],
                              device half* rows [[buffer(2)]], constant Hy4MlaWriteP& p [[buffer(3)]],
                              uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint pos = p.pos0 + t;
  device half* row = rows + size_t(p.ring ? pos % p.cap : pos) * kMlaW;
  const device float* lr = lat + size_t(t) * p.lat_ld;
  NI_UNROLL for (uint u = 0; u < 8; ++u) row[tid * 8 + u] = half(hy4_bf16_rne(lr[tid * 8 + u]));
  float cs, sn;
  hy4_rope_cs(pos, tid / 2, p.theta, p.rope_bf16, cs, sn);
  const float x = kr[size_t(t) * p.kr_ld + p.kr_off + tid];
  const float xp = simd_shuffle_xor(x, 1);
  const float y = (tid & 1) ? x * cs + xp * sn : x * cs - xp * sn;
  row[kMlaL + tid] = half(hy4_bf16_rne(y));
}

// Queries. mode 0 (absorbed): qh half [T*H][576] gets the roped rope dims at [512, 576), times qscale (the nope part
// comes from hy4_mla_gemm mode 0). mode 1 (expanded): qo fp32 [T][H][256] = [q_nope | rope(q_rope)].
// grid: (H, T) x 256.
struct Hy4MlaQP {
  uint T, H, pos0, q_ld, mode, rope_bf16;
  float theta, qscale;
};
[[kernel]] void hy4_mla_qprep(const device float* q [[buffer(0)]], device void* dst [[buffer(1)]],
                              constant Hy4MlaQP& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, t = tg.y;
  const bool rot = tid >= 192;
  float cs = 1.0f, sn = 0.0f;
  if (rot) hy4_rope_cs(p.pos0 + t, (tid - 192) / 2, p.theta, p.rope_bf16, cs, sn);
  float x = q[size_t(t) * p.q_ld + h * 256 + tid];
  const float xp = simd_shuffle_xor(x, 1);
  if (rot) x = (tid & 1) ? x * cs + xp * sn : x * cs - xp * sn;
  if (p.mode == 0) {
    if (rot) ((device half*)dst)[(size_t(t) * p.H + h) * kMlaW + kMlaL + tid - 192] = half(x * p.qscale);
  } else {
    ((device float*)dst)[(size_t(t) * p.H + h) * 256 + tid] = x;
  }
}

// Per-head products with kv_b (bf16 [H*448, 512]: per head 192 K-nope rows, then 256 V rows), fp32 MMA, for the
// absorbed prefill (NI_HY4_MLA_PREFILL=absorbed, a measuring switch; T <= 16 runs hy4_mla_uk / hy4_mla_uv).
// mode 0 (absorb): qh[(m*H + h)*576 + n] = half(qscale * sum_k A[m, k] W[h*448 + k, n]), A = q nope (k < 192, n < 512).
// mode 1 (V up): out[m*out_ld + h*256 + n] = sigmoid(gate[m*gate_ld + h*256 + n]) * sum_k A[m, k] W[h*448 + 192 + n, k],
//   A = latent outputs (k < 512, n < 256), out fp32 or half.
// A[m, k] at a[m*a_ld + h*a_hs + k]. grid: (N/32, ceil(M/32), H) x 128. N, K multiples of 32.
struct Hy4MlaGemmP {
  uint M, N, K, H, a_ld, a_hs, mode, out_half, gate_ld, out_ld;
  float qscale;
};
[[kernel, max_total_threads_per_threadgroup(128)]] void hy4_mla_gemm(
    const device float* A [[buffer(0)]], const device bfloat* W [[buffer(1)]], device void* out [[buffer(2)]],
    const device float* gate [[buffer(4)]], constant Hy4MlaGemmP& p [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BM = 32, BN = 32, BK = 32, LD = 36;
  threadgroup float As[BM * LD], Bs[BK * LD];
  const uint n0 = tg.x * BN, m0 = tg.y * BM, h = tg.z;
  const uint roff = p.mode == 1 ? 192 : 0;
  simdgroup_float8x8 acc[4];
  NI_UNROLL for (uint j = 0; j < 4; ++j) acc[j] = simdgroup_float8x8(0);
  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    NI_UNROLL for (uint u = 0; u < 8; ++u) {
      const uint idx = tid + 128 * u, r = idx / 32, c = idx % 32, m = m0 + r;
      As[r * LD + c] = m < p.M ? A[size_t(m) * p.a_ld + h * p.a_hs + k0 + c] : 0.0f;
    }
    NI_UNROLL for (uint u = 0; u < 8; ++u) {
      const uint idx = tid + 128 * u;
      if (p.mode == 0) {
        const uint kk = idx / 32, nn = idx % 32;
        Bs[kk * LD + nn] = float(W[size_t(h * 448 + k0 + kk) * kMlaL + n0 + nn]);
      } else {
        const uint nn = idx / 32, kk = idx % 32;
        Bs[kk * LD + nn] = float(W[size_t(h * 448 + roff + n0 + nn) * kMlaL + k0 + kk]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (uint kk = 0; kk < BK / 8; ++kk) {
      simdgroup_float8x8 a;
      simdgroup_load(a, As + sg * 8 * LD + kk * 8, LD);
      NI_UNROLL for (uint j = 0; j < 4; ++j) {
        simdgroup_float8x8 b;
        simdgroup_load(b, Bs + kk * 8 * LD + j * 8, LD);
        simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4, fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const uint m = m0 + sg * 8 + frow;
  if (m >= p.M) return;
  NI_UNROLL for (uint j = 0; j < 4; ++j)
    NI_UNROLL for (uint i = 0; i < 2; ++i) {
      const uint n = n0 + j * 8 + fcol + i;
      const float v = acc[j].thread_elements()[i];
      if (p.mode == 0) {
        ((device half*)out)[(size_t(m) * p.H + h) * kMlaW + n] = half(v * p.qscale);
      } else {
        const float y = v * sigmoid(gate[size_t(m) * p.gate_ld + h * 256 + n]);
        const size_t o = size_t(m) * p.out_ld + h * 256 + n;
        if (p.out_half) ((device half*)out)[o] = half(y);
        else ((device float*)out)[o] = y;
      }
    }
}

// hy4_mla_expand's rope dims: K[h][m][192 + j] = the latent row's k_rope j. grid: (ceil(M*64/256), H) x 256.
struct Hy4MlaKrP {
  uint M, cap, ring, kbase;
};
[[kernel]] void hy4_mla_kr(const device half* rows [[buffer(0)]], device half* K [[buffer(1)]],
                           constant Hy4MlaKrP& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint m = gid.x / 64, j = gid.x % 64, h = gid.y;
  if (m >= p.M) return;
  const uint pos = p.kbase + m;
  K[(size_t(h) * p.M + m) * 256 + 192 + j] = rows[size_t(p.ring ? pos % p.cap : pos) * kMlaW + kMlaL + j];
}

// Absorbed attention over the latent rows. Rows r = t*H + h of qh (half [R][576], scaled by log2e/16); a threadgroup
// takes 16 rows (one token t: H % 16 == 0) x the key tiles of its split (grid.y), 8 simdgroups = 2 row groups of 8
// rows x 4 slices of the 576 score dims (144 each) and of the 512 output dims (128 each). Scores are computed
// transposed (S^T = K Q^T: K rows load as contiguous pairs), the 4 slices' partials meet in threadgroup memory, read
// back in P layout; O += P V with V = the latent rows' first 512 dims. Tiles of 16 keys, 16-aligned.
// nsplit 1: olat [R][512] = O / l with the sink in the softmax. nsplit > 1: part_o [nsplit][R][512], part_ml (m, l)
// without the sink, for hy4_mla_reduce. grid: (R/16, nsplit) x 256.
struct Hy4MlaAttnP {
  uint T, H, pos0, cap, ring, win, nsplit, per, sink_t, sel_one;
};
// SEL (hy4_mla_attn_sel, an indexed layer past 2048 keys; linear cache, no window). The keys come in entries of 4:
// tile it takes entries 4 it .. 4 it + 3 (key j: entry 4 it + j / 4, slot j % 4); a masked key reads row 0.
// sel_one 0 (the target's rows): token t's entries are the n = min(G, 511) + 1 groups of sel[t] (uint [T][512],
//   ascending, G = (pos0 + t) / 4); the causal mask cuts the last (own) group.
// sel_one 1 (the drafter's block): every token shares the selection sel[0] of the row at pos0 - 1 (n = min((pos0 - 1)
//   / 4, 511) + 1 groups), cut to positions < pos0, then ceil(T / 4) entries of the block's own positions pos0..,
//   causal.
NI_INLINE uint hy4_mla_selpos(const device uint* srow, uint n, uint cut, bool one, uint pos0, uint it, uint key) {
  const uint e = it * 4 + key / 4;
  if (e >= n) return one ? pos0 + (e - n) * 4 + key % 4 : 0xffffffffu;  // the block (past kend: masked) or padding
  const uint ps = srow[e] * 4 + key % 4;
  return ps < cut ? ps : 0xffffffffu;
}
template <bool SEL>
[[kernel, max_total_threads_per_threadgroup(256)]] void hy4_mla_attn_impl(
    const device half* qh [[buffer(0)]], const device half* rows [[buffer(1)]], const device void* sinks [[buffer(2)]],
    device float* olat [[buffer(3)]], device float* part_o [[buffer(4)]], device float2* part_ml [[buffer(5)]],
    constant Hy4MlaAttnP& p [[buffer(6)]], const device uint* sel [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[2][2][4][16][8];  // [buffer][row group][slice][key][row]: 8 KB
  constexpr float kNeg = -1.0e30f;
  const uint rg = sg / 4, dg = sg % 4;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4, fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const uint R = p.T * p.H;
  const uint nrb = R / 16;
  const uint r0 = (nrb - 1 - tg.x) * 16;  // latest tokens (most keys) first
  const uint t = r0 / p.H;
  const uint qr0 = r0 + rg * 8;
  const uint kend = p.pos0 + t + 1;
  const uint lo = p.win && kend > p.win ? kend - p.win : 0;
  const bool one = SEL && p.sel_one;
  const uint nsel = !SEL ? 0 : one ? min((p.pos0 - 1) / 4, kIxSel - 1) + 1 : min((kend - 1) / 4, kIxSel - 1) + 1;
  const uint scut = one ? p.pos0 : 0xffffffffu;
  const device uint* srow = sel + (one ? 0 : size_t(t) * kIxSel);
  const uint it1 = SEL ? (nsel + (one ? (p.T + 3) / 4 : 0) + 3) / 4 : (kend + 15) / 16;
  const uint ib = min((SEL ? 0 : lo / 16) + tg.y * p.per, it1), ie = min(ib + p.per, it1);

  // Wide loads: a lane takes 8 (4) consecutive dims of a K/V row with one 16 (8) byte load and spreads them over 4
  // (2) fragments, element pair j to fragment j: fragment j's column fc + i (fc = 0, 2, 4, 6) stands for dim
  // (fc / 2) * 8 + 2j + i of the group's 32 (16: (fc / 2) * 4 + 2j + i). Q^T's rows and O's columns follow suit.
  // Score dims of slice dg: [144 dg, 144 dg + 144) = 4 groups of 32, then one of 16; output dims: [128 dg, +128).
  simdgroup_half8x8 qt[18];
  {
    const uint b = dg * 144;
    NI_UNROLL for (uint g = 0; g < 4; ++g)
      NI_UNROLL for (uint j = 0; j < 4; ++j)
        NI_UNROLL for (uint i = 0; i < 2; ++i)
          qt[g * 4 + j].thread_elements()[i] =
              qh[size_t(qr0 + fcol + i) * kMlaW + b + g * 32 + (frow / 2) * 8 + 2 * j + frow % 2];
    NI_UNROLL for (uint j = 0; j < 2; ++j)
      NI_UNROLL for (uint i = 0; i < 2; ++i)
        qt[16 + j].thread_elements()[i] = qh[size_t(qr0 + fcol + i) * kMlaW + b + 128 + (frow / 2) * 4 + 2 * j + frow % 2];
  }
  simdgroup_float8x8 o[16];
  NI_UNROLL for (uint j = 0; j < 16; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = kNeg, l_row = 0.0f;
  if (p.nsplit == 1) {
    m_row = hy4_ld(sinks, (qr0 + frow) % p.H, p.sink_t) * M_LOG2E_F;
    l_row = 1.0f;
  }
  for (uint it = ib; it < ie; ++it) {
    const uint k0 = it * 16, buf = it & 1;
    const device half* kr = rows + size_t(SEL ? 0 : p.ring ? k0 % p.cap : k0) * kMlaW;
    uint prow[2] = {0, 0};  // SEL: the rows of keys kf * 8 + frow (masked keys read row 0)
    if (SEL)
      NI_UNROLL for (uint kf = 0; kf < 2; ++kf) {
        const uint ps = hy4_mla_selpos(srow, nsel, scut, one, p.pos0, it, kf * 8 + frow);
        prow[kf] = ps < kend ? ps : 0;
      }
    simdgroup_float8x8 s[2] = {simdgroup_float8x8(0), simdgroup_float8x8(0)};
    NI_UNROLL for (uint kf = 0; kf < 2; ++kf) {
      const device half* krow = kr + (SEL ? prow[kf] : kf * 8 + frow) * kMlaW + dg * 144;
      NI_UNROLL for (uint g = 0; g < 4; ++g) {
        const uint4 w = *(const device uint4*)(krow + g * 32 + (fcol / 2) * 8);
        NI_UNROLL for (uint j = 0; j < 4; ++j) {
          simdgroup_half8x8 a;
          const half2 k2 = as_type<half2>(w[j]);
          a.thread_elements()[0] = k2.x;
          a.thread_elements()[1] = k2.y;
          simdgroup_multiply_accumulate(s[kf], a, qt[g * 4 + j], s[kf]);
        }
      }
      const uint2 w = *(const device uint2*)(krow + 128 + (fcol / 2) * 4);
      NI_UNROLL for (uint j = 0; j < 2; ++j) {
        simdgroup_half8x8 a;
        const half2 k2 = as_type<half2>(w[j]);
        a.thread_elements()[0] = k2.x;
        a.thread_elements()[1] = k2.y;
        simdgroup_multiply_accumulate(s[kf], a, qt[16 + j], s[kf]);
      }
    }
    NI_UNROLL for (uint kf = 0; kf < 2; ++kf)
      NI_UNROLL for (uint i = 0; i < 2; ++i) xs[buf][rg][dg][kf * 8 + frow][fcol + i] = s[kf].thread_elements()[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 pf[2];
    float mx = kNeg;
    NI_UNROLL for (uint c = 0; c < 2; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        const uint key = c * 8 + fcol + i, k = SEL ? hy4_mla_selpos(srow, nsel, scut, one, p.pos0, it, key) : k0 + key;
        float v = xs[buf][rg][0][key][frow] + xs[buf][rg][1][key][frow] + xs[buf][rg][2][key][frow] +
                  xs[buf][rg][3][key][frow];
        if (k < lo || k >= kend) v = kNeg;
        pf[c].thread_elements()[i] = v;
        mx = max(mx, v);
      }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = metal::fast::exp2(m_row - m_new);
    float ps = 0.0f;
    NI_UNROLL for (uint c = 0; c < 2; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        const float e = metal::fast::exp2(pf[c].thread_elements()[i] - m_new);
        pf[c].thread_elements()[i] = e;
        ps += e;
      }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    NI_UNROLL for (uint j = 0; j < 16; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    simdgroup_half8x8 ph[2];
    NI_UNROLL for (uint c = 0; c < 2; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) ph[c].thread_elements()[i] = half(pf[c].thread_elements()[i]);
    NI_UNROLL for (uint c = 0; c < 2; ++c)
      NI_UNROLL for (uint g = 0; g < 4; ++g) {
        const uint4 w =
            *(const device uint4*)(kr + (SEL ? prow[c] : c * 8 + frow) * kMlaW + dg * 128 + g * 32 + (fcol / 2) * 8);
        NI_UNROLL for (uint j = 0; j < 4; ++j) {
          simdgroup_half8x8 b;
          const half2 v2 = as_type<half2>(w[j]);
          b.thread_elements()[0] = v2.x;
          b.thread_elements()[1] = v2.y;
          simdgroup_multiply_accumulate(o[g * 4 + j], ph[c], b, o[g * 4 + j]);
        }
      }
  }
  const uint row = qr0 + frow;
  const float inv = p.nsplit == 1 ? 1.0f / l_row : 1.0f;
  device float* dst = (p.nsplit == 1 ? olat + size_t(row) * kMlaL : part_o + (size_t(tg.y) * R + row) * kMlaL) +
                      dg * 128 + (fcol / 2) * 8;
  NI_UNROLL for (uint g = 0; g < 4; ++g) {
    const float4 a = float4(o[g * 4].thread_elements()[0], o[g * 4].thread_elements()[1],
                            o[g * 4 + 1].thread_elements()[0], o[g * 4 + 1].thread_elements()[1]);
    const float4 b = float4(o[g * 4 + 2].thread_elements()[0], o[g * 4 + 2].thread_elements()[1],
                            o[g * 4 + 3].thread_elements()[0], o[g * 4 + 3].thread_elements()[1]);
    *(device float4*)(dst + g * 32) = a * inv;
    *(device float4*)(dst + g * 32 + 4) = b * inv;
  }
  if (p.nsplit > 1) {
    const size_t pr = size_t(tg.y) * R + row;
    if (dg == 0 && fcol == 0) part_ml[pr] = float2(m_row, l_row);
  }
}
template [[host_name("hy4_mla_attn")]] [[kernel]] void hy4_mla_attn_impl<false>(
    const device half*, const device half*, const device void*, device float*, device float*, device float2*,
    constant Hy4MlaAttnP&, const device uint*, uint2, uint, uint);
template [[host_name("hy4_mla_attn_sel")]] [[kernel]] void hy4_mla_attn_impl<true>(
    const device half*, const device half*, const device void*, device float*, device float*, device float2*,
    constant Hy4MlaAttnP&, const device uint*, uint2, uint, uint);

// olat[r] = (sum_s 2^(m_s - M) o_s) / (2^(sink - M) + sum_s 2^(m_s - M) l_s). grid: R x 128 (4 dims a thread).
[[kernel]] void hy4_mla_reduce(const device float* part_o [[buffer(0)]], const device float2* part_ml [[buffer(1)]],
                               const device void* sinks [[buffer(2)]], device float* olat [[buffer(3)]],
                               constant Hy4MlaAttnP& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  const uint R = p.T * p.H;
  const float sk = hy4_ld(sinks, r % p.H, p.sink_t) * M_LOG2E_F;
  float M = sk;
  for (uint s = 0; s < p.nsplit; ++s) M = max(M, part_ml[size_t(s) * R + r].x);
  float den = metal::fast::exp2(sk - M);
  float4 num = 0.0f;
  for (uint s = 0; s < p.nsplit; ++s) {
    const float2 ml = part_ml[size_t(s) * R + r];
    const float w = metal::fast::exp2(ml.x - M);
    den += w * ml.y;
    num += w * *(const device float4*)(part_o + (size_t(s) * R + r) * kMlaL + tid * 4);
  }
  *(device float4*)(olat + size_t(r) * kMlaL + tid * 4) = num / den;
}

// NI_KV_I8 only: half rows [H*M][256] (hy4_mla_expand's K or V) into the cache row layout. grid: H*M x 256.
[[kernel]] void hy4_mla_pack(const device half* src [[buffer(0)]], device half* dst [[buffer(1)]],
                             uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float s[32];
  // the row's byte offset in 64 bits (r * KVQ_ROW wraps a uint once H * M passes about 15.8M rows)
  device half* row = (device half*)((device char*)dst + size_t(r) * KVQ_ROW);
  kvq_store1(row, 0, s, tid, float(src[size_t(r) * 256 + tid]));
}

// Small-T products with kv_b (T <= 16: two 8-row fragments of tokens). Weight tiles are staged through threadgroup
// memory with wide coalesced loads (the rows' 64-byte pieces a fragment load would take run at half the bandwidth),
// the next tile's loads issued before the current tile's products.
struct Hy4MlaSmallP {
  uint T, H, q_ld, out_half, gate_ld, out_ld;
  float qscale;
};
// hy4_mla_uk: qh[(t*H + h)*576 + n] = half(qscale * sum_{k<192} q[t*q_ld + h*256 + k] W[h*448 + k, n]). A threadgroup
// takes 128 columns n (a simdgroup 32), in tiles of 32 k x 128 n (8 KB). grid: (4, H) x 128.
[[kernel, max_total_threads_per_threadgroup(128)]] void hy4_mla_uk(
    const device float* q [[buffer(0)]], const device bfloat* W [[buffer(1)]], device half* qh [[buffer(2)]],
    constant Hy4MlaSmallP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LD = 136;  // bf16 a tile row (128 + 8 pad)
  threadgroup bfloat ws[32 * LD];
  const uint h = tg.y, nb = tg.x * 128, n0 = sg * 32;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4, fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const uint nt = p.T > 8 ? 2 : 1;
  simdgroup_float8x8 acc[2][4];
  NI_UNROLL for (uint a = 0; a < 2; ++a) NI_UNROLL for (uint j = 0; j < 4; ++j) acc[a][j] = simdgroup_float8x8(0);
  // thread: rows tid / 16 + 8u (u < 4), 8 columns at (tid % 16) * 8
  const device bfloat* wsrc = W + size_t(h * 448 + tid / 16) * kMlaL + nb + (tid % 16) * 8;
  uint4 r[4];
  NI_UNROLL for (uint u = 0; u < 4; ++u) r[u] = *(const device uint4*)(wsrc + size_t(u * 8) * kMlaL);
  const device float* qb = q + h * 256 + fcol;
  for (uint k0 = 0; k0 < 192; k0 += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (uint u = 0; u < 4; ++u) *(threadgroup uint4*)(ws + (tid / 16 + u * 8) * LD + (tid % 16) * 8) = r[u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 32 < 192)
      NI_UNROLL for (uint u = 0; u < 4; ++u) r[u] = *(const device uint4*)(wsrc + size_t(k0 + 32 + u * 8) * kMlaL);
    NI_UNROLL for (uint kk = 0; kk < 32; kk += 8) {
      simdgroup_float8x8 b[4];
      NI_UNROLL for (uint j = 0; j < 4; ++j) {
        const float2 w2 = float2(*(const threadgroup bfloat2*)(ws + (kk + frow) * LD + n0 + j * 8 + fcol));
        b[j].thread_elements()[0] = w2.x;
        b[j].thread_elements()[1] = w2.y;
      }
      for (uint a = 0; a < nt; ++a) {
        const uint t = a * 8 + frow;
        simdgroup_float8x8 af;
        const float2 q2 = t < p.T ? *(const device float2*)(qb + size_t(t) * p.q_ld + k0 + kk) : float2(0);
        af.thread_elements()[0] = q2.x;
        af.thread_elements()[1] = q2.y;
        NI_UNROLL for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[a][j], af, b[j], acc[a][j]);
      }
    }
  }
  for (uint a = 0; a < nt; ++a) {
    const uint t = a * 8 + frow;
    if (t >= p.T) continue;
    device half* dst = qh + (size_t(t) * p.H + h) * kMlaW + nb + n0 + fcol;
    NI_UNROLL for (uint j = 0; j < 4; ++j)
      *(device half2*)(dst + j * 8) =
          half2(float2(acc[a][j].thread_elements()[0], acc[a][j].thread_elements()[1]) * p.qscale);
  }
}

// hy4_mla_uv: out[t*out_ld + h*256 + n] = sigmoid(gate[t*gate_ld + h*256 + n]) * sum_{k<512} olat[(t*H + h)*512 + k]
// W[h*448 + 192 + n, k], computed transposed (rows n, columns t). A threadgroup takes 16 rows n, in tiles of 16 n x
// 128 k (4 KB); simdgroup sg takes the tile's k [32 sg, 32 sg + 32); the 4 partial sums meet in threadgroup memory.
// grid: (16, H) x 128.
[[kernel, max_total_threads_per_threadgroup(128)]] void hy4_mla_uv(
    const device float* olat [[buffer(0)]], const device bfloat* W [[buffer(1)]], device void* out [[buffer(2)]],
    const device float* gate [[buffer(3)]], constant Hy4MlaSmallP& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LD = 136;
  threadgroup bfloat ws[16 * LD];
  threadgroup float red[4][2][2][64];
  const uint h = tg.y, n0 = tg.x * 16;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4, fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const uint nt = p.T > 8 ? 2 : 1;
  simdgroup_float8x8 acc[2][2];
  NI_UNROLL for (uint a = 0; a < 2; ++a) NI_UNROLL for (uint j = 0; j < 2; ++j) acc[a][j] = simdgroup_float8x8(0);
  // thread: rows tid / 16 + 8u (u < 2), 8 k at (tid % 16) * 8
  const device bfloat* wsrc = W + size_t(h * 448 + 192 + n0 + tid / 16) * kMlaL + (tid % 16) * 8;
  uint4 r[2];
  NI_UNROLL for (uint u = 0; u < 2; ++u) r[u] = *(const device uint4*)(wsrc + size_t(u * 8) * kMlaL);
  for (uint k0 = 0; k0 < kMlaL; k0 += 128) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (uint u = 0; u < 2; ++u) *(threadgroup uint4*)(ws + (tid / 16 + u * 8) * LD + (tid % 16) * 8) = r[u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 128 < kMlaL)
      NI_UNROLL for (uint u = 0; u < 2; ++u) r[u] = *(const device uint4*)(wsrc + size_t(u * 8) * kMlaL + k0 + 128);
    NI_UNROLL for (uint kk = 0; kk < 32; kk += 8) {
      const uint kt = sg * 32 + kk;  // within the tile
      simdgroup_float8x8 a[2];
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        const float2 w2 = float2(*(const threadgroup bfloat2*)(ws + (i * 8 + frow) * LD + kt + fcol));
        a[i].thread_elements()[0] = w2.x;
        a[i].thread_elements()[1] = w2.y;
      }
      for (uint j = 0; j < nt; ++j) {
        simdgroup_float8x8 b;
        NI_UNROLL for (uint e = 0; e < 2; ++e) {
          const uint t = j * 8 + fcol + e;
          b.thread_elements()[e] = t < p.T ? olat[(size_t(t) * p.H + h) * kMlaL + k0 + kt + frow] : 0.0f;
        }
        NI_UNROLL for (uint i = 0; i < 2; ++i) simdgroup_multiply_accumulate(acc[i][j], a[i], b, acc[i][j]);
      }
    }
  }
  for (uint j = 0; j < nt; ++j)
    NI_UNROLL for (uint i = 0; i < 2; ++i) simdgroup_store(acc[i][j], &red[sg][i][j][0], 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 128 threads: (i, j, row-in-fragment, column-in-fragment) = 2 x 2 x 8 x 8 / 2 elements; thread takes 2
  NI_UNROLL for (uint u = 0; u < 2; ++u) {
    const uint x = tid * 2 + u;  // < 256
    const uint i = x / 128, j = (x / 64) % 2, e = x % 64, rr = e / 8, cc = e % 8;
    const uint t = j * 8 + cc, n = n0 + i * 8 + rr;
    if (j >= nt || t >= p.T) continue;
    const float v = red[0][i][j][e] + red[1][i][j][e] + red[2][i][j][e] + red[3][i][j][e];
    const float y = v * sigmoid(gate[size_t(t) * p.gate_ld + h * 256 + n]);
    const size_t o = size_t(t) * p.out_ld + h * 256 + n;
    if (p.out_half) ((device half*)out)[o] = half(y);
    else ((device float*)out)[o] = y;
  }
}

// Expand-from-cache GEMM (prefill): for the M positions kbase.., K[h][m][n] (n < 192) and V[h][m][n - 192] (half
// [H][M][256] each) = sum_k row(kbase + m)[k] W[h*448 + n, k], k < 512. Half MMA, fp32 accumulation; W's bf16 values
// go to half (exact in half's normal range; below 2^-14 they lose low bits, a part in 1e4 of these weights' scale).
// Tile 64 m x 64 n, 32 k a step, 4 simdgroups of 32 x 32. grid: (7, ceil(M/64), H) x 128.
struct Hy4MlaExpP {
  uint M, cap, ring, kbase;
};
[[kernel, max_total_threads_per_threadgroup(128)]] void hy4_mla_expand(
    const device half* rows [[buffer(0)]], const device bfloat* W [[buffer(1)]], device half* Kh [[buffer(2)]],
    device half* Vh [[buffer(3)]], constant Hy4MlaExpP& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LD = 40;  // halves a tile row (32 + 8 pad)
  threadgroup half As[64 * LD], Bs[64 * LD];  // [m][k], [n][k]: 10 KB
  const uint n0 = tg.x * 64, m0 = tg.y * 64, h = tg.z;
  const uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
  // loads: thread -> row tid / 2 (of 64), 16 k at (tid % 2) * 16
  const uint lr = tid / 2, lk = (tid % 2) * 16;
  const uint m = min(m0 + lr, p.M - 1), pos = p.kbase + m;
  const device half* ar = rows + size_t(p.ring ? pos % p.cap : pos) * kMlaW + lk;
  const device bfloat* br = W + size_t(h * 448 + n0 + lr) * kMlaL + lk;
  uint4 ra[2], rb[2];
  NI_UNROLL for (uint u = 0; u < 2; ++u) {
    ra[u] = *(const device uint4*)(ar + u * 8);
    rb[u] = *(const device uint4*)(br + u * 8);
  }
  simdgroup_float8x8 acc[4][4];
  NI_UNROLL for (uint i = 0; i < 4; ++i) NI_UNROLL for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);
  for (uint k0 = 0; k0 < kMlaL; k0 += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous step's reads are done
    NI_UNROLL for (uint u = 0; u < 2; ++u) {
      *(threadgroup uint4*)(&As[lr * LD + lk + u * 8]) = ra[u];
      half hb[8];
      NI_UNROLL for (uint w = 0; w < 4; ++w) {
        const float2 f = float2(as_type<bfloat2>(rb[u][w]));
        hb[2 * w] = half(f.x);
        hb[2 * w + 1] = half(f.y);
      }
      *(threadgroup half4*)(&Bs[lr * LD + lk + u * 8]) = half4(hb[0], hb[1], hb[2], hb[3]);
      *(threadgroup half4*)(&Bs[lr * LD + lk + u * 8 + 4]) = half4(hb[4], hb[5], hb[6], hb[7]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 32 < kMlaL)
      NI_UNROLL for (uint u = 0; u < 2; ++u) {
        ra[u] = *(const device uint4*)(ar + k0 + 32 + u * 8);
        rb[u] = *(const device uint4*)(br + k0 + 32 + u * 8);
      }
    NI_UNROLL for (uint kk = 0; kk < 32; kk += 8) {
      simdgroup_half8x8 a[4], b[4];
      NI_UNROLL for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], &As[(sm + i * 8) * LD + kk], LD);
      NI_UNROLL for (uint j = 0; j < 4; ++j)
        simdgroup_load(b[j], &Bs[(sn + j * 8) * LD + kk], LD, ulong2(0, 0), true);
      NI_UNROLL for (uint i = 0; i < 4; ++i)
        NI_UNROLL for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4, fcol = (qd & 2) * 2 + (lane % 2) * 2;
  NI_UNROLL for (uint i = 0; i < 4; ++i) {
    const uint mm = m0 + sm + i * 8 + frow;
    if (mm >= p.M) continue;
    NI_UNROLL for (uint j = 0; j < 4; ++j) {
      const uint n = n0 + sn + j * 8 + fcol;
      const half2 v = half2(float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]));
      if (n < 192) *(device half2*)(Kh + (size_t(h) * p.M + mm) * 256 + n) = v;
      else *(device half2*)(Vh + (size_t(h) * p.M + mm) * 256 + n - 192) = v;
    }
  }
}

// ---- compressed indexer (semantics: hy4_ops.h, "compressed indexer") ----
// Caches: tok half [cap][128] token keys, grp half [cap/4][128] group keys (pooled, normed, roped), every value a bf16
// value (exact in half in its normal range).
constant constexpr uint kIxD = 128, kIxH = 16;

NI_INLINE float hy4_tg_sum128(float v, threadgroup float* s, uint tid) {  // 128 threads
  v = simd_sum(v);
  if (tid % 32 == 0) s[tid / 32] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = s[0] + s[1] + s[2] + s[3];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}
// rotate the last 64 of 128 dims (pair (64 + 2i, 64 + 2i + 1) at frequency i), bf16 in, bf16 out; d = the thread's dim
NI_INLINE float hy4_ix_rope(float x, uint d, uint pos, float theta, uint rbf) {
  const float xp = simd_shuffle_xor(x, 1);
  if (d < 64) return x;
  float cs, sn;
  hy4_rope_cs(pos, (d - 64) / 2, theta, rbf, cs, sn);
  return hy4_bf16_rne((d & 1) ? x * cs + xp * sn : x * cs - xp * sn);
}

// token keys: tok[pos0 + t] = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), x = bf16(tk row). grid: T x 128.
struct Hy4IxTokP {
  uint pos0, tk_ld, tk_off, w_t;
  float eps;
};
[[kernel]] void hy4_ix_tok(const device float* tk [[buffer(0)]], const device void* w [[buffer(1)]],
                           device half* tok [[buffer(2)]], constant Hy4IxTokP& p [[buffer(3)]],
                           uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float s[4];
  const float x = hy4_bf16_rne(tk[size_t(t) * p.tk_ld + p.tk_off + tid]);
  const float ss = hy4_tg_sum128(x * x, s, tid);
  const float n = hy4_bf16_rne(x * metal::precise::rsqrt(ss / 128.0f + p.eps));
  tok[size_t(p.pos0 + t) * kIxD + tid] = half(hy4_bf16_rne(hy4_ld(w, tid, p.w_t) * n));
}

// group keys g0 + tg of the token keys (gated pooling, key_norm, rope at 4g). grid: ngroups x 128 (thread = dim).
struct Hy4IxPoolP {
  uint g0, kn_t;
  float eps, theta;
  uint rope_bf16;
};
[[kernel]] void hy4_ix_pool(const device half* tok [[buffer(0)]], const device float* wv [[buffer(1)]],
                            const device float* wg [[buffer(2)]], const device float* ape [[buffer(3)]],
                            const device void* kn [[buffer(4)]], device half* grp [[buffer(5)]],
                            constant Hy4IxPoolP& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                            uint d [[thread_index_in_threadgroup]]) {
  threadgroup float x[4][kIxD];
  threadgroup float s[4];
  const uint g = p.g0 + tg;
  NI_UNROLL for (uint i = 0; i < 4; ++i) x[i][d] = float(tok[size_t(g * 4 + i) * kIxD + d]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float v[4] = {0, 0, 0, 0}, a[4] = {0, 0, 0, 0};
  const device float4* rv = (const device float4*)(wv + size_t(d) * kIxD);
  const device float4* rg = (const device float4*)(wg + size_t(d) * kIxD);
  for (uint k = 0; k < kIxD / 4; ++k) {
    const float4 bv = rv[k], bg = rg[k];
    NI_UNROLL for (uint i = 0; i < 4; ++i) {
      const float4 xi = float4(x[i][4 * k], x[i][4 * k + 1], x[i][4 * k + 2], x[i][4 * k + 3]);
      v[i] += dot(bv, xi);
      a[i] += dot(bg, xi);
    }
  }
  float m = -INFINITY;
  NI_UNROLL for (uint i = 0; i < 4; ++i) {
    a[i] += ape[i * kIxD + d];
    m = max(m, a[i]);
  }
  float es = 0.0f, pv = 0.0f;
  NI_UNROLL for (uint i = 0; i < 4; ++i) {
    a[i] = metal::precise::exp(a[i] - m);
    es += a[i];
  }
  NI_UNROLL for (uint i = 0; i < 4; ++i) pv += v[i] * (a[i] / es);
  const float ss = hy4_tg_sum128(pv * pv, s, d);
  const float n = hy4_bf16_rne(hy4_ld(kn, d, p.kn_t) * (pv * metal::precise::rsqrt(ss / 128.0f + p.eps)));
  grp[size_t(g) * kIxD + d] = half(hy4_ix_rope(n, d, g * 4, p.theta, p.rope_bf16));
}

// queries: qx half [T][16][128] = bf16(rope(bf16(qi))) at pos0 + t; hwx [T][16] = bf16(hw) * 16^-0.5 * 128^-0.5.
// grid: (16, T) x 128.
struct Hy4IxQP {
  uint pos0, qi_ld, qi_off, hw_ld, hw_off, rope_bf16;
  float theta;
};
[[kernel]] void hy4_ix_q(const device float* qi [[buffer(0)]], const device float* hw [[buffer(1)]],
                         device half* qx [[buffer(2)]], device float* hwx [[buffer(3)]],
                         constant Hy4IxQP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint d [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, t = tg.y;
  const float x = hy4_bf16_rne(qi[size_t(t) * p.qi_ld + p.qi_off + h * kIxD + d]);
  qx[(size_t(t) * kIxH + h) * kIxD + d] = half(hy4_ix_rope(x, d, p.pos0 + t, p.theta, p.rope_bf16));
  if (d == 0) hwx[t * kIxH + h] = hy4_bf16_rne(hw[size_t(t) * p.hw_ld + p.hw_off + h]) * 0.02209708691207961f;
}

// logits [T][gld]: logit(t, g) = sum_h hwx[t][h] * relu(qx[t][h] . grp[g]) for g < G_t = (pos0 + t) / 4, rows with
// G_t > 511 only (the others recall every group). A threadgroup takes 32 groups x 4 rows (simdgroup = row: 16 heads x
// 32 groups, half MMA, fp32 accumulation). grid: (ceil(Gmax / 32), ceil(T / 4)) x 128.
struct Hy4IxScoreP {
  uint T, pos0, gld;
};
[[kernel, max_total_threads_per_threadgroup(128)]] void hy4_ix_score(
    const device half* qx [[buffer(0)]], const device float* hwx [[buffer(1)]], const device half* grp [[buffer(2)]],
    device float* logits [[buffer(3)]], constant Hy4IxScoreP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LD = 136;
  threadgroup half ks[32 * LD];  // 8.5 KB
  const uint g0 = tg.x * 32, tlast = min(tg.y * 4 + 3, p.T - 1);
  const uint gmax = (p.pos0 + tlast) / 4;  // the most groups a row of this threadgroup scores
  if (g0 >= gmax || gmax <= kIxSel - 1) return;
  // stage 32 group keys: thread -> key tid / 4, 32 dims at (tid % 4) * 32
  {
    const uint g = min(g0 + tid / 4, gmax - 1);
    const device uint4* src = (const device uint4*)(grp + size_t(g) * kIxD + (tid % 4) * 32);
    threadgroup uint4* dst = (threadgroup uint4*)(ks + (tid / 4) * LD + (tid % 4) * 32);
    NI_UNROLL for (uint u = 0; u < 4; ++u) dst[u] = src[u];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint t = tg.y * 4 + sg;
  if (t >= p.T) return;
  const uint G = (p.pos0 + t) / 4;
  if (G <= kIxSel - 1 || g0 >= G) return;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4, fcol = (qd & 2) * 2 + (lane % 2) * 2;
  simdgroup_float8x8 acc[2][4];
  NI_UNROLL for (uint a = 0; a < 2; ++a) NI_UNROLL for (uint j = 0; j < 4; ++j) acc[a][j] = simdgroup_float8x8(0);
  const device half* qb = qx + size_t(t) * kIxH * kIxD;
  NI_UNROLL for (uint d0 = 0; d0 < kIxD; d0 += 8) {
    simdgroup_half8x8 qa[2], kb[4];
    NI_UNROLL for (uint a = 0; a < 2; ++a) {
      const half2 q2 = *(const device half2*)(qb + (a * 8 + frow) * kIxD + d0 + fcol);
      qa[a].thread_elements()[0] = q2.x;
      qa[a].thread_elements()[1] = q2.y;
    }
    NI_UNROLL for (uint j = 0; j < 4; ++j) simdgroup_load(kb[j], ks + j * 8 * LD + d0, LD, ulong2(0, 0), true);
    NI_UNROLL for (uint a = 0; a < 2; ++a)
      NI_UNROLL for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[a][j], qa[a], kb[j], acc[a][j]);
  }
  const float w0 = hwx[t * kIxH + frow], w1 = hwx[t * kIxH + 8 + frow];
  NI_UNROLL for (uint j = 0; j < 4; ++j)
    NI_UNROLL for (uint i = 0; i < 2; ++i) {
      float v = w0 * max(acc[0][j].thread_elements()[i], 0.0f) + w1 * max(acc[1][j].thread_elements()[i], 0.0f);
      v += simd_shuffle_xor(v, 2);
      v += simd_shuffle_xor(v, 4);
      v += simd_shuffle_xor(v, 16);
      const uint g = g0 + j * 8 + fcol + i;
      if ((lane & 22) == 0 && g < G) logits[size_t(t) * p.gld + g] = v;
    }
}

// selection: sel[t][0 .. n) ascending, n = min(G, 511) + 1, the last G (G = (pos0 + t) / 4). G > 511: the 511 groups
// of highest logit (radix select on order-preserving keys, -0 as +0; ties: lower g first). grid: T x 256.
struct Hy4IxSelP {
  uint pos0, gld;
};
NI_INLINE uint hy4_ix_key(float f) {
  uint u = as_type<uint>(f);
  if ((u << 1) == 0) u = 0;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
[[kernel, max_total_threads_per_threadgroup(256)]] void hy4_ix_select(
    const device float* logits [[buffer(0)]], device uint* sel [[buffer(1)]], constant Hy4IxSelP& p [[buffer(2)]],
    uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint sh[4];
  threadgroup uint scan[2][8];
  const uint G = (p.pos0 + t) / 4;
  device uint* out = sel + size_t(t) * kIxSel;
  if (G <= kIxSel - 1) {
    for (uint i = tid; i <= G; i += 256) out[i] = i;
    return;
  }
  const device float* lg = logits + size_t(t) * p.gld;
  uint prefix = 0, mask = 0, want = kIxSel - 1;
  for (int shift = 24; shift >= 0; shift -= 8) {
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = tid; g < G; g += 256) {
      const uint k = hy4_ix_key(lg[g]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint above = 0, b = 255;
      for (;; --b) {
        const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (above + c >= want || b == 0) break;
        above += c;
      }
      sh[0] = b;
      sh[1] = above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= sh[0] << shift;
    mask |= 255u << shift;
    want -= sh[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // keys > prefix: all taken; keys == prefix: the first `want` in g order
  uint eq_seen = 0, n_out = 0;
  for (uint base = 0; base < G; base += 256) {
    const uint g = base + tid;
    const uint k = g < G ? hy4_ix_key(lg[g]) : 0u;
    const bool gt = g < G && k > prefix, eq = g < G && k == prefix;
    const uint eqx = simd_prefix_exclusive_sum(uint(eq)), eqs = simd_sum(uint(eq));
    if (lane == 0) scan[0][sg] = eqs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint eq_before = eq_seen + eqx, eq_tot = 0;
    for (uint i = 0; i < 8; ++i) {
      if (i < sg) eq_before += scan[0][i];
      eq_tot += scan[0][i];
    }
    const bool take = gt || (eq && eq_before < want);
    const uint tx = simd_prefix_exclusive_sum(uint(take)), ts = simd_sum(uint(take));
    if (lane == 0) scan[1][sg] = ts;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = n_out + tx, tot = 0;
    for (uint i = 0; i < 8; ++i) {
      if (i < sg) at += scan[1][i];
      tot += scan[1][i];
    }
    if (take && at < kIxSel - 1) out[at] = g;
    eq_seen += eq_tot;
    n_out += tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) out[kIxSel - 1] = G;
}

// One selection row for the drafter: dst[0 .. 512) = sel[row], row = idx ? min(idx[0], T - 1) : fixed. grid: 512.
struct Hy4IxPickP {
  uint T, row, has_idx;
};
[[kernel]] void hy4_ix_pick(const device uint* sel [[buffer(0)]], const device int* idx [[buffer(1)]],
                            device uint* dst [[buffer(2)]], constant Hy4IxPickP& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  const uint row = p.has_idx ? min(uint(max(idx[0], 0)), p.T - 1) : p.row;
  dst[i] = sel[size_t(row) * kIxSel + i];
}
