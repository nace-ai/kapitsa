#include "common.h"

// ---- Qwen3.8-Flash-Next (qwen4_exp): hyper-connections, the n-gram embedding (PLE), the sparse-attention indexer ----
// The residual stream is S = 4 streams of D = 2560 ([s0 | s1 | s2 | s3], fp32). Norm weights are the checkpoint's
// zero-centred w applied as 1 + w; the model passes them as fp32 (1 + w) vectors.

// Sum of a float4 across a threadgroup of up to 1024 threads in one exchange (scratch: >= 32 float4)
NI_INLINE float4 tg_sum4(float4 v, threadgroup float4* scratch, uint tid, uint ntg) {
  const uint lane = tid & 31, sg = tid >> 5, nsg = (ntg + 31) >> 5;
  v = float4(simd_sum(v.x), simd_sum(v.y), simd_sum(v.z), simd_sum(v.w));
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 r = lane < nsg ? scratch[lane] : float4(0.0f);
  r = float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}
NI_INLINE float4 sigmoid4(float4 x) { return 1.0f / (1.0f + metal::fast::exp(-x)); }

struct HcParams {
  uint R;        // rows
  uint D;        // stream width
  uint S;        // streams
  uint has_y;    // first X += inj_prev[s] * y (the previous block's output), then y = 0
  uint has_inj;  // write the block's injection weights inj[s] = 2 sigmoid(W_inj[s] . Xn / S)
  float eps;
};

// Gated residual, first part, per row (one threadgroup of 1024 threads per row):
//   X += inj_prev[s] * y (has_y; y then zeroed for the next block's accumulation)
//   Xn = per-stream RMSNorm(X) * w       inj = 2 sigmoid(W_inj Xn / S) (has_inj)
// Xn: fp32 (decode / verify rows, the GEMV input) or half (prefill rows, the GEMM input).
// (S = 4 streams: their sums of squares and the four injection dots each go through one tg_sum4)
template <typename XN>
NI_INLINE void hc_pre_impl(device float* X, device float* y, const device float* inj_prev, const device float* w,
                           const device bfloat* winj, device XN* xn, device float* inj, constant HcParams& p, uint r,
                           uint tid, uint ntg, threadgroup float4* scratch) {
  const uint SD = p.S * p.D, D = p.D;
  device float* x = X + r * SD;
  device float* yr = y + r * D;
  const float4 a = p.has_y ? float4(inj_prev[r * 4], inj_prev[r * 4 + 1], inj_prev[r * 4 + 2], inj_prev[r * 4 + 3])
                           : float4(0.0f);
  float4 ss = 0.0f;
  for (uint c = tid; c < D; c += ntg) {  // channel c of every stream
    float4 v = float4(x[c], x[D + c], x[2 * D + c], x[3 * D + c]);
    if (p.has_y) {
      v += a * yr[c];
      x[c] = v.x, x[D + c] = v.y, x[2 * D + c] = v.z, x[3 * D + c] = v.w;
      yr[c] = 0.0f;  // (only this thread reads channel c of y)
    }
    ss += v * v;
  }
  const float4 inv = metal::precise::rsqrt(tg_sum4(ss, scratch, tid, ntg) / float(D) + p.eps);
  float4 dot = 0.0f;
  for (uint c = tid; c < D; c += ntg) {
    float4 v = float4(x[c], x[D + c], x[2 * D + c], x[3 * D + c]) * inv * float4(w[c], w[D + c], w[2 * D + c], w[3 * D + c]);
    xn[r * SD + c] = XN(v.x);
    xn[r * SD + D + c] = XN(v.y);
    xn[r * SD + 2 * D + c] = XN(v.z);
    xn[r * SD + 3 * D + c] = XN(v.w);
    if (p.has_inj) {
      const uint i0 = c, i1 = D + c, i2 = 2 * D + c, i3 = 3 * D + c;
      dot += v.x * float4(winj[i0], winj[SD + i0], winj[2 * SD + i0], winj[3 * SD + i0]);
      dot += v.y * float4(winj[i1], winj[SD + i1], winj[2 * SD + i1], winj[3 * SD + i1]);
      dot += v.z * float4(winj[i2], winj[SD + i2], winj[2 * SD + i2], winj[3 * SD + i2]);
      dot += v.w * float4(winj[i3], winj[SD + i3], winj[2 * SD + i3], winj[3 * SD + i3]);
    }
  }
  if (p.has_inj) {
    const float4 d = tg_sum4(dot, scratch, tid, ntg);
    if (tid == 0) *(device float4*)(inj + r * 4) = 2.0f * sigmoid4(d / float(p.S));
  }
}

#define NI_HC_PRE(NAME, XN)                                                                                    \
  [[kernel]] void NAME(device float* X [[buffer(0)]], device float* y [[buffer(1)]],                           \
                       const device float* inj_prev [[buffer(2)]], const device float* w [[buffer(3)]],        \
                       const device bfloat* winj [[buffer(4)]], device XN* xn [[buffer(5)]],                   \
                       device float* inj [[buffer(6)]], constant HcParams& p [[buffer(7)]],                    \
                       uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],      \
                       uint ntg [[threads_per_threadgroup]]) {                                                 \
    threadgroup float4 scratch[32];                                                                            \
    hc_pre_impl<XN>(X, y, inj_prev, w, winj, xn, inj, p, r, tid, ntg, scratch);                                \
  }
NI_HC_PRE(hc_pre, float)
NI_HC_PRE(hc_pre_h, half)

// The gated residual's first part with the previous MoE block's combine sum folded in (NI_HC_PRECOMB):
// the separate moe_combine dispatch is skipped and each row's threadgroup computes that sum itself —
// phase A replicates moe_combine's arithmetic verbatim (the same float4 ops, ascending routed slots
// then the shared slot, the same token-major y addressing) into threadgroup memory, and phase B is
// hc_pre_impl's loop reading that sum in place of y. yb_ is neither read nor zeroed here: with the
// combine skipped it stays zero from the previous gated residual's zeroing until the next mixer
// output overwrites it. Bitwise identical to moe_combine + hc_pre{,_h} by construction, and
// compared on a GPU: Flash-Next Q2 logit_dump, M5 Pro, 25 Oct, every row identical (#567). y / ys / wts bind as moe_combine's (ys = the
// shared rows, or the shared-fold view of y); the host gate holds cp.D <= 3968 (cp.k is unbounded —
// phase A's slot loop takes any k).
struct HcCombParams {
  uint D;  // stream width (the same value as HcParams.D; the layout of moe_combine's CombineParams)
  uint k;  // routed slots; the shared slot's weight is wts[t * (k + 1) + k]
};

// phase B: hc_pre_impl verbatim except the injected y comes from the threadgroup ysum4 (no y zeroing)
template <typename XN>
NI_INLINE void hc_pre_comb_impl(device float* X, threadgroup float4* ysum4, const device float* inj_prev,
                                const device float* w, const device bfloat* winj, device XN* xn, device float* inj,
                                constant HcParams& p, uint r, uint tid, uint ntg, threadgroup float4* scratch) {
  const uint SD = p.S * p.D, D = p.D;
  device float* x = X + r * SD;
  const float4 a = p.has_y ? float4(inj_prev[r * 4], inj_prev[r * 4 + 1], inj_prev[r * 4 + 2], inj_prev[r * 4 + 3])
                           : float4(0.0f);
  float4 ss = 0.0f;
  for (uint c = tid; c < D; c += ntg) {  // channel c of every stream
    float4 v = float4(x[c], x[D + c], x[2 * D + c], x[3 * D + c]);
    if (p.has_y) {
      v += a * ysum4[c >> 2][c & 3];
      x[c] = v.x, x[D + c] = v.y, x[2 * D + c] = v.z, x[3 * D + c] = v.w;
    }
    ss += v * v;
  }
  const float4 inv = metal::precise::rsqrt(tg_sum4(ss, scratch, tid, ntg) / float(D) + p.eps);
  float4 dot = 0.0f;
  for (uint c = tid; c < D; c += ntg) {
    float4 v = float4(x[c], x[D + c], x[2 * D + c], x[3 * D + c]) * inv * float4(w[c], w[D + c], w[2 * D + c], w[3 * D + c]);
    xn[r * SD + c] = XN(v.x);
    xn[r * SD + D + c] = XN(v.y);
    xn[r * SD + 2 * D + c] = XN(v.z);
    xn[r * SD + 3 * D + c] = XN(v.w);
    if (p.has_inj) {
      const uint i0 = c, i1 = D + c, i2 = 2 * D + c, i3 = 3 * D + c;
      dot += v.x * float4(winj[i0], winj[SD + i0], winj[2 * SD + i0], winj[3 * SD + i0]);
      dot += v.y * float4(winj[i1], winj[SD + i1], winj[2 * SD + i1], winj[3 * SD + i1]);
      dot += v.z * float4(winj[i2], winj[SD + i2], winj[2 * SD + i2], winj[3 * SD + i2]);
      dot += v.w * float4(winj[i3], winj[SD + i3], winj[2 * SD + i3], winj[3 * SD + i3]);
    }
  }
  if (p.has_inj) {
    const float4 d = tg_sum4(dot, scratch, tid, ntg);
    if (tid == 0) *(device float4*)(inj + r * 4) = 2.0f * sigmoid4(d / float(p.S));
  }
}

#define NI_HC_PRE_COMB(NAME, XN)                                                                               \
  [[kernel]] void NAME(device float* X [[buffer(0)]], const device float* inj_prev [[buffer(1)]],              \
                       const device float* w [[buffer(2)]], const device bfloat* winj [[buffer(3)]],           \
                       device XN* xn [[buffer(4)]], device float* inj [[buffer(5)]],                           \
                       constant HcParams& p [[buffer(6)]], const device float* y [[buffer(7)]],                \
                       const device float* ys [[buffer(8)]], const device float* wts [[buffer(9)]],            \
                       constant HcCombParams& cp [[buffer(10)]], uint r [[threadgroup_position_in_grid]],      \
                       uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {       \
    threadgroup float4 scratch[32];                                                                            \
    threadgroup float4 ysum4[992];  /* 15,872 B; with the scratch 16,384 B static (the cap, read as 32,768) */ \
    if (tid < cp.D / 4) {           /* moe_combine's accumulation verbatim: ascending slots, shared last */    \
      const uint d4 = tid;                                                                                     \
      float4 acc = 0.0f;                                                                                       \
      for (uint s = 0; s < cp.k; ++s)                                                                          \
        acc += wts[r * (cp.k + 1) + s] * ((const device float4*)(y + (r * cp.k + s) * cp.D))[d4];              \
      acc += wts[r * (cp.k + 1) + cp.k] * ((const device float4*)(ys + r * cp.D))[d4];                         \
      ysum4[d4] = acc;                                                                                         \
    }                                                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
    hc_pre_comb_impl<XN>(X, ysum4, inj_prev, w, winj, xn, inj, p, r, tid, ntg, scratch);                       \
  }
NI_HC_PRE_COMB(hc_pre_comb, float)
NI_HC_PRE_COMB(hc_pre_comb_h, half)


// h = silu(h / S) in place (the low-rank mix's hidden), n values. grid: ceil(n / 256) x 256.
[[kernel]] void hc_act(device float* h [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& inv_s [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < n) h[i] = silu(h[i] * inv_s);
}
// the same into half rows (prefill: the up projection's GEMM input)
[[kernel]] void hc_act_h(const device float* h [[buffer(0)]], device half* o [[buffer(1)]], constant uint& n [[buffer(2)]],
                         constant float& inv_s [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) o[i] = half(silu(h[i] * inv_s));
}

// The block input u[r, c] = mean_s sigmoid(m[r, s, c]) * Xn[r, s, c] (m = W_up silu(W_down Xn / S)).
// grid: (ceil(D / 256), R) x 256. u rows `u_ld` apart.
template <typename XN, typename U>
NI_INLINE void hc_mix_impl(const device float* m, const device XN* xn, device U* u, constant HcParams& p, uint u_ld,
                           uint c, uint r) {
  if (c >= p.D) return;
  const uint SD = p.S * p.D;
  float acc = 0.0f;
  for (uint s = 0; s < p.S; ++s) acc += sigmoid(m[r * SD + s * p.D + c]) * float(xn[r * SD + s * p.D + c]);
  u[r * u_ld + c] = U(acc / float(p.S));
}
#define NI_HC_MIX(NAME, XN, U)                                                                                 \
  [[kernel]] void NAME(const device float* m [[buffer(0)]], const device XN* xn [[buffer(1)]],                 \
                       device U* u [[buffer(2)]], constant HcParams& p [[buffer(3)]], constant uint& u_ld [[buffer(4)]], \
                       uint2 gid [[thread_position_in_grid]]) {                                                \
    hc_mix_impl<XN, U>(m, xn, u, p, u_ld, gid.x, gid.y);                                                       \
  }
NI_HC_MIX(hc_mix, float, float)
NI_HC_MIX(hc_mix_h, half, half)

// ---- PLE: the n-gram embedding's gate, norms and dilated causal conv ----
// Inputs per row: key = key_proj(emb) [S*D], val = value_proj(emb) [D], X (the residual, updated in place).
//   kn = RMSNorm_s(key) * wk;  qn = RMSNorm_s(X) * wq;  g_s = <kn_s, qn_s> / sqrt(D);  g_s = sign(g) sqrt(max(|g|, 1e-6))
//   gv_s = sigmoid(g_s) * val;  gvn = RMSNorm_s(gv) * wc;  c = silu(sum_j cw[:, j] * gvn[t - DIL * (K - 1 - j)])
//   X += gv + c
// The conv reads the gvn of earlier positions from `ring` (RING slots, position mod RING; zero before 0).
struct PleParams {
  uint R;       // rows
  uint D, S;
  uint pos0;    // position of row 0
  float eps;
  uint ring;    // slots in the ring (a power of two >= R + (K - 1) * DIL)
  uint dil;     // conv dilation (3)
};

// Decode / verify rows (R <= 8): one threadgroup per stream walks the rows in order; thread i owns channels
// 4i..4i+3 of its stream in every row, so the conv's taps of earlier rows (written by the same thread) need no
// barrier. grid: S threadgroups x D/4 threads.
[[kernel]] void ple_rows(const device float* key [[buffer(0)]], const device float* val [[buffer(1)]],
                         device float* X [[buffer(2)]], const device float* wk [[buffer(3)]],
                         const device float* wq [[buffer(4)]], const device float* wc [[buffer(5)]],
                         const device bfloat* cw [[buffer(6)]], device float* ring [[buffer(7)]],
                         constant PleParams& p [[buffer(8)]], uint s [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const uint SD = p.S * p.D, c0 = s * p.D + tid * 4;
  const float4 k_w = *(const device float4*)(wk + c0), q_w = *(const device float4*)(wq + c0);
  const float4 c_w = *(const device float4*)(wc + c0);
  // conv taps j = 0..3 of the 4 channels (cw[c, j]: MLX layout [C, K, 1]) as one float4 per tap
  const float4 t0 = float4(*(const device bfloat4*)(cw + c0 * 4)), t1 = float4(*(const device bfloat4*)(cw + c0 * 4 + 4));
  const float4 t2 = float4(*(const device bfloat4*)(cw + c0 * 4 + 8)), t3 = float4(*(const device bfloat4*)(cw + c0 * 4 + 12));
  const float4 cwj[4] = {float4(t0.x, t1.x, t2.x, t3.x), float4(t0.y, t1.y, t2.y, t3.y), float4(t0.z, t1.z, t2.z, t3.z),
                         float4(t0.w, t1.w, t2.w, t3.w)};
  const float isd = metal::precise::rsqrt(float(p.D));
  for (uint r = 0; r < p.R; ++r) {
    const uint pos = p.pos0 + r;
    const float4 kv = *(const device float4*)(key + r * SD + c0);
    device float4* xp = (device float4*)(X + r * SD + c0);
    const float4 xv = *xp;
    const float kss = tg_sum(dot(kv, kv), scratch, tid, ntg);
    const float xss = tg_sum(dot(xv, xv), scratch, tid, ntg);
    const float4 kn = kv * metal::precise::rsqrt(kss / float(p.D) + p.eps) * k_w;
    const float4 qn = xv * metal::precise::rsqrt(xss / float(p.D) + p.eps) * q_w;
    float g = tg_sum(dot(kn, qn), scratch, tid, ntg) * isd;
    g = sign(g) * metal::precise::sqrt(max(fabs(g), 1e-6f));
    const float4 gv = sigmoid(g) * *(const device float4*)(val + r * p.D + tid * 4);
    const float gss = tg_sum(dot(gv, gv), scratch, tid, ntg);
    const float4 gvn = gv * metal::precise::rsqrt(gss / float(p.D) + p.eps) * c_w;
    *(device float4*)(ring + (pos % p.ring) * SD + c0) = gvn;
    float4 acc = cwj[3] * gvn;
    for (uint j = 0; j < 3; ++j) {
      const uint back = p.dil * (3 - j);
      if (pos >= back) acc += cwj[j] * *(const device float4*)(ring + ((pos - back) % p.ring) * SD + c0);
    }
    *xp = xv + gv + float4(silu(acc.x), silu(acc.y), silu(acc.z), silu(acc.w));
  }
}

// Prefill rows, step 1: gv and gvn of every row into gvb / gvnb [T, S*D]. grid: (T, S) x D/4 threads.
[[kernel]] void ple_gate_rows(const device float* key [[buffer(0)]], const device float* val [[buffer(1)]],
                              const device float* X [[buffer(2)]], const device float* wk [[buffer(3)]],
                              const device float* wq [[buffer(4)]], const device float* wc [[buffer(5)]],
                              device float* gvb [[buffer(6)]], device float* gvnb [[buffer(7)]],
                              constant PleParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const uint r = tg.x, s = tg.y, SD = p.S * p.D, c0 = s * p.D + tid * 4, ntg = ntg2.x;
  const float4 kv = *(const device float4*)(key + r * SD + c0), xv = *(const device float4*)(X + r * SD + c0);
  const float kss = tg_sum(dot(kv, kv), scratch, tid, ntg);
  const float xss = tg_sum(dot(xv, xv), scratch, tid, ntg);
  const float4 kn = kv * metal::precise::rsqrt(kss / float(p.D) + p.eps) * *(const device float4*)(wk + c0);
  const float4 qn = xv * metal::precise::rsqrt(xss / float(p.D) + p.eps) * *(const device float4*)(wq + c0);
  float g = tg_sum(dot(kn, qn), scratch, tid, ntg) * metal::precise::rsqrt(float(p.D));
  g = sign(g) * metal::precise::sqrt(max(fabs(g), 1e-6f));
  const float4 gv = sigmoid(g) * *(const device float4*)(val + r * p.D + tid * 4);
  const float gss = tg_sum(dot(gv, gv), scratch, tid, ntg);
  *(device float4*)(gvb + r * SD + c0) = gv;
  *(device float4*)(gvnb + r * SD + c0) = gv * metal::precise::rsqrt(gss / float(p.D) + p.eps) * *(const device float4*)(wc + c0);
}

// Prefill rows, step 2: X += gv + silu(conv(gvn)), taps from gvnb (this pass's rows) or the ring (earlier
// positions). grid: (S*D / 256, T) x 256.
[[kernel]] void ple_conv_rows(const device float* gvb [[buffer(0)]], const device float* gvnb [[buffer(1)]],
                              device float* X [[buffer(2)]], const device bfloat* cw [[buffer(3)]],
                              const device float* ring [[buffer(4)]], constant PleParams& p [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y, SD = p.S * p.D;
  if (c >= SD) return;
  const uint pos = p.pos0 + r;
  float acc = float(cw[c * 4 + 3]) * gvnb[r * SD + c];
  for (uint j = 0; j < 3; ++j) {
    const uint back = p.dil * (3 - j);
    if (pos < back) continue;
    const float v = r >= back ? gvnb[(r - back) * SD + c] : ring[((pos - back) % p.ring) * SD + c];
    acc += float(cw[c * 4 + j]) * v;
  }
  X[r * SD + c] += gvb[r * SD + c] + silu(acc);
}

// Prefill rows, step 3: the ring after the pass: slot k holds the latest position < end (end = pos0 + T, or a
// checkpoint's position inside the pass when writing a snapshot ring `dst` from the live ring `src`) congruent to
// k: from gvnb when it is in this pass, else src's slot (positions before the pass). grid: (S*D / 256, ring) x 256.
[[kernel]] void ple_ring_rows(const device float* gvnb [[buffer(0)]], const device float* src [[buffer(1)]],
                              device float* dst [[buffer(2)]], constant PleParams& p [[buffer(3)]],
                              constant uint& end [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, k = gid.y, SD = p.S * p.D;
  if (c >= SD || end == 0) return;
  const uint q = end - 1 - ((end - 1 - k) % p.ring);  // the latest position < end with q % ring == k
  if (q >= p.pos0 && q < end) dst[k * SD + c] = gvnb[(q - p.pos0) * SD + c];
  else if (src != dst) dst[k * SD + c] = src[k * SD + c];
}

// ---- Sparse-attention indexer: index queries, raw index keys, block keys ----
// iqk rows = [iq head 0..NH-1 (HD each) | ik (HD)]. Index queries: RMSNorm * wq, partial NeoX RoPE (rot dims) at the
// row's position (M-RoPE for image rows); the raw index key goes to the cache ik [pos, HD] (half). Block j (tokens 4j..4j+3) gets its key
// once complete: RMSNorm(mean(ik[4j..4j+3])) * wk, RoPE at position 4j -> bk [j, HD] (fp32).
struct QsaParams {
  uint R;       // rows
  uint NH;      // index query heads (4)
  uint HD;      // index head dim (128)
  uint rot;     // rotary dims (64)
  uint pos0;
  uint ratio;   // tokens per block (4)
  uint ld;      // floats per iqk row
  float eps;
  float theta;
};

// RoPE of cache row `pos`: at its M-RoPE position for image rows (rows < mr.mn take (t, h, w) by the interleaved
// sections, as attn_prep_m), else at pos + mr.delta (text: mn = delta = 0)
NI_INLINE float qsa_rope(threadgroup float* row, float x, uint tid, uint rot, uint pos, float theta,
                         const device int* mpos, constant MropeParams& mr) {
  const uint hr = rot / 2;
  if (tid >= rot) return x;
  const uint i = tid % hr;
  int rp;
  if (pos < mr.mn) {
    const uint a = (i % 3 == 1 && i < mr.sec1) ? 1u : (i % 3 == 2 && i < mr.sec2) ? 2u : 0u;
    rp = mpos[pos * 3 + a];
  } else {
    rp = int(pos) + mr.delta;
  }
  const float ang = float(rp) * metal::precise::pow(theta, -2.0f * float(i) / float(rot));
  const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
  return tid < hr ? row[tid] * cs - row[tid + hr] * sn : row[tid] * cs + row[tid - hr] * sn;
}

// grid: (NH + 1, R) x HD threads
[[kernel]] void qsa_prep(const device float* iqk [[buffer(0)]], const device float* wq [[buffer(1)]],
                         device float* iq [[buffer(2)]], device half* ik [[buffer(3)]],
                         constant QsaParams& p [[buffer(4)]], const device int* mpos [[buffer(5)]],
                         constant MropeParams& mr [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float row[256];
  const uint h = tg.x, r = tg.y, pos = p.pos0 + r;
  const float x = iqk[r * p.ld + h * p.HD + tid];
  if (h == p.NH) {
    ik[pos * p.HD + tid] = half(x);
    return;
  }
  const float ss = tg_sum(x * x, scratch, tid, p.HD);
  row[tid] = x * metal::precise::rsqrt(ss / float(p.HD) + p.eps) * wq[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  iq[(r * p.NH + h) * p.HD + tid] = qsa_rope(row, row[tid], tid, p.rot, pos, p.theta, mpos, mr);
}

// grid: R x HD threads (rows that complete a block write its key)
[[kernel]] void qsa_block(const device half* ik [[buffer(0)]], const device float* wk [[buffer(1)]],
                          device float* bk [[buffer(2)]], constant QsaParams& p [[buffer(3)]],
                          const device int* mpos [[buffer(4)]], constant MropeParams& mr [[buffer(5)]],
                          uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float row[256];
  const uint pos = p.pos0 + r;
  if ((pos + 1) % p.ratio) return;
  const uint j = pos / p.ratio, t0 = j * p.ratio;
  float m = 0.0f;
  for (uint t = 0; t < p.ratio; ++t) m += float(ik[(t0 + t) * p.HD + tid]);
  m /= float(p.ratio);
  const float ss = tg_sum(m * m, scratch, tid, p.HD);
  row[tid] = m * metal::precise::rsqrt(ss / float(p.HD) + p.eps) * wk[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bk[j * p.HD + tid] = qsa_rope(row, row[tid], tid, p.rot, t0, p.theta, mpos, mr);
}

// ---- Sparse-attention selection (queries that see more than budget + ratio - 1 tokens) ----
// Block scores of row r (position pos0 + r, nb = (pos + 1) / ratio complete blocks):
//   score[r, j] = sum_h relu(iq[r, h] . bk[j]) / sqrt(HD), j < nb  (fp32). grid: (ceil(nbmax / 256), R) x 256
struct QsaSelParams {
  uint R;
  uint NH, HD;
  uint pos0;
  uint ratio;
  uint topb;     // blocks kept (budget / ratio)
  uint s_ld;     // floats per score row
  uint i_ld;     // entries per index row (>= topb * ratio + ratio - 1)
};
[[kernel]] void qsa_score(const device float* iq [[buffer(0)]], const device float* bk [[buffer(1)]],
                          device float* score [[buffer(2)]], constant QsaSelParams& p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float4 q4[8 * 32];  // NH <= 8 heads of HD = 128
  const uint r = tg.y, j = tg.x * 256 + tid, H4 = p.HD / 4;
  for (uint i = tid; i < p.NH * H4; i += 256) q4[i] = ((const device float4*)(iq + r * p.NH * p.HD))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nb = (p.pos0 + r + 1) / p.ratio;
  if (j >= nb) return;
  const device float4* k4 = (const device float4*)(bk + j * p.HD);
  float acc = 0.0f;
  for (uint h = 0; h < p.NH; ++h) {
    float d = 0.0f;
    for (uint i = 0; i < H4; ++i) d += dot(q4[h * H4 + i], k4[i]);
    acc += max(d, 0.0f);
  }
  score[r * p.s_ld + j] = acc * metal::precise::rsqrt(float(p.HD));
}

// The same scores for four rows per threadgroup, in one pass over the key blocks (NI_QSA_ROWFUSE). The four rows'
// query matrices sit in threadgroup memory and each sub-chunk of QSA_SUB key blocks is staged once, so a block's bytes
// are read once instead of once per row: at a verify pass of 8 rows two threadgroups cover them, which reads the key
// bytes 4x fewer times than the per-row kernel does. A score is computed by exactly the expression qsa_score uses on
// the same values (same per-head dot, same max, same order, same scale), so the results are expected bit-identical;
// the logit md5 is the gate, and tools/fn_check.cpp compares the two kernels score for score.
// The 16,384 B of static threadgroup memory (8 KB of queries + 8 KB of keys) is the most this macOS's validation layer
// accepts — it counts the static size twice against the 32,768 B cap — so four rows of four indexer heads is the widest
// block that can also run under the layer.
constant constexpr uint QSA_RPG = 4;    // rows per threadgroup
constant constexpr uint QSA_SUB = 16;   // key blocks staged per sub-chunk
constant constexpr uint QSA_NSUB = 8;   // sub-chunks per threadgroup: 128 blocks
constant constexpr uint QSA_NT = QSA_RPG * QSA_SUB;  // one thread per (row, key)
[[kernel]] void qsa_score_rows(const device float* iq [[buffer(0)]], const device float* bk [[buffer(1)]],
                               device float* score [[buffer(2)]], constant QsaSelParams& p [[buffer(3)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint H4 = p.HD / 4, r0 = tg.y * QSA_RPG, nr = min(QSA_RPG, p.R - r0);
  threadgroup float4 q4[QSA_RPG * 4 * 32];  // the rows' queries: NH <= 4 heads of HD = 128 (8 KB)
  threadgroup float4 kb[QSA_SUB * 32];      // one sub-chunk of key blocks (8 KB)
  const device float4* iq4 = (const device float4*)(iq + r0 * p.NH * p.HD);
  for (uint i = tid; i < nr * p.NH * H4; i += QSA_NT) q4[i] = iq4[i];  // a row's query is NH * HD floats: contiguous
  const uint nbtop = (p.pos0 + p.R) / p.ratio;  // the last row's block count: the valid extent of bk
  for (uint c = 0; c < QSA_NSUB; ++c) {
    const uint j0 = (tg.x * QSA_NSUB + c) * QSA_SUB;
    for (uint i = tid; i < QSA_SUB * H4; i += QSA_NT) {
      const uint j = j0 + i / H4;
      kb[i] = j < nbtop ? ((const device float4*)bk)[j * H4 + i % H4] : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint it = tid; it < nr * QSA_SUB; it += QSA_NT) {
      const uint r = it / QSA_SUB, k = it % QSA_SUB, j = j0 + k;
      if (j >= (p.pos0 + r0 + r + 1) / p.ratio) continue;  // this row's own causal limit
      const threadgroup float4* qr = q4 + r * p.NH * H4;
      const threadgroup float4* k4 = kb + k * H4;
      float acc = 0.0f;
      for (uint h = 0; h < p.NH; ++h) {
        float d = 0.0f;
        for (uint i = 0; i < H4; ++i) d += dot(qr[h * H4 + i], k4[i]);
        acc += max(d, 0.0f);
      }
      score[(r0 + r) * p.s_ld + j] = acc * metal::precise::rsqrt(float(p.HD));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The key positions row r attends to: the topb best-scoring blocks (ties: the lowest block indices) as 4 positions
// each, then the incomplete block's 0..ratio-1 positions; every position 0..pos when nb <= topb (dense). idx [R, i_ld],
// cnt [R]; positions in ascending block order. One threadgroup of 1024 threads per row: an MSB-first radix select of
// the topb-th largest score (scores are >= 0, so their bits order like the values), then thread t keeps blocks of its
// contiguous range [t * per, (t + 1) * per) above the threshold and the lowest-index ones equal to it, and writes them
// at offsets from a scan of the per-thread counts.
// MASK (qsa_select_mask): the selection as a bit row instead (idx [R, i_ld] words; bit k of word k / 32 = key k;
// i_ld = ceil(keys / 32)), every bit set for dense rows (the causal limit applies in the attention kernel).
template <bool MASK>
NI_INLINE void qsa_select_impl(const device float* score, device uint* idx, device uint* cnt, constant QsaSelParams& p,
                               uint r, uint tid, uint ntg, threadgroup atomic_uint* hist, threadgroup uint* scan,
                               threadgroup uint* sel) {
  const uint pos = p.pos0 + r, n = pos + 1, nb = n / p.ratio;
  device uint* out = idx + r * p.i_ld;
  if (nb <= p.topb) {
    if (MASK) {
      for (uint i = tid; i < p.i_ld; i += ntg) out[i] = 0xffffffffu;
    } else {
      for (uint i = tid; i < n; i += ntg) out[i] = i;
      if (tid == 0) cnt[r] = n;
    }
    return;
  }
  if (MASK)
    for (uint i = tid; i < p.i_ld; i += ntg) out[i] = 0u;
  const device uint* key = (const device uint*)(score + r * p.s_ld);
  // radix select: prefix = the threshold's bits found so far, rank = how many of the topb are still to place
  uint prefix = 0, pmask = 0, rank = p.topb;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = tid; i < 256; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < nb; j += ntg) {
      const uint k = key[j];
      if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {  // the digit holding the rank-th largest, from the top
      uint acc = 0, d = 255;
      for (;; --d) {
        const uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
        if (acc + c >= rank || d == 0) {
          sel[1] = rank - acc;
          break;
        }
        acc += c;
      }
      sel[0] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= sel[0] << shift;
    pmask |= 255u << shift;
    rank = sel[1];
  }
  // prefix = the threshold key; rank = how many blocks equal to it are kept (the lowest indices)
  const uint per = (nb + ntg - 1) / ntg, j0 = min(tid * per, nb), j1 = min(j0 + per, nb);
  uint gt = 0, eq = 0;
  for (uint j = j0; j < j1; ++j) {
    gt += key[j] > prefix;
    eq += key[j] == prefix;
  }
  // exclusive scan of eq (which ties are kept), then of the kept counts (output offsets)
  scan[tid] = eq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint off = 1; off < ntg; off <<= 1) {
    const uint v = tid >= off ? scan[tid - off] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[tid] += v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint eq_before = scan[tid] - eq;
  const uint eq_keep = eq_before >= rank ? 0u : min(eq, rank - eq_before);
  const uint keep = gt + eq_keep;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  scan[tid] = keep;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint off = 1; off < ntg; off <<= 1) {
    const uint v = tid >= off ? scan[tid - off] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[tid] += v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint o = (scan[tid] - keep) * p.ratio, e_left = eq_keep;
  if (MASK) threadgroup_barrier(mem_flags::mem_device);  // the row's words are zero before any bit is set
  for (uint j = j0; j < j1; ++j) {
    const uint k = key[j];
    const bool take = k > prefix || (k == prefix && e_left > 0);
    if (k == prefix && take) --e_left;
    if (!take) continue;
    if (MASK) {  // ratio = 4 bits of one word (blocks start at multiples of 4)
      atomic_fetch_or_explicit((device atomic_uint*)(out + j * p.ratio / 32), ((1u << p.ratio) - 1u) << (j * p.ratio % 32),
                               memory_order_relaxed);
    } else {
      for (uint t = 0; t < p.ratio; ++t) out[o + t] = j * p.ratio + t;
    }
    o += p.ratio;
  }
  const uint tail0 = nb * p.ratio, base = p.topb * p.ratio;
  if (MASK) {
    if (tid == 0)
      for (uint k = tail0; k < n; ++k) atomic_fetch_or_explicit((device atomic_uint*)(out + k / 32), 1u << (k % 32), memory_order_relaxed);
    return;
  }
  for (uint i = tid; tail0 + i < n; i += ntg) out[base + i] = tail0 + i;
  if (tid == 0) cnt[r] = base + (n - tail0);
}
#define NI_QSA_SELECT(NAME, MASK)                                                                              \
  [[kernel]] void NAME(const device float* score [[buffer(0)]], device uint* idx [[buffer(1)]],                \
                       device uint* cnt [[buffer(2)]], constant QsaSelParams& p [[buffer(3)]],                 \
                       uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],      \
                       uint ntg [[threads_per_threadgroup]]) {                                                 \
    threadgroup atomic_uint hist[256];                                                                         \
    threadgroup uint scan[1024];                                                                               \
    threadgroup uint sel[2];                                                                                   \
    qsa_select_impl<MASK>(score, idx, cnt, p, r, tid, ntg, hist, scan, sel);                                   \
  }
NI_QSA_SELECT(qsa_select, false)
NI_QSA_SELECT(qsa_select_mask, true)

// ---- The fused QSA chain (NI_QSA_FUSE): qsa_prep + qsa_block + qsa_score + qsa_select in one dispatch ----
// One threadgroup of 1024 threads per row; grid: R x 1024. Intended to be bitwise identical to the four-dispatch
// chain (the argument below; fn_check kernels' test_qsa_all compares the two byte for byte, not yet run):
// - Head h's qsa_prep work runs in the thread slice [h * HD, (h + 1) * HD). tg_sum is purely positional (lane =
//   tid & 31, sg = tid >> 5, the partials in scratch[sg], read back by lanes < nsg), so a slice's four simdgroup
//   partials land in the same scratch slots and lanes as qsa_prep's HD-thread threadgroup and reduce the same
//   values in the same order; slices h >= NH (and the threads >= HD of the block phase) only march the barriers.
//   The rope'd queries stay in row4: the iq buffer's row, in threadgroup memory.
// - A block whose last key is >= pos0 cannot be read from bk here: this dispatch is still writing it (in another
//   threadgroup) and there is no grid-wide sync. So every row re-derives those blocks' means itself: float(half(x))
//   is bitwise the ik entry qsa_prep wrote for a key >= pos0, keys below pos0 come from ik (earlier dispatches),
//   and the mean sum, /= ratio, tg_sum, wk scale and rope run exactly as in qsa_block. The row that completes a
//   block publishes it to bk, exactly as qsa_block's completing row does.
// - The score dots are per-thread and position-independent; they read row4 and bk4 or bk, bitwise the values
//   qsa_score reads from iq and bk. The scores go to the score buffer, so qsa_select_impl below runs verbatim at
//   ntg = 1024 after a device-scope barrier, exactly as the qsa_select dispatch would.
// Host gate (qsa_fusable): R <= 16, NH <= 8, HD == 128, ratio a power of two in [2, 16] — so the blocks re-derived
// per row are at most R / 2 <= 8 (bk4 holds 9) and the eight head slices fit the 1024 threads.
[[kernel]] void qsa_all(const device float* iqk [[buffer(0)]], const device float* wq [[buffer(1)]],
                        const device float* wk [[buffer(2)]], device half* ik [[buffer(3)]],
                        device float* bk [[buffer(4)]], device float* score [[buffer(5)]],
                        device uint* idx [[buffer(6)]], device uint* cnt [[buffer(7)]],
                        constant QsaParams& p [[buffer(8)]], const device int* mpos [[buffer(9)]],
                        constant MropeParams& mr [[buffer(10)]], constant QsaSelParams& ps [[buffer(11)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float4 row4[8 * 32];  // the row's rope'd index queries (qsa_score's q4): NH <= 8 heads of HD = 128
  threadgroup float brow[128];      // the pre-rope row of the block being re-derived
  threadgroup float4 bk4[9 * 32];   // re-derived keys of the blocks not complete below pos0
  threadgroup atomic_uint hist[256];
  threadgroup uint scan[1024];
  threadgroup uint sel[2];
  threadgroup float* const row = (threadgroup float*)row4;
  const uint HD = p.HD, H4 = HD / 4, pos = p.pos0 + r;
  // -- qsa_prep: head h = tid / HD, dim d = tid % HD (its (h, r) threadgroup, as a thread slice) --
  const uint h = tid / HD, d = tid - h * HD;
  const bool qh = h < p.NH;
  const float x = qh ? iqk[r * p.ld + h * HD + d] : 0.0f;
  const float ss = tg_sum(x * x, scratch + 4 * h, tid - HD * h, HD);
  if (qh) row[h * HD + d] = x * metal::precise::rsqrt(ss / float(HD) + p.eps) * wq[d];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float iq0 = 0.0f;
  if (qh) iq0 = qsa_rope(row + h * HD, row[h * HD + d], d, p.rot, pos, p.theta, mpos, mr);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (qh) row[h * HD + d] = iq0;
  if (tid < HD) ik[pos * HD + tid] = half(iqk[r * p.ld + p.NH * HD + tid]);  // qsa_prep's last threadgroup
  // -- qsa_block: the blocks whose last key is >= pos0, re-derived (bitwise the ik values) --
  const uint nb = (pos + 1) / p.ratio;          // blocks row r sees
  const uint jold = min(p.pos0 / p.ratio, nb);  // blocks complete below pos0: read from bk
  const bool done = (pos + 1) % p.ratio == 0;   // this row completes block pos / ratio
  for (uint j = jold; j < nb; ++j) {
    const uint t0 = j * p.ratio, b = j - jold;
    float m = 0.0f;
    if (tid < HD) {
      for (uint t = 0; t < p.ratio; ++t) {
        const uint tp = t0 + t;
        m += tp < p.pos0 ? float(ik[tp * HD + tid]) : float(half(iqk[(tp - p.pos0) * p.ld + p.NH * HD + tid]));
      }
      m /= float(p.ratio);
    }
    const float bss = tg_sum(m * m, scratch, tid, HD);  // the threads >= HD march; all read scratch[0 .. 3]
    if (tid < HD) brow[tid] = m * metal::precise::rsqrt(bss / float(HD) + p.eps) * wk[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HD) {
      const float v = qsa_rope(brow, brow[tid], tid, p.rot, t0, p.theta, mpos, mr);
      ((threadgroup float*)bk4)[b * HD + tid] = v;
      if (done && j == pos / p.ratio) bk[j * HD + tid] = v;  // exactly one row completes each block
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // -- qsa_score: thread j scores block j (j += ntg past the first 1024) --
  threadgroup_barrier(mem_flags::mem_threadgroup);  // row4 (and bk4, when the loop ran) ready
  for (uint j = tid; j < nb; j += ntg) {
    float acc = 0.0f;
    if (j < jold) {
      const device float4* k4 = (const device float4*)(bk + j * HD);
      for (uint hh = 0; hh < p.NH; ++hh) {
        float dd = 0.0f;
        for (uint i = 0; i < H4; ++i) dd += dot(row4[hh * H4 + i], k4[i]);
        acc += max(dd, 0.0f);
      }
    } else {
      const threadgroup float4* k4 = bk4 + (j - jold) * H4;
      for (uint hh = 0; hh < p.NH; ++hh) {
        float dd = 0.0f;
        for (uint i = 0; i < H4; ++i) dd += dot(row4[hh * H4 + i], k4[i]);
        acc += max(dd, 0.0f);
      }
    }
    score[r * ps.s_ld + j] = acc * metal::precise::rsqrt(float(HD));
  }
  // -- qsa_select, verbatim (this row's scores made visible by the device-scope barrier) --
  threadgroup_barrier(mem_flags::mem_device);
  qsa_select_impl<false>(score, idx, cnt, ps, r, tid, ntg, hist, scan, sel);
}

// Attention of R rows over their own key lists (idx / cnt from qsa_select): split-K partials as attn_rows_partial
// (combined by attn_rows_reduce with the gate), chunk c of row r covering list entries [c * chunk, (c + 1) * chunk).
// grid: (nchunks, Hkv, R) x 32 * G threads
constant constexpr uint kSpTile = 16;
constant constexpr uint kSpHD = 256;
struct AttnSparseParams {
  uint Hq, Hkv;
  uint chunk;
  uint max_seq;
  float scale;
  uint nchunks;
  uint i_ld;
};
[[kernel]] void attn_rows_sparse(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                 const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                 device float* part_ml [[buffer(4)]], constant AttnSparseParams& p [[buffer(5)]],
                                 const device uint* idx [[buffer(6)]], const device uint* cnt [[buffer(7)]],
                                 uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup half ktile[kSpTile * kSpHD];
  threadgroup half vtile[kSpTile * kSpHD];
  const uint ntg = ntg3.x;
  const uint chunk = tg.x, kvh = tg.y, r = tg.z;
  const uint G = p.Hq / p.Hkv;
  const uint hq = kvh * G + sg;
  const uint start = chunk * p.chunk, end = min(start + p.chunk, cnt[r]);
  const device uint* list = idx + r * p.i_ld;
  float qv[8], acc[8];
  for (int i = 0; i < 8; ++i) {
    qv[i] = q[(r * p.Hq + hq) * kSpHD + lane * 8 + i] * p.scale;
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const device half* kbase = kc + kvh * p.max_seq * kSpHD;
  const device half* vbase = vc + kvh * p.max_seq * kSpHD;
  for (uint t0 = start; t0 < end; t0 += kSpTile) {
    const uint n = min(kSpTile, end - t0);
    for (uint i = tid; i < n * kSpHD / 4; i += ntg) {
      const uint row = list[t0 + i / (kSpHD / 4)], c = i % (kSpHD / 4);
      ((threadgroup half4*)ktile)[i] = ((const device half4*)(kbase + row * kSpHD))[c];
      ((threadgroup half4*)vtile)[i] = ((const device half4*)(vbase + row * kSpHD))[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < G) {
      for (uint j = 0; j < n; ++j) {
        const threadgroup half* kr = ktile + j * kSpHD + lane * 8;
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += qv[i] * float(kr[i]);
        s = simd_sum(s);
        const float mn = max(m, s);
        const float corr = metal::fast::exp(m - mn);
        const float e = metal::fast::exp(s - mn);
        l = l * corr + e;
        const threadgroup half* vr = vtile + j * kSpHD + lane * 8;
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * corr + e * float(vr[i]);
        m = mn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg < G) {
    const uint base = (r * p.Hq + hq) * p.nchunks + chunk;
    for (int i = 0; i < 8; ++i) part_o[base * kSpHD + lane * 8 + i] = acc[i];
    if (lane == 0) {
      part_ml[base * 2 + 0] = m;
      part_ml[base * 2 + 1] = l;
    }
  }
}

// The same for prefill rows, one chunk per row (no partials): out[r, h*D + d] = attention * sigmoid(gate).
// grid: (Hkv, R) x 32 * G threads
[[kernel]] void attn_sparse_rows(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                 const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                                 device float* out [[buffer(4)]], constant AttnSparseParams& p [[buffer(5)]],
                                 const device uint* idx [[buffer(6)]], const device uint* cnt [[buffer(7)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint2 ntg2 [[threads_per_threadgroup]]) {
  threadgroup half ktile[kSpTile * kSpHD];
  threadgroup half vtile[kSpTile * kSpHD];
  const uint ntg = ntg2.x;
  const uint kvh = tg.x, r = tg.y;
  const uint G = p.Hq / p.Hkv;
  const uint hq = kvh * G + sg;
  const uint end = cnt[r];
  const device uint* list = idx + r * p.i_ld;
  float qv[8], acc[8];
  for (int i = 0; i < 8; ++i) {
    qv[i] = q[(r * p.Hq + hq) * kSpHD + lane * 8 + i] * p.scale;
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const device half* kbase = kc + kvh * p.max_seq * kSpHD;
  const device half* vbase = vc + kvh * p.max_seq * kSpHD;
  for (uint t0 = 0; t0 < end; t0 += kSpTile) {
    const uint n = min(kSpTile, end - t0);
    for (uint i = tid; i < n * kSpHD / 4; i += ntg) {
      const uint row = list[t0 + i / (kSpHD / 4)], c = i % (kSpHD / 4);
      ((threadgroup half4*)ktile)[i] = ((const device half4*)(kbase + row * kSpHD))[c];
      ((threadgroup half4*)vtile)[i] = ((const device half4*)(vbase + row * kSpHD))[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < G) {
      for (uint j = 0; j < n; ++j) {
        const threadgroup half* kr = ktile + j * kSpHD + lane * 8;
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += qv[i] * float(kr[i]);
        s = simd_sum(s);
        const float mn = max(m, s);
        const float corr = metal::fast::exp(m - mn);
        const float e = metal::fast::exp(s - mn);
        l = l * corr + e;
        const threadgroup half* vr = vtile + j * kSpHD + lane * 8;
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * corr + e * float(vr[i]);
        m = mn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg < G)
    for (int i = 0; i < 8; ++i) {
      const uint d = lane * 8 + i;
      out[(r * p.Hq + hq) * kSpHD + d] = acc[i] / l * sigmoid(qg[(r * p.Hq + hq) * 2 * kSpHD + kSpHD + d]);
    }
}

// ---- residual-stream moves (rows of S*D; grid: (ceil(D / 256), R) x 256, a thread per channel of a row) ----
// X[r, s, c] = y[r, c] for every stream (the embedding repeated), then y = 0
[[kernel]] void hc_expand(device float* X [[buffer(0)]], device float* y [[buffer(1)]], constant HcParams& p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.D) return;
  const float v = y[r * p.D + c];
  for (uint s = 0; s < p.S; ++s) X[(r * p.S + s) * p.D + c] = v;
  y[r * p.D + c] = 0.0f;
}
// X[r, s, c] += inj[r, s] * y[r, c], then y = 0 (a block's output into the streams)
[[kernel]] void hc_inject(device float* X [[buffer(0)]], device float* y [[buffer(1)]], const device float* inj [[buffer(2)]],
                          constant HcParams& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.D) return;
  const float v = y[r * p.D + c];
  for (uint s = 0; s < p.S; ++s) X[(r * p.S + s) * p.D + c] += inj[r * p.S + s] * v;
  y[r * p.D + c] = 0.0f;
}

// ---- DFlash2 capture of a layer's output (dflash_config.target_hidden_reduction stream_mean) ----
// The mean of row r's S streams after the layer, X + inj * y with the block output y not yet injected (has_y; the
// next block's mix injects it), summed in fp32, as bf16 in columns [col0, col0 + D) of dst row dst_row0 + r (rows ld
// apart). Reads only: the target's state is untouched. grid: (ceil(D / 256), R) x 256.
struct HcCaptureParams {
  uint R, D, S, ld, col0, dst_row0, has_y;
};
[[kernel]] void hc_capture_mean(const device float* X [[buffer(0)]], const device float* y [[buffer(1)]],
                                const device float* inj [[buffer(2)]], device bfloat* dst [[buffer(3)]],
                                constant HcCaptureParams& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.D || r >= p.R) return;
  const float v = p.has_y ? y[r * p.D + c] : 0.0f;
  float sum = 0.0f;
  for (uint s = 0; s < p.S; ++s) sum += X[(r * p.S + s) * p.D + c] + inj[r * p.S + s] * v;
  dst[size_t(p.dst_row0 + r) * p.ld + p.col0 + c] = bfloat(sum / float(p.S));
}

// ---- MTP head input: en = RMSNorm(embedding row) * we, hn = RMSNorm(the 4-stream state, ONE RMS over S*D) * wh ----
// (both weights fp32, 1 + w). grid: R x 1024; en rows D apart, hn rows S*D apart.
template <typename O>
NI_INLINE void mtp4_pre_impl(const device float* e, const device float* h, const device float* we, const device float* wh,
                             device O* en, device O* hn, constant HcParams& p, uint r, uint tid, uint ntg,
                             threadgroup float* scratch) {
  const uint SD = p.S * p.D;
  float se = 0.0f, sh = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) se += e[r * p.D + i] * e[r * p.D + i];
  for (uint i = tid; i < SD; i += ntg) sh += h[r * SD + i] * h[r * SD + i];
  const float ie = metal::precise::rsqrt(tg_sum(se, scratch, tid, ntg) / float(p.D) + p.eps);
  const float ih = metal::precise::rsqrt(tg_sum(sh, scratch, tid, ntg) / float(SD) + p.eps);
  for (uint i = tid; i < p.D; i += ntg) en[r * p.D + i] = O(e[r * p.D + i] * ie * we[i]);
  for (uint i = tid; i < SD; i += ntg) hn[r * SD + i] = O(h[r * SD + i] * ih * wh[i]);
}
#define NI_MTP4_PRE(NAME, O)                                                                                   \
  [[kernel]] void NAME(const device float* e [[buffer(0)]], const device float* h [[buffer(1)]],               \
                       const device float* we [[buffer(2)]], const device float* wh [[buffer(3)]],             \
                       device O* en [[buffer(4)]], device O* hn [[buffer(5)]], constant HcParams& p [[buffer(6)]], \
                       uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],      \
                       uint ntg [[threads_per_threadgroup]]) {                                                 \
    threadgroup float scratch[32];                                                                             \
    mtp4_pre_impl<O>(e, h, we, wh, en, hn, p, r, tid, ntg, scratch);                                           \
  }
NI_MTP4_PRE(mtp4_pre, float)
NI_MTP4_PRE(mtp4_pre_h, half)

// X[r, s, c] = pe[r, c] + ph[r * S + s, c] (fc_embedding of the embedding, shared fc_hidden of each stream).
// grid: (ceil(D / 256), R) x 256
[[kernel]] void mtp4_fuse(const device float* pe [[buffer(0)]], const device float* ph [[buffer(1)]],
                          device float* X [[buffer(2)]], constant HcParams& p [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.D) return;
  const float v = pe[r * p.D + c];
  for (uint s = 0; s < p.S; ++s) X[(r * p.S + s) * p.D + c] = v + ph[(r * p.S + s) * p.D + c];
}

// ---- The gated residual of one row in two dispatches (decode): hc_down, then hc_up ----
// hc_down: X' = X + inj_prev * y (has_y), the per-stream inverse RMS of X', h = silu(W_down Xn / S) with Xn = X' * inv *
// w (computed on the fly; no Xn buffer), and (the last threadgroup) X' into X2, inv, inj = 2 sigmoid(W_inj Xn / S).
// Threadgroup t < rank / 8: rows 8t..8t+7, its 8 simdgroups each a slice of the K = S*D inputs, summed in threadgroup
// memory. W_down: MLX affine BITS-bit, group 64. grid: rank / 8 + 1 threadgroups x 256 threads.
struct HcRowParams {
  uint D, S;
  uint rank;
  uint has_y, has_inj;
  float eps;
};
template <int BITS>
NI_INLINE void hc_down_impl(const device float* X, const device float* y, const device float* inj_prev,
                            const device float* w, const device uint32_t* W, const device bfloat* Ws,
                            const device bfloat* Wb, const device bfloat* winj, device float* X2, device float* h,
                            device float* inv_out, device float* inj, constant HcRowParams& p, uint tg, uint tid,
                            uint sg, uint lane, threadgroup float* scratch, threadgroup float* part) {
  const uint SD = p.S * p.D;
  float ss[4] = {0, 0, 0, 0};
  for (uint i = tid; i < SD; i += 256) {
    const uint s = i / p.D;
    const float v = X[i] + (p.has_y ? inj_prev[s] * y[i - s * p.D] : 0.0f);
    ss[s] += v * v;
  }
  float inv[4];
  for (uint s = 0; s < p.S; ++s) inv[s] = metal::precise::rsqrt(tg_sum(ss[s], scratch, tid, 256) / float(p.D) + p.eps);
  if (tg == p.rank / 8) {  // the residual, the norms, the injection weights
    float dot[4] = {0, 0, 0, 0};
    for (uint i = tid; i < SD; i += 256) {
      const uint s = i / p.D;
      const float v = X[i] + (p.has_y ? inj_prev[s] * y[i - s * p.D] : 0.0f);
      X2[i] = v;
      if (p.has_inj) {
        const float xn = v * inv[s] * w[i];
        for (uint q = 0; q < p.S; ++q) dot[q] += float(winj[q * SD + i]) * xn;
      }
    }
    for (uint q = 0; q < p.S; ++q) {
      const float d = p.has_inj ? tg_sum(dot[q], scratch, tid, 256) : 0.0f;
      if (tid == 0) {
        inv_out[q] = inv[q];
        if (p.has_inj) inj[q] = 2.0f * sigmoid(d / float(p.S));
      }
    }
    return;
  }
  constexpr uint PER = 32 / BITS, WPL = 8 / PER;  // 8 values per lane step: WPL words of a row
  const uint slice = SD / 8, k_begin = sg * slice, wpr = SD / PER, gpr = SD / 64;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint k0 = k_begin + lane * 8; k0 < k_begin + slice; k0 += 256) {
    const uint s = k0 / p.D, c0 = k0 - s * p.D;
    const float a = p.has_y ? inj_prev[s] : 0.0f, is = inv[s];
    const float4 x0 = *(const device float4*)(X + k0), x1 = *(const device float4*)(X + k0 + 4);
    const float4 y0 = *(const device float4*)(y + c0), y1 = *(const device float4*)(y + c0 + 4);
    const float4 w0 = *(const device float4*)(w + k0), w1 = *(const device float4*)(w + k0 + 4);
    const float4 n0 = (x0 + a * y0) * is * w0, n1 = (x1 + a * y1) * is * w1;
    const float xv[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
    const float xs = n0.x + n0.y + n0.z + n0.w + n1.x + n1.y + n1.z + n1.w;
    const uint g = k0 / 64;
    for (uint r = 0; r < 8; ++r) {
      const uint row = tg * 8 + r;
      float d = 0.0f;
      for (uint wi = 0; wi < WPL; ++wi) {
        const uint word = W[row * wpr + k0 / PER + wi];
        for (uint j = 0; j < PER; ++j) d += float((word >> (BITS * j)) & ((1u << BITS) - 1u)) * xv[wi * PER + j];
      }
      acc[r] += float(Ws[row * gpr + g]) * d + float(Wb[row * gpr + g]) * xs;
    }
  }
  for (uint r = 0; r < 8; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) part[sg * 8 + r] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8) {
    float t = 0.0f;
    for (uint q = 0; q < 8; ++q) t += part[q * 8 + tid];
    h[tg * 8 + tid] = silu(t / float(p.S));
  }
}
#define NI_HC_DOWN(NAME, BITS)                                                                                 \
  [[kernel]] void NAME(const device float* X [[buffer(0)]], const device float* y [[buffer(1)]],               \
                       const device float* inj_prev [[buffer(2)]], const device float* w [[buffer(3)]],        \
                       const device uint32_t* W [[buffer(4)]], const device bfloat* Ws [[buffer(5)]],          \
                       const device bfloat* Wb [[buffer(6)]], const device bfloat* winj [[buffer(7)]],         \
                       device float* X2 [[buffer(8)]], device float* h [[buffer(9)]], device float* inv [[buffer(10)]], \
                       device float* inj [[buffer(11)]], constant HcRowParams& p [[buffer(12)]],               \
                       uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],     \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    threadgroup float scratch[32];                                                                             \
    threadgroup float part[64];                                                                                \
    hc_down_impl<BITS>(X, y, inj_prev, w, W, Ws, Wb, winj, X2, h, inv, inj, p, tg, tid, sg, lane, scratch, part); \
  }
NI_HC_DOWN(hc_down_b4, 4)
NI_HC_DOWN(hc_down_b8, 8)

// hc_up: m = W_up h for the 4 rows (s*D + c) of channel c, u[c] = mean_s sigmoid(m_s) * X'[s, c] * inv[s] * w[s, c];
// X' (X2) goes back into X, and y is zeroed (has_y: hc_down has read it). A simdgroup per channel, 8 lanes per row
// (each rank / 8 inputs). W_up: MLX affine BITS-bit, group 64. grid: D / 8 threadgroups x 256 threads.
template <int BITS>
NI_INLINE void hc_up_impl(const device float* h, const device float* X2, const device float* inv, const device float* w,
                          const device uint32_t* W, const device bfloat* Ws, const device bfloat* Wb, device float* u,
                          device float* X, device float* y, constant HcRowParams& p, uint tg, uint tid, uint sg,
                          uint lane, threadgroup float* hs) {
  for (uint i = tid; i < p.rank; i += 256) hs[i] = h[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg * 8 + sg, s = lane / 8, sub = lane % 8;
  if (c >= p.D) return;
  constexpr uint PER = 32 / BITS;
  const uint row = s * p.D + c, per = p.rank / 8, k0 = sub * per, wpr = p.rank / PER, gpr = p.rank / 64;
  float m = 0.0f;
  if (s < p.S)
    for (uint k = k0; k < k0 + per; k += PER) {
      const uint word = W[row * wpr + k / PER];
      for (uint j = 0; j < PER; ++j) {
        const uint kk = k + j, g = kk / 64;
        m += (float(Ws[row * gpr + g]) * float((word >> (BITS * j)) & ((1u << BITS) - 1u)) + float(Wb[row * gpr + g])) * hs[kk];
      }
    }
  // the 8 lanes of a row, then the S rows of the channel
  m += simd_shuffle_xor(m, 1);
  m += simd_shuffle_xor(m, 2);
  m += simd_shuffle_xor(m, 4);
  float v = 0.0f;
  if (sub == 0 && s < p.S) {
    const float x = X2[row];
    v = sigmoid(m) * x * inv[s] * w[row];
    X[row] = x;
  }
  v = simd_sum(v);
  if (lane == 0) {
    u[c] = v / float(p.S);
    if (p.has_y) y[c] = 0.0f;
  }
}
#define NI_HC_UP(NAME, BITS)                                                                                   \
  [[kernel]] void NAME(const device float* h [[buffer(0)]], const device float* X2 [[buffer(1)]],              \
                       const device float* inv [[buffer(2)]], const device float* w [[buffer(3)]],             \
                       const device uint32_t* W [[buffer(4)]], const device bfloat* Ws [[buffer(5)]],          \
                       const device bfloat* Wb [[buffer(6)]], device float* u [[buffer(7)]], device float* X [[buffer(8)]], \
                       device float* y [[buffer(9)]], constant HcRowParams& p [[buffer(10)]],                  \
                       uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],     \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    threadgroup float hs[512];                                                                                 \
    hc_up_impl<BITS>(h, X2, inv, w, W, Ws, Wb, u, X, y, p, tg, tid, sg, lane, hs);                             \
  }
NI_HC_UP(hc_up_b4, 4)
NI_HC_UP(hc_up_b8, 8)

// ---- The gated residual's low-rank mix of R rows on parallel GEMVs (after hc_pre has written Xn) ----
// MLX affine BITS-bit codes (4, 6 or 8; group 64) in runs of 16 values: 16 * BITS bits, so a run starting at a
// multiple of 16 values starts on a word and stays in one 64-group.
template <int BITS>
NI_INLINE void unpack16(const device uint32_t* w, thread float* q) {
  constexpr uint NW = 16 * BITS / 32, MASK = (1u << BITS) - 1u;
  uint words[NW];
  for (uint i = 0; i < NW; ++i) words[i] = w[i];
  for (uint j = 0; j < 16; ++j) {
    const uint bit = j * BITS, wi = bit / 32, sh = bit % 32;
    uint v = words[wi] >> sh;
    if (sh + BITS > 32) v |= words[wi + 1] << (32 - sh);
    q[j] = float(v & MASK);
  }
}

// hc_down_sk: part[c, r, n] = W_down[n, chunk c] . Xn[r, chunk c] (chunks of 1024 inputs; summed by hc_up_mix).
// A simdgroup per (row n, chunk), lanes 32 inputs each (half a 64-group). grid: (N / 8, KC, R) x 256 threads.
struct HcMixParams {
  uint R, D, S;
  uint rank;      // N of W_down, K of W_up
  uint kc;        // chunks of 1024 inputs (S * D / 1024)
  uint u_ld;      // floats between u rows
};
template <int BITS>
NI_INLINE void hc_down_sk_impl(const device float* xn, const device uint32_t* W, const device bfloat* Ws,
                               const device bfloat* Wb, device float* part, constant HcMixParams& p, uint3 tg, uint sg,
                               uint lane) {
  const uint SD = p.S * p.D, n = tg.x * 8 + sg, c = tg.y, r = tg.z;
  const uint k0 = c * 1024 + lane * 32;
  const device float4* x4 = (const device float4*)(xn + r * SD + k0);
  float xv[32];
  float xs = 0.0f;
  for (uint i = 0; i < 8; ++i) {
    const float4 v = x4[i];
    xv[4 * i] = v.x, xv[4 * i + 1] = v.y, xv[4 * i + 2] = v.z, xv[4 * i + 3] = v.w;
    xs += v.x + v.y + v.z + v.w;
  }
  const device uint32_t* w = W + size_t(n) * (SD * BITS / 32) + k0 * BITS / 32;
  float q[16], d = 0.0f;
  const uint g = n * (SD / 64) + k0 / 64;
  for (uint h = 0; h < 2; ++h) {
    unpack16<BITS>(w + h * (16 * BITS / 32), q);
    for (uint j = 0; j < 16; ++j) d += q[j] * xv[h * 16 + j];
  }
  const float acc = simd_sum(float(Ws[g]) * d + float(Wb[g]) * xs);
  if (lane == 0) part[(c * p.R + r) * p.rank + n] = acc;
}
#define NI_HC_DOWN_SK(NAME, BITS)                                                                              \
  [[kernel]] void NAME(const device float* xn [[buffer(0)]], const device uint32_t* W [[buffer(1)]],           \
                       const device bfloat* Ws [[buffer(2)]], const device bfloat* Wb [[buffer(3)]],           \
                       device float* part [[buffer(4)]], constant HcMixParams& p [[buffer(5)]],                \
                       uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    hc_down_sk_impl<BITS>(xn, W, Ws, Wb, part, p, tg, sg, lane);                                               \
  }
NI_HC_DOWN_SK(hc_down_sk_b4, 4)
NI_HC_DOWN_SK(hc_down_sk_b6, 6)
NI_HC_DOWN_SK(hc_down_sk_b8, 8)

// hc_up_mix: h = silu(sum_c part[c] / S) (staged per threadgroup), m = W_up h for the S rows (s*D + ch) of a
// channel, u[ch] = mean_s sigmoid(m_s) * Xn[s*D + ch]. A simdgroup per 2 channels: 8 rows of 4 lanes, each lane a
// quarter of the rank inputs in runs of 16. grid: (D / 16, R) x 256 threads; rank % 64 == 0.
template <int BITS>
NI_INLINE void hc_up_mix_impl(const device float* part, const device float* xn, const device uint32_t* W,
                              const device bfloat* Ws, const device bfloat* Wb, device float* u, constant HcMixParams& p,
                              uint2 tg, uint tid, uint sg, uint lane, threadgroup float* hs, threadgroup float* hsum) {
  const uint r = tg.y, SD = p.S * p.D, K = p.rank;
  for (uint i = tid; i < K; i += 256) {
    float t = 0.0f;
    for (uint c = 0; c < p.kc; ++c) t += part[(c * p.R + r) * K + i];
    hs[i] = silu(t / float(p.S));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < K / 16; i += 256) {
    float t = 0.0f;
    for (uint j = 0; j < 16; ++j) t += hs[i * 16 + j];
    hsum[i] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint rl = lane / 4, qq = lane % 4;           // row of the simdgroup (channel rl / 4, stream rl % 4), quarter
  const uint ch = (tg.x * 8 + sg) * 2 + rl / 4, s = rl % 4;
  const uint row = s * p.D + ch, per = K / 4;        // inputs per lane (a multiple of 16)
  const device uint32_t* w = W + size_t(row) * (K * BITS / 32);
  float q[16], m = 0.0f;
  for (uint k = qq * per; k < (qq + 1) * per; k += 16) {
    unpack16<BITS>(w + k * BITS / 32, q);
    float d = 0.0f;
    const uint g = row * (K / 64) + k / 64;
    for (uint j = 0; j < 16; ++j) d += q[j] * hs[k + j];
    m += float(Ws[g]) * d + float(Wb[g]) * hsum[k / 16];
  }
  m += simd_shuffle_xor(m, 1);
  m += simd_shuffle_xor(m, 2);
  float v = qq == 0 ? sigmoid(m) * xn[r * SD + row] : 0.0f;
  v += simd_shuffle_xor(v, 4);
  v += simd_shuffle_xor(v, 8);
  if ((lane & 15) == 0) u[r * p.u_ld + ch] = v / float(p.S);
}
#define NI_HC_UP_MIX(NAME, BITS)                                                                               \
  [[kernel]] void NAME(const device float* part [[buffer(0)]], const device float* xn [[buffer(1)]],           \
                       const device uint32_t* W [[buffer(2)]], const device bfloat* Ws [[buffer(3)]],          \
                       const device bfloat* Wb [[buffer(4)]], device float* u [[buffer(5)]],                   \
                       constant HcMixParams& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],       \
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup float hs[1024];                                                                                \
    threadgroup float hsum[64];                                                                                \
    hc_up_mix_impl<BITS>(part, xn, W, Ws, Wb, u, p, tg, tid, sg, lane, hs, hsum);                              \
  }
NI_HC_UP_MIX(hc_up_mix_b4, 4)
NI_HC_UP_MIX(hc_up_mix_b6, 6)
NI_HC_UP_MIX(hc_up_mix_b8, 8)

// ---- The whole gated residual of R rows in two dispatches (hc_pre folded in; NI_HC_FOLD) ----
// The per-stream norm is linear in the row once its inverse RMS is known, so hc_down_sk2 runs on the unnormalized
// v = X + a_s y (times the norm weight) and keeps each stream's share of a 1024-input chunk apart (a chunk spans at
// most two streams: part2[c, r, n, j], j = stream - the chunk's first stream); the threadgroups of row block 0 also
// write v (X2), the chunk's sums of squares and its four injection dots per stream. hc_up_mix2 turns those into the
// inverse RMS, scales and sums the partials, and writes u, the injection weights, X = X2 and y = 0.
struct HcFoldParams {
  uint R, D, S;
  uint rank;      // N of W_down, K of W_up
  uint kc;        // chunks of 1024 inputs (S * D / 1024)
  uint u_ld;      // floats between u rows
  uint has_y, has_inj;
  float eps;
};
NI_INLINE float bf_dot4(const device bfloat* w, float4 v) {
  return v.x * float(w[0]) + v.y * float(w[1]) + v.z * float(w[2]) + v.w * float(w[3]);
}
// One row r of hc_down_sk2 (the partials of weight row n over chunk c; also zqwen4v.metal's multi-row form, so
// that a verify row's arithmetic is a decode step's)
// hc_down_sk2 in two parts (also zqwen4v.metal's forms, so that their arithmetic per row is the same): the lane's 32
// inputs of chunk c for row r (v = (X + a y) * w and their sum), then weight row n's partials over them
NI_INLINE void hc_down_sk2_x(const device float* X, const device float* y, const device float* inj_prev,
                             const device float* w, constant HcFoldParams& p, uint c, uint r, uint lane,
                             thread float (&xv)[32], thread float& xs) {
  const uint SD = p.S * p.D;
  const uint k0 = c * 1024 + lane * 32, s = k0 / p.D, c0 = k0 - s * p.D;
  const float a = p.has_y ? inj_prev[r * 4 + s] : 0.0f;
  const device float4* x4 = (const device float4*)(X + r * SD + k0);
  const device float4* y4 = (const device float4*)(y + r * p.D + c0);
  const device float4* w4 = (const device float4*)(w + k0);
  xs = 0.0f;
  for (uint i = 0; i < 8; ++i) {
    const float4 v = (x4[i] + a * y4[i]) * w4[i];
    xv[4 * i] = v.x, xv[4 * i + 1] = v.y, xv[4 * i + 2] = v.z, xv[4 * i + 3] = v.w;
    xs += v.x + v.y + v.z + v.w;
  }
}
template <int BITS>
NI_INLINE void hc_down_sk2_dot(const device uint32_t* W, const device bfloat* Ws, const device bfloat* Wb,
                               device float* part, constant HcFoldParams& p, uint n, uint c, uint r, uint lane,
                               thread const float (&xv)[32], float xs) {
  const uint SD = p.S * p.D;
  const uint s_lo = c * 1024 / p.D;
  const uint k0 = c * 1024 + lane * 32, s = k0 / p.D;
  const device uint32_t* wr = W + size_t(n) * (SD * BITS / 32) + k0 * BITS / 32;
  float q[16], d = 0.0f;
  const uint g = n * (SD / 64) + k0 / 64;
  for (uint h = 0; h < 2; ++h) {
    unpack16<BITS>(wr + h * (16 * BITS / 32), q);
    for (uint j = 0; j < 16; ++j) d += q[j] * xv[h * 16 + j];
  }
  const float val = float(Ws[g]) * d + float(Wb[g]) * xs;
  const float p0 = simd_sum(s == s_lo ? val : 0.0f), p1 = simd_sum(s == s_lo ? 0.0f : val);
  if (lane == 0) *(device float2*)(part + ((c * p.R + r) * p.rank + n) * 2) = float2(p0, p1);
}
// One row r of hc_down_sk2's row block 0: X2 = v, the chunk's per-stream sums of squares and injection dots
NI_INLINE void hc_down_sk2_stat(const device float* X, const device float* y, const device float* inj_prev,
                                const device float* w, const device bfloat* winj, device float* X2, device float* stat,
                                constant HcFoldParams& p, uint c, uint r, uint tid, threadgroup float4* scratch) {
  const uint SD = p.S * p.D;
  const uint s_lo = c * 1024 / p.D;
  const uint k = c * 1024 + tid * 4, s = k / p.D, c0 = k - s * p.D;
  const float a = p.has_y ? inj_prev[r * 4 + s] : 0.0f;
  const float4 v = *(const device float4*)(X + r * SD + k) + a * *(const device float4*)(y + r * p.D + c0);
  *(device float4*)(X2 + r * SD + k) = v;
  const float4 vw = v * *(const device float4*)(w + k);
  float4 dj = 0.0f;  // the four injection dots
  if (p.has_inj) {
    const device bfloat* wi = winj + k;
    dj = float4(bf_dot4(wi, vw), bf_dot4(wi + SD, vw), bf_dot4(wi + 2 * SD, vw), bf_dot4(wi + 3 * SD, vw));
  }
  const bool lo = s == s_lo;
  const float ss = metal::dot(v, v);
  const float4 d0 = tg_sum4(lo ? dj : float4(0.0f), scratch, tid, 256);
  const float4 d1 = tg_sum4(lo ? float4(0.0f) : dj, scratch, tid, 256);
  const float4 q2 = tg_sum4(float4(lo ? ss : 0.0f, lo ? 0.0f : ss, 0.0f, 0.0f), scratch, tid, 256);
  if (tid == 0) {
    device float* st = stat + (r * p.kc + c) * 10;  // ss[2], dots[2][4]
    st[0] = q2.x, st[1] = q2.y;
    *(device float4*)(st + 2) = d0;
    *(device float4*)(st + 6) = d1;
  }
}
template <int BITS>
NI_INLINE void hc_down_sk2_impl(const device float* X, const device float* y, const device float* inj_prev,
                                const device float* w, const device bfloat* winj, const device uint32_t* W,
                                const device bfloat* Ws, const device bfloat* Wb, device float* part, device float* X2,
                                device float* stat, constant HcFoldParams& p, uint3 tg, uint tid, uint sg, uint lane,
                                threadgroup float4* scratch) {
  float xv[32], xs;
  hc_down_sk2_x(X, y, inj_prev, w, p, tg.y, tg.z, lane, xv, xs);
  hc_down_sk2_dot<BITS>(W, Ws, Wb, part, p, tg.x * 8 + sg, tg.y, tg.z, lane, xv, xs);
  if (tg.x != 0) return;
  hc_down_sk2_stat(X, y, inj_prev, w, winj, X2, stat, p, tg.y, tg.z, tid, scratch);
}
#define NI_HC_DOWN_SK2(NAME, BITS)                                                                             \
  [[kernel]] void NAME(const device float* X [[buffer(0)]], const device float* y [[buffer(1)]],               \
                       const device float* inj_prev [[buffer(2)]], const device float* w [[buffer(3)]],        \
                       const device bfloat* winj [[buffer(4)]], const device uint32_t* W [[buffer(5)]],        \
                       const device bfloat* Ws [[buffer(6)]], const device bfloat* Wb [[buffer(7)]],           \
                       device float* part [[buffer(8)]], device float* X2 [[buffer(9)]],                       \
                       device float* stat [[buffer(10)]], constant HcFoldParams& p [[buffer(11)]],             \
                       uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],    \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    threadgroup float4 scratch[32];                                                                            \
    hc_down_sk2_impl<BITS>(X, y, inj_prev, w, winj, W, Ws, Wb, part, X2, stat, p, tg, tid, sg, lane, scratch); \
  }
NI_HC_DOWN_SK2(hc_down_sk2_b4, 4)
NI_HC_DOWN_SK2(hc_down_sk2_b6, 6)
NI_HC_DOWN_SK2(hc_down_sk2_b8, 8)

// hc_up_mix2 for row r in two parts (also zqwen4v.metal's multi-row form): the row's inverse RMS, injection weights
// (threadgroup 0), h and its 16-sums in threadgroup memory; then the up projection, gates and outputs of the
// threadgroup's channels
NI_INLINE void hc_up_mix2_prep(const device float* part, const device float* stat, device float* inj,
                               constant HcFoldParams& p, uint tgx, uint r, uint tid, threadgroup float* hs,
                               threadgroup float* hsum, threadgroup float* inv) {
  const uint K = p.rank;
  if (tid < 4) {  // stream tid's inverse RMS; then its share of the injection dots (threadgroup 0)
    float ss = 0.0f;
    for (uint c = 0; c < p.kc; ++c) {
      const uint s_lo = c * 1024 / p.D;
      if (s_lo == tid) ss += stat[(r * p.kc + c) * 10];
      else if (s_lo + 1 == tid) ss += stat[(r * p.kc + c) * 10 + 1];
    }
    inv[tid] = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tgx == 0 && p.has_inj && tid < 4) {  // inj[q] = 2 sigmoid(sum_s inv[s] dot_s[q] / S)
    float d = 0.0f;
    for (uint c = 0; c < p.kc; ++c) {
      const uint s_lo = c * 1024 / p.D;
      const device float* st = stat + (r * p.kc + c) * 10;
      d += inv[s_lo] * st[2 + tid] + (s_lo + 1 < p.S ? inv[s_lo + 1] * st[6 + tid] : 0.0f);
    }
    inj[r * 4 + tid] = 2.0f * sigmoid(d / float(p.S));
  }
  for (uint i = tid; i < K; i += 256) {
    float t = 0.0f;
    for (uint c = 0; c < p.kc; ++c) {
      const uint s_lo = c * 1024 / p.D;
      const float2 pp = *(const device float2*)(part + ((c * p.R + r) * K + i) * 2);
      t += inv[s_lo] * pp.x + (s_lo + 1 < p.S ? inv[s_lo + 1] * pp.y : 0.0f);
    }
    hs[i] = silu(t / float(p.S));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < K / 16; i += 256) {
    float t = 0.0f;
    for (uint j = 0; j < 16; ++j) t += hs[i * 16 + j];
    hsum[i] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
template <int BITS>
NI_INLINE void hc_up_mix2_out(const device float* X2, const device float* w, const device uint32_t* W,
                              const device bfloat* Ws, const device bfloat* Wb, device float* u, device float* X,
                              device float* y, constant HcFoldParams& p, uint tgx, uint r, uint sg, uint lane,
                              threadgroup const float* hs, threadgroup const float* hsum, threadgroup const float* inv) {
  const uint SD = p.S * p.D, K = p.rank;
  const uint rl = lane / 4, qq = lane % 4;
  const uint ch = (tgx * 8 + sg) * 2 + rl / 4, s = rl % 4;
  const uint row = s * p.D + ch, per = K / 4;
  const device uint32_t* wr = W + size_t(row) * (K * BITS / 32);
  float q[16], m = 0.0f;
  for (uint k = qq * per; k < (qq + 1) * per; k += 16) {
    unpack16<BITS>(wr + k * BITS / 32, q);
    float d = 0.0f;
    const uint g = row * (K / 64) + k / 64;
    // fma spelled out so the multi-row form (zqwen4v.metal hc_up_mix4) compiles to the same order under fast math
    for (uint j = 0; j < 16; ++j) d = fma(q[j], hs[k + j], d);
    m = fma(float(Wb[g]), hsum[k / 16], fma(float(Ws[g]), d, m));
  }
  m += simd_shuffle_xor(m, 1);
  m += simd_shuffle_xor(m, 2);
  float v = 0.0f;
  if (qq == 0) {
    const float x = X2[r * SD + row];
    v = sigmoid(m) * x * inv[s] * w[row];
    X[r * SD + row] = x;
  }
  v += simd_shuffle_xor(v, 4);
  v += simd_shuffle_xor(v, 8);
  if ((lane & 15) == 0) {
    u[r * p.u_ld + ch] = v / float(p.S);
    if (p.has_y) y[r * p.D + ch] = 0.0f;
  }
}
template <int BITS>
NI_INLINE void hc_up_mix2_impl(const device float* part, const device float* X2, const device float* stat,
                               const device float* w, const device uint32_t* W, const device bfloat* Ws,
                               const device bfloat* Wb, device float* u, device float* X, device float* y,
                               device float* inj, constant HcFoldParams& p, uint2 tg, uint tid, uint sg, uint lane,
                               threadgroup float* hs, threadgroup float* hsum, threadgroup float* inv) {
  hc_up_mix2_prep(part, stat, inj, p, tg.x, tg.y, tid, hs, hsum, inv);
  hc_up_mix2_out<BITS>(X2, w, W, Ws, Wb, u, X, y, p, tg.x, tg.y, sg, lane, hs, hsum, inv);
}
#define NI_HC_UP_MIX2(NAME, BITS)                                                                              \
  [[kernel]] void NAME(const device float* part [[buffer(0)]], const device float* X2 [[buffer(1)]],           \
                       const device float* stat [[buffer(2)]], const device float* w [[buffer(3)]],            \
                       const device uint32_t* W [[buffer(4)]], const device bfloat* Ws [[buffer(5)]],          \
                       const device bfloat* Wb [[buffer(6)]], device float* u [[buffer(7)]],                   \
                       device float* X [[buffer(8)]], device float* y [[buffer(9)]], device float* inj [[buffer(10)]], \
                       constant HcFoldParams& p [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],     \
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup float hs[1024];                                                                                \
    threadgroup float hsum[64];                                                                                \
    threadgroup float inv[4];                                                                                  \
    hc_up_mix2_impl<BITS>(part, X2, stat, w, W, Ws, Wb, u, X, y, inj, p, tg, tid, sg, lane, hs, hsum, inv);    \
  }
NI_HC_UP_MIX2(hc_up_mix2_b4, 4)
NI_HC_UP_MIX2(hc_up_mix2_b6, 6)
NI_HC_UP_MIX2(hc_up_mix2_b8, 8)

// ---- Router top-k for 512 experts (Qwen3.8-Flash-Next), one row per simdgroup ----
// The k largest logits are the k largest probabilities: selected on the logits (16 per lane, no exponentials), then
// weighted by softmax over the selected (norm: the top-k renormalization; else over all 512). wts[k] = sigmoid of the
// shared-expert gate. Rows: idx [R, k], wts [R, k + 1]. grid: R x 32 threads.
[[kernel]] void moe_topk512(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                            device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                            constant TopkParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const device float4* l4 = (const device float4*)(logits + r * 512 + lane * 16);
  const float4 a = l4[0], b = l4[1], c = l4[2], d = l4[3];
  float v[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
  float mx = v[0];
  for (uint i = 1; i < 16; ++i) mx = max(mx, v[i]);
  mx = simd_max(mx);
  float total = 0.0f;  // the full softmax denominator (no renormalization only)
  if (!p.norm) {
    for (uint i = 0; i < 16; ++i) total += metal::precise::exp(v[i] - mx);
    total = simd_sum(total);
  }
  uint taken = 0;
  float sel = 0.0f, my_w = 0.0f;
  uint my_i = 0;
  for (uint q = 0; q < p.k; ++q) {
    float best = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = 0; i < 16; ++i)
      if (!(taken & (1u << i)) && v[i] > best) {
        best = v[i];
        bi = lane * 16 + i;
      }
    const float g = simd_max(best);
    const uint win = simd_min(best == g ? bi : 0xffffffffu);
    if (win / 16 == lane) taken |= 1u << (win % 16);
    const float e = metal::precise::exp(g - mx);
    sel += e;
    if (lane == q) my_w = e, my_i = win;
  }
  if (lane < p.k) {
    idx[r * p.k + lane] = my_i;
    wts[r * (p.k + 1) + lane] = my_w / (p.norm ? sel : total);
  }
  if (lane == p.k) wts[r * (p.k + 1) + p.k] = sigmoid(shared[r]);
}
