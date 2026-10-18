#include "common.h"

// ---- Qwen3.8-Flash-Next (qwen4_exp): hyper-connections, the n-gram embedding (PLE), the sparse-attention indexer ----
// The residual stream is S = 4 streams of D = 2560 ([s0 | s1 | s2 | s3], fp32). Norm weights are the checkpoint's
// zero-centred w applied as 1 + w; the model passes them as fp32 (1 + w) vectors.

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
template <typename XN>
NI_INLINE void hc_pre_impl(device float* X, device float* y, const device float* inj_prev, const device float* w,
                           const device bfloat* winj, device XN* xn, device float* inj, constant HcParams& p, uint r,
                           uint tid, uint ntg, threadgroup float* scratch) {
  const uint SD = p.S * p.D;
  device float* x = X + r * SD;
  device float* yr = y + r * p.D;
  float ss[4] = {0, 0, 0, 0};
  for (uint i = tid; i < SD; i += ntg) {
    const uint s = i / p.D;
    float v = x[i];
    if (p.has_y) {
      v += inj_prev[r * p.S + s] * yr[i - s * p.D];
      x[i] = v;
    }
    ss[s] += v * v;
  }
  float inv[4];
  for (uint s = 0; s < p.S; ++s) inv[s] = metal::precise::rsqrt(tg_sum(ss[s], scratch, tid, ntg) / float(p.D) + p.eps);
  if (p.has_y) {  // every thread's reads of y are done (tg_sum's barriers)
    for (uint i = tid; i < p.D; i += ntg) yr[i] = 0.0f;
  }
  float dot[4] = {0, 0, 0, 0};
  for (uint i = tid; i < SD; i += ntg) {
    const uint s = i / p.D;
    const float v = x[i] * inv[s] * w[i];
    xn[r * SD + i] = XN(v);
    if (p.has_inj)
      for (uint q = 0; q < p.S; ++q) dot[q] += float(winj[q * SD + i]) * v;
  }
  if (p.has_inj)
    for (uint q = 0; q < p.S; ++q) {
      const float d = tg_sum(dot[q], scratch, tid, ntg);
      if (tid == 0) inj[r * p.S + q] = 2.0f * sigmoid(d / float(p.S));
    }
}

#define NI_HC_PRE(NAME, XN)                                                                                    \
  [[kernel]] void NAME(device float* X [[buffer(0)]], device float* y [[buffer(1)]],                           \
                       const device float* inj_prev [[buffer(2)]], const device float* w [[buffer(3)]],        \
                       const device bfloat* winj [[buffer(4)]], device XN* xn [[buffer(5)]],                   \
                       device float* inj [[buffer(6)]], constant HcParams& p [[buffer(7)]],                    \
                       uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],      \
                       uint ntg [[threads_per_threadgroup]]) {                                                 \
    threadgroup float scratch[32];                                                                             \
    hc_pre_impl<XN>(X, y, inj_prev, w, winj, xn, inj, p, r, tid, ntg, scratch);                                \
  }
NI_HC_PRE(hc_pre, float)
NI_HC_PRE(hc_pre_h, half)

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
// row's position; the raw index key goes to the cache ik [pos, HD] (half). Block j (tokens 4j..4j+3) gets its key
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

NI_INLINE float qsa_rope(threadgroup float* row, float x, uint tid, uint rot, uint pos, float theta) {
  const uint hr = rot / 2;
  if (tid >= rot) return x;
  const uint i = tid % hr;
  const float ang = float(pos) * metal::precise::pow(theta, -2.0f * float(i) / float(rot));
  const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
  return tid < hr ? row[tid] * cs - row[tid + hr] * sn : row[tid] * cs + row[tid - hr] * sn;
}

// grid: (NH + 1, R) x HD threads
[[kernel]] void qsa_prep(const device float* iqk [[buffer(0)]], const device float* wq [[buffer(1)]],
                         device float* iq [[buffer(2)]], device half* ik [[buffer(3)]],
                         constant QsaParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
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
  iq[(r * p.NH + h) * p.HD + tid] = qsa_rope(row, row[tid], tid, p.rot, pos, p.theta);
}

// grid: R x HD threads (rows that complete a block write its key)
[[kernel]] void qsa_block(const device half* ik [[buffer(0)]], const device float* wk [[buffer(1)]],
                          device float* bk [[buffer(2)]], constant QsaParams& p [[buffer(3)]],
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
  bk[j * p.HD + tid] = qsa_rope(row, row[tid], tid, p.rot, t0, p.theta);
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

// The key positions row r attends to: the topb best-scoring blocks (ties: the lowest block indices) as 4 positions
// each, then the incomplete block's 0..ratio-1 positions; every position 0..pos when nb <= topb (dense). idx [R, i_ld],
// cnt [R]; positions in ascending block order. One threadgroup of 1024 threads per row: an MSB-first radix select of
// the topb-th largest score (scores are >= 0, so their bits order like the values), then thread t keeps blocks of its
// contiguous range [t * per, (t + 1) * per) above the threshold and the lowest-index ones equal to it, and writes them
// at offsets from a scan of the per-thread counts.
[[kernel]] void qsa_select(const device float* score [[buffer(0)]], device uint* idx [[buffer(1)]],
                           device uint* cnt [[buffer(2)]], constant QsaSelParams& p [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint scan[1024];
  threadgroup uint sel_digit, sel_rank;
  const uint pos = p.pos0 + r, n = pos + 1, nb = n / p.ratio;
  device uint* out = idx + r * p.i_ld;
  if (nb <= p.topb) {
    for (uint i = tid; i < n; i += ntg) out[i] = i;
    if (tid == 0) cnt[r] = n;
    return;
  }
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
          sel_rank = rank - acc;
          break;
        }
        acc += c;
      }
      sel_digit = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= sel_digit << shift;
    pmask |= 255u << shift;
    rank = sel_rank;
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
  for (uint j = j0; j < j1; ++j) {
    const uint k = key[j];
    const bool take = k > prefix || (k == prefix && e_left > 0);
    if (k == prefix && take) --e_left;
    if (!take) continue;
    for (uint t = 0; t < p.ratio; ++t) out[o + t] = j * p.ratio + t;
    o += p.ratio;
  }
  const uint tail0 = nb * p.ratio, base = p.topb * p.ratio;
  for (uint i = tid; tail0 + i < n; i += ntg) out[base + i] = tail0 + i;
  if (tid == 0) cnt[r] = base + (n - tail0);
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
