#include "common.h"

// ---- Mixture of experts (Qwen3.5/3.6 MoE): softmax router, top-k, SwiGLU experts + gated shared expert ----

struct TopkParams {
  uint E;     // experts (multiple of 32, <= 512)
  uint k;     // active experts
  uint norm;  // renormalize top-k probabilities
};

// One simdgroup. logits: [E] router logits, shared: [1] shared-expert gate logit.
// Writes idx[k] and wts[k+1] (wts[k] = sigmoid(shared gate)).
[[kernel]] void moe_topk(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                         device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                         constant TopkParams& p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint MAXPER = 16;
  const uint per = p.E / 32;
  float v[MAXPER];
  float mx = -INFINITY;
  for (uint i = 0; i < per; ++i) {
    v[i] = logits[lane * per + i];
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  for (uint i = 0; i < per; ++i) {
    v[i] = metal::precise::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  for (uint i = 0; i < per; ++i) v[i] /= sum;

  uint taken = 0;
  float sel_sum = 0.0f;
  float my_w[16];
  uint my_i[16];
  for (uint r = 0; r < p.k; ++r) {
    float best = -1.0f;
    uint bi = 0xffffffff;
    for (uint i = 0; i < per; ++i)
      if (!(taken & (1u << i)) && v[i] > best) {
        best = v[i];
        bi = lane * per + i;
      }
    float gbest = simd_max(best);
    uint cand = best == gbest ? bi : 0xffffffff;
    uint win = simd_min(cand);
    if (win / per == lane && win != 0xffffffff) taken |= 1u << (win % per);
    my_w[r] = gbest;
    my_i[r] = win;
    sel_sum += gbest;
  }
  if (lane == 0) {
    for (uint r = 0; r < p.k; ++r) {
      idx[r] = my_i[r];
      wts[r] = p.norm ? my_w[r] / sel_sum : my_w[r];
    }
    wts[p.k] = sigmoid(shared[0]);
  }
}


struct MoeParams {
  uint D;    // hidden size (input of gate/up, output of down)
  uint I;    // expert intermediate size
  uint SI;   // shared expert intermediate size
  uint k;    // active experts
  float eps; // fused input RMSNorm
};

// grid: x = row tiles over max(I, SI) (ROWS*simdgroups rows each), y = slot (0..k-1 routed, k shared).
// h[slot, row] = silu(gate_row . xn) * (up_row . xn), xn = rmsnorm(x) * nw.
[[kernel]] void moe_gate_up(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],
                            const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],
                            const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],
                            const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],
                            const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],
                            const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],
                            const device float* x [[buffer(12)]], const device bfloat* nw [[buffer(13)]],
                            const device uint* idx [[buffer(14)]], device float* h [[buffer(15)]],
                            constant MoeParams& p [[buffer(16)]], uint2 tgp [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPL = 16, GS = 64;
  const uint slot = tgp.y;
  const bool shared = slot == p.k;
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / GS;
  const device uint32_t *G = gw, *U = uw;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    uint e = idx[slot];
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  NormCtx<1> nc = rms_ctx<1>(x, K, K, p.eps, lane);
  float ag[ROWS][1], au[ROWS][1];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    ag[r][0] = au[r][0] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    XChunk<4, VPL, 1> c;
    load_x<4, VPL, 1, true>(c, x, K, k0, nc, nw);
    const uint g = k0 / GS;
    for (int r = 0; r < ROWS; ++r) {
      uint ro = rows[r];
      dot_row<4, VPL, 1>(c, G + ro * wpr + k0 / 8, float(Gs[ro * gpr + g]), float(Gb[ro * gpr + g]), ag[r]);
      dot_row<4, VPL, 1>(c, U + ro * wpr + k0 / 8, float(Us[ro * gpr + g]), float(Ub[ro * gpr + g]), au[r]);
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float gv = simd_sum(ag[r][0]), uv = simd_sum(au[r][0]);
    if (lane == 0 && row0 + r < N) h[slot * max(p.I, p.SI) + row0 + r] = silu(gv) * uv;
  }
}

// grid: x = row tiles over D. x[row] += sum_slot wts[slot] * (down_slot[row] . h[slot]).
// Expert intermediate sizes must be a multiple of 512 (one 16-values-per-lane step).
[[kernel]] void moe_down(const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],
                         const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],
                         const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],
                         const device float* h [[buffer(6)]], const device uint* idx [[buffer(7)]],
                         const device float* wts [[buffer(8)]], device float* x [[buffer(9)]],
                         constant MoeParams& p [[buffer(10)]], uint tgp [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPL = 16, GS = 64;
  const uint row0 = (tgp * nsg + sg) * ROWS;
  if (row0 >= p.D) return;
  const uint hstride = max(p.I, p.SI);
  NormCtx<1> nc;
  float acc[ROWS];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, p.D - 1);
  }
  for (uint slot = 0; slot <= p.k; ++slot) {
    const bool shared = slot == p.k;
    const uint K = shared ? p.SI : p.I;
    const uint wpr = K / 8, gpr = K / GS;
    const device uint32_t* W = shared ? sdw : dw + idx[slot] * p.D * wpr;
    const device bfloat* S = shared ? sds : dsc + idx[slot] * p.D * gpr;
    const device bfloat* B = shared ? sdb : dbi + idx[slot] * p.D * gpr;
    const float ws = wts[slot];
    for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
      XChunk<4, VPL, 1> c;
      load_x<4, VPL, 1, false>(c, h + slot * hstride, K, k0, nc, nullptr);
      const uint g = k0 / GS;
      for (int r = 0; r < ROWS; ++r) {
        float a[1] = {0.0f};
        dot_row<4, VPL, 1>(c, W + rows[r] * wpr + k0 / 8, float(S[rows[r] * gpr + g]), float(B[rows[r] * gpr + g]), a);
        acc[r] += ws * a[0];
      }
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    if (lane == 0 && row0 + r < p.D) x[row0 + r] += v;
  }
}

// ---- Prefill MoE helpers ----

// Router top-k for T tokens: one simdgroup per token (grid x = T, 32 threads).
[[kernel]] void moe_topk_rows(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                              device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                              constant TopkParams& p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint per = p.E / 32;
  float v[16];
  float mx = -INFINITY;
  for (uint i = 0; i < per; ++i) {
    v[i] = logits[t * p.E + lane * per + i];
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  for (uint i = 0; i < per; ++i) {
    v[i] = metal::precise::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  for (uint i = 0; i < per; ++i) v[i] /= sum;
  uint taken = 0;
  float sel_sum = 0.0f;
  float my_w[16];
  uint my_i[16];
  for (uint r = 0; r < p.k; ++r) {
    float best = -1.0f;
    uint bi = 0xffffffff;
    for (uint i = 0; i < per; ++i)
      if (!(taken & (1u << i)) && v[i] > best) { best = v[i]; bi = lane * per + i; }
    float gbest = simd_max(best);
    uint win = simd_min(best == gbest ? bi : 0xffffffff);
    if (win != 0xffffffff && win / per == lane) taken |= 1u << (win % per);
    my_w[r] = gbest;
    my_i[r] = win;
    sel_sum += gbest;
  }
  if (lane == 0) {
    for (uint r = 0; r < p.k; ++r) {
      idx[t * p.k + r] = my_i[r];
      wts[t * (p.k + 1) + r] = p.norm ? my_w[r] / sel_sum : my_w[r];
    }
    wts[t * (p.k + 1) + p.k] = sigmoid(shared[t]);
  }
}

// moe_topk_rows for E = 256 (8 logits per lane) with the loops unrolled and 8 tokens per 256-thread
// threadgroup (one simdgroup each) instead of a 32-thread threadgroup per token: the same operations
// per token in the same order (NI_TOPK8). grid: ceil(T / 8) x 256 threads.
[[kernel]] void moe_topk_rows8(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                               device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                               constant TopkParams& p [[buffer(4)]], constant uint& T [[buffer(5)]],
                               uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint per = 8;
  const uint t = tg * 8 + sg;
  if (t >= T) return;
  float v[per];
  float mx = -INFINITY;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < per; ++i) {
    v[i] = logits[t * 256 + lane * per + i];
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < per; ++i) {
    v[i] = metal::precise::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < per; ++i) v[i] /= sum;
  uint taken = 0;
  float sel_sum = 0.0f;
  float my_w[16];
  uint my_i[16];
  for (uint r = 0; r < p.k; ++r) {
    float best = -1.0f;
    uint bi = 0xffffffff;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < per; ++i)
      if (!(taken & (1u << i)) && v[i] > best) { best = v[i]; bi = lane * per + i; }
    float gbest = simd_max(best);
    uint win = simd_min(best == gbest ? bi : 0xffffffff);
    if (win != 0xffffffff && win / per == lane) taken |= 1u << (win % per);
    my_w[r] = gbest;
    my_i[r] = win;
    sel_sum += gbest;
  }
  if (lane == 0) {
    for (uint r = 0; r < p.k; ++r) {
      idx[t * p.k + r] = my_i[r];
      wts[t * (p.k + 1) + r] = p.norm ? my_w[r] / sel_sum : my_w[r];
    }
    wts[t * (p.k + 1) + p.k] = sigmoid(shared[t]);
  }
}

// Groups the T*k (token, slot) pairs by expert. One threadgroup of 1024 threads.
// Outputs: sorted_pair[P] (pair ids grouped by expert), sorted_tok[P] (= pair / k),
// tiles[3*i] = (expert, begin, end) row ranges of at most BM sorted rows, ntiles[0].
struct BucketParams {
  uint P;   // pairs = T * k
  uint k;
  uint E;
  uint BM;
};

[[kernel]] void moe_bucket(const device uint* idx [[buffer(0)]], device uint* sorted_pair [[buffer(1)]],
                           device uint* sorted_tok [[buffer(2)]], device uint* tiles [[buffer(3)]],
                           device uint* ntiles [[buffer(4)]], constant BucketParams& p [[buffer(5)]],
                           uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[512];
  threadgroup uint off[513];
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) atomic_fetch_add_explicit(&cnt[idx[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0, nt = 0;
    for (uint e = 0; e < p.E; ++e) {
      uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      off[e] = acc;
      for (uint r = 0; r < c; r += p.BM) {
        tiles[nt * 3 + 0] = e;
        tiles[nt * 3 + 1] = acc + r;
        tiles[nt * 3 + 2] = acc + min(r + p.BM, c);
        ++nt;
      }
      acc += c;
    }
    ntiles[0] = nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) {
    uint e = idx[i];
    uint pos = off[e] + atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
    sorted_pair[pos] = i;
    sorted_tok[pos] = i / p.k;
  }
}

// h[row] = silu(g[row]) * u[row] as half (rows of width I). grid: rows x I/4 threads.
[[kernel]] void swiglu_half(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                            device half* h [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  h[i] = half(silu(g[i]) * u[i]);
}

// x[t, :] += sum_s wts[t, s] * y[t*k + s, :] + wts[t, k] * ys[t, :]. grid: (D/4, T)
struct CombineParams {
  uint D;
  uint k;
};
[[kernel]] void moe_combine(const device float* y [[buffer(0)]], const device float* ys [[buffer(1)]],
                            const device float* wts [[buffer(2)]], device float* x [[buffer(3)]],
                            constant CombineParams& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  uint d4 = gid.x, t = gid.y;
  if (d4 >= p.D / 4) return;
  float4 acc = 0.0f;
  for (uint s = 0; s < p.k; ++s) acc += wts[t * (p.k + 1) + s] * ((const device float4*)(y + (t * p.k + s) * p.D))[d4];
  acc += wts[t * (p.k + 1) + p.k] * ((const device float4*)(ys + t * p.D))[d4];
  ((device float4*)(x + t * p.D))[d4] += acc;
}

// ---- v2 MoE decode: top-k fused into gate/up, x staged once per threadgroup ----
// grid: x = row tiles of (nsg * ROWS) rows over max(I, SI), y = slot (0..k-1 routed, k = shared).
// Every threadgroup re-derives the routing from the router logits (256 floats; cheap) so no
// separate top-k dispatch is needed; threadgroup (0,0) publishes idx/wts for moe_down2.
NI_INLINE void route_topk(const device float* logits, const device float* shared, uint E, uint k, bool norm,
                          threadgroup uint* tidx, threadgroup float* twts, uint lane) {
  const uint per = E / 32;
  float v[16];
  float mx = -INFINITY;
  for (uint i = 0; i < per; ++i) {
    v[i] = logits[lane * per + i];
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  for (uint i = 0; i < per; ++i) {
    v[i] = metal::precise::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  uint taken = 0;
  float sel = 0.0f;
  for (uint r = 0; r < k; ++r) {
    float best = -1.0f;
    uint bi = 0xffffffff;
    for (uint i = 0; i < per; ++i)
      if (!(taken & (1u << i)) && v[i] > best) { best = v[i]; bi = lane * per + i; }
    float gbest = simd_max(best);
    uint win = simd_min(best == gbest ? bi : 0xffffffff);
    if (win != 0xffffffff && win / per == lane) taken |= 1u << (win % per);
    if (lane == 0) { tidx[r] = win; twts[r] = gbest / sum; }
    sel += gbest / sum;
  }
  if (lane == 0) {
    if (norm)
      for (uint r = 0; r < k; ++r) twts[r] /= sel;
    twts[k] = sigmoid(shared[0]);
  }
}

[[kernel]] void moe_gate_up2(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],
                             const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],
                             const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],
                             const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],
                             const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],
                             const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],
                             const device float* xn [[buffer(12)]], const device float* logits [[buffer(13)]],
                             const device float* shared_logit [[buffer(14)]], device float* h [[buffer(15)]],
                             constant MoeParams& p [[buffer(16)]], device uint* idx_out [[buffer(17)]],
                             device float* wts_out [[buffer(18)]], constant uint& E [[buffer(19)]],
                             uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint2 ntg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 2;
  const uint ntg = ntg2.x;
  threadgroup float xs[4096];
  threadgroup uint tidx[16];
  threadgroup float twts[17];
  const uint slot = tgp.y;
  const bool shared = slot == p.k;
  if (sg == 0) route_topk(logits, shared_logit, E, p.k, true, tidx, twts, lane);
  for (uint i = tid; i < p.D / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)xn)[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tgp.x == 0 && tgp.y == 0 && tid <= p.k) {
    if (tid < p.k) idx_out[tid] = tidx[tid];
    wts_out[tid] = twts[tid];
  }
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / 64;
  const device uint32_t *G = gw, *U = uw;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    uint e = tidx[slot];
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  float ag[ROWS], au[ROWS];
  qmv2_rows<4, 16, ROWS, 64>(G, Gs, Gb, xs, N, K, row0, lane, ag);
  qmv2_rows<4, 16, ROWS, 64>(U, Us, Ub, xs, N, K, row0, lane, au);
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < N) h[slot * max(p.I, p.SI) + row0 + r] = silu(ag[r]) * au[r];
}

template <int NI_GU_ROWS>
NI_INLINE void moe_gate_up3_impl(const device uint32_t* gw, const device bfloat* gsc,
                             const device bfloat* gbi, const device uint32_t* uw,
                             const device bfloat* usc, const device bfloat* ubi,
                             const device uint32_t* sgw, const device bfloat* sgs,
                             const device bfloat* sgb, const device uint32_t* suw,
                             const device bfloat* sus, const device bfloat* sub,
                             const device float* xn, const device float* logits,
                             const device float* shared_logit, device float* h,
                             constant MoeParams& p, const device uint* idx_out,
                             const device float* wts_out, constant uint& E,
                             uint2 tgp, uint tid,
                             uint2 ntg2, uint sg,
                             uint nsg, uint lane, threadgroup float* xs, threadgroup uint* tidx, threadgroup float* twts) {
  constexpr int ROWS = NI_GU_ROWS;
  const uint ntg = ntg2.x;
  const uint slot = tgp.y;
  const bool shared = slot == p.k;
  if (tid <= p.k) {
    if (tid < p.k) tidx[tid] = idx_out[tid];  // precomputed by moe_topk
    twts[tid] = wts_out[tid];
  }
  for (uint i = tid; i < p.D / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)xn)[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / 64;
  const device uint32_t *G = gw, *U = uw;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    uint e = tidx[slot];
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  float ag[ROWS], au[ROWS];
  qmv2_rows<4, 16, ROWS, 64>(G, Gs, Gb, xs, N, K, row0, lane, ag);
  qmv2_rows<4, 16, ROWS, 64>(U, Us, Ub, xs, N, K, row0, lane, au);
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < N) h[slot * max(p.I, p.SI) + row0 + r] = silu(ag[r]) * au[r];
}


#define NI_GU3(NAME, R)                                                                                    \
  [[kernel]] void NAME(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],      \
                       const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],      \
                       const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],       \
                       const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],     \
                       const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],     \
                       const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],     \
                       const device float* xn [[buffer(12)]], const device float* logits [[buffer(13)]],     \
                       const device float* shared_logit [[buffer(14)]], device float* h [[buffer(15)]],      \
                       constant MoeParams& p [[buffer(16)]], const device uint* idx_out [[buffer(17)]],      \
                       const device float* wts_out [[buffer(18)]], constant uint& E [[buffer(19)]],          \
                       uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint2 ntg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {    \
    threadgroup float xs[4096];                                                                             \
    threadgroup uint tidx[16];                                                                              \
    threadgroup float twts[17];                                                                             \
    moe_gate_up3_impl<R>(gw, gsc, gbi, uw, usc, ubi, sgw, sgs, sgb, suw, sus, sub, xn, logits, shared_logit, h, p, \
                         idx_out, wts_out, E, tgp, tid, ntg2, sg, nsg, lane, xs, tidx, twts);               \
  }
NI_GU3(moe_gate_up3_r2, 2)
NI_GU3(moe_gate_up3_r4, 4)

// grid: row tiles over D. x[row] += sum_slot wts[slot] * (down_slot[row] . h[slot]); h staged once.
// moe_down2 stages (k + 1) * max(I, SI) <= 4608 floats, moe_down2_w (wide experts: Qwen3.8-Flash-Next's 11 x 640)
// <= 7168 (checked on the host); the same arithmetic.
NI_INLINE void moe_down2_impl(const device uint32_t* dw, const device bfloat* dsc, const device bfloat* dbi,
                              const device uint32_t* sdw, const device bfloat* sds, const device bfloat* sdb,
                              const device float* h, const device uint* idx, const device float* wts, device float* x,
                              constant MoeParams& p, uint tgp, uint tid, uint ntg, uint sg, uint nsg, uint lane,
                              threadgroup float* hs) {
  constexpr int ROWS = 2;
  const uint hstride = max(p.I, p.SI);
  for (uint i = tid; i < (p.k + 1) * hstride / 4; i += ntg) ((threadgroup float4*)hs)[i] = ((const device float4*)h)[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = (tgp * nsg + sg) * ROWS;
  if (row0 >= p.D) return;
  float tot[ROWS] = {0.0f, 0.0f};
  for (uint slot = 0; slot <= p.k; ++slot) {
    const bool shared = slot == p.k;
    const uint K = shared ? p.SI : p.I;
    const uint wpr = K / 8, gpr = K / 64;
    const device uint32_t* W = shared ? sdw : dw + idx[slot] * p.D * wpr;
    const device bfloat* S = shared ? sds : dsc + idx[slot] * p.D * gpr;
    const device bfloat* B = shared ? sdb : dbi + idx[slot] * p.D * gpr;
    float a[ROWS];
    qmv2_rows<4, 16, ROWS, 64>(W, S, B, hs + slot * hstride, p.D, K, row0, lane, a);
    const float ws = wts[slot];
    for (int r = 0; r < ROWS; ++r) tot[r] += ws * a[r];
  }
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < p.D) x[row0 + r] += tot[r];
}

#define NI_MOE_DOWN2(NAME, HS)                                                                                 \
  [[kernel]] void NAME(const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],       \
                       const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],      \
                       const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],        \
                       const device float* h [[buffer(6)]], const device uint* idx [[buffer(7)]],             \
                       const device float* wts [[buffer(8)]], device float* x [[buffer(9)]],                  \
                       constant MoeParams& p [[buffer(10)]], uint tgp [[threadgroup_position_in_grid]],       \
                       uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],        \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                             \
    threadgroup float hs[HS];                                                                                 \
    moe_down2_impl(dw, dsc, dbi, sdw, sds, sdb, h, idx, wts, x, p, tgp, tid, ntg, sg, nsg, lane, hs);         \
  }
NI_MOE_DOWN2(moe_down2, 9 * 512)
NI_MOE_DOWN2(moe_down2_w, 7168)

// Faster single-simdgroup top-k for E = 256, k <= 8: compile-time sizes, fast exp, one pass of
// simd reductions per round.
[[kernel]] void moe_topk256(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                            device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                            constant TopkParams& p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER = 8;
  float4 a = ((const device float4*)logits)[lane * 2], b = ((const device float4*)logits)[lane * 2 + 1];
  float v[PER] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float mx = v[0];
  for (uint i = 1; i < PER; ++i) mx = max(mx, v[i]);
  mx = simd_max(mx);
  float sum = 0.0f;
  for (uint i = 0; i < PER; ++i) {
    v[i] = metal::fast::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  float sel = 0.0f, wsel[8];
  uint isel[8];
  for (uint r = 0; r < p.k; ++r) {
    float best = -1.0f;
    uint bi = 0;
    for (uint i = 0; i < PER; ++i)
      if (v[i] > best) { best = v[i]; bi = i; }
    float g = simd_max(best);
    uint win = simd_min(best == g ? lane * PER + bi : 0xffffffffu);
    if (win / PER == lane) v[win % PER] = -1.0f;
    wsel[r] = g;
    isel[r] = win;
    sel += g;
  }
  if (lane < p.k) {
    float w = 0.0f;
    uint ii = 0;
    for (uint r = 0; r < p.k; ++r)
      if (r == lane) { w = wsel[r]; ii = isel[r]; }
    idx[lane] = ii;
    wts[lane] = p.norm ? w / sel : w / sum;
  }
  if (lane == p.k) wts[p.k] = sigmoid(shared[0]);
}
