// DSpark block drafter (Nemotron 3.5 Lightning): the block attention, causal within the block; the mask rows of the
// block; the Markov head's greedy walk over each slot's candidates; the fused passes of its MLP and commit. The rest of
// the pass runs on LiLiCorr's kernels (nh_lili.metal: capture, candidates, dynamic depth, acceptance).

// ---- block attention: 8 query rows at positions P..P+7 over the ring's context [lo, P) and the block's rows up to
// their own (causal), sliding window W, per-head sink logits. q, k, v are the block's raw q|k|v product rows (qkv
// [B][ld]: q heads at column 0, k heads at Hq HD, v heads at (Hq + Hkv) HD); q and the block's k get their per-head
// RMSNorm (qn, kn) and NeoX RoPE here. The context is split in NS pieces (one threadgroup each per (kv head, row):
// nh_dsp_attn_part writes each piece's softmax state), nh_dsp_attn_fin combines them with the block's keys and the
// sink and writes the rows in half (the o product's input). One simdgroup per query head (G = Hq / Hkv per
// threadgroup), lanes hold HD / 32 = 4 dims. (One threadgroup per (kv head, row) over the whole context before: 16
// threadgroups, ~37 us per layer in the pipeline on Lightning, M5 Max.) ----
struct NhDspAttnP {
  uint P, lo, W, Hq, Hkv, B, NS, ld;
  float scale, eps, theta;
};
// the block's RoPE table: tab[r][i] = (cos, sin) of (P + r) theta^(-2i / HD), i < HD / 2 (the values nh_qk_norm_rope
// computes per element). grid: 1 x (B HD / 2) threads
[[kernel]] void nh_dsp_rope_tab(device float2* tab [[buffer(0)]], constant NhDspAttnP& p [[buffer(1)]],
                                uint t [[thread_position_in_grid]]) {
  constexpr uint HD = 128;
  if (t >= p.B * HD / 2) return;
  const uint r = t / (HD / 2), i = t % (HD / 2);
  const float inv = metal::precise::pow(p.theta, -2.0f * float(i) / float(HD));
  const float ang = float(p.P + r) * inv;
  tab[t] = float2(metal::precise::cos(ang), metal::precise::sin(ang));
}
// a head's 4 dims of lane: RMSNorm over the head (w: the norm weight), NeoX RoPE with the row's table cs (dims d and
// d + 64 are lanes l and l ^ 16)
NI_INLINE float4 nh_dsp_norm_rope(float4 x, const device float* w, const device float2* cs, float eps, uint lane) {
  constexpr uint HD = 128;
  const float ss = simd_sum(dot(x, x));
  x = x * (1.0f / metal::precise::sqrt(ss / float(HD) + eps)) * ((const device float4*)w)[lane];
  const float4 o = simd_shuffle_xor(x, 16);
  float4 y;
  for (uint c = 0; c < 4; ++c) {
    const uint d = lane * 4 + c;
    const float2 t = cs[d % (HD / 2)];
    y[c] = d < HD / 2 ? x[c] * t.x - o[c] * t.y : x[c] * t.x + o[c] * t.y;
  }
  return y;
}
// the online-softmax state of one query head over a piece: m, l, acc (4 dims per lane)
struct NhDspSm {
  float m, l;
  float4 acc;
};
NI_INLINE void nh_dsp_sm_add(thread NhDspSm& st, float s, float4 v) {
  const float mn = max(st.m, s);
  const float corr = metal::fast::exp(st.m - mn), e = metal::fast::exp(s - mn);
  st.l = st.l * corr + e;
  st.acc = st.acc * corr + e * v;
  st.m = mn;
}

// grid (Hkv, B, NS) x 32 G; part: [B][Hq][NS][HD + 2] (acc, m, l)
[[kernel]] void nh_dsp_attn_part(const device float* qkv [[buffer(0)]], const device float* qn [[buffer(1)]],
                                 const device half* rk [[buffer(2)]], const device half* rv [[buffer(3)]],
                                 device float* part [[buffer(4)]], constant NhDspAttnP& p [[buffer(5)]],
                                 const device float2* tab [[buffer(6)]],
                                 uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint3 ntg3 [[threads_per_threadgroup]]) {
  constexpr uint HD = 128, KT = 32;
  threadgroup half4 kt[KT * HD / 4];
  threadgroup half4 vt[KT * HD / 4];
  const uint kvh = tg.x, r = tg.y, sp = tg.z, G = p.Hq / p.Hkv, h = kvh * G + sg;
  const uint qpos = p.P + r;
  const uint n = p.P - p.lo, per = (n + p.NS - 1) / p.NS;
  const uint c1 = min(p.P, p.lo + (sp + 1) * per);
  const uint c0 = max(p.lo + sp * per, qpos + 1 > p.W ? qpos + 1 - p.W : 0u);  // (the window's first key)
  const float4 qr = ((const device float4*)(qkv + size_t(r) * p.ld + h * HD))[lane];
  const float4 qv = nh_dsp_norm_rope(qr, qn, tab + r * (HD / 2), p.eps, lane) * p.scale;
  NhDspSm st{-INFINITY, 0.0f, float4(0.0f)};
  const device half4* rkb = (const device half4*)(rk + size_t(kvh) * p.W * HD);
  const device half4* rvb = (const device half4*)(rv + size_t(kvh) * p.W * HD);
  for (uint t0 = c0; t0 < c1; t0 += KT) {
    const uint nt = min(KT, c1 - t0);
    for (uint i = tid; i < nt * HD / 4; i += ntg3.x) {
      const uint kk = i / (HD / 4), c = i % (HD / 4);
      const uint slot = (t0 + kk) % p.W;
      kt[i] = rkb[slot * HD / 4 + c];
      vt[i] = rvb[slot * HD / 4 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the tile's scores (lane j holds key j's), one rescale per tile, then the values weighted
    float sj = -INFINITY;
    for (uint j = 0; j < nt; ++j) {
      const float s = simd_sum(dot(qv, float4(kt[j * HD / 4 + lane])));
      if (lane == j) sj = s;
    }
    const float mn = max(st.m, simd_max(sj));
    const float corr = metal::fast::exp(st.m - mn), ej = lane < nt ? metal::fast::exp(sj - mn) : 0.0f;
    st.l = st.l * corr + simd_sum(ej);
    st.acc *= corr;
    for (uint j = 0; j < nt; ++j) st.acc += simd_shuffle(ej, j) * float4(vt[j * HD / 4 + lane]);
    st.m = mn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device float* o = part + ((size_t(r) * p.Hq + h) * p.NS + sp) * (HD + 2);
  ((device float4*)o)[lane] = st.acc;
  if (lane == 0) {
    o[HD] = st.m;
    o[HD + 1] = st.l;
  }
}

// grid (Hkv, B) x 32 G; out half [B][Hq HD]. NS <= 32
[[kernel]] void nh_dsp_attn_fin(const device float* qkv [[buffer(0)]], const device float* qn [[buffer(1)]],
                                const device float* kn [[buffer(2)]], const device float* sinks [[buffer(3)]],
                                const device float* part [[buffer(4)]], device half* out [[buffer(5)]],
                                constant NhDspAttnP& p [[buffer(6)]], const device float2* tab [[buffer(7)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint HD = 128;
  const uint kvh = tg.x, r = tg.y, G = p.Hq / p.Hkv, h = kvh * G + sg;
  const float4 qr = ((const device float4*)(qkv + size_t(r) * p.ld + h * HD))[lane];
  const float4 qv = nh_dsp_norm_rope(qr, qn, tab + r * (HD / 2), p.eps, lane) * p.scale;
  // the context pieces: lane s holds piece s's (m, l)
  const device float* pp = part + (size_t(r) * p.Hq + h) * p.NS * (HD + 2);
  const float ms = lane < p.NS ? pp[lane * (HD + 2) + HD] : -INFINITY;
  const float M = simd_max(ms);
  const float c = ms > -INFINITY ? metal::fast::exp(ms - M) : 0.0f;
  NhDspSm st{M, simd_sum(c > 0.0f ? pp[lane * (HD + 2) + HD + 1] * c : 0.0f), float4(0.0f)};
  for (uint s = 0; s < p.NS; ++s) {
    const float cs = simd_shuffle(c, s);
    if (cs > 0.0f) st.acc += ((const device float4*)(pp + s * (HD + 2)))[lane] * cs;
  }
  // the block's rows up to this one (their k normalized and rotated here)
  for (uint j = 0; j <= r && j < p.B; ++j) {
    const device float* row = qkv + size_t(j) * p.ld;
    const float4 kr = ((const device float4*)(row + (p.Hq + kvh) * HD))[lane];
    const float4 k4 = nh_dsp_norm_rope(kr, kn, tab + j * (HD / 2), p.eps, lane);
    nh_dsp_sm_add(st, simd_sum(dot(qv, k4)), ((const device float4*)(row + (p.Hq + p.Hkv + kvh) * HD))[lane]);
  }
  const float sk = sinks[h];
  const float mn = max(st.m, sk);
  const float corr = metal::fast::exp(st.m - mn);
  st.l = st.l * corr + metal::fast::exp(sk - mn);
  st.acc *= corr;
  ((device half4*)(out + (size_t(r) * p.Hq + h) * HD))[lane] = half4(st.acc / st.l);
}

// ---- rows 1..n of x = the mask token's embedding (the drafter's own row; the anchor row 0 comes from the target's
// embedding, which the drafter's equals everywhere else). grid: (n, ceil(D / 256)) x 256 ----
[[kernel]] void nh_dsp_mask_rows(device float* x [[buffer(0)]], const device float* row [[buffer(1)]],
                                 constant uint& D [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint r = tg.x + 1, d = tg.y * 256 + t2.x;
  if (d < D) x[size_t(r) * D + d] = row[d];
}

// ---- Markov walk: slot s picks argmax_c clp[s][c] + gs * W2[cand[s][c]] . W1[prev] over its K candidates (prev:
// the anchor tokens[P], then the previous pick), writes tokens[P + 1 + s] and picks[s], and the corrected log-probs
// qlp[s][c] = log(e^(clp + d) / (sum over the candidates + the mass outside them)) for the dynamic depth.
// W1 BF16 [V][R]; W2 NVFP4 [V][R] (codes [V][R / 2], UE4M3 scales [V][R / 16], tensor scale gs). R % 16 == 0,
// R <= 1024. grid: 1 x (32 K) threads (simdgroup c: candidate c). ----
struct NhDspWalkP {
  uint S, K, R, P;
  float gs;
};
[[kernel]] void nh_dsp_walk(const device int* cand [[buffer(0)]], const device float* clp [[buffer(1)]],
                            const device ushort* w1 [[buffer(2)]], const device uchar* w2q [[buffer(3)]],
                            const device uchar* w2s [[buffer(4)]], device int* tokens [[buffer(5)]],
                            device uint* picks [[buffer(6)]], device float* qlp [[buffer(7)]],
                            constant NhDspWalkP& p [[buffer(8)]], uint tid [[thread_position_in_threadgroup]],
                            uint c [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint ntg [[threads_per_threadgroup]]) {
  threadgroup float e[1024];
  threadgroup float sc[32];
  threadgroup int prev_t;
  if (tid == 0) prev_t = tokens[p.P];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 0; s < p.S; ++s) {
    const uint prev = uint(prev_t);
    for (uint k = tid; k < p.R; k += ntg) e[k] = as_type<float>(uint(w1[size_t(prev) * p.R + k]) << 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint id = uint(cand[s * p.K + c]);
    float d = 0.0f;
    for (uint b = lane; b < p.R / 16; b += 32) {  // one 16-element block per lane: byte j = elements j, j + 8
      const uint2 q2 = *(const device uint2*)(w2q + size_t(id) * (p.R / 2) + b * 8);
      const float bs = nh_ue4m3(w2s[size_t(id) * (p.R / 16) + b]);
      float acc = 0.0f;
      for (uint j = 0; j < 8; ++j) {
        const uint byte = ((j < 4 ? q2.x : q2.y) >> (8 * (j & 3))) & 0xFFu;
        acc += nh_fp4_lut[byte & 15u] * e[b * 16 + j] + nh_fp4_lut[byte >> 4] * e[b * 16 + j + 8];
      }
      d += acc * bs;
    }
    d = simd_sum(d);
    if (lane == 0) sc[c] = clp[s * p.K + c] + d * p.gs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint best = 0;
      float mx = sc[0], base = 0.0f;
      for (uint k = 1; k < p.K; ++k)
        if (sc[k] > mx) { mx = sc[k]; best = k; }
      float den = 0.0f;
      for (uint k = 0; k < p.K; ++k) {
        den += metal::precise::exp(sc[k] - mx);
        base += metal::precise::exp(clp[s * p.K + k]);
      }
      den += max(0.0f, 1.0f - base) * metal::precise::exp(-mx);
      const float lden = mx + metal::precise::log(den);
      for (uint k = 0; k < p.K; ++k) qlp[s * p.K + k] = sc[k] - lden;
      picks[s] = best;
      prev_t = cand[s * p.K + best];
      tokens[p.P + 1 + s] = prev_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- double-buffered decode: the parameters of a drafting half encoded before the host knows the verify's outcome.
// From the acceptance a (acc[0]) of the verify at P: m = a + 1 rows committed (positions P .. P + a), the anchor
// P' = P + m. out: one 256-byte block each (bound as the kernels' constant parameters): [0] the anchor's embedding
// (NhEmbedParams {D, tok_off P'}), [64] the block attention (NhDspAttnP {P', lo, W, Hq, Hkv, B, NS, ld, scale, eps,
// theta}), [128] nh_dsp_walk (NhDspWalkP {S, K, R, P', gs}), [192] the commit's ring-write grid (m, 2 Hkv, 1).
// lo = max(0, P' - W, hi - W) as the host computes it (hi: one past the highest position the ring held before the
// commit). One thread. ----
struct NhDspCycP {
  uint P, hi, W, D, Hq, Hkv, B, S, K, R, NS, ld;
  float eps, theta, scale, gs;
};
[[kernel]] void nh_dsp_cycle(const device int* acc [[buffer(0)]], device uint* out [[buffer(1)]],
                             constant NhDspCycP& p [[buffer(2)]]) {
  const uint m = uint(acc[0]) + 1u, P = p.P + m;
  const uint lo = max(P > p.W ? P - p.W : 0u, p.hi > p.W ? p.hi - p.W : 0u);
  device uint* e = out;
  e[0] = p.D;
  e[1] = P;
  device uint* a = out + 64;
  a[0] = P;
  a[1] = lo;
  a[2] = p.W;
  a[3] = p.Hq;
  a[4] = p.Hkv;
  a[5] = p.B;
  a[6] = p.NS;
  a[7] = p.ld;
  a[8] = as_type<uint>(p.scale);
  a[9] = as_type<uint>(p.eps);
  a[10] = as_type<uint>(p.theta);
  device uint* w = out + 128;
  w[0] = p.S;
  w[1] = p.K;
  w[2] = p.R;
  w[3] = P;
  w[4] = as_type<uint>(p.gs);
  device uint* g = out + 192;
  g[0] = m;
  g[1] = 2u * p.Hkv;
  g[2] = 1u;
}

// ---- the MLP's middle: x [R][ld] holds gate (columns 0..I-1) and up (I..2I-1) as the fused gate|up product left them
// (tensor scales not applied: gs_g, gs_u here), h = silu(gate) * up, then h to half with a per-row scale that keeps
// it in half's range (as nh_rows_half_scaled: s = max(1, max|h| / 16384), y = half(h / s), scale[r] = s) for the down
// product. The same values as the products with their scales, nh_swiglu and nh_rows_half_scaled. grid: R x 1024 ----
struct NhDspSwiP {
  uint I, ld;
  float gs_g, gs_u;
};
[[kernel]] void nh_dsp_swiglu_half(const device float* x [[buffer(0)]], device half* y [[buffer(1)]],
                                   device float* scale [[buffer(2)]], constant NhDspSwiP& p [[buffer(3)]],
                                   uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
  threadgroup float scratch[32];
  const device float* xr = x + size_t(r) * p.ld;
  float m = 0.0f;
  for (uint k = tid; k < p.I; k += ntg) m = max(m, fabs(silu(xr[k] * p.gs_g) * (xr[p.I + k] * p.gs_u)));
  m = tg_max(m, scratch, tid, ntg);
  const float s = max(1.0f, m / 16384.0f), inv = 1.0f / s;
  for (uint k = tid; k < p.I; k += ntg) y[size_t(r) * p.I + k] = half(silu(xr[k] * p.gs_g) * (xr[p.I + k] * p.gs_u) * inv);
  if (tid == 0) scale[r] = s;
}

// ---- the residual add of the down product scaled back: x[r] += t[r] * scale[r]. grid: (R, ceil(D / 256)) x 256 ----
[[kernel]] void nh_dsp_scale_add(device float* x [[buffer(0)]], const device float* t [[buffer(1)]],
                                 const device float* scale [[buffer(2)]], constant uint& D [[buffer(3)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint2 t2 [[thread_position_in_threadgroup]]) {
#pragma clang fp contract(off)  // (the product rounded before the add, as the two kernels did)
  const uint r = tg.x, d = tg.y * 256 + t2.x;
  if (d < D) x[size_t(r) * D + d] += t[size_t(r) * D + d] * scale[r];
}

// ---- a commit's context K/V into the ring: kv [R][ld] fp32 (k heads at columns 0.., v heads at Hkv * HD..), k
// per-head RMSNorm (weight w) and NeoX RoPE at position pos0 + row, both to half at slot = position % W (as
// nh_qk_norm_rope and nh_lili_ring_write, the same values). grid: (R, 2 Hkv) x HD (HD = 128) ----
struct NhDspKvP {
  uint pos0, W, Hkv, HD, ld;
  float eps, theta;
};
[[kernel]] void nh_dsp_commit_kv(const device float* kv [[buffer(0)]], const device float* w [[buffer(1)]],
                                 device half* rk [[buffer(2)]], device half* rv [[buffer(3)]],
                                 constant NhDspKvP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 t2 [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float row[256];
  const uint r = tg.x, hh = tg.y % p.Hkv, d = t2.x;
  const bool isv = tg.y >= p.Hkv;
  const uint slot = (p.pos0 + r) % p.W;
  float v = kv[size_t(r) * p.ld + (isv ? p.Hkv * p.HD : 0u) + hh * p.HD + d];
  if (!isv) {
    const float ss = tg_sum(v * v, scratch, d, p.HD);
    v = v * (1.0f / metal::precise::sqrt(ss / float(p.HD) + p.eps)) * w[d];
    row[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_d = p.HD / 2, i = d % half_d;
    const float inv = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.HD));
    const float ang = float(p.pos0 + r) * inv;
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    v = d < half_d ? row[d] * cs - row[d + half_d] * sn : row[d] * cs + row[d - half_d] * sn;
  }
  (isv ? rv : rk)[(size_t(hh) * p.W + slot) * p.HD + d] = half(v);
}

// ---- a fixed verify depth (NI_DSPARK_DEPTH, measurements): live[0] = d + 1. One thread ----
[[kernel]] void nh_dsp_fixed_depth(device uint* live [[buffer(0)]], constant uint& d [[buffer(1)]]) { live[0] = d + 1; }

// ---- the Markov walk in two passes (nh_dsp_walk's values): nh_dsp_walk_dots computes every dot the walk can need,
// D[s][j][c] = W2[cand[s][c]] . W1[prev] for prev = the anchor (s = 0, j = 0) or slot s - 1's candidate j, all at
// once (grid S x 32 K: simdgroup c of threadgroup s, each lane one 16-element block of W2's row, as nh_dsp_walk);
// nh_dsp_walk_pick then walks the slots on one simdgroup (lane c: candidate c), the sums over candidates in their
// order on lane 0. (nh_dsp_walk: one threadgroup, 7 slots of loads and reductions in series, 30 us isolated.) ----
[[kernel]] void nh_dsp_walk_dots(const device int* cand [[buffer(0)]], const device ushort* w1 [[buffer(1)]],
                                 const device uchar* w2q [[buffer(2)]], const device uchar* w2s [[buffer(3)]],
                                 const device int* tokens [[buffer(4)]], device float* D [[buffer(5)]],
                                 constant NhDspWalkP& p [[buffer(6)]], uint s [[threadgroup_position_in_grid]],
                                 uint c [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (c >= p.K) return;
  const uint id = uint(cand[s * p.K + c]);
  const uint nj = s == 0 ? 1u : p.K;
  for (uint j = 0; j < nj; ++j) {
    const uint prev = s == 0 ? uint(tokens[p.P]) : uint(cand[(s - 1) * p.K + j]);
    const device ushort* e = w1 + size_t(prev) * p.R;
    float d = 0.0f;
    for (uint b = lane; b < p.R / 16; b += 32) {
      const uint2 q2 = *(const device uint2*)(w2q + size_t(id) * (p.R / 2) + b * 8);
      const float bs = nh_ue4m3(w2s[size_t(id) * (p.R / 16) + b]);
      float acc = 0.0f;
      for (uint jj = 0; jj < 8; ++jj) {
        const uint byte = ((jj < 4 ? q2.x : q2.y) >> (8 * (jj & 3))) & 0xFFu;
        acc += nh_fp4_lut[byte & 15u] * as_type<float>(uint(e[b * 16 + jj]) << 16) +
               nh_fp4_lut[byte >> 4] * as_type<float>(uint(e[b * 16 + jj + 8]) << 16);
      }
      d += acc * bs;
    }
    d = simd_sum(d);
    if (lane == 0) D[(s * p.K + j) * p.K + c] = d;
  }
}

[[kernel]] void nh_dsp_walk_pick(const device int* cand [[buffer(0)]], const device float* clp [[buffer(1)]],
                                 const device float* D [[buffer(2)]], device int* tokens [[buffer(3)]],
                                 device uint* picks [[buffer(4)]], device float* qlp [[buffer(5)]],
                                 constant NhDspWalkP& p [[buffer(6)]], uint lane [[thread_index_in_simdgroup]]) {
  uint j = 0;  // the previous slot's pick (the anchor's row for slot 0)
  for (uint s = 0; s < p.S; ++s) {
    const bool on = lane < p.K;
    const float cl = on ? clp[s * p.K + lane] : 0.0f;
    const float sc = on ? cl + D[(s * p.K + j) * p.K + lane] * p.gs : -INFINITY;
    const float mx = simd_max(sc);
    const uint best = simd_min(on && sc == mx ? lane : 0xffffffffu);  // the first maximum
    const float ex = metal::precise::exp(sc - mx), eb = metal::precise::exp(cl);
    float den = 0.0f, base = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
      den += simd_shuffle(ex, k);
      base += simd_shuffle(eb, k);
    }
    den += max(0.0f, 1.0f - base) * metal::precise::exp(-mx);
    const float lden = mx + metal::precise::log(den);
    if (on) qlp[s * p.K + lane] = sc - lden;
    if (lane == 0) {
      picks[s] = best;
      tokens[p.P + 1 + s] = cand[s * p.K + best];
    }
    j = best;
  }
}
