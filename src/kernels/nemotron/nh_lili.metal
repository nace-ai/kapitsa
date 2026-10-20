// LiLiCorr block drafter for Nemotron 3.5 Bolt (GGUF architecture "dflash" with a LiLiCorr head): a 5-layer
// DFlash2-style drafter (grouped
// dynamic convolutions around attention and FFN, attention sinks, sliding window, context K/V from the target's
// hidden states) proposes top-8 candidates for 7 slots; the LiLiCorr head scores candidate transitions; a greedy
// walk over the lattice gives the draft.

// ---- capture of target layer inputs (fp32 rows -> columns [slot * D, (slot + 1) * D) of the feature rows) ----
// grid: (R, ceil(D / 256)) x 256. Rows src_row0 + r of x land in feature row dst_row0 + r.
struct NhCapP {
  uint D, ld, slot, dst_row0, src_row0;
};
[[kernel]] void nh_capture(const device float* x [[buffer(0)]], device float* feat [[buffer(1)]],
                           constant NhCapP& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint r = tg.x, d = tg.y * 256 + t2.x;
  if (d < p.D) feat[size_t(p.dst_row0 + r) * p.ld + p.slot * p.D + d] = x[size_t(p.src_row0 + r) * p.D + d];
}

// ---- DFlash2 grouped dynamic convolution inside the block (kernel 2): ----
// y[i, c] = sum_{tap < 2} (base[side, tap, c] + dyn[i, side, tap, c / gs]) * x[i - tap, c]   (x[-1] = 0)
// dyn rows: [2 sides][2 taps][D / gs groups]. grid: (R, ceil(D / 256)) x 256.
struct NhDconvP {
  uint D, gs, side, R;
};
[[kernel]] void nh_dconv(const device float* x [[buffer(0)]], const device float* dyn [[buffer(1)]],
                         const device float* base [[buffer(2)]], device float* y [[buffer(3)]],
                         constant NhDconvP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint i = tg.x, c = tg.y * 256 + t2.x;
  if (c >= p.D) return;
  const uint G = p.D / p.gs, g = c / p.gs;
  const device float* dr = dyn + size_t(i) * 4 * G;
  float v = (base[(p.side * 2 + 0) * p.D + c] + dr[(p.side * 2 + 0) * G + g]) * x[size_t(i) * p.D + c];
  if (i > 0) v += (base[(p.side * 2 + 1) * p.D + c] + dr[(p.side * 2 + 1) * G + g]) * x[size_t(i - 1) * p.D + c];
  y[size_t(i) * p.D + c] = v;
}

// ---- per-head RMSNorm (weight [HD]) and NeoX RoPE at position pos0 + row; rope = 0 skips the rotation ----
// x [R][H][HD] in place. grid: (R, H) x HD threads (HD = 128)
struct NhQkP {
  uint H, HD, pos0, rope;
  float eps, theta;
};
[[kernel]] void nh_qk_norm_rope(device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                                constant NhQkP& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint2 t2 [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float row[256];
  const uint r = tg.x, h = tg.y, d = t2.x;
  device float* xp = x + (size_t(r) * p.H + h) * p.HD;
  float v = xp[d];
  const float ss = tg_sum(v * v, scratch, d, p.HD);
  v = v * (1.0f / metal::precise::sqrt(ss / float(p.HD) + p.eps)) * w[d];
  if (p.rope) {
    row[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_d = p.HD / 2, i = d % half_d;
    const float inv = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.HD));
    const float ang = float(p.pos0 + r) * inv;
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    v = d < half_d ? row[d] * cs - row[d + half_d] * sn : row[d] * cs + row[d - half_d] * sn;
  }
  xp[d] = v;
}

// ---- context K/V into the drafter's ring (slot = position % W): k, v [R][Hkv * HD] fp32 -> half ----
// grid: (R, 2 * Hkv) x HD. ring: [layer][Hkv][W][HD] (this call gets the layer's base)
struct NhRingKvP {
  uint pos0, W, Hkv, HD;
};
[[kernel]] void nh_lili_ring_write(const device float* k [[buffer(0)]], const device float* v [[buffer(1)]],
                                   device half* rk [[buffer(2)]], device half* rv [[buffer(3)]],
                                   constant NhRingKvP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint r = tg.x, hh = tg.y % p.Hkv, d = t2.x;
  const bool isv = tg.y >= p.Hkv;
  const uint slot = (p.pos0 + r) % p.W;
  const float val = (isv ? v : k)[size_t(r) * p.Hkv * p.HD + hh * p.HD + d];
  (isv ? rv : rk)[(size_t(hh) * p.W + slot) * p.HD + d] = half(val);
}

// ---- block attention: 8 query rows at positions P..P+7 over the ring's context [lo, P) and the block's own
// 8 rows (non-causal), sliding window W (query p sees keys k > p - W), per-head sink logits. One threadgroup per
// (kv head, row): G = Hq / Hkv simdgroups (one per query head), lanes hold HD / 32 = 4 dims. ----
struct NhLiliAttnP {
  uint P, lo, W, Hq, Hkv, B;  // B: block rows
  float scale;
};
[[kernel]] void nh_lili_attn(const device float* q [[buffer(0)]], const device float* kb [[buffer(1)]],
                             const device float* vb [[buffer(2)]], const device half* rk [[buffer(3)]],
                             const device half* rv [[buffer(4)]], const device float* sinks [[buffer(5)]],
                             device float* out [[buffer(6)]], constant NhLiliAttnP& p [[buffer(7)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint2 ntg2 [[threads_per_threadgroup]]) {
  constexpr uint HD = 128, KT = 32;
  threadgroup half4 kt[KT * HD / 4];
  threadgroup half4 vt[KT * HD / 4];
  const uint kvh = tg.x, r = tg.y, G = p.Hq / p.Hkv, h = kvh * G + sg;
  const uint qpos = p.P + r;
  const uint klo = max(p.lo, qpos + 1 > p.W ? qpos + 1 - p.W : 0u);  // first context key in the window
  const float4 qv = ((const device float4*)(q + (size_t(r) * p.Hq + h) * HD))[lane] * p.scale;
  float4 acc = 0.0f;
  float m = -INFINITY, l = 0.0f;
  const device half4* rkb = (const device half4*)(rk + size_t(kvh) * p.W * HD);
  const device half4* rvb = (const device half4*)(rv + size_t(kvh) * p.W * HD);
  for (uint t0 = klo; t0 < p.P; t0 += KT) {
    const uint n = min(KT, p.P - t0);
    for (uint i = tid; i < n * HD / 4; i += ntg2.x) {
      const uint kk = i / (HD / 4), c = i % (HD / 4);
      const uint slot = (t0 + kk) % p.W;
      kt[i] = rkb[slot * HD / 4 + c];
      vt[i] = rvb[slot * HD / 4 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < n; ++j) {
      const float s = simd_sum(dot(qv, float4(kt[j * HD / 4 + lane])));
      const float mn = max(m, s);
      const float corr = metal::fast::exp(m - mn), e = metal::fast::exp(s - mn);
      l = l * corr + e;
      acc = acc * corr + e * float4(vt[j * HD / 4 + lane]);
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint j = 0; j < p.B; ++j) {  // the block's own rows (fp32, not yet in the ring)
    const float4 kv4 = ((const device float4*)(kb + (size_t(j) * p.Hkv + kvh) * HD))[lane];
    const float s = simd_sum(dot(qv, kv4));
    const float mn = max(m, s);
    const float corr = metal::fast::exp(m - mn), e = metal::fast::exp(s - mn);
    l = l * corr + e;
    acc = acc * corr + e * ((const device float4*)(vb + (size_t(j) * p.Hkv + kvh) * HD))[lane];
    m = mn;
  }
  // sink: one more logit in the denominator
  const float sk = sinks[h];
  const float mn = max(m, sk);
  const float corr = metal::fast::exp(m - mn);
  l = l * corr + metal::fast::exp(sk - mn);
  acc *= corr;
  ((device float4*)(out + (size_t(r) * p.Hq + h) * HD))[lane] = acc / l;
}

// ---- y = silu(a) * b, n elements ----
[[kernel]] void nh_swiglu(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                          device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = silu(a[i]) * b[i];
}

// ---- candidates: per slot row (logits rows 1..S of the pass), the top-K tokens in descending order with their
// log-probabilities (log-softmax over the full row). Stage 1: per (row, chunk of 2048) the chunk's top K and
// (max, sum exp); grid (nchunks, S) x 256. Stage 2: per row; grid S x 32. ----
struct NhCandP {
  uint V, K, nchunks, row0, ld;
};
[[kernel]] void nh_cand1(const device float* logits [[buffer(0)]], device float* cv [[buffer(1)]],
                         device uint* ci [[buffer(2)]], device float2* ms [[buffer(3)]],
                         constant NhCandP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint2 t2 [[thread_position_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float scratch[32];
  const uint tid = t2.x, ntg = ntg2.x, c = tg.x, s = tg.y;
  const device float* lg = logits + size_t(p.row0 + s) * p.ld;
  const uint base = c * 2048 + tid * 8;
  float v[8];
  float mx = -INFINITY;
  for (int j = 0; j < 8; ++j) {
    v[j] = base + j < p.V ? lg[base + j] : -INFINITY;
    mx = max(mx, v[j]);
  }
  const float cmax = tg_max(mx, scratch, tid, ntg);
  float se = 0.0f;
  for (int j = 0; j < 8; ++j) se += v[j] == -INFINITY ? 0.0f : metal::precise::exp(v[j] - cmax);
  se = tg_sum(se, scratch, tid, ntg);
  for (uint r = 0; r < p.K; ++r) {
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (int j = 0; j < 8; ++j)
      if (v[j] > bv) { bv = v[j]; bi = base + j; }
    // argmax over the threadgroup, lowest index on ties
    float wv = bv;
    uint wi = bi;
    for (uint off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(wv, off);
      const uint oi = simd_shuffle_down(wi, off);
      if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = wv; si[tid >> 5] = wi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
      wv = tid < ntg / 32 ? sv[tid] : -INFINITY;
      wi = tid < ntg / 32 ? si[tid] : 0xffffffffu;
      for (uint off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(wv, off);
        const uint oi = simd_shuffle_down(wi, off);
        if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
      }
      if (tid == 0) { sv[0] = wv; si[0] = wi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    wv = sv[0];
    wi = si[0];
    if (wi >= base && wi < base + 8) v[wi - base] = -INFINITY;
    if (tid == 0) {
      cv[(size_t(s) * p.nchunks + c) * p.K + r] = wv;
      ci[(size_t(s) * p.nchunks + c) * p.K + r] = wi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) ms[size_t(s) * p.nchunks + c] = float2(cmax, se);
}

// stage 2: one simdgroup per row merges its nchunks <= 64 sorted chunk lists (value descending, lowest id first on
// ties): lane l holds the heads of chunks l and l + 32, and each of the K rounds takes the best head by (value
// descending, id ascending, chunk ascending) with a simdgroup reduction, as a sequential merge takes them; the
// log-sum-exp over the chunks' (max, sum exp) in chunk order on lane 0. Writes cand ids (as ints) and log-probs
// [S][K]. (One thread per row merging the lists: 117 us isolated / ~90 us in the pipeline on Lightning's DSpark pass,
// 7 rows x 64 chunks x 16; the same output.) grid: S x 32 ----
[[kernel]] void nh_cand2(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                         const device float2* ms [[buffer(2)]], device int* cand [[buffer(3)]],
                         device float* clp [[buffer(4)]], constant NhCandP& p [[buffer(5)]],
                         uint s [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
  threadgroup float tv[64 * 32];
  threadgroup uint ti[64 * 32];
  const uint E = p.nchunks * p.K;
  for (uint i0 = 0; i0 < E; i0 += 32 * 8) {  // 8 loads in flight per lane before the stores
    float v[8];
    uint id[8];
    for (uint u = 0; u < 8; ++u) {
      const uint i = i0 + u * 32 + lane;
      v[u] = i < E ? cv[size_t(s) * E + i] : 0.0f;
      id[u] = i < E ? ci[size_t(s) * E + i] : 0u;
    }
    for (uint u = 0; u < 8; ++u) {
      const uint i = i0 + u * 32 + lane;
      if (i < E) {
        tv[i] = v[u];
        ti[i] = id[u];
      }
    }
  }
  // the log-sum-exp: the chunks' terms on their lanes, summed in chunk order on lane 0
  const float2 m0 = lane < p.nchunks ? ms[s * p.nchunks + lane] : float2(-INFINITY, 0.0f);
  const float2 m1 = lane + 32 < p.nchunks ? ms[s * p.nchunks + lane + 32] : float2(-INFINITY, 0.0f);
  const float M = simd_max(max(m0.x, m1.x));
  const float e0 = metal::precise::exp(m0.x - M), e1 = metal::precise::exp(m1.x - M);
  float S = 0.0f;
  for (uint c = 0; c < p.nchunks; ++c) {
    const float vx = simd_shuffle(c < 32 ? m0.x : m1.x, c & 31), vy = simd_shuffle(c < 32 ? m0.y : m1.y, c & 31);
    const float ec = simd_shuffle(c < 32 ? e0 : e1, c & 31);
    if (vx > -INFINITY) S += vy * ec;
  }
  const float lse = M + metal::precise::log(S);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint h0 = 0, h1 = 0;  // the heads of chunks lane, lane + 32
  for (uint r = 0; r < p.K; ++r) {
    // this lane's best head (an exhausted or missing chunk: none)
    float bv = -INFINITY;
    uint bi = 0xffffffffu, bc = 0xffffffffu;
    if (lane < p.nchunks && h0 < p.K) {
      bv = tv[lane * p.K + h0];
      bi = ti[lane * p.K + h0];
      bc = lane;
    }
    if (lane + 32 < p.nchunks && h1 < p.K) {
      const float v = tv[(lane + 32) * p.K + h1];
      const uint id = ti[(lane + 32) * p.K + h1];
      if (bc == 0xffffffffu || v > bv || (v == bv && id < bi)) {
        bv = v;
        bi = id;
        bc = lane + 32;
      }
    }
    for (uint off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_xor(bv, off);
      const uint oi = simd_shuffle_xor(bi, off), oc = simd_shuffle_xor(bc, off);
      if (oc != 0xffffffffu && (bc == 0xffffffffu || ov > bv || (ov == bv && (oi < bi || (oi == bi && oc < bc))))) {
        bv = ov;
        bi = oi;
        bc = oc;
      }
    }
    if (bc == 0xffffffffu) {  // every chunk exhausted (K > the candidates there are): as the merge, nothing found
      bv = -INFINITY;
      bi = 0xffffffffu;
    } else if (bc == lane) {
      ++h0;
    } else if (bc == lane + 32) {
      ++h1;
    }
    if (lane == 0) {
      cand[s * p.K + r] = int(bi);
      clp[s * p.K + r] = bv - lse;
    }
  }
}

// ---- LiLiCorr input states: x0[n] = token_proj(embed(cand)) + pass_proj(hidden of the slot) + feature MLP +
// slot / rank embeddings (+ biases). Nodes n = slot * K + rank. ----
// feature MLP first layer: f = LayerNorm(5 features) * w + b; h1 = silu(W1 f + b1) [N][Hh]. grid: N x 256
struct NhLiliFeatP {
  uint K, Hh;
  float eps;
};
[[kernel]] void nh_lili_feat1(const device float* clp [[buffer(0)]], const device float* ln_w [[buffer(1)]],
                              const device float* ln_b [[buffer(2)]], const device float* w1 [[buffer(3)]],
                              const device float* b1 [[buffer(4)]], device float* h1 [[buffer(5)]],
                              constant NhLiliFeatP& p [[buffer(6)]], uint n [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  const uint s = n / p.K, rank = n % p.K;
  const float lp = clp[n], lp_top = clp[s * p.K];
  float f[5] = {lp, metal::precise::exp(lp), lp - lp_top, p.K > 1 ? float(rank) / float(p.K - 1) : 0.0f,
                rank == 0 ? 1.0f : 0.0f};
  float mean = 0.0f;
  for (int i = 0; i < 5; ++i) mean += f[i];
  mean /= 5.0f;
  float var = 0.0f;
  for (int i = 0; i < 5; ++i) var += (f[i] - mean) * (f[i] - mean);
  var /= 5.0f;
  const float inv = 1.0f / metal::precise::sqrt(var + p.eps);
  for (int i = 0; i < 5; ++i) f[i] = (f[i] - mean) * inv * ln_w[i] + ln_b[i];
  for (uint o = tid; o < p.Hh; o += ntg) {
    float v = b1[o];
    for (int i = 0; i < 5; ++i) v += w1[o * 5 + i] * f[i];
    h1[size_t(n) * p.Hh + o] = silu(v);
  }
}

// x0[n, :] = tok[n] + tok_b + pass[s] + pass_b + feat[n] + feat_b + slot_emb[s] + rank_emb[rank]. grid: N x 256
[[kernel]] void nh_lili_x0(const device float* tok [[buffer(0)]], const device float* tok_b [[buffer(1)]],
                           const device float* pass [[buffer(2)]], const device float* pass_b [[buffer(3)]],
                           const device float* feat [[buffer(4)]], const device float* feat_b [[buffer(5)]],
                           const device float* slot_emb [[buffer(6)]], const device float* rank_emb [[buffer(7)]],
                           device float* x0 [[buffer(8)]], constant NhLiliFeatP& p [[buffer(9)]],
                           uint n [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]]) {
  const uint s = n / p.K, rank = n % p.K;
  for (uint o = tid; o < p.Hh; o += ntg)
    x0[size_t(n) * p.Hh + o] = tok[size_t(n) * p.Hh + o] + tok_b[o] + pass[size_t(s) * p.Hh + o] + pass_b[o] +
                               feat[size_t(n) * p.Hh + o] + feat_b[o] + slot_emb[s * p.Hh + o] + rank_emb[rank * p.Hh + o];
}

// y = x + b (rows of D), optionally silu: y = silu(x + b). grid: ceil(R * D / 256) x 256
struct NhBiasP {
  uint D, n, act;  // act: 0 none, 1 silu
};
[[kernel]] void nh_bias_act(device float* x [[buffer(0)]], const device float* b [[buffer(1)]],
                            constant NhBiasP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  float v = x[i] + b[i % p.D];
  if (p.act == 1) v = silu(v);
  x[i] = v;
}

// x += y + b (residual with a bias). grid: ceil(n / 256) x 256
[[kernel]] void nh_add_bias(device float* x [[buffer(0)]], const device float* y [[buffer(1)]],
                            const device float* b [[buffer(2)]], constant NhBiasP& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < p.n) x[i] += y[i] + b[i % p.D];
}

// ---- LiLiCorr attention over the N nodes: qkv [N][3 * Hh] (+ bias already added), heads of 128 dims, additive
// bias [heads][N][N]; out [N][Hh]. One simdgroup per (head, query): lanes take keys j = lane, lane + 32 for the
// scores and 4 dims each for the output. grid: (heads, N) x 32 ----
struct NhLiliMhaP {
  uint N, Hh, heads;
  float scale;
};
[[kernel]] void nh_lili_mha(const device float* qkv [[buffer(0)]], const device float* bias [[buffer(1)]],
                            device float* out [[buffer(2)]], constant NhLiliMhaP& p [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint HD = 128;
  const uint h = tg.x, i = tg.y;
  const device float* qi = qkv + size_t(i) * 3 * p.Hh + h * HD;
  float s[2];
  for (int u = 0; u < 2; ++u) {
    const uint j = lane + 32 * uint(u);
    float v = -INFINITY;
    if (j < p.N) {
      const device float* kj = qkv + size_t(j) * 3 * p.Hh + p.Hh + h * HD;
      float d = 0.0f;
      for (uint t = 0; t < HD; t += 4) d += dot(*(const device float4*)(qi + t), *(const device float4*)(kj + t));
      v = d * p.scale + bias[(size_t(h) * p.N + i) * p.N + j];
    }
    s[u] = v;
  }
  const float mx = simd_max(max(s[0], s[1]));
  float e0 = s[0] == -INFINITY ? 0.0f : metal::precise::exp(s[0] - mx);
  float e1 = s[1] == -INFINITY ? 0.0f : metal::precise::exp(s[1] - mx);
  const float sum = simd_sum(e0 + e1);
  e0 /= sum;
  e1 /= sum;
  float4 acc = 0.0f;
  for (uint j = 0; j < p.N; ++j) {
    const float pj = simd_shuffle(j < 32 ? e0 : e1, ushort(j & 31));
    acc += pj * *(const device float4*)(qkv + size_t(j) * 3 * p.Hh + 2 * p.Hh + h * HD + lane * 4);
  }
  *(device float4*)(out + size_t(i) * p.Hh + h * HD + lane * 4) = acc;
}

// factor input: f[n] = [x[n], a, x[n] * a] (3 * Hh). grid: N x 256
[[kernel]] void nh_lili_factor_in(const device float* x [[buffer(0)]], const device float* a [[buffer(1)]],
                                  device float* f [[buffer(2)]], constant NhLiliFeatP& p [[buffer(3)]],
                                  uint n [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                  uint ntg [[threads_per_threadgroup]]) {
  for (uint o = tid; o < p.Hh; o += ntg) {
    const float xv = x[size_t(n) * p.Hh + o], av = a[o];
    f[size_t(n) * 3 * p.Hh + o] = xv;
    f[size_t(n) * 3 * p.Hh + p.Hh + o] = av;
    f[size_t(n) * 3 * p.Hh + 2 * p.Hh + o] = xv * av;
  }
}

// y = (x + b) / max(||x + b||, eps), rows of D. grid: R x 256
struct NhL2P {
  uint D;
  float eps;
};
[[kernel]] void nh_bias_l2norm(device float* x [[buffer(0)]], const device float* b [[buffer(1)]],
                               constant NhL2P& p [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  device float* xr = x + size_t(r) * p.D;
  float ss = 0.0f;
  for (uint o = tid; o < p.D; o += ntg) {
    const float v = xr[o] + b[o];
    xr[o] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = 1.0f / max(metal::precise::sqrt(ss), p.eps);
  for (uint o = tid; o < p.D; o += ntg) xr[o] *= inv;
}

// ---- scores and the greedy walk: first[c] = ls * in[0, c] . anchor_out; trans[s][c][prev] =
// ls * in[s, c] . out[s - 1, prev]; prev = argmax first, then argmax_c trans[s][c][prev]; tokens[P + 1 + s] =
// cand[s][prev]. One threadgroup of 64 threads per pass (K * K = 64 dot products per slot). ----
struct NhWalkP {
  uint S, K, D, P;
  float ls;
};
// grid: 1 x (32 K) threads: simdgroup c computes candidate c's dot products. picks[s] = the chosen candidate.
[[kernel]] void nh_lili_walk(const device float* in [[buffer(0)]], const device float* outv [[buffer(1)]],
                             const device float* aout [[buffer(2)]], const device int* cand [[buffer(3)]],
                             device int* tokens [[buffer(4)]], device float* scores [[buffer(5)]],
                             constant NhWalkP& p [[buffer(6)]], device uint* picks [[buffer(7)]],
                             uint tid [[thread_position_in_threadgroup]], uint c [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sc[64];
  threadgroup uint prev_s;
  // slot 0 against the anchor
  {
    float d = 0.0f;
    for (uint o = lane; o < p.D; o += 32) d += in[size_t(c) * p.D + o] * aout[o];
    d = simd_sum(d);
    if (lane == 0) {
      sc[c] = p.ls * d;
      scores[c] = sc[c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint b = 0;
    for (uint k = 1; k < p.K; ++k)
      if (sc[k] > sc[b]) b = k;
    prev_s = b;
    picks[0] = b;
    tokens[p.P + 1] = cand[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 1; s < p.S; ++s) {
    const uint prev = prev_s;
    // candidate c of slot s given the previous slot's chosen candidate
    float d = 0.0f;
    const device float* a = in + (size_t(s) * p.K + c) * p.D;
    const device float* b = outv + (size_t(s - 1) * p.K + prev) * p.D;
    for (uint o = lane; o < p.D; o += 32) d += a[o] * b[o];
    d = simd_sum(d);
    if (lane == 0) {
      sc[c] = p.ls * d;
      scores[s * p.K + c] = sc[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint bb = 0;
      for (uint k = 1; k < p.K; ++k)
        if (sc[k] > sc[bb]) bb = k;
      prev_s = bb;
      picks[s] = bb;
      tokens[p.P + 1 + s] = cand[s * p.K + bb];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- dynamic verify depth: rows R = K + 1 of the coming verify, K in [0, S], maximizing the expected tokens per cost
// (1 + sum_{i <= K} prod_{s < i} p_s) / (1 + ratio (K + 1)), p_s = the drafter's probability of its pick at slot s
// (exp of its log-prob over the full head). ratio = per-row cost (the routed experts) over the fixed cost of a cycle;
// 0: K = S. One thread. ----
// mode 1: p_s = the LiLiCorr walk's softmax probability of its pick given the previous pick (scores, buffer 4);
// mode 2: the geometric mean of the two.
struct NhDepthP {
  uint S, K;
  float ratio;
  uint mode;
};
[[kernel]] void nh_lili_depth(const device float* clp [[buffer(0)]], const device uint* picks [[buffer(1)]],
                              device uint* live [[buffer(2)]], constant NhDepthP& p [[buffer(3)]],
                              const device float* scores [[buffer(4)]]) {
  if (p.ratio <= 0.0f) {
    live[0] = p.S + 1;
    return;
  }
  float prod = 1.0f, exp_tokens = 1.0f, best = 1.0f / (1.0f + p.ratio);
  uint best_k = 0;
  for (uint s = 0; s < p.S; ++s) {
    const float pm = metal::precise::exp(clp[s * p.K + picks[s]]);
    float pw = 1.0f;
    if (p.mode != 0) {  // softmax over the slot's K walk scores
      float mx = -INFINITY;
      for (uint k = 0; k < p.K; ++k) mx = max(mx, scores[s * p.K + k]);
      float den = 0.0f;
      for (uint k = 0; k < p.K; ++k) den += metal::precise::exp(scores[s * p.K + k] - mx);
      pw = metal::precise::exp(scores[s * p.K + picks[s]] - mx) / den;
    }
    prod *= p.mode == 0 ? pm : p.mode == 1 ? pw : metal::precise::sqrt(pm * pw);
    exp_tokens += prod;
    const float v = exp_tokens / (1.0f + p.ratio * float(s + 2));
    if (v > best) { best = v; best_k = s + 1; }
  }
  live[0] = best_k + 1;
}

// ---- verify acceptance: samples[r] = the target's token after row r (sampled with the request's settings);
// a = longest prefix with samples[i] == tokens[P + 1 + i] (i < K); tokens[P + 1 + a] = samples[a]; acc = {a, K}.
// One thread. ----
struct NhAcceptP {
  uint P, K;
};
[[kernel]] void nh_accept(const device int* samples [[buffer(0)]], device int* tokens [[buffer(1)]],
                          device int* acc [[buffer(2)]], constant NhAcceptP& p [[buffer(3)]],
                          const device uint* live [[buffer(4)]]) {
  const uint K = min(p.K, live[0] - 1u);  // drafts the block verified (the drafter's dynamic depth)
  uint a = 0;
  while (a < K && samples[a] == tokens[p.P + 1 + a]) ++a;
  tokens[p.P + 1 + a] = samples[a];
  acc[0] = int(a);
}

// ---- the drafting block's tokens: [tokens[P], mask x (B - 1)]. grid: 1 x B ----
struct NhBlockTokP {
  uint P, mask, B;
};
[[kernel]] void nh_block_tok(const device int* tokens [[buffer(0)]], device int* btok [[buffer(1)]],
                             constant NhBlockTokP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.B) btok[i] = i == 0 ? tokens[p.P] : int(p.mask);
}

// ---- anchor slots: rows r of g (positions pos0 + r) within the last `nslots` positions -> anchor[pos % nslots] ----
// grid: (R, ceil(D / 256)) x 256
struct NhAnchorP {
  uint pos0, R, D, nslots;
};
[[kernel]] void nh_anchor_write(const device float* g [[buffer(0)]], device float* anchor [[buffer(1)]],
                                constant NhAnchorP& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint r = tg.x, d = tg.y * 256 + t2.x;
  if (d >= p.D || r + p.nslots < p.R) return;
  anchor[size_t((p.pos0 + r) % p.nslots) * p.D + d] = g[size_t(r) * p.D + d];
}

// ---- fp32 rows -> half with a per-row scale that keeps them in half's range: y = half(x / s), s = max(1,
// max|x| / 16384) into scale[r] (raw residual-stream features can exceed 65504). grid: R x 1024 ----
struct NhHalfScaledP {
  uint D;
};
[[kernel]] void nh_rows_half_scaled(const device float* x [[buffer(0)]], device half* y [[buffer(1)]],
                                    device float* scale [[buffer(2)]], constant NhHalfScaledP& p [[buffer(3)]],
                                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* xr = x + size_t(r) * p.D;
  float m = 0.0f;
  for (uint k = tid; k < p.D; k += ntg) m = max(m, fabs(xr[k]));
  m = tg_max(m, scratch, tid, ntg);
  const float s = max(1.0f, m / 16384.0f), inv = 1.0f / s;
  for (uint k = tid; k < p.D; k += ntg) y[size_t(r) * p.D + k] = half(xr[k] * inv);
  if (tid == 0) scale[r] = s;
}

// rows *= scale[r]. grid: (R, ceil(D / 256)) x 256
[[kernel]] void nh_rows_scale(device float* x [[buffer(0)]], const device float* scale [[buffer(1)]],
                              constant NhHalfScaledP& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint r = tg.x, d = tg.y * 256 + t2.x;
  if (d < p.D) x[size_t(r) * p.D + d] *= scale[r];
}
