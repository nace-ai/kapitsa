// DSpark block drafter (Nemotron 3.5 Lightning): the block attention, causal within the block; the mask rows of the
// block; the Markov head's greedy walk over each slot's candidates. The rest of the pass runs on LiLiCorr's kernels
// (nh_lili.metal: capture, q/k norm + RoPE, ring, candidates, dynamic depth, acceptance).

// ---- block attention: 8 query rows at positions P..P+7 over the ring's context [lo, P) and the block's rows up to
// their own (causal), sliding window W, per-head sink logits. As nh_lili_attn: one threadgroup per (kv head, row),
// G = Hq / Hkv simdgroups (one per query head), lanes hold HD / 32 = 4 dims. ----
struct NhDspAttnP {  // (nh_lili_attn's parameters)
  uint P, lo, W, Hq, Hkv, B;
  float scale;
};
[[kernel]] void nh_dsp_attn(const device float* q [[buffer(0)]], const device float* kb [[buffer(1)]],
                            const device float* vb [[buffer(2)]], const device half* rk [[buffer(3)]],
                            const device half* rv [[buffer(4)]], const device float* sinks [[buffer(5)]],
                            device float* out [[buffer(6)]], constant NhDspAttnP& p [[buffer(7)]],
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
  for (uint j = 0; j <= r && j < p.B; ++j) {  // the block's rows up to this one (fp32, not in the ring)
    const float4 kv4 = ((const device float4*)(kb + (size_t(j) * p.Hkv + kvh) * HD))[lane];
    const float s = simd_sum(dot(qv, kv4));
    const float mn = max(m, s);
    const float corr = metal::fast::exp(m - mn), e = metal::fast::exp(s - mn);
    l = l * corr + e;
    acc = acc * corr + e * ((const device float4*)(vb + (size_t(j) * p.Hkv + kvh) * HD))[lane];
    m = mn;
  }
  const float sk = sinks[h];
  const float mn = max(m, sk);
  const float corr = metal::fast::exp(m - mn);
  l = l * corr + metal::fast::exp(sk - mn);
  acc *= corr;
  ((device float4*)(out + (size_t(r) * p.Hq + h) * HD))[lane] = acc / l;
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
// (NhEmbedParams {D, tok_off P'}), [64] q norm + RoPE (NhQkP {Hq, 128, P', 1, eps, theta}), [128] k (the same, Hkv),
// [192] nh_dsp_attn (NhDspAttnP {P', lo, W, Hq, Hkv, B, scale}), [256] nh_dsp_walk (NhDspWalkP {S, K, R, P', gs}),
// [320] the commit's ring-write grid (m, 2 Hkv, 1). lo = max(0, P' - W, hi - W) as the host computes it (hi: one
// past the highest position the ring held before the commit). One thread. ----
struct NhDspCycP {
  uint P, hi, W, D, Hq, Hkv, B, S, K, R;
  float eps, theta, scale, gs;
};
[[kernel]] void nh_dsp_cycle(const device int* acc [[buffer(0)]], device uint* out [[buffer(1)]],
                             constant NhDspCycP& p [[buffer(2)]]) {
  const uint m = uint(acc[0]) + 1u, P = p.P + m;
  const uint lo = max(P > p.W ? P - p.W : 0u, p.hi > p.W ? p.hi - p.W : 0u);
  device uint* e = out;
  e[0] = p.D;
  e[1] = P;
  for (uint i = 0; i < 2; ++i) {
    device uint* q = out + 64 * (1 + i);
    q[0] = i == 0 ? p.Hq : p.Hkv;
    q[1] = 128u;
    q[2] = P;
    q[3] = 1u;
    q[4] = as_type<uint>(p.eps);
    q[5] = as_type<uint>(p.theta);
  }
  device uint* a = out + 192;
  a[0] = P;
  a[1] = lo;
  a[2] = p.W;
  a[3] = p.Hq;
  a[4] = p.Hkv;
  a[5] = p.B;
  a[6] = as_type<uint>(p.scale);
  device uint* w = out + 256;
  w[0] = p.S;
  w[1] = p.K;
  w[2] = p.R;
  w[3] = P;
  w[4] = as_type<uint>(p.gs);
  device uint* g = out + 320;
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
