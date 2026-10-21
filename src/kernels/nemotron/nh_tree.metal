// KFlash trees for the Nemotron-H block drafters (DSpark, LiLiCorr): each drafter's candidate lattice as conditional
// log-probabilities (tlq = the shared builder's children-rows), and the accepted path's verify features for the
// drafter's ring commit. The tree descriptor, builder and acceptance are src/kernels/kflash.h's (kf_build, kf_accept);
// the verify's tree kernels (nh_ssm1_tree, nh_attn_partial_tree, nh_kv_tree_compact) are in nh_ops.metal.

// ---- DSpark's lattice: tlq[s][p][c] = the Markov head's corrected log-probability of candidate c of slot s after
// candidate p of slot s - 1 (slot 0: after the anchor, p = 0), normalized over the slot's K candidates plus the mass
// outside them as nh_dsp_walk normalizes its picks. grid: (1 + (S - 1) K) x (32 K) threads: threadgroup g is slot 0
// for g = 0, else slot 1 + (g - 1) / K after candidate (g - 1) % K; simdgroup c computes candidate c. ----
struct NhDspTableP {
  uint S, K, R, P;
  float gs;
};
[[kernel]] void nh_dsp_table(const device int* cand [[buffer(0)]], const device float* clp [[buffer(1)]],
                             const device ushort* w1 [[buffer(2)]], const device uchar* w2q [[buffer(3)]],
                             const device uchar* w2s [[buffer(4)]], const device int* tokens [[buffer(5)]],
                             device float* tlq [[buffer(6)]], constant NhDspTableP& p [[buffer(7)]],
                             uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                             uint c [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float e[1024];
  threadgroup float sc[32];
  const uint s = g == 0 ? 0u : 1u + (g - 1) / p.K, pp = g == 0 ? 0u : (g - 1) % p.K;
  const uint prev = uint(g == 0 ? tokens[p.P] : cand[(s - 1) * p.K + pp]);
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
    float mx = sc[0], base = 0.0f, den = 0.0f;
    for (uint k = 1; k < p.K; ++k) mx = max(mx, sc[k]);
    for (uint k = 0; k < p.K; ++k) {
      den += metal::precise::exp(sc[k] - mx);
      base += metal::precise::exp(clp[s * p.K + k]);
    }
    den += max(0.0f, 1.0f - base) * metal::precise::exp(-mx);
    const float lden = mx + metal::precise::log(den);
    for (uint k = 0; k < p.K; ++k) tlq[(s * p.K + pp) * p.K + k] = sc[k] - lden;
  }
}

// ---- LiLiCorr's lattice: tlq[s][p][c] = log softmax over the slot's K candidates of the head's transition scores
// (slot 0: ls in[0, c] . aout against the anchor; slot s: ls in[s, c] . out[s - 1, p]), plus the log of the drafter's
// probability mass on those K candidates (clp: log-probabilities over its full head), so that a slot whose K
// candidates miss the target's token is not counted as certain. grid: as nh_dsp_table. ----
struct NhLiliTableP {
  uint S, K, D;
  float ls;
};
[[kernel]] void nh_lili_table(const device float* in [[buffer(0)]], const device float* outv [[buffer(1)]],
                              const device float* aout [[buffer(2)]], const device float* clp [[buffer(3)]],
                              device float* tlq [[buffer(4)]], constant NhLiliTableP& p [[buffer(5)]],
                              uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint c [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sc[32];
  const uint s = g == 0 ? 0u : 1u + (g - 1) / p.K, pp = g == 0 ? 0u : (g - 1) % p.K;
  const device float* a = in + (size_t(s) * p.K + c) * p.D;
  const device float* b = s == 0 ? aout : outv + (size_t(s - 1) * p.K + pp) * p.D;
  float d = 0.0f;
  for (uint o = lane; o < p.D; o += 32) d += a[o] * b[o];
  d = simd_sum(d);
  if (lane == 0) sc[c] = p.ls * d;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float mx = sc[0], den = 0.0f, mass = 0.0f;
    for (uint k = 1; k < p.K; ++k) mx = max(mx, sc[k]);
    for (uint k = 0; k < p.K; ++k) {
      den += metal::precise::exp(sc[k] - mx);
      mass += metal::precise::exp(clp[s * p.K + k]);
    }
    const float l = mx + metal::precise::log(den) - metal::precise::log(min(1.0f, mass));
    for (uint k = 0; k < p.K; ++k) tlq[(s * p.K + pp) * p.K + k] = sc[k] - l;
  }
}

// (the tree builder and the tree's acceptance are the shared kf_build / kf_accept of src/kernels/kflash.h)

// ---- the accepted path's verify features (rows path[0 .. m) of feat) as rows 0 .. m of out, for the drafter's ring
// commit (the rows plain decoding of those tokens would have captured). grid: (m, ceil(ld / 256)) x 256 ----
struct NhKfGatherP {
  uint ld;
};
[[kernel]] void nh_kf_gather(const device float* feat [[buffer(0)]], device float* out [[buffer(1)]],
                             const device int* path [[buffer(2)]], constant NhKfGatherP& p [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint i = tg.x, c = tg.y * 256 + t2.x;
  if (c < p.ld) out[size_t(i) * p.ld + c] = feat[size_t(path[i]) * p.ld + c];
}
