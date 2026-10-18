// Nemotron-H prefill attention: causal flash attention, head dim 128, GQA, no positional encoding.
// One threadgroup = 32 query rows of one query head (4 simdgroups x 8 rows); K/V tiles of 32 keys from the cache
// (half [Hkv][cap][128]) staged in threadgroup memory; online softmax in base 2 with fp32 statistics.
// q: fp32 [T][Hq * 128] (rows at positions pos0 + t); out: half [T][Hq * 128]. grid: (ceil(T / 32), Hq) x 128.
struct NhAttnPfP {
  uint T, pos0, cap, Hq, Hkv;
  float scale;
};

constant constexpr uint kPfD = 128, kPfLD = kPfD + 8, kPfBQ = 32, kPfBK = 32;

[[kernel, max_total_threads_per_threadgroup(128)]] void nh_attn_prefill(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device half* out [[buffer(3)]], constant NhAttnPfP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half qs[kPfBQ * kPfLD];
  threadgroup half kt[kPfBK * kPfLD];
  threadgroup half vt[kPfBK * kPfLD];
  const uint qb = tg.x, h = tg.y, kvh = h / (p.Hq / p.Hkv);
  const uint t0 = qb * kPfBQ;
  const float qscale = p.scale * 1.4426950408889634f;  // log2(e): exp2 below
  // Q tile -> half (pre-scaled)
  for (uint i = tid; i < kPfBQ * kPfD / 4; i += 128) {
    const uint r = i / (kPfD / 4), c = (i % (kPfD / 4)) * 4;
    float4 v = t0 + r < p.T ? *(const device float4*)(q + (size_t(t0 + r) * p.Hq + h) * kPfD + c) : float4(0);
    v *= qscale;
    *(threadgroup half4*)(qs + r * kPfLD + c) = half4(v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[kPfD / 8];
  for (uint d = 0; d < kPfD / 8; ++d) simdgroup_load(qf[d], qs + (sg * 8) * kPfLD + d * 8, kPfLD);
  simdgroup_float8x8 o[kPfD / 8];
  for (uint d = 0; d < kPfD / 8; ++d) o[d] = simdgroup_float8x8(0);
  // this lane's row of every 8x8 fragment and its two columns
  const uint qq = lane / 4;
  const uint rr = (qq & 4) + (lane / 2) % 4, cc = (qq & 2) * 2 + (lane % 2) * 2;
  const uint my_t = t0 + sg * 8 + rr;                  // query row of this lane's elements
  const uint my_lim = p.pos0 + my_t;                   // last key it may see
  float m = -1e30f, l = 0.0f;
  const uint t_last = min(t0 + kPfBQ, p.T) - 1;
  const uint kend = p.pos0 + t_last + 1;               // keys the tile's last row sees
  const uint kmin_full = p.pos0 + t0;                  // keys below this are visible to every row of the tile
  const device half* kb = kc + size_t(kvh) * p.cap * kPfD;
  const device half* vb = vc + size_t(kvh) * p.cap * kPfD;
  for (uint k0 = 0; k0 < kend; k0 += kPfBK) {
    for (uint i = tid; i < kPfBK * kPfD / 8; i += 128) {
      const uint r = i / (kPfD / 8), c = (i % (kPfD / 8)) * 8;
      const bool ok = k0 + r < kend;
      *(threadgroup uint4*)(kt + r * kPfLD + c) = ok ? *(const device uint4*)(kb + size_t(k0 + r) * kPfD + c) : uint4(0);
      *(threadgroup uint4*)(vt + r * kPfLD + c) = ok ? *(const device uint4*)(vb + size_t(k0 + r) * kPfD + c) : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 sf[kPfBK / 8];
    for (uint j = 0; j < kPfBK / 8; ++j) {
      sf[j] = simdgroup_float8x8(0);
      for (uint d = 0; d < kPfD / 8; ++d) {
        simdgroup_half8x8 kf;
        simdgroup_load(kf, kt + (j * 8) * kPfLD + d * 8, kPfLD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
      }
    }
    // mask (diagonal tiles only) and the tile's row max
    const bool diag = k0 + kPfBK > kmin_full;
    float tmax = -1e30f;
    for (uint j = 0; j < kPfBK / 8; ++j)
      for (uint e = 0; e < 2; ++e) {
        float v = sf[j].thread_elements()[e];
        const uint key = k0 + j * 8 + cc + e;
        if (diag && (key > my_lim || key >= kend)) v = -1e30f;
        sf[j].thread_elements()[e] = v;
        tmax = max(tmax, v);
      }
    tmax = max(tmax, simd_shuffle_xor(tmax, 1));
    tmax = max(tmax, simd_shuffle_xor(tmax, 8));
    const float mn = max(m, tmax);
    const float corr = exp2(m - mn);
    float psum = 0.0f;
    simdgroup_half8x8 pf[kPfBK / 8];
    for (uint j = 0; j < kPfBK / 8; ++j)
      for (uint e = 0; e < 2; ++e) {
        const float v = sf[j].thread_elements()[e];
        const float pv = v <= -1e29f ? 0.0f : exp2(v - mn);
        psum += pv;
        pf[j].thread_elements()[e] = half(pv);
      }
    psum += simd_shuffle_xor(psum, 1);
    psum += simd_shuffle_xor(psum, 8);
    l = l * corr + psum;
    m = mn;
    for (uint d = 0; d < kPfD / 8; ++d) {
      o[d].thread_elements()[0] *= corr;
      o[d].thread_elements()[1] *= corr;
    }
    for (uint j = 0; j < kPfBK / 8; ++j)
      for (uint d = 0; d < kPfD / 8; ++d) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, vt + (j * 8) * kPfLD + d * 8, kPfLD);
        simdgroup_multiply_accumulate(o[d], pf[j], vf, o[d]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (my_t < p.T) {
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    device half* dst = out + (size_t(my_t) * p.Hq + h) * kPfD;
    for (uint d = 0; d < kPfD / 8; ++d)
      for (uint e = 0; e < 2; ++e) dst[d * 8 + cc + e] = half(o[d].thread_elements()[e] * inv);
  }
}
