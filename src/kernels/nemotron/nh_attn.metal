// Nemotron-H prefill attention: causal flash attention, head dim 128, GQA, no positional encoding.
// One threadgroup = 32 query rows of one query head (4 simdgroups x 8 rows); K/V tiles of 32 keys from the cache
// (half [Hkv][cap][128]) staged in threadgroup memory; online softmax in base 2 with fp32 statistics. Q goes from
// device memory straight into each lane's fragment elements, and every fixed-trip loop is fully unrolled so the
// fragment arrays stay in registers (as attn.metal's v11 found): 2.1x the rolled loops' speed (Lightning's 16K-token
// prefill 8.9 against 16.0 s), the same arithmetic (bitwise-equal output).
// q: fp32 [T][Hq * 128] (rows at positions pos0 + t); out: half [T][Hq * 128]. grid: (ceil(T / 32), Hq) x 128.
struct NhAttnPfP {
  uint T, pos0, cap, Hq, Hkv;
  float scale;
};

constant constexpr uint kPfD = 128, kPfLD = kPfD + 8, kPfBQ = 32, kPfBK = 32;

#ifndef NI_UNROLL
#define NI_UNROLL _Pragma("clang loop unroll(full)")
#endif

NI_INLINE void nh_attn_prefill_impl(const device float* q, const device half* kc, const device half* vc,
                                    device half* out, constant NhAttnPfP& p, uint2 tg, uint tid, uint sg, uint lane,
                                    threadgroup half* kt, threadgroup half* vt) {
  const uint qb = tg.x, h = tg.y, kvh = h / (p.Hq / p.Hkv);
  const uint t0 = qb * kPfBQ, sr = sg;  // this simdgroup: rows t0 + 8 sr ..
  const float qscale = p.scale * 1.4426950408889634f;
  const uint qq = lane / 4;
  const uint rr = (qq & 4) + (lane / 2) % 4, cc = (qq & 2) * 2 + (lane % 2) * 2;
  const uint my_t = t0 + sr * 8 + rr;
  simdgroup_half8x8 qf[kPfD / 8];
  {
    const bool ok = my_t < p.T;
    const device float* qr = q + (size_t(ok ? my_t : 0) * p.Hq + h) * kPfD + cc;
    NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) {
      const float2 v = ok ? *(const device float2*)(qr + d * 8) : float2(0);
      qf[d].thread_elements()[0] = half(v.x * qscale);
      qf[d].thread_elements()[1] = half(v.y * qscale);
    }
  }
  simdgroup_float8x8 o[kPfD / 8];
  NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) o[d] = simdgroup_float8x8(0);
  const uint my_lim = p.pos0 + my_t;
  float m = -1e30f, l = 0.0f;
  const uint t_last = min(t0 + kPfBQ, p.T) - 1;
  const uint kend = p.pos0 + t_last + 1;
  const uint kmin_full = p.pos0 + t0;
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
    // (the 4 key fragments' chains interleaved; each accumulator takes its 16 MMAs in order)
    simdgroup_float8x8 sf[kPfBK / 8];
    NI_UNROLL for (uint j = 0; j < kPfBK / 8; ++j) sf[j] = simdgroup_float8x8(0);
    NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d)
        NI_UNROLL for (uint j = 0; j < kPfBK / 8; ++j) {
        simdgroup_half8x8 kf;
        simdgroup_load(kf, kt + (j * 8) * kPfLD + d * 8, kPfLD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
      }
    const bool diag = k0 + kPfBK > kmin_full;
    float tmax = -1e30f;
    NI_UNROLL for (uint j = 0; j < kPfBK / 8; ++j)
      NI_UNROLL for (uint e = 0; e < 2; ++e) {
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
    NI_UNROLL for (uint j = 0; j < kPfBK / 8; ++j)
      NI_UNROLL for (uint e = 0; e < 2; ++e) {
        const float v = sf[j].thread_elements()[e];
        const float pv = v <= -1e29f ? 0.0f : exp2(v - mn);
        psum += pv;
        pf[j].thread_elements()[e] = half(pv);
      }
    psum += simd_shuffle_xor(psum, 1);
    psum += simd_shuffle_xor(psum, 8);
    l = l * corr + psum;
    m = mn;
    NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) {
      o[d].thread_elements()[0] *= corr;
      o[d].thread_elements()[1] *= corr;
    }
    NI_UNROLL for (uint j = 0; j < kPfBK / 8; ++j)
      NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, vt + (j * 8) * kPfLD + d * 8, kPfLD);
        simdgroup_multiply_accumulate(o[d], pf[j], vf, o[d]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (my_t < p.T) {
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    device half* dst = out + (size_t(my_t) * p.Hq + h) * kPfD;
    NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d)
      NI_UNROLL for (uint e = 0; e < 2; ++e) dst[d * 8 + cc + e] = half(o[d].thread_elements()[e] * inv);
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void nh_attn_prefill(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device half* out [[buffer(3)]], constant NhAttnPfP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half kt[kPfBK * kPfLD];
  threadgroup half vt[kPfBK * kPfLD];
  nh_attn_prefill_impl(q, kc, vc, out, p, tg, tid, sg, lane, kt, vt);
}
