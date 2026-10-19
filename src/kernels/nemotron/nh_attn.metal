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

// The same attention with a threadgroup per (8 QR query rows, NSG query heads of one KV head): the K / V tiles (BK
// keys) staged once for NSG heads (one per simdgroup), the Q fragments loaded straight from q (fp32 -> half in
// registers), so the threadgroup memory is the K / V tiles alone, and a first row block qb0 (a long pass splits into
// several dispatches, each in a command buffer of its own). The engine runs QR 1, NSG 4, BK 32: the same values as
// nh_attn_prefill bit for bit. Its fixed-trip loops fully unrolled (the arrays then stay in registers) as
// nh_attn_prefill's. Measured and not faster (tools/nh_x8_bench attn, 2048 rows after 14336
// positions, 327 ms for nh_attn_prefill): 8 heads per threadgroup 334 ms, K staged transposed 330, 16-key tiles (3
// threadgroups per core) 379-382, two row fragments per simdgroup 533-643.
// grid: (row blocks, Hq / NSG) x 32 NSG, row block qb = tg.x + p.qb0; Hq / Hkv % NSG == 0.
struct NhAttnPf2P {
  uint T, pos0, cap, Hq, Hkv;
  float scale;
  uint qb0;
};

template <int QR, int NSG, int BK>
NI_INLINE void nh_attn_pf2_impl(const device float* q, const device half* kc, const device half* vc, device half* out,
                                constant NhAttnPf2P& p, uint2 tg, uint tid, uint sg, uint lane, threadgroup half* kt,
                                threadgroup half* vt) {
  constexpr uint BQ = 8 * QR, NT = 32 * NSG;
  const uint qb = tg.x + p.qb0, h = tg.y * NSG + sg, kvh = h / (p.Hq / p.Hkv);
  const uint t0 = qb * BQ;
  if (t0 >= p.T) return;
  const float qscale = p.scale * 1.4426950408889634f;
  const uint qq = lane / 4;
  const uint rr = (qq & 4) + (lane / 2) % 4, cc = (qq & 2) * 2 + (lane % 2) * 2;
  // Q fragments: rows t0 + 8 f + rr, columns d * 8 + cc (+1), pre-scaled, as half (rows past T read row T - 1)
  simdgroup_half8x8 qf[QR][kPfD / 8];
  NI_UNROLL for (int f = 0; f < QR; ++f) {
    const uint t = min(t0 + 8 * f + rr, p.T - 1);
    const device float* qr = q + (size_t(t) * p.Hq + h) * kPfD + cc;
    NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) {
      const float2 v = *(const device float2*)(qr + d * 8) * qscale;
      qf[f][d].thread_elements()[0] = half(v.x);
      qf[f][d].thread_elements()[1] = half(v.y);
    }
  }
  simdgroup_float8x8 o[QR][kPfD / 8];
  NI_UNROLL for (int f = 0; f < QR; ++f)
    NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) o[f][d] = simdgroup_float8x8(0);
  float m[QR], l[QR];
  uint my_t[QR], my_lim[QR];
  NI_UNROLL for (int f = 0; f < QR; ++f) {
    m[f] = -1e30f;
    l[f] = 0.0f;
    my_t[f] = t0 + 8 * f + rr;
    my_lim[f] = p.pos0 + my_t[f];
  }
  const uint t_last = min(t0 + BQ, p.T) - 1;
  const uint kend = p.pos0 + t_last + 1;
  const uint kmin_full = p.pos0 + t0;
  const device half* kb = kc + size_t(kvh) * p.cap * kPfD;
  const device half* vb = vc + size_t(kvh) * p.cap * kPfD;
  for (uint k0 = 0; k0 < kend; k0 += BK) {
    for (uint i = tid; i < BK * kPfD / 8; i += NT) {
      const uint r = i / (kPfD / 8), c = (i % (kPfD / 8)) * 8;
      const bool ok = k0 + r < kend;
      *(threadgroup uint4*)(kt + r * kPfLD + c) = ok ? *(const device uint4*)(kb + size_t(k0 + r) * kPfD + c) : uint4(0);
      *(threadgroup uint4*)(vt + r * kPfLD + c) = ok ? *(const device uint4*)(vb + size_t(k0 + r) * kPfD + c) : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 sf[QR][BK / 8];
    NI_UNROLL for (uint j = 0; j < BK / 8; ++j) {
      NI_UNROLL for (int f = 0; f < QR; ++f) sf[f][j] = simdgroup_float8x8(0);
      NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) {
        simdgroup_half8x8 kf;
        simdgroup_load(kf, kt + (j * 8) * kPfLD + d * 8, kPfLD, ulong2(0, 0), true);
        NI_UNROLL for (int f = 0; f < QR; ++f) simdgroup_multiply_accumulate(sf[f][j], qf[f][d], kf, sf[f][j]);
      }
    }
    const bool diag = k0 + BK > kmin_full;
    simdgroup_half8x8 pf[QR][BK / 8];
    NI_UNROLL for (int f = 0; f < QR; ++f) {
      float tmax = -1e30f;
      NI_UNROLL for (uint j = 0; j < BK / 8; ++j)
        NI_UNROLL for (uint e = 0; e < 2; ++e) {
          float v = sf[f][j].thread_elements()[e];
          const uint key = k0 + j * 8 + cc + e;
          if (diag && (key > my_lim[f] || key >= kend)) v = -1e30f;
          sf[f][j].thread_elements()[e] = v;
          tmax = max(tmax, v);
        }
      tmax = max(tmax, simd_shuffle_xor(tmax, 1));
      tmax = max(tmax, simd_shuffle_xor(tmax, 8));
      const float mn = max(m[f], tmax);
      const float corr = exp2(m[f] - mn);
      float psum = 0.0f;
      NI_UNROLL for (uint j = 0; j < BK / 8; ++j)
        NI_UNROLL for (uint e = 0; e < 2; ++e) {
          const float v = sf[f][j].thread_elements()[e];
          const float pv = v <= -1e29f ? 0.0f : exp2(v - mn);
          psum += pv;
          pf[f][j].thread_elements()[e] = half(pv);
        }
      psum += simd_shuffle_xor(psum, 1);
      psum += simd_shuffle_xor(psum, 8);
      l[f] = l[f] * corr + psum;
      m[f] = mn;
      NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) {
        o[f][d].thread_elements()[0] *= corr;
        o[f][d].thread_elements()[1] *= corr;
      }
    }
    NI_UNROLL for (uint j = 0; j < BK / 8; ++j)
      NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, vt + (j * 8) * kPfLD + d * 8, kPfLD);
        NI_UNROLL for (int f = 0; f < QR; ++f) simdgroup_multiply_accumulate(o[f][d], pf[f][j], vf, o[f][d]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  NI_UNROLL for (int f = 0; f < QR; ++f)
    if (my_t[f] < p.T) {
      const float inv = l[f] > 0.0f ? 1.0f / l[f] : 0.0f;
      device half* dst = out + (size_t(my_t[f]) * p.Hq + h) * kPfD;
      NI_UNROLL for (uint d = 0; d < kPfD / 8; ++d)
        NI_UNROLL for (uint e = 0; e < 2; ++e) dst[d * 8 + cc + e] = half(o[f][d].thread_elements()[e] * inv);
    }
}

#define NH_ATTN_PF2(NAME, QR, NSG, BK)                                                                             \
  [[kernel, max_total_threads_per_threadgroup(32 * NSG)]] void NAME(                                               \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      device half* out [[buffer(3)]], constant NhAttnPf2P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]]) {                                                                  \
    threadgroup half kt[BK * kPfLD];                                                                               \
    threadgroup half vt[BK * kPfLD];                                                                               \
    nh_attn_pf2_impl<QR, NSG, BK>(q, kc, vc, out, p, tg, tid, sg, lane, kt, vt);                                   \
  }
// name: nh_attn_pf2_q<QR>_h<NSG>_k<BK>
NH_ATTN_PF2(nh_attn_pf2_q1_h4_k32, 1, 4, 32)
NH_ATTN_PF2(nh_attn_pf2_q1_h5_k32, 1, 5, 32)  // 5 query heads per KV head (Nemotron Nano 4B / 9B v2: 40 over 8)
