#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- ink2_attn_t_rows: 2-8 row verify attention on the GPU's matrix units (Metal 4 tensor_ops; Apple10+) ----
// Per (chunk, Hkv), loads each BK-key K/V tile ONCE into threadgroup memory for all 3 query heads x all T<=8 rows
// (24x less KV DRAM traffic than per-(row, head) scalar attention). Hi+lo half Q (~22-bit Q), fp32 online softmax.
// Output partials (pm in natural log, pl, po) match ink2_attn_verify_reduce.
// grid: (nchunks, Hkv, 1) x 96 threads (3 simdgroups, 1 per query head g = 0..2).

struct Ink2AttnVfTP {
  uint T, pos0, k_first, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks;
  float scale;
};

template <int BK, int LD>
NI_INLINE void ink2_attn_t_rows_impl(const device float* q, const device half* kc, const device half* vc,
                                     const device float* rb, device float* pm, device float* pl, device float* po,
                                     constant Ink2AttnVfTP& p, uint2 tg, uint sg, ushort lane,
                                     threadgroup half* kt, threadgroup half* vt) {
  constexpr int D = 128, G = 3, KT = BK / 16, NS = 3;
  const uint cid = tg.x, kvh = tg.y, tid = sg * 32u + lane;
  const uint hd = kvh * G + sg;
  const uint R = min(p.T, 8u);
  const uint cs = p.k_first + cid * p.chunk, ce = min(cs + p.chunk, p.pos0 + R);
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  const uint tk = uint(fm);
  const uint t_clamped = min(tk, R - 1u);
  const uint lim = p.pos0 + t_clamped;
  const device float* rbh = rb + (size_t(t_clamped) * p.Hq + hd) * p.rel_extent;

  const float qscale = p.scale * M_LOG2E_F;
  nhf_h8 qf[D / 16], ql[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) {
    const float4 x = *(const device float4*)(q + (size_t(t_clamped) * p.Hq + hd) * D + id * 16 + fn) * qscale;
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; ++j) {
      qf[id][j] = half(x[j]);
      ql[id][j] = half(x[j] - float(qf[id][j]));
      qf[id][4 + j] = half(0.0h);
      ql[id][4 + j] = half(0.0h);
    }
  }

  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m = -1e30f, l = 0.0f;

  for (uint k0 = cs; k0 < ce; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * (D / 8); i += 32u * NS) {
      const uint r = i / (D / 8), cc = (i % (D / 8)) * 8u, key = k0 + r;
      uint4 kv4 = uint4(0), vv4 = uint4(0);
      if (key < ce) {
        const size_t at = (size_t(key) * p.Hkv + kvh) * D + cc;
        kv4 = *(const device uint4*)(kc + at);
        vv4 = *(const device uint4*)(vc + at);
      }
      *(threadgroup uint4*)(kt + r * LD + cc) = kv4;
      *(threadgroup uint4*)(vt + r * LD + cc) = vv4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    nhf_f8 sf[KT > 1 ? KT : 2];
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < (KT > 1 ? KT : 2); ++f) sf[f] = nhf_f8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; f += 2)
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) {
        const nhf_h8 ka = qa_load_kv<LD>(kt, 16 * f, id * 16, fm, fn);
        const nhf_h8 kb2 = qa_load_kv<LD>(kt, KT > 1 ? 16 * f + 16 : 0, id * 16, fm, fn);
        nhf_mma<half, half, true>(sf[f], sf[f + 1], qf[id], ka, kb2);
        nhf_mma<half, half, true>(sf[f], sf[f + 1], ql[id], ka, kb2);
      }

    const bool near_rb = p.rel_extent > 0 && (k0 + BK + p.rel_extent > p.pos0);
    float mx = -1e30f;
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f)
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; ++j) {
        float v = sf[f][j];
        const uint key = k0 + 16 * f + fn + j;
        if (key >= ce || key > lim) {
          v = -1e30f;
        } else if (near_rb) {
          const uint dist = lim - key;
          if (dist < p.rel_extent) v += rbh[dist] * M_LOG2E_F;
        }
        sf[f][j] = v;
        sf[f][4 + j] = 0.0f;
        mx = max(mx, v);
      }
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float mn = max(m, mx);
    const float corr = (l == 0.0f) ? 0.0f : exp2(m - mn);
    m = mn;
    float ps = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f)
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; ++j) {
        const float v = sf[f][j];
        const float pv = v <= -1e29f ? 0.0f : exp2(v - mn);
        sf[f][j] = pv;
        ps += pv;
      }
    ps += simd_shuffle_xor(ps, ushort(1));
    ps += simd_shuffle_xor(ps, ushort(8));
    l = l * corr + ps;
    if (!simd_all(corr == 1.0f)) {
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id)
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 4; ++e) o[id][e] *= corr;
    }
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) {
      const nhf_h8 pf = nhf_h8(sf[f]);
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; id += 2) {
        const nhf_h8 v0 = qa_load_kv<LD>(vt, 16 * f, id * 16, fm, fn);
        const nhf_h8 v1 = qa_load_kv<LD>(vt, 16 * f, id * 16 + 16, fm, fn);
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
      }
    }
  }

  if (tk < R) {
    const size_t idx = (size_t(tk) * p.Hq + hd) * p.nchunks + cid;
    if (fn == 0) {
      pm[idx] = (l == 0.0f) ? -INFINITY : (m * M_LN2_F);
      pl[idx] = l;
    }
    device float* dst = po + idx * D;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id) {
      *(device float4*)(dst + id * 16 + fn) = float4(o[id][0], o[id][1], o[id][2], o[id][3]);
    }
  }
}

#define INK2_ATTN_T_ROWS(NAME, BK, LD)                                                                                 \
  [[kernel, max_total_threads_per_threadgroup(96)]] void NAME(                                                         \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],   \
      const device float* rb [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],            \
      device float* po [[buffer(6)]], constant Ink2AttnVfTP& p [[buffer(7)]],                                          \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                           \
      uint lane [[thread_index_in_simdgroup]]) {                                                                       \
    threadgroup half kt[BK * LD], vt[BK * LD];                                                                         \
    ink2_attn_t_rows_impl<BK, LD>(q, kc, vc, rb, pm, pl, po, p, tg, sg, ushort(lane), kt, vt);                         \
  }

INK2_ATTN_T_ROWS(ink2_attn_t_rows_k32, 32, 136)
INK2_ATTN_T_ROWS(ink2_attn_t_rows_k16, 16, 128)
