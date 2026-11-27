#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- Inkling-2-Mini prefill attention on the GPU's matrix units (Metal 4 tensor_ops; Apple10+) ----
//
// Mirrors qattn_t_flint.metal (16x16 register tiles, nhf_mma from nh_attn_t.metal, qa_load_kv from qattn_t.metal;
// filename sorts after both) for Inkling-2's attention geometry: D = 128, G = Hq / Hkv = 3 (24 query / 8 KV heads),
// [kv_cap, Hkv, D] half KV cache (ring buffer when is_swa != 0), per-(token, head) float sinks, learned relative
// position bias rb [T, Hq, rel_extent], and sigmoid output gate [T, Hq, D] -> float out [T, Hq, D].
// Each simdgroup takes 16 consecutive tokens of 1 query head; a threadgroup of NB * 3 simdgroups takes 16 * NB tokens
// across all 3 query heads of 1 KV head, reusing each BK-key K/V tile 16 * NB * 3 times.
// Online softmax in base 2 with fp32 statistics. grid: (ceil(T / (16 * NB)), Hkv) x 32 * NB * 3 threads.

struct Ink2AttnTP {  // = Ink2AttnPfP (inkling2/ink2.metal)
  uint T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa;
  float scale;
};

template <int NB, int BK, int LD>
NI_INLINE void qattn_ink2_impl(const device float* q, const device half* kc, const device half* vc,
                               const device float* rb, const device float* sinks, const device float* gate,
                               device float* out, constant Ink2AttnTP& p, uint2 tg, uint sg, ushort lane,
                               threadgroup half* kt, threadgroup half* vt) {
  constexpr int D = 128, G = 3, NS = NB * G, KT = BK / 16, T_TG = 16 * NB;
  const uint ntg = (p.T + T_TG - 1) / T_TG;
  const uint kvh = tg.y, tgt0 = (ntg - 1 - tg.x) * T_TG;
  const uint tb = sg / G, gh = sg % G;
  const uint t0 = tgt0 + tb * 16;
  const uint hd = kvh * G + gh;
  const bool live = t0 < p.T;
  const uint tid = sg * 32 + lane;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  uint tk[2], lim[2], lo[2];
  const device float* rbh[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    tk[i] = t0 + fm + 8 * i;
    const uint t_clamped = min(tk[i], p.T - 1);
    lim[i] = p.pos0 + t_clamped;
    lo[i] = (p.is_swa && lim[i] + 1 > p.swa) ? (lim[i] + 1 - p.swa) : 0u;
    rbh[i] = rb + (size_t(t_clamped) * p.Hq + hd) * p.rel_extent;
  }
  const uint kend = p.pos0 + min(tgt0 + T_TG, p.T);
  const uint kfirst = (p.is_swa && p.pos0 + tgt0 + 1 > p.swa) ? (p.pos0 + tgt0 + 1 - p.swa) : 0u;
  const uint sfirst = p.pos0 + min(t0, p.T - 1);
  const uint slast = p.pos0 + min(t0 + 15, p.T - 1);
  const uint slo = (p.is_swa && slast + 1 > p.swa) ? (slast + 1 - p.swa) : 0u;
  const uint sflo = (p.is_swa && sfirst + 1 > p.swa) ? (sfirst + 1 - p.swa) : 0u;

  const float qscale = p.scale * M_LOG2E_F;
  nhf_h8 qf[D / 16], ql[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x =
          *(const device float4*)(q + (size_t(min(tk[i], p.T - 1)) * p.Hq + hd) * D + id * 16 + fn) * qscale;
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; ++j) {
        qf[id][4 * i + j] = half(x[j]);
        ql[id][4 * i + j] = half(x[j] - float(qf[id][4 * i + j]));
      }
    }

  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2], l[2] = {1.0f, 1.0f};
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) m[i] = sinks[size_t(min(tk[i], p.T - 1)) * p.Hq + hd] * M_LOG2E_F;

  for (uint k0 = kfirst / BK * BK; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * (D / 8); i += 32 * NS) {
      const uint r = i / (D / 8), c = (i % (D / 8)) * 8, key = k0 + r;
      uint4 kv4 = uint4(0), vv4 = uint4(0);
      if (key >= kfirst && key < kend) {
        const uint slot = p.is_swa ? (key % p.kv_cap) : key;
        const size_t at = (size_t(slot) * p.Hkv + kvh) * D + c;
        kv4 = *(const device uint4*)(kc + at);
        vv4 = *(const device uint4*)(vc + at);
      }
      *(threadgroup uint4*)(kt + r * LD + c) = kv4;
      *(threadgroup uint4*)(vt + r * LD + c) = vv4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || k0 > slast || k0 + BK <= sflo) continue;

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

    const bool edge = k0 + BK - 1 > sfirst || k0 < slo;
    const bool near_rb = p.rel_extent > 0 && (k0 + BK + p.rel_extent > sfirst);
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; ++j) {
          float v = sf[f][4 * i + j];
          const uint key = k0 + 16 * f + fn + j;
          if (edge && (key > lim[i] || key < lo[i])) {
            v = -1e30f;
          } else if (near_rb) {
            const uint dist = lim[i] - key;
            if (dist < p.rel_extent) v += rbh[i][dist] * M_LOG2E_F;
          }
          sf[f][4 * i + j] = v;
          mx = max(mx, v);
        }
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float mn = max(m[i], mx);
      corr[i] = exp2(m[i] - mn);
      m[i] = mn;
      float ps = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; ++j) {
          const float v = sf[f][4 * i + j];
          const float pv = v <= -1e29f ? 0.0f : exp2(v - mn);
          sf[f][4 * i + j] = pv;
          ps += pv;
        }
      ps += simd_shuffle_xor(ps, ushort(1));
      ps += simd_shuffle_xor(ps, ushort(8));
      l[i] = l[i] * corr[i] + ps;
    }
    if (!simd_all(corr[0] == 1.0f && corr[1] == 1.0f)) {
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id)
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; ++e) o[id][e] *= corr[e >> 2];
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

  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    if (!live || tk[i] >= p.T) continue;
    const float inv = 1.0f / l[i];
    const size_t base = (size_t(tk[i]) * p.Hq + hd) * D;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id) {
      const size_t off = base + id * 16 + fn;
      const float4 r = float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv;
      const float4 g4 = *(const device float4*)(gate + off);
      const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                               metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
      *(device float4*)(out + off) = r * sig;
    }
  }
}

#define QATTN_INK2(NAME, NB, BK, LD)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(32 * (NB) * 3)]] void NAME(                                              \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],   \
      const device float* rb [[buffer(3)]], const device float* sinks [[buffer(4)]],                                   \
      const device float* gate [[buffer(5)]], device float* out [[buffer(6)]], constant Ink2AttnTP& p [[buffer(7)]],   \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                           \
      uint lane [[thread_index_in_simdgroup]]) {                                                                       \
    threadgroup half kt[BK * LD], vt[BK * LD];                                                                         \
    qattn_ink2_impl<NB, BK, LD>(q, kc, vc, rb, sinks, gate, out, p, tg, sg, ushort(lane), kt, vt);                     \
  }

QATTN_INK2(qattn_ink2_s6k32p, 2, 32, 136)
QATTN_INK2(qattn_ink2_s3k16, 1, 16, 128)
