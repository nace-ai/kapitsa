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

// Cooperative-tensor lane layouts of matmul2d(16, 32, 16, execution_simdgroup). LAY 0: the Apple10 (M5) layout the
// kernel was written for (lane = rows fm, fm + 8 x 4 consecutive columns fn..fn+3). LAY 1: the pre-Apple10 layout (M1,
// probed with get_multidimensional_index: simdgroup_matrix 8x8 tiles, lane = rows fm, fm + 8 x columns fn, fn+1, fn+8,
// fn+9 with fn = 2 (lane & 1) + 4 ((lane >> 3) & 1); the 16-element right/destination tensors interleave the two 16-col
// halves as [k or m half][c0 0..3 | c1 0..3]; the right operand's rows are k (not n) for both transposes).
template <int LAY> NI_INLINE short ink2a_fn(ushort lane) {
  const short qid = lane >> 2;
  return LAY ? short(2 * (lane & 1) + 4 * ((lane >> 3) & 1)) : short(((qid & 2) | (lane & 1)) * 4);
}
template <int LAY> NI_INLINE short ink2a_col(short fn, short j) { return LAY ? short(fn + (j & 1) + 8 * (j >> 1)) : short(fn + j); }

template <int LAY, typename AT, typename BT, bool TB>
NI_INLINE void ink2a_mma(thread nhf_f8& c0, thread nhf_f8& c1, thread const vec<AT, 8>& a, thread const vec<BT, 8>& b0,
                         thread const vec<BT, 8>& b1) {
  if (LAY == 0) {
    nhf_mma<AT, BT, TB>(c0, c1, a, b0, b1);
    return;
  }
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 16, false, TB, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
  auto cc = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(ca)>,
                                                           remove_addrspace_t<decltype(cb)>, float>();
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i)
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; ++j) {
      ca[4 * i + j] = a[4 * i + j];
      cb[8 * i + j] = b0[4 * i + j];
      cb[8 * i + 4 + j] = b1[4 * i + j];
      cc[8 * i + j] = c0[4 * i + j];
      cc[8 * i + 4 + j] = c1[4 * i + j];
    }
  op.run(ca, cb, cc);
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i)
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; ++j) {
      c0[4 * i + j] = cc[8 * i + j];
      c1[4 * i + j] = cc[8 * i + 4 + j];
    }
}

// K fragment (right operand of S = Q K^T, keys r0 + n, dims c0 + k): LAY 1 holds k = fm + 8i, n = col(j).
// KTR: the K tile is staged transposed ([D][BK + 8] halves), so the M1 layout's 4 keys of one dim are 2 half2 reads.
template <int LD, int LAY, int KTR, int BK>
NI_INLINE nhf_h8 ink2a_load_k(const threadgroup half* base, uint r0, uint c0, short fm, short fn) {
  if (LAY == 0) return qa_load_kv<LD>(base, r0, c0, fm, fn);
  nhf_h8 v;
  if (KTR) {
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i)
      _Pragma("clang loop unroll(full)")
      for (short h = 0; h < 2; ++h) {
        const half2 x = *(const threadgroup half2*)(base + (c0 + fm + 8 * i) * (BK + 8) + r0 + fn + 8 * h);
        v[4 * i + 2 * h] = x.x;
        v[4 * i + 2 * h + 1] = x.y;
      }
    return v;
  }
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i)
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; ++j) v[4 * i + j] = base[(r0 + ink2a_col<1>(fn, j)) * LD + c0 + fm + 8 * i];
  return v;
}
// V fragment (right operand of O += P V, keys r0 + k, dims c0 + n) and any row-major fragment: rows fm + 8i, cols col(j)
template <int LD, int LAY>
NI_INLINE nhf_h8 ink2a_load_v(const threadgroup half* base, uint r0, uint c0, short fm, short fn) {
  if (LAY == 0) return qa_load_kv<LD>(base, r0, c0, fm, fn);
  nhf_h8 v;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i)
    _Pragma("clang loop unroll(full)")
    for (short h = 0; h < 2; ++h) {
      const half2 x = *(const threadgroup half2*)(base + (r0 + fm + 8 * i) * LD + c0 + fn + 8 * h);
      v[4 * i + 2 * h] = x.x;
      v[4 * i + 2 * h + 1] = x.y;
    }
  return v;
}

template <int NB, int BK, int LD, int LAY, int KTR, int QS>
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
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ink2a_fn<LAY>(lane);
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
      const device float* qr = q + (size_t(min(tk[i], p.T - 1)) * p.Hq + hd) * D + id * 16 + fn;
      const float4 x = (LAY ? float4(*(const device float2*)qr, *(const device float2*)(qr + 8))
                            : *(const device float4*)qr) * qscale;
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
      if (KTR) {
        const half4 a = as_type<half4>(kv4.xy), b = as_type<half4>(kv4.zw);
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 4; ++e) {
          kt[(c + e) * (BK + 8) + r] = a[e];
          kt[(c + 4 + e) * (BK + 8) + r] = b[e];
        }
      } else {
        *(threadgroup uint4*)(kt + r * LD + c) = kv4;
      }
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
        const nhf_h8 ka = ink2a_load_k<LD, LAY, KTR, BK>(kt, 16 * f, id * 16, fm, fn);
        const nhf_h8 kb2 = ink2a_load_k<LD, LAY, KTR, BK>(kt, KT > 1 ? 16 * f + 16 : 0, id * 16, fm, fn);
        if (QS == 2) {  // one fp32 product (q hi + lo as float, K's halves widened)
          ink2a_mma<LAY, float, float, true>(sf[f], sf[f + 1], nhf_f8(qf[id]) + nhf_f8(ql[id]), nhf_f8(ka), nhf_f8(kb2));
        } else {
          ink2a_mma<LAY, half, half, true>(sf[f], sf[f + 1], qf[id], ka, kb2);
          if (QS) ink2a_mma<LAY, half, half, true>(sf[f], sf[f + 1], ql[id], ka, kb2);
        }
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
          const uint key = k0 + 16 * f + ink2a_col<LAY>(fn, j);
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
        const nhf_h8 v0 = ink2a_load_v<LD, LAY>(vt, 16 * f, id * 16, fm, fn);
        const nhf_h8 v1 = ink2a_load_v<LD, LAY>(vt, 16 * f, id * 16 + 16, fm, fn);
        ink2a_mma<LAY, half, half, false>(o[id], o[id + 1], pf, v0, v1);
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
      const float4 g4 = LAY ? float4(*(const device float2*)(gate + off), *(const device float2*)(gate + off + 8))
                            : *(const device float4*)(gate + off);
      const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                               metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
      if (LAY) {
        *(device float2*)(out + off) = (r * sig).xy;
        *(device float2*)(out + off + 8) = (r * sig).zw;
      } else {
        *(device float4*)(out + off) = r * sig;
      }
    }
  }
}

#define QATTN_INK2(NAME, NB, BK, LD, LAY, KTR, QS)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(32 * (NB) * 3)]] void NAME(                                              \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],   \
      const device float* rb [[buffer(3)]], const device float* sinks [[buffer(4)]],                                   \
      const device float* gate [[buffer(5)]], device float* out [[buffer(6)]], constant Ink2AttnTP& p [[buffer(7)]],   \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                           \
      uint lane [[thread_index_in_simdgroup]]) {                                                                       \
    threadgroup half kt[(KTR) ? 128 * ((BK) + 8) : (BK) * (LD)], vt[BK * LD];                                                                         \
    qattn_ink2_impl<NB, BK, LD, LAY, KTR, QS>(q, kc, vc, rb, sinks, gate, out, p, tg, sg, ushort(lane), kt, vt);                     \
  }

QATTN_INK2(qattn_ink2_s6k32p, 2, 32, 136, 0, 0, 1)
QATTN_INK2(qattn_ink2_s3k16, 1, 16, 128, 0, 0, 1)
QATTN_INK2(qattn_ink2_s6k32p_l1, 2, 32, 136, 1, 0, 1)
QATTN_INK2(qattn_ink2_s3k16_l1, 1, 16, 128, 1, 0, 1)
