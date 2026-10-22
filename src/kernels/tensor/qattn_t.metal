#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- qattn_t: Qwen3.5/3.6 prefill attention (head dim 256, 4 or 8 query heads per KV head, sigmoid output gate) on the GPU's
// matrix units (Metal 4 tensor_ops; Apple10 = M5). Buffers and result as attn_prefill11r (q fp32 [T][Hq][256], half KV
// cache [Hkv][max_seq][256], gate in qg [T][Hq][2][256], fp32 out [T][Hq][256]); products as nh_attn_t.metal (16 x 16
// fragments in registers, nhf_mma). A simdgroup takes 4 consecutive tokens x the KV head's 4 query heads = 16 rows
// (row r = token t0 + r / 4, head kvh * 4 + r % 4); the rows' causal limits differ by token.
// grid: (ceil(T / (4 NS)), Hkv) x 32 NS threads.

struct QaParams {  // = AttnPrefParams
  uint T, Hq, Hkv, pos0, max_seq;
  float scale;
};


// a fragment of 16 tile rows r0 + [0, 16) x 16 columns c0 + [0, 16) from threadgroup memory
template <int LD>
NI_INLINE nhf_h8 qa_load_kv(const threadgroup half* base, uint r0, uint c0, short fm, short fn) {
  nhf_h8 v;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const half4 x = *(const threadgroup half4*)(base + (r0 + fm + 8 * i) * LD + c0 + fn);
    v[4 * i] = x.x;
    v[4 * i + 1] = x.y;
    v[4 * i + 2] = x.z;
    v[4 * i + 3] = x.w;
  }
  return v;
}

template <int NS, int BK, int LD, int G>   // G query heads per KV head (4 on the 4B, 8 on the 35B)
NI_INLINE void qattn_t_impl(const device float* q, const device half* kc, const device half* vc, const device float* qg,
                            device float* out, constant QaParams& p, uint2 tg, uint sg, ushort lane, threadgroup half* kt,
                            threadgroup half* vt) {
  constexpr int D = 256, KT = BK / 16, TT = 16 / G, SH = G == 8 ? 3 : 2;   // TT tokens x G heads per 16-row tile
  const uint t0 = (tg.x * NS + sg) * TT, kvh = tg.y;
  const bool live = t0 < p.T;
  const uint tid = sg * 32 + lane;
  const uint tgl = p.pos0 + min(tg.x * NS * TT + NS * TT - 1, p.T - 1);   // the threadgroup's last key
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  uint tk[2], hd[2], lim[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const short r = fm + 8 * i;
    tk[i] = t0 + (r >> SH);
    hd[i] = kvh * G + (r & (G - 1));
    lim[i] = p.pos0 + min(tk[i], p.T - 1);
  }
  const uint first = p.pos0 + t0;                    // the block's first token: its limit
    nhf_h8 qf[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x = *(const device float4*)(q + (size_t(min(tk[i], p.T - 1)) * p.Hq + hd[i]) * D + id * 16 + fn) * p.scale;
      qf[id][4 * i] = half(x.x);
      qf[id][4 * i + 1] = half(x.y);
      qf[id][4 * i + 2] = half(x.z);
      qf[id][4 * i + 3] = half(x.w);
    }
  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
  const device half* kb = kc + size_t(kvh) * p.max_seq * D;
  const device half* vb = vc + size_t(kvh) * p.max_seq * D;
  for (uint k0 = 0; k0 <= tgl; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous step is done with the tiles
    for (uint i = tid; i < BK * 32; i += 32 * NS) {   // BK rows x 32 half8
      const uint r = i >> 5, c = (i & 31) * 8;
      const bool ok = k0 + r <= tgl;
      *(threadgroup uint4*)(kt + r * LD + c) = ok ? *(const device uint4*)(kb + size_t(k0 + r) * D + c) : uint4(0);
      *(threadgroup uint4*)(vt + r * LD + c) = ok ? *(const device uint4*)(vb + size_t(k0 + r) * D + c) : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;   // (tile loads and barriers are uniform over the threadgroup)
    nhf_f8 sf[KT];
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) sf[f] = nhf_f8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; f += 2)
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) {
        const nhf_h8 ka = qa_load_kv<LD>(kt, 16 * f, id * 16, fm, fn);
        const nhf_h8 kb2 = qa_load_kv<LD>(kt, 16 * f + 16, id * 16, fm, fn);
        nhf_mma<half, half, true>(sf[f], sf[f + 1], qf[id], ka, kb2);
      }
    const bool diag = k0 + BK - 1 > first;
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; ++j) {
          float v = sf[f][4 * i + j] * M_LOG2E_F;
          if (diag && k0 + 16 * f + fn + j > lim[i]) v = -1e30f;
          sf[f][4 * i + j] = v;
          mx = max(mx, v);
        }
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float mn = max(m[i], mx);
      corr[i] = metal::fast::exp2(m[i] - mn);
      m[i] = mn;
      float ps = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; ++j) {
          sf[f][4 * i + j] = metal::fast::exp2(sf[f][4 * i + j] - mn);
          ps += sf[f][4 * i + j];
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
    const size_t row = size_t(tk[i]) * p.Hq + hd[i];
    const device float* gate = qg + row * 2 * D + D;
    device float* dst = out + row * D;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id) {
      const float4 g = *(const device float4*)(gate + id * 16 + fn);
      const float4 r = float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv;
      *(device float4*)(dst + id * 16 + fn) = float4(r.x * sigmoid(g.x), r.y * sigmoid(g.y), r.z * sigmoid(g.z), r.w * sigmoid(g.w));
    }
  }
}

#define QATTN_T(NAME, NS, BK, LD, G)                                                                                  \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                                \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      const device float* qg [[buffer(3)]], device float* out [[buffer(4)]], constant QaParams& p [[buffer(5)]],     \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]]) {                                                                     \
    threadgroup half kt[BK * LD], vt[BK * LD];                                                                 \
    qattn_t_impl<NS, BK, LD, G>(q, kc, vc, qg, out, p, tg, sg, ushort(lane), kt, vt);                                 \
  }
QATTN_T(qattn_t_s8k16, 8, 16, 256, 4)
QATTN_T(qattn_t_s4k16, 4, 16, 256, 4)
QATTN_T(qattn_t_s16k16, 16, 16, 256, 4)
QATTN_T(qattn_t_s2k16, 2, 16, 256, 4)
// G = 8 (Qwen3.6-35B-A3B, 16 query heads over 2 KV heads): a 16-row tile holds 2 tokens x the KV head's 8
// query heads, so a threadgroup of NS simdgroups covers 2 NS tokens. s4 (8 tokens per threadgroup, 128
// threads) was the best of the shapes measured at T 2112 against 16493 keys: s4 56.63 ms, s8 61.46, s16 58.40,
// against attn_prefill11r's 79.64 ms; BK 32 (s8 79.91, s16 69.40) is slower, its two tiles need 32 KB of
// threadgroup memory. A shape dropped from this file needs its line back for tools/bench_attn.cpp, which
// looks kernels up by name.
QATTN_T(qattn_t_g8s4k16, 4, 16, 256, 8)
