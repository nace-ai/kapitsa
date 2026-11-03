#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- attn8t: the 8-row KFlash tree verify's history attention on the GPU's matrix units (Apple10 = M5) ----
//
// attn8m_partial_zone (verify8.metal, TREE 3) on matmul2d: a threadgroup per (history chunk, KV head) serves the KV
// head's G query heads x the verify's 8 rows, every K / V byte of its chunk read for all of them. attn8m runs that on
// 8 x 8 simdgroup MMAs, which execute on the shader ALUs (~15 TFLOP/s on the M5 Max): at long context the verify's
// attention is bound by them (the 35B at 258K: ~36 of a 47 ms cycle). Here the rows are (row t, head h) pairs, t G + h,
// in 16-row tiles; a pair of simdgroups takes a tile, one half of the head dim each (the registers of Q and O for 128
// dims, as qattn_t_d2), and per 32 keys: S = Q K^T over its half (nh_attn_t's 16 x 32 x 16 products), the two halves'
// scores added through threadgroup memory (a + b in one, b + a in the other: the same value), the log2-domain online
// softmax in fp32, O += P V for its half (P as half, as nh_attn_t). The keys, masks and partials are attn8m's (TREE 3):
// row t's keys below tree_zone(its logical position pos0 + depth), in the sequence's own fixed chunks; it writes the
// unnormalized O, the max as a natural log and the sum where attn8m_partial_zone does, for attn8_tree_zone and
// attn_rows_reduce. A row's numbers depend on its position alone (each output of a matmul2d sums its own row's
// products), not on the tree, as attn8m's. Half K / V rows only (NI_KV_I8's int8 rows keep attn8m).
// grid: (nchunks, Hkv) x 32 * 2 NT threads, NT = ceil(8 G / 16) tiles.

struct V8tAttnP {  // = AttnRowsParams (verify.metal)
  uint Hq, Hkv, pos0, chunk, max_seq;
  float scale;
  uint nchunks;
};
constant constexpr uint kV8tKfMask = 65;          // kflash.h's kKfMask: tree[kKfMask + r], bit j: row j is r or an ancestor
constant constexpr float kV8tNeg = -1.0e30f;      // attn.metal's kNeg11
NI_INLINE uint v8t_zone(uint q) { return q >= 7u ? (q - 7u) / 32u * 32u : 0u; }  // verify8.metal's tree_zone

// a fragment of 16 cache rows r0 + [0, 16) x 16 columns c0 + [0, 16) of 256-wide rows (rows past lim read as 0)
NI_INLINE nhf_h8 v8t_load_kv(const device half* base, uint r0, uint c0, uint lim, short fm, short fn) {
  nhf_h8 v;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const uint r = r0 + fm + 8 * i;
    const half4 x = r <= lim ? *(const device half4*)(base + size_t(r) * 256 + c0 + fn) : half4(0);
    v[4 * i] = x.x;
    v[4 * i + 1] = x.y;
    v[4 * i + 2] = x.z;
    v[4 * i + 3] = x.w;
  }
  return v;
}

template <int G>
NI_INLINE void attn8t_zone_impl(const device float* q, const device half* kc, const device half* vc,
                                device float* part_o, device float* part_ml, constant V8tAttnP& p, uint R,
                                const device int* tree, threadgroup float* xs, uint2 tg, uint sg, ushort lane) {
  constexpr int D = 256, DH = 128, BK = 32, KT = BK / 16;
  const uint chunk = tg.x, kvh = tg.y, pi = sg >> 1, dh = sg & 1, d0 = dh * DH;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  // this lane's two rows (t G + h = 16 pi + fm + 8 i): their verify row, query head and key limit
  uint t[2], hq[2], lim[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const uint rr = 16 * pi + fm + 8 * i;
    t[i] = rr / G;
    hq[i] = kvh * G + rr % G;
    const uint anc = t[i] < R ? uint(tree[kV8tKfMask + t[i]]) : 0u;
    lim[i] = t[i] < R ? v8t_zone(p.pos0 + (anc ? uint(popcount(anc)) - 1u : 0u)) : 0u;
  }
  const uint start = chunk * p.chunk, kend = min(start + p.chunk, v8t_zone(p.pos0 + 7u));
  // Q as attn8m's: half(q scale), this simdgroup's 128 dims
  nhf_h8 qf[DH / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < DH / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x = t[i] < R ? *(const device float4*)(q + (size_t(t[i]) * p.Hq + hq[i]) * D + d0 + id * 16 + fn) *
                                      p.scale
                                : float4(0.0f);
      qf[id][4 * i] = half(x.x);
      qf[id][4 * i + 1] = half(x.y);
      qf[id][4 * i + 2] = half(x.z);
      qf[id][4 * i + 3] = half(x.w);
    }
  nhf_f8 o[DH / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < DH / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {kV8tNeg, kV8tNeg}, l[2] = {0.0f, 0.0f};
  const device half* kb = kc + size_t(kvh) * p.max_seq * D;
  const device half* vb = vc + size_t(kvh) * p.max_seq * D;
  threadgroup float* xme = xs + (sg * 32 + lane) * 16;
  const threadgroup float* xpa = xs + ((sg ^ 1) * 32 + lane) * 16;
  const uint last = kend > 0 ? kend - 1 : 0;
  for (uint k0 = start; k0 < kend; k0 += BK) {
    nhf_f8 sf[KT];
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) sf[f] = nhf_f8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < DH / 16; ++id) {
      const nhf_h8 ka = v8t_load_kv(kb, k0, d0 + id * 16, last, fm, fn);
      const nhf_h8 kb2 = v8t_load_kv(kb, k0 + 16, d0 + id * 16, last, fm, fn);
      nhf_mma<half, half, true>(sf[0], sf[1], qf[id], ka, kb2);
    }
    // the pair's two halves of the scores: mine + my partner's
    threadgroup_barrier(mem_flags::mem_threadgroup);  // (the previous step's reads of the exchange are done)
    _Pragma("clang loop unroll(full)")
    for (short e = 0; e < 8; ++e) {
      xme[e] = sf[0][e];
      xme[8 + e] = sf[1][e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (short e = 0; e < 8; ++e) {
      sf[0][e] += xpa[e];
      sf[1][e] += xpa[8 + e];
    }
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = kV8tNeg;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; ++j) {
          float v = sf[f][4 * i + j] * M_LOG2E_F;
          const uint key = k0 + 16 * f + fn + j;
          if (key >= lim[i] || key >= kend) v = kV8tNeg;
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
      for (short id = 0; id < DH / 16; ++id)
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; ++e) o[id][e] *= corr[e >> 2];
    }
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) {
      const nhf_h8 pf = nhf_h8(sf[f]);
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < DH / 16; id += 2) {
        const nhf_h8 v0 = v8t_load_kv(vb, k0 + 16 * f, d0 + id * 16, last, fm, fn);
        const nhf_h8 v1 = v8t_load_kv(vb, k0 + 16 * f, d0 + id * 16 + 16, last, fm, fn);
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
      }
    }
  }
  // attn8m_partial_zone's partials: a row with no key below its zone in this chunk writes O 0, max -inf, sum 0
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    if (t[i] >= R) continue;
    const size_t base = (size_t(t[i]) * p.Hq + hq[i]) * p.nchunks + chunk;
    const bool any = start < lim[i];
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < DH / 16; ++id)
      *(device float4*)(part_o + base * D + d0 + id * 16 + fn) =
          any ? float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) : float4(0.0f);
    if (dh == 0 && fn == 0) {
      part_ml[base * 2 + 0] = any ? m[i] * M_LN2_F : -INFINITY;
      part_ml[base * 2 + 1] = any ? l[i] : 0.0f;
    }
  }
}

#define ATTN8T_ZONE(NAME, G)                                                                                          \
  [[kernel, max_total_threads_per_threadgroup(64 * ((8 * G + 15) / 16))]] void NAME(                                \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      device float* part_o [[buffer(3)]], device float* part_ml [[buffer(4)]], constant V8tAttnP& p [[buffer(5)]],   \
      constant uint& R [[buffer(6)]], const device int* tree [[buffer(7)]],                                          \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                        \
      uint lane [[thread_index_in_simdgroup]]) {                                                                     \
    threadgroup float xs[2 * ((8 * G + 15) / 16) * 32 * 16];                                                         \
    attn8t_zone_impl<G>(q, kc, vc, part_o, part_ml, p, R, tree, xs, tg, sg, ushort(lane));                           \
  }
ATTN8T_ZONE(attn8t_partial_zone_g8, 8)  // Qwen3.6 35B-A3B, Clover: 16 query heads over 2 KV heads, 4 tiles
