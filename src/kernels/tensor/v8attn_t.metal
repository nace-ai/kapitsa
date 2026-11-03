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

// TREE 3: attn8m_partial_zone's keys and masks (above). TREE 1, a wide tree verify (attn8m_partial_tree's tiles: 16 or
// 32 rows as 8-row tiles, the grid's x the chunks x the live tiles, tiles fastest): a threadgroup takes one 8-row tile
// of a chunk, its rows see the history (< pos0) and of the block their ancestors and themselves (the mask's bits), the
// keys up to pos0 + R.
template <int G, int TREE>
NI_INLINE void attn8t_impl(const device float* q, const device half* kc, const device half* vc,
                           device float* part_o, device float* part_ml, constant V8tAttnP& p, uint R,
                           const device int* tree, threadgroup float* xs, uint2 tg, uint ntgx, uint sg, ushort lane) {
  constexpr int D = 256, DH = 128, BK = 32, KT = BK / 16;
  const uint nt = TREE == 1 && ntgx > p.nchunks ? ntgx / p.nchunks : 1u;
  const uint chunk = nt > 1 ? tg.x / nt : tg.x, tile = nt > 1 ? tg.x % nt : 0u;
  if (TREE == 1 && R > 8) R = min(R, 8 * nt);  // (a wide verify: the live tiles' rows)
  const uint kvh = tg.y, pi = sg >> 1, dh = sg & 1, d0 = dh * DH;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  // this lane's two rows (t G + h = 16 pi + fm + 8 i within the tile's 8 rows): their verify row, query head, ancestor
  // mask and (TREE 3) key limit
  uint t[2], hq[2], anc[2], lim[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const uint rr = 16 * pi + fm + 8 * i;
    t[i] = rr < 8u * G ? 8 * tile + rr / G : R;  // (rows past the tile's 8 G: none)
    hq[i] = kvh * G + rr % G;
    anc[i] = t[i] < R ? uint(tree[kV8tKfMask + t[i]]) : 0u;
    lim[i] = TREE == 3 && t[i] < R ? v8t_zone(p.pos0 + (anc[i] ? uint(popcount(anc[i])) - 1u : 0u)) : 0u;
  }
  const uint start = chunk * p.chunk;
  const uint kend = min(start + p.chunk, TREE == 3 ? v8t_zone(p.pos0 + 7u) : p.pos0 + R);
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
          if (TREE == 3 ? key >= lim[i] : key > p.pos0 && !(key - p.pos0 < 32u && ((anc[i] >> (key - p.pos0)) & 1u)))
            v = kV8tNeg;
          if (key >= kend) v = kV8tNeg;
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
    // P as half (TREE 3, as nh_attn_t); TREE 1 (the 27B, whose decode paths agree to ~1e-6 KL) as a half pair, hi + lo,
    // so that O carries attn8m's fp32 P to ~22 bits and only the sums' order differs
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) {
      const nhf_h8 pf = nhf_h8(sf[f]);
      nhf_h8 pl;
      if (TREE == 1) pl = nhf_h8(sf[f] - nhf_f8(pf));
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < DH / 16; id += 2) {
        const nhf_h8 v0 = v8t_load_kv(vb, k0 + 16 * f, d0 + id * 16, last, fm, fn);
        const nhf_h8 v1 = v8t_load_kv(vb, k0 + 16 * f, d0 + id * 16 + 16, last, fm, fn);
        if (TREE == 1) nhf_mma<half, half, false>(o[id], o[id + 1], pl, v0, v1);
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
      }
    }
  }
  // attn8m_partial_zone's partials: a row with no key below its zone in this chunk writes O 0, max -inf, sum 0
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    if (t[i] >= R) continue;
    const size_t base = (size_t(t[i]) * p.Hq + hq[i]) * p.nchunks + chunk;
    const bool any = TREE == 3 ? start < lim[i]
                               : start <= p.pos0 || (start - p.pos0 < 32u && (anc[i] >> (start - p.pos0)) != 0u);
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

#define ATTN8T(NAME, G, TREE)                                                                                         \
  [[kernel, max_total_threads_per_threadgroup(64 * ((8 * G + 15) / 16))]] void NAME(                                \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      device float* part_o [[buffer(3)]], device float* part_ml [[buffer(4)]], constant V8tAttnP& p [[buffer(5)]],   \
      constant uint& R [[buffer(6)]], const device int* tree [[buffer(7)]],                                          \
      uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],                               \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                        \
    threadgroup float xs[2 * ((8 * G + 15) / 16) * 32 * 16];                                                         \
    attn8t_impl<G, TREE>(q, kc, vc, part_o, part_ml, p, R, tree, xs, tg, ntg.x, sg, ushort(lane));                   \
  }
ATTN8T(attn8t_partial_zone_g8, 8, 3)  // Qwen3.6 35B-A3B, Clover: 16 query heads over 2 KV heads, 4 tiles
ATTN8T(attn8t_partial_tree_g6, 6, 1)  // Qwen3.8 27B's wide verify: 24 query heads over 4 KV heads, 3 tiles, P as hi + lo
