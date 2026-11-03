#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- dory_attn_t: Dory's prefill attention on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5).
// dory_attn_prefill (src/kernels/dory/dory_attn.metal) with its products on matmul2d instead of 8x8x8 simdgroup
// MMAs: the same buffers (q fp32 [T][Hq][256] already scaled, keys from two places - the layer's cache for positions
// before the chunk, `ring` slots for a sliding-window layer, and the chunk's own [Hkv][T][256] for the rest -, fp32
// out [T][Hq][256], no gate; dory_gate_norm follows), the same half operands and the same masked/exponentiated fp32
// scores; only the summation inside the matrix unit differs. Products as src/kernels/tensor/qattn_t.metal (16 x 16
// fragments in registers, nhf_mma): a 16-key tile feeds the 16 x 32 product's first half only, since matmul2d wants
// 32 in one of its dims, and the second half is dropped - a 32-key tile fills it but doubles the tile buffer, which
// costs more occupancy than the waste (qattn_t measures the same on the 35B).
// A simdgroup takes TT = 16 / G consecutive tokens x the KV head's G query heads = 16 rows (row r = token t0 + r / TT,
// head kvh * G + r % G), so the rows' causal and window limits differ by token; the threadgroup is NS of those (NS 8
// default, 32 tokens and 256 threads; NI_DORY_ATTN_NS=4 halves both) and the latest blocks go first.
// grid: (ceil(T / (TT NS)), Hkv) x 32 NS threads. Measured on an M5 Pro against dory_attn_prefill (isolated
// dispatches): 717 -> 579 ms of attention at 4,132 tokens and 6,644 -> 5,468 ms at 16,493 (1.24x and 1.22x), the
// whole prefill 2,636 -> 2,534 ms and 14,160 -> 13,250 (a median of interleaved runs); the rows it produces are
// within 1.7e-6 mean KL of the simdgroup kernel's. 16 KB of threadgroup memory, so GPU shader validation and the
// debug layer (which count that memory twice) hold as they do for the 16-key dory_attn_prefill_k16.

struct DaParams {  // = DoryAttnPfP
  uint T, pos0, Hq, Hkv;
  uint rows, ring, window;
  float scale;
};

// a fragment of 16 tile rows r0 + [0, 16) x 16 columns c0 + [0, 16) from threadgroup memory
template <int LD>
NI_INLINE nhf_h8 da_load_kv(const threadgroup half* base, uint r0, uint c0, short fm, short fn) {
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

// the row of key position `key` for kv head kvh: the chunk's own K / V when key is in it, else the cache's slot
NI_INLINE const device half* da_kv_row(const device half* cache, const device half* own, uint key, constant DaParams& p,
                                       uint kvh) {
  if (key >= p.pos0) return own + (size_t(kvh) * p.T + (key - p.pos0)) * 256;
  return cache + (size_t(kvh) * p.rows + (p.ring ? key % p.rows : key)) * 256;
}

template <int NS, int BK, int LD, int G>   // G query heads per KV head (4 on the model this runs on)
NI_INLINE void dory_attn_t_impl(const device float* q, const device half* kc, const device half* vc,
                                const device half* ck, const device half* cv, device float* out, constant DaParams& p,
                                uint2 tg, uint sg, ushort lane, threadgroup half* kt, threadgroup half* vt) {
  constexpr int D = 256, KT = BK / 16, TT = 16 / G, SH = G == 8 ? 3 : 2;   // TT tokens x G heads per 16-row tile
  const uint ntg = (p.T + NS * TT - 1) / (NS * TT);
  const uint t0 = (ntg - 1 - tg.x) * NS * TT + sg * TT;   // heaviest (latest) blocks first
  const bool live = t0 < p.T;
  const uint tid = sg * 32 + lane;
  // the threadgroup's key span: its last token decides how far the tiles go, its first one where they start
  const uint tf = (ntg - 1 - tg.x) * NS * TT;
  const uint tl = min(tf + NS * TT, p.T);                     // first token after the block
  const uint kend = p.pos0 + tl;                              // keys needed (exclusive)
  const uint kfirst = p.window && p.pos0 + tf + 1 > p.window ? p.pos0 + tf + 1 - p.window : 0;
  const uint kt0 = kfirst / BK * BK;
  const uint kt1 = (kend + BK - 1) / BK * BK;
  const uint kvh = tg.y;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  uint tk[2], hd[2], lim[2], klo[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const short r = fm + 8 * i;
    tk[i] = t0 + (r >> SH);
    hd[i] = kvh * G + (r & (G - 1));
    lim[i] = p.pos0 + min(tk[i], p.T - 1);   // the last key this row sees
    klo[i] = p.window && lim[i] + 1 > p.window ? lim[i] + 1 - p.window : 0;
  }
  nhf_h8 qf[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const size_t qo = (size_t(min(tk[i], p.T - 1)) * p.Hq + hd[i]) * D + id * 16 + fn;
      const device float4* xp = (const device float4*)(q + qo);
      const float4 x = *xp * p.scale;
      qf[id][4 * i] = half(x.x);
      qf[id][4 * i + 1] = half(x.y);
      qf[id][4 * i + 2] = half(x.z);
      qf[id][4 * i + 3] = half(x.w);
    }
  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
  for (uint k0 = kt0; k0 < kt1; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous step is done with the tiles
    for (uint i = tid; i < BK * 32; i += 32 * NS) {   // BK rows x 32 half8
      const uint r = i >> 5, c = (i & 31) * 8;
      const bool ok = k0 + r < kend;
      *(threadgroup uint4*)(kt + r * LD + c) =
          ok ? *(const device uint4*)(da_kv_row(kc, ck, k0 + r, p, kvh) + c) : uint4(0);
      *(threadgroup uint4*)(vt + r * LD + c) =
          ok ? *(const device uint4*)(da_kv_row(vc, cv, k0 + r, p, kvh) + c) : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;   // (tile loads and barriers are uniform over the threadgroup)
    nhf_f8 sf[KT];
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) sf[f] = nhf_f8(0.0f);
    // (matmul2d wants 32 in one of M / N / K, so a 16-key tile fills the 16 x 32 product's first half only: the
    // second one is computed from zeros and dropped. A 32-key tile would fill it but doubles the tile buffer, which
    // costs more occupancy than the waste - qattn_t.metal measures the same on the 35B)
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; f += 2) {
      nhf_f8 tail = nhf_f8(0.0f);
      thread nhf_f8& c1 = f + 1 < KT ? sf[f + 1] : tail;
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) {
        const nhf_h8 ka = da_load_kv<LD>(kt, 16 * f, id * 16, fm, fn);
        const nhf_h8 kb2 = f + 1 < KT ? da_load_kv<LD>(kt, 16 * (f + 1), id * 16, fm, fn) : nhf_h8(0);
        nhf_mma<half, half, true>(sf[f], c1, qf[id], ka, kb2);
      }
    }
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; ++j) {
          const uint key = k0 + 16 * f + fn + j;
          float v = sf[f][4 * i + j] * M_LOG2E_F;
          if (key > lim[i] || key < klo[i] || key >= kend) v = -1e30f;
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
        const nhf_h8 v0 = da_load_kv<LD>(vt, 16 * f, id * 16, fm, fn);
        const nhf_h8 v1 = da_load_kv<LD>(vt, 16 * f, id * 16 + 16, fm, fn);
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    if (!live || tk[i] >= p.T) continue;
    const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
    device float* dst = out + (size_t(tk[i]) * p.Hq + hd[i]) * D;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id)
      *(device float4*)(dst + id * 16 + fn) =
          float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv;
  }
}

#define DORY_ATTN_T(NAME, NS, BK, LD, G)                                                                              \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                                   \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],  \
      const device half* ck [[buffer(3)]], const device half* cv [[buffer(4)]], device float* out [[buffer(5)]],      \
      constant DaParams& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup half kt[BK * LD], vt[BK * LD];                                                                        \
    dory_attn_t_impl<NS, BK, LD, G>(q, kc, vc, ck, cv, out, p, tg, sg, ushort(lane), kt, vt);                         \
  }
// 16-key tiles (16 KB of threadgroup memory for both tiles); a threadgroup covers TT NS tokens of one KV head. The
// models this runs on have 4 query heads per KV head (G = 4); the kernel stays generic in G, but no instance exists
// for another ratio until a model needs one.
DORY_ATTN_T(dory_attn_t_g4s4k16, 4, 16, 256, 4)
DORY_ATTN_T(dory_attn_t_g4s8k16, 8, 16, 256, 4)

// ---- dory_attn_td2: the head dim split over a pair of simdgroups (qattn_t.metal's qattn_t_d2, on Dory's buffers). Each
// simdgroup holds half of Q and of O: 64 accumulator floats a lane instead of dory_attn_t_impl's 128, the register bound
// that kernel runs into. The pair adds its two partial score tiles through threadgroup memory (a + b in one, b + a in
// the other: the same value), so both run the same online softmax; each multiplies P by its half of V. Keys, masks,
// the sliding window, the two key sources and the fp32 store are dory_attn_t_impl's; the scores' sum over the head dim
// is two halves added, so only the order of that sum differs from it.
// NS simdgroups = NS / 2 pairs x 16 rows per threadgroup (16 rows = 16 / G tokens x the KV head's G query heads), the
// latest blocks first. grid: (ceil(T * G / (8 NS)), Hkv) x 32 NS threads. 24,576 B of threadgroup memory (two 16-key
// half tiles and the pairs' fp32 score exchange): the validation layer, which counts it twice, cannot dispatch it, so the
// host takes dory_attn_t_impl under MTL_SHADER_VALIDATION.
template <int NS, int G>
NI_INLINE void dory_attn_td2_impl(const device float* q, const device half* kc, const device half* vc,
                                  const device half* ck, const device half* cv, device float* out, constant DaParams& p,
                                  uint2 tg, uint sg, ushort lane, threadgroup half* kt, threadgroup half* vt,
                                  threadgroup float* xs) {
  constexpr int D = 256, DH = 128, BK = 16, LD = 256, NP = NS / 2, TB = NP * 16 / G;   // TB tokens a threadgroup
  const uint ntg = (p.T + TB - 1) / TB;
  const uint bx = ntg - 1 - tg.x;                    // heaviest (latest) blocks first
  const uint pi = sg >> 1, d0 = (sg & 1) * DH;
  const uint R0 = (bx * NP + pi) * 16, t0 = R0 / G, kvh = tg.y;
  const bool live = R0 < p.T * G;
  const uint tid = sg * 32 + lane;
  // the threadgroup's key span: its last token decides how far the tiles go, its first one where they start
  const uint tf = bx * TB, tl = min(tf + TB, p.T);
  const uint kend = p.pos0 + tl;
  const uint kfirst = p.window && p.pos0 + tf + 1 > p.window ? p.pos0 + tf + 1 - p.window : 0;
  const uint kt0 = kfirst / BK * BK;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  uint tk[2], hd[2], lim[2], klo[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const uint r = R0 + fm + 8 * i;
    tk[i] = r / G;
    hd[i] = kvh * G + r % G;
    lim[i] = p.pos0 + min(tk[i], p.T - 1);   // the last key this row sees
    klo[i] = p.window && lim[i] + 1 > p.window ? lim[i] + 1 - p.window : 0;
  }
  const uint first = p.pos0 + t0;
  nhf_h8 qf[DH / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < DH / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x =
          *(const device float4*)(q + (size_t(min(tk[i], p.T - 1)) * p.Hq + hd[i]) * D + d0 + id * 16 + fn) * p.scale;
      qf[id][4 * i] = half(x.x);
      qf[id][4 * i + 1] = half(x.y);
      qf[id][4 * i + 2] = half(x.z);
      qf[id][4 * i + 3] = half(x.w);
    }
  nhf_f8 o[DH / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < DH / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
  threadgroup float4* xme = (threadgroup float4*)(xs + sg * 256) + lane * 2;
  const threadgroup float4* xpa = (const threadgroup float4*)(xs + (sg ^ 1) * 256) + lane * 2;
  for (uint k0 = kt0; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous step is done with the tiles
    for (uint i = tid; i < BK * 32; i += 32 * NS) {   // BK rows x 32 half8
      // (da_kv_row's choice as an offset and a base, both loads issued before either store: 13% faster at 254K
      // positions than calling it once per buffer)
      const uint r = i >> 5, c = (i & 31) * 8, key = k0 + r;
      const bool ok = key < kend, own = key >= p.pos0;
      const size_t off = own ? (size_t(kvh) * p.T + (key - p.pos0)) * 256 + c
                             : (size_t(kvh) * p.rows + (p.ring ? key % p.rows : key)) * 256 + c;
      const uint4 kx = ok ? *(const device uint4*)((own ? ck : kc) + off) : uint4(0);
      const uint4 vx = ok ? *(const device uint4*)((own ? cv : vc) + off) : uint4(0);
      *(threadgroup uint4*)(kt + r * LD + c) = kx;
      *(threadgroup uint4*)(vt + r * LD + c) = vx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (matmul2d wants 32 in one of M / N / K: the 16-key tile fills the 16 x 32 product's first half, the second is
    // the same keys again and dropped)
    nhf_f8 s = nhf_f8(0.0f), s1 = nhf_f8(0.0f);
    if (live) {
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < DH / 16; ++id) {
        const nhf_h8 ka = da_load_kv<LD>(kt, 0, d0 + id * 16, fm, fn);
        nhf_mma<half, half, true>(s, s1, qf[id], ka, ka);
      }
      xme[0] = float4(s[0], s[1], s[2], s[3]);
      xme[1] = float4(s[4], s[5], s[6], s[7]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;   // (tile loads and barriers are uniform over the threadgroup)
    const float4 pa = xpa[0], pb = xpa[1];
    s[0] += pa.x; s[1] += pa.y; s[2] += pa.z; s[3] += pa.w;
    s[4] += pb.x; s[5] += pb.y; s[6] += pb.z; s[7] += pb.w;
    // a tile past the pair's first token or below a window's start can hold keys a row must not see
    const bool msk = k0 + BK - 1 > first || p.window;
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; ++j) {
        const uint key = k0 + fn + j;
        float v = s[4 * i + j] * M_LOG2E_F;
        if (msk && (key > lim[i] || key < klo[i])) v = -1e30f;
        s[4 * i + j] = v;
        mx = max(mx, v);
      }
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float mn = max(m[i], mx);
      corr[i] = metal::fast::exp2(m[i] - mn);
      m[i] = mn;
      float ps = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; ++j) {
        s[4 * i + j] = metal::fast::exp2(s[4 * i + j] - mn);
        ps += s[4 * i + j];
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
    const nhf_h8 pf = nhf_h8(s);
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < DH / 16; id += 2) {
      const nhf_h8 v0 = da_load_kv<LD>(vt, 0, d0 + id * 16, fm, fn);
      const nhf_h8 v1 = da_load_kv<LD>(vt, 0, d0 + id * 16 + 16, fm, fn);
      nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
    }
  }
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    if (!live || tk[i] >= p.T) continue;
    const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
    device float* dst = out + (size_t(tk[i]) * p.Hq + hd[i]) * D + d0;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < DH / 16; ++id)
      *(device float4*)(dst + id * 16 + fn) =
          float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv;
  }
}

#define DORY_ATTN_TD2(NAME, NS, G)                                                                                    \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                                   \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],  \
      const device half* ck [[buffer(3)]], const device half* cv [[buffer(4)]], device float* out [[buffer(5)]],      \
      constant DaParams& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup half kt[16 * 256], vt[16 * 256];                                                                      \
    threadgroup float xs[NS * 256];                                                                                   \
    static_assert(sizeof(kt) + sizeof(vt) + sizeof(xs) <= 32768, "dory_attn_td2: threadgroup memory over 32 KB");    \
    dory_attn_td2_impl<NS, G>(q, kc, vc, ck, cv, out, p, tg, sg, ushort(lane), kt, vt, xs);                          \
  }
// 4 pairs x 16 rows = 64 (token, head) rows, 16 tokens a threadgroup (Dory: 4 query heads per KV head)
DORY_ATTN_TD2(dory_attn_td2_g4s8, 8, 4)
