#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- Nemotron-H prefill attention on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5) ----
//
// Causal attention of T new rows (positions pos0 + t) over the KV cache (half [Hkv][cap][128], written up to
// pos0 + T), head dim 128, 16 query heads per KV head (Lightning: 32 over 2). A simdgroup takes one token's 16 query
// heads: the 16 rows of its tiles all have the token's causal limit. Everything stays in registers as 16 x 16
// fragments (lane: rows fm, fm + 8, columns fn .. fn + 3), the layout of matmul2d's cooperative tensors for a
// 16 x 32 x 16 product (the fragment layout and the register-level products follow MLX's steel NAX attention,
// steel/attn/kernels/steel_attention_nax.h; see THIRD_PARTY_NOTICES): per 32 keys S = Q K^T (8 products over the
// head dim), the online softmax in fp32 (row reductions over lanes ^ 1 and ^ 8), O += P V (P half, 8 products).
// Output: half [T][Hq][128] (nh_attn_pf2's). grid: (ceil(T / NS), Hkv) x 32 NS threads, tokens of block tg.x + qb0.

struct NhAttnTP {  // = NhAttnPf2P
  uint T, pos0, cap, Hq, Hkv;
  float scale;
  uint qb0;
};

typedef vec<half, 8> nhf_h8;
typedef vec<float, 8> nhf_f8;

// C[16 x 32] (two fragments) += A[16 x 16] B[16 x 32] (B as two fragments: [k][n], or [n][k] with TB)
template <typename AT, typename BT, bool TB>
NI_INLINE void nhf_mma(thread nhf_f8& c0, thread nhf_f8& c1, thread const vec<AT, 8>& a, thread const vec<BT, 8>& b0,
                       thread const vec<BT, 8>& b1) {
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 16, false, TB, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
  auto cc = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(ca)>,
                                                           remove_addrspace_t<decltype(cb)>, float>();
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; ++i) {
    ca[i] = a[i];
    cb[i] = b0[i];
    cb[8 + i] = b1[i];
    cc[i] = c0[i];
    cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; ++i) {
    c0[i] = cc[i];
    c1[i] = cc[8 + i];
  }
}

// a fragment of 16 cache rows r0 + [0, 16) x 16 columns c0 + [0, 16) (rows past lim read as 0)
NI_INLINE nhf_h8 nhf_load_kv(const device half* base, uint r0, uint c0, uint lim, short fm, short fn) {
  nhf_h8 v;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const uint r = r0 + fm + 8 * i;
    const half4 x = r <= lim ? *(const device half4*)(base + size_t(r) * 128 + c0 + fn) : half4(0);
    v[4 * i] = x.x;
    v[4 * i + 1] = x.y;
    v[4 * i + 2] = x.z;
    v[4 * i + 3] = x.w;
  }
  return v;
}

template <int NS, int BK>
NI_INLINE void nh_attn_t_impl(const device float* q, const device half* kc, const device half* vc, device half* out,
                              constant NhAttnTP& p, uint2 tg, uint sg, ushort lane) {
  constexpr int D = 128, G = 16, KT = BK / 16;
  const uint t = (tg.x + p.qb0) * NS + sg, kvh = tg.y;
  if (t >= p.T) return;
  const uint lim = p.pos0 + t;  // the token's last key
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  // Q (pre-scaled by scale x log2 e, as half): rows = the 16 heads, 8 fragments over the head dim
  const float qscale = p.scale * 1.4426950408889634f;
  const device float* qp = q + (size_t(t) * p.Hq + kvh * G) * D;
  nhf_h8 qf[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x = *(const device float4*)(qp + (fm + 8 * i) * D + id * 16 + fn) * qscale;
      qf[id][4 * i] = half(x.x);
      qf[id][4 * i + 1] = half(x.y);
      qf[id][4 * i + 2] = half(x.z);
      qf[id][4 * i + 3] = half(x.w);
    }
  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
  const device half* kb = kc + size_t(kvh) * p.cap * D;
  const device half* vb = vc + size_t(kvh) * p.cap * D;
  for (uint k0 = 0; k0 <= lim; k0 += BK) {
    nhf_f8 sf[KT];
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) sf[f] = nhf_f8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; f += 2)
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) {
        const nhf_h8 ka = nhf_load_kv(kb, k0 + 16 * f, id * 16, lim, fm, fn);
        const nhf_h8 kb2 = nhf_load_kv(kb, k0 + 16 * f + 16, id * 16, lim, fm, fn);
        nhf_mma<half, half, true>(sf[f], sf[f + 1], qf[id], ka, kb2);
      }
    if (k0 + BK - 1 > lim) {
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; ++e)
          if (k0 + 16 * f + fn + (e & 3) > lim) sf[f][e] = -1e30f;
    }
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        mx = max(mx, max(max(sf[f][4 * i], sf[f][4 * i + 1]), max(sf[f][4 * i + 2], sf[f][4 * i + 3])));
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
          sf[f][4 * i + j] = exp2(sf[f][4 * i + j] - mn);
          ps += sf[f][4 * i + j];
        }
      ps += simd_shuffle_xor(ps, ushort(1));
      ps += simd_shuffle_xor(ps, ushort(8));
      l[i] = l[i] * corr[i] + ps;
    }
    // (no row's max moved: the accumulators keep their scale)
    if (!simd_all(corr[0] == 1.0f && corr[1] == 1.0f)) {
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id)
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; ++e) o[id][e] *= corr[e >> 2];
    }
    // P as half (the operand layout of half x half products is the fragment layout; a float P's is not)
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) {
      const nhf_h8 pf = nhf_h8(sf[f]);
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; id += 2) {
        const nhf_h8 v0 = nhf_load_kv(vb, k0 + 16 * f, id * 16, lim, fm, fn);
        const nhf_h8 v1 = nhf_load_kv(vb, k0 + 16 * f, id * 16 + 16, lim, fm, fn);
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
      }
    }
  }
  device half* op = out + (size_t(t) * p.Hq + kvh * G) * D;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id)
      *(device half4*)(op + (fm + 8 * i) * D + id * 16 + fn) =
          half4(float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv);
  }
}

#define NH_ATTN_T(NAME, NS, BK)                                                                                      \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                                \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      device half* out [[buffer(3)]], constant NhAttnTP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]], \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                      \
    nh_attn_t_impl<NS, BK>(q, kc, vc, out, p, tg, sg, ushort(lane));                                                 \
  }
// 4 tokens per threadgroup, 32 keys per step (tools/nh_pf_bench, 2048 rows after 14336 positions: 12.6-12.9 ms; 2 or 8
// tokens the same, 12.6-14.5; 64 keys per step 22.4-27.1, relaxed-precision products 13.0-14.8)
NH_ATTN_T(nh_attn_t_s4, 4, 32)

// ---- nh_attn_ts: nh_attn_t with the K / V tiles staged in threadgroup memory for the threadgroup's NS tokens ----
//
// nh_attn_t's simdgroups each read their K / V fragments from device memory, so every threadgroup streams the whole
// cache prefix on its own; past what the GPU's caches hold, that is the prefill's limit (one layer, 2048 rows: 12.8 ms
// after 14336 positions, 40 TFLOP/s, but 187 / 521 / 1266 ms after 131072 / 262144 / 491520, 24 / 17 / 13 TFLOP/s; 2, 8
// or 16 tokens per threadgroup the same). Here the threadgroup's threads load each 32-key tile of K and of V once (16 KB)
// and its NS simdgroups (a token each, as nh_attn_t's) read their fragments from it. A token's products, softmax and
// output are nh_attn_t_impl's: a key past the token's limit is masked to a zero weight (the tile holds the
// threadgroup's keys up to its last token's), so the output is nh_attn_t_s4's bit for bit.
// grid: (ceil(T / NS), Hkv) x 32 NS threads, tokens of block tg.x + qb0.

// a fragment of 16 tile rows r0 + [0, 16) x 16 columns c0 + [0, 16) from a staged tile (rows of 128 halves)
NI_INLINE nhf_h8 nhf_load_tg(const threadgroup half* base, uint r0, uint c0, short fm, short fn) {
  nhf_h8 v;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const half4 x = *(const threadgroup half4*)(base + (r0 + fm + 8 * i) * 128 + c0 + fn);
    v[4 * i] = x.x;
    v[4 * i + 1] = x.y;
    v[4 * i + 2] = x.z;
    v[4 * i + 3] = x.w;
  }
  return v;
}

template <int NS, int BK>
NI_INLINE void nh_attn_ts_impl(const device float* q, const device half* kc, const device half* vc, device half* out,
                               constant NhAttnTP& p, uint2 tg, uint sg, ushort lane, threadgroup half* kt,
                               threadgroup half* vt) {
  constexpr int D = 128, G = 16, KT = BK / 16;
  const uint b = tg.x + p.qb0, t = b * NS + sg, kvh = tg.y;
  if (b * NS >= p.T) return;  // (the whole threadgroup)
  const bool live = t < p.T;
  const uint lim = p.pos0 + min(t, p.T - 1);                   // the token's last key
  const uint tgl = p.pos0 + min(b * NS + NS - 1, p.T - 1);     // the threadgroup's
  const uint tid = sg * 32 + lane;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  const float qscale = p.scale * 1.4426950408889634f;
  const device float* qp = q + (size_t(min(t, p.T - 1)) * p.Hq + kvh * G) * D;
  nhf_h8 qf[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x = *(const device float4*)(qp + (fm + 8 * i) * D + id * 16 + fn) * qscale;
      qf[id][4 * i] = half(x.x);
      qf[id][4 * i + 1] = half(x.y);
      qf[id][4 * i + 2] = half(x.z);
      qf[id][4 * i + 3] = half(x.w);
    }
  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
  const device half* kb = kc + size_t(kvh) * p.cap * D;
  const device half* vb = vc + size_t(kvh) * p.cap * D;
  for (uint k0 = 0; k0 <= tgl; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous tile's reads are done
    for (uint i = tid; i < uint(BK * D / 8); i += 32 * NS) {
      const uint r = i / (D / 8), c = (i % (D / 8)) * 8;
      const bool ok = k0 + r <= tgl;
      *(threadgroup uint4*)(kt + r * D + c) = ok ? *(const device uint4*)(kb + size_t(k0 + r) * D + c) : uint4(0);
      *(threadgroup uint4*)(vt + r * D + c) = ok ? *(const device uint4*)(vb + size_t(k0 + r) * D + c) : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || k0 > lim) continue;  // (tile loads and barriers stay uniform over the threadgroup)
    nhf_f8 sf[KT];
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) sf[f] = nhf_f8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; f += 2)
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) {
        const nhf_h8 ka = nhf_load_tg(kt, 16 * f, id * 16, fm, fn);
        const nhf_h8 kb2 = nhf_load_tg(kt, 16 * f + 16, id * 16, fm, fn);
        nhf_mma<half, half, true>(sf[f], sf[f + 1], qf[id], ka, kb2);
      }
    if (k0 + BK - 1 > lim) {
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; ++e)
          if (k0 + 16 * f + fn + (e & 3) > lim) sf[f][e] = -1e30f;
    }
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        mx = max(mx, max(max(sf[f][4 * i], sf[f][4 * i + 1]), max(sf[f][4 * i + 2], sf[f][4 * i + 3])));
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
          sf[f][4 * i + j] = exp2(sf[f][4 * i + j] - mn);
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
        const nhf_h8 v0 = nhf_load_tg(vt, 16 * f, id * 16, fm, fn);
        const nhf_h8 v1 = nhf_load_tg(vt, 16 * f, id * 16 + 16, fm, fn);
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
      }
    }
  }
  if (!live) return;
  device half* op = out + (size_t(t) * p.Hq + kvh * G) * D;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id)
      *(device half4*)(op + (fm + 8 * i) * D + id * 16 + fn) =
          half4(float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv);
  }
}

#define NH_ATTN_TS(NAME, NS, BK)                                                                                     \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                                \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      device half* out [[buffer(3)]], constant NhAttnTP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]], \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                      \
    threadgroup half kt[BK * 128], vt[BK * 128];                                                                     \
    nh_attn_ts_impl<NS, BK>(q, kc, vc, out, p, tg, sg, ushort(lane), kt, vt);                                       \
  }
// 8 tokens per threadgroup (2048 rows after 14336 / 131072 / 262144 / 491520 positions: 11.0 / 98.5 / 201 / 384 ms, 43-47
// TFLOP/s; 16 tokens 12.3 / 106 / 212 / 398, 32 tokens 14.5 / 124 / 248 / 463)
NH_ATTN_TS(nh_attn_ts_s8, 8, 32)

// ---- the same products for a verify: R rows over chunks of the cache (split-K), nh_attn_partial's partials ----
//
// A verify's R rows (positions pos0 + r) each see the history [0, pos0) and, of the block, themselves and the rows
// before them (a chain) or their ancestors (a KFlash tree: the mask's bits). nh_attn_partial runs a simdgroup per query
// head and a threadgroup per row, a dot product and a simd_sum per key: at long context that is the verify's cost (an
// 8-row tree over Lightning's 271K-token context: ~80 ms of a 96 ms cycle). Here a simdgroup takes one row's 16 query
// heads over one chunk of keys [c chunk, min((c + 1) chunk, pos0 + r + 1)) on the matrix units, as nh_attn_t_impl takes
// a token's, and writes the chunk's unnormalized O, its max (natural log) and its sum where nh_attn_partial does, for
// nh_attn_reduce. One layer, 8 rows of which 6 live, at 32K / 128K / 256K / 512K positions (M5 Max, the engine's chunks):
// 0.17 / 0.61 / 1.19 / 2.35 ms against nh_attn_partial_tree's 1.47 / 5.52 / 11.0 / 22.2; 128 or 256 chunks instead of 64
// were slower (3.4-3.6 ms at 512K). Q as a half pair (QS) costs ~45% over Q as half alone (1.62 ms at 512K) and cuts the
// output's error against an fp64 reference 4-6x, to 1e-5 - 4e-4 (nh_attn_partial's fp32: ~1e-7; what is left is P as
// half, as nh_attn_t_impl's). grid: (nchunks, Hkv, ceil(R / NS)) x 32 NS threads.
struct NhAttnTPartP {  // = NhAttnParams (nemotron/nh_ops.metal)
  uint R, pos0, cap, Hq, Hkv, chunk, nchunks;
  float scale;
};
constant constexpr uint kNhtKfMask = 65;  // kflash.h's kKfMask: tree[kKfMask + r], bit j: row j is r or an ancestor

template <int NS, int BK, bool TREE, bool QS>
NI_INLINE void nh_attn_t_part_impl(const device float* q, const device half* kc, const device half* vc,
                                   device float* part_o, device float* part_ml, constant NhAttnTPartP& p,
                                   const device int* tree, uint3 tg, uint sg, ushort lane) {
  constexpr int D = 128, G = 16, KT = BK / 16;
  const uint r = tg.z * NS + sg, kvh = tg.y, c = tg.x;
  if (r >= p.R) return;
  const uint start = c * p.chunk, end = min(start + p.chunk, p.pos0 + r + 1);
  if (start >= end) return;  // (past the row's keys: nh_attn_reduce reads only its chunks)
  const uint last = end - 1;
  const uint mrow = TREE ? uint(tree[kNhtKfMask + r]) : 0u;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  // a tree's rows past its live rows (tree[0], kKfN, as kf_live reads it; best first, so no live row descends from
  // them): no attention, a zero output (O 0, max 0, sum 1), which their logits never use
  if (TREE && r >= uint(max(tree[0], 1))) {
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const size_t base = (size_t(r) * p.Hq + kvh * G + fm + 8 * i) * p.nchunks + c;
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) *(device float4*)(part_o + base * D + id * 16 + fn) = float4(0.0f);
      if (fn == 0) {
        part_ml[base * 2 + 0] = 0.0f;
        part_ml[base * 2 + 1] = 1.0f;
      }
    }
    return;
  }
  const float qscale = p.scale * 1.4426950408889634f;
  const device float* qp = q + (size_t(r) * p.Hq + kvh * G) * D;
  // Q as half, and with QS its remainder too (Q = hi + lo, both half: the products then carry ~22 bits of Q, not 11)
  nhf_h8 qf[D / 16], ql[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x = *(const device float4*)(qp + (fm + 8 * i) * D + id * 16 + fn) * qscale;
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; ++j) {
        qf[id][4 * i + j] = half(x[j]);
        if (QS) ql[id][4 * i + j] = half(x[j] - float(qf[id][4 * i + j]));
      }
    }
  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
  const device half* kb = kc + size_t(kvh) * p.cap * D;
  const device half* vb = vc + size_t(kvh) * p.cap * D;
  for (uint k0 = start; k0 <= last; k0 += BK) {
    nhf_f8 sf[KT];
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; ++f) sf[f] = nhf_f8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; f += 2)
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) {
        const nhf_h8 ka = nhf_load_kv(kb, k0 + 16 * f, id * 16, last, fm, fn);
        const nhf_h8 kb2 = nhf_load_kv(kb, k0 + 16 * f + 16, id * 16, last, fm, fn);
        if (QS) nhf_mma<half, half, true>(sf[f], sf[f + 1], ql[id], ka, kb2);
        nhf_mma<half, half, true>(sf[f], sf[f + 1], qf[id], ka, kb2);
      }
    // keys past the chunk's end, and of the block those that are not the row's ancestors (a tree)
    if (k0 + BK - 1 > last || (TREE && k0 + BK > p.pos0)) {
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; ++e) {
          const uint key = k0 + 16 * f + fn + (e & 3);
          if (key > last || (TREE && key >= p.pos0 && !((mrow >> (key - p.pos0)) & 1u))) sf[f][e] = -1e30f;
        }
    }
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < KT; ++f)
        mx = max(mx, max(max(sf[f][4 * i], sf[f][4 * i + 1]), max(sf[f][4 * i + 2], sf[f][4 * i + 3])));
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
          sf[f][4 * i + j] = v <= -1e29f ? 0.0f : exp2(v - mn);
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
        const nhf_h8 v0 = nhf_load_kv(vb, k0 + 16 * f, id * 16, last, fm, fn);
        const nhf_h8 v1 = nhf_load_kv(vb, k0 + 16 * f, id * 16 + 16, last, fm, fn);
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
      }
    }
  }
  // the partials: O unnormalized (scaled by 2^-m), m as a natural log (nh_attn_reduce's exp), l
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const size_t base = (size_t(r) * p.Hq + kvh * G + fm + 8 * i) * p.nchunks + c;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id)
      *(device float4*)(part_o + base * D + id * 16 + fn) =
          float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]);
    if (fn == 0) {
      part_ml[base * 2 + 0] = m[i] * 0.6931471805599453f;
      part_ml[base * 2 + 1] = l[i];
    }
  }
}

#define NH_ATTN_T_PART(NAME, NS, BK, TREE, QS)                                                                       \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                                \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      device float* part_o [[buffer(3)]], device float* part_ml [[buffer(4)]], constant NhAttnTPartP& p [[buffer(5)]], \
      const device int* tree [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                      \
    nh_attn_t_part_impl<NS, BK, TREE, QS>(q, kc, vc, part_o, part_ml, p, tree, tg, sg, ushort(lane));                \
  }
NH_ATTN_T_PART(nh_attn_t_part, 4, 32, false, true)
NH_ATTN_T_PART(nh_attn_t_part_tree, 4, 32, true, true)
