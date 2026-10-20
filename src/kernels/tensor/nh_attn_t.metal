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
