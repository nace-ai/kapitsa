#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- Flint prefill attention on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5) ----
// flint_attn_prefill's (flint/flint.metal) buffers, parameters and result on qattn_t.metal's register tiles (16 x 16
// fragments, nhf_mma from nh_attn_t.metal, qa_load_kv from qattn_t.metal; this file sorts after both). Head dim 128,
// G = 8 query heads per KV head: a 16-row tile holds 2 tokens x the KV head's 8 heads (row r = token t0 + r / 8, head
// kvh * 8 + r % 8); a threadgroup of NS simdgroups takes 2 NS tokens. Keys before the chunk come from the cache (a ring:
// slot = position % rows), the chunk's own from ck / cv ([Hkv][T][128]); row t sees keys (pos - window, pos] (all up to
// pos without a window). Online softmax in base 2 with fp32 statistics, every row starting from its head's sink
// (max = sink, sum = 1). q: fp32 [T][Hq][128] (scaled by scale * log2 e before rounding to half, as flint_attn_prefill);
// out: half [T][Hq][128]. grid: (ceil(T / (2 NS)), Hkv) x 32 NS threads; the latest token blocks start first.
// G = 7 (qattn_flint_*_g7: a dense Llama on Flint's kernels, Nanbeige4.3, whose sinks are -inf): the tile's rows keep
// 8 slots a token, the eighth a copy of the seventh head's that is computed and not written.

struct FlintAttnTP {  // = FlintAttnPfP
  uint T, pos0, Hq, Hkv;
  uint rows, ring, window;
  float scale;
};

template <int NS, int BK, int LD, int G = 8>
NI_INLINE void qattn_flint_impl(const device float* q, const device half* kc, const device half* vc,
                                const device half* ck, const device half* cv, const device bfloat* sinks,
                                device half* out, constant FlintAttnTP& p, uint2 tg, uint sg, ushort lane,
                                threadgroup half* kt, threadgroup half* vt) {
  constexpr int D = 128, KT = BK / 16, TT = 2;
  static_assert(G == 7 || G == 8, "a row's 8 slots hold the KV head's 7 or 8 query heads");
  const uint ntg = (p.T + NS * TT - 1) / (NS * TT);
  const uint kvh = tg.y, tgt0 = (ntg - 1 - tg.x) * NS * TT, t0 = tgt0 + sg * TT;   // heaviest (latest) blocks first
  const bool live = t0 < p.T;
  const uint tid = sg * 32 + lane;
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  uint tk[2], hd[2], lim[2], lo[2];
  bool slot[2];  // the row is one of the KV head's heads (G 7: the eighth slot of a token is not)
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    const short r = fm + 8 * i;
    tk[i] = t0 + (r >> 3);
    slot[i] = (r & 7) < G;
    hd[i] = kvh * G + min(r & 7, G - 1);
    lim[i] = p.pos0 + min(tk[i], p.T - 1);
    lo[i] = p.window && lim[i] + 1 > p.window ? lim[i] + 1 - p.window : 0u;
  }
  // the threadgroup's keys [kfirst, kend); the simdgroup's: a tile is skipped past the last token's limit or below the
  // first token's window start, and masked where it crosses the first token's limit or the last token's window start
  const uint kend = p.pos0 + min(tgt0 + NS * TT, p.T);
  const uint kfirst = p.window && p.pos0 + tgt0 + 1 > p.window ? p.pos0 + tgt0 + 1 - p.window : 0u;
  const uint sfirst = p.pos0 + min(t0, p.T - 1);
  const uint slast = p.pos0 + min(t0 + TT - 1, p.T - 1);
  const uint slo = p.window && slast + 1 > p.window ? slast + 1 - p.window : 0u;
  const uint sflo = p.window && sfirst + 1 > p.window ? sfirst + 1 - p.window : 0u;   // the first token's: one key lower
  const float qscale = p.scale * M_LOG2E_F;
  nhf_h8 qf[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x = *(const device float4*)(q + (size_t(min(tk[i], p.T - 1)) * p.Hq + hd[i]) * D + id * 16 + fn) * qscale;
      qf[id][4 * i] = half(x.x);
      qf[id][4 * i + 1] = half(x.y);
      qf[id][4 * i + 2] = half(x.z);
      qf[id][4 * i + 3] = half(x.w);
    }
  nhf_f8 o[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < D / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2], l[2] = {1.0f, 1.0f};
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) m[i] = float(sinks[hd[i]]) * M_LOG2E_F;
  const device half* kb = kc + size_t(kvh) * p.rows * D;
  const device half* vb = vc + size_t(kvh) * p.rows * D;
  const device half* ckb = ck + size_t(kvh) * p.T * D;
  const device half* cvb = cv + size_t(kvh) * p.T * D;
  for (uint k0 = kfirst / BK * BK; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous step is done with the tiles
    for (uint i = tid; i < BK * (D / 8); i += 32 * NS) {   // BK rows x 16 half8
      const uint r = i / (D / 8), c = (i % (D / 8)) * 8, key = k0 + r;
      uint4 kv4 = uint4(0), vv4 = uint4(0);
      if (key >= kfirst && key < kend) {  // (keys outside are zeros: a ring slot past the window holds another's)
        const bool own = key >= p.pos0;
        const size_t at = own ? size_t(key - p.pos0) * D + c : size_t(p.ring ? key % p.rows : key) * D + c;
        kv4 = *(const device uint4*)((own ? ckb : kb) + at);
        vv4 = *(const device uint4*)((own ? cvb : vb) + at);
      }
      *(threadgroup uint4*)(kt + r * LD + c) = kv4;
      *(threadgroup uint4*)(vt + r * LD + c) = vv4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || k0 > slast || k0 + BK <= sflo) continue;   // (tile loads and barriers are uniform over the threadgroup)
    nhf_f8 sf[KT > 1 ? KT : 2];   // (BK 16: the product's second 16 keys are a copy of the first, dropped)
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < (KT > 1 ? KT : 2); ++f) sf[f] = nhf_f8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < KT; f += 2)
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < D / 16; ++id) {
        const nhf_h8 ka = qa_load_kv<LD>(kt, 16 * f, id * 16, fm, fn);
        const nhf_h8 kb2 = KT > 1 ? qa_load_kv<LD>(kt, 16 * f + 16, id * 16, fm, fn) : ka;
        nhf_mma<half, half, true>(sf[f], sf[f + 1], qf[id], ka, kb2);
      }
    const bool edge = k0 + BK - 1 > sfirst || k0 < slo;   // some key of the tile is masked for some row
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
          if (edge && (key > lim[i] || key < lo[i])) v = -1e30f;
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
    if (!live || tk[i] >= p.T || !slot[i]) continue;
    const float inv = 1.0f / l[i];
    device half* dst = out + (size_t(tk[i]) * p.Hq + hd[i]) * D;
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < D / 16; ++id) {
      const float4 r = float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv;
      *(device half4*)(dst + id * 16 + fn) = half4(r);
    }
  }
}

#define QATTN_FLINT(NAME, NS, BK, LD, G)                                                                               \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                                 \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      const device half* ck [[buffer(3)]], const device half* cv [[buffer(4)]],                                      \
      const device bfloat* sinks [[buffer(5)]], device half* out [[buffer(6)]], constant FlintAttnTP& p [[buffer(7)]], \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                        \
      uint lane [[thread_index_in_simdgroup]]) {                                                                     \
    threadgroup half kt[BK * LD], vt[BK * LD];                                                                       \
    qattn_flint_impl<NS, BK, LD, G>(q, kc, vc, ck, cv, sinks, out, p, tg, sg, ushort(lane), kt, vt);                \
  }
// NS 8 / BK 32 with rows padded to 136 halves (kt / vt bank spread) is the measured best on the M5 Pro;
// small_tiles() (shader validation, NI_FLINT_K16=1) takes NS 4 / BK 16, half the threadgroup memory.
QATTN_FLINT(qattn_flint_s8k32p, 8, 32, 136, 8)
QATTN_FLINT(qattn_flint_s4k16, 4, 16, 128, 8)
QATTN_FLINT(qattn_flint_s8k32p_g7, 8, 32, 136, 7)
QATTN_FLINT(qattn_flint_s4k16_g7, 4, 16, 128, 7)
