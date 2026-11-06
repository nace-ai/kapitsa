#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;


// ---- qattn_t_sparse_g12d2: Qwen3.8-Flash-Next's long-context prefill attention over per-row key lists (NI_QSA_TOPS)
// on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5, macOS 26) ----
//
// The tensor-op form of attn_sparse_rows (qwen4.metal): buffers, parameters and result as that kernel's (q fp32
// [R][Hq][256], half KV cache [Hkv][max_seq][256], gate in qg [R][Hq][2][256], fp32 out [R][Hq][256], key list of row
// r = idx[r * i_ld + 0 .. cnt[r])). attn_sparse_rows walks a row's ~2048 listed keys one at a time on the shader ALUs
// (a dot product, a simd_sum and two exps per key and head): about 3 TFLOP/s, the largest prefill kernel at 32K.
// Here a threadgroup takes one row and one KV head: a pair of simdgroups, each holding half the head dim (128) of the
// row's 12 query heads as the 16 rows of its fragments (rows 12..15 repeat head 11 and are not written). The pair walks
// the list 32 keys at a time: S = Q K^T (8 products over its half of the head dim, K gathered straight from the cache
// into 16 x 16 fragments by the listed rows), the two partial score tiles added through threadgroup memory (a + b in
// one, b + a in the other: the same value), the same online softmax in fp32 in both, and O += P V over its half of V
// (P as hi + lo halves, as Q). Products and fragment layout as qattn_t.metal / nh_attn_t.metal (nhf_mma); the
// head-dim split as qattn_t_d2 (64 accumulator floats a lane instead of 128). The rows are one token, so there is no
// causal cut: a list holds only keys at or before its row. The softmax accumulates in 32-key tiles instead of key by
// key and Q and P enter the products as two halves each (hi + lo, about 22 bits, #585), so the outputs are not bitwise
// those of attn_sparse_rows (docs/usage.md, NI_QSA_TATTN).
// Static threadgroup memory 4,096 B (xs). grid: (R, Hkv) x 64 threads.

struct QsaTParams {  // = AttnSparseParams
  uint Hq, Hkv;
  uint chunk;
  uint max_seq;
  float scale;
  uint nchunks;
  uint i_ld;
};

template <int G>
NI_INLINE void qattn_t_sparse_d2_impl(const device float* q, const device half* kc, const device half* vc,
                                      const device float* qg, device float* out, constant QsaTParams& p,
                                      const device uint* idx, const device uint* cnt, uint R, uint2 tg, uint sg,
                                      ushort lane, threadgroup float* xs) {
  constexpr int D = 256, DH = 128, BK = 32;
  const uint r = tg.x, kvh = tg.y, d0 = sg * DH;
  if (r >= R) return;  // never taken (grid.x == R); the whole threadgroup would leave together
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  uint hd[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) hd[i] = kvh * G + min(uint(fm + 8 * i), uint(G - 1));
  // Q and P enter the products as two halves each, hi + lo (#585): ~22 bits of Q and P, near attn_sparse_rows'
  // fp32, for twice the S and PV products. With one half each, a 64K prompt's first decode row moved to <|im_end|>.
  nhf_h8 qf[DH / 16], ql[DH / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < DH / 16; ++id)
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      const float4 x = *(const device float4*)(q + (size_t(r) * p.Hq + hd[i]) * D + d0 + id * 16 + fn) * p.scale;
      _Pragma("clang loop unroll(full)")
      for (short c = 0; c < 4; ++c) {
        qf[id][4 * i + c] = half(x[c]);
        ql[id][4 * i + c] = half(x[c] - float(qf[id][4 * i + c]));
      }
    }
  nhf_f8 o[DH / 16];
  _Pragma("clang loop unroll(full)")
  for (short id = 0; id < DH / 16; ++id) o[id] = nhf_f8(0.0f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
  const device half* kb = kc + size_t(kvh) * p.max_seq * D + d0;
  const device half* vb = vc + size_t(kvh) * p.max_seq * D + d0;
  const device uint* list = idx + size_t(r) * p.i_ld;
  const uint n = cnt[r];
  threadgroup float4* xme = (threadgroup float4*)(xs + sg * 512) + lane * 4;
  const threadgroup float4* xpa = (const threadgroup float4*)(xs + (sg ^ 1) * 512) + lane * 4;
  for (uint k0 = 0; k0 < n; k0 += BK) {
    const device half* kr[4];
    const device half* vr[4];
    bool ok[4];
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; ++j) {
      const uint e = k0 + fm + 8 * j;
      ok[j] = e < n;
      const uint row = ok[j] ? list[e] : list[0];  // lists are never empty (qsa_select: topb * ratio + tail)
      kr[j] = kb + size_t(row) * D + fn;
      vr[j] = vb + size_t(row) * D + fn;
    }
    nhf_f8 sf[2] = {nhf_f8(0.0f), nhf_f8(0.0f)};
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < DH / 16; ++id) {
      nhf_h8 ka, kb2;
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 2; ++i) {
        const half4 x = *(const device half4*)(kr[i] + id * 16);
        const half4 y = *(const device half4*)(kr[2 + i] + id * 16);
        ka[4 * i] = x.x, ka[4 * i + 1] = x.y, ka[4 * i + 2] = x.z, ka[4 * i + 3] = x.w;
        kb2[4 * i] = y.x, kb2[4 * i + 1] = y.y, kb2[4 * i + 2] = y.z, kb2[4 * i + 3] = y.w;
      }
      nhf_mma<half, half, true>(sf[0], sf[1], qf[id], ka, kb2);
      nhf_mma<half, half, true>(sf[0], sf[1], ql[id], ka, kb2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the partner has read the previous tile's partials
    xme[0] = float4(sf[0][0], sf[0][1], sf[0][2], sf[0][3]);
    xme[1] = float4(sf[0][4], sf[0][5], sf[0][6], sf[0][7]);
    xme[2] = float4(sf[1][0], sf[1][1], sf[1][2], sf[1][3]);
    xme[3] = float4(sf[1][4], sf[1][5], sf[1][6], sf[1][7]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < 2; ++f) {
      const float4 pa = xpa[2 * f], pb = xpa[2 * f + 1];
      sf[f][0] += pa.x; sf[f][1] += pa.y; sf[f][2] += pa.z; sf[f][3] += pa.w;
      sf[f][4] += pb.x; sf[f][5] += pb.y; sf[f][6] += pb.z; sf[f][7] += pb.w;
    }
    const bool tail = k0 + BK > n;
    float corr[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; ++i) {
      float mx = -1e30f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < 2; ++f)
        _Pragma("clang loop unroll(full)")
        for (short jj = 0; jj < 4; ++jj) {
          float v = sf[f][4 * i + jj] * M_LOG2E_F;
          if (tail && k0 + 16 * f + fn + jj >= n) v = -1e30f;
          sf[f][4 * i + jj] = v;
          mx = max(mx, v);
        }
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float mn = max(m[i], mx);
      corr[i] = metal::fast::exp2(m[i] - mn);
      m[i] = mn;
      float ps = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (short f = 0; f < 2; ++f)
        _Pragma("clang loop unroll(full)")
        for (short jj = 0; jj < 4; ++jj) {
          sf[f][4 * i + jj] = metal::fast::exp2(sf[f][4 * i + jj] - mn);
          ps += sf[f][4 * i + jj];
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
    for (short f = 0; f < 2; ++f) {
      const nhf_h8 pf = nhf_h8(sf[f]);
      nhf_h8 pl;
      _Pragma("clang loop unroll(full)")
      for (short e = 0; e < 8; ++e) pl[e] = half(sf[f][e] - float(pf[e]));
      _Pragma("clang loop unroll(full)")
      for (short id = 0; id < DH / 16; id += 2) {
        nhf_h8 v0, v1;
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < 2; ++i) {
          const half4 x = *(const device half4*)(vr[2 * f + i] + id * 16);
          const half4 y = *(const device half4*)(vr[2 * f + i] + id * 16 + 16);
          const bool z = !ok[2 * f + i];
          v0[4 * i] = z ? half(0) : x.x, v0[4 * i + 1] = z ? half(0) : x.y;
          v0[4 * i + 2] = z ? half(0) : x.z, v0[4 * i + 3] = z ? half(0) : x.w;
          v1[4 * i] = z ? half(0) : y.x, v1[4 * i + 1] = z ? half(0) : y.y;
          v1[4 * i + 2] = z ? half(0) : y.z, v1[4 * i + 3] = z ? half(0) : y.w;
        }
        nhf_mma<half, half, false>(o[id], o[id + 1], pf, v0, v1);
        nhf_mma<half, half, false>(o[id], o[id + 1], pl, v0, v1);
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; ++i) {
    if (fm + 8 * i >= G) continue;
    const float inv = 1.0f / l[i];
    const size_t row = size_t(r) * p.Hq + hd[i];
    _Pragma("clang loop unroll(full)")
    for (short id = 0; id < DH / 16; ++id) {
      const uint d = d0 + id * 16 + fn;
      const float4 g = *(const device float4*)(qg + row * 2 * D + D + d);
      float4 v = float4(o[id][4 * i], o[id][4 * i + 1], o[id][4 * i + 2], o[id][4 * i + 3]) * inv;
      v *= 1.0f / (1.0f + metal::fast::exp(-g));
      *(device float4*)(out + row * D + d) = v;
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(64)]] void qattn_t_sparse_g12d2(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* qg [[buffer(3)]], device float* out [[buffer(4)]], constant QsaTParams& p [[buffer(5)]],
    const device uint* idx [[buffer(6)]], const device uint* cnt [[buffer(7)]], constant uint& R [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[2 * 512];
  static_assert(sizeof(xs) == 4096, "qattn_t_sparse_g12d2: the docs state 4,096 B of threadgroup memory");
  qattn_t_sparse_d2_impl<12>(q, kc, vc, qg, out, p, idx, cnt, R, tg, sg, ushort(lane), xs);
}
