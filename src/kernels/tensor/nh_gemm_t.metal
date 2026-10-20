#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- Nemotron-H prefill GEMMs on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5) ----
//
// nh_gemm_* (nemotron/nh_gemm.metal) with the products on matmul2d instead of 8x8x8 simdgroup MMAs: the same tiles
// of dequantized half weights and half inputs in threadgroup memory (the same decode, so the same half operands),
// fp32 accumulators, the same epilogues and MoE modes; only the summation order inside the matrix unit differs.
// Signature and grid as nh_gemm_* (NhGemmP [[4]], tiles / ntiles / sorted [[5-7]], gse [[9]]); K % 32 == 0, X rows
// 16-byte aligned. Tiles: 64 rows x 64 columns x 32 K per step, 4 simdgroups of 32 x 32 (one matmul2d each per step),
// double-buffered (a step's loads issued before the previous step's product). tools/nh_pf_bench, Lightning's shapes
// at 2048 rows (M5 Max, two runs): Mamba in_proj FP8 10304 x 2688 1.90-1.95 ms (58-60 TFLOP/s; the matrix units'
// peak is ~64, bench_tops) against nh_gemm_fp8_m64's 7.48 (15), out_proj 0.76-0.77 / 3.07, shared expert up NVFP4
// 0.78-0.80 / 2.82-2.83, attention q BF16 0.76-0.82 / 3.13-3.26; routed experts (12288 pairs, uniformly random) up
// 3.12-3.94 against nh_x4_moe_up_m32's 9.26-9.29, down 4.01-4.63 / nh_gemm_fp4_moe_dn_m32's 10.35-12.89. Slower
// (single-buffered): 64 x 128 and 128 x 64 tiles (in_proj 2.46 against 2.10 ms), 64 K per step (2.22). The rms error
// against fp64 sums is below the simdgroup kernels' (in_proj 4.6e-7 against 6.8e-7 of the outputs' rms).

struct NhGemmTP {  // = NhGemmP
  uint M, N, K;
  uint x_ld, y_ld;
  float gs;
  uint epi;
  uint q_estride, s_estride;
  uint k;
};

constant float nht_fp4_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

NI_INLINE float nht_ue4m3(uint b) { return (b == 0x7Fu) ? 0.0f : float(as_type<half>(ushort(b << 7))) * 128.0f; }

// 16 consecutive weights of row n at k as half (nh_dq16's decode): FP4 (1, layout L0; 6, L2), BF16 (4), FP8 (5). In two
// steps: the raw bytes (nht_raw: loads only, issued a step ahead) and their decode into threadgroup memory (two
// 16-byte stores)
struct NhtRaw {
  uint4 a, b;  // FP8: a (16 codes); FP4: a.xy (8 code bytes), a.z (the block scale); BF16: a, b
};
template <int FMT>
NI_INLINE NhtRaw nht_raw(const device uchar* q, const device uchar* s, uint K, uint n, uint k) {
  NhtRaw r;
  if (FMT == 1 || FMT == 6) {
    r.a.xy = *(const device uint2*)(q + size_t(n) * (K / 2) + k / 2);
    r.a.z = s[size_t(n) * (K / 16) + k / 16];
  } else if (FMT == 4) {
    const device uint4* wp = (const device uint4*)(q + size_t(n) * K * 2 + k * 2);
    r.a = wp[0];
    r.b = wp[1];
  } else {
    r.a = *(const device uint4*)(q + size_t(n) * K + k);
  }
  return r;
}
template <int FMT>
NI_INLINE void nht_dec(const NhtRaw r, threadgroup half* dst) {
  half w[16];
  if (FMT == 1) {
    const float sc = nht_ue4m3(r.a.z);
    for (int j = 0; j < 8; ++j) {
      const uint b = ((j < 4 ? r.a.x : r.a.y) >> (8 * (j & 3))) & 0xFFu;
      w[j] = half(nht_fp4_lut[b & 15u] * sc);
      w[j + 8] = half(nht_fp4_lut[b >> 4] * sc);
    }
  } else if (FMT == 6) {
    // layout L2 (nh_x_fp4.metal): nh_x4_l2m's pairs (elements 4 h + j, 8 + 4 h + j of word h) as E2M1 x 2^-14 times
    // the scale as E4M3 x 2^7 in half: E2M1 x E4M3 x 2^-7 exactly (subnormals included), the epilogue's gs x 128
    // restoring FMT 1's sums to the bit
    const half sc = r.a.z == 0x7Fu ? half(0) : as_type<half>(ushort(r.a.z << 7)) * 32768.0h;
    for (int h = 0; h < 2; ++h) {
      const uint v = h ? r.a.y : r.a.x;
      const half2 p0 = as_type<half2>(((v & 0x000F000Fu) * 0x8100u) & 0x8E008E00u) * sc;
      const half2 p1 = as_type<half2>(((v & 0x00F000F0u) * 0x0810u) & 0x8E008E00u) * sc;
      const half2 p2 = as_type<half2>(((v & 0x0F000F00u) * 0x0081u) & 0x8E008E00u) * sc;
      const half2 p3 = as_type<half2>((((v >> 4) & 0x0F000F00u) * 0x0081u) & 0x8E008E00u) * sc;
      w[4 * h] = p0.x, w[4 * h + 1] = p1.x, w[4 * h + 2] = p2.x, w[4 * h + 3] = p3.x;
      w[8 + 4 * h] = p0.y, w[8 + 4 * h + 1] = p1.y, w[8 + 4 * h + 2] = p2.y, w[8 + 4 * h + 3] = p3.y;
    }
  } else if (FMT == 4) {
    for (int h = 0; h < 2; ++h) {
      const uint4 v = h ? r.b : r.a;
      for (int i = 0; i < 4; ++i) {
        w[h * 8 + 2 * i] = half(as_type<float>(v[i] << 16));
        w[h * 8 + 2 * i + 1] = half(as_type<float>(v[i] & 0xFFFF0000u));
      }
    }
  } else {
    for (int m = 0; m < 4; ++m) {
      const uint v = r.a[m];
      const half2 a = as_type<half2>(((v & 0x00800080u) << 8) | ((v & 0x007F007Fu) << 7));
      const half2 b = as_type<half2>((v & 0x80008000u) | ((v >> 1) & 0x3F803F80u));
      w[2 * m] = a.x;
      w[2 * m + 8] = a.y;
      w[2 * m + 1] = b.x;
      w[2 * m + 9] = b.y;
    }
  }
  *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
  *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
  *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
  *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
}

// A tile of MR rows (tokens) x BN columns: MR 64 as 2 x 2 simdgroups of 32 x 32, MR 16 / 32 as 1 x 4 of MR x 16.
// (Expert tails on smaller tiles, 16 / 32 / 48 rows of a 64-row tile, measured the same: 2.9-4.2 ms either way
// for the routed up projection, run to run.)
template <int FMT, int MODE, int MR, int BN, int BK>
NI_INLINE void nht_tile(const device half* X, const device uchar* q, const device uchar* s, device void* Y,
                        constant NhGemmTP& p, const device uint* sorted, uint m0, uint M, uint n0, float gs, uint tid,
                        uint sg, threadgroup half* xs, threadgroup half* ws) {
  constexpr int GN = MR >= 64 ? 2 : 4, SM = MR / (4 / GN), SN = BN / GN;
  const uint sm = sg / GN, sn = sg % GN;
  constexpr auto desc =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tg_t;
  auto tX0 = tg_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW0 = tg_t(ws + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto tX1 = tg_t(xs + MR * BK + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW1 = tg_t(ws + BN * BK + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tX0), decltype(tW0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  // X: MR rows x BK halfs per step, 8 halfs (one uint4) per thread and pass
  constexpr int XN = MR * BK / 8, XP = (XN + 127) / 128;  // uint4s, passes
  uint xrow[XP], xk[XP], xr[XP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < XP; ++i) {
    const uint j = tid + 128 * i;
    xr[i] = j / (BK / 8);
    xk[i] = (j % (BK / 8)) * 8;
    const uint m = m0 + xr[i];
    xrow[i] = j < XN && m < M ? (MODE == 1 ? sorted[m] / p.k : m) : 0xffffffffu;
  }
  // W: BN rows x BK per step, 16 per thread and pass
  constexpr int WP = BN * BK / 16 / 128;
  uint wr[WP], wk[WP], wn[WP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < WP; ++i) {
    const uint j = tid + 128 * i;
    wr[i] = j / (BK / 16);
    wk[i] = (j % (BK / 16)) * 16;
    wn[i] = min(n0 + wr[i], p.N - 1);
  }
  // double-buffered: step i + 1's loads are issued before step i's product, its tiles written to the other buffer
  // after it (one barrier per step)
  uint4 xv[XP];
  NhtRaw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(X + size_t(xrow[i]) * p.x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) wv[i] = nht_raw<FMT>(q, s, p.K, wn[i], k0 + wk[i]);
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) *(threadgroup uint4*)(xs + b * MR * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) nht_dec<FMT>(wv[i], ws + b * BN * BK + wr[i] * BK + wk[i]);
  };
  load(0);
  store(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint steps = p.K / BK;
  for (uint st = 0; st < steps; ++st) {
    const bool more = st + 1 < steps;
    if (more) load((st + 1) * BK);
    if (st & 1) op.run(tX1, tW1, acc);
    else op.run(tX0, tW0, acc);
    if (more) store((st + 1) & 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    const uint n = n0 + sn * SN + ix[0], m = m0 + sm * SM + ix[1];
    if (m >= M || n >= p.N) continue;
    const uint yrow = MODE == 2 ? sorted[m] : m;
    float v = acc[i] * gs;
    const size_t o = size_t(yrow) * p.y_ld + n;
    if (p.epi == 0) ((device float*)Y)[o] = v;
    else if (p.epi == 1) ((device float*)Y)[o] += v;
    else if (p.epi == 2) {
      v = max(v, 0.0f);
      ((device half*)Y)[o] = half(v * v);
    } else ((device half*)Y)[o] = half(v);
  }
}

template <int FMT, int MODE, int BM, int BN, int BK>
NI_INLINE void nht_gemm_impl(const device half* X, const device uchar* Wq, const device uchar* Ws, device void* Y,
                             constant NhGemmTP& p, const device uint4* tiles, const device uint* ntiles,
                             const device uint* sorted, const device float* gse, uint2 tg, uint tid, uint sg,
                             threadgroup half* xs, threadgroup half* ws) {
  const uint n0 = tg.x * BN;
  if (MODE == 0) {
    nht_tile<FMT, 0, BM, BN, BK>(X, Wq, Ws, Y, p, sorted, tg.y * BM, p.M, n0, FMT == 6 ? p.gs * 128.0f : p.gs, tid,
                                 sg, xs, ws);
    return;
  }
  if (tg.y >= ntiles[0]) return;
  const uint4 t = tiles[tg.y];
  const device uchar* q = Wq + size_t(t.x) * p.q_estride;
  const device uchar* s = Ws + size_t(t.x) * p.s_estride;
  const float gs = FMT == 1 ? p.gs * gse[t.x] : FMT == 6 ? p.gs * gse[t.x] * 128.0f : p.gs;
  // rows [t.y, t.y + t.z) of the sorted pair list
  nht_tile<FMT, MODE, BM, BN, BK>(X, q, s, Y, p, sorted, t.y, t.y + t.z, n0, gs, tid, sg, xs, ws);
}

#define NH_GEMM_T(NAME, FMT, MODE, BM, BN, BK)                                                                      \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]], \
      device void* Y [[buffer(3)]], constant NhGemmTP& p [[buffer(4)]], const device uint4* tiles [[buffer(5)]],  \
      const device uint* ntiles [[buffer(6)]], const device uint* sorted [[buffer(7)]],                          \
      const device float* gse [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup half xs[2 * BM * BK];                                                                            \
    threadgroup half ws[2 * BN * BK];                                                                            \
    nht_gemm_impl<FMT, MODE, BM, BN, BK>(X, Wq, Ws, Y, p, tiles, ntiles, sorted, gse, tg, tid, sg, xs, ws);      \
  }

NH_GEMM_T(nh_gemm_t_fp8_m64, 5, 0, 64, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_m64, 1, 0, 64, 64, 32)
NH_GEMM_T(nh_gemm_t_bf16w_m64, 4, 0, 64, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_moe_up_m16, 1, 1, 16, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_moe_dn_m16, 1, 2, 16, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_moe_up_m32, 1, 1, 32, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_moe_dn_m32, 1, 2, 32, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_moe_up_m64, 1, 1, 64, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_moe_dn_m64, 1, 2, 64, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4_moe_dn_m64n128, 1, 2, 64, 128, 32)
NH_GEMM_T(nh_gemm_t_fp4l2_m64, 6, 0, 64, 64, 32)  // NVFP4 in layout L2
NH_GEMM_T(nh_gemm_t_fp4l2_moe_up_m16, 6, 1, 16, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4l2_moe_dn_m16, 6, 2, 16, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4l2_moe_up_m32, 6, 1, 32, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4l2_moe_dn_m32, 6, 2, 32, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4l2_moe_up_m64, 6, 1, 64, 64, 32)
NH_GEMM_T(nh_gemm_t_fp4l2_moe_dn_m64n128, 6, 2, 64, 128, 32)
