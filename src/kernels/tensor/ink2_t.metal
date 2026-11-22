#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- Inkling-2-Mini prefill GEMMs on Apple10+ matrix units (Metal 4 tensor_ops) ----
//
// Double-buffered 64x64x32 (and 16/32x64x32 for grouped MoE) matmul2d tiles with FP16 threadgroup operands and FP32
// accumulators, mirroring nh_gemm_t.metal for Inkling-2's weight formats:
//   FMT 0: BF16 dense weights [N, K], float X [M, x_ld] -> float Y [M, y_ld] (MODE 0, optional add)
//   FMT 1: NVFP4 MoE experts [E, N, K/2] (adjacent nibbles 2b, 2b+1) + F8E4M3 block-16 scales + per-expert outer[e]
//   FMT 2: MXFP8 MoE experts [E, N, K] (F8E4M3) + UE8M0 block-32 scales
// Modes:
//   MODE 0: dense GEMM (float X -> float Y)
//   MODE 1: grouped MoE w13 (float X indexed by sorted[m] / k -> float Y in sorted order m)
//   MODE 2: grouped MoE w2  (half X in sorted order m -> float Y scattered to sorted[m])

struct Ink2tGemmP {
  uint M, N, K;
  uint x_ld, y_ld;
  uint add;
  uint q_estride, s_estride;
  uint k;
};

constant float ink2t_e2m1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

NI_INLINE float ink2t_e4m3(uint b) {
  const ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return float(as_type<half>(h)) * 256.0f;
}

NI_INLINE float ink2t_ue8m0(uint b) {
  return b == 0u ? 5.8774718e-39f : as_type<float>(b << 23);
}

struct Ink2tRaw {
  uint4 a, b;
};

template <int FMT>
NI_INLINE Ink2tRaw ink2t_raw(const device uchar* q, const device uchar* s, uint K, uint n, uint k) {
  Ink2tRaw r;
  if (FMT == 0) {
    const device uint4* wp = (const device uint4*)(q + size_t(n) * K * 2 + k * 2);
    r.a = wp[0];
    r.b = wp[1];
  } else if (FMT == 1) {
    r.a.xy = *(const device uint2*)(q + size_t(n) * (K / 2) + k / 2);
    r.a.z = s[size_t(n) * (K / 16) + k / 16];
  } else {
    r.a = *(const device uint4*)(q + size_t(n) * K + k);
    r.b.x = s[size_t(n) * (K / 32) + k / 32];
  }
  return r;
}

template <int FMT>
NI_INLINE void ink2t_dec(const Ink2tRaw r, threadgroup half* dst) {
  half w[16];
  if (FMT == 0) {
    for (int h = 0; h < 2; ++h) {
      const uint4 v = h ? r.b : r.a;
      for (int i = 0; i < 4; ++i) {
        w[h * 8 + 2 * i] = half(as_type<float>(v[i] << 16));
        w[h * 8 + 2 * i + 1] = half(as_type<float>(v[i] & 0xFFFF0000u));
      }
    }
  } else if (FMT == 1) {
    const float sc = ink2t_e4m3(r.a.z);
    for (int j = 0; j < 8; ++j) {
      const uint b = ((j < 4 ? r.a.x : r.a.y) >> (8 * (j & 3))) & 0xFFu;
      w[2 * j] = half(ink2t_e2m1[b & 15u] * sc);
      w[2 * j + 1] = half(ink2t_e2m1[b >> 4] * sc);
    }
  } else {
    const float sc = ink2t_ue8m0(r.b.x);
    for (int m = 0; m < 4; ++m) {
      const uint v = r.a[m];
      w[4 * m + 0] = half(ink2t_e4m3(v & 0xFFu) * sc);
      w[4 * m + 1] = half(ink2t_e4m3((v >> 8) & 0xFFu) * sc);
      w[4 * m + 2] = half(ink2t_e4m3((v >> 16) & 0xFFu) * sc);
      w[4 * m + 3] = half(ink2t_e4m3((v >> 24) & 0xFFu) * sc);
    }
  }
  *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
  *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
  *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
  *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
}

template <int FMT, int MODE, int MR, int BN, int BK>
NI_INLINE void ink2t_tile(const device void* X, const device uchar* q, const device uchar* s, device float* Y,
                          constant Ink2tGemmP& p, const device uint* sorted, uint m0, uint M, uint n0, float gs,
                          uint tid, uint sg, threadgroup half* xs, threadgroup half* ws) {
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

  constexpr int XN = MR * BK / 8, XP = (XN + 127) / 128;
  uint xrow[XP], xk[XP], xr[XP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < XP; ++i) {
    const uint j = tid + 128 * i;
    xr[i] = j / (BK / 8);
    xk[i] = (j % (BK / 8)) * 8;
    const uint m = m0 + xr[i];
    xrow[i] = j < XN && m < M ? (MODE == 1 ? sorted[m] / p.k : m) : 0xffffffffu;
  }

  constexpr int WP = BN * BK / 16 / 128;
  uint wr[WP], wk[WP], wn[WP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < WP; ++i) {
    const uint j = tid + 128 * i;
    wr[i] = j / (BK / 16);
    wk[i] = (j % (BK / 16)) * 16;
    wn[i] = min(n0 + wr[i], p.N - 1);
  }

  half4 xh0[XP], xh1[XP];
  Ink2tRaw wv[WP];
  auto load = [&](uint k0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i) {
      if (xrow[i] != 0xffffffffu) {
        if (MODE == 2) {
          const device half* xp = (const device half*)X + size_t(xrow[i]) * p.x_ld + k0 + xk[i];
          xh0[i] = *(const device half4*)(xp);
          xh1[i] = *(const device half4*)(xp + 4);
        } else {
          const device float* xp = (const device float*)X + size_t(xrow[i]) * p.x_ld + k0 + xk[i];
          xh0[i] = half4(*(const device float4*)(xp));
          xh1[i] = half4(*(const device float4*)(xp + 4));
        }
      } else {
        xh0[i] = half4(0.0h);
        xh1[i] = half4(0.0h);
      }
    }
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) wv[i] = ink2t_raw<FMT>(q, s, p.K, wn[i], k0 + wk[i]);
  };
  auto store = [&](uint b) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!(XN % 128) || tid + 128 * i < XN) {
        threadgroup half* dst = xs + b * MR * BK + xr[i] * BK + xk[i];
        *(threadgroup half4*)(dst) = xh0[i];
        *(threadgroup half4*)(dst + 4) = xh1[i];
      }
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) ink2t_dec<FMT>(wv[i], ws + b * BN * BK + wr[i] * BK + wk[i]);
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
    const float v = acc[i] * gs;
    const size_t o = size_t(yrow) * p.y_ld + n;
    if (MODE == 0 && p.add) Y[o] += v;
    else Y[o] = v;
  }
}

template <int FMT, int MODE, int BM, int BN, int BK>
NI_INLINE void ink2t_gemm_impl(const device void* X, const device uchar* Wq, const device uchar* Ws, device float* Y,
                               constant Ink2tGemmP& p, const device uint4* tiles, const device uint* ntiles,
                               const device uint* sorted, const device float* gse, uint2 tg, uint tid, uint sg,
                               threadgroup half* xs, threadgroup half* ws) {
  const uint n0 = tg.x * BN;
  if (MODE == 0) {
    ink2t_tile<FMT, 0, BM, BN, BK>(X, Wq, Ws, Y, p, sorted, tg.y * BM, p.M, n0, 1.0f, tid, sg, xs, ws);
    return;
  }
  if (tg.y >= ntiles[0]) return;
  const uint4 t = tiles[tg.y];
  const device uchar* q = Wq + size_t(t.x) * p.q_estride;
  const device uchar* s = Ws + size_t(t.x) * p.s_estride;
  const float gs = FMT == 1 ? gse[t.x] : 1.0f;
  ink2t_tile<FMT, MODE, BM, BN, BK>(X, q, s, Y, p, sorted, t.y, t.y + t.z, n0, gs, tid, sg, xs, ws);
}

#define INK2T_GEMM(NAME, FMT, MODE, BM, BN, BK)                                                                        \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                        \
      const device void* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]],  \
      device float* Y [[buffer(3)]], constant Ink2tGemmP& p [[buffer(4)]], const device uint4* tiles [[buffer(5)]],    \
      const device uint* ntiles [[buffer(6)]], const device uint* sorted [[buffer(7)]],                                \
      const device float* gse [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],                                \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                          \
    threadgroup half xs[2 * BM * BK];                                                                                  \
    threadgroup half ws[2 * BN * BK];                                                                                  \
    ink2t_gemm_impl<FMT, MODE, BM, BN, BK>(X, Wq, Ws, Y, p, tiles, ntiles, sorted, gse, tg, tid, sg, xs, ws);          \
  }

INK2T_GEMM(ink2_t_bf16_m64, 0, 0, 64, 64, 32)
INK2T_GEMM(ink2_t_nvfp4_moe_w13_m16, 1, 1, 16, 64, 32)
INK2T_GEMM(ink2_t_nvfp4_moe_w13_m32, 1, 1, 32, 64, 32)
INK2T_GEMM(ink2_t_nvfp4_moe_w13_m64, 1, 1, 64, 64, 32)
INK2T_GEMM(ink2_t_nvfp4_moe_w2_m16, 1, 2, 16, 64, 32)
INK2T_GEMM(ink2_t_nvfp4_moe_w2_m32, 1, 2, 32, 64, 32)
INK2T_GEMM(ink2_t_nvfp4_moe_w2_m64, 1, 2, 64, 64, 32)
INK2T_GEMM(ink2_t_mxfp8_moe_w13_m16, 2, 1, 16, 64, 32)
INK2T_GEMM(ink2_t_mxfp8_moe_w13_m32, 2, 1, 32, 64, 32)
INK2T_GEMM(ink2_t_mxfp8_moe_w13_m64, 2, 1, 64, 64, 32)
INK2T_GEMM(ink2_t_mxfp8_moe_w2_m16, 2, 2, 16, 64, 32)
INK2T_GEMM(ink2_t_mxfp8_moe_w2_m32, 2, 2, 32, 64, 32)
INK2T_GEMM(ink2_t_mxfp8_moe_w2_m64, 2, 2, 64, 64, 32)
