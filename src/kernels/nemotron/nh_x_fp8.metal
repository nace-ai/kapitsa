// FP8 E4M3 products (one scale per tensor) for Nemotron-H's FP8 matrices (Nemotron 3.5 Super-VL: Mamba in_proj
// [18560][4096] and out_proj [4096][8192], attention q/k/v/o, the LM head [131072][4096]): one-row decode, 8-row
// verify, up to 32 rows, on the designs of nh_x_fp4.metal (same signatures, grids and split-K contract; its lane
// layout, MMA, store and staging helpers). Measured with tools/nh_x4_bench --fmt fp8.
//
// Weights: one byte per element in the loader's order (nh_repack_rows): in each 16-element group, byte 4 m + 0..3
// holds elements 2 m, 2 m + 1, 2 m + 8, 2 m + 9 (m = 0..3). An E4M3 byte b read as the half bits
// ((b & 0x80) << 8) | ((b & 0x7F) << 7) is its value x 2^-8, exactly (normals and subnormals alike; E4M3FN's NaN code,
// which ModelOpt never writes, is not special). So bytes 0, 2 of a word are the half2 (element 2 m, 2 m + 8) and bytes
// 1, 3 (2 m + 1, 2 m + 9): the pairs (i, i + 8) that nh_x4_v8 / nh_x4_mm take as A operands for NVFP4, whose B loads
// and X staging are reused as they are. The host's tensor scale includes the 2^8 (NhMat::gs = weight_scale x 256).
//
// Kernels:
//   nh_x8_mv[_add|_relu2]  R = 1, NhMvParams [[4]] (x fp32 [K], y fp32 [N]); buffer 1 (scales) unused. grid
//                          ceil(N / (NR x simdgroups)) x 32 x simdgroups (the engine: NR 4, 2 simdgroups)
//   nh_x8_v8               R <= 8, NhV8P [[4]] (X half [R][x_ld], Y per p.epi). grid ceil(N / 16) x 256
//   nh_x8_v8_n1            the same at 8 weight rows per threadgroup (one fragment per simdgroup: twice the
//                          threadgroups, each output to the bit as nh_x8_v8's). grid ceil(N / 8) x 256. The M5:
//                          tools/nh_x8_bench, Lightning's out_proj 2688x4096 24.6-24.7 vs 34.2-42.1 us, in_proj
//                          10304x2688 52.5-52.6 vs 54.1-59.1
//   nh_x8_mm16/24/32       R <= 16 / 24 / 32, NhV8P [[4]], part [[5]], cnt [[6]] as nh_x4_mm16/24/32
//   nh_x8_v8_t, nh_x8_mm16/24/32_t   the same for K % 128 != 0 (K % 32 == 0: the Nano 4B's hidden 3136): the last
//                          K % 128 inputs as one more step, its 32-element groups past K zero (read nothing); the full
//                          steps as the kernels above
// Requirements: K % 128 == 0 for v8 / mm (K % 32 for mv and the _t kernels); X rows 16-byte aligned (x_ld % 8 == 0).

// bytes 0, 2 / 1, 3 of w as half2, x 2^-8
NI_INLINE half2 nh_x8_02(uint w) { return as_type<half2>(((w & 0x00800080u) << 8) | ((w & 0x007F007Fu) << 7)); }
NI_INLINE half2 nh_x8_13(uint w) { return as_type<half2>((w & 0x80008000u) | ((w >> 1) & 0x3F803F80u)); }

// the A pair of fragment fr (0..15) from a lane's 32 code bytes (c0: group 0, c1: group 1): group fr / 8, pair fr % 8
NI_INLINE half2 nh_x8_pair(uint4 c0, uint4 c1, uint fr) {
  const uint w = (fr < 8 ? c0 : c1)[(fr & 7) >> 1];
  return (fr & 1) ? nh_x8_13(w) : nh_x8_02(w);
}

// ---- decode (R = 1): a simdgroup computes NR rows; lane l covers k = 32 l + 1024 i (two groups: two uint4 of codes)
template <int NR, int EPI>
NI_INLINE void nh_x8_mv_impl(const device uchar* q, const device float* x, device float* y, uint N, uint K, float gs,
                             uint row0, uint lane) {
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  if (row0 < N) {
    const device uchar* qr[NR];
    for (int j = 0; j < NR; ++j) qr[j] = q + size_t(min(row0 + uint(j), N - 1)) * K;
    for (uint k0 = lane * 32; k0 < K; k0 += 1024) {
      float4 xv[8];
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 8; ++i) xv[i] = ((const device float4*)(x + k0))[i];
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NR; ++j) {
        const uint4 c[2] = {*(const device uint4*)(qr[j] + k0), *(const device uint4*)(qr[j] + k0 + 16)};
        float d = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (int g = 0; g < 2; ++g)
          _Pragma("clang loop unroll(full)")
          for (int m = 0; m < 4; ++m) {
            const half2 a = nh_x8_02(c[g][m]), b = nh_x8_13(c[g][m]);  // (2m, 2m + 8), (2m + 1, 2m + 9)
            const float4 lo = xv[4 * g + (m >> 1)], hi = xv[4 * g + 2 + (m >> 1)];
            const int o = 2 * (m & 1);
            d = fma(float(a.x), lo[o], d);
            d = fma(float(b.x), lo[o + 1], d);
            d = fma(float(a.y), hi[o], d);
            d = fma(float(b.y), hi[o + 1], d);
          }
        acc[j] += d;
      }
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * gs;
    const uint n = row0 + uint(j);
    if (lane == 0 && n < N) {
      if (EPI == 1) v += y[n];
      if (EPI == 2) {
        v = max(v, 0.0f);
        v = v * v;
      }
      y[n] = v;
    }
  }
}

#define NH_X8_MV(NAME, NR, EPI)                                                                                    \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                   \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                        \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],         \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    nh_x8_mv_impl<NR, EPI>(q, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * NR, lane);                                \
  }
NH_X8_MV(nh_x8_mv, 4, 0)
NH_X8_MV(nh_x8_mv_add, 4, 1)
NH_X8_MV(nh_x8_mv_relu2, 4, 2)
NH_X8_MV(nh_x8_mv_n2, 2, 0)
NH_X8_MV(nh_x8_mv_n8, 8, 0)
NH_X8_MV(nh_x8_mv_add_n2, 2, 1)

// ---- 8 rows (verify): nh_x4_v8's design; the lane of column pair c reads 32 code bytes per row and step (elements
// 32 c .. 32 c + 31 of the 128-element step: groups 2c, 2c + 1) ----
// The 16 fragments of one 128-element step: X pairs a0 / a1 (token fn), b0 / b1 (token fn + 1), the NF rows' 32 code
// bytes c0 / c1 (groups 2c, 2c + 1)
template <int NF>
NI_INLINE void nh_x8_v8_frags(uint4 a0, uint4 a1, uint4 b0, uint4 b1, thread const uint4* c0, thread const uint4* c1,
                              thread float2* acc) {
  _Pragma("clang loop unroll(full)")
  for (int fr = 0; fr < 16; ++fr) {
    const uint wa = (fr < 8 ? a0 : a1)[(fr & 7) >> 1], wb = (fr < 8 ? b0 : b1)[(fr & 7) >> 1];
    const half2 bf = half2(as_type<half2>(wa)[fr & 1], as_type<half2>(wb)[fr & 1]);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) nh_x4_mma(acc[f], nh_x8_pair(c0[f], c1[f], uint(fr)), bf);
  }
}

template <bool TAIL, int NF = 2>
NI_INLINE void nh_x8_v8_impl(const device half* X, const device uchar* q, device void* Y, constant NhV8P& p, uint tg,
                             uint sg, uint nsg, uint lane, threadgroup float2* red) {
  const uint3 lc = nh_x4_lane(lane);
  const uint fm = lc.x, fn = lc.y, c = lc.z;
  const uint ks = sg, KS = nsg;
  const uint n0 = tg * 8 * NF;
  const uint N = p.N, K = p.K;
  const bool l0 = fn < p.R, l1 = fn + 1 < p.R;
  const uint bo = 32 * (fm >> 1) + 8 * (fm & 1);
  const device half* xa = X + size_t(l0 ? fn : 0) * p.x_ld + bo;
  const device half* xb = X + size_t(l1 ? fn + 1 : 0) * p.x_ld + bo;
  const device uchar* qr[NF];
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = min(n0 + 8 * f + fm, N - 1);
    qr[f] = q + size_t(n) * K + 32 * c;
    acc[f] = float2(0.0f);
  }
  if (n0 < N) {
    for (uint st = ks; st < K / 128; st += KS) {
      const uint k0 = st * 128;
      uint4 a0 = *(const device uint4*)(xa + k0), a1 = *(const device uint4*)(xa + k0 + 16);
      uint4 b0 = *(const device uint4*)(xb + k0), b1 = *(const device uint4*)(xb + k0 + 16);
      uint4 c0[NF], c1[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        c0[f] = *(const device uint4*)(qr[f] + k0);
        c1[f] = *(const device uint4*)(qr[f] + k0 + 16);
      }
      if (!l0) a0 = a1 = uint4(0);
      if (!l1) b0 = b1 = uint4(0);
      nh_x8_v8_frags<NF>(a0, a1, b0, b1, c0, c1, acc);
    }
    // TAIL: the last K % 128 inputs (K % 32 == 0) as one more step, the simdgroup next in the interleave's: its
    // 32-element groups at and past K are zero and read nothing (a lane's A pairs: its column pair c's group; its B
    // rows: k-slot fm's group, fm >> 1)
    if (TAIL && K % 128 != 0 && (K / 128) % KS == ks) {
      const uint k0 = K / 128 * 128, ng = (K % 128) / 32;
      const bool xok = (fm >> 1) < ng, wok = c < ng;
      uint4 a0 = uint4(0), a1 = uint4(0), b0 = uint4(0), b1 = uint4(0);
      if (xok && l0) a0 = *(const device uint4*)(xa + k0), a1 = *(const device uint4*)(xa + k0 + 16);
      if (xok && l1) b0 = *(const device uint4*)(xb + k0), b1 = *(const device uint4*)(xb + k0 + 16);
      uint4 c0[NF], c1[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        c0[f] = wok ? *(const device uint4*)(qr[f] + k0) : uint4(0);
        c1[f] = wok ? *(const device uint4*)(qr[f] + k0 + 16) : uint4(0);
      }
      nh_x8_v8_frags<NF>(a0, a1, b0, b1, c0, c1, acc);
    }
  }
  if (ks)
    for (int f = 0; f < NF; ++f) red[(sg * NF + uint(f)) * 32 + lane] = acc[f];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks) return;
  for (uint j = 1; j < KS; ++j)
    for (int f = 0; f < NF; ++f) acc[f] += red[(j * NF + uint(f)) * 32 + lane];
  for (int f = 0; f < NF; ++f) {
    const uint n = n0 + 8 * f + fm;
    if (n >= N) continue;
    if (l0) nh_x4_store(Y, p, n, fn, acc[f].x * p.gs);
    if (l1) nh_x4_store(Y, p, n, fn + 1, acc[f].y * p.gs);
  }
}

// grid: ceil(N / 16) threadgroups x 256 threads (8 simdgroups split K)
[[kernel, max_total_threads_per_threadgroup(256)]] void nh_x8_v8(
    const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 2 * 32];
  nh_x8_v8_impl<false>(X, Wq, Y, p, tg.x, sg, nsg, lane, red);
}
[[kernel, max_total_threads_per_threadgroup(256)]] void nh_x8_v8_n1(
    const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 32];
  nh_x8_v8_impl<false, 1>(X, Wq, Y, p, tg.x, sg, nsg, lane, red);
}
[[kernel, max_total_threads_per_threadgroup(256)]] void nh_x8_v8_t(
    const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 2 * 32];
  nh_x8_v8_impl<true>(X, Wq, Y, p, tg.x, sg, nsg, lane, red);
}

// ---- up to 32 rows: nh_x4_mm's design (X staged per 128-element step in the A operands' order, split K summed in a
// fixed order by the last part to arrive) ----
// TAIL: K % 128 != 0 (K % 32 == 0): the last K % 128 inputs are one more step (in the last part), staged with its
// 16-element blocks at and past K zero and its column pairs' codes there zero (none of them read)
template <int TB, bool TAIL>
NI_INLINE void nh_x8_mm_impl(const device half* X, const device uchar* q, device void* Y, constant NhV8P& p, uint tile,
                             uint pi, uint SK, uint sg, uint nsg, uint lane, uint tid, threadgroup half* xs,
                             device float2* part, device atomic_uint* cnt, threadgroup uint* arrival) {
  constexpr int NF = 2;
  const uint3 lc = nh_x4_lane(lane);
  const uint fm = lc.x, fn = lc.y, c = lc.z;
  const uint n0 = (tile * nsg + sg) * 8 * NF;
  const uint N = p.N, K = p.K, nthr = nsg * 32, S_all = TAIL ? (K + 127) / 128 : K / 128, per = (S_all + SK - 1) / SK;
  const uint st_lo = min(pi * per, S_all), S = min(st_lo + per, S_all) - st_lo;
  const device uchar* qr[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = min(n0 + 8 * f + fm, N - 1);
    qr[f] = q + size_t(n) * K + 32 * c + st_lo * 128;
  }
  const device half* xp = X + st_lo * 128;
  simdgroup_float8x8 acc[NF][TB];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < TB; ++b) acc[f][b] = simdgroup_float8x8(0);
  for (uint st = 0; st < S; ++st) {
    // the step's inputs before K: 128, or with TAIL the last step's K % 128 (a multiple of 32)
    const uint valid = TAIL ? min(128u, K - (st_lo + st) * 128) : 128u;
    for (uint i = tid; i < 8 * TB * 8; i += nthr) {
      const uint t = i >> 3, pb = i & 7;
      uint4 v0 = uint4(0), v1 = uint4(0);
      if (t < p.R && (!TAIL || 16 * pb < valid)) {
        const device uint4* src = (const device uint4*)(xp + size_t(t) * p.x_ld + st * 128 + 16 * pb);
        v0 = src[0];
        v1 = src[1];
      }
      threadgroup uint* d = (threadgroup uint*)(xs + t * kX4Ld + 64 * (pb & 1) + 2 * (pb >> 1));
      d[0] = (v0.x & 0xFFFFu) | (v1.x << 16);
      d[4] = (v0.x >> 16) | (v1.x & 0xFFFF0000u);
      d[8] = (v0.y & 0xFFFFu) | (v1.y << 16);
      d[12] = (v0.y >> 16) | (v1.y & 0xFFFF0000u);
      d[16] = (v0.z & 0xFFFFu) | (v1.z << 16);
      d[20] = (v0.z >> 16) | (v1.z & 0xFFFF0000u);
      d[24] = (v0.w & 0xFFFFu) | (v1.w << 16);
      d[28] = (v0.w >> 16) | (v1.w & 0xFFFF0000u);
    }
    uint4 c0[NF], c1[NF];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      if (!TAIL || 32 * c < valid) {
        c0[f] = *(const device uint4*)(qr[f] + st * 128);
        c1[f] = *(const device uint4*)(qr[f] + st * 128 + 16);
      } else {
        c0[f] = c1[f] = uint4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (int fr = 0; fr < 16; ++fr) {
      simdgroup_half8x8 A[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        const half2 av = nh_x8_pair(c0[f], c1[f], uint(fr));
        A[f].thread_elements()[0] = av.x;
        A[f].thread_elements()[1] = av.y;
      }
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < TB; ++b) {
        simdgroup_half8x8 B;
        simdgroup_load(B, xs + (8 * b) * kX4Ld + fr * 8, kX4Ld, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < NF; ++f) simdgroup_multiply_accumulate(acc[f][b], A[f], B, acc[f][b]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (SK > 1) {
#if NH_FENCES
    device float2* base = part + size_t(tile) * SK * nsg * (NF * TB * 32);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f)
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < TB; ++b)
        base[((pi * nsg + sg) * NF + uint(f)) * (TB * 32) + uint(b) * 32 + lane] =
            float2(acc[f][b].thread_elements()[0], acc[f][b].thread_elements()[1]);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival[0] = atomic_fetch_add_explicit(cnt + tile, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival[0] != SK - 1) return;
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f)
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < TB; ++b) {
        float2 v = float2(0.0f);
        for (uint j = 0; j < SK; ++j) v += base[((j * nsg + sg) * NF + uint(f)) * (TB * 32) + uint(b) * 32 + lane];
        acc[f][b].thread_elements()[0] = v.x;
        acc[f][b].thread_elements()[1] = v.y;
      }
    if (tid == 0) atomic_store_explicit(cnt + tile, 0u, memory_order_relaxed);
#else
    return;  // (no merge without device-scope fences: the engine runs SK = 1, NH_FENCES in nh_common.h)
#endif
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = n0 + 8 * f + fm;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < TB; ++b)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint t = 8 * b + fn + e;
        if (t < p.R) nh_x4_store(Y, p, n, t, acc[f][b].thread_elements()[e] * p.gs);
      }
  }
}

// grid: (SK, ceil(N / 64)) threadgroups x 128 threads; part / cnt as nh_x4_mm16/24/32
#define NH_X8_MM(NAME, TB, TAIL)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
                       const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]],                   \
                       const device uchar* Ws [[buffer(2)]], device void* Y [[buffer(3)]],                        \
                       constant NhV8P& p [[buffer(4)]], device float2* part [[buffer(5)]],                        \
                       device atomic_uint* cnt [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],          \
                       uint2 ntg [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],            \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       uint tid [[thread_index_in_threadgroup]]) {                                                 \
    threadgroup half xs[8 * TB * kX4Ld];                                                                           \
    threadgroup uint arrival[1];                                                                                   \
    if (ntg.x > 16u) return;                                                                                       \
    nh_x8_mm_impl<TB, TAIL>(X, Wq, Y, p, tg.y, tg.x, ntg.x, sg, nsg, lane, tid, xs, part, cnt, arrival);           \
  }
NH_X8_MM(nh_x8_mm16, 2, false)
NH_X8_MM(nh_x8_mm24, 3, false)
NH_X8_MM(nh_x8_mm32, 4, false)
NH_X8_MM(nh_x8_mm16_t, 2, true)
NH_X8_MM(nh_x8_mm24_t, 3, true)
NH_X8_MM(nh_x8_mm32_t, 4, true)
