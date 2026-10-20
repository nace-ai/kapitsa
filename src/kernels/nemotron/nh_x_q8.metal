// Q8 (int8 codes, one half scale per 32 elements) products of up to 8 rows on nh_x8_v8's design (nh_x_fp8.metal):
// the DSpark drafter's attention projections (Lightning: q|k|v 4608x2688, o 2688x4096), BF16 in the checkpoint and
// Q8 as loaded. nh_v8_q8 (nh_v8.metal) reads 8 code bytes and a 2-byte scale per lane and step, and its X operands
// one half at a time; this reads 32 code bytes per lane and row (uint4 x 2) and X as uint4, as the x8 kernel does.
//
// Weights: the codes repacked as nh_x8 expects (nh_xq8_repack): in each 16-element group, byte 4 m + 0..3 holds
// elements 2 m, 2 m + 1, 2 m + 8, 2 m + 9; the scales as Q8 has them (half [N][K / 32]). A lane's 32 code bytes per
// row and step are one Q8 block (elements 32 c .. 32 c + 31 of the step), so its A pairs are its codes (exact in
// half: int8 + 1152 as half bits, then - 1152) times its block's scale, rounded to half (a relative 2^-12 per weight,
// against nh_v8_q8's exact integer MMA chains scaled in fp32: a drafter's weights only propose tokens).
//   nh_xq8_v8   R <= 8, NhV8P [[4]] (X half [R][x_ld], Y per p.epi), scales [[2]]. grid ceil(N / 16) x 256
// Requirements: K % 128 == 0; X rows 16-byte aligned (x_ld % 8 == 0); |scale| x 128 within half's range.

// bytes 0, 2 / 1, 3 of w (int8) as half2
NI_INLINE half2 nh_xq8_02(uint w) {
  return as_type<half2>(((w & 0x00FF00FFu) ^ 0x00800080u) | 0x64006400u) - half2(1152.0h);
}
NI_INLINE half2 nh_xq8_13(uint w) {
  return as_type<half2>((((w >> 8) & 0x00FF00FFu) ^ 0x00800080u) | 0x64006400u) - half2(1152.0h);
}

template <int NF>
NI_INLINE void nh_xq8_v8_frags(uint4 a0, uint4 a1, uint4 b0, uint4 b1, thread const uint4* c0, thread const uint4* c1,
                               thread const half* sc, thread float2* acc) {
  _Pragma("clang loop unroll(full)")
  for (int fr = 0; fr < 16; ++fr) {
    const uint wa = (fr < 8 ? a0 : a1)[(fr & 7) >> 1], wb = (fr < 8 ? b0 : b1)[(fr & 7) >> 1];
    const half2 bf = half2(as_type<half2>(wa)[fr & 1], as_type<half2>(wb)[fr & 1]);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      const uint w = (fr < 8 ? c0[f] : c1[f])[(fr & 7) >> 1];
      nh_x4_mma(acc[f], ((fr & 1) ? nh_xq8_13(w) : nh_xq8_02(w)) * sc[f], bf);
    }
  }
}

template <int NF>
NI_INLINE void nh_xq8_v8_impl(const device half* X, const device uchar* q, const device half* s, device void* Y,
                              constant NhV8P& p, uint2 tg, uint sg, uint nsg, uint lane, threadgroup float2* red) {
  const uint3 lc = nh_x4_lane(lane);
  const uint fm = lc.x, fn = lc.y, c = lc.z;
  const uint ks = sg, KS = nsg;
  const uint n0 = tg.x * 8 * NF;
  const uint N = p.N, K = p.K;
  const bool l0 = fn < p.R, l1 = fn + 1 < p.R;
  const uint bo = 32 * (fm >> 1) + 8 * (fm & 1);
  const device half* xa = X + size_t(l0 ? fn : 0) * p.x_ld + bo;
  const device half* xb = X + size_t(l1 ? fn + 1 : 0) * p.x_ld + bo;
  const device uchar* qr[NF];
  const device half* sr[NF];
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = min(n0 + 8 * f + fm, N - 1);
    qr[f] = q + size_t(n) * K + 32 * c;
    sr[f] = s + size_t(n) * (K / 32) + c;
    acc[f] = float2(0.0f);
  }
  if (n0 < N) {
    for (uint st = ks; st < K / 128; st += KS) {
      const uint k0 = st * 128;
      uint4 a0 = *(const device uint4*)(xa + k0), a1 = *(const device uint4*)(xa + k0 + 16);
      uint4 b0 = *(const device uint4*)(xb + k0), b1 = *(const device uint4*)(xb + k0 + 16);
      uint4 c0[NF], c1[NF];
      half sc[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        c0[f] = *(const device uint4*)(qr[f] + k0);
        c1[f] = *(const device uint4*)(qr[f] + k0 + 16);
        sc[f] = sr[f][k0 / 32];
      }
      if (!l0) a0 = a1 = uint4(0);
      if (!l1) b0 = b1 = uint4(0);
      nh_xq8_v8_frags<NF>(a0, a1, b0, b1, c0, c1, sc, acc);
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

[[kernel, max_total_threads_per_threadgroup(256)]] void nh_xq8_v8(
    const device half* X [[buffer(0)]], const device uchar* q [[buffer(1)]], const device half* s [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 2 * 32];
  nh_xq8_v8_impl<2>(X, q, s, Y, p, tg, sg, nsg, lane, red);
}
// 8 weight rows per threadgroup (twice the threadgroups; each output as nh_xq8_v8's, bit for bit). grid ceil(N / 8)
[[kernel, max_total_threads_per_threadgroup(256)]] void nh_xq8_v8_n1(
    const device half* X [[buffer(0)]], const device uchar* q [[buffer(1)]], const device half* s [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 32];
  nh_xq8_v8_impl<1>(X, q, s, Y, p, tg, sg, nsg, lane, red);
}
