// BF16 matrices up to 8 rows (verify, drafter blocks) on nh_x4_v8's design: Nemotron 3.5 Lightning's attention
// projections (q [4096][2688], k / v [256][2688], o [2688][4096]), which the generic tile GEMM (nh_gemm_bf16w) ran at
// ~110 GB/s for 8 rows. Weights as stored (bf16 bits, row-major); a lane's A pair (elements i, i + 8 of a 16-element
// block) is converted to half as nh_gemm_bf16w converts (exact for |w| in [6.1e-5, 65504]), so verify and prefill take
// the same operands. Same signature, grid and requirements as nh_x8_v8 (buffer 2 unused; K % 128 == 0, x_ld % 8 == 0):
//   nh_xb_v8   R <= 8, NhV8P [[4]] (X half [R][x_ld], Y per p.epi). grid ceil(N / 16) x 256

NI_INLINE half nh_xb_h(uint w, uint hi) { return half(as_type<float>(hi ? (w & 0xFFFF0000u) : (w << 16))); }

NI_INLINE void nh_xb_v8_impl(const device half* X, const device uchar* q, device void* Y, constant NhV8P& p, uint tg,
                             uint sg, uint nsg, uint lane, threadgroup float2* red) {
  constexpr int NF = 2;
  const uint3 lc = nh_x4_lane(lane);
  const uint fm = lc.x, fn = lc.y, c = lc.z;
  const uint ks = sg, KS = nsg;
  const uint n0 = tg * 8 * NF;
  const uint N = p.N, K = p.K;
  const bool l0 = fn < p.R, l1 = fn + 1 < p.R;
  const uint bo = 32 * (fm >> 1) + 8 * (fm & 1);
  const device half* xa = X + size_t(l0 ? fn : 0) * p.x_ld + bo;
  const device half* xb = X + size_t(l1 ? fn + 1 : 0) * p.x_ld + bo;
  const device uint4* qr[NF];
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = min(n0 + 8 * f + fm, N - 1);
    qr[f] = (const device uint4*)(q + (size_t(n) * K + 32 * c) * 2);  // elements 32 c .. 32 c + 31 of each step
    acc[f] = float2(0.0f);
  }
  if (n0 < N) {
    for (uint st = ks; st < K / 128; st += KS) {
      const uint k0 = st * 128;
      uint4 a0 = *(const device uint4*)(xa + k0), a1 = *(const device uint4*)(xa + k0 + 16);
      uint4 b0 = *(const device uint4*)(xb + k0), b1 = *(const device uint4*)(xb + k0 + 16);
      uint4 w[NF][4];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f)
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 4; ++i) w[f][i] = qr[f][k0 / 8 + i];
      if (!l0) a0 = a1 = uint4(0);
      if (!l1) b0 = b1 = uint4(0);
      _Pragma("clang loop unroll(full)")
      for (int fr = 0; fr < 16; ++fr) {
        const uint wa = (fr < 8 ? a0 : a1)[(fr & 7) >> 1], wb = (fr < 8 ? b0 : b1)[(fr & 7) >> 1];
        const half2 bf = half2(as_type<half2>(wa)[fr & 1], as_type<half2>(wb)[fr & 1]);
        // A pair: local elements e = 16 (fr >> 3) + (fr & 7) and e + 8; word e / 2 of the lane's 16, half e % 2
        const uint e = 16 * (uint(fr) >> 3) + (uint(fr) & 7);
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < NF; ++f) {
          const uint w0 = w[f][e >> 3][(e >> 1) & 3], w1 = w[f][(e + 8) >> 3][((e + 8) >> 1) & 3];
          nh_x4_mma(acc[f], half2(nh_xb_h(w0, e & 1), nh_xb_h(w1, e & 1)), bf);
        }
      }
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
[[kernel, max_total_threads_per_threadgroup(256)]] void nh_xb_v8(
    const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 2 * 32];
  nh_xb_v8_impl(X, Wq, Y, p, tg.x, sg, nsg, lane, red);
}
