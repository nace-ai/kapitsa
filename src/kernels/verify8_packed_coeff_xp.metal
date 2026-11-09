// External complete-cost XP + temporary-SB experiment. Append after frozen C1 MSL.
// Dependencies: qr_rc, qr_pair, v8pc_coeff, QmmParams, silu and NI_INLINE from C1.
// No persistent coefficient cache. One 64 MiB W/SB arena and one <=48 MiB XP slot.
// Host must order old readers -> all producers -> consumer -> scratch reuse.
// Only ORDER0, BK32, original dense32x128/GU32x64 geometry is instantiated.
// CPU address/source checks are not Metal numeric or performance validation.

// C1 unpack with only its coefficient destination converted to original qmr operands.
[[kernel]] void qmr_xpsb_unpack(
    const device uint* payload [[buffer(0)]],
    const device uint* headers [[buffer(1)]],
    device uint* weights [[buffer(2)]],
    device half2* cache [[buffer(3)]],
    constant uint2& shape [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint N = shape.x, K = shape.y, groups = K / 64;
  if (!N || !groups || K % 64 || ulong(i) >= ulong(N) * groups) return;
  const uint row = i / groups, g = i % groups;
  const uint block = (row / 16) * ((groups + 1) / 2) + g / 2;
  const device uint* record = payload + ulong(block) * 272;
  const uint fm = row & 7u, f = (row >> 3) & 1u;
  const uint offset = ((g & 1u) * 2 + f) * 64 + ((fm & 4u) << 3) + ((fm & 3u) << 2);
  const uint4 lo = *(const device uint4*)(record + offset);
  const uint4 hi = *(const device uint4*)(record + offset + 16);
  device uint4* dst = (device uint4*)(weights + ulong(i) * 8);
  dst[0] = lo;
  dst[1] = hi;
  const uint2 sb = v8pc_coeff(payload, record, headers[block], row & 15u);
  const uint shift = (g & 1u) * 16;
  const bfloat scale = as_type<bfloat>(ushort(sb.x >> shift));
  const bfloat bias = as_type<bfloat>(ushort(sb.y >> shift));
  cache[i] = half2(half(float(scale) * 4096.0f), half(float(bias)));
}

// Identical cached-SB loader from the preceding external prototype.
template <int FA, int BK, int NP>
NI_INLINE void qrsbc_load_w(const device uint32_t* const thread* W, const device half2* const thread* SB,
                         thread const uint* wrow, uint wpr, uint gpr,
                         uint k0, uint c, thread uint* wr, thread half2* sb) {
  constexpr int WPL = BK / 32;
  const uint g = (k0 + 8 * WPL * c) / 64;
  _Pragma("clang loop unroll(full)")
  for (int p = 0; p < NP; ++p)
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      const int i = p * FA + a;
      const device uint* wp = W[p] + wrow[a] * wpr + k0 / 8 + WPL * c;
      if (WPL == 4) {
        uint4 v = *(const device uint4*)wp;
        wr[i * 4 + 0] = v.x; wr[i * 4 + 1] = v.y; wr[i * 4 + 2] = v.z; wr[i * 4 + 3] = v.w;
      } else if (WPL == 2) {
        uint2 v = *(const device uint2*)wp;
        wr[i * 2 + 0] = v.x; wr[i * 2 + 1] = v.y;
      } else {
        wr[i] = wp[0];
      }
      if (BK >= 64 || (k0 & 63) == 0)
        sb[i] = SB[p][wrow[a] * gpr + g];
    }
}

// Historical XP path, mechanically adapted; see source-parity-v1.json.

// Experimental: materialize B-operand half2 pairs once per activation matrix.
// Format [M/8 tile][K/32 tile][four MMA fragments][32 lanes][two halfs].
// Round M to 32 so the main kernel may compute and discard padded output rows.
struct QrXpParams { uint M, K, x_ld; };
[[kernel]] void qmr_xpsb_pack_x(const device half* X [[buffer(0)]], device uint* XP [[buffer(1)]],
                           constant QrXpParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint mp = (p.M + 31) / 32 * 32;
  if (i >= mp * p.K / 2) return;
  const uint lane = i % 32, f = i / 32 % 4, kb = i / 128 % (p.K / 32), mb = i / (4 * p.K);
  const uint2 rc = qr_rc(lane);
  const uint m = mb * 8 + rc.y, k = kb * 32 + 8 * (rc.x / 2) + f + 4 * (rc.x % 2);
  const half2 v(m < p.M ? X[m * p.x_ld + k] : half(0), m + 1 < p.M ? X[(m + 1) * p.x_ld + k] : half(0));
  XP[i] = as_type<uint>(v);
}

template <int BM, int BN, int WM, int WN, int NP>
NI_INLINE void qrxsbc_acc(const device uint* XP, const device uint* const thread* W,
                       const device half2* const thread* SB,
                       uint N, uint K, uint m0, uint n0, uint sgm, uint sgn, uint lane, uint2 rc,
                       thread simdgroup_float8x8 (&acc)[NP * (BN / WN / 8)][BM / WM / 8]) {
  constexpr int TM = BM / WM, TN = BN / WN, FB = TM / 8, FA = TN / 8;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < NP * FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) acc[a][b] = simdgroup_float8x8(0);
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, N - 1);
  uint wr[NP * FA];
  half2 sb[NP * FA];
  const uint mb = (m0 + sgm * TM) / 8, xp_stride = K * 4;
  for (uint k0 = 0; k0 < K; k0 += 32) {
    qrsbc_load_w<FA, 32, NP>(W, SB, wrow, K / 8, K / 64, k0, rc.y / 2, wr, sb);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 4; ++f) {
      simdgroup_half8x8 af[NP * FA];
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < NP * FA; ++a) {
        const half2 v = qr_pair(wr[a], f, sb[a]);
        af[a].thread_elements()[0] = v.x;
        af[a].thread_elements()[1] = v.y;
      }
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < FB; ++b) {
        const half2 v = as_type<half2>(XP[(mb + b) * xp_stride + k0 * 4 + f * 32 + lane]);
        simdgroup_half8x8 bf;
        bf.thread_elements()[0] = v.x;
        bf.thread_elements()[1] = v.y;
        _Pragma("clang loop unroll(full)")
        for (int a = 0; a < NP * FA; ++a) simdgroup_multiply_accumulate(acc[a][b], af[a], bf, acc[a][b]);
      }
    }
  }
}

#define NI_QMRXPSB(NAME, BM, BN, WM, WN)                                                            \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device uint* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                         \
      const device half2* SBd [[buffer(2)]],                        \
      device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                   \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    constexpr int TM = BM / WM, TN = BN / WN;                                                              \
    const device uint32_t* W[1] = {Wd};                                                                    \
    const device half2* SB[1] = {SBd};                                                                      \
    const uint sgm = sg / WN, sgn = sg % WN;                               \
    const uint m0 = tg.y * BM, n0 = tg.x * BN;                                                             \
    const uint2 rc = qr_rc(lane);                                                                          \
    simdgroup_float8x8 acc[TN / 8][TM / 8];                                                                \
    qrxsbc_acc<BM, BN, WM, WN, 1>(X, W, SB, p.N, p.K, m0, n0, sgm, sgn, lane, rc, acc); \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int a = 0; a < TN / 8; ++a) {                                                                     \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                         \
      if (n >= p.N) continue;                                                                              \
      _Pragma("clang loop unroll(full)")                                                                   \
      for (int b = 0; b < TM / 8; ++b)                                                                     \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int e = 0; e < 2; ++e) {                                                                      \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                 \
          if (m < p.M) {                                                                                   \
            device float* dst = Y + m * p.y_ld + n;                                                        \
            const float v = acc[a][b].thread_elements()[e];                                                \
            *dst = p.add ? *dst + v : v;                                                                   \
          }                                                                                                \
        }                                                                                                  \
    }                                                                                                      \
  }

#define NI_QMR_GUXPSB(NAME, BM, BN, WM, WN)                                                               \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                    \
      const device uint* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],                         \
      const device half2* SBg [[buffer(2)]],                        \
      device half* H [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                    \
      const device uint32_t* Wu [[buffer(6)]], const device half2* SBu [[buffer(7)]],                      \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    constexpr int TM = BM / WM, TN = BN / WN, FA = TN / 8;                                                 \
    const device uint32_t* W[2] = {Wg, Wu};                                                                \
    const device half2* SB[2] = {SBg, SBu};                                                                  \
    const uint sgm = sg / WN, sgn = sg % WN;                               \
    const uint m0 = tg.y * BM, n0 = tg.x * BN;                                                             \
    const uint2 rc = qr_rc(lane);                                                                          \
    simdgroup_float8x8 acc[2 * FA][TM / 8];                                                                \
    qrxsbc_acc<BM, BN, WM, WN, 2>(X, W, SB, p.N, p.K, m0, n0, sgm, sgn, lane, rc, acc); \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int a = 0; a < FA; ++a) {                                                                         \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                         \
      if (n >= p.N) continue;                                                                              \
      _Pragma("clang loop unroll(full)")                                                                   \
      for (int b = 0; b < TM / 8; ++b)                                                                     \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int e = 0; e < 2; ++e) {                                                                      \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                 \
          const float g = acc[a][b].thread_elements()[e], u = acc[FA + a][b].thread_elements()[e];        \
          if (m < p.M) H[m * p.y_ld + n] = half(silu(g) * u);                                              \
        }                                                                                                  \
    }                                                                                                      \
  }

NI_QMRXPSB(qmr_xpsb_k32_w14_m32_n128, 32, 128, 1, 4)
NI_QMR_GUXPSB(qmr_xpsb_gu_k32_w14_m32_n64, 32, 64, 1, 4)

#undef NI_QMRXPSB
#undef NI_QMR_GUXPSB
