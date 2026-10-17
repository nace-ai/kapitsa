// IQ2_XXS kernels for Nemotron 3.5 Bolt (helper set, every symbol nh_x2_*): decode matvecs, routed experts.
//
// Decode of one grid point (8 weights) without per-weight conversions or selects:
//   - The grid point's 8 magnitude bytes come from nh_iq2xxs_grid as a uint2 (one 8-byte gather; 16-byte gathers,
//     e.g. the grid as halves, cost about 4x more on the M4 Pro).
//   - A magnitude byte b (8, 25 or 43) read as the bits of a half is the subnormal b * 2^-24, exact; the kernels scale
//     by 2^24 once at the end. So `word & 0x807F807F` turns two bytes of the grid word into a half2 of weights.
//   - Signs: bit 7 of every byte is free (b < 64). Each element's sign goes into bit 7 of another byte of the same
//     32-bit word (one multiply spreads the 7 stored sign bits, popcount gives the parity bit), so that word & M gives
//     elements (0, 2) / (4, 6) with their signs and rotate(word, 24) & M elements (1, 3) / (5, 7).
//   - fp32 FMAs read the halves directly (a 16-bit source costs nothing extra), fp32 activations and accumulators.
// About 26 instructions and one 8-byte gather per 8 weights, against ~45 and a gather for nh_mv1_iq2_c. On the M4 Pro
// these kernels are ALU-bound near 175-195 GB/s; every load instruction is expensive (a threadgroup or 16-byte
// table lookup costs more than ten ALU instructions), so rows per simdgroup (NR) are as many as fit in registers
// (the 32 activations of a lane's sub-block stay in registers for all NR rows), and 32-thread threadgroups schedule
// better than 128-thread ones (5-10 %).
//
// Lane mapping (decode): one simdgroup, NR consecutive rows; lane l takes the 32-element sub-blocks l, l + 32, ... of
// every row (full 1024-element steps); the K % 1024 tail of the NR rows is spread over the lanes (item t: row t / T,
// sub-block t % T), so K = 5376 (shared down) or 2304 (routed down) keep every lane busy.

// d + the 8 weights of one grid point (G: grid word pair; s7: its 7 sign bits) . (xa, xb), weights as subnormals
NI_INLINE float nh_x2_dot8(uint2 G, uint s7, float4 xa, float4 xb, float d) {
  // lo: e0 sign -> bit 15, e1 -> 23, e2 -> 31, e3 -> 7; hi: e4 -> 15, e5 -> 23, e6 -> 31, e7 (parity) -> 7
  const uint lo_m = (s7 * 0x20408010u) & 0x80808080u;
  const uint hi_m = ((s7 * 0x02040800u) | (popcount(s7) << 7)) & 0x80808080u;
  const uint lo = G.x ^ lo_m, hi = G.y ^ hi_m;
  const uint M = 0x807F807Fu;
  const half2 w0 = as_type<half2>(lo & M), w1 = as_type<half2>(rotate(lo, 24u) & M);  // (e0, e2), (e1, e3)
  const half2 w2 = as_type<half2>(hi & M), w3 = as_type<half2>(rotate(hi, 24u) & M);  // (e4, e6), (e5, e7)
  d = fma(float(w0.x), xa.x, d);
  d = fma(float(w1.x), xa.y, d);
  d = fma(float(w0.y), xa.z, d);
  d = fma(float(w1.y), xa.w, d);
  d = fma(float(w2.x), xb.x, d);
  d = fma(float(w3.x), xb.y, d);
  d = fma(float(w2.y), xb.z, d);
  d = fma(float(w3.y), xb.w, d);
  return d;
}

// one 32-element sub-block (codes aux) . xv[0..7], times 2^-24 (one FMA chain)
NI_INLINE float nh_x2_dot32(uint2 aux, thread const float4* xv) {
  constant uint2* gt = (constant uint2*)nh_iq2xxs_grid;
  float d = nh_x2_dot8(gt[aux.x & 0xFFu], extract_bits(aux.y, 0, 7), xv[0], xv[1], 0.0f);
  d = nh_x2_dot8(gt[extract_bits(aux.x, 8, 8)], extract_bits(aux.y, 7, 7), xv[2], xv[3], d);
  d = nh_x2_dot8(gt[extract_bits(aux.x, 16, 8)], extract_bits(aux.y, 14, 7), xv[4], xv[5], d);
  return nh_x2_dot8(gt[aux.x >> 24], extract_bits(aux.y, 21, 7), xv[6], xv[7], d);
}

// the sub-block scale (0.5 + s4) / 4 as the subnormal half (2 s4 + 1) 2^-24 (times 2^21: with the weights' 2^-24,
// results are scaled by 2^45 at the end)
NI_INLINE float nh_x2_sub(uint auxy) { return float(as_type<half>(ushort((auxy >> 27) | 1u))); }
constant constexpr float kNhX2Scale = 35184372088832.0f;  // 2^45

// y[n] = epi(gs * W[n, :] . x) for rows row0 .. row0 + NR - 1 (clamped to N; EPI 0 store, 1 add, 2 relu^2), one
// simdgroup. Same contract as nh_mv1_iq2v<NR, EPI> (K % 32 == 0).
template <int NR, int EPI>
NI_INLINE void nh_x2_rows(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                          uint N, uint K, float gs, uint row0, uint lane) {
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  const uint KF = K & ~1023u;
  const device uchar* qr[NR];
  const device half* sr[NR];
  for (int j = 0; j < NR; ++j) {
    const uint n = min(row0 + j, N - 1);
    qr[j] = q + size_t(n) * (K / 4);
    sr[j] = (const device half*)s + size_t(n) * (K / 256);
  }
  for (uint k0 = lane * 32; k0 < KF; k0 += 1024) {
    const device float4* xp = (const device float4*)(x + k0);
    float4 xv[8];
    for (int i = 0; i < 8; ++i) xv[i] = xp[i];
    for (int j = 0; j < NR; ++j) {
      const uint2 aux = *(const device uint2*)(qr[j] + k0 / 4);
      acc[j] = fma(float(sr[j][k0 / 256]) * nh_x2_sub(aux.y), nh_x2_dot32(aux, xv), acc[j]);
    }
  }
  if (KF < K) {  // the tail sub-blocks of the NR rows over the lanes
    const uint T = (K - KF) / 32;
    for (uint t = lane; t < NR * T; t += 32) {
      const uint jt = t / T, k0 = KF + (t - jt * T) * 32;
      const device float4* xp = (const device float4*)(x + k0);
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = xp[i];
      const device uchar* qt = qr[0];
      const device half* st = sr[0];
      for (int j = 1; j < NR; ++j)
        if (jt == uint(j)) {
          qt = qr[j];
          st = sr[j];
        }
      const uint2 aux = *(const device uint2*)(qt + k0 / 4);
      const float v = float(st[k0 / 256]) * nh_x2_sub(aux.y) * nh_x2_dot32(aux, xv);
      for (int j = 0; j < NR; ++j) acc[j] += jt == uint(j) ? v : 0.0f;
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * (gs * kNhX2Scale);
    const uint n = row0 + j;
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

// ---- one-row matvecs, the signature of nh_mv1_iq2_c[_add|_relu2] (NhMvParams): grid ceil(N / NR) x 32 threads ----
#define NH_X2_MV(NAME, NR, EPI)                                                                                    \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                   \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                        \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],         \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    nh_x2_rows<NR, EPI>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * NR, lane);                                \
  }
NH_X2_MV(nh_x2_mv4, 4, 0)
NH_X2_MV(nh_x2_mv4_relu2, 4, 2)  // shared expert up (5376 x 4096)
NH_X2_MV(nh_x2_mv3_add, 3, 1)    // shared expert down (4096 x 5376)
NH_X2_MV(nh_x2_mv4_add, 4, 1)

// ---- routed experts per (pair, NR rows), the signatures of nh_moe2_up_iq2 / nh_moe2_dn_iq2 (NhMoeParams):
// up grid (ceil(I / 4), R * k) x 32 threads; down grid (ceil(D / NR), R * k) x 32 threads; then nh_moe2_sum ----
[[kernel]] void nh_x2_moe_up4(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                              const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                              const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],
                              const device uint* live [[buffer(7)]], uint3 tgp [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint pair = tgp.y, r = pair / p.k;
  if (r >= live[0]) return;
  const uint e = uint(eid[pair]);
  nh_x2_rows<4, 2>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D,
                   h + size_t(pair) * p.h_stride, p.I, p.D, 1.0f, (tgp.x * nsg + sg) * 4, lane);
}

#define NH_X2_MOE_DN(NAME, NR)                                                                                     \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                   \
                       const device float* h [[buffer(2)]], device float* part [[buffer(3)]],                     \
                       const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],                 \
                       constant NhMoeParams& p [[buffer(6)]], const device uint* live [[buffer(8)]],              \
                       uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],    \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
    const uint pair = tgp.y;                                                                                       \
    if (pair / p.k >= live[0]) return;                                                                             \
    const uint e = uint(eid[pair]);                                                                                \
    nh_x2_rows<NR, 0>(q + size_t(e) * p.dn_q_stride, s + size_t(e) * p.dn_s_stride, h + size_t(pair) * p.h_stride, \
                      part + size_t(pair) * p.D, p.D, p.I, ew[pair], (tgp.x * nsg + sg) * NR, lane);               \
  }
NH_X2_MOE_DN(nh_x2_moe_dn3, 3)
NH_X2_MOE_DN(nh_x2_moe_dn4, 4)
