// IQ2_XXS kernels for Nemotron 3.5 Bolt (helper set, every symbol nh_x2_*): decode matvecs, routed experts for one
// and for several token rows (verify), the shared expert for up to 32 token rows on simdgroup MMAs.
//
// Decode of one grid point (8 weights) without per-weight conversions or selects:
//   - The grid point's 8 magnitude bytes come from nh_iq2xxs_grid as a uint2 (one 8-byte gather; 16-byte gathers,
//     e.g. the grid as halves, cost about 4x more on the M4 Pro, and threadgroup tables are slower still).
//   - A magnitude byte b (8, 25 or 43) read as the bits of a half is the subnormal b * 2^-24, exact (fp32 FMAs and the
//     simdgroup MMA take fp16 subnormals as they are); results are scaled by 2^24 once at the end. So
//     `word & 0x807F807F` turns two bytes of a grid word into a half2 of weights.
//   - Signs: bit 7 of every byte is free (b < 64). Each element's sign goes into bit 7 of another byte of the same
//     32-bit word (one multiply spreads the 7 stored sign bits, popcount gives the parity bit), so word & M gives
//     elements (0, 2) / (4, 6) with their signs and rotate(word, 24) & M elements (1, 3) / (5, 7).
//   - fp32 FMAs read the halves directly (a 16-bit source costs nothing extra); fp32 activations and accumulators.
// About 26 instructions and one 8-byte gather per 8 weights, against ~45 and a gather for nh_mv1_iq2_c. On the M4 Pro
// the one-row kernels are ALU-bound near 165-195 GB/s; every load instruction is expensive (a threadgroup or 16-byte
// table lookup costs more than ten ALU instructions), so rows per simdgroup (NR) are as many as fit in registers
// (the 32 activations of a lane's sub-block stay in registers for all NR rows), and 32-thread threadgroups schedule
// better than 128-thread ones (5-10 %).
//
// Lane mapping (one simdgroup, NR consecutive rows): lane l takes the 32-element sub-blocks l, l + 32, ... of every row
// (full 1024-element steps); the K % 1024 tail of the NR rows is spread over the lanes (item t: row t / T, sub-block
// t % T), so K = 5376 (shared down) or 1536 / 2304 / 2560 (routed down) keep every lane busy. K % 256 == 0 (the
// IQ2_XXS block: one half scale per 256 weights), N >= 1.

// the 8 weights of one grid point (G: its grid word pair; s7: its 7 stored sign bits) as 4 half2 words, x 2^-24:
// (e0, e2), (e1, e3), (e4, e6), (e5, e7)
NI_INLINE uint4 nh_x2_dec8(uint2 G, uint s7) {
  // lo: e0 sign -> bit 15, e1 -> 23, e2 -> 31, e3 -> 7; hi: e4 -> 15, e5 -> 23, e6 -> 31, e7 (parity) -> 7
  const uint lo_m = (s7 * 0x20408010u) & 0x80808080u;
  const uint hi_m = ((s7 * 0x02040800u) | (popcount(s7) << 7)) & 0x80808080u;
  const uint lo = G.x ^ lo_m, hi = G.y ^ hi_m;
  const uint M = 0x807F807Fu;
  return uint4(lo & M, rotate(lo, 24u) & M, hi & M, rotate(hi, 24u) & M);
}

// d + the decoded grid point w . (xa, xb) (elements 0..3, 4..7)
NI_INLINE float nh_x2_fma8(uint4 w, float4 xa, float4 xb, float d) {
  const half2 w0 = as_type<half2>(w.x), w1 = as_type<half2>(w.y), w2 = as_type<half2>(w.z), w3 = as_type<half2>(w.w);
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

// one 32-element sub-block (codes aux) . xv[0..7], x 2^-24 (one FMA chain)
NI_INLINE float nh_x2_dot32(uint2 aux, thread const float4* xv) {
  constant uint2* gt = (constant uint2*)nh_iq2xxs_grid;
  float d = nh_x2_fma8(nh_x2_dec8(gt[aux.x & 0xFFu], extract_bits(aux.y, 0, 7)), xv[0], xv[1], 0.0f);
  d = nh_x2_fma8(nh_x2_dec8(gt[extract_bits(aux.x, 8, 8)], extract_bits(aux.y, 7, 7)), xv[2], xv[3], d);
  d = nh_x2_fma8(nh_x2_dec8(gt[extract_bits(aux.x, 16, 8)], extract_bits(aux.y, 14, 7)), xv[4], xv[5], d);
  return nh_x2_fma8(nh_x2_dec8(gt[aux.x >> 24], extract_bits(aux.y, 21, 7)), xv[6], xv[7], d);
}

// the sub-block scale (0.5 + s4) / 4 as the subnormal half (2 s4 + 1) 2^-24 (= 2^-21 of the scale; with the weights'
// 2^-24, results are scaled by 2^45 at the end)
NI_INLINE float nh_x2_sub(uint auxy) { return float(as_type<half>(ushort((auxy >> 27) | 1u))); }
constant constexpr float kNhX2Scale = 35184372088832.0f;  // 2^45

// ---------------------------------------------------------------------------------------------------------------------
// One token row: y[n] = epi(gs * W[n, :] . x) for rows row0 .. row0 + NR - 1 (clamped to N; EPI 0 store, 1 add,
// 2 relu^2), one simdgroup. Same contract as nh_mv1_iq2v<NR, EPI> (drop-in for the fused decode kernels), plus gs.
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

// The same for K <= 1024 (the routed up projection: K = latent 1024): a lane's 32 activations are loaded once and
// serve NB batches of NR rows (rows row0 .. row0 + NR NB - 1), so the activation loads and the per-simdgroup setup are
// spread over NR NB rows (routed up 13-15 % faster with NR 4, NB 2 than nh_x2_rows<4>). K > 1024 falls back to
// nh_x2_rows per batch.
template <int NR, int NB, int EPI>
NI_INLINE void nh_x2_rows1k(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                            uint N, uint K, float gs, uint row0, uint lane) {
  if (K > 1024) {
    for (int b = 0; b < NB; ++b) nh_x2_rows<NR, EPI>(q, s, x, y, N, K, gs, row0 + uint(b * NR), lane);
    return;
  }
  const uint k0 = lane * 32;
  const bool on = k0 < K;  // lanes past K read sub-block 0 and add nothing
  const uint kk = on ? k0 : 0u;
  float4 xv[8];
  const device float4* xp = (const device float4*)(x + kk);
  for (int i = 0; i < 8; ++i) xv[i] = xp[i];
  for (int b = 0; b < NB; ++b) {
    float acc[NR];
    for (int j = 0; j < NR; ++j) {
      const uint n = min(row0 + uint(b * NR + j), N - 1);
      const uint2 aux = *(const device uint2*)(q + size_t(n) * (K / 4) + kk / 4);
      const float v = float(((const device half*)s)[size_t(n) * (K / 256) + kk / 256]) * nh_x2_sub(aux.y) *
                      nh_x2_dot32(aux, xv);
      acc[j] = on ? v : 0.0f;
    }
    for (int j = 0; j < NR; ++j) {
      float v = simd_sum(acc[j]) * (gs * kNhX2Scale);
      const uint n = row0 + uint(b * NR + j);
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

// up for K <= 1024 (latent 1024), 8 rows per simdgroup: grid (ceil(I / 8), R * k) x 32 threads
[[kernel]] void nh_x2_moe_up8(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                              const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                              const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],
                              const device uint* live [[buffer(7)]], uint3 tgp [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint pair = tgp.y, r = pair / p.k;
  if (r >= live[0]) return;
  const uint e = uint(eid[pair]);
  nh_x2_rows1k<4, 2, 2>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D,
                        h + size_t(pair) * p.h_stride, p.I, p.D, 1.0f, (tgp.x * nsg + sg) * 8, lane);
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

// ---------------------------------------------------------------------------------------------------------------------
// Routed experts for several token rows (verify): the (row, expert) pairs grouped by expert into tiles of <= 2 pairs
// (nh_x2_bucket); a tile's threadgroup decodes each grid point once for both of its tokens. A verify block's pairs
// average ~1.4 tokens per expert (8 rows) to ~2 (32 rows): bigger tiles measured slower (registers and token loads
// under a branch), and so did separate dispatches for single and double tiles.
//
// tokens t < c (c <= CM, uniform): y_t[n] = epi(gs_t * W[n, :] . x_t), x_t = xb + xo[t], y_t = yb + yo[t] (floats),
// rows row0 .. row0 + NR - 1. xo / yo / gs of t >= c must be valid (the kernels repeat the last token).
template <int NR, int CM, int EPI>
NI_INLINE void nh_x2_rows_c(const device uchar* q, const device uchar* s, const device float* xb,
                            thread const uint (&xo)[CM], device float* yb, thread const uint (&yo)[CM],
                            thread const float (&gs)[CM], uint c, uint N, uint K, uint row0, uint lane) {
  float acc[NR][CM];
  for (int j = 0; j < NR; ++j)
    for (int t = 0; t < CM; ++t) acc[j][t] = 0.0f;
  constant uint2* gt = (constant uint2*)nh_iq2xxs_grid;
  const uint KF = K & ~1023u;
  uint rq[NR];
  for (int j = 0; j < NR; ++j) rq[j] = min(row0 + j, N - 1);
  for (uint k0 = lane * 32; k0 < KF; k0 += 1024) {
    uint2 aux[NR];
    float db[NR];
    for (int j = 0; j < NR; ++j) {
      aux[j] = *(const device uint2*)(q + size_t(rq[j]) * (K / 4) + k0 / 4);
      db[j] = float(((const device half*)s)[size_t(rq[j]) * (K / 256) + k0 / 256]) * nh_x2_sub(aux[j].y);
    }
    float d[NR][CM];
    for (int j = 0; j < NR; ++j)
      for (int t = 0; t < CM; ++t) d[j][t] = 0.0f;
#pragma clang loop unroll(full)
    for (int l = 0; l < 4; ++l) {
      uint4 w[NR];
      for (int j = 0; j < NR; ++j)
        w[j] = nh_x2_dec8(gt[extract_bits(aux[j].x, 8 * l, 8)], extract_bits(aux[j].y, 7 * l, 7));
      for (int t = 0; t < CM; ++t) {
        if (t < int(c)) {
          const device float4* xp = (const device float4*)(xb + xo[t] + k0) + 2 * l;
          const float4 xa = xp[0], xc = xp[1];
          for (int j = 0; j < NR; ++j) d[j][t] = nh_x2_fma8(w[j], xa, xc, d[j][t]);
        }
      }
    }
    for (int j = 0; j < NR; ++j)
      for (int t = 0; t < CM; ++t) acc[j][t] = fma(db[j], d[j][t], acc[j][t]);
  }
  if (KF < K) {  // tail sub-blocks of the NR rows over the lanes
    const uint T = (K - KF) / 32;
    for (uint it = lane; it < NR * T; it += 32) {
      const uint jt = it / T, k0 = KF + (it - jt * T) * 32;
      uint rr = rq[0];
      for (int j = 1; j < NR; ++j)
        if (jt == uint(j)) rr = rq[j];
      const uint2 aux = *(const device uint2*)(q + size_t(rr) * (K / 4) + k0 / 4);
      const float db = float(((const device half*)s)[size_t(rr) * (K / 256) + k0 / 256]) * nh_x2_sub(aux.y);
      float d[CM];
      for (int t = 0; t < CM; ++t) d[t] = 0.0f;
#pragma clang loop unroll(full)
      for (int l = 0; l < 4; ++l) {
        const uint4 w = nh_x2_dec8(gt[extract_bits(aux.x, 8 * l, 8)], extract_bits(aux.y, 7 * l, 7));
        for (int t = 0; t < CM; ++t)
          if (t < int(c)) {
            const device float4* xp = (const device float4*)(xb + xo[t] + k0) + 2 * l;
            d[t] = nh_x2_fma8(w, xp[0], xp[1], d[t]);
          }
      }
      for (int j = 0; j < NR; ++j)
        for (int t = 0; t < CM; ++t) acc[j][t] += jt == uint(j) ? db * d[t] : 0.0f;
    }
  }
  for (int j = 0; j < NR; ++j)
    for (int t = 0; t < CM; ++t) {
      if (t >= int(c)) break;
      float v = simd_sum(acc[j][t]) * (gs[t] * kNhX2Scale);
      const uint n = row0 + j;
      if (lane == 0 && n < N) {
        if (EPI == 2) {
          v = max(v, 0.0f);
          v = v * v;
        }
        yb[yo[t] + n] = v;
      }
    }
}

// Pairs grouped by expert: sorted[...] = pair indices grouped by expert, tiles[t] = (expert, first sorted index,
// pairs <= BM, 0), and the indirect dispatch arguments of the up and down kernels: args[0..3] = (up_x, tiles, 1, 0),
// args[4..7] = (dn_x, tiles, 1, 0). With k, the pairs of rows >= live[0] (a verify block's dynamic depth) are left
// out. One threadgroup of 1024 threads; E <= 1024. (nh_moe_bucket's algorithm, plus the dispatch arguments.)
struct NhX2BucketP {
  uint P, E, BM, k;  // pairs (rows * k), experts, pairs per tile, experts per row
  uint up_x, dn_x;   // threadgroups along the output rows: ceil(I / 4), ceil(D / 4)
};
[[kernel]] void nh_x2_bucket(const device int* eid [[buffer(0)]], device uint* sorted [[buffer(1)]],
                             device uint4* tiles [[buffer(2)]], device uint* args [[buffer(3)]],
                             device uint* rank [[buffer(4)]], constant NhX2BucketP& p [[buffer(5)]],
                             const device uint* live [[buffer(6)]], uint tid [[thread_position_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[1024];
  threadgroup uint off[1024], toff[1024];
  const uint P = p.k && live[0] < p.P / p.k ? live[0] * p.k : p.P;
  for (uint e = tid; e < 1024; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < P; i += ntg) rank[i] = atomic_fetch_add_explicit(&cnt[uint(eid[i])], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint o = 0, t = 0;
    for (uint e = 0; e < p.E; ++e) {
      const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      off[e] = o;
      toff[e] = t;
      o += c;
      t += (c + p.BM - 1) / p.BM;
    }
    args[0] = p.up_x;
    args[1] = t;
    args[2] = 1;
    args[3] = 0;
    args[4] = p.dn_x;
    args[5] = t;
    args[6] = 1;
    args[7] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.E; e += ntg) {
    const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
    for (uint j = 0; j * p.BM < c; ++j) tiles[toff[e] + j] = uint4(e, off[e] + j * p.BM, min(p.BM, c - j * p.BM), 0);
  }
  for (uint i = tid; i < P; i += ntg) sorted[off[uint(eid[i])] + rank[i]] = i;
}

// up over the tiles: h[pair] = relu(W_up[e] . lat[pair / k])^2. dispatch_indirect(args, 0) x 32 threads
[[kernel]] void nh_x2_moeb_up(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                              const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                              const device uint4* tiles [[buffer(4)]], const device uint* sorted [[buffer(5)]],
                              constant NhMoeParams& p [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint4 tl = tiles[tgp.y];
  uint xo[2], yo[2];
  float gs[2];
  for (int t = 0; t < 2; ++t) {
    const uint pr = sorted[tl.y + min(uint(t), tl.z - 1)];
    xo[t] = (pr / p.k) * p.D;
    yo[t] = pr * p.h_stride;
    gs[t] = 1.0f;
  }
  nh_x2_rows_c<4, 2, 2>(q + size_t(tl.x) * p.up_q_stride, s + size_t(tl.x) * p.up_s_stride, lat, xo, h, yo, gs, tl.z,
                        p.I, p.D, (tgp.x * nsg + sg) * 4, lane);
}

// down over the tiles: part[pair] = ew[pair] * W_dn[e] . h[pair]. dispatch_indirect(args, 16) x 32 threads; then
// nh_moe2_sum
[[kernel]] void nh_x2_moeb_dn(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                              const device float* h [[buffer(2)]], device float* part [[buffer(3)]],
                              const device uint4* tiles [[buffer(4)]], const device uint* sorted [[buffer(5)]],
                              const device float* ew [[buffer(6)]], constant NhMoeParams& p [[buffer(7)]],
                              uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint4 tl = tiles[tgp.y];
  uint xo[2], yo[2];
  float gs[2];
  for (int t = 0; t < 2; ++t) {
    const uint pr = sorted[tl.y + min(uint(t), tl.z - 1)];
    xo[t] = pr * p.h_stride;
    yo[t] = pr * p.D;
    gs[t] = ew[pr];
  }
  nh_x2_rows_c<4, 2, 0>(q + size_t(tl.x) * p.dn_q_stride, s + size_t(tl.x) * p.dn_s_stride, h, xo, part, yo, gs, tl.z,
                        p.D, p.I, (tgp.x * nsg + sg) * 4, lane);
}

// ---------------------------------------------------------------------------------------------------------------------
// Shared expert for up to 8 NT token rows (verify): Y[t][n] = epi(gs * W[n, :] . X[t, :]) on 8x8x8 simdgroup MMAs,
// one simdgroup per 8 output rows. A = the decoded weights (8 rows x 8 k, fp16 subnormals), B = X^T (8 k x 8 tokens,
// half), C fp32; per 32-element sub-block 4 MMA steps, closed by acc += sub-block scale * chain.
// Lane (fm, fn) holds A row fm, k-slots fn, fn + 1; step s, k-slot j is element 8 (j >> 1) + P[s][j & 1] of the
// sub-block, P = {(0,2), (1,3), (4,6), (5,7)}: so the lane's A at step s is word s of its decoded grid point (group
// fn / 2) as nh_x2_dec8 returns it, and its B (k-slot fm, tokens fn, fn + 1) comes from 8 consecutive halves of each
// token row. Codes are loaded SB sub-blocks at a time (fewer, wider loads).
// Epilogues (NhV8P.epi): 0 store fp32, 1 add fp32, 2 relu^2 -> half, 3 store half. Rows of X past R - 1 read row R - 1.
template <bool ACC>
NI_INLINE float2 nh_x2_mma(uint a, half2 b, float2 c) {
  simdgroup_half8x8 A, B;
  simdgroup_float8x8 C;
  const half2 ah = as_type<half2>(a);
  A.thread_elements()[0] = ah.x;
  A.thread_elements()[1] = ah.y;
  B.thread_elements()[0] = b.x;
  B.thread_elements()[1] = b.y;
  if (ACC) {
    C.thread_elements()[0] = c.x;
    C.thread_elements()[1] = c.y;
    simdgroup_multiply_accumulate(C, A, B, C);
  } else {
    simdgroup_multiply(C, A, B);
  }
  return float2(C.thread_elements()[0], C.thread_elements()[1]);
}

template <int NT, int SB>
NI_INLINE void nh_x2_vrows(const device half* X, const device uchar* q, const device uchar* s, device void* Y,
                           constant NhV8P& p, uint blk, uint lane) {
  const uint q4 = lane >> 2;
  const uint fm = (q4 & 4u) | ((lane >> 1) & 3u), fn = ((q4 & 2u) << 1) | ((lane & 1u) << 1);
  const uint n0 = blk * 8;
  if (n0 >= p.N) return;
  const uint K = p.K;
  uint xo[NT][2];  // B: token rows 8 tt + fn (+ 1), k offset 8 (fm >> 1) of each sub-block
  for (int tt = 0; tt < NT; ++tt)
    for (int c = 0; c < 2; ++c) xo[tt][c] = min(uint(8 * tt) + fn + uint(c), p.R - 1) * p.x_ld + 8 * (fm >> 1);
  const bool odd = (fm & 1u) != 0;
  const uint n = min(n0 + fm, p.N - 1);
  const device uchar* qr = q + size_t(n) * (K / 4);
  const device half* sr = (const device half*)s + size_t(n) * (K / 256);
  constant uint2* gt = (constant uint2*)nh_iq2xxs_grid;
  const uint g = fn >> 1;
  float2 acc[NT];
  for (int tt = 0; tt < NT; ++tt) acc[tt] = float2(0.0f);
  for (uint k1 = 0; k1 < K; k1 += 32 * SB) {  // (K % 256 == 0: SB = 2 sub-blocks always fit)
    uint2 auxs[SB];
    if (SB == 2) {
      const uint4 a0 = *(const device uint4*)(qr + k1 / 4);
      auxs[0] = a0.xy;
      auxs[SB - 1] = a0.zw;
    } else {
      auxs[0] = *(const device uint2*)(qr + k1 / 4);
    }
    const float dd = float(sr[k1 / 256]);
    for (int u = 0; u < SB; ++u) {
      const uint k0 = k1 + 32 * u;
      // B for the 4 steps first (independent of the codes): half2(token fn, token fn + 1) at elements P[s][fm & 1]
      half2 b0[NT], b1[NT], b2[NT], b3[NT];
      for (int tt = 0; tt < NT; ++tt) {
        const uint4 x0 = *(const device uint4*)(X + xo[tt][0] + k0), x1 = *(const device uint4*)(X + xo[tt][1] + k0);
        const half2 a0 = as_type<half2>(odd ? x0.y : x0.x), a1 = as_type<half2>(odd ? x0.w : x0.z);
        const half2 c0 = as_type<half2>(odd ? x1.y : x1.x), c1 = as_type<half2>(odd ? x1.w : x1.z);
        b0[tt] = half2(a0.x, c0.x);
        b1[tt] = half2(a0.y, c0.y);
        b2[tt] = half2(a1.x, c1.x);
        b3[tt] = half2(a1.y, c1.y);
      }
      const uint2 aux = auxs[u];
      const float db = dd * nh_x2_sub(aux.y);
      const uint4 w = nh_x2_dec8(gt[extract_bits(aux.x, 8 * g, 8)], extract_bits(aux.y, 7 * g, 7));
      for (int tt = 0; tt < NT; ++tt) {
        float2 c = nh_x2_mma<false>(w.x, b0[tt], float2(0.0f));
        c = nh_x2_mma<true>(w.y, b1[tt], c);
        c = nh_x2_mma<true>(w.z, b2[tt], c);
        c = nh_x2_mma<true>(w.w, b3[tt], c);
        acc[tt] = fma(float2(db), c, acc[tt]);
      }
    }
  }
  if (n0 + fm >= p.N) return;
  for (int tt = 0; tt < NT; ++tt) {
    const float2 a = acc[tt] * (p.gs * kNhX2Scale);
    for (int c = 0; c < 2; ++c) {
      const uint t = 8 * tt + fn + uint(c);
      if (t >= p.R) continue;
      float v = c ? a.y : a.x;
      const size_t o = size_t(t) * p.y_ld + n0 + fm;
      if (p.epi == 0) ((device float*)Y)[o] = v;
      else if (p.epi == 1) ((device float*)Y)[o] += v;
      else if (p.epi == 2) {
        v = max(v, 0.0f);
        ((device half*)Y)[o] = half(v * v);
      } else ((device half*)Y)[o] = half(v);
    }
  }
}

// the nh_v8_<fmt>_s<NB> signature (buffer 8, the half grid, is not read): grid ceil(N / 8) x 32 threads.
// v8: 1 <= R <= 8 token rows, v16: R <= 16, v32: R <= 32. K % 256 == 0.
#define NH_X2_V(NAME, NT, SB)                                                                                      \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]],                   \
                       const device uchar* Ws [[buffer(2)]], device void* Y [[buffer(3)]],                        \
                       constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],               \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    nh_x2_vrows<NT, SB>(X, Wq, Ws, Y, p, tg.x * nsg + sg, lane);                                                   \
  }
NH_X2_V(nh_x2_v8, 1, 2)
NH_X2_V(nh_x2_v16, 2, 1)
NH_X2_V(nh_x2_v32, 4, 1)
