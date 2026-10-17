// Single-row decode matvecs (R = 1), tuned per format: y[n] = gs * W[n, :] . x (+ y[n] with ADD, relu^2 with EPI 2).
// One simdgroup per NR rows, 4 simdgroups per threadgroup; lanes split K in 32-element steps.
//   FP4: a 256-entry threadgroup table maps a code byte to its two values (low nibble = element j, high = j + 8),
//        so a pair of weights costs a byte extract and one table load.
//   IQ2: the 256 grid points as floats (8 KB) and the 128 sign patterns as +-1 floats (4 KB) in threadgroup memory.

NI_INLINE void nh_fp4_pair_lut(threadgroup float2* lut, uint tid, uint ntg) {
  for (uint b = tid; b < 256; b += ntg) lut[b] = float2(nh_fp4_lut[b & 15u], nh_fp4_lut[b >> 4]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <int NR, int EPI>
NI_INLINE void nh_mv1_fp4(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                          constant NhMvParams& p, uint row0, uint lane, threadgroup const float2* lut) {
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  const uint K = p.K;
  if (row0 < p.N) {
    uint rows[NR];
    for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, p.N - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
      const device float4* xp = (const device float4*)(x + k0);
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = xp[i];
      for (int j = 0; j < NR; ++j) {
        const size_t n = rows[j];
        const uint4 c = *(const device uint4*)(q + n * (K / 2) + k0 / 2);
        const uint sb = *(const device ushort*)(s + n * (K / 16) + k0 / 16);
        // sub-block 0: bytes c.x, c.y -> elements 0..7 (low) and 8..15 (high); sub-block 1: c.z, c.w
        float d0 = 0.0f, d1 = 0.0f;
        for (int b = 0; b < 4; ++b) {
          const float2 a = lut[(c.x >> (8 * b)) & 0xFFu];   // elements b, b + 8
          const float2 e = lut[(c.y >> (8 * b)) & 0xFFu];   // elements 4 + b, 12 + b
          d0 += a.x * xv[0][b] + a.y * xv[2][b] + e.x * xv[1][b] + e.y * xv[3][b];
          const float2 a2 = lut[(c.z >> (8 * b)) & 0xFFu];
          const float2 e2 = lut[(c.w >> (8 * b)) & 0xFFu];
          d1 += a2.x * xv[4][b] + a2.y * xv[6][b] + e2.x * xv[5][b] + e2.y * xv[7][b];
        }
        acc[j] += d0 * nh_ue4m3(sb & 0xFFu) + d1 * nh_ue4m3(sb >> 8);
      }
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * p.gs;
    const uint n = row0 + j;
    if (lane == 0 && n < p.N) {
      if (EPI == 1) v += y[n];
      if (EPI == 2) { v = max(v, 0.0f); v = v * v; }
      y[n] = v;
    }
  }
}

NI_INLINE void nh_iq2_tables(threadgroup float4* tg_grid, threadgroup float4* tg_sign, uint tid, uint ntg) {
  for (uint i = tid; i < 256; i += ntg) {
    const ulong g = nh_iq2xxs_grid[i];
    tg_grid[i * 2 + 0] = float4(float(uint(g) & 0xFFu), float(uint(g >> 8) & 0xFFu), float(uint(g >> 16) & 0xFFu),
                                float(uint(g >> 24) & 0xFFu));
    tg_grid[i * 2 + 1] = float4(float(uint(g >> 32) & 0xFFu), float(uint(g >> 40) & 0xFFu), float(uint(g >> 48) & 0xFFu),
                                float(uint(g >> 56) & 0xFFu));
  }
  for (uint i = tid; i < 128; i += ntg) {
    const uint sg = i | ((popcount(i) & 1u) << 7);
    float4 a, b;
    for (int j = 0; j < 4; ++j) {
      a[j] = (sg >> j) & 1u ? -1.0f : 1.0f;
      b[j] = (sg >> (j + 4)) & 1u ? -1.0f : 1.0f;
    }
    tg_sign[i * 2 + 0] = a;
    tg_sign[i * 2 + 1] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// dot of one IQ2 sub-block (32 weights: aux) with 32 inputs xv (signs applied to x), without the scales
NI_INLINE float nh_iq2_dot32(uint2 aux, thread const float4* xv, threadgroup const float4* tg_grid,
                             threadgroup const float4* tg_sign) {
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const uint gi = (aux.x >> (8 * l)) & 0xFFu, si = (aux.y >> (7 * l)) & 127u;
    const float4 g0 = tg_grid[gi * 2], g1 = tg_grid[gi * 2 + 1];
    const float4 s0 = tg_sign[si * 2], s1 = tg_sign[si * 2 + 1];
    d += dot(g0, xv[l * 2] * s0) + dot(g1, xv[l * 2 + 1] * s1);
  }
  return d;
}

template <int NR, int EPI>
NI_INLINE void nh_mv1_iq2(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                          uint N, uint K, float gs, uint row0, uint lane, threadgroup const float4* tg_grid,
                          threadgroup const float4* tg_sign) {
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  if (row0 < N) {
    uint rows[NR];
    for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, N - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
      const device float4* xp = (const device float4*)(x + k0);
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = xp[i];
      for (int j = 0; j < NR; ++j) {
        const size_t n = rows[j];
        const uint2 aux = *(const device uint2*)(q + n * (K / 4) + k0 / 4);
        const float db = float(((const device half*)s)[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
        acc[j] += db * nh_iq2_dot32(aux, xv, tg_grid, tg_sign);
      }
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * gs;
    const uint n = row0 + j;
    if (lane == 0 && n < N) {
      if (EPI == 1) v += y[n];
      if (EPI == 2) { v = max(v, 0.0f); v = v * v; }
      y[n] = v;
    }
  }
}

#define NH_MV1_FP4(NAME, NR, EPI)                                                                                 \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                  \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                       \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],        \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],       \
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],         \
                       uint3 ntg3 [[threads_per_threadgroup]]) {                                                  \
    threadgroup float2 lut[256];                                                                                  \
    nh_fp4_pair_lut(lut, tid, ntg3.x);                                                                            \
    nh_mv1_fp4<NR, EPI>(q, s, x, y, p, (tgp.x * nsg + sg) * NR, lane, lut);                                       \
  }
NH_MV1_FP4(nh_mv1_fp4, 4, 0)
NH_MV1_FP4(nh_mv1_fp4_add, 4, 1)
NH_MV1_FP4(nh_mv1_fp4_n8, 8, 0)
NH_MV1_FP4(nh_mv1_fp4_n8_add, 8, 1)

#define NH_MV1_IQ2(NAME, NR, EPI)                                                                                 \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                  \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                       \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],        \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],       \
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],         \
                       uint3 ntg3 [[threads_per_threadgroup]]) {                                                  \
    threadgroup float4 tg_grid[512];                                                                              \
    threadgroup float4 tg_sign[256];                                                                              \
    nh_iq2_tables(tg_grid, tg_sign, tid, ntg3.x);                                                                 \
    nh_mv1_iq2<NR, EPI>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * NR, lane, tg_grid, tg_sign);             \
  }
NH_MV1_IQ2(nh_mv1_iq2, 4, 0)
NH_MV1_IQ2(nh_mv1_iq2_add, 4, 1)
NH_MV1_IQ2(nh_mv1_iq2_relu2, 4, 2)

// routed experts, R = 1, float tables: up (relu^2) for the k selected experts, down with the weighted sum
[[kernel]] void nh_moe1_up_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                               const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                               const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],
                               uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup float4 tg_grid[512];
  threadgroup float4 tg_sign[256];
  nh_iq2_tables(tg_grid, tg_sign, tid, ntg3.x);
  const uint pair = tgp.y, r = pair / p.k, e = uint(eid[pair]);
  nh_mv1_iq2<4, 2>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D,
                   h + size_t(pair) * p.h_stride, p.I, p.D, 1.0f, (tgp.x * nsg + sg) * 4, lane, tg_grid, tg_sign);
}

[[kernel]] void nh_moe1_down_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                 const device float* h [[buffer(2)]], device float* o [[buffer(3)]],
                                 const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],
                                 constant NhMoeParams& p [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],
                                 uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup float4 tg_grid[512];
  threadgroup float4 tg_sign[256];
  nh_iq2_tables(tg_grid, tg_sign, tid, ntg3.x);
  const uint r = tgp.y, row0 = (tgp.x * nsg + sg) * 4;
  float tot[4] = {0, 0, 0, 0};
  if (row0 < p.D) {
    uint rows[4];
    for (int j = 0; j < 4; ++j) rows[j] = min(row0 + j, p.D - 1);
    for (uint jj = 0; jj < p.k; ++jj) {
      const uint pair = r * p.k + jj, e = uint(eid[pair]);
      const device uchar* qe = q + size_t(e) * p.dn_q_stride;
      const device half* se = (const device half*)(s + size_t(e) * p.dn_s_stride);
      const device float* x = h + size_t(pair) * p.h_stride;
      const float w = ew[pair];
      for (uint k0 = lane * 32; k0 < p.I; k0 += 32 * 32) {
        const device float4* xp = (const device float4*)(x + k0);
        float4 xv[8];
        for (int i = 0; i < 8; ++i) xv[i] = xp[i];
        for (int j = 0; j < 4; ++j) {
          const size_t n = rows[j];
          const uint2 aux = *(const device uint2*)(qe + n * (p.I / 4) + k0 / 4);
          const float db = float(se[n * (p.I / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
          tot[j] += w * db * nh_iq2_dot32(aux, xv, tg_grid, tg_sign);
        }
      }
    }
  }
  for (int j = 0; j < 4; ++j) {
    const float v = simd_sum(tot[j]);
    if (lane == 0 && row0 + j < p.D) o[r * p.D + row0 + j] = v;
  }
}

// ---- table-source variants (benchmarks): the IQ2 grid read straight from constant memory (no per-threadgroup copy),
// or float grid / sign tables from a device buffer filled once at load (grid [256][8], signs [128][8]) ----
NI_INLINE float nh_iq2_dot32_c(uint2 aux, thread const float4* xv) {
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const ulong g = nh_iq2xxs_grid[(aux.x >> (8 * l)) & 0xFFu];
    const uint s7 = (aux.y >> (7 * l)) & 127u;
    const uint sg = s7 | ((popcount(s7) & 1u) << 7);
    const uint lo = uint(g), hi = uint(g >> 32);
    const float4 g0 = float4(float(lo & 0xFFu), float((lo >> 8) & 0xFFu), float((lo >> 16) & 0xFFu), float(lo >> 24));
    const float4 g1 = float4(float(hi & 0xFFu), float((hi >> 8) & 0xFFu), float((hi >> 16) & 0xFFu), float(hi >> 24));
    const float4 s0 = float4((sg & 1u) ? -1.0f : 1.0f, (sg & 2u) ? -1.0f : 1.0f, (sg & 4u) ? -1.0f : 1.0f, (sg & 8u) ? -1.0f : 1.0f);
    const float4 s1 = float4((sg & 16u) ? -1.0f : 1.0f, (sg & 32u) ? -1.0f : 1.0f, (sg & 64u) ? -1.0f : 1.0f, (sg & 128u) ? -1.0f : 1.0f);
    d += dot(g0, xv[l * 2] * s0) + dot(g1, xv[l * 2 + 1] * s1);
  }
  return d;
}

NI_INLINE float nh_iq2_dot32_d(uint2 aux, thread const float4* xv, const device float4* tab) {
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const uint gi = (aux.x >> (8 * l)) & 0xFFu, si = (aux.y >> (7 * l)) & 127u;
    const float4 g0 = tab[gi * 2], g1 = tab[gi * 2 + 1];
    const float4 s0 = tab[512 + si * 2], s1 = tab[512 + si * 2 + 1];
    d += dot(g0, xv[l * 2] * s0) + dot(g1, xv[l * 2 + 1] * s1);
  }
  return d;
}

// MODE 0: constant grid, 1: device tables (buffer 5)
template <int NR, int MODE, int EPI = 0>
NI_INLINE void nh_mv1_iq2v(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                           uint N, uint K, float gs, uint row0, uint lane, const device float4* tab) {
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  if (row0 < N) {
    uint rows[NR];
    for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, N - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
      const device float4* xp = (const device float4*)(x + k0);
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = xp[i];
      for (int j = 0; j < NR; ++j) {
        const size_t n = rows[j];
        const uint2 aux = *(const device uint2*)(q + n * (K / 4) + k0 / 4);
        const float db = float(((const device half*)s)[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
        acc[j] += db * (MODE == 0 ? nh_iq2_dot32_c(aux, xv) : nh_iq2_dot32_d(aux, xv, tab));
      }
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * gs;
    const uint n = row0 + j;
    if (lane == 0 && n < N) {
      if (EPI == 1) v += y[n];
      if (EPI == 2) { v = max(v, 0.0f); v = v * v; }
      y[n] = v;
    }
  }
}

[[kernel]] void nh_mv1_iq2_c_add(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                 const device float* x [[buffer(2)]], device float* y [[buffer(3)]],
                                 constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  nh_mv1_iq2v<4, 0, 1>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane, nullptr);
}

[[kernel]] void nh_mv1_iq2_c_relu2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                   const device float* x [[buffer(2)]], device float* y [[buffer(3)]],
                                   constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  nh_mv1_iq2v<4, 0, 2>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane, nullptr);
}

[[kernel]] void nh_mv1_iq2_c(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                             const device float* x [[buffer(2)]], device float* y [[buffer(3)]],
                             constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  nh_mv1_iq2v<4, 0>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane, nullptr);
}

[[kernel]] void nh_mv1_iq2_d(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                             const device float* x [[buffer(2)]], device float* y [[buffer(3)]],
                             constant NhMvParams& p [[buffer(4)]], const device float4* tab [[buffer(5)]],
                             uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  nh_mv1_iq2v<4, 1>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane, tab);
}

// fills the device tables of nh_mv1_iq2_d: grid [256][8] floats, signs [128][8] floats. grid: 1 x 256
[[kernel]] void nh_iq2_fill_tables(device float* tab [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const ulong g = nh_iq2xxs_grid[i];
  for (int j = 0; j < 8; ++j) tab[i * 8 + j] = float(uint(g >> (8 * j)) & 0xFFu);
  if (i < 128) {
    const uint sg = i | ((popcount(i) & 1u) << 7);
    for (int j = 0; j < 8; ++j) tab[2048 + i * 8 + j] = (sg >> j) & 1u ? -1.0f : 1.0f;
  }
}

// ---- persistent variants: a few threadgroups per core build their table once and walk the row blocks ----
// (grid: min(row blocks, ~4 per GPU core) threadgroups of 256 threads; NR rows per simdgroup per block)
NI_INLINE float nh_iq2_dot32_t(uint2 aux, thread const float4* xv, threadgroup const ulong* tgrid) {
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const ulong g = tgrid[(aux.x >> (8 * l)) & 0xFFu];
    const uint s7 = (aux.y >> (7 * l)) & 127u;
    const uint sg = s7 | ((popcount(s7) & 1u) << 7);
    const uint lo = uint(g), hi = uint(g >> 32);
    const float4 x0 = xv[l * 2], x1 = xv[l * 2 + 1];
    // signs on x: flip the float sign bit
    const float4 a = float4(as_type<float>(as_type<uint>(x0.x) ^ ((sg & 1u) << 31)),
                            as_type<float>(as_type<uint>(x0.y) ^ ((sg & 2u) << 30)),
                            as_type<float>(as_type<uint>(x0.z) ^ ((sg & 4u) << 29)),
                            as_type<float>(as_type<uint>(x0.w) ^ ((sg & 8u) << 28)));
    const float4 b = float4(as_type<float>(as_type<uint>(x1.x) ^ ((sg & 16u) << 27)),
                            as_type<float>(as_type<uint>(x1.y) ^ ((sg & 32u) << 26)),
                            as_type<float>(as_type<uint>(x1.z) ^ ((sg & 64u) << 25)),
                            as_type<float>(as_type<uint>(x1.w) ^ ((sg & 128u) << 24)));
    d += dot(float4(float(lo & 0xFFu), float((lo >> 8) & 0xFFu), float((lo >> 16) & 0xFFu), float(lo >> 24)), a) +
         dot(float4(float(hi & 0xFFu), float((hi >> 8) & 0xFFu), float((hi >> 16) & 0xFFu), float(hi >> 24)), b);
  }
  return d;
}

template <int NR, int EPI>
NI_INLINE void nh_mv1_iq2p_impl(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                                uint N, uint K, float gs, uint tg0, uint ntgs, uint sg, uint nsg, uint lane,
                                threadgroup const ulong* tgrid) {
  const uint per_block = NR * nsg;
  for (uint blk = tg0; blk * per_block < N; blk += ntgs) {
    const uint row0 = blk * per_block + sg * NR;
    float acc[NR];
    for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
    if (row0 < N) {
      uint rows[NR];
      for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, N - 1);
      for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
        const device float4* xp = (const device float4*)(x + k0);
        float4 xv[8];
        for (int i = 0; i < 8; ++i) xv[i] = xp[i];
        for (int j = 0; j < NR; ++j) {
          const size_t n = rows[j];
          const uint2 aux = *(const device uint2*)(q + n * (K / 4) + k0 / 4);
          const float db = float(((const device half*)s)[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
          acc[j] += db * nh_iq2_dot32_t(aux, xv, tgrid);
        }
      }
    }
    for (int j = 0; j < NR; ++j) {
      float v = simd_sum(acc[j]) * gs;
      const uint n = row0 + j;
      if (lane == 0 && n < N) {
        if (EPI == 1) v += y[n];
        if (EPI == 2) { v = max(v, 0.0f); v = v * v; }
        y[n] = v;
      }
    }
  }
}

#define NH_MV1_IQ2P(NAME, NR, EPI)                                                                                \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                  \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                       \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],        \
                       uint3 ntgs3 [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],         \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],          \
                       uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {        \
    threadgroup ulong tgrid[256];                                                                                 \
    nh_iq2_grid_to_tg(tgrid, tid, ntg3.x);                                                                        \
    nh_mv1_iq2p_impl<NR, EPI>(q, s, x, y, p.N, p.K, p.gs, tgp.x, ntgs3.x, sg, nsg, lane, tgrid);                  \
  }
NH_MV1_IQ2P(nh_mv1_iq2p, 4, 0)
NH_MV1_IQ2P(nh_mv1_iq2p_add, 4, 1)
NH_MV1_IQ2P(nh_mv1_iq2p_relu2, 4, 2)
NH_MV1_IQ2P(nh_mv1_iq2p_n2, 2, 0)

template <int NR, int EPI>
NI_INLINE void nh_mv1_fp4p_impl(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                                constant NhMvParams& p, uint tg0, uint ntgs, uint sg, uint nsg, uint lane,
                                threadgroup const float2* lut) {
  const uint per_block = NR * nsg, K = p.K, N = p.N;
  for (uint blk = tg0; blk * per_block < N; blk += ntgs) {
    const uint row0 = blk * per_block + sg * NR;
    float acc[NR];
    for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
    if (row0 < N) {
      uint rows[NR];
      for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, N - 1);
      for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
        const device float4* xp = (const device float4*)(x + k0);
        float4 xv[8];
        for (int i = 0; i < 8; ++i) xv[i] = xp[i];
        for (int j = 0; j < NR; ++j) {
          const size_t n = rows[j];
          const uint4 c = *(const device uint4*)(q + n * (K / 2) + k0 / 2);
          const uint sb = *(const device ushort*)(s + n * (K / 16) + k0 / 16);
          float d0 = 0.0f, d1 = 0.0f;
          for (int b = 0; b < 4; ++b) {
            const float2 a = lut[(c.x >> (8 * b)) & 0xFFu], e = lut[(c.y >> (8 * b)) & 0xFFu];
            d0 += a.x * xv[0][b] + a.y * xv[2][b] + e.x * xv[1][b] + e.y * xv[3][b];
            const float2 a2 = lut[(c.z >> (8 * b)) & 0xFFu], e2 = lut[(c.w >> (8 * b)) & 0xFFu];
            d1 += a2.x * xv[4][b] + a2.y * xv[6][b] + e2.x * xv[5][b] + e2.y * xv[7][b];
          }
          acc[j] += d0 * nh_ue4m3(sb & 0xFFu) + d1 * nh_ue4m3(sb >> 8);
        }
      }
    }
    for (int j = 0; j < NR; ++j) {
      float v = simd_sum(acc[j]) * p.gs;
      const uint n = row0 + j;
      if (lane == 0 && n < N) {
        if (EPI == 1) v += y[n];
        y[n] = v;
      }
    }
  }
}

#define NH_MV1_FP4P(NAME, NR, EPI)                                                                                \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                  \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                       \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],        \
                       uint3 ntgs3 [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],         \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],          \
                       uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {        \
    threadgroup float2 lut[256];                                                                                  \
    nh_fp4_pair_lut(lut, tid, ntg3.x);                                                                            \
    nh_mv1_fp4p_impl<NR, EPI>(q, s, x, y, p, tgp.x, ntgs3.x, sg, nsg, lane, lut);                                 \
  }
NH_MV1_FP4P(nh_mv1_fp4p, 4, 0)
NH_MV1_FP4P(nh_mv1_fp4p_add, 4, 1)
NH_MV1_FP4P(nh_mv1_fp4p_n2, 2, 0)

// ---- routed experts, persistent: items are (pair, 4 rows) for up, (row r, 4 output rows) for down ----
[[kernel]] void nh_moe1p_up_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                                const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],
                                constant uint& npairs [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
                                uint3 ntgs3 [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup ulong tgrid[256];
  nh_iq2_grid_to_tg(tgrid, tid, ntg3.x);
  const uint blocks_per_pair = (p.I + 3) / 4, total = npairs * blocks_per_pair;
  const uint K = p.D;
  for (uint item = tgp.x * nsg + sg; item < total; item += ntgs3.x * nsg) {
    const uint pair = item / blocks_per_pair, row0 = (item % blocks_per_pair) * 4;
    const uint r = pair / p.k, e = uint(eid[pair]);
    const device uchar* qe = q + size_t(e) * p.up_q_stride;
    const device half* se = (const device half*)(s + size_t(e) * p.up_s_stride);
    const device float* x = lat + r * p.D;
    float acc[4] = {0, 0, 0, 0};
    uint rows[4];
    for (int j = 0; j < 4; ++j) rows[j] = min(row0 + j, p.I - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
      const device float4* xp = (const device float4*)(x + k0);
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = xp[i];
      for (int j = 0; j < 4; ++j) {
        const size_t n = rows[j];
        const uint2 aux = *(const device uint2*)(qe + n * (K / 4) + k0 / 4);
        const float db = float(se[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
        acc[j] += db * nh_iq2_dot32_t(aux, xv, tgrid);
      }
    }
    for (int j = 0; j < 4; ++j) {
      const float v = max(simd_sum(acc[j]), 0.0f);
      if (lane == 0 && row0 + j < p.I) h[size_t(pair) * p.h_stride + row0 + j] = v * v;
    }
  }
}

[[kernel]] void nh_moe1p_down_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                  const device float* h [[buffer(2)]], device float* o [[buffer(3)]],
                                  const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],
                                  constant NhMoeParams& p [[buffer(6)]], constant uint& nrows [[buffer(7)]],
                                  uint3 tgp [[threadgroup_position_in_grid]], uint3 ntgs3 [[threadgroups_per_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],
                                  uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup ulong tgrid[256];
  nh_iq2_grid_to_tg(tgrid, tid, ntg3.x);
  const uint blocks = (p.D + 3) / 4, total = nrows * blocks, K = p.I;
  for (uint item = tgp.x * nsg + sg; item < total; item += ntgs3.x * nsg) {
    const uint r = item / blocks, row0 = (item % blocks) * 4;
    uint rows[4];
    for (int j = 0; j < 4; ++j) rows[j] = min(row0 + j, p.D - 1);
    float tot[4] = {0, 0, 0, 0};
    for (uint jj = 0; jj < p.k; ++jj) {
      const uint pair = r * p.k + jj, e = uint(eid[pair]);
      const device uchar* qe = q + size_t(e) * p.dn_q_stride;
      const device half* se = (const device half*)(s + size_t(e) * p.dn_s_stride);
      const device float* x = h + size_t(pair) * p.h_stride;
      const float w = ew[pair];
      for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
        const device float4* xp = (const device float4*)(x + k0);
        float4 xv[8];
        for (int i = 0; i < 8; ++i) xv[i] = xp[i];
        for (int j = 0; j < 4; ++j) {
          const size_t n = rows[j];
          const uint2 aux = *(const device uint2*)(qe + n * (K / 4) + k0 / 4);
          const float db = float(se[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
          tot[j] += w * db * nh_iq2_dot32_t(aux, xv, tgrid);
        }
      }
    }
    for (int j = 0; j < 4; ++j) {
      const float v = simd_sum(tot[j]);
      if (lane == 0 && row0 + j < p.D) o[r * p.D + row0 + j] = v;
    }
  }
}

// ---- routed experts, v2 (decode and verify rows): no per-threadgroup table, the down projection split per
// (pair, 16 output rows) with weighted partial sums, summed per row by nh_moe2_sum ----
// up: grid (ceil(I / 16), R * k) x 128; down: grid (ceil(D / 16), R * k) x 128; sum: grid R x 256
// live[0]: the rows a verify block keeps (the drafter's dynamic depth; 0xffffffff: all) - pairs of later rows return
[[kernel]] void nh_moe2_up_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                               const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                               const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],
                               const device uint* live [[buffer(7)]],
                               uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint pair = tgp.y, r = pair / p.k;
  if (r >= live[0]) return;
  const uint e = uint(eid[pair]);
  nh_mv1_iq2v<4, 0, 2>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D,
                       h + size_t(pair) * p.h_stride, p.I, p.D, 1.0f, (tgp.x * nsg + sg) * 4, lane, nullptr);
}

[[kernel]] void nh_moe2_dn_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                               const device float* h [[buffer(2)]], device float* part [[buffer(3)]],
                               const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],
                               constant NhMoeParams& p [[buffer(6)]], const device uint* live [[buffer(8)]],
                               uint3 tgp [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint pair = tgp.y;
  if (pair / p.k >= live[0]) return;
  const uint e = uint(eid[pair]);
  const float gs = ew[pair];
  nh_mv1_iq2v<4, 0, 0>(q + size_t(e) * p.dn_q_stride, s + size_t(e) * p.dn_s_stride, h + size_t(pair) * p.h_stride,
                       part + size_t(pair) * p.D, p.D, p.I, gs, (tgp.x * nsg + sg) * 4, lane, nullptr);
}

[[kernel]] void nh_moe2_sum(const device float* part [[buffer(0)]], device float* o [[buffer(1)]],
                            constant NhMoeParams& p [[buffer(2)]], const device uint* live [[buffer(3)]],
                            uint r [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  if (r >= live[0]) return;
  for (uint d = tid; d < p.D; d += ntg) {
    float v = 0.0f;
    for (uint j = 0; j < p.k; ++j) v += part[(size_t(r) * p.k + j) * p.D + d];
    o[r * p.D + d] = v;
  }
}

// ---- IQ2_XXS with lookup tables in the constant address space (filled once by nh_iq2_fill_tables2):
//   [0, 4096)      grid points as 8 halves (uint4 each), 256
//   [4096, 6144)   sign masks of the 128 sign indices (parity bit included) for 8 halves (0x8000 per negative), uint4
//   [6144, 14336)  grid points as 8 floats (2 float4 each)
//   [14336, 18432) sign masks for 8 floats (2 uint4 each)
// Variant H: half grid ^ half mask, converted to float. Variant F: float grid ^ float mask. ----
[[kernel]] void nh_iq2_fill_tables2(device uchar* tab [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const ulong g = nh_iq2xxs_grid[i];
  device half* gh = (device half*)tab;
  device float* gf = (device float*)(tab + 6144);
  for (int j = 0; j < 8; ++j) {
    gh[i * 8 + j] = half(float(uint(g >> (8 * j)) & 0xFFu));
    gf[i * 8 + j] = float(uint(g >> (8 * j)) & 0xFFu);
  }
  if (i < 128) {
    const uint sg = i | ((popcount(i) & 1u) << 7);
    device ushort* mh = (device ushort*)(tab + 4096);
    device uint* mf = (device uint*)(tab + 14336);
    for (int j = 0; j < 8; ++j) {
      mh[i * 8 + j] = (sg >> j) & 1u ? 0x8000u : 0u;
      mf[i * 8 + j] = (sg >> j) & 1u ? 0x80000000u : 0u;
    }
  }
}

NI_INLINE float nh_iq2_dot32_h(uint2 aux, thread const float4* xv, constant uchar* tab) {
  constant uint4* gh = (constant uint4*)tab;
  constant uint4* mh = (constant uint4*)(tab + 4096);
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const uint4 w = gh[(aux.x >> (8 * l)) & 0xFFu] ^ mh[(aux.y >> (7 * l)) & 127u];
    d += dot(float4(as_type<half4>(w.xy)), xv[2 * l]) + dot(float4(as_type<half4>(w.zw)), xv[2 * l + 1]);
  }
  return d;
}

NI_INLINE float nh_iq2_dot32_f(uint2 aux, thread const float4* xv, constant uchar* tab) {
  constant uint4* gf = (constant uint4*)(tab + 6144);
  constant uint4* mf = (constant uint4*)(tab + 14336);
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const uint gi = (aux.x >> (8 * l)) & 0xFFu, si = (aux.y >> (7 * l)) & 127u;
    d += dot(as_type<float4>(gf[2 * gi] ^ mf[2 * si]), xv[2 * l]) +
         dot(as_type<float4>(gf[2 * gi + 1] ^ mf[2 * si + 1]), xv[2 * l + 1]);
  }
  return d;
}

// MLX-style: float grid from the table, sign bits made arithmetically
NI_INLINE float nh_iq2_dot32_a(uint2 aux, thread const float4* xv, constant uchar* tab) {
  constant uint4* gf = (constant uint4*)(tab + 6144);
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const uint gi = (aux.x >> (8 * l)) & 0xFFu, s7 = (aux.y >> (7 * l)) & 127u;
    const uint sg = s7 | ((popcount(s7) & 1u) << 7);
    const uint4 k0 = (uint4(sg) << uint4(31, 30, 29, 28)) & 0x80000000u;
    const uint4 k1 = (uint4(sg) << uint4(27, 26, 25, 24)) & 0x80000000u;
    d += dot(as_type<float4>(gf[2 * gi] ^ k0), xv[2 * l]) + dot(as_type<float4>(gf[2 * gi + 1] ^ k1), xv[2 * l + 1]);
  }
  return d;
}

// MODE: 0 H, 1 F, 2 A. NR rows per simdgroup, lanes over K in 32-element steps.
template <int NR, int MODE, int EPI>
NI_INLINE void nh_mv1_iq2t(const device uchar* q, const device uchar* s, const device float* x, device float* y, uint N,
                           uint K, float gs, uint row0, uint lane, constant uchar* tab) {
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  if (row0 < N) {
    uint rows[NR];
    for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, N - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
      const device float4* xp = (const device float4*)(x + k0);
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = xp[i];
      for (int j = 0; j < NR; ++j) {
        const size_t n = rows[j];
        const uint2 aux = *(const device uint2*)(q + n * (K / 4) + k0 / 4);
        const float db = float(((const device half*)s)[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
        acc[j] += db * (MODE == 0 ? nh_iq2_dot32_h(aux, xv, tab) : MODE == 1 ? nh_iq2_dot32_f(aux, xv, tab)
                                                                          : nh_iq2_dot32_a(aux, xv, tab));
      }
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * gs;
    const uint n = row0 + j;
    if (lane == 0 && n < N) {
      if (EPI == 1) v += y[n];
      if (EPI == 2) { v = max(v, 0.0f); v = v * v; }
      y[n] = v;
    }
  }
}

#define NH_MV1_IQ2T(NAME, NR, MODE, EPI)                                                                            \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                    \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                         \
                       constant NhMvParams& p [[buffer(4)]], constant uchar* tab [[buffer(5)]],                    \
                       uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
    nh_mv1_iq2t<NR, MODE, EPI>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * NR, lane, tab);                     \
  }
NH_MV1_IQ2T(nh_mv1_iq2_h, 4, 0, 0)
NH_MV1_IQ2T(nh_mv1_iq2_h_add, 4, 0, 1)
NH_MV1_IQ2T(nh_mv1_iq2_h_relu2, 4, 0, 2)
NH_MV1_IQ2T(nh_mv1_iq2_f, 4, 1, 0)
NH_MV1_IQ2T(nh_mv1_iq2_f_add, 4, 1, 1)
NH_MV1_IQ2T(nh_mv1_iq2_f_relu2, 4, 1, 2)
NH_MV1_IQ2T(nh_mv1_iq2_a, 4, 2, 0)
NH_MV1_IQ2T(nh_mv1_iq2_a_add, 4, 2, 1)
NH_MV1_IQ2T(nh_mv1_iq2_a_relu2, 4, 2, 2)
NH_MV1_IQ2T(nh_mv1_iq2_h2, 2, 0, 0)
NH_MV1_IQ2T(nh_mv1_iq2_h8, 8, 0, 0)

// routed experts with the table variants (as nh_moe2_*: up per (pair, rows), down per (pair, rows) into partials)
#define NH_MOE2_T(SUF, MODE)                                                                                         \
  [[kernel]] void nh_moe2_up_iq2_##SUF(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],      \
                                      const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],          \
                                      const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],    \
                                      constant uchar* tab [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]], \
                                      uint sg [[simdgroup_index_in_threadgroup]],                                    \
                                      uint nsg [[simdgroups_per_threadgroup]],                                       \
                                      uint lane [[thread_index_in_simdgroup]]) {                                     \
    const uint pair = tgp.y, r = pair / p.k, e = uint(eid[pair]);                                                    \
    nh_mv1_iq2t<4, MODE, 2>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D,            \
                            h + size_t(pair) * p.h_stride, p.I, p.D, 1.0f, (tgp.x * nsg + sg) * 4, lane, tab);       \
  }                                                                                                                  \
  [[kernel]] void nh_moe2_dn_iq2_##SUF(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],      \
                                      const device float* h [[buffer(2)]], device float* part [[buffer(3)]],         \
                                      const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],     \
                                      constant NhMoeParams& p [[buffer(6)]], constant uchar* tab [[buffer(7)]],      \
                                      uint3 tgp [[threadgroup_position_in_grid]],                                    \
                                      uint sg [[simdgroup_index_in_threadgroup]],                                    \
                                      uint nsg [[simdgroups_per_threadgroup]],                                       \
                                      uint lane [[thread_index_in_simdgroup]]) {                                     \
    const uint pair = tgp.y, e = uint(eid[pair]);                                                                    \
    nh_mv1_iq2t<4, MODE, 0>(q + size_t(e) * p.dn_q_stride, s + size_t(e) * p.dn_s_stride,                            \
                            h + size_t(pair) * p.h_stride, part + size_t(pair) * p.D, p.D, p.I, ew[pair],            \
                            (tgp.x * nsg + sg) * 4, lane, tab);                                                      \
  }
NH_MOE2_T(h, 0)
NH_MOE2_T(f, 1)
NH_MOE2_T(a, 2)

// ---- IQ2_XXS with the grid as 2-bit magnitude codes (0, 1, 2 = 8, 25, 43): 256 x 16 bits = 512 bytes, 8 cache
// lines, so a simdgroup's divergent lookups touch few lines. Filled by nh_iq2_fill_codes (device buffer bound in the
// constant address space, or copied to threadgroup memory: variant T). ----
[[kernel]] void nh_iq2_fill_codes(device ushort* tab [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const ulong g = nh_iq2xxs_grid[i];
  ushort c = 0;
  for (int j = 0; j < 8; ++j) {
    const uint m = uint(g >> (8 * j)) & 0xFFu;
    c |= ushort((m == 8u ? 0u : m == 25u ? 1u : 2u) << (2 * j));
  }
  tab[i] = c;
}

template <typename TAB>
NI_INLINE float nh_iq2_dot32_q(uint2 aux, thread const float4* xv, TAB tab) {
  float d = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const uint c = tab[(aux.x >> (8 * l)) & 0xFFu];
    const uint s7 = (aux.y >> (7 * l)) & 127u;
    const uint sg = s7 | ((popcount(s7) & 1u) << 7);
    float w[8];
    for (int i = 0; i < 8; ++i) {
      const uint m = (0x2B1908u >> (8u * ((c >> (2 * i)) & 3u))) & 0xFFu;
      w[i] = as_type<float>(as_type<uint>(float(m)) | (((sg >> i) & 1u) << 31));
    }
    d += dot(float4(w[0], w[1], w[2], w[3]), xv[2 * l]) + dot(float4(w[4], w[5], w[6], w[7]), xv[2 * l + 1]);
  }
  return d;
}

template <int NR, int EPI, typename TAB>
NI_INLINE void nh_mv1_iq2q(const device uchar* q, const device uchar* s, const device float* x, device float* y, uint N,
                           uint K, float gs, uint row0, uint lane, TAB tab) {
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  if (row0 < N) {
    uint rows[NR];
    for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, N - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
      const device float4* xp = (const device float4*)(x + k0);
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = xp[i];
      for (int j = 0; j < NR; ++j) {
        const size_t n = rows[j];
        const uint2 aux = *(const device uint2*)(q + n * (K / 4) + k0 / 4);
        const float db = float(((const device half*)s)[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
        acc[j] += db * nh_iq2_dot32_q(aux, xv, tab);
      }
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * gs;
    const uint n = row0 + j;
    if (lane == 0 && n < N) {
      if (EPI == 1) v += y[n];
      if (EPI == 2) { v = max(v, 0.0f); v = v * v; }
      y[n] = v;
    }
  }
}

#define NH_MV1_IQ2Q(NAME, NR, EPI)                                                                                  \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                    \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                         \
                       constant NhMvParams& p [[buffer(4)]], constant ushort* tab [[buffer(5)]],                   \
                       uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
    nh_mv1_iq2q<NR, EPI>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * NR, lane, tab);                           \
  }                                                                                                                 \
  [[kernel]] void NAME##_t(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                \
                           const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                     \
                           constant NhMvParams& p [[buffer(4)]], constant ushort* tab [[buffer(5)]],               \
                           uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                           uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],       \
                           uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {     \
    threadgroup ushort tt[256];                                                                                     \
    for (uint i = tid; i < 256; i += ntg3.x) tt[i] = tab[i];                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                \
    nh_mv1_iq2q<NR, EPI>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * NR, lane, (threadgroup const ushort*)tt); \
  }
NH_MV1_IQ2Q(nh_mv1_iq2_q, 4, 0)
NH_MV1_IQ2Q(nh_mv1_iq2_q_add, 4, 1)
NH_MV1_IQ2Q(nh_mv1_iq2_q_relu2, 4, 2)
NH_MV1_IQ2Q(nh_mv1_iq2_q8r, 8, 0)

// ---- routed experts of a verify block (<= 8 token rows), pairs grouped by expert (nh_moe_bucket tiles of <= 8 pairs):
// each expert row is decoded once per 2 of its tokens (the constant-grid dot, two input vectors), instead of once per
// (token, expert) pair. up: Y = h rows in sorted order (relu^2), X = latent rows of the tokens; down: X = h rows in
// sorted order, Y = out rows of the pairs (unweighted; nh_moe_combine weights and sums them).
// grid: (ceil(rows / 16), tiles * 4) x 128 threads (the 4 = token pairs of a tile of <= 8) ----
NI_INLINE float nh_iq2_dot32_c2(uint2 aux, thread const float4* xa, thread const float4* xb, thread float& db_b) {
  float da = 0.0f, d2 = 0.0f;
  for (int l = 0; l < 4; ++l) {
    const ulong g = nh_iq2xxs_grid[(aux.x >> (8 * l)) & 0xFFu];
    const uint s7 = (aux.y >> (7 * l)) & 127u;
    const uint sg = s7 | ((popcount(s7) & 1u) << 7);
    const uint lo = uint(g), hi = uint(g >> 32);
    const float4 g0 = float4(float(lo & 0xFFu), float((lo >> 8) & 0xFFu), float((lo >> 16) & 0xFFu), float(lo >> 24));
    const float4 g1 = float4(float(hi & 0xFFu), float((hi >> 8) & 0xFFu), float((hi >> 16) & 0xFFu), float(hi >> 24));
    const float4 s0 = float4((sg & 1u) ? -1.0f : 1.0f, (sg & 2u) ? -1.0f : 1.0f, (sg & 4u) ? -1.0f : 1.0f, (sg & 8u) ? -1.0f : 1.0f);
    const float4 s1 = float4((sg & 16u) ? -1.0f : 1.0f, (sg & 32u) ? -1.0f : 1.0f, (sg & 64u) ? -1.0f : 1.0f, (sg & 128u) ? -1.0f : 1.0f);
    const float4 w0 = g0 * s0, w1 = g1 * s1;
    da += dot(w0, xa[l * 2]) + dot(w1, xa[l * 2 + 1]);
    d2 += dot(w0, xb[l * 2]) + dot(w1, xb[l * 2 + 1]);
  }
  db_b = d2;
  return da;
}

template <bool DOWN>
NI_INLINE void nh_moe3(const device uchar* q, const device uchar* s, const device float* X, device float* Y,
                       const device uint4* tiles, const device uint* ntiles, const device uint* sorted,
                       constant NhMoeParams& p, uint3 tgp, uint sg, uint nsg, uint lane) {
  const uint tile = tgp.y >> 2, b = tgp.y & 3u;
  if (tile >= ntiles[0]) return;
  const uint4 t = tiles[tile];  // expert, first sorted pair, pairs
  if (2 * b >= t.z) return;
  const uint nt = min(2u, t.z - 2 * b);
  const uint N = DOWN ? p.D : p.I, K = DOWN ? p.I : p.D;
  const device uchar* qe = q + size_t(t.x) * (DOWN ? p.dn_q_stride : p.up_q_stride);
  const device uchar* se = s + size_t(t.x) * (DOWN ? p.dn_s_stride : p.up_s_stride);
  uint xrow[2], yrow[2];
  for (uint c = 0; c < 2; ++c) {
    const uint j = t.y + 2 * b + min(c, nt - 1);
    xrow[c] = DOWN ? j : sorted[j] / p.k;
    yrow[c] = DOWN ? sorted[j] : j;
  }
  const size_t xs = DOWN ? p.h_stride : p.D;      // floats per X row
  const size_t ys = DOWN ? p.D : p.h_stride;      // floats per Y row
  const uint row0 = (tgp.x * nsg + sg) * 4;
  float acc[4][2];
  for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = 0.0f;
  if (row0 < N) {
    uint rows[4];
    for (int j = 0; j < 4; ++j) rows[j] = min(row0 + j, N - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {
      float4 xa[8], xb[8];
      const device float4* pa = (const device float4*)(X + xrow[0] * xs + k0);
      const device float4* pb = (const device float4*)(X + xrow[1] * xs + k0);
      for (int i = 0; i < 8; ++i) {
        xa[i] = pa[i];
        xb[i] = pb[i];
      }
      for (int j = 0; j < 4; ++j) {
        const size_t n = rows[j];
        const uint2 aux = *(const device uint2*)(qe + n * (K / 4) + k0 / 4);
        const float db = float(((const device half*)se)[n * (K / 256) + k0 / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
        float d2;
        const float d1 = nh_iq2_dot32_c2(aux, xa, xb, d2);
        acc[j][0] += db * d1;
        acc[j][1] += db * d2;
      }
    }
  }
  for (int j = 0; j < 4; ++j)
    for (uint c = 0; c < nt; ++c) {
      float v = simd_sum(acc[j][c]);
      const uint n = row0 + j;
      if (lane == 0 && n < N) {
        if (!DOWN) { v = max(v, 0.0f); v = v * v; }
        Y[yrow[c] * ys + n] = v;
      }
    }
}

[[kernel]] void nh_moe3_up_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                               const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                               const device uint4* tiles [[buffer(4)]], const device uint* ntiles [[buffer(5)]],
                               const device uint* sorted [[buffer(6)]], constant NhMoeParams& p [[buffer(7)]],
                               uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  nh_moe3<false>(q, s, lat, h, tiles, ntiles, sorted, p, tgp, sg, nsg, lane);
}

[[kernel]] void nh_moe3_dn_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                               const device float* h [[buffer(2)]], device float* out [[buffer(3)]],
                               const device uint4* tiles [[buffer(4)]], const device uint* ntiles [[buffer(5)]],
                               const device uint* sorted [[buffer(6)]], constant NhMoeParams& p [[buffer(7)]],
                               uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  nh_moe3<true>(q, s, h, out, tiles, ntiles, sorted, p, tgp, sg, nsg, lane);
}

// ---- FP4 single-row matvec with the nibble values from a 16-entry table held across the simdgroup's lanes
// (simd_shuffle per nibble: no threadgroup table, no barrier). Same lane / row layout as nh_mv1_fp4. ----
template <int NR, int EPI>
NI_INLINE void nh_mv1_fp4x(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                           constant NhMvParams& p, uint row0, uint lane) {
  const float tab = nh_fp4_lut[lane & 15u];  // E2M1 x 2; nh_ue4m3 halves the scale
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  const uint K = p.K;
  const bool live = row0 < p.N;
  uint rows[NR];
  for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, p.N - 1);
  for (uint k0 = lane * 32; k0 < K; k0 += 32 * 32) {  // (K / 32 is a multiple of 32 for these shapes)
    const device float4* xp = (const device float4*)(x + k0);
    float4 xv[8];
    for (int i = 0; i < 8; ++i) xv[i] = xp[i];
    for (int j = 0; j < NR; ++j) {
      const size_t n = rows[j];
      const uint4 c = *(const device uint4*)(q + n * (K / 2) + k0 / 2);
      const uint sb = *(const device ushort*)(s + n * (K / 16) + k0 / 16);
      float d0 = 0.0f, d1 = 0.0f;
      const uint w[4] = {c.x, c.y, c.z, c.w};
      for (int wi = 0; wi < 4; ++wi)
        for (int b = 0; b < 4; ++b) {
          const uint byte = (w[wi] >> (8 * b)) & 0xFFu;
          const float lo = simd_shuffle(tab, ushort(byte & 15u)), hi = simd_shuffle(tab, ushort(byte >> 4));
          // word wi of the 16-byte step: sub-block wi >> 1, bytes (wi & 1) * 4 + b = elements i (low), i + 8 (high)
          const uint i = (wi & 1) * 4 + b;
          const float xl = xv[(wi >> 1) * 4 + (i >> 2)][i & 3], xh = xv[(wi >> 1) * 4 + 2 + (i >> 2)][i & 3];
          if (wi < 2) d0 += lo * xl + hi * xh;
          else d1 += lo * xl + hi * xh;
        }
      acc[j] += d0 * nh_ue4m3(sb & 0xFFu) + d1 * nh_ue4m3(sb >> 8);
    }
  }
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * p.gs;
    const uint n = row0 + j;
    if (live && lane == 0 && n < p.N) {
      if (EPI == 1) v += y[n];
      y[n] = v;
    }
  }
}
#define NH_MV1_FP4X(NAME, NR, EPI)                                                                                \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                  \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                       \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],        \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],       \
                       uint lane [[thread_index_in_simdgroup]]) {                                                 \
    nh_mv1_fp4x<NR, EPI>(q, s, x, y, p, (tgp.x * nsg + sg) * NR, lane);                                           \
  }
NH_MV1_FP4X(nh_mv1_fp4x, 4, 0)
NH_MV1_FP4X(nh_mv1_fp4x_add, 4, 1)
NH_MV1_FP4X(nh_mv1_fp4x_n8, 8, 0)
NH_MV1_FP4X(nh_mv1_fp4x_n8_add, 8, 1)
