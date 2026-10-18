// Single-row decode matvecs (R = 1), tuned per format: y[n] = gs * W[n, :] . x (+ y[n] with ADD, relu^2 with EPI 2).
// One simdgroup per NR rows, 4 simdgroups per threadgroup; lanes split K in 32-element steps.
//   FP4: a 256-entry threadgroup table maps a code byte to its two values (low nibble = element j, high = j + 8),
//        so a pair of weights costs a byte extract and one table load.
//   IQ2: each grid point read from the constant table nh_iq2xxs_grid (8 magnitude bytes), its signs applied to the
//        inputs. On the M4 Pro this beat every table we tried (float or half grids and sign masks in constant,
//        threadgroup or device memory, the float lattice with XORed sign bits, 2-bit magnitude codes), split-K and
//        8-lane row groups: 107 against 34-99 GB/s on the shared expert's 5376 x 4096; the decode is ALU-bound.

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

template <int NR, int EPI = 0>
NI_INLINE void nh_mv1_iq2v(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                           uint N, uint K, float gs, uint row0, uint lane) {
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
        acc[j] += db * nh_iq2_dot32_c(aux, xv);
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
  nh_mv1_iq2v<4, 1>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane);
}

[[kernel]] void nh_mv1_iq2_c_relu2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                   const device float* x [[buffer(2)]], device float* y [[buffer(3)]],
                                   constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  nh_mv1_iq2v<4, 2>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane);
}

[[kernel]] void nh_mv1_iq2_c(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                             const device float* x [[buffer(2)]], device float* y [[buffer(3)]],
                             constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  nh_mv1_iq2v<4>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane);
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
  nh_mv1_iq2v<4, 2>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D,
                    h + size_t(pair) * p.h_stride, p.I, p.D, 1.0f, (tgp.x * nsg + sg) * 4, lane);
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
  nh_mv1_iq2v<4>(q + size_t(e) * p.dn_q_stride, s + size_t(e) * p.dn_s_stride, h + size_t(pair) * p.h_stride,
                 part + size_t(pair) * p.D, p.D, p.I, gs, (tgp.x * nsg + sg) * 4, lane);
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

// The same sums spread over threadgroups (many experts per row: one threadgroup per row read 22 x 1024 partials alone
// in ~25 us). grid: (ceil(D / 64), R) x 64 threads; the same order of additions.
[[kernel]] void nh_moe2_sum2(const device float* part [[buffer(0)]], device float* o [[buffer(1)]],
                             constant NhMoeParams& p [[buffer(2)]], const device uint* live [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]],
                             uint2 ntg [[threads_per_threadgroup]]) {
  const uint r = tg.y, d = tg.x * ntg.x + tid.x;
  if (r >= live[0] || d >= p.D) return;
  float v = 0.0f;
  for (uint j = 0; j < p.k; ++j) v += part[(size_t(r) * p.k + j) * p.D + d];
  o[r * p.D + d] = v;
}

// The IQ2_XXS grid points as 8 halves each (4 KB): the A operands of the 8-row MMA kernels and the prefill GEMMs'
// weight tiles read it from the constant address space. grid: 1 x 256 threads.
[[kernel]] void nh_iq2_fill_tables2(device uchar* tab [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const ulong g = nh_iq2xxs_grid[i];
  device half* gh = (device half*)tab;
  for (int j = 0; j < 8; ++j) gh[i * 8 + j] = half(float(uint(g >> (8 * j)) & 0xFFu));
}
