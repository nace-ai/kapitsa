// Nemotron-H (GGUF) kernels: weight formats and the matrix x vector(s) products of decode and verify.
//
// Weights are repacked at load from llama.cpp's block layouts into separate code and scale arrays:
//   Q8   codes char  [N][K], scales half [N][K/32]                    w = s * q
//   FP4  codes uchar [N][K/2], scales uchar (UE4M3) [N][K/16]        w = e2m1(q) * ue4m3(s) * tensor scale
//        (NVFP4; in a 16-element sub-block, byte i holds element i in its low and element i + 8 in its high nibble)
//   IQ2  codes uchar [N][K/256][64], scales half [N][K/256]           IQ2_XXS: per 32 elements 4 grid indices,
//        4 x 7 sign bits and a 4-bit scale; w = d * (0.5 + s4) / 4 * grid * sign
//   F32  float [N][K];  BF16  bfloat bits [N][K]
// The formats follow ggml's definitions (ggml-common.h, ggml-quants.c; MIT): the same values, other kernels.
// Activations are fp32. y[r, n] = gs * sum_k W[n, k] x[r, k] for R activation rows.


struct NhMvParams {
  uint N, K;
  uint x_stride, y_stride;  // floats between activation rows / output rows
  float gs;                 // tensor scale (NVFP4), else 1
  uint R;                   // live activation rows (<= the kernel's RMAX)
};

// One simdgroup computes NR consecutive output rows for RMAX activation rows. Lanes split K in SPAN-element
// steps (lane l covers k0 = l * SPAN + i * 32 * SPAN); each step is consumed in 8-element chunks: the 8 inputs
// of every activation row are loaded once and used for the NR weight rows.
// FMT: 0 Q8 (SPAN 32), 1 FP4 (SPAN 32), 2 IQ2 (SPAN 32), 3 F32 (SPAN 8), 4 BF16 (SPAN 8).
template <int FMT, int NR, int RMAX>
NI_INLINE void nh_mv_dot(const device uchar* q, const device uchar* s, const device float* x, uint K, uint xs, uint R,
                         uint row0, uint N, uint lane, threadgroup const ulong* tgrid, thread float (&acc)[NR][RMAX]) {
  constexpr uint SPAN = FMT >= 3 ? 8 : 32;
  uint rows[NR];
  for (int j = 0; j < NR; ++j) rows[j] = min(row0 + j, N - 1);
  for (uint k0 = lane * SPAN; k0 < K; k0 += 32 * SPAN) {
    // the step's codes and scales per weight row
    uint4 cq[NR], cq2[NR];
    uint2 c2[NR];
    float sc[NR][2];
    for (int j = 0; j < NR; ++j) {
      const size_t n = rows[j];
      if (FMT == 0) {
        cq[j] = *(const device uint4*)(q + n * K + k0);
        cq2[j] = *(const device uint4*)(q + n * K + k0 + 16);
        sc[j][0] = float(((const device half*)s)[n * (K / 32) + k0 / 32]);
      } else if (FMT == 1) {
        cq[j] = *(const device uint4*)(q + n * (K / 2) + k0 / 2);
        const uint b = *(const device ushort*)(s + n * (K / 16) + k0 / 16);
        sc[j][0] = nh_ue4m3(b & 0xFFu);
        sc[j][1] = nh_ue4m3(b >> 8);
      } else if (FMT == 2) {
        c2[j] = *(const device uint2*)(q + n * (K / 4) + k0 / 4);
        sc[j][0] = float(((const device half*)s)[n * (K / 256) + k0 / 256]) * (0.5f + float(c2[j].y >> 28)) * 0.25f;
      }
    }
    for (uint c = 0; c < SPAN / 8; ++c) {
      float xv[RMAX][8];
      for (int r = 0; r < RMAX; ++r) {
        const device float4* xp = (const device float4*)(x + uint(r < int(R) ? r : 0) * xs + k0 + c * 8);
        const float4 a = xp[0], b = xp[1];
        xv[r][0] = a.x; xv[r][1] = a.y; xv[r][2] = a.z; xv[r][3] = a.w;
        xv[r][4] = b.x; xv[r][5] = b.y; xv[r][6] = b.z; xv[r][7] = b.w;
      }
      for (int j = 0; j < NR; ++j) {
        float w[8];
        float scale = 1.0f;
        if (FMT == 0) {
          // 32 codes: bytes 0-15 in cq, 16-31 in cq2
          const uint4 v = c < 2 ? cq[j] : cq2[j];
          const uint lo = (c & 1u) ? v.z : v.x, hi = (c & 1u) ? v.w : v.y;
          for (int i = 0; i < 4; ++i) {
            w[i] = float(as_type<char>(uchar(lo >> (8 * i))));
            w[i + 4] = float(as_type<char>(uchar(hi >> (8 * i))));
          }
          scale = sc[j][0];
        } else if (FMT == 1) {
          // sub-block c / 2: bytes (c / 2) * 8 .. +8 of the step; chunk parity picks low or high nibbles
          const uint lo = (c >> 1) == 0 ? cq[j].x : cq[j].z, hi = (c >> 1) == 0 ? cq[j].y : cq[j].w;
          const uint sh = (c & 1u) * 4u;
          for (int i = 0; i < 4; ++i) {
            w[i] = nh_fp4_lut[(lo >> (8 * i + sh)) & 15u];
            w[i + 4] = nh_fp4_lut[(hi >> (8 * i + sh)) & 15u];
          }
          scale = sc[j][c >> 1];
        } else if (FMT == 2) {
          nh_iq2_decode8((c2[j].x >> (8 * c)) & 0xFFu, (c2[j].y >> (7 * c)) & 127u, 1.0f, tgrid, w);
          scale = sc[j][0];
        } else if (FMT == 3) {
          const device float4* wp = (const device float4*)((const device float*)q + size_t(rows[j]) * K + k0);
          const float4 a = wp[0], b = wp[1];
          w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
        } else {
          const uint4 v = *(const device uint4*)((const device ushort*)q + size_t(rows[j]) * K + k0);
          for (int i = 0; i < 4; ++i) {
            w[2 * i] = as_type<float>(v[i] << 16);
            w[2 * i + 1] = as_type<float>(v[i] & 0xFFFF0000u);
          }
        }
        for (int r = 0; r < RMAX; ++r) {
          float d = 0.0f;
          for (int i = 0; i < 8; ++i) d += w[i] * xv[r][i];
          acc[j][r] += scale * d;
        }
      }
    }
  }
}

// Epilogues: 0 store, 1 add into y (residual), 2 relu^2 then store
template <int NR, int RMAX, int EPI>
NI_INLINE void nh_mv_store(thread float (&acc)[NR][RMAX], device float* y, uint N, uint ys, uint R, float gs,
                           uint row0, uint lane) {
  for (int j = 0; j < NR; ++j)
    for (int r = 0; r < RMAX; ++r) {
      float v = simd_sum(acc[j][r]) * gs;
      const uint n = row0 + j;
      if (lane == 0 && n < N && uint(r) < R) {
        device float* yp = y + r * ys + n;
        if (EPI == 1) v += *yp;
        if (EPI == 2) { v = max(v, 0.0f); v = v * v; }
        *yp = v;
      }
    }
}

// grid: ceil(N / (NR * simdgroups)) threadgroups
#define NH_MV(NAME, FMT, NR, RMAX, EPI)                                                                         \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                     \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],      \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],     \
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],       \
                       uint3 ntg3 [[threads_per_threadgroup]]) {                                                \
    threadgroup ulong tgrid[FMT == 2 ? 256 : 1];                                                                \
    if (FMT == 2) nh_iq2_grid_to_tg(tgrid, tid, ntg3.x);                                                        \
    const uint row0 = (tgp.x * nsg + sg) * NR;                                                                  \
    float acc[NR][RMAX];                                                                                        \
    for (int j = 0; j < NR; ++j)                                                                                \
      for (int r = 0; r < RMAX; ++r) acc[j][r] = 0.0f;                                                          \
    if (row0 < p.N) nh_mv_dot<FMT, NR, RMAX>(q, s, x, p.K, p.x_stride, p.R, row0, p.N, lane, tgrid, acc);       \
    nh_mv_store<NR, RMAX, EPI>(acc, y, p.N, p.y_stride, p.R, p.gs, row0, lane);                                 \
  }

// name: nh_mv_<fmt>_r<rows>[_add|_relu2]; rows per simdgroup: 4 (1-2 activation rows), 2 (4), 1 (8)
#define NH_MV_FMT(F, FMT)                                  \
  NH_MV(nh_mv_##F##_r1, FMT, 4, 1, 0)                      \
  NH_MV(nh_mv_##F##_r1_add, FMT, 4, 1, 1)                  \
  NH_MV(nh_mv_##F##_r1_relu2, FMT, 4, 1, 2)                \
  NH_MV(nh_mv_##F##_r2, FMT, 4, 2, 0)                      \
  NH_MV(nh_mv_##F##_r2_add, FMT, 4, 2, 1)                  \
  NH_MV(nh_mv_##F##_r2_relu2, FMT, 4, 2, 2)                \
  NH_MV(nh_mv_##F##_r4, FMT, 2, 4, 0)                      \
  NH_MV(nh_mv_##F##_r4_add, FMT, 2, 4, 1)                  \
  NH_MV(nh_mv_##F##_r4_relu2, FMT, 2, 4, 2)                \
  NH_MV(nh_mv_##F##_r8, FMT, 1, 8, 0)                      \
  NH_MV(nh_mv_##F##_r8_add, FMT, 1, 8, 1)                  \
  NH_MV(nh_mv_##F##_r8_relu2, FMT, 1, 8, 2)
NH_MV_FMT(q8, 0)
NH_MV_FMT(fp4, 1)
NH_MV_FMT(iq2, 2)
NH_MV_FMT(f32, 3)
NH_MV_FMT(bf16, 4)

// ---- Routed experts (IQ2) for R activation rows: each row has its own k experts ----
// up:   h[r, j, :] = relu(up_{e(r,j)} . lat[r])^2           grid: (ceil(I / rows per tg), R * k)
// down: o[r, :]    = sum_j w(r,j) * down_{e(r,j)} . h[r, j]  grid: (ceil(D / rows per tg), R)
struct NhMoeParams {
  uint I;            // expert width
  uint D;            // latent size
  uint k;            // experts per row
  uint up_q_stride, up_s_stride;    // bytes between experts' up codes / scales
  uint dn_q_stride, dn_s_stride;
  uint h_stride;     // floats between (row, slot) hidden vectors (>= I)
};

[[kernel]] void nh_moe_up_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                              const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                              const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],
                              uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup ulong tgrid[256];
  nh_iq2_grid_to_tg(tgrid, tid, ntg3.x);
  const uint pair = tgp.y, r = pair / p.k;
  const uint e = uint(eid[pair]);
  const uint row0 = (tgp.x * nsg + sg) * 4;
  float acc[4][1] = {{0}, {0}, {0}, {0}};
  if (row0 < p.I)
    nh_mv_dot<2, 4, 1>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D, p.D, p.D, 1,
                       row0, p.I, lane, tgrid, acc);
  for (int j = 0; j < 4; ++j) {
    float v = max(simd_sum(acc[j][0]), 0.0f);
    if (lane == 0 && row0 + j < p.I) h[size_t(pair) * p.h_stride + row0 + j] = v * v;
  }
}

[[kernel]] void nh_moe_down_iq2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                const device float* h [[buffer(2)]], device float* o [[buffer(3)]],
                                const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],
                                constant NhMoeParams& p [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],
                                uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup ulong tgrid[256];
  nh_iq2_grid_to_tg(tgrid, tid, ntg3.x);
  const uint r = tgp.y;
  const uint row0 = (tgp.x * nsg + sg) * 4;
  float tot[4] = {0, 0, 0, 0};
  for (uint j = 0; j < p.k; ++j) {
    const uint pair = r * p.k + j;
    const uint e = uint(eid[pair]);
    float acc[4][1] = {{0}, {0}, {0}, {0}};
    if (row0 < p.D)
      nh_mv_dot<2, 4, 1>(q + size_t(e) * p.dn_q_stride, s + size_t(e) * p.dn_s_stride, h + size_t(pair) * p.h_stride,
                         p.I, p.I, 1, row0, p.D, lane, tgrid, acc);
    const float w = ew[pair];
    for (int i = 0; i < 4; ++i) tot[i] += w * acc[i][0];
  }
  for (int i = 0; i < 4; ++i) {
    const float v = simd_sum(tot[i]);
    if (lane == 0 && row0 + i < p.D) o[r * p.D + row0 + i] = v;
  }
}
