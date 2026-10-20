// Nemotron-H kernels: weight formats and the generic matrix x vector(s) products (decode fallbacks, the router).
//
// Weights are repacked at load from the checkpoint's blocks into separate code and scale arrays:
//   Q8   codes char  [N][K], scales half [N][K/32]                    w = s * q
//   FP4  codes uchar [N][K/2], scales uchar (UE4M3) [N][K/16]        w = e2m1(q) * ue4m3(s) * tensor scale
//        (NVFP4; in a 16-element sub-block, byte i holds element i in its low and element i + 8 in its high nibble)
//   IQ2  codes uchar [N][K/256][64], scales half [N][K/256]           IQ2_XXS: per 32 elements 4 grid indices,
//        4 x 7 sign bits and a 4-bit scale; w = d * (0.5 + s4) / 4 * grid * sign
//   F32  float [N][K];  BF16  bfloat bits [N][K]
//   FP8  E4M3 bytes [N][K], each 16-element group reordered (byte 4 m + 0..3 = elements 2 m, 2 m + 1, 2 m + 8,
//        2 m + 9; nh_x_fp8.metal), read as half x 2^-8; w = e4m3(q) * tensor scale (the host's gs includes 2^8)
// Q8 and IQ2 are the GGUF Q8_0 / IQ2_XXS blocks, FP4 is NVFP4, FP8 FP8 E4M3, as NVIDIA's checkpoints store them.
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
// FMT: 0 Q8 (SPAN 32), 1 FP4 (SPAN 32), 2 IQ2 (SPAN 32), 3 F32 (SPAN 8), 4 BF16 (SPAN 8), 5 FP8 (SPAN 16).
template <int FMT, int NR, int RMAX>
NI_INLINE void nh_mv_dot(const device uchar* q, const device uchar* s, const device float* x, uint K, uint xs, uint R,
                         uint row0, uint N, uint lane, threadgroup const ulong* tgrid, thread float (&acc)[NR][RMAX]) {
  constexpr uint SPAN = FMT == 5 ? 16 : FMT >= 3 ? 8 : 32;
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
        } else if (FMT == 5) {
          // chunk c (elements 8c .. 8c + 7 of the group): bytes 2c, 2c + 1 of each word m are elements 8c + 2m, + 1
          const uint4 v = *(const device uint4*)(q + size_t(rows[j]) * K + k0);
          for (int m = 0; m < 4; ++m) {
            const half2 a = as_type<half2>(((v[m] & 0x00800080u) << 8) | ((v[m] & 0x007F007Fu) << 7));
            const half2 b = as_type<half2>((v[m] & 0x80008000u) | ((v[m] >> 1) & 0x3F803F80u));
            w[2 * m] = float(c ? a.y : a.x);
            w[2 * m + 1] = float(c ? b.y : b.x);
          }
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
NH_MV_FMT(fp8, 5)
// one BF16 row at 2 weight rows per simdgroup (twice the threadgroups; each row's sums as nh_mv_bf16_r1's)
NH_MV(nh_mv_bf16_r1_n2, 4, 2, 1, 0)
NH_MV(nh_mv_bf16_r1_add_n2, 4, 2, 1, 1)

// Attention's q, k and v (one input row, three BF16 matrices; k and v of one shape) in one dispatch: threadgroups
// [0, tq) compute q, then tk of k, then tk of v, each row as nh_mv_bf16_r1_n2 computes it. grid: (tq + 2 tk) x 128
[[kernel]] void nh_mv_bf16_qkv_n2(const device uchar* wq [[buffer(0)]], const device uchar* wk [[buffer(1)]],
                                  const device float* x [[buffer(2)]], device float* yq [[buffer(3)]],
                                  constant NhMvParams& pq [[buffer(4)]], const device uchar* wv [[buffer(5)]],
                                  device float* yk [[buffer(6)]], device float* yv [[buffer(7)]],
                                  constant NhMvParams& pk [[buffer(8)]], uint3 tgp [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NR = 2;
  const uint tq = (pq.N + NR * nsg - 1) / (NR * nsg), tk = (pk.N + NR * nsg - 1) / (NR * nsg);
  const bool isq = tgp.x < tq, isk = !isq && tgp.x < tq + tk;
  const uint t = isq ? tgp.x : isk ? tgp.x - tq : tgp.x - tq - tk;
  const device uchar* w = isq ? wq : isk ? wk : wv;
  device float* y = isq ? yq : isk ? yk : yv;
  const uint N = isq ? pq.N : pk.N, K = isq ? pq.K : pk.K;
  const uint row0 = (t * nsg + sg) * NR;
  float acc[NR][1];
  for (int j = 0; j < NR; ++j) acc[j][0] = 0.0f;
  if (row0 < N) nh_mv_dot<4, NR, 1>(w, w, x, K, K, 1, row0, N, lane, nullptr, acc);
  nh_mv_store<NR, 1, 0>(acc, y, N, N, 1, isq ? pq.gs : pk.gs, row0, lane);
}

// BF16 for 2-8 activation rows (the MoE router in verify) on the M5, the same outputs to the bit as nh_mv_bf16_r<n>
// (and the one-row router's nh_mv_dot): per lane the same chain, acc += d_i over the steps i (k0 = 8 lane + 256 i,
// d_i the step's 8 products summed from 0), then the sum over the lanes; but a simdgroup takes NR = 2 weight rows for
// TG = 2 of the activation rows (each activation load serves 2 rows, each weight load 2 activation rows) instead of 1
// weight row for all 8: Bolt's 512 x 4096 router in an 8-row verify 12.2 against 46 us per layer (NI_PROFILE; NR x TG
// 3 x 2 12.1, 2 x 4 12.5, 1 x 4 13.2, 4 x 1 13.9, 1 x 2 14.0, 2 x 1 15.2, 8 x 1 15.5, 4 x 2 19.3, 4 x 4 21.8; a
// split of the steps over the simdgroups through threadgroup memory 27-44).
// grid: ceil(N / (4 NR)) x ceil(R / TG) threadgroups x 128 threads; K % 256 == 0.
constant constexpr int kNhRtNR = 2, kNhRtTG = 2;
[[kernel]] void nh_mv_bf16_rt(const device uchar* q [[buffer(0)]], const device float* x [[buffer(2)]],
                              device float* y [[buffer(3)]], constant NhMvParams& p [[buffer(4)]],
                              uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NR = kNhRtNR, TG = kNhRtTG;
  const uint K = p.K, n0 = (tgp.x * nsg + sg) * NR, r0 = tgp.y * TG;
  if (n0 >= p.N || r0 >= p.R) return;
  float acc[NR][TG];
  for (int j = 0; j < NR; ++j)
    for (int t = 0; t < TG; ++t) acc[j][t] = 0.0f;
  const device ushort* qr[NR];
  for (int j = 0; j < NR; ++j) qr[j] = (const device ushort*)q + size_t(min(n0 + j, p.N - 1)) * K;
  for (uint k0 = lane * 8; k0 < K; k0 += 256) {
    float xv[TG][8];
    for (int t = 0; t < TG; ++t) {
      const device float4* xp = (const device float4*)(x + min(r0 + t, p.R - 1) * p.x_stride + k0);
      const float4 a = xp[0], b = xp[1];
      xv[t][0] = a.x; xv[t][1] = a.y; xv[t][2] = a.z; xv[t][3] = a.w;
      xv[t][4] = b.x; xv[t][5] = b.y; xv[t][6] = b.z; xv[t][7] = b.w;
    }
    for (int j = 0; j < NR; ++j) {
      const uint4 v = *(const device uint4*)(qr[j] + k0);
      float w[8];
      for (int u = 0; u < 4; ++u) {
        w[2 * u] = as_type<float>(v[u] << 16);
        w[2 * u + 1] = as_type<float>(v[u] & 0xFFFF0000u);
      }
      for (int t = 0; t < TG; ++t) {
        float d = 0.0f;
        for (int u = 0; u < 8; ++u) d += w[u] * xv[t][u];
        acc[j][t] += 1.0f * d;
      }
    }
  }
  for (int j = 0; j < NR; ++j)
    for (int t = 0; t < TG; ++t) {
      const float v = simd_sum(acc[j][t]) * p.gs;
      if (lane == 0 && n0 + j < p.N && r0 + t < p.R) y[(r0 + t) * p.y_stride + n0 + j] = v;
    }
}

// ---- Routed experts (IQ2) for R activation rows, each with its own k experts: the parameters of nh_moe2_* ----
struct NhMoeParams {
  uint I;            // expert width
  uint D;            // latent size
  uint k;            // experts per row
  uint up_q_stride, up_s_stride;    // bytes between experts' up codes / scales
  uint dn_q_stride, dn_s_stride;
  uint h_stride;     // floats between (row, slot) hidden vectors (>= I)
};
