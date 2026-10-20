// Nemotron-H prefill / verify GEMMs: Y[M, N] = epi(gs * X[M, K] W[N, K]^T), X half, W in the repacked formats
// (nh_mv.metal), dequantized to half into a threadgroup tile, simdgroup-matrix products with fp32 accumulators.
// Tiles: 64 rows (tokens) x 64 columns (outputs) x 32 K per step, 4 simdgroups of 32 x 32.
// MoE mode: the rows of a tile are one expert's (token, expert) pairs (tile list from nh_moe_bucket):
//   MODE 1 (up):   X row = sorted[i] / k (the token), Y row i (sorted order), W = expert e
//   MODE 2 (down): X row i, Y row = sorted[i] (the pair), W = expert e
//   NVFP4 expert stacks (nh_gemm_fp4_moe_*) scale expert e's products by its tensor scale gse[e] (buffer 9) in the
//   epilogue, before relu^2: the half intermediate of the up projection stays in range.
// Epilogues: 0 store fp32, 1 add into fp32, 2 relu^2 then store half, 3 store half.

struct NhGemmP {
  uint M, N, K;
  uint x_ld, y_ld;
  float gs;
  uint epi;
  uint q_estride, s_estride;  // bytes between experts' codes / scales (MoE)
  uint k;                     // experts per token (MoE up)
};

constant constexpr int kGBM = 64, kGBN = 64, kGBK = 32, kGLD = kGBK + 8;

// 16 consecutive weights of row n at k (k % 16 == 0) as half, scaled by their block scale (not by gs). IQ2 decodes
// through the constant tables of nh_iq2_fill_tables2 (grid points as halves ^ sign masks).
template <int FMT>
NI_INLINE void nh_dq16(const device uchar* q, const device uchar* s, uint K, uint n, uint k, constant uchar* tab,
                       threadgroup half* dst) {
  if (FMT == 0) {  // Q8
    const uint4 c = *(const device uint4*)(q + size_t(n) * K + k);
    const float sc = float(((const device half*)s)[size_t(n) * (K / 32) + k / 32]);
    const uint w[4] = {c.x, c.y, c.z, c.w};
    for (int i = 0; i < 4; ++i)
      for (int b = 0; b < 4; ++b) dst[i * 4 + b] = half(float(as_type<char>(uchar(w[i] >> (8 * b)))) * sc);
  } else if (FMT == 1) {  // FP4: one 16-element sub-block, byte j = elements j (low) and j + 8 (high)
    const uint2 c = *(const device uint2*)(q + size_t(n) * (K / 2) + k / 2);
    const float sc = nh_ue4m3(s[size_t(n) * (K / 16) + k / 16]);
    for (int j = 0; j < 8; ++j) {
      const uint b = ((j < 4 ? c.x : c.y) >> (8 * (j & 3))) & 0xFFu;
      dst[j] = half(nh_fp4_lut[b & 15u] * sc);
      dst[j + 8] = half(nh_fp4_lut[b >> 4] * sc);
    }
  } else if (FMT == 6) {  // FP4 in layout L2 (nh_x_fp4.metal): word h holds element 4h + j in nibble j and 8 + 4h + j
                          // in nibble 4 + j, codes as (m2 m1 m0 s); the same values as FMT 1
    const uint2 c = *(const device uint2*)(q + size_t(n) * (K / 2) + k / 2);
    const float sc = nh_ue4m3(s[size_t(n) * (K / 16) + k / 16]);
    for (int e = 0; e < 16; ++e) {
      const uint v = (((e >> 2) & 1 ? c.y : c.x) >> (4 * ((e & 3) + ((e >> 3) << 2)))) & 15u;
      dst[e] = half(nh_fp4_lut[(v >> 1) | ((v & 1u) << 3)] * sc);
    }
  } else if (FMT == 4) {  // BF16 -> half (exact for |w| in [6.1e-5, 65504])
    const device uint4* wp = (const device uint4*)(q + size_t(n) * K * 2 + k * 2);
    for (int h = 0; h < 2; ++h) {
      const uint4 v = wp[h];
      for (int i = 0; i < 4; ++i) {
        dst[h * 8 + 2 * i] = half(as_type<float>(v[i] << 16));
        dst[h * 8 + 2 * i + 1] = half(as_type<float>(v[i] & 0xFFFF0000u));
      }
    }
  } else if (FMT == 5) {  // FP8: one reordered 16-element group, E4M3 x 2^-8 as half bits (gs includes 2^8)
    const uint4 v = *(const device uint4*)(q + size_t(n) * K + k);
    for (int m = 0; m < 4; ++m) {
      const half2 a = as_type<half2>(((v[m] & 0x00800080u) << 8) | ((v[m] & 0x007F007Fu) << 7));
      const half2 b = as_type<half2>((v[m] & 0x80008000u) | ((v[m] >> 1) & 0x3F803F80u));
      dst[2 * m] = a.x;
      dst[2 * m + 8] = a.y;
      dst[2 * m + 1] = b.x;
      dst[2 * m + 9] = b.y;
    }
  } else {  // IQ2: half of a 32-element sub-block (grid points l0, l0 + 1)
    const uint k32 = k & ~31u, l0 = (k & 16u) ? 2u : 0u;
    const uint2 aux = *(const device uint2*)(q + size_t(n) * (K / 4) + k32 / 4);
    const float db = float(((const device half*)s)[size_t(n) * (K / 256) + k / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
    constant uint4* gh = (constant uint4*)tab;
    for (uint l = l0; l < l0 + 2; ++l) {
      // the grid point's magnitudes as halves (one lookup), the signs XORed in (parity bit from popcount)
      const uint s7 = (aux.y >> (7 * l)) & 127u;
      const uint sg = s7 | ((popcount(s7) & 1u) << 7);
      const uint4 sm = ((uint4(sg) >> uint4(0, 2, 4, 6)) & 1u) << 15 | ((uint4(sg) >> uint4(1, 3, 5, 7)) & 1u) << 31;
      const uint4 w = gh[(aux.x >> (8 * l)) & 0xFFu] ^ sm;
      *(threadgroup half4*)(dst + (l - l0) * 8) = half4(float4(as_type<half4>(w.xy)) * db);
      *(threadgroup half4*)(dst + (l - l0) * 8 + 4) = half4(float4(as_type<half4>(w.zw)) * db);
    }
  }
}

#ifndef NI_UNROLL
#define NI_UNROLL _Pragma("clang loop unroll(full)")
#endif
template <int FMT, int MODE, int BM>
NI_INLINE void nh_gemm_impl(const device half* X, const device uchar* Wq, const device uchar* Ws, device void* Y,
                            constant NhGemmP& p, const device uint4* tiles, const device uint* ntiles,
                            const device uint* sorted, uint2 tg, uint tid, uint sg, uint lane, threadgroup half* xs,
                            threadgroup half* ws, constant uchar* tab, const device float* gse = nullptr) {
  constexpr int FM = BM / 16;  // row fragments per simdgroup (2 x 2 simdgroups: BM / 2 rows x 32 columns each)
  uint m0 = tg.y * BM, M = p.M, n0 = tg.x * kGBN;
  const device uchar* q = Wq;
  const device uchar* s = Ws;
  float gs = p.gs;
  if (MODE != 0) {
    if (tg.y >= ntiles[0]) return;
    const uint4 t = tiles[tg.y];
    q += size_t(t.x) * p.q_estride;
    s += size_t(t.x) * p.s_estride;
    m0 = t.y;
    M = t.y + t.z;  // rows [t.y, t.y + t.z) of the sorted pair list
    if (FMT == 1 || FMT == 6) gs *= gse[t.x];
  }
  const uint sm = sg / 2, sn = sg % 2;
  simdgroup_float8x8 acc[FM][4];
  NI_UNROLL for (int i = 0; i < FM; ++i)
    NI_UNROLL for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);
  // X: BM rows x 32 halfs per K step; 128 threads take BM * 32 / 128 halfs each (8 bytes x BM / 16)
  constexpr int XH = BM * 32 / 128;               // halfs per thread: 4 (BM 16), 8 (32), 16 (64)
  const uint xr_ = tid / (32 / XH), xk = (tid % (32 / XH)) * XH;
  const uint xm = m0 + xr_;
  uint xrow = xm;
  if (MODE == 1 && xm < M) xrow = sorted[xm] / p.k;
  // W: 64 rows x 32 per K step, 16 per thread
  const uint tr = tid / 2, tk = (tid % 2) * 16;
  const uint wn = min(n0 + tr, p.N - 1);
  for (uint k0 = 0; k0 < p.K; k0 += kGBK) {
    if (XH == 16) {
      uint4 x0 = uint4(0), x1 = uint4(0);
      if (xm < M) {
        const device uint4* xp = (const device uint4*)(X + size_t(xrow) * p.x_ld + k0 + xk);
        x0 = xp[0];
        x1 = xp[1];
      }
      *(threadgroup uint4*)(xs + xr_ * kGLD + xk) = x0;
      *(threadgroup uint4*)(xs + xr_ * kGLD + xk + 8) = x1;
    } else if (XH == 8) {
      const uint4 x0 = xm < M ? *(const device uint4*)(X + size_t(xrow) * p.x_ld + k0 + xk) : uint4(0);
      *(threadgroup uint4*)(xs + xr_ * kGLD + xk) = x0;
    } else {
      const uint2 x0 = xm < M ? *(const device uint2*)(X + size_t(xrow) * p.x_ld + k0 + xk) : uint2(0);
      *(threadgroup uint2*)(xs + xr_ * kGLD + xk) = x0;
    }
    nh_dq16<FMT>(q, s, p.K, wn, k0 + tk, tab, ws + tr * kGLD + tk);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (int kk = 0; kk < kGBK / 8; ++kk) {
      simdgroup_half8x8 a[FM], b[4];
      NI_UNROLL for (int i = 0; i < FM; ++i) simdgroup_load(a[i], xs + (sm * (BM / 2) + i * 8) * kGLD + kk * 8, kGLD);
      NI_UNROLL for (int j = 0; j < 4; ++j) simdgroup_load(b[j], ws + (sn * 32 + j * 8) * kGLD + kk * 8, kGLD, ulong2(0, 0), true);
      NI_UNROLL for (int i = 0; i < FM; ++i)
        NI_UNROLL for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // store: lane element (row rc.x, cols rc.y, rc.y + 1) of each 8x8 fragment
  const uint qq = lane / 4;
  const uint rr = (qq & 4) + (lane / 2) % 4, cc = (qq & 2) * 2 + (lane % 2) * 2;
  NI_UNROLL for (int i = 0; i < FM; ++i) {
    const uint m = m0 + sm * (BM / 2) + i * 8 + rr;
    if (m >= M) continue;
    const uint yrow = MODE == 2 ? sorted[m] : m;
    NI_UNROLL for (int j = 0; j < 4; ++j)
      NI_UNROLL for (int e = 0; e < 2; ++e) {
        const uint n = n0 + sn * 32 + j * 8 + cc + e;
        if (n >= p.N) continue;
        float v = acc[i][j].thread_elements()[e] * gs;
        const size_t o = size_t(yrow) * p.y_ld + n;
        if (p.epi == 0) ((device float*)Y)[o] = v;
        else if (p.epi == 1) ((device float*)Y)[o] += v;
        else if (p.epi == 2) { v = max(v, 0.0f); ((device half*)Y)[o] = half(v * v); }
        else ((device half*)Y)[o] = half(v);
      }
  }
}

#define NH_GEMM(NAME, FMT, MODE, BM)                                                                             \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]], \
      device void* Y [[buffer(3)]], constant NhGemmP& p [[buffer(4)]], const device uint4* tiles [[buffer(5)]],   \
      const device uint* ntiles [[buffer(6)]], const device uint* sorted [[buffer(7)]],                          \
      constant uchar* tab [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],                              \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]) {                                                                 \
    threadgroup half xs[BM * kGLD];                                                                              \
    threadgroup half ws[kGBN * kGLD];                                                                            \
    nh_gemm_impl<FMT, MODE, BM>(X, Wq, Ws, Y, p, tiles, ntiles, sorted, tg, tid, sg, lane, xs, ws, tab);         \
  }

// name: nh_gemm_<fmt>[_moe_up|_moe_dn]_m<rows per tile>
NH_GEMM(nh_gemm_q8_m64, 0, 0, 64)
NH_GEMM(nh_gemm_q8_m32, 0, 0, 32)
NH_GEMM(nh_gemm_q8_m16, 0, 0, 16)
NH_GEMM(nh_gemm_fp4_m64, 1, 0, 64)
NH_GEMM(nh_gemm_fp4_m32, 1, 0, 32)
NH_GEMM(nh_gemm_fp4_m16, 1, 0, 16)
NH_GEMM(nh_gemm_fp4l2_m64, 6, 0, 64)  // NVFP4 in layout L2 (Dory on the M1)
NH_GEMM(nh_gemm_fp4l2_m32, 6, 0, 32)
NH_GEMM(nh_gemm_fp4l2_m16, 6, 0, 16)
NH_GEMM(nh_gemm_iq2_m64, 2, 0, 64)
NH_GEMM(nh_gemm_iq2_m32, 2, 0, 32)
NH_GEMM(nh_gemm_iq2_m16, 2, 0, 16)
NH_GEMM(nh_gemm_iq2_moe_up_m64, 2, 1, 64)
NH_GEMM(nh_gemm_iq2_moe_up_m32, 2, 1, 32)
NH_GEMM(nh_gemm_iq2_moe_up_m16, 2, 1, 16)
NH_GEMM(nh_gemm_iq2_moe_dn_m64, 2, 2, 64)
NH_GEMM(nh_gemm_iq2_moe_dn_m32, 2, 2, 32)
NH_GEMM(nh_gemm_iq2_moe_dn_m16, 2, 2, 16)
NH_GEMM(nh_gemm_fp8_m64, 5, 0, 64)
NH_GEMM(nh_gemm_fp8_m32, 5, 0, 32)
NH_GEMM(nh_gemm_fp8_m16, 5, 0, 16)
NH_GEMM(nh_gemm_bf16w_m64, 4, 0, 64)
NH_GEMM(nh_gemm_bf16w_m32, 4, 0, 32)
NH_GEMM(nh_gemm_bf16w_m16, 4, 0, 16)

// NVFP4 expert stacks: the experts' tensor scales gse (buffer 9)
#define NH_GEMM_GS(NAME, FMT, MODE, BM)                                                                          \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]], \
      device void* Y [[buffer(3)]], constant NhGemmP& p [[buffer(4)]], const device uint4* tiles [[buffer(5)]],   \
      const device uint* ntiles [[buffer(6)]], const device uint* sorted [[buffer(7)]],                          \
      constant uchar* tab [[buffer(8)]], const device float* gse [[buffer(9)]],                                  \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                       \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                     \
    threadgroup half xs[BM * kGLD];                                                                              \
    threadgroup half ws[kGBN * kGLD];                                                                            \
    nh_gemm_impl<FMT, MODE, BM>(X, Wq, Ws, Y, p, tiles, ntiles, sorted, tg, tid, sg, lane, xs, ws, tab, gse);    \
  }
NH_GEMM_GS(nh_gemm_fp4_moe_up_m64, 1, 1, 64)
NH_GEMM_GS(nh_gemm_fp4_moe_up_m32, 1, 1, 32)
NH_GEMM_GS(nh_gemm_fp4_moe_up_m16, 1, 1, 16)
NH_GEMM_GS(nh_gemm_fp4_moe_dn_m64, 1, 2, 64)
NH_GEMM_GS(nh_gemm_fp4_moe_dn_m32, 1, 2, 32)
NH_GEMM_GS(nh_gemm_fp4_moe_dn_m16, 1, 2, 16)
NH_GEMM_GS(nh_gemm_fp4l2_moe_up_m64, 6, 1, 64)  // expert stacks in layout L2
NH_GEMM_GS(nh_gemm_fp4l2_moe_up_m32, 6, 1, 32)
NH_GEMM_GS(nh_gemm_fp4l2_moe_up_m16, 6, 1, 16)
NH_GEMM_GS(nh_gemm_fp4l2_moe_dn_m64, 6, 2, 64)
NH_GEMM_GS(nh_gemm_fp4l2_moe_dn_m32, 6, 2, 32)
NH_GEMM_GS(nh_gemm_fp4l2_moe_dn_m16, 6, 2, 16)

// ---- F32 GEMM (the router: Y[M, N] = X[M, K] W[N, K]^T, X fp32, W fp32 or bf16, fp32 products) ----
// grid: (ceil(N / 32), ceil(M / 32)) x 128 threads; 4 simdgroups of 16 x 16 (2 x 2 fragments)
struct NhGemmF32P {
  uint M, N, K, x_ld, y_ld;
};
NI_INLINE float4 nh_load4(const device float* w) { return *(const device float4*)w; }
NI_INLINE float4 nh_load4(const device ushort* w) {
  const uint2 v = *(const device uint2*)w;
  return float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xFFFF0000u), as_type<float>(v.y << 16),
                as_type<float>(v.y & 0xFFFF0000u));
}
template <typename WT>
NI_INLINE void nh_gemm_f32_impl(const device float* X, const device WT* W, device float* Y, constant NhGemmF32P& p,
                                uint2 tg, uint tid, uint sg, uint lane, threadgroup float* xs, threadgroup float* wsh) {
  constexpr int BM = 32, BN = 32, BK = 32, LD = BK + 4;
  const uint m0 = tg.y * BM, n0 = tg.x * BN, sm = sg / 2, sn = sg % 2;
  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0);
  const uint tr = tid / 4, tc = (tid % 4) * 8;
  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    for (int h = 0; h < 8; h += 4) {
      const float4 xv = m0 + tr < p.M ? *(const device float4*)(X + size_t(m0 + tr) * p.x_ld + k0 + tc + h) : float4(0);
      *(threadgroup float4*)(xs + tr * LD + tc + h) = xv;
      const float4 wv = n0 + tr < p.N ? nh_load4(W + size_t(n0 + tr) * p.K + k0 + tc + h) : float4(0);
      *(threadgroup float4*)(wsh + tr * LD + tc + h) = wv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BK / 8; ++kk) {
      simdgroup_float8x8 a[2], b[2];
      for (int i = 0; i < 2; ++i) simdgroup_load(a[i], xs + (sm * 16 + i * 8) * LD + kk * 8, LD);
      for (int j = 0; j < 2; ++j) simdgroup_load(b[j], wsh + (sn * 16 + j * 8) * LD + kk * 8, LD, ulong2(0, 0), true);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint qq = lane / 4;
  const uint rr = (qq & 4) + (lane / 2) % 4, cc = (qq & 2) * 2 + (lane % 2) * 2;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int e = 0; e < 2; ++e) {
        const uint m = m0 + sm * 16 + i * 8 + rr, n = n0 + sn * 16 + j * 8 + cc + e;
        if (m < p.M && n < p.N) Y[size_t(m) * p.y_ld + n] = acc[i][j].thread_elements()[e];
      }
}
#define NH_GEMM_F32(NAME, WT)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                 \
      const device float* X [[buffer(0)]], const device WT* W [[buffer(1)]], device float* Y [[buffer(2)]],     \
      constant NhGemmF32P& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]]) {                                                                \
    threadgroup float xs[32 * 36];                                                                              \
    threadgroup float wsh[32 * 36];                                                                             \
    nh_gemm_f32_impl<WT>(X, W, Y, p, tg, tid, sg, lane, xs, wsh);                                               \
  }
NH_GEMM_F32(nh_gemm_f32, float)
NH_GEMM_F32(nh_gemm_bf16, ushort)

// ---- MoE bucketing: the (token, expert) pairs grouped by expert, and the GEMM tile list ----
// sorted[offset[e] + rank] = pair; tiles[t] = (expert, first sorted row, rows <= 64, 0); ntiles[0] = count.
// One threadgroup of 1024 threads (E <= 1024).
struct NhBucketP {
  uint P, E, BM;  // BM: rows per tile
  uint k;         // pairs per token row: with live (buffer 6), the pairs of rows >= live[0] are left out
};
[[kernel]] void nh_moe_bucket(const device int* eid [[buffer(0)]], device uint* sorted [[buffer(1)]],
                              device uint4* tiles [[buffer(2)]], device uint* ntiles [[buffer(3)]],
                              device uint* rank [[buffer(4)]], constant NhBucketP& p [[buffer(5)]],
                              const device uint* live [[buffer(6)]],
                              uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
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
    ntiles[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.E; e += ntg) {
    const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
    for (uint j = 0; j * p.BM < c; ++j) tiles[toff[e] + j] = uint4(e, off[e] + j * p.BM, min(p.BM, c - j * p.BM), 0);
  }
  for (uint i = tid; i < P; i += ntg) sorted[off[uint(eid[i])] + rank[i]] = i;
}

// lo[t, d] = half(sum_j w[t, j] * out[t * k + j, d]). grid: T threadgroups x 256 threads
struct NhCombineP {
  uint D, k;
};
[[kernel]] void nh_moe_combine(const device float* out [[buffer(0)]], const device float* ew [[buffer(1)]],
                               device half* lo [[buffer(2)]], constant NhCombineP& p [[buffer(3)]],
                               uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                               uint ntg [[threads_per_threadgroup]]) {
  for (uint d = tid; d < p.D; d += ntg) {
    float v = 0.0f;
    for (uint j = 0; j < p.k; ++j) v += ew[t * p.k + j] * out[(size_t(t) * p.k + j) * p.D + d];
    lo[size_t(t) * p.D + d] = half(v);
  }
}

// rows -> half, optionally RMS-normalized with weight w: y[r, :] = half(x[r, :] (* rsqrt(mean(x^2) + eps) * w)).
// grid: R threadgroups x 1024 threads
struct NhHalfP {
  uint D, norm;
  float eps;
  uint x_ld;
};
[[kernel]] void nh_rows_half(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                             device half* y [[buffer(2)]], constant NhHalfP& p [[buffer(3)]],
                             uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* xr = x + size_t(r) * p.x_ld;
  float inv = 1.0f;
  if (p.norm) {
    float ss = 0.0f;
    for (uint k = tid; k < p.D; k += ntg) ss += xr[k] * xr[k];
    ss = tg_sum(ss, scratch, tid, ntg);
    inv = 1.0f / metal::precise::sqrt(ss / float(p.D) + p.eps);
  }
  for (uint k = tid; k < p.D; k += ntg) y[size_t(r) * p.D + k] = half(p.norm ? xr[k] * inv * w[k] : xr[k]);
}

// The grouped RMSNorm of the Mamba output into half rows (out_proj's input). grid: (R * groups) x 256 threads
[[kernel]] void nh_gnorm_half(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                              device half* y [[buffer(2)]], constant NhGnormParams& p [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const uint ng = p.D / p.G, r = tg / ng, g = tg % ng;
  const device float* xr = x + size_t(r) * p.D + g * p.G;
  float ss = 0.0f;
  for (uint k = tid; k < p.G; k += ntg) ss += xr[k] * xr[k];
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = 1.0f / metal::precise::sqrt(ss / float(p.G) + p.eps);
  for (uint k = tid; k < p.G; k += ntg) y[size_t(r) * p.D + g * p.G + k] = half(xr[k] * inv * w[g * p.G + k]);
}
