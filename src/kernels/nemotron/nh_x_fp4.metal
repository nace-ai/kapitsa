// NVFP4 products for Nemotron-H's Mamba-2 projections (in_proj [18560][4096], out_proj [4096][8192]) and any other
// NVFP4 matrix with K % 128 == 0 (the LiLiCorr drafter's): one-row decode, 8-row verify, up to 32 rows (short prefill,
// drafter blocks). Measured with tools/nh_x4_bench (M4 Pro; the variants that lost are listed at the end).
//
// Weights: codes [N][K/2] bytes, a 16-element block in 8 bytes; UE4M3 block scales [N][K/16]; the tensor scale gs
// applied once at the end. Two layouts of a block's 8 code bytes:
//   L0 (the default repack): byte i = element i (low nibble) | element i + 8 (high nibble), codes as ModelOpt
//      writes them (s e1 e0 m). Kernels *_l0: a threadgroup table maps a code byte to its two values as half2.
//   L2: little-endian word h (0, 1): nibble j (0..3) = element 4h + j, nibble 4 + j = element 8 + 4h + j, each code
//      stored as (m2 m1 m0 s) = ((c & 7) << 1) | (c >> 3). The pair (nibble j, nibble j + 4) of a word then becomes
//      the half2 bits of E2M1 x 2^-14 with three integer ops (nh_x4_l2m): the nibbles masked in place and one
//      multiply by 2^a + 2^b put the magnitude at half bits 9-11 and the sign at bit 15 (the shifted copies never
//      overlap, so there are no carries); E2M1 0.5 lands on a half subnormal, which this GPU keeps (fast math
//      included; nh_x4_bench probe). 1-7% faster than L0 below; on the M1 Pro 12-16% at Dory's shapes, which repacks
//      to L2 on the M1's GPU family (Dory::load). Every kernel's output equals its L0 twin's to the bit.
// Products: decode in fp32 (activations fp32, fp32 sums); multi-row on 8x8x8 simdgroup MMAs with the decoded weights
// x their block scale as the A operand (exact in half: <= 6 significant bits), X (half rows) as B, fp32 accumulators.
//
// Kernels (signatures of nh_mv1 / nh_v8, so enc_mv / enc_gemm swap the name and the grid; L2 / L0):
//   nh_x4_mv[_l0][_add]   R = 1, NhMvParams [[4]] (x fp32 [K], y fp32 [N]; _add: y += ...). grid ceil(N / 8) x 64
//                         (4 rows per simdgroup; any simdgroup count works: rows per threadgroup = 4 x simdgroups)
//   nh_x4_v8[_l0]         R <= 8, NhV8P [[4]] (X half [R][x_ld], Y per p.epi as nh_v8). grid ceil(N / 16) x 256
//                         (16 rows per threadgroup, 8 simdgroups splitting K)
//   nh_x4_mm16/24/32[_l0] R <= 16 / 24 / 32, NhV8P [[4]], plus part [[5]] and cnt [[6]] (split K, see below).
//                         grid (SK, ceil(N / 64)) x 128 with SK parts of K: 4 for K <= 8192 (in_proj, out_proj),
//                         8 above (1 without the buffers: then they are not touched, but must still be bound);
//                         part >= SK x ceil(N / 64) x 2048 x TB bytes (TB = 2 / 3 / 4), cnt >= ceil(N / 64) uints
//   nh_x4_v8[_l0]_t, nh_x4_mm16/24/32[_l0]_t   the same for K % 128 != 0 (K % 32 == 0: the Nano 9B v2's MLP down,
//                         K 15680): the last K % 128 inputs as one more step, its 32-element groups past K zero (read
//                         nothing); the full steps as the kernels above
// Requirements: K % 128 == 0 for v8 / mm (K % 32 for mv and the _t kernels); X rows 16-byte aligned (x_ld % 8 == 0);
// UE4M3 scale bytes < 0x7F (0x7F is NaN in E4M3FN, which ModelOpt never writes; nh_x4_mv maps it to 0 like nh_ue4m3,
// the MMA kernels do not).

// ---- shared pieces ----
// E2M1 code -> its value x 2 (0, 1, 2, 3, 4, 6, 8, 12 and negatives) as a constant expression; a code byte (L0) ->
// (low nibble, high nibble) values as half2
#define NH_X4_E(n) ((((n) & 8) ? -1.0f : 1.0f) * float(((n) & 7) < 2 ? ((n) & 7) : (2 + ((n) & 1)) << ((((n) & 7) >> 1) - 1)))
#define NH_X4_H2(b) half2(half(NH_X4_E((b) & 15)), half(NH_X4_E(((b) >> 4) & 15)))
#define NH_X4_R4(M, b) M(b), M(b + 1), M(b + 2), M(b + 3)
#define NH_X4_R16(M, b) NH_X4_R4(M, b), NH_X4_R4(M, b + 4), NH_X4_R4(M, b + 8), NH_X4_R4(M, b + 12)
#define NH_X4_R256(M)                                                                                              \
  NH_X4_R16(M, 0), NH_X4_R16(M, 16), NH_X4_R16(M, 32), NH_X4_R16(M, 48), NH_X4_R16(M, 64), NH_X4_R16(M, 80),       \
      NH_X4_R16(M, 96), NH_X4_R16(M, 112), NH_X4_R16(M, 128), NH_X4_R16(M, 144), NH_X4_R16(M, 160),                \
      NH_X4_R16(M, 176), NH_X4_R16(M, 192), NH_X4_R16(M, 208), NH_X4_R16(M, 224), NH_X4_R16(M, 240)
constant half2 nh_x4_luth[256] = {NH_X4_R256(NH_X4_H2)};

NI_INLINE void nh_x4_lut_to_tg(threadgroup half2* tlh, uint tid, uint nthr) {
  for (uint b = tid; b < 256; b += nthr) tlh[b] = nh_x4_luth[b];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// L2: the pair (element 4h + j, element 8 + 4h + j) of block word h, as half bits of E2M1 x 2^-14
NI_INLINE half2 nh_x4_l2m(uint w, uint j) {
  if (j == 0) return as_type<half2>(((w & 0x000F000Fu) * 0x8100u) & 0x8E008E00u);
  if (j == 1) return as_type<half2>(((w & 0x00F000F0u) * 0x0810u) & 0x8E008E00u);
  if (j == 2) return as_type<half2>(((w & 0x0F000F00u) * 0x0081u) & 0x8E008E00u);
  return as_type<half2>((((w >> 4) & 0x0F000F00u) * 0x0081u) & 0x8E008E00u);
}

// L (layout 0 / 2): the A-operand pair of word w, position j (L0: byte j, values E2M1 x 2; L2: nibbles j, j + 4)
template <int L>
NI_INLINE half2 nh_x4_pair(uint w, uint j, threadgroup const half2* tlh) {
  if (L == 0) return tlh[(w >> (8 * j)) & 0xFFu];
  return nh_x4_l2m(w, j);
}

// UE4M3 byte -> its value (bias 7, subnormals m 2^-9; 0x7F -> 0 like nh_ue4m3)
NI_INLINE float nh_x4_e4m3(uint b) {
  const float v = float(as_type<half>(ushort(b << 7))) * 256.0f;
  return b == 0x7Fu ? 0.0f : v;
}
// A step's two block scales (bytes of sw) as half2 such that pair x scale = E2M1 x E4M3 x 2^-7 (L2) or x 1 (L0):
// the UE4M3 bits at half bits 7-14 are E4M3 x 2^-8 (subnormals included)
template <int L>
NI_INLINE half2 nh_x4_sc2(uint sw) {
  return as_type<half2>(((sw & 0xFFu) << 7) | ((sw & 0xFF00u) << 15)) * half2(L == 0 ? 128.0h : 32768.0h);
}
constant constexpr float nh_x4_post_l[3] = {1.0f, 1.0f, 128.0f};  // MMA products x this x gs

NI_INLINE void nh_x4_mma(thread float2& c, half2 a, half2 b) {
  simdgroup_half8x8 A, B;
  simdgroup_float8x8 C;
  A.thread_elements()[0] = a.x;
  A.thread_elements()[1] = a.y;
  B.thread_elements()[0] = b.x;
  B.thread_elements()[1] = b.y;
  C.thread_elements()[0] = c.x;
  C.thread_elements()[1] = c.y;
  simdgroup_multiply_accumulate(C, A, B, C);
  c = float2(C.thread_elements()[0], C.thread_elements()[1]);
}

// Y[t][n] per NhV8P.epi: 0 store fp32, 1 add fp32, 2 relu^2 -> half, 3 store half
NI_INLINE void nh_x4_store(device void* Y, constant NhV8P& p, uint n, uint t, float v) {
  const size_t o = size_t(t) * p.y_ld + n;
  if (p.epi == 0) ((device float*)Y)[o] = v;
  else if (p.epi == 1) ((device float*)Y)[o] += v;
  else if (p.epi == 2) { v = max(v, 0.0f); ((device half*)Y)[o] = half(v * v); }
  else ((device half*)Y)[o] = half(v);
}

// 8x8 fragment lane layout (as nh_v8): lane holds row fm, columns fn, fn + 1; c = fn / 2 is its column pair
NI_INLINE uint3 nh_x4_lane(uint lane) {
  const uint q4 = lane >> 2;
  const uint fm = (q4 & 4u) | ((lane >> 1) & 3u), fn = ((q4 & 2u) << 1) | ((lane & 1u) << 1);
  return uint3(fm, fn, fn >> 1);
}

// ---- decode (R = 1): a simdgroup computes 4 rows; lane l covers k = 32 l + 1024 i (two blocks: one uint4 of codes,
// two scale bytes) and pairs elements (i, i + 8) of a block with x[i], x[i + 8] (L0 byte i, L2 word i / 4 position
// i % 4). ----
template <int L>
NI_INLINE float2 nh_x4_pf(uint w, uint j, threadgroup const half2* tlh) {
  return float2(nh_x4_pair<L>(w, j, tlh));
}

template <int L, int EPI>
NI_INLINE void nh_x4_mv_impl(const device uchar* q, const device uchar* s, const device float* x, device float* y,
                             uint N, uint K, float gs, uint row0, uint lane, threadgroup const half2* tlh) {
  constexpr int NR = 4;
  float acc[NR];
  for (int j = 0; j < NR; ++j) acc[j] = 0.0f;
  if (row0 < N) {
    uint rows[NR];
    for (int j = 0; j < NR; ++j) rows[j] = min(row0 + uint(j), N - 1);
    for (uint k0 = lane * 32; k0 < K; k0 += 1024) {
      float4 xv[8];
      _Pragma("clang loop unroll(full)")
      // L2: x scaled by 2^15 so every product is L0's (E2M1 x 2^-14 x 2^15 x = E2M1 x 2 x), exactly: unscaled,
      // inputs below ~1e-34 (a gated attention head that is shut) give products under fp32's normal range,
      // flushed to zero where L0's stay normal, and a flushed term can break an FMA rounding tie differently
      for (int i = 0; i < 8; ++i) xv[i] = ((const device float4*)(x + k0))[i] * (L == 0 ? 1.0f : 32768.0f);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NR; ++j) {
        const size_t n = rows[j];
        const uint4 c = *(const device uint4*)(q + n * (K / 2) + k0 / 2);
        const uint sb = *(const device ushort*)(s + n * (K / 16) + k0 / 16);
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < 2; ++b) {
          const uint w0 = c[2 * b], w1 = c[2 * b + 1];
          float d = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (int jj = 0; jj < 4; ++jj) {
            const float2 lo = nh_x4_pf<L>(w0, uint(jj), tlh);  // elements jj, jj + 8
            const float2 hi = nh_x4_pf<L>(w1, uint(jj), tlh);  // elements 4 + jj, 12 + jj
            d = fma(lo.x, xv[4 * b][jj], d);
            d = fma(lo.y, xv[4 * b + 2][jj], d);
            d = fma(hi.x, xv[4 * b + 1][jj], d);
            d = fma(hi.y, xv[4 * b + 3][jj], d);
          }
          acc[j] = fma(d, nh_x4_e4m3((sb >> (8 * b)) & 0xFFu), acc[j]);
        }
      }
    }
  }
  const float post = gs * 0.5f;  // products E2M1 x 2 x x (L0's pairs; L2's with x x 2^15)
  for (int j = 0; j < NR; ++j) {
    float v = simd_sum(acc[j]) * post;
    const uint n = row0 + uint(j);
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

#define NH_X4_MV(NAME, L, EPI)                                                                                     \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                   \
                       const device float* x [[buffer(2)]], device float* y [[buffer(3)]],                        \
                       constant NhMvParams& p [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],         \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {        \
    threadgroup half2 tlh[L == 0 ? 256 : 1];                                                                       \
    if (L == 0) nh_x4_lut_to_tg(tlh, tid, nsg * 32);                                                               \
    nh_x4_mv_impl<L, EPI>(q, s, x, y, p.N, p.K, p.gs, (tgp.x * nsg + sg) * 4, lane, tlh);                          \
  }
NH_X4_MV(nh_x4_mv, 2, 0)
NH_X4_MV(nh_x4_mv_add, 2, 1)
NH_X4_MV(nh_x4_mv_relu2, 2, 2)  // (a dense model's MLP up in layout L2)
NH_X4_MV(nh_x4_mv_l0, 0, 0)
NH_X4_MV(nh_x4_mv_l0_add, 0, 1)
NH_X4_MV(nh_x4_mv_l0_relu2, 0, 2)

// ---- NVFP4 routed experts per (row, expert) pair (decode, verify): the signatures of nh_moe2_up_iq2 /
// nh_moe2_dn_iq2 plus the experts' tensor scales gse [[9]] (scaled before relu^2, so the intermediate never holds the
// unscaled products). up: h[pair] = relu(gse[e] W_up[e] . lat[row])^2, grid (ceil(I / (4 x simdgroups)), pairs);
// down: part[pair] = ew[pair] gse[e] W_dn[e] . h[pair], grid (ceil(D / (4 x simdgroups)), pairs); then nh_moe2_sum2.
// The engine runs 8 simdgroups (32 rows per threadgroup: the table fill is spread over more rows; one token's 22
// pairs 538 against 312 GB/s with 2 simdgroups, tools/nh_x8_bench); a verify on the M5 4 (tools/nh_moe_bench: up +
// down + sum 223 against 230 us per layer). Grouping a verify block's pairs by expert into tiles of <= 2 tokens (each
// weight row decoded once for both, as nh_x2_moeb_*) measured slower: 0.97 against 0.83 ms for 8 rows; on the M5 too,
// with each pair's sums kept bitwise (206 against 199 us per layer, recorded routings: a verify layer's 23.5 pairs
// read 17.4 experts, and an expert's second read mostly hits the cache).
// Pairs of rows >= live[0] return. ----
// L: the experts' layout (L2: no table; the products are L0's to the bit, as nh_x4_mv_impl's)
#define NH_X4_MOE_UP(NAME, L)                                                                                      \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                   \
                       const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],                      \
                       const device int* eid [[buffer(4)]], constant NhMoeParams& p [[buffer(5)]],                \
                       const device uint* live [[buffer(7)]], const device float* gse [[buffer(9)]],              \
                       uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       uint tid [[thread_index_in_threadgroup]]) {                                                 \
    threadgroup half2 tlh[L == 0 ? 256 : 1];                                                                       \
    if (L == 0) nh_x4_lut_to_tg(tlh, tid, nsg * 32);                                                               \
    const uint pair = tgp.y, r = pair / p.k;                                                                       \
    if (r >= live[0]) return;                                                                                      \
    const uint e = uint(eid[pair]);                                                                                \
    nh_x4_mv_impl<L, 2>(q + size_t(e) * p.up_q_stride, s + size_t(e) * p.up_s_stride, lat + r * p.D,               \
                        h + size_t(pair) * p.h_stride, p.I, p.D, gse[e], (tgp.x * nsg + sg) * 4, lane, tlh);       \
  }
#define NH_X4_MOE_DN(NAME, L)                                                                                      \
  [[kernel]] void NAME(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],                   \
                       const device float* h [[buffer(2)]], device float* part [[buffer(3)]],                     \
                       const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],                  \
                       constant NhMoeParams& p [[buffer(6)]], const device uint* live [[buffer(8)]],              \
                       const device float* gse [[buffer(9)]], uint3 tgp [[threadgroup_position_in_grid]],          \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {        \
    threadgroup half2 tlh[L == 0 ? 256 : 1];                                                                       \
    if (L == 0) nh_x4_lut_to_tg(tlh, tid, nsg * 32);                                                               \
    const uint pair = tgp.y;                                                                                       \
    if (pair / p.k >= live[0]) return;                                                                             \
    const uint e = uint(eid[pair]);                                                                                \
    nh_x4_mv_impl<L, 0>(q + size_t(e) * p.dn_q_stride, s + size_t(e) * p.dn_s_stride,                              \
                        h + size_t(pair) * p.h_stride, part + size_t(pair) * p.D, p.D, p.I, ew[pair] * gse[e],     \
                        (tgp.x * nsg + sg) * 4, lane, tlh);                                                        \
  }
NH_X4_MOE_UP(nh_x4_moe_up, 0)
NH_X4_MOE_UP(nh_x4_moe_up_l2, 2)
NH_X4_MOE_DN(nh_x4_moe_dn, 0)
NH_X4_MOE_DN(nh_x4_moe_dn_l2, 2)

// ---- 8 rows (verify): Y^T[n, t] = W[n, :] . X^T[:, t] on 8x8x8 MMAs, W decoded into registers as A, the token rows
// as B read straight from X ----
// A 128-element step of a row is 8 blocks; the lane of column pair c owns blocks 2c, 2c + 1 (one uint4 of codes, two
// scale bytes) and fragment fr (0..15) takes pair fr % 8 of block 2c + fr / 8: elements (block, fr % 8) and + 8, so
// k(fr, j) = 32 (j >> 1) + 16 (fr >> 3) + 8 (j & 1) + (fr & 7) for k-slot j. B's lane (k-slot fm, tokens fn, fn + 1)
// then needs X[t][k0 + 32 (fm >> 1) + 8 (fm & 1) + 0..7] and + 16: two uint4 loads per token row and step, its
// half2 pairs built from 16-bit halves. A simdgroup computes 2 fragments (16 rows), 8 simdgroups split the steps
// (interleaved) and are reduced through threadgroup memory: 16 rows touched per simdgroup and step streams better than
// 32 (the stream probe: 32 rows 219 GB/s, 16 rows 227-233, 4-8 rows 240-247).
// The 16 fragments of one 128-element step: X pairs a0 / a1 (token fn), b0 / b1 (token fn + 1), the NF rows' codes cw
// and scale bytes sw
template <int L, int NF>
NI_INLINE void nh_x4_v8_frags(uint4 a0, uint4 a1, uint4 b0, uint4 b1, thread const uint4* cw, thread const uint* sw,
                              thread float2* acc, threadgroup const half2* tlh) {
  half2 sc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) sc[f] = nh_x4_sc2<L>(sw[f]);
  _Pragma("clang loop unroll(full)")
  for (int fr = 0; fr < 16; ++fr) {
    const uint wa = (fr < 8 ? a0 : a1)[(fr & 7) >> 1], wb = (fr < 8 ? b0 : b1)[(fr & 7) >> 1];
    const half2 bf = half2(as_type<half2>(wa)[fr & 1], as_type<half2>(wb)[fr & 1]);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      const half2 av = nh_x4_pair<L>(cw[f][fr >> 2], uint(fr & 3), tlh) * sc[f][fr >> 3];
      nh_x4_mma(acc[f], av, bf);
    }
  }
}

template <int L, bool TAIL>
NI_INLINE void nh_x4_v8_impl(const device half* X, const device uchar* q, const device uchar* s, device void* Y,
                             constant NhV8P& p, uint tg, uint sg, uint nsg, uint lane, threadgroup float2* red,
                             threadgroup const half2* tlh) {
  constexpr int NF = 2;
  const uint3 lc = nh_x4_lane(lane);
  const uint fm = lc.x, fn = lc.y, c = lc.z;
  const uint ks = sg, KS = nsg;  // the simdgroups split K (at most 8: max_total_threads_per_threadgroup(256))
  const uint n0 = tg * 8 * NF;
  const uint N = p.N, K = p.K;
  const bool l0 = fn < p.R, l1 = fn + 1 < p.R;
  const uint bo = 32 * (fm >> 1) + 8 * (fm & 1);
  const device half* xa = X + size_t(l0 ? fn : 0) * p.x_ld + bo;
  const device half* xb = X + size_t(l1 ? fn + 1 : 0) * p.x_ld + bo;
  const device uchar* qr[NF];
  const device uchar* sr[NF];
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = min(n0 + 8 * f + fm, N - 1);
    qr[f] = q + size_t(n) * (K / 2) + 16 * c;
    sr[f] = s + size_t(n) * (K / 16) + 2 * c;
    acc[f] = float2(0.0f);
  }
  if (n0 < N) {
    for (uint st = ks; st < K / 128; st += KS) {
      const uint k0 = st * 128;
      uint4 a0 = *(const device uint4*)(xa + k0), a1 = *(const device uint4*)(xa + k0 + 16);
      uint4 b0 = *(const device uint4*)(xb + k0), b1 = *(const device uint4*)(xb + k0 + 16);
      uint4 cw[NF];
      uint sw[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        cw[f] = *(const device uint4*)(qr[f] + k0 / 2);
        sw[f] = *(const device ushort*)(sr[f] + k0 / 16);
      }
      if (!l0) a0 = a1 = uint4(0);
      if (!l1) b0 = b1 = uint4(0);
      nh_x4_v8_frags<L, NF>(a0, a1, b0, b1, cw, sw, acc, tlh);
    }
    // TAIL: the last K % 128 inputs (K % 32 == 0) as one more step, the simdgroup next in the interleave's: its
    // 32-element groups at and past K are zero and read nothing (a lane's A pairs: its column pair c's group; its B
    // rows: k-slot fm's group, fm >> 1)
    if (TAIL && K % 128 != 0 && (K / 128) % KS == ks) {
      const uint k0 = K / 128 * 128, ng = (K % 128) / 32;
      const bool xok = (fm >> 1) < ng, wok = c < ng;
      uint4 a0 = uint4(0), a1 = uint4(0), b0 = uint4(0), b1 = uint4(0);
      if (xok && l0) a0 = *(const device uint4*)(xa + k0), a1 = *(const device uint4*)(xa + k0 + 16);
      if (xok && l1) b0 = *(const device uint4*)(xb + k0), b1 = *(const device uint4*)(xb + k0 + 16);
      uint4 cw[NF];
      uint sw[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        cw[f] = wok ? *(const device uint4*)(qr[f] + k0 / 2) : uint4(0);
        sw[f] = wok ? uint(*(const device ushort*)(sr[f] + k0 / 16)) : 0u;
      }
      nh_x4_v8_frags<L, NF>(a0, a1, b0, b1, cw, sw, acc, tlh);
    }
  }
  if (ks)
    for (int f = 0; f < NF; ++f) red[(sg * NF + uint(f)) * 32 + lane] = acc[f];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks) return;
  for (uint j = 1; j < KS; ++j)
    for (int f = 0; f < NF; ++f) acc[f] += red[(j * NF + uint(f)) * 32 + lane];
  const float post = p.gs * nh_x4_post_l[L];
  for (int f = 0; f < NF; ++f) {
    const uint n = n0 + 8 * f + fm;
    if (n >= N) continue;
    if (l0) nh_x4_store(Y, p, n, fn, acc[f].x * post);
    if (l1) nh_x4_store(Y, p, n, fn + 1, acc[f].y * post);
  }
}

// grid: ceil(N / 16) threadgroups x 256 threads (8 simdgroups split K; fewer work too, more are refused)
#define NH_X4_V8(NAME, L, TAIL)                                                                                    \
  [[kernel, max_total_threads_per_threadgroup(256)]] void NAME(                                                    \
                       const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]],                   \
                       const device uchar* Ws [[buffer(2)]], device void* Y [[buffer(3)]],                        \
                       constant NhV8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {        \
    threadgroup float2 red[8 * 2 * 32];                                                                            \
    threadgroup half2 tlh[L == 0 ? 256 : 1];                                                                       \
    if (L == 0) nh_x4_lut_to_tg(tlh, tid, nsg * 32);                                                               \
    nh_x4_v8_impl<L, TAIL>(X, Wq, Ws, Y, p, tg.x, sg, nsg, lane, red, tlh);                                        \
  }
NH_X4_V8(nh_x4_v8, 2, false)
NH_X4_V8(nh_x4_v8_l0, 0, false)
NH_X4_V8(nh_x4_v8_t, 2, true)
NH_X4_V8(nh_x4_v8_l0_t, 0, true)

// ---- up to 32 rows: TB blocks of 8 token rows. A threadgroup of 4 simdgroups computes 64 rows (2 fragments of 8 per
// simdgroup) for all TB token blocks, so each weight is decoded once per threadgroup for all rows; X of a 128-element
// step is staged in threadgroup memory in the order the A operands need: xs[t][fr * 8 + j] = X[t][k0 + k(fr, j)]
// (a step's block p of token t: two uint4 loads, eight 32-bit stores of half pairs; k-slots 2c', 2c' + 1 of
// fragments 8 (p & 1) + v, c' = p >> 1); B fragments by simdgroup_load.
// Split K (deterministic): the grid's x dimension is SK parts of K per row tile (y). With SK > 1 every part writes its
// fragments to part (buffer 5) and counts itself in cnt[tile] (buffer 6: zero before the first call, the last part
// resets it); the last part to arrive sums the SK partials in part order and stores. SK 4 is best or within 1% on
// in_proj (2-8) and out_proj (2-8), 8 on the drafter's K 12288 / 24576. ----
constant constexpr uint kX4Ld = 136;  // halfs per staged token row (128 + 8: conflict-free fragment loads)

// TAIL: K % 128 != 0 (K % 32 == 0): the last K % 128 inputs are one more step (in the last part), staged with its
// 16-element blocks at and past K zero and its column pairs' codes there zero (none of them read)
template <int TB, int L, bool TAIL>
NI_INLINE void nh_x4_mm_impl(const device half* X, const device uchar* q, const device uchar* s, device void* Y,
                             constant NhV8P& p, uint tile, uint pi, uint SK, uint sg, uint nsg, uint lane, uint tid,
                             threadgroup half* xs, threadgroup const half2* tlh, device float2* part,
                             device atomic_uint* cnt, threadgroup uint* arrival) {
  constexpr int NF = 2;
  const uint3 lc = nh_x4_lane(lane);
  const uint fm = lc.x, fn = lc.y, c = lc.z;
  const uint n0 = (tile * nsg + sg) * 8 * NF;
  const uint N = p.N, K = p.K, nthr = nsg * 32, S_all = TAIL ? (K + 127) / 128 : K / 128, per = (S_all + SK - 1) / SK;
  const uint st_lo = min(pi * per, S_all), S = min(st_lo + per, S_all) - st_lo;  // this part's steps
  const device uchar* qr[NF];
  const device uchar* sr[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = min(n0 + 8 * f + fm, N - 1);
    qr[f] = q + size_t(n) * (K / 2) + 16 * c + st_lo * 64;
    sr[f] = s + size_t(n) * (K / 16) + 2 * c + st_lo * 8;
  }
  const device half* xp = X + st_lo * 128;
  simdgroup_float8x8 acc[NF][TB];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < TB; ++b) acc[f][b] = simdgroup_float8x8(0);
  for (uint st = 0; st < S; ++st) {
    // the step's inputs before K: 128, or with TAIL the last step's K % 128 (a multiple of 32)
    const uint valid = TAIL ? min(128u, K - (st_lo + st) * 128) : 128u;
    // stage the step: item i = (token t, block pb)
    for (uint i = tid; i < 8 * TB * 8; i += nthr) {
      const uint t = i >> 3, pb = i & 7;
      uint4 v0 = uint4(0), v1 = uint4(0);
      if (t < p.R && (!TAIL || 16 * pb < valid)) {
        const device uint4* src = (const device uint4*)(xp + size_t(t) * p.x_ld + st * 128 + 16 * pb);
        v0 = src[0];
        v1 = src[1];
      }
      threadgroup uint* d = (threadgroup uint*)(xs + t * kX4Ld + 64 * (pb & 1) + 2 * (pb >> 1));
      d[0] = (v0.x & 0xFFFFu) | (v1.x << 16);
      d[4] = (v0.x >> 16) | (v1.x & 0xFFFF0000u);
      d[8] = (v0.y & 0xFFFFu) | (v1.y << 16);
      d[12] = (v0.y >> 16) | (v1.y & 0xFFFF0000u);
      d[16] = (v0.z & 0xFFFFu) | (v1.z << 16);
      d[20] = (v0.z >> 16) | (v1.z & 0xFFFF0000u);
      d[24] = (v0.w & 0xFFFFu) | (v1.w << 16);
      d[28] = (v0.w >> 16) | (v1.w & 0xFFFF0000u);
    }
    uint4 cw[NF];
    uint sw[NF];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      if (!TAIL || 32 * c < valid) {
        cw[f] = *(const device uint4*)(qr[f] + st * 64);
        sw[f] = *(const device ushort*)(sr[f] + st * 8);
      } else {
        cw[f] = uint4(0);
        sw[f] = 0u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half2 sc[NF];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) sc[f] = nh_x4_sc2<L>(sw[f]);
    _Pragma("clang loop unroll(full)")
    for (int fr = 0; fr < 16; ++fr) {
      simdgroup_half8x8 A[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        const half2 av = nh_x4_pair<L>(cw[f][fr >> 2], uint(fr & 3), tlh) * sc[f][fr >> 3];
        A[f].thread_elements()[0] = av.x;
        A[f].thread_elements()[1] = av.y;
      }
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < TB; ++b) {
        simdgroup_half8x8 B;
        simdgroup_load(B, xs + (8 * b) * kX4Ld + fr * 8, kX4Ld, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < NF; ++f) simdgroup_multiply_accumulate(acc[f][b], A[f], B, acc[f][b]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (SK > 1) {
#if NH_FENCES
    device float2* base = part + size_t(tile) * SK * nsg * (NF * TB * 32);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f)
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < TB; ++b)
        base[((pi * nsg + sg) * NF + uint(f)) * (TB * 32) + uint(b) * 32 + lane] =
            float2(acc[f][b].thread_elements()[0], acc[f][b].thread_elements()[1]);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival[0] = atomic_fetch_add_explicit(cnt + tile, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival[0] != SK - 1) return;
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f)
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < TB; ++b) {
        float2 v = float2(0.0f);
        for (uint j = 0; j < SK; ++j) v += base[((j * nsg + sg) * NF + uint(f)) * (TB * 32) + uint(b) * 32 + lane];
        acc[f][b].thread_elements()[0] = v.x;
        acc[f][b].thread_elements()[1] = v.y;
      }
    if (tid == 0) atomic_store_explicit(cnt + tile, 0u, memory_order_relaxed);
#else
    return;  // (no merge without device-scope fences: the engine runs SK = 1, NH_FENCES in nh_common.h)
#endif
  }
  const float post = p.gs * nh_x4_post_l[L];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = n0 + 8 * f + fm;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < TB; ++b)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint t = 8 * b + fn + e;
        if (t < p.R) nh_x4_store(Y, p, n, t, acc[f][b].thread_elements()[e] * post);
      }
  }
}

// grid: (SK, ceil(N / 64)) threadgroups x 128 threads. Buffers 5 (part: SK x ceil(N / 64) x 2048 TB bytes) and 6
// (cnt: ceil(N / 64) uints, zero before the first call; every call leaves them zero) must be bound; with SK = 1 they
// are not touched. Two split dispatches that may run at the same time need their own part / cnt.
#define NH_X4_MM(NAME, TB, L, TAIL)                                                                                \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
                       const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]],                   \
                       const device uchar* Ws [[buffer(2)]], device void* Y [[buffer(3)]],                        \
                       constant NhV8P& p [[buffer(4)]], device float2* part [[buffer(5)]],                        \
                       device atomic_uint* cnt [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],          \
                       uint2 ntg [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],            \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       uint tid [[thread_index_in_threadgroup]]) {                                                 \
    threadgroup half xs[8 * TB * kX4Ld];                                                                           \
    threadgroup half2 tlh[L == 0 ? 256 : 1];                                                                       \
    threadgroup uint arrival[1];                                                                                   \
    if (ntg.x > 16u) return; /* SK > 16: not a split grid (a 1-D dispatch?); write nothing */                      \
    if (L == 0) nh_x4_lut_to_tg(tlh, tid, nsg * 32);                                                               \
    nh_x4_mm_impl<TB, L, TAIL>(X, Wq, Ws, Y, p, tg.y, tg.x, ntg.x, sg, nsg, lane, tid, xs, tlh, part, cnt, arrival); \
  }
NH_X4_MM(nh_x4_mm16, 2, 2, false)
NH_X4_MM(nh_x4_mm24, 3, 2, false)
NH_X4_MM(nh_x4_mm32, 4, 2, false)
NH_X4_MM(nh_x4_mm16_l0, 2, 0, false)
NH_X4_MM(nh_x4_mm24_l0, 3, 0, false)
NH_X4_MM(nh_x4_mm32_l0, 4, 0, false)
NH_X4_MM(nh_x4_mm16_t, 2, 2, true)
NH_X4_MM(nh_x4_mm24_t, 3, 2, true)
NH_X4_MM(nh_x4_mm32_t, 4, 2, true)
NH_X4_MM(nh_x4_mm16_l0_t, 2, 0, true)
NH_X4_MM(nh_x4_mm24_l0_t, 3, 0, true)
NH_X4_MM(nh_x4_mm32_l0_t, 4, 0, true)

// Measured and lost (nh_x4_bench, rotating weights): decode with float2 byte tables in threadgroup (2 KB: bank
// conflicts) or constant memory, 8 rows per simdgroup, 64 elements per lane, the L1 bit trick with shifts or one
// multiply after a shift, L2 with two shifts and masks (register spills); 8 rows: 4 fragments per simdgroup (32 rows
// per step stream at 219 GB/s), 1 fragment, 2 column slices per threadgroup, contiguous k-slices, prefetching the next
// step (spills), X staged in threadgroup memory (barriers), constant-memory tables; up to 32 rows: 4 fragments per
// simdgroup (spills), 16-bit staging stores, k-groups inside a threadgroup instead of split K (out_proj 0.406 vs
// 0.332 ms at 32 rows). A token-pair-interleaved X (B pairs without packing) would gain 1.5-2%.

// The same down projection, and the row's sum over its k pairs by the last of the k threadgroups of each 32-row block
// to finish (an arrival counter per (row, block) in cnt [[11]]: zero before the first call, every call leaves it zero):
// lo[row] = sum_j part[row * k + j] in pair order, the same additions as nh_moe2_sum2, without its dispatch (22 x 1024
// partials of a one-token layer read again by every simdgroup of nh_moe_d1: 28 us). 8 simdgroups (32 rows) per
// threadgroup; grid (ceil(D / 32), pairs).
[[kernel]] void nh_x4_moe_dn_sum(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                                 const device float* h [[buffer(2)]], device float* part [[buffer(3)]],
                                 const device int* eid [[buffer(4)]], const device float* ew [[buffer(5)]],
                                 constant NhMoeParams& p [[buffer(6)]], const device uint* live [[buffer(8)]],
                                 const device float* gse [[buffer(9)]], device float* lo [[buffer(10)]],
                                 device atomic_uint* cnt [[buffer(11)]], uint3 tgp [[threadgroup_position_in_grid]],
                                 uint3 ntg [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half2 tlh[256];
  threadgroup uint arrival[1];
  nh_x4_lut_to_tg(tlh, tid, nsg * 32);
  const uint pair = tgp.y, r = pair / p.k;
  if (r >= live[0]) return;
  const uint e = uint(eid[pair]);
  nh_x4_mv_impl<0, 0>(q + size_t(e) * p.dn_q_stride, s + size_t(e) * p.dn_s_stride, h + size_t(pair) * p.h_stride,
                      part + size_t(pair) * p.D, p.D, p.I, ew[pair] * gse[e], (tgp.x * nsg + sg) * 4, lane, tlh);
#if NH_FENCES  // (else the partials only: nh_x4_moe_dn's, NH_FENCES in nh_common.h)
  device atomic_uint* c = cnt + r * ntg.x + tgp.x;
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    arrival[0] = atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (arrival[0] != p.k - 1) return;
  const uint rows = nsg * 4, d = tgp.x * rows + tid;
  if (tid < rows && d < p.D) {
    float v = 0.0f;
    for (uint j = 0; j < p.k; ++j) v += part[(size_t(r) * p.k + j) * p.D + d];
    lo[r * p.D + d] = v;
  }
  if (tid == 0) atomic_store_explicit(c, 0u, memory_order_relaxed);
#endif
}

// ---- NVFP4 routed experts for prefill: the pairs grouped by expert (nh_moe_bucket tiles of <= 8 TB rows), each tile
// on nh_x4_mm's design (weights decoded once per threadgroup into the A operands, the tile's token rows staged in
// threadgroup memory per 128-element step, no split K): 4 simdgroups x 16 output rows per threadgroup.
//   MODE 1 (up):   X row = sorted[m] / k (plat_, half [T][K]), Y row m (sorted order), half relu^2(gse[e] W x)
//   MODE 2 (down): X row m (sorted order, half [P][K]), Y row = sorted[m] (the pair), fp32 gse[e] W x
// grid: (ceil(N / 64), max tiles) x 128 threads; tiles past ntiles[0] return. K % 128 == 0.
template <int TB, int MODE, int L>
NI_INLINE void nh_x4_moe_mm_impl(const device half* X, const device uchar* q, const device uchar* s, device void* Y,
                                 constant NhGemmP& p, const device uint4* tiles, const device uint* ntiles,
                                 const device uint* sorted, const device float* gse, uint ntile, uint tile, uint sg,
                                 uint nsg, uint lane, uint tid, threadgroup half* xs, threadgroup const half2* tlh,
                                 threadgroup uint* xrow) {
  constexpr int NF = 2;
  if (tile >= ntiles[0]) return;
  const uint4 t = tiles[tile];
  const uint e = t.x, m0 = t.y, rows = t.z;
  const uint3 lc = nh_x4_lane(lane);
  const uint fm = lc.x, fn = lc.y, c = lc.z;
  const uint n0 = (ntile * nsg + sg) * 8 * NF;
  const uint N = p.N, K = p.K, nthr = nsg * 32;
  // the tile's X rows
  for (uint i = tid; i < 8 * TB; i += nthr) xrow[i] = i < rows ? (MODE == 1 ? sorted[m0 + i] / p.k : m0 + i) : 0xffffffffu;
  const device uchar* qe = q + size_t(e) * p.q_estride;
  const device uchar* se = s + size_t(e) * p.s_estride;
  const device uchar* qr[NF];
  const device uchar* sr[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = min(n0 + 8 * f + fm, N - 1);
    qr[f] = qe + size_t(n) * (K / 2) + 16 * c;
    sr[f] = se + size_t(n) * (K / 16) + 2 * c;
  }
  simdgroup_float8x8 acc[NF][TB];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < TB; ++b) acc[f][b] = simdgroup_float8x8(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = 0; st < K / 128; ++st) {
    for (uint i = tid; i < 8 * TB * 8; i += nthr) {
      const uint tr = i >> 3, pb = i & 7, xr = xrow[tr];
      uint4 v0 = uint4(0), v1 = uint4(0);
      if (xr != 0xffffffffu) {
        const device uint4* src = (const device uint4*)(X + size_t(xr) * p.x_ld + st * 128 + 16 * pb);
        v0 = src[0];
        v1 = src[1];
      }
      threadgroup uint* d = (threadgroup uint*)(xs + tr * kX4Ld + 64 * (pb & 1) + 2 * (pb >> 1));
      d[0] = (v0.x & 0xFFFFu) | (v1.x << 16);
      d[4] = (v0.x >> 16) | (v1.x & 0xFFFF0000u);
      d[8] = (v0.y & 0xFFFFu) | (v1.y << 16);
      d[12] = (v0.y >> 16) | (v1.y & 0xFFFF0000u);
      d[16] = (v0.z & 0xFFFFu) | (v1.z << 16);
      d[20] = (v0.z >> 16) | (v1.z & 0xFFFF0000u);
      d[24] = (v0.w & 0xFFFFu) | (v1.w << 16);
      d[28] = (v0.w >> 16) | (v1.w & 0xFFFF0000u);
    }
    uint4 cw[NF];
    uint sw[NF];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      cw[f] = *(const device uint4*)(qr[f] + st * 64);
      sw[f] = *(const device ushort*)(sr[f] + st * 8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half2 sc[NF];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) sc[f] = nh_x4_sc2<L>(sw[f]);
    _Pragma("clang loop unroll(full)")
    for (int fr = 0; fr < 16; ++fr) {
      simdgroup_half8x8 A[NF];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        const half2 av = nh_x4_pair<L>(cw[f][fr >> 2], uint(fr & 3), tlh) * sc[f][fr >> 3];
        A[f].thread_elements()[0] = av.x;
        A[f].thread_elements()[1] = av.y;
      }
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < TB; ++b) {
        simdgroup_half8x8 B;
        simdgroup_load(B, xs + (8 * b) * kX4Ld + fr * 8, kX4Ld, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < NF; ++f) simdgroup_multiply_accumulate(acc[f][b], A[f], B, acc[f][b]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float post = gse[e] * nh_x4_post_l[L];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = n0 + 8 * f + fm;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < TB; ++b)
      _Pragma("clang loop unroll(full)")
      for (int ee = 0; ee < 2; ++ee) {
        const uint r = 8 * b + fn + ee;
        if (r >= rows) continue;
        float v = acc[f][b].thread_elements()[ee] * post;
        if (MODE == 1) {
          v = max(v, 0.0f);
          ((device half*)Y)[size_t(m0 + r) * p.y_ld + n] = half(v * v);
        } else {
          ((device float*)Y)[size_t(sorted[m0 + r]) * p.y_ld + n] = v;
        }
      }
  }
}

#define NH_X4_MOE_MM(NAME, TB, MODE, L)                                                                            \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                    \
      const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]], const device uchar* Ws [[buffer(2)]], \
      device void* Y [[buffer(3)]], constant NhGemmP& p [[buffer(4)]], const device uint4* tiles [[buffer(5)]],   \
      const device uint* ntiles [[buffer(6)]], const device uint* sorted [[buffer(7)]],                          \
      const device float* gse [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],                          \
      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {                       \
    threadgroup half xs[8 * TB * kX4Ld];                                                                           \
    threadgroup half2 tlh[L == 0 ? 256 : 1];                                                                       \
    threadgroup uint xrow[8 * TB];                                                                                 \
    if (L == 0) nh_x4_lut_to_tg(tlh, tid, nsg * 32);                                                               \
    nh_x4_moe_mm_impl<TB, MODE, L>(X, Wq, Ws, Y, p, tiles, ntiles, sorted, gse, tg.x, tg.y, sg, nsg, lane, tid,    \
                                   xs, tlh, xrow);                                                                 \
  }
NH_X4_MOE_MM(nh_x4_moe_up_m16, 2, 1, 0)
NH_X4_MOE_MM(nh_x4_moe_up_m32, 4, 1, 0)
NH_X4_MOE_MM(nh_x4_moe_dn_m16, 2, 2, 0)
NH_X4_MOE_MM(nh_x4_moe_dn_m32, 4, 2, 0)
NH_X4_MOE_MM(nh_x4_moe_up_m16_l2, 2, 1, 2)  // experts in layout L2
NH_X4_MOE_MM(nh_x4_moe_up_m32_l2, 4, 1, 2)
NH_X4_MOE_MM(nh_x4_moe_dn_m16_l2, 2, 2, 2)
NH_X4_MOE_MM(nh_x4_moe_dn_m32_l2, 4, 2, 2)
