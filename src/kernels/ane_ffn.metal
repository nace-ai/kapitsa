#include "common.h"

// ---- GPU side of the GPU + Neural Engine split of the dense prefill FFN (NI_ANE=1) ----
// The ANE multiplies int8 weights and int8 activations after a block-diagonal Hadamard rotation
// v -> D H v / sqrt(n) (H Walsh-Hadamard in natural order, D fixed random signs), which keeps dot
// products of rows rotated alike and spreads outliers before the per-token / per-row int8 scales.
// FFN inputs and gate/up weight rows rotate in blocks of 128 along the hidden dim; down's weight
// rows rotate in blocks of 512 along the ANE's intermediate channels (the ANE program rotates its
// intermediate the same way). int8 codes are read by the ANE as code/128 (fp16 sums cannot
// overflow), so every scale carries a factor 128. Weights: MLX affine 4-bit, group 64 (w uint32 [N][K/8], value j
// of a word at bits 4j, scales and biases bf16 [N][K/64]) here; the other formats (bf16, MLX affine 3/5/6/8-bit,
// 4-bit group 32: wfmt.metal's packing) through ane_wf4, instantiated in wfmt.metal's second library.
// ane_rotate_block below is adapted from Splash's ane_ffn_rotate_block (incoai/splash, Apache-2.0).
// Nace AI changes: helper/variable names, direct indexing of v[k], an explicit ushort shuffle mask,
// and a literal 128-element rotation unit. License notice: THIRD_PARTY_NOTICES.

constant constexpr float kAnePeak = 127.0f;
constant constexpr float kAneUnit = 128.0f;
constant constexpr float kAneFloor = 0x1p-14f;  // least peak: fp16's least normal value

// One block of K*128 values held 4 per lane in each 128 (value k*128 + lane*4 + e): 2 butterfly
// stages in registers, 5 across the simdgroup, then log2(K) across the k's. Natural-order WHT.
template <uint K>
NI_INLINE void ane_rotate_block(thread float (&v)[K][4], uint lane, const device float* sign) {
  for (uint k = 0; k < K; ++k) {
    for (uint hs = 1; hs < 4; hs <<= 1)
      for (uint e = 0; e < 4; ++e)
        if (!(e & hs)) {
          const float a = v[k][e], b = v[k][e + hs];
          v[k][e] = a + b;
          v[k][e + hs] = a - b;
        }
    for (uint m = 1; m < 32; m <<= 1)
      for (uint e = 0; e < 4; ++e) {
        const float o = simd_shuffle_xor(v[k][e], ushort(m));
        v[k][e] = (lane & m) ? o - v[k][e] : v[k][e] + o;
      }
  }
  for (uint hs = 1; hs < K; hs <<= 1)
    for (uint k = 0; k < K; ++k)
      if (!(k & hs))
        for (uint e = 0; e < 4; ++e) {
          const float a = v[k][e], b = v[k + hs][e];
          v[k][e] = a + b;
          v[k + hs][e] = a - b;
        }
  const float norm = rsqrt(float(K * 128));
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e) v[k][e] *= sign[k * 128 + lane * 4 + e] * norm;
}

// fp16 inf/NaN by exponent bits (fast math may fold isfinite to true)
NI_INLINE bool ane_bad(half v) { return (as_type<ushort>(v) & 0x7c00) == 0x7c00; }

// Rotates each token row of X (half [T][D], post-norm FFN input) in 128-blocks, scales it to
// +-127 by its absmax and writes it to R (half [T][D]) and tx[t] = scale * 128 (the ANE's token
// scale). Rows t in [T, rows) get tx = 0 (padding of the ANE function). grid: rows threadgroups x
// 256 threads; D/128 blocks split evenly over the simdgroups, at most 8 each (D = 5120: 5).
struct AneRotParams {
  uint T, D;
};
[[kernel]] void ane_rotate(const device half* X [[buffer(0)]], const device float* sign [[buffer(1)]],
                           device half* R [[buffer(2)]], device half* tx [[buffer(3)]],
                           constant AneRotParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float peaks[32];
  if (row >= p.T) {
    if (sg == 0 && lane == 0) tx[row] = half(0);
    return;
  }
  constexpr uint MAXB = 8;
  const uint nb = p.D / 128 / nsg;  // blocks per simdgroup
  float v[MAXB][4];
  float peak = 0.0f;
  for (uint i = 0; i < MAXB; ++i) {
    if (i >= nb) break;
    const uint base = row * p.D + (sg * nb + i) * 128 + lane * 4;
    const half4 h = *(const device half4*)(X + base);
    float b[1][4] = {{float(h.x), float(h.y), float(h.z), float(h.w)}};
    ane_rotate_block<1>(b, lane, sign);
    for (uint e = 0; e < 4; ++e) {
      v[i][e] = b[0][e];
      peak = max(peak, fabs(b[0][e]));
    }
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[sg] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint g = 0; g < nsg; ++g) peak = max(peak, peaks[g]);
  const float scale = max(peak, kAneFloor) / kAnePeak, inv = 1.0f / scale;
  for (uint i = 0; i < MAXB; ++i) {
    if (i >= nb) break;
    const uint base = row * p.D + (sg * nb + i) * 128 + lane * 4;
    *(device half4*)(R + base) = half4(v[i][0] * inv, v[i][1] * inv, v[i][2] * inv, v[i][3] * inv);
  }
  if (sg == 0 && lane == 0) tx[row] = half(scale * kAneUnit);
}

// Rotated rows R[t][ch0 + c] (t < T; zeros for T <= t < tiles*32) -> int8 out[c][t] (channel-major,
// `stride` bytes per channel). grid: (ceil(rows/32), channels/32) x (32, 8) threads.
struct AnePackParams {
  uint T, D, ch0, stride;
};
[[kernel]] void ane_pack(const device half* R [[buffer(0)]], device char* out [[buffer(1)]],
                         constant AnePackParams& p [[buffer(2)]], uint2 tile [[threadgroup_position_in_grid]],
                         uint2 pos [[thread_position_in_threadgroup]]) {
  threadgroup half st[32][33];
  const uint t0 = tile.x * 32, c0 = tile.y * 32;
  for (uint j = pos.y; j < 32; j += 8) {
    const uint t = t0 + j;
    st[j][pos.x] = t < p.T ? R[t * p.D + p.ch0 + c0 + pos.x] : half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = pos.y; j < 32; j += 8)
    out[(c0 + j) * p.stride + t0 + pos.x] = char(clamp(rint(float(st[pos.x][j])), -kAnePeak, kAnePeak));
}

// Weight rows [row0, row0 + rows) of an MLX 4-bit matrix over inputs [col0, col0 + width).
struct AneWParams {
  uint row0, col0, width;
  uint wpr, gpr;      // uint32 words / scale groups per source row
  uint stride;        // int8 output bytes per row
  uint sstride;       // halves between rows of the scale surface
};

// 4 dequantized values at inputs [k, k+4) of row n (k a multiple of 4)
NI_INLINE float4 ane_mlx4(const device uint32_t* W, const device bfloat* S, const device bfloat* B, uint n, uint k,
                          uint wpr, uint gpr) {
  const uint q = ((const device ushort*)(W + n * wpr))[k / 4];
  const float s = float(S[n * gpr + k / 64]), b = float(B[n * gpr + k / 64]);
  return float4(float(q & 15u), float((q >> 4) & 15u), float((q >> 8) & 15u), float(q >> 12)) * s + b;
}

// ane_mlx4 for format F (wfmt.metal's: F-bit codes as a little-endian bit stream per row, w = s * q + b per group
// of GS; F = 16 bf16): 4 values at inputs [k, k+4) of row n, k a multiple of 4.
template <int F, int GS>
NI_INLINE float4 ane_wf4(const device uint32_t* W, const device bfloat* S, const device bfloat* B, uint n, uint k,
                         uint wpr, uint gpr) {
  if (F == 4 && GS == 64) return ane_mlx4(W, S, B, n, k, wpr, gpr);
  const device uint32_t* row = W + n * wpr;
  if (F == 16) {
    const uint2 u = *(const device uint2*)(row + k / 2);
    return float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u), as_type<float>(u.y << 16),
                  as_type<float>(u.y & 0xFFFF0000u));
  }
  const uint bit = uint(F) * k, w0 = bit / 32, sh = bit % 32;
  uint v = row[w0] >> sh;
  if (sh + 4 * F > 32) v |= row[w0 + 1] << (32 - sh);
  const uint m = (1u << F) - 1u;
  const float s = float(S[n * gpr + k / GS]), b = float(B[n * gpr + k / GS]);
  return float4(float(v & m), float((v >> F) & m), float((v >> (2 * F)) & m), float((v >> (3 * F)) & m)) * s + b;
}

// Per-row int8 scale of the rotated rows (once, at load): absmax over the row's [col0, col0 +
// width) after the rotation in K*128 blocks, / 127 * 128, as half. One simdgroup per row, 8 rows
// per threadgroup. grid: rows/8 x 256 threads.
template <uint K, int F = 4, int GS = 64>
NI_INLINE void ane_row_scale_body(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                                  device half* rs, const device float* sign, constant AneWParams& p, uint tg,
                                  uint sg, uint lane) {
  const uint r = tg * 8 + sg, n = p.row0 + r;
  float peak = 0.0f;
  for (uint blk = 0; blk < p.width; blk += K * 128) {
    float v[K][4];
    for (uint k = 0; k < K; ++k) {
      const float4 f = ane_wf4<F, GS>(W, S, B, n, p.col0 + blk + k * 128 + lane * 4, p.wpr, p.gpr);
      for (uint e = 0; e < 4; ++e) v[k][e] = f[e];
    }
    ane_rotate_block<K>(v, lane, sign);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(v[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) rs[r] = half(max(peak, kAneFloor) / kAnePeak * kAneUnit);
}

// Stages rows [row0, +rows) x inputs [col0, +width) as rotated int8 under the row scales `rs`
// (from ane_row_scale) into out (stride bytes per row) and copies the scales into the ANE's scale
// surface. One simdgroup per row and K*128-block; grid: (rows/8, width/(K*128)) x 256 threads.
template <uint K, int F = 4, int GS = 64>
NI_INLINE void ane_stage_body(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                              const device half* rs, device char* out, device half* sout, const device float* sign,
                              constant AneWParams& p, uint2 tg, uint sg, uint lane) {
  const uint r = tg.x * 8 + sg, n = p.row0 + r, blk = tg.y * K * 128;
  float v[K][4];
  for (uint k = 0; k < K; ++k) {
    const float4 f = ane_wf4<F, GS>(W, S, B, n, p.col0 + blk + k * 128 + lane * 4, p.wpr, p.gpr);
    for (uint e = 0; e < 4; ++e) v[k][e] = f[e];
  }
  ane_rotate_block<K>(v, lane, sign);
  const half scale = rs[r];
  const float inv = kAneUnit / float(scale);
  for (uint k = 0; k < K; ++k)
    *(device char4*)(out + ulong(r) * p.stride + blk + k * 128 + lane * 4) =
        char4(clamp(rint(float4(v[k][0], v[k][1], v[k][2], v[k][3]) * inv), -kAnePeak, kAnePeak));
  if (tg.y == 0 && lane == 0) sout[r * p.sstride] = scale;
}

#define NI_ANE_W(SUFFIX, K)                                                                                    \
  [[kernel]] void ane_row_scale_##SUFFIX(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]], \
                                         const device bfloat* B [[buffer(2)]], device half* rs [[buffer(3)]],      \
                                         const device float* sign [[buffer(4)]], constant AneWParams& p [[buffer(5)]], \
                                         uint tg [[threadgroup_position_in_grid]],                                  \
                                         uint sg [[simdgroup_index_in_threadgroup]],                                \
                                         uint lane [[thread_index_in_simdgroup]]) {                                 \
    ane_row_scale_body<K>(W, S, B, rs, sign, p, tg, sg, lane);                                                     \
  }                                                                                                                \
  [[kernel]] void ane_stage_##SUFFIX(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],    \
                                     const device bfloat* B [[buffer(2)]], const device half* rs [[buffer(3)]],       \
                                     device char* out [[buffer(4)]], device half* sout [[buffer(5)]],               \
                                     const device float* sign [[buffer(6)]], constant AneWParams& p [[buffer(7)]],  \
                                     uint2 tg [[threadgroup_position_in_grid]],                                     \
                                     uint sg [[simdgroup_index_in_threadgroup]],                                    \
                                     uint lane [[thread_index_in_simdgroup]]) {                                     \
    ane_stage_body<K>(W, S, B, rs, out, sout, sign, p, tg, sg, lane);                                              \
  }

NI_ANE_W(b128, 1)  // gate/up rows over the hidden inputs
NI_ANE_W(b512, 4)  // down rows over the ANE's intermediate channels

// The other weight formats: ane_row_scale_<fmt>_b128 ... (instantiated in wfmt.metal under NI_WF_INST).
#define NI_ANE_WF(SFX, F, GS)                                                                                     \
  [[kernel]] void ane_row_scale_##SFX##_b128(const device uint32_t* W [[buffer(0)]],                              \
                                             const device bfloat* S [[buffer(1)]], const device bfloat* B [[buffer(2)]], \
                                             device half* rs [[buffer(3)]], const device float* sign [[buffer(4)]], \
                                             constant AneWParams& p [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], \
                                             uint sg [[simdgroup_index_in_threadgroup]],                          \
                                             uint lane [[thread_index_in_simdgroup]]) {                           \
    ane_row_scale_body<1, F, GS>(W, S, B, rs, sign, p, tg, sg, lane);                                             \
  }                                                                                                               \
  [[kernel]] void ane_row_scale_##SFX##_b512(const device uint32_t* W [[buffer(0)]],                              \
                                             const device bfloat* S [[buffer(1)]], const device bfloat* B [[buffer(2)]], \
                                             device half* rs [[buffer(3)]], const device float* sign [[buffer(4)]], \
                                             constant AneWParams& p [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], \
                                             uint sg [[simdgroup_index_in_threadgroup]],                          \
                                             uint lane [[thread_index_in_simdgroup]]) {                           \
    ane_row_scale_body<4, F, GS>(W, S, B, rs, sign, p, tg, sg, lane);                                             \
  }                                                                                                               \
  [[kernel]] void ane_stage_##SFX##_b128(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]], \
                                         const device bfloat* B [[buffer(2)]], const device half* rs [[buffer(3)]],  \
                                         device char* out [[buffer(4)]], device half* sout [[buffer(5)]],          \
                                         const device float* sign [[buffer(6)]], constant AneWParams& p [[buffer(7)]], \
                                         uint2 tg [[threadgroup_position_in_grid]],                                \
                                         uint sg [[simdgroup_index_in_threadgroup]],                               \
                                         uint lane [[thread_index_in_simdgroup]]) {                                \
    ane_stage_body<1, F, GS>(W, S, B, rs, out, sout, sign, p, tg, sg, lane);                                      \
  }                                                                                                               \
  [[kernel]] void ane_stage_##SFX##_b512(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]], \
                                         const device bfloat* B [[buffer(2)]], const device half* rs [[buffer(3)]],  \
                                         device char* out [[buffer(4)]], device half* sout [[buffer(5)]],          \
                                         const device float* sign [[buffer(6)]], constant AneWParams& p [[buffer(7)]], \
                                         uint2 tg [[threadgroup_position_in_grid]],                                \
                                         uint sg [[simdgroup_index_in_threadgroup]],                               \
                                         uint lane [[thread_index_in_simdgroup]]) {                                \
    ane_stage_body<4, F, GS>(W, S, B, rs, out, sout, sign, p, tg, sg, lane);                                      \
  }

// x[t][c] += Y[c][t] * Y[D][t] * tx[t] in fp32 for t < T (Y = the ANE's partial down projection,
// fp16 [D+1][ystride], row D = per-token intermediate scale; tx = per-token input scale). A
// non-finite partial or scale of a row of the chunk sets *status. grid: (ceil(T/32), D/32) x (32, 8).
struct AneJoinParams {
  uint T, D, ystride;
};
[[kernel]] void ane_join(device float* X [[buffer(0)]], const device half* Y [[buffer(1)]],
                         const device half* tx [[buffer(2)]], device atomic_uint* status [[buffer(3)]],
                         constant AneJoinParams& p [[buffer(4)]], uint2 tile [[threadgroup_position_in_grid]],
                         uint2 pos [[thread_position_in_threadgroup]]) {
  threadgroup float st[32][33];
  threadgroup float sc[32];
  const uint t0 = tile.x * 32, c0 = tile.y * 32, t = t0 + pos.x;
  const bool in = t < p.T;
  bool bad = false;
  for (uint j = pos.y; j < 32; j += 8) {
    const half v = Y[(c0 + j) * p.ystride + t];
    bad |= in && ane_bad(v);
    st[j][pos.x] = float(v);
  }
  if (pos.y == 0) {
    const half hs = Y[p.D * p.ystride + t], s = tx[t];
    bad |= in && (ane_bad(hs) || ane_bad(s));
    sc[pos.x] = float(hs) * float(s);
  }
  if (bad) atomic_store_explicit(status, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = pos.y; j < 32; j += 8) {
    const uint tt = t0 + j;
    if (tt >= p.T) continue;
    X[tt * p.D + c0 + pos.x] += st[pos.x][j] * sc[j];
  }
}
