// ---- GPU side of Nemotron-H's GPU + Neural Engine prefill split (src/model/nemotron_ane.cpp) ----
// The scheme of src/kernels/ane_ffn.metal (W8A8 after a block Hadamard rotation v -> D H v / sqrt(n),
// per-row weight and per-token input int8 scales, int8 codes read by the ANE as code / 128), for the checkpoint's
// formats: the Mamba in / out projections (NVFP4 in Bolt, FP8 in Super-VL) and the shared expert (IQ2_XXS in Bolt,
// NVFP4 in Super-VL). Weight rows rotate in blocks of
// 128 along their inputs (in_proj, out_proj, shared up) or of 256 along the shared expert's intermediate (its down
// projection, matching the ANE program's rotation of the intermediate). The activations are rotated and packed by
// ane_rotate / ane_pack of the base library.

constant constexpr float kNhAnePeak = 127.0f;
constant constexpr float kNhAneUnit = 128.0f;
constant constexpr float kNhAneFloor = 0x1p-14f;

// ane_ffn.metal's ane_rotate_block (natural-order Walsh-Hadamard of K*128 values held 4 per lane, then signs and
// 1/sqrt(n)): the same arithmetic, so weights and activations rotate alike
template <uint K>
NI_INLINE void nh_rot_block(thread float (&v)[K][4], uint lane, const device float* sign) {
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

// 4 weights of row n at inputs [k, k + 4) (k % 4 == 0), dequantized
template <int FMT>
NI_INLINE float4 nh_w4(const device uchar* q, const device uchar* s, float gs, uint K, uint n, uint k) {
  if (FMT == 1) {  // FP4: byte i of a 16-block holds elements i (low nibble) and i + 8 (high)
    const uint blk = k / 16, i0 = k % 16;
    const uint w = *(const device uint*)(q + size_t(n) * (K / 2) + blk * 8 + (i0 & 7u));
    const uint sh = i0 >= 8 ? 4u : 0u;
    const float sc = nh_ue4m3(s[size_t(n) * (K / 16) + blk]) * gs;
    return float4(nh_fp4_lut[(w >> sh) & 15u], nh_fp4_lut[(w >> (8 + sh)) & 15u], nh_fp4_lut[(w >> (16 + sh)) & 15u],
                  nh_fp4_lut[(w >> (24 + sh)) & 15u]) * sc;
  } else if (FMT == 5) {  // FP8: 16-element groups, byte 4 m + (0, 1, 2, 3) = elements 2 m, 2 m + 1, 2 m + 8, 2 m + 9
    const device uchar* g = q + size_t(n) * K + (k & ~15u);
    const uint i0 = k % 16;
    float4 r;
    for (uint e = 0; e < 4; ++e) {
      const uint el = i0 + e, b = g[4 * ((el & 7u) >> 1) + (el & 1u) + 2 * (el >> 3)];
      r[e] = float(as_type<half>(ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7))));  // x 2^-8: gs holds 2^8
    }
    return r * gs;
  } else {  // IQ2: sub-block k / 32, grid point (k % 32) / 8, elements k % 8 .. + 4
    const uint2 aux = *(const device uint2*)(q + size_t(n) * (K / 4) + (k / 32) * 8);
    const float db = float(((const device half*)s)[size_t(n) * (K / 256) + k / 256]) * (0.5f + float(aux.y >> 28)) * 0.25f;
    const uint l = (k % 32) / 8, e0 = k % 8;
    const ulong g = nh_iq2xxs_grid[(aux.x >> (8 * l)) & 0xFFu];
    const uint s7 = (aux.y >> (7 * l)) & 127u;
    const uint sg = s7 | ((popcount(s7) & 1u) << 7);
    float4 r;
    for (uint e = 0; e < 4; ++e) {
      const float m = float(uint(g >> (8 * (e0 + e))) & 0xFFu) * db;
      r[e] = (sg >> (e0 + e)) & 1u ? -m : m;
    }
    return r;
  }
}

struct NhAneWP {
  uint row0;      // first source row
  uint col0;      // first input
  uint width;     // inputs (a multiple of 128 K)
  uint K;         // source row length
  uint stride;    // int8 output bytes per row (staging)
  uint sstride;   // halves between rows of the scale surface (staging)
  float gs;       // tensor scale (FP4)
};

// Per-row int8 scale of the rotated rows: absmax over [col0, col0 + width) after the rotation in K*128 blocks,
// / 127 * 128, as half. One simdgroup per row, 8 rows per threadgroup. grid: rows / 8 x 256 threads.
template <int FMT, uint KB>
NI_INLINE void nh_ane_rs(const device uchar* q, const device uchar* s, device half* rs, const device float* sign,
                         constant NhAneWP& p, uint tg, uint sg, uint lane) {
  const uint r = tg * 8 + sg, n = p.row0 + r;
  float peak = 0.0f;
  for (uint blk = 0; blk < p.width; blk += KB * 128) {
    float v[KB][4];
    for (uint k = 0; k < KB; ++k) {
      const float4 f = nh_w4<FMT>(q, s, p.gs, p.K, n, p.col0 + blk + k * 128 + lane * 4);
      for (uint e = 0; e < 4; ++e) v[k][e] = f[e];
    }
    nh_rot_block<KB>(v, lane, sign);
    for (uint k = 0; k < KB; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(v[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) rs[r] = half(max(peak, kNhAneFloor) / kNhAnePeak * kNhAneUnit);
}

// Rows [row0, + rows) x inputs [col0, + width) as rotated int8 under the row scales rs into out (stride bytes per
// row), the scales into the scale surface. One simdgroup per row and KB*128 block; grid: (rows / 8, width / (KB*128))
// x 256 threads.
template <int FMT, uint KB>
NI_INLINE void nh_ane_stage(const device uchar* q, const device uchar* s, const device half* rs, device char* out,
                            device half* sout, const device float* sign, constant NhAneWP& p, uint2 tg, uint sg,
                            uint lane) {
  const uint r = tg.x * 8 + sg, n = p.row0 + r, blk = tg.y * KB * 128;
  float v[KB][4];
  for (uint k = 0; k < KB; ++k) {
    const float4 f = nh_w4<FMT>(q, s, p.gs, p.K, n, p.col0 + blk + k * 128 + lane * 4);
    for (uint e = 0; e < 4; ++e) v[k][e] = f[e];
  }
  nh_rot_block<KB>(v, lane, sign);
  const half scale = rs[r];
  const float inv = kNhAneUnit / float(scale);
  for (uint k = 0; k < KB; ++k)
    *(device char4*)(out + ulong(r) * p.stride + blk + k * 128 + lane * 4) =
        char4(clamp(rint(float4(v[k][0], v[k][1], v[k][2], v[k][3]) * inv), -kNhAnePeak, kNhAnePeak));
  if (tg.y == 0 && lane == 0) sout[r * p.sstride] = scale;
}

#define NH_ANE_W(SFX, FMT, KB)                                                                                     \
  [[kernel]] void nh_ane_rs_##SFX(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],        \
                                  device half* rs [[buffer(2)]], const device float* sign [[buffer(3)]],           \
                                  constant NhAneWP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],     \
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    nh_ane_rs<FMT, KB>(q, s, rs, sign, p, tg, sg, lane);                                                            \
  }                                                                                                                \
  [[kernel]] void nh_ane_stage_##SFX(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],     \
                                     const device half* rs [[buffer(2)]], device char* out [[buffer(3)]],          \
                                     device half* sout [[buffer(4)]], const device float* sign [[buffer(5)]],      \
                                     constant NhAneWP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], \
                                     uint sg [[simdgroup_index_in_threadgroup]],                                   \
                                     uint lane [[thread_index_in_simdgroup]]) {                                    \
    nh_ane_stage<FMT, KB>(q, s, rs, out, sout, sign, p, tg, sg, lane);                                            \
  }
NH_ANE_W(fp4_b128, 1, 1)  // in_proj / out_proj rows (Bolt); shared up rows (Super-VL)
NH_ANE_W(fp8_b128, 5, 1)  // in_proj / out_proj rows (Super-VL)
NH_ANE_W(iq2_b128, 2, 1)  // shared up rows (Bolt)
NH_ANE_W(iq2_b256, 2, 2)  // shared down rows over the rotated intermediate (Bolt)
NH_ANE_W(fp4_b256, 1, 2)  // the same, NVFP4 (Super-VL)

// A projection's ANE rows into the model's buffers: Y fp16 [rows][ystride] channel-major (row c = output col0 + c),
// tx the per-token input scale. ADD 0: X[t][col0 + c] = Y[c][t] * tx[t]; 1: +=. A non-finite value sets *status.
// grid: (ceil(T / 32), ceil(rows / 32)) x (32, 8).
struct NhAneJoinP {
  uint T, rows, ystride, ld, col0;
};
template <int ADD>
NI_INLINE void nh_ane_join_impl(device float* X, const device half* Y, const device half* tx, device atomic_uint* status,
                                constant NhAneJoinP& p, uint2 tile, uint2 pos, threadgroup float (&st)[32][33],
                                threadgroup float* sc) {
  const uint t0 = tile.x * 32, c0 = tile.y * 32, t = t0 + pos.x;
  const bool in = t < p.T;
  bool bad = false;
  for (uint j = pos.y; j < 32; j += 8) {
    const half v = c0 + j < p.rows ? Y[(c0 + j) * p.ystride + t] : half(0);
    bad |= in && (as_type<ushort>(v) & 0x7c00) == 0x7c00;
    st[j][pos.x] = float(v);
  }
  if (pos.y == 0) {
    const half s = tx[t];
    bad |= in && (as_type<ushort>(s) & 0x7c00) == 0x7c00;
    sc[pos.x] = float(s);
  }
  if (bad) atomic_store_explicit(status, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = pos.y; j < 32; j += 8) {
    const uint tt = t0 + j;
    if (tt >= p.T || c0 + pos.x >= p.rows) continue;
    device float* o = X + size_t(tt) * p.ld + p.col0 + c0 + pos.x;
    const float v = st[pos.x][j] * sc[j];
    if (ADD) *o += v;
    else *o = v;
  }
}
[[kernel]] void nh_ane_join_store(device float* X [[buffer(0)]], const device half* Y [[buffer(1)]],
                                  const device half* tx [[buffer(2)]], device atomic_uint* status [[buffer(3)]],
                                  constant NhAneJoinP& p [[buffer(4)]], uint2 tile [[threadgroup_position_in_grid]],
                                  uint2 pos [[thread_position_in_threadgroup]]) {
  threadgroup float st[32][33];
  threadgroup float sc[32];
  nh_ane_join_impl<0>(X, Y, tx, status, p, tile, pos, st, sc);
}
[[kernel]] void nh_ane_join_add(device float* X [[buffer(0)]], const device half* Y [[buffer(1)]],
                                const device half* tx [[buffer(2)]], device atomic_uint* status [[buffer(3)]],
                                constant NhAneJoinP& p [[buffer(4)]], uint2 tile [[threadgroup_position_in_grid]],
                                uint2 pos [[thread_position_in_threadgroup]]) {
  threadgroup float st[32][33];
  threadgroup float sc[32];
  nh_ane_join_impl<1>(X, Y, tx, status, p, tile, pos, st, sc);
}
