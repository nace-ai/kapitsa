#include "common.h"

// ---- 8-row verify kernels (DFlash2 block verify; any R <= 8 token rows, padded to 8) ----
//
// The target verifies R <= 8 rows at once (anchor + up to 7 drafts). 8 rows are exactly one 8x8
// simdgroup_matrix dimension, so every projection runs as Y^T[n, t] = W[n, :] . X^T[:, t] on 8x8x8
// MMAs with the weight as the A operand and the 8 token rows as B:
//  * A holds the INTEGER CODES of the MLX affine weights, exact in half as 1024 + q
//    (`(codes & 0x000F000F) | 0x64006400`: two integer ops per two weights, no dequant FMA);
//  * one MMA chain per 64-group, closed by an fp32 epilogue per group and output:
//      y += s * dot + (b - 1024 s) * sum_k x[k]
//    with the group sums of the (half-rounded) inputs precomputed per token row.
//  The Apple9 register-tile technique: integer codes as exact MMA operands, one fp32 epilogue per
//  coefficient group, reading the MLX weight layout directly (no repacked weight image), with half
//  activations (11-bit mantissa).
//
// Activations: an "X8 table" per 64-group holds the 8 rows' inputs already in the per-lane B-fragment
// order, so a lane reads its 16 operand halfs of a group with two 16-byte loads:
//   table[(g * 32 + lane) * 16 + 2 f + e] = half(x[fn + e][64 g + k8(f, fm)])     (lane = (fm, fn))
//   sums[g * 8 + t] = sum over the group of float(half(x[t][k]))
// where k8(f, j) = 16 (j / 2) + 8 (f / 4) + (f % 4) + 4 (j % 2) is the physical input of k-slot j of
// fragment f. That permutation makes a lane's A elements of all 8 fragments of a group 16 contiguous
// 4-bit codes of its weight row (8 bytes, one uint2 load: word 0 -> fragments 0-3, word 1 -> 4-7, pair
// (nibble s, nibble s + 4) of a word = slots (2q, 2q + 1)), the layout opt/shortpf's qmms uses.
// Rows t >= R of the table are zero.
//
// simdgroup 8x8 lane layout: lane holds row fm = (q & 4) | ((lane >> 1) & 3), columns fn, fn + 1 with
// fn = ((q & 2) << 1) | ((lane & 1) << 1), q = lane >> 2.

constant constexpr uint kV8 = 8;  // token rows of a verify tile

NI_INLINE uint2 v8_lane(uint lane) {
  const uint q = lane >> 2;
  return uint2((q & 4) | ((lane >> 1) & 3), ((q & 2) << 1) | ((lane & 1) << 1));
}
NI_INLINE uint v8_lane_of(uint fm, uint fn) {
  return ((fm >> 2) << 4) | ((fn >> 2) << 3) | ((fm & 3) << 1) | ((fn >> 1) & 1);
}
// physical input (within a 64-group) -> (fragment f, k-slot j)
NI_INLINE uint2 v8_fj(uint kk) { return uint2(4 * ((kk >> 3) & 1) + (kk & 3), 2 * (kk >> 4) + ((kk >> 2) & 1)); }
// table offset (in halfs) of input kk of group g for token row t
NI_INLINE uint v8_tab(uint g, uint kk, uint t) {
  const uint2 fj = v8_fj(kk);
  return (g * 32 + v8_lane_of(fj.y, t & 6u)) * 16 + 2 * fj.x + (t & 1u);
}

// One 8x8x8 MMA on plain-register operands: c += a x b (fp32 accumulate). The simdgroup_matrix
// objects live only inside the call, so c stays an ordinary register pair.
NI_INLINE void v8_mma(thread float2& c, half2 a, half2 b) {
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

// The A operand (1024 + code pairs) of fragment f from this lane's words of a group.
template <int BITS>
NI_INLINE half2 v8_codes(uint4 w, int f) {
  if (BITS == 4) {
    const uint word = f < 4 ? w.x : w.y;
    return as_type<half2>(((word >> (4 * (f & 3))) & 0x000F000Fu) | 0x64006400u);
  } else {
    const uint wa = w[2 * (f >> 2)], wb = w[2 * (f >> 2) + 1], sh = 8 * (f & 3);
    return as_type<half2>(((wa >> sh) & 0xFFu) | (((wb >> sh) & 0xFFu) << 16) | 0x64006400u);
  }
}

NI_INLINE float v8_bf(ushort b) { return as_type<float>(uint(b) << 16); }

// Subnormal-code operand (CM 1): the bare code bits q are the half subnormal q * 2^-24, exact, so the
// MMA chain sums q * x * 2^-24 in fp32 with no 1024 * sum(x) term to cancel; the epilogue scales by
// s * 2^24. One integer op fewer per MMA (no OR) and ~10 more bits of the chain's result survive.
template <int BITS>
NI_INLINE half2 v8_codes_sn(uint4 w, int f) {
  if (BITS == 4) {
    const uint word = f < 4 ? w.x : w.y;
    return as_type<half2>((word >> (4 * (f & 3))) & 0x000F000Fu);
  } else {
    const uint wa = w[2 * (f >> 2)], wb = w[2 * (f >> 2) + 1], sh = 8 * (f & 3);
    return as_type<half2>(((wa >> sh) & 0xFFu) | (((wb >> sh) & 0xFFu) << 16));
  }
}
constant constexpr float kV8Sub = 16777216.0f;  // 2^24

// ---- input tables ----
struct X8PrepParams {
  uint K;
  uint R;         // live rows (t >= R are written as zeros)
  uint x_ld;      // floats per input row
  uint norm;      // 1: RMSNorm with weight w
  uint write_xn;  // 1: also store the (normalized) fp32 rows into xn (row stride xn_ld)
  float eps;
  uint xn_ld;
  uint write_lo;  // 1: also the residual table lo = half(x - hi) and its group sums (two-term inputs)
  uint unroll;    // x8_prep2 with norm: 4 independent sum-of-squares chains (more loads in flight)
  uint write_t;   // x8_prep2: also the T table (buffer 8) of the tensor_ops GEMM (tensor/verify8t.metal)
};

// The T table of qmm8t (src/kernels/tensor/verify8t.metal): half rows [8][K], each 64-group's inputs permuted so
// that slot 4 c + 16 jb + kk holds input 16 c + 8 (jb >> 1) + 2 kk + (jb & 1) (the matrix unit's operand layout).
NI_INLINE uint v8t_input_of_slot(uint s) {
  const uint c = (s >> 2) & 3, jb = (s >> 4) & 3, kk = s & 3;
  return 16 * c + 8 * (jb >> 1) + 2 * kk + (jb & 1);
}

// grid: 8 threadgroups (token rows) x 256 threads.
[[kernel]] void x8_prep(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                        device half* table [[buffer(2)]], device float* sums [[buffer(3)]],
                        device float* xn [[buffer(4)]], constant X8PrepParams& p [[buffer(5)]],
                        device half* table_lo [[buffer(6)]], device float* sums_lo [[buffer(7)]],
                        uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  const bool live = t < p.R;
  const device float* xr = x + t * p.x_ld;
  float inv = 1.0f;
  if (p.norm) {
    float ss = 0.0f;
    if (live)
      for (uint i = tid; i < p.K / 4; i += ntg) {
        const float4 v = ((const device float4*)xr)[i];
        ss += dot(v, v);
      }
    ss = tg_sum(ss, red, tid, ntg);
    inv = metal::precise::rsqrt(ss / float(p.K) + p.eps);
  }
  const uint G = p.K / 64;
  for (uint g = sg; g < G; g += nsg) {
    const uint kk = 2 * lane;
    float2 v = live ? *(const device float2*)(xr + g * 64 + kk) : float2(0.0f);
    if (p.norm) v = v * inv * float2(float(w[g * 64 + kk]), float(w[g * 64 + kk + 1]));
    if (p.write_xn && live) *(device float2*)(xn + t * p.xn_ld + g * 64 + kk) = v;
    const half2 h = half2(v);
    const uint o0 = v8_tab(g, kk, t), o1 = v8_tab(g, kk + 1, t);
    table[o0] = h.x;
    table[o1] = h.y;
    const float s = simd_sum(float(h.x) + float(h.y));
    if (lane == 0) sums[g * kV8 + t] = s;
    if (p.write_lo) {
      const half2 l = half2(v - float2(h));
      table_lo[o0] = l.x;
      table_lo[o1] = l.y;
      const float sl = simd_sum(float(l.x) + float(l.y));
      if (lane == 0) sums_lo[g * kV8 + t] = sl;
    }
  }
}

// x8_prep2: same outputs as x8_prep, 4 groups (256 inputs) x 8 rows per threadgroup, so the grid
// covers the GPU and the table goes out as coalesced 16-byte stores (staged through threadgroup
// memory). With norm, every threadgroup computes the 8 row norms itself (each reads the rows once).
// grid: K / 256 threadgroups x 256 threads (K a multiple of 256); x row tiles (grid y, v8_tiles: rows 8 y ..
// 8 y + 7 into tile y's tables, K * 8 halfs / K / 64 * 8 sums apart; the T table's rows 8 y ..).
[[kernel]] void x8_prep2(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                         device half* table [[buffer(2)]], device float* sums [[buffer(3)]],
                         device float* xn [[buffer(4)]], constant X8PrepParams& p [[buffer(5)]],
                         device half* table_lo [[buffer(6)]], device float* sums_lo [[buffer(7)]],
                         device half* table_t [[buffer(8)]],
                         uint2 tg2 [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half stg[2][8][256];
  threadgroup float inv_s[8];
  const uint tg = tg2.x, tile = tg2.y;
  if (tile) {  // a row tile past the first: its rows and tables
    x += 8 * tile * p.x_ld;
    xn += 8 * tile * p.xn_ld;
    table += tile * p.K * 8;
    table_lo += tile * p.K * 8;
    sums += tile * (p.K / 64) * kV8;
    sums_lo += tile * (p.K / 64) * kV8;
    table_t += 8 * tile * p.K;
  }
  const uint t = sg;  // simdgroup = token row
  const bool live = 8 * tile + t < p.R;
  const device float* xr = x + t * p.x_ld;
  if (p.norm) {
    float ss = 0.0f;
    if (live && p.unroll && p.K % 512 == 0) {
      float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
      for (uint i = lane; i < p.K / 4; i += 128) {
        const float4 v0 = ((const device float4*)xr)[i], v1 = ((const device float4*)xr)[i + 32];
        const float4 v2 = ((const device float4*)xr)[i + 64], v3 = ((const device float4*)xr)[i + 96];
        a0 = fma(v0, v0, a0);
        a1 = fma(v1, v1, a1);
        a2 = fma(v2, v2, a2);
        a3 = fma(v3, v3, a3);
      }
      const float4 a = (a0 + a1) + (a2 + a3);
      ss = (a.x + a.y) + (a.z + a.w);
    } else if (live) {
      for (uint i = lane; i < p.K / 4; i += 32) {
        const float4 v = ((const device float4*)xr)[i];
        ss += dot(v, v);
      }
    }
    ss = simd_sum(ss);
    if (lane == 0) inv_s[t] = metal::precise::rsqrt(ss / float(p.K) + p.eps);
  }
  const uint k0 = tg * 256 + lane * 8;  // this lane's 8 inputs of row t
  float v[8];
  if (live) {
    const float4 a = *(const device float4*)(xr + k0), b = *(const device float4*)(xr + k0 + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  } else {
    for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  }
  if (p.norm) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[t];
    for (int i = 0; i < 8; ++i) v[i] = v[i] * inv * float(w[k0 + i]);
  }
  if (p.write_xn && live) {
    *(device float4*)(xn + t * p.xn_ld + k0) = float4(v[0], v[1], v[2], v[3]);
    *(device float4*)(xn + t * p.xn_ld + k0 + 4) = float4(v[4], v[5], v[6], v[7]);
  }
  float sh = 0.0f, sl = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const half h = half(v[i]);
    stg[0][t][lane * 8 + i] = h;
    sh += float(h);
    if (p.write_lo) {
      const half l = half(v[i] - float(h));
      stg[1][t][lane * 8 + i] = l;
      sl += float(l);
    }
  }
  // group sums: the 8 lanes of a group (lane / 8) in a fixed order
  sh += simd_shuffle_xor(sh, 1u);
  sh += simd_shuffle_xor(sh, 2u);
  sh += simd_shuffle_xor(sh, 4u);
  sl += simd_shuffle_xor(sl, 1u);
  sl += simd_shuffle_xor(sl, 2u);
  sl += simd_shuffle_xor(sl, 4u);
  if ((lane & 7) == 0) {
    sums[(tg * 4 + lane / 8) * kV8 + t] = sh;
    if (p.write_lo) sums_lo[(tg * 4 + lane / 8) * kV8 + t] = sl;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // table: 4 groups x 32 lanes x 16 halfs; thread = (group, lane', half of the 16)
  const uint gi = tid / 64, l2 = (tid % 64) / 2, hf = tid % 2;
  const uint2 rc = v8_lane(l2);
  const uint fm = rc.x, fn = rc.y;
  for (int which = 0; which < (p.write_lo ? 2 : 1); ++which) {
    half o[8];
    for (int jj = 0; jj < 8; ++jj) {
      const uint j = hf * 8 + jj, f = j / 2, e = j % 2;
      const uint kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
      o[jj] = stg[which][fn + e][gi * 64 + kk];
    }
    device half* dst = (which ? table_lo : table) + ((tg * 4 + gi) * 32 + l2) * 16 + hf * 8;
    *(device uint4*)dst = uint4(as_type<uint>(half2(o[0], o[1])), as_type<uint>(half2(o[2], o[3])),
                                as_type<uint>(half2(o[4], o[5])), as_type<uint>(half2(o[6], o[7])));
  }
  if (p.write_t) {  // T table: thread = (row tid / 32, 8 slots of this threadgroup's 256)
    const uint tt = tid / 32, s0 = (tid % 32) * 8, gb = s0 & ~63u;
    half o[8];
    for (int i = 0; i < 8; ++i) o[i] = stg[0][tt][gb + v8t_input_of_slot((s0 + i) & 63)];
    *(device uint4*)(table_t + tt * p.K + tg * 256 + s0) =
        uint4(as_type<uint>(half2(o[0], o[1])), as_type<uint>(half2(o[2], o[3])), as_type<uint>(half2(o[4], o[5])),
              as_type<uint>(half2(o[6], o[7])));
  }
}

// ---- the GEMM core ----
// acc[f] (lane's C[fm][fn], C[fm][fn+1] of fragment f) += sum over groups [g0, g1) of the fragment's
// weight rows (wp/sp/bp: this lane's row, already offset to the lane's words) times the table.
// X2 (precise mode, for small matrices such as the MoE router): exact codes q (the 1024 offset is
// subtracted in half, exactly) and two-term inputs hi + lo (lo = half(x - hi)), i.e. ~22-bit inputs
// and no 1024 * sum cancellation; twice the MMAs.
template <int BITS, int NF, bool X2 = false>
NI_INLINE void v8_run(const device uint32_t* const thread* wp, const device ushort* const thread* sp,
                      const device ushort* const thread* bp, const device half* table, const device float* sums,
                      const device half* table_lo, const device float* sums_lo, uint g0, uint g1, uint lane,
                      uint fn, thread float2* acc) {
  constexpr uint WPG = 64 * BITS / 32;  // words per group and row
  for (uint g = g0; g < g1; ++g) {
    const device half4* tb = (const device half4*)(table + (g * 32 + lane) * 16);
    const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
    const half2 b[8] = {x0.xy, x0.zw, x1.xy, x1.zw, x2.xy, x2.zw, x3.xy, x3.zw};
    float2 sx = *(const device float2*)(sums + g * kV8 + fn);
    half2 bl[8];
    if (X2) {
      const device half4* tl = (const device half4*)(table_lo + (g * 32 + lane) * 16);
      const half4 l0 = tl[0], l1 = tl[1], l2 = tl[2], l3 = tl[3];
      bl[0] = l0.xy; bl[1] = l0.zw; bl[2] = l1.xy; bl[3] = l1.zw;
      bl[4] = l2.xy; bl[5] = l2.zw; bl[6] = l3.xy; bl[7] = l3.zw;
      sx += *(const device float2*)(sums_lo + g * kV8 + fn);
    }
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      uint4 w;
      if (BITS == 4) {
        const uint2 v = *(const device uint2*)(wp[f] + g * WPG);
        w = uint4(v, 0u, 0u);
      } else {
        w = *(const device uint4*)(wp[f] + g * WPG);
      }
      const float s = v8_bf(sp[f][g]), bi = v8_bf(bp[f][g]);
      float2 dot = float2(0.0f);
      if (X2) {
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 8; ++j) {
          const half2 a = v8_codes<BITS>(w, j) - half2(1024.0h);
          v8_mma(dot, a, b[j]);
          v8_mma(dot, a, bl[j]);
        }
        acc[f] = fma(dot, float2(s), acc[f]);
        acc[f] = fma(sx, float2(bi), acc[f]);
      } else {
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 8; ++j) v8_mma(dot, v8_codes<BITS>(w, j), b[j]);
        acc[f] = fma(dot, float2(s), acc[f]);
        acc[f] = fma(sx, float2(fma(-1024.0f, s, bi)), acc[f]);
      }
    }
  }
}

// Software-pipelined variant of v8_run (no X2): group g + 1's weights, coefficients and inputs are
// loaded before group g's MMAs run, so every simdgroup keeps two groups of loads in flight.
template <int BITS, int NF>
struct V8Ops {
  uint4 w[NF];
  ushort s[NF], b[NF];
  uint4 t0, t1;
  float2 sx;
};
template <int BITS, int NF>
NI_INLINE void v8_load(thread V8Ops<BITS, NF>& o, const device uint32_t* const thread* wp,
                       const device ushort* const thread* sp, const device ushort* const thread* bp,
                       const device half* table, const device float* sums, uint g, uint lane, uint fn) {
  constexpr uint WPG = 64 * BITS / 32;
  const device uint4* tb = (const device uint4*)(table + (g * 32 + lane) * 16);
  o.t0 = tb[0];
  o.t1 = tb[1];
  o.sx = *(const device float2*)(sums + g * kV8 + fn);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    if (BITS == 4) o.w[f] = uint4(*(const device uint2*)(wp[f] + g * WPG), 0u, 0u);
    else o.w[f] = *(const device uint4*)(wp[f] + g * WPG);
    o.s[f] = sp[f][g];
    o.b[f] = bp[f][g];
  }
}
template <int BITS, int NF, bool SPLIT = false, int ABL = 0, int CM = 0>
NI_INLINE void v8_compute(thread const V8Ops<BITS, NF>& o, thread float2* acc) {
  half2 b[8];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 4; ++j) {
    b[j] = as_type<half2>(o.t0[j]);
    b[j + 4] = as_type<half2>(o.t1[j]);
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const float s = v8_bf(o.s[f]), bi = v8_bf(o.b[f]);
    float2 dot = float2(0.0f);
    if (ABL == 1) {  // ablation: no MMAs, same operands consumed
      half2 t = half2(0.0h);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; ++j) t += v8_codes<BITS>(o.w[f], j) * b[j];
      dot = float2(t);
    } else if (SPLIT) {  // two independent MMA chains per fragment (more MMAs in flight)
      float2 d1 = float2(0.0f);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; j += 2) {
        v8_mma(dot, v8_codes<BITS>(o.w[f], j), b[j]);
        v8_mma(d1, v8_codes<BITS>(o.w[f], j + 1), b[j + 1]);
      }
      dot += d1;
    } else if (CM == 1) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; ++j) v8_mma(dot, v8_codes_sn<BITS>(o.w[f], j), b[j]);
    } else {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; ++j) v8_mma(dot, v8_codes<BITS>(o.w[f], j), b[j]);
    }
    if (CM != 0) {
      acc[f] = fma(dot, float2(s * kV8Sub), acc[f]);
      acc[f] = fma(o.sx, float2(bi), acc[f]);
    } else {
      acc[f] = fma(dot, float2(s), acc[f]);
      acc[f] = fma(o.sx, float2(fma(-1024.0f, s, bi)), acc[f]);
    }
  }
}
// ABL (bench-only ablations): 1 = no MMAs, 2 = every group reads group 0's weights (cache-resident:
// no DRAM streaming), so the kernel time is the MMA/ALU side alone.
template <int BITS, int NF, bool SPLIT = false, int ABL = 0, int CM = 0>
NI_INLINE void v8_run_pf(const device uint32_t* const thread* wp, const device ushort* const thread* sp,
                         const device ushort* const thread* bp, const device half* table, const device float* sums,
                         uint g0, uint g1, uint lane, uint fn, thread float2* acc) {
  if (g0 >= g1) return;
  V8Ops<BITS, NF> cur, nxt;
  v8_load<BITS, NF>(cur, wp, sp, bp, table, sums, ABL == 2 ? 0 : g0, lane, fn);
  for (uint g = g0; g < g1; ++g) {
    if (g + 1 < g1) v8_load<BITS, NF>(nxt, wp, sp, bp, table, sums, ABL == 2 ? 0 : g + 1, lane, fn);
    v8_compute<BITS, NF, SPLIT, ABL, CM>(cur, acc);
    cur = nxt;
  }
}

// Two groups per step (4-bit): the scale and bias pairs of groups g, g + 1 come in one 4-byte load each
// (g even), so a fragment-group needs 2 loads instead of 3, and the loop runs half the iterations (27B, 8-row verify
// with CM 1: -0.7 to -0.9 ms per DFlash2 cycle; a prefetching version of it took 96 registers and was no faster).
// An odd first or last group runs alone.
template <int NF>
struct V8Ops2 {
  uint2 w[2][NF];
  uint s[NF], b[NF];
  uint4 t[2][2];
  float2 sx[2];
};
template <int NF>
NI_INLINE void v8_load2(thread V8Ops2<NF>& o, const device uint32_t* const thread* wp, const device ushort* const thread* sp,
                        const device ushort* const thread* bp, const device half* table, const device float* sums,
                        uint g, uint lane, uint fn) {
  _Pragma("clang loop unroll(full)")
  for (int h = 0; h < 2; ++h) {
    const device uint4* tb = (const device uint4*)(table + ((g + h) * 32 + lane) * 16);
    o.t[h][0] = tb[0];
    o.t[h][1] = tb[1];
    o.sx[h] = *(const device float2*)(sums + (g + h) * kV8 + fn);
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    o.w[0][f] = *(const device uint2*)(wp[f] + g * 8);
    o.w[1][f] = *(const device uint2*)(wp[f] + g * 8 + 8);
    o.s[f] = *(const device uint*)(sp[f] + g);
    o.b[f] = *(const device uint*)(bp[f] + g);
  }
}
template <int NF, int CM>
NI_INLINE void v8_compute2(thread const V8Ops2<NF>& o, thread float2* acc) {
  _Pragma("clang loop unroll(full)")
  for (int h = 0; h < 2; ++h) {
    V8Ops<4, NF> c;
    c.t0 = o.t[h][0];
    c.t1 = o.t[h][1];
    c.sx = o.sx[h];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      c.w[f] = uint4(o.w[h][f], 0u, 0u);
      c.s[f] = ushort(h ? o.s[f] >> 16 : o.s[f] & 0xffffu);
      c.b[f] = ushort(h ? o.b[f] >> 16 : o.b[f] & 0xffffu);
    }
    v8_compute<4, NF, false, 0, CM>(c, acc);
  }
}
template <int NF, int CM>
NI_INLINE void v8_run_pair(const device uint32_t* const thread* wp, const device ushort* const thread* sp,
                           const device ushort* const thread* bp, const device half* table, const device float* sums,
                           uint g0, uint g1, uint lane, uint fn, thread float2* acc) {
  if ((g0 & 1u) && g0 < g1) {  // pairs start at an even group (4-byte aligned scale / bias pairs)
    V8Ops<4, NF> c;
    v8_load<4, NF>(c, wp, sp, bp, table, sums, g0, lane, fn);
    v8_compute<4, NF, false, 0, CM>(c, acc);
    ++g0;
  }
  const uint gp = g0 + ((g1 - g0) & ~1u);  // pairs end here
  for (uint g = g0; g < gp; g += 2) {
    V8Ops2<NF> cur;
    v8_load2<NF>(cur, wp, sp, bp, table, sums, g, lane, fn);
    v8_compute2<NF, CM>(cur, acc);
  }
  if (gp < g1) {
    V8Ops<4, NF> c;
    v8_load<4, NF>(c, wp, sp, bp, table, sums, gp, lane, fn);
    v8_compute<4, NF, false, 0, CM>(c, acc);
  }
}

template <int BITS, int NF, bool SPLIT = false>
NI_INLINE void v8_run_lc(const device uint32_t* const thread* wp, const device ushort* const thread* sp,
                         const device ushort* const thread* bp, const device half* table, const device float* sums,
                         uint g0, uint g1, uint lane, uint fn, thread float2* acc) {
  for (uint g = g0; g < g1; ++g) {
    V8Ops<BITS, NF> cur;
    v8_load<BITS, NF>(cur, wp, sp, bp, table, sums, g, lane, fn);
    v8_compute<BITS, NF, SPLIT>(cur, acc);
  }
}

// Table staged in threadgroup memory: the cs simdgroups of a threadgroup (all on the same K range,
// ks = 1) share each chunk of CG groups of the X8 table, so the table is read from L2 once per
// threadgroup instead of once per simdgroup (per group: 1 KB of table vs 256 B of weights per
// fragment). ABL 3 (bench-only): no table reads at all (B from group 0, staged once).
template <int BITS, int NF, int CG, int ABL = 0>
NI_INLINE void v8_run_tg(const device uint32_t* const thread* wp, const device ushort* const thread* sp,
                         const device ushort* const thread* bp, const device half* table, const device float* sums,
                         threadgroup uint4* tt, threadgroup float2* ts, uint g0, uint g1, uint tid, uint nthr,
                         uint lane, uint fn, thread float2* acc) {
  constexpr uint WPG = 64 * BITS / 32;
  for (uint gc = g0; gc < g1; gc += CG) {
    const uint n = min(uint(CG), g1 - gc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ABL != 3 || gc == g0) {
      const device uint4* src = (const device uint4*)(table + gc * 512);
      for (uint i = tid; i < n * 64; i += nthr) tt[i] = src[i];
      const device float2* ssrc = (const device float2*)(sums + gc * kV8);
      for (uint i = tid; i < n * 4; i += nthr) ts[i] = ssrc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint gi = 0; gi < n; ++gi) {
      const uint g = gc + gi, gt = ABL == 3 ? 0 : gi;
      V8Ops<BITS, NF> o;
      o.t0 = tt[gt * 64 + lane * 2];
      o.t1 = tt[gt * 64 + lane * 2 + 1];
      o.sx = ts[gt * 4 + fn / 2];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) {
        if (BITS == 4) o.w[f] = uint4(*(const device uint2*)(wp[f] + g * WPG), 0u, 0u);
        else o.w[f] = *(const device uint4*)(wp[f] + g * WPG);
        o.s[f] = sp[f][g];
        o.b[f] = bp[f][g];
      }
      v8_compute<BITS, NF>(o, acc);
    }
  }
}

// Fixed-order sum of the ks k-slices of each column block (simdgroups sg with the same sg / ks).
// Afterwards the simdgroup with kslice 0 holds the total. Every simdgroup must call it.
template <int NF>
NI_INLINE void v8_reduce(thread float2* acc, threadgroup float2* red, uint ks, uint sg, uint lane) {
  const uint kslice = sg % ks;
  if (kslice)
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) red[(sg * NF + f) * 32 + lane] = acc[f];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (kslice == 0)
    for (uint j = 1; j < ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NF; ++f) acc[f] += red[((sg + j) * NF + f) * 32 + lane];
}

struct Qmm8Params {
  uint K;
  uint R;          // rows stored (t < R)
  uint nseg;       // segments sharing the input table, <= 4
  uint ks;         // k-slices per column block; simdgroups per threadgroup = cs * ks
  uint N[4];
  uint y_ld[4];
  uint add[4];     // 1: y += result (residual)
  uint tg_begin[5];
};

// Row tiles (KFlash tree verifies wider than 8 rows: 16 / 32 rows as 2 / 4 tiles of 8 rows): a GEMM's grid then holds
// tg_begin[nseg] threadgroups per tile with the tiles fastest (threadgroup x = column block * nt + tile), so the tiles
// of a column block run side by side and its weights come from DRAM once (the other tiles read them from the caches).
// Tile t reads its own X8 table (the tables of the tiles follow each other: K * 8 halfs and K / 64 * 8 sums apart) and
// writes rows 8 t .. 8 t + 7. A grid of tg_begin[nseg] threadgroups is the one-tile GEMM, unchanged.
NI_INLINE uint v8_tiles(uint ntgs, uint per_tile) { return ntgs > per_tile ? ntgs / per_tile : 1u; }
// The rows a tree verify of capacity R computes: up to 8, the tree's live rows (kf_live); a wider verify (16 / 32 rows) runs the
// tiles of its tree's live rows (the size rule's n = tree[kKfN], whole 8-row tiles: 8 ceil(n / 8), at most R).
NI_INLINE uint kf_live_rows(const device int* tree, uint R) {
  if (R <= 8) return kf_live(tree, R);
  const uint n = uint(max(tree[kKfN], 1));
  return min(R, (n + 7) / 8 * 8);
}

// Dense projections, up to 4 segments sharing the input (e.g. GDN qkv|z|b|a): segment i's
// threadgroups [tg_begin[i], tg_begin[i+1]) each cover cs column blocks of 8 * NF rows.
// V: 0 = v8_run, 1 = prefetching (v8_run_pf), 2 = v8_run_lc + split chains, 3 = prefetching + split chains
template <int BITS, int NF, bool X2, int V = 0>
NI_INLINE void qmm8_impl(const device uint32_t* const thread* Wv, const device ushort* const thread* Sv,
                         const device ushort* const thread* Bv, device float* const thread* Yv,
                         const device half* table, const device float* sums, const device half* table_lo,
                         const device float* sums_lo, constant Qmm8Params& p, threadgroup float2* red, uint tg,
                         uint sg, uint nsg, uint lane, threadgroup uint4* tt = nullptr,
                         threadgroup float2* ts = nullptr, uint ntgs = 0) {
  // row tiles (v8_tiles): this threadgroup's tile and its tables
  const uint nt = v8_tiles(ntgs, p.tg_begin[p.nseg]);
  const uint tile = nt > 1 ? tg % nt : 0u;
  if (nt > 1) {
    tg /= nt;
    const uint tw = tile * (p.K / 64);
    table += tw * 512;
    sums += tw * kV8;
    table_lo += tw * 512;
    sums_lo += tw * kV8;
  }
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint cs = nsg / p.ks, kslice = sg % p.ks;
  const uint n0 = ((tg - p.tg_begin[si]) * cs + sg / p.ks) * 8 * NF;
  const uint N = p.N[si], gpr = p.K / 64, wpr = p.K * BITS / 32;
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* wp[NF];
  const device ushort* sp[NF];
  const device ushort* bp[NF];
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint row = min(n0 + 8 * f + fm, N - 1);
    wp[f] = Wv[si] + row * wpr + q * (BITS == 4 ? 2 : 4);
    sp[f] = Sv[si] + row * gpr;
    bp[f] = Bv[si] + row * gpr;
    acc[f] = float2(0.0f);
  }
  if (X2 || V == 0) v8_run<BITS, NF, X2>(wp, sp, bp, table, sums, table_lo, sums_lo, g0, g1, lane, fn, acc);
  else if (V == 1) v8_run_pf<BITS, NF>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 2) v8_run_lc<BITS, NF, true>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 3) v8_run_pf<BITS, NF, true>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 4) v8_run_pf<BITS, NF, false, 1>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 5) v8_run_pf<BITS, NF, false, 2>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 8) v8_run_pf<BITS, NF, false, 0, 1>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 13) v8_run_pair<NF, 1>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 14) v8_run_pair<NF, 0>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (V == 6) v8_run_tg<BITS, NF, 8>(wp, sp, bp, table, sums, tt, ts, g0, g1, sg * 32 + lane, nsg * 32, lane, fn, acc);
  else v8_run_tg<BITS, NF, 8, 3>(wp, sp, bp, table, sums, tt, ts, g0, g1, sg * 32 + lane, nsg * 32, lane, fn, acc);
  if (p.ks > 1) {
    v8_reduce<NF>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  device float* Y = Yv[si];
  const uint y_ld = p.y_ld[si];
  const bool add = p.add[si] != 0;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint n = n0 + 8 * f + fm;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint t = 8 * tile + fn + e;
      if (t < p.R) {
        device float* d = Y + t * y_ld + n;
        *d = add ? *d + acc[f][e] : acc[f][e];
      }
    }
  }
}

#define V8_STAGED_(V, ...) ((V) == 6 || (V) == 7)
#define V8_STAGED(...) V8_STAGED_(__VA_ARGS__ + 0, 0)
#define NI_QMM8(NAME, BITS, NF, X2, ...) NI_QMM8_SG(NAME, 8, BITS, NF, X2, ##__VA_ARGS__)
// MAXSG: simdgroups a threadgroup may have (cs * ks), sizes the k-slice reduction buffer
#define NI_QMM8_SG(NAME, MAXSG, BITS, NF, X2, ...)                                                                   \
  [[kernel]] void NAME(const device uint32_t* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],       \
                       const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                \
                       const device uint32_t* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],       \
                       const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                \
                       const device uint32_t* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],       \
                       const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],              \
                       const device uint32_t* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],     \
                       const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],              \
                       const device half* table [[buffer(16)]], const device float* sums [[buffer(17)]],     \
                       constant Qmm8Params& p [[buffer(18)]], const device half* table_lo [[buffer(19)]],     \
                       const device float* sums_lo [[buffer(20)]], uint tg [[threadgroup_position_in_grid]], \
                       uint ntgs [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {   \
    threadgroup float2 red[MAXSG * NF * 32];                                                                 \
    threadgroup uint4 tt[(V8_STAGED(__VA_ARGS__)) ? 8 * 64 : 1];                                              \
    threadgroup float2 ts[(V8_STAGED(__VA_ARGS__)) ? 8 * 4 : 1];                                              \
    const device uint32_t* Wv[4] = {W0, W1, W2, W3};                                                          \
    const device ushort* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device ushort* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmm8_impl<BITS, NF, X2, ##__VA_ARGS__>(Wv, Sv, Bv, Yv, table, sums, table_lo, sums_lo, p, red, tg, sg, nsg, lane, \
                                           tt, ts, ntgs);                                                      \
  }

NI_QMM8(qmm8_b4_nf1, 4, 1, false)
NI_QMM8(qmm8_b4_nf2, 4, 2, false)
NI_QMM8(qmm8_b4_nf4, 4, 4, false)
NI_QMM8(qmm8_b8_nf1, 8, 1, false)
NI_QMM8(qmm8_b8_nf2, 8, 2, false)
NI_QMM8(qmm8x2_b4_nf1, 4, 1, true)
NI_QMM8(qmm8x2_b8_nf1, 8, 1, true)
NI_QMM8(qmm8pf_b4_nf1, 4, 1, false, 1)
NI_QMM8(qmm8pf_b4_nf2, 4, 2, false, 1)  // the default 4-bit GEMM
NI_QMM8(qmm8pf_b4_nf4, 4, 4, false, 1)
NI_QMM8(qmm8s_b4_nf2, 4, 2, false, 2)
NI_QMM8(qmm8pfs_b4_nf2, 4, 2, false, 3)
NI_QMM8(qmm8sn_b4_nf1, 4, 1, false, 8)    // subnormal-code operands (CM 1), prefetching loop
NI_QMM8(qmm8sn_b4_nf2, 4, 2, false, 8)
NI_QMM8(qmm8b_b4_nf1, 4, 1, false, 14)     // CM 0, two groups per step
NI_QMM8(qmm8b_b4_nf2, 4, 2, false, 14)
NI_QMM8(qmm8snb_b4_nf1, 4, 1, false, 13)   // CM 1, two groups per step (the default dense GEMM)
NI_QMM8(qmm8snb_b4_nf2, 4, 2, false, 13)
NI_QMM8_SG(qmm8snb16_b4_nf1, 16, 4, 1, false, 13)  // up to 16 simdgroups (k-slices) per threadgroup
NI_QMM8_SG(qmm8snb16_b4_nf2, 16, 4, 2, false, 13)
// Experiments measured and dropped (not instantiated to keep the library small; add the line to
// reproduce): NI_QMM8(qmm8abl1_b4_nf2, 4, 2, false, 4)  no MMAs; NI_QMM8(qmm8abl2_b4_nf2, 4, 2, false, 5)
// cache-resident weights; NI_QMM8(qmm8t_b4_nf2, 4, 2, false, 6) threadgroup-staged table (ks = 1);
// NI_QMM8(qmm8abl3_b4_nf2, 4, 2, false, 7) staged table without table reads.

// ---- gate/up + SwiGLU, writing the down projection's input table ----
// h[t][n] = half(silu(x_t . Wg[n]) * (x_t . Wu[n])) for all 8 rows, stored straight into an X8 table of
// width N (the down GEMM's K) with its group sums: no fp32 intermediate, no prepare dispatch.
// A threadgroup is 4 simdgroups x 16 columns = one 64-group of h: simdgroup s owns columns
// 16 s .. 16 s + 15 of gate and up (fragments 0-1 gate, 2-3 up, sharing the B operand).
// Dense MLP: grid (N / 64, 1). MoE experts: grid (N / 64, tiles) with tiles[y] = expert id
// (0xffffffff: the shared expert, weight set 2), each tile writing its own table (h_stride halfs,
// hs_stride floats apart); tiles >= ntiles[0] exit (the grid may be sized for the worst case).
struct Qmm8GuParams {
  uint K;
  uint N;          // columns per matrix (multiple of 64)
  uint w_stride;   // words between stacked experts
  uint s_stride;   // groups between stacked experts
  uint moe;        // 1: tiles/ntiles select the expert per grid row
  uint h_stride;   // halfs between tiles' output tables
  uint hs_stride;  // floats between tiles' output sums
};

// NP fragments (8 columns each) of gate and of up per simdgroup, 8 / NP column slices, KS k-slices:
// 8 / NP * KS simdgroups per threadgroup, simdgroup sg = column slice sg / KS, k-slice sg % KS (the
// k-slices' partials are summed in a fixed order before the epilogue, as qmm8's in-threadgroup split).
template <int PF, int NP, int KS, int CM = 0>  // PF: 0 = v8_run, 1 = prefetching, 2 = two groups per step
NI_INLINE void qmm8_gu_impl(const device uint32_t* Wg, const device ushort* Sg, const device ushort* Bg,
                            const device uint32_t* Wu, const device ushort* Su, const device ushort* Bu,
                            const device uint32_t* SWg, const device ushort* SSg, const device ushort* SBg,
                            const device uint32_t* SWu, const device ushort* SSu, const device ushort* SBu,
                            const device half* table, const device float* sums, device half* hT, device float* hS,
                            constant Qmm8GuParams& p, const device uint* tiles, const device uint* ntiles, uint2 tg,
                            uint sg, uint lane, threadgroup float* red, threadgroup float2* kred) {
  constexpr uint NSG = 8 / NP;
  uint wo = 0, so = 0;
  bool shared = false;
  if (p.moe) {
    if (tg.y >= ntiles[0]) return;
    const uint e = tiles[tg.y];
    shared = e == 0xffffffffu;
    if (!shared) {
      wo = e * p.w_stride;
      so = e * p.s_stride;
    }
  }
  const device uint32_t* G = (shared ? SWg : Wg) + wo;
  const device uint32_t* U = (shared ? SWu : Wu) + wo;
  const device ushort *Gs = (shared ? SSg : Sg) + so, *Gb = (shared ? SBg : Bg) + so;
  const device ushort *Us = (shared ? SSu : Su) + so, *Ub = (shared ? SBu : Bu) + so;
  const uint gpr = p.K / 64, wpr = p.K / 8;
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint ci = sg / KS, kslice = sg % KS;
  const uint c0 = tg.x * 64 + ci * 8 * NP;
  const uint gper = (gpr + KS - 1) / KS, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* wp[2 * NP];
  const device ushort* sp[2 * NP];
  const device ushort* bp[2 * NP];
  float2 acc[2 * NP];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NP; ++f) {
    const uint row = c0 + 8 * f + fm;
    wp[f] = G + row * wpr + 2 * q;
    sp[f] = Gs + row * gpr;
    bp[f] = Gb + row * gpr;
    wp[f + NP] = U + row * wpr + 2 * q;
    sp[f + NP] = Us + row * gpr;
    bp[f + NP] = Ub + row * gpr;
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 2 * NP; ++f) acc[f] = float2(0.0f);
  if (PF == 2) v8_run_pair<2 * NP, CM>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else if (PF) v8_run_pf<4, 2 * NP, false, 0, CM>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
  else v8_run<4, 2 * NP>(wp, sp, bp, table, sums, table, sums, g0, g1, lane, fn, acc);
  if (KS > 1) v8_reduce<2 * NP>(acc, kred, KS, sg, lane);  // every simdgroup passes the same barriers
  const bool lead = kslice == 0;                            // holds the column slice's totals
  device half* ht = hT + tg.y * p.h_stride;
  float2 ps = float2(0.0f);
  if (lead) {
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NP; ++f) {
      const uint kk = ci * 8 * NP + 8 * f + fm;
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const half h = half(silu(acc[f][e]) * acc[f + NP][e]);
        ht[v8_tab(tg.x, kk, fn + e)] = h;
        ps[e] += float(h);
      }
    }
  }
  // sum over the simdgroup's columns: lanes sharing fn differ in the fm bits (lane bits 1, 2, 4)
  ps += simd_shuffle_xor(ps, 2u);
  ps += simd_shuffle_xor(ps, 4u);
  ps += simd_shuffle_xor(ps, 16u);
  if (lead && fm == 0) {
    red[ci * kV8 + fn] = ps.x;
    red[ci * kV8 + fn + 1] = ps.y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < kV8) {
    float t = red[lane];
    for (uint i = 1; i < NSG; ++i) t += red[i * kV8 + lane];
    hS[tg.y * p.hs_stride + tg.x * kV8 + lane] = t;
  }
}

#define NI_QMM8_GU(NAME, PF, NP, KS, ...)                                                                       \
  [[kernel]] void NAME(const device uint32_t* Wg [[buffer(0)]], const device ushort* Sg [[buffer(1)]],          \
                       const device ushort* Bg [[buffer(2)]], const device uint32_t* Wu [[buffer(3)]],          \
                       const device ushort* Su [[buffer(4)]], const device ushort* Bu [[buffer(5)]],            \
                       const device uint32_t* SWg [[buffer(6)]], const device ushort* SSg [[buffer(7)]],        \
                       const device ushort* SBg [[buffer(8)]], const device uint32_t* SWu [[buffer(9)]],        \
                       const device ushort* SSu [[buffer(10)]], const device ushort* SBu [[buffer(11)]],        \
                       const device half* table [[buffer(12)]], const device float* sums [[buffer(13)]],        \
                       device half* hT [[buffer(14)]], device float* hS [[buffer(15)]],                         \
                       constant Qmm8GuParams& p [[buffer(16)]], const device uint* tiles [[buffer(17)]],        \
                       const device uint* ntiles [[buffer(18)]], uint2 tg [[threadgroup_position_in_grid]],     \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {   \
    threadgroup float red[8 * kV8];                                                                            \
    threadgroup float2 kred[KS > 1 ? (8 / NP) * KS * 2 * NP * 32 : 1];                                          \
    qmm8_gu_impl<PF, NP, KS, ##__VA_ARGS__>(Wg, Sg, Bg, Wu, Su, Bu, SWg, SSg, SBg, SWu, SSu, SBu, table, sums, hT, hS, p, \
                             tiles, ntiles, tg, sg, lane, red, kred);                                          \
  }
NI_QMM8_GU(qmm8_gu_b4, false, 2, 1)       // 4 simdgroups x (2 gate + 2 up fragments)
NI_QMM8_GU(qmm8_gu_pf_b4, true, 2, 1)
NI_QMM8_GU(qmm8_gu1_b4, false, 1, 1)      // 8 simdgroups x (1 gate + 1 up fragment)
NI_QMM8_GU(qmm8_gu1_pf_b4, true, 1, 1)
NI_QMM8_GU(qmm8_gu1k2_b4, true, 1, 2)     // + in-threadgroup K split (16 / 32 simdgroups)
NI_QMM8_GU(qmm8_gu1k4_b4, true, 1, 4)
NI_QMM8_GU(qmm8_gu2k2_b4, true, 2, 2)     // 4 column slices x 2 k-slices (8 simdgroups)
NI_QMM8_GU(qmm8_gu2k4_b4, true, 2, 4)
NI_QMM8_GU(qmm8_gu1k2sn_b4, true, 1, 2, 1)  // subnormal-code operands (CM 1)
NI_QMM8_GU(qmm8_gu1k2b_b4, 2, 1, 2, 0)      // two groups per step
NI_QMM8_GU(qmm8_gu1k2snb_b4, 2, 1, 2, 1)

// ---- MoE for 8 rows ----
// moe8_plan (one threadgroup, E threads; with topk, the router top-k first, one simdgroup per row): the distinct
// experts of the R rows' top-k (ascending ids). Experts used by more than `split` rows become MMA tiles 0 .. U-1 (all 8
// rows computed, the live ones stored), then the shared expert tile U (unless shs); slot[u * 8 + t] = the top-k slot
// of row t that routes to tile u's expert (0xffffffff: row t does not use it; the shared tile: slot k for t < R).
// Experts used by 1 .. small (<= 2) rows become one small tile s (the scalar passes: just those rows, no padded MMA
// work), experts with small < rows <= split ceil(rows / 2) of them (2 rows each), and with shs the shared expert
// ceil(R / 2) (expert 0xffffffff): sm[s] = expert, sm[72 + 2s + i] = row, sm[216 + 2s + i] = slot (0xffffffff past
// the tile's rows). sep puts the split experts' and the shared expert's tiles in a second list sm2 (greedy verifies:
// list A on moe8s2, list B on moe8e). ntiles = {MMA tiles, S (list A), S2 (list B)}. Writes the indirect grids of the
// six expert passes: {gu_gx, max(ntiles[0], 1), 1}, {dn_gx, ..}, {sgu_gx, max(S, 1), 1}, {sdn_gx, ..},
// {gu_gx, max(S2, 1), 1}, {edn_gx, ..}.
constant constexpr uint kV8SM = 72;  // small tiles at most: 64 routed (row, expert) pairs + the shared expert's 4
struct Moe8PlanParams {
  uint R, k, E;
  uint gu_gx;   // grid x of the gate/up pass (I / 64)
  uint dn_gx;   // grid x of the down pass
  uint layer;   // stats[layer] = distinct routed experts, for diagnostics
  uint small;   // experts with <= small rows take the scalar passes (0: none, max 2)
  uint sgu_gx, sdn_gx;
  uint rlim;    // 1: only rows < min(R, rows[0]) (a drafter's dynamic verify depth, a KFlash tree's live rows): the
                // router's top-k, the routed experts and the shared expert; later rows are dropped (garbage)
  uint split;   // experts with small < rows <= split become ceil(rows / 2) small tiles (>= small; MMA-exact passes)
  uint shs;     // 1: the shared expert as ceil(R / 2) small tiles (expert 0xffffffff) after the routed ones, no MMA tile
  uint sep;     // 1: the split experts' (and the shared expert's) tiles in list B (sm2, ntiles[2]: MMA-exact passes)
                // while list A has the 1-2-row experts (greedy: moe8s2)
  uint edn_gx;  // grid x of list B's down pass
  uint topk;    // 1: the router top-k first (topk256_row, one simdgroup per row; E == 256): no moe_topk256_rows pass
};

// n 2-row small tiles of expert e from s0 on (its rows in order; slots from my[t]) in tile list sm
NI_INLINE void v8_put_tiles(device uint* sm, uint s0, uint n, uint e, thread const uint (&my)[8]) {
  for (uint q = 0; q < n; ++q) sm[s0 + q] = e;
  uint i = 0;  // the expert's i-th row goes to tile s0 + i / 2, position i % 2
  for (uint t = 0; t < 8; ++t)
    if (my[t] != 0xffffffffu) {
      sm[kV8SM + 2 * s0 + i] = t;
      sm[3 * kV8SM + 2 * s0 + i] = my[t];
      ++i;
    }
  for (; i < 2 * n; ++i) {
    sm[kV8SM + 2 * s0 + i] = 0xffffffffu;
    sm[3 * kV8SM + 2 * s0 + i] = 0xffffffffu;
  }
}

[[kernel]] void moe8_plan(device uint* ridx [[buffer(0)]], device uint* tiles [[buffer(1)]],
                          device uint* ntiles [[buffer(2)]], device uint* slot [[buffer(3)]],
                          device uint* args [[buffer(4)]], constant Moe8PlanParams& p [[buffer(5)]],
                          device uint* stats [[buffer(6)]], device uint* sm [[buffer(7)]],
                          const device uint* rows_lim [[buffer(8)]], device uint* sm2 [[buffer(9)]],
                          const device float* rlog [[buffer(10)]], const device float* slog [[buffer(11)]],
                          device float* rwts [[buffer(12)]], constant TopkParams& tp [[buffer(13)]],
                          uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint ids[64];
  threadgroup uint cnt[4][32];
  threadgroup uint total[4];
  const uint R = p.rlim ? max(min(p.R, rows_lim[0]), 1u) : p.R;
  const uint n = R * p.k;
  if (p.topk) {
    if (sg < R) topk256_row(rlog, slog, ridx, rwts, tp, sg, lane, ids);
  } else if (tid < n) {
    ids[tid] = ridx[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // thread e: its slots per row (rows use each expert at most once)
  const uint e = tid;
  uint my[8];
  uint rows = 0;
  for (uint t = 0; t < 8; ++t) {
    my[t] = 0xffffffffu;
    if (t < R && e < p.E)
      for (uint j = 0; j < p.k; ++j)
        if (ids[t * p.k + j] == e) my[t] = j;
    rows += my[t] != 0xffffffffu ? 1u : 0u;
  }
  // small tiles: 1 for rows <= small; ceil(rows / 2) for small < rows <= split (MMA-exact passes), in list A or (sep)
  // list B; the rest MMA tiles
  const bool spl = rows > p.small && rows <= p.split;
  const uint na = rows > 0 && rows <= p.small ? 1u : spl && !p.sep ? (rows + 1) / 2 : 0u;
  const uint nb = spl && p.sep ? (rows + 1) / 2 : 0u;
  const bool mma = rows > 0 && na == 0 && nb == 0;
  const uint pre_m = simd_prefix_exclusive_sum(mma ? 1u : 0u);
  const uint pre_a = simd_prefix_exclusive_sum(na);
  const uint pre_b = simd_prefix_exclusive_sum(nb);
  const uint used = simd_sum(rows > 0 ? 1u : 0u);  // distinct experts (stats)
  if (lane == 31) {
    cnt[0][sg] = pre_m + (mma ? 1u : 0u);
    cnt[1][sg] = pre_a + na;
    cnt[2][sg] = pre_b + nb;
    cnt[3][sg] = used;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4) {
    uint a = 0;
    for (uint i = 0; i < (ntg + 31) / 32; ++i) {
      const uint c = cnt[tid][i];
      cnt[tid][i] = a;
      a += c;
    }
    total[tid] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (mma) {
    const uint u = cnt[0][sg] + pre_m;
    tiles[u] = e;
    for (uint t = 0; t < 8; ++t) slot[u * 8 + t] = my[t];
  }
  if (na) v8_put_tiles(sm, cnt[1][sg] + pre_a, na, e, my);
  if (nb) v8_put_tiles(sm2, cnt[2][sg] + pre_b, nb, e, my);
  const uint U = total[0], nsh = p.shs ? (R + 1) / 2 : 0u;
  const uint SA = total[1] + (p.shs && !p.sep ? nsh : 0u), SB = total[2] + (p.shs && p.sep ? nsh : 0u);
  if (p.shs) {  // the shared expert: rows 0 .. R - 1 (its slot: k), expert 0xffffffff
    device uint* l = p.sep ? sm2 : sm;
    const uint s0 = p.sep ? total[2] : total[1];
    if (tid < nsh) l[s0 + tid] = 0xffffffffu;
    if (tid < 2 * nsh) {
      l[kV8SM + 2 * s0 + tid] = tid < R ? tid : 0xffffffffu;
      l[3 * kV8SM + 2 * s0 + tid] = tid < R ? p.k : 0xffffffffu;
    }
  } else if (tid < 8) {
    slot[U * 8 + tid] = tid < R ? p.k : 0xffffffffu;
  }
  if (tid == 0) {
    if (!p.shs) tiles[U] = 0xffffffffu;
    const uint U1 = p.shs ? U : U + 1;
    ntiles[0] = U1;
    ntiles[1] = SA;
    ntiles[2] = SB;
    stats[p.layer] = total[3];
    args[0] = p.gu_gx; args[1] = max(U1, 1u); args[2] = 1;
    args[3] = p.dn_gx; args[4] = max(U1, 1u); args[5] = 1;
    args[6] = p.sgu_gx; args[7] = max(SA, 1u); args[8] = 1;
    args[9] = p.sdn_gx; args[10] = max(SA, 1u); args[11] = 1;
    args[12] = p.gu_gx; args[13] = max(SB, 1u); args[14] = 1;  // list B: moe8e_gu_r2 (I / 64), moe8e_down_r2
    args[15] = p.edn_gx; args[16] = max(SB, 1u); args[17] = 1;
  }
}
// ---- scalar expert passes for experts used by 1-2 rows ----
struct Moe8sParams {
  uint K, N;            // input width, output columns
  uint w_stride, s_stride;
  uint k;
  uint x_ld;            // floats per input row
};

// Expert down projections: tile u reads its h table (h_stride / hs_stride apart) and writes row t's
// output to out[(t * k + slot) * N] for the rows that route to it (shared tile: sh_out[t * N]).
// grid (N / (8 NF cs), tiles), cs = simdgroups per threadgroup.
struct Moe8DownParams {
  uint K;          // expert intermediate (input width)
  uint N;          // hidden (output width)
  uint w_stride, s_stride;
  uint h_stride, hs_stride;
  uint k;
};

template <int NF>
NI_INLINE void moe8_down_impl(const device uint32_t* W, const device ushort* S, const device ushort* B,
                              const device uint32_t* SW, const device ushort* SS, const device ushort* SB,
                              const device half* hT, const device float* hS, device float* out,
                              device float* sh_out, constant Moe8DownParams& p, const device uint* tiles,
                              const device uint* ntiles, const device uint* slot, uint2 tg, uint sg, uint nsg,
                              uint lane) {
  if (tg.y >= ntiles[0]) return;
  const uint e = tiles[tg.y];
  const bool shared = e == 0xffffffffu;
  const device uint32_t* Wb = shared ? SW : W + e * p.w_stride;
  const device ushort* Sb = shared ? SS : S + e * p.s_stride;
  const device ushort* Bb = shared ? SB : B + e * p.s_stride;
  const uint gpr = p.K / 64, wpr = p.K / 8;
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint n0 = (tg.x * nsg + sg) * 8 * NF;
  const device uint32_t* wp[NF];
  const device ushort* sp[NF];
  const device ushort* bp[NF];
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint row = n0 + 8 * f + fm;
    wp[f] = Wb + row * wpr + 2 * q;
    sp[f] = Sb + row * gpr;
    bp[f] = Bb + row * gpr;
    acc[f] = float2(0.0f);
  }
  const device half* ht = hT + tg.y * p.h_stride;
  const device float* hs = hS + tg.y * p.hs_stride;
  v8_run<4, NF>(wp, sp, bp, ht, hs, ht, hs, 0, gpr, lane, fn, acc);
  _Pragma("clang loop unroll(full)")
  for (int e2 = 0; e2 < 2; ++e2) {
    const uint t = fn + e2, j = slot[tg.y * 8 + t];
    if (j == 0xffffffffu) continue;
    device float* o = shared ? sh_out + t * p.N : out + (t * p.k + j) * p.N;
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) o[n0 + 8 * f + fm] = acc[f][e2];
  }
}

#define NI_MOE8_DOWN(NAME, NF)                                                                                  \
  [[kernel]] void NAME(const device uint32_t* W [[buffer(0)]], const device ushort* S [[buffer(1)]],           \
                       const device ushort* B [[buffer(2)]], const device uint32_t* SW [[buffer(3)]],          \
                       const device ushort* SS [[buffer(4)]], const device ushort* SB [[buffer(5)]],           \
                       const device half* hT [[buffer(6)]], const device float* hS [[buffer(7)]],              \
                       device float* out [[buffer(8)]], device float* sh_out [[buffer(9)]],                    \
                       constant Moe8DownParams& p [[buffer(10)]], const device uint* tiles [[buffer(11)]],     \
                       const device uint* ntiles [[buffer(12)]], const device uint* slot [[buffer(13)]],       \
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {     \
    moe8_down_impl<NF>(W, S, B, SW, SS, SB, hT, hS, out, sh_out, p, tiles, ntiles, slot, tg, sg, nsg, lane);   \
  }
NI_MOE8_DOWN(moe8_down_nf1, 1)
NI_MOE8_DOWN(moe8_down_nf2, 2)

// x[t] += sum_j w[t][j] * out[t * k + j] + w[t][k] * sh_out[t]   (slot order, then the shared expert).
// grid (N / 256, R) x 64 threads (float4 per thread).
struct Moe8CombineParams {
  uint N, k;
};
[[kernel]] void moe8_combine(const device float4* out [[buffer(0)]], const device float4* sh_out [[buffer(1)]],
                             const device float* wts [[buffer(2)]], device float4* x [[buffer(3)]],
                             constant Moe8CombineParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint t = tg.y, i = tg.x * 64 + tid2.x, n4 = p.N / 4;
  if (i >= n4) return;
  float4 acc = float4(0.0f);
  for (uint j = 0; j < p.k; ++j) acc += wts[t * (p.k + 1) + j] * out[(t * p.k + j) * n4 + i];
  acc += wts[t * (p.k + 1) + p.k] * sh_out[t * n4 + i];
  x[t * n4 + i] += acc;
}

// The tile's (<= 2) input rows are staged once per threadgroup in threadgroup memory and each simdgroup
// produces ROWS output columns from register-dequantized weights (AND + convert per value, one FMA per
// value and row, x pre-scaled for the shift-free nibbles as qmv2_rows), so a lightly used expert costs
// its bytes, not an 8-row MMA tile. (A first version read the inputs from device memory per 2 columns,
// 4x the weight traffic, and was slower than the MMA tiles.)
template <int ROWS>
NI_INLINE void v8s_dot2t(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         threadgroup const float* x0, threadgroup const float* x1, uint K, uint row0, uint lane,
                         thread float (&acc)[ROWS][2]) {
  constexpr int VPL = 16;
  const uint wpr = K / 8, gpr = K / 64;
  for (int r = 0; r < ROWS; ++r) acc[r][0] = acc[r][1] = 0.0f;
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[2][VPL];
    float xs[2];
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < 2; ++t) {
      threadgroup const float* xr = t ? x1 : x0;
      xs[t] = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < VPL / 4; ++i) {
        const float4 v = ((threadgroup const float4*)(xr + k0))[i];
        xs[t] += (v.x + v.y) + (v.z + v.w);
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (4 * ((i * 4 + e) % 8))));
      }
    }
    const uint g = k0 / 64, wo = k0 / 8;
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < ROWS; ++r) {
      const uint row = row0 + r;
      const uint2 wv = *(const device uint2*)(w + row * wpr + wo);
      float d0 = 0.0f, d1 = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 16; ++j) {
        const float q = float((j < 8 ? wv.x : wv.y) & (15u << (4 * (j % 8))));
        d0 = fma(q, xv[0][j], d0);
        d1 = fma(q, xv[1][j], d1);
      }
      const float s = float(scales[row * gpr + g]), b = float(biases[row * gpr + g]);
      acc[r][0] += s * d0 + b * xs[0];
      acc[r][1] += s * d1 + b * xs[1];
    }
  }
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < ROWS; ++r) {
    acc[r][0] = simd_sum(acc[r][0]);
    acc[r][1] = simd_sum(acc[r][1]);
  }
}

// gate/up: grid (N / (4 * nsg), max(S, 1)); x rows staged (K <= 2048).
[[kernel]] void moe8s2_gu(const device uint32_t* Wg [[buffer(0)]], const device bfloat* Sg [[buffer(1)]],
                          const device bfloat* Bg [[buffer(2)]], const device uint32_t* Wu [[buffer(3)]],
                          const device bfloat* Su [[buffer(4)]], const device bfloat* Bu [[buffer(5)]],
                          const device float* xn [[buffer(6)]], device float* hs [[buffer(7)]],
                          constant Moe8sParams& p [[buffer(8)]], const device uint* sm [[buffer(9)]],
                          const device uint* ntiles [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[2 * 2048];
  if (tg.y >= ntiles[1]) return;
  const uint e = sm[tg.y], t0 = sm[kV8SM + 2 * tg.y], t1r = sm[kV8SM + 2 * tg.y + 1];
  const bool two = t1r != 0xffffffffu;
  const uint t1 = two ? t1r : t0;
  for (uint i = tid; i < p.K / 4; i += 32 * nsg) {
    ((threadgroup float4*)xs)[i] = ((const device float4*)(xn + t0 * p.x_ld))[i];
    ((threadgroup float4*)(xs + p.K))[i] = ((const device float4*)(xn + t1 * p.x_ld))[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int ROWS = 4;
  const uint row0 = (tg.x * nsg + sg) * ROWS;
  if (row0 >= p.N) return;
  float ag[ROWS][2], au[ROWS][2];
  v8s_dot2t<ROWS>(Wg + e * p.w_stride, Sg + e * p.s_stride, Bg + e * p.s_stride, xs, xs + p.K, p.K, row0, lane, ag);
  v8s_dot2t<ROWS>(Wu + e * p.w_stride, Su + e * p.s_stride, Bu + e * p.s_stride, xs, xs + p.K, p.K, row0, lane, au);
  if (lane < ROWS) {
    float g0 = ag[0][0], u0 = au[0][0], g1 = ag[0][1], u1 = au[0][1];
    for (int r = 1; r < ROWS; ++r)
      if (uint(r) == lane) { g0 = ag[r][0]; u0 = au[r][0]; g1 = ag[r][1]; u1 = au[r][1]; }
    hs[(2 * tg.y) * p.N + row0 + lane] = silu(g0) * u0;
    if (two) hs[(2 * tg.y + 1) * p.N + row0 + lane] = silu(g1) * u1;
  }
}

// down: grid (N / (4 * nsg), max(S, 1)); h rows staged (K <= 2048).
[[kernel]] void moe8s2_down(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],
                            const device bfloat* B [[buffer(2)]], const device float* hs [[buffer(3)]],
                            device float* out [[buffer(4)]], constant Moe8sParams& p [[buffer(5)]],
                            const device uint* sm [[buffer(6)]], const device uint* ntiles [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[2 * 2048];
  if (tg.y >= ntiles[1]) return;
  const uint e = sm[tg.y];
  const uint t0 = sm[kV8SM + 2 * tg.y], t1r = sm[kV8SM + 2 * tg.y + 1];
  const uint j0 = sm[3 * kV8SM + 2 * tg.y], j1 = sm[3 * kV8SM + 2 * tg.y + 1];
  const bool two = t1r != 0xffffffffu;
  for (uint i = tid; i < p.K / 4; i += 32 * nsg) {
    ((threadgroup float4*)xs)[i] = ((const device float4*)(hs + (2 * tg.y) * p.K))[i];
    ((threadgroup float4*)(xs + p.K))[i] = ((const device float4*)(hs + (2 * tg.y + (two ? 1 : 0)) * p.K))[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int ROWS = 4;
  const uint row0 = (tg.x * nsg + sg) * ROWS;
  if (row0 >= p.N) return;
  float a[ROWS][2];
  v8s_dot2t<ROWS>(W + e * p.w_stride, S + e * p.s_stride, B + e * p.s_stride, xs, xs + p.K, p.K, row0, lane, a);
  if (lane < ROWS) {
    float a0 = a[0][0], a1 = a[0][1];
    for (int r = 1; r < ROWS; ++r)
      if (uint(r) == lane) { a0 = a[r][0]; a1 = a[r][1]; }
    out[(t0 * p.k + j0) * p.N + row0 + lane] = a0;
    if (two) out[(t1r * p.k + j1) * p.N + row0 + lane] = a1;
  }
}

// ---- MMA-exact scalar passes (moe8e) for 2-row tiles ----
// A sampled (T > 0) verify needs row-independent MoE numerics: a row's expert outputs may not depend on which other
// rows route to the same expert, so it cannot mix the scalar passes above (fp32 inputs, their own sums) with the MMA
// tiles. These passes give the 1-2 rows of a small tile bitwise the values an 8-row MMA tile gives them, at about the
// scalar passes' cost, so sampled verifies run every expert as 2-row tiles (ceil(rows / 2) per expert, the shared
// expert too: the tiles of one expert run side by side and its re-reads come from the caches; BENCH_REPLAY: 51 tiles
// of 39 experts took what 39 distinct tiles take).
// simdgroup_multiply_accumulate (half x half + float, 8x8x8) adds an output's 8 products (exact in fp32) to C one at a
// time in k order, rounding to fp32 after each (tools/mma_probe: every one of 2^19 random outputs on the M5 Max;
// mma_chain_check below re-checks it on the GPU at hand before these passes are used). A tile's 64-group of one
// (output, row) is then a chain of 64 fp32 FMAs in the order of its 8 MMAs (m) and their k-slots (s):
//   step 8 m + s adds code(k) * x[k],  k = 16 (s / 2) + 8 (m / 4) + (m % 4) + 4 (s % 2)
// (word 2 (s / 2) + m / 4 of the group, nibble (m % 4) + 4 (s % 2)), followed by the tile kernel's group epilogue.
// Lanes run one group's chains each (the gate/up pass: the 32 lanes are the 32 groups of a 2048-wide row, 1 KB
// coalesced weight reads); the epilogue chains over the groups then run one per lane from threadgroup memory.
NI_INLINE constexpr uint v8e_word(uint m, uint s) { return 2 * (s / 2) + m / 4; }
NI_INLINE constexpr uint v8e_nib(uint m, uint s) { return (m % 4) + 4 * (s % 2); }
NI_INLINE constexpr uint v8e_k(uint m, uint s) { return 8 * v8e_word(m, s) + v8e_nib(m, s); }

struct Moe8eParams {
  uint K, N;  // input width, output columns
  uint w_stride, s_stride;
  uint k;     // top-k (down: the output slots)
};

// The passes over tiles of up to RT rows. RT 2: moe8_plan's small tiles. RT 4 compiles too (4 simdgroups per gate/up
// threadgroup, for its threadgroup memory) but 4-row tiles ran slower than two 2-row tiles each (35B, 64 tokens
// teacher-forced, sampled-verify numerics: 11.98 vs 10.81 ms per verify: too few threadgroups for the few such
// experts). Measured and dropped for the 2-row passes: 4 or 16 simdgroups per gate/up threadgroup (+0.08, +2.1 ms
// per verify), the next pass's weights requested before the current epilogue (+0.03 to +0.08), a persistent grid of
// 120-320 threadgroups (+0.14 to +0.25), 8 simdgroups per down threadgroup (+0.1), 512-byte coalesced weight loads
// with lane-pair swaps (+0.03). Tile list (MT tiles at most): sm[s] = expert,
// sm[MT + RT s + i] = row i, sm[MT + MT RT + RT s + i] = its top-k slot (0xffffffff past the tile's rows, which come
// first). h and its group sums per tile row: hs[(RT s + i) * N], hsum[(RT s + i) * N / 64].
NI_INLINE constexpr uint v8e_mt(uint rt) { return rt == 2 ? kV8SM : 32; }
NI_INLINE constexpr uint v8e_lrt(uint rt) { return rt == 2 ? 1 : 2; }  // log2(RT)
// the tile's rows (dead entries: row 0) and live row count
template <uint RT>
NI_INLINE uint v8e_tile_rows(const device uint* sm, uint s, thread uint (&t)[RT]) {
  constexpr uint MT = v8e_mt(RT);
  uint n = 0;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < RT; ++i) {
    t[i] = sm[MT + RT * s + i];
    n += t[i] != 0xffffffffu ? 1u : 0u;
  }
  _Pragma("clang loop unroll(full)")
  for (uint i = 1; i < RT; ++i) t[i] = i < n ? t[i] : t[0];
  return n;
}

// gate/up + SwiGLU: grid (N / 64, max(tiles, 1)) x 32 NSG threads; simdgroup sg owns columns 64 / NSG sg ..
// (the tile kernel's column slices of 8), 2 per pass: gate and up rows of both = 4 weight rows, 4 RT chains per lane.
template <uint RT, uint NSG>
NI_INLINE void moe8e_gu_t(const device uint32_t* Wg, const device ushort* Sg, const device ushort* Bg,
                          const device uint32_t* Wu, const device ushort* Su, const device ushort* Bu,
                          const device half* table, const device float* sums, device float* hs, device float* hsum,
                          constant Moe8eParams& p, const device uint* sm, const device uint* ntiles,
                          const device uint32_t* SWg, const device ushort* SSg, const device ushort* SBg,
                          const device uint32_t* SWu, const device ushort* SSu, const device ushort* SBu, uint2 tg, uint tid,
                          uint sg, uint lane, threadgroup vec<half, RT>* xa, threadgroup vec<half, RT>* xb,
                          threadgroup float* dots, threadgroup float* hv, threadgroup float* red,
                          threadgroup float* tsx) {
  constexpr uint CPS = 64 / NSG, P = CPS / 2, LRT = v8e_lrt(RT);
  constexpr uint gpr = 32, wpr = 256;  // K == 2048: lane = group
  if (tg.y >= ntiles[RT == 2 ? 1 : 2]) return;
  const uint e = sm[tg.y];
  uint t[RT];
  const uint nr = v8e_tile_rows<RT>(sm, tg.y, t);
  if (tid < 32)
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < RT; ++r) tsx[r * 32 + tid] = sums[tid * kV8 + t[r]];
  for (uint i = tid; i < 32 * 32; i += 32 * NSG) {  // [m * 4 + slot pair j][group]: steps 2j (xa), 2j + 1 (xb)
    const uint sp = i / 32, g = i % 32, m = sp / 4, j = sp % 4, ka = v8e_k(m, 2 * j), kb = v8e_k(m, 2 * j + 1);
    vec<half, RT> a, b;
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < RT; ++r) {
      a[r] = table[v8_tab(g, ka, t[r])];
      b[r] = table[v8_tab(g, kb, t[r])];
    }
    xa[i] = a;
    xb[i] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool shr = e == 0xffffffffu;  // the shared expert (its own weights)
  const device uint32_t* G = shr ? SWg : Wg + e * p.w_stride;
  const device uint32_t* U = shr ? SWu : Wu + e * p.w_stride;
  const device ushort *Gs = shr ? SSg : Sg + e * p.s_stride, *Gb = shr ? SBg : Bg + e * p.s_stride;
  const device ushort *Us = shr ? SSu : Su + e * p.s_stride, *Ub = shr ? SBu : Bu + e * p.s_stride;
  threadgroup float* dt = dots + sg * (2 * 2 * RT * 32);
  // epilogue lane: (column, gate/up, row, k-slice of the tile kernel's 2-way K split)
  const uint eslice = lane & 1, etrow = (lane >> 1) & (RT - 1), emat = (lane >> (1 + LRT)) & 1,
             ecol = (lane >> (2 + LRT)) & 1;
  for (uint pp = 0; pp < P; ++pp) {
    const uint c0 = tg.x * 64 + sg * CPS + pp * 2;
    uint4 w[4][2];  // gate c0, gate c0 + 1, up c0, up c0 + 1: this lane's group
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; ++r) {
      const device uint4* src = (const device uint4*)((r < 2 ? G : U) + (c0 + (r & 1)) * wpr + lane * 8);
      w[r][0] = src[0];
      w[r][1] = src[1];
    }
    // the epilogue lane's 16 scales and biases, requested ahead of the chains
    const uint en = c0 + ecol;
    const device uint4* esp = (const device uint4*)((emat ? Us : Gs) + en * gpr + eslice * 16);
    const device uint4* ebp = (const device uint4*)((emat ? Ub : Gb) + en * gpr + eslice * 16);
    const uint4 es0 = esp[0], es1 = esp[1], eb0 = ebp[0], eb1 = ebp[1];
    float d[4][RT];
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; ++r)
      _Pragma("clang loop unroll(full)")
      for (uint q = 0; q < RT; ++q) d[r][q] = 0.0f;
    // MMAs 0-3 take the group's even words, 4-7 the odd ones; a code pair (one shift + AND) is slots 2j, 2j + 1
    // (a rolled loop: the fully unrolled 32 steps made 3x the code and ran slower)
    for (uint mh = 0; mh < 2; ++mh) {
      uint4 wv[4];
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; ++r)
        wv[r] = mh ? uint4(w[r][0].y, w[r][0].w, w[r][1].y, w[r][1].w) : uint4(w[r][0].x, w[r][0].z, w[r][1].x, w[r][1].z);
      for (uint m4 = 0; m4 < 4; ++m4) {
        const uint m = mh * 4 + m4, sh = 4 * m4;
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) {
          const vec<half, RT> x0 = xa[(m * 4 + j) * 32 + lane], x1 = xb[(m * 4 + j) * 32 + lane];
          _Pragma("clang loop unroll(full)")
          for (int r = 0; r < 4; ++r) {
            const half2 c = as_type<half2>((wv[r][j] >> sh) & 0x000F000Fu);
            _Pragma("clang loop unroll(full)")
            for (uint q = 0; q < RT; ++q) d[r][q] = fma(float(c.x), float(x0[q]), d[r][q]);
            _Pragma("clang loop unroll(full)")
            for (uint q = 0; q < RT; ++q) d[r][q] = fma(float(c.y), float(x1[q]), d[r][q]);
          }
        }
      }
    }
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; ++r)
      _Pragma("clang loop unroll(full)")
      for (uint q = 0; q < RT; ++q) dt[(((r & 1) * 2 + (r >> 1)) * RT + q) * 32 + lane] = d[r][q];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
      const threadgroup float* dd = dt + ((ecol * 2 + emat) * RT + etrow) * 32 + eslice * 16;
      const threadgroup float* sx = tsx + etrow * 32 + eslice * 16;
      float acc = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < 16; ++i) {
        const uint sw = (i < 8 ? es0 : es1)[(i % 8) / 2], bw = (i < 8 ? eb0 : eb1)[(i % 8) / 2];
        const ushort sb = ushort(i & 1 ? sw >> 16 : sw & 0xffffu), bb = ushort(i & 1 ? bw >> 16 : bw & 0xffffu);
        acc = fma(dd[i], v8_bf(sb) * kV8Sub, acc);
        acc = fma(sx[i], v8_bf(bb), acc);
      }
      acc += simd_shuffle_xor(acc, 1u);  // the k-slices' sum (v8_reduce)
      const float o = simd_shuffle_xor(acc, 1u << (1 + LRT));
      const half h = half(silu(emat ? o : acc) * (emat ? acc : o));
      if (lane < 8 * RT && emat == 0 && eslice == 0) {
        hv[(sg * CPS + pp * 2 + ecol) * RT + etrow] = float(h);
        if (etrow < nr) hs[(RT * tg.y + etrow) * p.N + en] = float(h);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {  // group sums of h: lane = (column fm of an 8-column slice, (slice, row) combo): the tile kernel's butterfly
    const uint2 rc = v8_lane(lane);
    const uint fm = rc.x, cl = rc.y >> 1;
    constexpr uint NC = CPS / 8 * RT;  // (slice, row) combos of this simdgroup
    _Pragma("clang loop unroll(full)")
    for (uint c0 = 0; c0 < NC; c0 += 4) {
      const uint c = c0 + cl, ci = c / RT, trow = c % RT;
      float ps = c < NC ? hv[(sg * CPS + ci * 8 + fm) * RT + trow] : 0.0f;
      ps += simd_shuffle_xor(ps, 2u);
      ps += simd_shuffle_xor(ps, 4u);
      ps += simd_shuffle_xor(ps, 16u);
      if (fm == 0 && c < NC) red[(sg * (CPS / 8) + ci) * kV8 + trow] = ps;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < RT) {
    float s = red[lane];
    for (uint i = 1; i < 8; ++i) s += red[i * kV8 + lane];
    if (lane < nr) hsum[(RT * tg.y + lane) * (p.N / 64) + tg.x] = s;
  }
}
#define NI_MOE8E_GU_T(NAME, RT, NSG)                                                                            \
  [[kernel]] void NAME(const device uint32_t* Wg [[buffer(0)]], const device ushort* Sg [[buffer(1)]],          \
                       const device ushort* Bg [[buffer(2)]], const device uint32_t* Wu [[buffer(3)]],          \
                       const device ushort* Su [[buffer(4)]], const device ushort* Bu [[buffer(5)]],            \
                       const device half* table [[buffer(6)]], const device float* sums [[buffer(7)]],          \
                       device float* hs [[buffer(8)]], device float* hsum [[buffer(9)]],                        \
                       constant Moe8eParams& p [[buffer(10)]], const device uint* sm [[buffer(11)]],            \
                       const device uint* ntiles [[buffer(12)]], const device uint32_t* SWg [[buffer(13)]],    \
                       const device ushort* SSg [[buffer(14)]], const device ushort* SBg [[buffer(15)]],        \
                       const device uint32_t* SWu [[buffer(16)]], const device ushort* SSu [[buffer(17)]],      \
                       const device ushort* SBu [[buffer(18)]], uint2 tg [[threadgroup_position_in_grid]],      \
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],    \
                       uint lane [[thread_index_in_simdgroup]]) {                                               \
    threadgroup vec<half, RT> xa[32 * 32], xb[32 * 32];                                                         \
    threadgroup float dots[NSG * 2 * 2 * RT * 32];                                                              \
    threadgroup float hv[64 * RT];                                                                              \
    threadgroup float red[8 * kV8];                                                                             \
    threadgroup float tsx[RT * 32];                                                                             \
    moe8e_gu_t<RT, NSG>(Wg, Sg, Bg, Wu, Su, Bu, table, sums, hs, hsum, p, sm, ntiles, SWg, SSg, SBg, SWu, SSu, SBu, tg,  \
                        tid, sg, lane, xa, xb, dots, hv, red, tsx);                                             \
  }
NI_MOE8E_GU_T(moe8e_gu_r2, 2, 8)

// down (K == 512): grid (N / 128, max(tiles, 1)) x 128 threads; lane = (column c of 4, group g of 8), columns
// n0 + 4 i + c (i < 4) per pass (4 RT chains), two passes of 16 columns per simdgroup. Codes as 1024 + q (the down
// tile's CM 0 operands), inputs h from moe8e_gu_t; writes out[(t * k + slot) * N] as moe8_down does.
template <uint RT>
NI_INLINE void moe8e_down_t(const device uint32_t* W, const device ushort* S, const device ushort* B,
                            const device float* hs, const device float* hsum, device float* out,
                            constant Moe8eParams& p, const device uint* sm, const device uint* ntiles,
                            const device uint32_t* SW, const device ushort* SS, const device ushort* SB,
                            device float* sh_out, uint2 tg,
                            uint tid, uint sg, uint lane, threadgroup vec<half, RT>* xa, threadgroup vec<half, RT>* xb,
                            threadgroup float* dots, threadgroup float* tsx) {
  constexpr uint MT = v8e_mt(RT), gpr = 8, wpr = 64, NSG = 4;
  if (tg.y >= ntiles[RT == 2 ? 1 : 2]) return;
  const uint e = sm[tg.y];
  uint t[RT], jj[RT];
  const uint nr = v8e_tile_rows<RT>(sm, tg.y, t);
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < RT; ++i) jj[i] = sm[MT + MT * RT + RT * tg.y + i];
  if (tid < 8)
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < RT; ++r) tsx[r * 8 + tid] = hsum[(RT * tg.y + (r < nr ? r : 0)) * gpr + tid];
  for (uint i = tid; i < 32 * 8; i += 32 * NSG) {
    const uint sp = i / 8, g = i % 8, m = sp / 4, j = sp % 4;
    const uint ka = 64 * g + v8e_k(m, 2 * j), kb = 64 * g + v8e_k(m, 2 * j + 1);
    vec<half, RT> a, b;
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < RT; ++r) {
      const device float* hr = hs + (RT * tg.y + (r < nr ? r : 0)) * p.K;
      a[r] = half(hr[ka]);
      b[r] = half(hr[kb]);
    }
    xa[i] = a;
    xb[i] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool shr = e == 0xffffffffu;  // the shared expert: its own weights, outputs to sh_out[t * N]
  const device uint32_t* Wb = shr ? SW : W + e * p.w_stride;
  const device ushort* Sb = shr ? SS : S + e * p.s_stride;
  const device ushort* Bb = shr ? SB : B + e * p.s_stride;
  const uint c4 = lane >> 3, g = lane & 7;
  threadgroup float* dt = dots + sg * (16 * RT * 8);
  for (uint pp = 0; pp < 2; ++pp) {
    const uint n0 = tg.x * 128 + sg * 32 + pp * 16;
    uint4 w[4][2];
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 4; ++c) {
      const device uint4* src = (const device uint4*)(Wb + (n0 + 4 * c + c4) * wpr + g * 8);
      w[c][0] = src[0];
      w[c][1] = src[1];
    }
    const uint en = n0 + (lane >> 1);  // the epilogue lane's column: its 8 scales and biases, requested ahead
    const uint4 es = *(const device uint4*)(Sb + en * gpr), eb = *(const device uint4*)(Bb + en * gpr);
    float d[4][RT];
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 4; ++c)
      _Pragma("clang loop unroll(full)")
      for (uint q = 0; q < RT; ++q) d[c][q] = 0.0f;
    for (uint mh = 0; mh < 2; ++mh) {
      uint4 wv[4];
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < 4; ++c)
        wv[c] = mh ? uint4(w[c][0].y, w[c][0].w, w[c][1].y, w[c][1].w) : uint4(w[c][0].x, w[c][0].z, w[c][1].x, w[c][1].z);
      for (uint m4 = 0; m4 < 4; ++m4) {
        const uint m = mh * 4 + m4, sh = 4 * m4;
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) {
          const vec<half, RT> x0 = xa[(m * 4 + j) * 8 + g], x1 = xb[(m * 4 + j) * 8 + g];
          _Pragma("clang loop unroll(full)")
          for (int c = 0; c < 4; ++c) {
            const half2 a = as_type<half2>(((wv[c][j] >> sh) & 0x000F000Fu) | 0x64006400u);
            _Pragma("clang loop unroll(full)")
            for (uint q = 0; q < RT; ++q) d[c][q] = fma(float(a.x), float(x0[q]), d[c][q]);
            _Pragma("clang loop unroll(full)")
            for (uint q = 0; q < RT; ++q) d[c][q] = fma(float(a.y), float(x1[q]), d[c][q]);
          }
        }
      }
    }
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 4; ++c)
      _Pragma("clang loop unroll(full)")
      for (uint q = 0; q < RT; ++q) dt[((4 * c + c4) * RT + q) * 8 + g] = d[c][q];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < RT / 2; ++u) {  // epilogue: lane = (column, row parity), rows lane & 1, + 2, ...
      const uint col = lane >> 1, trow = (lane & 1) + 2 * u;
      const threadgroup float* dd = dt + (col * RT + trow) * 8;
      const threadgroup float* hsr = tsx + trow * 8;
      float acc = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint gi = 0; gi < gpr; ++gi) {
        const uint sw = es[gi / 2], bw = eb[gi / 2];
        const float s = v8_bf(ushort(gi & 1 ? sw >> 16 : sw & 0xffffu)), bi = v8_bf(ushort(gi & 1 ? bw >> 16 : bw & 0xffffu));
        acc = fma(dd[gi], s, acc);
        acc = fma(hsr[gi], fma(-1024.0f, s, bi), acc);
      }
      const uint tr = (lane & 1) ? t[1 + 2 * u] : t[2 * u], js = (lane & 1) ? jj[1 + 2 * u] : jj[2 * u];
      if (trow < nr) (shr ? sh_out + tr * p.N : out + (tr * p.k + js) * p.N)[en] = acc;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
#define NI_MOE8E_DOWN_T(NAME, RT)                                                                               \
  [[kernel]] void NAME(const device uint32_t* W [[buffer(0)]], const device ushort* S [[buffer(1)]],           \
                       const device ushort* B [[buffer(2)]], const device float* hs [[buffer(3)]],             \
                       const device float* hsum [[buffer(4)]], device float* out [[buffer(5)]],                \
                       constant Moe8eParams& p [[buffer(6)]], const device uint* sm [[buffer(7)]],             \
                       const device uint* ntiles [[buffer(8)]], const device uint32_t* SW [[buffer(9)]],       \
                       const device ushort* SS [[buffer(10)]], const device ushort* SB [[buffer(11)]],         \
                       device float* sh_out [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],         \
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup vec<half, RT> xa[32 * 8], xb[32 * 8];                                                          \
    threadgroup float dots[4 * 16 * RT * 8];                                                                   \
    threadgroup float tsx[RT * 8];                                                                             \
    moe8e_down_t<RT>(W, S, B, hs, hsum, out, p, sm, ntiles, SW, SS, SB, sh_out, tg, tid, sg, lane, xa, xb, dots, tsx); \
  }
NI_MOE8E_DOWN_T(moe8e_down_r2, 2)

// GPU self-check of the MMA rounding the moe8e passes rely on: grid (64) x 32 threads; threadgroup i chains two
// MMAs on pseudo-random operands (A: 4-bit codes as subnormal halves for even i, as 1024 + q for odd i; B: halves
// over a wide exponent range; C: fp32) and compares each element with the 16-step fp32 FMA chain; mismatches are
// counted into bad[0].
NI_INLINE uint v8e_hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
NI_INLINE uint v8e_q(uint i, uint mm, uint r, uint c) { return v8e_hash(i * 977u + mm * 131u + r * 8u + c) & 15u; }
NI_INLINE half v8e_a(uint i, uint q) { return (i & 1) ? half(1024.0h + half(q)) : as_type<half>(ushort(q)); }
NI_INLINE float v8e_af(uint i, uint q) { return (i & 1) ? float(1024u + q) : float(q) * as_type<float>((127u - 24u) << 23); }
NI_INLINE half v8e_b(uint i, uint mm, uint r, uint c) {
  const uint h = v8e_hash(0x9e3779b9u ^ (i * 1543u + mm * 257u + r * 8u + c));
  const float v = float(1024u + (h & 1023u)) * as_type<float>((127u - 10u - 14u + ((h >> 10) % 19u)) << 23);
  return half((h >> 31) ? -v : v);
}
[[kernel]] void mma_chain_check(device atomic_uint* bad [[buffer(0)]], uint i [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y;
  float2 c = float2(as_type<float>((v8e_hash(i * 31u + lane) & 0x807fffffu) | 0x3c000000u),
                    as_type<float>((v8e_hash(i * 37u + lane + 7u) & 0x807fffffu) | 0x3a800000u));
  float2 ref = c;
  for (uint mm = 0; mm < 2; ++mm) {
    v8_mma(c, half2(v8e_a(i, v8e_q(i, mm, fm, fn)), v8e_a(i, v8e_q(i, mm, fm, fn + 1))),
           half2(v8e_b(i, mm, fm, fn), v8e_b(i, mm, fm, fn + 1)));
    for (uint e2 = 0; e2 < 2; ++e2)
      for (uint kk = 0; kk < 8; ++kk) ref[e2] = fma(v8e_af(i, v8e_q(i, mm, fm, kk)), float(v8e_b(i, mm, kk, fn + e2)), ref[e2]);
  }
  const uint n = (as_type<uint>(c.x) != as_type<uint>(ref.x) ? 1u : 0u) + (as_type<uint>(c.y) != as_type<uint>(ref.y) ? 1u : 0u);
  if (n) atomic_fetch_add_explicit(bad, n, memory_order_relaxed);
}

// ---- attention for R <= 8 rows sharing each K/V tile ----
// As attn_rows_partial (same partial layout, reduced by attn_rows_reduce), but one threadgroup serves
// RB rows x G query heads of one kv head and chunk, so each K/V tile is read once per RB rows instead
// of once per row. Row r sees positions < pos0 + r + 1. grid (nchunks, Hkv, ceil(R / RB)),
// 32 * G * RB threads.
[[kernel]] void attn8_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                              device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                              constant uint& R [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup half ktile[kTile * kHD];
  threadgroup half vtile[kTile * kHD];
  const uint ntg = ntg3.x;
  const uint chunk = tg.x, kvh = tg.y;
  const uint G = p.Hq / p.Hkv, RB = nsg / G;
  const uint hq = kvh * G + sg % G, r = tg.z * RB + sg / G;
  const bool live = r < R && sg < G * RB;
  const uint rmax = min(tg.z * RB + RB, R);  // rows of this threadgroup: tg.z * RB .. rmax - 1
  const uint start = chunk * p.chunk;
  const uint end = min(start + p.chunk, p.pos0 + rmax);  // the last row's keys bound the tiles
  const uint L = p.pos0 + r + 1;                          // this row's visible keys
  float qv[8], acc[8];
  for (int i = 0; i < 8; ++i) {
    qv[i] = live ? q[(r * p.Hq + hq) * kHD + lane * 8 + i] * p.scale : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const device half* kbase = kc + kvh * p.max_seq * kHD;
  const device half* vbase = vc + kvh * p.max_seq * kHD;
  for (uint t0 = start; t0 < end; t0 += kTile) {
    const uint n = min(kTile, end - t0);
    for (uint i = tid; i < n * kHD / 4; i += ntg) {
      ((threadgroup half4*)ktile)[i] = ((const device half4*)(kbase + t0 * kHD))[i];
      ((threadgroup half4*)vtile)[i] = ((const device half4*)(vbase + t0 * kHD))[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live) {
      const uint nj = t0 < L ? min(n, L - t0) : 0u;
      for (uint j = 0; j < nj; ++j) {
        const threadgroup half* kr = ktile + j * kHD + lane * 8;
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += qv[i] * float(kr[i]);
        s = simd_sum(s);
        float mn = max(m, s);
        float corr = metal::fast::exp(m - mn);
        float e = metal::fast::exp(s - mn);
        l = l * corr + e;
        const threadgroup half* vr = vtile + j * kHD + lane * 8;
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * corr + e * float(vr[i]);
        m = mn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (live) {
    uint base = (r * p.Hq + hq) * p.nchunks + chunk;
    for (int i = 0; i < 8; ++i) part_o[base * kHD + lane * 8 + i] = acc[i];
    if (lane == 0) {
      part_ml[base * 2 + 0] = m;
      part_ml[base * 2 + 1] = l;
    }
  }
}

// ---- attention for R <= 8 rows on simdgroup MMAs (attn8m) ----
// One threadgroup per (history chunk, kv head) serves all G query heads of the group x 8 rows:
// 2G simdgroups = G heads x 2 halves of the head dim, each holding its head's 8 rows as ONE fragment
// row block (Q: its 128 dims, half; O: its 128 dims, fp32), so every K/V tile is read once for all
// rows and heads. Inner loop = attn_prefill11's (MLX steel-attention layout: S = Q K^T per dim half,
// partial scores summed through threadgroup memory, log2-domain online softmax, fp32 P as the A
// operand of P V), with flash-decoding chunks over the history. Writes attn_rows_partial's layout
// (unnormalized o, natural-log max m, sum l), reduced by attn_rows_reduce. Row t sees positions
// <= pos0 + t. chunk must be a multiple of 32. grid (nchunks, Hkv), 64 * G threads (<= 1024).
// TREE (attn8m_partial_tree, KFlash tree verifies): the rows are a tree whose node t sits in cache row pos0 + t;
// row t sees the history (< pos0) and the rows of its ancestors and itself (tree[kKfMask + t], the KFlash tree
// ancestor-or-self mask), not the rows before it. TREE 1 = that, the block rows as keys of the MMA tiles (their sums
// and tiles follow the rows' slots: the same path inside two trees rounds a bit differently). A tree of more than 8 rows
// (a wide verify, TREE 1 only) runs as 8-row tiles: the grid holds nchunks * nt threadgroups along x, tiles fastest (x =
// chunk * nt + tile: a chunk's K / V tiles are read by its row tiles side by side), and only the live tiles' rows
// (8 nt <= R; every live row's ancestors are live) are keys. TREE 2 = the history
// alone (keys < pos0, every row sees all of them: its tiles follow pos0 only); the block's keys come from
// attn8_tree_block, in an order that follows the path alone, as one more chunk.
// KFlash tree verifies, TREE 3 (NI_V8_TREE_ATTN=2): a row's keys at logical positions below tree_zone(q) go through the
// MMA tiles, whose chunks are the sequence's own absolute ones (a fixed size, tiles aligned at multiples of 32), the
// keys from tree_zone(q) to q through attn8_tree_zone, one by one in position order; both depend on q alone, so a row's
// numbers do not depend on the tree it sits in or on the anchor that splits history from block (zone keys are cache
// rows below the anchor, block rows from it; a row's block keys are its path's rows). The zone starts 7 + 1 keys
// behind q at least, so every MMA key is in the cache whatever the tree.
NI_INLINE uint tree_zone(uint q) { return q >= 7u ? (q - 7u) / 32u * 32u : 0u; }

template <uint TREE>
NI_INLINE void attn8m_partial_impl(const device float* q, const device half* kc, const device half* vc,
                                   device float* part_o, device float* part_ml, constant AttnRowsParams& p, uint R,
                                   const device int* tree, threadgroup uint4* kvm, uint2 tg, uint tid, uint sg,
                                   uint lane, uint ntgx = 0) {
  constexpr uint BK = 32, NC = BK / 8;
  constexpr uint KLD = attn11_kld<BK>();
  threadgroup half* kv = (threadgroup half*)kvm;
  threadgroup float2* xw = (threadgroup float2*)kvm;  // score exchange reuses the K tile
  const uint nt = TREE && ntgx > p.nchunks ? ntgx / p.nchunks : 1u;
  const uint chunk = nt > 1 ? tg.x / nt : tg.x, tile = nt > 1 ? tg.x % nt : 0u, kvh = tg.y;
  if (TREE && R > 8) R = min(R, 8 * nt);  // a wide verify (R = its capacity): the live tiles' rows
  const uint G = p.Hq / p.Hkv;
  const uint h = sg / 2, dh = sg % 2, hq = kvh * G + h;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const uint t = 8 * tile + frow;                   // this lane's row
  const uint qpos = p.pos0 + t;
  const uint anc = (TREE == 1 || TREE == 3) && t < R ? uint(tree[kKfMask + t]) : 0u;  // tree: the block rows this row sees
  // TREE 3: row t's MMA keys are those below tree_zone(its logical position pos0 + depth) -- the position alone, not the
  // tree or the anchor; the keys from there to the row itself are attn8_tree_zone's
  const uint lim = TREE == 3 ? (t < R ? tree_zone(p.pos0 + (anc ? uint(popcount(anc)) - 1u : 0u)) : 0u) : 0u;
  const uint start = chunk * p.chunk;
  const uint kend = min(start + p.chunk, TREE == 2 ? p.pos0 : TREE == 3 ? tree_zone(p.pos0 + 7u) : p.pos0 + R);
  const uint ntiles = kend > start ? (kend - start + BK - 1) / BK : 0u;
  simdgroup_half8x8 qf[kDH / 8];
  {
    const device float* qr = q + (min(t, R - 1) * p.Hq + hq) * kHD + dh * kDH + fcol;
    NI_UNROLL for (uint d = 0; d < kDH / 8; ++d) {
      const half2 v = t < R ? half2(*(const device float2*)(qr + d * 8) * p.scale) : half2(0);
      qf[d].thread_elements()[0] = v.x;
      qf[d].thread_elements()[1] = v.y;
    }
  }
  simdgroup_float8x8 o[kDH / 8];
  NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = kNeg11, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;
  const threadgroup half* kfr = kv + (dh * kDH + frow) * KLD + fcol;
  const threadgroup half* vfr = kv + frow * kVLD11 + dh * kDH + fcol;
  const bool loader = tid < 256;
  for (uint it = 0; it < ntiles; ++it) {
    const uint k0 = start + it * BK;
    uint4 r[BK / 8];
    if (loader) attn11_fetch_k<BK>(r, kb, k0, kend, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (loader) attn11_store_k<BK>(kv, r, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s[NC];
    NI_UNROLL for (uint c = 0; c < NC; ++c) s[c] = simdgroup_float8x8(0);
    NI_UNROLL for (uint d = 0; d < kDH / 8; ++d) {
      simdgroup_half8x8 kf[NC];
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        const half2 k2 = *(const threadgroup half2*)(kfr + d * 8 * KLD + c * 8);
        kf[c].thread_elements()[0] = k2.x;
        kf[c].thread_elements()[1] = k2.y;
      }
      NI_UNROLL for (uint c = 0; c < NC; ++c) simdgroup_multiply_accumulate(s[c], qf[d], kf[c], s[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // K reads done: the exchange reuses the tile
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      xw[(sg * NC + c) * 32 + lane] = float2(s[c].thread_elements()[0], s[c].thread_elements()[1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (uint c = 0; c < NC; ++c) {
      const float2 o2 = xw[((sg ^ 1) * NC + c) * 32 + lane];
      s[c].thread_elements()[0] += o2.x;
      s[c].thread_elements()[1] += o2.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // exchange reads done: V may overwrite
    if (loader) {
      attn11_fetch_v<BK>(r, vb, k0, kend, tid);
      attn11_store_v<BK>(kv, r, tid);
    }
    const bool diag = TREE == 3   ? k0 + BK > tree_zone(p.pos0)
                      : TREE == 2 ? k0 + BK > p.pos0
                                  : k0 + BK - 1 > p.pos0;  // tile reaches past row 0's position
    float mx = kNeg11;
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        float v = s[c].thread_elements()[i] * M_LOG2E_F;
        const uint key = k0 + c * 8 + fcol + i;
        if (TREE == 3) {
          if (diag && key >= lim) v = kNeg11;
        } else if (TREE == 2) {
          if (diag && key >= p.pos0) v = kNeg11;  // (the tile's keys past the history)
        } else if (TREE == 1) {
          const uint j = key - p.pos0;  // block row (key > pos0 here)
          if (diag && key > p.pos0 && !(j < kKfMaxRows && ((anc >> j) & 1u))) v = kNeg11;
        } else if (diag && key > qpos) {
          v = kNeg11;
        }
        s[c].thread_elements()[i] = v;
        mx = max(mx, v);
      }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = metal::fast::exp2(m_row - m_new);
    float ps = 0.0f;
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        const float e = metal::fast::exp2(s[c].thread_elements()[i] - m_new);
        s[c].thread_elements()[i] = e;
        ps += e;
      }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // V tile visible
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) {
      simdgroup_half8x8 vf[NC];
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        const half2 v2 = *(const threadgroup half2*)(vfr + c * 8 * kVLD11 + j * 8);
        vf[c].thread_elements()[0] = v2.x;
        vf[c].thread_elements()[1] = v2.y;
      }
      NI_UNROLL for (uint c = 0; c < NC; ++c) simdgroup_multiply_accumulate(o[j], s[c], vf[c], o[j]);
    }
  }
  if (t < R) {
    const uint base = (t * p.Hq + hq) * p.nchunks + chunk;
    // the chunk holds at least one key this row sees (tree: a chunk past pos0 holds every block row after its start)
    const bool any = TREE == 3   ? start < lim
                     : TREE == 2 ? start < p.pos0
                     : TREE == 1 ? start <= p.pos0 || (start - p.pos0 < kKfMaxRows && (anc >> (start - p.pos0)) != 0u)
                                 : start <= qpos;
    device float* dst = part_o + base * kHD + dh * kDH + fcol;
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j)
      *(device float2*)(dst + j * 8) =
          any ? float2(o[j].thread_elements()[0], o[j].thread_elements()[1]) : float2(0.0f);
    if (dh == 0 && (lane & 9) == 0) {  // one lane per row (fcol == 0)
      part_ml[base * 2 + 0] = any ? m_row * M_LN2_F : -INFINITY;
      part_ml[base * 2 + 1] = any ? l_row : 0.0f;
    }
  }
}
[[kernel]] void attn8m_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                               const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                               device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                               constant uint& R [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  attn8m_partial_impl<0>(q, kc, vc, part_o, part_ml, p, R, nullptr, kvm, tg, tid, sg, lane);
}
[[kernel]] void attn8m_partial_tree(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                    const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                    device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                                    constant uint& R [[buffer(6)]], const device int* tree [[buffer(7)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],
                                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  attn8m_partial_impl<1>(q, kc, vc, part_o, part_ml, p, R, tree, kvm, tg, tid, sg, lane, ntg.x);
}
[[kernel]] void attn8m_partial_zone(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                    const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                    device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                                    constant uint& R [[buffer(6)]], const device int* tree [[buffer(7)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  attn8m_partial_impl<3>(q, kc, vc, part_o, part_ml, p, R, tree, kvm, tg, tid, sg, lane);
}
[[kernel]] void attn8m_partial_hist(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                    const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                    device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                                    constant uint& R [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  attn8m_partial_impl<2>(q, kc, vc, part_o, part_ml, p, R, nullptr, kvm, tg, tid, sg, lane);
}

// The tree block's keys (cache rows pos0 .. pos0 + 7) for attn8m_partial_hist: one more partial chunk (index
// p.nchunks - 1) per (row, head) in attn_rows_partial's layout. A row's keys are its ancestors' rows and its own
// (tree[kKfMask + t]), taken in slot order (root to leaf), their scores and the sums over them in fp32, so the
// numbers follow the path alone, not the slots the other rows of the tree take. grid (Hq, R) x one simdgroup.
[[kernel]] void attn8_tree_block(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                 const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                 device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                                 const device int* tree [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint hq = tg.x, t = tg.y;
  const uint kvh = hq / (p.Hq / p.Hkv);
  const uint anc = uint(tree[kKfMask + t]);
  float qv[8];
  for (uint i = 0; i < 8; ++i) qv[i] = float(half(q[(t * p.Hq + hq) * kHD + lane * 8 + i] * p.scale));
  const device half* kb = kc + (kvh * p.max_seq + p.pos0) * kHD + lane * 8;
  const device half* vb = vc + (kvh * p.max_seq + p.pos0) * kHD + lane * 8;
  float s[8];
  float m = -INFINITY;
  for (uint j = 0; j < 8; ++j) {
    s[j] = -INFINITY;
    if ((anc >> j) & 1u) {  // the same for every lane: the simd_sum below is uniform
      float a = 0.0f;
      for (uint i = 0; i < 8; ++i) a = fma(qv[i], float(kb[j * kHD + i]), a);
      s[j] = simd_sum(a);
      m = max(m, s[j]);
    }
  }
  float l = 0.0f, acc[8];
  for (uint i = 0; i < 8; ++i) acc[i] = 0.0f;
  for (uint j = 0; j < 8; ++j) {
    if ((anc >> j) & 1u) {
      const float e = metal::precise::exp(s[j] - m);
      l += e;
      for (uint i = 0; i < 8; ++i) acc[i] = fma(e, float(vb[j * kHD + i]), acc[i]);
    }
  }
  const uint base = (t * p.Hq + hq) * p.nchunks + (p.nchunks - 1);
  for (uint i = 0; i < 8; ++i) part_o[base * kHD + lane * 8 + i] = acc[i];
  if (lane == 0) {
    part_ml[base * 2 + 0] = m;
    part_ml[base * 2 + 1] = l;
  }
}

// The keys from tree_zone(q) up to q (q = pos0 + the row's depth): below pos0 they are cache rows, then the row's
// ancestors and the row itself (slot order = depth order). Both tail kernels use this fp32 online softmax in position
// order so the result depends on the row's position and path alone. acc, m and l stay in the caller's thread storage.
NI_INLINE void attn8_tree_tail(const device float* q, const device half* kc, const device half* vc,
                               constant AttnRowsParams& p, const device int* tree, uint hq, uint t, uint lane,
                               thread float* acc, thread float& m, thread float& l) {
  const uint kvh = hq / (p.Hq / p.Hkv);
  const uint anc = uint(tree[kKfMask + t]);
  const uint dep = anc ? uint(popcount(anc)) - 1u : 0u;
  uint arow[8];
  {
    uint n = 0;
    for (uint j = 0; j < 8; ++j)
      if ((anc >> j) & 1u) arow[n++] = j;
    for (; n < 8; ++n) arow[n] = 0;
  }
  float qv[8];
  for (uint i = 0; i < 8; ++i) qv[i] = float(half(q[(t * p.Hq + hq) * kHD + lane * 8 + i] * p.scale));
  const device half* kb = kc + kvh * p.max_seq * kHD + lane * 8;
  const device half* vb = vc + kvh * p.max_seq * kHD + lane * 8;
  const uint qp = p.pos0 + dep;
  m = -INFINITY; l = 0.0f;
  for (uint i = 0; i < 8; ++i) acc[i] = 0.0f;
  for (uint k = tree_zone(qp); k <= qp; ++k) {
    const uint row = k < p.pos0 ? k : p.pos0 + arow[k - p.pos0];
    float a = 0.0f;
    for (uint i = 0; i < 8; ++i) a = fma(qv[i], float(kb[row * kHD + i]), a);
    const float sc = simd_sum(a);
    const float m_new = max(m, sc);
    const float corr = m == -INFINITY ? 0.0f : metal::precise::exp(m - m_new);
    const float e = metal::precise::exp(sc - m_new);
    l = l * corr + e;
    for (uint i = 0; i < 8; ++i) acc[i] = fma(e, float(vb[row * kHD + i]), acc[i] * corr);
    m = m_new;
  }
}

// Write the tail as the last partial chunk (index p.nchunks - 1) for attn_rows_reduce.
// grid (Hq, R) x one simdgroup.
[[kernel]] void attn8_tree_zone(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                                const device int* tree [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const uint hq = tg.x, t = tg.y;
  float acc[8], m, l;
  attn8_tree_tail(q, kc, vc, p, tree, hq, t, lane, acc, m, l);
  const uint base = (t * p.Hq + hq) * p.nchunks + (p.nchunks - 1);
  for (uint i = 0; i < 8; ++i) part_o[base * kHD + lane * 8 + i] = acc[i];
  if (lane == 0) {
    part_ml[base * 2 + 0] = m;
    part_ml[base * 2 + 1] = l;
  }
}

// The tree-zone tail stays in each lane's registers, then joins the history partials in the original order.
// One simdgroup owns a (row, head), with eight output dimensions per lane; no global tail write or extra dispatch.
[[kernel]] void attn8_tree_zone_reduce(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                       const device half* vc [[buffer(2)]], const device float* part_o [[buffer(3)]],
                                       const device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                                       const device int* tree [[buffer(6)]], const device float* qg [[buffer(7)]],
                                       device float* out [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  const uint hq = tg.x, t = tg.y;
  float acc[8], m, l;
  attn8_tree_tail(q, kc, vc, p, tree, hq, t, lane, acc, m, l);
  const uint base = (t * p.Hq + hq) * p.nchunks;
  float M = -INFINITY;
  for (uint c = 0; c + 1 < p.nchunks; ++c) M = max(M, part_ml[(base + c) * 2]);
  M = max(M, m);
  float num[8], den = 0.0f;
  for (uint i = 0; i < 8; ++i) num[i] = 0.0f;
  for (uint c = 0; c + 1 < p.nchunks; ++c) {
    const float mc = part_ml[(base + c) * 2];
    if (mc == -INFINITY) continue;
    const float w = metal::fast::exp(mc - M);
    for (uint i = 0; i < 8; ++i) num[i] += w * part_o[(base + c) * kHD + lane * 8 + i];
    den += w * part_ml[(base + c) * 2 + 1];
  }
  const float w = metal::fast::exp(m - M);
  den += w * l;
  for (uint i = 0; i < 8; ++i) {
    num[i] += w * acc[i];
    const float gate = qg[(t * p.Hq + hq) * 2 * kHD + kHD + lane * 8 + i];
    out[(t * p.Hq + hq) * kHD + lane * 8 + i] = (num[i] / den) * sigmoid(gate);
  }
}

// ---- KFlash tree verifies: q / k norm + RoPE and the K/V cache write of a tree verify's rows ----
// attn_prep / attn_prep_m (same buffers 0-10, MropeParams from mm.metal) for a tree (tree = the KFlash tree descriptor at
// buffer 11): row t's K and V go to cache row pos + t, but q and k rotate at the position its token takes, pos +
// depth(t) (rows past the images' positions: + delta, as attn_prep_m). grid: x = Hq + 2 Hkv, y = rows; head_dim
// threads.
[[kernel]] void attn_prep_tree(const device float* qg [[buffer(0)]], const device float* kr [[buffer(1)]],
                               const device float* vr [[buffer(2)]], const device bfloat* qn_w [[buffer(3)]],
                               const device bfloat* kn_w [[buffer(4)]], device float* q_out [[buffer(5)]],
                               device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                               constant AttnPrepParams& p [[buffer(8)]], const device int* mpos [[buffer(9)]],
                               constant MropeParams& mr [[buffer(10)]], const device int* tree [[buffer(11)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  threadgroup float scratch[32];
  threadgroup float row[512];
  const uint h = tg.x, t = tg.y, D = p.D;
  const uint pos = p.pos + t;                    // cache row
  const uint tpos = p.pos + uint(tree[kKfDep + t]);  // the position of its token
  if (h >= p.Hq + p.Hkv) {  // value head: plain copy into the cache
    uint hv = h - p.Hq - p.Hkv;
    vc[(hv * p.max_seq + pos) * D + tid] = half(vr[(t * p.Hkv + hv) * D + tid]);
    return;
  }
  const bool is_q = h < p.Hq;
  float x = is_q ? qg[(t * p.Hq + h) * 2 * D + tid] : kr[(t * p.Hkv + (h - p.Hq)) * D + tid];
  float ss = tg_sum(x * x, scratch, tid, D);
  x = x * metal::precise::rsqrt(ss / float(D) + p.eps) * float(is_q ? qn_w[tid] : kn_w[tid]);
  row[tid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_rot = p.rot / 2;
  if (tid < p.rot) {
    uint i = tid % half_rot;
    int rp;
    if (tpos < mr.mn) {
      const uint a = (i % 3 == 1 && i < mr.sec1) ? 1u : (i % 3 == 2 && i < mr.sec2) ? 2u : 0u;
      rp = mpos[tpos * 3 + a];
    } else {
      rp = int(tpos) + mr.delta;
    }
    float inv_freq = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.rot));
    float ang = float(rp) * inv_freq;
    float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    if (tid < half_rot) x = row[tid] * cs - row[tid + half_rot] * sn;
    else x = row[tid] * cs + row[tid - half_rot] * sn;
  }
  if (is_q) q_out[(t * p.Hq + h) * D + tid] = x;
  else kc[((h - p.Hq) * p.max_seq + pos) * D + tid] = half(x);
}

// ---- KFlash tree verifies wider than 8 rows: this cycle's row tiles, on the GPU ----
// The host encodes a wide tree verify (16 / 32 rows, Qwen35::kMaxTreeRows) before the drafting pass has chosen the
// tree's live rows (the size rule runs in kf_build), so every dispatch whose grid scales with the rows takes its
// threadgroup counts from args (dispatchThreadgroups indirect), and this kernel, the verify's first, writes them from
// the tree's live rows (kf_live_rows: rows = 8 nt, nt live 8-row tiles). desc[0].x: the grid count; desc[1 + j] =
// {x, y, z, flags}, flags 16: x *= ceil(nt / 2) (the tile-pair qmm8t2), 1: x *= nt (GEMMs, the tree attention: tiles fastest along x), 2: y *= nt (row-tile grids:
// X8 tables, SwiGLU, gated norm), 4: y = rows (per-row grids), 8: x = rows. args: 4 words per grid. grid: 1 x 64.
[[kernel]] void v8_tree_plan(const device int* tree [[buffer(0)]], const device uint4* desc [[buffer(1)]],
                             device uint4* args [[buffer(2)]], constant uint& R [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  const uint n = desc[0].x, rows = kf_live_rows(tree, R), nt = max(rows / 8, 1u);
  for (uint j = i; j < n; j += 64) {
    const uint4 d = desc[1 + j];
    uint x = d.x, y = d.y;
    if (d.w & 1u) x *= nt;
    if (d.w & 16u) x *= (nt + 1) / 2;
    if (d.w & 2u) y *= nt;
    if (d.w & 4u) y = rows;
    if (d.w & 8u) x = rows;
    args[j] = uint4(x, y, d.z, 0u);
  }
}

// After a tree verify's acceptance: the accepted path's K / V rows (tree[kKfPath + i] = its i-th row's cache row -
// pos, a = tree[kKfA] drafts; kf_accept) move to the rows of their positions pos + i, in order (a row's destination
// is below every later row's source). grid (1, 2 Hkv) x D: K heads, then V heads.
struct KvTreeParams {
  uint pos, Hkv, D, max_seq;
};
[[kernel]] void kv_tree_compact(device half* kc [[buffer(0)]], device half* vc [[buffer(1)]],
                                const device int* tree [[buffer(2)]], constant KvTreeParams& p [[buffer(3)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint h = tg.y % p.Hkv;
  device half* c = (tg.y < p.Hkv ? kc : vc) + size_t(h) * p.max_seq * p.D + d;
  const uint a = uint(tree[kKfA]);
  for (uint i = 1; i <= a; ++i) {
    const uint s = uint(tree[kKfPath + i]);
    if (s != i) c[size_t(p.pos + i) * p.D] = c[size_t(p.pos + s) * p.D];
  }
}

// ---- Gated DeltaNet verify step for up to 8 rows with one normalization pass ----
// gdn_step_verify's arithmetic (fold the pending tape rows, then the R new rows without committing
// them, recording their tape), but the gated RMSNorm of all R rows runs once after the scan from
// threadgroup memory instead of two threadgroup barriers per row. grid: Hv threadgroups x 1024.
[[kernel]] void gdn_step_verify8(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                                 const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],
                                 const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                                 const device float* z [[buffer(6)]], const device bfloat* norm_w [[buffer(7)]],
                                 device float* out [[buffer(8)]], device float* pend_k [[buffer(9)]],
                                 device float* pend_g [[buffer(10)]], device float* pend_d [[buffer(11)]],
                                 constant GdnVerifyParams& p [[buffer(12)]], uint h [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float os[8][128];
  constexpr uint D = 128;
  const uint hk = h / (p.Hv / p.Hk);
  device float* S = state + h * D * D;
  float4 st[4];
  for (int r = 0; r < 4; ++r) st[r] = *(device float4*)(S + (sg * 4 + r) * D + lane * 4);
  for (uint t = 0; t < p.n_pend; ++t) {
    float4 kk = *(const device float4*)(pend_k + (t * p.Hv + h) * D + lane * 4);
    float gg = pend_g[t * p.Hv + h];
    for (int r = 0; r < 4; ++r) st[r] = st[r] * gg + kk * pend_d[(t * p.Hv + h) * D + sg * 4 + r];
  }
  if (p.n_pend > 0)
    for (int r = 0; r < 4; ++r) *(device float4*)(S + (sg * 4 + r) * D + lane * 4) = st[r];
  threadgroup_barrier(mem_flags::mem_device);  // all tape reads done before it is overwritten
  for (uint t = 0; t < p.R; ++t) {
    const float4 q4 = *(const device float4*)(q + (t * p.Hk + hk) * D + lane * 4);
    const float4 k4 = *(const device float4*)(k + (t * p.Hk + hk) * D + lane * 4);
    const float gg = g[t * p.Hv + h], be = beta[t * p.Hv + h];
    const float4 v4 = *(const device float4*)(v + (t * p.Hv + h) * D + sg * 4);
    if (sg == 0) *(device float4*)(pend_k + (t * p.Hv + h) * D + lane * 4) = k4;
    if (sg == 0 && lane == 0) pend_g[t * p.Hv + h] = gg;
    float o[4], dl[4];
    for (int r = 0; r < 4; ++r) {
      st[r] *= gg;
      const float kv = simd_sum(dot(st[r], k4));
      dl[r] = (v4[r] - kv) * be;
      st[r] += k4 * dl[r];
      o[r] = simd_sum(dot(st[r], q4));
    }
    if (lane == 0) {
      *(device float4*)(pend_d + (t * p.Hv + h) * D + sg * 4) = float4(dl[0], dl[1], dl[2], dl[3]);
      *(threadgroup float4*)&os[t][sg * 4] = float4(o[0], o[1], o[2], o[3]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // gated RMSNorm: simdgroup t normalizes row t (lane: 4 consecutive channels)
  if (sg < p.R) {
    const uint t = sg;
    const float4 ov = *(threadgroup float4*)&os[t][lane * 4];
    const float tot = simd_sum(dot(ov, ov));
    const float inv = metal::precise::rsqrt(tot / float(D) + p.eps);
    const uint base = (t * p.Hv + h) * D + lane * 4;
    const float4 zz = *(const device float4*)(z + base);
    float4 r;
    for (int i = 0; i < 4; ++i) r[i] = ov[i] * inv * float(norm_w[lane * 4 + i]) * silu(zz[i]);
    *(device float4*)(out + base) = r;
  }
}

// SwiGLU of the 8 rows' fp32 gate / up projections into the down GEMM's X8 table (x8_prep2's layout and order:
// table entries half(silu(g) * u), sums of the halfs per 64 inputs): the split MLP path, where gate and up run
// as two segments of one qmm8 dispatch. grid: K / 256 threadgroups x 256 threads (K = intermediate width); x row
// tiles (grid y: rows 8 y .. 8 y + 7 into tile y's table, as x8_prep2).
struct SwigluX8Params {
  uint K;   // intermediate width (row length of g and u)
  uint R;   // live rows (t >= R are zeros)
  uint write_t;  // 1: also the T table of qmm8t (buffer 5; x8_prep2's write_t layout, rows of K inputs)
};
[[kernel]] void swiglu_x8(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                          device half* table [[buffer(2)]], device float* sums [[buffer(3)]],
                          constant SwigluX8Params& p [[buffer(4)]], device half* table_t [[buffer(5)]],
                          uint2 tg2 [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half stg[8][256];
  const uint tg = tg2.x, tile = tg2.y;
  if (tile) {
    g += 8 * tile * p.K;
    u += 8 * tile * p.K;
    table += tile * p.K * 8;
    sums += tile * (p.K / 64) * kV8;
    table_t += 8 * tile * p.K;
  }
  const uint t = sg;
  const bool live = 8 * tile + t < p.R;
  const uint k0 = tg * 256 + lane * 8;
  float v[8];
  if (live) {
    const float4 g0 = *(const device float4*)(g + t * p.K + k0), g1 = *(const device float4*)(g + t * p.K + k0 + 4);
    const float4 u0 = *(const device float4*)(u + t * p.K + k0), u1 = *(const device float4*)(u + t * p.K + k0 + 4);
    for (int i = 0; i < 4; ++i) {
      v[i] = silu(g0[i]) * u0[i];
      v[4 + i] = silu(g1[i]) * u1[i];
    }
  } else {
    for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  }
  float sh = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const half h = half(v[i]);
    stg[t][lane * 8 + i] = h;
    sh += float(h);
  }
  sh += simd_shuffle_xor(sh, 1u);
  sh += simd_shuffle_xor(sh, 2u);
  sh += simd_shuffle_xor(sh, 4u);
  if ((lane & 7) == 0) sums[(tg * 4 + lane / 8) * kV8 + t] = sh;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gi = tid / 64, l2 = (tid % 64) / 2, hf = tid % 2;
  const uint2 rc = v8_lane(l2);
  const uint fm = rc.x, fn = rc.y;
  half o[8];
  for (int jj = 0; jj < 8; ++jj) {
    const uint j = hf * 8 + jj, f = j / 2, e = j % 2;
    const uint kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
    o[jj] = stg[fn + e][gi * 64 + kk];
  }
  device half* dst = table + ((tg * 4 + gi) * 32 + l2) * 16 + hf * 8;
  *(device uint4*)dst = uint4(as_type<uint>(half2(o[0], o[1])), as_type<uint>(half2(o[2], o[3])),
                              as_type<uint>(half2(o[4], o[5])), as_type<uint>(half2(o[6], o[7])));
  if (p.write_t) {  // T table: thread = (row tid / 32, 8 slots of this threadgroup's 256)
    const uint tt = tid / 32, s0 = (tid % 32) * 8, gb = s0 & ~63u;
    half q[8];
    for (int i = 0; i < 8; ++i) q[i] = stg[tt][gb + v8t_input_of_slot((s0 + i) & 63)];
    *(device uint4*)(table_t + tt * p.K + tg * 256 + s0) =
        uint4(as_type<uint>(half2(q[0], q[1])), as_type<uint>(half2(q[2], q[3])), as_type<uint>(half2(q[4], q[5])),
              as_type<uint>(half2(q[6], q[7])));
  }
}

// A value row's rank-1 update, st + k d, as one fused multiply-add on the decayed state (st already times g): the step,
// the fold of the previous cycle's accepted rows and the replay of a jumping row's ancestry all end in this one
// operation, so a row has the same state whether the tree reaches it in sequence or by a replay. (Left to fast math,
// the fold's and the replay's st g + k d may fuse the g product and round the k d one, the step's st + k d the
// other way round.) The 35B's router top-k flips on a last-bit difference between two trees' rows of one path.
NI_INLINE float4 gdn_update(float4 st_g, float4 k, float d) { return fma(k, float4(d), st_g); }

// ---- Gated DeltaNet verify step split over value rows (no norm) + the gated norm into the X8 table ----
// The recurrence of a value head is independent per value row (S[v] = S[v] g + k d[v], d[v] = (v[v] - S[v].k) beta,
// o[v] = S[v].q), so a head's 128 rows are spread over 128 / (4 nsg) threadgroups of nsg simdgroups x 4 rows
// (gdn_step_verify8's per-row arithmetic: bitwise the same o and tape), which fills the GPU (48 threadgroups of
// 1024 threads left it underused). The per-head tape rows k and g (read by every block for the fold) are written
// by the block that arrives last at the counter cnt[h] (it also resets it): every block has read the old ones by
// then. Raw outputs o -> out [R][Hv][128] (fp32); gdn_gnorm_x8 applies the gated RMSNorm. grid (Hv, blocks).
// TREE (gdn_step_verify8s_tree, KFlash tree verifies): the rows are a tree (p.tree; row t's parent tree[t]; its live
// rows alone, the rows in sequence being this step's cost) and the pending rows a tree cycle's accepted path (p.pidx:
// tape rows path[0 .. n_pend)). A row whose parent is the row before it continues from that row's state; any other
// row first rebuilds its parent's state from the base (kept in registers): the folds of its ancestors' tape entries
// (st = st g + k d, rows in slot order = root to leaf; d: the tape rows this simdgroup wrote).
template <bool TREE>
NI_INLINE void gdn_step_verify8s_impl(const device float* q, const device float* k, const device float* v,
                                      const device float* g, const device float* beta, device float* state,
                                      device float* out, device float* pend_k, device float* pend_g,
                                      device float* pend_d, constant GdnVerifyParams& p, device atomic_uint* cnt,
                                      const device int* tree, const device int* path, threadgroup uint& last,
                                      uint2 tg, uint2 ntg, uint sg, uint nsg, uint lane, uint tid) {
  constexpr uint D = 128;
  const uint h = tg.x, hk = h / (p.Hv / p.Hk);
  const uint row0 = (tg.y * nsg + sg) * 4;  // this simdgroup's 4 value rows
  device float* S = state + h * D * D;
  float4 st[4];
  for (int r = 0; r < 4; ++r) st[r] = *(device float4*)(S + (row0 + r) * D + lane * 4);
  const bool pin = TREE && p.pin != 0;
  for (uint t0 = 0; t0 < p.n_pend; ++t0) {
    const uint t = TREE && p.pidx ? uint(path[t0]) : t0;
    float4 kk = *(const device float4*)(pend_k + (t * p.Hv + h) * D + lane * 4);
    float gg = pend_g[t * p.Hv + h];
    for (int r = 0; r < 4; ++r) {
      const float d = pend_d[(t * p.Hv + h) * D + row0 + r];
      if (pin) st[r] = gdn_update(st[r] * gg, kk, d);
      else st[r] = st[r] * gg + kk * d;
    }
  }
  if (p.n_pend > 0)
    for (int r = 0; r < 4; ++r) *(device float4*)(S + (row0 + r) * D + lane * 4) = st[r];
  threadgroup_barrier(mem_flags::mem_device);  // this block's tape reads are done
  if (tid == 0) last = atomic_fetch_add_explicit(&cnt[h], 1u, memory_order_relaxed) == ntg.y - 1 ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool writer = last != 0;  // every block of the head has read the old k / g tape
  const uint rebuild = TREE && p.tree ? uint(tree[kKfJump]) : 0u;  // tree: rows whose parent is not the row before them
  // a wide tree verify (R = its capacity, > 8 rows): the live tiles' rows (kf_live_rows); else the tree's live rows
  const uint R = TREE && p.tree ? kf_live_rows(tree, p.R) : p.R;
  float4 st0[4];  // tree: the base state
  if (TREE)
    for (int r = 0; r < 4; ++r) st0[r] = st[r];
  for (uint t = 0; t < R; ++t) {
    if (TREE && ((rebuild >> t) & 1u)) {
      simdgroup_barrier(mem_flags::mem_device);  // this simdgroup's tape rows of the earlier rows
      for (int r = 0; r < 4; ++r) st[r] = st0[r];
      const uint anc = uint(tree[kKfMask + t]);
      for (uint j = 0; j < t; ++j) {
        if (!((anc >> j) & 1u)) continue;
        const float4 kj = *(const device float4*)(k + (j * p.Hk + hk) * D + lane * 4);
        const float gj = g[j * p.Hv + h];
        const float4 dj = *(const device float4*)(pend_d + (j * p.Hv + h) * D + row0);
        for (int r = 0; r < 4; ++r) {
          if (pin) {
            st[r] = gdn_update(st[r] * gj, kj, dj[r]);
          } else {
            st[r] *= gj;
            st[r] += kj * dj[r];
          }
        }
      }
    }
    const float4 q4 = *(const device float4*)(q + (t * p.Hk + hk) * D + lane * 4);
    const float4 k4 = *(const device float4*)(k + (t * p.Hk + hk) * D + lane * 4);
    const float gg = g[t * p.Hv + h], be = beta[t * p.Hv + h];
    const float4 v4 = *(const device float4*)(v + (t * p.Hv + h) * D + row0);
    if (writer && sg == 0) *(device float4*)(pend_k + (t * p.Hv + h) * D + lane * 4) = k4;
    if (writer && sg == 0 && lane == 0) pend_g[t * p.Hv + h] = gg;
    float o[4], dl[4];
    for (int r = 0; r < 4; ++r) {
      st[r] *= gg;
      const float kv = simd_sum(dot(st[r], k4));
      dl[r] = (v4[r] - kv) * be;
      if (pin) st[r] = gdn_update(st[r], k4, dl[r]);
      else st[r] += k4 * dl[r];
      o[r] = simd_sum(dot(st[r], q4));
    }
    if (lane == 0) {
      *(device float4*)(pend_d + (t * p.Hv + h) * D + row0) = float4(dl[0], dl[1], dl[2], dl[3]);
      *(device float4*)(out + (t * p.Hv + h) * D + row0) = float4(o[0], o[1], o[2], o[3]);
    }
  }
  if (writer && tid == 0) atomic_store_explicit(&cnt[h], 0u, memory_order_relaxed);
}
[[kernel]] void gdn_step_verify8s(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                                  const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],
                                  const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                                  device float* out [[buffer(6)]], device float* pend_k [[buffer(9)]],
                                  device float* pend_g [[buffer(10)]], device float* pend_d [[buffer(11)]],
                                  constant GdnVerifyParams& p [[buffer(12)]], device atomic_uint* cnt [[buffer(13)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint last;
  gdn_step_verify8s_impl<false>(q, k, v, g, beta, state, out, pend_k, pend_g, pend_d, p, cnt, nullptr, nullptr, last,
                                tg, ntg, sg, nsg, lane, tid);
}
[[kernel]] void gdn_step_verify8s_tree(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                                       const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],
                                       const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                                       device float* out [[buffer(6)]], device float* pend_k [[buffer(9)]],
                                       device float* pend_g [[buffer(10)]], device float* pend_d [[buffer(11)]],
                                       constant GdnVerifyParams& p [[buffer(12)]],
                                       device atomic_uint* cnt [[buffer(13)]], const device int* tree [[buffer(14)]],
                                       const device int* path [[buffer(15)]], uint2 tg [[threadgroup_position_in_grid]],
                                       uint2 ntg [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint last;
  gdn_step_verify8s_impl<true>(q, k, v, g, beta, state, out, pend_k, pend_g, pend_d, p, cnt, tree, path, last, tg, ntg,
                               sg, nsg, lane, tid);
}

// The gated RMSNorm of gdn_step_verify8 (bitwise its values) from the raw outputs, written as the out_proj's X8
// table with group sums in x8_prep2's addition order (8 consecutive inputs sequentially: this lane's 4, then the odd
// neighbour's 4; then lanes 2, 4, 8 apart = x8_prep2's 1, 2, 4), i.e. bitwise the table x8_prep2 would build from
// gdn_step_verify8's fp32 rows. grid Hv x 256 (simdgroup = row).
// TW (gdn_gnorm_x8t): threads 128-255 also write the T table of qmm8t (buffer 6, rows of Hv * 128 inputs)
// Row tiles (grid y, v8_tiles): rows 8 y .. 8 y + 7 into tile y's tables.
template <bool TW>
NI_INLINE void gdn_gnorm_x8_impl(const device float* o_raw, const device float* z, const device bfloat* norm_w,
                                 device half* table, device float* sums, constant GdnVerifyParams& p,
                                 device half* table_t, threadgroup half (*hs)[128], uint h, uint tid, uint sg,
                                 uint lane, uint tile = 0) {
  constexpr uint D = 128;
  if (tile) {
    const uint K = p.Hv * D;
    o_raw += 8 * tile * K;
    z += 8 * tile * K;
    table += tile * K * 8;
    sums += tile * (K / 64) * kV8;
    table_t += 8 * tile * K;
  }
  const uint t = sg;
  float4 r = float4(0.0f);
  if (8 * tile + t < p.R) {
    const float4 ov = *(const device float4*)(o_raw + (t * p.Hv + h) * D + lane * 4);
    const float tot = simd_sum(dot(ov, ov));
    const float inv = metal::precise::rsqrt(tot / float(D) + p.eps);
    const float4 zz = *(const device float4*)(z + (t * p.Hv + h) * D + lane * 4);
    for (int i = 0; i < 4; ++i) r[i] = ov[i] * inv * float(norm_w[lane * 4 + i]) * silu(zz[i]);
  }
  const half4 hv = half4(r);
  *(threadgroup half4*)&hs[t][lane * 4] = hv;
  float a = 0.0f;
  for (int i = 0; i < 4; ++i) a += float(hv[i]);
  float b[4];
  for (int i = 0; i < 4; ++i) b[i] = simd_shuffle_down(float(hv[i]), 1u);
  for (int i = 0; i < 4; ++i) a += b[i];
  a += simd_shuffle_xor(a, 2u);
  a += simd_shuffle_xor(a, 4u);
  a += simd_shuffle_xor(a, 8u);
  if ((lane & 15) == 0) sums[(2 * h + lane / 16) * kV8 + t] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128) {
    const uint gi = tid / 64, l2 = (tid % 64) / 2, hf = tid % 2;
    const uint2 rc = v8_lane(l2);
    const uint fm = rc.x, fn = rc.y;
    half o8[8];
    for (int jj = 0; jj < 8; ++jj) {
      const uint j = hf * 8 + jj, f = j / 2, e = j % 2;
      const uint kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
      o8[jj] = hs[fn + e][gi * 64 + kk];
    }
    device half* dst = table + ((2 * h + gi) * 32 + l2) * 16 + hf * 8;
    *(device uint4*)dst = uint4(as_type<uint>(half2(o8[0], o8[1])), as_type<uint>(half2(o8[2], o8[3])),
                                as_type<uint>(half2(o8[4], o8[5])), as_type<uint>(half2(o8[6], o8[7])));
  } else if (TW) {
    const uint i = tid - 128, tt = i / 16, s0 = (i % 16) * 8, gb = s0 & ~63u;
    half o8[8];
    for (int j = 0; j < 8; ++j) o8[j] = hs[tt][gb + v8t_input_of_slot((s0 + j) & 63)];
    *(device uint4*)(table_t + tt * (p.Hv * D) + h * D + s0) =
        uint4(as_type<uint>(half2(o8[0], o8[1])), as_type<uint>(half2(o8[2], o8[3])),
              as_type<uint>(half2(o8[4], o8[5])), as_type<uint>(half2(o8[6], o8[7])));
  }
}
[[kernel]] void gdn_gnorm_x8(const device float* o_raw [[buffer(0)]], const device float* z [[buffer(1)]],
                             const device bfloat* norm_w [[buffer(2)]], device half* table [[buffer(3)]],
                             device float* sums [[buffer(4)]], constant GdnVerifyParams& p [[buffer(5)]],
                             uint2 hg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half hs[8][128];
  gdn_gnorm_x8_impl<false>(o_raw, z, norm_w, table, sums, p, table, hs, hg.x, tid, sg, lane, hg.y);
}
[[kernel]] void gdn_gnorm_x8t(const device float* o_raw [[buffer(0)]], const device float* z [[buffer(1)]],
                              const device bfloat* norm_w [[buffer(2)]], device half* table [[buffer(3)]],
                              device float* sums [[buffer(4)]], constant GdnVerifyParams& p [[buffer(5)]],
                              device half* table_t [[buffer(6)]],
                              uint2 hg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half hs[8][128];
  gdn_gnorm_x8_impl<true>(o_raw, z, norm_w, table, sums, p, table_t, hs, hg.x, tid, sg, lane, hg.y);
}
