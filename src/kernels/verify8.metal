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
//  This is Splash's Apple9 register-tile idea (incoai/splash, Apache-2.0,
//  runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal: integer codes as exact bf16 MMA operands,
//  one fp32 epilogue per coefficient group); the code here is our own and reads the MLX weight layout
//  directly (no repacked weight image), with half activations (11-bit mantissa) instead of bf16.
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
};

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
template <int BITS, int NF, bool SPLIT = false>
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
    if (SPLIT) {  // two independent MMA chains per fragment (more MMAs in flight)
      float2 d1 = float2(0.0f);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; j += 2) {
        v8_mma(dot, v8_codes<BITS>(o.w[f], j), b[j]);
        v8_mma(d1, v8_codes<BITS>(o.w[f], j + 1), b[j + 1]);
      }
      dot += d1;
    } else {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; ++j) v8_mma(dot, v8_codes<BITS>(o.w[f], j), b[j]);
    }
    acc[f] = fma(dot, float2(s), acc[f]);
    acc[f] = fma(o.sx, float2(fma(-1024.0f, s, bi)), acc[f]);
  }
}
template <int BITS, int NF, bool SPLIT = false>
NI_INLINE void v8_run_pf(const device uint32_t* const thread* wp, const device ushort* const thread* sp,
                         const device ushort* const thread* bp, const device half* table, const device float* sums,
                         uint g0, uint g1, uint lane, uint fn, thread float2* acc) {
  if (g0 >= g1) return;
  V8Ops<BITS, NF> cur, nxt;
  v8_load<BITS, NF>(cur, wp, sp, bp, table, sums, g0, lane, fn);
  for (uint g = g0; g < g1; ++g) {
    if (g + 1 < g1) v8_load<BITS, NF>(nxt, wp, sp, bp, table, sums, g + 1, lane, fn);
    v8_compute<BITS, NF, SPLIT>(cur, acc);
    cur = nxt;
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

// Dense projections, up to 4 segments sharing the input (e.g. GDN qkv|z|b|a): segment i's
// threadgroups [tg_begin[i], tg_begin[i+1]) each cover cs column blocks of 8 * NF rows.
// V: 0 = v8_run, 1 = prefetching (v8_run_pf), 2 = v8_run_lc + split chains, 3 = prefetching + split chains
template <int BITS, int NF, bool X2, int V = 0>
NI_INLINE void qmm8_impl(const device uint32_t* const thread* Wv, const device ushort* const thread* Sv,
                         const device ushort* const thread* Bv, device float* const thread* Yv,
                         const device half* table, const device float* sums, const device half* table_lo,
                         const device float* sums_lo, constant Qmm8Params& p, threadgroup float2* red, uint tg,
                         uint sg, uint nsg, uint lane) {
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
  else v8_run_pf<BITS, NF, true>(wp, sp, bp, table, sums, g0, g1, lane, fn, acc);
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
      const uint t = fn + e;
      if (t < p.R) {
        device float* d = Y + t * y_ld + n;
        *d = add ? *d + acc[f][e] : acc[f][e];
      }
    }
  }
}

#define NI_QMM8(NAME, BITS, NF, X2, ...)                                                                             \
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
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],  \
                       uint lane [[thread_index_in_simdgroup]]) {                                            \
    threadgroup float2 red[8 * NF * 32];                                                                     \
    const device uint32_t* Wv[4] = {W0, W1, W2, W3};                                                          \
    const device ushort* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device ushort* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmm8_impl<BITS, NF, X2, ##__VA_ARGS__>(Wv, Sv, Bv, Yv, table, sums, table_lo, sums_lo, p, red, tg, sg, nsg, lane); \
  }

NI_QMM8(qmm8_b4_nf1, 4, 1, false)
NI_QMM8(qmm8_b4_nf2, 4, 2, false)
NI_QMM8(qmm8_b4_nf4, 4, 4, false)
NI_QMM8(qmm8_b8_nf1, 8, 1, false)
NI_QMM8(qmm8_b8_nf2, 8, 2, false)
NI_QMM8(qmm8x2_b4_nf1, 4, 1, true)
NI_QMM8(qmm8x2_b8_nf1, 8, 1, true)
NI_QMM8(qmm8pf_b4_nf1, 4, 1, false, 1)
NI_QMM8(qmm8pf_b4_nf2, 4, 2, false, 1)
NI_QMM8(qmm8pf_b4_nf4, 4, 4, false, 1)
NI_QMM8(qmm8s_b4_nf1, 4, 1, false, 2)
NI_QMM8(qmm8s_b4_nf2, 4, 2, false, 2)
NI_QMM8(qmm8s_b4_nf4, 4, 4, false, 2)
NI_QMM8(qmm8pfs_b4_nf1, 4, 1, false, 3)
NI_QMM8(qmm8pfs_b4_nf2, 4, 2, false, 3)
NI_QMM8(qmm8pfs_b4_nf4, 4, 4, false, 3)

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

[[kernel]] void qmm8_gu_b4(const device uint32_t* Wg [[buffer(0)]], const device ushort* Sg [[buffer(1)]],
                           const device ushort* Bg [[buffer(2)]], const device uint32_t* Wu [[buffer(3)]],
                           const device ushort* Su [[buffer(4)]], const device ushort* Bu [[buffer(5)]],
                           const device uint32_t* SWg [[buffer(6)]], const device ushort* SSg [[buffer(7)]],
                           const device ushort* SBg [[buffer(8)]], const device uint32_t* SWu [[buffer(9)]],
                           const device ushort* SSu [[buffer(10)]], const device ushort* SBu [[buffer(11)]],
                           const device half* table [[buffer(12)]], const device float* sums [[buffer(13)]],
                           device half* hT [[buffer(14)]], device float* hS [[buffer(15)]],
                           constant Qmm8GuParams& p [[buffer(16)]], const device uint* tiles [[buffer(17)]],
                           const device uint* ntiles [[buffer(18)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[4 * kV8];
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
  const uint c0 = tg.x * 64 + sg * 16;
  const device uint32_t* wp[4];
  const device ushort* sp[4];
  const device ushort* bp[4];
  float2 acc[4];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 2; ++f) {
    const uint row = c0 + 8 * f + fm;
    wp[f] = G + row * wpr + 2 * q;
    sp[f] = Gs + row * gpr;
    bp[f] = Gb + row * gpr;
    wp[f + 2] = U + row * wpr + 2 * q;
    sp[f + 2] = Us + row * gpr;
    bp[f + 2] = Ub + row * gpr;
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 4; ++f) acc[f] = float2(0.0f);
  v8_run<4, 4>(wp, sp, bp, table, sums, table, sums, 0, gpr, lane, fn, acc);
  device half* ht = hT + tg.y * p.h_stride;
  float2 ps = float2(0.0f);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 2; ++f) {
    const uint kk = sg * 16 + 8 * f + fm;
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const half h = half(silu(acc[f][e]) * acc[f + 2][e]);
      ht[v8_tab(tg.x, kk, fn + e)] = h;
      ps[e] += float(h);
    }
  }
  // sum over the simdgroup's 16 columns: lanes sharing fn differ in the fm bits (lane bits 1, 2, 4)
  ps += simd_shuffle_xor(ps, 2u);
  ps += simd_shuffle_xor(ps, 4u);
  ps += simd_shuffle_xor(ps, 16u);
  if (fm == 0) {
    red[sg * kV8 + fn] = ps.x;
    red[sg * kV8 + fn + 1] = ps.y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < kV8)
    hS[tg.y * p.hs_stride + tg.x * kV8 + lane] = ((red[lane] + red[kV8 + lane]) + red[2 * kV8 + lane]) + red[3 * kV8 + lane];
}

// ---- MoE for 8 rows ----
// moe8_plan (one threadgroup, E threads): the distinct experts of the R rows' top-k (ascending ids).
// Experts used by more than `small` rows become MMA tiles 0 .. U-1 (all 8 rows computed, the live
// ones stored), the shared expert tile U; slot[u * 8 + t] = the top-k slot of row t that routes to
// tile u's expert (0xffffffff: row t does not use it; the shared tile: slot k for t < R). Experts
// used by 1 .. small (<= 2) rows become scalar tiles s (register-dequant matvecs over just those
// rows, no padded MMA work): sm[s] = expert, sm[64 + 2s + i] = row, sm[192 + 2s + i] = slot
// (0xffffffff past the tile's rows). ntiles = {U + 1, S}. Writes the indirect grids of the four
// expert passes: {gu_gx, U + 1, 1}, {dn_gx, U + 1, 1}, {sgu_gx, max(S, 1), 1}, {sdn_gx, max(S, 1), 1}.
struct Moe8PlanParams {
  uint R, k, E;
  uint gu_gx;   // grid x of the gate/up pass (I / 64)
  uint dn_gx;   // grid x of the down pass
  uint layer;   // stats[layer] = distinct routed experts, for diagnostics
  uint small;   // experts with <= small rows take the scalar passes (0: none, max 2)
  uint sgu_gx, sdn_gx;
};

[[kernel]] void moe8_plan(const device uint* ridx [[buffer(0)]], device uint* tiles [[buffer(1)]],
                          device uint* ntiles [[buffer(2)]], device uint* slot [[buffer(3)]],
                          device uint* args [[buffer(4)]], constant Moe8PlanParams& p [[buffer(5)]],
                          device uint* stats [[buffer(6)]], device uint* sm [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint ids[64];
  threadgroup uint cnt[2][32];
  threadgroup uint total[2];
  const uint n = p.R * p.k;
  if (tid < n) ids[tid] = ridx[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // thread e: its slots per row (rows use each expert at most once)
  const uint e = tid;
  uint my[8];
  uint rows = 0;
  for (uint t = 0; t < 8; ++t) {
    my[t] = 0xffffffffu;
    if (t < p.R && e < p.E)
      for (uint j = 0; j < p.k; ++j)
        if (ids[t * p.k + j] == e) my[t] = j;
    rows += my[t] != 0xffffffffu ? 1u : 0u;
  }
  const bool scal = rows > 0 && rows <= p.small, mma = rows > p.small;
  const uint pre_m = simd_prefix_exclusive_sum(mma ? 1u : 0u);
  const uint pre_s = simd_prefix_exclusive_sum(scal ? 1u : 0u);
  if (lane == 31) {
    cnt[0][sg] = pre_m + (mma ? 1u : 0u);
    cnt[1][sg] = pre_s + (scal ? 1u : 0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2) {
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
  if (scal) {
    const uint si = cnt[1][sg] + pre_s;
    sm[si] = e;
    uint i = 0;
    for (uint t = 0; t < 8; ++t)
      if (my[t] != 0xffffffffu) {
        sm[64 + 2 * si + i] = t;
        sm[192 + 2 * si + i] = my[t];
        ++i;
      }
    for (; i < 2; ++i) {
      sm[64 + 2 * si + i] = 0xffffffffu;
      sm[192 + 2 * si + i] = 0xffffffffu;
    }
  }
  const uint U = total[0], S = total[1];
  if (tid < 8) slot[U * 8 + tid] = tid < p.R ? p.k : 0xffffffffu;
  if (tid == 0) {
    tiles[U] = 0xffffffffu;
    ntiles[0] = U + 1;
    ntiles[1] = S;
    stats[p.layer] = U + S;
    args[0] = p.gu_gx; args[1] = U + 1; args[2] = 1;
    args[3] = p.dn_gx; args[4] = U + 1; args[5] = 1;
    args[6] = p.sgu_gx; args[7] = max(S, 1u); args[8] = 1;
    args[9] = p.sdn_gx; args[10] = max(S, 1u); args[11] = 1;
  }
}

// ---- scalar expert passes for experts used by 1-2 rows ----
// The weights are read once for the tile's rows (dequantized in registers: AND + convert per value,
// one FMA per value and row, x pre-scaled for the shift-free nibbles as qmv2_rows), so a lightly used
// expert costs its bytes, not an 8-row MMA tile. ROWS output columns per simdgroup.
template <int ROWS>
NI_INLINE void v8s_dot2(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                        const device float* x0, const device float* x1, uint K, uint row0, uint lane,
                        thread float (&acc)[ROWS][2]) {
  constexpr int VPL = 16;
  const uint wpr = K / 8, gpr = K / 64;
  for (int r = 0; r < ROWS; ++r) acc[r][0] = acc[r][1] = 0.0f;
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[2][VPL];
    float xs[2];
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < 2; ++t) {
      const device float* xr = t ? x1 : x0;
      xs[t] = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < VPL / 4; ++i) {
        const float4 v = ((const device float4*)(xr + k0))[i];
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

struct Moe8sParams {
  uint K, N;            // input width, output columns
  uint w_stride, s_stride;
  uint k;
  uint x_ld;            // floats per input row
};

// gate/up + SwiGLU of scalar tile s: hs[(2 s + i) * N + n] = silu(g) * u for its rows i < 2.
// grid (N / (2 * nsg), max(S, 1)); xn: [8][x_ld] normalized fp32 rows.
[[kernel]] void moe8s_gu(const device uint32_t* Wg [[buffer(0)]], const device bfloat* Sg [[buffer(1)]],
                         const device bfloat* Bg [[buffer(2)]], const device uint32_t* Wu [[buffer(3)]],
                         const device bfloat* Su [[buffer(4)]], const device bfloat* Bu [[buffer(5)]],
                         const device float* xn [[buffer(6)]], device float* hs [[buffer(7)]],
                         constant Moe8sParams& p [[buffer(8)]], const device uint* sm [[buffer(9)]],
                         const device uint* ntiles [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[1]) return;
  const uint e = sm[tg.y], t0 = sm[64 + 2 * tg.y], t1r = sm[64 + 2 * tg.y + 1];
  const bool two = t1r != 0xffffffffu;
  const uint t1 = two ? t1r : t0;
  constexpr int ROWS = 2;
  const uint row0 = (tg.x * nsg + sg) * ROWS;
  if (row0 >= p.N) return;
  const device float* xs = xn + t0 * p.x_ld;
  const device float* xs1 = xn + t1 * p.x_ld;
  float ag[ROWS][2], au[ROWS][2];
  v8s_dot2<ROWS>(Wg + e * p.w_stride, Sg + e * p.s_stride, Bg + e * p.s_stride, xs, xs1, p.K, row0, lane, ag);
  v8s_dot2<ROWS>(Wu + e * p.w_stride, Su + e * p.s_stride, Bu + e * p.s_stride, xs, xs1, p.K, row0, lane, au);
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r) {
      hs[(2 * tg.y) * p.N + row0 + r] = silu(ag[r][0]) * au[r][0];
      if (two) hs[(2 * tg.y + 1) * p.N + row0 + r] = silu(ag[r][1]) * au[r][1];
    }
}

// down of scalar tile s: out[(t * k + slot) * N + n] for its rows. grid (N / (2 * nsg), max(S, 1)).
[[kernel]] void moe8s_down(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],
                           const device bfloat* B [[buffer(2)]], const device float* hs [[buffer(3)]],
                           device float* out [[buffer(4)]], constant Moe8sParams& p [[buffer(5)]],
                           const device uint* sm [[buffer(6)]], const device uint* ntiles [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[1]) return;
  const uint e = sm[tg.y];
  const uint t0 = sm[64 + 2 * tg.y], t1r = sm[64 + 2 * tg.y + 1];
  const uint j0 = sm[192 + 2 * tg.y], j1 = sm[192 + 2 * tg.y + 1];
  const bool two = t1r != 0xffffffffu;
  constexpr int ROWS = 2;
  const uint row0 = (tg.x * nsg + sg) * ROWS;
  if (row0 >= p.N) return;
  const device float* xs = hs + (2 * tg.y) * p.K;
  const device float* xs1 = hs + (2 * tg.y + (two ? 1 : 0)) * p.K;
  float a[ROWS][2];
  v8s_dot2<ROWS>(W + e * p.w_stride, S + e * p.s_stride, B + e * p.s_stride, xs, xs1, p.K, row0, lane, a);
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r) {
      out[(t0 * p.k + j0) * p.N + row0 + r] = a[r][0];
      if (two) out[(t1r * p.k + j1) * p.N + row0 + r] = a[r][1];
    }
}

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

// ---- captures for a speculative drafter (DFlash2) ----
// capt[t][slot * D + i] = bf16(x[t][i]) for t < R: the residual stream after a target layer.
// grid (D / 256, R) x 256.
struct Capture8Params {
  uint D, ld, slot;
};
[[kernel]] void capture8(const device float* x [[buffer(0)]], device bfloat* capt [[buffer(1)]],
                         constant Capture8Params& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint i = tg.x * 256 + tid2.x, t = tg.y;
  if (i < p.D) capt[t * p.ld + p.slot * p.D + i] = bfloat(x[t * p.D + i]);
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
[[kernel]] void attn8m_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                               const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                               device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                               constant uint& R [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BK = 32, NC = BK / 8;
  constexpr uint KLD = attn11_kld<BK>();
  threadgroup uint4 kvm[attn11_kv<BK>() / 8];
  threadgroup half* kv = (threadgroup half*)kvm;
  threadgroup float2* xw = (threadgroup float2*)kvm;  // score exchange reuses the K tile
  const uint chunk = tg.x, kvh = tg.y;
  const uint G = p.Hq / p.Hkv;
  const uint h = sg / 2, dh = sg % 2, hq = kvh * G + h;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const uint t = frow;                              // this lane's row
  const uint qpos = p.pos0 + t;
  const uint start = chunk * p.chunk;
  const uint kend = min(start + p.chunk, p.pos0 + R);
  const uint ntiles = (kend - start + BK - 1) / BK;
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
    const bool diag = k0 + BK - 1 > p.pos0;          // tile reaches past row 0's position
    float mx = kNeg11;
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        float v = s[c].thread_elements()[i] * M_LOG2E_F;
        if (diag && k0 + c * 8 + fcol + i > qpos) v = kNeg11;
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
    const bool any = start <= qpos;  // the chunk holds at least one key this row sees
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
