#include "common.h"

// ==== Weight formats beyond the specialized 4-bit / 8-bit group-64 kernels ====
//
// A linear's format F:
//   F = 2..8  MLX affine F-bit: the codes of a row form a little-endian bit stream in uint32 words (value j of a
//             row at bits [F*j, F*j + F)), w = scale * q + bias per group of GS consecutive values, bf16 scales
//             and biases ([N, K/GS]). This is MLX's packing for every width (for 4 and 8 bits it is the familiar
//             "value j of a word at bits F*j"), see mlx/backend/metal/kernels/quantized.h.
//   F = 16    bf16 weights [N, K] (no scales, no biases): full precision.
// The kernels in the other files keep their specialized forms for 4-bit and 8-bit group-64 weights; the kernels
// here take any format, so every path of a model runs whatever its checkpoint holds (bf16, 8-bit, 6-bit, mixed).
// Name suffix of a format: _f16 (bf16) or _f{F}g{GS}, e.g. qmvg_f6g64_r2.
//
// Every lane works on runs of 16 consecutive values starting at a multiple of 16 (decode / verify) or of 8
// (prefill tiles): a run's codes span 16F bits at bit 16F*m, word aligned for even F and 16-bit aligned for odd F.

// words per row of K values
NI_INLINE uint wf_wpr(int F, uint K) { return F == 16 ? K / 2 : K * uint(F) / 32; }

// Extracts N codes of F bits from an aligned bit window w[0..] (code j at bits [F*j, F*j + F)).
template <int F, int N, int NW>
NI_INLINE void wf_extract(thread const uint (&w)[NW], thread uint* q) {
  constexpr uint M = (1u << F) - 1u;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < N; ++j) {
    const int b = F * j, wi = b / 32, s = b % 32;
    uint v = w[wi] >> s;
    if (s + F > 32) v |= w[wi + 1] << (32 - s);
    q[j] = v & M;
  }
}

// Codes [k0, k0 + 16) of a row (F <= 8, k0 % 16 == 0). May read one word past the run (tensors are padded).
template <int F>
NI_INLINE void wf_codes16(const device uint32_t* row, uint k0, thread uint* q) {
  constexpr uint M = (1u << F) - 1u;
  if (F == 4) {
    const uint2 w = *(const device uint2*)(row + k0 / 8);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 8; ++j) {
      q[j] = (w.x >> (4 * j)) & M;
      q[8 + j] = (w.y >> (4 * j)) & M;
    }
  } else if (F == 8) {
    const uint4 w = *(const device uint4*)(row + k0 / 4);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 4; ++j) q[4 * i + j] = (w[i] >> (8 * j)) & M;
  } else if (F % 2 == 0) {  // 2, 6: F/2 whole words
    constexpr int NW = F / 2 + 1;
    const device uint32_t* p = row + k0 * uint(F) / 32;
    uint w[NW];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NW - 1; ++i) w[i] = p[i];
    w[NW - 1] = 0u;
    wf_extract<F, 16, NW>(w, q);
  } else {  // 3, 5, 7: a (F+1)/2-word window, shifted down by 16 bits when the run starts mid-word
    constexpr int NW = (F + 1) / 2 + 1;
    const uint bit0 = uint(F) * k0;
    const device uint32_t* p = row + bit0 / 32;
    uint w[NW];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NW - 1; ++i) w[i] = p[i];
    w[NW - 1] = 0u;
    if (bit0 & 16u)
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < NW - 1; ++i) w[i] = (w[i] >> 16) | (w[i + 1] << 16);
    wf_extract<F, 16, NW>(w, q);
  }
}

// Codes [k0, k0 + 8) of a row (F <= 8, k0 % 8 == 0): 8F bits at a byte-aligned offset, within two words.
template <int F>
NI_INLINE void wf_codes8(const device uint32_t* row, uint k0, thread uint* q) {
  constexpr uint M = (1u << F) - 1u;
  if (F == 4) {
    const uint w = row[k0 / 8];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 8; ++j) q[j] = (w >> (4 * j)) & M;
  } else if (F == 8) {
    const uint2 w = *(const device uint2*)(row + k0 / 4);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 4; ++j) {
      q[j] = (w.x >> (8 * j)) & M;
      q[4 + j] = (w.y >> (8 * j)) & M;
    }
  } else {
    const uint bit0 = uint(F) * k0, sh = bit0 % 32;
    const device uint32_t* p = row + bit0 / 32;
    const uint a = p[0], b = (sh + 8 * F > 32) ? p[1] : 0u;
    uint w[3];
    w[0] = sh ? (a >> sh) | (b << (32 - sh)) : a;
    w[1] = sh ? (b >> sh) : b;
    w[2] = 0u;
    wf_extract<F, 8, 3>(w, q);
  }
}

// Values [k0, k0 + 16) of a row as floats: integer codes (F <= 8) or the bf16 weights (F = 16).
template <int F>
NI_INLINE void wf_vals16(const device uint32_t* row, uint k0, thread float* v) {
  if (F == 16) {
    const device uint4* p = (const device uint4*)(row + k0 / 2);
    const uint4 a = p[0], b = p[1];
    const uint w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 8; ++i) {
      v[2 * i] = as_type<float>(w[i] << 16);
      v[2 * i + 1] = as_type<float>(w[i] & 0xFFFF0000u);
    }
  } else {
    uint q[16];
    wf_codes16<F>(row, k0, q);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 16; ++j) v[j] = float(q[j]);
  }
}

// One weight of a row (F, GS): for lookups (embeddings).
template <int F>
NI_INLINE float wf_one(const device uint32_t* W, const device bfloat* S, const device bfloat* B, uint row, uint k,
                       uint K, uint GS) {
  if (F == 16) {
    const device ushort* p = (const device ushort*)W;
    return as_type<float>(uint(p[row * K + k]) << 16);
  }
  const uint bit = uint(F) * k, sh = bit % 32;
  const device uint32_t* p = W + row * wf_wpr(F, K) + bit / 32;
  uint v = p[0] >> sh;
  if (sh + F > 32) v |= p[1] << (32 - sh);
  const uint gi = row * (K / GS) + k / GS;
  return float(S[gi]) * float(v & ((1u << F) - 1u)) + float(B[gi]);
}

// ---- decode / small-batch dot products ----
// acc[r] += W[rows[r], kt + k] . xs[k] for k < tile (x staged in threadgroup memory; no simd reduction).
template <int F, int GS, int ROWS>
NI_INLINE void wf_rows_acc(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                           threadgroup const float* xs, uint K, uint kt, uint tile, thread const uint* rows, uint lane,
                           thread float* acc) {
  const uint wpr = wf_wpr(F, K), gpr = F == 16 ? 1u : K / uint(GS);
  for (uint k0 = lane * 16; k0 < tile; k0 += 512) {
    float xv[16];
    float xsum = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i) {
      const float4 v = ((threadgroup const float4*)(xs + k0))[i];
      xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;
      xsum += v.x + v.y + v.z + v.w;
    }
    const uint kg = kt + k0;
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < ROWS; ++r) {
      float wv[16];
      wf_vals16<F>(w + rows[r] * wpr, kg, wv);
      float d = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 16; ++j) d = fma(wv[j], xv[j], d);
      if (F == 16) {
        acc[r] += d;
      } else {
        const uint gi = rows[r] * gpr + kg / uint(GS);
        acc[r] += float(scales[gi]) * d + float(biases[gi]) * xsum;
      }
    }
  }
}

// y[n] (+)= W[n] . (norm?) x for one input vector, any K (x staged in 4096-wide tiles, like qmv3).
// QmvParams / buffers as qmv2_* and qmv3_*: w, s, b, x, y, params, resid, norm weight.
template <int F, int GS, int ROWS, bool ADD, bool NORM>
NI_INLINE void qmvg_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         const device float* x, device float* y, uint N, uint K, float eps, const device float* resid,
                         const device bfloat* nw, threadgroup float* xs, threadgroup float* red, uint row0, uint tid,
                         uint ntg, uint lane) {
  const float inv = NORM ? rms_inv_device(x, K, eps, red, tid, ntg) : 1.0f;
  uint rows[ROWS];
  float acc[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, N - 1);
    acc[r] = 0.0f;
  }
  for (uint kt = 0; kt < K; kt += 4096) {
    const uint tile = min(4096u, K - kt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_tile(x, kt, tile, NORM, inv, nw, xs, tid, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < N) wf_rows_acc<F, GS, ROWS>(w, scales, biases, xs, K, kt, tile, rows, lane, acc);
  }
  if (row0 >= N) return;
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    const uint row = row0 + r;
    if (lane == 0 && row < N) {
      if (ADD) v += resid[row];
      y[row] = v;
    }
  }
}

#define NI_QMVG(NAME, F, GS, ROWS, ADD, NORM)                                                                \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]],    \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],        \
                       device float* y [[buffer(4)]], constant QmvParams& p [[buffer(5)]],                    \
                       const device float* resid [[buffer(6)]], const device bfloat* nw [[buffer(7)]],        \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],    \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {     \
    threadgroup float xs[4096];                                                                              \
    threadgroup float red[32];                                                                               \
    const uint row0 = (tgp * nsg + sg) * ROWS;                                                               \
    qmvg_impl<F, GS, ROWS, ADD, NORM>(w, scales, biases, x, y, p.N, p.K, p.eps, resid, nw, xs, red, row0, tid, \
                                      ntg, lane);                                                            \
  }

// Multi-segment (up to 4 matrices of one format sharing x; QmvMulti2Params, buffers as qmv3_multi_r2):
// optional input norm, optional normalized-x output.
template <int F, int GS, int ROWS>
NI_INLINE void qmvg_multi_impl(const device uint32_t* const thread* Wv, const device bfloat* const thread* Sv,
                               const device bfloat* const thread* Bv, device float* const thread* Yv,
                               const device float* x, constant QmvMulti2Params& p, const device bfloat* nw,
                               device float* xn_out, threadgroup float* xs, threadgroup float* red, uint tgp, uint tid,
                               uint ntg, uint sg, uint nsg, uint lane) {
  const bool norm = p.norm != 0;
  const float inv = norm ? rms_inv_device(x, p.K, p.eps, red, tid, ntg) : 1.0f;
  if (p.write_xn && tgp == 0)
    for (uint i = tid; i < p.K; i += ntg) xn_out[i] = norm ? x[i] * inv * float(nw[i]) : x[i];
  uint seg = 0;
  for (uint i = 1; i < p.nseg; ++i)
    if (tgp >= p.tg_begin[i]) seg = i;
  const uint N = p.N[seg];
  const uint row0 = ((tgp - p.tg_begin[seg]) * nsg + sg) * ROWS;
  uint rows[ROWS];
  float acc[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, N - 1);
    acc[r] = 0.0f;
  }
  for (uint kt = 0; kt < p.K; kt += 4096) {
    const uint tile = min(4096u, p.K - kt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_tile(x, kt, tile, norm, inv, nw, xs, tid, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < N) wf_rows_acc<F, GS, ROWS>(Wv[seg], Sv[seg], Bv[seg], xs, p.K, kt, tile, rows, lane, acc);
  }
  if (row0 >= N) return;
  for (int r = 0; r < ROWS; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0 && row0 + r < N) Yv[seg][row0 + r] = v;
  }
}

#define NI_QMVG_MULTI(NAME, F, GS)                                                                         \
  [[kernel]] void NAME(                                                                                    \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                      \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                               \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                      \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                               \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                      \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                             \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                    \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                             \
      const device float* x [[buffer(16)]], constant QmvMulti2Params& p [[buffer(17)]],                    \
      const device bfloat* nw [[buffer(18)]], device float* xn_out [[buffer(19)]],                         \
      uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                 \
      uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                    \
      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                  \
    threadgroup float xs[4096];                                                                             \
    threadgroup float red[32];                                                                              \
    const device uint32_t* Wv[4] = {w0, w1, w2, w3};                                                        \
    const device bfloat* Sv[4] = {s0, s1, s2, s3};                                                          \
    const device bfloat* Bv[4] = {b0, b1, b2, b3};                                                          \
    device float* Yv[4] = {y0, y1, y2, y3};                                                                 \
    qmvg_multi_impl<F, GS, 2>(Wv, Sv, Bv, Yv, x, p, nw, xn_out, xs, red, tgp, tid, ntg, sg, nsg, lane);     \
  }

// Dense SwiGLU gate+up for one vector (as mlp_gate_up): h[i] = silu(gate_i . xn) * (up_i . xn), xn = rmsnorm(x).
template <int F, int GS>
NI_INLINE void mlpg_gate_up_impl(const device uint32_t* gw, const device bfloat* gs, const device bfloat* gb,
                                 const device uint32_t* uw, const device bfloat* us, const device bfloat* ub,
                                 const device float* x, const device bfloat* nw, device float* h,
                                 constant QmvParams& p, threadgroup float* xs, threadgroup float* red, uint tgp,
                                 uint tid, uint ntg, uint sg, uint nsg, uint lane) {
  constexpr int ROWS = 2;
  const float inv = rms_inv_device(x, p.K, p.eps, red, tid, ntg);
  const uint row0 = (tgp * nsg + sg) * ROWS;
  uint rows[ROWS];
  float ag[ROWS], au[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, p.N - 1);
    ag[r] = au[r] = 0.0f;
  }
  for (uint kt = 0; kt < p.K; kt += 4096) {
    const uint tile = min(4096u, p.K - kt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_tile(x, kt, tile, true, inv, nw, xs, tid, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < p.N) {
      wf_rows_acc<F, GS, ROWS>(gw, gs, gb, xs, p.K, kt, tile, rows, lane, ag);
      wf_rows_acc<F, GS, ROWS>(uw, us, ub, xs, p.K, kt, tile, rows, lane, au);
    }
  }
  if (row0 >= p.N) return;
  for (int r = 0; r < ROWS; ++r) {
    const float g = simd_sum(ag[r]), u = simd_sum(au[r]);
    if (lane == 0 && row0 + r < p.N) h[row0 + r] = silu(g) * u;
  }
}

#define NI_MLPG(NAME, F, GS)                                                                                 \
  [[kernel]] void NAME(const device uint32_t* gw [[buffer(0)]], const device bfloat* gs [[buffer(1)]],       \
                       const device bfloat* gb [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],       \
                       const device bfloat* us [[buffer(4)]], const device bfloat* ub [[buffer(5)]],         \
                       const device float* x [[buffer(6)]], const device bfloat* nw [[buffer(7)]],           \
                       device float* h [[buffer(8)]], constant QmvParams& p [[buffer(9)]],                   \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],  \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],    \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {   \
    threadgroup float xs[4096];                                                                             \
    threadgroup float red[32];                                                                              \
    mlpg_gate_up_impl<F, GS>(gw, gs, gb, uw, us, ub, x, nw, h, p, xs, red, tgp, tid, ntg, sg, nsg, lane);   \
  }

// ---- token embedding: x[t] = E[tokens[tok_off + t]] (EmbedParams; grid: (D/8 + 63)/64 x T, 64 threads) ----
template <int F>
NI_INLINE void embedg_impl(const device uint32_t* w, const device bfloat* s, const device bfloat* b,
                           const device int* tokens, device float* x, constant EmbedParams& p, uint2 gid) {
  const uint i = gid.x, t = gid.y;
  if (i >= p.D / 8) return;
  const uint tok = uint(tokens[p.tok_off + t]);
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) x[t * p.D + i * 8 + j] = wf_one<F>(w, s, b, tok, i * 8 + j, p.D, p.gs);
}

#define NI_EMBEDG(NAME, F)                                                                                   \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],         \
                       const device bfloat* b [[buffer(2)]], const device int* tokens [[buffer(3)]],         \
                       device float* x [[buffer(4)]], constant EmbedParams& p [[buffer(5)]],                 \
                       uint2 gid [[thread_position_in_grid]]) {                                              \
    embedg_impl<F>(w, s, b, tokens, x, p, gid);                                                              \
  }

// DFlash2's block embedding + norm (dfl_embed_norm) over a target embedding table in format F.
template <int F>
NI_INLINE void dfl_embed_normg_impl(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                                    const device int* tokens, device float* h, const device bfloat* nw,
                                    device float* n, device half* xh, device float* xs, constant DflEmbedParams& p,
                                    const device uint* gst, uint r, uint tid, uint ntg, uint sg, uint nsg,
                                    uint lane, threadgroup float* red) {
  const uint H = p.H;
  const uint tok = r == 0 ? uint(tokens[dfl_pos(p.tok_off, gst)]) : p.mask;
  device float* hr = h + r * H;
  float ss = 0.0f;
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint k = g * 64 + 2 * lane;
    const float2 v = float2(wf_one<F>(W, S, B, tok, k, H, p.gs), wf_one<F>(W, S, B, tok, k + 1, H, p.gs));
    *(device float2*)(hr + k) = v;
    ss += v.x * v.x + v.y * v.y;
  }
  ss = tg_sum(ss, red, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(H) + p.eps);
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint k = g * 64 + 2 * lane;
    const float2 v = *(device float2*)(hr + k);
    const float2 o = float2(v.x * inv * float(nw[k]), v.y * inv * float(nw[k + 1]));
    *(device float2*)(n + r * H + k) = o;
    dfl_put2(xh, xs, r, H, g, lane, o.x, o.y);
  }
}

#define NI_DFL_EMBEDG(NAME, F)                                                                               \
  [[kernel]] void NAME(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],         \
                       const device bfloat* B [[buffer(2)]], const device int* tokens [[buffer(3)]],         \
                       device float* h [[buffer(4)]], const device bfloat* nw [[buffer(5)]],                 \
                       device float* n [[buffer(6)]], device half* xh [[buffer(7)]],                         \
                       device float* xs [[buffer(8)]], constant DflEmbedParams& p [[buffer(9)]],             \
                       const device uint* gst [[buffer(10)]], uint r [[threadgroup_position_in_grid]],       \
                       uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],       \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],  \
                       uint lane [[thread_index_in_simdgroup]]) {                                            \
    threadgroup float red[32];                                                                               \
    dfl_embed_normg_impl<F>(W, S, B, tokens, h, nw, n, xh, xs, p, gst, r, tid, ntg, sg, nsg, lane, red);      \
  }

// ---- per-row MoE (the verify.metal moe_*_rows layout) with any expert / shared-expert formats ----
// gate/up grid: (row tiles over max(I, SI), k + 1 slots, R rows); xn [R, D]; h [R, k+1, Imax]; idx [R, k].
template <int FE, int GE, int FS, int GSS>
NI_INLINE void moeg_gu_rows_impl(const device uint32_t* gw, const device bfloat* gsc, const device bfloat* gbi,
                                 const device uint32_t* uw, const device bfloat* usc, const device bfloat* ubi,
                                 const device uint32_t* sgw, const device bfloat* sgs, const device bfloat* sgb,
                                 const device uint32_t* suw, const device bfloat* sus, const device bfloat* sub,
                                 const device float* xn, device float* h, constant MoeParams& p, const device uint* idx,
                                 uint3 tgp, uint tid, uint ntg, uint sg, uint nsg, uint lane, threadgroup float* xs) {
  constexpr int ROWS = 2;
  const uint slot = tgp.y, r = tgp.z;
  const bool shared = slot == p.k;
  for (uint i = tid; i < p.D / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)(xn + r * p.D))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D;
  uint rows[ROWS];
  for (int q = 0; q < ROWS; ++q) rows[q] = min(row0 + q, N - 1);
  float ag[ROWS] = {0.0f, 0.0f}, au[ROWS] = {0.0f, 0.0f};
  if (shared) {
    wf_rows_acc<FS, GSS, ROWS>(sgw, sgs, sgb, xs, K, 0, K, rows, lane, ag);
    wf_rows_acc<FS, GSS, ROWS>(suw, sus, sub, xs, K, 0, K, rows, lane, au);
  } else {
    const uint e = idx[r * p.k + slot];
    const uint wo = e * N * wf_wpr(FE, K), so = FE == 16 ? 0u : e * N * (K / uint(GE));
    wf_rows_acc<FE, GE, ROWS>(gw + wo, gsc + so, gbi + so, xs, K, 0, K, rows, lane, ag);
    wf_rows_acc<FE, GE, ROWS>(uw + wo, usc + so, ubi + so, xs, K, 0, K, rows, lane, au);
  }
  const uint Imax = max(p.I, p.SI);
  for (int q = 0; q < ROWS; ++q) {
    const float g = simd_sum(ag[q]), u = simd_sum(au[q]);
    if (lane == 0 && row0 + q < N) h[(r * (p.k + 1) + slot) * Imax + row0 + q] = silu(g) * u;
  }
}

#define NI_MOEG_GU_ROWS(NAME, FE, GE, FS, GSS)                                                                \
  [[kernel]] void NAME(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],       \
                       const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],       \
                       const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],        \
                       const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],      \
                       const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],      \
                       const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],      \
                       const device float* xn [[buffer(12)]], device float* h [[buffer(13)]],                 \
                       constant MoeParams& p [[buffer(14)]], const device uint* idx [[buffer(15)]],           \
                       uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],  \
                       uint3 ntg3 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],    \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {     \
    threadgroup float xs[4096];                                                                              \
    moeg_gu_rows_impl<FE, GE, FS, GSS>(gw, gsc, gbi, uw, usc, ubi, sgw, sgs, sgb, suw, sus, sub, xn, h, p, idx, \
                                       tgp, tid, ntg3.x, sg, nsg, lane, xs);                                  \
  }

// down grid: (row tiles over D, R): x[r, row] += sum_slot wts[r, slot] * (down_slot[row] . h[r, slot]).
template <int FE, int GE, int FD, int GD>
NI_INLINE void moeg_down_rows_impl(const device uint32_t* dw, const device bfloat* dsc, const device bfloat* dbi,
                                   const device uint32_t* sdw, const device bfloat* sds, const device bfloat* sdb,
                                   const device float* h, const device uint* idx, const device float* wts,
                                   device float* x, constant MoeRowsDownParams& p, uint2 tgp, uint tid, uint ntg,
                                   uint sg, uint nsg, uint lane, threadgroup float* hs) {
  constexpr int ROWS = 2;
  const uint r = tgp.y;
  const uint Imax = max(p.I, p.SI);
  for (uint i = tid; i < (p.k + 1) * Imax / 4; i += ntg)
    ((threadgroup float4*)hs)[i] = ((const device float4*)(h + r * (p.k + 1) * Imax))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= p.D) return;
  uint rows[ROWS];
  for (int q = 0; q < ROWS; ++q) rows[q] = min(row0 + q, p.D - 1);
  float tot[ROWS] = {0.0f, 0.0f};
  for (uint slot = 0; slot <= p.k; ++slot) {
    float a[ROWS] = {0.0f, 0.0f};
    if (slot == p.k) {
      wf_rows_acc<FD, GD, ROWS>(sdw, sds, sdb, hs + slot * Imax, p.SI, 0, p.SI, rows, lane, a);
    } else {
      const uint e = idx[r * p.k + slot];
      const uint wo = e * p.D * wf_wpr(FE, p.I), so = FE == 16 ? 0u : e * p.D * (p.I / uint(GE));
      wf_rows_acc<FE, GE, ROWS>(dw + wo, dsc + so, dbi + so, hs + slot * Imax, p.I, 0, p.I, rows, lane, a);
    }
    const float ws = wts[r * (p.k + 1) + slot];
    for (int q = 0; q < ROWS; ++q) tot[q] += ws * a[q];
  }
  for (int q = 0; q < ROWS; ++q) {
    const float v = simd_sum(tot[q]);
    if (lane == 0 && row0 + q < p.D) x[r * p.D + row0 + q] += v;
  }
}

#define NI_MOEG_DOWN_ROWS(NAME, FE, GE, FD, GD, HS)                                                             \
  [[kernel]] void NAME(const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],       \
                       const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],      \
                       const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],        \
                       const device float* h [[buffer(6)]], const device uint* idx [[buffer(7)]],             \
                       const device float* wts [[buffer(8)]], device float* x [[buffer(9)]],                  \
                       constant MoeRowsDownParams& p [[buffer(10)]], uint2 tgp [[threadgroup_position_in_grid]], \
                       uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]],      \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                             \
    threadgroup float hs[HS];                                                                                \
    moeg_down_rows_impl<FE, GE, FD, GD>(dw, dsc, dbi, sdw, sds, sdb, h, idx, wts, x, p, tgp, tid, ntg2.x, sg, nsg, \
                                        lane, hs);                                                           \
  }

// ---- 8-row verify GEMM (verify8.metal's qmm8 layout: X8 tables, Qmm8Params, buffers 0-20) for any format ----
// Lane (fm, fn): weight row fm of the 8-row block, its 16 inputs 64g + 16q (q = fn/2) of each 64-group g, as
// qmm8_*. F = 16: the bf16 weights as half A operands, the MMA chain runs over all groups. F <= 8 with groups of
// >= 64: exact code operands 1024 + q and qmm8's per-group fp32 epilogue s * dot + (b - 1024 s) * sum(x).
// Groups of 32 (the lanes of a row hold two groups): codes dequantized to half (s * q + b) before the MMA.
template <int F, int GS, int NF>
NI_INLINE void v8g_run(const device uint32_t* const thread* wrow, const device ushort* const thread* srow,
                       const device ushort* const thread* brow, const device half* table, const device float* sums,
                       uint g0, uint g1, uint lane, uint fn, uint q, thread float2* acc) {
  for (uint g = g0; g < g1; ++g) {
    const device half4* tb = (const device half4*)(table + (g * 32 + lane) * 16);
    const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
    const half2 b[8] = {x0.xy, x0.zw, x1.xy, x1.zw, x2.xy, x2.zw, x3.xy, x3.zw};
    const uint k0 = g * 64 + 16 * q;
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      if (F == 16) {
        const device uint4* p = (const device uint4*)(wrow[f] + k0 / 2);
        const uint4 u0 = p[0], u1 = p[1];
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 8; ++j) {
          const uint4 u = j < 4 ? u0 : u1;
          const int jj = j % 4;
          const uint w0 = u[jj / 2], w1 = u[2 + jj / 2];
          const float a0 = (jj % 2) ? as_type<float>(w0 & 0xFFFF0000u) : as_type<float>(w0 << 16);
          const float a1 = (jj % 2) ? as_type<float>(w1 & 0xFFFF0000u) : as_type<float>(w1 << 16);
          v8_mma(acc[f], half2(half(a0), half(a1)), b[j]);
        }
      } else {
        uint c[16];
        wf_codes16<F>(wrow[f], k0, c);
        if (GS >= 64) {
          float2 dot = float2(0.0f);
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < 8; ++j) {
            const int i0 = 8 * (j / 4) + j % 4;
            v8_mma(dot, as_type<half2>(c[i0] | (c[i0 + 4] << 16) | 0x64006400u), b[j]);
          }
          const uint gi = g * 64 / uint(GS);
          const float s = v8_bf(srow[f][gi]), bi = v8_bf(brow[f][gi]);
          const float2 sx = *(const device float2*)(sums + g * kV8 + fn);
          acc[f] = fma(dot, float2(s), acc[f]);
          acc[f] = fma(sx, float2(fma(-1024.0f, s, bi)), acc[f]);
        } else {
          const uint gi = k0 / uint(GS);
          const half2 s2 = half2(half(v8_bf(srow[f][gi]))), b2 = half2(half(v8_bf(brow[f][gi])));
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < 8; ++j) {
            const int i0 = 8 * (j / 4) + j % 4;
            const half2 cv = as_type<half2>(c[i0] | (c[i0 + 4] << 16) | 0x64006400u) - half2(1024.0h);
            v8_mma(acc[f], fma(cv, s2, b2), b[j]);
          }
        }
      }
    }
  }
}

template <int F, int GS, int NF>
NI_INLINE void qmm8g_impl(const device uint32_t* const thread* Wv, const device ushort* const thread* Sv,
                          const device ushort* const thread* Bv, device float* const thread* Yv,
                          const device half* table, const device float* sums, constant Qmm8Params& p,
                          threadgroup float2* red, uint tg, uint sg, uint nsg, uint lane, uint ntgs) {
  // row tiles as qmm8 (verify8.metal v8_tiles): this threadgroup's tile and its tables
  const uint nt = v8_tiles(ntgs, p.tg_begin[p.nseg]);
  const uint tile = nt > 1 ? tg % nt : 0u;
  if (nt > 1) {
    tg /= nt;
    table += tile * (p.K / 64) * 512;
    sums += tile * (p.K / 64) * kV8;
  }
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint cs = nsg / p.ks, kslice = sg % p.ks;
  const uint n0 = ((tg - p.tg_begin[si]) * cs + sg / p.ks) * 8 * NF;
  const uint N = p.N[si], gpr = p.K / 64, wpr = wf_wpr(F, p.K), spr = F == 16 ? 1u : p.K / uint(GS);
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* wrow[NF];
  const device ushort* srow[NF];
  const device ushort* brow[NF];
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    const uint row = min(n0 + 8 * f + fm, N - 1);
    wrow[f] = Wv[si] + row * wpr;
    srow[f] = Sv[si] + row * spr;
    brow[f] = Bv[si] + row * spr;
    acc[f] = float2(0.0f);
  }
  v8g_run<F, GS, NF>(wrow, srow, brow, table, sums, g0, g1, lane, fn, q, acc);
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

#define NI_QMM8G(NAME, F, GS, NF)                                                                            \
  [[kernel]] void NAME(const device uint32_t* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],       \
                       const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                \
                       const device uint32_t* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],       \
                       const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                \
                       const device uint32_t* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],       \
                       const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],              \
                       const device uint32_t* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],     \
                       const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],              \
                       const device half* table [[buffer(16)]], const device float* sums [[buffer(17)]],     \
                       constant Qmm8Params& p [[buffer(18)]], uint tg [[threadgroup_position_in_grid]],      \
                       uint ntgs [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {   \
    threadgroup float2 red[8 * NF * 32];                                                                     \
    const device uint32_t* Wv[4] = {W0, W1, W2, W3};                                                          \
    const device ushort* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device ushort* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmm8g_impl<F, GS, NF>(Wv, Sv, Bv, Yv, table, sums, p, red, tg, sg, nsg, lane, ntgs);                     \
  }

// ---- 8-row MoE verify (verify8.metal's moe8_* passes: experts de-duplicated over the 8 rows) for any formats ----
// moe8g_gu: gate/up + SwiGLU of every touched expert's tile (moe8_plan's tiles; the shared expert is tile
// 0xffffffff) into its h table, as qmm8_gu1k2 (8 column slices x KS k-slices); routed experts in FE/GE, the shared
// expert in FS/GSS. Qmm8GuParams: w_stride / s_stride in words / groups of the routed format.
template <int FE, int GE, int FS, int GSS, int KS>
NI_INLINE void moe8g_gu_impl(const device uint32_t* Wg, const device ushort* Sg, const device ushort* Bg,
                             const device uint32_t* Wu, const device ushort* Su, const device ushort* Bu,
                             const device uint32_t* SWg, const device ushort* SSg, const device ushort* SBg,
                             const device uint32_t* SWu, const device ushort* SSu, const device ushort* SBu,
                             const device half* table, const device float* sums, device half* hT, device float* hS,
                             constant Qmm8GuParams& p, const device uint* tiles, const device uint* ntiles, uint2 tg,
                             uint sg, uint lane, threadgroup float* red, threadgroup float2* kred) {
  constexpr uint NSG = 8;
  if (tg.y >= ntiles[0]) return;
  const uint e = tiles[tg.y];
  const bool shared = e == 0xffffffffu;
  const uint gpr = p.K / 64;
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint ci = sg / KS, kslice = sg % KS;
  const uint row = tg.x * 64 + ci * 8 + fm;
  const uint gper = (gpr + KS - 1) / KS, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  float2 acc[2] = {float2(0.0f), float2(0.0f)};
  if (shared) {
    const uint wpr = wf_wpr(FS, p.K), spr = FS == 16 ? 1u : p.K / uint(GSS);
    const device uint32_t* wr[2] = {SWg + row * wpr, SWu + row * wpr};
    const device ushort* sr[2] = {SSg + row * spr, SSu + row * spr};
    const device ushort* br[2] = {SBg + row * spr, SBu + row * spr};
    v8g_run<FS, GSS, 2>(wr, sr, br, table, sums, g0, g1, lane, fn, q, acc);
  } else {
    const uint wpr = wf_wpr(FE, p.K), spr = FE == 16 ? 1u : p.K / uint(GE);
    const uint wo = e * p.w_stride, so = e * p.s_stride;
    const device uint32_t* wr[2] = {Wg + wo + row * wpr, Wu + wo + row * wpr};
    const device ushort* sr[2] = {Sg + so + row * spr, Su + so + row * spr};
    const device ushort* br[2] = {Bg + so + row * spr, Bu + so + row * spr};
    v8g_run<FE, GE, 2>(wr, sr, br, table, sums, g0, g1, lane, fn, q, acc);
  }
  if (KS > 1) v8_reduce<2>(acc, kred, KS, sg, lane);
  const bool lead = kslice == 0;
  device half* ht = hT + tg.y * p.h_stride;
  float2 ps = float2(0.0f);
  if (lead) {
    const uint kk = ci * 8 + fm;
    _Pragma("clang loop unroll(full)")
    for (int k2 = 0; k2 < 2; ++k2) {
      const half h = half(silu(acc[0][k2]) * acc[1][k2]);
      ht[v8_tab(tg.x, kk, fn + k2)] = h;
      ps[k2] += float(h);
    }
  }
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

#define NI_MOE8G_GU(NAME, FE, GE, FS, GSS)                                                                     \
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
    threadgroup float2 kred[8 * 2 * 2 * 32];                                                                   \
    moe8g_gu_impl<FE, GE, FS, GSS, 2>(Wg, Sg, Bg, Wu, Su, Bu, SWg, SSg, SBg, SWu, SSu, SBu, table, sums, hT, hS, p, \
                                      tiles, ntiles, tg, sg, lane, red, kred);                                 \
  }

// moe8g_down: each tile's down projection to per (row, slot) outputs (as moe8_down_nf2), any formats.
template <int FE, int GE, int FD, int GD, int NF>
NI_INLINE void moe8g_down_impl(const device uint32_t* W, const device ushort* S, const device ushort* B,
                               const device uint32_t* SW, const device ushort* SS, const device ushort* SB,
                               const device half* hT, const device float* hS, device float* out,
                               device float* sh_out, constant Moe8DownParams& p, const device uint* tiles,
                               const device uint* ntiles, const device uint* slot, uint2 tg, uint sg, uint nsg,
                               uint lane) {
  if (tg.y >= ntiles[0]) return;
  const uint e = tiles[tg.y];
  const bool shared = e == 0xffffffffu;
  const uint gpr = p.K / 64;
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint n0 = (tg.x * nsg + sg) * 8 * NF;
  const device half* ht = hT + tg.y * p.h_stride;
  const device float* hs = hS + tg.y * p.hs_stride;
  float2 acc[NF];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) acc[f] = float2(0.0f);
  if (shared) {
    const uint wpr = wf_wpr(FD, p.K), spr = FD == 16 ? 1u : p.K / uint(GD);
    const device uint32_t* wr[NF];
    const device ushort* sr[NF];
    const device ushort* br[NF];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      const uint row = n0 + 8 * f + fm;
      wr[f] = SW + row * wpr;
      sr[f] = SS + row * spr;
      br[f] = SB + row * spr;
    }
    v8g_run<FD, GD, NF>(wr, sr, br, ht, hs, 0, gpr, lane, fn, q, acc);
  } else {
    const uint wpr = wf_wpr(FE, p.K), spr = FE == 16 ? 1u : p.K / uint(GE);
    const device uint32_t* wr[NF];
    const device ushort* sr[NF];
    const device ushort* br[NF];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) {
      const uint row = n0 + 8 * f + fm;
      wr[f] = W + e * p.w_stride + row * wpr;
      sr[f] = S + e * p.s_stride + row * spr;
      br[f] = B + e * p.s_stride + row * spr;
    }
    v8g_run<FE, GE, NF>(wr, sr, br, ht, hs, 0, gpr, lane, fn, q, acc);
  }
  _Pragma("clang loop unroll(full)")
  for (int e2 = 0; e2 < 2; ++e2) {
    const uint t = fn + e2, j = slot[tg.y * 8 + t];
    if (j == 0xffffffffu) continue;
    device float* o = shared ? sh_out + t * p.N : out + (t * p.k + j) * p.N;
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < NF; ++f) o[n0 + 8 * f + fm] = acc[f][e2];
  }
}

#define NI_MOE8G_DOWN(NAME, FE, GE, FD, GD)                                                                    \
  [[kernel]] void NAME(const device uint32_t* W [[buffer(0)]], const device ushort* S [[buffer(1)]],           \
                       const device ushort* B [[buffer(2)]], const device uint32_t* SW [[buffer(3)]],          \
                       const device ushort* SS [[buffer(4)]], const device ushort* SB [[buffer(5)]],           \
                       const device half* hT [[buffer(6)]], const device float* hS [[buffer(7)]],              \
                       device float* out [[buffer(8)]], device float* sh_out [[buffer(9)]],                    \
                       constant Moe8DownParams& p [[buffer(10)]], const device uint* tiles [[buffer(11)]],     \
                       const device uint* ntiles [[buffer(12)]], const device uint* slot [[buffer(13)]],       \
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {     \
    moe8g_down_impl<FE, GE, FD, GD, 2>(W, S, B, SW, SS, SB, hT, hS, out, sh_out, p, tiles, ntiles, slot, tg, sg, nsg, \
                                       lane);                                                                  \
  }

// ---- prefill GEMMs on the qmr design (qmm3.metal) for any format: qrf_* ----
// Same tiles, lane layout and X staging as qmr (BK = 32: the lane's 8 weights of a row per K step are
// k0 + 8c .. + 7). The weights go to registers as the four A pairs (w_f, w_{f+4}) of the lane's fragments:
// bf16 converted to half pairs at load; F <= 8 codes as pair words q_f | q_{f+4} << 16, made into the half
// 0.25 + q/4096 (exact for q < 1024) and dequantized with one half2 fma (scale * 4096, bias), as qr_pair.
// MFAST (the bf16 kernels): grid x runs over the M tiles, so the threadgroups running together share a
// weight tile from the cache instead of streaming the whole matrix once per M tile (bf16 weights are 4x
// the bytes of 4-bit ones).
NI_INLINE float bf_lo(uint w) { return as_type<float>(w << 16); }
NI_INLINE float bf_hi(uint w) { return as_type<float>(w & 0xFFFF0000u); }

template <int F, int GS, int FA, int NP>
NI_INLINE void qrf_load_w(const device uint32_t* const thread* W, const device bfloat* const thread* S,
                          const device bfloat* const thread* B, thread const uint* wrow, uint wpr, uint spr, uint k0,
                          uint c, thread uint* wr, thread half2* sb) {
  const uint kk = k0 + 8 * c;
  _Pragma("clang loop unroll(full)")
  for (int p = 0; p < NP; ++p)
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      const int i = p * FA + a;
      const device uint32_t* row = W[p] + wrow[a] * wpr;
      if (F == 16) {
        const uint4 u = *(const device uint4*)(row + kk / 2);
        wr[i * 4 + 0] = as_type<uint>(half2(half(bf_lo(u.x)), half(bf_lo(u.z))));
        wr[i * 4 + 1] = as_type<uint>(half2(half(bf_hi(u.x)), half(bf_hi(u.z))));
        wr[i * 4 + 2] = as_type<uint>(half2(half(bf_lo(u.y)), half(bf_lo(u.w))));
        wr[i * 4 + 3] = as_type<uint>(half2(half(bf_hi(u.y)), half(bf_hi(u.w))));
      } else {
        uint qv[8];
        wf_codes8<F>(row, kk, qv);
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < 4; ++f) wr[i * 4 + f] = qv[f] | (qv[f + 4] << 16) | 0x34003400u;
        if (GS <= 32 || (k0 % uint(GS)) == 0) {
          const uint gi = wrow[a] * spr + kk / uint(GS);
          sb[i] = half2(half(float(S[p][gi]) * 4096.0f), half(float(B[p][gi])));
        }
      }
    }
}

template <int F, int FA, int FB, int TM>
NI_INLINE void qrf_mma(thread simdgroup_float8x8 (&acc)[FA][FB], threadgroup const half* xs, uint sgm,
                       thread const uint* wr, thread const half2* sb) {
  constexpr int LD = 32 + 8;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 4; ++f) {
    simdgroup_half8x8 af[FA];
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      half2 v = as_type<half2>(wr[a * 4 + f]);
      if (F != 16) v = fma(v - half2(0.25h), half2(sb[a].x), half2(sb[a].y));
      af[a].thread_elements()[0] = v.x;
      af[a].thread_elements()[1] = v.y;
    }
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) {
      simdgroup_half8x8 bf;
      simdgroup_load(bf, xs + (sgm * TM + b * 8) * LD + 8 * f, LD, ulong2(0, 0), true);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FA; ++a) simdgroup_multiply_accumulate(acc[a][b], af[a], bf, acc[a][b]);
    }
  }
}

// qr_acc for format F (BK = 32): acc[p*FA + a][b] over the whole K.
template <int F, int GS, int BM, int BN, int WM, int WN, int NP>
NI_INLINE void qrf_acc(const device half* X, const device uint32_t* const thread* W,
                       const device bfloat* const thread* S, const device bfloat* const thread* B, uint M, uint N,
                       uint K, uint x_ld, uint m0, uint n0, threadgroup half* xs, uint tid, uint sgm, uint sgn,
                       uint2 rc, thread simdgroup_float8x8 (&acc)[NP * (BN / WN / 8)][BM / WM / 8], uint w_ld = 0,
                       uint s_ld = 0) {
  constexpr int BK = 32, NT = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int FB = TM / 8, FA = TN / 8;
  constexpr int CPR = BK / 8, XL = (BM * CPR + NT - 1) / NT;
  const uint c = rc.y / 2;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < NP * FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FB; ++b) acc[a][b] = simdgroup_float8x8(0);
  // row strides: the rows' own K, or (w_ld, s_ld: words / groups) for the leading K columns of wider rows
  const uint wpr = w_ld ? w_ld : wf_wpr(F, K), spr = F == 16 ? 1u : s_ld ? s_ld : K / uint(GS), x_ld8 = x_ld / 8;
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, N - 1);
  const device uint4* X4 = (const device uint4*)X;
  uint4 xr[XL];
  uint wr[NP * FA * 4];
  half2 sb[NP * FA];
  for (uint k0 = 0; k0 < K; k0 += BK) {
    qr_load_x<BM, BK, NT>(X4, x_ld8, M, m0, k0, tid, xr);
    qrf_load_w<F, GS, FA, NP>(W, S, B, wrow, wpr, spr, k0, c, wr, sb);
    qr_store_x<BM, BK, NT>(xs, tid, xr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qrf_mma<F, NP * FA, FB, TM>(acc, xs, sgm, wr, sb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Y (+)= X W^T (QmmParams, buffers 0-5 as qmr_*).
#define NI_QRF(NAME, F, GS, BM, BN, WM, WN, MFAST)                                                           \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                       \
      const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                            \
      const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],                           \
      device float* Y [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                      \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                     \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                   \
    constexpr int TM = BM / WM, TN = BN / WN;                                                                 \
    threadgroup uint4 xs4[BM * (32 + 8) / 8];                                                                 \
    const device uint32_t* W[1] = {Wd};                                                                       \
    const device bfloat* S[1] = {Sd};                                                                         \
    const device bfloat* B[1] = {Bd};                                                                         \
    const uint mt = MFAST ? tg.x : tg.y, nt = MFAST ? tg.y : tg.x;                                            \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = mt * BM, n0 = nt * BN;                                      \
    const uint2 rc = qr_rc(lane);                                                                             \
    simdgroup_float8x8 acc[TN / 8][TM / 8];                                                                   \
    qrf_acc<F, GS, BM, BN, WM, WN, 1>(X, W, S, B, p.M, p.N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4, tid, \
                                      sgm, sgn, rc, acc);                                                     \
    _Pragma("clang loop unroll(full)")                                                                        \
    for (int a = 0; a < TN / 8; ++a) {                                                                        \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                            \
      if (n >= p.N) continue;                                                                                 \
      _Pragma("clang loop unroll(full)")                                                                      \
      for (int b = 0; b < TM / 8; ++b)                                                                        \
        _Pragma("clang loop unroll(full)")                                                                    \
        for (int e = 0; e < 2; ++e) {                                                                         \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                    \
          if (m < p.M) {                                                                                      \
            device float* dst = Y + m * p.y_ld + n;                                                           \
            const float v = acc[a][b].thread_elements()[e];                                                   \
            *dst = p.add ? *dst + v : v;                                                                      \
          }                                                                                                   \
        }                                                                                                     \
    }                                                                                                         \
  }

// Y (+)= X W[:, :K]^T over the leading K columns of rows w_ld words / s_ld groups long (QmmLdParams, as
// qmm_b4_m64_n64_ld): the GPU's share of the down projection in the Neural Engine split.
#define NI_QRF_LD(NAME, F, GS, BM, BN, WM, WN, MFAST)                                                        \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                       \
      const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                            \
      const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],                           \
      device float* Y [[buffer(4)]], constant QmmLdParams& pl [[buffer(5)]],                                   \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                     \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                   \
    constexpr int TM = BM / WM, TN = BN / WN;                                                                 \
    constant QmmParams& p = pl.q;                                                                             \
    threadgroup uint4 xs4[BM * (32 + 8) / 8];                                                                 \
    const device uint32_t* W[1] = {Wd};                                                                       \
    const device bfloat* S[1] = {Sd};                                                                         \
    const device bfloat* B[1] = {Bd};                                                                         \
    const uint mt = MFAST ? tg.x : tg.y, nt = MFAST ? tg.y : tg.x;                                            \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = mt * BM, n0 = nt * BN;                                      \
    const uint2 rc = qr_rc(lane);                                                                             \
    simdgroup_float8x8 acc[TN / 8][TM / 8];                                                                   \
    qrf_acc<F, GS, BM, BN, WM, WN, 1>(X, W, S, B, p.M, p.N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4, tid, \
                                      sgm, sgn, rc, acc, pl.w_ld, pl.s_ld);                                   \
    _Pragma("clang loop unroll(full)")                                                                        \
    for (int a = 0; a < TN / 8; ++a) {                                                                        \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                            \
      if (n >= p.N) continue;                                                                                 \
      _Pragma("clang loop unroll(full)")                                                                      \
      for (int b = 0; b < TM / 8; ++b)                                                                        \
        _Pragma("clang loop unroll(full)")                                                                    \
        for (int e = 0; e < 2; ++e) {                                                                         \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                    \
          if (m < p.M) {                                                                                      \
            device float* dst = Y + m * p.y_ld + n;                                                           \
            const float v = acc[a][b].thread_elements()[e];                                                   \
            *dst = p.add ? *dst + v : v;                                                                      \
          }                                                                                                   \
        }                                                                                                     \
    }                                                                                                         \
  }

// Fused gate/up + SwiGLU (as qmr_gu_*): H[m, n] = half(silu(X Wg^T) * (X Wu^T)).
#define NI_QRF_GU(NAME, F, GS, BM, BN, WM, WN, MFAST)                                                        \
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void NAME(                                       \
      const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],                            \
      const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],                           \
      device half* H [[buffer(4)]], constant QmmParams& p [[buffer(5)]],                                       \
      const device uint32_t* Wu [[buffer(6)]], const device bfloat* Su [[buffer(7)]],                         \
      const device bfloat* Bu [[buffer(8)]],                                                                  \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                     \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                   \
    constexpr int TM = BM / WM, TN = BN / WN, FA = TN / 8;                                                    \
    threadgroup uint4 xs4[BM * (32 + 8) / 8];                                                                 \
    const device uint32_t* W[2] = {Wg, Wu};                                                                   \
    const device bfloat* S[2] = {Sg, Su};                                                                     \
    const device bfloat* B[2] = {Bg, Bu};                                                                     \
    const uint mt = MFAST ? tg.x : tg.y, nt = MFAST ? tg.y : tg.x;                                            \
    const uint sgm = sg / WN, sgn = sg % WN, m0 = mt * BM, n0 = nt * BN;                                      \
    const uint2 rc = qr_rc(lane);                                                                             \
    simdgroup_float8x8 acc[2 * FA][TM / 8];                                                                   \
    qrf_acc<F, GS, BM, BN, WM, WN, 2>(X, W, S, B, p.M, p.N, p.K, p.x_ld, m0, n0, (threadgroup half*)xs4, tid, \
                                      sgm, sgn, rc, acc);                                                     \
    _Pragma("clang loop unroll(full)")                                                                        \
    for (int a = 0; a < FA; ++a) {                                                                            \
      const uint n = n0 + sgn * TN + a * 8 + rc.x;                                                            \
      if (n >= p.N) continue;                                                                                 \
      _Pragma("clang loop unroll(full)")                                                                      \
      for (int b = 0; b < TM / 8; ++b)                                                                        \
        _Pragma("clang loop unroll(full)")                                                                    \
        for (int e = 0; e < 2; ++e) {                                                                         \
          const uint m = m0 + sgm * TM + b * 8 + rc.y + e;                                                    \
          const float g = acc[a][b].thread_elements()[e], u = acc[FA + a][b].thread_elements()[e];           \
          if (m < p.M) H[m * p.y_ld + n] = half(silu(g) * u);                                                 \
        }                                                                                                     \
    }                                                                                                         \
  }

// MoE experts on moe_bucket2 tiles (as qrm_*): any expert format; the shared expert (riding along as
// expert shared_e) must have the experts' format.
template <int F, int GS, int FL, int BN, int WN, int NP>
NI_INLINE void qrmf_acc(const device half* X, const device uint* xrows, bool gather, uint x_ld, uint m0, uint rows,
                        const device uint32_t* const thread* W, const device bfloat* const thread* S,
                        const device bfloat* const thread* B, uint N, uint K, uint n0, threadgroup half* xs,
                        uint tid, uint sgn, uint2 rc, thread simdgroup_float8x8 (&acc)[NP * (BN / WN / 8)][FL]) {
  constexpr int BK = 32, NT = WN * 32, TN = BN / WN, FA = TN / 8, BMl = FL * 8;
  constexpr int CPR = BK / 8, XL = (BMl * CPR + NT - 1) / NT;
  const uint c = rc.y / 2;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < NP * FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FL; ++b) acc[a][b] = simdgroup_float8x8(0);
  const uint wpr = wf_wpr(F, K), spr = F == 16 ? 1u : K / uint(GS);
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, N - 1);
  uint xo[XL];
  qrm_xoff<BMl, BK, NT>(xrows, gather, x_ld / 8, m0, rows, tid, xo);
  const device uint4* X4 = (const device uint4*)X;
  uint4 xr[XL];
  uint wr[NP * FA * 4];
  half2 sb[NP * FA];
  for (uint k0 = 0; k0 < K; k0 += BK) {
    qrm_load_x<BMl, BK, NT>(X4, xo, k0, xr);
    qrf_load_w<F, GS, FA, NP>(W, S, B, wrow, wpr, spr, k0, c, wr, sb);
    qr_store_x<BMl, BK, NT>(xs, tid, xr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qrf_mma<F, NP * FA, FL, BMl>(acc, xs, 0, wr, sb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// H: the products silu(gate) * up as half, or as fp32 (HT float: Flint's, whose products can pass half's range and are
// stored scaled by flint_swiglu_scaled)
template <int F, int GS, int FL, int BN, int WN, typename HT = half>
NI_INLINE void qrmf_gu_body(const device half* X, const device uint* xrows, uint x_ld, uint m0, uint rows,
                            const device uint32_t* const thread* W, const device bfloat* const thread* S,
                            const device bfloat* const thread* B, device HT* H, uint y_ld, uint N, uint K, uint n0,
                            threadgroup half* xs, uint tid, uint sgn, uint2 rc) {
  constexpr int TN = BN / WN, FA = TN / 8;
  simdgroup_float8x8 acc[2 * FA][FL];
  qrmf_acc<F, GS, FL, BN, WN, 2>(X, xrows, true, x_ld, m0, rows, W, S, B, N, K, n0, xs, tid, sgn, rc, acc);
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) {
    const uint n = n0 + sgn * TN + a * 8 + rc.x;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FL; ++b)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint m = b * 8 + rc.y + e;
        const float g = acc[a][b].thread_elements()[e], u = acc[FA + a][b].thread_elements()[e];
        if (m < rows) H[(m0 + m) * y_ld + n] = HT(silu(g) * u);
      }
  }
}

template <int F, int GS, int FL, int BN, int WN>
NI_INLINE void qrmf_dn_body(const device half* X, uint x_ld, uint m0, uint rows, const device uint32_t* const thread* W,
                            const device bfloat* const thread* S, const device bfloat* const thread* B,
                            device float* Y, const device uint* yrows, uint y_ld, uint N, uint K, uint n0,
                            threadgroup half* xs, uint tid, uint sgn, uint2 rc) {
  constexpr int TN = BN / WN, FA = TN / 8;
  simdgroup_float8x8 acc[FA][FL];
  qrmf_acc<F, GS, FL, BN, WN, 1>(X, nullptr, false, x_ld, m0, rows, W, S, B, N, K, n0, xs, tid, sgn, rc, acc);
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < FL; ++b)
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint m = b * 8 + rc.y + e;
      if (m < rows) {
        device float* yr = Y + yrows[m0 + m] * y_ld + n0 + sgn * TN + rc.x;
        _Pragma("clang loop unroll(full)")
        for (int a = 0; a < FA; ++a) yr[a * 8] = acc[a][b].thread_elements()[e];
      }
    }
}

// grid (N / BN, max tiles), 32 * WN threads; Qmm2Params (w_expert_stride in words, s_expert_stride in groups).
#define NI_QRMF_GU(NAME, F, GS, FB, BN, WN) NI_QRMF_GU_T(NAME, F, GS, FB, BN, WN, half)
#define NI_QRMF_GU_T(NAME, F, GS, FB, BN, WN, HT)                                                            \
  [[kernel, max_total_threads_per_threadgroup(WN * 32)]] void NAME(                                          \
      const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],                          \
      const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],                         \
      const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],                       \
      const device bfloat* Bu [[buffer(6)]], device HT* H [[buffer(7)]], constant Qmm2Params& p [[buffer(8)]], \
      const device uint* tiles [[buffer(9)]], const device uint* ntiles [[buffer(10)]],                     \
      const device uint* xrows [[buffer(11)]], const device uint32_t* SWg [[buffer(12)]],                   \
      const device bfloat* SSg [[buffer(13)]], const device bfloat* SBg [[buffer(14)]],                     \
      const device uint32_t* SWu [[buffer(15)]], const device bfloat* SSu [[buffer(16)]],                   \
      const device bfloat* SBu [[buffer(17)]], uint2 tg [[threadgroup_position_in_grid]],                   \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                 \
      uint lane [[thread_index_in_simdgroup]]) {                                                            \
    if (tg.y >= ntiles[0]) return;                                                                          \
    threadgroup uint4 xs4[8 * FB * (32 + 8) / 8];                                                           \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;          \
    const bool sh = e == p.shared_e;                                                                        \
    const uint wo = sh ? 0 : e * p.w_expert_stride, so = sh ? 0 : e * p.s_expert_stride;                    \
    const device uint32_t* W[2] = {(sh ? SWg : Wg) + wo, (sh ? SWu : Wu) + wo};                             \
    const device bfloat* S[2] = {(sh ? SSg : Sg) + so, (sh ? SSu : Su) + so};                               \
    const device bfloat* B[2] = {(sh ? SBg : Bg) + so, (sh ? SBu : Bu) + so};                               \
    const uint n0 = tg.x * BN;                                                                              \
    const uint2 rc = qr_rc(lane);                                                                           \
    threadgroup half* xs = (threadgroup half*)xs4;                                                          \
    const uint live = (rows + 7) / 8;                                                                       \
    if (FB >= 4 && live >= 4)                                                                               \
      qrmf_gu_body<F, GS, (FB >= 4 ? 4 : 1), BN, WN, HT>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K, \
                                                     n0, xs, tid, sg, rc);                                  \
    else if (FB >= 3 && live == 3)                                                                          \
      qrmf_gu_body<F, GS, (FB >= 3 ? 3 : 1), BN, WN, HT>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K, \
                                                     n0, xs, tid, sg, rc);                                  \
    else if (FB >= 2 && live == 2)                                                                          \
      qrmf_gu_body<F, GS, (FB >= 2 ? 2 : 1), BN, WN, HT>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K, \
                                                     n0, xs, tid, sg, rc);                                  \
    else                                                                                                    \
      qrmf_gu_body<F, GS, 1, BN, WN, HT>(X, xrows, p.x_ld, m0, rows, W, S, B, H, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                     rc);                                                                   \
  }

#define NI_QRMF_DN(NAME, F, GS, FB, BN, WN)                                                                  \
  [[kernel, max_total_threads_per_threadgroup(WN * 32)]] void NAME(                                          \
      const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                          \
      const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],                         \
      device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]], const device uint* tiles [[buffer(6)]], \
      const device uint* ntiles [[buffer(7)]], const device uint* yrows [[buffer(8)]],                      \
      const device uint32_t* SWd [[buffer(9)]], const device bfloat* SSd [[buffer(10)]],                    \
      const device bfloat* SBd [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],                   \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                 \
      uint lane [[thread_index_in_simdgroup]]) {                                                            \
    if (tg.y >= ntiles[0]) return;                                                                          \
    threadgroup uint4 xs4[8 * FB * (32 + 8) / 8];                                                           \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;          \
    const bool sh = e == p.shared_e;                                                                        \
    const device uint32_t* W[1] = {sh ? SWd : Wd + e * p.w_expert_stride};                                  \
    const device bfloat* S[1] = {sh ? SSd : Sd + e * p.s_expert_stride};                                    \
    const device bfloat* B[1] = {sh ? SBd : Bd + e * p.s_expert_stride};                                    \
    const uint n0 = tg.x * BN;                                                                              \
    const uint2 rc = qr_rc(lane);                                                                           \
    threadgroup half* xs = (threadgroup half*)xs4;                                                          \
    const uint live = (rows + 7) / 8;                                                                       \
    if (FB >= 4 && live >= 4)                                                                               \
      qrmf_dn_body<F, GS, (FB >= 4 ? 4 : 1), BN, WN>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K, \
                                                     n0, xs, tid, sg, rc);                                  \
    else if (FB >= 3 && live == 3)                                                                          \
      qrmf_dn_body<F, GS, (FB >= 3 ? 3 : 1), BN, WN>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K, \
                                                     n0, xs, tid, sg, rc);                                  \
    else if (FB >= 2 && live == 2)                                                                          \
      qrmf_dn_body<F, GS, (FB >= 2 ? 2 : 1), BN, WN>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K, \
                                                     n0, xs, tid, sg, rc);                                  \
    else                                                                                                    \
      qrmf_dn_body<F, GS, 1, BN, WN>(X, p.x_ld, m0, rows, W, S, B, Y, yrows, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                     rc);                                                                   \
  }

// ---- short prefills (qmms.metal's register-streaming layout, QmmsParams) for any format: qmmsf_* ----
// The lane's A elements of the 8 fragments of 64-group g of a row: values 64g + 16q + 8(f/4) + f%4 (+ 4),
// dequantized to half (s * q + b in one half2 fma, as qs_dequant) or bf16 converted.
template <int F, int GS>
NI_INLINE void qsf_dequant(const device uint32_t* row, const device bfloat* srow, const device bfloat* brow, uint g,
                           uint q, thread half2* a) {
  const uint k0 = g * 64 + 16 * q;
  if (F == 16) {
    const device uint4* p = (const device uint4*)(row + k0 / 2);
    const uint4 u0 = p[0], u1 = p[1];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; ++f) {
      const uint4 u = f < 4 ? u0 : u1;
      const int jj = f % 4;
      const uint w0 = u[jj / 2], w1 = u[2 + jj / 2];
      a[f] = half2(half((jj % 2) ? bf_hi(w0) : bf_lo(w0)), half((jj % 2) ? bf_hi(w1) : bf_lo(w1)));
    }
  } else {
    uint c[16];
    wf_codes16<F>(row, k0, c);
    const uint gi = k0 / uint(GS);
    const half2 s2 = half2(half(float(srow[gi]))), b2 = half2(half(float(brow[gi])));
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; ++f) {
      const int i0 = 8 * (f / 4) + f % 4;
      a[f] = fma(as_type<half2>(c[i0] | (c[i0 + 4] << 16) | 0x64006400u) - half2(1024.0h), s2, b2);
    }
  }
}

template <int F, int GS, int NM, int TB>
NI_INLINE void qsf_run(const device uint32_t* const thread* W, const device bfloat* const thread* S,
                       const device bfloat* const thread* B, uint row, uint wpr, uint spr, const device half* X,
                       uint x_ld, uint T, uint g0, uint g1, uint lane, thread simdgroup_float8x8 (&acc)[NM][TB]) {
  const uint2 rc = qs_lane_rc(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const device half* xr[TB][2];
  _Pragma("clang loop unroll(full)")
  for (int tb = 0; tb < TB; ++tb) {
    const uint t0 = tb * 8 + fn;
    xr[tb][0] = t0 < T ? X + t0 * x_ld : nullptr;
    xr[tb][1] = t0 + 1 < T ? X + (t0 + 1) * x_ld : nullptr;
  }
  _Pragma("clang loop unroll(full)")
  for (int m = 0; m < NM; ++m)
    _Pragma("clang loop unroll(full)")
    for (int tb = 0; tb < TB; ++tb) acc[m][tb] = simdgroup_float8x8(0);
  for (uint g = g0; g < g1; ++g) {
    half2 a[NM][8];
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < NM; ++m) qsf_dequant<F, GS>(W[m] + row * wpr, S[m] + row * spr, B[m] + row * spr, g, q, a[m]);
    _Pragma("clang loop unroll(full)")
    for (int tb = 0; tb < TB; ++tb) {
      half2 b[8];
      qs_gather(xr[tb][0], xr[tb][1], g, fm, b);
      _Pragma("clang loop unroll(full)")
      for (int m = 0; m < NM; ++m) qs_mma8(acc[m][tb], a[m], b);
    }
  }
}

template <int F, int GS, int TB>
NI_INLINE void qmmsf_impl(const device uint32_t* const* Wv, const device bfloat* const* Sv,
                          const device bfloat* const* Bv, device float* const* Yv, const device half* X,
                          constant QmmsParams& p, threadgroup float2* red, uint tg, uint sg, uint nsg, uint lane,
                          uint tgy) {
  const uint r0 = tgy * 8 * TB;
  if (r0 >= p.T) return;
  const uint T = min(p.T - r0, uint(8 * TB));
  X += r0 * p.x_ld;
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint rbs = nsg / p.ks;
  const uint rb = (tg - p.tg_begin[si]) * rbs + sg / p.ks, kslice = sg % p.ks;
  const uint N = p.N[si], gpr = p.K / 64, wpr = wf_wpr(F, p.K), spr = F == 16 ? 1u : p.K / uint(GS);
  const uint2 rc = qs_lane_rc(lane);
  const uint n = rb * 8 + rc.x;
  const uint row = min(n, N - 1);
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* W[1] = {Wv[si]};
  const device bfloat* S[1] = {Sv[si]};
  const device bfloat* B[1] = {Bv[si]};
  simdgroup_float8x8 acc[1][TB];
  qsf_run<F, GS, 1, TB>(W, S, B, row, wpr, spr, X, p.x_ld, T, g0, g1, lane, acc);
  if (p.ks > 1) {
    qs_reduce<1, TB>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  if (n >= N) return;
  const uint y_ld = p.y_ld[si];
  device float* Y = Yv[si] + r0 * y_ld;
  const bool add = p.add[si];
  _Pragma("clang loop unroll(full)")
  for (int tb = 0; tb < TB; ++tb)
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint t = tb * 8 + rc.y + e;
      if (t < T) {
        device float* d = Y + t * y_ld + n;
        const float v = acc[0][tb].thread_elements()[e];
        *d = add ? *d + v : v;
      }
    }
}

#define NI_QMMSF(NAME, F, GS, TB)                                                                              \
  [[kernel]] void NAME(const device uint32_t* W0 [[buffer(0)]], const device bfloat* S0 [[buffer(1)]],         \
                       const device bfloat* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                 \
                       const device uint32_t* W1 [[buffer(4)]], const device bfloat* S1 [[buffer(5)]],         \
                       const device bfloat* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                 \
                       const device uint32_t* W2 [[buffer(8)]], const device bfloat* S2 [[buffer(9)]],         \
                       const device bfloat* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],               \
                       const device uint32_t* W3 [[buffer(12)]], const device bfloat* S3 [[buffer(13)]],       \
                       const device bfloat* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],               \
                       const device half* X [[buffer(16)]], constant QmmsParams& p [[buffer(17)]],             \
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {     \
    threadgroup float2 red[8 * TB * 32];                                                                     \
    const device uint32_t* Wv[4] = {W0, W1, W2, W3};                                                          \
    const device bfloat* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device bfloat* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    qmmsf_impl<F, GS, TB>(Wv, Sv, Bv, Yv, X, p, red, tg.x, sg, nsg, lane, tg.y);                             \
  }

template <int F, int GS, int TB>
NI_INLINE void qmmsf_gu_impl(const device uint32_t* Wg, const device bfloat* Sg, const device bfloat* Bg,
                             const device uint32_t* Wu, const device bfloat* Su, const device bfloat* Bu,
                             device half* H, const device half* X, constant QmmsParams& p, threadgroup float2* red,
                             uint tg, uint sg, uint nsg, uint lane, uint tgy) {
  const uint r0 = tgy * 8 * TB;
  if (r0 >= p.T) return;
  const uint T = min(p.T - r0, uint(8 * TB));
  X += r0 * p.x_ld;
  H += r0 * p.y_ld[0];
  const uint rbs = nsg / p.ks;
  const uint rb = tg * rbs + sg / p.ks, kslice = sg % p.ks;
  const uint N = p.N[0], gpr = p.K / 64, wpr = wf_wpr(F, p.K), spr = F == 16 ? 1u : p.K / uint(GS);
  const uint2 rc = qs_lane_rc(lane);
  const uint n = rb * 8 + rc.x;
  const uint row = min(n, N - 1);
  const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* W[2] = {Wg, Wu};
  const device bfloat* S[2] = {Sg, Su};
  const device bfloat* B[2] = {Bg, Bu};
  simdgroup_float8x8 acc[2][TB];
  qsf_run<F, GS, 2, TB>(W, S, B, row, wpr, spr, X, p.x_ld, T, g0, g1, lane, acc);
  if (p.ks > 1) {
    qs_reduce<2, TB>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  if (n >= N) return;
  const uint ld = p.y_ld[0];
  _Pragma("clang loop unroll(full)")
  for (int tb = 0; tb < TB; ++tb)
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint t = tb * 8 + rc.y + e;
      if (t < T) {
        const float gv = acc[0][tb].thread_elements()[e], uv = acc[1][tb].thread_elements()[e];
        H[t * ld + n] = half(silu(gv) * uv);
      }
    }
}

#define NI_QMMSF_GU(NAME, F, GS, TB)                                                                           \
  [[kernel]] void NAME(const device uint32_t* Wg [[buffer(0)]], const device bfloat* Sg [[buffer(1)]],         \
                       const device bfloat* Bg [[buffer(2)]], device half* H [[buffer(3)]],                   \
                       const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],         \
                       const device bfloat* Bu [[buffer(6)]], const device half* X [[buffer(16)]],             \
                       constant QmmsParams& p [[buffer(17)]], uint2 tg [[threadgroup_position_in_grid]],       \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],    \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup float2 red[8 * 2 * TB * 32];                                                                 \
    qmmsf_gu_impl<F, GS, TB>(Wg, Sg, Bg, Wu, Su, Bu, H, X, p, red, tg.x, sg, nsg, lane, tg.y);               \
  }

// MoE experts at small T (qmms_moe_* layout: <= 8-row tiles of one expert, Qmm2Params) for any expert format.
template <int F, int GS>
NI_INLINE void qmmsf_moe_gu(const device half* X, const device uint32_t* Wg, const device bfloat* Sg,
                            const device bfloat* Bg, const device uint32_t* Wu, const device bfloat* Su,
                            const device bfloat* Bu, device half* H, constant Qmm2Params& p, const device uint* tiles,
                            const device uint* ntiles, const device uint* xrows, const device uint32_t* SWg,
                            const device bfloat* SSg, const device bfloat* SBg, const device uint32_t* SWu,
                            const device bfloat* SSu, const device bfloat* SBu, uint2 tg, uint sg, uint nsg,
                            uint lane) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const bool sh = e == p.shared_e;
  const uint wo = sh ? 0 : e * p.w_expert_stride, so = sh ? 0 : e * p.s_expert_stride;
  const device uint32_t* W[2] = {(sh ? SWg : Wg) + wo, (sh ? SWu : Wu) + wo};
  const device bfloat* S[2] = {(sh ? SSg : Sg) + so, (sh ? SSu : Su) + so};
  const device bfloat* B[2] = {(sh ? SBg : Bg) + so, (sh ? SBu : Bu) + so};
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;
  const uint gpr = p.K / 64, wpr = wf_wpr(F, p.K), spr = F == 16 ? 1u : p.K / uint(GS);
  const device half* x0 = rc.y < rows ? X + xrows[m0 + rc.y] * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + xrows[m0 + rc.y + 1] * p.x_ld : nullptr;
  simdgroup_float8x8 ag = simdgroup_float8x8(0), au = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a0[8], a1[8], b[8];
    qsf_dequant<F, GS>(W[0] + n * wpr, S[0] + n * spr, B[0] + n * spr, g, q, a0);
    qsf_dequant<F, GS>(W[1] + n * wpr, S[1] + n * spr, B[1] + n * spr, g, q, a1);
    qs_gather(x0, x1, g, fm, b);
    qs_mma8(ag, a0, b);
    qs_mma8(au, a1, b);
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 2; ++j) {
    const uint r = rc.y + j;
    if (r < rows) H[(m0 + r) * p.y_ld + n] = half(silu(ag.thread_elements()[j]) * au.thread_elements()[j]);
  }
}

template <int F, int GS>
NI_INLINE void qmmsf_moe_dn(const device half* X, const device uint32_t* Wd, const device bfloat* Sd,
                            const device bfloat* Bd, device float* Y, constant Qmm2Params& p, const device uint* tiles,
                            const device uint* ntiles, const device uint* yrows, const device uint32_t* SWd,
                            const device bfloat* SSd, const device bfloat* SBd, uint2 tg, uint sg, uint nsg,
                            uint lane) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const bool sh = e == p.shared_e;
  const device uint32_t* W = sh ? SWd : Wd + e * p.w_expert_stride;
  const device bfloat* S = sh ? SSd : Sd + e * p.s_expert_stride;
  const device bfloat* B = sh ? SBd : Bd + e * p.s_expert_stride;
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;
  const uint gpr = p.K / 64, wpr = wf_wpr(F, p.K), spr = F == 16 ? 1u : p.K / uint(GS);
  const device half* x0 = rc.y < rows ? X + (m0 + rc.y) * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + (m0 + rc.y + 1) * p.x_ld : nullptr;
  simdgroup_float8x8 acc = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a[8], b[8];
    qsf_dequant<F, GS>(W + n * wpr, S + n * spr, B + n * spr, g, q, a);
    qs_gather(x0, x1, g, fm, b);
    qs_mma8(acc, a, b);
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 2; ++j) {
    const uint r = rc.y + j;
    if (r < rows) Y[yrows[m0 + r] * p.y_ld + n] = acc.thread_elements()[j];
  }
}

#define NI_QMMSF_MOE(SFX, F, GS)                                                                               \
  [[kernel]] void qmmsf_moe_gu_##SFX(                                                                          \
      const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],                             \
      const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],                            \
      const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],                          \
      const device bfloat* Bu [[buffer(6)]], device half* H [[buffer(7)]], constant Qmm2Params& p [[buffer(8)]], \
      const device uint* tiles [[buffer(9)]], const device uint* ntiles [[buffer(10)]],                        \
      const device uint* xrows [[buffer(11)]], const device uint32_t* SWg [[buffer(12)]],                      \
      const device bfloat* SSg [[buffer(13)]], const device bfloat* SBg [[buffer(14)]],                        \
      const device uint32_t* SWu [[buffer(15)]], const device bfloat* SSu [[buffer(16)]],                      \
      const device bfloat* SBu [[buffer(17)]], uint2 tg [[threadgroup_position_in_grid]],                      \
      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]]) {                                                               \
    qmmsf_moe_gu<F, GS>(X, Wg, Sg, Bg, Wu, Su, Bu, H, p, tiles, ntiles, xrows, SWg, SSg, SBg, SWu, SSu, SBu, tg, sg, \
                        nsg, lane);                                                                            \
  }                                                                                                            \
  [[kernel]] void qmmsf_moe_dn_##SFX(                                                                          \
      const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],                             \
      const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],                            \
      device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]], const device uint* tiles [[buffer(6)]], \
      const device uint* ntiles [[buffer(7)]], const device uint* yrows [[buffer(8)]],                         \
      const device uint32_t* SWd [[buffer(9)]], const device bfloat* SSd [[buffer(10)]],                       \
      const device bfloat* SBd [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],                      \
      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]]) {                                                               \
    qmmsf_moe_dn<F, GS>(X, Wd, Sd, Bd, Y, p, tiles, ntiles, yrows, SWd, SSd, SBd, tg, sg, nsg, lane);           \
  }

// ---- load-time conversions (tests and tools) ----
// Dequantizes an affine F-bit matrix [N, K] (group GS) to bf16 (round to nearest even), one thread per 16 values.
template <int F, int GS>
NI_INLINE void wf_dequant_bf16(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                               device ushort* out, uint K, uint i) {
  const uint per_row = K / 16, row = i / per_row, k0 = (i % per_row) * 16;
  uint c[16];
  wf_codes16<F>(W + row * wf_wpr(F, K), k0, c);
  const uint gi = row * (K / uint(GS)) + k0 / uint(GS);
  const float s = float(S[gi]), b = float(B[gi]);
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 16; ++j) out[row * K + k0 + j] = as_type<ushort>(bfloat(fma(s, float(c[j]), b)));
}

#define NI_WF_DEQUANT(SFX, F, GS)                                                                              \
  [[kernel]] void wf_dequant_##SFX(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]], \
                                   const device bfloat* B [[buffer(2)]], device ushort* out [[buffer(3)]],      \
                                   constant uint2& nk [[buffer(4)]], uint i [[thread_position_in_grid]]) {      \
    if (i < nk.x * (nk.y / 16)) wf_dequant_bf16<F, GS>(W, S, B, out, nk.y, i);                                  \
  }

// The bf16 GEMMs of the image encoders (src/vision): in the base library, so a 4-bit model's first image does not
// wait for the second library's compile (~4-5 s the first time a build runs it).
NI_QRF(qrf_f16_m32_n128_mf, 16, 64, 32, 128, 1, 4, true)
NI_QRF(qrf_f16_m16_n64, 16, 64, 16, 64, 1, 4, false)
NI_QRF(qrf_f16_m8_n64, 16, 64, 8, 64, 1, 2, false)

#ifdef NI_WF_INST  // compiled into the second library only (Gpu::compile_formats)
// ---- instantiations ----
// Formats: f16 = bf16, f8g64, f6g64, f5g64, f3g64, f4g32 (4-bit group 32, e.g. the MTPLX 27B pack). 4-bit group 64
// keeps the specialized kernels everywhere, and 8-bit group 64 wherever one exists (qmv3_b8, qmv2_multi, qmm8_b8,
// qmms_b8, qmr8), so each family below lists only the formats without one (each format costs ~0.3 s of the
// first, uncached library compile).
#define NI_WF_LOW(X) X(f6g64, 6, 64) X(f5g64, 5, 64) X(f3g64, 3, 64) X(f4g32, 4, 32)
#define NI_WF_NO8(X) X(f16, 16, 64) NI_WF_LOW(X)
#define NI_WF_ALL(X) NI_WF_NO8(X) X(f8g64, 8, 64)
#define NI_WF_MOE(X) X(f16, 16, 64) X(f8g64, 8, 64) X(f6g64, 6, 64) X(f5g64, 5, 64) X(f3g64, 3, 64)

#define NI_WF_DECODE(SFX, F, GS)                                     \
  NI_QMVG(qmvg_##SFX##_r2, F, GS, 2, false, false)                   \
  NI_QMVG(qmvg_##SFX##_r4, F, GS, 4, false, false)                   \
  NI_QMVG(qmvg_##SFX##_r2_add, F, GS, 2, true, false)                \
  NI_QMVG(qmvg_##SFX##_r2_norm, F, GS, 2, false, true)               \
  NI_QMVG(qmvg_##SFX##_r4_norm, F, GS, 4, false, true)               \
  NI_QMVG_MULTI(qmvg_multi_##SFX, F, GS)                             \
  NI_QMM8G(qmm8g_##SFX##_nf1, F, GS, 1)                              \
  NI_QMM8G(qmm8g_##SFX##_nf2, F, GS, 2)
NI_WF_NO8(NI_WF_DECODE)

#define NI_WF_LOOKUP(SFX, F, GS)                                     \
  NI_MLPG(mlp_gate_up_##SFX, F, GS)                                  \
  NI_EMBEDG(embed_##SFX, F)                                          \
  NI_DFL_EMBEDG(dfl_embed_norm_##SFX, F)
NI_WF_ALL(NI_WF_LOOKUP)

#define NI_WF_PREFILL(SFX, F, GS)                                    \
  NI_QRF(qrf_##SFX##_m32_n128, F, GS, 32, 128, 1, 4, false)          \
  NI_QRF_GU(qrf_gu_##SFX##_m32_n64, F, GS, 32, 64, 1, 4, false)      \
  NI_QRF_GU(qrf_gu_##SFX##_m16_n64, F, GS, 16, 64, 1, 4, false)      \
  NI_QRF_GU(qrf_gu_##SFX##_m8_n32, F, GS, 8, 32, 1, 2, false)
NI_WF_NO8(NI_WF_PREFILL)
#define NI_WF_PREFILL_SMALL(SFX, F, GS)                              \
  NI_QRF(qrf_##SFX##_m16_n64, F, GS, 16, 64, 1, 4, false)            \
  NI_QRF(qrf_##SFX##_m8_n64, F, GS, 8, 64, 1, 2, false)
NI_WF_LOW(NI_WF_PREFILL_SMALL)  // bf16's are in the base library (above)

#define NI_WF_MOE_PREFILL(SFX, F, GS)                                \
  NI_QRMF_GU(qrmf_gu_##SFX##_m32_n64, F, GS, 4, 64, 4)               \
  NI_QRMF_DN(qrmf_dn_##SFX##_m32_n128, F, GS, 4, 128, 4)             \
  NI_QMMSF_MOE(SFX, F, GS)
NI_WF_MOE(NI_WF_MOE_PREFILL)
// Flint's 8-bit routed experts: the gate / up products in fp32 (model/flint.cpp, enc_moe_g_prefill)
NI_QRMF_GU_T(qrmf_gu_f8g64_m32_n64_f32, 8, 64, 4, 64, 4, float)

// bf16 large-M tiles with M tiles along grid x (weights shared through the cache; qrf_f16_m32_n128_mf: base library)
NI_QRF(qrf_f16_m64_n64_mf, 16, 64, 64, 64, 1, 4, true)
NI_QRF_GU(qrf_gu_f16_m32_n64_mf, 16, 64, 32, 64, 1, 4, true)
// The other formats' large tiles with M tiles along grid x too: the M1's (Apple7) default (enc_qrf)
#define NI_WF_PREFILL_MF(SFX, F, GS) NI_QRF(qrf_##SFX##_m32_n128_mf, F, GS, 32, 128, 1, 4, true)
NI_WF_LOW(NI_WF_PREFILL_MF)

#define NI_WF_SHORT(SFX, F, GS)                                                                             \
  NI_QMMSF(qmmsf_##SFX##_t1, F, GS, 1) NI_QMMSF(qmmsf_##SFX##_t2, F, GS, 2) NI_QMMSF(qmmsf_##SFX##_t3, F, GS, 3) \
  NI_QMMSF(qmmsf_##SFX##_t4, F, GS, 4) NI_QMMSF(qmmsf_##SFX##_t5, F, GS, 5) NI_QMMSF(qmmsf_##SFX##_t6, F, GS, 6) \
  NI_QMMSF(qmmsf_##SFX##_t7, F, GS, 7) NI_QMMSF(qmmsf_##SFX##_t8, F, GS, 8)                                   \
  NI_QMMSF_GU(qmmsf_gu_##SFX##_t1, F, GS, 1) NI_QMMSF_GU(qmmsf_gu_##SFX##_t2, F, GS, 2)                       \
  NI_QMMSF_GU(qmmsf_gu_##SFX##_t3, F, GS, 3) NI_QMMSF_GU(qmmsf_gu_##SFX##_t4, F, GS, 4)                       \
  NI_QMMSF_GU(qmmsf_gu_##SFX##_t5, F, GS, 5) NI_QMMSF_GU(qmmsf_gu_##SFX##_t6, F, GS, 6)                       \
  NI_QMMSF_GU(qmmsf_gu_##SFX##_t7, F, GS, 7) NI_QMMSF_GU(qmmsf_gu_##SFX##_t8, F, GS, 8)
NI_WF_NO8(NI_WF_SHORT)

// 8-row MoE verify: routed experts x shared expert formats (gate/up; down)
#define NI_MOE8G(SFX, FE, GE, FS, GSS) NI_MOE8G_GU(moe8g_gu_##SFX, FE, GE, FS, GSS) \
                                       NI_MOE8G_DOWN(moe8g_down_##SFX, FE, GE, FS, GSS)
NI_MOE8G(f16_f16, 16, 64, 16, 64)
NI_MOE8G(f8g64_f8g64, 8, 64, 8, 64)
NI_MOE8G(f6g64_f6g64, 6, 64, 6, 64)
NI_MOE8G(f6g64_f8g64, 6, 64, 8, 64)
NI_MOE8G(f5g64_f5g64, 5, 64, 5, 64)
NI_MOE8G(f5g64_f8g64, 5, 64, 8, 64)
NI_MOE8G(f4g64_f8g64, 4, 64, 8, 64)
NI_MOE8G(f3g64_f8g64, 3, 64, 8, 64)

// MoE rows: routed experts x shared expert formats seen in MLX packs
NI_MOEG_GU_ROWS(moeg_gu_rows_f16_f16, 16, 64, 16, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f16_f16, 16, 64, 16, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f8g64_f8g64, 8, 64, 8, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f8g64_f8g64, 8, 64, 8, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f4g64_f8g64, 4, 64, 8, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f4g64_f8g64, 4, 64, 8, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f6g64_f6g64, 6, 64, 6, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f6g64_f6g64, 6, 64, 6, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f6g64_f8g64, 6, 64, 8, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f6g64_f8g64, 6, 64, 8, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f5g64_f5g64, 5, 64, 5, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f5g64_f5g64, 5, 64, 5, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f5g64_f8g64, 5, 64, 8, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f5g64_f8g64, 5, 64, 8, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f3g64_f8g64, 3, 64, 8, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f3g64_f8g64, 3, 64, 8, 64, 9 * 512)
NI_MOEG_GU_ROWS(moeg_gu_rows_f4g64_f4g64, 4, 64, 4, 64)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f4g64_f4g64, 4, 64, 4, 64, 9 * 512)
// wide experts ((k + 1) * max(I, SI) <= 7168 staged floats: Qwen3.8-Flash-Next's 11 x 640)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f4g64_f8g64_w, 4, 64, 8, 64, 7168)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f3g64_f8g64_w, 3, 64, 8, 64, 7168)
NI_MOEG_DOWN_ROWS(moeg_down_rows_f8g64_f8g64_w, 8, 64, 8, 64, 7168)  // (Flint's 7 x 1024)

// Neural Engine split (ane_ffn.metal): weight staging from each format, and the GPU's share of the down projection
#define NI_WF_ANE(SFX, F, GS) NI_ANE_WF(SFX, F, GS) NI_QRF_LD(qrf_##SFX##_m32_n128_ld, F, GS, 32, 128, 1, 4, false)
NI_WF_ALL(NI_WF_ANE)
NI_QRF_LD(qrf_f16_m32_n128_mf_ld, 16, 64, 32, 128, 1, 4, true)

NI_WF_DEQUANT(f4g64, 4, 64)
NI_WF_DEQUANT(f8g64, 8, 64)
NI_WF_DEQUANT(f4g32, 4, 32)

#endif  // NI_WF_INST
