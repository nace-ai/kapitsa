#include "common.h"

// ---- Qwen3.8-Flash-Next speculative verify: kernels for 2-4 token rows ----
// (The file name sorts last: the kernel sources are concatenated in name order, and these use helpers of qwen4.metal,
// verify.metal and wfmt.metal.)
// Kernels whose per-row code is the decode kernels' (qmvs_multi, hc_*3, moe4_*) give verify rows bitwise equal to
// decode steps (tools/fn_check, FN_ROWS=1). qmvt_rows (qmvp_multi, 4 rows): each simdgroup reads its weight rows once
// for NT token rows, x from device memory; the same formula as qmv2_rows (16 values per lane, lanes 512 values apart,
// the shift-free dequant), which fast math may sum in another order (~1e-7 relative).
struct QmvtParams {
  uint K, R, nseg, x_ld;
  uint N[4];
  uint tg_begin[5];
  uint bits[4];
  uint y_ld[4];
  uint add[4];
};

template <int BITS, int NT, int ROWS>
NI_INLINE void qmvt_rows(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         const device float* x, uint x_ld, uint R, uint N, uint K, uint row0, uint lane,
                         thread float (&acc)[ROWS][NT]) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int VPL = 16;
  constexpr int WORDS = VPL / PER_WORD;
  const uint wpr = K / PER_WORD, gpr = K / 64;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL];
    float xsum[NT];
    for (int t = 0; t < NT; ++t) {
      const device float4* xp = (const device float4*)(x + min(uint(t), R - 1) * x_ld + k0);
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = xp[i];
        xsum[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e)
          xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
      }
    }
    const uint g = k0 / 64, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * wpr + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      }
      const float s = float(scales[rows[r] * gpr + g]), b = float(biases[rows[r] * gpr + g]);
      float d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) {
          const float wf = float(wv[i] & (MASK << (BITS * j)));
          for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
        }
      for (int t = 0; t < NT; ++t) acc[r][t] += s * d[t] + b * xsum[t];
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = simd_sum(acc[r][t]);
}

template <int BITS, int NT, int ROWS>
NI_INLINE void qmvt_store(const device uint32_t* w, const device bfloat* s, const device bfloat* b,
                          const device float* x, device float* y, uint x_ld, uint y_ld, uint R, uint N, uint K,
                          bool add, uint row0, uint lane) {
  float acc[ROWS][NT];
  qmvt_rows<BITS, NT, ROWS>(w, s, b, x, x_ld, R, N, K, row0, lane, acc);
  if (lane != 0) return;
  for (int t = 0; t < NT; ++t) {
    if (uint(t) >= R) break;
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < N) {
        device float* dst = y + t * y_ld + row0 + r;
        *dst = add ? *dst + acc[r][t] : acc[r][t];
      }
  }
}

// ---- Token rows as separate simdgroups: one row of x per simdgroup, the weight block shared through the core's
// cache (each simdgroup's arithmetic is the decode kernels'). qmvs_multi: grid (R, threadgroups) of 2 simdgroups x 4
// weight rows (the R threadgroups of a block adjacent, as qmvrc1); up to four 4/8-bit matrices sharing x.
template <int BITS>
NI_INLINE void qmvs_row(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                        const device float* x, device float* y, uint N, uint K, bool add, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int VPL = 16, ROWS = 4;
  constexpr int WORDS = VPL / PER_WORD;
  const uint wpr = K / PER_WORD, gpr = K / 64;
  float acc[ROWS];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[VPL];
    float xsum = 0.0f;
    const device float4* xp = (const device float4*)(x + k0);
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = xp[i];
      xsum += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) xv[i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
    }
    const uint g = k0 / 64, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * wpr + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      }
      float d = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * xv[i * PER_WORD + j];
      acc[r] += float(scales[rows[r] * gpr + g]) * d + float(biases[rows[r] * gpr + g]) * xsum;
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    const float v = simd_sum(acc[r]);
    const uint row = row0 + r;
    if (lane == 0 && row < N) y[row] = add ? y[row] + v : v;
  }
}

#define NI_QMVS_MULTI(NAME)                                                                               \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvtParams& p [[buffer(17)]],                       \
      uint2 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],            \
      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    const uint t = tgp.x;                                                                                \
    if (t >= p.R) return;                                                                                \
    uint seg = 0;                                                                                        \
    for (uint i = 1; i < p.nseg; ++i)                                                                    \
      if (tgp.y >= p.tg_begin[i]) seg = i;                                                                \
    const uint row0 = ((tgp.y - p.tg_begin[seg]) * nsg + sg) * 4;                                        \
    const uint N = p.N[seg];                                                                              \
    if (row0 >= N) return;                                                                                \
    const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;                        \
    const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;                          \
    const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;                          \
    device float* Y = (seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3) + t * p.y_ld[seg];             \
    if (p.bits[seg] == 8) qmvs_row<8>(W, S, B, x + t * p.x_ld, Y, N, p.K, p.add[seg] != 0, row0, lane);    \
    else qmvs_row<4>(W, S, B, x + t * p.x_ld, Y, N, p.K, p.add[seg] != 0, row0, lane);                     \
  }
NI_QMVS_MULTI(qmvs_multi)


// Any MLX affine format of wfmt.metal (6-bit attention projections): per row the arithmetic of wf_rows_acc (qmvg_*).
template <int F>
NI_INLINE void qmvs_row_wf(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                           const device float* x, device float* y, uint N, uint K, bool add, uint row0, uint lane) {
  constexpr int ROWS = 4;
  const uint wpr = wf_wpr(F, K), gpr = K / 64;
  float acc[ROWS];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * 16; k0 < K; k0 += 512) {
    float xv[16];
    float xsum = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i) {
      const float4 v = ((const device float4*)(x + k0))[i];
      xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;
      xsum += v.x + v.y + v.z + v.w;
    }
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < ROWS; ++r) {
      float wv[16];
      wf_vals16<F>(w + rows[r] * wpr, k0, wv);
      float d = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 16; ++j) d = fma(wv[j], xv[j], d);
      const uint gi = rows[r] * gpr + k0 / 64;
      acc[r] += float(scales[gi]) * d + float(biases[gi]) * xsum;
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    const float v = simd_sum(acc[r]);
    const uint row = row0 + r;
    if (lane == 0 && row < N) y[row] = add ? y[row] + v : v;
  }
}
// segments of one wfmt format F (group 64), grid (R, threadgroups) of 2 simdgroups x 4 rows
#define NI_QMVS_MULTI_WF(NAME, F)                                                                         \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvtParams& p [[buffer(17)]],                       \
      uint2 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],            \
      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
    const uint t = tgp.x;                                                                                \
    if (t >= p.R) return;                                                                                \
    uint seg = 0;                                                                                        \
    for (uint i = 1; i < p.nseg; ++i)                                                                    \
      if (tgp.y >= p.tg_begin[i]) seg = i;                                                                \
    const uint row0 = ((tgp.y - p.tg_begin[seg]) * nsg + sg) * 4;                                        \
    const uint N = p.N[seg];                                                                              \
    if (row0 >= N) return;                                                                                \
    const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;                        \
    const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;                          \
    const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;                          \
    device float* Y = (seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3) + t * p.y_ld[seg];             \
    qmvs_row_wf<F>(W, S, B, x + t * p.x_ld, Y, N, p.K, p.add[seg] != 0, row0, lane);                     \
  }
NI_QMVS_MULTI_WF(qmvs_multi_f6g64, 6)
NI_QMVS_MULTI_WF(qmvs_multi_f5g64, 5)
NI_QMVS_MULTI_WF(qmvs_multi_f3g64, 3)

// ---- The folded gated residual (hc_down_sk2 / hc_up_mix2, qwen4.metal) with more work per threadgroup ----
// The one-row kernels' code per (row, weight row) (hc_down_sk2_x / _dot / _stat, hc_up_mix2_prep / _out), so the
// results are theirs bit for bit, for decode (R = 1) and verify rows (R <= 4) alike:
//  hc_down_sk3<NR>: a simdgroup loads its 32 inputs per lane once and runs NR weight rows over them (hc_down_sk2 loads
//    them for every weight row: 16x the weights' bytes from the caches); each threadgroup does every row.
//    grid (rank / (8 NR), S * D / 1024), 256 threads.
//  hc_up_mix3<CPT>: a threadgroup stages h (prep) once for CPT channel blocks of 16 and every row.
//    grid (D / (16 CPT)), 256 threads; rank <= 512.
template <int BITS, int NR>
NI_INLINE void hc_down_sk3_impl(const device float* X, const device float* y, const device float* inj_prev,
                                const device float* w, const device bfloat* winj, const device uint32_t* W,
                                const device bfloat* Ws, const device bfloat* Wb, device float* part, device float* X2,
                                device float* stat, constant HcFoldParams& p, uint2 tg, uint tid, uint sg, uint lane,
                                threadgroup float4* scratch) {
  for (uint r = 0; r < p.R; ++r) {
    float xv[32], xs;
    hc_down_sk2_x(X, y, inj_prev, w, p, tg.y, r, lane, xv, xs);
    for (uint j = 0; j < uint(NR); ++j) hc_down_sk2_dot<BITS>(W, Ws, Wb, part, p, (tg.x * 8 + sg) * NR + j, tg.y, r, lane, xv, xs);
  }
  if (tg.x != 0) return;
  for (uint r = 0; r < p.R; ++r) hc_down_sk2_stat(X, y, inj_prev, w, winj, X2, stat, p, tg.y, r, tid, scratch);
}
#define NI_HC_DOWN_SK3(NAME, BITS, NR)                                                                         \
  [[kernel]] void NAME(const device float* X [[buffer(0)]], const device float* y [[buffer(1)]],               \
                       const device float* inj_prev [[buffer(2)]], const device float* w [[buffer(3)]],        \
                       const device bfloat* winj [[buffer(4)]], const device uint32_t* W [[buffer(5)]],        \
                       const device bfloat* Ws [[buffer(6)]], const device bfloat* Wb [[buffer(7)]],           \
                       device float* part [[buffer(8)]], device float* X2 [[buffer(9)]],                       \
                       device float* stat [[buffer(10)]], constant HcFoldParams& p [[buffer(11)]],             \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],    \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    threadgroup float4 scratch[32];                                                                            \
    hc_down_sk3_impl<BITS, NR>(X, y, inj_prev, w, winj, W, Ws, Wb, part, X2, stat, p, tg, tid, sg, lane, scratch); \
  }
NI_HC_DOWN_SK3(hc_down_sk3_n2_b6, 6, 2)
NI_HC_DOWN_SK3(hc_down_sk3_n2_b4, 4, 2)
NI_HC_DOWN_SK3(hc_down_sk3_n2_b8, 8, 2)

#define NI_HC_UP_MIX3(NAME, BITS, CPT)                                                                         \
  [[kernel]] void NAME(const device float* part [[buffer(0)]], const device float* X2 [[buffer(1)]],           \
                       const device float* stat [[buffer(2)]], const device float* w [[buffer(3)]],            \
                       const device uint32_t* W [[buffer(4)]], const device bfloat* Ws [[buffer(5)]],          \
                       const device bfloat* Wb [[buffer(6)]], device float* u [[buffer(7)]],                   \
                       device float* X [[buffer(8)]], device float* y [[buffer(9)]], device float* inj [[buffer(10)]], \
                       constant HcFoldParams& p [[buffer(11)]], uint tg [[threadgroup_position_in_grid]],      \
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup float hs[4][512];                                                                              \
    threadgroup float hsum[4][32];                                                                             \
    threadgroup float inv[4][4];                                                                               \
    for (uint r = 0; r < p.R; ++r) hc_up_mix2_prep(part, stat, inj, p, tg, r, tid, hs[r], hsum[r], inv[r]);    \
    for (uint r = 0; r < p.R; ++r)                                                                             \
      for (uint j = 0; j < uint(CPT); ++j)                                                                     \
        hc_up_mix2_out<BITS>(X2, w, W, Ws, Wb, u, X, y, p, tg * CPT + j, r, sg, lane, hs[r], hsum[r], inv[r]); \
  }
NI_HC_UP_MIX3(hc_up_mix3_c2_b6, 6, 2)
NI_HC_UP_MIX3(hc_up_mix3_c2_b4, 4, 2)
NI_HC_UP_MIX3(hc_up_mix3_c2_b8, 8, 2)

// ---- The MoE of R <= 4 rows with the decode kernels' arithmetic (moe_gate_up3 / moe_down2_w) ----
// moe4_unique: the distinct experts of the rows' top-k (idx [R, k]): ue[u], the rows that chose it (bit mask) and
// each such row's slot; n[0] = U. One thread (R * k <= 64 entries).
[[kernel]] void moe4_unique(const device uint* idx [[buffer(0)]], device uint* ue [[buffer(1)]],
                            device uint* umask [[buffer(2)]], device uchar* uslot [[buffer(3)]],
                            device uint* n [[buffer(4)]], constant uint2& rk [[buffer(5)]],
                            uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const uint R = rk.x, k = rk.y;
  uint U = 0;
  for (uint r = 0; r < R; ++r)
    for (uint s = 0; s < k; ++s) {
      const uint e = idx[r * k + s];
      uint u = 0;
      while (u < U && ue[u] != e) ++u;
      if (u == U) {
        ue[U] = e;
        umask[U] = 0;
        ++U;
      }
      umask[u] |= 1u << r;
      uslot[u * 4 + r] = uchar(s);
    }
  n[0] = U;
}

// gate/up: h[(r * (k + 1) + slot) * max(I, SI) + i] = silu(gate_e . x_r) * (up_e . x_r). grid (R, max(I, SI) / 16,
// R * k + 1): z = a distinct expert (the last: the shared one); the R threadgroups of an expert's tile are adjacent,
// so rows that chose the same expert share its weights through the cache. 256 threads.
[[kernel]] void moe4_gate_up(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],
                             const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],
                             const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],
                             const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],
                             const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],
                             const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],
                             const device float* xn [[buffer(12)]], device float* h [[buffer(13)]],
                             constant MoeParams& p [[buffer(14)]], const device uint* ue [[buffer(15)]],
                             const device uint* umask [[buffer(16)]], const device uchar* uslot [[buffer(17)]],
                             const device uint* un [[buffer(18)]], constant uint& R [[buffer(19)]],
                             uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint3 ntg3 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 2;
  threadgroup float xs[4096];
  const uint r = tgp.x, u = tgp.z, ntg = ntg3.x;
  const bool shared = u == R * p.k;
  uint slot = p.k, e = 0;
  if (!shared) {
    if (u >= un[0] || !((umask[u] >> r) & 1u)) return;  // uniform across the threadgroup
    slot = uslot[u * 4 + r];
    e = ue[u];
  }
  for (uint i = tid; i < p.D / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)(xn + r * p.D))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.y * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / 64;
  const device uint32_t *G = gw, *U = uw;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  float ag[ROWS], au[ROWS];
  qmv2_rows<4, 16, ROWS, 64>(G, Gs, Gb, xs, N, K, row0, lane, ag);
  qmv2_rows<4, 16, ROWS, 64>(U, Us, Ub, xs, N, K, row0, lane, au);
  if (lane == 0)
    for (int q = 0; q < ROWS; ++q)
      if (row0 + q < N) h[(r * (p.k + 1) + slot) * max(p.I, p.SI) + row0 + q] = silu(ag[q]) * au[q];
}

// down: x[r, d] += sum_slot wts[r, slot] * (down_slot[d] . h[r, slot]) (moe_down2_w's arithmetic, h staged per row).
// grid (R, D / 16), 256 threads; (k + 1) * max(I, SI) <= HS (moe4_down_s: a small instance for shader validation).
#define NI_MOE4_DOWN(NAME, HS)                                                                                  \
  [[kernel]] void NAME(const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],        \
                       const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],       \
                       const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],         \
                       const device float* h [[buffer(6)]], const device uint* idx [[buffer(7)]],              \
                       const device float* wts [[buffer(8)]], device float* x [[buffer(9)]],                   \
                       constant MoeParams& p [[buffer(10)]], uint2 tgp [[threadgroup_position_in_grid]],       \
                       uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]],       \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],    \
                       uint lane [[thread_index_in_simdgroup]]) {                                              \
    threadgroup float hs[HS];                                                                                  \
    const uint r = tgp.x;                                                                                      \
    moe_down2_impl(dw, dsc, dbi, sdw, sds, sdb, h + r * (p.k + 1) * max(p.I, p.SI), idx + r * p.k,             \
                   wts + r * (p.k + 1), x + r * p.D, p, tgp.y, tid, ntg2.x, sg, nsg, lane, hs);                \
  }
NI_MOE4_DOWN(moe4_down, 7168)
NI_MOE4_DOWN(moe4_down_s, 2048)

// ---- Token pairs: each simdgroup reads its 4 weight rows once for two token rows (qmvt_rows, NT = 2); the
// ceil(R / 2) pair simdgroups of a weight block share it through the core's cache. grid (threadgroups) of
// ceil(R / 2) x 2 simdgroups (sg = block * pairs + pair); segments as qmvs_multi.
#define NI_QMVP_MULTI(NAME)                                                                               \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvtParams& p [[buffer(17)]],                       \
      uint tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],             \
      uint lane [[thread_index_in_simdgroup]]) {                                                         \
    const uint P = (p.R + 1) / 2, pair = sg % P, blk = sg / P;                                           \
    uint seg = 0;                                                                                        \
    for (uint i = 1; i < p.nseg; ++i)                                                                    \
      if (tgp >= p.tg_begin[i]) seg = i;                                                                  \
    const uint row0 = ((tgp - p.tg_begin[seg]) * 2 + blk) * 4;                                           \
    const uint N = p.N[seg];                                                                              \
    if (row0 >= N) return;                                                                                \
    const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;                        \
    const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;                          \
    const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;                          \
    device float* Y = (seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3) + 2 * pair * p.y_ld[seg];      \
    const uint Rp = min(2u, p.R - 2 * pair);                                                             \
    const device float* xp = x + 2 * pair * p.x_ld;                                                      \
    if (p.bits[seg] == 8) qmvt_store<8, 2, 4>(W, S, B, xp, Y, p.x_ld, p.y_ld[seg], Rp, N, p.K, p.add[seg] != 0, row0, lane); \
    else qmvt_store<4, 2, 4>(W, S, B, xp, Y, p.x_ld, p.y_ld[seg], Rp, N, p.K, p.add[seg] != 0, row0, lane);  \
  }
NI_QMVP_MULTI(qmvp_multi)
