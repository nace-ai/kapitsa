#include "common.h"

// ---- Small-batch ("verify") kernels: R <= 4 rows processed against one weight read ----
// Used by speculative decoding (the trunk verifies the pending token plus drafts) and by the MTP
// head (committed-history rows). Rows are [R, K] fp32 in device memory.

// y[r] = rmsnorm(x[r]) * w   (fp32). grid: R threadgroups x 256.
struct RowsNormParams {
  uint D;
  float eps;
};
[[kernel]] void rows_norm_f32(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant RowsNormParams& p [[buffer(3)]],
                              uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const device float* xr = x + r * p.D;
  float ss = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) ss += xr[i] * xr[i];
  ss = tg_sum(ss, red, tid, ntg);
  float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  for (uint i = tid; i < p.D; i += ntg) y[r * p.D + i] = xr[i] * inv * float(w[i]);
}

// MTP input: u[r] = [rmsnorm(e[r]) * we, rmsnorm(h[r]) * wh]  ([R, 2D]). grid: R x 256.
[[kernel]] void mtp_concat_norm(const device float* e [[buffer(0)]], const device float* h [[buffer(1)]],
                                const device bfloat* we [[buffer(2)]], const device bfloat* wh [[buffer(3)]],
                                device float* u [[buffer(4)]], constant RowsNormParams& p [[buffer(5)]],
                                uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const device float* er = e + r * p.D;
  const device float* hr = h + r * p.D;
  float se = 0.0f, sh = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) {
    se += er[i] * er[i];
    sh += hr[i] * hr[i];
  }
  se = tg_sum(se, red, tid, ntg);
  sh = tg_sum(sh, red, tid, ntg);
  float ie = metal::precise::rsqrt(se / float(p.D) + p.eps), ih = metal::precise::rsqrt(sh / float(p.D) + p.eps);
  for (uint i = tid; i < p.D; i += ntg) {
    u[r * 2 * p.D + i] = er[i] * ie * float(we[i]);
    u[r * 2 * p.D + p.D + i] = hr[i] * ih * float(wh[i]);
  }
}

// y[t, n] (+)= W[n] . x[t] for t < R (NT = compile-time max rows). x read from device (L1-cached).
struct QmvrParams {
  uint N;
  uint K;
  uint R;       // valid rows (<= NT)
  uint x_ld;    // floats per x row
  uint y_ld;    // floats per y row
  uint add;
};

template <int BITS, int GS, int NT, int ROWS>
NI_INLINE void qmvr_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         const device float* x, device float* y, constant QmvrParams& p, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int VPL = 16;
  constexpr int WORDS = VPL / PER_WORD;
  const uint K = p.K;
  const uint wpr = K / PER_WORD, gpr = K / GS;
  float acc[ROWS][NT];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, p.N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL];
    float xsum[NT];
    for (int t = 0; t < NT; ++t) {
      const uint tt = min(uint(t), p.R - 1);
      const device float4* xp = (const device float4*)(x + tt * p.x_ld + k0);
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = xp[i];
        xsum[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
      }
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
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
      float wf[VPL];
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) wf[i * PER_WORD + j] = float(wv[i] & (MASK << (BITS * j)));
      for (int t = 0; t < NT; ++t) {
        float d = 0.0f;
        for (int j = 0; j < VPL; ++j) d += wf[j] * xv[t][j];
        acc[r][t] += s * d + b * xsum[t];
      }
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N && uint(t) < p.R) {
        device float* dst = y + t * p.y_ld + row;
        *dst = p.add ? *dst + v : v;
      }
    }
}

#define NI_QMVR(NAME, BITS, GS, NT, ROWS)                                                                 \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],      \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                 \
    if (row0 >= p.N) return;                                                                             \
    qmvr_impl<BITS, GS, NT, ROWS>(w, s, b, x, y, p, row0, lane);                                         \
  }

NI_QMVR(qmvr_b4_g64_t2, 4, 64, 2, 2)
NI_QMVR(qmvr_b4_g64_t3, 4, 64, 3, 2)
NI_QMVR(qmvr_b4_g64_t4, 4, 64, 4, 2)
NI_QMVR(qmvr_b4_g64_t1, 4, 64, 1, 2)
NI_QMVR(qmvr_b8_g64_t1, 8, 64, 1, 2)
NI_QMVR(qmvr_b8_g64_t2, 8, 64, 2, 2)
NI_QMVR(qmvr_b8_g64_t3, 8, 64, 3, 2)
NI_QMVR(qmvr_b8_g64_t4, 8, 64, 4, 2)

// ---- per-row causal decode attention: row r is at position pos0 + r ----
// part_o: [R, Hq, nchunks, D], part_ml: [R, Hq, nchunks, 2]
struct AttnRowsParams {
  uint Hq;
  uint Hkv;
  uint pos0;     // position of row 0
  uint chunk;
  uint max_seq;
  float scale;
  uint nchunks;
};

[[kernel]] void attn_rows_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                  const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                  device float* part_ml [[buffer(4)]], constant AttnRowsParams& p [[buffer(5)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup half ktile[kTile * kHD];
  threadgroup half vtile[kTile * kHD];
  const uint ntg = ntg3.x;
  const uint chunk = tg.x, kvh = tg.y, r = tg.z;
  const uint G = p.Hq / p.Hkv;
  const uint hq = kvh * G + sg;
  const uint L = p.pos0 + r + 1;
  const uint start = chunk * p.chunk;
  const uint end = min(start + p.chunk, L);
  float qv[8], acc[8];
  for (int i = 0; i < 8; ++i) {
    qv[i] = q[(r * p.Hq + hq) * kHD + lane * 8 + i] * p.scale;
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const device half* kbase = kc + kvh * p.max_seq * kHD;
  const device half* vbase = vc + kvh * p.max_seq * kHD;
  for (uint t0 = start; t0 < end; t0 += kTile) {
    uint n = min(kTile, end - t0);
    for (uint i = tid; i < n * kHD / 4; i += ntg) {
      ((threadgroup half4*)ktile)[i] = ((const device half4*)(kbase + t0 * kHD))[i];
      ((threadgroup half4*)vtile)[i] = ((const device half4*)(vbase + t0 * kHD))[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < G) {
      for (uint j = 0; j < n; ++j) {
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
  if (sg < G) {
    uint base = (r * p.Hq + hq) * p.nchunks + chunk;
    for (int i = 0; i < 8; ++i) part_o[base * kHD + lane * 8 + i] = acc[i];
    if (lane == 0) {
      part_ml[base * 2 + 0] = m;
      part_ml[base * 2 + 1] = l;
    }
  }
}

// grid: (Hq, R) x 256. out[r, h*D + d] = combined * sigmoid(gate[r, h, d])
[[kernel]] void attn_rows_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                 const device float* qg [[buffer(2)]], device float* out [[buffer(3)]],
                                 constant AttnRowsParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint h = tg.x, r = tg.y, d = tid2.x;
  const uint base = (r * p.Hq + h) * p.nchunks;
  float M = -INFINITY;
  for (uint c = 0; c < p.nchunks; ++c) M = max(M, part_ml[(base + c) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint c = 0; c < p.nchunks; ++c) {
    float mc = part_ml[(base + c) * 2];
    if (mc == -INFINITY) continue;
    float w = metal::fast::exp(mc - M);
    num += w * part_o[(base + c) * kHD + d];
    den += w * part_ml[(base + c) * 2 + 1];
  }
  float gate = qg[(r * p.Hq + h) * 2 * kHD + kHD + d];
  out[(r * p.Hq + h) * kHD + d] = (num / den) * sigmoid(gate);
}

// ---- per-row MoE (R tokens, each with its own k experts) ----
// gate/up grid: (row tiles over max(I,SI), k+1 slots, R rows). xn: [R, D] normalized inputs.
// h: [R, k+1, Imax]. idx: [R, k].
template <int GS, int SBITS>
NI_INLINE void moe_gu_rows_impl(const device uint32_t* gw, const device bfloat* gsc, const device bfloat* gbi,
                                const device uint32_t* uw, const device bfloat* usc, const device bfloat* ubi,
                                const device uint32_t* sgw, const device bfloat* sgs, const device bfloat* sgb,
                                const device uint32_t* suw, const device bfloat* sus, const device bfloat* sub,
                                const device float* xn, device float* h, constant MoeParams& p,
                                const device uint* idx, uint3 tgp, uint tid, uint ntg, uint sg, uint nsg, uint lane,
                                threadgroup float* xs) {
  constexpr int ROWS = 2;
  const uint slot = tgp.y, r = tgp.z;
  const bool shared = slot == p.k;
  for (uint i = tid; i < p.D / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)(xn + r * p.D))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / GS;
  const device uint32_t *G = gw, *U = uw;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    uint e = idx[r * p.k + slot];
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  float ag[ROWS], au[ROWS];
  if (shared) {
    qmv2_rows<SBITS, 16, ROWS, 64>(G, Gs, Gb, xs, N, K, row0, lane, ag);
    qmv2_rows<SBITS, 16, ROWS, 64>(U, Us, Ub, xs, N, K, row0, lane, au);
  } else {
    qmv2_rows<4, 16, ROWS, GS>(G, Gs, Gb, xs, N, K, row0, lane, ag);
    qmv2_rows<4, 16, ROWS, GS>(U, Us, Ub, xs, N, K, row0, lane, au);
  }
  const uint Imax = max(p.I, p.SI);
  if (lane == 0)
    for (int q = 0; q < ROWS; ++q)
      if (row0 + q < N) h[(r * (p.k + 1) + slot) * Imax + row0 + q] = silu(ag[q]) * au[q];
}

#define NI_MOE_GU_ROWS(NAME, GS, SBITS)                                                                           \
  [[kernel]] void NAME(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],      \
                       const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],      \
                       const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],       \
                       const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],     \
                       const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],     \
                       const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],     \
                       const device float* xn [[buffer(12)]], device float* h [[buffer(13)]],                \
                       constant MoeParams& p [[buffer(14)]], const device uint* idx [[buffer(15)]],          \
                       uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint3 ntg3 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],   \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {    \
    threadgroup float xs[4096];                                                                             \
    moe_gu_rows_impl<GS, SBITS>(gw, gsc, gbi, uw, usc, ubi, sgw, sgs, sgb, suw, sus, sub, xn, h, p, idx, tgp, tid,  \
                         ntg3.x, sg, nsg, lane, xs);                                                        \
  }
NI_MOE_GU_ROWS(moe_gate_up_rows_g64, 64, 4)
NI_MOE_GU_ROWS(moe_gate_up_rows_g32_s8, 32, 8)

// down grid: (row tiles over D, R). x[r, row] += sum_slot wts[r, slot] * (down_slot[row] . h[r, slot]).
// The shared expert's down projection may use a different quantization (sbits/sgs below).
struct MoeRowsDownParams {
  uint D, I, SI, k;
  uint shared_bits;  // 4 or 8
};
template <int GS, int SBITS>
NI_INLINE void moe_down_rows_impl(const device uint32_t* dw, const device bfloat* dsc, const device bfloat* dbi,
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
  float tot[ROWS] = {0.0f, 0.0f};
  for (uint slot = 0; slot <= p.k; ++slot) {
    float a[ROWS];
    if (slot == p.k) {
      qmv2_rows<SBITS, 16, ROWS, 64>(sdw, sds, sdb, hs + slot * Imax, p.D, p.SI, row0, lane, a);
    } else {
      const uint K = p.I, wpr = K / 8, gpr = K / GS;
      const uint e = idx[r * p.k + slot];
      qmv2_rows<4, 16, ROWS, GS>(dw + e * p.D * wpr, dsc + e * p.D * gpr, dbi + e * p.D * gpr, hs + slot * Imax, p.D,
                                 K, row0, lane, a);
    }
    const float ws = wts[r * (p.k + 1) + slot];
    for (int q = 0; q < ROWS; ++q) tot[q] += ws * a[q];
  }
  if (lane == 0)
    for (int q = 0; q < ROWS; ++q)
      if (row0 + q < p.D) x[r * p.D + row0 + q] += tot[q];
}

#define NI_MOE_DOWN_ROWS(NAME, GS, SBITS, HS)                                                                \
  [[kernel]] void NAME(const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],      \
                       const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],     \
                       const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],       \
                       const device float* h [[buffer(6)]], const device uint* idx [[buffer(7)]],            \
                       const device float* wts [[buffer(8)]], device float* x [[buffer(9)]],                 \
                       constant MoeRowsDownParams& p [[buffer(10)]], uint2 tgp [[threadgroup_position_in_grid]], \
                       uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]],     \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],  \
                       uint lane [[thread_index_in_simdgroup]]) {                                            \
    threadgroup float hs[HS];                                                                               \
    moe_down_rows_impl<GS, SBITS>(dw, dsc, dbi, sdw, sds, sdb, h, idx, wts, x, p, tgp, tid, ntg2.x, sg, nsg, lane, hs); \
  }
// (k + 1) * max(I, SI) <= 4608 staged floats; the _w forms (wide experts: Qwen3.8-Flash-Next) <= 7168
NI_MOE_DOWN_ROWS(moe_down_rows_g64, 64, 4, 9 * 512)
NI_MOE_DOWN_ROWS(moe_down_rows_g32_s8, 32, 8, 9 * 512)
NI_MOE_DOWN_ROWS(moe_down_rows_g64_w, 64, 4, 7168)

// Router top-k of row r with E = 256 (one simdgroup): idx[r * k ..], wts[r * (k + 1) ..] (the shared expert's gate
// last); ids (optional): the chosen experts again, for a caller in the same threadgroup (moe8_plan's fused top-k).
NI_INLINE void topk256_row(const device float* logits, const device float* shared, device uint* idx, device float* wts,
                           constant TopkParams& p, uint r, uint lane, threadgroup uint* ids) {
  constexpr uint PER = 8;
  const device float* lg = logits + r * 256;
  float4 a = ((const device float4*)lg)[lane * 2], b = ((const device float4*)lg)[lane * 2 + 1];
  float v[PER] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float mx = v[0];
  for (uint i = 1; i < PER; ++i) mx = max(mx, v[i]);
  mx = simd_max(mx);
  float sum = 0.0f;
  for (uint i = 0; i < PER; ++i) {
    v[i] = metal::fast::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  float sel = 0.0f, wsel[8];
  uint isel[8];
  for (uint q = 0; q < p.k; ++q) {
    float best = -1.0f;
    uint bi = 0;
    for (uint i = 0; i < PER; ++i)
      if (v[i] > best) { best = v[i]; bi = i; }
    float g = simd_max(best);
    uint win = simd_min(best == g ? lane * PER + bi : 0xffffffffu);
    if (win / PER == lane) v[win % PER] = -1.0f;
    wsel[q] = g;
    isel[q] = win;
    sel += g;
  }
  if (lane < p.k) {
    float w = 0.0f;
    uint ii = 0;
    for (uint q = 0; q < p.k; ++q)
      if (q == lane) { w = wsel[q]; ii = isel[q]; }
    idx[r * p.k + lane] = ii;
    if (ids) ids[r * p.k + lane] = ii;
    wts[r * (p.k + 1) + lane] = p.norm ? w / sel : w / sum;
  }
  if (lane == p.k) wts[r * (p.k + 1) + p.k] = sigmoid(shared[r]);
}

// Router top-k for R rows with E = 256 (one simdgroup per row).
[[kernel]] void moe_topk256_rows(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                                 device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                                 constant TopkParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  topk256_row(logits, shared, idx, wts, p, r, lane, nullptr);
}

// ---- Gated DeltaNet verify with a rollback tape ----
// The committed ("base") conv window and recurrent state lag behind by the rows accepted in the
// previous cycle (n_pend). Each verify first folds those rows in (conv: raw rows; recurrence: the
// recorded (k, g, delta) per row, a rank-1 update each) and stores the new base, then evaluates
// the R new rows without writing the state, recording their tape. Accepted rows are folded in by
// the next call, rejected ones are simply never applied. R = 0 just flushes the pending rows.
// KFlash tree verifies (Qwen35::set_verify_tree): the R rows are a tree, row t's parent row tree[kKfPar + t] (the KFlash
// tree descriptor, kflash.h; row 0, the anchor, has none): a row's conv window and recurrent state are its ancestors' fold. Its
// accepted path's rows are folded by the next pass in path order: tape rows path[0 .. n_pend) (pidx; kf_accept).
struct GdnVerifyParams {
  uint R;
  uint n_pend;
  uint Hk;
  uint Hv;
  uint C;
  float eps;
  uint tree;  // the rows form a tree (the *_tree kernels)
  uint pidx;  // the pending rows are tape rows path[0 .. n_pend), not 0 .. n_pend
  uint pin;   // gdn_step_verify8s_tree: the recurrence's update in one explicit form (gdn_update), in the step, the fold and
              // the replay of a jumping row alike
};

// grid: C/128 threadgroups x 128 threads (as gdn_prep).
template <bool TREE>
NI_INLINE void gdn_prep_verify_impl(const device float* qkv, const device float* a, const device float* bb,
                                    device float* conv_state, const device bfloat* conv_w,
                                    const device bfloat* A_log, const device bfloat* dt_bias, device float* q_out,
                                    device float* k_out, device float* v_out, device float* g_out,
                                    device float* beta_out, device float* pend_qkv, constant GdnVerifyParams& p,
                                    const device int* tree, const device int* path, uint tg, uint tid,
                                    threadgroup float* scratch) {
  constexpr uint D = 128;
  const uint c = tg * D + tid;
  const uint C = p.C;
  const uint qk = p.Hk * D;
  const bool is_q = c < qk, is_k = c >= qk && c < 2 * qk;
  const uint R = TREE && p.tree ? kf_live(tree, p.R) : p.R;  // a tree: its live rows
  // fold pending rows into the committed window
  float w3[3] = {conv_state[0 * C + c], conv_state[1 * C + c], conv_state[2 * C + c]};
  for (uint t = 0; t < p.n_pend; ++t) {
    w3[0] = w3[1];
    w3[1] = w3[2];
    w3[2] = pend_qkv[(TREE && p.pidx ? uint(path[t]) : t) * C + c];
  }
  for (uint r = 0; r < 3; ++r) conv_state[r * C + c] = w3[r];
  float w0 = conv_w[c * 4 + 0], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], wq = conv_w[c * 4 + 3];
  float h0 = w3[0], h1 = w3[1], h2 = w3[2];
  float tw1[8], tw2[8], tin[8];  // tree rows: each row's window (its last two entries) and input, by row
  for (uint t = 0; t < R; ++t) {
    float in3 = qkv[t * C + c];
    pend_qkv[t * C + c] = in3;
    if (TREE && p.tree && t > 0) {  // the parent's window shifted by the parent's input
      const uint pr = uint(tree[kKfPar + t]);
      h0 = tw1[pr];
      h1 = tw2[pr];
      h2 = tin[pr];
    }
    if (TREE) {
      tw1[t] = h1;
      tw2[t] = h2;
      tin[t] = in3;
    }
    float o = silu(w0 * h0 + w1 * h1 + w2 * h2 + wq * in3);
    h0 = h1;
    h1 = h2;
    h2 = in3;
    if (is_q || is_k) {
      float ss = tg_sum(o * o, scratch, tid, D);
      o *= metal::precise::rsqrt(ss + 1e-6f);
      if (is_q) q_out[t * qk + c] = o * metal::precise::rsqrt(float(D));
      else k_out[t * qk + (c - qk)] = o;
    } else {
      v_out[t * (C - 2 * qk) + (c - 2 * qk)] = o;
    }
  }
  if (tg == 0 && tid < p.Hv) {
    float A = metal::precise::exp(float(A_log[tid]));
    float dtb = float(dt_bias[tid]);
    for (uint t = 0; t < R; ++t) {
      g_out[t * p.Hv + tid] = metal::precise::exp(-A * softplus(a[t * p.Hv + tid] + dtb));
      beta_out[t * p.Hv + tid] = sigmoid(bb[t * p.Hv + tid]);
    }
  }
}
[[kernel]] void gdn_prep_verify(const device float* qkv [[buffer(0)]], const device float* a [[buffer(1)]],
                                const device float* bb [[buffer(2)]], device float* conv_state [[buffer(3)]],
                                const device bfloat* conv_w [[buffer(4)]], const device bfloat* A_log [[buffer(5)]],
                                const device bfloat* dt_bias [[buffer(6)]], device float* q_out [[buffer(7)]],
                                device float* k_out [[buffer(8)]], device float* v_out [[buffer(9)]],
                                device float* g_out [[buffer(10)]], device float* beta_out [[buffer(11)]],
                                device float* pend_qkv [[buffer(12)]], constant GdnVerifyParams& p [[buffer(13)]],
                                uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[32];
  gdn_prep_verify_impl<false>(qkv, a, bb, conv_state, conv_w, A_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
                              pend_qkv, p, nullptr, nullptr, tg, tid, scratch);
}
// tree rows (p.tree; the live rows alone) and / or pending rows on a tree's path (p.pidx): tree = the KFlash tree
// descriptor, path = kf_accept's (encode_flush folds a tree cycle's accepted rows with it)
[[kernel]] void gdn_prep_verify_tree(const device float* qkv [[buffer(0)]], const device float* a [[buffer(1)]],
                                     const device float* bb [[buffer(2)]], device float* conv_state [[buffer(3)]],
                                     const device bfloat* conv_w [[buffer(4)]],
                                     const device bfloat* A_log [[buffer(5)]],
                                     const device bfloat* dt_bias [[buffer(6)]], device float* q_out [[buffer(7)]],
                                     device float* k_out [[buffer(8)]], device float* v_out [[buffer(9)]],
                                     device float* g_out [[buffer(10)]], device float* beta_out [[buffer(11)]],
                                     device float* pend_qkv [[buffer(12)]], constant GdnVerifyParams& p [[buffer(13)]],
                                     const device int* tree [[buffer(14)]], const device int* path [[buffer(15)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[32];
  gdn_prep_verify_impl<true>(qkv, a, bb, conv_state, conv_w, A_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
                             pend_qkv, p, tree, path, tg, tid, scratch);
}

// grid: Hv threadgroups x 1024 threads (as gdn_step).
// pend_k: [Rmax, Hv, 128], pend_g: [Rmax, Hv], pend_d: [Rmax, Hv, 128]
// PIDX: the pending rows are tape rows path[0 .. n_pend) when p.pidx (a tree cycle's accepted path; rows are a chain)
template <bool SIG, bool PIDX = false>
NI_INLINE void gdn_step_verify_impl(const device float* q, const device float* k, const device float* v,
                                    const device float* g, const device float* beta, device float* state,
                                    const device float* z, const device bfloat* norm_w, device float* out,
                                    device float* pend_k, device float* pend_g, device float* pend_d,
                                    constant GdnVerifyParams& p, uint h, uint sg, uint lane,
                                    threadgroup float* scratch, const device int* path = nullptr) {
  constexpr uint D = 128;
  const uint hk = h / (p.Hv / p.Hk);
  device float* S = state + h * D * D;
  float4 st[4];
  for (int r = 0; r < 4; ++r) st[r] = *(device float4*)(S + (sg * 4 + r) * D + lane * 4);
  // fold pending (accepted) rows from the tape
  for (uint t0 = 0; t0 < p.n_pend; ++t0) {
    const uint t = PIDX && p.pidx ? uint(path[t0]) : t0;
    float4 kk = *(const device float4*)(pend_k + (t * p.Hv + h) * D + lane * 4);
    float gg = pend_g[t * p.Hv + h];
    for (int r = 0; r < 4; ++r) st[r] = st[r] * gg + kk * pend_d[(t * p.Hv + h) * D + sg * 4 + r];
  }
  if (p.n_pend > 0)
    for (int r = 0; r < 4; ++r) *(device float4*)(S + (sg * 4 + r) * D + lane * 4) = st[r];
  threadgroup_barrier(mem_flags::mem_device);  // all tape reads done before it is overwritten
  float nw[4];
  for (int r = 0; r < 4; ++r) nw[r] = float(norm_w[sg * 4 + r]);
  for (uint t = 0; t < p.R; ++t) {
    float4 q4 = *(const device float4*)(q + (t * p.Hk + hk) * D + lane * 4);
    float4 k4 = *(const device float4*)(k + (t * p.Hk + hk) * D + lane * 4);
    float gg = g[t * p.Hv + h], be = beta[t * p.Hv + h];
    if (sg == 0) *(device float4*)(pend_k + (t * p.Hv + h) * D + lane * 4) = k4;
    if (sg == 0 && lane == 0) pend_g[t * p.Hv + h] = gg;
    float o[4];
    float ss = 0.0f;
    for (int r = 0; r < 4; ++r) {
      st[r] *= gg;
      float kv = simd_sum(dot(st[r], k4));
      float delta = (v[(t * p.Hv + h) * D + sg * 4 + r] - kv) * be;
      if (lane == 0) pend_d[(t * p.Hv + h) * D + sg * 4 + r] = delta;
      st[r] += k4 * delta;
      o[r] = simd_sum(dot(st[r], q4));
      ss += o[r] * o[r];
    }
    if (lane == 0) scratch[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(scratch[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::precise::rsqrt(tot / float(D) + p.eps);
    if (lane < 4) {
      uint dv = sg * 4 + lane;
      float ov = lane == 0 ? o[0] : lane == 1 ? o[1] : lane == 2 ? o[2] : o[3];
      float nv = lane == 0 ? nw[0] : lane == 1 ? nw[1] : lane == 2 ? nw[2] : nw[3];
      uint idx = (t * p.Hv + h) * D + dv;
      out[idx] = ov * inv * nv * gdn_gate<SIG>(z[idx]);
    }
  }
}

#define NI_GDN_STEP_VERIFY(NAME, SIG)                                                                         \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],             \
                       const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],             \
                       const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],            \
                       const device float* z [[buffer(6)]], const device bfloat* norm_w [[buffer(7)]],       \
                       device float* out [[buffer(8)]], device float* pend_k [[buffer(9)]],                  \
                       device float* pend_g [[buffer(10)]], device float* pend_d [[buffer(11)]],             \
                       constant GdnVerifyParams& p [[buffer(12)]], uint h [[threadgroup_position_in_grid]],  \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float scratch[32];                                                                           \
    gdn_step_verify_impl<SIG>(q, k, v, g, beta, state, z, norm_w, out, pend_k, pend_g, pend_d, p, h, sg, lane, \
                              scratch);                                                                      \
  }
NI_GDN_STEP_VERIFY(gdn_step_verify, false)
NI_GDN_STEP_VERIFY(gdn_step_verify_sg, true)
// pending rows on a tree cycle's path (encode_flush after a KFlash tree cycle): path = kf_accept's
[[kernel]] void gdn_step_verify_pidx(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                                     const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],
                                     const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                                     const device float* z [[buffer(6)]], const device bfloat* norm_w [[buffer(7)]],
                                     device float* out [[buffer(8)]], device float* pend_k [[buffer(9)]],
                                     device float* pend_g [[buffer(10)]], device float* pend_d [[buffer(11)]],
                                     constant GdnVerifyParams& p [[buffer(12)]], const device int* path [[buffer(13)]],
                                     uint h [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  gdn_step_verify_impl<false, true>(q, k, v, g, beta, state, z, norm_w, out, pend_k, pend_g, pend_d, p, h, sg, lane,
                                    scratch, path);
}

// ---- per-row sampling ----
// topk_stage1 with a row dimension: grid (ceil(V/chunk), R).
NI_INLINE void topk_stage1_rows_impl(const device float* logits, device float* cv, device uint* ci,
                                     constant TopkStageParams& p, uint2 tg, uint2 tid2, uint2 ntg2,
                                     threadgroup float* sv, threadgroup uint* si) {
  const uint tid = tid2.x, ntg = ntg2.x, r = tg.y, ntgs = (p.V + p.chunk - 1) / p.chunk;
  const device float* lg = logits + size_t(r) * p.V;
  float v[8];
  uint base = tg.x * p.chunk + tid * 8;
  for (int j = 0; j < 8; ++j) v[j] = base + j < p.V ? lg[base + j] : -INFINITY;
  for (uint q = 0; q < p.K; ++q) {
    float bv = -INFINITY;
    uint bi = 0xffffffff;
    for (int j = 0; j < 8; ++j)
      if (v[j] > bv) { bv = v[j]; bi = base + j; }
    tg_argmax(bv, bi, sv, si, tid, ntg);
    if (bi >= base && bi < base + 8) v[bi - base] = -INFINITY;
    if (tid == 0) {
      cv[(r * ntgs + tg.x) * p.K + q] = bv;
      ci[(r * ntgs + tg.x) * p.K + q] = bi;
    }
  }
}
[[kernel]] void topk_stage1_rows(const device float* logits [[buffer(0)]], device float* cv [[buffer(1)]],
                                 device uint* ci [[buffer(2)]], constant TopkStageParams& p [[buffer(3)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                                 uint2 ntg2 [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  topk_stage1_rows_impl(logits, cv, ci, p, tg, tid2, ntg2, sv, si);
}
// a tree verify's rows (KFlash): its live rows alone (tree = the KFlash tree descriptor)
[[kernel]] void topk_stage1_rows_tree(const device float* logits [[buffer(0)]], device float* cv [[buffer(1)]],
                                      device uint* ci [[buffer(2)]], constant TopkStageParams& p [[buffer(3)]],
                                      const device int* tree [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                      uint2 tid2 [[thread_position_in_threadgroup]],
                                      uint2 ntg2 [[threads_per_threadgroup]], uint2 ntgs [[threadgroups_per_grid]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  if (tg.y >= kf_live(tree, ntgs.y)) return;
  topk_stage1_rows_impl(logits, cv, ci, p, tg, tid2, ntg2, sv, si);
}

// Draft confidence for the verify budget (src/model/verify_budget.h): the largest candidate probability of
// one row of logits. One threadgroup of 1024 over V floats: each thread keeps a running max and a sum of
// exp(v - max), the simdgroups reduce their pairs, simdgroup 0 reduces those, and lane 0 writes
// exp(max - (max + log(sum))) = 1 / sum. Non-finite logits leave their thread's pair alone (-inf, 0), so a
// poisoned row can only reach the host as a confidence outside [0, 1], which observe() drops. (The budget
// itself is ported from Splash's runtime/model/VerifyBudget.{hpp,cpp}, Apache-2.0; Splash's draft selector
// writes the confidence inside its sampler, this kernel reads the draft's logits row instead so the sampler
// itself stays untouched.)
struct DraftConfParams {
  uint V;
};
[[kernel]] void draft_conf_rows(const device float* logits [[buffer(0)]], device float* conf [[buffer(1)]],
                                constant DraftConfParams& p [[buffer(2)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  float m = -INFINITY, s = 0.0f;
  for (uint i = tid; i < p.V; i += ntg) {
    const float v = logits[i];
    if (v > m) {
      s = s * exp(m - v) + 1.0f;
      m = v;
    } else if (v > -INFINITY) {
      s += exp(v - m);
    }
  }
  for (uint off = 16; off > 0; off >>= 1) {
    const float om = simd_shuffle_down(m, off), os = simd_shuffle_down(s, off);
    if (om > m) {
      s = s * exp(m - om) + os;
      m = om;
    } else if (om > -INFINITY) {
      s += os * exp(om - m);
    }
  }
  threadgroup float sm[32];
  threadgroup float ss[32];
  if (lane == 0) {
    sm[sg] = m;
    ss[sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const uint nsg = (ntg + 31) >> 5;
    m = lane < nsg ? sm[lane] : -INFINITY;
    s = lane < nsg ? ss[lane] : 0.0f;
    for (uint off = 16; off > 0; off >>= 1) {
      const float om = simd_shuffle_down(m, off), os = simd_shuffle_down(s, off);
      if (om > m) {
        s = s * exp(m - om) + os;
        m = om;
      } else if (om > -INFINITY) {
        s += os * exp(om - m);
      }
    }
    if (lane == 0) conf[0] = 1.0f / s;
  }
}

// Per-row stage 2: grid R threadgroups x 1024. Row r reads its candidates, samples with RNG step
// (step + r) and writes tokens[out_index + r]. Indices >= remap_from map to remap_base + (i - remap_from)
// (draft heads that score a vocabulary subset made of two row ranges).
struct SampleRowsParams {
  uint n;        // candidates per row
  uint K;
  float temp;
  float top_p;
  uint seed;
  uint step;
  uint out_index;
  uint remap_from;  // 0 = no remap
  uint remap_base;
  uint table_from;  // 0 = none; indices >= table_from are table[i - table_from] (a draft head's gathered rows)
};
// A row's sample from its K best candidates (topv / topi, value desc, id asc): top-p on the untempered distribution,
// then temperature, with RNG step (p.step + rkey; the row r, or a tree verify row's node depth); writes tokens[p.out_index + r]. (One thread.)
NI_INLINE void sample_rows_pick(threadgroup const float* topv, threadgroup const uint* topi, uint K,
                                constant SampleRowsParams& p, uint r, uint rkey, const device uint* table,
                                device int* tokens) {
  uint chosen = topi[0];
  if (p.temp > 0.0f && K > 1) {
    // top-p is taken on the untempered distribution (llama.cpp's and mlx-lm's order: top-k,
    // top-p, then temperature), then the kept candidates are sampled at temperature p.temp.
    float probs[64];
    float tot = 0.0f;
    for (uint q = 0; q < K; ++q) {
      probs[q] = metal::precise::exp(topv[q] - topv[0]);
      tot += probs[q];
    }
    uint keep = K;
    float cum = 0.0f;
    for (uint q = 0; q < K; ++q) {
      cum += probs[q] / tot;
      if (cum >= p.top_p) { keep = q + 1; break; }
    }
    float kt = 0.0f;
    for (uint q = 0; q < keep; ++q) {
      probs[q] = metal::precise::exp((topv[q] - topv[0]) / p.temp);
      kt += probs[q];
    }
    float u = float(pcg_hash(p.seed ^ pcg_hash(p.step + rkey)) & 0xffffff) / 16777216.0f * kt;
    chosen = topi[keep - 1];
    float acc = 0.0f;
    for (uint q = 0; q < keep; ++q) {
      acc += probs[q];
      if (u < acc) { chosen = topi[q]; break; }
    }
  }
  if (p.table_from && chosen >= p.table_from) chosen = table[chosen - p.table_from];
  else if (p.remap_from && chosen >= p.remap_from) chosen = p.remap_base + (chosen - p.remap_from);
  tokens[p.out_index + r] = int(chosen);
}

// rkey: the row's RNG step offset (r; a tree verify's row: its node's depth)
NI_INLINE void sample_stage2_rows_impl(const device float* cv, const device uint* ci, device int* tokens,
                                       constant SampleRowsParams& p, const device uint* table, uint r, uint rkey,
                                       uint tid, uint ntg, threadgroup float* sv, threadgroup uint* si,
                                       threadgroup float* topv, threadgroup uint* topi) {
  constexpr int PER = 4;
  float v[PER];
  uint id[PER];
  for (int j = 0; j < PER; ++j) {
    uint k = tid * PER + j;
    v[j] = k < p.n ? cv[r * p.n + k] : -INFINITY;
    id[j] = k < p.n ? ci[r * p.n + k] : 0xffffffff;
  }
  uint K = max(1u, min(p.K, 64u));
  if (p.temp <= 0.0f) K = 1;
  for (uint q = 0; q < K; ++q) {
    float bv = -INFINITY;
    uint bi = 0xffffffff;
    int bj = -1;
    for (int j = 0; j < PER; ++j)
      if (v[j] > bv || (v[j] == bv && id[j] < bi)) { bv = v[j]; bi = id[j]; bj = j; }
    float wv = bv;
    uint wi = bi;
    tg_argmax(wv, wi, sv, si, tid, ntg);
    if (bj >= 0 && bi == wi && bv == wv) v[bj] = -INFINITY;
    if (tid == 0) { topv[q] = wv; topi[q] = wi; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) sample_rows_pick(topv, topi, K, p, r, rkey, table, tokens);
}
[[kernel]] void sample_stage2_rows(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                                   device int* tokens [[buffer(2)]], constant SampleRowsParams& p [[buffer(3)]],
                                   const device uint* table [[buffer(4)]],
                                   uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float topv[64];
  threadgroup uint topi[64];
  sample_stage2_rows_impl(cv, ci, tokens, p, table, r, r, tid, ntg, sv, si, topv, topi);
}
// A tree verify's rows (DFlash2 tree drafting): row r draws with the uniform of the position its token takes, step +
// depth of node r (tree[kKfDep + r], the KFlash tree descriptor), the same uniform plain decoding's row at that position draws:
// siblings share it, and only one of them can lie on the accepted path. The live rows alone (no sample past them).
[[kernel]] void sample_stage2_rows_tree(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                                        device int* tokens [[buffer(2)]], constant SampleRowsParams& p [[buffer(3)]],
                                        const device uint* table [[buffer(4)]], const device int* tree [[buffer(5)]],
                                        uint r [[threadgroup_position_in_grid]],
                                        uint tid [[thread_position_in_threadgroup]],
                                        uint ntg [[threads_per_threadgroup]], uint nr [[threadgroups_per_grid]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float topv[64];
  threadgroup uint topi[64];
  if (r >= kf_live(tree, nr)) return;
  sample_stage2_rows_impl(cv, ci, tokens, p, table, r, uint(tree[kKfDep + r]), tid, ntg, sv, si, topv, topi);
}

// The same two passes on simdgroup reductions (no threadgroup barrier per candidate): the same candidates in the same
// order, so the same tokens. topk_stage1_rows_sg: each simdgroup's 256 logits give their K best (K rounds of a
// simdgroup argmax, value desc, id asc), simdgroup 0 merges the 8 sorted lists. sample_stage2_rows_sg: one simdgroup
// merges the ntgs <= 64 sorted lists of stage 1 (K <= 32 each; lane l: lists l and l + 32) into the row's K best, then
// samples as sample_stage2_rows (the candidates loaded by all its threads first). grids as theirs.
NI_INLINE void sg_argmax(thread float& v, thread uint& i) {
  for (uint off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_xor(v, off);
    const uint oi = simd_shuffle_xor(i, off);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
}
[[kernel]] void topk_stage1_rows_sg(const device float* logits [[buffer(0)]], device float* cv [[buffer(1)]],
                                    device uint* ci [[buffer(2)]], constant TopkStageParams& p [[buffer(3)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                                    uint2 ntg2 [[threads_per_threadgroup]]) {
  threadgroup float lv[8][64];
  threadgroup uint li[8][64];
  const uint tid = tid2.x, sg = tid / 32, lane = tid % 32, nsg = ntg2.x / 32, r = tg.y;
  const uint ntgs = (p.V + p.chunk - 1) / p.chunk, K = min(p.K, 64u);
  const device float* lg = logits + size_t(r) * p.V;
  float v[8];
  const uint base = tg.x * p.chunk + tid * 8;
  for (int j = 0; j < 8; ++j) v[j] = base + j < p.V ? lg[base + j] : -INFINITY;
  float bv = -INFINITY;
  uint bi = 0xffffffff;
  for (int j = 0; j < 8; ++j)
    if (v[j] > bv) { bv = v[j]; bi = base + j; }
  for (uint q = 0; q < K; ++q) {
    float wv = bv;
    uint wi = bi;
    sg_argmax(wv, wi);
    if (lane == 0) { lv[sg][q] = wv; li[sg][q] = wi; }
    if (bi == wi && wi >= base && wi < base + 8) {  // the winner's lane: the next of its 8
      v[wi - base] = -INFINITY;
      bv = -INFINITY;
      bi = 0xffffffff;
      for (int j = 0; j < 8; ++j)
        if (v[j] > bv) { bv = v[j]; bi = base + j; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg) return;
  uint pos = 0;
  for (uint q = 0; q < K; ++q) {
    float hv = lane < nsg && pos < K ? lv[lane][pos] : -INFINITY;
    uint hi = lane < nsg && pos < K ? li[lane][pos] : 0xffffffff;
    const float mv = hv;
    const uint mi = hi;
    sg_argmax(hv, hi);
    if (lane == 0) {
      cv[(r * ntgs + tg.x) * p.K + q] = hv;
      ci[(r * ntgs + tg.x) * p.K + q] = hi;
    }
    if (lane < nsg && mi == hi && mv == hv) ++pos;
  }
}
[[kernel]] void sample_stage2_rows_sg(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                                      device int* tokens [[buffer(2)]], constant SampleRowsParams& p [[buffer(3)]],
                                      const device uint* table [[buffer(4)]],
                                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                      uint ntg [[threads_per_threadgroup]]) {
  threadgroup float topv[64];
  threadgroup uint topi[64];
  threadgroup float lv[64 * 32];  // the row's candidates (p.n = lists x p.K <= 64 x 32)
  threadgroup uint li[64 * 32];
  uint K = max(1u, min(p.K, 64u));
  if (p.temp <= 0.0f) K = 1;
  const uint L = p.n / p.K;  // stage 1's lists, p.K candidates each, sorted
  for (uint k = tid; k < p.n; k += ntg) {
    lv[k] = cv[size_t(r) * p.n + k];
    li[k] = ci[size_t(r) * p.n + k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid >= 32) return;
  const uint lane = tid;
  uint pa = 0, pb = 0;  // positions in lists lane, lane + 32
  const uint la = lane, lb = lane + 32;
  for (uint q = 0; q < K; ++q) {
    const bool ha = la < L && pa < p.K, hb = lb < L && pb < p.K;
    const float va = ha ? lv[la * p.K + pa] : -INFINITY, vb = hb ? lv[lb * p.K + pb] : -INFINITY;
    const uint ia = ha ? li[la * p.K + pa] : 0xffffffff, ib = hb ? li[lb * p.K + pb] : 0xffffffff;
    const bool a = va > vb || (va == vb && ia < ib);
    float wv = a ? va : vb;
    uint wi = a ? ia : ib;
    const float mv = wv;
    const uint mi = wi;
    sg_argmax(wv, wi);
    if (lane == 0) { topv[q] = wv; topi[q] = wi; }
    if (mi == wi && mv == wv && (a ? ha : hb)) {
      if (a) ++pa;
      else ++pb;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sample_rows_pick(topv, topi, K, p, r, r, table, tokens);
}

// ---- qmvr v2: x rows staged in threadgroup memory (half), 4 weight rows per simdgroup ----
// Per weight value: one AND + convert, then NT FMAs (x pre-scaled for the shift-free dequant).
template <int BITS, int GS, int NT, int ROWS>
NI_INLINE void qmvr2_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                          threadgroup const half* xs, device float* y, constant QmvrParams& p, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int VPL = 16;
  constexpr int WORDS = VPL / PER_WORD;
  const uint K = p.K;
  const uint wpr = K / PER_WORD, gpr = K / GS;
  float acc[ROWS][NT];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, p.N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL];
    float xsum[NT];
    for (int t = 0; t < NT; ++t) {
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = float4(((threadgroup const half4*)(xs + t * K + k0))[i]);
        xsum[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
      }
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
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
          float wf = float(wv[i] & (MASK << (BITS * j)));
          for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
        }
      for (int t = 0; t < NT; ++t) acc[r][t] += s * d[t] + b * xsum[t];
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N && uint(t) < p.R) {
        device float* dst = y + t * p.y_ld + row;
        *dst = p.add ? *dst + v : v;
      }
    }
}

#define NI_QMVR2(NAME, BITS, GS, NT, ROWS)                                                                \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],      \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[NT * 4096];                                                                      \
    for (uint i = tid; i < NT * p.K / 4; i += ntg) {                                                     \
      uint t = i / (p.K / 4), c = i % (p.K / 4);                                                         \
      uint tt = min(t, p.R - 1);                                                                         \
      ((threadgroup half4*)(xs + t * p.K))[c] = half4(((const device float4*)(x + tt * p.x_ld))[c]);     \
    }                                                                                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                     \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                 \
    if (row0 >= p.N) return;                                                                             \
    qmvr2_impl<BITS, GS, NT, ROWS>(w, s, b, xs, y, p, row0, lane);                                       \
  }

NI_QMVR2(qmvr2_b4_g64_t2, 4, 64, 2, 4)
NI_QMVR2(qmvr2_b4_g64_t3, 4, 64, 3, 4)
NI_QMVR2(qmvr2_b4_g64_t4, 4, 64, 4, 4)
NI_QMVR2(qmvr2_b8_g64_t2, 8, 64, 2, 4)
NI_QMVR2(qmvr2_b8_g64_t3, 8, 64, 3, 4)
NI_QMVR2(qmvr2_b8_g64_t4, 8, 64, 4, 4)

// ---- qmvm: small-batch quantized matmul on simdgroup matrices (R <= 8 token rows) ----
// Y^T[N, R] = W[N, K] * X^T[K, R]. A simdgroup owns 8 weight rows; per 8x8 block the A fragment
// holds integer codes (exact in half), B holds 8 k x 8 token rows of x. Each 64-wide group is
// accumulated separately and scaled at the end: y += s * (sum q x) + b * (sum x).
// Fragment layout: lane owns row fm = (q & 4) + (lane/2)%4 and cols fn, fn+1 = (q & 2)*2 + (lane%2)*2.
template <int BITS>
NI_INLINE void qmvm_impl(const device uint8_t* w, const device bfloat* scales, const device bfloat* biases,
                         threadgroup const half* xs, threadgroup const float* xg, device float* y,
                         constant QmvrParams& p, uint n0, uint lane) {
  constexpr uint GS = 64;
  const uint qd = lane / 4;
  const uint fm = (qd & 4) + (lane / 2) % 4;
  const uint fn = (qd & 2) * 2 + (lane % 2) * 2;
  const uint K = p.K, R = p.R;
  const uint n = min(n0 + fm, p.N - 1);
  const uint row_bytes = K * BITS / 8, gpr = K / GS;
  const device uint8_t* wr = w + n * row_bytes;
  float tot0 = 0.0f, tot1 = 0.0f;
  for (uint g = 0; g < gpr; ++g) {
    simdgroup_float8x8 acc = simdgroup_float8x8(0);
    for (uint kb = 0; kb < GS / 8; ++kb) {
      const uint k = g * GS + kb * 8 + fn;  // this lane's two k values: k, k+1
      simdgroup_half8x8 a, b;
      if (BITS == 4) {
        uint8_t byte = wr[k / 2];
        a.thread_elements()[0] = half(byte & 15);
        a.thread_elements()[1] = half(byte >> 4);
      } else {
        a.thread_elements()[0] = half(wr[k]);
        a.thread_elements()[1] = half(wr[k + 1]);
      }
      // B[kk][r]: lane owns kk = fm, r = fn, fn+1
      const uint kk = g * GS + kb * 8 + fm;
      b.thread_elements()[0] = fn < R ? xs[fn * K + kk] : half(0);
      b.thread_elements()[1] = fn + 1 < R ? xs[(fn + 1) * K + kk] : half(0);
      simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    const float s = float(scales[n * gpr + g]), bi = float(biases[n * gpr + g]);
    tot0 += s * acc.thread_elements()[0] + bi * (fn < R ? xg[fn * gpr + g] : 0.0f);
    tot1 += s * acc.thread_elements()[1] + bi * (fn + 1 < R ? xg[(fn + 1) * gpr + g] : 0.0f);
  }
  if (n0 + fm < p.N) {
    if (fn < R) {
      device float* d = y + fn * p.y_ld + n0 + fm;
      *d = p.add ? *d + tot0 : tot0;
    }
    if (fn + 1 < R) {
      device float* d = y + (fn + 1) * p.y_ld + n0 + fm;
      *d = p.add ? *d + tot1 : tot1;
    }
  }
}

#define NI_QMVM(NAME, BITS)                                                                               \
  [[kernel]] void NAME(const device uint8_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],       \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[12288];                                                                          \
    threadgroup float xg[4 * 64];                                                                        \
    const uint gpr = p.K / 64;                                                                           \
    for (uint i = tid; i < p.R * p.K / 4; i += ntg) {                                                    \
      uint t = i / (p.K / 4), c = i % (p.K / 4);                                                         \
      ((threadgroup half4*)(xs + t * p.K))[c] = half4(((const device float4*)(x + t * p.x_ld))[c]);      \
    }                                                                                                    \
    for (uint i = tid; i < p.R * gpr; i += ntg) {                                                        \
      uint t = i / gpr, gg = i % gpr;                                                                     \
      float sum = 0.0f;                                                                                  \
      for (uint k = 0; k < 64; ++k) sum += x[t * p.x_ld + gg * 64 + k];                                  \
      xg[t * gpr + gg] = sum;                                                                            \
    }                                                                                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                     \
    uint n0 = (tgp * nsg + sg) * 8;                                                                      \
    if (n0 >= p.N) return;                                                                               \
    qmvm_impl<BITS>(w, s, b, xs, xg, y, p, n0, lane);                                                    \
  }

NI_QMVM(qmvm_b4_g64, 4)
NI_QMVM(qmvm_b8_g64, 8)

// qmvm v2: each lane loads its weight row 16 bytes at a time and dequantizes in registers
// (a = s*q + b as half), so one accumulator covers the whole K and no group sums are needed.
template <int BITS>
NI_INLINE void qmvm2_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                          threadgroup const half* xs, device float* y, constant QmvrParams& p, uint n0, uint lane) {
  constexpr uint GS = 64;
  constexpr uint PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  const uint qd = lane / 4;
  const uint fm = (qd & 4) + (lane / 2) % 4;
  const uint fn = (qd & 2) * 2 + (lane % 2) * 2;
  const uint K = p.K, R = p.R;
  const uint n = min(n0 + fm, p.N - 1);
  const uint wpr = K / PER_WORD, gpr = K / GS;
  const device uint32_t* wr = w + n * wpr;
  const bool r0ok = fn < R, r1ok = fn + 1 < R;
  simdgroup_float8x8 acc = simdgroup_float8x8(0);
  for (uint g = 0; g < gpr; ++g) {
    const float s = float(scales[n * gpr + g]), bi = float(biases[n * gpr + g]);
    // 64 values of this row: 8 words (4-bit) or 16 words (8-bit)
    for (uint half_ = 0; half_ < GS / 32; ++half_) {
      const uint kbase = g * GS + half_ * 32;
      uint4 wv[BITS / 4];
      for (uint i = 0; i < BITS / 4; ++i) wv[i] = *(const device uint4*)(wr + kbase / PER_WORD + i * 4);
      for (uint kb = 0; kb < 4; ++kb) {  // 4 blocks of 8 k
        const uint k8 = kb * 8 + fn;     // index within the 32-value chunk of this lane's pair
        uint q0, q1;
        if (BITS == 4) {
          uint word = wv[0][kb];
          q0 = (word >> (4 * fn)) & MASK;
          q1 = (word >> (4 * (fn + 1))) & MASK;
        } else {
          uint word = wv[k8 / 16][(k8 % 16) / 4];
          q0 = (word >> (8 * (k8 % 4))) & MASK;
          q1 = (word >> (8 * ((k8 + 1) % 4))) & MASK;
        }
        simdgroup_half8x8 a, b;
        a.thread_elements()[0] = half(s * float(q0) + bi);
        a.thread_elements()[1] = half(s * float(q1) + bi);
        const uint kk = kbase + kb * 8 + fm;
        b.thread_elements()[0] = r0ok ? xs[fn * K + kk] : half(0);
        b.thread_elements()[1] = r1ok ? xs[(fn + 1) * K + kk] : half(0);
        simdgroup_multiply_accumulate(acc, a, b, acc);
      }
    }
  }
  if (n0 + fm < p.N) {
    if (r0ok) {
      device float* d = y + fn * p.y_ld + n0 + fm;
      *d = p.add ? *d + acc.thread_elements()[0] : acc.thread_elements()[0];
    }
    if (r1ok) {
      device float* d = y + (fn + 1) * p.y_ld + n0 + fm;
      *d = p.add ? *d + acc.thread_elements()[1] : acc.thread_elements()[1];
    }
  }
}

#define NI_QMVM2(NAME, BITS)                                                                              \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],      \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[12288];                                                                          \
    for (uint i = tid; i < p.R * p.K / 4; i += ntg) {                                                    \
      uint t = i / (p.K / 4), c = i % (p.K / 4);                                                         \
      ((threadgroup half4*)(xs + t * p.K))[c] = half4(((const device float4*)(x + t * p.x_ld))[c]);      \
    }                                                                                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                     \
    uint n0 = (tgp * nsg + sg) * 8;                                                                      \
    if (n0 >= p.N) return;                                                                               \
    qmvm2_impl<BITS>(w, s, b, xs, y, p, n0, lane);                                                       \
  }

NI_QMVM2(qmvm2_b4_g64, 4)
NI_QMVM2(qmvm2_b8_g64, 8)

template <int BITS, int GS, int NT, int ROWS, int VPL>
NI_INLINE void qmvr3_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                          threadgroup const half* xs, device float* y, constant QmvrParams& p, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int WORDS = VPL / PER_WORD;
  const uint K = p.K;
  const uint wpr = K / PER_WORD, gpr = K / GS;
  float acc[ROWS][NT];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, p.N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL];
    float xsum[NT];
    for (int t = 0; t < NT; ++t) {
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = float4(((threadgroup const half4*)(xs + t * K + k0))[i]);
        xsum[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
      }
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * wpr + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else if (WORDS == 2) {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      } else {
        wv[0] = wp[0];
      }
      const float s = float(scales[rows[r] * gpr + g]), b = float(biases[rows[r] * gpr + g]);
      float d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) {
          float wf = float(wv[i] & (MASK << (BITS * j)));
          for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
        }
      for (int t = 0; t < NT; ++t) acc[r][t] += s * d[t] + b * xsum[t];
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N && uint(t) < p.R) {
        device float* dst = y + t * p.y_ld + row;
        *dst = p.add ? *dst + v : v;
      }
    }
}

#define NI_QMVR3(NAME, BITS, GS, NT, ROWS, VPL)                                                                \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],      \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[NT * 4096];                                                                      \
    for (uint i = tid; i < NT * p.K / 4; i += ntg) {                                                     \
      uint t = i / (p.K / 4), c = i % (p.K / 4);                                                         \
      uint tt = min(t, p.R - 1);                                                                         \
      ((threadgroup half4*)(xs + t * p.K))[c] = half4(((const device float4*)(x + tt * p.x_ld))[c]);     \
    }                                                                                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                     \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                 \
    if (row0 >= p.N) return;                                                                             \
    qmvr3_impl<BITS, GS, NT, ROWS, VPL>(w, s, b, xs, y, p, row0, lane);                                       \
  }


NI_QMVR3(qmvr3_b4_t3_r2_v16, 4, 64, 3, 2, 16)
NI_QMVR3(qmvr3_b4_t3_r4_v8, 4, 64, 3, 4, 8)
NI_QMVR3(qmvr3_b4_t3_r2_v8, 4, 64, 3, 2, 8)
NI_QMVR3(qmvr3_b4_t4_r2_v8, 4, 64, 4, 2, 8)
NI_QMVR3(qmvr3_b4_t4_r4_v8, 4, 64, 4, 4, 8)
NI_QMVR3(qmvr3_b4_t2_r4_v16, 4, 64, 2, 4, 16)
NI_QMVR3(qmvr3_b4_t2_r8_v8, 4, 64, 2, 8, 8)

// ---- fused small-batch projections: (norm) + up to 4 matrices sharing the input rows ----
// x rows are staged once per threadgroup (fp32, optionally RMS-normalized per row); NT <= 3.
struct QmvrMultiParams {
  uint K;
  uint R;
  uint nseg;
  uint norm;
  uint write_xn;
  float eps;
  uint N[4];
  uint tg_begin[5];
  uint bits[4];
  uint add[4];
};

template <int BITS, int NT, int ROWS>
NI_INLINE void qmvr_tg_rows(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                            threadgroup const float* xs, uint K, uint N, uint R, uint row0, uint lane,
                            thread float (*acc)[NT]) {
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
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = ((threadgroup const float4*)(xs + t * K + k0))[i];
        xsum[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
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
          float wf = float(wv[i] & (MASK << (BITS * j)));
          for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
        }
      for (int t = 0; t < NT; ++t) acc[r][t] += s * d[t] + b * xsum[t];
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = simd_sum(acc[r][t]);
}

template <int NT>
NI_INLINE void qmvr_multi_impl(const device uint32_t* const* W, const device bfloat* const* S,
                               const device bfloat* const* B, device float* const* Y, const device float* x,
                               constant QmvrMultiParams& p, const device bfloat* nw, uint tgp, uint tid, uint ntg,
                               uint sg, uint nsg, uint lane, threadgroup float* xs, threadgroup float* red,
                               device float* xn_out) {
  constexpr int ROWS = 4;
  const uint K = p.K;
  for (uint t = 0; t < uint(NT); ++t) {
    const uint tt = min(t, p.R - 1);
    float ss = 0.0f;
    for (uint i = tid; i < K / 4; i += ntg) {
      float4 v = ((const device float4*)(x + tt * K))[i];
      ((threadgroup float4*)(xs + t * K))[i] = v;
      ss += dot(v, v);
    }
    if (p.norm) {
      ss = tg_sum(ss, red, tid, ntg);
      float inv = metal::precise::rsqrt(ss / float(K) + p.eps);
      for (uint i = tid; i < K; i += ntg) xs[t * K + i] = xs[t * K + i] * inv * float(nw[i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (p.write_xn && tgp == 0)
    for (uint i = tid; i < p.R * K; i += ntg) xn_out[i] = xs[i];
  uint seg = 0;
  for (uint i = 1; i < p.nseg; ++i)
    if (tgp >= p.tg_begin[i]) seg = i;
  const uint N = p.N[seg];
  const uint row0 = ((tgp - p.tg_begin[seg]) * nsg + sg) * ROWS;
  if (row0 >= N) return;
  float acc[ROWS][NT];
  if (p.bits[seg] == 8) qmvr_tg_rows<8, NT, ROWS>(W[seg], S[seg], B[seg], xs, K, N, p.R, row0, lane, acc);
  else qmvr_tg_rows<4, NT, ROWS>(W[seg], S[seg], B[seg], xs, K, N, p.R, row0, lane, acc);
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      for (int t = 0; t < NT; ++t)
        if (row0 + r < N && uint(t) < p.R) {
          device float* d = Y[seg] + t * N + row0 + r;
          *d = p.add[seg] ? *d + acc[r][t] : acc[r][t];
        }
}

#define NI_QMVR_MULTI(NAME, NT)                                                                          \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvrMultiParams& p [[buffer(17)]],                  \
      const device bfloat* nw [[buffer(18)]], device float* xn_out [[buffer(19)]],                       \
      uint tgp [[threadgroup_position_in_grid]],                                                         \
      uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],                    \
      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],               \
      uint lane [[thread_index_in_simdgroup]]) {                                                         \
    threadgroup float xs[NT * 2048];                                                                      \
    threadgroup float red[32];                                                                            \
    const device uint32_t* W[4] = {w0, w1, w2, w3};                                                       \
    const device bfloat* S[4] = {s0, s1, s2, s3};                                                         \
    const device bfloat* B[4] = {b0, b1, b2, b3};                                                         \
    device float* Y[4] = {y0, y1, y2, y3};                                                                \
    qmvr_multi_impl<NT>(W, S, B, Y, x, p, nw, tgp, tid, ntg, sg, nsg, lane, xs, red, xn_out);                     \
  }

NI_QMVR_MULTI(qmvr_multi_t2, 2)
NI_QMVR_MULTI(qmvr_multi_t3, 3)

// ---- MoE over R rows with shared expert reads (dedupe) ----
// moe_unique_rows: one thread. ue[U] = distinct experts across the rows, uw[u * R + r] = row r's
// routing weight for that expert (0 if unused); un[0] = U.
struct MoeUniqParams {
  uint R;
  uint k;
};
[[kernel]] void moe_unique_rows(const device uint* idx [[buffer(0)]], const device float* wts [[buffer(1)]],
                                device uint* ue [[buffer(2)]], device float* uw [[buffer(3)]],
                                device uint* un [[buffer(4)]], constant MoeUniqParams& p [[buffer(5)]],
                                uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  uint U = 0;
  for (uint r = 0; r < p.R; ++r)
    for (uint s = 0; s < p.k; ++s) {
      uint e = idx[r * p.k + s];
      uint u = 0;
      while (u < U && ue[u] != e) ++u;
      if (u == U) {
        ue[U] = e;
        for (uint q = 0; q < p.R; ++q) uw[U * p.R + q] = 0.0f;
        ++U;
      }
      uw[u * p.R + r] = wts[r * (p.k + 1) + s];
    }
  un[0] = U;
}

// gate/up: grid (row tiles over max(I, SI), R*k + 1, 1); y index u (last = shared expert, 4-bit).
// h[u][r][Imax] = silu(gate . x_r) * (up . x_r)
template <int NT>
NI_INLINE void moe_gu_dedupe_impl(const device uint32_t* gw, const device bfloat* gsc, const device bfloat* gbi,
                                  const device uint32_t* uw_, const device bfloat* usc, const device bfloat* ubi,
                                  const device uint32_t* sgw, const device bfloat* sgs, const device bfloat* sgb,
                                  const device uint32_t* suw, const device bfloat* sus, const device bfloat* sub,
                                  const device float* xn, device float* h, constant MoeParams& p,
                                  const device uint* ue, const device uint* un, uint R, uint2 tgp, uint tid,
                                  uint ntg, uint sg, uint nsg, uint lane, threadgroup float* xs) {
  constexpr int ROWS = 4;
  const uint u = tgp.y;
  const uint U = un[0];
  const bool shared = u == R * p.k;
  if (!shared && u >= U) return;
  for (uint i = tid; i < uint(NT) * p.D / 4; i += ntg) {
    uint t = i / (p.D / 4), c = i % (p.D / 4);
    ((threadgroup float4*)(xs + t * p.D))[c] = ((const device float4*)(xn + min(t, R - 1) * p.D))[c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / 64;
  const device uint32_t *G = gw, *Up = uw_;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; Up = suw; Us = sus; Ub = sub;
  } else {
    uint e = ue[u];
    G += e * N * wpr; Up += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  float ag[ROWS][NT], au[ROWS][NT];
  qmvr_tg_rows<4, NT, ROWS>(G, Gs, Gb, xs, K, N, R, row0, lane, ag);
  qmvr_tg_rows<4, NT, ROWS>(Up, Us, Ub, xs, K, N, R, row0, lane, au);
  const uint Imax = max(p.I, p.SI);
  if (lane == 0)
    for (int q = 0; q < ROWS; ++q)
      for (int t = 0; t < NT; ++t)
        if (row0 + q < N && uint(t) < R) h[(u * R + t) * Imax + row0 + q] = silu(ag[q][t]) * au[q][t];
}

// down: grid (row tiles over D). x[r, row] += sum_u uw[u, r] * (down_u[row] . h[u][r]) + shared.
template <int NT>
NI_INLINE void moe_down_dedupe_impl(const device uint32_t* dw, const device bfloat* dsc, const device bfloat* dbi,
                                    const device uint32_t* sdw, const device bfloat* sds, const device bfloat* sdb,
                                    const device float* h, const device uint* ue, const device float* uw,
                                    const device uint* un, const device float* wts, device float* x,
                                    constant MoeParams& p, uint R, uint tgp, uint tid, uint ntg, uint sg, uint nsg,
                                    uint lane, threadgroup float* hs) {
  constexpr int ROWS = 4;
  const uint U = un[0];
  const uint Imax = max(p.I, p.SI);
  const uint row0 = (tgp * nsg + sg) * ROWS;
  float tot[ROWS][NT];
  for (int q = 0; q < ROWS; ++q)
    for (int t = 0; t < NT; ++t) tot[q][t] = 0.0f;
  for (uint u = 0; u <= U; ++u) {
    const bool shared = u == U;
    const uint hu = shared ? R * p.k : u;
    const uint K = shared ? p.SI : p.I;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(NT) * K / 4; i += ntg) {
      uint t = i / (K / 4), c = i % (K / 4);
      ((threadgroup float4*)(hs + t * K))[c] = ((const device float4*)(h + (hu * R + min(t, R - 1)) * Imax))[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= p.D) continue;
    float a[ROWS][NT];
    if (shared) {
      qmvr_tg_rows<4, NT, ROWS>(sdw, sds, sdb, hs, K, p.D, R, row0, lane, a);
      for (int q = 0; q < ROWS; ++q)
        for (int t = 0; t < NT; ++t) tot[q][t] += (uint(t) < R ? wts[t * (p.k + 1) + p.k] : 0.0f) * a[q][t];
    } else {
      const uint wpr = K / 8, gpr = K / 64, e = ue[u];
      qmvr_tg_rows<4, NT, ROWS>(dw + e * p.D * wpr, dsc + e * p.D * gpr, dbi + e * p.D * gpr, hs, K, p.D, R, row0,
                                lane, a);
      for (int q = 0; q < ROWS; ++q)
        for (int t = 0; t < NT; ++t) tot[q][t] += (uint(t) < R ? uw[u * R + t] : 0.0f) * a[q][t];
    }
  }
  if (row0 < p.D && lane == 0)
    for (int q = 0; q < ROWS; ++q)
      for (int t = 0; t < NT; ++t)
        if (row0 + q < p.D && uint(t) < R) x[t * p.D + row0 + q] += tot[q][t];
}

#define NI_MOE_DEDUPE(SUFFIX, NT)                                                                          \
  [[kernel]] void moe_gate_up_dedupe_##SUFFIX(                                                              \
      const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],                      \
      const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw_ [[buffer(3)]],                     \
      const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],                       \
      const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],                     \
      const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],                     \
      const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],                     \
      const device float* xn [[buffer(12)]], device float* h [[buffer(13)]], constant MoeParams& p [[buffer(14)]], \
      const device uint* ue [[buffer(15)]], const device uint* un [[buffer(16)]], constant uint& R [[buffer(17)]], \
      uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                 \
      uint2 ntg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                   \
      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                    \
    threadgroup float xs[NT * 2048];                                                                         \
    moe_gu_dedupe_impl<NT>(gw, gsc, gbi, uw_, usc, ubi, sgw, sgs, sgb, suw, sus, sub, xn, h, p, ue, un, R, tgp, \
                           tid, ntg2.x, sg, nsg, lane, xs);                                                   \
  }                                                                                                          \
  [[kernel]] void moe_down_dedupe_##SUFFIX(                                                                 \
      const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],                      \
      const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],                     \
      const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],                       \
      const device float* h [[buffer(6)]], const device uint* ue [[buffer(7)]], const device float* uw [[buffer(8)]], \
      const device uint* un [[buffer(9)]], const device float* wts [[buffer(10)]], device float* x [[buffer(11)]], \
      constant MoeParams& p [[buffer(12)]], constant uint& R [[buffer(13)]],                                \
      uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                  \
      uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                     \
      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                    \
    threadgroup float hs[NT * 512];                                                                          \
    moe_down_dedupe_impl<NT>(dw, dsc, dbi, sdw, sds, sdb, h, ue, uw, un, wts, x, p, R, tgp, tid, ntg, sg, nsg, \
                             lane, hs);                                                                      \
  }

NI_MOE_DEDUPE(t2, 2)
NI_MOE_DEDUPE(t3, 3)

// ---- qmvr4: small-batch matvec for any K (x rows staged per 4096-wide tile as half) ----
template <int BITS, int NT, int ROWS, int VPL = 16>
NI_INLINE void qmvr4_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                          const device float* x, device float* y, constant QmvrParams& p, threadgroup half* xs,
                          uint row0, uint tid, uint ntg, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int WORDS = VPL / PER_WORD;
  constexpr uint TILE = 4096;
  const uint K = p.K;
  const uint wpr = K / PER_WORD, gpr = K / 64;
  float acc[ROWS][NT];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, p.N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  for (uint kt = 0; kt < K; kt += TILE) {
    const uint tile = min(TILE, K - kt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(NT) * tile / 4; i += ntg) {
      uint t = i / (tile / 4), c = i % (tile / 4);
      uint tt = min(t, p.R - 1);
      ((threadgroup half4*)(xs + t * TILE))[c] = half4(((const device float4*)(x + tt * p.x_ld + kt))[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= p.N) continue;
    for (uint k0 = lane * VPL; k0 < tile; k0 += 32 * VPL) {
      float xv[NT][VPL];
      float xsum[NT];
      for (int t = 0; t < NT; ++t) {
        xsum[t] = 0.0f;
        for (int i = 0; i < VPL / 4; ++i) {
          float4 v = float4(((threadgroup const half4*)(xs + t * TILE + k0))[i]);
          xsum[t] += v.x + v.y + v.z + v.w;
          for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
        }
      }
      const uint kg = kt + k0;
      const uint g = kg / 64, wo = kg / PER_WORD;
      for (int r = 0; r < ROWS; ++r) {
        const device uint32_t* wp = w + rows[r] * wpr + wo;
        uint32_t wv[WORDS];
        if (WORDS == 4) {
          uint4 v = *(const device uint4*)wp;
          wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
        } else if (WORDS == 2) {
          uint2 v = *(const device uint2*)wp;
          wv[0] = v.x; wv[1] = v.y;
        } else {
          wv[0] = wp[0];
        }
        const float s = float(scales[rows[r] * gpr + g]), b = float(biases[rows[r] * gpr + g]);
        float d[NT];
        for (int t = 0; t < NT; ++t) d[t] = 0.0f;
        for (int i = 0; i < WORDS; ++i)
          for (int j = 0; j < PER_WORD; ++j) {
            float wf = float(wv[i] & (MASK << (BITS * j)));
            for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
          }
        for (int t = 0; t < NT; ++t) acc[r][t] += s * d[t] + b * xsum[t];
      }
    }
  }
  if (row0 >= p.N) return;
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N && uint(t) < p.R) {
        device float* dst = y + t * p.y_ld + row;
        *dst = p.add ? *dst + v : v;
      }
    }
}

#define NI_QMVR4(NAME, BITS, NT, ROWS, ...)                                                                    \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],      \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[NT * 4096];                                                                      \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                 \
    qmvr4_impl<BITS, NT, ROWS __VA_OPT__(,) __VA_ARGS__>(w, s, b, x, y, p, xs, row0, tid, ntg, lane);                              \
  }

NI_QMVR4(qmvr4_b4_t1, 4, 1, 4)
NI_QMVR4(qmvr4_b4_t2, 4, 2, 4)
NI_QMVR4(qmvr4_b4_t3, 4, 3, 4)
NI_QMVR4(qmvr4_b8_t1, 8, 1, 4)
NI_QMVR4(qmvr4_b8_t2, 8, 2, 4)
NI_QMVR4(qmvr4_b8_t3, 8, 3, 4)
NI_QMVR4(qmvr4_b4_t3_v8, 4, 3, 4, 8)
NI_QMVR4(qmvr4_b4_t4_v8, 4, 4, 4, 8)
NI_QMVR4(qmvr4_b8_t4, 8, 4, 4)

// MTP input when the head expects the trunk's final-normed hidden ("post_norm" contract):
// u[r] = [rmsnorm(e[r]) * we, rmsnorm(rmsnorm(h[r]) * wf) * wh]. grid: R x 256.
[[kernel]] void mtp_concat_norm_post(const device float* e [[buffer(0)]], const device float* h [[buffer(1)]],
                                     const device bfloat* we [[buffer(2)]], const device bfloat* wh [[buffer(3)]],
                                     device float* u [[buffer(4)]], constant RowsNormParams& p [[buffer(5)]],
                                     const device bfloat* wf [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const device float* er = e + r * p.D;
  const device float* hr = h + r * p.D;
  float se = 0.0f, sh = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) {
    se += er[i] * er[i];
    sh += hr[i] * hr[i];
  }
  se = tg_sum(se, red, tid, ntg);
  sh = tg_sum(sh, red, tid, ntg);
  const float ie = metal::precise::rsqrt(se / float(p.D) + p.eps), ih = metal::precise::rsqrt(sh / float(p.D) + p.eps);
  float s2 = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) {
    float v = hr[i] * ih * float(wf[i]);
    s2 += v * v;
  }
  s2 = tg_sum(s2, red, tid, ntg);
  const float i2 = metal::precise::rsqrt(s2 / float(p.D) + p.eps);
  for (uint i = tid; i < p.D; i += ntg) {
    u[r * 2 * p.D + i] = er[i] * ie * float(we[i]);
    u[r * 2 * p.D + p.D + i] = hr[i] * ih * float(wf[i]) * i2 * float(wh[i]);
  }
}

// h[i] = silu(g[i]) * u[i] (fp32). grid: n threads.
[[kernel]] void swiglu_f32(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                           device float* h [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i < n) h[i] = silu(g[i]) * u[i];
}

// ---- Row-major small batches ("rowcache"): one single-row matvec threadgroup per (row, block),
// with the row index in the FASTEST grid dimension. The R threadgroups that read the same weight
// block run back to back, so weights stream from DRAM once and the other rows hit the caches,
// while each threadgroup keeps the register footprint of a single-row kernel.
// grid: (R, row blocks). x/y/resid rows are x_ld / y_ld apart.
#define NI_QMV2_ROWCACHE(NAME, BITS, ROWS, ADD, NORM)                                                    \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],     \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],                \
                       const device float* resid [[buffer(6)]], const device bfloat* nw [[buffer(7)]],     \
                       uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint2 ntg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    threadgroup float xs[4096];                                                                           \
    threadgroup float red[32];                                                                            \
    const uint r = tgp.x;                                                                                 \
    if (r >= p.R) return;                                                                                 \
    uint row0 = (tgp.y * nsg + sg) * ROWS;                                                                \
    device float* yr = y + r * p.y_ld;                                                                    \
    if (p.K <= 4096)                                                                                      \
      qmv2_impl<BITS, 16, ROWS, 64, ADD, NORM>(w, scales, biases, x + r * p.x_ld, yr, p.N, p.K, 1e-6f,   \
                                               ADD ? resid + r * p.y_ld : yr, nw, xs, red, row0, tid,     \
                                               ntg2.x, lane);                                             \
    else                                                                                                  \
      qmv3_impl<BITS, ROWS, ADD, NORM>(w, scales, biases, x + r * p.x_ld, yr, p.N, p.K, 1e-6f,           \
                                       ADD ? resid + r * p.y_ld : yr, nw, xs, red, row0, tid, ntg2.x, lane); \
  }

NI_QMV2_ROWCACHE(qmvrc_b4_r2, 4, 2, false, false)
NI_QMV2_ROWCACHE(qmvrc_b4_r4, 4, 4, false, false)
NI_QMV2_ROWCACHE(qmvrc_b4_r2_add, 4, 2, true, false)
NI_QMV2_ROWCACHE(qmvrc_b8_r2, 8, 2, false, false)
NI_QMV2_ROWCACHE(qmvrc_b8_r2_add, 8, 2, true, false)

// Multi-segment version (fused projections sharing the input rows), optional per-row norm and
// normalized-row output. grid: (R, sum of segment blocks).
struct QmvrcMultiParams {
  uint K;
  uint R;
  uint nseg;
  uint norm;
  uint write_xn;
  float eps;
  uint N[4];
  uint tg_begin[5];
  uint bits[4];
};
[[kernel]] void qmvrc_multi_r2(
    const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],
    const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],
    const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],
    const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],
    const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],
    const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],
    const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],
    const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],
    const device float* x [[buffer(16)]], constant QmvrcMultiParams& p [[buffer(17)]],
    const device bfloat* nw [[buffer(18)]], device float* xn_out [[buffer(19)]],
    uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint2 ntg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 2;
  threadgroup float xs[4096];
  threadgroup float red[32];
  const uint ntg = ntg2.x;
  const uint r = tgp.x;
  if (r >= p.R) return;
  const device float* xr = x + r * p.K;
  const bool norm = p.norm != 0;
  const float inv = norm ? rms_inv_device(xr, p.K, p.eps, red, tid, ntg) : 1.0f;
  if (p.write_xn && tgp.y == 0)
    for (uint i = tid; i < p.K; i += ntg) xn_out[r * p.K + i] = norm ? xr[i] * inv * float(nw[i]) : xr[i];
  uint seg = 0;
  for (uint i = 1; i < p.nseg; ++i)
    if (tgp.y >= p.tg_begin[i]) seg = i;
  const uint N = p.N[seg];
  const uint row0 = ((tgp.y - p.tg_begin[seg]) * nsg + sg) * ROWS;
  const device uint32_t* W = seg == 0 ? w0 : seg == 1 ? w1 : seg == 2 ? w2 : w3;
  const device bfloat* S = seg == 0 ? s0 : seg == 1 ? s1 : seg == 2 ? s2 : s3;
  const device bfloat* B = seg == 0 ? b0 : seg == 1 ? b1 : seg == 2 ? b2 : b3;
  device float* Y = (seg == 0 ? y0 : seg == 1 ? y1 : seg == 2 ? y2 : y3) + r * N;
  uint rows[ROWS];
  float acc[ROWS];
  for (int q = 0; q < ROWS; ++q) { rows[q] = min(row0 + q, N - 1); acc[q] = 0.0f; }
  for (uint kt = 0; kt < p.K; kt += 4096) {
    const uint tile = min(4096u, p.K - kt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_tile(xr, kt, tile, norm, inv, nw, xs, tid, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < N) {
      if (p.bits[seg] == 8) qmv3_tile_rows<8, ROWS>(W, S, B, xs, p.K, kt, tile, N, rows, lane, acc);
      else qmv3_tile_rows<4, ROWS>(W, S, B, xs, p.K, kt, tile, N, rows, lane, acc);
    }
  }
  if (row0 >= N) return;
  for (int q = 0; q < ROWS; ++q) {
    float v = simd_sum(acc[q]);
    if (lane == 0 && row0 + q < N) Y[row0 + q] = v;
  }
}

// ---- MoE verify with row-major caching ----
// After moe_unique_rows: ue[U] distinct experts, uw[u * R + r] row weights (0 = unused).
// gate/up grid: (R, row tiles over max(I, SI), R*k + 1): threadgroup (r, t, u) computes rows of expert u
// for token row r only if r uses it (the shared expert, u = R*k, serves every row). Rows are the
// fastest grid dimension, so the R threadgroups of one expert tile run back to back and share the
// weight reads through the caches. h[u][r][Imax].
[[kernel]] void moe_gate_up_rc(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],
                               const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw_ [[buffer(3)]],
                               const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],
                               const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],
                               const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],
                               const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],
                               const device float* xn [[buffer(12)]], device float* h [[buffer(13)]],
                               constant MoeParams& p [[buffer(14)]], const device uint* ue [[buffer(15)]],
                               const device float* uwts [[buffer(16)]], const device uint* un [[buffer(17)]],
                               constant uint& R [[buffer(18)]], uint3 tgp [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 2;
  threadgroup float xs[4096];
  const uint r = tgp.x, u = tgp.z;
  if (r >= R) return;
  const bool shared = u == R * p.k;
  if (!shared && (u >= un[0] || uwts[u * R + r] == 0.0f)) return;
  for (uint i = tid; i < p.D / 4; i += ntg3.x) ((threadgroup float4*)xs)[i] = ((const device float4*)(xn + r * p.D))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.y * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / 64;
  const device uint32_t *G = gw, *U = uw_;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    uint e = ue[u];
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  float ag[ROWS], au[ROWS];
  qmv2_rows<4, 16, ROWS, 64>(G, Gs, Gb, xs, N, K, row0, lane, ag);
  qmv2_rows<4, 16, ROWS, 64>(U, Us, Ub, xs, N, K, row0, lane, au);
  const uint Imax = max(p.I, p.SI);
  if (lane == 0)
    for (int q = 0; q < ROWS; ++q)
      if (row0 + q < N) h[(u * R + r) * Imax + row0 + q] = silu(ag[q]) * au[q];
}

// down grid: (R, row tiles over D). x[r] += sum over experts u used by r of uw[u,r] * down_u(h[u][r])
// + shared. Rows fastest again, so both rows' threadgroups of a tile reuse each shared expert's rows.
[[kernel]] void moe_down_rc(const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],
                            const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],
                            const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],
                            const device float* h [[buffer(6)]], const device uint* ue [[buffer(7)]],
                            const device float* uwts [[buffer(8)]], const device uint* un [[buffer(9)]],
                            const device float* wts [[buffer(10)]], device float* x [[buffer(11)]],
                            constant MoeParams& p [[buffer(12)]], constant uint& R [[buffer(13)]],
                            uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint2 ntg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 2;
  threadgroup float hs[512];
  const uint r = tgp.x;
  if (r >= R) return;
  const uint U = un[0];
  const uint Imax = max(p.I, p.SI);
  const uint row0 = (tgp.y * nsg + sg) * ROWS;
  float tot[ROWS] = {0.0f, 0.0f};
  for (uint u = 0; u <= U; ++u) {
    const bool shared = u == U;
    const float wgt = shared ? wts[r * (p.k + 1) + p.k] : uwts[u * R + r];
    if (wgt == 0.0f) continue;  // uniform across the threadgroup: no barrier divergence
    const uint hu = shared ? R * p.k : u;
    const uint K = shared ? p.SI : p.I;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < K / 4; i += ntg2.x) ((threadgroup float4*)hs)[i] = ((const device float4*)(h + (hu * R + r) * Imax))[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= p.D) continue;
    float a[ROWS];
    if (shared) {
      qmv2_rows<4, 16, ROWS, 64>(sdw, sds, sdb, hs, p.D, K, row0, lane, a);
    } else {
      const uint wpr = K / 8, gpr = K / 64, e = ue[u];
      qmv2_rows<4, 16, ROWS, 64>(dw + e * p.D * wpr, dsc + e * p.D * gpr, dbi + e * p.D * gpr, hs, p.D, K, row0, lane, a);
    }
    for (int q = 0; q < ROWS; ++q) tot[q] += wgt * a[q];
  }
  if (row0 < p.D && lane == 0)
    for (int q = 0; q < ROWS; ++q)
      if (row0 + q < p.D) x[r * p.D + row0 + q] += tot[q];
}

// Row-major small batch with small threadgroups (2 simdgroups x 4 rows, x read per simdgroup from
// device memory, no staging or barriers): grid (R, N / 8). Consecutive threadgroups (the R token
// rows of one 8-row weight block) tend to land on the same core and share the block in its cache.
#define NI_QMV_RC1(NAME, BITS, ADD)                                                                       \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],     \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],                \
                       uint2 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    const uint r = tgp.x;                                                                                 \
    if (r >= p.R) return;                                                                                 \
    QmvParams q;                                                                                          \
    q.N = p.N; q.K = p.K; q.x_stride = p.x_ld; q.y_stride = p.y_ld; q.eps = 0.0f;                         \
    uint row0 = (tgp.y * nsg + sg) * 4;                                                                   \
    qmv_impl_rc<BITS, 16, 4, 64, ADD>(w, scales, biases, x + r * p.x_ld, y + r * p.y_ld, q, row0, lane);  \
  }

template <int BITS, int VPL, int ROWS, int GS, bool ADD>
NI_INLINE void qmv_impl_rc(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                           const device float* x, device float* y, thread const QmvParams& p, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  const uint K = p.K, words_per_row = K / PER_WORD, groups_per_row = K / GS;
  float acc[ROWS];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, p.N - 1);
  }
  NormCtx<1> nc;
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    XChunk<BITS, VPL, 1> c;
    load_x<BITS, VPL, 1, false>(c, x, K, k0, nc, nullptr);
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r)
      dot_row<BITS, VPL, 1>(c, w + rows[r] * words_per_row + wo, float(scales[rows[r] * groups_per_row + g]),
                            float(biases[rows[r] * groups_per_row + g]), &acc[r]);
  }
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    uint row = row0 + r;
    if (lane == 0 && row < p.N) y[row] = ADD ? y[row] + v : v;
  }
}

NI_QMV_RC1(qmvrc1_b4, 4, false)
NI_QMV_RC1(qmvrc1_b4_add, 4, true)
NI_QMV_RC1(qmvrc1_b8, 8, false)
NI_QMV_RC1(qmvrc1_b8_add, 8, true)

// Two token rows per threadgroup (in registers) and row pairs in the fastest grid dimension:
// grid (ceil(R/2), N / 8), 2 simdgroups x 4 weight rows. R = 3 or 4 -> two threadgroups per weight
// block, adjacent in dispatch order, sharing the block through the core's cache.
template <int BITS, int VPL, int ROWS, bool ADD>
NI_INLINE void qmv_rc2_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                            const device float* x0, const device float* x1, device float* y0, device float* y1,
                            bool two, uint N, uint K, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int WORDS = VPL / PER_WORD;
  const uint wpr = K / PER_WORD, gpr = K / 64;
  float a0[ROWS], a1[ROWS];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    a0[r] = a1[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float v0[VPL], v1[VPL];
    float s0 = 0.0f, s1 = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 p0 = ((const device float4*)(x0 + k0))[i];
      float4 p1 = ((const device float4*)(x1 + k0))[i];
      s0 += p0.x + p0.y + p0.z + p0.w;
      s1 += p1.x + p1.y + p1.z + p1.w;
      for (int e = 0; e < 4; ++e) {
        const float sc = float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
        v0[i * 4 + e] = p0[e] * sc;
        v1[i * 4 + e] = p1[e] * sc;
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
      float d0 = 0.0f, d1 = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) {
          float wf = float(wv[i] & (MASK << (BITS * j)));
          d0 += wf * v0[i * PER_WORD + j];
          d1 += wf * v1[i * PER_WORD + j];
        }
      const float sc = float(scales[rows[r] * gpr + g]), bi = float(biases[rows[r] * gpr + g]);
      a0[r] += sc * d0 + bi * s0;
      a1[r] += sc * d1 + bi * s1;
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float u0 = simd_sum(a0[r]), u1 = simd_sum(a1[r]);
    uint row = row0 + r;
    if (lane == 0 && row < N) {
      y0[row] = ADD ? y0[row] + u0 : u0;
      if (two) y1[row] = ADD ? y1[row] + u1 : u1;
    }
  }
}

#define NI_QMV_RC2(NAME, BITS, ADD)                                                                       \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],     \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],                \
                       uint2 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    const uint r0 = tgp.x * 2;                                                                            \
    if (r0 >= p.R) return;                                                                                \
    const bool two = r0 + 1 < p.R;                                                                        \
    const uint r1 = two ? r0 + 1 : r0;                                                                    \
    uint row0 = (tgp.y * nsg + sg) * 4;                                                                   \
    qmv_rc2_impl<BITS, 16, 4, ADD>(w, scales, biases, x + r0 * p.x_ld, x + r1 * p.x_ld, y + r0 * p.y_ld,  \
                                   y + r1 * p.y_ld, two, p.N, p.K, row0, lane);                          \
  }

NI_QMV_RC2(qmvrc2_b4, 4, false)
NI_QMV_RC2(qmvrc2_b4_add, 4, true)
NI_QMV_RC2(qmvrc2_b8, 8, false)
NI_QMV_RC2(qmvrc2_b8_add, 8, true)

// Same as qmvrc1 but x rows are half (half the cache footprint of the activations, as MLX's
// bf16 vectors), so several token rows' threadgroups can keep sharing the weight block.
template <int BITS, int VPL, int ROWS, bool ADD>
NI_INLINE void qmv_rch_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                            const device half* x, device float* y, uint N, uint K, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
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
    float xs = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = float4(((const device half4*)(x + k0))[i]);
      xs += v.x + v.y + v.z + v.w;
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
      acc[r] += float(scales[rows[r] * gpr + g]) * d + float(biases[rows[r] * gpr + g]) * xs;
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    uint row = row0 + r;
    if (lane == 0 && row < N) y[row] = ADD ? y[row] + v : v;
  }
}

#define NI_QMV_RCH(NAME, BITS, ADD)                                                                       \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device half* x [[buffer(3)]],      \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],                \
                       uint2 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    const uint r = tgp.x;                                                                                 \
    if (r >= p.R) return;                                                                                 \
    uint row0 = (tgp.y * nsg + sg) * 4;                                                                   \
    qmv_rch_impl<BITS, 16, 4, ADD>(w, scales, biases, x + r * p.x_ld, y + r * p.y_ld, p.N, p.K, row0, lane); \
  }

NI_QMV_RCH(qmvrch_b4, 4, false)
NI_QMV_RCH(qmvrch_b4_add, 4, true)
NI_QMV_RCH(qmvrch_b8, 8, false)
NI_QMV_RCH(qmvrch_b8_add, 8, true)

// One threadgroup per block of ROWS weight rows; simdgroup s handles token row s (R simdgroups),
// so the R reads of each weight row happen on one core at the same time (one DRAM read, R-1 L1
// hits) while every simdgroup keeps a single-row register footprint. x rows are half.
// grid: (N / ROWS) threadgroups of 32 * R threads.
#define NI_QMV_SGR(NAME, BITS, ROWS, ADD)                                                                 \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device half* x [[buffer(3)]],      \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],                \
                       uint tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint lane [[thread_index_in_simdgroup]]) {                                          \
    if (sg >= p.R) return;                                                                                \
    qmv_rch_impl<BITS, 16, ROWS, ADD>(w, scales, biases, x + sg * p.x_ld, y + sg * p.y_ld, p.N, p.K,      \
                                      tgp * ROWS, lane);                                                  \
  }

NI_QMV_SGR(qmvsg_b4_r4, 4, 4, false)
NI_QMV_SGR(qmvsg_b4_r8, 4, 8, false)
NI_QMV_SGR(qmvsg_b4_r2, 4, 2, false)
NI_QMV_SGR(qmvsg_b4_r4_add, 4, 4, true)
NI_QMV_SGR(qmvsg_b8_r4, 8, 4, false)
NI_QMV_SGR(qmvsg_b8_r4_add, 8, 4, true)

// Small batch with half-precision inner products: each 16-bit half of a weight word holds 4
// nibbles; (w16 & mask_j) converts exactly to half (<= 61440) and x is pre-scaled by 2^-4j in half,
// so the per-value work is AND + convert + half FMA, shared dequant across NT rows, 16-value
// partial sums in half then promoted to float with the group scale.
template <int NT, int ROWS>
NI_INLINE void qmvh_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         threadgroup const half* xs, device float* y, constant QmvrParams& p, uint row0, uint lane) {
  constexpr int VPL = 16;
  const uint K = p.K;
  const uint wpr = K / 8, gpr = K / 64;
  float acc[ROWS][NT];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, p.N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    half xv[NT][VPL];
    float xsum[NT];
    for (int t = 0; t < NT; ++t) {
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        half4 v = ((threadgroup const half4*)(xs + t * K + k0))[i];
        xsum[t] += float(v.x) + float(v.y) + float(v.z) + float(v.w);
        xv[t][i * 4 + 0] = v.x;
        xv[t][i * 4 + 1] = v.y * half(1.0h / 16.0h);
        xv[t][i * 4 + 2] = v.z * half(1.0h / 256.0h);
        xv[t][i * 4 + 3] = v.w * half(1.0h / 4096.0h);
      }
    }
    const uint g = k0 / 64, wo = k0 / 8;
    for (int r = 0; r < ROWS; ++r) {
      uint2 wv = *(const device uint2*)(w + rows[r] * wpr + wo);
      ushort4 h4 = as_type<ushort4>(wv);
      half d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0h;
      for (int i = 0; i < 4; ++i) {
        ushort u = h4[i];
        half w0 = half(u & 0x000f), w1 = half(u & 0x00f0), w2 = half(u & 0x0f00), w3 = half(u & 0xf000);
        for (int t = 0; t < NT; ++t)
          d[t] += w0 * xv[t][i * 4] + w1 * xv[t][i * 4 + 1] + w2 * xv[t][i * 4 + 2] + w3 * xv[t][i * 4 + 3];
      }
      const float s = float(scales[rows[r] * gpr + g]), b = float(biases[rows[r] * gpr + g]);
      for (int t = 0; t < NT; ++t) acc[r][t] += s * float(d[t]) + b * xsum[t];
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N && uint(t) < p.R) {
        device float* dst = y + t * p.y_ld + row;
        *dst = p.add ? *dst + v : v;
      }
    }
}

#define NI_QMVH(NAME, NT, ROWS)                                                                           \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],      \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[NT * 4096];                                                                      \
    for (uint i = tid; i < NT * p.K / 4; i += ntg) {                                                     \
      uint t = i / (p.K / 4), c = i % (p.K / 4);                                                         \
      ((threadgroup half4*)(xs + t * p.K))[c] = half4(((const device float4*)(x + min(t, p.R - 1) * p.x_ld))[c]); \
    }                                                                                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                     \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                 \
    if (row0 >= p.N) return;                                                                             \
    qmvh_impl<NT, ROWS>(w, s, b, xs, y, p, row0, lane);                                                  \
  }

NI_QMVH(qmvh_t3_r4, 3, 4)
NI_QMVH(qmvh_t4_r4, 4, 4)
NI_QMVH(qmvh_t3_r2, 3, 2)
NI_QMVH(qmvh_t4_r2, 4, 2)
NI_QMVH(qmvh_t2_r4, 2, 4)

// SHIFT (the qmvhts_* kernels, Apple10 GPUs): each nibble shifted down to 0..15 and x unscaled. The masked form
// is right as written, but Apple10's fast-math compiler (M5) reassociates the 3- and 4-row builds (and qmvh_*'s)
// so that the 2^-4j pre-scales come after the masked-weight products are summed (2 rows keep them): those sums
// (up to 61440 |x| per term) overflow half once |x| passes ~0.4, and inf - inf gives NaN rows (finite under
// NI_MATH_SAFE=1, and with explicit fma()s or x pre-scaled in threadgroup memory). Weights of 0..15 leave no
// common factor to move, and the shifts cost nothing measurable there.
template <int NT, int ROWS, bool SHIFT = false>
NI_INLINE void qmvht_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         const device float* x, threadgroup half* xs, device float* y, constant QmvrParams& p, uint row0,
                          uint tid, uint ntg, uint lane) {
  constexpr uint TILE = NT <= 2 ? 4096 : NT == 3 ? 4096 : 2048;
  constexpr int VPL = 16;
  const uint K = p.K;
  const uint wpr = K / 8, gpr = K / 64;
  float acc[ROWS][NT];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, p.N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  for (uint kt = 0; kt < K; kt += TILE) {
  const uint tile = min(TILE, K - kt);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < uint(NT) * tile / 4; i += ntg) {
    uint t = i / (tile / 4), c = i % (tile / 4);
    ((threadgroup half4*)(xs + t * TILE))[c] = half4(((const device float4*)(x + min(t, p.R - 1) * p.x_ld + kt))[c]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (row0 >= p.N) continue;
  for (uint k0 = lane * VPL; k0 < tile; k0 += 32 * VPL) {
    half xv[NT][VPL];
    float xsum[NT];
    for (int t = 0; t < NT; ++t) {
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        half4 v = ((threadgroup const half4*)(xs + t * TILE + k0))[i];
        xsum[t] += float(v.x) + float(v.y) + float(v.z) + float(v.w);
        if (SHIFT) {
          xv[t][i * 4 + 0] = v.x;
          xv[t][i * 4 + 1] = v.y;
          xv[t][i * 4 + 2] = v.z;
          xv[t][i * 4 + 3] = v.w;
        } else {
          xv[t][i * 4 + 0] = v.x;
          xv[t][i * 4 + 1] = v.y * half(1.0h / 16.0h);
          xv[t][i * 4 + 2] = v.z * half(1.0h / 256.0h);
          xv[t][i * 4 + 3] = v.w * half(1.0h / 4096.0h);
        }
      }
    }
    const uint g = (kt + k0) / 64, wo = (kt + k0) / 8;
    for (int r = 0; r < ROWS; ++r) {
      uint2 wv = *(const device uint2*)(w + rows[r] * wpr + wo);
      ushort4 h4 = as_type<ushort4>(wv);
      half d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0h;
      for (int i = 0; i < 4; ++i) {
        ushort u = h4[i];
        half w0, w1, w2, w3;
        if (SHIFT)
          w0 = half(u & 0x000f), w1 = half((u >> 4) & 0x000f), w2 = half((u >> 8) & 0x000f), w3 = half(u >> 12);
        else
          w0 = half(u & 0x000f), w1 = half(u & 0x00f0), w2 = half(u & 0x0f00), w3 = half(u & 0xf000);
        for (int t = 0; t < NT; ++t)
          d[t] += w0 * xv[t][i * 4] + w1 * xv[t][i * 4 + 1] + w2 * xv[t][i * 4 + 2] + w3 * xv[t][i * 4 + 3];
      }
      const float s = float(scales[rows[r] * gpr + g]), b = float(biases[rows[r] * gpr + g]);
      for (int t = 0; t < NT; ++t) acc[r][t] += s * float(d[t]) + b * xsum[t];
    }
  }
  }
  if (row0 >= p.N) return;
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N && uint(t) < p.R) {
        device float* dst = y + t * p.y_ld + row;
        *dst = p.add ? *dst + v : v;
      }
    }
}


#define NI_QMVHT(NAME, NT, ROWS, SHIFT)                                                                   \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],      \
                       const device bfloat* b [[buffer(2)]], const device float* x [[buffer(3)]],         \
                       device float* y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],               \
                       uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],  \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half xs[NT * (NT <= 3 ? 4096 : 2048)];                                                   \
    uint row0 = (tgp * nsg + sg) * ROWS;                                                                 \
    qmvht_impl<NT, ROWS, SHIFT>(w, s, b, x, xs, y, p, row0, tid, ntg, lane);                             \
  }
NI_QMVHT(qmvht_t2_r4, 2, 4, false)
NI_QMVHT(qmvht_t3_r4, 3, 4, false)
NI_QMVHT(qmvht_t4_r4, 4, 4, false)
NI_QMVHT(qmvht_t4_r2, 4, 2, false)
NI_QMVHT(qmvhts_t3_r4, 3, 4, true)  // Apple10 GPUs (M5)
