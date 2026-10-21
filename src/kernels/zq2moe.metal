#include "common.h"

// ---- Routed experts in Q2_0 (Qwen3.8-Flash-Next on 48 GB Macs; internal/FLASH_48GB.md) ----
// Q2_0 is ggml's type 42: blocks of 64 weights, an fp16 d and 16 bytes of 2-bit codes, LSB first, weight = (code - 1) d.
// The codes are in MLX's 2-bit order (code j of a word at bits 2j), so a block is MLX's 2-bit affine group of 64 with
// scale d and bias -d. An expert matrix [N, K] is kept as two planes: codes [N][K/16] uint32 and d [N][K/64] half
// (2.25 bits a weight). The three matrices of an expert are either planes stacked over the experts ([E][N][...]) or one
// slot of an arena read through the layer's slot table (SLOT): gate codes, gate d, up codes, up d, down codes, down d
// at the byte offsets of Q20Slot. A slot word is chunk << 28 | page: 16 KiB page `page` of arena chunk 0 or 1 (an arena
// over maxBufferLength takes two buffers). The arithmetic of the 4-bit kernels each family mirrors is unchanged except
// for the dequantization: the decode / verify matvecs mask codes in place against x pre-scaled by 4^-j (qmv2_rows),
// the prefill MMA paths dequantize (code - 1) d exactly in one half FMA. The shared expert (4-bit group 64) runs in the
// decode / verify kernels as slot k; the prefill kernels take routed tiles only (the shared expert on the dense path).
// Buffers of the routed experts, w0..w3: stacked gate/up: gate codes, gate d, up codes, up d; stacked down: down codes,
// down d, (unused); SLOT: arena chunk 0, chunk 1, the layer's slot table (E words), (unused).

struct Q20Slot {
  uint gc, gd, uc, ud, dc, dd;  // byte offsets of the planes inside a slot
};

struct Q20Mat {
  const device uint32_t* c;  // codes, K/16 words a row
  const device half* d;      // K/64 scales a row
};

// Expert e's matrix. Stacked: base0 / base1 = its codes / d plane, cs / ds = the planes' per-expert strides (words /
// halfs). SLOT: base0 / base1 = arena chunks, coff / doff = the matrix's offsets inside a slot.
template <bool SLOT>
NI_INLINE Q20Mat q20_mat(const device uchar* base0, const device uchar* base1, const device uint* slots, uint e,
                         uint cs, uint ds, uint coff, uint doff) {
  if (SLOT) {
    const uint sw = slots[e];
    const device uchar* b = ((sw >> 28) != 0u ? base1 : base0) + ulong(sw & 0x0FFFFFFFu) * 16384ul;
    return Q20Mat{(const device uint32_t*)(b + coff), (const device half*)(b + doff)};
  }
  return Q20Mat{(const device uint32_t*)base0 + ulong(e) * cs, (const device half*)base1 + ulong(e) * ds};
}

// acc[r] = row (row0 + r) of m . xs for ROWS rows (qmv2_rows<2, VPL, ROWS, 64> with scale d and bias -d: each lane's
// VPL codes masked in place against x pre-scaled by 4^-j, then d * (sum - xsum) per lane chunk).
template <int VPL, int ROWS>
NI_INLINE void q20_rows(Q20Mat m, threadgroup const float* xs, uint N, uint K, uint row0, uint lane, thread float* acc) {
  constexpr int WORDS = VPL / 16;
  const uint wpr = K / 16, gpr = K / 64;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[VPL];
    float xsum = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      const float4 v = ((threadgroup const float4*)(xs + k0))[i];
      xsum += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) xv[i * 4 + e] = v[e] * float(1.0f / float(1u << (2 * ((i * 4 + e) % 16))));
    }
    const uint g = k0 / 64, wo = k0 / 16;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = m.c + rows[r] * wpr + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        const uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else if (WORDS == 2) {
        const uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      } else {
        wv[0] = wp[0];
      }
      float d = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < 16; ++j) d += float(wv[i] & (3u << (2 * j))) * xv[i * 16 + j];
      acc[r] += float(m.d[rows[r] * gpr + g]) * (d - xsum);
    }
  }
  for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}

// ---- Decode and verify: moe_gate_up3 / moe_down2_w / moe4_* grids and arithmetic ----
// Instances: VPL 16 values a lane, ROWS 4 rows a simdgroup (fn_q2_bench tune on the M4 Pro, interleaved, 21 rounds: a
// decode call 93.2 us against 105.9 with ROWS 2, outputs bit for bit equal; VPL 32 or 64 slower). Per row the sums are
// those of ROWS 2.

// This simdgroup's ROWS gate/up rows of expert e (or of the shared expert): out[i] = silu(gate_i . x) * (up_i . x),
// x staged in xs. N = I (SI for the shared expert), K = D.
template <bool SLOT, int VPL, int ROWS>
NI_INLINE void q20_gu_tile(const device uchar* w0, const device uchar* w1, const device uchar* w2,
                           const device uchar* w3, const device uint32_t* sgw, const device bfloat* sgs,
                           const device bfloat* sgb, const device uint32_t* suw, const device bfloat* sus,
                           const device bfloat* sub, threadgroup const float* xs, device float* out,
                           constant MoeParams& p, uint e, bool shared, constant Q20Slot& L, uint tgx, uint sg,
                           uint nsg, uint lane) {
  const uint N = shared ? p.SI : p.I, K = p.D;
  const uint row0 = (tgx * nsg + sg) * ROWS;
  if (row0 >= N) return;
  float ag[ROWS], au[ROWS];
  if (shared) {
    qmv2_rows<4, 16, ROWS, 64>(sgw, sgs, sgb, xs, N, K, row0, lane, ag);
    qmv2_rows<4, 16, ROWS, 64>(suw, sus, sub, xs, N, K, row0, lane, au);
  } else {
    const device uint* slots = (const device uint*)w2;
    const uint cs = N * (K / 16), ds = N * (K / 64);
    const Q20Mat G = q20_mat<SLOT>(w0, w1, slots, e, cs, ds, L.gc, L.gd);
    const Q20Mat U = q20_mat<SLOT>(SLOT ? w0 : w2, SLOT ? w1 : w3, slots, e, cs, ds, L.uc, L.ud);
    q20_rows<VPL, ROWS>(G, xs, N, K, row0, lane, ag);
    q20_rows<VPL, ROWS>(U, xs, N, K, row0, lane, au);
  }
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < N) out[row0 + r] = silu(ag[r]) * au[r];
}

// Decode gate/up of one row: grid (max(I, SI) / (8 ROWS), k + 1), 256 threads, ROWS rows a simdgroup; slot y < k is
// expert idx[y], slot k the shared expert; h[slot * max(I, SI) + i]. Buffers: 0-3 the routed experts (see the top), 4-9
// the shared expert's gate and up (w, scales, biases), 10 xn [D], 11 h, 12 MoeParams, 13 idx [k], 14 Q20Slot.
#define NI_Q20_GU(NAME, SLOT, VPL, ROWS)                                                                               \
  [[kernel]] void NAME(const device uchar* w0 [[buffer(0)]], const device uchar* w1 [[buffer(1)]],                 \
                       const device uchar* w2 [[buffer(2)]], const device uchar* w3 [[buffer(3)]],                 \
                       const device uint32_t* sgw [[buffer(4)]], const device bfloat* sgs [[buffer(5)]],           \
                       const device bfloat* sgb [[buffer(6)]], const device uint32_t* suw [[buffer(7)]],           \
                       const device bfloat* sus [[buffer(8)]], const device bfloat* sub [[buffer(9)]],             \
                       const device float* xn [[buffer(10)]], device float* h [[buffer(11)]],                      \
                       constant MoeParams& p [[buffer(12)]], const device uint* idx [[buffer(13)]],                \
                       constant Q20Slot& L [[buffer(14)]], uint2 tgp [[threadgroup_position_in_grid]],             \
                       uint tid [[thread_index_in_threadgroup]], uint2 ntg2 [[threads_per_threadgroup]],           \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    threadgroup float xs[4096];                                                                                    \
    const uint slot = tgp.y;                                                                                       \
    const bool shared = slot == p.k;                                                                               \
    for (uint i = tid; i < p.D / 4; i += ntg2.x) ((threadgroup float4*)xs)[i] = ((const device float4*)xn)[i];     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
    q20_gu_tile<SLOT, VPL, ROWS>(w0, w1, w2, w3, sgw, sgs, sgb, suw, sus, sub, xs, h + slot * max(p.I, p.SI), p,   \
                             shared ? 0u : idx[slot], shared, L, tgp.x, sg, nsg, lane);                           \
  }
NI_Q20_GU(moe_q20_gate_up, false, 16, 4)
NI_Q20_GU(moe_q20_gate_up_slot, true, 16, 4)

// The distinct experts of R <= 8 rows' top-k (idx [R, k], R * k <= 128), in first-occurrence order: ue[u], the rows
// that chose it (umask bit r) and each such row's slot (uslot[u * 8 + r]); n[0] = U. One simdgroup.
[[kernel]] void moe_q20_unique(const device uint* idx [[buffer(0)]], device uint* ue [[buffer(1)]],
                               device uint* umask [[buffer(2)]], device uchar* uslot [[buffer(3)]],
                               device uint* n [[buffer(4)]], constant uint2& rk [[buffer(5)]],
                               uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint upos[512];
  threadgroup atomic_uint um[128];
  const uint R = rk.x, k = rk.y, P = R * k;
  uint U = 0;
  for (uint i0 = 0; i0 < P; i0 += 32) {
    const uint i = i0 + lane;
    const uint e = i < P ? idx[i] : 0u;
    bool first = i < P;
    for (uint j = 0; j < i && first; ++j) first = idx[j] != e;
    const uint u = U + simd_prefix_exclusive_sum(first ? 1u : 0u);
    if (first) {
      upos[e] = u;
      ue[u] = e;
      atomic_store_explicit(&um[u], 0u, memory_order_relaxed);
    }
    U += simd_sum(first ? 1u : 0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < P; i += 32) {
    const uint u = upos[idx[i]], r = i / k;
    atomic_fetch_or_explicit(&um[u], 1u << r, memory_order_relaxed);
    uslot[u * 8 + r] = uchar(i % k);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint u = lane; u < U; u += 32) umask[u] = atomic_load_explicit(&um[u], memory_order_relaxed);
  if (lane == 0) n[0] = U;
}

// Verify gate/up of R <= 8 rows: grid (R, max(I, SI) / (8 ROWS), R * k + 1), 256 threads; z = a distinct expert u of
// moe_q20_unique (the R threadgroups of an expert's tile adjacent: rows that chose it share its weights through the
// cache) or the shared expert (z = R * k). h[(r * (k + 1) + slot) * max(I, SI) + i]. Buffers: 0-9 as the decode
// kernel, 10 xn [R, D], 11 h, 12 MoeParams, 13 ue, 14 umask, 15 uslot, 16 un, 17 R, 18 Q20Slot.
#define NI_Q20V_GU(NAME, SLOT, VPL, ROWS)                                                                              \
  [[kernel]] void NAME(const device uchar* w0 [[buffer(0)]], const device uchar* w1 [[buffer(1)]],                 \
                       const device uchar* w2 [[buffer(2)]], const device uchar* w3 [[buffer(3)]],                 \
                       const device uint32_t* sgw [[buffer(4)]], const device bfloat* sgs [[buffer(5)]],           \
                       const device bfloat* sgb [[buffer(6)]], const device uint32_t* suw [[buffer(7)]],           \
                       const device bfloat* sus [[buffer(8)]], const device bfloat* sub [[buffer(9)]],             \
                       const device float* xn [[buffer(10)]], device float* h [[buffer(11)]],                      \
                       constant MoeParams& p [[buffer(12)]], const device uint* ue [[buffer(13)]],                 \
                       const device uint* umask [[buffer(14)]], const device uchar* uslot [[buffer(15)]],          \
                       const device uint* un [[buffer(16)]], constant uint& R [[buffer(17)]],                      \
                       constant Q20Slot& L [[buffer(18)]], uint3 tgp [[threadgroup_position_in_grid]],             \
                       uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]],           \
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],        \
                       uint lane [[thread_index_in_simdgroup]]) {                                                  \
    threadgroup float xs[4096];                                                                                    \
    const uint r = tgp.x, u = tgp.z;                                                                               \
    const bool shared = u == R * p.k;                                                                              \
    uint slot = p.k, e = 0;                                                                                        \
    if (!shared) {                                                                                                 \
      if (u >= un[0] || !((umask[u] >> r) & 1u)) return; /* uniform across the threadgroup */                     \
      slot = uslot[u * 8 + r];                                                                                     \
      e = ue[u];                                                                                                   \
    }                                                                                                              \
    for (uint i = tid; i < p.D / 4; i += ntg3.x)                                                                   \
      ((threadgroup float4*)xs)[i] = ((const device float4*)(xn + r * p.D))[i];                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
    q20_gu_tile<SLOT, VPL, ROWS>(w0, w1, w2, w3, sgw, sgs, sgb, suw, sus, sub, xs,                                 \
                             h + (r * (p.k + 1) + slot) * max(p.I, p.SI), p, e, shared, L, tgp.y, sg, nsg, lane); \
  }
NI_Q20V_GU(moe_q20v_gate_up, false, 16, 4)
NI_Q20V_GU(moe_q20v_gate_up_slot, true, 16, 4)

// Down of R <= 8 rows (decode: R = 1), moe_down2_w's arithmetic per row: grid (R, D / (8 ROWS)), 256 threads; row r's h
// ((k + 1) * max(I, SI) <= 7168 floats) staged in threadgroup memory 0, sized by the host to what the shapes stage
// (Flash-Next: 7040 floats; small test shapes stay inside the shader validator's limit, which counts it twice);
// y[r, d] += sum_slot wts[r, slot] * (down_slot[d] . h[r, slot]), slot k the shared expert. Buffers: 0-2 the routed
// experts' down (see the top), 3-5 the shared expert's down (w, scales, biases), 6 h, 7 idx [R, k], 8 wts [R, k + 1],
// 9 y [R, D], 10 MoeParams, 11 Q20Slot.
#define NI_Q20_DN(NAME, SLOT, VPL, ROWS)                                                                               \
  [[kernel]] void NAME(const device uchar* w0 [[buffer(0)]], const device uchar* w1 [[buffer(1)]],                 \
                       const device uchar* w2 [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],             \
                       const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],             \
                       const device float* h [[buffer(6)]], const device uint* idx [[buffer(7)]],                  \
                       const device float* wts [[buffer(8)]], device float* y [[buffer(9)]],                       \
                       constant MoeParams& p [[buffer(10)]], constant Q20Slot& L [[buffer(11)]],                   \
                       uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],       \
                       uint2 ntg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],         \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       threadgroup float* hs [[threadgroup(0)]]) {                                                 \
    const uint r = tgp.x, hstride = max(p.I, p.SI);                                                                \
    const device float* hr = h + r * (p.k + 1) * hstride;                                                          \
    for (uint i = tid; i < (p.k + 1) * hstride / 4; i += ntg2.x)                                                   \
      ((threadgroup float4*)hs)[i] = ((const device float4*)hr)[i];                                                \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
    const uint row0 = (tgp.y * nsg + sg) * ROWS;                                                                   \
    if (row0 >= p.D) return;                                                                                       \
    const device uint* ri = idx + r * p.k;                                                                         \
    const device float* rw = wts + r * (p.k + 1);                                                                  \
    float tot[ROWS];                                                                                               \
    for (int q = 0; q < ROWS; ++q) tot[q] = 0.0f;                                                                  \
    for (uint slot = 0; slot <= p.k; ++slot) {                                                                     \
      float a[ROWS];                                                                                               \
      if (slot == p.k) {                                                                                           \
        qmv2_rows<4, 16, ROWS, 64>(sdw, sds, sdb, hs + slot * hstride, p.D, p.SI, row0, lane, a);                  \
      } else {                                                                                                     \
        const Q20Mat W = q20_mat<SLOT>(w0, w1, (const device uint*)w2, ri[slot], p.D * (p.I / 16),                 \
                                       p.D * (p.I / 64), L.dc, L.dd);                                              \
        q20_rows<VPL, ROWS>(W, hs + slot * hstride, p.D, p.I, row0, lane, a);                                      \
      }                                                                                                            \
      const float ws = rw[slot];                                                                                   \
      for (int q = 0; q < ROWS; ++q) tot[q] += ws * a[q];                                                          \
    }                                                                                                              \
    if (lane == 0)                                                                                                 \
      for (int q = 0; q < ROWS; ++q)                                                                               \
        if (row0 + q < p.D) y[r * p.D + row0 + q] += tot[q];                                                       \
  }
NI_Q20_DN(moe_q20_down, false, 16, 4)
NI_Q20_DN(moe_q20_down_slot, true, 16, 4)

// ---- Prefill at small T: qmms_moe2_*'s tiles of <= 8 sorted pair rows (moe_bucket3, BM 8, no shared rows) ----
// One A-fragment element pair of qs_dequant's k order (fragment f, element e <-> code 8(f/4) + f%4 + 4e of the lane's
// 16 codes, one word w of the 64-block): half2(1024 + code) - 1024, times d plus -d: (code - 1) d, exact in one FMA.
NI_INLINE half2 q20s_pair(uint w, uint f, half2 s2, half2 b2) {
  const uint x = w >> (2 * (8 * (f / 4) + f % 4));
  const uint v = (x & 0x3u) | ((x & 0x300u) << 8) | 0x64006400u;
  return fma(as_type<half2>(v) - half2(1024.0h), s2, b2);
}

// qs_dequant for Q2_0: this lane's A elements of the 8 fragments of 64-block g of weight row `wrow`
NI_INLINE void q20s_dequant(const device uint32_t* wrow, half d, uint g, uint q, thread half2* a) {
  const uint w = wrow[g * 4 + q];
  const half2 s2 = half2(d), b2 = half2(-d);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) a[f] = q20s_pair(w, uint(f), s2, b2);
}

// qv_dot for Q2_0: the same half weights as q20s_dequant, products and sums in fp32
template <int R>
NI_INLINE void q20v_dot(const device uint32_t* wrow, half d, uint g, uint q, thread const float (&xs)[R][16],
                        thread float (&acc)[R]) {
  const uint w = wrow[g * 4 + q];
  const half2 s2 = half2(d), b2 = half2(-d);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) {
    const half2 wv = q20s_pair(w, uint(f), s2, b2);
    const float a0 = float(wv.x), a1 = float(wv.y);
    const int c = 8 * (f / 4) + f % 4;
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < R; ++t) acc[t] = fma(a1, xs[t][c + 4], fma(a0, xs[t][c], acc[t]));
  }
}

// qv_moe_gu / qv_moe_dn for Q2_0 (experts of 1-2 rows by FMAs; see qmms_moe2_gu_b4)
template <int R>
NI_INLINE void q20v_gu(const device half* X, const device uint* xrows, uint x_ld, uint m0, uint rows, Q20Mat G,
                       Q20Mat U, device half* H, uint y_ld, uint n, uint K, uint lane) {
  const uint2 rc = qs_lane_rc(lane);
  const uint q = rc.y / 2, gpr = K / 64, wpr = K / 16;
  const device half* xr[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) xr[t] = X + xrows[m0 + min(uint(t), rows - 1)] * x_ld;
  float ag[R], au[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) ag[t] = au[t] = 0.0f;
  for (uint g = 0; g < gpr; ++g) {
    float xs[R][16];
    qv_load_x<R>(xr, g, q, xs);
    q20v_dot<R>(G.c + n * wpr, G.d[n * gpr + g], g, q, xs, ag);
    q20v_dot<R>(U.c + n * wpr, U.d[n * gpr + g], g, q, xs, au);
  }
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) {
    ag[t] += simd_shuffle_xor(ag[t], 1);
    ag[t] += simd_shuffle_xor(ag[t], 8);
    au[t] += simd_shuffle_xor(au[t], 1);
    au[t] += simd_shuffle_xor(au[t], 8);
  }
  if ((lane & 9) == 0)
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < R; ++t)
      if (uint(t) < rows) H[(m0 + t) * y_ld + n] = half(silu(ag[t]) * au[t]);
}

template <int R>
NI_INLINE void q20v_dn(const device half* X, uint x_ld, uint m0, uint rows, Q20Mat W, device float* Y,
                       const device uint* yrows, uint y_ld, uint n, uint K, uint lane) {
  const uint2 rc = qs_lane_rc(lane);
  const uint q = rc.y / 2, gpr = K / 64, wpr = K / 16;
  const device half* xr[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) xr[t] = X + (m0 + min(uint(t), rows - 1)) * x_ld;
  float acc[R];
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) acc[t] = 0.0f;
  for (uint g = 0; g < gpr; ++g) {
    float xs[R][16];
    qv_load_x<R>(xr, g, q, xs);
    q20v_dot<R>(W.c + n * wpr, W.d[n * gpr + g], g, q, xs, acc);
  }
  _Pragma("clang loop unroll(full)")
  for (int t = 0; t < R; ++t) {
    acc[t] += simd_shuffle_xor(acc[t], 1);
    acc[t] += simd_shuffle_xor(acc[t], 8);
  }
  if ((lane & 9) == 0)
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < R; ++t)
      if (uint(t) < rows) Y[yrows[m0 + t] * y_ld + n] = acc[t];
}

// Gate/up (+ SwiGLU) on tiles of <= 8 rows: grid (N / 8 / nsg, max tiles), 32 * nsg threads; H[sorted row, n] half.
// Buffers: 0 X (half [T, x_ld]), 1-4 the routed experts' gate/up (see the top), 5 H, 6 Qmm2Params (expert strides in
// words / halfs), 7 tiles, 8 ntiles, 9 xrows, 10 Q20Slot.
template <bool SLOT>
NI_INLINE void q20s_moe_gu(const device half* X, const device uchar* w0, const device uchar* w1,
                           const device uchar* w2, const device uchar* w3, device half* H, constant Qmm2Params& p,
                           const device uint* tiles, const device uint* ntiles, const device uint* xrows,
                           constant Q20Slot& L, uint2 tg, uint sg, uint nsg, uint lane) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const device uint* slots = (const device uint*)w2;
  const Q20Mat G = q20_mat<SLOT>(w0, w1, slots, e, p.w_expert_stride, p.s_expert_stride, L.gc, L.gd);
  const Q20Mat U = q20_mat<SLOT>(SLOT ? w0 : w2, SLOT ? w1 : w3, slots, e, p.w_expert_stride, p.s_expert_stride,
                                 L.uc, L.ud);
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;  // N % 8 == 0 (checked on the host)
  // FMA for experts of at most 2 rows (their first and only tile), as qmms_moe2_gu_b4
  const bool first = tg.y == 0 || tiles[(tg.y - 1) * 3] != e;
  if (first && rows == 1) {
    q20v_gu<1>(X, xrows, p.x_ld, m0, rows, G, U, H, p.y_ld, n, p.K, lane);
    return;
  }
  if (first && rows == 2) {
    q20v_gu<2>(X, xrows, p.x_ld, m0, rows, G, U, H, p.y_ld, n, p.K, lane);
    return;
  }
  const uint gpr = p.K / 64, wpr = p.K / 16;
  const device half* x0 = rc.y < rows ? X + xrows[m0 + rc.y] * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + xrows[m0 + rc.y + 1] * p.x_ld : nullptr;
  simdgroup_float8x8 ag = simdgroup_float8x8(0), au = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a0[8], a1[8], b[8];
    q20s_dequant(G.c + n * wpr, G.d[n * gpr + g], g, q, a0);
    q20s_dequant(U.c + n * wpr, U.d[n * gpr + g], g, q, a1);
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

// Down on tiles of <= 8 rows: Y[yrows[sorted row], n] fp32. Buffers: 0 X (= the gate/up kernel's H), 1-3 the routed
// experts' down (see the top), 4 Y, 5 Qmm2Params, 6 tiles, 7 ntiles, 8 yrows, 9 Q20Slot.
template <bool SLOT>
NI_INLINE void q20s_moe_dn(const device half* X, const device uchar* w0, const device uchar* w1,
                           const device uchar* w2, device float* Y, constant Qmm2Params& p, const device uint* tiles,
                           const device uint* ntiles, const device uint* yrows, constant Q20Slot& L, uint2 tg,
                           uint sg, uint nsg, uint lane) {
  if (tg.y >= ntiles[0]) return;
  uint e, m0, rows;
  qs_tile(p, tiles, tg.y, e, m0, rows);
  const Q20Mat W =
      q20_mat<SLOT>(w0, w1, (const device uint*)w2, e, p.w_expert_stride, p.s_expert_stride, L.dc, L.dd);
  const uint2 rc = qs_lane_rc(lane);
  const uint n = (tg.x * nsg + sg) * 8 + rc.x;
  const bool first = tg.y == 0 || tiles[(tg.y - 1) * 3] != e;
  if (first && rows == 1) {
    q20v_dn<1>(X, p.x_ld, m0, rows, W, Y, yrows, p.y_ld, n, p.K, lane);
    return;
  }
  if (first && rows == 2) {
    q20v_dn<2>(X, p.x_ld, m0, rows, W, Y, yrows, p.y_ld, n, p.K, lane);
    return;
  }
  const uint gpr = p.K / 64, wpr = p.K / 16;
  const device half* x0 = rc.y < rows ? X + (m0 + rc.y) * p.x_ld : nullptr;
  const device half* x1 = rc.y + 1 < rows ? X + (m0 + rc.y + 1) * p.x_ld : nullptr;
  simdgroup_float8x8 acc = simdgroup_float8x8(0);
  const uint fm = rc.x, q = rc.y / 2;
  for (uint g = 0; g < gpr; ++g) {
    half2 a[8], b[8];
    q20s_dequant(W.c + n * wpr, W.d[n * gpr + g], g, q, a);
    qs_gather(x0, x1, g, fm, b);
    qs_mma8(acc, a, b);
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 2; ++j) {
    const uint r = rc.y + j;
    if (r < rows) Y[yrows[m0 + r] * p.y_ld + n] = acc.thread_elements()[j];
  }
}

#define NI_Q20S_MOE(GU, DN, SLOT)                                                                                  \
  [[kernel]] void GU(const device half* X [[buffer(0)]], const device uchar* w0 [[buffer(1)]],                     \
                     const device uchar* w1 [[buffer(2)]], const device uchar* w2 [[buffer(3)]],                   \
                     const device uchar* w3 [[buffer(4)]], device half* H [[buffer(5)]],                           \
                     constant Qmm2Params& p [[buffer(6)]], const device uint* tiles [[buffer(7)]],                 \
                     const device uint* ntiles [[buffer(8)]], const device uint* xrows [[buffer(9)]],              \
                     constant Q20Slot& L [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],                \
                     uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                    \
    q20s_moe_gu<SLOT>(X, w0, w1, w2, w3, H, p, tiles, ntiles, xrows, L, tg, sg, nsg, lane);                        \
  }                                                                                                                \
  [[kernel]] void DN(const device half* X [[buffer(0)]], const device uchar* w0 [[buffer(1)]],                     \
                     const device uchar* w1 [[buffer(2)]], const device uchar* w2 [[buffer(3)]],                   \
                     device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]],                          \
                     const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],              \
                     const device uint* yrows [[buffer(8)]], constant Q20Slot& L [[buffer(9)]],                    \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],        \
                     uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {           \
    q20s_moe_dn<SLOT>(X, w0, w1, w2, Y, p, tiles, ntiles, yrows, L, tg, sg, nsg, lane);                            \
  }
NI_Q20S_MOE(qmms_moe_gu_q20, qmms_moe_dn_q20, false)
NI_Q20S_MOE(qmms_moe_gu_q20_slot, qmms_moe_dn_q20_slot, true)

// ---- Prefill: the qrm design (qmm_moe3.metal) with Q2_0 weights, BK = 64 ----
// At BK = 64 a lane's 16 codes of a K step are one word of its row (word k0/16 + c), and qmr's k order for BK = 64
// (fragment f, column 2c + e <-> k0 + 16c + 8(f/4) + f%4 + 4e) puts codes j = 8(f/4) + f%4 and j + 4 of that word in a
// fragment's element pair: X is staged exactly as qr_store_x<.., 64, ..> stages it for the 4-bit kernels.
// One element pair: half2 0.25 + code/4096 (0x3400 | code), minus 0.25, times 4096 d, minus d: (code - 1) d, exact.
NI_INLINE half2 q20r_pair(uint w, uint f, half2 sb) {
  const uint x = w >> (2 * (8 * (f / 4) + f % 4));
  const uint t = (x & 0x3u) | ((x & 0x300u) << 8) | 0x34003400u;
  return fma(as_type<half2>(t) - half2(0.25h), half2(sb.x), half2(sb.y));
}

// This lane's word of its FA rows of each of the NP matrices for the K step at k0, and (4096 d, -d)
template <int FA, int NP>
NI_INLINE void q20r_load_w(thread const Q20Mat* M, thread const uint* wrow, uint wpr, uint gpr, uint k0, uint c,
                           thread uint* wr, thread half2* sb) {
  const uint g = k0 / 64;
  _Pragma("clang loop unroll(full)")
  for (int p = 0; p < NP; ++p)
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      const int i = p * FA + a;
      wr[i] = M[p].c[wrow[a] * wpr + k0 / 16 + c];
      const half d = M[p].d[wrow[a] * gpr + g];
      sb[i] = half2(d * 4096.0h, -d);
    }
}

template <int FA, int FB, int TM>
NI_INLINE void q20r_mma(thread simdgroup_float8x8 (&acc)[FA][FB], threadgroup const half* xs, uint sgm,
                        thread const uint* wr, thread const half2* sb) {
  constexpr int LD = 64 + 8;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) {
    simdgroup_half8x8 af[FA];
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FA; ++a) {
      const half2 v = q20r_pair(wr[a], uint(f), sb[a]);
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

// qrm_acc for Q2_0 at BK = 64
template <int FL, int BN, int WN, int NP>
NI_INLINE void q20rm_acc(const device half* X, const device uint* xrows, bool gather, uint x_ld, uint m0, uint rows,
                         thread const Q20Mat* M, uint N, uint K, uint n0, threadgroup half* xs, uint tid, uint sgn,
                         uint2 rc, thread simdgroup_float8x8 (&acc)[NP * (BN / WN / 8)][FL]) {
  constexpr int BK = 64, NT = WN * 32, TN = BN / WN, FA = TN / 8, BMl = FL * 8;
  constexpr int CPR = BK / 8, XL = (BMl * CPR + NT - 1) / NT;
  const uint c = rc.y / 2;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < NP * FA; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FL; ++b) acc[a][b] = simdgroup_float8x8(0);
  const uint wpr = K / 16, gpr = K / 64;
  uint wrow[FA];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) wrow[a] = min(n0 + sgn * TN + a * 8 + rc.x, N - 1);
  uint xo[XL];
  qrm_xoff<BMl, BK, NT>(xrows, gather, x_ld / 8, m0, rows, tid, xo);
  const device uint4* X4 = (const device uint4*)X;
  uint4 xr[XL];
  uint wr[NP * FA];
  half2 sb[NP * FA];
  for (uint k0 = 0; k0 < K; k0 += BK) {
    qrm_load_x<BMl, BK, NT>(X4, xo, k0, xr);
    q20r_load_w<FA, NP>(M, wrow, wpr, gpr, k0, c, wr, sb);
    qr_store_x<BMl, BK, NT>(xs, tid, xr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q20r_mma<NP * FA, FL, BMl>(acc, xs, 0, wr, sb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// gate/up body for FL live fragments: H[m0 + m, n] = half(silu(g) * u)
template <int FL, int BN, int WN>
NI_INLINE void q20rm_gu_body(const device half* X, const device uint* xrows, uint x_ld, uint m0, uint rows,
                             thread const Q20Mat* M, device half* H, uint y_ld, uint N, uint K, uint n0,
                             threadgroup half* xs, uint tid, uint sgn, uint2 rc) {
  constexpr int TN = BN / WN, FA = TN / 8;
  simdgroup_float8x8 acc[2 * FA][FL];
  q20rm_acc<FL, BN, WN, 2>(X, xrows, true, x_ld, m0, rows, M, N, K, n0, xs, tid, sgn, rc, acc);
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FA; ++a) {
    const uint n = n0 + sgn * TN + a * 8 + rc.x;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FL; ++b)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint m = b * 8 + rc.y + e;
        const float g = acc[a][b].thread_elements()[e], u = acc[FA + a][b].thread_elements()[e];
        if (m < rows) H[(m0 + m) * y_ld + n] = half(silu(g) * u);
      }
  }
}

// down body for FL live fragments: Y[yrows[m0 + m], n] = acc
template <int FL, int BN, int WN>
NI_INLINE void q20rm_dn_body(const device half* X, uint x_ld, uint m0, uint rows, thread const Q20Mat* M,
                             device float* Y, const device uint* yrows, uint y_ld, uint N, uint K, uint n0,
                             threadgroup half* xs, uint tid, uint sgn, uint2 rc) {
  constexpr int TN = BN / WN, FA = TN / 8;
  simdgroup_float8x8 acc[FA][FL];
  q20rm_acc<FL, BN, WN, 1>(X, nullptr, false, x_ld, m0, rows, M, N, K, n0, xs, tid, sgn, rc, acc);
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

// Gate/up (+ SwiGLU) on moe_bucket3 tiles of at most 8 * FB routed rows: grid (N / BN, max tiles), 32 * WN threads.
// Buffers as qmms_moe_gu_q20 (0 X, 1-4 the experts, 5 H, 6 Qmm2Params, 7 tiles, 8 ntiles, 9 xrows, 10 Q20Slot).
#define NI_Q20RM_GU(NAME, SLOT, FB, BN, WN)                                                                        \
  [[kernel, max_total_threads_per_threadgroup(WN * 32)]] void NAME(                                                 \
      const device half* X [[buffer(0)]], const device uchar* w0 [[buffer(1)]], const device uchar* w1 [[buffer(2)]], \
      const device uchar* w2 [[buffer(3)]], const device uchar* w3 [[buffer(4)]], device half* H [[buffer(5)]],    \
      constant Qmm2Params& p [[buffer(6)]], const device uint* tiles [[buffer(7)]],                                \
      const device uint* ntiles [[buffer(8)]], const device uint* xrows [[buffer(9)]],                             \
      constant Q20Slot& L [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                        \
      uint lane [[thread_index_in_simdgroup]]) {                                                                   \
    if (tg.y >= ntiles[0]) return;                                                                                 \
    threadgroup uint4 xs4[8 * FB * (64 + 8) / 8];                                                                  \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;                 \
    const device uint* slots = (const device uint*)w2;                                                             \
    const Q20Mat M[2] = {                                                                                          \
        q20_mat<SLOT>(w0, w1, slots, e, p.w_expert_stride, p.s_expert_stride, L.gc, L.gd),                         \
        q20_mat<SLOT>(SLOT ? w0 : w2, SLOT ? w1 : w3, slots, e, p.w_expert_stride, p.s_expert_stride, L.uc, L.ud)}; \
    const uint n0 = tg.x * BN;                                                                                     \
    const uint2 rc = qr_rc(lane);                                                                                  \
    threadgroup half* xs = (threadgroup half*)xs4;                                                                 \
    const uint live = (rows + 7) / 8;                                                                              \
    if (FB >= 4 && live >= 4)                                                                                      \
      q20rm_gu_body<(FB >= 4 ? 4 : 1), BN, WN>(X, xrows, p.x_ld, m0, rows, M, H, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                               rc);                                                                \
    else if (FB >= 3 && live == 3)                                                                                 \
      q20rm_gu_body<(FB >= 3 ? 3 : 1), BN, WN>(X, xrows, p.x_ld, m0, rows, M, H, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                               rc);                                                                \
    else if (FB >= 2 && live == 2)                                                                                 \
      q20rm_gu_body<(FB >= 2 ? 2 : 1), BN, WN>(X, xrows, p.x_ld, m0, rows, M, H, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                               rc);                                                                \
    else                                                                                                           \
      q20rm_gu_body<1, BN, WN>(X, xrows, p.x_ld, m0, rows, M, H, p.y_ld, p.N, p.K, n0, xs, tid, sg, rc);          \
  }

// Down to fp32 rows by pair on the same tiles: grid (N / BN, max tiles). Buffers as qmms_moe_dn_q20 (0 X = H, 1-3 the
// experts, 4 Y, 5 Qmm2Params, 6 tiles, 7 ntiles, 8 yrows, 9 Q20Slot).
#define NI_Q20RM_DN(NAME, SLOT, FB, BN, WN)                                                                        \
  [[kernel, max_total_threads_per_threadgroup(WN * 32)]] void NAME(                                                 \
      const device half* X [[buffer(0)]], const device uchar* w0 [[buffer(1)]], const device uchar* w1 [[buffer(2)]], \
      const device uchar* w2 [[buffer(3)]], device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]],   \
      const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],                             \
      const device uint* yrows [[buffer(8)]], constant Q20Slot& L [[buffer(9)]],                                   \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                         \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                       \
    if (tg.y >= ntiles[0]) return;                                                                                 \
    threadgroup uint4 xs4[8 * FB * (64 + 8) / 8];                                                                  \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;                 \
    const Q20Mat M[1] = {                                                                                          \
        q20_mat<SLOT>(w0, w1, (const device uint*)w2, e, p.w_expert_stride, p.s_expert_stride, L.dc, L.dd)};      \
    const uint n0 = tg.x * BN;                                                                                     \
    const uint2 rc = qr_rc(lane);                                                                                  \
    threadgroup half* xs = (threadgroup half*)xs4;                                                                 \
    const uint live = (rows + 7) / 8;                                                                              \
    if (FB >= 4 && live >= 4)                                                                                      \
      q20rm_dn_body<(FB >= 4 ? 4 : 1), BN, WN>(X, p.x_ld, m0, rows, M, Y, yrows, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                               rc);                                                                \
    else if (FB >= 3 && live == 3)                                                                                 \
      q20rm_dn_body<(FB >= 3 ? 3 : 1), BN, WN>(X, p.x_ld, m0, rows, M, Y, yrows, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                               rc);                                                                \
    else if (FB >= 2 && live == 2)                                                                                 \
      q20rm_dn_body<(FB >= 2 ? 2 : 1), BN, WN>(X, p.x_ld, m0, rows, M, Y, yrows, p.y_ld, p.N, p.K, n0, xs, tid, sg, \
                                               rc);                                                                \
    else                                                                                                           \
      q20rm_dn_body<1, BN, WN>(X, p.x_ld, m0, rows, M, Y, yrows, p.y_ld, p.N, p.K, n0, xs, tid, sg, rc);          \
  }

// naming as qrm: qrm_{gu,dn}_q20_m{8*FB}_n{BN}; 4 simdgroups along N (128 threads)
NI_Q20RM_GU(qrm_gu_q20_m32_n64, false, 4, 64, 4)
NI_Q20RM_GU(qrm_gu_q20_m32_n64_slot, true, 4, 64, 4)
NI_Q20RM_DN(qrm_dn_q20_m32_n128, false, 4, 128, 4)
NI_Q20RM_DN(qrm_dn_q20_m32_n128_slot, true, 4, 128, 4)
