#include "common.h"

// ---- Routed experts in EXL3 (turboderp's trellis format; Sushi's Qwen3.8-Flash-Next packs, src/model/moe_exl3.h) ----
// A projection [in, out] of one expert is tiles of 16 x 16 weights, trellis [in/16][out/16][n] uint16: n halfwords
// (16 n bits, n / 16 bits a weight) per tile. Weight t of a tile (0..255) is the 16-bit window of the tile's bitstream
// (each 32-bit word MSB first, the stream tail-biting: it wraps) ending at bit ((t + 1) n) >> 4, masked to the pack's
// window and decoded by the MCG codebook to a half; its place in the tile is the tensor-core order: lane l's weights
// 8l .. 8l + 7 are rows 2q, 2q+1, 2q+8, 2q+9 (q = l % 4) of columns l / 4 (the first four) and l / 4 + 8 (the rest).
// The projection is y = svh * H128(W^T H128(suh * x)), H128 the orthonormal Hadamard transform of each block of 128,
// suh / svh per expert (f16), the transformed input and output rounded to f16 as the reference rounds them (Sushi's
// expert_exl3.zig: prepareInput, finishOutput); the inner products are fp32. The SwiGLU between stays fp32 up to the
// down projection's transform, as Sushi's chain does: an f16 product overflows on large activations (Sushi saw gate and
// up near 400 on MiMo, a product past 65504), so the down input is rounded to f16 only after H128, where suh has scaled
// it down. Rates n 16-64 (K1-K4: a lane's 8 codewords lie in the 64 bits ending at its last one).

struct Exl3Params {
  uint D, I, k;     // hidden, expert width, top-k
  uint ng, nu, nd;  // halfwords per 256-weight tile: gate, up, down
  uint mask;        // the codeword window: (1 << window) - 1
  uint rows;        // rows of the pass: R k routed slots (decode / verify), sorted rows or pairs (prefill)
};

constant constexpr float kExl3Had = 0.08838834764831845f;  // 1 / sqrt(128)

// The MCG codebook: the codeword times 0xCBAC1FED, masked and xored into two halves, their f32 sum as a half
NI_INLINE float exl3_mcg(uint cw) {
  const half2 h = as_type<half2>(0x3B603B60u ^ ((cw * 0xCBAC1FEDu) & 0x8FFF8FFFu));
  return float(half(float(h.x) + float(h.y)));
}

// The 8 codewords of lane `lane` in a tile at words w (n / 2 of them): weights 8 lane .. 8 lane + 7, read from the 64
// bits of the stream ending at the lane's last bit (three words: the window rarely starts on a word boundary)
NI_INLINE void exl3_lane_cw(const device uint* w, uint n, uint lane, thread uint* cw) {
  const uint wc = n / 2, total = 16 * n, t0 = lane * 8;
  const uint E = (((t0 + 8) * n) >> 4) + total;  // one past the last bit, a lap up (the stream wraps at lane 0)
  const uint i2 = (E - 1) / 32, shift = (i2 + 1) * 32 - E;
  const ulong a = (ulong(w[(i2 - 1) % wc]) << 32) | ulong(w[i2 % wc]);
  const ulong win = shift ? (a >> shift) | (ulong(w[(i2 - 2) % wc]) << (64 - shift)) : a;
  for (uint m = 0; m < 8; ++m) cw[m] = uint(win >> (E - total - (((t0 + m + 1) * n) >> 4))) & 0xFFFFu;
}

// H128 of the 128 values a simdgroup holds, 4 a lane (values 4 lane .. 4 lane + 3): the butterflies in the natural
// (Sylvester) order, scaled by 1 / sqrt(128)
NI_INLINE void exl3_h128(thread float* v, uint lane) {
  float a = v[0], b = v[1];
  v[0] = a + b, v[1] = a - b;
  a = v[2], b = v[3];
  v[2] = a + b, v[3] = a - b;
  a = v[0], b = v[2];
  v[0] = a + b, v[2] = a - b;
  a = v[1], b = v[3];
  v[1] = a + b, v[3] = a - b;
  for (ushort h = 1; h < 32; h <<= 1) {
    const bool hi = (lane & h) != 0;
    for (int m = 0; m < 4; ++m) {
      const float o = simd_shuffle_xor(v[m], h);
      v[m] = hi ? o - v[m] : v[m] + o;
    }
  }
  for (int m = 0; m < 4; ++m) v[m] *= kExl3Had;
}

// x (fp32 [D]) as f16(H128(f16(x) * suh)) into xs (threadgroup, D floats): NSG simdgroups, a 128-block each in turn
NI_INLINE void exl3_prep(const device float* x, const device half* suh, threadgroup float* xs, uint D, uint sg,
                         uint nsg, uint lane) {
  for (uint b = sg; b < D / 128; b += nsg) {
    const uint c0 = b * 128 + lane * 4;
    float v[4];
    for (int m = 0; m < 4; ++m) v[m] = float(half(x[c0 + m])) * float(suh[c0 + m]);
    exl3_h128(v, lane);
    for (int m = 0; m < 4; ++m) xs[c0 + m] = float(half(v[m]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The 16 inner products of one tile column (w: its first tile, tstride words between tiles down the column, ktiles of
// them) with xs: lanes with lane % 4 == 0 return columns lane / 4 (x) and lane / 4 + 8 (y)
NI_INLINE float2 exl3_col(const device uint* w, uint n, uint tstride, uint ktiles, threadgroup const float* xs,
                          uint mask, uint lane) {
  const uint q = lane & 3;
  float2 acc = 0.0f;
  for (uint tk = 0; tk < ktiles; ++tk) {
    uint cw[8];
    exl3_lane_cw(w + tk * tstride, n, lane, cw);
    threadgroup const float* xr = xs + tk * 16 + 2 * q;
    const float x0 = xr[0], x1 = xr[1], x2 = xr[8], x3 = xr[9];
    acc.x += x0 * exl3_mcg(cw[0] & mask) + x1 * exl3_mcg(cw[1] & mask) + x2 * exl3_mcg(cw[2] & mask) +
             x3 * exl3_mcg(cw[3] & mask);
    acc.y += x0 * exl3_mcg(cw[4] & mask) + x1 * exl3_mcg(cw[5] & mask) + x2 * exl3_mcg(cw[6] & mask) +
             x3 * exl3_mcg(cw[7] & mask);
  }
  acc.x += simd_shuffle_xor(acc.x, 1);
  acc.x += simd_shuffle_xor(acc.x, 2);
  acc.y += simd_shuffle_xor(acc.y, 1);
  acc.y += simd_shuffle_xor(acc.y, 2);
  return acc;
}

// ---- Decode / verify: R rows' k routed slots (s = r k + j, expert idx[s]) ----

// Gate and up: grid (ceil(I / 128), R k, 2: gate / up) of 8 simdgroups. The threadgroup prepares row s / k of x (fp32
// [R, D]) with the slot's suh (D floats of threadgroup memory), then each simdgroup a 16-column tile: hg / hu [R k, I]
// (fp32, the inner products before the output transform).
[[kernel]] void exl3_gu_rows(const device uint* gt [[buffer(0)]], const device half* gsuh [[buffer(1)]],
                             const device uint* ut [[buffer(2)]], const device half* usuh [[buffer(3)]],
                             const device float* x [[buffer(4)]], const device uint* idx [[buffer(5)]],
                             device float* hg [[buffer(6)]], device float* hu [[buffer(7)]],
                             constant Exl3Params& p [[buffer(8)]], threadgroup float* xs [[threadgroup(0)]],
                             uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint s = tg.y, e = idx[s];
  const bool up = tg.z != 0;
  exl3_prep(x + ulong(s / p.k) * p.D, (up ? usuh : gsuh) + ulong(e) * p.D, xs, p.D, sg, 8, lane);
  const uint ot = p.I / 16, tn = tg.x * 8 + sg;
  if (tn >= ot) return;
  const uint n = up ? p.nu : p.ng, hw = n / 2;
  const device uint* w = (up ? ut : gt) + ulong(e) * (p.D / 16) * ot * hw + tn * hw;
  const float2 a = exl3_col(w, n, ot * hw, p.D / 16, xs, p.mask, lane);
  if ((lane & 3) == 0) {
    device float* o = (up ? hu : hg) + ulong(s) * p.I + tn * 16 + lane / 4;
    o[0] = a.x;
    o[8] = a.y;
  }
}

// Between gate/up and down, a row and a 128-block of I at a time: g = f16(svh_g * H128(hg)), u = f16(svh_u * H128(hu)),
// m = silu(g) u (fp32), the down projection's input f16(H128(m * suh_d)) into md (half [rows, I]). The row's expert:
// idx[s], or for sorted prefill rows (sorted != 0) idx[yrows[s]]. grid (I / 128, ceil(rows / 4)) of 4 simdgroups.
[[kernel]] void exl3_mid(const device float* hg [[buffer(0)]], const device float* hu [[buffer(1)]],
                         const device half* gsvh [[buffer(2)]], const device half* usvh [[buffer(3)]],
                         const device half* dsuh [[buffer(4)]], const device uint* idx [[buffer(5)]],
                         const device uint* yrows [[buffer(6)]], device half* md [[buffer(7)]],
                         constant Exl3Params& p [[buffer(8)]], constant uint& sorted [[buffer(9)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint s = tg.y * 4 + sg;
  if (s >= p.rows) return;
  const uint e = idx[sorted ? yrows[s] : s];
  const uint c0 = tg.x * 128 + lane * 4;
  const ulong o = ulong(s) * p.I + c0, oe = ulong(e) * p.I + c0;
  float g[4], u[4];
  for (int m = 0; m < 4; ++m) g[m] = hg[o + m], u[m] = hu[o + m];
  exl3_h128(g, lane);
  exl3_h128(u, lane);
  float d[4];
  for (int m = 0; m < 4; ++m) {
    const float gv = float(half(g[m] * float(gsvh[oe + m]))), uv = float(half(u[m] * float(usvh[oe + m])));
    d[m] = silu(gv) * uv * float(dsuh[oe + m]);
  }
  exl3_h128(d, lane);
  for (int m = 0; m < 4; ++m) md[o + m] = half(d[m]);
}

// Down: md (half [R k, I], prepared) through slot s's expert: yd [R k, D] (fp32, before the output transform).
// grid (ceil(D / 128), R k) of 8 simdgroups, I floats of threadgroup memory.
[[kernel]] void exl3_dn_rows(const device uint* dt [[buffer(0)]], const device half* md [[buffer(1)]],
                             const device uint* idx [[buffer(2)]], device float* yd [[buffer(3)]],
                             constant Exl3Params& p [[buffer(4)]], threadgroup float* xs [[threadgroup(0)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint s = tg.y, e = idx[s];
  for (uint i = tid; i < p.I; i += 256) xs[i] = float(md[ulong(s) * p.I + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint ot = p.D / 16, tn = tg.x * 8 + sg;
  if (tn >= ot) return;
  const uint hw = p.nd / 2;
  const device uint* w = dt + ulong(e) * (p.I / 16) * ot * hw + tn * hw;
  const float2 a = exl3_col(w, p.nd, ot * hw, p.I / 16, xs, p.mask, lane);
  if ((lane & 3) == 0) {
    device float* o = yd + ulong(s) * p.D + tn * 16 + lane / 4;
    o[0] = a.x;
    o[8] = a.y;
  }
}

// The rows' sums: y[r] += sum_j wts[r, j] f16(svh_d * H128(yd[r k + j])) + wts[r, k] ys[r] (ys: the shared expert's
// output, fp32 [R, D]; wts [R, k + 1]). grid (D / 512, R) of 4 simdgroups, a 128-block each.
[[kernel]] void exl3_fin_rows(const device float* yd [[buffer(0)]], const device half* dsvh [[buffer(1)]],
                              const device uint* idx [[buffer(2)]], const device float* wts [[buffer(3)]],
                              const device float* ys [[buffer(4)]], device float* y [[buffer(5)]],
                              constant Exl3Params& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.y, c0 = (tg.x * 4 + sg) * 128 + lane * 4;
  if (c0 >= p.D) return;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint j = 0; j < p.k; ++j) {
    const uint s = r * p.k + j;
    const ulong oe = ulong(idx[s]) * p.D + c0;
    float v[4];
    for (int m = 0; m < 4; ++m) v[m] = yd[ulong(s) * p.D + c0 + m];
    exl3_h128(v, lane);
    const float wj = wts[r * (p.k + 1) + j];
    for (int m = 0; m < 4; ++m) acc[m] += wj * float(half(v[m] * float(dsvh[oe + m])));
  }
  const float ws = wts[r * (p.k + 1) + p.k];
  for (int m = 0; m < 4; ++m) y[ulong(r) * p.D + c0 + m] += acc[m] + ws * ys[ulong(r) * p.D + c0 + m];
}

// ---- Prefill: moe_bucket3's tiles of <= 32 sorted rows of one expert ----
// tiles[3i] = (expert, begin, end) of sorted rows; xrows[s] the token of sorted row s, yrows[s] its pair (t k + j).

constant constexpr uint kExl3Bm = 32, kExl3Bn = 64, kExl3Xld = 136, kExl3Wld = 72;

// The tile's rows times columns n0 .. n0 + 64 of an expert's projection (wbase: its trellis, ot tile columns, n
// halfwords a tile), over in = 128 * kblocks inputs: for each 128-block the rows' inputs into Xs (PREP: rows of x by
// token, f16(H128(f16(x) * suh)); else the prepared rows of x as they are), then for each of its 8 rows of tiles the
// 4 tiles of the columns decoded into Ws (a simdgroup each) and 8 x 8 simdgroup MMAs: simdgroup sg accumulates all
// 32 rows of columns n0 + 16 sg .. + 16.
template <bool PREP>
NI_INLINE void exl3_mm(const device uint* wbase, uint n, uint ot, uint kblocks, const device half* suh,
                       const device half* x, uint x_ld, const device uint* xrows, uint b0, uint nr, uint tn0,
                       uint mask, threadgroup half* Xs, threadgroup half* Ws, thread simdgroup_float8x8 (&acc)[4][2],
                       uint sg, uint lane) {
  const uint hw = n / 2, q = lane & 3, c = lane >> 2;
  for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = simdgroup_float8x8(0);
  for (uint kb = 0; kb < kblocks; ++kb) {
    for (uint i = sg; i < kExl3Bm; i += 4) {
      float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (i < nr) {
        const uint c0 = kb * 128 + lane * 4;
        const device half* xr = x + ulong(PREP ? xrows[b0 + i] : b0 + i) * x_ld + c0;
        if (PREP) {
          for (int m = 0; m < 4; ++m) v[m] = float(xr[m]) * float(suh[c0 + m]);
          exl3_h128(v, lane);
        } else {
          for (int m = 0; m < 4; ++m) v[m] = float(xr[m]);
        }
      }
      for (int m = 0; m < 4; ++m) Xs[i * kExl3Xld + lane * 4 + m] = half(v[m]);
    }
    for (uint kt = 0; kt < 8; ++kt) {
      uint cw[8];
      exl3_lane_cw(wbase + ((kb * 8 + kt) * ot + tn0 + sg) * hw, n, lane, cw);
      threadgroup half* wt = Ws + sg * 16 + c;
      const uint rr[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
      for (int m = 0; m < 4; ++m) {
        wt[rr[m] * kExl3Wld] = half(exl3_mcg(cw[m] & mask));
        wt[rr[m] * kExl3Wld + 8] = half(exl3_mcg(cw[m + 4] & mask));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint kk = 0; kk < 2; ++kk) {
        simdgroup_half8x8 b0f, b1f;
        simdgroup_load(b0f, Ws + kk * 8 * kExl3Wld + sg * 16, kExl3Wld);
        simdgroup_load(b1f, Ws + kk * 8 * kExl3Wld + sg * 16 + 8, kExl3Wld);
        for (int i = 0; i < 4; ++i) {
          simdgroup_half8x8 af;
          simdgroup_load(af, Xs + i * 8 * kExl3Xld + kt * 16 + kk * 8, kExl3Xld);
          simdgroup_multiply_accumulate(acc[i][0], af, b0f, acc[i][0]);
          simdgroup_multiply_accumulate(acc[i][1], af, b1f, acc[i][1]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// acc staged through Cs (the Xs space) and stored for the tile's live rows: row i to out + orow(i) * ld + n0
NI_INLINE void exl3_mm_store(thread simdgroup_float8x8 (&acc)[4][2], threadgroup float* Cs, uint sg, uint tid,
                             uint nr, device float* out, uint ld, uint n0, const device uint* rows, uint b0) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Cs + i * 8 * kExl3Bn + sg * 16 + j * 8, kExl3Bn);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = tid; t < kExl3Bm * kExl3Bn; t += 128) {
    const uint i = t / kExl3Bn, cc = t % kExl3Bn;
    if (i < nr) out[ulong(rows ? rows[b0 + i] : b0 + i) * ld + n0 + cc] = Cs[t];
  }
}

// Gate / up: grid (I / 64, max tiles, 2: gate / up) of 4 simdgroups; xh (half [T, D]) rows by token; hg / hu (fp32
// [sorted rows, I], before the output transform)
[[kernel]] void exl3_gu_mm(const device uint* gt [[buffer(0)]], const device half* gsuh [[buffer(1)]],
                           const device uint* ut [[buffer(2)]], const device half* usuh [[buffer(3)]],
                           const device half* xh [[buffer(4)]], const device uint* tiles [[buffer(5)]],
                           const device uint* ntiles [[buffer(6)]], const device uint* xrows [[buffer(7)]],
                           device float* hg [[buffer(8)]], device float* hu [[buffer(9)]],
                           constant Exl3Params& p [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  alignas(16) threadgroup half Xs[kExl3Bm * kExl3Xld];  // also the fp32 output staging (exl3_mm_store)
  threadgroup half Ws[16 * kExl3Wld];
  const uint e = tiles[tg.y * 3], b0 = tiles[tg.y * 3 + 1], nr = tiles[tg.y * 3 + 2] - b0;
  const bool up = tg.z != 0;
  const uint ot = p.I / 16, n = up ? p.nu : p.ng;
  const device uint* wbase = (up ? ut : gt) + ulong(e) * (p.D / 16) * ot * (n / 2);
  simdgroup_float8x8 acc[4][2];
  exl3_mm<true>(wbase, n, ot, p.D / 128, (up ? usuh : gsuh) + ulong(e) * p.D, xh, p.D, xrows, b0, nr, tg.x * 4,
                p.mask, Xs, Ws, acc, sg, lane);
  exl3_mm_store(acc, (threadgroup float*)Xs, sg, tid, nr, up ? hu : hg, p.I, tg.x * kExl3Bn, nullptr, b0);
}

// Down: grid (D / 64, max tiles) of 4 simdgroups; md (half [sorted rows, I], prepared) into yd by pair (fp32 [P, D],
// before the output transform)
[[kernel]] void exl3_dn_mm(const device uint* dt [[buffer(0)]], const device half* md [[buffer(1)]],
                           const device uint* tiles [[buffer(2)]], const device uint* ntiles [[buffer(3)]],
                           const device uint* yrows [[buffer(4)]], device float* yd [[buffer(5)]],
                           constant Exl3Params& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  alignas(16) threadgroup half Xs[kExl3Bm * kExl3Xld];  // also the fp32 output staging (exl3_mm_store)
  threadgroup half Ws[16 * kExl3Wld];
  const uint e = tiles[tg.y * 3], b0 = tiles[tg.y * 3 + 1], nr = tiles[tg.y * 3 + 2] - b0;
  const uint ot = p.D / 16;
  const device uint* wbase = dt + ulong(e) * (p.I / 16) * ot * (p.nd / 2);
  simdgroup_float8x8 acc[4][2];
  exl3_mm<false>(wbase, p.nd, ot, p.I / 128, nullptr, md, p.I, nullptr, b0, nr, tg.x * 4, p.mask, Xs, Ws, acc, sg,
                 lane);
  exl3_mm_store(acc, (threadgroup float*)Xs, sg, tid, nr, yd, p.D, tg.x * kExl3Bn, yrows, b0);
}

// The pairs' output transform in place: yd[q] = f16(svh_d * H128(yd[q])) with pair q's expert idx[q] (moe_combine then
// sums them). grid (D / 128, ceil(P / 4)) of 4 simdgroups, a pair each.
[[kernel]] void exl3_fin_pairs(device float* yd [[buffer(0)]], const device half* dsvh [[buffer(1)]],
                               const device uint* idx [[buffer(2)]], constant Exl3Params& p [[buffer(3)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint q = tg.y * 4 + sg;
  if (q >= p.rows) return;
  const uint c0 = tg.x * 128 + lane * 4;
  const ulong o = ulong(q) * p.D + c0, oe = ulong(idx[q]) * p.D + c0;
  float v[4];
  for (int m = 0; m < 4; ++m) v[m] = yd[o + m];
  exl3_h128(v, lane);
  for (int m = 0; m < 4; ++m) yd[o + m] = float(half(v[m] * float(dsvh[oe + m])));
}
