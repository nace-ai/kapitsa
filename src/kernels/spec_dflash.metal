#include "common.h"

// ---- DFlash2 block drafter ----
// One drafting pass = the block [anchor, mask x 7] at positions P..P+7 through NL qwen3-style
// layers with two-tap grouped dynamic convolutions, bidirectional attention over the committed
// context ring + the block itself, the target's LM head, per-position top-16 and the rank-256
// selector walk. Fixed geometry: 8 rows, 32 q heads / 8 kv heads x 128, window 2048, top-16.
// The file sorts after sample.metal in the embedded library (it uses pcg_hash and tg_argmax).
//
// 8-row GEMM operand ("X8 table"): Xh half [8][ld] row-major plus Xs float [K/64][8] = the sum of
// the 64 half-rounded inputs of each (group, row), which the fp32 epilogue of dfl_qmm8 needs
// for the affine bias term. Producers write both.

constant constexpr uint kDflKvHeads = 8;
constant constexpr uint kDflHeadDim = 128;
constant constexpr uint kDflRing = 2048;  // ring slots = window; slot = position mod 2048
constant constexpr uint kDflSaltWalk = 0x57A1C3E5u;  // uniform stream of the sampled walk (SPEC §8.2)

// The "T table" of the tensor-op GEMM qmm8t (src/kernels/tensor/verify8t.metal): the same 8 rows of halfs with each
// 64-group permuted to the matrix units' k-slots (input i -> slot 4 (i / 16) + 16 jb + kk, jb = 2 ((i % 16) / 8) +
// i % 2, kk = (i % 8) / 2), stored right after the row-major table (xh + 8 ld). Every X8 producer writes both.
NI_INLINE uint dfl_tslot(uint i) {
  const uint rem = i & 15u;
  return 4u * (i >> 4) + 16u * ((((rem >> 3) << 1)) | (rem & 1u)) + ((rem & 7u) >> 1);
}

// Group g (64 values) of row r: two values per lane -> half table (+ its T table) + exact sum of the halfs.
NI_INLINE void dfl_put2(device half* xh, device float* xs, uint r, uint ld, uint g, uint lane, float v0, float v1) {
  const half2 hv = half2(half(v0), half(v1));
  *(device half2*)(xh + r * ld + g * 64 + 2 * lane) = hv;
  device half* t = xh + (8 + r) * ld + g * 64;
  t[dfl_tslot(2 * lane)] = hv.x;
  t[dfl_tslot(2 * lane + 1)] = hv.y;
  const float s = simd_sum(float(hv.x) + float(hv.y));
  if (lane == 0) xs[g * 8 + r] = s;
}

// NeoX-halves RoPE of one 128-dim head held in threadgroup memory (SPEC §5): inv_freq computed
// on the host in fp32 like HF, angle = float(pos) * inv_freq in fp32, precise cos/sin.
NI_INLINE float dfl_rope(threadgroup const float* t, uint d, float pos, const device float* inv_freq) {
  const uint i = d & 63;
  const float ang = pos * inv_freq[i];
  const float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
  return d < 64 ? t[d] * c - t[d + 64] * s : t[d] * c + t[d - 64] * s;
}

// simdgroup 8x8 matrix lane layout (as MLX steel): lane holds row fm, columns fn and fn + 1.
NI_INLINE uint2 dfl_lane_rc(uint lane) {
  const uint q = lane / 4;
  return uint2((q & 4) + (lane / 2) % 4, (q & 2) * 2 + (lane % 2) * 2);
}

NI_INLINE float dfl_uniform(uint seed, uint pos, uint salt) {
  return float(pcg_hash(seed ^ pcg_hash(pos) ^ salt) & 0xffffffu) / 16777216.0f;
}

// GPU-driven drafting (double-buffered decode loop): a position parameter equal to kDflGpu means "read
// it from the drafter state gst", which the previous cycle's dfl_accept wrote: gst[0] = anchor
// position P of the next pass, gst[1] = rows m to commit to the ring, gst[2] = their first position.
// The host can then commit the next cycle's ring commit + drafting pass before it has read the
// accepted count of the current cycle.
constant constexpr uint kDflGpu = 0xffffffffu;
NI_INLINE uint dfl_pos(uint p, const device uint* gst) { return p == kDflGpu ? gst[0] : p; }
// split plan of the draft attention for a GPU-side P: context [lo, P), lo = max(P - 2047, lo_min),
// fixed `chunk` positions per split; the host plan (dfl_attn_plan) is used when P is host-known.
// chunk == 0: the smallest multiple of 32 that splits the context into at most `nsplit` (the grid's) splits.
NI_INLINE void dfl_attn_geom(const device uint* gst, thread uint& P, thread uint& lo, thread uint& chunk, thread uint& nsplit) {
  if (P != kDflGpu) return;
  P = gst[0];
  lo = min(max(P > 2047u ? P - 2047u : 0u, lo), P);
  if (chunk == 0) chunk = max(32u, ((P - lo + nsplit - 1) / nsplit + 31) / 32 * 32);
  nsplit = max(1u, (P - lo + chunk - 1) / chunk);
}

// ---- target capture (SPEC §6.1): fp32 residual rows -> bf16 columns [col0, col0 + H) ----
struct DflCaptureParams {
  uint H, ld, col0, dst_row0;
};
// grid: (ceil(H / 1024), rows) x 256
[[kernel]] void dfl_capture(const device float* x [[buffer(0)]], device bfloat* dst [[buffer(1)]],
                            constant DflCaptureParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x * 4, r = gid.y;
  if (c >= p.H) return;
  const float4 v = *(const device float4*)(x + size_t(r) * p.H + c);
  *(device bfloat4*)(dst + size_t(p.dst_row0 + r) * p.ld + p.col0 + c) = bfloat4(v);
}

// ---- row kernels (one threadgroup of 256 per block row) ----
struct DflEmbedParams {
  uint H, gs, tok_off, mask;
  float eps;
};
// h[r] = E[t_r] (t_0 = tokens[tok_off], t_1..7 = mask), n = RMSNorm(h; w) -> n (fp32) and X8 table.
[[kernel]] void dfl_embed_norm(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],
                               const device bfloat* B [[buffer(2)]], const device int* tokens [[buffer(3)]],
                               device float* h [[buffer(4)]], const device bfloat* nw [[buffer(5)]],
                               device float* n [[buffer(6)]], device half* xh [[buffer(7)]],
                               device float* xs [[buffer(8)]], constant DflEmbedParams& p [[buffer(9)]],
                               const device uint* gst [[buffer(10)]],
                               uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  const uint H = p.H, gpr = H / p.gs;
  const uint tok = r == 0 ? uint(tokens[dfl_pos(p.tok_off, gst)]) : p.mask;
  device float* hr = h + r * H;
  float ss = 0.0f;
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint k = g * 64 + 2 * lane;
    const uint word = W[size_t(tok) * (H / 8) + k / 8], sh = 4 * (k % 8);
    const float sc = float(S[size_t(tok) * gpr + k / p.gs]), bi = float(B[size_t(tok) * gpr + k / p.gs]);
    const float2 v = float2(sc * float((word >> sh) & 15u) + bi, sc * float((word >> (sh + 4)) & 15u) + bi);
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

struct DflNormParams {
  uint H;
  float eps;
  float xscale;   // x is multiplied by this first (exact power of two)
  uint write_n;   // also write the normalized rows (fp32) to n
};
// n = RMSNorm(xscale * x[r]; w) -> X8 table (and fp32 n).
[[kernel]] void dfl_norm_x8(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                            device float* n [[buffer(2)]], device half* xh [[buffer(3)]], device float* xs [[buffer(4)]],
                            constant DflNormParams& p [[buffer(5)]], uint r [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  const uint H = p.H;
  const device float* xr = x + r * H;
  float ss = 0.0f;
  for (uint g = sg; g < H / 64; g += nsg) {
    const float2 v = *(const device float2*)(xr + g * 64 + 2 * lane) * p.xscale;
    ss += v.x * v.x + v.y * v.y;
  }
  ss = tg_sum(ss, red, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(H) + p.eps);
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint k = g * 64 + 2 * lane;
    const float2 v = *(const device float2*)(xr + k) * p.xscale;
    const float2 o = float2(v.x * inv * float(w[k]), v.y * inv * float(w[k + 1]));
    if (p.write_n) *(device float2*)(n + r * H + k) = o;
    dfl_put2(xh, xs, r, H, g, lane, o.x, o.y);
  }
}

// DFlash (v1) drafters, plain residual sublayers: h[r] += u[r]; n = RMSNorm(h[r]; w) -> fp32 n and the X8 table
// (dfl_conv_fin_norm without the block conv). grid: 8 threadgroups x 256.
[[kernel]] void dfl_res_norm(device float* h [[buffer(0)]], const device float* u [[buffer(1)]],
                             const device bfloat* w [[buffer(2)]], device float* n [[buffer(3)]],
                             device half* xh [[buffer(4)]], device float* xs [[buffer(5)]],
                             constant DflNormParams& p [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  const uint H = p.H;
  device float* hr = h + r * H;
  float ss = 0.0f;
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint c = g * 64 + 2 * lane;
    const float2 v = *(device float2*)(hr + c) + *(const device float2*)(u + r * H + c);
    *(device float2*)(hr + c) = v;
    ss += v.x * v.x + v.y * v.y;
  }
  ss = tg_sum(ss, red, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(H) + p.eps);
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint c = g * 64 + 2 * lane;
    const float2 v = *(device float2*)(hr + c);
    const float2 o = float2(v.x * inv * float(w[c]), v.y * inv * float(w[c + 1]));
    *(device float2*)(n + r * H + c) = o;
    dfl_put2(xh, xs, r, H, g, lane, o.x, o.y);
  }
}

// Two-tap grouped dynamic convolution (SPEC §4.1). D [8][4G] = n . P^T, columns
// [stage][tap][group]; base bf16 [stage][tap][H]; tap 0 = current row, tap 1 = previous row (zero
// for row 0); the coefficients come from the current row.
struct DflConvParams {
  uint H, G;      // hidden, conv groups (H / 16)
  float eps;
};
// Prepare stage: a[r] = (b00 + D[r,0,g]) x[r] + (b01 + D[r,1,g]) x[r-1] -> X8 table.
[[kernel]] void dfl_conv_prep(const device float* x [[buffer(0)]], const device float* D [[buffer(1)]],
                              const device bfloat* base [[buffer(2)]], device half* xh [[buffer(3)]],
                              device float* xs [[buffer(4)]], constant DflConvParams& p [[buffer(5)]],
                              uint r [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint H = p.H, G = p.G;
  const device float* Dr = D + r * 4 * G;
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint c = g * 64 + 2 * lane, cg = c / 16;
    const float d0 = Dr[cg], d1 = Dr[G + cg];
    const float2 xc = *(const device float2*)(x + r * H + c);
    const float2 xp = r > 0 ? *(const device float2*)(x + (r - 1) * H + c) : float2(0.0f);
    const float2 b0 = float2(base[c], base[c + 1]), b1 = float2(base[H + c], base[H + c + 1]);
    const float2 a = (b0 + d0) * xc + (b1 + d1) * xp;
    dfl_put2(xh, xs, r, H, g, lane, a.x, a.y);
  }
}

// Finish stage + residual, then the next RMSNorm:
//   h[r] += (b10 + D[r,2,g]) u[r] + (b11 + D[r,3,g]) u[r-1];  n = RMSNorm(h[r]; w) -> n, X8 table.
[[kernel]] void dfl_conv_fin_norm(device float* h [[buffer(0)]], const device float* u [[buffer(1)]],
                                  const device float* D [[buffer(2)]], const device bfloat* base [[buffer(3)]],
                                  const device bfloat* w [[buffer(4)]], device float* n [[buffer(5)]],
                                  device half* xh [[buffer(6)]], device float* xs [[buffer(7)]],
                                  constant DflConvParams& p [[buffer(8)]], uint r [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  const uint H = p.H, G = p.G;
  const device float* Dr = D + r * 4 * G;
  device float* hr = h + r * H;
  float ss = 0.0f;
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint c = g * 64 + 2 * lane, cg = c / 16;
    const float d0 = Dr[2 * G + cg], d1 = Dr[3 * G + cg];
    const float2 uc = *(const device float2*)(u + r * H + c);
    const float2 up = r > 0 ? *(const device float2*)(u + (r - 1) * H + c) : float2(0.0f);
    const float2 b0 = float2(base[2 * H + c], base[2 * H + c + 1]), b1 = float2(base[3 * H + c], base[3 * H + c + 1]);
    const float2 v = *(device float2*)(hr + c) + ((b0 + d0) * uc + (b1 + d1) * up);
    *(device float2*)(hr + c) = v;
    ss += v.x * v.x + v.y * v.y;
  }
  ss = tg_sum(ss, red, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(H) + p.eps);
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint c = g * 64 + 2 * lane;
    const float2 v = *(device float2*)(hr + c);
    const float2 o = float2(v.x * inv * float(w[c]), v.y * inv * float(w[c + 1]));
    *(device float2*)(n + r * H + c) = o;
    dfl_put2(xh, xs, r, H, g, lane, o.x, o.y);
  }
}

// ---- attention ----
// cos/sin of the 8 block positions P + r for the 64 frequencies (shared by every draft layer):
// cs[r * 64 + i] = (cos, sin)(float(P + r) * inv_freq[i]). grid 1 x 512.
[[kernel]] void dfl_rope_table(const device float* inv_freq [[buffer(0)]], device float2* cs [[buffer(1)]],
                               constant uint& P [[buffer(2)]], const device uint* gst [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= 512) return;
  const float ang = float(dfl_pos(P, gst) + i / 64) * inv_freq[i % 64];
  cs[i] = float2(metal::precise::cos(ang), metal::precise::sin(ang));
}

struct DflQkParams {
  uint P;        // position of block row 0
  float eps;
  uint q_ld, kv_ld;
};
// q/k per-head RMSNorm + RoPE at P + r (table from dfl_rope_table); v copied.
// grid: (32 + 8 heads, 8 rows) x 128. Q [32][8][128], Kb/Vb [8][8][128] (head-major, fp32).
[[kernel]] void dfl_qk_prep(const device float* q [[buffer(0)]], const device float* kv [[buffer(1)]],
                            const device bfloat* qn [[buffer(2)]], const device bfloat* kn [[buffer(3)]],
                            const device float2* cs [[buffer(4)]], device float* Q [[buffer(5)]],
                            device float* Kb [[buffer(6)]], device float* Vb [[buffer(7)]],
                            constant DflQkParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint d [[thread_index_in_threadgroup]]) {
  threadgroup float red[32];
  threadgroup float t[128];
  const uint hh = tg.x, r = tg.y;
  const bool isq = hh < 32;
  const uint head = isq ? hh : hh - 32;
  const float x = isq ? q[r * p.q_ld + head * 128 + d] : kv[r * p.kv_ld + head * 128 + d];
  const float ss = tg_sum(x * x, red, d, 128);
  t[d] = x * metal::precise::rsqrt(ss / 128.0f + p.eps) * float((isq ? qn : kn)[d]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float2 c = cs[r * 64 + (d & 63)];
  const float o = d < 64 ? t[d] * c.x - t[d + 64] * c.y : t[d] * c.x + t[d - 64] * c.y;
  if (isq) {
    Q[(head * 8 + r) * 128 + d] = o;
  } else {
    Kb[(head * 8 + r) * 128 + d] = o;
    Vb[(head * 8 + r) * 128 + d] = kv[r * p.kv_ld + 1024 + head * 128 + d];
  }
}

struct DflAttnParams {
  uint P;       // block row 0 position; the context is positions [lo, P)
  uint lo;      // max(0, P - 2047): first position any block row can see
  uint chunk;   // context positions per split (multiple of 16)
  uint nsplit;
  float scale;
};

// Draft attention, one split of the context of one kv head (SPEC §5): 4 simdgroups = the 4 q heads
// sharing the kv head, 8 block rows each. Context key j is visible to row r iff j >= P + r - 2047
// (a per-row window, equal to REF/MLX's query - key < 2048); the 8 block keys
// are visible to every row (is_causal = false) and are added by the last split. fp32 scores and
// online softmax; partials are combined by dfl_attn_reduce in split order.
// Tiles of 16 keys: lane (rg, kl) scores rows 4rg..4rg+3 against key kl; for P.V each lane owns
// 4 of the 128 dims of all 8 rows.
[[kernel]] void dfl_attn(const device float* Q [[buffer(0)]], const device bfloat* ringK [[buffer(1)]],
                         const device bfloat* ringV [[buffer(2)]], const device float* Kb [[buffer(3)]],
                         const device float* Vb [[buffer(4)]], device float* part_o [[buffer(5)]],
                         device float* part_ml [[buffer(6)]], constant DflAttnParams& p [[buffer(7)]], const device uint* gst [[buffer(8)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint TK = 16, KLD = 129;
  threadgroup float qs[4 * 8 * 128];
  threadgroup float ks[TK * KLD];
  threadgroup float ps[4 * 8 * TK];
  const uint split = tg.x, g = tg.y, h = g * 4 + sg;
  uint P = p.P, lo = p.lo, chunk = p.chunk, nsplit = p.nsplit;
  dfl_attn_geom(gst, P, lo, chunk, nsplit);
  if (split >= nsplit) return;
  for (uint i = tid; i < 4 * 8 * 128; i += 128) qs[i] = Q[g * 4 * 1024 + i];
  const device bfloat* Kr = ringK + size_t(g) * kDflRing * 128;
  const device bfloat* Vr = ringV + size_t(g) * kDflRing * 128;
  const uint j_begin = lo + split * chunk;
  const uint j_end = min(j_begin + chunk, P);
  const uint nctx = j_end > j_begin ? (j_end - j_begin + TK - 1) / TK : 0;
  const uint ntiles = nctx + (split + 1 == nsplit ? 1u : 0u);
  const uint rg = lane / 16, kl = lane % 16;
  float M[8], L[8];
  float4 O[8];
  for (int r = 0; r < 8; ++r) {
    M[r] = -INFINITY;
    L[r] = 0.0f;
    O[r] = float4(0.0f);
  }
  const threadgroup float* qh = qs + sg * 1024;
  threadgroup float* pw = ps + sg * 8 * TK;
  for (uint t = 0; t < ntiles; ++t) {
    const bool blk = t == nctx;
    const uint j0 = j_begin + t * TK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const uint key = tid / 8, d0 = (tid % 8) * 16;
      threadgroup float* kd = ks + key * KLD + d0;
      if (blk) {
        for (int i = 0; i < 4; ++i) {
          const float4 v = key < 8 ? *(const device float4*)(Kb + (g * 8 + key) * 128 + d0 + 4 * i) : float4(0.0f);
          kd[4 * i + 0] = v.x; kd[4 * i + 1] = v.y; kd[4 * i + 2] = v.z; kd[4 * i + 3] = v.w;
        }
      } else {
        const uint j = j0 + key;
        for (int i = 0; i < 4; ++i) {
          const float4 v = j < j_end ? float4(*(const device bfloat4*)(Kr + (j & (kDflRing - 1)) * 128 + d0 + 4 * i))
                                     : float4(0.0f);
          kd[4 * i + 0] = v.x; kd[4 * i + 1] = v.y; kd[4 * i + 2] = v.z; kd[4 * i + 3] = v.w;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const threadgroup float* kk = ks + kl * KLD;
    const threadgroup float* q0 = qh + (4 * rg) * 128;
    for (uint d = 0; d < 128; d += 4) {
      const float4 kv = float4(kk[d], kk[d + 1], kk[d + 2], kk[d + 3]);
      for (int i = 0; i < 4; ++i) sc[i] += dot(kv, *(const threadgroup float4*)(q0 + i * 128 + d));
    }
    const uint j = j0 + kl;
    const bool kvalid = blk ? kl < 8 : j < j_end;
    for (int i = 0; i < 4; ++i) {
      const uint r = 4 * rg + i;
      const bool vis = kvalid && (blk || j + (kDflRing - 1) >= P + r);
      sc[i] = vis ? sc[i] * p.scale : -INFINITY;
    }
    float tm[4], tmo[4];
    for (int i = 0; i < 4; ++i) {
      float v = sc[i];
      v = max(v, simd_shuffle_xor(v, 8));
      v = max(v, simd_shuffle_xor(v, 4));
      v = max(v, simd_shuffle_xor(v, 2));
      v = max(v, simd_shuffle_xor(v, 1));
      tm[i] = v;
      tmo[i] = simd_shuffle_xor(v, 16);
    }
    float mnew[8], alpha[8];
    for (int r = 0; r < 8; ++r) {
      const uint i = uint(r) & 3u;
      const bool own = (uint(r) >> 2) == rg;
      const float tmax = own ? tm[i] : tmo[i];
      mnew[r] = max(M[r], tmax);
      alpha[r] = mnew[r] == -INFINITY ? 1.0f : exp(M[r] - mnew[r]);
    }
    float pe[4];
    for (int i = 0; i < 4; ++i) {
      const float mr = rg ? mnew[4 + i] : mnew[i];
      pe[i] = sc[i] == -INFINITY ? 0.0f : exp(sc[i] - mr);
      pw[(4 * rg + i) * TK + kl] = pe[i];
    }
    float ts[4], tso[4];
    for (int i = 0; i < 4; ++i) {
      float v = pe[i];
      v += simd_shuffle_xor(v, 8);
      v += simd_shuffle_xor(v, 4);
      v += simd_shuffle_xor(v, 2);
      v += simd_shuffle_xor(v, 1);
      ts[i] = v;
      tso[i] = simd_shuffle_xor(v, 16);
    }
    for (int r = 0; r < 8; ++r) {
      const uint i = uint(r) & 3u;
      const bool own = (uint(r) >> 2) == rg;
      L[r] = L[r] * alpha[r] + (own ? ts[i] : tso[i]);
      M[r] = mnew[r];
      O[r] *= alpha[r];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint nk = blk ? 8u : min(TK, j_end - j0);
    for (uint kq = 0; kq < nk; ++kq) {
      const float4 v = blk ? *(const device float4*)(Vb + (g * 8 + kq) * 128 + 4 * lane)
                           : float4(*(const device bfloat4*)(Vr + ((j0 + kq) & (kDflRing - 1)) * 128 + 4 * lane));
      for (int r = 0; r < 8; ++r) O[r] += pw[r * TK + kq] * v;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int r = 0; r < 8; ++r) {
    const size_t base = (size_t(split) * 32 + h) * 8 + r;
    *(device float4*)(part_o + base * 128 + 4 * lane) = O[r];
    if (lane == 0) {
      part_ml[base * 2] = M[r];
      part_ml[base * 2 + 1] = L[r];
    }
  }
}

// Same contract as dfl_attn with simdgroup-matrix tiles (NI_DFLASH_ATTN=mma): per simdgroup one q
// head x the 8 block rows; Q kept as 16 half 8x8 fragments, tiles of 32 keys staged as half in
// threadgroup memory; S = Q K^T and O += P V with fp32 accumulation; online softmax on the S
// fragments (a lane holds rows fm, keys fn/fn+1: row reductions are xor-shuffles over lane bits 0, 3).
[[kernel]] void dfl_attn_mma(const device float* Q [[buffer(0)]], const device bfloat* ringK [[buffer(1)]],
                             const device bfloat* ringV [[buffer(2)]], const device float* Kb [[buffer(3)]],
                             const device float* Vb [[buffer(4)]], device float* part_o [[buffer(5)]],
                             device float* part_ml [[buffer(6)]], constant DflAttnParams& p [[buffer(7)]], const device uint* gst [[buffer(8)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint TK = 32, LD = 136;
  threadgroup half ks[TK * LD];
  threadgroup half vs[TK * LD];
  const uint split = tg.x, g = tg.y, h = g * 4 + sg;
  uint P = p.P, lo = p.lo, chunk = p.chunk, nsplit = p.nsplit;
  dfl_attn_geom(gst, P, lo, chunk, nsplit);
  if (split >= nsplit) return;
  const uint2 rc = dfl_lane_rc(lane);
  const uint fm = rc.x, fn = rc.y;
  // Q fragments: rows = block rows, cols = 8 dims of chunk c
  simdgroup_half8x8 qf[16];
  {
    const device float* qr = Q + (size_t(h) * 8 + fm) * 128 + fn;
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 16; ++c) {
      qf[c].thread_elements()[0] = half(qr[c * 8] * p.scale);
      qf[c].thread_elements()[1] = half(qr[c * 8 + 1] * p.scale);
    }
  }
  const device bfloat* Kr = ringK + size_t(g) * kDflRing * 128;
  const device bfloat* Vr = ringV + size_t(g) * kDflRing * 128;
  const uint j_begin = lo + split * chunk;
  const uint j_end = min(j_begin + chunk, P);
  const uint nctx = j_end > j_begin ? (j_end - j_begin + TK - 1) / TK : 0;
  const uint ntiles = nctx + (split + 1 == nsplit ? 1u : 0u);
  simdgroup_float8x8 of[16];
  _Pragma("clang loop unroll(full)")
  for (int c = 0; c < 16; ++c) of[c] = simdgroup_float8x8(0);
  float M = -INFINITY, L = 0.0f;  // row fm (replicated over the lanes holding row fm)
  for (uint t = 0; t < ntiles; ++t) {
    const bool blk = t == nctx;
    const uint j0 = j_begin + t * TK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {  // stage 32 keys x 128 of K and V (each thread: one key, 32 dims)
      const uint key = tid / 4, d0 = (tid % 4) * 32;
      threadgroup half* kd = ks + key * LD + d0;
      threadgroup half* vd = vs + key * LD + d0;
      if (blk) {
        for (int i = 0; i < 8; ++i) {
          const float4 kv = key < 8 ? *(const device float4*)(Kb + (g * 8 + key) * 128 + d0 + 4 * i) : float4(0.0f);
          const float4 vv = key < 8 ? *(const device float4*)(Vb + (g * 8 + key) * 128 + d0 + 4 * i) : float4(0.0f);
          *(threadgroup half4*)(kd + 4 * i) = half4(kv);
          *(threadgroup half4*)(vd + 4 * i) = half4(vv);
        }
      } else {
        const uint j = j0 + key;
        const bool ok = j < j_end;
        const size_t off = size_t(j & (kDflRing - 1)) * 128 + d0;
        for (int i = 0; i < 8; ++i) {
          const float4 kv = ok ? float4(*(const device bfloat4*)(Kr + off + 4 * i)) : float4(0.0f);
          const float4 vv = ok ? float4(*(const device bfloat4*)(Vr + off + 4 * i)) : float4(0.0f);
          *(threadgroup half4*)(kd + 4 * i) = half4(kv);
          *(threadgroup half4*)(vd + 4 * i) = half4(vv);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 sf[4];
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb) {
      sf[kb] = simdgroup_float8x8(0);
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < 16; ++c) {
        simdgroup_half8x8 kt;
        simdgroup_load(kt, ks + kb * 8 * LD + c * 8, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[kb], qf[c], kt, sf[kb]);
      }
    }
    // mask + row max over the 32 keys
    float sv[8];
    float tmax = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint kk = kb * 8 + fn + e, j = j0 + kk;
        const bool vis = blk ? kk < 8 : (j < j_end && j + (kDflRing - 1) >= P + fm);
        const float v = vis ? sf[kb].thread_elements()[e] : -INFINITY;
        sv[kb * 2 + e] = v;
        tmax = max(tmax, v);
      }
    tmax = max(tmax, simd_shuffle_xor(tmax, 1));
    tmax = max(tmax, simd_shuffle_xor(tmax, 8));
    const float mnew = max(M, tmax);
    const float alpha = mnew == -INFINITY ? 1.0f : exp(M - mnew);
    float tsum = 0.0f;
    simdgroup_half8x8 pf[4];
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const float v = sv[kb * 2 + e];
        const float pe = v == -INFINITY ? 0.0f : exp(v - mnew);
        tsum += pe;
        pf[kb].thread_elements()[e] = half(pe);
      }
    tsum += simd_shuffle_xor(tsum, 1);
    tsum += simd_shuffle_xor(tsum, 8);
    L = L * alpha + tsum;
    M = mnew;
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 16; ++c) {
      of[c].thread_elements()[0] *= alpha;
      of[c].thread_elements()[1] *= alpha;
    }
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb)
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < 16; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, vs + kb * 8 * LD + c * 8, LD);
        simdgroup_multiply_accumulate(of[c], pf[kb], vf, of[c]);
      }
  }
  const size_t base = (size_t(split) * 32 + h) * 8 + fm;
  _Pragma("clang loop unroll(full)")
  for (int c = 0; c < 16; ++c)
    *(device float2*)(part_o + base * 128 + c * 8 + fn) =
        float2(of[c].thread_elements()[0], of[c].thread_elements()[1]);
  if (fn == 0) {
    part_ml[base * 2] = M;
    part_ml[base * 2 + 1] = L;
  }
}

// dfl_qk_prep fused into dfl_attn_mma (one dispatch fewer per layer): every threadgroup normalizes and rotates its
// 4 q heads' rows itself (lane-local: a lane's 32 elements of row fm hold both RoPE halves, the row sum of squares
// is an xor-shuffle over lane bits 0, 3), and the split holding the block tile its kv head's 8 block keys (via
// threadgroup memory). Same math as dfl_qk_prep + dfl_attn_mma; q / kv are the projections' fp32 rows.
struct DflAttnQkParams {
  uint P, lo, chunk, nsplit;  // as DflAttnParams (chunk 0: dfl_attn_geom's split plan)
  float scale, eps;
  uint q_ld, kv_ld;
};
[[kernel]] void dfl_attn_qk(const device float* q [[buffer(0)]], const device float* kv [[buffer(1)]],
                            const device bfloat* qn [[buffer(2)]], const device bfloat* kn [[buffer(3)]],
                            const device float2* cs [[buffer(4)]], const device bfloat* ringK [[buffer(5)]],
                            const device bfloat* ringV [[buffer(6)]], device float* part_o [[buffer(7)]],
                            device float* part_ml [[buffer(8)]], constant DflAttnQkParams& p [[buffer(9)]],
                            const device uint* gst [[buffer(10)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint TK = 32, LD = 136;
  threadgroup half ks[TK * LD];
  threadgroup half vs[TK * LD];
  threadgroup float kf[8 * 128];  // the block keys after k_norm (before RoPE)
  const uint split = tg.x, g = tg.y, h = g * 4 + sg;
  uint P = p.P, lo = p.lo, chunk = p.chunk, nsplit = p.nsplit;
  dfl_attn_geom(gst, P, lo, chunk, nsplit);
  if (split >= nsplit) return;
  const bool last = split + 1 == nsplit;
  const uint2 rc = dfl_lane_rc(lane);
  const uint fm = rc.x, fn = rc.y;
  simdgroup_half8x8 qf[16];
  {  // Q = RoPE(RMSNorm_head(q; q_norm)) * scale, row fm (= block row) of head h, dims 8c + fn + e
    const device float* qr = q + fm * p.q_ld + h * 128 + fn;
    float2 v[16];
    float ss = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 16; ++c) {
      v[c] = *(const device float2*)(qr + c * 8);
      ss += v[c].x * v[c].x + v[c].y * v[c].y;
    }
    ss += simd_shuffle_xor(ss, 1);
    ss += simd_shuffle_xor(ss, 8);
    const float inv = metal::precise::rsqrt(ss / 128.0f + p.eps);
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 16; ++c) {
      v[c].x *= inv * float(qn[c * 8 + fn]);
      v[c].y *= inv * float(qn[c * 8 + fn + 1]);
    }
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 8; ++c) {
      const float4 t = *(const device float4*)(cs + fm * 64 + c * 8 + fn);  // (cos, sin) of dims d, d + 1
      const float2 a = v[c], b = v[c + 8];
      qf[c].thread_elements()[0] = half((a.x * t.x - b.x * t.y) * p.scale);
      qf[c].thread_elements()[1] = half((a.y * t.z - b.y * t.w) * p.scale);
      qf[c + 8].thread_elements()[0] = half((b.x * t.x + a.x * t.y) * p.scale);
      qf[c + 8].thread_elements()[1] = half((b.y * t.z + a.y * t.w) * p.scale);
    }
  }
  if (last) {  // block key `key` (thread: 8 dims d0..d0+7): k_norm into kf
    const uint key = tid / 16, d0 = (tid % 16) * 8;
    const device float* kr = kv + key * p.kv_ld + g * 128 + d0;
    const float4 a = *(const device float4*)kr, b = *(const device float4*)(kr + 4);
    float ss = dot(a, a) + dot(b, b);
    ss += simd_shuffle_xor(ss, 1);
    ss += simd_shuffle_xor(ss, 2);
    ss += simd_shuffle_xor(ss, 4);
    ss += simd_shuffle_xor(ss, 8);
    const float inv = metal::precise::rsqrt(ss / 128.0f + p.eps);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i) {
      kf[key * 128 + d0 + i] = a[i] * inv * float(kn[d0 + i]);
      kf[key * 128 + d0 + 4 + i] = b[i] * inv * float(kn[d0 + 4 + i]);
    }
  }
  const device bfloat* Kr = ringK + size_t(g) * kDflRing * 128;
  const device bfloat* Vr = ringV + size_t(g) * kDflRing * 128;
  const uint j_begin = lo + split * chunk;
  const uint j_end = min(j_begin + chunk, P);
  const uint nctx = j_end > j_begin ? (j_end - j_begin + TK - 1) / TK : 0;
  const uint ntiles = nctx + (last ? 1u : 0u);
  simdgroup_float8x8 of[16];
  _Pragma("clang loop unroll(full)")
  for (int c = 0; c < 16; ++c) of[c] = simdgroup_float8x8(0);
  float M = -INFINITY, L = 0.0f;
  for (uint t = 0; t < ntiles; ++t) {
    const bool blk = t == nctx;
    const uint j0 = j_begin + t * TK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {  // stage 32 keys x 128 of K and V (each thread: one key, 32 dims)
      const uint key = tid / 4, d0 = (tid % 4) * 32;
      threadgroup half* kd = ks + key * LD + d0;
      threadgroup half* vd = vs + key * LD + d0;
      if (blk) {
        for (int i = 0; i < 32; ++i) {
          float kk = 0.0f, vv = 0.0f;
          if (key < 8) {
            const uint d = d0 + i, j = d & 63;
            const float2 c = cs[key * 64 + j];
            const threadgroup float* kr = kf + key * 128;
            kk = d < 64 ? kr[d] * c.x - kr[d + 64] * c.y : kr[d] * c.x + kr[d - 64] * c.y;
            vv = kv[key * p.kv_ld + 1024 + g * 128 + d];
          }
          kd[i] = half(kk);
          vd[i] = half(vv);
        }
      } else {
        const uint j = j0 + key;
        const bool ok = j < j_end;
        const size_t off = size_t(j & (kDflRing - 1)) * 128 + d0;
        for (int i = 0; i < 8; ++i) {
          const float4 kv4 = ok ? float4(*(const device bfloat4*)(Kr + off + 4 * i)) : float4(0.0f);
          const float4 vv4 = ok ? float4(*(const device bfloat4*)(Vr + off + 4 * i)) : float4(0.0f);
          *(threadgroup half4*)(kd + 4 * i) = half4(kv4);
          *(threadgroup half4*)(vd + 4 * i) = half4(vv4);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 sf[4];
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb) {
      sf[kb] = simdgroup_float8x8(0);
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < 16; ++c) {
        simdgroup_half8x8 kt;
        simdgroup_load(kt, ks + kb * 8 * LD + c * 8, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[kb], qf[c], kt, sf[kb]);
      }
    }
    float sv[8];
    float tmax = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint kk = kb * 8 + fn + e, j = j0 + kk;
        const bool vis = blk ? kk < 8 : (j < j_end && j + (kDflRing - 1) >= P + fm);
        const float v = vis ? sf[kb].thread_elements()[e] : -INFINITY;
        sv[kb * 2 + e] = v;
        tmax = max(tmax, v);
      }
    tmax = max(tmax, simd_shuffle_xor(tmax, 1));
    tmax = max(tmax, simd_shuffle_xor(tmax, 8));
    const float mnew = max(M, tmax);
    const float alpha = mnew == -INFINITY ? 1.0f : exp(M - mnew);
    float tsum = 0.0f;
    simdgroup_half8x8 pf[4];
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const float v = sv[kb * 2 + e];
        const float pe = v == -INFINITY ? 0.0f : exp(v - mnew);
        tsum += pe;
        pf[kb].thread_elements()[e] = half(pe);
      }
    tsum += simd_shuffle_xor(tsum, 1);
    tsum += simd_shuffle_xor(tsum, 8);
    L = L * alpha + tsum;
    M = mnew;
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 16; ++c) {
      of[c].thread_elements()[0] *= alpha;
      of[c].thread_elements()[1] *= alpha;
    }
    _Pragma("clang loop unroll(full)")
    for (int kb = 0; kb < 4; ++kb)
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < 16; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, vs + kb * 8 * LD + c * 8, LD);
        simdgroup_multiply_accumulate(of[c], pf[kb], vf, of[c]);
      }
  }
  const size_t base = (size_t(split) * 32 + h) * 8 + fm;
  _Pragma("clang loop unroll(full)")
  for (int c = 0; c < 16; ++c)
    *(device float2*)(part_o + base * 128 + c * 8 + fn) =
        float2(of[c].thread_elements()[0], of[c].thread_elements()[1]);
  if (fn == 0) {
    part_ml[base * 2] = M;
    part_ml[base * 2 + 1] = L;
  }
}

struct DflAttnRedParams {
  uint nsplit;
  uint P, lo, chunk;  // P == kDflGpu: the split plan of dfl_attn_geom (nsplit derived from gst)
};
// Combines the splits in order; writes o as the o_proj X8 table (K = 4096). grid (32, 8) x 128.
[[kernel]] void dfl_attn_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                device half* xh [[buffer(2)]], device float* xs [[buffer(3)]],
                                constant DflAttnRedParams& p [[buffer(4)]], const device uint* gst [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint d [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[4];
  const uint h = tg.x, r = tg.y;
  uint P = p.P, lo = p.lo, chunk = p.chunk, nsplit = p.nsplit;
  dfl_attn_geom(gst, P, lo, chunk, nsplit);
  float mx = -INFINITY;
  for (uint s = 0; s < nsplit; ++s) mx = max(mx, part_ml[((s * 32 + h) * 8 + r) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint s = 0; s < nsplit; ++s) {
    const size_t base = (size_t(s) * 32 + h) * 8 + r;
    const float m = part_ml[base * 2];
    if (m == -INFINITY) continue;
    const float w = exp(m - mx);
    num += w * part_o[base * 128 + d];
    den += w * part_ml[base * 2 + 1];
  }
  const half o = half(num / den);
  xh[r * 4096 + h * 128 + d] = o;
  xh[(8 + r) * 4096 + h * 128 + (d & 64u) + dfl_tslot(d & 63u)] = o;  // T table
  const float sm = simd_sum(float(o));
  if (lane == 0) red[sg] = sm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (d == 0) {
    xs[(2 * h) * 8 + r] = red[0] + red[1];
    xs[(2 * h + 1) * 8 + r] = red[2] + red[3];
  }
}

// ---- context commit (SPEC §6.2) ----
struct DflCommitParams {
  uint rows, pos0, kv_ld;
  float eps;
};
// kv [rows][kv_ld]: draft layer l's k at l*2048 + h*128, v at l*2048 + 1024 + h*128 (c . [W_k|W_v]^T).
// K = RoPE(RMSNorm_head(k; k_norm_l), pos), V = v -> ring slot pos mod 2048 (bf16).
// grid: (rows, NL * 8) x 128.
[[kernel]] void dfl_ctx_commit(const device float* kv [[buffer(0)]], const device bfloat* kn [[buffer(1)]],
                               const device float* inv_freq [[buffer(2)]], device bfloat* ringK [[buffer(3)]],
                               device bfloat* ringV [[buffer(4)]], constant DflCommitParams& p [[buffer(5)]],
                               const device uint* gst [[buffer(6)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  threadgroup float red[32];
  threadgroup float t[128];
  const uint r = tg.x, l = tg.y / kDflKvHeads, hh = tg.y % kDflKvHeads;
  const bool gpu = p.rows == kDflGpu;  // rows / positions from the drafter state (grid: 8 rows)
  if (r >= (gpu ? gst[1] : p.rows)) return;
  const device float* src = kv + size_t(r) * p.kv_ld + l * 2048 + hh * 128;
  const float x = src[d];
  const float ss = tg_sum(x * x, red, d, 128);
  t[d] = x * metal::precise::rsqrt(ss / 128.0f + p.eps) * float(kn[l * 128 + d]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint pos = (gpu ? gst[2] : p.pos0) + r;
  const float o = dfl_rope(t, d, float(pos), inv_freq);
  const size_t dst = ((size_t(l) * kDflKvHeads + hh) * kDflRing + (pos & (kDflRing - 1))) * 128 + d;
  ringK[dst] = bfloat(o);
  ringV[dst] = bfloat(src[1024 + d]);
}

// Captured rows (bf16, <= 8) -> X8 table scaled by `scale` (an exact power of two keeps the
// residual stream's outliers inside half range; the GEMM epilogue multiplies back).
struct DflX8Params {
  uint K, m, src_ld;
  float scale;
  uint gather;  // 1: row r is source row gst[4 + r] (a tree cycle's accepted path, kf_accept)
};
// grid (8 rows, column blocks) x 256: block y converts groups y * 8 + sg, then every 8 * blocks.
[[kernel]] void dfl_x8_from_bf16(const device bfloat* src [[buffer(0)]], device half* xh [[buffer(1)]],
                                 device float* xs [[buffer(2)]], constant DflX8Params& p [[buffer(3)]],
                                 const device uint* gst [[buffer(4)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint m = p.m == kDflGpu ? gst[1] : p.m;
  const uint r = tg.x;
  const uint rs = p.gather && r < m ? gst[4 + r] : r;
  for (uint g = tg.y * nsg + sg; g < p.K / 64; g += nsg * ntgs.y) {
    const uint c = g * 64 + 2 * lane;
    float2 v = float2(0.0f);
    if (r < m) v = float2(*(const device bfloat2*)(src + size_t(rs) * p.src_ld + c)) * p.scale;
    dfl_put2(xh, xs, r, p.K, g, lane, v.x, v.y);
  }
}

// Prefill context rows: bf16 -> half (scaled), row-major for the prefill GEMM. grid (ceil(K/1024), rows) x 256.
struct DflRowsParams {
  uint K;
  float scale;
};
[[kernel]] void dfl_rows_bf16_half(const device bfloat* src [[buffer(0)]], device half* dst [[buffer(1)]],
                                   constant DflRowsParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x * 4, r = gid.y;
  if (c >= p.K) return;
  const float4 v = float4(*(const device bfloat4*)(src + size_t(r) * p.K + c)) * p.scale;
  *(device half4*)(dst + size_t(r) * p.K + c) = half4(v);
}

// Prefill: c = RMSNorm(xscale * x[r]; w) -> half rows (+ fp32 rows when write_f). grid rows x 256.
[[kernel]] void dfl_rows_norm_half(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                                   device half* y [[buffer(2)]], device float* yf [[buffer(3)]],
                                   constant DflNormParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const device float* xr = x + size_t(r) * p.H;
  float ss = 0.0f;
  for (uint i = tid; i < p.H; i += ntg) {
    const float v = xr[i] * p.xscale;
    ss += v * v;
  }
  ss = tg_sum(ss, red, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  for (uint i = tid; i < p.H; i += ntg) {
    const float o = xr[i] * p.xscale * inv * float(w[i]);
    y[size_t(r) * p.H + i] = half(o);
    if (p.write_n) yf[size_t(r) * p.H + i] = o;
  }
}

// ---- 8-row affine-quantized GEMM: Y[8, N] = oscale * X[8, K] . W[N, K]^T ----
// MLX affine 4/8-bit weights, group 64. A simdgroup owns 8 weight rows; per 64-group every lane
// loads 16 contiguous codes of its row (opt/shortpf qmms layout: lane row fm, slot q = fn/2,
// fragment f, element e <-> k = 64g + 16q + 8(f/4) + f%4 + 4e) and turns them into exact half
// MMA operands (code | 0x6400 = 1024 + code, minus 1024: exact). 8 MMAs per group into a zeroed fp32 C, then the fp32 epilogue
//   acc += s . C + b . sum(x over the group)
// applies the scale/bias once per group (no half rounding of dequantized weights). X is gathered
// from the row-major half table with four 8-byte loads per lane per group.
// Segments: up to 4 matrices sharing X (one dispatch); ks simdgroups split K (fixed-order sum).
struct DflQmm8Params {
  uint K, nseg, ks, x_ld;
  uint tg_begin[5];
  uint N[4];
  uint bits[4];
  uint y_ld[4];
  float oscale[4];
  uint old4;  // 1: 4-bit GEMMs on dfl_q8_run (the core before dfl_q8_run4)
};


template <int BITS>
NI_INLINE void dfl_codes(const device uint32_t* wrow, uint g, uint q, thread half2* a) {
  if (BITS == 4) {
    const uint2 w = *(const device uint2*)(wrow + g * 8 + 2 * q);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; ++f) {
      const uint word = f < 4 ? w.x : w.y;
      const uint v = ((word >> (4 * (f % 4))) & 0x000F000Fu) | 0x64006400u;
      a[f] = as_type<half2>(v) - half2(1024.0h);
    }
  } else {
    const uint4 w = *(const device uint4*)(wrow + g * 16 + 4 * q);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; ++f) {
      const uint wa = w[2 * (f / 4)], wb = w[2 * (f / 4) + 1], sh = 8 * (f % 4);
      const uint v = ((wa >> sh) & 0xFFu) | (((wb >> sh) & 0xFFu) << 16) | 0x64006400u;
      a[f] = as_type<half2>(v) - half2(1024.0h);
    }
  }
}

// acc[m][f] (tokens fn, fn+1 of weight row rows[f]) over groups [g0, g1) for NM matrices sharing
// rows; RF fragments of 8 rows per simdgroup share each X gather.
template <int BITS, int NM, int RF, bool SN = false>  // SN (4-bit): subnormal codes, as dfl_q8_run4
NI_INLINE void dfl_q8_run(const device uint32_t* const* W, const device bfloat* const* S, const device bfloat* const* B,
                          thread const uint* rows, uint K, const device half* X, uint x_ld, const device float* XS,
                          uint g0, uint g1, uint lane, thread float2 (&acc)[NM][RF]) {
  const uint2 rc = dfl_lane_rc(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint wpr = K * BITS / 32, gpr = K / 64;
  const device half* x0 = X + fn * x_ld + 16 * (fm / 2) + 4 * (fm % 2);
  const device half* x1 = x0 + x_ld;
  _Pragma("clang loop unroll(full)")
  for (int m = 0; m < NM; ++m)
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < RF; ++r) acc[m][r] = float2(0.0f);
  for (uint g = g0; g < g1; ++g) {
    half2 b[8];
    {
      const half4 c00 = *(const device half4*)(x0 + g * 64), c01 = *(const device half4*)(x0 + g * 64 + 8);
      const half4 c10 = *(const device half4*)(x1 + g * 64), c11 = *(const device half4*)(x1 + g * 64 + 8);
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < 4; ++f) {
        b[f] = half2(c00[f], c10[f]);
        b[f + 4] = half2(c01[f], c11[f]);
      }
    }
    const float2 sx = *(const device float2*)(XS + g * 8 + fn);
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < NM; ++m)
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < RF; ++r) {
        half2 a[8];
        if (SN && BITS == 4) {
          const uint2 w = *(const device uint2*)(W[m] + size_t(rows[r]) * wpr + g * 8 + 2 * q);
          _Pragma("clang loop unroll(full)")
          for (int f = 0; f < 8; ++f) a[f] = as_type<half2>(((f < 4 ? w.x : w.y) >> (4 * (f % 4))) & 0x000F000Fu);
        } else {
          dfl_codes<BITS>(W[m] + size_t(rows[r]) * wpr, g, q, a);
        }
        simdgroup_float8x8 C = simdgroup_float8x8(0);
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < 8; ++f) {
          simdgroup_half8x8 Am, Bm;
          Am.thread_elements()[0] = a[f].x;
          Am.thread_elements()[1] = a[f].y;
          Bm.thread_elements()[0] = b[f].x;
          Bm.thread_elements()[1] = b[f].y;
          simdgroup_multiply_accumulate(C, Am, Bm, C);
        }
        const float s = float(S[m][size_t(rows[r]) * gpr + g]) * (SN && BITS == 4 ? 16777216.0f : 1.0f),
                    bb = float(B[m][size_t(rows[r]) * gpr + g]);
        acc[m][r].x += s * C.thread_elements()[0] + bb * sx.x;
        acc[m][r].y += s * C.thread_elements()[1] + bb * sx.y;
      }
  }
}

// 4-bit, one fragment (RF 1), NM matrices sharing rows: dfl_q8_run with two groups per step (scale and bias
// pairs in one 4-byte load each), the next step's weights / coefficients / sums loaded before this step's MMAs, and
// the codes as half subnormals q * 2^-24 (the bare code bits: no OR / subtract per pair; the fp32 chain is the
// 2^-24-scaled one, so with s * 2^24 in the epilogue the sums agree with dfl_q8_run's up to fast-math contraction).
// Same group order and epilogue. bench_dfl (M5 Max, DRAM-bound rotation): 512 x 2048 5.7 -> 4.4 us, q|kv 6144 x 2048
// 22.1 -> 20.7, down 2048 x 6144 24.8 -> 22.0, the reduced LM head 98304 x 2048 261 -> 254; in the decode loop the
// 35B drafting pass -0.03 to -0.05 ms (NI_AB, 3 runs). Not for the fused gate/up (NM 2): slower there.
NI_INLINE void dfl_codes_sn(uint2 w, thread half2* a) {
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) a[f] = as_type<half2>(((f < 4 ? w.x : w.y) >> (4 * (f % 4))) & 0x000F000Fu);
}
NI_INLINE void dfl_xg(const device half* x0, const device half* x1, uint g, thread half2* b) {
  const half4 c00 = *(const device half4*)(x0 + g * 64), c01 = *(const device half4*)(x0 + g * 64 + 8);
  const half4 c10 = *(const device half4*)(x1 + g * 64), c11 = *(const device half4*)(x1 + g * 64 + 8);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 4; ++f) {
    b[f] = half2(c00[f], c10[f]);
    b[f + 4] = half2(c01[f], c11[f]);
  }
}
NI_INLINE float2 dfl_mma8(thread const half2* a, thread const half2* b) {
  simdgroup_float8x8 C = simdgroup_float8x8(0);
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) {
    simdgroup_half8x8 Am, Bm;
    Am.thread_elements()[0] = a[f].x;
    Am.thread_elements()[1] = a[f].y;
    Bm.thread_elements()[0] = b[f].x;
    Bm.thread_elements()[1] = b[f].y;
    simdgroup_multiply_accumulate(C, Am, Bm, C);
  }
  return float2(C.thread_elements()[0], C.thread_elements()[1]);
}
template <int NM>
NI_INLINE void dfl_q8_run4(const device uint32_t* const* W, const device bfloat* const* S, const device bfloat* const* B,
                           uint row, uint K, const device half* X, uint x_ld, const device float* XS, uint g0, uint g1,
                           uint lane, thread float2 (&acc)[NM][1]) {
  const uint2 rc = dfl_lane_rc(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint wpr = K / 8, gpr = K / 64;
  const device half* x0 = X + fn * x_ld + 16 * (fm / 2) + 4 * (fm % 2);
  const device half* x1 = x0 + x_ld;
  const device uint32_t* wr[NM];
  const device bfloat* sr[NM];
  const device bfloat* br[NM];
  _Pragma("clang loop unroll(full)")
  for (int m = 0; m < NM; ++m) {
    wr[m] = W[m] + size_t(row) * wpr + 2 * q;
    sr[m] = S[m] + size_t(row) * gpr;
    br[m] = B[m] + size_t(row) * gpr;
    acc[m][0] = float2(0.0f);
  }
  constexpr float kSn = 16777216.0f;  // 2^24
  uint g = g0;
  if (gpr % 2 == 0 && g0 % 2 == 0 && g0 + 1 < g1) {  // pairs (4-byte scale / bias pairs need an even group)
    uint2 wa[NM], wb[NM];
    uint sp[NM], bp[NM];
    float4 sxp;
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < NM; ++m) {
      wa[m] = *(const device uint2*)(wr[m] + g * 8);
      wb[m] = *(const device uint2*)(wr[m] + g * 8 + 8);
      sp[m] = *(const device uint*)(sr[m] + g);
      bp[m] = *(const device uint*)(br[m] + g);
    }
    sxp = float4(*(const device float2*)(XS + g * 8 + fn), *(const device float2*)(XS + g * 8 + 8 + fn));
    for (; g + 1 < g1; g += 2) {
      uint2 w0[NM], w1[NM];
      uint su[NM], bu[NM];
      const float4 sx = sxp;
      _Pragma("clang loop unroll(full)")
      for (int m = 0; m < NM; ++m) {
        w0[m] = wa[m]; w1[m] = wb[m]; su[m] = sp[m]; bu[m] = bp[m];
      }
      if (g + 3 < g1) {
        _Pragma("clang loop unroll(full)")
        for (int m = 0; m < NM; ++m) {
          wa[m] = *(const device uint2*)(wr[m] + (g + 2) * 8);
          wb[m] = *(const device uint2*)(wr[m] + (g + 2) * 8 + 8);
          sp[m] = *(const device uint*)(sr[m] + g + 2);
          bp[m] = *(const device uint*)(br[m] + g + 2);
        }
        sxp = float4(*(const device float2*)(XS + (g + 2) * 8 + fn), *(const device float2*)(XS + (g + 2) * 8 + 8 + fn));
      }
      half2 b0[8], b1[8];
      dfl_xg(x0, x1, g, b0);
      dfl_xg(x0, x1, g + 1, b1);
      _Pragma("clang loop unroll(full)")
      for (int m = 0; m < NM; ++m) {
        const float2 s2 = float2(as_type<bfloat2>(su[m])) * kSn, b2 = float2(as_type<bfloat2>(bu[m]));
        half2 a[8];
        dfl_codes_sn(w0[m], a);
        const float2 c0 = dfl_mma8(a, b0);
        dfl_codes_sn(w1[m], a);
        const float2 c1 = dfl_mma8(a, b1);
        acc[m][0] += s2.x * c0 + b2.x * sx.xy;
        acc[m][0] += s2.y * c1 + b2.y * sx.zw;
      }
    }
  }
  for (; g < g1; ++g) {  // an odd count's last group (or an odd start)
    half2 b0[8];
    dfl_xg(x0, x1, g, b0);
    const float2 sx = *(const device float2*)(XS + g * 8 + fn);
    _Pragma("clang loop unroll(full)")
    for (int m = 0; m < NM; ++m) {
      half2 a[8];
      dfl_codes_sn(*(const device uint2*)(wr[m] + g * 8), a);
      const float2 c = dfl_mma8(a, b0);
      acc[m][0] += (float(sr[m][g]) * kSn) * c + float(br[m][g]) * sx;
    }
  }
}

// grid: tg_begin[nseg] threadgroups x 256 (8 simdgroups = 8/ks row blocks of 8 * RF rows).
template <int RF>
NI_INLINE void dfl_qmm8_impl(const device uint32_t* const* Wv, const device bfloat* const* Sv,
                             const device bfloat* const* Bv, device float* const* Yv, const device half* X,
                             const device float* XS, constant DflQmm8Params& p, threadgroup float2* red, uint tg,
                             uint sg, uint lane) {
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint ks = p.ks, rbs = 8 / ks;
  const uint rb = (tg - p.tg_begin[si]) * rbs + sg / ks, kslice = sg % ks;
  const uint N = p.N[si], gpr = p.K / 64;
  const uint2 rc = dfl_lane_rc(lane);
  uint n[RF], rows[RF];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < RF; ++r) {
    n[r] = (rb * RF + r) * 8 + rc.x;
    rows[r] = min(n[r], N - 1);
  }
  const uint gper = (gpr + ks - 1) / ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* W[1] = {Wv[si]};
  const device bfloat* S[1] = {Sv[si]};
  const device bfloat* B[1] = {Bv[si]};
  float2 acc[1][RF];
  if (p.bits[si] == 4) {
    if (RF == 1 && !p.old4) {
      float2 a1[1][1];
      dfl_q8_run4<1>(W, S, B, rows[0], p.K, X, p.x_ld, XS, g0, g1, lane, a1);
      acc[0][0] = a1[0][0];
    } else {
      dfl_q8_run<4, 1, RF>(W, S, B, rows, p.K, X, p.x_ld, XS, g0, g1, lane, acc);
    }
  } else {
    dfl_q8_run<8, 1, RF>(W, S, B, rows, p.K, X, p.x_ld, XS, g0, g1, lane, acc);
  }
  if (ks > 1) {
    if (kslice)
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < RF; ++r) red[(sg * RF + r) * 32 + lane] = acc[0][r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kslice) return;
    for (uint j = 1; j < ks; ++j)
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < RF; ++r) acc[0][r] += red[((sg + j) * RF + r) * 32 + lane];
  }
  device float* Y = Yv[si];
  const uint ld = p.y_ld[si];
  const float os = p.oscale[si];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < RF; ++r) {
    if (n[r] >= N) continue;
    Y[rc.y * ld + n[r]] = os * acc[0][r].x;
    Y[(rc.y + 1) * ld + n[r]] = os * acc[0][r].y;
  }
}

#define DFL_QMM8(NAME, RF)                                                                                   \
  [[kernel]] void NAME(const device uint32_t* W0 [[buffer(0)]], const device bfloat* S0 [[buffer(1)]],         \
                       const device bfloat* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],                 \
                       const device uint32_t* W1 [[buffer(4)]], const device bfloat* S1 [[buffer(5)]],         \
                       const device bfloat* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],                 \
                       const device uint32_t* W2 [[buffer(8)]], const device bfloat* S2 [[buffer(9)]],         \
                       const device bfloat* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],               \
                       const device uint32_t* W3 [[buffer(12)]], const device bfloat* S3 [[buffer(13)]],       \
                       const device bfloat* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],               \
                       const device half* X [[buffer(16)]], const device float* XS [[buffer(17)]],             \
                       constant DflQmm8Params& p [[buffer(18)]], uint tg [[threadgroup_position_in_grid]],     \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float2 red[8 * RF * 32];                                                                     \
    const device uint32_t* Wv[4] = {W0, W1, W2, W3};                                                          \
    const device bfloat* Sv[4] = {S0, S1, S2, S3};                                                            \
    const device bfloat* Bv[4] = {B0, B1, B2, B3};                                                            \
    device float* Yv[4] = {Y0, Y1, Y2, Y3};                                                                   \
    dfl_qmm8_impl<RF>(Wv, Sv, Bv, Yv, X, XS, p, red, tg, sg, lane);                                          \
  }

DFL_QMM8(dfl_qmm8, 1)
DFL_QMM8(dfl_qmm8_r2, 2)

// Conv projection + dfl_conv_prep in one dispatch. The projection's rows are stored permuted (DFlash2 constructor):
// row block b < H / 64 holds D columns [4b, 4b + 4) of stage 0 tap 0 and the same 4 groups of tap 1, i.e. every
// coefficient the 64 inputs of X8 group b need, so the simdgroup that finishes block b writes
//   a[r] = (b00 + D[r,0,g]) x[r] + (b01 + D[r,1,g]) x[r-1]
// for those 64 inputs and all 8 rows (dfl_conv_prep's math) into a second X8 table (the first is this GEMM's input).
// Blocks >= H / 64 are the stage-1 columns 2G.. in order. D is written in the unpermuted order.
// grid / threads as dfl_qmm8 (one segment, RF 1).
[[kernel]] void dfl_qmm8_cp(const device uint32_t* W0 [[buffer(0)]], const device bfloat* S0 [[buffer(1)]],
                            const device bfloat* B0 [[buffer(2)]], device float* D [[buffer(3)]],
                            const device half* X [[buffer(16)]], const device float* XS [[buffer(17)]],
                            constant DflQmm8Params& p [[buffer(18)]], const device float* x [[buffer(19)]],
                            const device bfloat* base [[buffer(20)]], device half* xh2 [[buffer(21)]],
                            device float* xs2 [[buffer(22)]], constant DflConvParams& cp [[buffer(23)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 32];
  threadgroup float dd[8][8][8];  // [row block in the threadgroup][weight row][token row]
  const uint ks = p.ks, rbs = 8 / ks;
  const uint rbl = sg / ks, rb = tg * rbs + rbl, kslice = sg % ks;
  const uint N = p.N[0], gpr = p.K / 64;
  const uint2 rc = dfl_lane_rc(lane);
  const uint n = rb * 8 + rc.x;
  const uint rows[1] = {min(n, N - 1)};
  const uint gper = (gpr + ks - 1) / ks, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* W[1] = {W0};
  const device bfloat* S[1] = {S0};
  const device bfloat* B[1] = {B0};
  float2 acc[1][1];
  if (p.bits[0] == 4 && !p.old4) dfl_q8_run4<1>(W, S, B, rows[0], p.K, X, p.x_ld, XS, g0, g1, lane, acc);
  else if (p.bits[0] == 4) dfl_q8_run<4, 1, 1>(W, S, B, rows, p.K, X, p.x_ld, XS, g0, g1, lane, acc);
  else dfl_q8_run<8, 1, 1>(W, S, B, rows, p.K, X, p.x_ld, XS, g0, g1, lane, acc);
  if (ks > 1) {
    if (kslice) red[sg * 32 + lane] = acc[0][0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kslice) return;
    for (uint j = 1; j < ks; ++j) acc[0][0] += red[(sg + j) * 32 + lane];
  }
  if (n >= N) return;  // N is a multiple of 8: the whole simdgroup
  const uint H = cp.H;
  // D in the unpermuted column order (dfl_conv_fin_norm reads stage 1; the debug dumps all of it)
  const uint col = rb >= H / 64 ? n : rc.x < 4 ? 4 * rb + rc.x : cp.G + 4 * rb + rc.x - 4;
  D[rc.y * p.y_ld[0] + col] = acc[0][0].x;
  D[(rc.y + 1) * p.y_ld[0] + col] = acc[0][0].y;
  if (rb >= H / 64) return;
  dd[rbl][rc.x][rc.y] = acc[0][0].x;
  dd[rbl][rc.x][rc.y + 1] = acc[0][0].y;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = rb * 64 + 2 * lane, j = lane / 8;  // input pair c, c + 1; its conv group 4 rb + j
  const float2 b0 = float2(base[c], base[c + 1]), b1 = float2(base[H + c], base[H + c + 1]);
  for (uint r = 0; r < 8; ++r) {
    const float d0 = dd[rbl][j][r], d1 = dd[rbl][4 + j][r];
    const float2 xc = *(const device float2*)(x + r * H + c);
    const float2 xp = r > 0 ? *(const device float2*)(x + (r - 1) * H + c) : float2(0.0f);
    const float2 a = (b0 + d0) * xc + (b1 + d1) * xp;
    dfl_put2(xh2, xs2, r, H, rb, lane, a.x, a.y);
  }
}

// SwiGLU of gate / up rows from a GEMM (fp32 [8][N] each, the matrix-unit path): H = half(silu(g) * u) as the
// down projection's X8 table (+ T table) and its group sums, as dfl_qmm8_gu's epilogue. grid (8 rows, ceil(N / 512))
// x 256: one 64-group per simdgroup.
[[kernel]] void dfl_swiglu_x8(const device float* gy [[buffer(0)]], const device float* uy [[buffer(1)]],
                              device half* xh [[buffer(2)]], device float* xs [[buffer(3)]],
                              constant uint& N [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint2 ntgs [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x;
  for (uint g = tg.y * nsg + sg; g < N / 64; g += nsg * ntgs.y) {
    const uint c = g * 64 + 2 * lane;
    const float2 a = *(const device float2*)(gy + r * N + c), b = *(const device float2*)(uy + r * N + c);
    dfl_put2(xh, xs, r, N, g, lane, silu(a.x) * b.x, silu(a.y) * b.y);
  }
}

// Gate/up + SwiGLU: H[8, N] = half(silu(X Wg^T) * (X Wu^T)) and HS[N/64][8] = sums of the half
// values per 64 columns (the down projection's X8 table). A threadgroup owns 64 output columns:
// 8 row blocks x KS simdgroups splitting K (fixed-order sum). grid: N/64 threadgroups x 256*KS.
template <int KS>
NI_INLINE void dfl_qmm8_gu_impl(const device uint32_t* Wg, const device bfloat* Sg, const device bfloat* Bg,
                                device half* Hh, const device uint32_t* Wu, const device bfloat* Su,
                                const device bfloat* Bu, device float* HS, const device half* X,
                                const device float* XS, constant DflQmm8Params& p, threadgroup float2* red,
                                threadgroup float* hred, uint tg, uint sg, uint lane) {
  const uint N = p.N[0], gpr = p.K / 64;
  const uint2 rc = dfl_lane_rc(lane);
  const uint rbl = sg / KS, kslice = sg % KS;
  const uint n = (tg * 8 + rbl) * 8 + rc.x;
  const uint rows[1] = {min(n, N - 1)};
  const uint gper = (gpr + KS - 1) / KS, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device uint32_t* W[2] = {Wg, Wu};
  const device bfloat* S[2] = {Sg, Su};
  const device bfloat* B[2] = {Bg, Bu};
  float2 acc[2][1];
  // subnormal codes only: dfl_q8_run4's pairs / prefetch measured slower here (6144 x 2048: 44.5 us; pairs +
  // prefetch 49.1, pairs 45.2; subnormal codes 44.2)
  if (p.bits[0] == 4 && !p.old4) dfl_q8_run<4, 2, 1, true>(W, S, B, rows, p.K, X, p.x_ld, XS, g0, g1, lane, acc);
  else if (p.bits[0] == 4) dfl_q8_run<4, 2, 1>(W, S, B, rows, p.K, X, p.x_ld, XS, g0, g1, lane, acc);
  else dfl_q8_run<8, 2, 1>(W, S, B, rows, p.K, X, p.x_ld, XS, g0, g1, lane, acc);
  if (KS > 1) {
    if (kslice) {
      red[(sg * 2 + 0) * 32 + lane] = acc[0][0];
      red[(sg * 2 + 1) * 32 + lane] = acc[1][0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kslice == 0)
      for (uint j = 1; j < KS; ++j) {
        acc[0][0] += red[((sg + j) * 2 + 0) * 32 + lane];
        acc[1][0] += red[((sg + j) * 2 + 1) * 32 + lane];
      }
  }
  if (kslice == 0) {
    const half h0 = half(silu(acc[0][0].x) * acc[1][0].x), h1 = half(silu(acc[0][0].y) * acc[1][0].y);
    const uint ld = p.y_ld[0];
    float2 t = float2(0.0f);
    if (n < N) {
      Hh[rc.y * ld + n] = h0;
      Hh[(rc.y + 1) * ld + n] = h1;
      const uint tn = (n & ~63u) + dfl_tslot(n & 63u);  // T table
      Hh[(8 + rc.y) * ld + tn] = h0;
      Hh[(9 + rc.y) * ld + tn] = h1;
      t = float2(float(h0), float(h1));
    }
    // sum over the 8 weight rows of the fragment (lane bits 1, 2, 4 select the row)
    t += simd_shuffle_xor(t, 2);
    t += simd_shuffle_xor(t, 4);
    t += simd_shuffle_xor(t, 16);
    if ((lane & 22u) == 0) {
      hred[rbl * 8 + rc.y] = t.x;
      hred[rbl * 8 + rc.y + 1] = t.y;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < 8) {
    float s = 0.0f;
    for (uint j = 0; j < 8; ++j) s += hred[j * 8 + lane];
    HS[tg * 8 + lane] = s;
  }
}

#define DFL_QMM8_GU(NAME, KS)                                                                                \
  [[kernel]] void NAME(const device uint32_t* Wg [[buffer(0)]], const device bfloat* Sg [[buffer(1)]],         \
                       const device bfloat* Bg [[buffer(2)]], device half* Hh [[buffer(3)]],                  \
                       const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],         \
                       const device bfloat* Bu [[buffer(6)]], device float* HS [[buffer(7)]],                 \
                       const device half* X [[buffer(16)]], const device float* XS [[buffer(17)]],             \
                       constant DflQmm8Params& p [[buffer(18)]], uint tg [[threadgroup_position_in_grid]],     \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float2 red[KS > 1 ? 8 * KS * 2 * 32 : 1];                                                    \
    threadgroup float hred[8 * 8];                                                                           \
    dfl_qmm8_gu_impl<KS>(Wg, Sg, Bg, Hh, Wu, Su, Bu, HS, X, XS, p, red, hred, tg, sg, lane);                \
  }

DFL_QMM8_GU(dfl_qmm8_gu, 1)
DFL_QMM8_GU(dfl_qmm8_gu_ks2, 2)
DFL_QMM8_GU(dfl_qmm8_gu_ks4, 4)

// ---- selector (SPEC §7) ----
struct DflMergeParams {
  uint n;   // stage-1 candidates per row
};
// Top-16 of the stage-1 candidates of one proposal row, ordered (value desc, id asc). grid 7 x 256.
[[kernel]] void dfl_merge16(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                            device float* ov [[buffer(2)]], device int* oi [[buffer(3)]],
                            constant DflMergeParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  constexpr int PER = 8;
  float v[PER];
  uint id[PER];
  for (int j = 0; j < PER; ++j) {
    const uint k = tid * PER + j;
    v[j] = k < p.n ? cv[r * p.n + k] : -INFINITY;
    id[j] = k < p.n ? ci[r * p.n + k] : 0xffffffffu;
  }
  for (uint q = 0; q < 16; ++q) {
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    int bj = -1;
    for (int j = 0; j < PER; ++j)
      if (v[j] > bv || (v[j] == bv && id[j] < bi)) {
        bv = v[j];
        bi = id[j];
        bj = j;
      }
    float wv = bv;
    uint wi = bi;
    tg_argmax(wv, wi, sv, si, tid, ntg);
    if (bj >= 0 && bi == wi && bv == wv) v[bj] = -INFINITY;
    if (tid == 0) {
      ov[r * 16 + q] = wv;
      oi[r * 16 + q] = int(wi);
    }
  }
}

// ---- top-16 with (value desc, id asc) order, simdgroup selection (NI_DFLASH_TOPK=1, default) ----
NI_INLINE bool dfl_before(float va, uint ia, float vb, uint ib) { return va > vb || (va == vb && ia < ib); }
#define DFL_CS(a, b)                                       \
  if (dfl_before(v[b], id[b], v[a], id[a])) {               \
    const float tv = v[a]; v[a] = v[b]; v[b] = tv;          \
    const uint ti = id[a]; id[a] = id[b]; id[b] = ti;       \
  }
NI_INLINE void dfl_sort8(thread float* v, thread uint* id) {  // Batcher network, 19 comparators
  DFL_CS(0, 1) DFL_CS(2, 3) DFL_CS(4, 5) DFL_CS(6, 7)
  DFL_CS(0, 2) DFL_CS(1, 3) DFL_CS(4, 6) DFL_CS(5, 7)
  DFL_CS(1, 2) DFL_CS(5, 6) DFL_CS(0, 4) DFL_CS(3, 7)
  DFL_CS(1, 5) DFL_CS(2, 6)
  DFL_CS(1, 4) DFL_CS(3, 6)
  DFL_CS(2, 4) DFL_CS(3, 5)
  DFL_CS(3, 4)
}
// Every lane holds a sorted run of 8; 16 rounds pick the simdgroup's best head and pop it.
// Lane q < 16 returns the q-th largest of the 256 (value, id).
NI_INLINE void dfl_simd_top16(thread float* v, thread uint* id, uint lane, thread float& ov, thread uint& oi) {
  ov = -INFINITY;
  oi = 0xffffffffu;
  for (uint q = 0; q < 16; ++q) {
    const float mv = simd_max(v[0]);
    const uint mi = simd_min(v[0] == mv ? id[0] : 0xffffffffu);
    if (lane == q) {
      ov = mv;
      oi = mi;
    }
    if (v[0] == mv && id[0] == mi) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 7; ++j) {
        v[j] = v[j + 1];
        id[j] = id[j + 1];
      }
      v[7] = -INFINITY;
      id[7] = 0xffffffffu;
    }
  }
}

struct DflTopParams {
  uint n;        // valid elements per row
  uint ld;       // row stride of src
  uint nchunks;  // threadgroups per row (2048 elements each)
};
// grid (nchunks, rows) x 256. out[(r * nchunks + chunk) * 16 + q], ordered. ids: has_id 0 = the index,
// 1 = src_id[r * ld + k], 2 = src_id[k] (one id map for every row: a reduced draft head's token ids).
[[kernel]] void dfl_top16(const device float* src [[buffer(0)]], const device uint* src_id [[buffer(1)]],
                          device float* ov [[buffer(2)]], device uint* oi [[buffer(3)]],
                          constant DflTopParams& p [[buffer(4)]], constant uint& has_id [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sv[8 * 16];
  threadgroup uint si[8 * 16];
  const uint r = tg.y, base = tg.x * 2048 + tid * 8;
  float v[8];
  uint id[8];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; ++j) {
    const uint k = base + j;
    const bool ok = k < p.n;
    v[j] = ok ? src[size_t(r) * p.ld + k] : -INFINITY;
    id[j] = ok ? (has_id == 2 ? src_id[k] : has_id ? src_id[size_t(r) * p.ld + k] : k) : 0xffffffffu;
  }
  dfl_sort8(v, id);
  float bv;
  uint bi;
  dfl_simd_top16(v, id, lane, bv, bi);
  if (lane < 16) {
    sv[sg * 16 + lane] = bv;
    si[sg * 16 + lane] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 4; ++j) {
    v[j] = sv[lane * 4 + j];
    id[j] = si[lane * 4 + j];
    v[4 + j] = -INFINITY;
    id[4 + j] = 0xffffffffu;
  }
  dfl_simd_top16(v, id, lane, bv, bi);
  if (lane < 16) {
    const size_t o = (size_t(r) * p.nchunks + tg.x) * 16 + lane;
    ov[o] = bv;
    oi[o] = bi;
  }
}

// Edge lattice E[p][j][k] = sum_r A[pred_j][r] s_{p+1}[r] Bc[C_p[k]][r] (pred = anchor for p = 0,
// C_{p-1}[j] otherwise; fp32 from the bf16 codebooks) -> edges [7][16][16] (NaN where p = 0, j > 0).
// grid: (7 positions p, m) threadgroups x 256; one simdgroup per dot product of 256, threadgroup y takes every m-th
// simdgroup's share (m 8: 4 dot products per simdgroup instead of 32; the codebook rows are DRAM reads).
[[kernel]] void dfl_lattice(const device int* cand_i [[buffer(0)]], const device float* sel [[buffer(1)]],
                            const device bfloat* A [[buffer(2)]], const device bfloat* Bc [[buffer(3)]],
                            const device int* tokens [[buffer(4)]], device float* edges [[buffer(5)]],
                            constant uint& P [[buffer(6)]], const device uint* gst [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint pp = tg.x;
  const uint ntask = pp == 0 ? 16u : 256u;
  if (pp == 0 && tg.y == 0)
    for (uint i = 16 + sg * 32 + lane; i < 256; i += nsg * 32) edges[i] = as_type<float>(0x7fc00000u);
  const device float* s = sel + (pp + 1) * 256 + lane * 8;
  const float4 s0 = *(const device float4*)s, s1 = *(const device float4*)(s + 4);
  for (uint t = tg.y * nsg + sg; t < ntask; t += nsg * ntgs.y) {
    const uint j = t / 16, k = t % 16;
    const uint pred = pp == 0 ? uint(tokens[dfl_pos(P, gst)]) : uint(cand_i[(pp - 1) * 16 + j]);
    const uint succ = uint(cand_i[pp * 16 + k]);
    const device bfloat* a = A + size_t(pred) * 256 + lane * 8;
    const device bfloat* b = Bc + size_t(succ) * 256 + lane * 8;
    const float4 a0 = float4(*(const device bfloat4*)a), a1 = float4(*(const device bfloat4*)(a + 4));
    const float4 b0 = float4(*(const device bfloat4*)b), b1 = float4(*(const device bfloat4*)(b + 4));
    const float acc = simd_sum(dot(a0 * s0, b0) + dot(a1 * s1, b1));
    if (lane == 0) edges[pp * 256 + j * 16 + k] = acc;
  }
}

struct DflSelectParams {
  uint P;          // anchor position: tokens[P] is the anchor
  float temp;      // 0: greedy walk (greedy acceptance)
  uint seed;
  uint out_index;  // tokens[out_index + p] = d_p
  float qtemp;     // temp > 0: the walk's own temperature T_q (0: argmax walk, q = one-hot)
};
// The walk over the lattice: S_p[k] = U_p[k] + E[p][j_p][k]; greedy = first argmax; sampled = first k
// with cumsum softmax(S_p / T_q) > u (else 15), q_p stored for the acceptance test. One thread.
// T_q is a free choice (the acceptance only needs q_p to be exactly the law d_p was drawn from); the
// spec's T_q = T is the default, T_q = 0 drafts the argmax chain with q = one-hot (accept iff u < p(d)).
[[kernel]] void dfl_walk(const device int* cand_i [[buffer(0)]], const device float* cand_v [[buffer(1)]],
                         const device float* E [[buffer(2)]], device int* tokens [[buffer(3)]],
                         device float* walk_s [[buffer(4)]], device float* qprob [[buffer(5)]],
                         device int* walk_k [[buffer(6)]], constant DflSelectParams& p [[buffer(7)]],
                         const device uint* gst [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const uint P = dfl_pos(p.P, gst), out_index = p.P == kDflGpu ? P + 1 : p.out_index;
  uint jp = 0;
  for (uint pp = 0; pp < 7; ++pp) {
    float S[16];
    float mx = -INFINITY;
    uint kb = 0;
    for (uint k = 0; k < 16; ++k) {
      S[k] = cand_v[pp * 16 + k] + E[pp * 256 + jp * 16 + k];
      walk_s[pp * 16 + k] = S[k];
      if (S[k] > mx) {
        mx = S[k];
        kb = k;
      }
    }
    if (p.temp > 0.0f && !(p.qtemp > 0.0f)) {  // argmax chain, q = point mass on it
      for (uint k = 0; k < 16; ++k) qprob[pp * 16 + k] = k == kb ? 1.0f : 0.0f;
    } else if (p.temp > 0.0f) {
      float q[16], tot = 0.0f;
      for (uint k = 0; k < 16; ++k) {
        q[k] = metal::precise::exp((S[k] - mx) / p.qtemp);
        tot += q[k];
      }
      const float u = dfl_uniform(p.seed, P + 1 + pp, kDflSaltWalk);
      float cum = 0.0f;
      kb = 0;
      bool found = false;
      for (uint k = 0; k < 16; ++k) {
        q[k] /= tot;
        cum += q[k];
        if (!found && q[k] > 0.0f) kb = k;  // fallback (u above the rounded total): last k with q > 0
        if (!found && cum > u && q[k] > 0.0f) {
          kb = k;
          found = true;
        }
        qprob[pp * 16 + k] = q[k];
      }
    } else {
      for (uint k = 0; k < 16; ++k) qprob[pp * 16 + k] = 0.0f;
    }
    walk_k[pp] = int(kb);
    jp = kb;
    tokens[out_index + pp] = cand_i[pp * 16 + kb];
  }
}

// ---- KFlash: the DFlash2 lattice as the shared tree builder's children tables (kflash.h, kf_build) ----
// Children-row (s * 16 + p) = the 16 candidates of position s after candidate p of position s - 1 (s = 0: after the
// anchor, p = 0): clq = log softmax_k(U_s[k] + E[s][p][k]) (the walk's scores; E row 0 under the anchor; DFlash v1
// drafters have E = 0), ctok = the candidates' ids. The rows' successors (kf::lattice_next) are constant.
// grid: 7 * 16 threadgroups x 32.
[[kernel]] void dfl_kf_table(const device int* cand_i [[buffer(0)]], const device float* cand_v [[buffer(1)]],
                             const device float* E [[buffer(2)]], device float* clq [[buffer(3)]],
                             device int* ctok [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint s = g / 16, pp = g % 16;
  const bool on = lane < 16;
  float v = -INFINITY;
  if (on) {
    const float e = E[s * 256 + (s == 0 ? 0u : pp) * 16 + lane];
    v = cand_v[s * 16 + lane] + (isnan(e) ? 0.0f : e);
  }
  const float mx = simd_max(v);
  const float z = simd_sum(on ? metal::precise::exp(v - mx) : 0.0f);
  if (on) {
    clq[g * 16 + lane] = v - mx - metal::precise::log(z);
    ctok[g * 16 + lane] = cand_i[s * 16 + lane];
  }
}

// The shared builder for a GPU-driven drafting pass: the anchor comes from the drafter state (kDflGpu), so the nodes'
// tokens go to tokens[gst[0] + r]. grid: 1 x 32.
[[kernel]] void kf_build_gpu(const device float* clq [[buffer(0)]], const device int* ctok [[buffer(1)]],
                             const device int* cnext [[buffer(2)]], constant KfParams& kp [[buffer(3)]],
                             device int* tree [[buffer(4)]], device int* tokens [[buffer(5)]],
                             device uint* live [[buffer(6)]], constant KfBuildP& p [[buffer(7)]],
                             const device uint* gst [[buffer(8)]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float fs[kKfMaxRows * 32], fq[kKfMaxRows * 32];
  threadgroup uint fe[kKfMaxRows * 32];
  threadgroup float ns[kKfMaxRows];
  threadgroup uint tdep[kKfMaxRows];
  threadgroup uint nf, nodes;
  kf_build_impl(clq, ctok, cnext, kp, tree, tokens, live, p.K, dfl_pos(p.P, gst), p.root, fs, fq, fe, ns, tdep, nf,
                nodes, lane);
}

// ---- acceptance (SPEC §8): verify rows -> accepted length a and the bonus / correction token ----
// Uniform streams (counter-based, keyed by seed, absolute position and purpose; SPEC §8.2): the walk
// draws proposal p with position P+1+p (kDflSaltWalk), acceptance of proposal i uses P+1+i
// (kDflSaltAcc), the residual / bonus draw P+1+a (kDflSaltRes). A position's acceptance and residual
// uniforms are consumed at most once over a whole generation (the next anchor sits past them); walk
// uniforms of proposals that were never evaluated are reused by the next cycle, which is harmless
// because nothing emitted depended on them.
constant constexpr uint kDflSaltAcc = 0x3C6EF372u;
constant constexpr uint kDflSaltRes = 0xA54FF53Au;

// Target distribution of every verify row, exactly the plain sampler's (sample_stage2 /
// sample_stage2_rows, verify.metal): the K best stage-1 candidates ordered (value desc, id asc), the
// top-p prefix on the untempered probabilities, then exp((l - l0) / T) normalized over that prefix.
// dist_id / dist_p [R][32] (zero past keep), dist_keep [R]. grid: R threadgroups x 1024.
struct DflDistParams {
  uint n;  // stage-1 candidates per row
  uint K;  // top-k (<= 32)
  float temp;
  float top_p;
};
[[kernel]] void dfl_dist_rows(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                              device int* dist_id [[buffer(2)]], device float* dist_p [[buffer(3)]],
                              device uint* dist_keep [[buffer(4)]], constant DflDistParams& p [[buffer(5)]],
                              uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float topv[32];
  threadgroup uint topi[32];
  constexpr int PER = 4;
  float v[PER];
  uint id[PER];
  for (int j = 0; j < PER; ++j) {
    uint k = tid * PER + j;
    v[j] = k < p.n ? cv[r * p.n + k] : -INFINITY;
    id[j] = k < p.n ? ci[r * p.n + k] : 0xffffffff;
  }
  const uint K = max(1u, min(p.K, 32u));
  for (uint q = 0; q < K; ++q) {  // the selection loop of sample_stage2_rows
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
  if (tid == 0) {
    float probs[32];
    uint keep = 1;
    float kt = 1.0f;
    probs[0] = 1.0f;
    if (p.temp > 0.0f && K > 1) {  // same arithmetic, same order as sample_stage2_rows
      float tot = 0.0f;
      for (uint q = 0; q < K; ++q) {
        probs[q] = metal::precise::exp(topv[q] - topv[0]);
        tot += probs[q];
      }
      keep = K;
      float cum = 0.0f;
      for (uint q = 0; q < K; ++q) {
        cum += probs[q] / tot;
        if (cum >= p.top_p) { keep = q + 1; break; }
      }
      kt = 0.0f;
      for (uint q = 0; q < keep; ++q) {
        probs[q] = metal::precise::exp((topv[q] - topv[0]) / p.temp);
        kt += probs[q];
      }
    }
    for (uint q = 0; q < 32; ++q) {
      dist_id[r * 32 + q] = q < keep ? int(topi[q]) : -1;
      dist_p[r * 32 + q] = q < keep ? probs[q] / kt : 0.0f;
    }
    dist_keep[r] = keep;
  }
}

// Accepts proposals d_i = tokens[P+1+i] (i < K) against the verify rows of [anchor, d_0 .. d_{K-1}]
// and writes the token that follows the accepted prefix to tokens[P+1+a]; res = {a, that token}.
//  T = 0: a = longest prefix with d_i == argmax_i (samp: the verify's own argmax, ties to the lowest id
//         as in plain decoding), token = argmax_a.
//  T > 0: exact speculative sampling (Leviathan et al. / Chen et al.; SPEC §8.2): accept d_i iff
//         u_i * q_i(d_i) < p_i(d_i), stop at the first rejection; then the token is drawn from
//         max(0, p_a - q_a) normalized (or p_a if that is empty), or from p_K after K acceptances.
//         p = dfl_dist_rows, q = the walk's distribution over its 16 candidates (qprob, walk_k).
// grid: 1 x 32 (one simdgroup; lane q holds candidate q of the current row).
struct DflAcceptParams {
  uint P;   // anchor position
  uint K;   // proposals verified (verify rows 0..K)
  float temp;
  uint seed;
  uint dyn;  // 1: only the first kdyn[0] proposals were verified (dfl_depth)
};
[[kernel]] void dfl_accept(device int* tokens [[buffer(0)]], const device int* samp [[buffer(1)]],
                           const device int* dist_id [[buffer(2)]], const device float* dist_p [[buffer(3)]],
                           const device int* cand_i [[buffer(4)]], const device float* qprob [[buffer(5)]],
                           const device int* walk_k [[buffer(6)]], device int* res [[buffer(7)]],
                           constant DflAcceptParams& p [[buffer(8)]], const device uint* kdyn [[buffer(9)]],
                           device uint* gst [[buffer(10)]], uint lane [[thread_index_in_simdgroup]]) {
  const uint K = p.dyn ? min(p.K, kdyn[0]) : p.K;
  if (p.temp <= 0.0f) {
    const bool stop = lane >= K || tokens[p.P + 1 + lane] != samp[lane];
    const uint a = simd_min(stop ? lane : 32u);
    if (lane == 0) {
      const int y = samp[a];
      tokens[p.P + 1 + a] = y;
      res[0] = int(a);
      res[1] = y;
      gst[0] = p.P + a + 1;  // next anchor
      gst[1] = a + 1;        // rows to commit to the ring (old anchor + accepted drafts)
      gst[2] = p.P;
    }
    return;
  }
  uint a = 0;
  for (; a < K; ++a) {
    const int d = tokens[p.P + 1 + a];
    const float pd = simd_sum(dist_id[a * 32 + lane] == d ? dist_p[a * 32 + lane] : 0.0f);
    const float qd = qprob[a * 16 + uint(walk_k[a])];
    const float u = dfl_uniform(p.seed, p.P + 1 + a, kDflSaltAcc);
    if (!(u * qd < pd)) break;
  }
  const int x = dist_id[a * 32 + lane];
  const float px = dist_p[a * 32 + lane];
  float w = px;
  if (a < K) {  // residual max(0, p_a - q_a): q_a(x) from the walk's 16 candidates of proposal a
    const int cid = lane < 16 ? cand_i[a * 16 + lane] : -2;
    const float cq = lane < 16 ? qprob[a * 16 + lane] : 0.0f;
    float qx = 0.0f;
    for (ushort k = 0; k < 16; ++k) {
      const int c = simd_shuffle(cid, k);
      const float cv = simd_shuffle(cq, k);
      if (c == x) qx = cv;
    }
    w = x >= 0 ? max(0.0f, px - qx) : 0.0f;
    if (!(simd_sum(w) > 0.0f)) w = px;  // p_a <= q_a everywhere (rejection only by rounding)
  }
  const float cum = simd_prefix_inclusive_sum(w);
  const float tot = simd_shuffle(cum, 31);
  const float u = dfl_uniform(p.seed, p.P + 1 + a, kDflSaltRes) * tot;
  uint sel = simd_min(cum > u && w > 0.0f ? lane : 32u);
  if (sel == 32u) sel = uint(simd_max(w > 0.0f ? int(lane) : -1));  // u above the rounded total
  const int y = simd_shuffle(x, ushort(sel));
  if (lane == 0) {
    tokens[p.P + 1 + a] = y;
    res[0] = int(a);
    res[1] = y;
    gst[0] = p.P + a + 1;
    gst[1] = a + 1;
    gst[2] = p.P;
  }
}

// Dynamic verify depth (MoE targets): stop verifying once the drafter is unlikely to have been right so far.
//  T > 0: verify proposal p iff prod_{j <= p} s_j >= tau, s_j = sum_k softmax(S_j)_k^2 (the chance that two draws
//         from the walk's distribution at j agree: a proxy for q_j's acceptance). Every s_j is a property of
//         proposal j's distribution, fixed before d_j is drawn, so the decision about d_p never depends on d_p
//         itself: given the accepted prefix d_0 .. d_{p-1} it is fixed, and each position is either an ordinary
//         speculative step or the bonus draw from p_p, so the emitted law stays exactly the target's. (A rule that
//         used c_p = the probability of d_p itself would skip unlikely draws of q_p and draw from p_p instead, which
//         is biased; dflash_accept_test's dynamic-depth cases detect that rule.) Replayed on 3490 cycles verified
//         at full depth (4 prompts x 6 seeds, 35B, T = 0.6, M5 Max, with the measured cost per depth), the old rule
//         (rule 2) gives 285.1 tok/s at tau 0.3, this one 297.2 at tau 0.3 (every prompt +2.6-4.6%); with the costs of
//         the MMA-exact expert passes (cheaper rows) 320.4 vs 339.8, tau 0.3 still the best of 0.15-0.5.
//  T = 0: verify proposal p iff prod_{j <= p} c_j >= tau, c_j = softmax(S_j)[k_j] (greedy output is the verify
//         argmax whatever is verified, so the proposal's own confidence may be used).
// Proposal 0 is always verified. kdyn = {K, K + 1}; the verify then routes experts only for the K + 1 live rows
// (moe8_plan row limit) and dfl_accept stops at K. rule: 0 = as above; 1 = prod_{j <= p} c_j at any T (tests only:
// biased at T > 0); 2 = the earlier T > 0 rule, prod_{j < p} q_j(d_j) >= tau (exact; NI_DFLASH_DYN_RULE=2).
// grid 1 x 32.
struct DflDepthParams {
  uint kmax;
  float tau, temp;
  uint rule;
};
[[kernel]] void dfl_depth(const device float* walk_s [[buffer(0)]], const device int* walk_k [[buffer(1)]],
                          const device float* qprob [[buffer(2)]], device uint* kdyn [[buffer(3)]],
                          constant DflDepthParams& p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
  const bool sampled = p.temp > 0.0f;
  float c = 1.0f;  // lane j < 7: the factor of proposal j
  if (lane < 7) {
    const int kj = walk_k[lane];
    if (sampled && p.rule == 2) {
      c = qprob[lane * 16 + kj];
    } else {
      float mx = -INFINITY;
      for (uint k = 0; k < 16; ++k) mx = max(mx, walk_s[lane * 16 + k]);
      float z = 0.0f, z2 = 0.0f;
      for (uint k = 0; k < 16; ++k) {
        const float e = exp(walk_s[lane * 16 + k] - mx);
        z += e;
        z2 += e * e;
      }
      c = sampled && p.rule == 0 ? z2 / (z * z) : exp(walk_s[lane * 16 + kj] - mx) / z;
    }
  }
  const float cum = sampled && p.rule == 2 ? simd_prefix_exclusive_product(c) : simd_prefix_inclusive_product(c);
  const bool keep = lane < p.kmax && (lane == 0 || cum >= p.tau);
  const uint k = simd_min(keep ? 32u : lane);  // leading proposals kept (cum is non-increasing)
  if (lane == 0) {
    kdyn[0] = k;
    kdyn[1] = k + 1;
  }
}

// ---- the verify GEMMs' input layout (verify8.metal "X8 table") ----
// The drafter's tables are row-major half [8][K]; the 8-row verify GEMMs (qmm8pf, qmm8_gu*) read the
// fragment-order table table[(g * 32 + lane) * 16 + 2 f + e] = x[fn + e][64 g + k8(f, fm)] with
// (fm, fn) = the lane's 8x8 position and k8(f, j) = 16 (j / 2) + 8 (f / 4) + (f % 4) + 4 (j % 2). The
// per-64 group sums have the same layout in both ([K/64][8]), so only the table is permuted.
// grid: K / 512 threadgroups x 256 (one simdgroup per group of 64 inputs).
struct DflPermParams {
  uint K;
};
[[kernel]] void dfl_x8_to_v8(const device half* xh [[buffer(0)]], device half* table [[buffer(1)]],
                             constant DflPermParams& p [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg * 8 + sg;
  if (g >= p.K / 64) return;
  const uint q = lane >> 2, fm = (q & 4) | ((lane >> 1) & 3), fn = ((q & 2) << 1) | ((lane & 1) << 1);
  half v[16];
  _Pragma("clang loop unroll(full)")
  for (uint f = 0; f < 8; ++f) {
    const uint kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
    v[2 * f] = xh[fn * p.K + g * 64 + kk];
    v[2 * f + 1] = xh[(fn + 1) * p.K + g * 64 + kk];
  }
  device half4* o = (device half4*)(table + (g * 32 + lane) * 16);
  o[0] = half4(v[0], v[1], v[2], v[3]);
  o[1] = half4(v[4], v[5], v[6], v[7]);
  o[2] = half4(v[8], v[9], v[10], v[11]);
  o[3] = half4(v[12], v[13], v[14], v[15]);
}
// the inverse (debug dumps of the drafter's intermediates in row-major order)
[[kernel]] void dfl_v8_to_x8(const device half* table [[buffer(0)]], device half* xh [[buffer(1)]],
                             constant DflPermParams& p [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg * 8 + sg;
  if (g >= p.K / 64) return;
  const uint q = lane >> 2, fm = (q & 4) | ((lane >> 1) & 3), fn = ((q & 2) << 1) | ((lane & 1) << 1);
  _Pragma("clang loop unroll(full)")
  for (uint f = 0; f < 8; ++f) {
    const uint kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
    xh[fn * p.K + g * 64 + kk] = table[(g * 32 + lane) * 16 + 2 * f];
    xh[(fn + 1) * p.K + g * 64 + kk] = table[(g * 32 + lane) * 16 + 2 * f + 1];
  }
}
