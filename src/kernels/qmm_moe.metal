#include "common.h"

// ---- Prefill MoE v2 (opt-in: NI_MOE_FUSED): grouped expert GEMMs without the fp32 round trip ----
// Same arithmetic per output element as qmm_tile (same dequantized halfs, same MMA order over K),
// restructured:
//  * gate and up share one X tile per K step; the epilogue applies SwiGLU and writes half h in
//    sorted-row order, so the down GEMM reads its rows contiguously (no hg/hu, no swiglu pass).
//  * every fragment loop is fully unrolled. qmm_tile's store loop is left rolled by the compiler
//    for 32-row tiles (FM=2), which moves the accumulators to stack memory: 2304 B/thread of spill
//    for m32_n64, 4352 for m32_n128 (the 2-3 TFLOP/s "tile anomaly"; m16/m64 tiles do not spill).
//  * row fragments past the tile's last row are skipped (uniform per simdgroup), so 64-row tiles
//    pay MMAs only for the rows they hold. Fragments alternate between the two simdgroup rows
//    (fragment f of row sgm = tile rows (2f+sgm)*8..) to balance partial tiles.
//  * each thread dequantizes 8 weights into one 16-byte threadgroup store; X moves as 16-byte
//    loads; gathered X row offsets are resolved once per tile, not per K step.
//  * results leave registers directly (simdgroup 8x8 lane layout: lane holds row
//    (lane/4 & 4) + (lane/2) % 4, columns 2*(lane/4 & 2) + 2*(lane % 2) and +1).
//  * the shared expert can ride along as expert index `shared_e` with its own weight buffers.
// Requirements (checked on the host): 4-bit weights, N % BN == 0, K % 32 == 0, group size % 32 == 0.

struct Qmm2Params {
  uint N;                // output columns per projection
  uint K;
  uint x_ld;             // halfs per X row
  uint y_ld;             // elements per output row
  uint gs;               // quantization group size
  uint w_expert_stride;  // words between stacked experts
  uint s_expert_stride;  // scale/bias elements between stacked experts
  uint shared_e;         // tiles with this expert index use the shared-expert weights
};

constant constexpr int Q2_BK = 32;
constant constexpr int Q2_LD = Q2_BK + 8;  // threadgroup row stride in halfs (80 B: conflict-free 8x8 loads)

// acc[p][f] += A_f * B_p^T over one K step for the first FL row fragments (FL <= FM).
template <int FL, int BN, int NP, int FM>
NI_INLINE void q2_mma(thread simdgroup_float8x8 (&acc)[NP][FM][BN / 16], threadgroup const half* xs,
                      threadgroup const half* ws, uint sgm, uint sgn) {
  constexpr int FN = BN / 16;
  for (int kk = 0; kk < Q2_BK; kk += 8) {
    simdgroup_half8x8 a[FL], b[NP][FN];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FL; ++f) simdgroup_load(a[f], xs + (2 * f + sgm) * 8 * Q2_LD + kk, Q2_LD);
    _Pragma("clang loop unroll(full)")
    for (int p = 0; p < NP; ++p)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < FN; ++j)
        simdgroup_load(b[p][j], ws + (p * BN + sgn * (BN / 2) + j * 8) * Q2_LD + kk, Q2_LD, ulong2(0, 0), true);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FL; ++f)
      _Pragma("clang loop unroll(full)")
      for (int p = 0; p < NP; ++p)
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[p][f][j], a[f], b[p][j], acc[p][f][j]);
  }
}

// acc[p] += X[rows, K] * Wp[n0 .. n0+BN, K]^T for NP matrices sharing X (4 simdgroups, 2x2 layout).
// X row r of the tile is xrows[m0 + r] with GATHER, else m0 + r. fm_live: row fragments this
// simdgroup computes. xs: BM*Q2_LD halfs, ws: NP*BN*Q2_LD halfs, both 16-byte aligned.
// Register notes (M4 Pro, from the pipeline's compiler stats): offsets that need no device load are
// recomputed per K step (cheaper than holding them); gathered X offsets are kept.
template <int BM, int BN, int NP, bool GATHER>
NI_INLINE void q2_tile(const device half* X, const device uint* xrows, uint x_ld, uint m0, uint rows,
                       const device uint32_t* const thread* W, const device bfloat* const thread* S,
                       const device bfloat* const thread* B, uint n0, uint K, uint gs,
                       thread simdgroup_float8x8 (&acc)[NP][BM / 16][BN / 16], threadgroup half* xs,
                       threadgroup half* ws, uint tid, uint sgm, uint sgn, int fm_live) {
  constexpr int FM = BM / 16, FN = BN / 16;
  constexpr int XL = (BM * 4 + 127) / 128;  // 16-byte X loads per thread per K step
  constexpr int WL = (BN * 4 + 127) / 128;  // packed words per thread per K step (per matrix)
  _Pragma("clang loop unroll(full)")
  for (int p = 0; p < NP; ++p)
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FM; ++f)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < FN; ++j) acc[p][f][j] = simdgroup_float8x8(0);

  // X: thread i loads 8 halfs (chunk i % 4) of tile row i / 4; gathered row offsets in uint4 units
  const device uint4* X4 = (const device uint4*)X;
  uint xoff[XL];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < XL; ++j) {
    uint i = tid + j * 128, r = i / 4;
    xoff[j] = GATHER && i < BM * 4 && r < rows ? xrows[m0 + r] * (x_ld / 8) + i % 4 : 0;
  }
  const uint wpr = K / 8, gpr = K / gs;

  for (uint k0 = 0; k0 < K; k0 += Q2_BK) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < XL; ++j) {
      uint i = tid + j * 128, r = i / 4;
      if (i < BM * 4) {
        uint o = GATHER ? xoff[j] : (m0 + r) * (x_ld / 8) + i % 4;
        *(threadgroup uint4*)(xs + r * Q2_LD + (i % 4) * 8) = r < rows ? X4[o + k0 / 8] : uint4(0);
      }
    }
    // W: thread i dequantizes word i % 4 (8 values) of tile row i / 4, for every matrix
    const uint g = k0 / gs;  // one group per row per step (gs is a multiple of 32)
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < WL; ++j) {
      uint i = tid + j * 128, n = n0 + i / 4;
      if (i < BN * 4) {
        _Pragma("clang loop unroll(full)")
        for (int p = 0; p < NP; ++p) {
          uint32_t word = W[p][n * wpr + k0 / 8 + i % 4];
          float sc = float(S[p][n * gpr + g]), bi = float(B[p][n * gpr + g]);
          half4 lo, hi;
          for (int q = 0; q < 4; ++q) {
            lo[q] = half(sc * float((word >> (4 * q)) & 15u) + bi);
            hi[q] = half(sc * float((word >> (4 * q + 16)) & 15u) + bi);
          }
          *(threadgroup uint4*)(ws + (p * BN + i / 4) * Q2_LD + (i % 4) * 8) =
              uint4(as_type<uint2>(lo), as_type<uint2>(hi));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // MMAs for the live row fragments only: one branch per K step, uniform per simdgroup (a
    // predicate per fragment inside the loop costs ~20 registers at BM=64)
    if (FM >= 4 && fm_live == 4) q2_mma<(FM < 4 ? FM : 4), BN, NP>(acc, xs, ws, sgm, sgn);
    else if (FM >= 3 && fm_live == 3) q2_mma<(FM < 3 ? FM : 3), BN, NP>(acc, xs, ws, sgm, sgn);
    else if (FM >= 2 && fm_live == 2) q2_mma<(FM < 2 ? FM : 2), BN, NP>(acc, xs, ws, sgm, sgn);
    else if (fm_live == 1) q2_mma<1, BN, NP>(acc, xs, ws, sgm, sgn);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Row fragments of simdgroup row sgm that hold tile rows (fragment f covers rows (2f+sgm)*8..+8).
template <int BM>
NI_INLINE int q2_live(uint rows, uint sgm) {
  int nf = int(rows + 7) / 8;
  return clamp((nf - int(sgm) + 1) / 2, 0, BM / 16);
}

// Lane position inside an 8x8 accumulator: row, first of two adjacent columns.
NI_INLINE uint2 q2_lane_rc(uint lane) {
  uint q = lane / 4;
  return uint2((q & 4) + (lane / 2) % 4, (q & 2) * 2 + (lane % 2) * 2);
}

// This lane's two accumulator elements (q2_lane_rc gives their position).
NI_INLINE float2 q2_elems(thread simdgroup_float8x8& m) {
  return float2(m.thread_elements()[0], m.thread_elements()[1]);
}

// Fused expert gate/up: H[m, n] = half(silu(X W_gate^T) * (X W_up^T)) for sorted pair rows m of
// each tile; X rows are gathered through xrows (token of each sorted pair). grid: (N/BN, max tiles),
// 128 threads. tiles[i] = (expert, row_begin, row_end), ntiles[0] = valid tiles.
#define NI_QMM_MOE_GU(NAME, BM, BN)                                                                        \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uint32_t* Wg [[buffer(1)]],         \
                       const device bfloat* Sg [[buffer(2)]], const device bfloat* Bg [[buffer(3)]],         \
                       const device uint32_t* Wu [[buffer(4)]], const device bfloat* Su [[buffer(5)]],       \
                       const device bfloat* Bu [[buffer(6)]], device half* H [[buffer(7)]],                  \
                       constant Qmm2Params& p [[buffer(8)]], const device uint* tiles [[buffer(9)]],         \
                       const device uint* ntiles [[buffer(10)]], const device uint* xrows [[buffer(11)]],    \
                       const device uint32_t* SWg [[buffer(12)]], const device bfloat* SSg [[buffer(13)]],   \
                       const device bfloat* SBg [[buffer(14)]], const device uint32_t* SWu [[buffer(15)]],   \
                       const device bfloat* SSu [[buffer(16)]], const device bfloat* SBu [[buffer(17)]],     \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],  \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    if (tg.y >= ntiles[0]) return;                                                                         \
    threadgroup uint4 xs4[BM * Q2_LD / 8];                                                                 \
    threadgroup uint4 ws4[2 * BN * Q2_LD / 8];                                                             \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;         \
    const bool sh = e == p.shared_e;                                                                       \
    const uint wo = sh ? 0 : e * p.w_expert_stride, so = sh ? 0 : e * p.s_expert_stride;                   \
    const device uint32_t* W[2] = {(sh ? SWg : Wg) + wo, (sh ? SWu : Wu) + wo};                            \
    const device bfloat* S[2] = {(sh ? SSg : Sg) + so, (sh ? SSu : Su) + so};                              \
    const device bfloat* B[2] = {(sh ? SBg : Bg) + so, (sh ? SBu : Bu) + so};                              \
    const uint sgm = sg / 2, sgn = sg % 2, n0 = tg.x * BN;                                                 \
    const int live = q2_live<BM>(rows, sgm);                                                               \
    simdgroup_float8x8 acc[2][BM / 16][BN / 16];                                                           \
    q2_tile<BM, BN, 2, true>(X, xrows, p.x_ld, m0, rows, W, S, B, n0, p.K, p.gs, acc, (threadgroup half*)xs4,    \
                       (threadgroup half*)ws4, tid, sgm, sgn, live);                                       \
    const uint2 rc = q2_lane_rc(lane);                                                                     \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int f = 0; f < BM / 16; ++f) {                                                                    \
      uint m = (2 * f + sgm) * 8 + rc.x;                                                                   \
      if (f < live && m < rows) {                                                                          \
        device half* hp = H + (m0 + m) * p.y_ld + n0 + sgn * (BN / 2) + rc.y;                              \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int j = 0; j < BN / 16; ++j) {                                                                \
          float2 gv = q2_elems(acc[0][f][j]), uv = q2_elems(acc[1][f][j]);                                 \
          *(device half2*)(hp + j * 8) = half2(half(silu(gv.x) * uv.x), half(silu(gv.y) * uv.y));          \
        }                                                                                                  \
      }                                                                                                    \
    }                                                                                                      \
  }

// Grouped GEMM to fp32 rows: Y[yrows[m], n] = H[m, :] W_e[n, :]^T for sorted rows m (H rows are
// contiguous per tile). Same grid/tiles as the gate/up kernel.
#define NI_QMM_MOE_DN(NAME, BM, BN)                                                                        \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uint32_t* Wd [[buffer(1)]],         \
                       const device bfloat* Sd [[buffer(2)]], const device bfloat* Bd [[buffer(3)]],         \
                       device float* Y [[buffer(4)]], constant Qmm2Params& p [[buffer(5)]],                  \
                       const device uint* tiles [[buffer(6)]], const device uint* ntiles [[buffer(7)]],      \
                       const device uint* yrows [[buffer(8)]], const device uint32_t* SWd [[buffer(9)]],     \
                       const device bfloat* SSd [[buffer(10)]], const device bfloat* SBd [[buffer(11)]],     \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],  \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    if (tg.y >= ntiles[0]) return;                                                                         \
    threadgroup uint4 xs4[BM * Q2_LD / 8];                                                                 \
    threadgroup uint4 ws4[BN * Q2_LD / 8];                                                                 \
    const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2] - m0;         \
    const bool sh = e == p.shared_e;                                                                       \
    const device uint32_t* W[1] = {sh ? SWd : Wd + e * p.w_expert_stride};                                 \
    const device bfloat* S[1] = {sh ? SSd : Sd + e * p.s_expert_stride};                                   \
    const device bfloat* B[1] = {sh ? SBd : Bd + e * p.s_expert_stride};                                   \
    const uint sgm = sg / 2, sgn = sg % 2, n0 = tg.x * BN;                                                 \
    const int live = q2_live<BM>(rows, sgm);                                                               \
    simdgroup_float8x8 acc[1][BM / 16][BN / 16];                                                           \
    q2_tile<BM, BN, 1, false>(X, nullptr, p.x_ld, m0, rows, W, S, B, n0, p.K, p.gs, acc, (threadgroup half*)xs4,  \
                       (threadgroup half*)ws4, tid, sgm, sgn, live);                                       \
    const uint2 rc = q2_lane_rc(lane);                                                                     \
    _Pragma("clang loop unroll(full)")                                                                     \
    for (int f = 0; f < BM / 16; ++f) {                                                                    \
      uint m = (2 * f + sgm) * 8 + rc.x;                                                                   \
      if (f < live && m < rows) {                                                                          \
        device float* yp = Y + yrows[m0 + m] * p.y_ld + n0 + sgn * (BN / 2) + rc.y;                        \
        _Pragma("clang loop unroll(full)")                                                                 \
        for (int j = 0; j < BN / 16; ++j) *(device float2*)(yp + j * 8) = q2_elems(acc[0][f][j]);          \
      }                                                                                                    \
    }                                                                                                      \
  }

NI_QMM_MOE_GU(qmm_moe_gu_b4_m16_n32, 16, 32)
NI_QMM_MOE_GU(qmm_moe_gu_b4_m32_n32, 32, 32)
NI_QMM_MOE_GU(qmm_moe_gu_b4_m64_n32, 64, 32)
NI_QMM_MOE_DN(qmm_moe_dn_b4_m16_n64, 16, 64)
NI_QMM_MOE_DN(qmm_moe_dn_b4_m32_n64, 32, 64)
NI_QMM_MOE_DN(qmm_moe_dn_b4_m64_n64, 64, 64)

// Groups the T*k (token, slot) pairs by expert like moe_bucket, and with S > 0 appends the shared
// expert as expert E over S = T token rows. Outputs per sorted row m: yrows[m] (pair id, or P + t
// for shared rows), xrows[m] (token); tiles[3*i] = (expert, begin, end) of at most BM rows, shared
// tiles last; ntiles[0]. One threadgroup of 1024 threads.
struct Bucket2Params {
  uint P;   // pairs = T * k
  uint k;
  uint E;
  uint BM;
  uint S;   // shared-expert rows (0: none)
};

[[kernel]] void moe_bucket2(const device uint* idx [[buffer(0)]], device uint* yrows [[buffer(1)]],
                            device uint* xrows [[buffer(2)]], device uint* tiles [[buffer(3)]],
                            device uint* ntiles [[buffer(4)]], constant Bucket2Params& p [[buffer(5)]],
                            uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[512];
  threadgroup uint off[513];
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) atomic_fetch_add_explicit(&cnt[idx[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0, nt = 0;
    for (uint e = 0; e < p.E; ++e) {
      uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      off[e] = acc;
      for (uint r = 0; r < c; r += p.BM) {
        tiles[nt * 3 + 0] = e;
        tiles[nt * 3 + 1] = acc + r;
        tiles[nt * 3 + 2] = acc + min(r + p.BM, c);
        ++nt;
      }
      acc += c;
    }
    for (uint r = 0; r < p.S; r += p.BM) {
      tiles[nt * 3 + 0] = p.E;
      tiles[nt * 3 + 1] = p.P + r;
      tiles[nt * 3 + 2] = p.P + min(r + p.BM, p.S);
      ++nt;
    }
    ntiles[0] = nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) {
    uint e = idx[i];
    uint pos = off[e] + atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
    yrows[pos] = i;
    xrows[pos] = i / p.k;
  }
  for (uint t = tid; t < p.S; t += ntg) {
    yrows[p.P + t] = p.P + t;
    xrows[p.P + t] = t;
  }
}

// moe_bucket2 with the per-expert offsets and tiles from parallel prefix sums instead of thread 0's
// loop over the experts (53-94 us per layer, ~1% of a short prefill): same outputs bit for bit (same
// tile order: expert-major, rows ascending; shared tiles last). E <= 512, one threadgroup of 1024.
[[kernel]] void moe_bucket3(const device uint* idx [[buffer(0)]], device uint* yrows [[buffer(1)]],
                            device uint* xrows [[buffer(2)]], device uint* tiles [[buffer(3)]],
                            device uint* ntiles [[buffer(4)]], constant Bucket2Params& p [[buffer(5)]],
                            uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint cnt[512];
  threadgroup uint off[512];
  threadgroup uint sums[2][16];
  threadgroup uint totals[2];
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) atomic_fetch_add_explicit(&cnt[idx[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // exclusive prefix sums of rows and tiles over the experts (threads tid < 512: 16 simdgroups)
  const uint c = tid < p.E ? atomic_load_explicit(&cnt[tid], memory_order_relaxed) : 0u;
  const uint nt = (c + p.BM - 1) / p.BM;
  const uint pc = simd_prefix_exclusive_sum(c), pt = simd_prefix_exclusive_sum(nt);
  if (tid < 512 && lane == 31) {
    sums[0][sg] = pc + c;
    sums[1][sg] = pt + nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const uint a = lane < 16 ? sums[0][lane] : 0u, b = lane < 16 ? sums[1][lane] : 0u;
    const uint ea = simd_prefix_exclusive_sum(a), eb = simd_prefix_exclusive_sum(b);
    if (lane < 16) {
      sums[0][lane] = ea;
      sums[1][lane] = eb;
    }
    if (lane == 31) {
      totals[0] = ea + a;
      totals[1] = eb + b;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.E) {
    const uint o = sums[0][sg] + pc, to = sums[1][sg] + pt;
    off[tid] = o;
    for (uint r = 0; r < nt; ++r) {
      tiles[(to + r) * 3 + 0] = tid;
      tiles[(to + r) * 3 + 1] = o + r * p.BM;
      tiles[(to + r) * 3 + 2] = o + min((r + 1) * p.BM, c);
    }
  }
  const uint nt_routed = totals[1], nt_shared = (p.S + p.BM - 1) / p.BM;
  for (uint r = tid; r < nt_shared; r += ntg) {
    tiles[(nt_routed + r) * 3 + 0] = p.E;
    tiles[(nt_routed + r) * 3 + 1] = p.P + r * p.BM;
    tiles[(nt_routed + r) * 3 + 2] = p.P + min((r + 1) * p.BM, p.S);
  }
  if (tid == 0) ntiles[0] = nt_routed + nt_shared;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) {
    uint e = idx[i];
    uint pos = off[e] + atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
    yrows[pos] = i;
    xrows[pos] = i / p.k;
  }
  for (uint t = tid; t < p.S; t += ntg) {
    yrows[p.P + t] = p.P + t;
    xrows[p.P + t] = t;
  }
}
