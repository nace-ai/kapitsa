#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---- Nemotron-H's 8-row NVFP4 products on the GPU's matrix units (Metal 4 tensor_ops; Apple10 = M5) ----
//
// nh_x4_v8 (nemotron/nh_x_fp4.metal) runs a verify's NVFP4 GEMMs (Lightning: the LM head, the shared expert; the
// DSpark drafter's NVFP4 matrices) as 8x8x8 simdgroup MMAs on the shader ALUs, which costs as much as the weight reads:
// with the weights in the cache (tools/nh_v8t_bench --same) the shared expert's up 3712x2688 takes 15.2 us there
// against 9.6 here. Here per simdgroup 16 weight rows; per 64-input group one matmul2d (M = 16 weight rows, N = 8
// token rows, K = 64) of the decoded weights (half, a cooperative tensor filled from registers: E2M1 x its block's
// UE4M3 scale, exact in half as in nh_x4_v8) by the inputs (a device tensor of the R live rows), accumulated in fp32
// by the matrix unit. The threadgroup's KS simdgroups take the groups in turn (g = sg, sg + KS, ...: together they
// read each row's codes front to back; a contiguous k-slice per simdgroup was 10-25% slower on fresh weights) and
// their sums are added in a fixed order. A token row's outputs depend on that row's inputs only (row-independent, as a
// verify must be); bitwise they differ from nh_x4_v8's (summation order).
//
// Operand layouts (tools/bench_tq probe tq_probe_wl16_h): lane role c = (lane & 1) | ((lane >> 3) & 1) << 1 holds
// inputs 4 c + 16 jb + kk (jb, kk < 4) of the group of weight rows r + 8 j (r = ((lane >> 1) & 3) | ((lane >> 4) &
// 1) << 2) as left elements jb * 8 + j * 4 + kk; destination element j * 4 + kk is token row 4 (lane & 1) + kk of
// weight row r + 8 j (lanes with bit 3 set hold the padding token rows 8-15).
// NVFP4 in layout L0 (byte i of a 16-block = element i | element i + 8 << 4): inputs 4 c .. 4 c + 3 of block jb are
// the low (c < 2) or high (c >= 2) nibbles of the block's word c & 1.
//
// Kernel (NhV8P [[4]], X half [R][x_ld], Y per p.epi as nh_x4_v8; codes [[1]], block scales [[2]]):
//   nh_x4t_v8_l0   grid ceil(N / 16) x (32 x KS): KS simdgroups (1..8) split K
// Requirements: K % 64 == 0.

struct NhV8tP {  // = NhV8P (nemotron/nh_v8.metal)
  uint N, K;
  uint x_ld;
  uint y_ld;
  uint R;
  float gs;
  uint epi;
};

// Y[t][n] per epi: 0 store fp32, 1 add fp32, 2 relu^2 -> half, 3 store half
NI_INLINE void nh_v8t_store(device void* Y, constant NhV8tP& p, uint n, uint t, float v) {
  const size_t o = size_t(t) * p.y_ld + n;
  if (p.epi == 0) ((device float*)Y)[o] = v;
  else if (p.epi == 1) ((device float*)Y)[o] += v;
  else if (p.epi == 2) { v = max(v, 0.0f); ((device half*)Y)[o] = half(v * v); }
  else ((device half*)Y)[o] = half(v);
}

NI_INLINE void nh_x4t_v8_l0_impl(const device half* X, const device uchar* q, const device uchar* s, device void* Y,
                                 constant NhV8tP& p, uint tg, uint sg, uint nsg, uint lane, threadgroup float* red) {
  constexpr int NT = 16, NJ = NT / 8;
  const uint N = p.N, K = p.K, gpr = K / 64, n0 = tg * NT;
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc =
      matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.get_left_input_cooperative_tensor<half, half, float>();
  // the inputs as a device tensor of R rows (the op reads no row >= R)
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
      (device half*)X, dextents<int32_t, 2>(int32_t(K), int32_t(p.R)), array<int32_t, 2>{1, int32_t(p.x_ld)});
  auto Rx0 = tX.slice<64, 8>(0, 0);
  auto acc = op.get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  // a lane's words of row r + 8 j in group g: wp[j][g * 8 + 2 jb] (block jb's word c & 1), the block scales sp[j][g]
  const device uint* wp[NJ];
  const device uint* sp[NJ];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    wp[j] = (const device uint*)(q + size_t(row) * (K / 2)) + (c & 1);
    sp[j] = (const device uint*)(s + size_t(row) * (K / 16));
  }
  const uint sh = 4 * (c >> 1);
  for (uint g = sg; g < gpr; g += nsg) {
    uint w[NJ][4], sw[NJ];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) w[j][jb] = wp[j][g * 8 + 2 * jb];
      sw[j] = sp[j][g];
    }
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NJ; ++j) {
      // the four block scales x 2^15 as two half2 (blocks 0, 2 and 1, 3; UE4M3 bits at half bits 7-14 are its value
      // x 2^-8), the codes as E2M1 x 2^-14: products E2M1 x E4M3 x 2^-7, exact in half
      const half2 sc02 = as_type<half2>((sw[j] & 0x00FF00FFu) << 7) * 32768.0h;
      const half2 sc13 = as_type<half2>((sw[j] & 0xFF00FF00u) >> 1) * 32768.0h;
      _Pragma("clang loop unroll(full)")
      for (int jb = 0; jb < 4; ++jb) {
        // the low / high nibbles of the word; codes 0, 2 and 1, 3 sit 16 bits apart: a half2 each with two masks and
        // shifts (magnitude bits to half bits 9-11, the sign to 15)
        const half sc = (jb & 1 ? sc13 : sc02)[jb >> 1];
        const uint u = w[j][jb] >> sh;
        const half2 h02 = as_type<half2>(((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12)) * sc;
        const half2 h13 = as_type<half2>(((u & 0x07000700u) << 1) | ((u & 0x08000800u) << 4)) * sc;
        const int i = jb * 8 + j * 4;
        Lw[i] = h02.x;
        Lw[i + 1] = h13.x;
        Lw[i + 2] = h02.y;
        Lw[i + 3] = h13.y;
      }
    }
    auto Rx = tX.slice<64, 8>(int32_t(g * 64), 0);
    op.run(Lw, Rx, acc);
  }
  // fixed-order sum of the k-slices (simdgroup 0 adds 1, 2, ... in turn)
  constexpr uint cap = NT / 2;
  if (nsg > 1) {
    if (sg)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < nsg; ++j)
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;  // padding token rows
  const float post = p.gs * 128.0f;
  const uint t0 = 4 * (lane & 1);
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 4; ++kk)
      if (t0 + kk < p.R) nh_v8t_store(Y, p, n, t0 + kk, acc[j * 4 + kk] * post);
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void nh_x4t_v8_l0(
    const device half* X [[buffer(0)]], const device uchar* q [[buffer(1)]], const device uchar* s [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8tP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[7 * 8 * 32];
  nh_x4t_v8_l0_impl(X, q, s, Y, p, tg, sg, nsg, lane, red);
}

// ---- a verify's MoE router and routing beside the shared expert (Lightning: F32 router, no latent) ----
// nh_moe_ru_v8t: threadgroups [0, E) the router's output row tg for the R rows (nh_mv_f32_sk's sums), the rest the
// shared expert's up product (nh_x4t_v8_l0, p.epi). nh_moe_rd_v8t: threadgroup 0 routes the live rows, one simdgroup
// per row (nh_moe_route1's selection), the rest the shared expert's down product (nh_x4t_v8_l0, p.epi). The router's
// and the routing's latency so runs beside the shared expert's weight reads, and no pass waits for another's
// threadgroups. grids: E + ceil(p.N / 16) (ru), 1 + ceil(p.N / 16) (rd) threadgroups x 256 threads (8 simdgroups, as
// enc_gemm's nh_x4t_v8_l0 by default); R <= 8 rows
struct NhRsuP {
  uint E, K;        // experts (router rows), hidden
  uint x_stride;    // floats between xn rows
  uint lg_stride;   // floats between logit rows
  uint R, k, norm;  // rows of the pass, experts per row, renormalize
  float scale, gs;  // routed scale, router tensor scale
};
[[kernel, max_total_threads_per_threadgroup(256)]] void nh_moe_ru_v8t(
    const device half* X [[buffer(0)]], const device uchar* q [[buffer(1)]], const device uchar* s [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8tP& p [[buffer(4)]], const device float* rw [[buffer(5)]],
    const device float* xn [[buffer(6)]], device float* logits [[buffer(7)]], constant NhRsuP& rp [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float red[7 * 8 * 32];
  threadgroup float rred[8][8];
  if (tg >= rp.E) nh_x4t_v8_l0_impl(X, q, s, Y, p, tg - rp.E, sg, nsg, lane, red);
  else nh_mv_f32_rows(rw, xn, logits, rp.K, rp.x_stride, rp.lg_stride, rp.gs, rp.R, tg, tid, sg, lane, ntg, rred);
}
[[kernel, max_total_threads_per_threadgroup(256)]] void nh_moe_rd_v8t(
    const device half* X [[buffer(0)]], const device uchar* q [[buffer(1)]], const device uchar* s [[buffer(2)]],
    device void* Y [[buffer(3)]], constant NhV8tP& p [[buffer(4)]], const device float* logits [[buffer(7)]],
    const device float* bias [[buffer(8)]], device int* eid [[buffer(9)]], device float* ew [[buffer(10)]],
    const device uint* live [[buffer(12)]], constant NhRsuP& rp [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[7 * 8 * 32];
  if (tg > 0) {
    nh_x4t_v8_l0_impl(X, q, s, Y, p, tg - 1, sg, nsg, lane, red);
    return;
  }
  const uint r = sg;
  if (r < rp.R && r < live[0])
    nh_route_row(logits + size_t(r) * rp.lg_stride, bias, rp.E, rp.k, rp.norm, rp.scale, lane, eid + r * rp.k,
                 ew + r * rp.k);
}
