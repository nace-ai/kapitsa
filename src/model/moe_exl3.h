// Routed experts in EXL3 (turboderp's trellis format; src/kernels/exl3moe.metal): the expert half of
// Qwen3.8-Flash-Next's MoE in Sushi's packs (beamster/Qwen3.8-Flash-Next-Sushi-*), for decode / verify (R <= 8 rows)
// and prefill (moe_bucket3's sorted tiles of 32 rows). The router, the top-k, the bucketing, the shared expert and the
// combine stay with the callers, as for Q2_0 (src/model/moe_q20.h).
// A projection [in, out] of expert e is three tensors: trellis uint16 [E, in/16, out/16, n] (n halfwords a 16 x 16 tile:
// n / 16 bits a weight), suh f16 [E, in] and svh f16 [E, out]; y = svh * H128(W^T H128(suh * x)). config.json's
// expert_quant names the codebook (MCG only here) and the codeword window ({"format": "exl3", "codebook": "mcg",
// "window": 15}).
#pragma once
#include "metal/gpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ni::exl3 {

// One projection of a layer's experts: the stacked trellis, suh and svh, and its rate (halfwords a tile)
struct Proj {
  Buffer t, suh, svh;
  uint32_t n = 0;
};

struct Experts {
  Proj g, u, d;
};

struct Shape {
  uint32_t D = 0, I = 0, k = 0, E = 0;
  uint32_t window = 16;  // codeword bits (8..16)
};

// Whether the kernels take these shapes and rates (why: the first reason they don't)
bool shapes_ok(const Shape& sh, const Experts& x, std::string* why = nullptr);

// The rate of a trellis tensor's shape [E, in/16, out/16, n]: n, or 0 when it is not one the kernels read (even, 16-64)
uint32_t rate_of(const std::vector<int64_t>& shape);

// The kernels of these paths (prewarm lists)
std::vector<std::string> kernel_names();

// Decode / verify, R <= 8 rows: x (fp32 [R, D]), the routing idx [R, k] and wts [R, k + 1] (wts[r, k]: the shared
// expert's weight), ys the shared expert's output (fp32 [R, D]); y [R, D] += the routed sum and wts[r, k] ys. Scratch:
// hg, hu (fp32, R k I), md (half, R k I), yd (fp32, R k D).
void enc_rows(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xn, const Buffer& idx,
              const Buffer& wts, const Buffer& ys, const Buffer& hg, const Buffer& hu, const Buffer& md,
              const Buffer& yd, const Buffer& y);

// Prefill on moe_bucket3's tiles of 32 rows with no shared rows (P = T k pairs, idx [P] each pair's expert): xh (half
// [T, D]) rows by token (xrows) through gate / up into hg / hu (fp32, sorted rows), md (half [P, I], sorted rows), down
// into yd (fp32 [P, D], rows by pair through yrows), each pair's output transform applied: moe_combine sums yd.
void enc_prefill(Encoder& e, const Experts& x, const Shape& sh, uint32_t P, uint32_t max_tiles, const Buffer& xh,
                 const Buffer& tiles, const Buffer& ntiles, const Buffer& xrows, const Buffer& yrows, const Buffer& idx,
                 const Buffer& hg, const Buffer& hu, const Buffer& md, const Buffer& yd);

// The rows of moe_bucket3's tiles enc_prefill takes
constexpr int kTileRows = 32;

// ---- CPU reference (tools/exl3_test.cpp, tools/exl3_check.mm) ----
// The MCG codebook's value of a codeword (already masked to the window), as f16 bits
uint16_t mcg(uint32_t cw);
// The 256 codewords of a tile (n halfwords at `words`) in stream order
void unpack_tile(const uint16_t* words, uint32_t n, uint16_t* cw);
// Codeword i's place in the 16 x 16 tile, row (input) * 16 + column (output)
uint32_t perm(uint32_t i);
// A tile's weights, [16 in][16 out] row-major, the codewords masked to `window` bits
void decode_tile(const uint16_t* words, uint32_t n, uint32_t window, float* w);
// H128 of v (128 values) in place, scaled by 1 / sqrt(128)
void h128(float* v);
// One projection of one expert for one input (in values): y (out values), with the reference's f16 roundings
void project(const uint16_t* trellis, const uint16_t* suh, const uint16_t* svh, uint32_t in, uint32_t out, uint32_t n,
             uint32_t window, const float* x, float* y);

}  // namespace ni::exl3
