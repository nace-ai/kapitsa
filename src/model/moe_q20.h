// Routed experts in Q2_0 (ggml type 42; src/kernels/zq2moe.metal): the expert half of Qwen3.8-Flash-Next's MoE for
// decode (one row), verify (R <= 8 rows) and prefill (moe_bucket3's sorted tiles), with the experts as stacked planes
// or as slots of an arena read through a slot table. The router, the top-k, the bucketing, the shared expert at
// prefill and the combine stay with the callers: they do not depend on the experts' format.
// Q2_0: blocks of 64 weights, an fp16 d and 2-bit codes in MLX's order, weight = (code - 1) d; a matrix [N, K] is the
// planes codes [N][K/16] uint32 and d [N][K/64] half (2.25 bits a weight).
#pragma once
#include "metal/gpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ni::q20 {

// Byte offsets of an expert's planes inside an arena slot, and the slot's size rounded up to the arena's 16 KiB pages.
// The three code planes come first (gate, up, down), then the three d planes: Flash-Next's code planes are 25 pages
// each, so a slot's code planes start on page boundaries and an aligned piece of a pack file can be read into them
// directly (src/model/expert_store.cpp).
struct SlotLayout {
  uint32_t gc = 0, gd = 0, uc = 0, ud = 0, dc = 0, dd = 0;
  uint32_t bytes = 0;
};
SlotLayout slot_layout(uint32_t D, uint32_t I);

// The kernels of these paths (prewarm lists): stacked planes, or with slot = true the slot-table twins
std::vector<std::string> kernel_names(bool slot);

// One layer's routed experts. Stacked: the planes, gate / up codes [E][I][D/16] and d [E][I][D/64], down codes
// [E][D][I/16] and d [E][D][I/64]. Arena: chunks a0 and a1 (a1 = a0 for one chunk) and the layer's slot table: E words
// chunk << 28 | page, page counted in 16 KiB from the chunk's start.
struct Experts {
  Buffer gc, gd, uc, ud, dc, dd;
  Buffer a0, a1, slots;
  bool arena = false;
  SlotLayout lay;
};

// The shared expert, 4-bit group 64: (w, scales, biases) of gate, up, down
struct Shared {
  Buffer gw, gs, gb, uw, us, ub, dw, ds, db;
};

struct Shape {
  uint32_t D = 0, I = 0, SI = 0, k = 0, E = 0;
};

// Whether the kernels take these shapes (why: the first reason they don't)
bool shapes_ok(const Shape& sh, std::string* why = nullptr);

// Decode: h (fp32 [(k + 1) * max(I, SI)]) from xn (fp32 [D]) and the routing idx [k]; y [D] += the routed experts and
// the shared expert (slot k) weighted by wts [k + 1].
void enc_decode(Encoder& e, const Experts& x, const Shared& s, const Shape& sh, const Buffer& xn, const Buffer& idx,
                const Buffer& wts, const Buffer& h, const Buffer& y);

// Verify, R <= 8 rows and R * k <= 128: xn [R, D], idx [R, k], wts [R, k + 1], h [R * (k + 1) * max(I, SI)], y [R, D].
// ue (R * k words), umask (R * k words), uslot (8 * R * k bytes) and un (one word): moe_q20_unique's scratch.
void enc_verify(Encoder& e, const Experts& x, const Shared& s, const Shape& sh, int R, const Buffer& xn,
                const Buffer& idx, const Buffer& wts, const Buffer& h, const Buffer& y, const Buffer& ue,
                const Buffer& umask, const Buffer& uslot, const Buffer& un);

// Prefill on moe_bucket3's tiles of bm rows with no shared rows: gate/up of xh (half [T, D]) rows gathered through
// xrows into H (half [pairs, I], sorted rows), down into Y (fp32 [pairs, D], rows by pair through yrows). bm 8: the
// small-T kernels (experts of 1-2 rows by FMAs), bm 32: the qrm-design kernels.
void enc_prefill(Encoder& e, const Experts& x, const Shape& sh, int bm, uint32_t max_tiles, const Buffer& xh,
                 const Buffer& H, const Buffer& Y, const Buffer& tiles, const Buffer& ntiles, const Buffer& xrows,
                 const Buffer& yrows);

}  // namespace ni::q20
