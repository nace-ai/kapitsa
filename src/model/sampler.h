// A request's top-k on the GPU's two-pass sampler (kapitsa#1286).
//
// The two-pass top-k sampler (topk_stage1 + sample_stage2 and their _rows forms, sample.metal / verify.metal) keeps at
// most kTopkMax candidates per row: stage 2 holds 4 per thread of its 1024 (4096) over stage 1's lists, topk_lists(),
// and the speculative acceptance (dfl_dist_rows, dfl_accept: two candidates per simdgroup lane) 64. So top_k 1..64
// is sampled exactly, with every speculative loop. top_k <= 0 still means 32 and a top_k above 64 is clamped to 64.
#pragma once
#include "model/model.h"

#include <cstdint>

namespace ni {

constexpr int kTopkMax = 64;
// what a request with top_k <= 0 decodes with (as before kapitsa#1286)
constexpr int kTopkUnset = 32;

// Stage 1's chunk for K candidates per list: 2048 logits (256 threads x 8) up to K = 32, as before kapitsa#1286; 4096
// (512 threads) above, so a 248K vocabulary's lists hold 61 x 64 <= 4096 for stage 2. The same top-K either way (every
// list keeps its chunk's K best; stage 2 picks by value desc, id asc).
inline uint32_t topk_chunk(uint32_t K) { return K <= 32 ? 2048u : 4096u; }
inline uint32_t topk_threads(uint32_t K) { return topk_chunk(K) / 8; }
// stage 1's lists for a V-logit row (fatal when lists x K is more than stage 2's 4096)
uint32_t topk_lists(uint32_t V, uint32_t K);

// The two-pass sampler's candidates per row: top_k (<= 0: 32; above 64: 64), 1 when greedy.
uint32_t topk_k(const SamplingParams& sp);

}  // namespace ni
