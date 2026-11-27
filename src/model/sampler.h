// A request's top-k on the GPU samplers (kapitsa#1286).
//
// The two-pass top-k sampler (topk_stage1 + sample_stage2 and their _rows forms, sample.metal / verify.metal) keeps at
// most kTopkMax candidates per row: stage 2 holds 4 per thread of its 1024 (4096) over stage 1's lists, topk_lists(),
// and the speculative acceptance (dfl_dist_rows, dfl_accept: two candidates per simdgroup lane) 64. A sampled request whose
// top_k it cannot hold (top_k <= 0: no truncation; above kTopkMax) is honored exactly instead: each token is sampled
// over the whole vocabulary (sample_full), in plain steps, or on Nemotron-H's KFlash tree loop in its verify
// (enc_sample_full_rows: each row at its position's plain RNG step). Greedy requests and top_k 1..kTopkMax take the
// two-pass sampler as before.
#pragma once
#include "metal/gpu.h"
#include "model/model.h"

#include <cstdint>

namespace ni {

constexpr int kTopkMax = 64;
// what a request with no top_k of its own decoded with before kapitsa#1286 (none: 0, clamped to 32)
constexpr int kTopkOmitted = 32;

// Stage 1's chunk for K candidates per list: 2048 logits (256 threads x 8) up to K = 32, as before kapitsa#1286; 4096
// (512 threads) above, so a 248K vocabulary's lists hold 61 x 64 <= 4096 for stage 2. The same top-K either way (every
// list keeps its chunk's K best; stage 2 picks by value desc, id asc).
inline uint32_t topk_chunk(uint32_t K) { return K <= 32 ? 2048u : 4096u; }
inline uint32_t topk_threads(uint32_t K) { return topk_chunk(K) / 8; }
// stage 1's lists for a V-logit row (fatal when lists x K is more than stage 2's 4096)
uint32_t topk_lists(uint32_t V, uint32_t K);

// The request is sampled over the whole vocabulary (sample_full), in plain steps.
inline bool full_vocab(const SamplingParams& sp) {
  return sp.temperature > 0 && (sp.top_k <= 0 || sp.top_k > kTopkMax);
}

// The top_k a request that names none gets from the model's generation_config.json (publisher: its top_k, 0 when it
// has none): the publisher's when the two-pass sampler holds it, else kTopkOmitted, what such a request decoded with
// before kapitsa#1286 (no top_k: 0, clamped to 32). So a request that leaves top_k out keeps the speculative loops it
// had; only a request's own top_k <= 0 or above kTopkMax takes the full-vocabulary path. (Qwen3.5/3.6/3.8 take
// ChatRequest's 20 and Clover decodes greedily: neither reads this.)
inline int omitted_top_k(int publisher) { return publisher >= 1 && publisher <= kTopkMax ? publisher : kTopkOmitted; }

// The two-pass sampler's candidates per row: top_k, 1 when greedy. A full-vocabulary request is routed to sample_full
// before any caller of this; one that still arrives gets kTopkOmitted (top_k <= 0) or kTopkMax and a warning (once),
// never a fatal: serve has no recovery from one.
uint32_t topk_k(const SamplingParams& sp);
// sp with top_k as topk_k() takes it: for a verify whose own sample is not used (Clover's check picks from the logits)
SamplingParams two_pass(const SamplingParams& sp);
// Whether a verify pass (encode_verify / encode_verify8 / encode_verify_chain: the two-pass sampler over each row) may
// sample for sp: false for a full-vocabulary request, which a verify would clamp. The vpf prefill's gate uses it;
// sample_topk_test --cpu pins its rule, and that two_pass(sp) (Clover's check) always passes it.
inline bool verify_pass_ok(const SamplingParams& sp) { return !full_vocab(sp); }

// One logits row (V floats) sampled over the whole vocabulary: tokens[out_index] = the sampled id, sprob[0] = the
// row's largest logit (as sample_stage2 writes). One threadgroup.
void enc_sample_full(Encoder& e, const Buffer& logits, uint32_t V, const Buffer& tokens, int out_index,
                     const Buffer& sprob, const SamplingParams& sp, uint32_t step);
// R consecutive logits rows sampled over the whole vocabulary in one dispatch (a threadgroup per row): row r into
// tokens[out_index + r], at RNG step step + dstep[out_index + r] (dstep: int32 at dstep_off; a KFlash tree verify's
// depths, so each row draws as the plain step at its position would)
void enc_sample_full_rows(Encoder& e, const Buffer& logits, uint32_t V, int R, const Buffer& tokens, int out_index,
                          const Buffer& sprob, const SamplingParams& sp, uint32_t step, const Buffer& dstep,
                          size_t dstep_off);

}  // namespace ni
