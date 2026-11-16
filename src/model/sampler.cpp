#include "model/sampler.h"

#include "model/kparams.h"
#include "util/log.h"

namespace ni {

uint32_t topk_k(const SamplingParams& sp) {
  if (sp.temperature <= 0) return 1;
  if (full_vocab(sp)) fatal("top_k %d reached the %d-candidate sampler: it is sampled over the whole vocabulary", sp.top_k,
                            kTopkMax);
  return uint32_t(sp.top_k);
}

uint32_t topk_lists(uint32_t V, uint32_t K) {
  const uint32_t chunk = topk_chunk(K), lists = (V + chunk - 1) / chunk;
  if (size_t(lists) * K > 4096) fatal("top_k %u over %u logits: %u lists, more than stage 2's 4096 candidates", K, V, lists);
  return lists;
}

void enc_sample_full(Encoder& e, const Buffer& logits, uint32_t V, const Buffer& tokens, int out_index,
                     const Buffer& sprob, const SamplingParams& sp, uint32_t step) {
  static const Pipeline pl = Gpu::get().pipeline("sample_full");
  const kp::SampleFull p{V, sp.top_k > 0 ? uint32_t(sp.top_k) : 0u, sp.temperature, sp.top_p, sp.seed, step,
                         uint32_t(out_index)};
  e.use(pl);
  e.buf(0, logits);
  e.buf(1, tokens);
  e.buf(2, sprob);
  e.val(3, p);
  e.dispatch(1, 1, 1, 1024);
}

}  // namespace ni
