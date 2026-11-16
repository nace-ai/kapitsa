#include "model/sampler.h"

#include "util/log.h"

#include <algorithm>

namespace ni {

uint32_t topk_lists(uint32_t V, uint32_t K) {
  const uint32_t chunk = topk_chunk(K), lists = (V + chunk - 1) / chunk;
  if (size_t(lists) * K > 4096) fatal("top_k %u over %u logits: %u lists, more than stage 2's 4096 candidates", K, V, lists);
  return lists;
}

uint32_t topk_k(const SamplingParams& sp) {
  if (sp.temperature <= 0) return 1;
  return uint32_t(std::clamp(sp.top_k <= 0 ? kTopkUnset : sp.top_k, 1, kTopkMax));
}

}  // namespace ni
