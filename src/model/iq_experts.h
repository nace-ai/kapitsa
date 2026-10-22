// Routed experts of GGUF imports kept in llama.cpp's IQ blocks (src/kernels/wfmt_iq.metal), as Flint loads them (and
// Qwen35::load_iq_experts, which does the same): each stacked matrix read from the WeightStore (where it is deferred),
// moved into the kernels' layout by parallel threads (gguf::iq_repack: the same bytes) and locked in memory as it is
// written.
#pragma once
#include "gguf/quant.h"
#include "loader/safetensors.h"
#include "metal/gpu.h"

#include <string>
#include <vector>

namespace ni {

// mlock of weight buffers, as Qwen35::load_iq_experts locks them (its comment has the measurements: without it, a 16 GB
// Mac swapped the freshly written experts and the first prompt paid it back). NI_MLOCK=0 leaves them pageable; a lock
// refused (past the wire limit) is logged and the load goes on without one.
class WeightLock {
 public:
  WeightLock();
  void wire(const Buffer& b);
  size_t locked() const { return locked_; }

 private:
  bool on_ = true;
  size_t locked_ = 0;
};

// The deferred tensor `name` of ws, `rows` rows of K values in IQ type `type` (the GGUF's blocks), in the kernels'
// layout: a new (untracked) GPU buffer, its sections in *lay. raw: scratch for the read.
Buffer iq_read_matrix(const WeightStore& ws, const std::string& name, uint32_t type, size_t rows, size_t K,
                      gguf::IqLayout* lay, std::vector<uint8_t>& raw);

}  // namespace ni
