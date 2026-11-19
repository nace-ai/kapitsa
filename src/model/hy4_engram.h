// Hy4-flash's Engram memory, the CPU half: the hashed n-gram ids of each position, the rows of the layer's table
// (MLX affine 4-bit group 32, 9.9 GB per layer, mapped read-only and read by page: it never enters GPU memory), the
// fp8 e4m3 round trip and the per-head RMSNorm. The result, [R, 1536] per pass, is what the GPU's Engram block
// projects. Reference: transformers' modeling_hy_v4_flash.py (revision c2ec8505), HYV4FlashEngramLayer up to and
// including lookup_per_head_norms, as the reference logits ran it (torch 2.15 on CUDA, bf16).
#pragma once
#include "model/ngram.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ni::hy4 {

class EngramCpu {
 public:
  static constexpr int kHeadDim = 96;  // engram_n_embed_per_ngram / engram_n_head_per_ngram

  // dir: the MLX pack folder (model.safetensors.index.json names each tensor's file); text_config: config.json's
  // "text_config", which must carry the converter's engram_hash_multipliers (NumPy's draw, not redone here);
  // layer_id: the 1-based config key (2, 12, 22, 32), whose tensors are model.layers.{layer_id - 1}.engram_layer.*
  EngramCpu(const std::string& dir, const nlohmann::json& text_config, int layer_id);

  int heads() const { return int(mod_.size()); }        // 16: (max_ngram_size - 1) * n_head_per_ngram
  int width() const { return heads() * kHeadDim; }     // 1536
  const NgramTable& table() const { return table_; }

  // Positions pos0 .. pos0 + R - 1 of toks (toks[0 .. pos0 + R) valid): out fp32 [R, width()] = the heads' rows in
  // hash order (n-gram-major), each after the bf16 read, the fp8 round trip and its RMSNorm, all bf16-exact values;
  // mask[r] = the reference's engram_mask (not a pad, image or video token). A masked row is zeros: the caller adds
  // nothing there. Thread-safe for concurrent calls on different outputs.
  void rows(const int32_t* toks, int pos0, int R, float* out, uint8_t* mask) const;

  // Table rows (offset included) of position pos: heads() values. Positions before 0 read the pad id.
  void ids(const int32_t* toks, int pos, int64_t* out) const;
  bool owns_ngram(int32_t tok) const { return tok != pad_ && tok != image_ && tok != video_; }

  // One row's bf16 values -> its fp8 round trip and head h's norm, in place (exposed for the check tool)
  void finish_head(float* x, int h) const;

 private:
  NgramTable table_;
  std::vector<int32_t> lookup_;  // token -> compressed id (lookup_table)
  std::vector<int64_t> mod_, off_;
  std::vector<float> gamma_;      // [heads, kHeadDim], the norms' bf16 weights
  int64_t mult_[8] = {};
  int ngram_ = 3, per_ngram_ = 8;
  int32_t pad_ = 120002, image_ = 120120, video_ = 120124;
};

}  // namespace ni::hy4
