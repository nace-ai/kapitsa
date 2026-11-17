// Memory reserved for the QMR prefill kernel's packed float16 scale/bias pairs.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>

namespace ni {

struct QmrCoeffBudget {
  size_t immutable = 0, scratch = 0;
};

class QmrCoeffHeaders {
 public:
  // MLX may put a projection's packed words and scales in different shard headers.
  void note(const std::string& name, bool u32, std::span<const int64_t> shape) {
    if (shape.size() != 2 || shape[0] <= 64 || shape[1] <= 0) return;
    if (u32 && name.ends_with(".weight"))
      matrices_[name.substr(0, name.size() - 7)].words = {shape[0], shape[1]};
    else if (name.ends_with(".scales"))
      matrices_[name.substr(0, name.size() - 7)].scales = {shape[0], shape[1]};
  }

  QmrCoeffBudget budget(bool cache_trunk) const {
    constexpr size_t limit = size_t(512) << 20;  // immutable images and one reusable scratch buffer together
    size_t images = 0, largest = 0;
    for (const auto& [name, matrix] : matrices_) {
      const auto [n, words] = matrix.words;
      const auto [rows, groups] = matrix.scales;
      // Of the supported affine formats, only 4-bit group-64 has eight packed uint32s per scale.
      if (!groups || n != rows || words % 8 || words / 8 != groups) continue;
      const size_t bytes = n > UINT32_MAX / groups ? limit + 1 : size_t(n) * size_t(groups) * 4;
      const bool gate_up = name.ends_with(".gate_proj") || name.ends_with(".up_proj");
      // Uncached objects and slices can reach the same encoder. Fused gate/up needs both images at once.
      largest = std::max(largest, std::min(bytes, limit) * (gate_up ? 2 : 1));
      if (cache_trunk && immutable_projection(name))
        images = std::min(limit + 1, images + bytes);
    }
    if (images > limit) images = 0;  // The loader only creates complete sets of immutable trunk images.
    return {images, std::min(largest, limit - images)};
  }

 private:
  static bool immutable_projection(const std::string& name) {
    if (name.find("layers.") == std::string::npos) return false;
    for (const char* suffix : {".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.out_proj",
                               ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj",
                               ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".mlp.shared_expert.gate_proj",
                               ".mlp.shared_expert.up_proj", ".mlp.shared_expert.down_proj"})
      if (name.ends_with(suffix)) return true;
    return false;
  }

  struct Shape { int64_t rows = 0, cols = 0; };
  struct Matrix { Shape words, scales; };
  std::map<std::string, Matrix> matrices_;
};

}  // namespace ni
