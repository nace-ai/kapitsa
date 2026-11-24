#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "gguf/gguf.h"
#include "kernels/inkling2/gguf_blocks.h"

#include <vector>

namespace ni::ink2 {
struct ExpertSource {
  const gguf::Tensor* weight;
  std::vector<float> outer;
  bool native_nvfp4() const { return weight->type == gguf::kNVFP4; }
};
struct ExpertSources { ExpertSource gu, down; };

inline ExpertSources read_expert_sources(const gguf::File& file, const std::string& prefix,
                                         size_t H, size_t I, size_t E) {
  auto require = [&](const std::string& name, std::initializer_list<uint64_t> shape) -> const gguf::Tensor& {
    const auto* t = file.find(name);
    if (!t || t->ne != std::vector<uint64_t>(shape))
      throw gguf::Error("Inkling2: missing or unexpected shape: " + name);
    return *t;
  };
  const auto& gu = require(prefix + "ffn_gate_up_exps.weight", {H, 2 * I, E});
  const auto& down = require(prefix + "ffn_down_exps.weight", {I, H, E});
  auto read = [&](const gguf::Tensor& weight, const char* scale_name) {
    ExpertSource source{&weight, {}};
    if (source.native_nvfp4()) {
      if (weight.ne[0] % 64) throw gguf::Error("Inkling2: unaligned NVFP4 row: " + weight.name);
    } else if (!ink2_gguf::block_bytes(weight.type) || weight.ne[0] % ink2_gguf::block_values(weight.type)) {
      throw gguf::Error("Inkling2: unsupported block type or row: " + weight.name);
    }
    if (source.native_nvfp4() || file.find(prefix + scale_name)) {
      const auto& scale = require(prefix + scale_name, {E});
      if (scale.type != gguf::kF32) throw gguf::Error("Inkling2 expert outer scales must be F32");
      source.outer.resize(E);
      file.read(scale, 0, source.outer.data(), E * sizeof(float));
    }
    return source;
  };
  return {read(gu, "ffn_gate_up_exps.scale"), read(down, "ffn_down_exps.scale")};
}

// GGUF gate rows precede up rows; native kernels consume gate/up pairs.
inline size_t native_gu_row(size_t gguf_row, size_t intermediate) {
  return 2 * (gguf_row % intermediate) + gguf_row / intermediate;
}
inline void unpack_nvfp4_row(const uint8_t* src, size_t K, uint8_t* q, uint8_t* scales) {
  for (size_t block = 0; block < K / 64; ++block) {
    const uint8_t* b = src + block * 36;
    for (size_t s = 0; s < 4; ++s) scales[block * 4 + s] = b[s] == 127 ? 0 : (b[s] & 127);
    for (size_t group = 0; group < 4; ++group)
      for (size_t pair = 0; pair < 8; ++pair) {
        const size_t i = 2 * pair;
        const uint8_t* v = b + 4 + group * 8;
        const unsigned shift = i < 8 ? 0 : 4;
        q[block * 32 + group * 8 + pair] = uint8_t(((v[i % 8] >> shift) & 15) |
                                                                (((v[(i + 1) % 8] >> shift) & 15) << 4));
      }
  }
}
inline void unpack_q8_row(const uint8_t* src, size_t K, uint8_t* q, uint16_t* scales) {
  for (size_t block = 0; block < K / 32; ++block) {
    std::memcpy(scales + block, src + block * 34, 2);
    std::memcpy(q + block * 32, src + block * 34 + 2, 32);
  }
}
}  // namespace ni::ink2
