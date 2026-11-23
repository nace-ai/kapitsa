#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ni::ink2 {
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
