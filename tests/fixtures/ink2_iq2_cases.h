#pragma once
#include "ink2_iq2_reference.h"
#include <vector>

namespace ink2_iq2_cases {
inline void fill_xs(uint8_t* b, uint16_t half, unsigned pattern) {
  std::memcpy(b, &half, 2);
  for (unsigned group = 0; group < 8; ++group) {
    const unsigned low = (pattern + group * 3) & 15u;
    const unsigned high = (low + 7) & 15u;
    b[66 + group] = uint8_t(low | (high << 4));
    for (unsigned lane = 0; lane < 4; ++lane) {
      const unsigned position = pattern + group * 4 + lane;
      const unsigned index = (97 * position + 11) & 511u;
      const unsigned signs = (73 * position + 29) & 127u;
      const uint16_t entry = uint16_t(index | (signs << 9));
      std::memcpy(b + 2 + 2 * (4 * group + lane), &entry, 2);
    }
  }
}

inline void decode_xs(const uint8_t* bytes, float* out, size_t count) {
  std::vector<ink2_iq2_reference::block_iq2_xs> blocks(count);
  std::memcpy(blocks.data(), bytes, count * 74);
  ink2_iq2_reference::dequantize_row_iq2_xs(blocks.data(), out, int64_t(count * 256));
}
}  // namespace ink2_iq2_cases
