#pragma once
#include <cstdint>
#include <cstring>

namespace ni {

inline float bf16_to_f32(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  // Keep NaNs out of the rounding carry, which can turn them into infinity or zero.
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u); // quiet NaN
  u += 0x7fff + ((u >> 16) & 1); // round to nearest even
  return uint16_t(u >> 16);
}

} // namespace ni
