#pragma once
// Scalar reads of the GGUF blocks used by Inkling2. Also compiled by the CPU reference test.
#ifdef __METAL_VERSION__
#define INK2_DEVICE device
#define INK2_CONSTANT constant
#else
#include <cstring>
#include "../../gguf/iq_grids.h"
#define INK2_DEVICE
#define INK2_CONSTANT inline constexpr
#endif

namespace ink2_gguf {
#ifndef __METAL_VERSION__
using ni::gguf::iq::kIq2xxs;
using ni::gguf::iq::kIq2xs;
#endif
INK2_CONSTANT int levels[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
inline unsigned u16(const INK2_DEVICE unsigned char* p) { return unsigned(p[0]) | (unsigned(p[1]) << 8); }
inline float f16(const INK2_DEVICE unsigned char* p) {
  const unsigned short h = static_cast<unsigned short>(u16(p));
#ifdef __METAL_VERSION__
  return float(as_type<half>(h));
#else
  _Float16 f;
  std::memcpy(&f, &h, 2);
  return float(f);
#endif
}
inline unsigned block_values(unsigned type) { return (type == 6 || type == 8 || type == 20) ? 32 : 256; }
inline unsigned block_bytes(unsigned type) {
  switch (type) {
    case 6: return 22;
    case 8: return 34;
    case 12: return 144;
    case 13: return 176;
    case 14: return 210;
    case 16: return 66;
    case 17: return 74;
    case 20: return 18;
    case 23: return 136;
    default: return 0;
  }
}
inline unsigned scale6(const INK2_DEVICE unsigned char* p, unsigned j) {
  return j < 4 ? p[j] & 63u : (p[j + 4] & 15u) | ((p[j - 4] >> 6) << 4);
}
inline unsigned min6(const INK2_DEVICE unsigned char* p, unsigned j) {
  return j < 4 ? p[j + 4] & 63u : (p[j + 4] >> 4) | ((p[j] >> 6) << 4);
}
inline float value(const INK2_DEVICE unsigned char* row, unsigned type, unsigned k) {
  const unsigned n = block_values(type), i = k % n;
  const INK2_DEVICE unsigned char* b = row + (k / n) * block_bytes(type);
  switch (type) {
    case 8: {
      const int q = b[2 + i] < 128 ? int(b[2 + i]) : int(b[2 + i]) - 256;
      return f16(b) * float(q);
    }
    case 6: {
      const unsigned q = (b[6 + i % 16] >> (4 * (i / 16))) & 15u;
      const unsigned hi = (b[2 + i / 8] >> (i % 8)) & 1u;
      return f16(b) * float(int(q | (hi << 4)) - 16);
    }
    case 12:
    case 13: {
      const unsigned group = i / 32, lane = i % 32;
      const unsigned qoff = type == 13 ? 48 : 16;
      unsigned q = (b[qoff + 32 * (group / 2) + lane] >> (4 * (group % 2))) & 15u;
      if (type == 13) q |= ((b[16 + lane] >> group) & 1u) << 4;
      const float d = f16(b) * float(scale6(b + 4, group));
      const float m = f16(b + 2) * float(min6(b + 4, group));
      return d * float(q) - m;
    }
    case 14: {
      const unsigned h = i / 128, j = i % 128, group = j / 32, lane = j % 32;
      const unsigned lo = (b[64 * h + (group % 2) * 32 + lane] >> (4 * (group / 2))) & 15u;
      const unsigned hi = (b[128 + 32 * h + lane] >> (2 * group)) & 3u;
      const unsigned sc = b[192 + 8 * h + lane / 16 + 2 * group];
      const int s = sc < 128 ? int(sc) : int(sc) - 256;
      return (f16(b + 208) * float(s)) * float(int(lo | (hi << 4)) - 32);
    }
    case 16: {
      const unsigned group = i / 32, lane = (i % 32) / 8, j = i % 8;
      const INK2_DEVICE unsigned char* a = b + 6 + 8 * group;
      const unsigned aux = u16(a) | (u16(a + 2) << 16);
      unsigned signs = (aux >> (7 * lane)) & 127u;
      unsigned parity = signs ^ (signs >> 4);
      parity ^= parity >> 2;
      parity ^= parity >> 1;
      signs |= (parity & 1u) << 7;
      const unsigned magnitude = unsigned((kIq2xxs[b[2 + 8 * group + lane]] >> (8 * j)) & 255u);
      const float d = f16(b) * (0.5f + float(aux >> 28)) * 0.25f;
      return d * float((signs & (1u << j)) ? -int(magnitude) : int(magnitude));
    }
    case 17: {
      const unsigned group = i / 32, lane = (i % 32) / 8, j = i % 8;
      const unsigned entry = u16(b + 2 + 2 * (4 * group + lane));
      unsigned signs = entry >> 9;
      unsigned parity = signs ^ (signs >> 4);
      parity ^= parity >> 2;
      parity ^= parity >> 1;
      signs |= (parity & 1u) << 7;
      const unsigned magnitude = unsigned((kIq2xs[entry & 511u] >> (8 * j)) & 255u);
      const unsigned scale = (b[66 + group] >> (4 * (lane / 2))) & 15u;
      const float d = f16(b) * (0.5f + float(scale)) * 0.25f;
      return d * float((signs & (1u << j)) ? -int(magnitude) : int(magnitude));
    }
    case 20: {
      const unsigned q = (b[2 + i % 16] >> (4 * (i / 16))) & 15u;
      return f16(b) * float(levels[q]);
    }
    case 23: {
      const unsigned group = i / 32, lane = i % 32;
      const unsigned low = (b[4 + group / 2] >> (4 * (group % 2))) & 15u;
      const unsigned high = (u16(b + 2) >> (2 * group)) & 3u;
      const float d = f16(b) * float(int(low | (high << 4)) - 32);
      const unsigned q = (b[8 + 16 * group + lane % 16] >> (4 * (lane / 16))) & 15u;
      return d * float(levels[q]);
    }
    default: return 0.0f;
  }
}
}  // namespace ink2_gguf
#undef INK2_DEVICE
#undef INK2_CONSTANT
