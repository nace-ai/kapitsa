// CPU IQ3_XXS decoding: integer oracle, exceptional scales, overlapping writes and guarded buffers.
#pragma once
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "gguf/iq_grids.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

namespace iq3_xxs_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

// The finite half significand times an odd scale (1..31) and grid magnitude (4..62) fits in 21 bits. Scaling by 1/4 and both
// multiplications are exact, so construct the expected fp32 representation with integer arithmetic.
inline uint32_t bits(uint16_t h, int scale, int q) {
  const unsigned e = (h >> 10) & 31, m = h & 1023;
  const uint32_t sign = uint32_t(((h >> 15) != 0) ^ (scale < 0) ^ (q < 0)) << 31;
  const uint32_t v = (e ? 1024 + m : m) * unsigned(scale < 0 ? -scale : scale) * unsigned(q < 0 ? -q : q);
  if (!v) return sign;
  const int top = 31 - __builtin_clz(v), power = e ? int(e) - 27 : -26;
  return sign | (uint32_t(top + power + 127) << 23) | ((v << (23 - top)) & 0x7fffffu);
}

inline void fixture(uint8_t* b, uint16_t h, unsigned pattern, uint32_t* want) {
  memset(b, 0, 98);
  memcpy(b, &h, 2);
  // Encode logical signs, including the implied parity lane, independently of the packed decoder.
  for (unsigned group = 0; group < 8; ++group) {
    const unsigned scale = (group * 13 + pattern * 7) % 16;
    uint32_t packed = scale << 28;
    for (unsigned pair = 0; pair < 4; ++pair) {
      const unsigned signs = (pattern + group * 17 + pair * 23) % 128;
      packed |= signs << (7 * pair);
      unsigned parity = 0;
      for (unsigned bit = 0; bit < 7; ++bit) parity ^= (signs >> bit) & 1;
      for (unsigned grid = 0; grid < 2; ++grid) {
        const unsigned index = (pattern * 97 + group * 131 + (2 * pair + grid) * 73) % 256;
        b[2 + group * 8 + pair * 2 + grid] = uint8_t(index);
        const uint32_t levels = ni::gguf::iq::kIq3xxs[index];
        for (unsigned lane = 0; lane < 4; ++lane) {
          const unsigned bit = grid * 4 + lane;
          const bool negative = bit == 7 ? parity : ((signs >> bit) & 1);
          const int magnitude = int((levels >> (8 * lane)) & 255);
          if ((h & 0x7c00u) != 0x7c00u)
            want[group * 32 + pair * 8 + bit] = bits(h, int(1 + 2 * scale), negative ? -magnitude : magnitude);
        }
      }
    }
    memcpy(b + 66 + 4 * group, &packed, 4);
  }
}

// Mirror the pre-change scalar expression, including sign multiplication, with its normal compiler FP semantics.
#pragma STDC FENV_ACCESS OFF
inline uint32_t signs8(uint32_t s) { return s | (uint32_t(__builtin_popcount(s) & 1) << 7); }
inline float iq_sign(uint32_t signs, int j) { return (signs >> j) & 1 ? -1.0f : 1.0f; }
inline float grid_byte(uint64_t g, int j) { return float(uint8_t(g >> (8 * j))); }
#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline))
#endif
inline void scalar(const uint8_t* b, float* y) {
  uint16_t h;
  memcpy(&h, b, 2);
  const float d = ni::gguf::from_f16(h);
  const uint8_t* qs = b + 2;
  const uint8_t* sas = b + 66;
  for (int ib = 0; ib < 8; ++ib) {
    uint32_t aux;
    memcpy(&aux, sas + 4 * ib, 4);
    const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    for (int l = 0; l < 4; ++l) {
      const uint32_t s = signs8((aux >> (7 * l)) & 127);
      const uint32_t g1 = ni::gguf::iq::kIq3xxs[qs[8 * ib + 2 * l]], g2 = ni::gguf::iq::kIq3xxs[qs[8 * ib + 2 * l + 1]];
      for (int j = 0; j < 4; ++j) {
        y[32 * ib + 8 * l + j] = db * grid_byte(g1, j) * iq_sign(s, j);
        y[32 * ib + 8 * l + j + 4] = db * grid_byte(g2, j) * iq_sign(s, j + 4);
      }
    }
  }
}
#pragma STDC FENV_ACCESS ON

inline size_t oracle() {
  alignas(16) uint8_t b[98];
  float got[256], want[256];
  uint32_t expected[256];
  size_t cases = 0;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(fesetround(mode) == 0, "IQ3_XXS rounding mode");
    for (unsigned h = 0; h < 65536; ++h) {
      fixture(b, uint16_t(h), h, expected);
      int flags = 0;
      if ((h & 0x7c00u) == 0x7c00u) {
        feclearexcept(FE_ALL_EXCEPT);
        scalar(b, want);
        flags = fetestexcept(FE_ALL_EXCEPT);
      } else memcpy(want, expected, sizeof(want));
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kIQ3_XXS, b, got, 256);
      if (memcmp(got, want, sizeof(got)) != 0) {
        for (unsigned i = 0; i < 256; ++i) {
          uint32_t actual, expected_bits;
          memcpy(&actual, got + i, 4);
          memcpy(&expected_bits, want + i, 4);
          if (actual != expected_bits) {
            printf("  IQ3_XXS mode=%d half=%04x lane=%u actual=%08x expected=%08x\n", mode, h, i, actual, expected_bits);
            break;
          }
        }
      }
      require(memcmp(got, want, sizeof(got)) == 0, "IQ3_XXS half-pattern bytes");
      require(fetestexcept(FE_ALL_EXCEPT) == flags, "IQ3_XXS half-pattern flags");
      ++cases;
    }
    for (uint16_t h : {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff}) {
      for (unsigned scale = 0; scale < 16; ++scale) {
        for (unsigned code = 0; code < 256; ++code) {
          memset(b + 2, int(code), 64);
          // Four distinct packed sign groups exercise shifts 0, 7, 14 and 21 next to the scale nibble.
          uint32_t packed = scale << 28;
          for (unsigned pair = 0; pair < 4; ++pair) packed |= ((code + pair * 31) % 128) << (7 * pair);
          for (unsigned group = 0; group < 8; ++group) memcpy(b + 66 + 4 * group, &packed, 4);
          memcpy(b, &h, 2);
          for (unsigned lane = 0; lane < 256; ++lane) {
            const unsigned signs = (code + (lane % 32 / 8) * 31) % 128;
            unsigned parity = 0;
            for (unsigned bit = 0; bit < 7; ++bit) parity ^= (signs >> bit) & 1;
            const bool negative = lane % 8 == 7 ? parity : ((signs >> (lane % 8)) & 1);
            const int magnitude = int((ni::gguf::iq::kIq3xxs[code] >> (8 * (lane % 4))) & 255);
            expected[lane] = bits(h, int(1 + 2 * scale), negative ? -magnitude : magnitude);
          }
          ni::gguf::dequantize(ni::gguf::kIQ3_XXS, b, got, 256);
          require(memcmp(got, expected, sizeof(got)) == 0, "IQ3_XXS scale/grid/sign bytes");
          ++cases;
        }
      }
    }
  }
  return cases;
}

inline size_t overlap_alignment() {
  alignas(16) std::array<uint8_t, 4096> got, want;
  uint32_t expected[256];
  size_t cases = 0;
  require(fesetround(FE_TONEAREST) == 0, "IQ3_XXS restore rounding mode");
  for (uint16_t h : {0x3555, 0x8000, 0x7c01}) {
    for (size_t src = 1024; src < 1040; ++src) {
      for (size_t dst = 0; dst <= 2060; dst += 4) {
        got.fill(0xa5);
        fixture(got.data() + src, h, unsigned(src + dst), expected);
        want = got;
        feclearexcept(FE_ALL_EXCEPT);
        scalar(want.data() + src, reinterpret_cast<float*>(want.data() + dst));
        const int flags = fetestexcept(FE_ALL_EXCEPT);
        feclearexcept(FE_ALL_EXCEPT);
        ni::gguf::dequantize(ni::gguf::kIQ3_XXS, got.data() + src, reinterpret_cast<float*>(got.data() + dst), 256);
        require(got == want && fetestexcept(FE_ALL_EXCEPT) == flags, "IQ3_XXS overlapping or unaligned buffers/flags");
        ++cases;
      }
    }
  }
  for (size_t blocks : {size_t(2), size_t(3)}) {
    for (size_t dst : {size_t(0), size_t(900), size_t(1024), size_t(1132), size_t(1356)}) {
      alignas(16) std::array<uint8_t, 8192> a, z;
      a.fill(0xa5);
      for (size_t i = 0; i < blocks; ++i) fixture(a.data() + 1024 + 98 * i, 0x3555, unsigned(i * 37), expected);
      z = a;
      feclearexcept(FE_ALL_EXCEPT);
      for (size_t i = 0; i < blocks; ++i)
        scalar(z.data() + 1024 + 98 * i, reinterpret_cast<float*>(z.data() + dst) + 256 * i);
      const int flags = fetestexcept(FE_ALL_EXCEPT);
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kIQ3_XXS, a.data() + 1024, reinterpret_cast<float*>(a.data() + dst), blocks * 256);
      require(a == z && fetestexcept(FE_ALL_EXCEPT) == flags, "IQ3_XXS sequential overlapping blocks/flags");
      ++cases;
    }
  }
  return cases;
}

inline void guard_pages() {
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  auto* src = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  auto* dst = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  require(src != MAP_FAILED && dst != MAP_FAILED, "IQ3_XXS guard mmap");
  require(mprotect(src + page, page, PROT_READ | PROT_WRITE) == 0 &&
          mprotect(dst + page, page, PROT_READ | PROT_WRITE) == 0, "IQ3_XXS guard mprotect");
  for (size_t blocks : {size_t(1), size_t(3)}) {
    for (bool end : {false, true}) {
      uint8_t* b = src + page + (end ? page - 98 * blocks : 0);
      float* y = reinterpret_cast<float*>(dst + page + (end ? page - 1024 * blocks : 0));
      std::array<uint32_t, 768> expected;
      for (size_t i = 0; i < blocks; ++i) fixture(b + i * 98, i % 2 ? 0x8000 : 0x3555, unsigned(i), expected.data() + i * 256);
      ni::gguf::dequantize(ni::gguf::kIQ3_XXS, b, y, blocks * 256);
      require(memcmp(y, expected.data(), blocks * 1024) == 0, "IQ3_XXS guard bytes");
    }
  }
  // The existing API truncates to whole blocks, including no access for a sub-block request.
  for (size_t n : {size_t(0), size_t(1), size_t(255)})
    ni::gguf::dequantize(ni::gguf::kIQ3_XXS, src, reinterpret_cast<float*>(dst), n);
  uint8_t* b = src + 2 * page - 98;
  float* y = reinterpret_cast<float*>(dst + 2 * page - 1024);
  uint32_t expected[256];
  fixture(b, 0x3555, 3, expected);
  ni::gguf::dequantize(ni::gguf::kIQ3_XXS, b, y, 511);
  require(memcmp(y, expected, sizeof(expected)) == 0, "IQ3_XXS partial trailing block truncation");
  munmap(src, page * 3);
  munmap(dst, page * 3);
}

inline int run() {
  fenv_t env;
  if (fegetenv(&env) != 0) return 1;
  int rc = 0;
  try {
    const size_t cases = oracle() + overlap_alignment();
    guard_pages();
    printf("  IQ3_XXS: %zu oracle/overlap cases and guarded single/multiple/empty/partial blocks PASS\n", cases);
  } catch (const std::exception& e) {
    printf("  FAILED: %s\n", e.what());
    rc = 1;
  }
  fesetenv(&env);
  return rc;
}
}  // namespace iq3_xxs_test
