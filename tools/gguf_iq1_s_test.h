// CPU IQ1_S decoding: integer oracle, exceptional scales, overlapping writes and guarded buffers.
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

namespace iq1_s_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

// Finite half times odd scale (1..15) times the signed ternary eighths is exact in fp32.
inline uint32_t bits(uint16_t h, int scale, int q) {
  const unsigned e = (h >> 10) & 31, m = h & 1023;
  const uint32_t sign = uint32_t(((h >> 15) != 0) ^ (scale < 0) ^ (q < 0)) << 31;
  const uint32_t v = (e ? 1024 + m : m) * unsigned(scale < 0 ? -scale : scale) * unsigned(q < 0 ? -q : q);
  if (!v) return sign;
  const int top = 31 - __builtin_clz(v), power = e ? int(e) - 28 : -27;
  return sign | (uint32_t(top + power + 127) << 23) | ((v << (23 - top)) & 0x7fffffu);
}

inline int level(uint64_t grid, unsigned lane) { return int(int8_t(uint8_t(grid >> (8 * lane)))); }

inline void fixture(uint8_t* b, uint16_t h, unsigned pattern, uint32_t* want) {
  memset(b, 0, 50);
  memcpy(b, &h, 2);
  for (unsigned group = 0; group < 8; ++group) {
    const unsigned scale = (pattern + group * 3) % 8, negative = ((pattern / 7) + group) % 2;
    uint16_t high = uint16_t((scale << 12) | (negative << 15));
    for (unsigned grid = 0; grid < 4; ++grid) {
      const unsigned index = (pattern * 97 + group * 131 + grid * 43) % 2048;
      b[2 + group * 4 + grid] = uint8_t(index);
      high |= uint16_t((index >> 8) << (3 * grid));
      if ((h & 0x7c00u) != 0x7c00u)
        for (unsigned lane = 0; lane < 8; ++lane)
          want[group * 32 + grid * 8 + lane] = bits(h, int(1 + 2 * scale),
              8 * level(ni::gguf::iq::kIq1s[index], lane) + (negative ? -1 : 1));
    }
    memcpy(b + 34 + 2 * group, &high, 2);
  }
}

// Mirror the pre-change scalar expression, including sign multiplication, with its normal compiler FP semantics.
#pragma STDC FENV_ACCESS OFF
#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline))
#endif
inline void scalar(const uint8_t* b, float* y) {
  uint16_t h;
  memcpy(&h, b, 2);
  const float d = ni::gguf::from_f16(h);
  for (int ib = 0; ib < 8; ++ib) {
    uint16_t high;
    memcpy(&high, b + 34 + 2 * ib, 2);
    const float dl = d * float(2 * ((high >> 12) & 7) + 1);
    const float delta = high & 0x8000 ? -0.125f : 0.125f;
    for (int l = 0; l < 4; ++l) {
      const uint64_t grid = ni::gguf::iq::kIq1s[b[2 + 4 * ib + l] | (((high >> (3 * l)) & 7) << 8)];
      for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = dl * (float(level(grid, unsigned(j))) + delta);
    }
  }
}
#pragma STDC FENV_ACCESS ON

inline size_t oracle() {
  alignas(16) uint8_t b[50];
  float got[256], want[256];
  uint32_t expected[256];
  size_t cases = 0;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(fesetround(mode) == 0, "IQ1_S rounding mode");
    for (unsigned h = 0; h < 65536; ++h) {
      fixture(b, uint16_t(h), h, expected);
      int flags = 0;
      if ((h & 0x7c00u) == 0x7c00u) {
        feclearexcept(FE_ALL_EXCEPT);
        scalar(b, want);
        flags = fetestexcept(FE_ALL_EXCEPT);
      } else memcpy(want, expected, sizeof(want));
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kIQ1_S, b, got, 256);
      if (memcmp(got, want, sizeof(got)) != 0) {
        for (unsigned i = 0; i < 256; ++i) {
          uint32_t actual, expected_bits;
          memcpy(&actual, got + i, 4);
          memcpy(&expected_bits, want + i, 4);
          if (actual != expected_bits) {
            printf("  IQ1_S mode=%d half=%04x lane=%u actual=%08x expected=%08x\n", mode, h, i, actual, expected_bits);
            break;
          }
        }
      }
      require(memcmp(got, want, sizeof(got)) == 0, "IQ1_S half-pattern bytes");
      require(fetestexcept(FE_ALL_EXCEPT) == flags, "IQ1_S half-pattern flags");
      ++cases;
    }
    for (uint16_t h : {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff}) {
      for (unsigned scale = 0; scale < 8; ++scale) {
        for (unsigned code = 0; code < 2048; ++code) {
          for (unsigned negative = 0; negative < 2; ++negative) {
            memcpy(b, &h, 2);
            memset(b + 2, int(code & 255), 32);
            const uint16_t high = uint16_t((code >> 8) * 585 | (scale << 12) | (negative << 15));
            for (unsigned group = 0; group < 8; ++group) memcpy(b + 34 + 2 * group, &high, 2);
            for (unsigned lane = 0; lane < 256; ++lane)
              expected[lane] = bits(h, int(1 + 2 * scale),
                  8 * level(ni::gguf::iq::kIq1s[code], lane % 8) + (negative ? -1 : 1));
            ni::gguf::dequantize(ni::gguf::kIQ1_S, b, got, 256);
            require(memcmp(got, expected, sizeof(got)) == 0, "IQ1_S scale/grid/delta Cartesian bytes");
            ++cases;
          }
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
  require(fesetround(FE_TONEAREST) == 0, "IQ1_S restore rounding mode");
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
        ni::gguf::dequantize(ni::gguf::kIQ1_S, got.data() + src, reinterpret_cast<float*>(got.data() + dst), 256);
        require(got == want && fetestexcept(FE_ALL_EXCEPT) == flags, "IQ1_S overlapping or unaligned buffers/flags");
        ++cases;
      }
    }
  }
  for (size_t blocks : {size_t(2), size_t(3)}) {
    for (size_t dst : {size_t(0), size_t(900), size_t(1024), size_t(1132), size_t(1356)}) {
      alignas(16) std::array<uint8_t, 8192> a, z;
      a.fill(0xa5);
      for (size_t i = 0; i < blocks; ++i) fixture(a.data() + 1024 + 50 * i, 0x3555, unsigned(i * 37), expected);
      z = a;
      feclearexcept(FE_ALL_EXCEPT);
      for (size_t i = 0; i < blocks; ++i)
        scalar(z.data() + 1024 + 50 * i, reinterpret_cast<float*>(z.data() + dst) + 256 * i);
      const int flags = fetestexcept(FE_ALL_EXCEPT);
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kIQ1_S, a.data() + 1024, reinterpret_cast<float*>(a.data() + dst), blocks * 256);
      require(a == z && fetestexcept(FE_ALL_EXCEPT) == flags, "IQ1_S sequential overlapping blocks/flags");
      ++cases;
    }
  }
  return cases;
}

inline void guard_pages() {
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  auto* src = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  auto* dst = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  require(src != MAP_FAILED && dst != MAP_FAILED, "IQ1_S guard mmap");
  require(mprotect(src + page, page, PROT_READ | PROT_WRITE) == 0 &&
          mprotect(dst + page, page, PROT_READ | PROT_WRITE) == 0, "IQ1_S guard mprotect");
  for (size_t blocks : {size_t(1), size_t(3)}) {
    for (bool end : {false, true}) {
      uint8_t* b = src + page + (end ? page - 50 * blocks : 0);
      float* y = reinterpret_cast<float*>(dst + page + (end ? page - 1024 * blocks : 0));
      std::array<uint32_t, 768> expected;
      for (size_t i = 0; i < blocks; ++i) fixture(b + i * 50, i % 2 ? 0x8000 : 0x3555, unsigned(i), expected.data() + i * 256);
      require(mprotect(src + page, page, PROT_READ) == 0, "IQ1_S read-only input");
      ni::gguf::dequantize(ni::gguf::kIQ1_S, b, y, blocks * 256);
      require(mprotect(src + page, page, PROT_READ | PROT_WRITE) == 0, "IQ1_S restore input protection");
      require(memcmp(y, expected.data(), blocks * 1024) == 0, "IQ1_S guard bytes");
    }
  }
  // The existing API truncates to whole blocks, including no access for a sub-block request.
  for (size_t n : {size_t(0), size_t(1), size_t(255)})
    ni::gguf::dequantize(ni::gguf::kIQ1_S, src, reinterpret_cast<float*>(dst), n);
  uint8_t* b = src + 2 * page - 50;
  float* y = reinterpret_cast<float*>(dst + 2 * page - 1024);
  uint32_t expected[256];
  fixture(b, 0x3555, 3, expected);
  ni::gguf::dequantize(ni::gguf::kIQ1_S, b, y, 511);
  require(memcmp(y, expected, sizeof(expected)) == 0, "IQ1_S partial trailing block truncation");
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
    printf("  IQ1_S: %zu oracle/overlap cases and guarded single/multiple/empty/partial blocks PASS\n", cases);
  } catch (const std::exception& e) {
    printf("  FAILED: %s\n", e.what());
    rc = 1;
  }
  fesetenv(&env);
  return rc;
}
}  // namespace iq1_s_test
