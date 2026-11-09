// CPU IQ4_XS decoding: integer oracle, exceptional scales, overlapping writes and guarded buffers.
#pragma once
#include "gguf/gguf.h"
#include "gguf/quant.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

namespace iq4_xs_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

// The finite half significand times a signed six-bit scale and a seven-bit level fits in 23 bits. Both scalar
// multiplications are exact, so construct the expected fp32 representation with integer arithmetic.
inline uint32_t bits(uint16_t h, int scale, int q) {
  const unsigned e = (h >> 10) & 31, m = h & 1023;
  const uint32_t sign = uint32_t(((h >> 15) != 0) ^ (scale < 0) ^ (q < 0)) << 31;
  const uint32_t v = (e ? 1024 + m : m) * unsigned(scale < 0 ? -scale : scale) * unsigned(q < 0 ? -q : q);
  if (!v) return sign;
  const int top = 31 - __builtin_clz(v), power = e ? int(e) - 25 : -24;
  return sign | (uint32_t(top + power + 127) << 23) | ((v << (23 - top)) & 0x7fffffu);
}

constexpr int levels[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline void fixture(uint8_t* b, uint16_t h, unsigned pattern, uint32_t* want) {
  memset(b, 0, 136);
  memcpy(b, &h, 2);
  // Encode logical output lanes; the oracle never decodes the packed scale or nibble layout.
  for (unsigned group = 0; group < 8; ++group) {
    const unsigned scale = (group * 13 + pattern * 19) % 64;
    b[4 + group / 2] |= uint8_t((scale & 15) << (4 * (group % 2)));
    b[2 + group / 4] |= uint8_t((scale >> 4) << (2 * (group % 4)));
    for (unsigned lane = 0; lane < 32; ++lane) {
      const unsigned code = (lane * 7 + group * 3 + pattern + lane / 16) % 16;
      b[8 + group * 16 + lane % 16] |= uint8_t(code << (4 * (lane / 16)));
      if ((h & 0x7c00u) != 0x7c00u) want[group * 32 + lane] = bits(h, int(scale) - 32, levels[code]);
    }
  }
}

#if defined(__clang__)
__attribute__((noinline, optnone))
#elif defined(__GNUC__)
__attribute__((noinline, optimize("O0")))
#endif
inline void scalar(const uint8_t* b, float* y) {
  uint16_t h, high;
  memcpy(&h, b, 2);
  memcpy(&high, b + 2, 2);
  _Float16 half_scale;
  memcpy(&half_scale, &h, 2);
  const volatile float d = float(half_scale);
  for (int group = 0; group < 8; ++group) {
    const int scale = ((b[4 + group / 2] >> (4 * (group % 2))) & 15) | (((high >> (2 * group)) & 3) << 4);
    const volatile float dl = d * float(scale - 32);
    for (int j = 0; j < 16; ++j) {
      // The second nibble is read after the first write, preserving the original alias order.
      y[32 * group + j] = dl * float(levels[b[8 + 16 * group + j] & 15]);
      y[32 * group + j + 16] = dl * float(levels[b[8 + 16 * group + j] >> 4]);
    }
  }
}

inline size_t oracle() {
  alignas(16) uint8_t b[136];
  float got[256], want[256];
  uint32_t expected[256];
  size_t cases = 0;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(fesetround(mode) == 0, "IQ4_XS rounding mode");
    for (unsigned h = 0; h < 65536; ++h) {
      fixture(b, uint16_t(h), h, expected);
      int flags = 0;
      if ((h & 0x7c00u) == 0x7c00u) {
        feclearexcept(FE_ALL_EXCEPT);
        scalar(b, want);
        flags = fetestexcept(FE_ALL_EXCEPT);
      } else memcpy(want, expected, sizeof(want));
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kIQ4_XS, b, got, 256);
      require(memcmp(got, want, sizeof(got)) == 0, "IQ4_XS half-pattern bytes");
      require(fetestexcept(FE_ALL_EXCEPT) == flags, "IQ4_XS half-pattern flags");
      ++cases;
    }
    for (uint16_t h : {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff}) {
      for (unsigned scale = 0; scale < 64; ++scale) {
        for (unsigned q = 0; q < 16; ++q) {
          memset(b + 8, int(q * 17), 128);
          memset(b + 4, int((scale & 15) * 17), 4);
          memset(b + 2, int((scale >> 4) * 85), 2);
          memcpy(b, &h, 2);
          const uint32_t v = bits(h, int(scale) - 32, levels[q]);
          for (auto& x : expected) x = v;
          ni::gguf::dequantize(ni::gguf::kIQ4_XS, b, got, 256);
          require(memcmp(got, expected, sizeof(got)) == 0, "IQ4_XS scale/code Cartesian bytes");
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
  require(fesetround(FE_TONEAREST) == 0, "IQ4_XS restore rounding mode");
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
        ni::gguf::dequantize(ni::gguf::kIQ4_XS, got.data() + src, reinterpret_cast<float*>(got.data() + dst), 256);
        require(got == want && fetestexcept(FE_ALL_EXCEPT) == flags, "IQ4_XS overlapping or unaligned buffers/flags");
        ++cases;
      }
    }
  }
  for (size_t blocks : {size_t(2), size_t(3)}) {
    for (size_t dst : {size_t(0), size_t(900), size_t(1024), size_t(1156), size_t(1432)}) {
      alignas(16) std::array<uint8_t, 8192> a, z;
      a.fill(0xa5);
      for (size_t i = 0; i < blocks; ++i) fixture(a.data() + 1024 + 136 * i, 0x3555, unsigned(i * 37), expected);
      z = a;
      feclearexcept(FE_ALL_EXCEPT);
      for (size_t i = 0; i < blocks; ++i)
        scalar(z.data() + 1024 + 136 * i, reinterpret_cast<float*>(z.data() + dst) + 256 * i);
      const int flags = fetestexcept(FE_ALL_EXCEPT);
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kIQ4_XS, a.data() + 1024, reinterpret_cast<float*>(a.data() + dst), blocks * 256);
      require(a == z && fetestexcept(FE_ALL_EXCEPT) == flags, "IQ4_XS sequential overlapping blocks/flags");
      ++cases;
    }
  }
  return cases;
}

inline void guard_pages() {
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  auto* src = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  auto* dst = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  require(src != MAP_FAILED && dst != MAP_FAILED, "IQ4_XS guard mmap");
  require(mprotect(src + page, page, PROT_READ | PROT_WRITE) == 0 &&
          mprotect(dst + page, page, PROT_READ | PROT_WRITE) == 0, "IQ4_XS guard mprotect");
  for (size_t blocks : {size_t(1), size_t(3)}) {
    for (bool end : {false, true}) {
      uint8_t* b = src + page + (end ? page - 136 * blocks : 0);
      float* y = reinterpret_cast<float*>(dst + page + (end ? page - 1024 * blocks : 0));
      std::array<uint32_t, 768> expected;
      for (size_t i = 0; i < blocks; ++i) fixture(b + i * 136, i % 2 ? 0x8000 : 0x3555, unsigned(i), expected.data() + i * 256);
      ni::gguf::dequantize(ni::gguf::kIQ4_XS, b, y, blocks * 256);
      require(memcmp(y, expected.data(), blocks * 1024) == 0, "IQ4_XS guard bytes");
    }
  }
  // The existing API truncates to whole blocks, including no access for a sub-block request.
  for (size_t n : {size_t(0), size_t(1), size_t(255)})
    ni::gguf::dequantize(ni::gguf::kIQ4_XS, src, reinterpret_cast<float*>(dst), n);
  uint8_t* b = src + 2 * page - 136;
  float* y = reinterpret_cast<float*>(dst + 2 * page - 1024);
  uint32_t expected[256];
  fixture(b, 0x3555, 3, expected);
  ni::gguf::dequantize(ni::gguf::kIQ4_XS, b, y, 511);
  require(memcmp(y, expected, sizeof(expected)) == 0, "IQ4_XS partial trailing block truncation");
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
    printf("  IQ4_XS: %zu oracle/overlap cases and guarded single/multiple/empty/partial blocks PASS\n", cases);
  } catch (const std::exception& e) {
    printf("  FAILED: %s\n", e.what());
    rc = 1;
  }
  fesetenv(&env);
  return rc;
}
}  // namespace iq4_xs_test
