// CPU Q6_K decoding: integer oracle, exceptional scales, overlapping writes and guarded buffers.
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

namespace q6_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

// The finite half significand times an int8 scale and a signed six-bit code fits in 23 bits. Both scalar
// multiplications are exact, so construct the expected fp32 representation with integer arithmetic.
inline uint32_t bits(uint16_t h, int scale, int q) {
  const unsigned e = (h >> 10) & 31, m = h & 1023;
  const uint32_t sign = uint32_t(((h >> 15) != 0) ^ (scale < 0) ^ (q < 0)) << 31;
  const uint32_t v = (e ? 1024 + m : m) * unsigned(scale < 0 ? -scale : scale) * unsigned(q < 0 ? -q : q);
  if (!v) return sign;
  const int top = 31 - __builtin_clz(v), power = e ? int(e) - 25 : -24;
  return sign | (uint32_t(top + power + 127) << 23) | ((v << (23 - top)) & 0x7fffffu);
}

inline void fixture(uint8_t* b, uint16_t h, unsigned pattern, uint32_t* want) {
  memset(b, 0, 210);
  memcpy(b + 208, &h, 2);
  for (unsigned lane = 0; lane < 256; ++lane) {
    const unsigned half = lane / 128, plane = lane / 32 % 4, j = lane % 32;
    const unsigned group = half * 8 + plane * 2 + j / 16;
    const uint8_t raw_scale = uint8_t(group * 37 + pattern * 19);
    const int scale = raw_scale < 128 ? int(raw_scale) : int(raw_scale) - 256;
    const unsigned code = (lane * 13 + pattern * 17 + lane / 16) % 64;
    b[half * 64 + (plane % 2) * 32 + j] |= uint8_t((code & 15) << (plane / 2 * 4));
    b[128 + half * 32 + j] |= uint8_t((code >> 4) << (plane * 2));
    b[192 + group] = raw_scale;
    if ((h & 0x7c00u) != 0x7c00u) want[lane] = bits(h, scale, int(code) - 32);
  }
}

#if defined(__clang__)
__attribute__((noinline, optnone))
#elif defined(__GNUC__)
__attribute__((noinline, optimize("O0")))
#endif
inline void scalar(const uint8_t* b, float* y) {
  uint16_t h;
  memcpy(&h, b + 208, 2);
  _Float16 half_scale;
  memcpy(&half_scale, &h, 2);
  const volatile float d = float(half_scale);
  for (int half = 0; half < 2; ++half) {
    for (int j = 0; j < 32; ++j) {
      // Read all four codes before writing, as the original overlapping-buffer decoder does.
      const volatile uint8_t lo = b[half * 64 + j], hi = b[half * 64 + 32 + j], high = b[128 + half * 32 + j];
      const int codes[] = {int((lo & 15) | ((high & 3) << 4)) - 32,
                           int((hi & 15) | (((high >> 2) & 3) << 4)) - 32,
                           int((lo >> 4) | (((high >> 4) & 3) << 4)) - 32,
                           int((hi >> 4) | ((high >> 6) << 4)) - 32};
      for (int p = 0; p < 4; ++p) {
        const volatile int8_t scale = *reinterpret_cast<const volatile int8_t*>(b + 192 + half * 8 + p * 2 + j / 16);
        y[half * 128 + p * 32 + j] = d * float(scale) * float(codes[p]);
      }
    }
  }
}

inline size_t oracle() {
  alignas(16) uint8_t b[210];
  float got[256], want[256];
  uint32_t expected[256];
  size_t cases = 0;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(fesetround(mode) == 0, "Q6_K rounding mode");
    for (unsigned h = 0; h < 65536; ++h) {
      fixture(b, uint16_t(h), h, expected);
      int flags = 0;
      if ((h & 0x7c00u) == 0x7c00u) {
        feclearexcept(FE_ALL_EXCEPT);
        scalar(b, want);
        flags = fetestexcept(FE_ALL_EXCEPT);
      } else memcpy(want, expected, sizeof(want));
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kQ6_K, b, got, 256);
      require(memcmp(got, want, sizeof(got)) == 0, "Q6_K half-pattern bytes");
      require(fetestexcept(FE_ALL_EXCEPT) == flags, "Q6_K half-pattern flags");
      ++cases;
    }
    for (uint16_t h : {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff}) {
      for (unsigned scale = 0; scale < 256; ++scale) {
        for (unsigned q = 0; q < 64; ++q) {
          memset(b, int((q & 15) * 17), 128);
          memset(b + 128, int((q >> 4) * 85), 64);
          memset(b + 192, int(scale), 16);
          memcpy(b + 208, &h, 2);
          const uint32_t v = bits(h, scale < 128 ? int(scale) : int(scale) - 256, int(q) - 32);
          for (auto& x : expected) x = v;
          ni::gguf::dequantize(ni::gguf::kQ6_K, b, got, 256);
          require(memcmp(got, expected, sizeof(got)) == 0, "Q6_K scale/code Cartesian bytes");
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
  require(fesetround(FE_TONEAREST) == 0, "Q6_K restore rounding mode");
  for (uint16_t h : {0x3555, 0x8000, 0x7c01}) {
    for (size_t src = 1024; src < 1040; ++src) {
      for (size_t dst = 0; dst <= 2060; dst += 4) {
        got.fill(0xa5);
        fixture(got.data() + src, h, unsigned(src + dst), expected);
        want = got;
        scalar(want.data() + src, reinterpret_cast<float*>(want.data() + dst));
        ni::gguf::dequantize(ni::gguf::kQ6_K, got.data() + src, reinterpret_cast<float*>(got.data() + dst), 256);
        require(got == want, "Q6_K overlapping or unaligned buffers");
        ++cases;
      }
    }
  }
  return cases;
}

inline void guard_pages() {
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  auto* src = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  auto* dst = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  require(src != MAP_FAILED && dst != MAP_FAILED, "Q6_K guard mmap");
  require(mprotect(src + page, page, PROT_READ | PROT_WRITE) == 0 &&
          mprotect(dst + page, page, PROT_READ | PROT_WRITE) == 0, "Q6_K guard mprotect");
  for (size_t blocks : {size_t(1), size_t(3)}) {
    for (bool end : {false, true}) {
      uint8_t* b = src + page + (end ? page - 210 * blocks : 0);
      float* y = reinterpret_cast<float*>(dst + page + (end ? page - 1024 * blocks : 0));
      std::array<uint32_t, 768> expected;
      for (size_t i = 0; i < blocks; ++i) fixture(b + i * 210, i % 2 ? 0x8000 : 0x3555, unsigned(i), expected.data() + i * 256);
      ni::gguf::dequantize(ni::gguf::kQ6_K, b, y, blocks * 256);
      require(memcmp(y, expected.data(), blocks * 1024) == 0, "Q6_K guard bytes");
    }
  }
  // The existing API truncates to whole blocks, including no access for a sub-block request.
  for (size_t n : {size_t(0), size_t(1), size_t(255)})
    ni::gguf::dequantize(ni::gguf::kQ6_K, src, reinterpret_cast<float*>(dst), n);
  uint8_t* b = src + 2 * page - 210;
  float* y = reinterpret_cast<float*>(dst + 2 * page - 1024);
  uint32_t expected[256];
  fixture(b, 0x3555, 3, expected);
  ni::gguf::dequantize(ni::gguf::kQ6_K, b, y, 511);
  require(memcmp(y, expected, sizeof(expected)) == 0, "Q6_K partial trailing block truncation");
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
    printf("  Q6_K: %zu oracle/overlap cases and guarded single/multiple/empty/partial blocks PASS\n", cases);
  } catch (const std::exception& e) {
    printf("  FAILED: %s\n", e.what());
    rc = 1;
  }
  fesetenv(&env);
  return rc;
}
}  // namespace q6_test
