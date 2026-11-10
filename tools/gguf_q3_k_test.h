// CPU Q3_K decoding: integer oracle, exceptional scales, overlapping writes and guarded buffers.
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

namespace q3_k_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

// The finite half significand times a signed 6-bit scale and signed 3-bit value fits in 24 bits. Both
// multiplications are exact, so construct the expected fp32 representation with integer arithmetic.
inline uint32_t bits(uint16_t h, int scale, int q) {
  const unsigned e = (h >> 10) & 31, m = h & 1023;
  const uint32_t sign = uint32_t(((h >> 15) != 0) ^ (scale < 0) ^ (q < 0)) << 31;
  const uint32_t v = (e ? 1024 + m : m) * unsigned(scale < 0 ? -scale : scale) * unsigned(q < 0 ? -q : q);
  if (!v) return sign;
  const int top = 31 - __builtin_clz(v), power = e ? int(e) - 25 : -24;
  return sign | (uint32_t(top + power + 127) << 23) | ((v << (23 - top)) & 0x7fffffu);
}

// Encode values in logical output order; the oracle never decodes the packed bit fields.
inline void fixture(uint8_t* b, uint16_t h, unsigned pattern, uint32_t* want,
                    int forced_scale = 99, int forced_code = 99) {
  memset(b, 0, 110);
  memcpy(b + 108, &h, 2);
  for (unsigned group = 0; group < 16; ++group) {
    const int scale = forced_scale == 99 ? int((pattern * 13 + group * 7) % 64) - 32 : forced_scale;
    const unsigned encoded = unsigned(scale + 32);
    b[96 + group % 8] |= uint8_t((encoded & 15) << (4 * (group / 8)));
    b[104 + group % 4] |= uint8_t((encoded >> 4) << (2 * (group / 4)));
    for (unsigned lane = 0; lane < 16; ++lane) {
      const int q = forced_code == 99 ? int((pattern + group * 3 + lane * 5) % 8) - 4 : forced_code;
      const unsigned output = group * 16 + lane;
      b[32 + (output / 128) * 32 + output % 32] |= uint8_t((unsigned(q) & 3) << (2 * ((output % 128) / 32)));
      if (q >= 0) b[output % 32] |= uint8_t(1u << (output / 32));
      if ((h & 0x7c00u) != 0x7c00u) want[output] = bits(h, scale, q);
    }
  }
}

// The pre-change scalar expression preserves exceptional values and sequential aliasing semantics.
#pragma STDC FENV_ACCESS OFF
inline float f16_at(const uint8_t* p) {
  uint16_t h;
  memcpy(&h, p, 2);
  return ni::gguf::from_f16(h);
}
#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline))
#endif
inline void scalar(const uint8_t* b, float* y) {
  const uint8_t* hm = b;
  const uint8_t* qs = b + 32;
  const uint8_t* s = b + 96;
  const float d = f16_at(b + 108);
  int scale[16];
  for (int i = 0; i < 4; ++i) {
    scale[i] = (s[i] & 0xF) | (((s[8 + i] >> 0) & 3) << 4);
    scale[4 + i] = (s[4 + i] & 0xF) | (((s[8 + i] >> 2) & 3) << 4);
    scale[8 + i] = (s[i] >> 4) | (((s[8 + i] >> 4) & 3) << 4);
    scale[12 + i] = (s[4 + i] >> 4) | (((s[8 + i] >> 6) & 3) << 4);
  }
  for (int k = 0; k < 16; ++k) {
    const int half = k / 8, j = (k % 8) / 2, second = k % 2;
    const int shift = 2 * j, bit = 4 * half + j;
    const uint8_t* q = qs + 32 * half + 16 * second;
    const uint8_t* h = hm + 16 * second;
    const float dl = d * float(scale[k] - 32);
    for (int l = 0; l < 16; ++l) {
      const int v = int((q[l] >> shift) & 3) - ((h[l] >> bit) & 1 ? 0 : 4);
      y[16 * k + l] = dl * float(v);
    }
  }
}

#pragma STDC FENV_ACCESS ON

inline size_t oracle() {
  alignas(16) uint8_t b[110];
  float got[256], want[256];
  uint32_t expected[256];
  size_t cases = 0;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(fesetround(mode) == 0, "Q3_K rounding mode");
    for (unsigned h = 0; h < 65536; ++h) {
      fixture(b, uint16_t(h), h, expected);
      int flags = 0;
      if ((h & 0x7c00u) == 0x7c00u) {
        feclearexcept(FE_ALL_EXCEPT);
        scalar(b, want);
        flags = fetestexcept(FE_ALL_EXCEPT);
      } else memcpy(want, expected, sizeof(want));
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kQ3_K, b, got, 256);
      if (memcmp(got, want, sizeof(got)) != 0) {
        for (unsigned i = 0; i < 256; ++i) {
          uint32_t actual, expected_bits;
          memcpy(&actual, got + i, 4);
          memcpy(&expected_bits, want + i, 4);
          if (actual != expected_bits) {
            printf("  Q3_K mode=%d half=%04x lane=%u actual=%08x expected=%08x\n", mode, h, i, actual, expected_bits);
            break;
          }
        }
      }
      require(memcmp(got, want, sizeof(got)) == 0, "Q3_K half-pattern bytes");
      require(fetestexcept(FE_ALL_EXCEPT) == flags, "Q3_K half-pattern flags");
      ++cases;
    }
    for (uint16_t h : {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff}) {
      for (int scale = -32; scale <= 31; ++scale) {
        for (int code = -4; code <= 3; ++code) {
          fixture(b, h, 0, expected, scale, code);
          feclearexcept(FE_ALL_EXCEPT);
          ni::gguf::dequantize(ni::gguf::kQ3_K, b, got, 256);
          require(memcmp(got, expected, sizeof(got)) == 0, "Q3_K scale/code Cartesian bytes");
          require(fetestexcept(FE_ALL_EXCEPT) == 0, "Q3_K exact finite flags");
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
  require(fesetround(FE_TONEAREST) == 0, "Q3_K restore rounding mode");
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
        ni::gguf::dequantize(ni::gguf::kQ3_K, got.data() + src, reinterpret_cast<float*>(got.data() + dst), 256);
        require(got == want && fetestexcept(FE_ALL_EXCEPT) == flags, "Q3_K overlapping or unaligned buffers/flags");
        ++cases;
      }
    }
  }
  for (size_t blocks : {size_t(2), size_t(3)}) {
    for (size_t dst : {size_t(0), size_t(900), size_t(1024), size_t(1132), size_t(1356)}) {
      alignas(16) std::array<uint8_t, 8192> a, z;
      a.fill(0xa5);
      for (size_t i = 0; i < blocks; ++i) fixture(a.data() + 1024 + 110 * i, 0x3555, unsigned(i * 37), expected);
      z = a;
      feclearexcept(FE_ALL_EXCEPT);
      for (size_t i = 0; i < blocks; ++i)
        scalar(z.data() + 1024 + 110 * i, reinterpret_cast<float*>(z.data() + dst) + 256 * i);
      const int flags = fetestexcept(FE_ALL_EXCEPT);
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kQ3_K, a.data() + 1024, reinterpret_cast<float*>(a.data() + dst), blocks * 256);
      require(a == z && fetestexcept(FE_ALL_EXCEPT) == flags, "Q3_K sequential overlapping blocks/flags");
      ++cases;
    }
  }
  return cases;
}

inline void guard_pages() {
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  auto* src = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  auto* dst = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  require(src != MAP_FAILED && dst != MAP_FAILED, "Q3_K guard mmap");
  require(mprotect(src + page, page, PROT_READ | PROT_WRITE) == 0 &&
          mprotect(dst + page, page, PROT_READ | PROT_WRITE) == 0, "Q3_K guard mprotect");
  for (size_t blocks : {size_t(1), size_t(3)}) {
    for (bool end : {false, true}) {
      uint8_t* b = src + page + (end ? page - 110 * blocks : 0);
      float* y = reinterpret_cast<float*>(dst + page + (end ? page - 1024 * blocks : 0));
      std::array<uint32_t, 768> expected;
      for (size_t i = 0; i < blocks; ++i) fixture(b + i * 110, i % 2 ? 0x8000 : 0x3555, unsigned(i), expected.data() + i * 256);
      ni::gguf::dequantize(ni::gguf::kQ3_K, b, y, blocks * 256);
      require(memcmp(y, expected.data(), blocks * 1024) == 0, "Q3_K guard bytes");
    }
  }
  // The existing API truncates to whole blocks, including no access for a sub-block request.
  for (size_t n : {size_t(0), size_t(1), size_t(255)})
    ni::gguf::dequantize(ni::gguf::kQ3_K, src, reinterpret_cast<float*>(dst), n);
  uint8_t* b = src + 2 * page - 110;
  float* y = reinterpret_cast<float*>(dst + 2 * page - 1024);
  uint32_t expected[256];
  fixture(b, 0x3555, 3, expected);
  ni::gguf::dequantize(ni::gguf::kQ3_K, b, y, 511);
  require(memcmp(y, expected, sizeof(expected)) == 0, "Q3_K partial trailing block truncation");
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
    printf("  Q3_K: %zu oracle/overlap cases and guarded single/multiple/empty/partial blocks PASS\n", cases);
  } catch (const std::exception& e) {
    printf("  FAILED: %s\n", e.what());
    rc = 1;
  }
  fesetenv(&env);
  return rc;
}
}  // namespace q3_k_test
