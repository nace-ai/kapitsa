// CPU Q5_K decoding: integer oracle, exceptional scales, overlapping writes and guarded buffers.
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

namespace q5_k_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

// Half values are integer multiples of 2^-24. Even the largest two scaled terms and their
// difference fit in int64, so round the exact integer difference directly to fp32.
inline int64_t units(uint16_t h) {
  const unsigned e = (h >> 10) & 31, m = h & 1023;
  const int64_t v = e ? int64_t(1024 + m) << (e - 1) : m;
  return h & 0x8000 ? -v : v;
}
inline uint32_t bits(uint16_t h, uint16_t hm, unsigned scale, unsigned minimum, unsigned q, int* flags) {
  const int64_t a = units(h) * scale * q, b = units(hm) * minimum, delta = a - b;
  const int mode = fegetround();
  if (!delta) {
    const bool same_zero_sign = !a && !b && ((h ^ hm) & 0x8000);
    return uint32_t(same_zero_sign ? bool(h & 0x8000) : mode == FE_DOWNWARD) << 31;
  }
  const bool negative = delta < 0;
  const uint64_t mag = uint64_t(negative ? -delta : delta);
  int top = 63 - __builtin_clzll(mag);
  const int shift = top > 23 ? top - 23 : 0;
  uint64_t mantissa = mag >> shift;
  const uint64_t remainder = mag - (mantissa << shift);
  if (remainder) {
    *flags |= FE_INEXACT;
    const uint64_t halfway = uint64_t(1) << (shift - 1);
    const bool up = mode == FE_TONEAREST ? (remainder > halfway || (remainder == halfway && (mantissa & 1)))
                   : mode == FE_UPWARD ? !negative : mode == FE_DOWNWARD && negative;
    mantissa += up;
    if (mantissa == (uint64_t(1) << 24)) { mantissa >>= 1; ++top; }
  }
  if (top < 23) mantissa <<= 23 - top;
  return (uint32_t(negative) << 31) | (uint32_t(top - 24 + 127) << 23) | (uint32_t(mantissa) & 0x7fffff);
}

// Pack logical output values; expected values do not decode the packed fields.
inline int fixture(uint8_t* b, uint16_t h, unsigned pattern, uint32_t* want,
                   uint16_t hm = 0x3555, int forced_scale = -1, int forced_min = -1, int forced_q = -1) {
  memset(b, 0, 176);
  memcpy(b, &h, 2);
  memcpy(b + 2, &hm, 2);
  int flags = 0;
  for (unsigned group = 0; group < 8; ++group) {
    const unsigned scale = forced_scale < 0 ? (pattern + group * 7) % 64 : unsigned(forced_scale);
    const unsigned minimum = forced_min < 0 ? (pattern * 3 + group * 11) % 64 : unsigned(forced_min);
    if (group < 4) {
      b[4 + group] |= uint8_t(scale);
      b[8 + group] |= uint8_t(minimum);
    } else {
      b[4 + group - 4] |= uint8_t((scale >> 4) << 6);
      b[4 + group] |= uint8_t((minimum >> 4) << 6);
      b[4 + group + 4] = uint8_t((scale & 15) | ((minimum & 15) << 4));
    }
    for (unsigned lane = 0; lane < 32; ++lane) {
      const unsigned q = forced_q < 0 ? (pattern + group * 3 + lane) % 32 : unsigned(forced_q);
      const unsigned output = group * 32 + lane;
      b[48 + (group / 2) * 32 + lane] |= uint8_t((q & 15) << (4 * (group % 2)));
      b[16 + lane] |= uint8_t((q >> 4) << group);
      if ((h & 0x7c00) != 0x7c00 && (hm & 0x7c00) != 0x7c00)
        want[output] = bits(h, hm, scale, minimum, q, &flags);
    }
  }
  return flags;
}

// Preserve the original scalar expression for exceptional scales and sequential overlapping writes.
#pragma STDC FENV_ACCESS OFF
inline float f16_at(const uint8_t* p) {
  uint16_t h;
  memcpy(&h, p, 2);
  return ni::gguf::from_f16(h);
}
inline void scale_min_k4(int j, const uint8_t* q, uint8_t* sc, uint8_t* m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

inline void scalar(const uint8_t* b, float* y) {
  const float d = f16_at(b), dmin = f16_at(b + 2);
  const uint8_t* sm = b + 4;
  const uint8_t* qh = b + 16;
  const uint8_t* qs = b + 48;
  for (int p = 0; p < 4; ++p)
    for (int h = 0; h < 2; ++h) {
      uint8_t sc, m;
      scale_min_k4(2 * p + h, sm, &sc, &m);
      const float dl = d * float(sc), ml = dmin * float(m);
      const int bit = 2 * p + h;
      for (int l = 0; l < 32; ++l) {
        const int lo = h ? qs[32 * p + l] >> 4 : qs[32 * p + l] & 0xF;
        y[64 * p + 32 * h + l] = dl * float(lo + (((qh[l] >> bit) & 1) << 4)) - ml;
      }
    }
}


#pragma STDC FENV_ACCESS ON

inline size_t oracle() {
  alignas(16) uint8_t b[176];
  float got[256], want[256];
  uint32_t expected[256];
  size_t cases = 0;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(fesetround(mode) == 0, "Q5_K rounding mode");
    for (unsigned h = 0; h < 65536; ++h) {
      for (bool swap : {false, true}) {
        int flags = fixture(b, swap ? 0x3555 : uint16_t(h), h, expected, swap ? uint16_t(h) : 0x3555);
        if ((h & 0x7c00) == 0x7c00) {
          feclearexcept(FE_ALL_EXCEPT);
          scalar(b, want);
          flags = fetestexcept(FE_ALL_EXCEPT);
        } else memcpy(want, expected, sizeof(want));
        feclearexcept(FE_ALL_EXCEPT);
        ni::gguf::dequantize(ni::gguf::kQ5_K, b, got, 256);
        if (memcmp(got, want, sizeof(got))) printf("  Q5_K mode=%d half=%04x swap=%d\n", mode, h, swap);
        require(memcmp(got, want, sizeof(got)) == 0, "Q5_K half-pattern bytes");
        require(fetestexcept(FE_ALL_EXCEPT) == flags, "Q5_K half-pattern flags");
        ++cases;
      }
    }
    for (uint16_t h : {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff}) {
      for (uint16_t hm : {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff}) {
        for (int scale = 0; scale < 64; ++scale) {
          for (int minimum = 0; minimum < 64; ++minimum) {
            const int flags = fixture(b, h, 0, expected, hm, scale, minimum);
            feclearexcept(FE_ALL_EXCEPT);
            ni::gguf::dequantize(ni::gguf::kQ5_K, b, got, 256);
            require(memcmp(got, expected, sizeof(got)) == 0, "Q5_K scale/min/code Cartesian bytes");
            require(fetestexcept(FE_ALL_EXCEPT) == flags, "Q5_K Cartesian rounding flags");
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
  require(fesetround(FE_TONEAREST) == 0, "Q5_K restore rounding mode");
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
        ni::gguf::dequantize(ni::gguf::kQ5_K, got.data() + src, reinterpret_cast<float*>(got.data() + dst), 256);
        require(got == want && fetestexcept(FE_ALL_EXCEPT) == flags, "Q5_K overlapping or unaligned buffers/flags");
        ++cases;
      }
    }
  }
  for (size_t blocks : {size_t(2), size_t(3)}) {
    for (size_t dst : {size_t(0), size_t(900), size_t(1024), size_t(1200), size_t(1276)}) {
      alignas(16) std::array<uint8_t, 8192> a, z;
      a.fill(0xa5);
      for (size_t i = 0; i < blocks; ++i) fixture(a.data() + 1024 + 176 * i, 0x3555, unsigned(i * 37), expected);
      z = a;
      feclearexcept(FE_ALL_EXCEPT);
      for (size_t i = 0; i < blocks; ++i)
        scalar(z.data() + 1024 + 176 * i, reinterpret_cast<float*>(z.data() + dst) + 256 * i);
      const int flags = fetestexcept(FE_ALL_EXCEPT);
      feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kQ5_K, a.data() + 1024, reinterpret_cast<float*>(a.data() + dst), blocks * 256);
      require(a == z && fetestexcept(FE_ALL_EXCEPT) == flags, "Q5_K sequential overlapping blocks/flags");
      ++cases;
    }
  }
  return cases;
}

inline void guard_pages() {
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  auto* src = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  auto* dst = static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
  require(src != MAP_FAILED && dst != MAP_FAILED, "Q5_K guard mmap");
  require(mprotect(src + page, page, PROT_READ | PROT_WRITE) == 0 &&
          mprotect(dst + page, page, PROT_READ | PROT_WRITE) == 0, "Q5_K guard mprotect");
  for (size_t blocks : {size_t(1), size_t(3)}) {
    for (bool end : {false, true}) {
      uint8_t* b = src + page + (end ? page - 176 * blocks : 0);
      float* y = reinterpret_cast<float*>(dst + page + (end ? page - 1024 * blocks : 0));
      std::array<uint32_t, 768> expected;
      for (size_t i = 0; i < blocks; ++i) fixture(b + i * 176, i % 2 ? 0x8000 : 0x3555, unsigned(i), expected.data() + i * 256);
      ni::gguf::dequantize(ni::gguf::kQ5_K, b, y, blocks * 256);
      require(memcmp(y, expected.data(), blocks * 1024) == 0, "Q5_K guard bytes");
    }
  }
  // The existing API truncates to whole blocks, including no access for a sub-block request.
  for (size_t n : {size_t(0), size_t(1), size_t(255)})
    ni::gguf::dequantize(ni::gguf::kQ5_K, src, reinterpret_cast<float*>(dst), n);
  uint8_t* b = src + 2 * page - 176;
  float* y = reinterpret_cast<float*>(dst + 2 * page - 1024);
  uint32_t expected[256];
  fixture(b, 0x3555, 3, expected);
  ni::gguf::dequantize(ni::gguf::kQ5_K, b, y, 511);
  require(memcmp(y, expected, sizeof(expected)) == 0, "Q5_K partial trailing block truncation");
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
    printf("  Q5_K: %zu oracle/overlap cases and guarded single/multiple/empty/partial blocks PASS\n", cases);
  } catch (const std::exception& e) {
    printf("  FAILED: %s\n", e.what());
    rc = 1;
  }
  fesetenv(&env);
  return rc;
}
}  // namespace q5_k_test
