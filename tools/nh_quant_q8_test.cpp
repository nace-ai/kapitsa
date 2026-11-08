// CPU Q8_0 load quantization: complete code/half-scale bytes, rounding, exceptional fallback and guards.
// No model or GPU. --finite-only omits the pre-existing nonfinite/domain-error conversion cases for sanitizers.
#include "model/nemotron_h.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
size_t checked = 0, exceptional = 0, skipped = 0;

// The former scalar routine is used only for its platform-specific exceptional behavior and overlapping buffers.
void legacy(const float* x, int64_t K, uint8_t* q, uint8_t* s) {
  for (int64_t b = 0; b < K / 32; ++b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
    const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    const __fp16 dh = __fp16(d);
    memcpy(s + b * 2, &dh, 2);
    for (int i = 0; i < 32; ++i) {
      const int v = int(std::lround(x[b * 32 + i] * id));
      q[b * 32 + i] = uint8_t(int8_t(std::clamp(v, -127, 127)));
    }
  }
}

bool check(const std::array<float, 32>& x, bool finite_only) {
  std::array<uint8_t, 34> q{}, expected{};
  std::array<uint8_t, 4> s{}, scale{};
  q.fill(0xa5); expected = q;
  s.fill(0x5a); scale = s;
  float magnitude = 0;
  bool finite = true;
  for (float v : x) {
    finite = finite && (std::bit_cast<uint32_t>(v) & 0x7f800000u) != 0x7f800000u;
    if (finite) magnitude = std::max(magnitude, std::fabs(v));
  }
  const float unit = magnitude / 127.0f;
  const float inverse = unit > 0 ? 1.0f / unit : 0;
  finite = finite && std::isfinite(inverse);
  int legacy_flags = 0;
  if (!finite) {
    if (finite_only) { ++skipped; return true; }
    feclearexcept(FE_ALL_EXCEPT);
    legacy(x.data(), 32, expected.data() + 1, scale.data() + 1);
    legacy_flags = fetestexcept(FE_ALL_EXCEPT);
    ++exceptional;
  } else {
    const _Float16 half = _Float16(unit);
    memcpy(scale.data() + 1, &half, 2);
    for (int i = 0; i < 32; ++i) {
      const float normalized = x[i] * inverse;
      // Double is exact for this float product. Magnitude+floor implements ties away, independently of lround.
      const int absolute = int(std::floor(std::fabs(double(normalized)) + 0.5));
      const int rounded = std::signbit(normalized) ? -absolute : absolute;
      expected[i + 1] = uint8_t(int8_t(std::clamp(rounded, -127, 127)));
    }
    std::array<uint8_t, 32> flag_codes{};
    std::array<uint8_t, 2> flag_scale{};
    feclearexcept(FE_ALL_EXCEPT);
    legacy(x.data(), 32, flag_codes.data(), flag_scale.data());
    legacy_flags = fetestexcept(FE_ALL_EXCEPT);
  }
  feclearexcept(FE_ALL_EXCEPT);
  ni::nh_quant_q8(x.data(), 32, q.data() + 1, s.data() + 1);
  const int flags = fetestexcept(FE_ALL_EXCEPT);
  ++checked;
  if (q != expected || s != scale || flags != legacy_flags) {
    printf("FAIL block %zu finite=%d flags=%d/%d rounding=%d\n", checked, int(finite), flags, legacy_flags, fegetround());
    return false;
  }
  return true;
}

bool boundaries() {
  std::mt19937 rng(713);
  for (int64_t K : {int64_t(0), int64_t(1), int64_t(31), int64_t(32), int64_t(33), int64_t(63),
                    int64_t(64), int64_t(65), int64_t(127), int64_t(128), int64_t(129), int64_t(4096)}) {
    for (int start = 0; start < 4; ++start) {
      std::vector<float> x(size_t(K) + 4);
      for (float& v : x) v = float(int(rng() % 2001) - 1000) / 16;
      for (int offset = 0; offset < 16; ++offset) {
        std::vector<uint8_t> q(size_t(K) + 64, 0xa5), expected = q;
        std::vector<uint8_t> s(size_t(K) / 16 + 64, 0x5a), scale = s;
        legacy(x.data() + start, K, expected.data() + offset, scale.data() + offset);
        ni::nh_quant_q8(x.data() + start, K, q.data() + offset, s.data() + offset);
        if (q != expected || s != scale) {
          printf("FAIL boundary K=%lld start=%d offset=%d\n", (long long)K, start, offset);
          return false;
        }
        ++checked;
      }
    }
  }
  // The code write zeroes a future float; a scale write replaces its exponent. Both remain finite.
  for (bool codes : {false, true}) {
    std::array<float, 32> a{}, b{};
    a[8] = 1.5f; a[31] = 127.0f; b = a;
    std::array<uint8_t, 32> qa{}, qb{};
    std::array<uint8_t, 2> sa{}, sb{};
    uint8_t* ar = reinterpret_cast<uint8_t*>(a.data());
    uint8_t* br = reinterpret_cast<uint8_t*>(b.data());
    legacy(a.data(), 32, codes ? ar + 32 : qa.data(), codes ? sa.data() : ar + 34);
    ni::nh_quant_q8(b.data(), 32, codes ? br + 32 : qb.data(), codes ? sb.data() : br + 34);
    if (memcmp(a.data(), b.data(), sizeof a) || qa != qb || sa != sb) {
      printf("FAIL overlapping %s\n", codes ? "codes" : "scale");
      return false;
    }
    ++checked;
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  const bool finite_only = argc == 2 && !strcmp(argv[1], "--finite-only");
  if (argc > 1 && !finite_only) { printf("usage: nh_quant_q8_test [--finite-only]\n"); return 2; }
  const int original_rounding = fegetround();
  std::mt19937 rng(31991);
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    fesetround(mode);
    // Every BF16 code, rotating through all lanes; mixtures also include signed finite control values.
    for (uint32_t word = 0; word < 65536; ++word) {
      std::array<float, 32> x{};
      for (int i = 0; i < 32; ++i) x[i] = float(i - 16) / 8;
      x[word % 32] = std::bit_cast<float>(word << 16);
      if (!check(x, finite_only)) return 1;
    }
    for (uint32_t bits : {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
                          0x00800000u, 0x80800000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
                          0xff800000u, 0x7f800001u, 0xff800001u, 0x7fc00000u, 0xffc00000u}) {
      for (int lane = 0; lane < 32; ++lane) {
        std::array<float, 32> x{};
        x[lane] = std::bit_cast<float>(bits);
        if (!check(x, finite_only)) return 1;
        x.fill(std::bit_cast<float>(bits));
        if (!check(x, finite_only)) return 1;
      }
    }
    for (int integer = -127; integer <= 126; ++integer) {
      for (int side = -1; side <= 1; ++side) {
        const float tie = float(integer) + 0.5f;
        std::array<float, 32> x{};
        x.fill(side == 0 ? tie : std::nextafter(tie, side < 0 ? -INFINITY : INFINITY));
        x[31] = 127;
        if (!check(x, finite_only)) return 1;
      }
    }
    for (int i = 0; i < 2000; ++i) {
      std::array<float, 32> x{};
      for (float& v : x) v = std::bit_cast<float>(uint32_t(rng()));
      if (!check(x, finite_only)) return 1;
    }
  }
  fesetround(FE_TONEAREST);
  if (!boundaries()) return 1;
  fesetround(original_rounding);
  printf("nh_quant_q8_test: %zu cases, %zu exceptional legacy comparisons, %zu skipped, ALL PASSED\n",
         checked, exceptional, skipped);
}
