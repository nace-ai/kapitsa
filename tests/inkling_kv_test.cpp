#include "model/inkling2.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#pragma STDC FENV_ACCESS ON

namespace {
int failures = 0;
size_t checked_rows = 0;
double max_error = 0.0, max_bound = 0.0;

#define CHECK(condition, what)                                                                  \
  do {                                                                                          \
    if (!(condition)) {                                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, what, #condition);          \
      ++failures;                                                                               \
    }                                                                                           \
  } while (false)

using Input = std::array<float, INK2_KV_HEAD_DIM>;
using Bytes = std::array<uint8_t, INK2_KV_I8_BYTES>;

static_assert(sizeof(_Float16) == 2 && sizeof(float) == 4 && sizeof(int8_t) == 1);
static_assert(INK2_KV_HEAD_DIM == 128 && INK2_KV_SCALE_OFFSET == 128);
static_assert(INK2_KV_PADDING_OFFSET == 132 && INK2_KV_PADDING_BYTES == 12);
static_assert(ni::ink2_kv_row_bytes(ni::Ink2KvFormat::I8) == 144);
static_assert(ni::ink2_kv_row_bytes(ni::Ink2KvFormat::F16) == 256);

// This codec models the Metal row contract. It does not execute either Metal library.
void reference_encode(const Input& input, uint8_t* dst) {
  Input half_input;
  float amax = 0.0f;
  for (size_t d = 0; d < half_input.size(); ++d) {
    half_input[d] = float(_Float16(input[d]));
    amax = std::max(amax, std::fabs(half_input[d]));
  }
  const float scale = amax == 0.0f ? 1.0f : amax / 127.0f;
  for (size_t d = 0; d < half_input.size(); ++d) {
    const float rounded = std::rint(half_input[d] / scale);
    const int8_t code = int8_t(std::clamp(rounded, -127.0f, 127.0f));
    std::memcpy(dst + d, &code, sizeof(code));
  }
  std::memcpy(dst + INK2_KV_SCALE_OFFSET, &scale, sizeof(scale));
  std::memset(dst + INK2_KV_PADDING_OFFSET, 0, INK2_KV_PADDING_BYTES);
}

Input reference_decode(const uint8_t* src) {
  float scale;
  std::memcpy(&scale, src + INK2_KV_SCALE_OFFSET, sizeof(scale));
  Input output;
  for (size_t d = 0; d < output.size(); ++d) {
    int8_t code;
    std::memcpy(&code, src + d, sizeof(code));
    output[d] = float(_Float16(float(code) * scale));
  }
  return output;
}

void check_round_trip(const Input& input, const uint8_t* row) {
  float scale;
  std::memcpy(&scale, row + INK2_KV_SCALE_OFFSET, sizeof(scale));
  CHECK(std::isfinite(scale) && scale > 0.0f, "positive finite FP32 scale");
  const Input output = reference_decode(row);
  for (size_t d = 0; d < input.size(); ++d) {
    int8_t code;
    std::memcpy(&code, row + d, sizeof(code));
    CHECK(code >= -127 && code <= 127, "signed code never uses -128");
    CHECK(std::isfinite(output[d]), "finite half-rounded reconstruction");
    const double half_input = float(_Float16(input[d]));
    const double product = double(code) * double(scale);
    const double half_rounding = std::max(0x1p-25, std::fabs(product) * 0x1p-11);
    const double fp32_rounding = 8.0 * std::numeric_limits<float>::epsilon() *
                                 std::max(std::fabs(half_input), double(scale));
    const double bound = std::fabs(double(input[d]) - half_input) + double(scale) * 0.5 +
                         half_rounding + fp32_rounding;
    const double error = std::fabs(double(output[d]) - double(input[d]));
    CHECK(error <= bound, "max error includes input half, quantization, FP32 and output half rounding");
    max_error = std::max(max_error, error);
    max_bound = std::max(max_bound, bound);
  }
  for (size_t p = INK2_KV_PADDING_OFFSET; p < INK2_KV_I8_BYTES; ++p)
    CHECK(row[p] == 0, "all twelve padding bytes are zero");
  ++checked_rows;
}

void geometry_and_zero() {
  CHECK(INK2_KV_I8_BYTES % 16 == 0, "I8 neighboring rows retain 16-byte alignment");
  CHECK(INK2_KV_SCALE_OFFSET % alignof(float) == 0, "scale is four-byte aligned");
  CHECK(INK2_KV_SCALE_OFFSET + sizeof(float) == INK2_KV_PADDING_OFFSET, "padding follows FP32 scale");
  CHECK(INK2_KV_F16_BYTES == INK2_KV_HEAD_DIM * sizeof(_Float16), "F16 row has 128 halves");
  CHECK(8 * ni::ink2_kv_row_bytes(ni::Ink2KvFormat::I8) == 1152, "eight-head global token stride");
  CHECK(8 * ni::ink2_kv_row_bytes(ni::Ink2KvFormat::F16) == 2048, "eight-head local token stride");
  Input input{};
  for (size_t d = 0; d < input.size(); d += 2) input[d] = -0.0f;
  Bytes row;
  row.fill(0xa5);
  reference_encode(input, row.data());
  for (size_t d = 0; d < INK2_KV_HEAD_DIM; ++d) CHECK(row[d] == 0, "zero row has zero codes");
  float scale;
  std::memcpy(&scale, row.data() + 128, sizeof(scale));
  CHECK(scale == 1.0f, "zero row stores scale one at byte 128");
  CHECK(std::bit_cast<uint32_t>(scale) == 0x3f800000u, "scale occupies four FP32 bytes");
  check_round_trip(input, row.data());
}

void signs_extrema_and_ties() {
  Input input{};
  input[0] = 65504.0f;
  input[1] = -65504.0f;
  input[2] = 0x1p-14f;
  input[3] = -0x1p-14f;
  input[4] = 0x1p-24f;
  input[5] = -0x1p-24f;
  Bytes row{};
  reference_encode(input, row.data());
  CHECK(row[0] == 0x7f && row[1] == 0x81, "finite half extrema use +127 and -127 codes");
  check_round_trip(input, row.data());

  input.fill(0x1p-24f);
  input[1] = -0x1p-24f;
  reference_encode(input, row.data());
  CHECK(row[0] == 0x7f && row[1] == 0x81, "smallest half subnormals retain their signs");
  CHECK(reference_decode(row.data())[0] == 0x1p-24f, "smallest half subnormal reconstructs");
  check_round_trip(input, row.data());

  for (int n = 0; n <= 126; ++n) {
    input.fill(0.0f);
    input[0] = float(n) + 0.5f;
    input[1] = -input[0];
    input.back() = 127.0f;
    reference_encode(input, row.data());
    int8_t positive, negative;
    std::memcpy(&positive, row.data(), 1);
    std::memcpy(&negative, row.data() + 1, 1);
    const int even = n + (n & 1);
    CHECK(positive == even && negative == -even, "positive and negative exact ties round to even");
    CHECK(row.back() == 0, "last padding byte does not retain sentinel data");
    check_round_trip(input, row.data());
  }
}

void half_rounding_and_scale_placement() {
  Input input{};
  input[0] = 127.02f;
  input[1] = 0.5001f;
  input[2] = -0.5001f;
  Bytes row{};
  reference_encode(input, row.data());
  float scale;
  std::memcpy(&scale, row.data() + INK2_KV_SCALE_OFFSET, sizeof(scale));
  CHECK(scale == 1.0f && row[1] == 0 && row[2] == 0, "input half rounding precedes maxabs and code rounding");
  CHECK(std::rint(input[1] / (input[0] / 127.0f)) == 1.0f, "fixture distinguishes unrounded input codec");
  check_round_trip(input, row.data());

  input.fill(0.0f);
  input[0] = 1.0f;
  input[1] = 0.5f;
  reference_encode(input, row.data());
  std::memcpy(&scale, row.data() + INK2_KV_SCALE_OFFSET, sizeof(scale));
  CHECK(scale == 1.0f / 127.0f, "non-half-representable scale retains FP32 precision");
  CHECK(scale != float(_Float16(scale)), "fixture distinguishes a half scale");
  CHECK(row[0] == 127 && row[1] == 64, "known positive non-tie and tie codes");
  const Input output = reference_decode(row.data());
  CHECK(output[1] == 0.50390625f, "dequantized value rounds to half");
  CHECK(output[1] != float(64) * scale, "fixture distinguishes unrounded reconstruction");
  check_round_trip(input, row.data());

  row.fill(0xa5);
  const int8_t negative = -127, positive = 63;
  const float known_scale = 0.25f;
  std::memcpy(row.data(), &negative, 1);
  std::memcpy(row.data() + 127, &positive, 1);
  std::memcpy(row.data() + 128, &known_scale, sizeof(known_scale));
  const Input known = reference_decode(row.data());
  CHECK(known.front() == -31.75f && known.back() == 15.75f, "decoder uses signed edge codes and byte-128 scale");
  CHECK(row[132] == 0xa5 && row[143] == 0xa5, "decoder ignores padding contents");
}

void independent_rows_and_sentinels() {
  constexpr size_t heads = 8, tokens = 3, rows = heads * tokens, guard = 16;
  std::array<std::vector<uint8_t>, 2> kv = {
      std::vector<uint8_t>(2 * guard + rows * INK2_KV_I8_BYTES, 0xa5),
      std::vector<uint8_t>(2 * guard + rows * INK2_KV_I8_BYTES, 0x5a)};
  for (size_t kind = 0; kind < kv.size(); ++kind) {
    for (size_t token = 0; token < tokens; ++token) {
      for (size_t head = 0; head < heads; ++head) {
        const size_t row = token * heads + head;
        const size_t offset = guard + row * INK2_KV_I8_BYTES;
        const float amplitude = float((kind + 1) * 32 + token * heads + head + 1);
        Input input;
        for (size_t d = 0; d < input.size(); ++d) input[d] = amplitude * float(int(d % 9) - 4) / 4.0f;
        input[0] = amplitude;
        input[1] = -amplitude;
        const auto before = kv;
        reference_encode(input, kv[kind].data() + offset);
        for (size_t other = 0; other < kv.size(); ++other)
          for (size_t byte = 0; byte < kv[other].size(); ++byte)
            if (other != kind || byte < offset || byte >= offset + INK2_KV_I8_BYTES)
              CHECK(kv[other][byte] == before[other][byte], "write preserves neighboring rows, other KV buffer and outer guards");
        float scale;
        std::memcpy(&scale, kv[kind].data() + offset + 128, sizeof(scale));
        CHECK(scale == amplitude / 127.0f, "K, V, token and head each have an independent scale");
        CHECK(kv[kind][offset] == 0x7f && kv[kind][offset + 1] == 0x81, "token-major row addressing keeps signed extrema");
        check_round_trip(input, kv[kind].data() + offset);
      }
    }
  }
  for (size_t kind = 0; kind < kv.size(); ++kind)
    for (size_t row = 0; row < rows; ++row) {
      const size_t offset = guard + row * INK2_KV_I8_BYTES;
      float scale;
      std::memcpy(&scale, kv[kind].data() + offset + 128, sizeof(scale));
      CHECK(scale == float((kind + 1) * 32 + row + 1) / 127.0f, "later writes leave earlier row scales intact");
      CHECK(reference_decode(kv[kind].data() + offset)[0] == float((kind + 1) * 32 + row + 1),
            "neighboring rows reconstruct their own maxima");
    }
}

void deterministic_random_rows() {
  uint32_t state = 0x772128u;
  for (int trial = 0; trial < 512; ++trial) {
    Input input;
    const float magnitude = std::ldexp(1.0f, trial % 33 - 20);
    for (float& value : input) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      value = magnitude * (float(int(state % 65537u) - 32768) / 32768.0f);
    }
    Bytes first, second;
    first.fill(0xa5);
    second.fill(0x5a);
    reference_encode(input, first.data());
    reference_encode(input, second.data());
    CHECK(first == second, "fixed-seed rows encode deterministically and overwrite the complete row");
    check_round_trip(input, first.data());
  }
}
}  // namespace

int main() {
  const int original_rounding = std::fegetround();
  if (original_rounding == -1 || std::fesetround(FE_TONEAREST) != 0) {
    std::fprintf(stderr, "inkling_kv_test: cannot establish FE_TONEAREST\n");
    return 1;
  }
  CHECK(std::fegetround() == FE_TONEAREST, "reference codec fixes nearest-even rounding mode");
  geometry_and_zero();
  signs_extrema_and_ties();
  half_rounding_and_scale_placement();
  independent_rows_and_sentinels();
  deterministic_random_rows();
  CHECK(std::fesetround(original_rounding) == 0, "restore caller rounding mode");
  std::printf("inkling_kv_test: CPU reference round-trip coverage, %zu rows, max error %.9g, max bound %.9g, %d failures\n",
              checked_rows, max_error, max_bound, failures);
  std::puts("Metal quantization and attention kernels were not executed by this test.");
  return failures ? 1 : 0;
}
