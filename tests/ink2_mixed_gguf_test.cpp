#include "gguf/quant.h"
#include "model/inkling2_gguf.h"
#include "fixtures/ink2_mixed_gguf.h"

#include <bit>
#include <cmath>
#include <cstdio>

namespace {
using namespace ink2_mixed_fixture;

float native_value(const uint8_t* codes, const uint8_t* scales, size_t k) {
  constexpr float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
  const uint8_t s = scales[k / 16];
  const int exponent = (s >> 3) & 15, mantissa = s & 7;
  const float block_scale = exponent ? std::ldexp(1.0f + float(mantissa) / 8, exponent - 7)
                                     : std::ldexp(float(mantissa), -9);
  return block_scale * levels[(codes[k / 2] >> ((k % 2) * 4)) & 15];
}

size_t projection(const ni::gguf::File& file, const ni::ink2::ExpertSource& source, bool gate_up) {
  const auto& weight = *source.weight;
  const size_t K = weight.ne[0], R = weight.ne[1];
  const auto& outer_bits = gate_up ? gu_outer : down_outer;
  if (source.native_nvfp4()) {
    check(source.outer.size() == E, "native outer shape");
    for (size_t expert = 0; expert < E; ++expert)
      check(std::bit_cast<uint32_t>(source.outer[expert]) == outer_bits[expert], "native outer bits/order");
  } else check(source.outer.empty(), "retained blocks must not receive native outer");
  std::vector<uint8_t> raw(weight.bytes), codes(E * R * K / 2), scales(E * R * K / 16);
  file.read(weight, 0, raw.data(), raw.size());
  check(std::memcmp(raw.data(), raw.data() + R * weight.row_bytes(), weight.row_bytes()) != 0,
        "fixture experts must differ");
  if (gate_up)
    check(std::memcmp(raw.data(), raw.data() + I * weight.row_bytes(), weight.row_bytes()) != 0,
          "fixture gate/up rows must differ");
  if (source.native_nvfp4()) {
    for (size_t r = 0; r < E * R; ++r) {
      const size_t dst = (r / R) * R + (gate_up ? ni::ink2::native_gu_row(r % R, I) : r % R);
      ni::ink2::unpack_nvfp4_row(raw.data() + r * weight.row_bytes(), K,
                                codes.data() + dst * K / 2, scales.data() + dst * K / 16);
    }
  }
  std::vector<float> reference(K);
  bool nonzero = false;
  for (size_t r = 0; r < E * R; ++r) {
    const auto* row = raw.data() + r * weight.row_bytes();
    ni::gguf::dequantize(weight.type, row, reference.data(), K);
    const size_t local = r % R;
    const size_t paired = local < I ? 2 * local : 2 * (local - I) + 1;
    const size_t dst = (r / R) * R + (gate_up ? paired : local);
    for (size_t k = 0; k < K; ++k) {
      float got = source.native_nvfp4()
          ? native_value(codes.data() + dst * K / 2, scales.data() + dst * K / 16, k)
          : ink2_gguf::value(row, weight.type, k);
      if (source.native_nvfp4()) {
        got *= source.outer[r / R];
        reference[k] *= std::bit_cast<float>(outer_bits[r / R]);
      }
      check(std::isfinite(got) && got == reference[k], "projection differs from reference dequantization");
      nonzero |= got != 0;
    }
  }
  check(nonzero, "fixture produced no signal");
  return E * R * K;
}
}  // namespace

bool check_mixed_gguf() {
  try {
    size_t checked = 0;
    // Mixed first: the old pair-wide decision rejects this down tensor as type 40.
    for (const auto& types : {std::array<uint32_t, 2>{ni::gguf::kIQ3_XXS, ni::gguf::kNVFP4},
                              {ni::gguf::kNVFP4, ni::gguf::kIQ3_XXS},
                              {ni::gguf::kIQ3_XXS, ni::gguf::kIQ3_XXS},
                              {ni::gguf::kNVFP4, ni::gguf::kNVFP4},
                              {ni::gguf::kIQ2_XXS, ni::gguf::kNVFP4}}) {
      Fixture fixture(types[0], types[1]);
      ni::gguf::File file(fixture.path);
      const auto sources = ni::ink2::read_expert_sources(file, "blk.9.", H, I, E);
      check(sources.gu.native_nvfp4() == (types[0] == ni::gguf::kNVFP4), "GU representation");
      check(sources.down.native_nvfp4() == (types[1] == ni::gguf::kNVFP4), "down representation");
      checked += projection(file, sources.gu, true);
      checked += projection(file, sources.down, false);
    }
    for (bool bad_gu : {false, true}) {
      for (Scale scale : {Scale::Missing, Scale::WrongType, Scale::WrongShape}) {
        Fixture fixture(bad_gu ? ni::gguf::kNVFP4 : ni::gguf::kIQ3_XXS,
                        bad_gu ? ni::gguf::kIQ3_XXS : ni::gguf::kNVFP4, scale, bad_gu);
        ni::gguf::File file(fixture.path);
        bool rejected = false;
        try { (void)ni::ink2::read_expert_sources(file, "blk.9.", H, I, E); }
        catch (const ni::gguf::Error&) { rejected = true; }
        check(rejected, "bad native outer accepted");
      }
    }
    std::printf("mixed GGUF: %zu values match CPU dequantization with nonunit per-expert outer scales; "
                "five projection pairs and six malformed scales passed\n", checked);
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mixed GGUF regression: %s\n", error.what());
    return false;
  }
}
