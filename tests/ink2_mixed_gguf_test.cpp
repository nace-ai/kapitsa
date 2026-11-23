#include "gguf/quant.h"
#include "model/inkling2_gguf.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace {
constexpr size_t H = 512, I = 256, E = 2;
constexpr std::array<uint32_t, E> gu_outer = {0x3e800001, 0x3fc00000};
constexpr std::array<uint32_t, E> down_outer = {0x3fe00000, 0x3d000001};

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template<class T> void put(std::ofstream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void string(std::ofstream& out, const std::string& value) {
  put<uint64_t>(out, value.size());
  out.write(value.data(), value.size());
}

enum class Scale { Good, Missing, WrongType, WrongShape };
struct Fixture {
  char path[64] = "/tmp/ink2-mixed-XXXXXX.gguf";
  Fixture(uint32_t gu, uint32_t down, Scale scale = Scale::Good, bool bad_gu = false) {
    const int fd = mkstemps(path, 5);
    if (fd < 0) throw std::runtime_error("mkstemps");
    close(fd);
    std::vector<ni::gguf::Tensor> tensors;
    uint64_t offset = 0;
    auto add = [&](const char* suffix, std::vector<uint64_t> shape, uint32_t type) {
      uint32_t n = 0, bytes = 0;
      check(ni::gguf::type_block(type, &n, &bytes), "fixture type");
      uint64_t elements = 1;
      for (auto v : shape) elements *= v;
      const uint64_t size = elements / n * bytes;
      tensors.push_back({std::string("blk.9.") + suffix, std::move(shape), type, 0, offset, size});
      offset = (offset + size + 31) & ~uint64_t(31);
    };
    add("ffn_gate_up_exps.weight", {H, 2 * I, E}, gu);
    add("ffn_down_exps.weight", {I, H, E}, down);
    auto add_scale = [&](const char* suffix, uint32_t type, bool malformed) {
      if (type != ni::gguf::kNVFP4 || (malformed && scale == Scale::Missing)) return;
      add(suffix, {malformed && scale == Scale::WrongShape ? E + 1 : E},
          malformed && scale == Scale::WrongType ? ni::gguf::kBF16 : ni::gguf::kF32);
    };
    add_scale("ffn_gate_up_exps.scale", gu, bad_gu);
    add_scale("ffn_down_exps.scale", down, !bad_gu);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    put<uint32_t>(out, 0x46554747); put<uint32_t>(out, 3);
    put<uint64_t>(out, tensors.size()); put<uint64_t>(out, 0);
    for (const auto& t : tensors) {
      string(out, t.name); put<uint32_t>(out, t.ne.size());
      for (auto v : t.ne) put<uint64_t>(out, v);
      put<uint32_t>(out, t.type); put<uint64_t>(out, t.offset);
    }
    const uint64_t start = (uint64_t(out.tellp()) + 31) & ~uint64_t(31);
    for (const auto& t : tensors) {
      std::vector<uint8_t> data(t.bytes);
      if (t.name.ends_with(".scale")) {
        if (t.type == ni::gguf::kF32) {
          const auto& bits = t.name.find("gate_up") != std::string::npos ? gu_outer : down_outer;
          std::memcpy(data.data(), bits.data(), sizeof(bits));
        }
      } else {
        uint32_t n = 0, bytes = 0;
        check(ni::gguf::type_block(t.type, &n, &bytes), "fixture block layout");
        for (size_t b = 0; b < data.size() / bytes; ++b) {
          auto* block = data.data() + b * bytes;
          for (size_t j = 0; j < bytes; ++j) block[j] = uint8_t(b * 73 + (b / 251) * 17 + j * 29 + 11);
          if (t.type != ni::gguf::kNVFP4) {
            const uint16_t d = uint16_t(0x2400 + b % 1024);
            std::memcpy(block, &d, sizeof(d));
          }
        }
      }
      out.seekp(start + t.offset);
      out.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    check(bool(out), "fixture write");
  }
  Fixture(const Fixture&) = delete;
  ~Fixture() { unlink(path); }
};

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
