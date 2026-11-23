#pragma once
#include "gguf/gguf.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <unistd.h>

namespace ink2_mixed_fixture {
constexpr size_t H = 512, I = 256, E = 2;
constexpr std::array<uint32_t, E> gu_outer = {0x3e800001, 0x3fc00000};
constexpr std::array<uint32_t, E> down_outer = {0x3fe00000, 0x3d000001};

inline void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template<class T> void put(std::ofstream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
inline void string(std::ofstream& out, const std::string& value) {
  put<uint64_t>(out, value.size());
  out.write(value.data(), value.size());
}

enum class Scale { Good, Missing, WrongType, WrongShape };
enum class Data { FullRange, Bounded };
struct Fixture {
  char path[64] = "/tmp/ink2-mixed-XXXXXX.gguf";
  Fixture(uint32_t gu, uint32_t down, Scale scale = Scale::Good, bool bad_gu = false,
          Data kind = Data::FullRange) {
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
            const uint16_t d = kind == Data::Bounded ? 0x1800 : uint16_t(0x2400 + b % 1024);
            std::memcpy(block, &d, sizeof(d));
          } else if (kind == Data::Bounded) {
            for (size_t j = 0; j < 4; ++j) block[j] = uint8_t(0x28 + (b + j) % 8);
          }
        }
      }
      out.seekp(start + t.offset);
      out.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    check(bool(out), "fixture write");
  }
  Fixture(uint32_t gu, uint32_t down, Data kind) : Fixture(gu, down, Scale::Good, false, kind) {}
  Fixture(const Fixture&) = delete;
  ~Fixture() { unlink(path); }
};
}  // namespace ink2_mixed_fixture
