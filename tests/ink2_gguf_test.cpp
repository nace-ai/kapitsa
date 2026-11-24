#include "gguf/gguf.h"
#include "model/inkling2_gguf.h"
#include "gguf/quant.h"
#include "kernels/inkling2/gguf_blocks.h"
#include "fixtures/ink2_iq3_reference.h"
#include "../tools/gguf_iq2_xs_test.h"
#include "fixtures/ink2_iq2_cases.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static bool check_iq3(const unsigned char* bytes, size_t count, size_t& checked) {
  std::vector<ink2_iq3_reference::block_iq3_xxs> blocks(count);
  std::memcpy(blocks.data(), bytes, count * 98);
  std::vector<float> ref(count * 256), bulk(count * 256);
  ink2_iq3_reference::dequantize_row_iq3_xxs(blocks.data(), ref.data(), ref.size());
  ni::gguf::dequantize(ni::gguf::kIQ3_XXS, bytes, bulk.data(), bulk.size());
  for (unsigned k = 0; k < ref.size(); ++k) {
    const float got = ink2_gguf::value(bytes, ni::gguf::kIQ3_XXS, k);
    if (std::bit_cast<uint32_t>(got) != std::bit_cast<uint32_t>(ref[k]) ||
        std::bit_cast<uint32_t>(bulk[k]) != std::bit_cast<uint32_t>(ref[k])) {
      std::fprintf(stderr, "pinned IQ3_XXS k=%u: read=%a bulk=%a llama=%a\n", k, got, bulk[k], ref[k]);
      return false;
    }
    ++checked;
  }
  return true;
}

static bool parser_fixture(uint32_t type, uint64_t columns, size_t payload, bool expect) {
  std::vector<uint8_t> bytes{'G', 'G', 'U', 'F'};
  auto put = [&](auto value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), p, p + sizeof(value));
  };
  put(uint32_t(3)); put(uint64_t(1)); put(uint64_t(0));
  put(uint64_t(1)); bytes.push_back('x'); put(uint32_t(2));
  put(columns); put(uint64_t(2)); put(type); put(uint64_t(0));
  const size_t data = (bytes.size() + 31) / 32 * 32;
  bytes.resize(data + payload);
  FILE* tmp = std::tmpfile();
  if (!tmp) return false;
  bool ok = false;
  if (std::fwrite(bytes.data(), 1, bytes.size(), tmp) == bytes.size() && std::fflush(tmp) == 0) {
    try {
      ni::gguf::File file("/dev/fd/" + std::to_string(fileno(tmp)));
      const auto* tensor = file.find("x");
      ok = expect && tensor && tensor->bytes == payload && tensor->row_bytes() == payload / 2;
    } catch (const std::exception&) {
      ok = !expect;
    }
  }
  std::fclose(tmp);
  if (!ok) std::fprintf(stderr, "GGUF parser type=%u columns=%llu payload=%zu expect=%d\n",
                        type, static_cast<unsigned long long>(columns), payload, expect);
  return ok;
}

int main() {
  for (uint32_t type : {16u, 17u, 22u}) {
    uint32_t n = 0, bytes = 0;
    if (!ni::gguf::type_block(type, &n, &bytes) ||
        !parser_fixture(type, n, bytes * 2, true) ||
        !parser_fixture(type, n - 1, bytes * 2, false) ||
        !parser_fixture(type, n, bytes * 2 - 1, false)) return 1;
  }
  if (ink2_gguf::block_values(18) != 256 || ink2_gguf::block_bytes(18) != 98) {
    std::fprintf(stderr, "IQ3_XXS layout: expected 256 values in 98 bytes\n");
    return 1;
  }
  std::mt19937 rng(754);
  size_t iq3_checked = 0;
  if (!check_iq3(ink2_iq3_reference::trained_blocks, 3, iq3_checked)) return 1;
  // Offset the input by one byte: no assumption that a GGUF row is naturally aligned.
  std::vector<unsigned char> storage(1 + 3 * 98);
  auto* blocks = storage.data() + 1;
  for (unsigned trial = 0; trial < 65536; ++trial) {
    if ((trial & 0x7c00u) == 0x7c00u) continue;  // Every finite half pattern, including signed zero.
    for (unsigned i = 0; i < 3 * 98; ++i) blocks[i] = static_cast<unsigned char>(rng());
    for (unsigned block = 0; block < 3; ++block) {
      const uint16_t h = uint16_t(trial);
      std::memcpy(blocks + block * 98, &h, 2);
      // Every codebook index, sign encoding and scale nibble occurs in every packed position.
      for (unsigned i = 0; i < 64; ++i) blocks[block * 98 + 2 + i] = uint8_t(trial + i * 73 + block * 97);
      for (unsigned group = 0; group < 8; ++group) {
        uint32_t aux = ((trial + group * 7 + block * 3) & 15u) << 28;
        for (unsigned pair = 0; pair < 4; ++pair) aux |= ((trial / 16 + pair * 23 + group * 17) & 127u) << (7 * pair);
        std::memcpy(blocks + block * 98 + 66 + group * 4, &aux, 4);
      }
    }
    if (!check_iq3(blocks, 3, iq3_checked)) return 1;
  }
  std::printf("IQ3_XXS: %zu CPU scalar/bulk values bitwise match pinned llama.cpp, including 768 trained values\n", iq3_checked);
  size_t checked = 0;
  for (unsigned type : {6u, 8u, 12u, 13u, 14u, 16u, 17u, 18u, 20u, 22u, 23u}) {
    const unsigned n = ink2_gguf::block_values(type), bytes = ink2_gguf::block_bytes(type);
    for (int trial = 0; trial < 1000; ++trial) {
      std::vector<unsigned char> b(bytes * 3);
      for (auto& v : b) v = static_cast<unsigned char>(rng());
      // Finite, nonzero half scales; exercise arbitrary bitplanes, signs, subscales and adjacent blocks.
      for (unsigned block = 0; block < 3; ++block) {
        const unsigned off = block * bytes + (type == 14 ? 208 : 0);
        b[off + 1] = static_cast<unsigned char>(0x08 + rng() % 0x68);
        if (type == 12 || type == 13) b[off + 3] = static_cast<unsigned char>(0x08 + rng() % 0x68);
      }
      std::vector<float> ref(n * 3);
      ni::gguf::dequantize(type, b.data(), ref.data(), ref.size());
      for (unsigned k = 0; k < ref.size(); ++k) {
        const float got = ink2_gguf::value(b.data(), type, k);
        if (got != ref[k]) {
          std::fprintf(stderr, "type %u trial %d k %u: %.9g != %.9g\n", type, trial, k, got, ref[k]);
          return 1;
        }
        ++checked;
      }
    }
  }
  const size_t oracle_start = checked;
  for (uint16_t half : {0u, 0x8000u, 1u, 0x8001u, 0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu}) {
    for (unsigned pattern = 0; pattern < 512; ++pattern) {
      std::vector<uint8_t> row(9 * 74);
      std::vector<uint32_t> want(2304);
      for (unsigned block = 0; block < 9; ++block)
        iq2_xs_test::fixture(row.data() + block * 74, half, pattern + block, want.data() + block * 256);
      for (unsigned k = 0; k < want.size(); ++k) {
        if (std::bit_cast<uint32_t>(ink2_gguf::value(row.data(), 17, k)) != want[k]) {
          std::fprintf(stderr, "IQ2_XS oracle half=%04x pattern=%u k=%u\n", half, pattern, k);
          return 1;
        }
        ++checked;
      }
    }
  }
  std::printf("IQ2_XS: %zu integer-oracle reads across 2304-value rows match\n", checked - oracle_start);
  for (unsigned type : {17u, 22u}) {
    const unsigned bytes = ink2_gguf::block_bytes(type);
    if (ink2_gguf::block_values(type) != 256 || bytes != (type == 17 ? 74 : 82)) return 1;
    size_t iq2_checked = 0;
    std::vector<uint8_t> iq2_storage(1 + 3 * bytes);
    uint8_t* iq2_bytes = iq2_storage.data() + 1;
    std::vector<float> iq2_ref(3 * 256), iq2_bulk(3 * 256);
    for (unsigned half = 0; half < 65536; ++half) {
      if ((half & 0x7c00u) == 0x7c00u) continue;
      for (unsigned block = 0; block < 3; ++block) {
        auto* b = iq2_bytes + block * bytes;
        if (type == 17) ink2_iq2_cases::fill_xs(b, uint16_t(half), half + block * 131);
        else ink2_iq2_cases::fill_s(b, uint16_t(half), half + block * 131);
      }
      if (type == 17) ink2_iq2_cases::decode_xs(iq2_bytes, iq2_ref.data(), 3);
      else ink2_iq2_cases::decode_s(iq2_bytes, iq2_ref.data(), 3);
      ni::gguf::dequantize(type, iq2_bytes, iq2_bulk.data(), iq2_bulk.size());
      for (unsigned k = 0; k < iq2_ref.size(); ++k) {
        const float got = ink2_gguf::value(iq2_bytes, type, k);
        if (std::bit_cast<uint32_t>(got) != std::bit_cast<uint32_t>(iq2_ref[k]) ||
            std::bit_cast<uint32_t>(iq2_bulk[k]) != std::bit_cast<uint32_t>(iq2_ref[k])) {
          std::fprintf(stderr, "pinned IQ2 type=%u half=%04x k=%u: read=%a bulk=%a llama=%a\n",
                       type, half, k, got, iq2_bulk[k], iq2_ref[k]);
          return 1;
        }
        ++iq2_checked;
      }
    }
    std::printf("IQ2 type=%u: %zu CPU scalar/bulk values bitwise match pinned llama.cpp\n", type, iq2_checked);
  }
  for (unsigned K : {64u, 1152u, 2304u}) {
    for (int trial = 0; trial < 100; ++trial) {
      std::vector<uint8_t> raw(K / 64 * 36), q(K / 2), scales(K / 16);
      for (auto& b : raw) b = uint8_t(rng());
      for (size_t block = 0; block < K / 64; ++block)
        for (size_t s = 0; s < 4; ++s) raw[block * 36 + s] = uint8_t(size_t(trial) * 4 + block * 4 + s);
      ni::ink2::unpack_nvfp4_row(raw.data(), K, q.data(), scales.data());
      std::vector<float> ref(K);
      ni::gguf::dequantize(ni::gguf::kNVFP4, raw.data(), ref.data(), K);
      constexpr float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
      for (unsigned k = 0; k < K; ++k) {
        const uint8_t s = scales[k / 16];
        const unsigned ex = (s >> 3) & 15, mant = s & 7;
        const float scale = ex ? std::ldexp(1.0f + float(mant) / 8, int(ex) - 7) : std::ldexp(float(mant), -9);
        const float got = scale * levels[(q[k / 2] >> ((k % 2) * 4)) & 15];
        if (got != ref[k]) {
          std::fprintf(stderr, "NVFP4 repack K=%u k=%u: %.9g != %.9g\n", K, k, got, ref[k]);
          return 1;
        }
        ++checked;
      }
      std::vector<uint8_t> q8(K / 32 * 34), codes(K);
      std::vector<uint16_t> half_scales(K / 32);
      for (auto& b : q8) b = uint8_t(rng());
      ni::ink2::unpack_q8_row(q8.data(), K, codes.data(), half_scales.data());
      for (unsigned block = 0; block < K / 32; ++block)
        if (std::memcmp(q8.data() + block * 34, &half_scales[block], 2) ||
            std::memcmp(q8.data() + block * 34 + 2, codes.data() + block * 32, 32)) return 1;
    }
  }
  for (size_t row = 0; row < 2304; ++row) {
    const size_t native = ni::ink2::native_gu_row(row, 1152);
    if (native / 2 + (native % 2) * 1152 != row) return 1;
  }
  std::printf("ink2 GGUF: %zu scalar reads/repacked NVFP4 values match CPU dequantization; Q8 bytes and 2304 gate/up rows match\n", checked);
}
