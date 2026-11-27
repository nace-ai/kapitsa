#include "gguf/quant.h"
#include "gguf/gguf.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstdint>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  constexpr size_t rows = 1024, width = 2048, bytes = rows * width / 32 * 22;
  std::vector<uint8_t> input(bytes + 1);
  uint32_t rng = 0x8fd38721;
  auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
  for (auto& x : input) x = uint8_t(next());
  for (size_t offset = 1; offset < input.size(); offset += 22) {
    const uint16_t half = uint16_t(0x2800 + next() % 0x2000);
    memcpy(input.data() + offset, &half, 2);
  }
  const auto original = input;
  std::vector<float> output(rows * width + 2, 12345.0f);
  auto decode = [&]() {
    for (size_t row = 0; row < rows; ++row)
      ni::gguf::dequantize(ni::gguf::kQ5_0, input.data() + 1 + row * width / 32 * 22,
                          output.data() + 1 + row * width, width);
  };
  decode();
  const size_t output_bytes = rows * width * sizeof(float);
  std::vector<char> gold(output_bytes);
  std::ifstream in(argv[1], std::ios::binary);
  if (in) {
    in.read(gold.data(), std::streamsize(gold.size()));
    if (in.gcount() != std::streamsize(gold.size()) || in.peek() != EOF) return 3;
  } else {
    memcpy(gold.data(), output.data() + 1, output_bytes);
    std::ofstream out(argv[1], std::ios::binary);out.write(gold.data(), std::streamsize(gold.size()));
    if (!out) return 4;
  }
  printf("{\"input_kind\":\"fixed-seed synthetic Q5_0 blocks\",\"rows\":%zu,\"width\":%zu,\"complete_output_bytes\":%zu,\"ns_per_row\":[", rows, width, output_bytes);
  for (int sample = 0; sample < 16; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < 64; ++iteration) decode();
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / (64 * rows);
    if (memcmp(gold.data(), output.data() + 1, output_bytes) || input != original ||
        output.front() != 12345.0f || output.back() != 12345.0f) return 5;
    for (size_t i = 1; i <= rows * width; ++i) if (!std::isfinite(output[i])) return 6;
    printf("%s%.9f", sample ? "," : "", ns);
  }
  std::ofstream out(argv[2], std::ios::binary);out.write(reinterpret_cast<const char*>(output.data() + 1), output_bytes);
  if (!out) return 7;
  puts("],\"complete_output_matches_reference\":true,\"input_unchanged\":true,\"finite\":true,\"guards_intact\":true}");
}
