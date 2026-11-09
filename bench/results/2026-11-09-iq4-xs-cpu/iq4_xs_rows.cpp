// Standalone measurement driver of the official GGUF CPU dequantize API.
// Every gate and file operation is outside the measured row loop.
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace gg = ni::gguf;

static std::vector<uint8_t> read_file(const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || in.tellg() != static_cast<std::streamoff>(expected)) throw std::runtime_error("input file size mismatch");
  std::vector<uint8_t> data(expected);
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size())))
    throw std::runtime_error("input file read failed");
  return data;
}
static void write_file(const std::string& path, const float* data, size_t count) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out || !out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float))))
    throw std::runtime_error("output file write failed");
}
static size_t number(const std::string& value) {
  size_t used = 0;
  unsigned long long n = std::stoull(value, &used);
  if (used != value.size() || value.empty() || value[0] == '-' || n > std::numeric_limits<size_t>::max())
    throw std::runtime_error("invalid positive integer");
  return static_cast<size_t>(n);
}
int main(int argc, char** argv) {
  try {
    std::string input, oracle, output;
    size_t rows = 0, cols = 0, passes = 0;
    bool make_oracle = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--make-oracle") { make_oracle = true; continue; }
      if (i + 1 >= argc) throw std::runtime_error("option has no value");
      const std::string value = argv[++i];
      if (arg == "--input") input = value;
      else if (arg == "--oracle") oracle = value;
      else if (arg == "--output") output = value;
      else if (arg == "--rows") rows = number(value);
      else if (arg == "--cols") cols = number(value);
      else if (arg == "--passes") passes = number(value);
      else throw std::runtime_error("unknown option");
    }
    if (input.empty() || output.empty() || !rows || !cols || cols % 256 ||
        rows > std::numeric_limits<size_t>::max() / cols / sizeof(float) ||
        (make_oracle ? passes != 0 : (oracle.empty() || passes != 16)))
      throw std::runtime_error("invalid IQ4_XS row configuration");
    const size_t row_bytes = cols / 256 * 136, values = rows * cols;
    const auto packed = read_file(input, rows * row_bytes);
    const auto packed_original = packed;
    std::vector<float> decoded(values);
    auto decode_rows = [&] {
      for (size_t row = 0; row < rows; ++row)
        gg::dequantize(gg::kIQ4_XS, packed.data() + row * row_bytes, decoded.data() + row * cols, cols);
    };
    auto finite = [&] {
      for (float value : decoded) if (!std::isfinite(value)) return false;
      return true;
    };
    json report = {{"schema", "kapitsa-iq4-xs-official-cpu-v1"}, {"mode", make_oracle ? "reference-oracle" : "timed-leg"},
                   {"api", "ni::gguf::dequantize(kIQ4_XS), one call per row"}, {"rows", rows}, {"cols", cols},
                   {"packed_row_bytes", row_bytes}, {"input_bytes", packed.size()}, {"output_bytes", values * sizeof(float)},
                   {"scope", "warm CPU row dequantization only; excludes file IO, hashing, gates and allocations"},
                   {"passes", json::array()}};
    if (make_oracle) {
      decode_rows();
      const bool input_equal = packed == packed_original, all_finite = finite();
      report["input_unchanged"] = input_equal; report["all_finite"] = all_finite;
      if (!input_equal || !all_finite) { std::cout << report.dump() << '\n'; return 1; }
      write_file(output, decoded.data(), values);
      report["written"] = true;
      std::cout << report.dump() << '\n'; return 0;
    }
    const auto expected = read_file(oracle, values * sizeof(float));
    decode_rows();  // One untimed warmup per process/leg.
    const bool warm_equal = !memcmp(decoded.data(), expected.data(), expected.size());
    report["warmup"] = {{"count", 1}, {"output_equal", warm_equal}, {"all_finite", finite()},
                         {"input_unchanged", packed == packed_original}};
    if (!warm_equal || !finite() || packed != packed_original) { std::cout << report.dump() << '\n'; return 1; }
    for (size_t pass = 0; pass < passes; ++pass) {
      const auto begin = std::chrono::steady_clock::now();
      decode_rows();
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
      const bool output_equal = !memcmp(decoded.data(), expected.data(), expected.size());
      const bool input_equal = packed == packed_original, all_finite = finite();
      report["passes"].push_back({{"pass", pass + 1}, {"elapsed_ns", elapsed},
                                  {"ns_per_row", double(elapsed) / double(rows)},
                                  {"output_equal", output_equal}, {"all_finite", all_finite}, {"input_unchanged", input_equal}});
      if (elapsed <= 0 || !output_equal || !input_equal || !all_finite) { std::cout << report.dump() << '\n'; return 1; }
    }
    write_file(output, decoded.data(), values);
    report["written"] = true;
    std::cout << report.dump() << '\n'; return 0;
  } catch (const std::exception& ex) {
    std::cerr << "CPU row driver: " << ex.what() << '\n'; return 2;
  }
}
