// MLX affine quantization of bf16 rows with Kapitsa's re-quantizer (src/gguf/quant.cpp: per group the best of the
// min-max grid, a symmetric grid and least-squares refits, after rounding scale and bias to bf16), for Python tools
// through ctypes (tools/convert_qwen4_exp.py, bench/quality/eval_qwen4_exp.py). Built by those tools on first use:
//   c++ -O3 -std=c++20 -shared -fPIC -Isrc -Ithird_party tools/mlx_quant.cpp src/gguf/quant.cpp -o <cache>/libmlxq.dylib
#include "gguf/quant.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

extern "C" {

// rows x cols bf16 values (row-major) -> MLX affine: words [rows, cols*bits/32] uint32, scales and biases
// [rows, cols/gs] bf16. Rows are split over `threads` threads (0: all cores).
int mlxq_quantize_bf16(const uint16_t* x, int64_t rows, int64_t cols, int bits, int gs, uint32_t* words,
                       uint16_t* scales, uint16_t* biases, int threads) {
  if (cols % gs || (cols * bits) % 32 || gs > 256 || bits < 2 || bits > 8) return 1;
  if (threads <= 0) threads = int(std::max(1u, std::thread::hardware_concurrency()));
  threads = int(std::min<int64_t>(threads, std::max<int64_t>(1, rows)));
  const int64_t wpr = cols * bits / 32, gpr = cols / gs;
  auto work = [&](int64_t r0, int64_t r1) {
    std::vector<float> f(static_cast<size_t>(cols));
    std::vector<uint8_t> codes(static_cast<size_t>(cols));
    for (int64_t r = r0; r < r1; ++r) {
      const uint16_t* src = x + r * cols;
      for (int64_t j = 0; j < cols; ++j) f[size_t(j)] = ni::gguf::from_bf16(src[j]);
      ni::gguf::quantize(f.data(), size_t(cols), bits, gs, codes.data(), scales + r * gpr, biases + r * gpr);
      ni::gguf::pack_codes(codes.data(), size_t(cols), bits, words + r * wpr);
    }
  };
  std::vector<std::thread> pool;
  const int64_t per = (rows + threads - 1) / threads;
  for (int t = 0; t < threads; ++t) {
    const int64_t r0 = t * per, r1 = std::min(rows, r0 + per);
    if (r0 < r1) pool.emplace_back(work, r0, r1);
  }
  for (auto& th : pool) th.join();
  return 0;
}

}  // extern "C"
