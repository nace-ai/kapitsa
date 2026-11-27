// CPU draft-row decoding: independent bit-stream packing, group boundaries and Buffer views. No model or GPU.
#include "model/dflash2.h"
#include "util/bf16.h"

#include <cstdio>
#include <random>
#include <vector>

int main() {
  using namespace ni;
  std::mt19937 rng(42);
  size_t cases = 0;
  for (uint32_t K : {64u, 128u, 192u, 2560u, 4096u})
    for (int gs : {32, 64, 128})
      for (int bits : {2, 3, 4, 5, 6, 7, 8, 16}) {
        if (K % uint32_t(gs)) continue;
        constexpr uint32_t N = 3;
        const size_t row_words = size_t(K) * bits / 32, gpr = K / gs;
        // Leave a prefix in each backing buffer to exercise its offset as well as the row stride.
        std::vector<uint32_t> weights(4 + N * row_words, 0);
        std::vector<uint16_t> scales(8 + N * gpr), biases(scales.size()), expected(size_t(N) * K);
        for (size_t i = 0; i < N * gpr; ++i) {
          scales[8 + i] = f32_to_bf16((int(rng() % 255) - 127) / 64.0f);
          biases[8 + i] = f32_to_bf16((int(rng() % 255) - 127) / 32.0f);
        }
        for (uint32_t row = 0; row < N; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            const uint32_t code = rng() & ((1u << bits) - 1);
            // Write each code one bit at a time, independent of the decoder's one/two-word extraction.
            for (int b = 0; b < bits; ++b) {
              const size_t pos = size_t(k) * bits + b;
              weights[4 + row * row_words + pos / 32] |= ((code >> b) & 1u) << (pos % 32);
            }
            const size_t g = 8 + row * gpr + k / gs;
            expected[size_t(row) * K + k] = bits == 16 ? uint16_t(code)
                : f32_to_bf16(bf16_to_f32(scales[g]) * float(code) + bf16_to_f32(biases[g]));
          }
        Tensor w, s, b;
        w.buf = {nullptr, reinterpret_cast<uint8_t*>(weights.data()), 16, N * row_words * 4};
        s.buf = {nullptr, reinterpret_cast<uint8_t*>(scales.data()), 16, N * gpr * 2};
        b.buf = {nullptr, reinterpret_cast<uint8_t*>(biases.data()), 16, N * gpr * 2};
        QLinear H;
        H.w = &w; H.s = &s; H.b = &b; H.N = N; H.K = K; H.bits = bits; H.gs = gs;
        for (uint32_t row = 0; row < N; ++row) {
          std::vector<uint16_t> out(K + 2, 0xa55a);
          dfl_row_bf16(H, row, out.data() + 1);
          if (out.front() != 0xa55a || out.back() != 0xa55a) {
            printf("FAIL guard: K=%u gs=%d bits=%d row=%u\n", K, gs, bits, row);
            return 1;
          }
          for (uint32_t k = 0; k < K; ++k)
            if (out[k + 1] != expected[size_t(row) * K + k]) {
              printf("FAIL value: K=%u gs=%d bits=%d row=%u k=%u\n", K, gs, bits, row, k);
              return 1;
            }
          ++cases;
        }
      }
  printf("dflash_row_test: %zu rows, ALL PASSED\n", cases);
}
