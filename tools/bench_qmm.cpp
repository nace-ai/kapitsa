// Quantized GEMM benchmark + correctness: Y[M,N] = X[M,K] (half) * W[N,K]^T (4-bit, g64).
//   bench_qmm [kernel-prefix=qmm_b4] ; kernels named <prefix>_m{BM}_n{BN}
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace ni;

struct QmmParams {
  uint32_t M, N, K, x_ld, y_ld, add, gs, w_expert_stride, s_expert_stride;
};

static uint16_t f2h(float f) {
  __fp16 h = (__fp16)f;
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
static float h2f(uint16_t u) {
  __fp16 h;
  memcpy(&h, &u, 2);
  return float(h);
}

int main(int argc, char** argv) {
  Gpu& g = Gpu::get();
  std::string prefix = argc > 1 ? argv[1] : "qmm_b4";
  std::vector<std::pair<int, int>> tiles = {{32, 32}, {64, 64}};
  if (argc > 2) {
    tiles.clear();
    for (int i = 2; i + 1 < argc; i += 2) tiles.push_back({atoi(argv[i]), atoi(argv[i + 1])});
  }
  std::mt19937 rng(7);
  struct Shape {
    uint32_t M, N, K;
  };
  std::vector<Shape> shapes = {{512, 8192, 2048}, {2048, 8192, 2048}, {2048, 2048, 4096}, {2048, 512, 2048}, {64, 512, 2048}};
  if (getenv("SHAPES27"))  // Qwen3.8-27B prefill: MLP gate/up, down, out_proj, GDN in_proj
    shapes = {{2048, 17408, 5120}, {2048, 5120, 17408}, {2048, 5120, 6144}, {2048, 16384, 5120}};
  if (getenv("SMALLM")) shapes = {{4, 8192, 2048}, {8, 8192, 2048}, {16, 8192, 2048}, {4, 17408, 5120}, {8, 5120, 17408}};
  for (auto sh : shapes) {
    const uint32_t M = sh.M, N = sh.N, K = sh.K;
    Buffer W = g.alloc(size_t(N) * K / 2), S = g.alloc(size_t(N) * K / 64 * 2), B = g.alloc(size_t(N) * K / 64 * 2);
    Buffer X = g.alloc(size_t(M) * K * 2), Y = g.alloc(size_t(M) * N * 4);
    for (size_t i = 0; i < size_t(N) * K / 8; ++i) W.as<uint32_t>()[i] = rng();
    std::uniform_real_distribution<float> d(-1, 1);
    for (size_t i = 0; i < size_t(N) * K / 64; ++i) {
      S.as<uint16_t>()[i] = f32_to_bf16(0.01f * (1.0f + d(rng)));
      B.as<uint16_t>()[i] = f32_to_bf16(0.05f * d(rng));
    }
    for (size_t i = 0; i < size_t(M) * K; ++i) X.as<uint16_t>()[i] = f2h(d(rng));
    for (auto [bm, bn] : tiles) {
      std::string name = prefix + "_m" + std::to_string(bm) + "_n" + std::to_string(bn);
      Pipeline p = g.pipeline(name);
      QmmParams prm{M, N, K, K, N, 0, 64, 0, 0};
      auto run = [&](int reps) {
        auto e = g.begin();
        e->use(p);
        for (int r = 0; r < reps; ++r) {
          e->buf(0, X); e->buf(1, W); e->buf(2, S); e->buf(3, B); e->buf(4, Y);
          e->val(5, prm);
          e->dispatch((N + bn - 1) / bn, (M + bm - 1) / bm, 1, p.max_threads >= 128 ? 128 : p.max_threads);
        }
        e->commit_and_wait();
        return e->gpu_ms();
      };
      run(2);
      int reps = std::max(3, int(2e9 / (2.0 * M * N * K)));
      double ms = run(reps) / reps;
      // spot-check 16 outputs
      double maxrel = 0;
      for (int t = 0; t < 16; ++t) {
        uint32_t m = rng() % M, n = rng() % N;
        double ref = 0;
        for (uint32_t k = 0; k < K; ++k) {
          uint32_t q = (W.as<uint32_t>()[(size_t(n) * K + k) / 8] >> (4 * (k % 8))) & 15;
          float wv = bf16_to_f32(S.as<uint16_t>()[(size_t(n) * K + k) / 64]) * q + bf16_to_f32(B.as<uint16_t>()[(size_t(n) * K + k) / 64]);
          ref += double(float(__fp16(wv))) * h2f(X.as<uint16_t>()[size_t(m) * K + k]);
        }
        double got = Y.as<float>()[size_t(m) * N + n];
        maxrel = std::max(maxrel, std::fabs(got - ref) / (std::fabs(ref) + 1.0));
      }
      printf("%-22s M=%5u N=%5u K=%5u  %8.1f us  %5.2f TFLOP/s  err=%.1e\n", name.c_str(), M, N, K, ms * 1000,
             2.0 * M * N * K / (ms * 1e-3) / 1e12, maxrel);
    }
    for (Buffer* b : {&W, &S, &B, &X, &Y}) g.release(*b);
  }
}
