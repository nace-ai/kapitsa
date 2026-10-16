// Small-batch (verify) matvec benchmark: R token rows against rotating 4-bit weight matrices.
//   bench_qmvr <kernel> <rows-per-simdgroup or 8 for qmvm> [R] [N] [K]
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"
#include <cmath>
#include <random>
#include <string>
#include <vector>
using namespace ni;
struct QmvrParams {
  uint32_t N, K, R, x_ld, y_ld, add;
};
int main(int argc, char** argv) {
  std::string kname = argv[1];
  uint32_t rows = atoi(argv[2]);
  uint32_t R = argc > 3 ? atoi(argv[3]) : 3, N = argc > 4 ? atoi(argv[4]) : 8192, K = argc > 5 ? atoi(argv[5]) : 2048;
  Gpu& g = Gpu::get();
  std::mt19937 rng(3);
  size_t wbytes = size_t(N) * K / 2, gbytes = size_t(N) * K / 64 * 2;
  int copies = std::max(1, int(1e9 / (wbytes + 2 * gbytes)));
  copies = std::min(copies, 200);
  std::vector<Buffer> W, S, B;
  for (int c = 0; c < copies; ++c) {
    W.push_back(g.alloc(wbytes)); S.push_back(g.alloc(gbytes)); B.push_back(g.alloc(gbytes));
    for (size_t i = 0; i < wbytes / 4; ++i) W.back().as<uint32_t>()[i] = rng();
    for (size_t i = 0; i < gbytes / 2; ++i) { S.back().as<uint16_t>()[i] = f32_to_bf16(0.01f); B.back().as<uint16_t>()[i] = f32_to_bf16(-0.05f); }
  }
  Buffer x = g.alloc(size_t(R) * K * 4), y = g.alloc(size_t(R) * N * 4);
  Buffer xh = g.alloc(size_t(R) * K * 2);
  std::normal_distribution<float> nd(0, 1);
  for (size_t i = 0; i < size_t(R) * K; ++i) {
    x.as<float>()[i] = nd(rng);
    xh.as<__fp16>()[i] = (__fp16)x.as<float>()[i];
    x.as<float>()[i] = float(xh.as<__fp16>()[i]);  // reference sees the same rounded inputs
  }
  Pipeline p = g.pipeline(kname);
  QmvrParams prm{N, K, R, K, N, 0};
  uint32_t nsg = (kname.rfind("qmvrc1", 0) == 0 || kname.rfind("qmvrc2", 0) == 0 || kname.rfind("qmvrch", 0) == 0) ? 2 : 8;
  const bool sgr = kname.rfind("qmvsg", 0) == 0;
  const bool xhalf = kname.rfind("qmvrch", 0) == 0 || sgr;
  auto run = [&]() {
    auto e = g.begin();
    e->use(p);
    for (int c = 0; c < copies; ++c) {
      e->buf(0, W[c]); e->buf(1, S[c]); e->buf(2, B[c]); e->buf(3, xhalf ? xh : x); e->buf(4, y);
      e->val(5, prm);
      if (kname.rfind("qmv2_", 0) == 0 || kname.rfind("qmvrc", 0) == 0) { e->buf(6, y); e->buf(7, y); }
      if (sgr) e->dispatch((N + rows - 1) / rows, 1, 1, 32 * R);
      else if (kname.rfind("qmvrc2", 0) == 0) e->dispatch((R + 1) / 2, (N + rows * nsg - 1) / (rows * nsg), 1, 32 * nsg);
      else if (kname.rfind("qmvrc", 0) == 0) e->dispatch(R, (N + rows * nsg - 1) / (rows * nsg), 1, 32 * nsg);
      else e->dispatch((N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
    }
    e->commit_and_wait();
    return e->gpu_ms();
  };
  run();
  double best = 1e9;
  for (int i = 0; i < 3; ++i) best = std::min(best, run());
  // check row 0..R-1 element 5 against CPU
  double maxerr = 0;
  for (uint32_t t = 0; t < (kname.rfind("qmv2_", 0) == 0 || kname.rfind("qmv3_", 0) == 0 ? 1u : R); ++t) {
    uint32_t n = 5;
    double ref = 0;
    for (uint32_t k = 0; k < K; ++k) {
      uint32_t q = (W.back().as<uint32_t>()[(size_t(n) * K + k) / 8] >> (4 * (k % 8))) & 15;
      ref += (0.01 * q - 0.05) * x.as<float>()[t * K + k];
    }
    maxerr = std::max(maxerr, std::fabs(ref - y.as<float>()[t * N + n]) / (std::fabs(ref) + 1));
  }
  printf("%-18s R=%u N=%u K=%u  %7.1f us/mat  %6.1f GB/s  err=%.1e\n", kname.c_str(), R, N, K, best * 1000 / copies,
         (wbytes + 2 * gbytes) * copies / (best * 1e-3) / 1e9, maxerr);
}
