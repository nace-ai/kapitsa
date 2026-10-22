// Quantized GEMM benchmark + correctness: Y[M,N] = X[M,K] (half) * W[N,K]^T (4-bit, g64).
//   bench_qmm [kernel-prefix=qmm_b4] [BM BN ...]   kernels named <prefix>_m{BM}_n{BN}
//   bench_qmm <kernel> [<kernel> ...]               full names ending in _m{BM}_n{BN}
// Threads per threadgroup: QMM_THREADS, else 32*WM*WN from a "_w<WM><WN>" name part, else 128.
// Kernels are timed interleaved (QMM_TRIALS rounds, default 5, each ~0.25 s of work in one command
// buffer); prints the median and best TFLOP/s and the max relative error over 64 sampled outputs
// (incl. the last row/column) against a double-precision reference on the dequantized halfs.
// qmr_gu_* kernels (fused gate/up/SwiGLU, half output) get a second weight set; their TFLOP/s
// count both projections. *_ld kernels (QmmLdParams) read the leading K columns of weight rows 2K long.
// QMM_SCALE=<f> multiplies the random scales (1e-3: below half's normal range).
// Shapes: default (35B-like), SHAPES27=1 (27B prefill), SMALLM=1, or QMM_SHAPES="M,N,K;M,N,K".
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

using namespace ni;
namespace ni { extern const char* const kTensorKernelSource; }

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

struct Kern {
  std::string name;
  int bm, bn;
  uint32_t threads;
  bool gu;  // fused gate/up/SwiGLU (qmr_gu_*): second weight set in buffers 6-8, half H output
  int bits; // 8 for qmr8_* / *_b8_* kernels, else 4
  bool ld;  // *_ld: QmmLdParams, weight / scale rows of 2K columns
};
static int bits_for(const std::string& n) { return n.rfind("qmr8", 0) == 0 || n.find("_b8_") != std::string::npos ? 8 : 4; }

static uint32_t threads_for(const std::string& name) {
  if (getenv("QMM_THREADS")) return atoi(getenv("QMM_THREADS"));
  std::smatch m;
  if (std::regex_search(name, m, std::regex("_w([0-9])([0-9])_"))) return 32 * std::stoi(m[1]) * std::stoi(m[2]);
  return 128;
}

int main(int argc, char** argv) {
  Gpu& g = Gpu::get();
  std::vector<Kern> kerns;
  const std::regex full(".*_m([0-9]+)_n([0-9]+)(_ld)?$");
  std::smatch mt;
  std::string a1 = argc > 1 ? argv[1] : "qmm_b4";
  if (std::regex_match(a1, mt, full)) {
    for (int i = 1; i < argc; ++i) {
      std::string n = argv[i];
      if (!std::regex_match(n, mt, full)) fatal("not a kernel name: %s", n.c_str());
      kerns.push_back({n, std::stoi(mt[1]), std::stoi(mt[2]), threads_for(n), (n.find("_gu_") != std::string::npos || n.rfind("qmtg", 0) == 0), bits_for(n),
                       mt[3].matched});
    }
  } else {
    std::vector<std::pair<int, int>> tiles = {{32, 32}, {64, 64}};
    if (argc > 2) {
      tiles.clear();
      for (int i = 2; i + 1 < argc; i += 2) tiles.push_back({atoi(argv[i]), atoi(argv[i + 1])});
    }
    for (auto [bm, bn] : tiles) {
      std::string n = a1 + "_m" + std::to_string(bm) + "_n" + std::to_string(bn);
      kerns.push_back({n, bm, bn, threads_for(n), (n.find("_gu_") != std::string::npos || n.rfind("qmtg", 0) == 0), bits_for(n), false});
    }
  }
  const int trials = getenv("QMM_TRIALS") ? atoi(getenv("QMM_TRIALS")) : 5;
  // QMM_SCALE: multiplies the random scales (e.g. 1e-3 puts them below half's normal range)
  const float scale_mul = getenv("QMM_SCALE") ? float(atof(getenv("QMM_SCALE"))) : 1.0f;
  std::mt19937 rng(7);
  struct Shape {
    uint32_t M, N, K;
  };
  std::vector<Shape> shapes = {{512, 8192, 2048}, {2048, 8192, 2048}, {2048, 2048, 4096}, {2048, 512, 2048}, {64, 512, 2048}};
  if (getenv("SHAPES27"))  // Qwen3.8-27B prefill: MLP gate/up, down, out_proj, GDN in_proj
    shapes = {{2048, 17408, 5120}, {2048, 5120, 17408}, {2048, 5120, 6144}, {2048, 16384, 5120}};
  if (getenv("SMALLM")) shapes = {{4, 8192, 2048}, {8, 8192, 2048}, {16, 8192, 2048}, {4, 17408, 5120}, {8, 5120, 17408}};
  if (const char* qs = getenv("QMM_SHAPES")) {
    shapes.clear();
    std::string s = qs;
    std::regex re("([0-9]+),([0-9]+),([0-9]+)");
    for (std::sregex_iterator it(s.begin(), s.end(), re), end; it != end; ++it)
      shapes.push_back({uint32_t(std::stoul((*it)[1])), uint32_t(std::stoul((*it)[2])), uint32_t(std::stoul((*it)[3]))});
  }
  for (auto sh : shapes) {
    const uint32_t M = sh.M, N = sh.N, K = sh.K;
    // weights sized for 8 bits (4-bit kernels read the first half), rows of 2K for the *_ld kernels' view
    Buffer W = g.alloc(size_t(N) * K * 2), S = g.alloc(size_t(N) * K / 64 * 4), B = g.alloc(size_t(N) * K / 64 * 4);
    Buffer X = g.alloc(size_t(M) * K * 2), Y = g.alloc(size_t(M) * N * 4);
    Buffer W2 = g.alloc(size_t(N) * K * 2), S2 = g.alloc(size_t(N) * K / 64 * 4), B2 = g.alloc(size_t(N) * K / 64 * 4);
    std::uniform_real_distribution<float> d(-1, 1);
    for (auto [w, sc, bi] : {std::tuple{&W, &S, &B}, std::tuple{&W2, &S2, &B2}}) {
      for (size_t i = 0; i < size_t(N) * K / 2; ++i) w->as<uint32_t>()[i] = rng();
      for (size_t i = 0; i < size_t(N) * K / 32; ++i) {
        sc->as<uint16_t>()[i] = f32_to_bf16(scale_mul * 0.01f * (1.0f + d(rng)));
        bi->as<uint16_t>()[i] = f32_to_bf16(0.05f * d(rng));
      }
    }
    for (size_t i = 0; i < size_t(M) * K; ++i) X.as<uint16_t>()[i] = f2h(d(rng));
    const double flop = 2.0 * M * N * K;
    const int reps = std::max(3, int(1.5e12 / flop));
    std::vector<Pipeline> pipes;
    for (auto& k : kerns) {
      if (k.name.rfind("qmt", 0) == 0) g.compile_extra("tensor", kTensorKernelSource, true);
      pipes.push_back(g.pipeline(k.name));
    }
    QmmParams prm{M, N, K, K, N, 0, 64, 0, 0};
    struct { QmmParams q; uint32_t w_ld, s_ld; } ldprm{prm, 2 * K / 8, 2 * K / 64};  // 4-bit rows of 2K
    auto run = [&](size_t ki, int n) {
      const Kern& k = kerns[ki];
      auto e = g.begin();
      e->use(pipes[ki]);
      for (int r = 0; r < n; ++r) {
        e->buf(0, X); e->buf(1, W); e->buf(2, S); e->buf(3, B); e->buf(4, Y);
        if (k.ld) e->val(5, ldprm); else e->val(5, prm);
        if (k.gu) { e->buf(6, W2); e->buf(7, S2); e->buf(8, B2); }
        const int cb = k.name.rfind("qmtg", 0) == 0 ? k.bn / 2 : k.bn;
        e->dispatch((N + cb - 1) / cb, (M + k.bm - 1) / k.bm, 1, std::min(k.threads, pipes[ki].max_threads));
      }
      e->commit_and_wait();
      return e->gpu_ms() / n;
    };
    std::vector<std::vector<double>> ms(kerns.size());
    for (size_t ki = 0; ki < kerns.size(); ++ki) run(ki, 1);
    for (int t = 0; t < trials; ++t)
      for (size_t ki = 0; ki < kerns.size(); ++ki) ms[ki].push_back(run(ki, reps));
    for (size_t ki = 0; ki < kerns.size(); ++ki) {
      memset(Y.ptr(), 0, size_t(M) * N * 4);
      run(ki, 1);
      double maxrel = 0;
      std::mt19937 r2(11);
      for (int t = 0; t < 64; ++t) {
        uint32_t m = t == 0 ? M - 1 : r2() % M, n = t == 0 ? N - 1 : r2() % N;
        const uint32_t wk = kerns[ki].ld ? 2 * K : K;  // weight row length
        auto dot = [&](const Buffer& w, const Buffer& sc, const Buffer& bi) {
          double r = 0;
          for (uint32_t k = 0; k < K; ++k) {
            const size_t ik = size_t(n) * wk + k;
            uint32_t q = kerns[ki].bits == 8 ? (w.as<uint32_t>()[ik / 4] >> (8 * (k % 4))) & 255
                                             : (w.as<uint32_t>()[ik / 8] >> (4 * (k % 8))) & 15;
            float wv = bf16_to_f32(sc.as<uint16_t>()[ik / 64]) * q + bf16_to_f32(bi.as<uint16_t>()[ik / 64]);
            r += double(float(__fp16(wv))) * h2f(X.as<uint16_t>()[size_t(m) * K + k]);
          }
          return r;
        };
        double ref = dot(W, S, B), got;
        if (kerns[ki].gu) {
          double gv = ref, uv = dot(W2, S2, B2);
          ref = gv / (1.0 + std::exp(-gv)) * uv;
          got = h2f(Y.as<uint16_t>()[size_t(m) * N + n]);
        } else {
          got = Y.as<float>()[size_t(m) * N + n];
        }
        maxrel = std::max(maxrel, std::fabs(got - ref) / (std::fabs(ref) + 1.0));
      }
      std::vector<double> v = ms[ki];
      std::sort(v.begin(), v.end());
      double med = v[v.size() / 2], best = v[0];
      const double kf = kerns[ki].gu ? 2.0 * flop : flop, tol = kerns[ki].gu ? 4e-3 : 1e-3;
      printf("%-26s M=%5u N=%5u K=%5u  %8.1f us  med %5.2f  best %5.2f TFLOP/s  err=%.1e%s\n", kerns[ki].name.c_str(), M, N,
             K, med * 1000, kf / (med * 1e-3) / 1e12, kf / (best * 1e-3) / 1e12, maxrel, maxrel > tol ? "  FAIL" : "");
    }
    for (Buffer* b : {&W, &S, &B, &W2, &S2, &B2, &X, &Y}) g.release(*b);
  }
}
