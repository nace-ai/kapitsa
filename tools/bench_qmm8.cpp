// dfl_qmm8 / dfl_qmm8_gu: correctness against a CPU fp64 reference and GPU timing.
//   bench_qmm8 [reps=20] [NxKxbits ...]   (default: the DFlash2 drafter shapes of both models)
// Random MLX-affine weights (codes, bf16 scales/biases), random half X rows. Each timed run is
// its own command buffer; the median GPU time is reported with the effective weight bandwidth.
// After a GPU warm-up, every shape runs its variants interleaved (K split ks in {auto,1,2,4,8} x
// rows per simdgroup rf in {1,2}); the first variant ("auto") is what the drafter uses.
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ni;

struct Mat {
  Tensor w, s, b;
  QLinear L;
};

static Mat make(uint32_t N, uint32_t K, int bits, std::mt19937& rng) {
  Gpu& g = Gpu::get();
  Mat m;
  const uint32_t wpr = K * bits / 32, gpr = K / 64;
  Buffer wb = g.alloc(size_t(N) * wpr * 4, "w", false), sb = g.alloc(size_t(N) * gpr * 2, "s", false),
         bb = g.alloc(size_t(N) * gpr * 2, "b", false);
  uint32_t* W = wb.as<uint32_t>();
  for (size_t i = 0; i < size_t(N) * wpr; ++i) W[i] = rng();
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(N) * gpr; ++i) {
    const float sc = 0.002f + 0.01f * std::fabs(nd(rng));
    sb.as<uint16_t>()[i] = f32_to_bf16(sc);
    bb.as<uint16_t>()[i] = f32_to_bf16(-sc * float((1 << bits) / 2) + 0.01f * nd(rng));
  }
  m.w = Tensor{wb, DType::U32, {int64_t(N), int64_t(wpr)}};
  m.s = Tensor{sb, DType::BF16, {int64_t(N), int64_t(gpr)}};
  m.b = Tensor{bb, DType::BF16, {int64_t(N), int64_t(gpr)}};
  m.L.w = &m.w;
  m.L.s = &m.s;
  m.L.b = &m.b;
  m.L.N = N;
  m.L.K = K;
  m.L.bits = bits;
  m.L.gs = 64;
  return m;
}

static double wval(const Mat& m, uint32_t n, uint32_t k) {
  const int bits = m.L.bits;
  const uint32_t wpr = m.L.K * bits / 32, per = 32 / bits;
  const uint32_t word = m.w.buf.as<uint32_t>()[size_t(n) * wpr + k / per];
  const uint32_t q = (word >> (bits * (k % per))) & ((1u << bits) - 1);
  const size_t gi = size_t(n) * (m.L.K / 64) + k / 64;
  return double(bf16_to_f32(m.s.buf.as<uint16_t>()[gi])) * q + bf16_to_f32(m.b.buf.as<uint16_t>()[gi]);
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int reps = argc > 1 ? atoi(argv[1]) : 20;
  struct Shape {
    uint32_t N, K;
    int bits;
    bool gu;
  };
  std::vector<Shape> shapes;
  for (int i = 2; i < argc; ++i) {
    Shape s{};
    char g[8] = {0};
    if (sscanf(argv[i], "%ux%ux%d%7s", &s.N, &s.K, &s.bits, g) < 3) fatal("shape NxKxbits[gu]");
    s.gu = std::string(g) == "gu";
    shapes.push_back(s);
  }
  if (shapes.empty())
    shapes = {{512, 2048, 4, false},    {6144, 2048, 4, false},  {2048, 4096, 4, false},  {6144, 2048, 4, true},
              {2048, 6144, 4, false},   {248320, 2048, 4, false}, {2048, 16384, 4, false}, {12288, 2048, 4, false},
              {1280, 5120, 4, false},   {6144, 5120, 4, false},  {5120, 4096, 4, false},  {17408, 5120, 4, true},
              {5120, 17408, 4, false},  {248320, 5120, 4, false}, {5120, 25600, 4, false}, {10240, 5120, 4, false},
              {6144, 2048, 8, false},   {17408, 5120, 8, true}};
  Gpu& g = Gpu::get();
  std::mt19937 rng(1234);
  bool warmed = false;
  for (const Shape& sh : shapes) {
    const uint32_t N = sh.N, K = sh.K;
    Mat A = make(N, K, sh.bits, rng), B;
    if (sh.gu) B = make(N, K, sh.bits, rng);
    A.L.w = &A.w; A.L.s = &A.s; A.L.b = &A.b;
    B.L.w = &B.w; B.L.s = &B.s; B.L.b = &B.b;
    Buffer xh = g.alloc(size_t(8) * K * 2), xs = g.alloc(size_t(K / 64) * 8 * 4);
    Buffer y = g.alloc(size_t(8) * N * 4), hh = g.alloc(size_t(16) * N * 2), hs = g.alloc(size_t(N / 64 + 1) * 8 * 4);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    __fp16* X = xh.as<__fp16>();
    for (size_t i = 0; i < size_t(8) * K; ++i) X[i] = __fp16(ud(rng));
    for (uint32_t gi = 0; gi < K / 64; ++gi)
      for (int t = 0; t < 8; ++t) {
        float s = 0;
        for (int k = 0; k < 64; ++k) s += float(X[t * K + gi * 64 + k]);
        xs.as<float>()[gi * 8 + t] = s;
      }
    struct Var {
      int ks, rf;
    };
    std::vector<Var> vars = {{0, 0}};
    if (!sh.gu)
      for (int rf : {1, 2})
        for (int ks : {1, 2, 4, 8}) vars.push_back({ks, rf});
    else
      for (int ks : {1, 2, 4}) vars.push_back({ks, 1});
    auto run = [&](Encoder& e, const Var& v) {
      if (sh.gu) dfl_enc_qmm8_gu(e, A.L, B.L, xh, xs, hh, hs, v.ks);
      else dfl_enc_qmm8(e, {{&A.L, y, N, 1.0f}}, xh, xs, K, K, v.ks, v.rf);
    };
    if (!warmed) {  // ~0.3 s of GPU work so the clocks are up before the first timing
      warmed = true;
      const double t0 = now_ms();
      while (now_ms() - t0 < 300) {
        auto e = g.begin();
        for (int i = 0; i < 8; ++i) run(*e, vars[0]);
        e->commit_and_wait();
      }
    }
    {
      auto e = g.begin();
      run(*e, vars[0]);
      e->commit_and_wait();
    }
    // reference on a sample of rows (all rows for small N)
    const uint32_t step = std::max<uint32_t>(1, N / 509);
    double num = 0, den = 0, maxe = 0;
    double hnum = 0, hden = 0;
    for (uint32_t n = 0; n < N; n += step)
      for (int t = 0; t < 8; ++t) {
        double ra = 0, rb = 0;
        for (uint32_t k = 0; k < K; ++k) {
          ra += wval(A, n, k) * double(X[t * K + k]);
          if (sh.gu) rb += wval(B, n, k) * double(X[t * K + k]);
        }
        double ref = ra, got = y.as<float>()[t * N + n];
        if (sh.gu) {
          ref = ra / (1.0 + std::exp(-ra)) * rb;
          got = double(hh.as<__fp16>()[t * N + n]);
        }
        num += (got - ref) * (got - ref);
        den += ref * ref;
        maxe = std::max(maxe, std::fabs(got - ref));
      }
    if (sh.gu) {  // HS sums of the half outputs
      for (uint32_t gi = 0; gi < N / 64; ++gi)
        for (int t = 0; t < 8; ++t) {
          double s = 0;
          for (int k = 0; k < 64; ++k) s += double(hh.as<__fp16>()[t * N + gi * 64 + k]);
          const double got = hs.as<float>()[gi * 8 + t];
          hnum += (got - s) * (got - s);
          hden += s * s;
        }
    }
    std::vector<std::vector<double>> ts(vars.size());
    for (int i = 0; i < reps + 2; ++i)
      for (size_t v = 0; v < vars.size(); ++v) {
        auto e = g.begin();
        run(*e, vars[v]);
        e->commit_and_wait();
        if (i >= 2) ts[v].push_back(e->gpu_ms());
      }
    const double bytes = double(N) * K * (sh.bits / 8.0 + 4.0 / 64) * (sh.gu ? 2 : 1);
    printf("%s %6ux%-6u b%d  rel err %.2e (max abs %.2e)%s\n", sh.gu ? "gu  " : "qmm8", N, K, sh.bits,
           std::sqrt(num / den), maxe,
           sh.gu ? (std::string("  HS rel ") + std::to_string(std::sqrt(hnum / std::max(hden, 1e-30)))).c_str() : "");
    for (size_t v = 0; v < vars.size(); ++v) {
      std::sort(ts[v].begin(), ts[v].end());
      const double med = ts[v][ts[v].size() / 2];
      char name[32];
      if (v == 0) snprintf(name, sizeof name, "auto");
      else snprintf(name, sizeof name, "ks%d rf%d", vars[v].ks, vars[v].rf);
      printf("    %-9s median %.4f ms (min %.4f)  %4.0f GB/s  %.2f TFLOP/s\n", name, med, ts[v].front(),
             bytes / med / 1e6, 2.0 * 8 * N * K * (sh.gu ? 2 : 1) / med / 1e9);
    }
  }
  return 0;
}
