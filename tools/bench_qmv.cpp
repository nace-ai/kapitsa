// Microbenchmark + correctness check for the quantized matvec kernels.
// Rotates through enough distinct weight matrices (~1 GB) that results reflect DRAM
// bandwidth, not the system-level cache.
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ni;

struct QmvParams {
  uint32_t N, K, x_stride, y_stride;
  float eps;
};

struct QMat {
  Buffer w, s, b;
};

static QMat make_mat(Gpu& g, uint32_t N, uint32_t K, int bits, int gs, std::mt19937& rng) {
  QMat m;
  size_t words = size_t(N) * K * bits / 32, groups = size_t(N) * K / gs;
  m.w = g.alloc(words * 4);
  m.s = g.alloc(groups * 2);
  m.b = g.alloc(groups * 2);
  auto* w = m.w.as<uint32_t>();
  for (size_t i = 0; i < words; ++i) w[i] = rng();
  std::uniform_real_distribution<float> d(-0.02f, 0.02f);
  auto* s = m.s.as<uint16_t>();
  auto* b = m.b.as<uint16_t>();
  for (size_t i = 0; i < groups; ++i) {
    s[i] = f32_to_bf16(std::fabs(d(rng)) * 0.1f);
    b[i] = f32_to_bf16(d(rng));
  }
  return m;
}

static float cpu_row(const QMat& m, uint32_t K, int bits, int gs, const float* x, uint32_t row) {
  const uint32_t* w = m.w.as<uint32_t>() + size_t(row) * K * bits / 32;
  const uint16_t* s = m.s.as<uint16_t>() + size_t(row) * K / gs;
  const uint16_t* b = m.b.as<uint16_t>() + size_t(row) * K / gs;
  int per = 32 / bits;
  uint32_t mask = (1u << bits) - 1;
  double acc = 0;
  for (uint32_t k = 0; k < K; ++k) {
    uint32_t q = (w[k / per] >> (bits * (k % per))) & mask;
    float wr = bf16_to_f32(s[k / gs]) * float(q) + bf16_to_f32(b[k / gs]);
    acc += double(wr) * x[k];
  }
  return float(acc);
}

int main(int argc, char** argv) {
  Gpu& g = Gpu::get();
  logi("device %s, working set %.1f GB", g.name().c_str(), g.recommended_working_set() / 1e9);
  std::mt19937 rng(1234);

  struct Case {
    const char* kernel;
    uint32_t N, K;
    int bits, gs, rows, nt;
  };
  std::vector<Case> cases;
  static std::vector<std::string> keep;  // kernel names of SHAPES cases
  if (const char* sh = getenv("SHAPES")) {
    // SHAPES=kernel:N:K:rows[,...] e.g. qmv2_b4_g64_r4:2560:4096:4 (simdgroups per threadgroup: argv[1])
    std::string all = sh;
    for (size_t p = 0; p < all.size();) {
      size_t q = all.find(',', p);
      std::string item = all.substr(p, q == std::string::npos ? std::string::npos : q - p);
      char name[128];
      unsigned N = 0, K = 0, r = 0;
      if (sscanf(item.c_str(), "%127[^:]:%u:%u:%u", name, &N, &K, &r) == 4) {
        keep.push_back(name);
        cases.push_back({nullptr, N, K, 4, 64, int(r), 1});
      }
      p = q == std::string::npos ? all.size() : q + 1;
    }
    for (size_t i = 0; i < cases.size(); ++i) cases[i].kernel = keep[i].c_str();
  } else if (getenv("V2")) {
    int r = atoi(getenv("V2"));
    std::string s = "qmv2_b4_g64_r" + std::to_string(r);
    static std::vector<std::string> names = {s, s + "_norm"};
    cases = {{names[0].c_str(), 8192, 2048, 4, 64, r, 1}, {names[1].c_str(), 8192, 2048, 4, 64, r, 1},
             {names[0].c_str(), 2048, 4096, 4, 64, r, 1}, {names[0].c_str(), 2048, 512, 4, 64, r, 1},
             {names[0].c_str(), 248320, 2048, 4, 64, r, 1}, {names[0].c_str(), 512, 2048, 4, 64, r, 1}};
  } else cases = {
      {"qmv_b4_g64_v16_r4_t1", 8192, 2048, 4, 64, 4, 1},  {"qmv_b4_g64_v32_r4_t1", 8192, 2048, 4, 64, 4, 1},
      {"qmv_b4_g64_v16_r8_t1", 8192, 2048, 4, 64, 8, 1},  {"qmv_b4_g64_v32_r2_t1", 8192, 2048, 4, 64, 2, 1},
      {"qmv_b4_g64_v16_r2_t1", 8192, 2048, 4, 64, 2, 1},  {"qmv_b4_g64_v16_r4_t1", 2048, 4096, 4, 64, 4, 1},
      {"qmv_b4_g64_v32_r4_t1", 2048, 4096, 4, 64, 4, 1},  {"qmv_b4_g64_v16_r4_t1", 2048, 512, 4, 64, 4, 1},
      {"qmv_b4_g64_v16_r4_t1", 248320, 2048, 4, 64, 4, 1}, {"qmv_b4_g64_v32_r4_t1", 248320, 2048, 4, 64, 4, 1},
      {"qmv_b8_g64_v16_r4_t1", 256, 2048, 8, 64, 4, 1},   {"qmv_b4_g64_v16_r4_t4", 8192, 2048, 4, 64, 4, 4},
  };
  int nsg = argc > 1 ? atoi(argv[1]) : 2;
  std::vector<uint16_t> nwv(8192, f32_to_bf16(1.0f));
  Buffer nw = g.alloc(8192 * 2);
  memcpy(nw.ptr(), nwv.data(), 8192 * 2);

  for (auto& c : cases) {
    size_t bytes_per = size_t(c.N) * c.K * c.bits / 8 + size_t(c.N) * c.K / c.gs * 4;
    int copies = std::max<int>(1, int(1.0e9 / bytes_per));
    copies = std::min(copies, 200);
    std::vector<QMat> mats;
    for (int i = 0; i < copies; ++i) mats.push_back(make_mat(g, c.N, c.K, c.bits, c.gs, rng));
    Buffer x = g.alloc(size_t(c.K) * c.nt * 4), y = g.alloc(size_t(c.N) * c.nt * 4);
    Buffer z = g.alloc(size_t(c.N) * c.nt * 4);  // the residual the _add kernels read: zero, so y = W x
    memset(z.ptr(), 0, z.size);
    std::normal_distribution<float> nd(0, 1);
    for (uint32_t i = 0; i < c.K * c.nt; ++i) x.as<float>()[i] = nd(rng);
    Pipeline p = g.pipeline(c.kernel);
    QmvParams prm{c.N, c.K, c.K, c.N, 1e-6f};
    uint32_t rows_per_tg = c.rows * nsg;
    uint32_t ntg = (c.N + rows_per_tg - 1) / rows_per_tg;

    auto run = [&](int reps) {
      auto e = g.begin();
      e->use(p);
      for (int r = 0; r < reps; ++r)
        for (auto& m : mats) {
          e->buf(0, m.w); e->buf(1, m.s); e->buf(2, m.b); e->buf(3, x); e->buf(4, y);
          e->val(5, prm); e->buf(6, z); e->buf(7, nw);
          e->dispatch(ntg, 1, 1, 32 * nsg);
        }
      e->commit_and_wait();
      return e->gpu_ms();
    };
    run(1); // warm
    double ms = 1e30;
    for (int it = 0; it < 3; ++it) ms = std::min(ms, run(1));
    double gbs = bytes_per * copies / (ms * 1e-3) / 1e9;

    // correctness on the last matrix (y holds its result)
    if (std::string(c.kernel).find("_norm") != std::string::npos) {
      double ss = 0;
      for (uint32_t i = 0; i < c.K; ++i) ss += double(x.as<float>()[i]) * x.as<float>()[i];
      float inv = float(1.0 / std::sqrt(ss / c.K + 1e-6));
      for (uint32_t i = 0; i < c.K; ++i) x.as<float>()[i] *= inv;  // reference uses normalized x
    }
    double maxrel = 0;
    for (uint32_t t = 0; t < (uint32_t)c.nt; ++t)
      for (uint32_t row : {0u, 1u, c.N / 2, c.N - 1}) {
        float ref = cpu_row(mats.back(), c.K, c.bits, c.gs, x.as<float>() + t * c.K, row);
        float got = y.as<float>()[t * c.N + row];
        maxrel = std::max(maxrel, double(std::fabs(ref - got) / (std::fabs(ref) + 1e-3)));
      }
    printf("%-26s N=%6u K=%5u x%-3d %7.1f us/mat  %6.1f GB/s  maxrel=%.2e\n", c.kernel, c.N, c.K, copies,
           ms * 1000 / copies, gbs, maxrel);
    for (auto& m : mats) { g.release(m.w); g.release(m.s); g.release(m.b); }
    g.release(x); g.release(y); g.release(z);
  }
}
