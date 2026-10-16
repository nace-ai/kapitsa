// 8-row verify GEMM (qmm8, src/kernels/verify8.metal) benchmark and correctness check.
//   bench_v8 <N> <K> [bits=4] [nf=2] [cs=0:auto] [ks=0:auto] [R=8]
//   bench_v8 sweep <27b|35b> [nf=2] [cs=0:auto]   model shapes: qmm8 (8 rows) vs the 1-row decode matvec
// KERNEL=<prefix> picks the variant (default qmm8pf, the model's default).
// Weights rotate through ~1 GB of distinct matrices so timings reflect DRAM, not the caches.
// Error: max over sampled outputs of |gpu - ref| / (|ref| + rms(ref)), ref in double from the
// same half-rounded inputs the kernel reads (so it measures the GEMM, not the input rounding).
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ni;

struct QMat {
  Buffer w, s, b;
};

static QMat make_mat(Gpu& g, uint32_t N, uint32_t K, int bits, std::mt19937& rng) {
  QMat m;
  size_t words = size_t(N) * K * bits / 32, groups = size_t(N) * K / 64;
  m.w = g.alloc(words * 4, "w", false);
  m.s = g.alloc(groups * 2, "s", false);
  m.b = g.alloc(groups * 2, "b", false);
  auto* w = m.w.as<uint32_t>();
  for (size_t i = 0; i < words; ++i) w[i] = rng();
  std::uniform_real_distribution<float> d(-0.02f, 0.02f);
  auto* s = m.s.as<uint16_t>();
  auto* b = m.b.as<uint16_t>();
  for (size_t i = 0; i < groups; ++i) {
    s[i] = f32_to_bf16(std::fabs(d(rng)) * 0.2f + 1e-4f);
    b[i] = f32_to_bf16(d(rng));
  }
  return m;
}

static double cpu_dot(const QMat& m, uint32_t K, int bits, const float* x, uint32_t row) {
  const uint32_t* w = m.w.as<uint32_t>() + size_t(row) * K * bits / 32;
  const uint16_t* s = m.s.as<uint16_t>() + size_t(row) * K / 64;
  const uint16_t* b = m.b.as<uint16_t>() + size_t(row) * K / 64;
  const int per = 32 / bits;
  const uint32_t mask = (1u << bits) - 1;
  double acc = 0;
  for (uint32_t k = 0; k < K; ++k) {
    uint32_t q = (w[k / per] >> (bits * (k % per))) & mask;
    acc += (double(bf16_to_f32(s[k / 64])) * q + bf16_to_f32(b[k / 64])) * x[k];
  }
  return acc;
}

struct Cfg {
  int nf = 2, cs = 0, ks = 0;  // 0: the model's plan
};

// Host plan (kept in sync with v8_plan in src/model/qwen35_v8.cpp): 4 k-slices when a slice keeps >= 4
// groups of 64, 8 for small grids; cs = 8 / ks column blocks per threadgroup unless given.
static int auto_ks(uint32_t N, uint32_t K, int nf) {
  const uint32_t blocks = (N + 8 * nf - 1) / (8 * nf), gpr = K / 64;
  int ks = gpr >= 16 ? 4 : gpr >= 8 ? 2 : 1;
  if (ks == 4 && gpr >= 32 && blocks * 4 < 1280) ks = 8;
  return ks;
}

struct Result {
  double us, gbs, err;
};

static Result run_qmm8(Gpu& g, uint32_t N, uint32_t K, int bits, Cfg c, uint32_t R, bool check) {
  std::mt19937 rng(7);
  const size_t mbytes = size_t(N) * K * bits / 8 + size_t(N) * K / 64 * 4;
  int copies = std::clamp(int(1.0e9 / mbytes), 1, 400);
  std::vector<QMat> mats;
  for (int i = 0; i < copies; ++i) mats.push_back(make_mat(g, N, K, bits, rng));
  // inputs: fp32 rows -> x8_prep -> table + sums
  const bool x2 = getenv("KERNEL") && !strcmp(getenv("KERNEL"), "qmm8x2");
  Buffer x = g.alloc(size_t(8) * K * 4), table = g.alloc(size_t(K) * 8 * 2), sums = g.alloc(size_t(K) / 64 * 8 * 4);
  Buffer table_lo = g.alloc(size_t(K) * 8 * 2), sums_lo = g.alloc(size_t(K) / 64 * 8 * 4);
  Buffer y = g.alloc(size_t(8) * N * 4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(8) * K; ++i) x.as<float>()[i] = nd(rng);
  {
    auto e = g.begin();
    const bool v2 = K % 256 == 0 && !getenv("PREP1");
    e->use(g.pipeline(v2 ? "x8_prep2" : "x8_prep"));
    kp::X8Prep pp{K, R, K, 0, 0, 1e-6f, K, x2 ? 1u : 0u};
    e->buf(0, x); e->buf(1, x); e->buf(2, table); e->buf(3, sums); e->buf(4, x);
    e->val(5, pp);
    e->buf(6, table_lo); e->buf(7, sums_lo);
    e->dispatch(v2 ? K / 256 : 8, 1, 1, 256);
    e->commit_and_wait();
  }
  // half-rounded inputs (what the kernel multiplies; two-term hi + lo for qmm8x2)
  std::vector<float> xh(size_t(8) * K);
  for (size_t i = 0; i < xh.size(); ++i) {
    const float v = x.as<float>()[i], h = float(__fp16(v));
    xh[i] = x2 ? h + float(__fp16(v - h)) : h;
  }
  if (check) {  // the table layout against a direct CPU construction
    int bad = 0;
    const __fp16* tb = table.as<__fp16>();
    for (uint32_t gi = 0; gi < K / 64; ++gi)
      for (uint32_t lane = 0; lane < 32; ++lane)
        for (uint32_t j = 0; j < 16; ++j) {
          uint32_t qd = lane >> 2, fm = (qd & 4) | ((lane >> 1) & 3), fn = ((qd & 2) << 1) | ((lane & 1) << 1);
          uint32_t f = j / 2, e = j % 2, t = fn + e;
          uint32_t kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
          float want = t < R ? float(__fp16(x.as<float>()[size_t(t) * K + gi * 64 + kk])) : 0.0f;
          if (float(tb[(gi * 32 + lane) * 16 + j]) != want) ++bad;
        }
    double serr = 0;
    for (uint32_t gi = 0; gi < K / 64; ++gi)
      for (uint32_t t = 0; t < 8; ++t) {
        double s = 0;
        for (int k = 0; k < 64; ++k) s += t < R ? float(__fp16(x.as<float>()[size_t(t) * K + gi * 64 + k])) : 0.0;
        serr = std::max(serr, std::fabs(s - sums.as<float>()[gi * 8 + t]) / (std::fabs(s) + 1));
      }
    if (bad || serr > 1e-5) logi("x8_prep: %d table mismatches, sum err %.1e", bad, serr);
  }
  const std::string kn = std::string(getenv("KERNEL") ? getenv("KERNEL") : bits == 4 ? "qmm8pf" : "qmm8") + "_b" + std::to_string(bits) +
                         "_nf" + std::to_string(c.nf);
  Pipeline p = g.pipeline(kn);
  const bool staged = kn.rfind("qmm8t_", 0) == 0 || kn.rfind("qmm8abl3_", 0) == 0;  // table staged: ks = 1
  const int ks = staged ? 1 : c.ks > 0 ? c.ks : auto_ks(N, K, c.nf);
  c.cs = std::max(1, std::min(c.cs > 0 ? c.cs : 8, 8 / ks));  // <= 8 simdgroups per threadgroup
  const uint32_t rows_tg = 8 * c.nf * c.cs, ntg = (N + rows_tg - 1) / rows_tg;
  kp::Qmm8 prm{};
  prm.K = K;
  prm.R = R;
  prm.nseg = 1;
  prm.ks = uint32_t(ks);
  prm.N[0] = N;
  prm.y_ld[0] = N;
  prm.add[0] = 0;
  prm.tg_begin[0] = 0;
  prm.tg_begin[1] = ntg;
  auto run = [&]() {
    auto e = g.begin();
    e->use(p);
    for (int i = 0; i < copies; ++i) {
      for (int sgi = 0; sgi < 4; ++sgi) {
        e->buf(sgi * 4 + 0, mats[i].w); e->buf(sgi * 4 + 1, mats[i].s); e->buf(sgi * 4 + 2, mats[i].b);
        e->buf(sgi * 4 + 3, y);
      }
      e->buf(16, table); e->buf(17, sums);
      e->val(18, prm);
      e->buf(19, table_lo); e->buf(20, sums_lo);
      e->dispatch(ntg, 1, 1, 32 * c.cs * ks);
    }
    e->commit_and_wait();
    return e->gpu_ms();
  };
  run();
  double best = 1e9;
  for (int i = 0; i < 4; ++i) best = std::min(best, run());
  double maxerr = 0;
  if (check) {
    const QMat& m = mats.back();
    std::vector<uint32_t> rows = {0, 1, 7, 8, 15, N / 2 + 3, N - 1};
    for (int i = 0; i < 12; ++i) rows.push_back(rng() % N);
    for (uint32_t t = 0; t < R; ++t) {
      double rms = 0;
      std::vector<double> ref(rows.size());
      for (size_t i = 0; i < rows.size(); ++i) {
        ref[i] = cpu_dot(m, K, bits, xh.data() + size_t(t) * K, rows[i]);
        rms += ref[i] * ref[i];
      }
      rms = std::sqrt(rms / rows.size());
      for (size_t i = 0; i < rows.size(); ++i)
        maxerr = std::max(maxerr, std::fabs(ref[i] - y.as<float>()[size_t(t) * N + rows[i]]) / (std::fabs(ref[i]) + rms));
    }
  }
  for (auto& m : mats) { g.release(m.w); g.release(m.s); g.release(m.b); }
  g.release(x); g.release(table); g.release(sums); g.release(y); g.release(table_lo); g.release(sums_lo);
  return {best * 1000 / copies, mbytes * copies / (best * 1e-3) / 1e9, maxerr};
}

// The 1-row decode matvec as Qwen35::enc_qmv picks it (no norm, no residual).
static Result run_qmv(Gpu& g, uint32_t N, uint32_t K, int bits) {
  std::mt19937 rng(9);
  const size_t mbytes = size_t(N) * K * bits / 8 + size_t(N) * K / 64 * 4;
  int copies = std::clamp(int(1.0e9 / mbytes), 1, 400);
  std::vector<QMat> mats;
  for (int i = 0; i < copies; ++i) mats.push_back(make_mat(g, N, K, bits, rng));
  Buffer x = g.alloc(size_t(K) * 4), y = g.alloc(size_t(N) * 4);
  for (uint32_t i = 0; i < K; ++i) x.as<float>()[i] = std::normal_distribution<float>(0, 1)(rng);
  const bool huge = N >= 65536;
  uint32_t rows = huge ? 4 : 2, nsg = huge ? 4 : 8;
  std::string name = "qmv2_b" + std::to_string(bits) + "_g64_r" + std::to_string(rows);
  if (K > 4096) {
    if (bits == 8) rows = 2;
    name = "qmv3_b" + std::to_string(bits) + "_r" + std::to_string(rows);
  }
  Pipeline p = g.pipeline(name);
  kp::Qmv prm{N, K, K, N, 1e-6f};
  auto run = [&]() {
    auto e = g.begin();
    e->use(p);
    for (int i = 0; i < copies; ++i) {
      e->buf(0, mats[i].w); e->buf(1, mats[i].s); e->buf(2, mats[i].b); e->buf(3, x); e->buf(4, y);
      e->val(5, prm);
      e->buf(6, y); e->buf(7, y);
      e->dispatch((N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
    }
    e->commit_and_wait();
    return e->gpu_ms();
  };
  run();
  double best = 1e9;
  for (int i = 0; i < 4; ++i) best = std::min(best, run());
  for (auto& m : mats) { g.release(m.w); g.release(m.s); g.release(m.b); }
  g.release(x); g.release(y);
  return {best * 1000 / copies, mbytes * copies / (best * 1e-3) / 1e9, 0};
}

int main(int argc, char** argv) {
  Gpu& g = Gpu::get();
  if (argc < 3) {
    fprintf(stderr, "usage: bench_v8 <N> <K> [bits] [nf] [cs] [ks] [R] | sweep <27b|35b> [nf] [cs]\n");
    return 1;
  }
  if (!strcmp(argv[1], "sweep")) {
    struct Shape {
      const char* what;
      uint32_t N, K;
      int bits;
    };
    std::vector<Shape> shapes;
    if (!strcmp(argv[2], "27b"))
      shapes = {{"gdn in_proj qkv|z|b|a", 16480, 5120, 4}, {"attn q|k|v", 14336, 5120, 4},
                {"gdn out / attn o", 5120, 6144, 4},       {"mlp gate+up", 34816, 5120, 4},
                {"mlp down", 5120, 17408, 4},              {"lm_head", 248320, 5120, 4}};
    else
      shapes = {{"gdn in_proj qkv|z|b|a", 12352, 2048, 4}, {"attn q|k|v", 9216, 2048, 4},
                {"gdn out / attn o", 2048, 4096, 4},       {"shared/expert gate+up", 1024, 2048, 4},
                {"shared/expert down", 2048, 512, 4},      {"router (8-bit)", 256, 2048, 8},
                {"lm_head", 248320, 2048, 4}};
    Cfg c;
    if (argc > 3) c.nf = atoi(argv[3]);
    if (argc > 4) c.cs = atoi(argv[4]);
    printf("%-24s %7s %6s | %9s %7s | %9s %7s %8s | %6s\n", "shape", "N", "K", "1-row us", "GB/s", "8-row us", "GB/s",
           "err", "ratio");
    for (auto& s : shapes) {
      Result a = s.bits == 4 ? run_qmv(g, s.N, s.K, s.bits) : Result{0, 0, 0};  // no 1-row 8-bit qmv2 kernel
      Result b = run_qmm8(g, s.N, s.K, s.bits, c, 8, true);
      printf("%-24s %7u %6u | %9.1f %7.1f | %9.1f %7.1f %8.1e | %6.2f\n", s.what, s.N, s.K, a.us, a.gbs, b.us, b.gbs,
             b.err, a.us > 0 ? b.us / a.us : 0.0);
      fflush(stdout);
    }
    return 0;
  }
  uint32_t N = atoi(argv[1]), K = atoi(argv[2]);
  int bits = argc > 3 ? atoi(argv[3]) : 4;
  Cfg c;
  if (argc > 4) c.nf = atoi(argv[4]);
  if (argc > 5) c.cs = atoi(argv[5]);
  if (argc > 6) c.ks = atoi(argv[6]);
  uint32_t R = argc > 7 ? atoi(argv[7]) : 8;
  Result b = run_qmm8(g, N, K, bits, c, R, true);
  const int ks = c.ks > 0 ? c.ks : auto_ks(N, K, c.nf);
  c.cs = std::max(1, std::min(c.cs > 0 ? c.cs : 8, 8 / ks));
  printf("qmm8 b%d nf%d cs%d ks%d R=%u N=%u K=%u  %8.1f us  %6.1f GB/s  err %.1e\n", bits, c.nf, c.cs, ks, R, N, K, b.us,
         b.gbs, b.err);
  if (getenv("BASE")) {
    Result a = run_qmv(g, N, K, bits);
    printf("qmv 1-row                 %8.1f us  %6.1f GB/s  -> ratio %.2f\n", a.us, a.gbs, b.us / a.us);
  }
  return 0;
}
