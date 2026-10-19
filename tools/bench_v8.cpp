// 8-row verify GEMM (qmm8, src/kernels/verify8.metal) benchmark and correctness check.
//   bench_v8 <N> <K> [bits=4] [nf=2] [cs=0:auto] [ks=0:auto] [R=8]
//   bench_v8 sweep <27b|35b> [nf=2] [cs=0:auto]   model shapes: qmm8 (8 rows) vs the 1-row decode matvec
//   bench_v8 dfl <27b|35b>   DFlash2 drafter shapes: the drafter's dfl_qmm8 / dfl_qmm8_gu vs qmm8pf /
//                            qmm8_gu1k2 on the same rotating weights (alternating runs, best of 6)
// KERNEL=<prefix> picks the variant (default qmm8pf, the model's default).
// Weights rotate through ~1 GB of distinct matrices so timings reflect DRAM, not the caches.
// Error: max over sampled outputs of |gpu - ref| / (|ref| + rms(ref)), ref in double from the
// same half-rounded inputs the kernel reads (so it measures the GEMM, not the input rounding).
#include "metal/gpu.h"
#include "model/dflash2.h"
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
namespace ni {
extern const char* const kTensorKernelSource;
}

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
  Buffer ttab = g.alloc(size_t(K) * 8 * 2);  // qmm8t's T table
  Buffer y = g.alloc(size_t(8) * N * 4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(8) * K; ++i) x.as<float>()[i] = nd(rng);
  {
    auto e = g.begin();
    const bool v2 = K % 256 == 0 && !getenv("PREP1");
    e->use(g.pipeline(v2 ? "x8_prep2" : "x8_prep"));
    kp::X8Prep pp{K, R, K, 0, 0, 1e-6f, K, x2 ? 1u : 0u, 0u, v2 ? 1u : 0u};
    e->buf(0, x); e->buf(1, x); e->buf(2, table); e->buf(3, sums); e->buf(4, x);
    e->val(5, pp);
    e->buf(6, table_lo); e->buf(7, sums_lo);
    e->buf(8, ttab);
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
  // KERNEL=qmm8t / qmm8tp (the tensor_ops GEMM): nf = NT rows per simdgroup (16, 32), cs = 1
  const bool tens = kn.rfind("qmm8t", 0) == 0;
  if (tens) g.compile_extra("tensor", kTensorKernelSource, true);
  Pipeline p = g.pipeline(tens ? std::string(getenv("KERNEL")) + "_b4_nt" + std::to_string(c.nf) : kn);
  const int ks = c.ks > 0 ? c.ks : auto_ks(N, K, c.nf);
  c.cs = tens ? 1 : std::max(1, std::min(c.cs > 0 ? c.cs : 8, 8 / ks));  // <= 8 simdgroups per threadgroup
  const uint32_t rows_tg = tens ? uint32_t(c.nf) : 8 * c.nf * c.cs, ntg = (N + rows_tg - 1) / rows_tg;
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
      e->buf(16, tens ? ttab : table); e->buf(17, sums);
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
  g.release(ttab);
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

// ---- drafter GEMM comparison: dfl_qmm8 (row-major half input table) vs qmm8pf (fragment-order table) ----
static QLinear qlin_of(const QMat& m, uint32_t N, uint32_t K, int bits, std::vector<std::unique_ptr<Tensor>>& keep) {
  QLinear L;
  keep.push_back(std::make_unique<Tensor>(Tensor{m.w, DType::U32, {int64_t(N), int64_t(K * bits / 32)}}));
  L.w = keep.back().get();
  keep.push_back(std::make_unique<Tensor>(Tensor{m.s, DType::BF16, {int64_t(N), int64_t(K / 64)}}));
  L.s = keep.back().get();
  keep.push_back(std::make_unique<Tensor>(Tensor{m.b, DType::BF16, {int64_t(N), int64_t(K / 64)}}));
  L.b = keep.back().get();
  L.N = N;
  L.K = K;
  L.bits = bits;
  L.gs = 64;
  return L;
}

// gu: gate and up of N x K each (fused SwiGLU kernels), else one N x K matrix
static void run_dfl_cmp(Gpu& g, const char* what, uint32_t N, uint32_t K, bool gu) {
  std::mt19937 rng(11);
  const int bits = 4;
  const size_t mbytes = (size_t(N) * K * bits / 8 + size_t(N) * K / 64 * 4) * (gu ? 2 : 1);
  const int copies = std::clamp(int(1.0e9 / mbytes), 1, 400);
  std::vector<QMat> mats;
  for (int i = 0; i < copies * (gu ? 2 : 1); ++i) mats.push_back(make_mat(g, N, K, bits, rng));
  std::vector<std::unique_ptr<Tensor>> keep;
  std::vector<QLinear> L;
  for (auto& m : mats) L.push_back(qlin_of(m, N, K, bits, keep));
  // inputs in both layouts (same values)
  Buffer x = g.alloc(size_t(8) * K * 4), table = g.alloc(size_t(K) * 8 * 2), sums = g.alloc(size_t(K) / 64 * 8 * 4);
  Buffer xh = g.alloc(size_t(8) * K * 2), xs = g.alloc(size_t(K) / 64 * 8 * 4);
  Buffer y = g.alloc(size_t(8) * N * 4), hh = g.alloc(size_t(16) * N * 2), hs = g.alloc(size_t(N) / 64 * 8 * 4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(8) * K; ++i) x.as<float>()[i] = nd(rng);
  for (uint32_t t = 0; t < 8; ++t)
    for (uint32_t k = 0; k < K; ++k) xh.as<__fp16>()[size_t(t) * K + k] = __fp16(x.as<float>()[size_t(t) * K + k]);
  for (uint32_t gi = 0; gi < K / 64; ++gi)
    for (uint32_t t = 0; t < 8; ++t) {
      float sm = 0;
      for (int k = 0; k < 64; ++k) sm += float(xh.as<__fp16>()[size_t(t) * K + gi * 64 + k]);
      xs.as<float>()[gi * 8 + t] = sm;
    }
  {
    auto e = g.begin();
    e->use(g.pipeline("x8_prep2"));
    kp::X8Prep pp{K, 8, K, 0, 0, 1e-6f, K, 0u};
    e->buf(0, x); e->buf(1, x); e->buf(2, table); e->buf(3, sums); e->buf(4, x);
    e->val(5, pp);
    e->buf(6, table); e->buf(7, sums); e->buf(8, table);
    e->dispatch(K / 256, 1, 1, 256);
    e->commit_and_wait();
  }
  auto run_dfl = [&]() {
    auto e = g.begin();
    for (int i = 0; i < copies; ++i) {
      if (gu) dfl_enc_qmm8_gu(*e, L[2 * i], L[2 * i + 1], xh, xs, hh, hs);
      else dfl_enc_qmm8(*e, {{&L[i], y, N, 1.0f}}, xh, xs, K, K);
    }
    e->commit_and_wait();
    return e->gpu_ms();
  };
  // qmm8pf with the model's plan (v8_plan: 2 column blocks x 4 k-slices, 8 k-slices for small grids)
  const int nf = 2, ks = auto_ks(N, K, nf), cs = std::max(1, 8 / ks);
  const uint32_t rows_tg = 8 * nf * cs, ntg = (N + rows_tg - 1) / rows_tg;
  Pipeline p8 = g.pipeline("qmm8pf_b4_nf2"), pgu = g.pipeline("qmm8_gu1k2_b4");
  kp::Qmm8 prm{};
  prm.K = K;
  prm.R = 8;
  prm.nseg = 1;
  prm.ks = uint32_t(ks);
  prm.N[0] = N;
  prm.y_ld[0] = N;
  prm.tg_begin[1] = ntg;
  auto run_v8 = [&]() {
    auto e = g.begin();
    for (int i = 0; i < copies; ++i) {
      if (gu) {
        kp::Qmm8Gu gp{K, N, 0, 0, 0, 0, 0};
        const QMat& a = mats[2 * i];
        const QMat& b = mats[2 * i + 1];
        e->use(pgu);
        e->buf(0, a.w); e->buf(1, a.s); e->buf(2, a.b); e->buf(3, b.w); e->buf(4, b.s); e->buf(5, b.b);
        e->buf(6, a.w); e->buf(7, a.s); e->buf(8, a.b); e->buf(9, b.w); e->buf(10, b.s); e->buf(11, b.b);
        e->buf(12, table); e->buf(13, sums); e->buf(14, hh); e->buf(15, hs);
        e->val(16, gp);
        e->buf(17, sums); e->buf(18, sums);
        e->dispatch(N / 64, 1, 1, 512);
      } else {
        e->use(p8);
        for (int sgi = 0; sgi < 4; ++sgi) {
          e->buf(sgi * 4 + 0, mats[i].w); e->buf(sgi * 4 + 1, mats[i].s); e->buf(sgi * 4 + 2, mats[i].b);
          e->buf(sgi * 4 + 3, y);
        }
        e->buf(16, table); e->buf(17, sums);
        e->val(18, prm);
        e->buf(19, table); e->buf(20, sums);
        e->dispatch(ntg, 1, 1, 32 * cs * ks);
      }
    }
    e->commit_and_wait();
    return e->gpu_ms();
  };
  run_dfl();
  run_v8();
  double bd = 1e9, bv = 1e9;
  for (int i = 0; i < 6; ++i) {
    bd = std::min(bd, run_dfl());
    bv = std::min(bv, run_v8());
  }
  const double ud = bd * 1000 / copies, uv = bv * 1000 / copies;
  printf("%-26s %7u x %-6u%s | dfl_qmm8%s %8.1f us %6.1f GB/s | %s %8.1f us %6.1f GB/s | v8/dfl %.3f\n", what, N, K,
         gu ? " x2" : "   ", gu ? "_gu" : "   ", ud, mbytes / (ud * 1e3), gu ? "qmm8_gu1k2" : "qmm8pf    ", uv,
         mbytes / (uv * 1e3), uv / ud);
  fflush(stdout);
  for (auto& m : mats) { g.release(m.w); g.release(m.s); g.release(m.b); }
  for (Buffer* b : {&x, &table, &sums, &xh, &xs, &y, &hh, &hs}) g.release(*b);
}

// ---- interleaved comparison of qmm8 kernel variants on one shape (same weights, alternating runs) ----
// bench_v8 cmp <N> <K> <reps> <kernel>[:nf][:cs][:ks] ...   (kernel = pipeline name prefix, e.g. qmm8snb_b4; names
// with "16" allow 16 simdgroups per threadgroup). Prints min / median us and GB/s per kernel.
static void run_cmp(Gpu& g, uint32_t N, uint32_t K, int reps, const std::vector<std::string>& specs) {
  std::mt19937 rng(7);
  const int bits = 4;
  const size_t mbytes = size_t(N) * K * bits / 8 + size_t(N) * K / 64 * 4;
  const int copies = std::clamp(int(1.0e9 / mbytes), 1, 400);
  std::vector<QMat> mats;
  for (int i = 0; i < copies; ++i) mats.push_back(make_mat(g, N, K, bits, rng));
  Buffer x = g.alloc(size_t(8) * K * 4), y = g.alloc(size_t(8) * N * 4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(8) * K; ++i) x.as<float>()[i] = nd(rng);
  Buffer tab = g.alloc(size_t(K) * 8 * 2), sum = g.alloc(size_t(K) / 64 * 8 * 4), ttab = g.alloc(size_t(K) * 8 * 2);
  {
    auto e = g.begin();
    e->use(g.pipeline("x8_prep2"));
    kp::X8Prep pp{K, 8, K, 0, 0, 1e-6f, K, 0u, 0u, 1u};
    e->buf(0, x); e->buf(1, x); e->buf(2, tab); e->buf(3, sum); e->buf(4, x);
    e->val(5, pp);
    e->buf(6, tab); e->buf(7, sum); e->buf(8, ttab);
    e->dispatch(K / 256, 1, 1, 256);
    e->commit_and_wait();
  }
  struct K8 {
    std::string name;
    Pipeline p;
    int nf, cs, ks;
    bool tens;
    uint32_t ntg;
    kp::Qmm8 prm;
    std::vector<double> t;
  };
  std::vector<K8> ks;
  for (auto& sp : specs) {
    K8 k{};
    std::vector<std::string> f;
    size_t a = 0;
    while (true) {
      size_t b = sp.find(':', a);
      f.push_back(sp.substr(a, b == std::string::npos ? std::string::npos : b - a));
      if (b == std::string::npos) break;
      a = b + 1;
    }
    k.tens = f[0].rfind("qmm8t", 0) == 0;  // qmm8t_b4 / qmm8tp_b4: nf = NT weight rows per simdgroup
    if (k.tens) g.compile_extra("tensor", kTensorKernelSource, true);
    k.nf = f.size() > 1 ? atoi(f[1].c_str()) : k.tens ? 16 : 2;
    k.name = f[0] + (k.tens ? "_nt" : "_nf") + std::to_string(k.nf);
    k.p = g.pipeline(k.name);
    k.ks = f.size() > 3 && atoi(f[3].c_str()) > 0 ? atoi(f[3].c_str()) : auto_ks(N, K, k.nf);
    const int maxsg = f[0].find("16_") != std::string::npos || f[0].size() > 2 && f[0].substr(f[0].size() - 2) == "16" ? 16 : 8;
    k.cs = k.tens ? 1 : std::max(1, std::min(f.size() > 2 && atoi(f[2].c_str()) > 0 ? atoi(f[2].c_str()) : 8, maxsg / k.ks));
    const uint32_t rows_tg = k.tens ? uint32_t(k.nf) : 8 * k.nf * k.cs;
    k.ntg = (N + rows_tg - 1) / rows_tg;
    k.prm = kp::Qmm8{};
    k.prm.K = K;
    k.prm.R = 8;
    k.prm.nseg = 1;
    k.prm.ks = uint32_t(k.ks);
    k.prm.N[0] = N;
    k.prm.y_ld[0] = N;
    k.prm.tg_begin[1] = k.ntg;
    ks.push_back(k);
  }
  auto run = [&](K8& k) {
    auto e = g.begin();
    e->use(k.p);
    for (int i = 0; i < copies; ++i) {
      for (int sgi = 0; sgi < 4; ++sgi) {
        e->buf(sgi * 4 + 0, mats[i].w); e->buf(sgi * 4 + 1, mats[i].s); e->buf(sgi * 4 + 2, mats[i].b);
        e->buf(sgi * 4 + 3, y);
      }
      e->buf(16, k.tens ? ttab : tab); e->buf(17, sum);
      e->val(18, k.prm);
      e->buf(19, tab); e->buf(20, sum);
      e->dispatch(k.ntg, 1, 1, 32 * k.cs * k.ks);
    }
    e->commit_and_wait();
    return e->gpu_ms() * 1000.0 / copies;
  };
  for (auto& k : ks) run(k);  // warm-up
  for (int r = 0; r < reps; ++r)
    for (auto& k : ks) k.t.push_back(run(k));
  double base_med = 0;
  for (size_t i = 0; i < ks.size(); ++i) {
    auto& k = ks[i];
    std::vector<double> t = k.t;
    std::sort(t.begin(), t.end());
    const double mn = t.front(), med = t[t.size() / 2];
    if (i == 0) base_med = med;
    printf("%-22s nf%d cs%d ks%d tg%5u mt%4u | min %8.1f us %6.1f GB/s | median %8.1f us %6.1f GB/s | med/first %.3f\n",
           k.name.c_str(), k.nf, k.cs, k.ks, k.ntg, k.p.max_threads, mn, mbytes / (mn * 1e3), med, mbytes / (med * 1e3),
           med / base_med);
  }
  fflush(stdout);
  for (auto& m : mats) { g.release(m.w); g.release(m.s); g.release(m.b); }
  for (Buffer* b : {&x, &y, &tab, &sum, &ttab}) g.release(*b);
}

int main(int argc, char** argv) {
  Gpu& g = Gpu::get();
  if (argc >= 6 && !strcmp(argv[1], "cmp")) {
    std::vector<std::string> specs;
    for (int i = 5; i < argc; ++i) specs.push_back(argv[i]);
    printf("cmp N=%s K=%s reps=%s\n", argv[2], argv[3], argv[4]);
    run_cmp(g, uint32_t(atoi(argv[2])), uint32_t(atoi(argv[3])), atoi(argv[4]), specs);
    return 0;
  }
  if (argc >= 3 && !strcmp(argv[1], "dfl")) {
    const bool b27 = !strcmp(argv[2], "27b");
    const uint32_t H = b27 ? 5120 : 2048, I = b27 ? 17408 : 6144, nH = b27 ? 25600 : 16384, NL = b27 ? 5 : 6;
    run_dfl_cmp(g, "conv kernel projection", 4 * (H / 16), H, false);
    run_dfl_cmp(g, "q|kv", 6144, H, false);
    run_dfl_cmp(g, "o", H, 4096, false);
    run_dfl_cmp(g, "gate+up (fused SwiGLU)", I, H, true);
    run_dfl_cmp(g, "down", H, I, false);
    run_dfl_cmp(g, "lm_head", 248320, H, false);
    run_dfl_cmp(g, "commit fc", H, nH, false);
    run_dfl_cmp(g, "commit k|v all layers", NL * 2048, H, false);
    return 0;
  }
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
