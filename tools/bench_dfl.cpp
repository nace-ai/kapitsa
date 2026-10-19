// DFlash2 drafter GEMM variants on one shape: dfl_qmm8 (reference outputs) against kernels from the library or
// from an extra Metal source file appended to it (BDFL_SRC=<file>), on rotating weights (~1 GB, so DRAM-bound).
//   bench_dfl <N> <K> <reps> <kernel>[:rows_per_tg[:threads[:ks]]] ...
// Every kernel takes dfl_qmm8's bindings (0-3 W S B Y, 16 X half table, 17 group sums, 18 DflQmm8Params) plus
// 19 = the same input as a fragment-order table (verify8.metal's X8 table). rows_per_tg sets tg_begin; threads per
// threadgroup default 256. "dfl_qmm8" alone uses the host plan (dfl_enc_qmm8). Prints min / median us, GB/s and the
// max |y - ref| / (|ref| + rms(ref)) against dfl_qmm8 (itself checked against a CPU double reference on sampled
// rows). BDFL_SRC4=<file>: Metal 4 variants (tensor ops; Gpu::compile_extra with metal4). BDFL_COPIES=n rotates n
// matrices instead of ~1 GB (n small: system-cache resident), BDFL_REPEAT=n encodes the rotation n times.
// Ablations measured with it (M5 Max, 6144 x 2048): weights cache-resident 16.2 us vs 21.2 streamed, and without
// the MMAs 6.9: the 8-row 4-bit GEMM is bound by the MMA rate about as much as by DRAM.
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: bench_dfl <N> <K> <reps> <kernel>[:rows_per_tg[:threads[:ks]]] ...\n");
    return 1;
  }
  const uint32_t N = uint32_t(atoi(argv[1])), K = uint32_t(atoi(argv[2]));
  const int reps = atoi(argv[3]);
  Gpu& g = Gpu::get();
  std::string extra;
  if (const char* src = getenv("BDFL_SRC")) {
    std::ifstream f(src);
    std::stringstream ss;
    ss << f.rdbuf();
    extra = ss.str();
  }
  g.compile(extra);
  static std::string src4;  // BDFL_SRC4=<file>: a Metal 4 library (tensor ops) of more variants
  if (const char* f4 = getenv("BDFL_SRC4")) {
    std::ifstream f(f4);
    std::stringstream ss;
    ss << f.rdbuf();
    src4 = ss.str();
    g.compile_extra("bench_dfl metal4", src4.c_str(), true);
  }
  std::mt19937 rng(11);
  const int bits = 4;
  const size_t wbytes = size_t(N) * K / 2, gbytes = size_t(N) * K / 64 * 2;
  const size_t mbytes = wbytes + 2 * gbytes;
  const int copies = getenv("BDFL_COPIES") ? atoi(getenv("BDFL_COPIES")) : std::clamp(int(1.0e9 / mbytes), 1, 400);
  struct M {
    Buffer w, s, b;
    QLinear L;
    std::unique_ptr<Tensor> tw, ts, tb;
  };
  std::vector<M> mats(copies);
  std::uniform_real_distribution<float> d(-0.02f, 0.02f);
  for (auto& m : mats) {
    m.w = g.alloc(wbytes, "w", false);
    m.s = g.alloc(gbytes, "s", false);
    m.b = g.alloc(gbytes, "b", false);
    for (size_t i = 0; i < wbytes / 4; ++i) m.w.as<uint32_t>()[i] = rng();
    for (size_t i = 0; i < gbytes / 2; ++i) {
      m.s.as<uint16_t>()[i] = f32_to_bf16(std::fabs(d(rng)) * 0.2f + 1e-4f);
      m.b.as<uint16_t>()[i] = f32_to_bf16(d(rng));
    }
    m.tw = std::make_unique<Tensor>(Tensor{m.w, DType::U32, {int64_t(N), int64_t(K / 8)}});
    m.ts = std::make_unique<Tensor>(Tensor{m.s, DType::BF16, {int64_t(N), int64_t(K / 64)}});
    m.tb = std::make_unique<Tensor>(Tensor{m.b, DType::BF16, {int64_t(N), int64_t(K / 64)}});
    m.L.w = m.tw.get();
    m.L.s = m.ts.get();
    m.L.b = m.tb.get();
    m.L.N = N;
    m.L.K = K;
    m.L.bits = 4;
    m.L.gs = 64;
  }
  // inputs: row-major half table + group sums, and the fragment-order table (verify8.metal) with the same values
  Buffer xh = g.alloc(size_t(8) * K * 2), xs = g.alloc(size_t(K) / 64 * 8 * 4), tab = g.alloc(size_t(8) * K * 2);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(8) * K; ++i) xh.as<__fp16>()[i] = __fp16(nd(rng));
  for (uint32_t gi = 0; gi < K / 64; ++gi)
    for (uint32_t t = 0; t < 8; ++t) {
      float sm = 0;
      for (int k = 0; k < 64; ++k) sm += float(xh.as<__fp16>()[size_t(t) * K + gi * 64 + k]);
      xs.as<float>()[gi * 8 + t] = sm;
    }
  for (uint32_t gi = 0; gi < K / 64; ++gi)
    for (uint32_t lane = 0; lane < 32; ++lane) {
      const uint32_t q = lane / 4, fm = (q & 4) + (lane / 2) % 4, fn = (q & 2) * 2 + (lane % 2) * 2;
      for (int f = 0; f < 8; ++f)
        for (int e = 0; e < 2; ++e) {
          const uint32_t kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
          tab.as<__fp16>()[(size_t(gi) * 32 + lane) * 16 + 2 * f + e] = xh.as<__fp16>()[size_t(fn + e) * K + gi * 64 + kk];
        }
    }
  Buffer yref = g.alloc(size_t(8) * N * 4), y = g.alloc(size_t(8) * N * 4);
  struct Var {
    std::string name;
    uint32_t rows = 16, threads = 256, ks = 0;
    Pipeline p;
    std::vector<double> t;
    double err = 0;
  };
  std::vector<Var> vars;
  for (int i = 4; i < argc; ++i) {
    Var v;
    std::string s = argv[i];
    std::vector<std::string> parts;
    size_t a = 0;
    while (true) {
      size_t b = s.find(':', a);
      parts.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
      if (b == std::string::npos) break;
      a = b + 1;
    }
    v.name = parts[0];
    if (parts.size() > 1) v.rows = uint32_t(atoi(parts[1].c_str()));
    if (parts.size() > 2) v.threads = uint32_t(atoi(parts[2].c_str()));
    if (parts.size() > 3) v.ks = uint32_t(atoi(parts[3].c_str()));
    if (v.name != "dfl_qmm8" || parts.size() > 1) v.p = g.pipeline(v.name);
    vars.push_back(std::move(v));
  }
  auto enc_one = [&](Encoder& e, Var& v, M& m, const Buffer& out) {
    if (!v.p.pso) {
      dfl_enc_qmm8(e, {{&m.L, out, N, 1.0f}}, xh, xs, K, K);
      return;
    }
    kp::DflQmm8 p{};
    p.K = K;
    p.nseg = 1;
    p.ks = v.ks ? v.ks : 1;
    p.x_ld = K;
    p.N[0] = N;
    p.bits[0] = 4;
    p.y_ld[0] = N;
    p.oscale[0] = 1.0f;
    const uint32_t ntg = (N + v.rows - 1) / v.rows;
    p.tg_begin[1] = ntg;
    e.use(v.p);
    for (int sgi = 0; sgi < 4; ++sgi) {
      e.buf(sgi * 4 + 0, m.w); e.buf(sgi * 4 + 1, m.s); e.buf(sgi * 4 + 2, m.b); e.buf(sgi * 4 + 3, out);
    }
    e.buf(16, xh); e.buf(17, xs);
    e.val(18, p);
    e.buf(19, tab);
    e.dispatch(ntg, 1, 1, v.threads);
  };
  {  // reference and errors (matrix 0)
    Var ref;
    auto e = g.begin();
    enc_one(*e, ref, mats[0], yref);
    e->commit_and_wait();
    double rms = 0;
    for (size_t i = 0; i < size_t(8) * N; ++i) rms += double(yref.as<float>()[i]) * yref.as<float>()[i];
    rms = std::sqrt(rms / (8.0 * N));
    {  // dfl_qmm8 against a CPU double reference on sampled rows (same half inputs)
      double e = 0;
      const uint32_t* w = mats[0].w.as<uint32_t>();
      const uint16_t* sc = mats[0].s.as<uint16_t>();
      const uint16_t* bi = mats[0].b.as<uint16_t>();
      for (uint32_t n = 0; n < N; n += std::max(1u, N / 61)) {
        for (int t = 0; t < 8; ++t) {
          double acc = 0;
          for (uint32_t k = 0; k < K; ++k) {
            const uint32_t q = (w[size_t(n) * (K / 8) + k / 8] >> (4 * (k % 8))) & 15u;
            acc += (double(bf16_to_f32(sc[size_t(n) * (K / 64) + k / 64])) * q + bf16_to_f32(bi[size_t(n) * (K / 64) + k / 64])) *
                   double(xh.as<__fp16>()[size_t(t) * K + k]);
          }
          e = std::max(e, std::fabs(acc - yref.as<float>()[size_t(t) * N + n]) / (std::fabs(acc) + rms));
        }
      }
      printf("dfl_qmm8 vs CPU (sampled rows): max rel err %.2e\n", e);
    }
    for (auto& v : vars) {
      memset(y.ptr(), 0, y.size);
      auto e2 = g.begin();
      enc_one(*e2, v, mats[0], y);
      e2->commit_and_wait();
      for (size_t i = 0; i < size_t(8) * N; ++i) {
        const double r = yref.as<float>()[i], a = y.as<float>()[i];
        v.err = std::max(v.err, std::fabs(a - r) / (std::fabs(r) + rms));
      }
    }
  }
  for (int r = 0; r < reps + 1; ++r)
    for (auto& v : vars) {
      static const int rep = getenv("BDFL_REPEAT") ? std::max(1, atoi(getenv("BDFL_REPEAT"))) : 1;
      auto e = g.begin();
      for (int k = 0; k < rep; ++k)
        for (auto& m : mats) enc_one(*e, v, m, y);
      e->commit_and_wait();
      if (r) v.t.push_back(e->gpu_ms() * 1000 / (copies * rep));
    }
  printf("N=%u K=%u (%.2f MB, %d copies)\n", N, K, mbytes / 1e6, copies);
  for (auto& v : vars) {
    std::sort(v.t.begin(), v.t.end());
    printf("  %-28s rows/tg %3u thr %4u ks %u: min %7.2f us  med %7.2f us  %6.1f GB/s  err %.2e\n", v.name.c_str(), v.rows,
           v.threads, v.ks, v.t[0], v.t[v.t.size() / 2], mbytes / (v.t[v.t.size() / 2] * 1e3), v.err);
  }
  return 0;
}
