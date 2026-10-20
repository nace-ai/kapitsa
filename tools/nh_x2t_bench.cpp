// Nemotron 3.5 Bolt's 8-row IQ2_XXS products on the M5's matrix units (nh_x2t_v8, src/kernels/tensor/nh_x2t.metal)
// against nh_x2_v8 (nemotron/nh_x_iq2.metal) on Bolt's shared-expert shapes: a check of each against an fp64 product of
// the dequantized weights (rel.err = max |y - ref| / max |ref|), a row-independence check (a token row's outputs, bit
// for bit, with the other rows' inputs changed and with fewer live rows), then timings: 10 back-to-back calls per
// command buffer rotating over copies of the weights (~1 GB per shape, so no call finds its weights in the system
// cache), old and new interleaved, median of --reps.
//   nh_x2t_bench [--reps n] [--small] [--same] [filter ...]
// (--small: tiny shapes, one weight copy, for runs with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1; --same: one copy of
// the weights, which then stay in the system cache: the compute side)
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
extern const char* const kTensorKernelSource;
}  // namespace ni
using namespace ni;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {  // splitmix64
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
float nrand() {
  float s = 0;
  for (int i = 0; i < 4; ++i) s += float(rnd() & 0xFFFF) / 65535.0f;
  return (s - 2.0f) * 1.7320508f;
}

std::vector<uint64_t> grid_from_source() {  // nh_iq2xxs_grid out of the embedded kernel source
  std::vector<uint64_t> g;
  const std::string src = kNhKernelSource;
  size_t p = src.find("nh_iq2xxs_grid[256] = {");
  p = src.find('{', p) + 1;
  while (g.size() < 256) {
    p = src.find("0x", p);
    g.push_back(std::stoull(src.substr(p + 2, 16), nullptr, 16));
    p += 18;
  }
  return g;
}

struct V8P {
  uint32_t N, K, x_ld, y_ld, R;
  float gs;
  uint32_t epi;
};

struct Shape {
  const char* name;
  uint32_t N, K;
};

void parallel_for(size_t n, const std::function<void(size_t, size_t)>& f) {
  const size_t nt = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (size_t t = 0; t < nt; ++t) th.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
  for (auto& x : th) x.join();
}

}  // namespace

int main(int argc, char** argv) {
  int reps = 15;
  bool small = false, same = false;
  std::vector<std::string> filt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    else if (a == "--small") small = true;
    else if (a == "--same") same = true;
    else filt.push_back(a);
  }
  if (reps < 1) fatal("nh_x2t_bench: --reps takes a positive count");
  Gpu& g = Gpu::get();
  if (!g.apple10()) fatal("nh_x2t_bench: needs the M5's matrix units (Apple10)");
  g.compile_extra("Nemotron-H", kNhKernelSource);
  g.compile_extra("tensor", kTensorKernelSource, true);
  const std::vector<uint64_t> grid = grid_from_source();
  // the shared expert's up and down
  const Shape big[] = {{"sh_up", 5376, 4096}, {"sh_dn", 4096, 5376}};
  const Shape tiny[] = {{"t1", 72, 256}, {"t2", 40, 512}, {"t3", 200, 1280}};
  std::vector<Shape> shapes(small ? std::begin(tiny) : std::begin(big), small ? std::end(tiny) : std::end(big));
  for (const Shape& sh : shapes) {
    if (!filt.empty() &&
        std::none_of(filt.begin(), filt.end(), [&](auto& f) { return std::string(sh.name).find(f) != std::string::npos; }))
      continue;
    const uint32_t N = sh.N, K = sh.K;
    const float gs = 0.37f;
    // random codes (any grid index, signs and sub-block scale) and half block scales
    const size_t qb = size_t(N) * K / 4, sb = size_t(N) * K / 256 * 2;
    const int copies = small || same ? 1 : std::clamp(int(1.0e9 / double(qb + sb)), 2, 400);
    Buffer q = g.alloc(qb * copies, "q", false), s = g.alloc(sb * copies, "s", false);
    for (size_t i = 0; i < qb / 8; ++i) ((uint64_t*)q.ptr())[i] = rnd();
    for (size_t i = 0; i < sb / 2; ++i) {
      const __fp16 d = __fp16(0.0005f + 0.002f * float(rnd() & 0xFFFF) / 65535.0f);
      memcpy(s.as<uint8_t>() + 2 * i, &d, 2);
    }
    for (int c = 1; c < copies; ++c) {
      memcpy(q.as<uint8_t>() + c * qb, q.ptr(), qb);
      memcpy(s.as<uint8_t>() + c * sb, s.ptr(), sb);
    }
    // inputs: 8 half rows, and a second set with rows 1..7 different
    const uint32_t x_ld = K + 8;
    Buffer x = g.alloc(size_t(8) * x_ld * 2), x2 = g.alloc(size_t(8) * x_ld * 2);
    std::vector<float> xf(size_t(8) * K);
    for (uint32_t t = 0; t < 8; ++t)
      for (uint32_t k = 0; k < x_ld; ++k) {
        const float v = k < K ? nrand() : 0.0f;
        const __fp16 h = __fp16(v);
        x.as<__fp16>()[size_t(t) * x_ld + k] = h;
        x2.as<__fp16>()[size_t(t) * x_ld + k] = t == 0 ? h : __fp16(k < K ? nrand() : 0.0f);
        if (k < K) xf[size_t(t) * K + k] = float(h);
      }
    std::vector<double> ref(size_t(8) * N);
    parallel_for(N, [&](size_t a, size_t b) {
      std::vector<double> w(K);
      for (size_t n = a; n < b; ++n) {
        const uint8_t* qr = q.as<uint8_t>() + n * (K / 4);
        const __fp16* sr = (const __fp16*)(s.as<uint8_t>() + n * (K / 256) * 2);
        for (uint32_t sbk = 0; sbk < K / 32; ++sbk) {
          uint32_t a0, a1;
          memcpy(&a0, qr + sbk * 8, 4);
          memcpy(&a1, qr + sbk * 8 + 4, 4);
          const double db = double(float(sr[sbk / 8])) * (0.5 + double(a1 >> 28)) * 0.25;
          for (int l = 0; l < 4; ++l) {
            const uint64_t gp = grid[(a0 >> (8 * l)) & 0xFF];
            const uint32_t s7 = (a1 >> (7 * l)) & 127, sg = s7 | ((__builtin_popcount(s7) & 1) << 7);
            for (int j = 0; j < 8; ++j) {
              const double m = double((gp >> (8 * j)) & 0xFF) * db;
              w[sbk * 32 + l * 8 + j] = ((sg >> j) & 1 ? -m : m) * gs;
            }
          }
        }
        for (uint32_t t = 0; t < 8; ++t) {
          double acc = 0;
          for (uint32_t k = 0; k < K; ++k) acc += w[k] * double(xf[size_t(t) * K + k]);
          ref[size_t(t) * N + n] = acc;
        }
      }
    });
    double rmax = 0;
    for (double v : ref) rmax = std::max(rmax, std::fabs(v));
    Buffer y = g.alloc(size_t(8) * N * 4), yb = g.alloc(size_t(8) * N * 4);
    struct Var {  // (label, kernel, rows per threadgroup, threads)
      std::string label, kern;
      uint32_t rows, thr;
    };
    std::vector<Var> vars{{"x2_v8", "nh_x2_v8", 8, 32}};
    for (int k : {2, 4, 8}) vars.push_back({"x2t_v8 ks" + std::to_string(k), "nh_x2t_v8", 16, uint32_t(32 * k)});
    auto run = [&](const Var& v, const Buffer& xin, const Buffer& out, uint32_t R, int copy, int calls) {
      Pipeline pl = g.pipeline(v.kern);
      const V8P p{N, K, x_ld, N, R, gs, 0};
      auto e = g.begin();
      for (int i = 0; i < calls; ++i) {
        const int c = (copy + i) % copies;
        e->use(pl);
        e->buf(0, xin);
        e->buf(1, q, size_t(c) * qb);
        e->buf(2, s, size_t(c) * sb);
        e->buf(3, out);
        e->val(4, p);
        e->dispatch((N + v.rows - 1) / v.rows, 1, 1, v.thr);
      }
      e->commit_and_wait();
      return e->gpu_ms();
    };
    printf("%s N=%u K=%u (%.1f MB of weights, %d copies)\n", sh.name, N, K, double(qb + sb) / 1e6, copies);
    for (const Var& v : vars) {
      memset(y.ptr(), 0, y.size);
      run(v, x, y, 8, 0, 1);
      double err = 0;
      for (size_t i = 0; i < size_t(8) * N; ++i) {
        const double d = std::fabs(double(y.as<float>()[i]) - ref[i]);
        if (!(d <= err)) err = d;  // (NaN too)
      }
      memset(yb.ptr(), 0, yb.size);
      run(v, x2, yb, 8, 0, 1);
      bool indep = memcmp(y.ptr(), yb.ptr(), size_t(N) * 4) == 0;
      memset(yb.ptr(), 0, yb.size);
      run(v, x, yb, 3, 0, 1);
      indep = indep && memcmp(y.ptr(), yb.ptr(), size_t(3) * N * 4) == 0;
      bool tail_clean = true;
      for (size_t i = size_t(3) * N; i < size_t(8) * N; ++i) tail_clean = tail_clean && yb.as<float>()[i] == 0.0f;
      printf("  %-20s rel.err %.3e  rows independent %s%s\n", v.label.c_str(), err / rmax, indep ? "yes" : "NO",
             tail_clean ? "" : " (WROTE ROWS >= R)");
    }
    if (!small) {
      std::vector<std::vector<double>> ts(vars.size());
      for (int rd = 0; rd < reps + 2; ++rd)
        for (size_t i = 0; i < vars.size(); ++i) {
          const double ms = run(vars[i], x, y, 8, int(rnd() % copies), 10);
          if (rd >= 2) ts[i].push_back(ms / 10);
        }
      for (size_t i = 0; i < vars.size(); ++i) {
        std::sort(ts[i].begin(), ts[i].end());
        const double ms = ts[i][ts[i].size() / 2];
        printf("  %-20s %8.1f us  %6.1f GB/s (min %.1f us)\n", vars[i].label.c_str(), ms * 1000,
               double(qb + sb) / ms / 1e6, ts[i][0] * 1000);
      }
    }
    for (Buffer* b : {&q, &s, &x, &x2, &y, &yb}) g.release(*b);  // (each shape's ~1 GB of copies)
  }
  return 0;
}
