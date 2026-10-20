// Nemotron-H's 8-row NVFP4 products on the M5's matrix units (nh_x4t_v8_l0, src/kernels/tensor/nh_v8t.metal) against
// nh_x4_v8_l0 on Nemotron 3.5 Lightning's shapes: a check of each against an fp64 product of the dequantized weights (rel.err = max |y - ref| / max |ref|), a row-independence check (a token row's outputs, bit for bit, with
// the other rows' inputs changed and with fewer live rows), then timings: 10 back-to-back calls per command buffer
// rotating over copies of the weights (~1 GB per shape, so no call finds its weights in the system cache), old and
// new interleaved, median of --reps.
//   nh_v8t_bench [--reps n] [--small] [--same] [--gpr] [--ks n] [filter ...]   (--ks: only that many simdgroups)
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
#include <iterator>
#include <fstream>
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
double e4m3(uint8_t x) {
  if (x == 0x7F) return 0.0;
  const int e = (x >> 3) & 0xF, m = x & 7;
  return e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + double(m) / 8.0, e - 7);
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

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

// ---- GPR / stack from a binary archive (ni_compile's parser) ----
uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint32_t le32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
uint64_t le64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }
bool macho_section(const uint8_t* m, size_t n, const char* seg, const char* sect, const uint8_t** out, size_t* len) {
  if (n < 32) return false;
  uint32_t ncmds = le32(m + 16);
  size_t off = 32;
  for (uint32_t i = 0; i < ncmds && off + 8 <= n; ++i) {
    uint32_t cmd = le32(m + off), csz = le32(m + off + 4);
    if (cmd == 0x19 && off + 72 <= n) {
      uint32_t nsect = le32(m + off + 64);
      for (uint32_t s = 0; s < nsect; ++s) {
        const uint8_t* sd = m + off + 72 + 80 * s;
        if (sd + 80 > m + n) return false;
        if (strncmp((const char*)sd, sect, 16) == 0 && strncmp((const char*)sd + 16, seg, 16) == 0) {
          uint64_t size = le64(sd + 40);
          uint32_t soff = le32(sd + 48);
          if (soff + size > n) return false;
          *out = m + soff;
          *len = size;
          return true;
        }
      }
    }
    off += csz;
  }
  return false;
}
void kernel_stats(const std::vector<uint8_t>& b, long* stack, long* gprs) {
  *stack = *gprs = -1;
  if (b.size() < 8) return;
  uint32_t nfat = be32(b.data() + 4);
  for (uint32_t i = 0; i < nfat && 8 + 20 * (i + 1) <= b.size(); ++i) {
    const uint8_t* e = b.data() + 8 + 20 * i;
    if (be32(e) != 0x01000013) continue;
    uint32_t off = be32(e + 8), size = be32(e + 12);
    if (off + size > b.size()) return;
    const uint8_t *inner, *md;
    size_t ninner, nmd;
    if (!macho_section(b.data() + off, size, "__TEXT", "__compute", &inner, &ninner)) return;
    if (!macho_section(inner, ninner, "__GPU_METADATA", "__compute", &md, &nmd)) return;
    std::vector<uint32_t> w(nmd / 4);
    for (size_t k = 0; k < w.size(); ++k) w[k] = le32(md + 4 * k);
    for (size_t k = 4; k + 2 < w.size(); ++k) {
      uint32_t f = w[k];
      if (f >> 24 != 1 || (((f >> 16) & 0xff) != 1 && ((f >> 16) & 0xff) != 3) || f == 0x01000000 || (f & 0xff) > 3)
        continue;
      uint32_t c = w[k + 1];
      bool seen = c > 0 && (w[k - 1] == c || w[k - 2] == c || w[k - 3] == c || w[k - 4] == c);
      if (seen && w[k + 2] >= 4 && w[k + 2] <= 256) { *stack = c; *gprs = w[k + 2]; return; }
      if (c <= 256) { *stack = 0; *gprs = c; return; }
    }
    return;
  }
}
std::string gpr_str(const std::string& fn) {
  const char* tmp = getenv("TMPDIR");
  const std::string path = std::string(tmp ? tmp : "/tmp") + "/v8t_" + fn + ".bin";
  if (!Gpu::get().archive(fn, path)) return "gpr ?";
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  remove(path.c_str());
  long st, gp;
  kernel_stats(b, &st, &gp);
  char buf[64];
  snprintf(buf, sizeof buf, "gpr %ld%s", gp, st > 0 ? (" stack " + std::to_string(st)).c_str() : "");
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  int reps = 15, ks = 0;
  bool small = false, same = false, gpr = false;
  std::vector<std::string> filt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    else if (a == "--ks" && i + 1 < argc) ks = atoi(argv[++i]);
    else if (a == "--small") small = true;
    else if (a == "--same") same = true;
    else if (a == "--gpr") gpr = true;
    else filt.push_back(a);
  }
  if (reps < 1 || ks < 0 || ks > 8) fatal("nh_v8t_bench: --reps takes a positive count, --ks 1-8");
  Gpu& g = Gpu::get();
  if (!g.apple10()) fatal("nh_v8t_bench: needs the M5's matrix units (Apple10)");
  g.compile_extra("Nemotron-H", kNhKernelSource);
  g.compile_extra("tensor", kTensorKernelSource, true);
  // the LM head, the shared expert's up and down, the drafter's 6144-row matrices
  const Shape big[] = {{"head", 131072, 2688}, {"sh_up", 3712, 2688}, {"sh_dn", 2688, 3712}, {"dsp_gu", 6144, 2688}};
  const Shape tiny[] = {{"t1", 72, 128}, {"t2", 40, 192}, {"t3", 200, 640}};
  std::vector<Shape> shapes(small ? std::begin(tiny) : std::begin(big), small ? std::end(tiny) : std::end(big));
  for (const Shape& sh : shapes) {
    if (!filt.empty() && std::none_of(filt.begin(), filt.end(), [&](auto& f) { return std::string(sh.name).find(f) != std::string::npos; }))
      continue;
    const uint32_t N = sh.N, K = sh.K;
    const float gs = 0.37f;
    // codes and UE4M3 block scales (mostly realistic, some subnormal and extreme), packed in layout L0
    std::vector<uint8_t> code(size_t(N) * K), sc(size_t(N) * K / 16);
    for (auto& c : code) c = uint8_t(rnd() & 15);
    for (auto& b : sc) {
      const uint64_t r = rnd();
      const int sel = int(r & 63);
      b = sel == 0 ? uint8_t((r >> 8) & 7) : sel == 1 ? uint8_t(0x70 + ((r >> 8) % 15)) : uint8_t(0x20 + ((r >> 8) % 0x50));
      if (b == 0x7F) b = 0x7E;
    }
    const size_t qb = size_t(N) * K / 2, sb = size_t(N) * K / 16;
    const int copies = small || same ? 1 : std::clamp(int(1.0e9 / double(qb + sb)), 2, 400);
    Buffer q = g.alloc(qb * copies, "q", false), s = g.alloc(sb * copies, "s", false);
    for (size_t b = 0; b < size_t(N) * K / 16; ++b)
      for (int i = 0; i < 8; ++i) q.as<uint8_t>()[b * 8 + i] = uint8_t(code[b * 16 + i] | (code[b * 16 + i + 8] << 4));
    memcpy(s.ptr(), sc.data(), sb);
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
        for (uint32_t k = 0; k < K; ++k) w[k] = kE2M1[code[n * K + k]] * e4m3(sc[(n * K + k) / 16]) * gs;
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
    // the variants: (label, kernel, rows per threadgroup, threads)
    struct Var {
      std::string label, kern;
      uint32_t rows, thr;
    };
    std::vector<Var> vars;
    if (K % 128 == 0) vars.push_back({"x4_v8_l0", "nh_x4_v8_l0", 16, 256});  // (its K % 128 == 0)
    for (int k : {2, 4, 8}) {
      if (ks && k != ks) continue;
      vars.push_back({"x4t_v8_l0 ks" + std::to_string(k), "nh_x4t_v8_l0", 16, uint32_t(32 * k)});
    }
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
      for (size_t i = 0; i < size_t(8) * N; ++i) err = std::max(err, std::fabs(double(y.as<float>()[i]) - ref[i]));
      // row independence: row 0 with the other rows changed, rows 0-2 with R = 3
      memset(yb.ptr(), 0, yb.size);
      run(v, x2, yb, 8, 0, 1);
      bool indep = memcmp(y.ptr(), yb.ptr(), size_t(N) * 4) == 0;
      memset(yb.ptr(), 0, yb.size);
      run(v, x, yb, 3, 0, 1);
      indep = indep && memcmp(y.ptr(), yb.ptr(), size_t(3) * N * 4) == 0;
      bool tail_clean = true;
      for (size_t i = size_t(3) * N; i < size_t(8) * N; ++i) tail_clean = tail_clean && yb.as<float>()[i] == 0.0f;
      printf("  %-20s rel.err %.3e  rows independent %s%s%s\n", v.label.c_str(), err / rmax, indep ? "yes" : "NO",
             tail_clean ? "" : " (WROTE ROWS >= R)", gpr ? ("  " + gpr_str(v.kern)).c_str() : "");
    }
    if (small) continue;
    std::vector<std::vector<double>> ts(vars.size());
    for (int rd = 0; rd < reps + 2; ++rd)
      for (size_t i = 0; i < vars.size(); ++i) {
        const double ms = run(vars[i], x, y, 8, int(rnd() % copies), 10);
        if (rd >= 2) ts[i].push_back(ms / 10);
      }
    for (size_t i = 0; i < vars.size(); ++i) {
      std::sort(ts[i].begin(), ts[i].end());
      const double ms = ts[i][ts[i].size() / 2];
      printf("  %-20s %8.1f us  %6.1f GB/s (min %.1f us)\n", vars[i].label.c_str(), ms * 1000, double(qb + sb) / ms / 1e6,
             ts[i][0] * 1000);
    }
  }
  return 0;
}
