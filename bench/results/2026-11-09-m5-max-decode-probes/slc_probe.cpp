// Scratch probe: does the M5 Max's system-level cache (SLC) serve a re-read faster than DRAM, and at which sizes?
//   zz_slc A  : warm (one buffer re-read) vs cold (rotating buffers far beyond the SLC) streaming reads
//   zz_slc B  : a dependent chain of small dispatches, then a consumer read of S MB; with / without a prefetch of
//               those bytes paired (same group) with the chain's steps
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace ni;

static const char* kSrc = R"MSL(
[[kernel]] void zz_rd(const device uint4* p [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  uint4 a = 0, b = 0, c = 0, d = 0;
  uint i = gid;
  for (; i + 3 * gsz < n; i += 4 * gsz) { a ^= p[i]; b ^= p[i + gsz]; c ^= p[i + 2 * gsz]; d ^= p[i + 3 * gsz]; }
  for (; i < n; i += gsz) a ^= p[i];
  a ^= b ^ c ^ d;
  if (all(a == uint4(0x9e3779b9u))) out[0] = 1;
}
// a latency-bound step: 32 threadgroups, each a short dependent loop over a few values, writes its slot
[[kernel]] void zz_step(device float* s [[buffer(0)]], constant uint& iters [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
  float v = s[tg * 256 + lid];
  for (uint k = 0; k < iters; ++k) v = fma(v, 0.999f, 0.001f);
  s[tg * 256 + lid] = v;
}
)MSL";

static double med(std::vector<double> t) {
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "A";
  Gpu& g = Gpu::get();
  g.compile(kSrc);
  Buffer out = g.alloc(64);
  Pipeline rd = g.pipeline("zz_rd"), st = g.pipeline("zz_step");
  const uint32_t tgs = uint32_t(g.core_count()) * 8;
  auto read = [&](Encoder& e, const Buffer& b) {
    e.use(rd);
    e.buf(0, b);
    e.buf(1, out);
    const uint32_t n = uint32_t(b.size / 16);
    e.val(2, n);
    e.dispatch(tgs, 1, 1, 256);
  };
  // 1.5 GB of eviction buffers
  std::vector<Buffer> ev;
  for (int i = 0; i < 12; ++i) {
    ev.push_back(g.alloc(size_t(128) << 20, nullptr, false));
    memset(ev.back().ptr(), i + 1, ev.back().size);
  }
  if (mode == "A") {
    printf("cores %d\n", g.core_count());
    for (double mb : {1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0}) {
      const size_t bytes = size_t(mb * (1 << 20)) / 4096 * 4096;
      const int copies = std::clamp(int(1.6e9 / double(bytes)), 4, 400);
      std::vector<Buffer> bufs;
      for (int c = 0; c < copies; ++c) {
        bufs.push_back(g.alloc(bytes, nullptr, false));
        memset(bufs.back().ptr(), c + 1, bytes);
      }
      std::vector<double> cold, warm;
      for (int rep = 0; rep < 5; ++rep) {
        {
          auto e = g.begin();
          for (auto& b : bufs) read(*e, b);
          e->commit_and_wait();
          cold.push_back(e->gpu_ms() / copies);
        }
        {
          auto e = g.begin();
          for (int c = 0; c < copies; ++c) read(*e, bufs[0]);
          e->commit_and_wait();
          warm.push_back(e->gpu_ms() / copies);
        }
      }
      const double c = med(cold), w = med(warm);
      printf("%6.1f MB  cold %8.2f us %6.1f GB/s | warm %8.2f us %6.1f GB/s | x%.2f\n", mb, c * 1e3,
             bytes / (c * 1e-3) / 1e9, w * 1e3, bytes / (w * 1e-3) / 1e9, c / w);
      for (auto& b : bufs) g.release(b);
    }
    return 0;
  }
  if (mode == "C") {  // prefetch A (S MB), stream X MB of other data, then read A: is A still warm?
    for (double amb : {4.0, 8.0, 14.0, 18.0}) {
      for (double xmb : {0.0, 24.0, 48.0, 72.0, 110.0}) {
        const size_t ab = size_t(amb * (1 << 20)) / 4096 * 4096, xb = size_t(std::max(xmb, 1.0) * (1 << 20)) / 4096 * 4096;
        const int layers = 16;
        std::vector<Buffer> A, X;
        for (int l = 0; l < layers; ++l) { A.push_back(g.alloc(ab, nullptr, false)); memset(A.back().ptr(), l + 1, ab);
                                           X.push_back(g.alloc(xb, nullptr, false)); memset(X.back().ptr(), l + 7, xb); }
        auto run = [&](bool pf) {
          auto e = g.begin();
          for (int l = 0; l < layers; ++l) {
            if (pf) read(*e, A[l]);
            if (xmb > 0) read(*e, X[l]);
            read(*e, A[l]);
          }
          e->commit_and_wait();
          return e->gpu_ms() / layers;
        };
        auto runx = [&]() {  // the X stream alone (and the A read alone)
          auto e = g.begin();
          for (int l = 0; l < layers; ++l) { if (xmb > 0) read(*e, X[l]); }
          e->commit_and_wait();
          return e->gpu_ms() / layers;
        };
        std::vector<double> t0, t1, tx;
        for (int r = 0; r < 5; ++r) { t0.push_back(run(false)); t1.push_back(run(true)); tx.push_back(runx()); }
        // with pf: A + X + A_warm; without: X + A_cold -> A_warm = t1 - tx - A_cold, A_cold = t0 - tx
        const double ac = med(t0) - med(tx), aw = med(t1) - med(tx) - ac;
        printf("A %4.0f MB after %5.0f MB stream: A cold %7.1f us, A again after prefetch %7.1f us (x%.2f)\n", amb, xmb, ac * 1e3, aw * 1e3, ac / aw);
        for (auto& b : A) g.release(b);
        for (auto& b : X) g.release(b);
      }
    }
    return 0;
  }
  // B: chain of `steps` dependent steps, then a read of S MB; the prefetch: S MB split over the steps, each piece in
  // the step's group
  const int steps = argc > 2 ? atoi(argv[2]) : 8;
  const uint32_t iters = argc > 3 ? uint32_t(atoi(argv[3])) : 2000;
  Buffer sbuf = g.alloc(32 * 256 * 4);
  memset(sbuf.ptr(), 0, sbuf.size);
  for (double mb : {2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0}) {
    const size_t bytes = size_t(mb * (1 << 20)) / (4096 * size_t(steps)) * 4096 * size_t(steps);
    const int layers = 24;  // distinct weight buffers, one per "layer", so every consumer read is cold unless prefetched
    std::vector<Buffer> w;
    for (int l = 0; l < layers; ++l) {
      w.push_back(g.alloc(bytes, nullptr, false));
      memset(w.back().ptr(), l + 3, bytes);
    }
    auto run = [&](int how) {  // 0 chain + read; 1 chain + paired prefetch + read; 2 chain only; 3 read only
      auto e = g.begin_grouped();
      for (int l = 0; l < layers; ++l) {
        if (how != 3)
          for (int s = 0; s < steps; ++s) {
            e->group_begin();
            e->use(st);
            e->buf(0, sbuf);
            e->val(1, iters);
            e->dispatch(32, 1, 1, 256);
            if (how == 1) {
              const size_t piece = bytes / size_t(steps);
              e->use(rd);
              e->buf(0, w[l].view(piece * size_t(s), piece));
              e->buf(1, out);
              const uint32_t n = uint32_t(piece / 16);
              e->val(2, n);
              e->dispatch(tgs, 1, 1, 256);
            }
            e->group_end();
          }
        if (how != 2) read(*e, w[l]);
      }
      e->commit_and_wait();
      return e->gpu_ms() / layers;
    };
    std::vector<double> t[4];
    for (int rep = 0; rep < 5; ++rep)
      for (int how = 0; how < 4; ++how) t[how].push_back(run(how));
    printf("%5.1f MB, %d steps x %u iters: chain %7.1f us, read %7.1f us, chain+read %7.1f us, chain+prefetch+read %7.1f us"
           " (saves %5.1f us of the read's %5.1f)\n",
           mb, steps, iters, med(t[2]) * 1e3, med(t[3]) * 1e3, med(t[0]) * 1e3, med(t[1]) * 1e3,
           (med(t[0]) - med(t[1])) * 1e3, med(t[3]) * 1e3);
    for (auto& b : w) g.release(b);
  }
  return 0;
}
