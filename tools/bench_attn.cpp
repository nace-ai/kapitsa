// Prefill attention microbenchmark: T queries (positions L-T..L-1) against L cached keys, D=256,
// 16 query heads / 2 kv heads (Qwen3.5/3.6 shapes; NI_HQ=24 NI_HKV=4 for the 27B's G=6).
// usage: bench_attn [T] [L] kernels...
// Each kernel's output is checked against a CPU reference (double precision) on a few sampled
// queries (block edges, the last partial block) x all heads: err = max |gpu - ref| / max |ref|.
#include "metal/gpu.h"
#include "util/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace ni;
struct AttnPrefParams {
  uint32_t T, Hq, Hkv, pos0, max_seq;
  float scale;
};
int main(int argc, char** argv) {
  uint32_t T = argc > 1 ? atoi(argv[1]) : 2048, L = argc > 2 ? atoi(argv[2]) : 16384;
  Gpu& g = Gpu::get();
  const uint32_t Hq = getenv("NI_HQ") ? atoi(getenv("NI_HQ")) : 16, Hkv = getenv("NI_HKV") ? atoi(getenv("NI_HKV")) : 2;
  const uint32_t D = 256, G = Hq / Hkv;
  Buffer q = g.alloc(size_t(T) * Hq * D * 4), qg = g.alloc(size_t(T) * Hq * 2 * D * 4), out = g.alloc(size_t(T) * Hq * D * 4);
  Buffer kc = g.alloc(size_t(Hkv) * L * D * 2), vc = g.alloc(size_t(Hkv) * L * D * 2);
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 1);
  for (size_t i = 0; i < size_t(T) * Hq * D; ++i) q.as<float>()[i] = nd(rng);
  for (size_t i = 0; i < size_t(T) * Hq * 2 * D; ++i) qg.as<float>()[i] = nd(rng);
  for (size_t i = 0; i < size_t(Hkv) * L * D; ++i) {
    __fp16 a = nd(rng), b = nd(rng);
    kc.as<__fp16>()[i] = a;
    vc.as<__fp16>()[i] = b;
  }
  AttnPrefParams p{T, Hq, Hkv, L - T, L, 1.0f / 16.0f};
  double pairs = double(T) * L - double(T) * T / 2;

  // CPU reference for sampled queries x all heads: block edges, the last (partial) block, and
  // queries in every 8-row group of a 32-query block (11, 20, 45, T-19, T-11) for v11's layout
  std::vector<uint32_t> ts;
  for (int64_t t : {int64_t(0), int64_t(1), int64_t(11), int64_t(20), int64_t(31), int64_t(32), int64_t(45),
                    int64_t(T / 2), int64_t(T) - 33, int64_t(T) - 19, int64_t(T) - 11, int64_t(T) - 2, int64_t(T) - 1})
    if (t >= 0 && t < T) ts.push_back(uint32_t(t));
  std::sort(ts.begin(), ts.end());
  ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
  std::vector<double> ref(ts.size() * Hq * D);
  double ref_max = 0;
  {
    const __fp16* K = kc.as<__fp16>();
    const __fp16* V = vc.as<__fp16>();
    std::vector<double> sc(L), o(D);
    for (size_t si = 0; si < ts.size(); ++si)
      for (uint32_t h = 0; h < Hq; ++h) {
        const uint32_t t = ts[si], kvh = h / G, n = p.pos0 + t + 1;
        const float* qr = q.as<float>() + (size_t(t) * Hq + h) * D;
        double mx = -1e300;
        for (uint32_t j = 0; j < n; ++j) {
          const __fp16* kr = K + (size_t(kvh) * L + j) * D;
          double s = 0;
          for (uint32_t d = 0; d < D; ++d) s += double(qr[d]) * double(kr[d]);
          sc[j] = s * p.scale;
          mx = std::max(mx, sc[j]);
        }
        double sum = 0;
        std::fill(o.begin(), o.end(), 0.0);
        for (uint32_t j = 0; j < n; ++j) {
          const double e = std::exp(sc[j] - mx);
          sum += e;
          const __fp16* vr = V + (size_t(kvh) * L + j) * D;
          for (uint32_t d = 0; d < D; ++d) o[d] += e * double(vr[d]);
        }
        for (uint32_t d = 0; d < D; ++d) {
          const double gt = qg.as<float>()[(size_t(t) * Hq + h) * 2 * D + D + d];
          const double r = o[d] / sum / (1.0 + std::exp(-gt));
          ref[(si * Hq + h) * D + d] = r;
          ref_max = std::max(ref_max, std::fabs(r));
        }
      }
  }

  for (int k = 3; k < argc; ++k) {
    std::string name = argv[k];
    Pipeline pl = g.pipeline(name);
    uint32_t qb = (name == "attn_prefill" || name == "attn_prefill5" || name == "attn_prefill6" || name.rfind("attn_prefill7", 0) == 0) ? 4 : 8;
    uint32_t th = name == "attn_prefill" ? 128 : 256, gy = Hkv;
    if (name.rfind("attn_prefill8_q", 0) == 0 || name.rfind("attn_prefill9_q", 0) == 0) { qb = atoi(name.c_str() + 15); th = qb * 64; }
    if (name.rfind("attn_prefill10_q", 0) == 0) { qb = atoi(name.c_str() + 16); th = qb * 64; }
    if (name.rfind("attn_prefill10u_q", 0) == 0) { qb = atoi(name.c_str() + 17); th = qb * 64; }
    if (name.rfind("attn_prefill11", 0) == 0) { qb = 32; th = 256; gy = Hq; }  // 32 queries of one q head
    auto run = [&]() {
      auto e = g.begin();
      e->use(pl);
      e->buf(0, q); e->buf(1, kc); e->buf(2, vc); e->buf(3, qg); e->buf(4, out);
      e->val(5, p);
      e->dispatch((T + qb - 1) / qb, gy, 1, th);
      e->commit_and_wait();
      return e->gpu_ms();
    };
    memset(out.ptr(), 0xff, out.size);  // NaN: rows a kernel fails to write show up as errors
    run();
    double err = 0;
    bool nan = false;  // sticky: one NaN (unwritten or broken output) makes err NaN
    for (size_t si = 0; si < ts.size(); ++si)
      for (uint32_t h = 0; h < Hq; ++h)
        for (uint32_t d = 0; d < D; ++d) {
          const double gv = out.as<float>()[(size_t(ts[si]) * Hq + h) * D + d];
          const double e = std::fabs(gv - ref[(si * Hq + h) * D + d]);
          if (std::isnan(e)) nan = true;
          else err = std::max(err, e);
        }
    if (nan) err = NAN;
    double best = 1e9;
    for (int i = 0; i < 3; ++i) best = std::min(best, run());
    double cs = 0;
    for (int i = 0; i < 1000; ++i) cs += out.as<float>()[i * 37 % (T * Hq * D)];
    printf("%-24s T=%u L=%u Hq=%u Hkv=%u  %8.1f ms  %5.2f TFLOP/s  err=%.2e  checksum=%.4f\n", name.c_str(), T, L, Hq,
           Hkv, best, pairs * Hq * 1024 / (best * 1e-3) / 1e12, err / ref_max, cs);
  }
}
