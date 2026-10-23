// Prefill attention microbenchmark: T queries (positions L-T..L-1) against L cached keys, D=256,
// 16 query heads / 2 kv heads (Qwen3.5/3.6 shapes; NI_HQ=24 NI_HKV=4 for the 27B's G=6).
// usage: bench_attn [T] [L] kernels...
// Short-prefill forms (NI_SHORTPF): attn_prefill11g (rows = (query, head) pairs of one KV head),
// attn_prefill11gs:NS (the same with the keys split over NS threadgroups + attn11_reduce;
// NS = auto: the engine's choice).
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
namespace ni { extern const char* const kTensorKernelSource; }
using ni::kTensorKernelSource;
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

  const uint32_t max_rows = 1024;  // nsplit * T rows per head (as the engine's part_o_)
  Buffer part_o = g.alloc(size_t(Hq) * max_rows * D * 4), part_ml = g.alloc(size_t(Hq) * max_rows * 2 * 4);
  for (int k = 3; k < argc; ++k) {
    std::string name = argv[k];
    uint32_t ns = 0;
    const bool gq = name.rfind("attn_prefill11g", 0) == 0;
    if (name.find(':') != std::string::npos) {
      std::string a = name.substr(name.find(':') + 1);
      name = name.substr(0, name.find(':'));
      const uint32_t nqb = gq ? (T * G + 31) / 32 : (T + 31) / 32, gy = gq ? Hkv : Hq, tiles = (L + 31) / 32;
      if (a == "auto") {
        ns = (100 + nqb * gy - 1) / (nqb * gy);
        ns = std::min({ns, tiles / 8, max_rows / T, 16u});
      } else {
        ns = atoi(a.c_str());
      }
      if (ns * T > max_rows) fatal("nsplit %u x T %u > %u", ns, T, max_rows);
    }
    if (name.rfind("qattn_t", 0) == 0) g.compile_extra("tensor", kTensorKernelSource, true);
    Pipeline pl = g.pipeline(name);
    static Pipeline red = g.pipeline("attn11_reduce");
    uint32_t qb = (name == "attn_prefill" || name == "attn_prefill5" || name == "attn_prefill6" || name.rfind("attn_prefill7", 0) == 0) ? 4 : 8;
    uint32_t th = name == "attn_prefill" ? 128 : 256, gy = Hkv;
    if (name.rfind("attn_prefill8_q", 0) == 0 || name.rfind("attn_prefill9_q", 0) == 0) { qb = atoi(name.c_str() + 15); th = qb * 64; }
    if (name.rfind("attn_prefill10_q", 0) == 0) { qb = atoi(name.c_str() + 16); th = qb * 64; }
    if (name.rfind("attn_prefill10u_q", 0) == 0) { qb = atoi(name.c_str() + 17); th = qb * 64; }
    if (name.rfind("attn_prefill11", 0) == 0) { qb = 32; th = 256; gy = Hq; }  // 32 queries of one q head  // 32 queries of one q head
    if (name.rfind("qattn_t", 0) == 0) {
      const uint32_t nsg = name.find("s16") != std::string::npos ? 16 : name.find("s2k") != std::string::npos ? 2 : name.find("s8") != std::string::npos ? 8 : 4;
      const bool g8 = name.find("g8") != std::string::npos;   // 2 tokens x 8 heads per tile instead of 4 x 4
      qb = (g8 ? 2 : 4) * nsg; th = 32 * nsg; gy = Hkv;
      if (name.find("d2") != std::string::npos) qb /= 2;   // a pair of simdgroups per 16-row tile
    }
    uint32_t nblk = gq ? (T * G + 31) / 32 : (T + qb - 1) / qb;
    if (name.rfind("qattn_t", 0) == 0 && name.find("d2s") != std::string::npos) {   // 8 NS (token, head) rows per group
      const uint32_t nsg = atoi(name.c_str() + name.find("d2s") + 3);
      th = 32 * nsg;
      nblk = (T * G + 8 * nsg - 1) / (8 * nsg);
    }
    if (gq) gy = Hkv;
    auto run = [&](int n = 1) {  // n back-to-back calls in one command buffer, ms per call
      auto e = g.begin();
      for (int r = 0; r < n; ++r) {
        e->use(pl);
        e->buf(0, q); e->buf(1, kc); e->buf(2, vc); e->buf(3, qg); e->buf(4, out);
        e->val(5, p);
        if (ns) {
          e->buf(6, part_o); e->buf(7, part_ml);
          e->val(8, ns);
          e->dispatch(nblk, gy, ns, th);
          e->use(red);
          e->buf(0, part_o); e->buf(1, part_ml); e->buf(2, qg); e->buf(3, out);
          e->val(4, p);
          e->val(5, ns);
          e->dispatch(T, Hq, 1, D);
        } else {
          e->dispatch(nblk, gy, 1, th);
        }
      }
      e->commit_and_wait();
      return e->gpu_ms() / n;
    };
    if (ns) name += ":" + std::to_string(ns);
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
    // vs the first kernel named: each output's distance from that kernel's, in half-precision steps at the reference's
    // magnitude (one step: the spacing of halfs around |ref|), max and the count past one step
    static std::vector<float> first_out;
    double vs_max = 0;
    size_t vs_over = 0;
    const size_t nout = size_t(T) * Hq * D;
    if (first_out.empty()) {
      first_out.assign(out.as<float>(), out.as<float>() + nout);
    } else {
      for (size_t i = 0; i < nout; ++i) {
        const double a = out.as<float>()[i], b = first_out[i], m = std::max(std::fabs(b), 6.1e-5);
        const double step = std::ldexp(1.0, std::ilogb(m) - 10);
        const double d = std::fabs(a - b) / step;
        vs_max = std::max(vs_max, d);
        vs_over += d > 1.0;
      }
    }
    // median of 9 trials of 8 calls each (BENCH_ATTN_REPS overrides the 8)
    static const int nrep = getenv("BENCH_ATTN_REPS") ? atoi(getenv("BENCH_ATTN_REPS")) : 8;
    std::vector<double> tv;
    for (int i = 0; i < 9; ++i) tv.push_back(run(nrep));
    std::sort(tv.begin(), tv.end());
    const double best = tv[tv.size() / 2];
    double cs = 0;
    for (int i = 0; i < 1000; ++i) cs += out.as<float>()[i * 37 % (T * Hq * D)];
    uint64_t hash = 1469598103934665603ull;  // FNV-1a over every output bit: equal hashes = bitwise-equal outputs
    const uint8_t* ob = (const uint8_t*)out.ptr();
    for (size_t i = 0; i < size_t(T) * Hq * D * 4; ++i) hash = (hash ^ ob[i]) * 1099511628211ull;
    printf("%-24s T=%u L=%u Hq=%u Hkv=%u  %8.3f ms  %5.2f TFLOP/s  err=%.2e  checksum=%.4f  bits=%016llx  vs first: max %.2f"
           " half steps, %zu of %zu past one\n", name.c_str(), T, L, Hq, Hkv, best, pairs * Hq * 1024 / (best * 1e-3) / 1e12,
           err / ref_max, cs, (unsigned long long)hash, vs_max, vs_over, nout);
  }
}
