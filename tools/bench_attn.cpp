// Prefill attention microbenchmark: T queries (positions L-T..L-1) against L cached keys, D=256,
// 16 query heads / 2 kv heads (Qwen3.5/3.6 shapes). usage: bench_attn [T] [L] kernels...
#include "metal/gpu.h"
#include "util/log.h"
#include <cmath>
#include <random>
#include <string>
using namespace ni;
struct AttnPrefParams {
  uint32_t T, Hq, Hkv, pos0, max_seq;
  float scale;
};
int main(int argc, char** argv) {
  uint32_t T = argc > 1 ? atoi(argv[1]) : 2048, L = argc > 2 ? atoi(argv[2]) : 16384;
  Gpu& g = Gpu::get();
  const uint32_t Hq = 16, Hkv = 2, D = 256;
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
  for (int k = 3; k < argc; ++k) {
    std::string name = argv[k];
    Pipeline pl = g.pipeline(name);
    uint32_t qb = (name == "attn_prefill" || name == "attn_prefill5" || name == "attn_prefill6" || name.rfind("attn_prefill7", 0) == 0) ? 4 : 8;
    uint32_t th = name == "attn_prefill" ? 128 : 256;
    if (name.rfind("attn_prefill8_q", 0) == 0 || name.rfind("attn_prefill9_q", 0) == 0) { qb = atoi(name.c_str() + 15); th = qb * 64; }
    if (name.rfind("attn_prefill10_q", 0) == 0) { qb = atoi(name.c_str() + 16); th = qb * 64; }
    auto run = [&]() {
      auto e = g.begin();
      e->use(pl);
      e->buf(0, q); e->buf(1, kc); e->buf(2, vc); e->buf(3, qg); e->buf(4, out);
      e->val(5, p);
      e->dispatch((T + qb - 1) / qb, Hkv, 1, th);
      e->commit_and_wait();
      return e->gpu_ms();
    };
    run();
    double best = 1e9;
    for (int i = 0; i < 3; ++i) best = std::min(best, run());
    double cs = 0;
    for (int i = 0; i < 1000; ++i) cs += out.as<float>()[i * 37 % (T * Hq * D)];
    printf("%-16s T=%u L=%u  %8.1f ms  %5.2f TFLOP/s  checksum=%.4f\n", name.c_str(), T, L, best,
           pairs * Hq * 1024 / (best * 1e-3) / 1e12, cs);
  }
}
