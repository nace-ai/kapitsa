// --gpu [--small]: original-vs-SIMD selection, tie/NaN/inf/zero cases and two-stage sampling.
// --bench: six interleaved original/new GPU-time pairs. Takes the GPU lock in both modes.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/log.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
using namespace ni;
namespace {
constexpr unsigned guard = 16;
constexpr uint32_t sentinel = 0x5a5a5a5a;
struct Guarded {
  Buffer storage, b;
  explicit Guarded(size_t n) {
    storage = Gpu::get().alloc((n + 2 * guard) * 4);
    std::fill_n(storage.as<uint32_t>(), n + 2 * guard, sentinel);
    b = storage.view(guard * 4, n * 4);
  }
  ~Guarded() { Gpu::get().release(storage); }
  bool intact() const {
    for (unsigned i = 0; i < guard; ++i)
      if (storage.as<uint32_t>()[i] != sentinel || storage.as<uint32_t>()[guard + b.size / 4 + i] != sentinel)
        return false;
    return true;
  }
};
float val(unsigned i, unsigned pattern) {
  if (pattern == 1) return float(i % 3);  // repeated ties
  if (pattern == 2) return i % 2 ? 0.0f : -0.0f;
  if (pattern == 3)
    return i % 37 == 0   ? INFINITY
           : i % 13 == 0 ? -INFINITY
           : i % 11 == 0 ? std::numeric_limits<float>::quiet_NaN()
                         : float(i % 17);
  if (pattern == 4) return -INFINITY;
  return float(int((i * 2246822519u ^ (i >> 4)) % 100003) - 50001) / 1000;
}
struct Sample {
  Gpu& g = Gpu::get();
  unsigned V, K, N;
  Buffer logits;
  Guarded cv[2], ci[2], tok[2], prob[2];
  Pipeline st1[2], st2[2];
  Sample(unsigned vocab, unsigned k)
      : V(vocab),
        K(k),
        N((V + 2047) / 2048 * K),
        cv{Guarded(N), Guarded(N)},
        ci{Guarded(N), Guarded(N)},
        tok{Guarded(1), Guarded(1)},
        prob{Guarded(1), Guarded(1)} {
    logits = g.alloc(V * 4);
    for (unsigned i = 0; i < V; ++i) logits.as<float>()[i] = val(i, 0);
    st1[0] = g.pipeline("topk_stage1");
    st1[1] = g.pipeline(K >= 8 ? "topk_stage1_rows_sg" : "topk_stage1");
    st2[0] = g.pipeline("sample_stage2");
    st2[1] = g.pipeline("sample_stage2");
  }
  ~Sample() { g.release(logits); }
  double run(unsigned v, unsigned phase, unsigned repeat = 1, float temp = .7f, float top_p = .9f,
             unsigned seed = 42) {
    auto e = g.begin();
    for (unsigned it = 0; it < repeat; ++it) {
      if (phase != 2) {
        e->use(st1[v]);
        e->buf(0, logits);
        e->buf(1, cv[v].b);
        e->buf(2, ci[v].b);
        e->val(3, kp::TopkStage{V, K, 2048});
        e->dispatch((V + 2047) / 2048, 1, 1, 256);
      }
      if (phase != 1) {
        e->use(st2[v]);
        e->buf(0, cv[v].b);
        e->buf(1, ci[v].b);
        e->buf(2, tok[v].b);
        e->buf(3, prob[v].b);
        e->val(4, kp::Sample{N, K, temp, top_p, seed, 17, 0});
        e->dispatch(1, 1, 1, 1024);
      }
    }
    e->commit_and_wait();
    return e->gpu_ms() * 1000 / repeat;
  }
  bool check(unsigned pattern) {
    for (unsigned i = 0; i < V; ++i) logits.as<float>()[i] = val(i, pattern);
    run(0, 1);
    run(1, 1);
    size_t diff = 0, cpu = 0, draw = 0, guards = 0;
    for (unsigned i = 0; i < N; ++i)
      diff += cv[0].b.as<uint32_t>()[i] != cv[1].b.as<uint32_t>()[i] ||
              ci[0].b.as<uint32_t>()[i] != ci[1].b.as<uint32_t>()[i];
    for (unsigned group = 0; group < (V + 2047) / 2048; ++group) {
      std::vector<unsigned> ids;
      for (unsigned i = group * 2048; i < std::min(V, (group + 1) * 2048); ++i)
        if (logits.as<float>()[i] > -INFINITY) ids.push_back(i);
      std::sort(ids.begin(), ids.end(), [&](unsigned a, unsigned b) {
        float x = logits.as<float>()[a], y = logits.as<float>()[b];
        return x > y || (x == y && a < b);
      });
      for (unsigned k = 0; k < K; ++k) {
        unsigned id = k < ids.size() ? ids[k] : 0xffffffffu;
        float value = k < ids.size() ? logits.as<float>()[id] : -INFINITY;
        unsigned i = group * K + k;
        cpu += ci[1].b.as<uint32_t>()[i] != id || cv[1].b.as<uint32_t>()[i] != std::bit_cast<uint32_t>(value);
      }
    }
    if (N <= 4096)
      for (float temp : {0.f, .7f, 1.2f})
        for (float top_p : {.1f, .9f, 1.f})
          for (unsigned seed : {0u, 1u, 42u, 999u}) {
            run(0, 2, 1, temp, top_p, seed);
            run(1, 2, 1, temp, top_p, seed);
            draw += tok[0].b.as<uint32_t>()[0] != tok[1].b.as<uint32_t>()[0] ||
                    prob[0].b.as<uint32_t>()[0] != prob[1].b.as<uint32_t>()[0];
          }
    for (unsigned v = 0; v < 2; ++v)
      guards += !cv[v].intact() || !ci[v].intact() || !tok[v].intact() || !prob[v].intact();
    printf("V=%u K=%u pattern=%u bitwise=%zu cpu=%zu draws=%zu guards=%zu\n", V, K, pattern, diff, cpu, draw,
           guards);
    return !diff && !cpu && !draw && !guards;
  }
};

}  // namespace
int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (strcmp(argv[1], "--gpu") && strcmp(argv[1], "--bench")) ||
      (argc == 3 && (strcmp(argv[1], "--gpu") || strcmp(argv[2], "--small"))))
    fatal("usage: sample_reduce_test --gpu [--small] | --bench");
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("cannot acquire GPU lock");
  auto& g = Gpu::get();
  g.compile();
  bool bench = !strcmp(argv[1], "--bench"), small = argc == 3;
  printf("GPU %s thermal=%s\n", g.name().c_str(), Gpu::thermal_state());
  unsigned failures = 0, cases = 0;

  for (unsigned V : {1u, 31u, 257u, 2047u, 2048u, 2049u, 32000u, 248320u}) {
    if (small && V > 257) continue;
    if (bench && V < 32000) continue;
    for (unsigned K : {1u, 2u, 4u, 7u, 8u, 16u, 31u, 32u, 64u}) {
      Sample c(V, K);
      if (bench) {
        for (unsigned phase : {1u, 2u}) {
          if (phase == 2 && c.N > 4096) continue;
          c.run(0, 1);
          c.run(1, 1);
          for (unsigned i = 0; i < 4; ++i) {
            c.run(0, phase, 128);
            c.run(1, phase, 128);
          }
          for (unsigned pair = 1; pair <= 6; ++pair)
            for (unsigned v = 0; v < 2; ++v)
              printf(
                  "{\"V\":%u,\"K\":%u,\"phase\":%u,\"pair\":%u,\"variant\":\"%c\",\"us\":%.6f,\"thermal\":\"%"
                  "s\"}\n",
                  V, K, phase, pair, v ? 'B' : 'A', c.run(v, phase, 128), Gpu::thermal_state());
        }
      } else
        for (unsigned p = 0; p < 5; ++p) {
          failures += !c.check(p);
          ++cases;
        }
      fflush(stdout);
    }
  }
  printf("%u cases, %u failures\n", cases, failures);
  return failures != 0;
}
