// --gpu [--small]: split-attention reduction against the original kernel bitwise and an fp64 reference.
// --bench: interleaved original/shared-statistics timings, including the 27B's 24-head layout.
// Both modes take the GPU lock. Run --gpu --small with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 first.
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
float value(unsigned row, unsigned c, unsigned d) {
  unsigned x = row * 2654435761u ^ c * 2246822519u ^ d * 3266489917u ^ 131u;
  x ^= x >> 16; x *= 2246822519u; x ^= x >> 13;
  return float(int(x & 65535u) - 32768) / 32768.0f;
}

struct Reduction {
  static constexpr unsigned D = 256, guard = 16;
  static constexpr float sentinel = 12345.0f;
  Gpu& g = Gpu::get();
  unsigned Hq, R, S;
  Buffer po, ml, qg, storage[2], out[2];
  Pipeline pipelines[2];

  Reduction(unsigned heads, unsigned rows, unsigned splits, unsigned pattern)
      : Hq(heads), R(rows), S(splits) {
    if (!S || R * S > 1024) fatal("reduction test exceeds the production scratch limit");
    const size_t nr = size_t(R) * Hq, n = nr * D;
    po = g.alloc(n * S * sizeof(float)); ml = g.alloc(nr * S * 2 * sizeof(float));
    qg = g.alloc(n * 2 * sizeof(float));
    for (int i = 0; i < 2; ++i) {
      storage[i] = g.alloc((n + 2 * guard) * sizeof(float));
      std::fill_n(storage[i].as<float>(), n + 2 * guard, sentinel);
      out[i] = storage[i].view(guard * sizeof(float), n * sizeof(float));
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::fill_n(qg.as<float>(), n * 2, nan);
    for (unsigned row = 0; row < nr; ++row) {
      for (unsigned d = 0; d < D; ++d) qg.as<float>()[row * 2 * D + D + d] = 8 * value(row, d, 17);
      for (unsigned c = 0; c < S; ++c) {
        const size_t slot = size_t(row) * S + c;
        const bool empty = pattern == 4 || (pattern == 1 && c % 3 == 0 && c != S - 1) ||
                           (pattern == 3 && c != S - 1);
        float m = 20 * value(row, c, 7) + float(row);
        if (pattern == 2) m = c % 2 ? 10000.0f : -10000.0f;  // ties and exponential underflow
        const float l = 1 + 32 * (1 + value(row, c, 11));
        ml.as<float>()[slot * 2] = empty ? -INFINITY : m;
        ml.as<float>()[slot * 2 + 1] = empty ? nan : l;
        for (unsigned d = 0; d < D; ++d)
          po.as<float>()[slot * D + d] = empty ? nan : l * value(row, c, d);
      }
    }
    pipelines[0] = g.pipeline("attn_rows_reduce");
    pipelines[1] = g.pipeline("attn_rows_reduce_shared");
  }
  ~Reduction() {
    for (Buffer* b : {&po, &ml, &qg, &storage[0], &storage[1]}) g.release(*b);
  }
  double run(int variant, int repeats = 1) {
    const kp::AttnRows p{Hq, 1, 0, 64, 64, 1.0f / 16, S};
    auto e = g.begin();
    for (int i = 0; i < repeats; ++i) {
      e->use(pipelines[variant]); e->buf(0, po); e->buf(1, ml); e->buf(2, qg); e->buf(3, out[variant]);
      e->val(4, p); e->dispatch(Hq, R, 1, D);
    }
    e->commit_and_wait();
    return e->gpu_ms() * 1000 / repeats;
  }
  bool check() {
    run(0); run(1);
    size_t different = 0, invalid = 0, guards = 0, empty_rows = 0;
    double max_error = 0;
    const size_t n = size_t(R) * Hq * D;
    for (int v = 0; v < 2; ++v)
      for (unsigned i = 0; i < guard; ++i) {
        guards += storage[v].as<float>()[i] != sentinel;
        guards += storage[v].as<float>()[guard + n + i] != sentinel;
      }
    for (unsigned row = 0; row < R * Hq; ++row) {
      double mx = -INFINITY, den = 0;
      std::vector<double> num(D, 0);
      for (unsigned c = 0; c < S; ++c) mx = std::max(mx, double(ml.as<float>()[(row * S + c) * 2]));
      if (mx == -INFINITY) ++empty_rows;
      for (unsigned c = 0; c < S; ++c) {
        const size_t slot = size_t(row) * S + c;
        const float m = ml.as<float>()[slot * 2];
        if (m == -INFINITY) continue;
        const double w = std::exp(double(m) - mx);
        den += w * ml.as<float>()[slot * 2 + 1];
        for (unsigned d = 0; d < D; ++d) num[d] += w * po.as<float>()[slot * D + d];
      }
      for (unsigned d = 0; d < D; ++d) {
        const size_t i = size_t(row) * D + d;
        const float a = out[0].as<float>()[i], b = out[1].as<float>()[i];
        if (mx == -INFINITY) {
          invalid += !std::isnan(a) || !std::isnan(b);
          continue;
        }
        different += std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b);
        invalid += !std::isfinite(a) || !std::isfinite(b);
        const double gate = 1 / (1 + std::exp(-double(qg.as<float>()[row * 2 * D + D + d])));
        const double ref = num[d] / den * gate;
        max_error = std::max(max_error, std::fabs(double(b) - ref) / (1 + std::fabs(ref)));
      }
    }
    printf("Hq=%u R=%u splits=%u: bitwise_diff=%zu nonfinite_mismatch=%zu guard_errors=%zu empty_rows=%zu "
           "fp64_scaled_error=%.3g\n", Hq, R, S, different, invalid, guards, empty_rows, max_error);
    return !different && !invalid && !guards && max_error < 5e-6;
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (strcmp(argv[1], "--gpu") && strcmp(argv[1], "--bench")) ||
      (argc == 3 && (strcmp(argv[1], "--gpu") || strcmp(argv[2], "--small"))))
    fatal("usage: attn_reduce_test --gpu [--small] | --bench");
  ni::bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD") && !lock.acquire()) fatal("cannot acquire GPU lock");
  Gpu& g = Gpu::get();
  g.compile();
  printf("GPU %s, shared reduction static threadgroup bytes %u\n", g.name().c_str(), g.pipeline("attn_rows_reduce_shared").tg_mem);
  if (!strcmp(argv[1], "--bench")) {
    { Reduction warmup(24, 8, 128, 0); warmup.run(0, 512); warmup.run(1, 512); }
    for (unsigned rows : {1u, 8u, 32u})
      for (unsigned splits : {1u, 2u, 3u, 4u, 7u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u}) {
        if (rows * splits > 1024) continue;
        Reduction test(24, rows, splits, 0);
        for (int i = 0; i < 4; ++i) { test.run(0, 128); test.run(1, 128); }
        for (int pair = 0; pair < 6; ++pair)
          for (int v = 0; v < 2; ++v) {
            const double us = test.run(v, 128);
            printf("{\"Hq\":24,\"R\":%u,\"splits\":%u,\"pair\":%d,\"variant\":\"%c\",\"us\":%.6f,"
                   "\"thermal\":\"%s\"}\n", rows, splits, pair + 1, v ? 'B' : 'A', us, Gpu::thermal_state());
            fflush(stdout);
          }
      }
    return 0;
  }
  const bool small = argc == 3;
  int failed = 0, cases = 0;
  for (unsigned rows : {1u, 3u, 8u, 32u})
    for (unsigned splits : {1u, 2u, 3u, 4u, 7u, 31u, 32u, 33u, 64u, 127u, 128u, 129u, 255u, 256u, 257u, 511u, 512u, 1024u}) {
      if (rows * splits > 1024 || (small && (rows > 3 || splits > 7))) continue;
      for (unsigned pattern = 0; pattern < 5; ++pattern) {
        Reduction test(rows == 8 ? 24 : 3, rows, splits, pattern);
        failed += !test.check(); ++cases;
      }
    }
  printf("%d cases, %d failures\n", cases, failed);
  return failed != 0;
}
