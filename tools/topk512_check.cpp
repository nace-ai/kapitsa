// Compare the production 512-expert router with the ordered reference and an fp64 oracle.
// --gpu [--small] checks outputs and guards; --bench reports interleaved kernel timings.
// Both modes hold the GPU lock. Run --gpu --small with both Metal validation layers first.
#include "bench/bench.h"
#include "metal/gpu.h"
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
// moe_topk512 from f22effb9, before reusing the first selection's maximum.
constexpr const char* reference_source = R"METAL(#include <metal_stdlib>
using namespace metal;
struct TopkParams { uint E, k, norm; };
inline float sigmoid(float x) { return 1.0f / (1.0f + metal::fast::exp(-x)); }
[[kernel]] void moe_topk512_reference(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                            device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                            constant TopkParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const device float4* l4 = (const device float4*)(logits + r * 512 + lane * 16);
  const float4 a = l4[0], b = l4[1], c = l4[2], d = l4[3];
  float v[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
  float mx = v[0];
  for (uint i = 1; i < 16; ++i) mx = max(mx, v[i]);
  mx = simd_max(mx);
  float total = 0.0f;  // the full softmax denominator (no renormalization only)
  if (!p.norm) {
    for (uint i = 0; i < 16; ++i) total += metal::precise::exp(v[i] - mx);
    total = simd_sum(total);
  }
  uint taken = 0;
  float sel = 0.0f, my_w = 0.0f;
  uint my_i = 0;
  for (uint q = 0; q < p.k; ++q) {
    float best = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = 0; i < 16; ++i)
      if (!(taken & (1u << i)) && v[i] > best) {
        best = v[i];
        bi = lane * 16 + i;
      }
    const float g = simd_max(best);
    const uint win = simd_min(best == g ? bi : 0xffffffffu);
    if (win / 16 == lane) taken |= 1u << (win % 16);
    const float e = metal::precise::exp(g - mx);
    sel += e;
    if (lane == q) my_w = e, my_i = win;
  }
  if (lane < p.k) {
    idx[r * p.k + lane] = my_i;
    wts[r * (p.k + 1) + lane] = my_w / (p.norm ? sel : total);
  }
  if (lane == p.k) wts[r * (p.k + 1) + p.k] = sigmoid(shared[r]);
}
)METAL";
struct Router {
  static constexpr unsigned E = 512, guard = 16;
  static constexpr uint32_t sentinel = 0x4abc1234;
  Gpu& g = Gpu::get();
  unsigned R, K, norm, pattern;
  Buffer logits, shared, ids_store[2], wts_store[2], ids[2], wts[2];
  Pipeline pipes[2];
  Router(unsigned rows, unsigned k, unsigned n, unsigned pattern) : R(rows), K(k), norm(n), pattern(pattern) {
    logits = g.alloc(size_t(R) * E * 4); shared = g.alloc(R * 4);
    for (unsigned r = 0; r < R; ++r) {
      shared.as<float>()[r] = float(int(r % 17) - 8) / 3;
      for (unsigned i = 0; i < E; ++i) {
        uint32_t h = (r + 1) * 2654435761u ^ i * 2246822519u;
        h ^= h >> 13;
        float x = float(int(h & 65535) - 32768) / 4096;
        if (pattern == 1) x = float(i % 7);
        if (pattern == 2) x = i % 2 ? 10000.0f : -10000.0f;
        if (pattern == 3) x = i % 2 ? 0.0f : -0.0f;
        if (pattern == 4 && i % 3) x = -INFINITY;
        if (pattern == 5 && i % 71 == 0) x = INFINITY;
        if (pattern == 6 && i % 71 == 0) x = std::numeric_limits<float>::quiet_NaN();
        if (pattern == 7) x = -INFINITY;
        if (pattern == 8) x = std::numeric_limits<float>::quiet_NaN();
        logits.as<float>()[size_t(r) * E + i] = x;
      }
    }
    for (int v = 0; v < 2; ++v) {
      ids_store[v] = g.alloc((size_t(R) * K + 2 * guard) * 4);
      wts_store[v] = g.alloc((size_t(R) * (K + 1) + 2 * guard) * 4);
      std::fill_n(ids_store[v].as<uint32_t>(), R * K + 2 * guard, sentinel);
      std::fill_n(wts_store[v].as<uint32_t>(), R * (K + 1) + 2 * guard, sentinel);
      ids[v] = ids_store[v].view(guard * 4, size_t(R) * K * 4);
      wts[v] = wts_store[v].view(guard * 4, size_t(R) * (K + 1) * 4);
    }
    pipes[0] = g.pipeline("moe_topk512_reference"); pipes[1] = g.pipeline(R > 1 ? "moe_topk512_rows" : "moe_topk512");
  }
  ~Router() { for (Buffer* b : {&logits, &shared, &ids_store[0], &ids_store[1], &wts_store[0], &wts_store[1]}) g.release(*b); }
  double run(int v, int repeats = 1) {
    const struct { unsigned E, k, norm; } p{E, K, norm};
    auto e = g.begin();
    for (int i = 0; i < repeats; ++i) {
      e->use(pipes[v]); e->buf(0, logits); e->buf(1, shared); e->buf(2, ids[v]); e->buf(3, wts[v]); e->val(4, p);
      e->dispatch(R, 1, 1, 32);
    }
    e->commit_and_wait(); return e->gpu_ms() * 1000 / repeats;
  }
  bool check() {
    run(0); run(1);
    unsigned id_diff = 0, weight_diff = 0, guard_errors = 0, oracle_ids = 0;
    double err = 0;
    for (unsigned i = 0; i < R * K; ++i) id_diff += ids[0].as<uint32_t>()[i] != ids[1].as<uint32_t>()[i];
    for (unsigned i = 0; i < R * (K + 1); ++i) {
      const float a = wts[0].as<float>()[i], b = wts[1].as<float>()[i];
      weight_diff += !(std::isnan(a) && std::isnan(b)) && std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b);
    }
    for (int v = 0; v < 2; ++v) for (unsigned i = 0; i < guard; ++i) {
      guard_errors += ids_store[v].as<uint32_t>()[i] != sentinel;
      guard_errors += ids_store[v].as<uint32_t>()[guard + R * K + i] != sentinel;
      guard_errors += wts_store[v].as<uint32_t>()[i] != sentinel;
      guard_errors += wts_store[v].as<uint32_t>()[guard + R * (K + 1) + i] != sentinel;
    }
    if (pattern < 5) for (unsigned r = 0; r < R; ++r) {
      std::vector<unsigned> order(E);
      for (unsigned i = 0; i < E; ++i) order[i] = i;
      const float* l = logits.as<float>() + size_t(r) * E;
      std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return l[a] > l[b]; });
      double den = 0, mx = l[order[0]];
      for (unsigned i = 0; i < (norm ? K : E); ++i) den += std::exp(double(l[order[i]]) - mx);
      for (unsigned q = 0; q < K; ++q) {
        oracle_ids += ids[1].as<uint32_t>()[r * K + q] != order[q];
        const double want = std::exp(double(l[order[q]]) - mx) / den;
        const float got = wts[1].as<float>()[r * (K + 1) + q];
        if (!std::isfinite(got)) err = INFINITY;
        else err = std::max(err, std::abs(got - want));
      }
    }
    printf("R=%u K=%u norm=%u pattern=%u ids=%u weights=%u guards=%u oracle_ids=%u fp64_abs=%.3g\n",
           R, K, norm, pattern, id_diff, weight_diff, guard_errors, oracle_ids, err);
    return !id_diff && !weight_diff && !guard_errors && !oracle_ids && err < 1e-5;
  }
};
}
int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (strcmp(argv[1], "--gpu") && strcmp(argv[1], "--bench")) ||
      (argc == 3 && (strcmp(argv[1], "--gpu") || strcmp(argv[2], "--small")))) fatal("usage: topk512_check --gpu [--small] | --bench");
  ni::bench::GpuLock lock;
  if (!ni::bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("cannot acquire GPU lock");
  Gpu& g = Gpu::get(); g.compile_extra("ordered router reference", reference_source);
  printf("GPU %s, threadgroup bytes A=%u B=%u\n", g.name().c_str(), g.pipeline("moe_topk512_reference").tg_mem, g.pipeline("moe_topk512").tg_mem);
  if (!strcmp(argv[1], "--bench")) {
    { Router warm(256, 10, 1, 0);
      for (int i = 0; i < 48; ++i) { warm.run(0, 8192); warm.run(1, 8192); }
    }
    for (unsigned rows : {1u, 2u, 3u, 4u, 8u, 16u, 32u, 64u, 256u}) for (unsigned norm : {0u, 1u}) {
      Router t(rows, 10, norm, 0);
      for (int i = 0; i < 4; ++i) { t.run(0, 256); t.run(1, 256); }
      for (int pair = 0; pair < 8; ++pair) for (int v = 0; v < 2; ++v) {
        const double us = t.run(v, 8192);
        printf("{\"R\":%u,\"norm\":%u,\"pair\":%d,\"variant\":\"%c\",\"us\":%.6f,\"thermal\":\"%s\"}\n", rows, norm, pair, v ? 'B' : 'A', us, Gpu::thermal_state());
        fflush(stdout);
      }
    }
    return 0;
  }
  int bad = 0, cases = 0;
  for (unsigned rows : {1u, 2u, 3u, 4u, 8u, 16u, 64u, 257u}) {
    if (argc == 3 && rows > 3) continue;
    for (unsigned k : {1u, 2u, 8u, 10u, 16u}) for (unsigned norm : {0u, 1u}) for (unsigned pattern = 0; pattern < 9; ++pattern) {
      Router t(rows, k, norm, pattern); bad += !t.check(); ++cases;
    }
  }
  printf("%d cases, %d failures\n", cases, bad); return bad != 0;
}
