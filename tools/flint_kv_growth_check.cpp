// Exercise Flint::reserve with an existing model, checking every cache byte outside the timed call.
//   flint_kv_growth_check MODEL [--small] [--rounds N]
// --small uses a 1024-row cache for shader validation; the default also crosses the copy batch boundary.
// Timings measure cache growth, not inference. The tool takes the GPU lock before loading the model.
#include "bench/bench.h"
#include "model/flint.h"
#include "util/log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ni;

namespace {
void require(bool ok, const char* what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
  }
}

unsigned char pattern(size_t offset, int layer, int kv, int head) {
  return static_cast<unsigned char>(offset * 13 + size_t(layer * 7 + kv * 19 + head * 23));
}

void fill(Seq& s, const FlintConfig& c) {
  for (int l = 0; l < c.layers; ++l)
    for (int kv = 0; kv < 2; ++kv) {
      auto& b = kv ? s.vc[size_t(l)] : s.kc[size_t(l)];
      const size_t stride = b.size / size_t(c.hkv);
      auto* p = b.as<unsigned char>();
      for (int h = 0; h < c.hkv; ++h)
        for (size_t i = 0; i < stride; ++i) p[size_t(h) * stride + i] = pattern(i, l, kv, h);
    }
}

void check(const Seq& s, const FlintConfig& c, int old, const std::vector<void*>& rings) {
  size_t ring = 0;
  for (int l = 0; l < c.layers; ++l)
    for (int kv = 0; kv < 2; ++kv) {
      const auto& b = kv ? s.vc[size_t(l)] : s.kc[size_t(l)];
      const size_t stride = b.size / size_t(c.hkv);
      const bool sliding = c.kind[size_t(l)] == 'S';
      const size_t retained = sliding ? stride : size_t(old) * kv_row_bytes(c.hd);
      require(sliding || stride == size_t(s.cap) * kv_row_bytes(c.hd), "full cache shape");
      if (sliding) require(b.mtl == rings[ring++], "ring identity changed");
      const auto* p = b.as<unsigned char>();
      for (int h = 0; h < c.hkv; ++h)
        for (size_t i = 0; i < stride; ++i)
          require(p[size_t(h) * stride + i] == (i < retained ? pattern(i, l, kv, h) : 0),
                  "retained bytes or zero tail differ");
    }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: flint_kv_growth_check MODEL [--small] [--rounds N]\n");
    return 2;
  }
  bool small = false;
  int rounds = 5;
  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--small") == 0) small = true;
    else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
      char* end = nullptr;
      const long n = std::strtol(argv[++i], &end, 10);
      if (!*argv[i] || *end || n < 1 || n > 100) return 2;
      rounds = int(n);
    } else return 2;
  }
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) {
    fprintf(stderr, "cannot take the GPU lock\n");
    return 1;
  }
  Gpu::AllocationScope scope;
  Flint model(argv[1], 8);
  Gpu& g = Gpu::get();
  const auto& c = model.config();
  const std::vector<int> capacities = small ? std::vector<int>{1024} : std::vector<int>{1024, 4096, 16384, 32768};
  for (int initial : capacities) {
    for (int run = 0; run < rounds; ++run) {
      auto s = model.new_seq(65536, initial);
      const int old = s->cap;
      for (const auto* v : {&s->kc, &s->vc})
        for (const auto& b : *v) {
          const auto* p = b.as<unsigned char>();
          require(std::all_of(p, p + b.size, [](unsigned char x) { return x == 0; }), "initial cache not zero");
        }
      fill(*s, c);
      std::vector<void*> rings, identities;
      for (int l = 0; l < c.layers; ++l)
        for (int kv = 0; kv < 2; ++kv) {
          const auto& b = kv ? s->vc[size_t(l)] : s->kc[size_t(l)];
          identities.push_back(b.mtl);
          if (c.kind[size_t(l)] == 'S') rings.push_back(b.mtl);
        }
      require(model.reserve(*s, old), "no-op reserve failed");
      require(!model.reserve(*s, s->limit + 1), "limit refusal failed");
      model.set_mem_budget(1);
      require(!model.reserve(*s, old + 1), "budget refusal failed");
      model.set_mem_budget(0);
      require(s->cap == old, "refusal changed capacity");
      size_t identity = 0;
      for (int l = 0; l < c.layers; ++l)
        for (int kv = 0; kv < 2; ++kv)
          require((kv ? s->vc[size_t(l)] : s->kc[size_t(l)]).mtl == identities[identity++],
                  "no-op or refusal changed a buffer");
      check(*s, c, old, rings);
      const auto start = std::chrono::steady_clock::now();
      require(model.reserve(*s, old + 1), "growth failed");
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      require(s->cap > old, "capacity did not grow");
      check(*s, c, old, rings);
      printf("{\"old\":%d,\"capacity\":%d,\"run\":%d,\"ms\":%.6f,\"bytes_match\":true,\"rings_unchanged\":true}\n",
             old, s->cap, run, ms);
      fflush(stdout);
      model.drop_kv(*s);
      g.release(s->tok);
    }
  }
  puts("PASS: initial zeros, no-op, limit and budget refusal, retained bytes, zero tails, ring identity and contents");
  return 0;
}
