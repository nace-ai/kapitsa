// --gpu [position]: the fused tree-zone tail/reduction against the separate kernels bitwise, and fp64.
// --bench P: interleaved separate/fused attention timings on the 27B's head layout. Both modes take the GPU lock.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kflash_budget.h"
#include "model/kparams.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

using namespace ni;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { printf("FAIL: %s\n", what); ++failures; }
}

// Deterministic values indexed by logical position, independent of the KV allocation's stride or tree slots.
float value(unsigned p, unsigned h, unsigned d, unsigned salt) {
  unsigned x = p * 2654435761u ^ h * 2246822519u ^ d * 3266489917u ^ salt;
  x ^= x >> 16; x *= 2246822519u; x ^= x >> 13;
  return float(int(x & 65535u) - 32768) / 32768.0f;
}

struct Attention {
  Gpu& g = Gpu::get();
  unsigned Hq, Hkv, R, P, cap, target;
  Buffer q, qg, k, v, po, ml, out, tree;
  Pipeline partial, tail, reduce, fused;
  Attention(unsigned pos, unsigned stride, unsigned heads, unsigned kvheads, const std::vector<int>& parents,
            const std::vector<int>& path)
      : Hq(heads), Hkv(kvheads), R(unsigned(parents.size())), P(pos - unsigned(path.size()) + 1), cap(stride),
        target(unsigned(path.back())) {
    q = g.alloc(8 * Hq * 256 * 4); qg = g.alloc(8 * Hq * 256 * 8); out = g.alloc(8 * Hq * 256 * 4);
    k = g.alloc(size_t(Hkv) * cap * 256 * 2); v = g.alloc(size_t(Hkv) * cap * 256 * 2);
    po = g.alloc(1024 * Hq * 256 * 4); ml = g.alloc(1024 * Hq * 8); tree = g.alloc(kf::kWords * 4);
    memset(qg.ptr(), 0, qg.size); memset(tree.ptr(), 0, tree.size);
    tree.as<int>()[kf::kN] = int(R);
    std::vector<unsigned> dep(R), salts(R, 271u);
    for (int r : path) salts[size_t(r)] = 0;
    for (unsigned r = 0; r < R; ++r) {
      const int parent = parents[r];
      dep[r] = parent < 0 ? 0 : dep[size_t(parent)] + 1;
      tree.as<int>()[kf::kMask + r] = (1u << r) | (parent < 0 ? 0 : tree.as<int>()[kf::kMask + parent]);
      for (unsigned h = 0; h < Hq; ++h)
        for (unsigned d = 0; d < 256; ++d) {
          q.as<float>()[(r * Hq + h) * 256 + d] = value(P + dep[r], h, d, 13 + salts[r]);
          qg.as<float>()[(r * Hq + h) * 512 + 256 + d] = 4 * value(P + dep[r], h, d, 131);
        }
    }
    for (unsigned h = 0; h < Hkv; ++h)
      for (unsigned row = 0; row < P + R; ++row) {
        const unsigned logical = row < P ? row : P + dep[row - P];
        const unsigned salt = row < P ? 0 : salts[row - P];
        for (unsigned d = 0; d < 256; ++d) {
          const size_t i = (size_t(h) * cap + row) * 256 + d;
          k.as<__fp16>()[i] = value(logical, h, d, 31 + salt);
          v.as<__fp16>()[i] = value(logical, h, d, 79 + salt);
        }
      }
    partial = g.pipeline("attn8m_partial_zone"); tail = g.pipeline("attn8_tree_zone"); reduce = g.pipeline("attn_rows_reduce");
    fused = g.pipeline("attn8_tree_zone_reduce");
  }
  ~Attention() { for (Buffer* b : {&q, &qg, &k, &v, &po, &ml, &out, &tree}) g.release(*b); }
  unsigned fixed_chunk() const {
    unsigned c = 128;
    while (((cap + c - 1) / c + 1) * R > 1024) c += 32;
    return c;
  }
  unsigned chunks() const {
    const unsigned c = fixed_chunk();
    return std::max(1u, (P + c - 1) / c);
  }
  double run(bool fuse, int reps = 1) {
    const unsigned nc = chunks();
    check((nc + 1) * R <= 1024, "GPU grid fits scratch");
    if ((nc + 1) * R > 1024) fatal("test grid exceeds scratch");
    kp::AttnRows p{Hq, Hkv, P, fixed_chunk(), cap, 1.0f / 16, nc + 1};
    auto e = g.begin();
    for (int i = 0; i < reps; ++i) {
      e->use(partial); e->buf(0, q); e->buf(1, k); e->buf(2, v); e->buf(3, po); e->buf(4, ml);
      e->val(5, p); e->val(6, R); e->buf(7, tree); e->dispatch(nc, Hkv, 1, 64 * Hq / Hkv);
      e->use(fuse ? fused : tail); e->buf(0, q); e->buf(1, k); e->buf(2, v); e->buf(3, po); e->buf(4, ml);
      if (fuse) { e->buf(7, qg); e->buf(8, out); }
      e->val(5, p); e->buf(6, tree); e->dispatch(Hq, R, 1, 32);
      if (!fuse) {
        e->use(reduce); e->buf(0, po); e->buf(1, ml); e->buf(2, qg); e->buf(3, out); e->val(4, p);
        e->dispatch(Hq, R, 1, 256);
      }
    }
    e->commit_and_wait();
    return e->gpu_ms() / reps;
  }
  std::vector<float> result() const {
    const float* a = out.as<float>();
    return {a, a + R * Hq * 256};
  }
  void reference(unsigned pos) const {
    double maxerr = 0, maxref = 0;
    for (unsigned h : {0u, Hq - 1}) {
      const unsigned kh = h / (Hq / Hkv);
      double sum = 0;
      std::vector<double> acc(256);
      for (unsigned key = 0; key <= pos; ++key) {
        double score = 0;
        for (unsigned d = 0; d < 256; ++d)
          score += double(__fp16(value(pos, h, d, 13) / 16)) * double(__fp16(value(key, kh, d, 31)));
        const double w = std::exp(score);
        sum += w;
        for (unsigned d = 0; d < 256; ++d) acc[d] += w * double(__fp16(value(key, kh, d, 79)));
      }
      for (unsigned d = 0; d < 256; ++d) {
        const double gate = 1 / (1 + std::exp(-4 * double(value(pos, h, d, 131))));
        const double ref = acc[d] / sum * gate;
        const double actual = out.as<float>()[(target * Hq + h) * 256 + d];
        check(std::isfinite(actual), "finite GPU output");
        maxerr = std::max(maxerr, std::fabs(actual - ref)); maxref = std::max(maxref, std::fabs(ref));
      }
    }
    check(maxerr / maxref < 2e-4, "attention against fp64");
    printf("  fp64 relative error %.3g\n", maxerr / maxref);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (strcmp(argv[1], "--gpu") && strcmp(argv[1], "--bench")))
    fatal("usage: tree_reduce_test --gpu [P] | --bench P");
  ni::bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD") && !lock.acquire()) fatal("cannot acquire GPU lock");
  const bool bench = !strcmp(argv[1], "--bench");
  if (bench && argc != 3) fatal("--bench needs a position");
  const unsigned pos = argc > 2 ? unsigned(atoi(argv[2])) : 39;
  if (pos < 7 || pos > 131064) fatal("position must be 7..131064");
  if (bench) {
    Attention a(pos, ((pos + 1023) / 1024 + 1) * 1024, 24, 4, {-1, 0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 6, 7});
    for (int i = 0; i < 4; ++i) { a.run(false, 16); a.run(true, 16); }
    for (bool fuse : {false, true, true, false}) {
      printf("P=%u mode=%s scratch_partials=%u ms=", pos, fuse ? "fused" : "separate", a.chunks() + (fuse ? 0 : 1));
      for (int i = 0; i < 7; ++i) printf("%.6f,", a.run(fuse, 512));
      printf("\n"); fflush(stdout);
    }
  } else {
    std::vector<unsigned> positions{7, 8, 31, 32, 38, 39, 4095, 8191, 16256, 16384, 19456, 32760, 131064};
    if (argc == 3) positions = {pos};
    const std::vector<std::vector<int>> parents{{-1, 0, 1, 2, 3, 4, 5, 6}, {-1}, {-1, 0, 1}, {-1, 0, 0, 2, 2, 4, 4, 6}};
    const std::vector<std::vector<int>> paths{{0, 1, 2, 3, 4, 5, 6, 7}, {0}, {0, 1, 2}, {0, 2, 4, 6}};
    for (unsigned pos : positions) {
      for (auto [hq, hkv] : {std::pair{4u, 1u}, {6u, 1u}, {8u, 1u}, {24u, 4u}})
        for (unsigned cap : {pos + 16, ((pos + 1023) / 1024 + 2) * 1024})
          for (size_t shape = 0; shape < parents.size(); ++shape) {
            Attention a(pos, cap, hq, hkv, parents[shape], paths[shape]);
            a.run(false);
            const auto old = a.result();
            const unsigned nc = a.chunks() + 1;
            for (unsigned row = 0; row < a.R * a.Hq; ++row) {
              const unsigned last = row * nc + nc - 1;
              a.ml.as<float>()[last * 2] = a.ml.as<float>()[last * 2 + 1] = std::numeric_limits<float>::quiet_NaN();
              std::fill_n(a.po.as<float>() + last * 256, 256, std::numeric_limits<float>::quiet_NaN());
            }
            a.run(true);
            const auto got = a.result();
            check(memcmp(old.data(), got.data(), old.size() * sizeof(float)) == 0, "fused tail matches separate kernels bitwise");
            if (shape == 0 && cap == pos + 16) a.reference(pos);
          }
      printf("Fused GPU P=%u: %d failures\n", pos, failures);
    }
  }
  return failures != 0;
}
