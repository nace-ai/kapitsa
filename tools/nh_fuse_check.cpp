// The fused small passes of Nemotron-H's multi-row (verify) path against the pairs of passes they replace, bit for
// bit, on random rows at small sizes (run it with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1):
//   nh_rows_half with its norm         = nh_rmsnorm, then nh_rows_half
//   nh_rmsnorm_h (fp32 and half rows)   = nh_rmsnorm, then nh_rows_half
//   nh_attn_reduce_h                    = nh_attn_reduce, then nh_rows_half
//   nh_fuse_check [D] [R]   (default D 2688, R 8)
#include "metal/gpu.h"
#include "util/log.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {
struct NormP {
  uint32_t D;
  float eps;
  uint32_t x_stride, y_stride;
};
struct HalfP {
  uint32_t D, norm;
  float eps;
  uint32_t x_ld;
};
struct AttnP {
  uint32_t R, pos0, cap, Hq, Hkv, chunk, nchunks;
  float scale;
};
bool any_different = false;  // the exit status: 1 when a fused kernel's output differs
bool same(const Buffer& a, const Buffer& b, size_t bytes) {
  const bool eq = memcmp(a.ptr(), b.ptr(), bytes) == 0;
  any_different |= !eq;
  return eq;
}
}  // namespace

int main(int argc, char** argv) {
  const uint32_t D = argc > 1 ? uint32_t(atoi(argv[1])) : 2688, R = argc > 2 ? uint32_t(atoi(argv[2])) : 8;
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const float eps = 1e-5f;
  Buffer x = g.alloc(size_t(R) * D * 4), w = g.alloc(size_t(D) * 4), y = g.alloc(size_t(R) * D * 4),
         y2 = g.alloc(size_t(R) * D * 4), h1 = g.alloc(size_t(R) * D * 2), h2 = g.alloc(size_t(R) * D * 2),
         h3 = g.alloc(size_t(R) * D * 2);
  for (size_t i = 0; i < size_t(R) * D; ++i) x.as<float>()[i] = nd(rng) * (1.0f + float(i % 7));
  for (uint32_t i = 0; i < D; ++i) w.as<float>()[i] = 1.0f + 0.1f * nd(rng);
  auto norm = [&](auto& e) {
    const NormP p{D, eps, D, D};
    e->use(g.pipeline("nh_rmsnorm"));
    e->buf(0, x);
    e->buf(1, w);
    e->buf(2, y);
    e->val(3, p);
    e->dispatch(R, 1, 1, 1024);
  };
  auto half = [&](auto& e, const Buffer& in, const Buffer& out, uint32_t n, uint32_t donorm) {
    const HalfP hp{n, donorm, eps, n};
    e->use(g.pipeline("nh_rows_half"));
    e->buf(0, in);
    e->buf(1, w);
    e->buf(2, out);
    e->val(3, hp);
    e->dispatch(R, 1, 1, 1024);
  };
  {
    auto e = g.begin();
    norm(e);
    half(e, y, h1, D, 0);  // the two passes
    half(e, x, h2, D, 1);  // one: nh_rows_half's norm
    const NormP p{D, eps, D, D};
    e->use(g.pipeline("nh_rmsnorm_h"));  // one: both outputs
    e->buf(0, x);
    e->buf(1, w);
    e->buf(2, y2);
    e->val(3, p);
    e->buf(4, h3);
    e->dispatch(R, 1, 1, 1024);
    e->commit_and_wait();
  }
  printf("nh_rows_half norm      vs nh_rmsnorm + nh_rows_half: %s\n", same(h1, h2, size_t(R) * D * 2) ? "identical" : "DIFFERENT");
  printf("nh_rmsnorm_h (fp32)    vs nh_rmsnorm:                %s\n", same(y, y2, size_t(R) * D * 4) ? "identical" : "DIFFERENT");
  printf("nh_rmsnorm_h (half)    vs nh_rmsnorm + nh_rows_half: %s\n", same(h1, h3, size_t(R) * D * 2) ? "identical" : "DIFFERENT");
  // attention reduce: Hq heads of 128, 3 chunks of 256 positions, rows at positions 300 + r
  const uint32_t Hq = 4, HD = 128, nch = 3, QW = Hq * HD;
  const AttnP ap{R, 300, 4096, Hq, 1, 256, nch, 0.088f};
  Buffer po = g.alloc(size_t(R) * Hq * nch * HD * 4), pml = g.alloc(size_t(R) * Hq * nch * 2 * 4),
         ao = g.alloc(size_t(R) * QW * 4), a1 = g.alloc(size_t(R) * QW * 2), a2 = g.alloc(size_t(R) * QW * 2);
  for (size_t i = 0; i < size_t(R) * Hq * nch * HD; ++i) po.as<float>()[i] = nd(rng);
  for (size_t i = 0; i < size_t(R) * Hq * nch; ++i) {
    pml.as<float>()[2 * i] = 3.0f * nd(rng);
    pml.as<float>()[2 * i + 1] = 1.0f + std::fabs(nd(rng));
  }
  {
    auto e = g.begin();
    e->use(g.pipeline("nh_attn_reduce"));
    e->buf(0, po);
    e->buf(1, pml);
    e->buf(2, ao);
    e->val(3, ap);
    e->dispatch(Hq, R, 1, HD);
    half(e, ao, a1, QW, 0);
    e->use(g.pipeline("nh_attn_reduce_h"));
    e->buf(0, po);
    e->buf(1, pml);
    e->buf(2, a2);
    e->val(3, ap);
    e->dispatch(Hq, R, 1, HD);
    e->commit_and_wait();
  }
  printf("nh_attn_reduce_h       vs nh_attn_reduce + nh_rows_half: %s\n", same(a1, a2, size_t(R) * QW * 2) ? "identical" : "DIFFERENT");
  return any_different ? 1 : 0;
}
