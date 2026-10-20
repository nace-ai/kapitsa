// The fused small passes of Nemotron-H's multi-row (verify) path against the pairs of passes they replace, bit for
// bit, on random rows at small sizes (run it with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1):
//   nh_rows_half with its norm         = nh_rmsnorm, then nh_rows_half
//   nh_rmsnorm_h (fp32 and half rows)   = nh_rmsnorm, then nh_rows_half
//   nh_attn_reduce_h                    = nh_attn_reduce, then nh_rows_half
//   nh_gnorm_half_ring                  = nh_gnorm_half, then nh_conv_ring
//   nh_xb_v8_qkv                        = nh_xb_v8 for q, k and v
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
struct GnormP {
  uint32_t D, G;
  float eps;
};
struct RingP {
  uint32_t T, pos0, zx_stride, DI, CH;
};
struct V8P {
  uint32_t N, K, x_ld, y_ld, R;
  float gs;
  uint32_t epi;
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
  // grouped norm of R rows of DI = 4 groups x 128 channels, with the ring writes of np = 3 pending rows (CH = DI + 256)
  {
    const uint32_t DI = 512, G = 128, CH = DI + 256, IP = 2 * DI + 256 + 16, np = 3, pos0 = 41;
    Buffer y = g.alloc(size_t(R) * DI * 4), gw = g.alloc(DI * 4), zp = g.alloc(size_t(np) * IP * 4),
           ring1 = g.alloc(size_t(4) * CH * 4), ring2 = g.alloc(size_t(4) * CH * 4), o1 = g.alloc(size_t(R) * DI * 2),
           o2 = g.alloc(size_t(R) * DI * 2);
    for (size_t i = 0; i < size_t(R) * DI; ++i) y.as<float>()[i] = nd(rng);
    for (uint32_t i = 0; i < DI; ++i) gw.as<float>()[i] = 1.0f + 0.1f * nd(rng);
    for (size_t i = 0; i < size_t(np) * IP; ++i) zp.as<float>()[i] = nd(rng);
    for (size_t i = 0; i < size_t(4) * CH; ++i) ring1.as<float>()[i] = ring2.as<float>()[i] = nd(rng);
    const GnormP gp{DI, G, eps};
    const RingP rp{np, pos0 - np, IP, DI, CH};
    auto e = g.begin();
    e->use(g.pipeline("nh_gnorm_half"));
    e->buf(0, y);
    e->buf(1, gw);
    e->buf(2, o1);
    e->val(3, gp);
    e->dispatch(R * (DI / G), 1, 1, 256);
    e->use(g.pipeline("nh_conv_ring"));
    e->buf(0, zp);
    e->buf(1, ring1);
    e->val(2, rp);
    e->dispatch((CH * 3 + 255) / 256, 1, 1, 256);
    e->use(g.pipeline("nh_gnorm_half_ring"));
    e->buf(0, y);
    e->buf(1, gw);
    e->buf(2, o2);
    e->val(3, gp);
    e->buf(4, zp);
    e->buf(5, ring2);
    e->val(6, rp);
    e->dispatch(R * (DI / G), 1, 1, 256);
    e->commit_and_wait();
    printf("nh_gnorm_half_ring     vs nh_gnorm_half + nh_conv_ring:  %s\n",
           same(o1, o2, size_t(R) * DI * 2) && same(ring1, ring2, size_t(4) * CH * 4) ? "identical" : "DIFFERENT");
  }
  // q [96][K], k / v [32][K] in BF16, K = 256
  {
    const uint32_t K = 256, NQ = 96, NK = 32;
    Buffer xh = g.alloc(size_t(R) * K * 2), wq = g.alloc(size_t(NQ) * K * 2), wk = g.alloc(size_t(NK) * K * 2),
           wv = g.alloc(size_t(NK) * K * 2);
    for (size_t i = 0; i < size_t(R) * K; ++i) xh.as<__fp16>()[i] = __fp16(nd(rng));
    for (Buffer* b : {&wq, &wk, &wv})
      for (size_t i = 0; i < b->size / 2; ++i) {
        const float f = 0.05f * nd(rng);
        uint32_t u;
        memcpy(&u, &f, 4);
        b->as<uint16_t>()[i] = uint16_t(u >> 16);
      }
    Buffer y1[3], y2[3];
    for (int i = 0; i < 3; ++i) {
      y1[i] = g.alloc(size_t(R) * NQ * 4);
      y2[i] = g.alloc(size_t(R) * NQ * 4);
    }
    const V8P pq{NQ, K, K, NQ, R, 1.0f, 0}, pk{NK, K, K, NK, R, 1.0f, 0};
    auto e = g.begin();
    const Buffer* w3[3] = {&wq, &wk, &wv};
    for (int i = 0; i < 3; ++i) {
      e->use(g.pipeline("nh_xb_v8"));
      e->buf(0, xh);
      e->buf(1, *w3[i]);
      e->buf(2, *w3[i]);
      e->buf(3, y1[i]);
      e->val(4, i ? pk : pq);
      e->dispatch(((i ? NK : NQ) + 15) / 16, 1, 1, 256);
    }
    e->use(g.pipeline("nh_xb_v8_qkv"));
    e->buf(0, xh);
    e->buf(1, wq);
    e->buf(2, wk);
    e->buf(3, y2[0]);
    e->val(4, pq);
    e->buf(5, wv);
    e->buf(6, y2[1]);
    e->buf(7, y2[2]);
    e->val(8, pk);
    e->dispatch((NQ + 15) / 16 + 2 * ((NK + 15) / 16), 1, 1, 256);
    e->commit_and_wait();
    printf("nh_xb_v8_qkv           vs nh_xb_v8 x 3:                  %s\n",
           same(y1[0], y2[0], size_t(R) * NQ * 4) && same(y1[1], y2[1], size_t(R) * NK * 4) &&
                   same(y1[2], y2[2], size_t(R) * NK * 4)
               ? "identical"
               : "DIFFERENT");
  }
  return any_different ? 1 : 0;
}
