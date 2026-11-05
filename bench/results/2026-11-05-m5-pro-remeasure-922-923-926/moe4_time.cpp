// #343: the verify MoE kernels at Flash-Next's shape (D 2560, I 640, SI 640, E 512, k 10), R rows, timed as N dispatches
// behind one command buffer with a barrier between them (GPU time of the buffer / N), alternating arms is the caller's job.
//   moe4_time KERNEL R [N=200] [REPS=7]   KERNEL: moe4_unique | moe4_gate_up | moe4_gate_up_d | moe4_down | moe4_down_d
// Takes no GPU lock itself: run under NI_LOCK_HELD=1 inside lockf. Output: "KERNEL R=r median us/call ... (reps ...)".
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace ni;
namespace {
std::mt19937 rng(7);
struct Buf {
  Buffer b; size_t n;
  Buf(size_t count, size_t elem = 4) : n(count) { b = Gpu::get().alloc(std::max<size_t>(count * elem, 16)); }
  void rand32() { for (size_t i = 0; i < n; ++i) b.as<uint32_t>()[i] = uint32_t(rng()); }
  void bf(float a, float c) { std::uniform_real_distribution<float> d(a, c); for (size_t i = 0; i < n; ++i) b.as<uint16_t>()[i] = f32_to_bf16(d(rng)); }
  void f(float a, float c) { std::uniform_real_distribution<float> d(a, c); for (size_t i = 0; i < n; ++i) b.as<float>()[i] = d(rng); }
};
}  // namespace
int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: moe4_time KERNEL R [N] [REPS]\n"); return 1; }
  const std::string kn = argv[1];
  const uint32_t R = uint32_t(atoi(argv[2]));
  const int N = argc > 3 ? atoi(argv[3]) : 200, REPS = argc > 4 ? atoi(argv[4]) : 7;
  const uint32_t D = 2560, I = 640, SI = 640, k = 10, E = 512, G = 64;
  Buf gw(size_t(E) * I * D / 8), gs(E * I * D / G, 2), gb(E * I * D / G, 2);
  Buf uw(size_t(E) * I * D / 8), us(E * I * D / G, 2), ub(E * I * D / G, 2);
  Buf dw(size_t(E) * D * I / 8), ds(E * D * I / G, 2), db(E * D * I / G, 2);
  Buf sgw(size_t(SI) * D / 8), sgs(SI * D / G, 2), sgb(SI * D / G, 2), suw(size_t(SI) * D / 8), sus(SI * D / G, 2), sub(SI * D / G, 2);
  Buf sdw(size_t(D) * SI / 8), sds(D * SI / G, 2), sdb(D * SI / G, 2);
  for (Buf* b : {&gw, &uw, &dw, &sgw, &suw, &sdw}) b->rand32();
  for (Buf* b : {&gs, &us, &ds, &sgs, &sus, &sds}) b->bf(0.002f, 0.01f);
  for (Buf* b : {&gb, &ub, &db, &sgb, &sub, &sdb}) b->bf(-0.06f, 0.0f);
  struct { uint32_t D, I, SI, k; float eps; } mp{D, I, SI, k, 1e-6f};
  Buf x(R * D), idx(R * k), wts(R * (k + 1)), y(R * D), hR(R * (k + 1) * I), ue(64), umask(64), uslot(64), un(4);
  x.f(-2, 2); wts.f(0.0f, 0.3f); y.f(-1, 1); hR.f(-1, 1);
  for (uint32_t r = 0; r < R; ++r) {  // experts distinct within a row, drawn from a pool of 24 so that rows share some
    std::vector<uint32_t> pool(24);
    for (uint32_t i = 0; i < 24; ++i) pool[i] = i * 21 % E;
    std::shuffle(pool.begin(), pool.end(), rng);
    for (uint32_t s = 0; s < k; ++s) idx.b.as<uint32_t>()[r * k + s] = pool[s];
  }
  const struct { uint32_t R, k; } rk{R, k};
  auto bind = [&](Encoder& e) {
    if (kn == "moe4_unique") {
      e.buf(0, idx.b); e.buf(1, ue.b); e.buf(2, umask.b); e.buf(3, uslot.b); e.buf(4, un.b); e.val(5, rk);
      e.dispatch(1, 1, 1, 32);
    } else if (kn == "moe4_gate_up" || kn == "moe4_gate_up_d") {
      e.buf(0, gw.b); e.buf(1, gs.b); e.buf(2, gb.b); e.buf(3, uw.b); e.buf(4, us.b); e.buf(5, ub.b);
      e.buf(6, sgw.b); e.buf(7, sgs.b); e.buf(8, sgb.b); e.buf(9, suw.b); e.buf(10, sus.b); e.buf(11, sub.b);
      e.buf(12, x.b); e.buf(13, hR.b); e.val(14, mp); e.buf(15, ue.b); e.buf(16, umask.b); e.buf(17, uslot.b);
      e.buf(18, un.b); e.val(19, R);
      e.dispatch(R, (std::max(I, SI) + 15) / 16, R * k + 1, 256);
    } else {
      e.buf(0, dw.b); e.buf(1, ds.b); e.buf(2, db.b); e.buf(3, sdw.b); e.buf(4, sds.b); e.buf(5, sdb.b);
      e.buf(6, hR.b); e.buf(7, idx.b); e.buf(8, wts.b); e.buf(9, y.b); e.val(10, mp);
      e.dispatch(R, (D + 15) / 16, 1, 256);
    }
  };
  {  // moe4_unique first, once, so ue / umask / uslot / un hold what the later kernels read
    auto e = Gpu::get().begin();
    e->use(Gpu::get().pipeline("moe4_unique"));
    e->buf(0, idx.b); e->buf(1, ue.b); e->buf(2, umask.b); e->buf(3, uslot.b); e->buf(4, un.b); e->val(5, rk);
    e->dispatch(1, 1, 1, 32);
    e->commit_and_wait();
  }
  std::vector<double> us_call;
  for (int rep = -1; rep < REPS; ++rep) {  // rep -1 warms up
    auto e = Gpu::get().begin();
    e->use(Gpu::get().pipeline(kn.c_str()));
    for (int i = 0; i < N; ++i) { bind(*e); if (i + 1 < N) e->barrier(); }
    e->commit_and_wait();
    if (rep >= 0) us_call.push_back(e->gpu_ms() * 1000.0 / N);
  }
  std::vector<double> s = us_call;
  std::sort(s.begin(), s.end());
  printf("%s R=%u median %.2f us/call (min %.2f max %.2f; reps", kn.c_str(), R, s[s.size() / 2], s.front(), s.back());
  for (double v : us_call) printf(" %.2f", v);
  printf(")\n");
  return 0;
}
