// Reference/main comparison for ordinary RoPE and interleaved M-RoPE, including untouched cache rows.
// --gpu --small runs first with Metal validation. --bench prints all interleaved samples.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/log.h"
#include "rope_pair_reference.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace ni;
int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (strcmp(argv[1], "--gpu") && strcmp(argv[1], "--bench")) ||
      (argc == 3 && (strcmp(argv[1], "--gpu") || strcmp(argv[2], "--small"))))
    fatal("usage: rope_pair_test --gpu [--small] | --bench");
  ni::bench::GpuLock lock;
  if (!ni::bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("cannot acquire GPU lock");
  auto& g = Gpu::get(); g.compile(reference_source);
  const bool bench = !strcmp(argv[1], "--bench"), small = argc == 3;
  printf("GPU %s thermal=%s\n", g.name().c_str(), Gpu::thermal_state());
  int failed = 0;
  for (unsigned D : {64u, 128u, 256u, 512u}) for (unsigned T : {1u, 8u, 128u, 512u}) {
    if (small && (D > 128 || T > 8)) continue;
    if (bench && D != 256) continue;
    for (unsigned rot : {0u, 2u, 32u, 64u, 128u, 256u, 512u}) for (unsigned mode : {0u, 1u, 2u}) {
      if (rot > D) continue;
      if (bench && (rot != 64 || mode == 2)) continue;
      const unsigned Hq = 24, Hkv = 4, pos = small ? 3 : 257, cap = pos + T + 2;
      const size_t nq = size_t(T) * Hq * D, nk = size_t(T) * Hkv * D, cache = size_t(cap) * Hkv * D;
      Buffer qg = g.alloc(nq * 8), kr = g.alloc(nk * 4), vr = g.alloc(nk * 4), w = g.alloc(D * 2);
      Buffer mp = g.alloc(cap * 3 * 4), q[2], k[2], v[2];
      for (Buffer* b : {&qg, &kr, &vr}) for (size_t i = 0; i < b->size / 4; ++i)
        b->as<float>()[i] = float(int(i * 73 % 521) - 260) / 71;
      for (unsigned i = 0; i < D; ++i) w.as<uint16_t>()[i] = 0x3f00 + i % 128;
      for (unsigned i = 0; i < cap * 3; ++i) mp.as<int>()[i] = int(i % 67) - 9;
      for (int arm = 0; arm < 2; ++arm) {
        q[arm] = g.alloc(nq * 4 + 64); k[arm] = g.alloc(cache * 2 + 64); v[arm] = g.alloc(cache * 2 + 64);
        for (Buffer* b : {&q[arm], &k[arm], &v[arm]}) memset(b->ptr(), 0x55, b->size);
      }
      Pipeline pipes[] = {g.pipeline(mode ? "attn_prep_m_reference" : "attn_prep_reference"),
                          g.pipeline(mode ? "attn_prep_m" : "attn_prep")};
      auto run = [&](int arm, int repeats) {
        kp::AttnPrep p{T,Hq,Hkv,D,rot,pos,cap,1e-6f,1000000.f};
        struct { unsigned mn; int delta; unsigned sec1, sec2; } mr{mode == 1 ? cap : pos + T / 2,-3,11,13};
        auto e = g.begin();
        for (int i = 0; i < repeats; ++i) {
          e->use(pipes[arm]); e->buf(0,qg); e->buf(1,kr); e->buf(2,vr); e->buf(3,w); e->buf(4,w);
          e->buf(5,q[arm]); e->buf(6,k[arm]); e->buf(7,v[arm]); e->val(8,p);
          if (mode) { e->buf(9,mp); e->val(10,mr); }
          e->dispatch(Hq + 2 * Hkv,T,1,D);
        }
        e->commit_and_wait(); return e->gpu_ms() * 1000 / repeats;
      };
      if (bench) {
        for (int warm = 0; warm < 4; ++warm) { run(0,64); run(1,64); }
        for (int pair = 0; pair < 6; ++pair) for (int arm = 0; arm < 2; ++arm)
          printf("{\"D\":%u,\"T\":%u,\"mode\":%u,\"pair\":%d,\"variant\":\"%c\",\"us\":%.6f,\"thermal\":\"%s\"}\n",
                 D,T,mode,pair+1,arm?'B':'A',run(arm,64),Gpu::thermal_state());
      } else {
        run(0,1); run(1,1);
        bool different = memcmp(q[0].ptr(),q[1].ptr(),q[0].size) || memcmp(k[0].ptr(),k[1].ptr(),k[0].size) ||
                         memcmp(v[0].ptr(),v[1].ptr(),v[0].size);
        size_t guards = 0;
        for (int arm = 0; arm < 2; ++arm) for (Buffer* b : {&q[arm],&k[arm],&v[arm]})
          for (size_t i = b->size-64; i < b->size; ++i) guards += b->as<uint8_t>()[i] != 0x55;
        printf("D=%u T=%u rot=%u mode=%u mismatch=%d guards=%zu\n",D,T,rot,mode,different,guards);
        failed += different || guards;
      }
      for (Buffer* b : {&qg,&kr,&vr,&w,&mp,&q[0],&q[1],&k[0],&k[1],&v[0],&v[1]}) g.release(*b);
      fflush(stdout);
    }
  }
  printf("failures=%d\n",failed); return failed != 0;
}
