// rows_norm_f32 + x8_prep2 versus the fused small-row preparation, bytewise and against fp64.
// Takes the GPU lock. Small mode must run first with both Metal validation layers.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace ni;

struct Guarded {
  Buffer storage, data;
  Guarded() = default;
  explicit Guarded(size_t bytes) {
    storage = Gpu::get().alloc(bytes + 128);
    data = storage.view(64, bytes);
    memset(storage.ptr(), 0x55, storage.size);
  }
  void release() { Gpu::get().release(storage); }
  bool guards() const {
    const auto* p = storage.as<unsigned char>();
    for (size_t i = 0; i < 64; ++i)
      if (p[i] != 0x55 || p[64 + data.size + i] != 0x55) return false;
    return true;
  }
};

struct Case {
  Gpu& gpu = Gpu::get();
  unsigned K, R, ld;
  bool lo, transposed, write_xn;
  float eps;
  Buffer x, w, normalized;
  std::array<std::array<Guarded, 6>, 2> out;
  Pipeline norm, prep, fused;
  Case(unsigned k, unsigned r, unsigned pattern, bool extras, bool output)
      : K(k), R(r), ld(k + (extras ? 16 : 0)), lo(extras), transposed(extras), write_xn(output),
        eps(pattern == 3 ? 1e-5f : 1e-6f) {
    x = gpu.alloc(size_t(R) * K * 4);
    w = gpu.alloc(K * 2);
    normalized = gpu.alloc(size_t(R) * K * 4);
    std::mt19937 rng(173 + K + R * 13 + pattern);
    std::uniform_real_distribution<float> u(-1, 1);
    for (unsigned i = 0; i < R * K; ++i) {
      float v = u(rng);
      if (pattern == 1) v = 0;
      if (pattern == 2) v *= 1e-5f;
      if (pattern == 3) v *= (i % 5 == 0 ? 100.0f : 0.001f);
      x.as<float>()[i] = v;
    }
    for (unsigned i = 0; i < K; ++i) w.as<uint16_t>()[i] = f32_to_bf16(1.0f + .8f * u(rng));
    const std::array<size_t, 6> sizes{size_t(K) * 8 * 2, size_t(K / 64) * 8 * 4,
                                    size_t(R) * ld * 4, size_t(K) * 8 * 2,
                                    size_t(K / 64) * 8 * 4, size_t(K) * 8 * 2};
    for (auto& a : out) for (size_t j = 0; j < a.size(); ++j) a[j] = Guarded(sizes[j]);
    norm = gpu.pipeline("rows_norm_f32"); prep = gpu.pipeline("x8_prep2");
    fused = gpu.pipeline("x8_prep2_rows_norm");
  }
  ~Case() {
    for (auto& a : out) for (auto& b : a) b.release();
    for (Buffer* b : {&x, &w, &normalized}) gpu.release(*b);
  }
  void encode(Encoder& e, unsigned arm) {
    // Production already retains rows_norm_f32's output. Do not time a second copy of it.
    const bool direct_norm = write_xn && ld == K;
    const Buffer& norm_out = direct_norm ? out[0][2].data : normalized;
    if (!arm) {
      const kp::RowsNorm np{K, eps};
      e.use(norm); e.buf(0, x); e.buf(1, w); e.buf(2, norm_out); e.val(3, np);
      e.dispatch(R, 1, 1, 256);
    }
    const kp::X8Prep p{K, R, K, arm, unsigned(write_xn && (arm || !direct_norm)), eps, ld,
                       unsigned(lo), 0, unsigned(transposed)};
    auto& a = out[arm];
    e.use(arm ? fused : prep); e.buf(0, arm ? x : norm_out); e.buf(1, w);
    e.buf(2, a[0].data); e.buf(3, a[1].data); e.buf(4, a[2].data); e.val(5, p);
    e.buf(6, a[3].data); e.buf(7, a[4].data); e.buf(8, a[5].data);
    e.dispatch(K / 256, 1, 1, 256);
  }
  double run(unsigned arm, unsigned reps = 1) {
    auto e = gpu.begin();
    for (unsigned i = 0; i < reps; ++i) encode(*e, arm);
    e->commit_and_wait();
    return e->gpu_ms() * 1000 / reps;
  }
  bool check() {
    run(0); run(1);
    bool ok = true;
    const char* names[] = {"table", "sums", "normalized", "lo", "lo_sums", "transposed"};
    for (size_t j = 0; j < out[0].size(); ++j) {
      size_t different = 0;
      for (size_t i = 0; i < out[0][j].data.size; ++i)
        different += out[0][j].data.as<unsigned char>()[i] != out[1][j].data.as<unsigned char>()[i];
      bool guards = out[0][j].guards() && out[1][j].guards();
      if (different || !guards) {
        printf("FAIL K=%u R=%u %s bytes=%zu guards=%d\n", K, R, names[j], different, guards);
        ok = false;
      }
    }
    // Independent fp64 RMSNorm check: catch equal-but-wrong and nonfinite results.
    if (write_xn) for (unsigned r = 0; r < R; ++r) {
      double ss = 0;
      for (unsigned i = 0; i < K; ++i) { double v = x.as<float>()[r * K + i]; ss += v * v; }
      double inv = 1 / std::sqrt(ss / K + eps);
      for (unsigned i = 0; i < K; ++i) {
        double ref = x.as<float>()[r * K + i] * inv * bf16_to_f32(w.as<uint16_t>()[i]);
        for (unsigned a = 0; a < 2; ++a) {
          float v = out[a][2].data.as<float>()[r * ld + i];
          if (!std::isfinite(v) || std::abs(v - ref) > 2e-5 * std::max(1.0, std::abs(ref))) ok = false;
        }
      }
    }
    return ok;
  }
};

int main(int argc, char** argv) {
  if (argc != 2) fatal("usage: norm_x8_test small|check|bench");
  const std::string mode = argv[1];
  if (mode != "small" && mode != "check" && mode != "bench") fatal("invalid mode");
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("cannot acquire GPU lock");
  auto& gpu = Gpu::get(); gpu.compile();
  printf("GPU=%s thermal=%s\n", gpu.name().c_str(), Gpu::thermal_state());
  if (mode != "bench") {
    unsigned passed = 0, failed = 0;
    const std::vector<unsigned> widths = mode == "small" ? std::vector<unsigned>{256,512} :
      std::vector<unsigned>{768,1024,2048,2560,4096,5120,9216};
    for (unsigned k : widths) for (unsigned r : {1u,2u,3u,8u})
      for (unsigned pattern = 0; pattern < 4; ++pattern) for (bool extras : {false,true}) {
        Case c(k, r, pattern, extras, true);
        if (c.check()) ++passed; else ++failed;
      }
    Case no_output(2048, 2, 0, false, false);
    if (no_output.check()) ++passed; else ++failed;
    printf("CHECK passed=%u failed=%u\n", passed, failed);
    return failed ? 1 : 0;
  }
  for (unsigned k : {2048u,2560u,4096u,5120u}) for (unsigned r : {1u,2u}) {
    Case c(k,r,0,false,true);
    if (!c.check()) return 1;
    for (unsigned i=0;i<4;++i) {c.run(0,16); c.run(1,16);}
    const unsigned reps = unsigned(std::clamp(15000.0 / std::max(c.run(0,32),c.run(1,32)),64.0,8192.0));
    for (unsigned pair=0;pair<32;++pair) {
      double time[2];
      for (unsigned j=0;j<2;++j) { unsigned arm=j ^ (pair & 1); time[arm]=c.run(arm,reps); }
      printf("PAIR K=%u R=%u pair=%u reps=%u A_us=%.6f B_us=%.6f thermal=%s\n",
             k,r,pair,reps,time[0],time[1],Gpu::thermal_state());
      fflush(stdout);
    }
  }
}
