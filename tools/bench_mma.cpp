// simdgroup-matrix peak microbenchmark: 8x8x8 MMAs on registers only (no memory traffic).
//   bench_mma [iters=4096] [threadgroups=2048] [threads=128]
// Prints TFLOP/s for half x half -> float, float x float -> float, half x half -> half.
#include "metal/gpu.h"
#include "util/log.h"

#include <cstdio>
#include <string>

using namespace ni;

static const char* kSrc = R"MSL(
template <typename TI, typename TA, int FA, int FB>
NI_INLINE void mma_loop(device float* out, uint iters, uint gid, uint lane) {
  simdgroup_matrix<TI, 8, 8> a[FA], b[FB];
  simdgroup_matrix<TA, 8, 8> acc[FA][FB];
  for (int i = 0; i < FA; ++i) a[i] = simdgroup_matrix<TI, 8, 8>(TI(0.001f * (i + 1)));
  for (int j = 0; j < FB; ++j) b[j] = simdgroup_matrix<TI, 8, 8>(TI(0.002f * (j + 1)));
  for (int i = 0; i < FA; ++i)
    for (int j = 0; j < FB; ++j) acc[i][j] = simdgroup_matrix<TA, 8, 8>(TA(0));
  for (uint it = 0; it < iters; ++it) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < FA; ++i)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < FB; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  float s = 0;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < FA; ++i)
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < FB; ++j) s += float(acc[i][j].thread_elements()[0]);
  if (s == 12345.0f) out[gid * 32 + lane] = s;
}
[[kernel]] void mma_peak_hf(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                            uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  mma_loop<half, float, 2, 4>(out, iters, gid, lane);
}
[[kernel]] void mma_peak_ff(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                            uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  mma_loop<float, float, 2, 4>(out, iters, gid, lane);
}
[[kernel]] void mma_peak_hh(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                            uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  mma_loop<half, half, 2, 4>(out, iters, gid, lane);
}
[[kernel]] void mma_peak_hf44(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                            uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  mma_loop<half, float, 4, 4>(out, iters, gid, lane);
}
[[kernel]] void mma_peak_hf11(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                            uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  mma_loop<half, float, 1, 1>(out, iters, gid, lane);
}
// plain FMA peak (float), 8 independent chains per thread
[[kernel]] void fma_peak(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                         uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float a[8];
  for (int i = 0; i < 8; ++i) a[i] = float(lane + i);
  const float m = 0.999f, c = 0.001f;
  for (uint it = 0; it < iters * 64; ++it) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 8; ++i) a[i] = fma(a[i], m, c);
  }
  float s = 0;
  for (int i = 0; i < 8; ++i) s += a[i];
  if (s == 12345.0f) out[gid * 32 + lane] = s;
}
)MSL";

int main(int argc, char** argv) {
  uint32_t iters = argc > 1 ? atoi(argv[1]) : 4096;
  uint32_t tgs = argc > 2 ? atoi(argv[2]) : 2048;
  uint32_t th = argc > 3 ? atoi(argv[3]) : 128;
  Gpu& g = Gpu::get();
  g.compile(kSrc);
  Buffer out = g.alloc(size_t(tgs) * 32 * 4 * 32);
  struct K { const char* name; double mmas_per_iter; double flop_per_mma; };
  for (K k : {K{"mma_peak_hf", 8, 1024}, K{"mma_peak_ff", 8, 1024}, K{"mma_peak_hh", 8, 1024},
              K{"mma_peak_hf44", 16, 1024}, K{"mma_peak_hf11", 1, 1024}, K{"fma_peak", 8 * 64, 64}}) {
    Pipeline p = g.pipeline(k.name);
    auto run = [&]() {
      auto e = g.begin();
      e->use(p);
      e->buf(0, out);
      e->val(1, iters);
      e->dispatch(tgs, 1, 1, th);
      e->commit_and_wait();
      return e->gpu_ms();
    };
    run();
    double best = 1e30;
    for (int r = 0; r < 3; ++r) best = std::min(best, run());
    double flop = double(tgs) * (th / 32) * iters * k.mmas_per_iter * k.flop_per_mma;
    printf("%-16s %8.3f ms  %6.2f TFLOP/s\n", k.name, best, flop / (best * 1e-3) / 1e12);
  }
}
