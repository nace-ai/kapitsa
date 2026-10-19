// Metal 4 tensor_ops matmul2d throughput microbenchmark (the M5's GPU matrix units), operands in threadgroup memory
// (shared by the threadgroup's 4 simdgroups), fp32 cooperative destination, one op per simdgroup (execution_simdgroup).
//   bench_tops [iters=2048] [threadgroups=1280] [kernel]
// M5 Max (40 cores): ~32 TFLOP/s when M or N is 8 (the unit's 16-row tile half used), 64-66 TFLOP/s at 16 x 64 x 64 and
// up, 51 at 64 x 64 x 64; uint8 x half ~30. bench_mma (simdgroup_matrix on the shader ALUs): 15.
#include "metal/gpu.h"
#include "util/log.h"

#include <cstdio>
#include <string>

using namespace ni;

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <int M, int N, int K, typename TL, typename TR, bool TRR>
inline void tp_loop(device float* out, uint iters, threadgroup TL* la, threadgroup TR* rb, uint lane) {
  for (uint i = lane; i < M * K; i += 32) la[i] = TL(float((i * 7) % 13) * 0.01f);
  for (uint i = lane; i < K * N; i += 32) rb[i] = TR(float((i * 5) % 11) * 0.01f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup TL, dextents<int32_t, 2>, tensor_inline>(la, dextents<int32_t, 2>(K, M));
  auto tB = TRR ? tensor<threadgroup TR, dextents<int32_t, 2>, tensor_inline>(rb, dextents<int32_t, 2>(K, N))
                : tensor<threadgroup TR, dextents<int32_t, 2>, tensor_inline>(rb, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(M, N, K, false, TRR, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) cT[i] = 0;
  for (uint it = 0; it < iters; ++it) op.run(tA, tB, cT);
  float s = 0;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) s += cT[i];
  if (s == 12345.0f) out[lane] = s;
}

#define TP(NAME, M, N, K, TL, TR, TRR)                                                                 \
  [[kernel]] void NAME(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],            \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup TL la[M * K];                                                                          \
    threadgroup TR rb[K * N];                                                                          \
    tp_loop<M, N, K, TL, TR, TRR>(out, iters, la, rb, lane);                                           \
  }

TP(tp_8_64_64, 8, 64, 64, half, half, false)
TP(tp_64_8_64, 64, 8, 64, half, half, false)
TP(tp_8_32_64, 8, 32, 64, half, half, false)
TP(tp_32_8_64, 32, 8, 64, half, half, false)
TP(tp_16_64_64, 16, 64, 64, half, half, false)
TP(tp_64_16_64, 64, 16, 64, half, half, false)
TP(tp_32_32_64, 32, 32, 64, half, half, false)
TP(tp_64_32_64, 64, 32, 64, half, half, false)
TP(tp_16_16_64, 16, 16, 64, half, half, false)
TP(tp_8_64_64t, 8, 64, 64, half, half, true)
TP(tp_64_64_64, 64, 64, 64, half, half, false)
TP(tp_8_64_32, 8, 64, 32, half, half, false)
TP(tp_8_64_64_u8, 8, 64, 64, half, uint8_t, false)
TP(tp_8_64_64_u8t, 8, 64, 64, half, uint8_t, true)
)MSL";

int main(int argc, char** argv) {
  const uint32_t iters = argc > 1 ? atoi(argv[1]) : 2048, ntg = argc > 2 ? atoi(argv[2]) : 1280;
  Gpu& g = Gpu::get();
  g.compile_extra("tops", kSrc, true);
  Buffer out = g.alloc(4096);
  struct S {
    const char* n;
    int m, nn, k;
  };
  const S shapes[] = {{"tp_8_64_64", 8, 64, 64},     {"tp_64_8_64", 64, 8, 64},     {"tp_8_32_64", 8, 32, 64},
                      {"tp_32_8_64", 32, 8, 64},     {"tp_16_64_64", 16, 64, 64},   {"tp_64_16_64", 64, 16, 64},
                      {"tp_32_32_64", 32, 32, 64},   {"tp_64_32_64", 64, 32, 64},   {"tp_16_16_64", 16, 16, 64},
                      {"tp_8_64_64t", 8, 64, 64},    {"tp_64_64_64", 64, 64, 64},
                      {"tp_8_64_32", 8, 64, 32},
                      {"tp_8_64_64_u8", 8, 64, 64},  {"tp_8_64_64_u8t", 8, 64, 64}};
  for (const S& s : shapes) {
    if (argc > 3 && std::string(argv[3]) != s.n) continue;
    Pipeline p = g.pipeline(s.n, false);
    if (!p.pso) {
      printf("%-18s (no pipeline)\n", s.n);
      continue;
    }
    auto run = [&]() {
      auto e = g.begin();
      e->use(p);
      e->buf(0, out);
      e->val(1, iters);
      e->dispatch(ntg, 1, 1, 128);
      e->commit_and_wait();
      return e->gpu_ms();
    };
    run();
    double best = 1e9;
    for (int i = 0; i < 3; ++i) best = std::min(best, run());
    const double flop = 2.0 * s.m * s.nn * s.k * iters * ntg * 4;
    printf("%-18s m%3d n%3d k%3d  %8.3f ms  %7.2f TFLOP/s  (mt %u)\n", s.n, s.m, s.nn, s.k, best, flop / best / 1e9,
           p.max_threads);
    fflush(stdout);
  }
  return 0;
}
