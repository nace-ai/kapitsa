// Metal 4 tensor_ops matmul2d throughput microbenchmark (the M5's GPU matrix units), operands in threadgroup memory
// (shared by the threadgroup's 4 simdgroups), fp32 cooperative destination, one op per simdgroup (execution_simdgroup).
//   bench_tops [iters=2048] [threadgroups=1280] [kernel]
// M5 Max (40 cores): ~32 TFLOP/s when M or N is 8 (the unit's 16-row tile half used), 64-66 TFLOP/s at 16 x 64 x 64 and
// up, 51 at 64 x 64 x 64; uint8 x half ~30. bench_mma (simdgroup_matrix on the shader ALUs): 15.
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cstdio>
#include <string>

using namespace ni;

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <int M, int N, int K, typename TL, typename TR, bool TRR, typename TD = float>
inline void tp_loop(device float* out, uint iters, threadgroup TL* la, threadgroup TR* rb, uint lane) {
  for (uint i = lane; i < M * K; i += 32) la[i] = TL(float((i * 7) % 13) * 0.01f);
  for (uint i = lane; i < K * N; i += 32) rb[i] = TR(float((i * 5) % 11) * 0.01f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup TL, dextents<int32_t, 2>, tensor_inline>(la, dextents<int32_t, 2>(K, M));
  auto tB = TRR ? tensor<threadgroup TR, dextents<int32_t, 2>, tensor_inline>(rb, dextents<int32_t, 2>(K, N))
                : tensor<threadgroup TR, dextents<int32_t, 2>, tensor_inline>(rb, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(M, N, K, false, TRR, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), TD>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) cT[i] = 0;
  for (uint it = 0; it < iters; ++it) op.run(tA, tB, cT);
  TD s = 0;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) s += cT[i];
  if (s == TD(12345)) out[lane] = float(s);
}

#define TPD(NAME, M, N, K, TL, TR, TRR, TD)                                                            \
  [[kernel]] void NAME(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],            \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup TL la[M * K];                                                                          \
    threadgroup TR rb[K * N];                                                                          \
    tp_loop<M, N, K, TL, TR, TRR, TD>(out, iters, la, rb, lane);                                       \
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
TPD(tp_32_32_64_i8, 32, 32, 64, int8_t, int8_t, false, int32_t)
TPD(tp_64_64_64_i8, 64, 64, 64, int8_t, int8_t, false, int32_t)
TPD(tp_32_32_128_i8, 32, 32, 128, int8_t, int8_t, false, int32_t)
TPD(tp_64_32_64_i8, 64, 32, 64, int8_t, int8_t, false, int32_t)
TP(tp_32_32_64_hi8, 32, 32, 64, half, int8_t, false)
TPD(tp_32_32_64_bf, 32, 32, 64, bfloat, bfloat, false, float)

// int8 x int8 with a fresh int32 accumulator per K step (one weight group of 64), rescaled into fp32 per element:
// acc += float(ia) * sx[row] * sw[col]  (W8A8 per-group activation scales, group-64 weight scales)
template <int M, int N, int K>
inline void tp_rs(device float* out, uint iters, threadgroup int8_t* la, threadgroup int8_t* rb, threadgroup float* sc,
                  uint sg, uint lane) {
  if (sg == 0) {  // one simdgroup fills the shared operands and scales
    for (uint i = lane; i < M * K; i += 32) la[i] = int8_t((i * 7) % 13);
    for (uint i = lane; i < K * N; i += 32) rb[i] = int8_t((i * 5) % 11);
    for (uint i = lane; i < 2 * (M + N); i += 32) sc[i] = 0.001f * float(i % 17);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(la, dextents<int32_t, 2>(K, M));
  auto tB = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(rb, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(M, N, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
  constexpr int CAP = M * N / 32;  // the destination's elements per lane (checked below)
  if (cT.get_capacity() != CAP) { out[lane] = -1; return; }  // the host reads this back and drops the row
  float acc[CAP];
  short2 ix[CAP];
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < CAP; ++i) {
    acc[i] = 0;
    const auto m = cT.get_multidimensional_index(i);
    ix[i] = short2(m[0], m[1]);
  }
  for (uint it = 0; it < iters; ++it) {
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < CAP; ++i) cT[i] = 0;
    op.run(tA, tB, cT);
    const uint o = (it & 1) * (M + N);
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < CAP; ++i) acc[i] += float(cT[i]) * (sc[o + ix[i].y] * sc[o + M + ix[i].x]);
  }
  float s = 0;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < CAP; ++i) s += acc[i];
  if (s == 12345.0f) out[lane] = s;
}
#define TRS(NAME, M, N, K)                                                                             \
  [[kernel]] void NAME(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],            \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup int8_t la[M * K];                                                                      \
    threadgroup int8_t rb[K * N];                                                                      \
    threadgroup float sc[2 * (M + N)];  /* two scale sets (row then column), alternated per iteration */                                                                 \
    tp_rs<M, N, K>(out, iters, la, rb, sc, sg, lane);                                             \
  }
TRS(tp_rs_32_32_64, 32, 32, 64)
TRS(tp_rs_64_32_64, 64, 32, 64)
TRS(tp_rs_64_64_64, 64, 64, 64)
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
                      {"tp_8_64_64_u8", 8, 64, 64},  {"tp_8_64_64_u8t", 8, 64, 64},
                      {"tp_32_32_64_i8", 32, 32, 64}, {"tp_64_64_64_i8", 64, 64, 64},
                      {"tp_32_32_128_i8", 32, 32, 128}, {"tp_64_32_64_i8", 64, 32, 64},
                      {"tp_32_32_64_hi8", 32, 32, 64}, {"tp_32_32_64_bf", 32, 32, 64},
                      {"tp_rs_32_32_64", 32, 32, 64}, {"tp_rs_64_32_64", 64, 32, 64}, {"tp_rs_64_64_64", 64, 64, 64}};
  for (const S& s : shapes) {
    if (argc > 3 && std::string(argv[3]) != s.n) continue;
    Pipeline p = g.pipeline(s.n, false);
    if (!p.pso) {
      printf("%-18s (no pipeline)\n", s.n);
      continue;
    }
    std::fill_n(out.as<float>(), 32, 0.0f);
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
    if (std::count(out.as<float>(), out.as<float>() + 32, -1.0f)) {  // tp_rs: destination capacity != M*N/32, nothing ran
      printf("%-18s (destination capacity mismatch, kernel did no work)\n", s.n);
      continue;
    }
    double best = 1e9;
    for (int i = 0; i < 3; ++i) best = std::min(best, run());
    const double flop = 2.0 * s.m * s.nn * s.k * iters * ntg * 4;
    printf("%-18s m%3d n%3d k%3d  %8.3f ms  %7.2f TFLOP/s  (mt %u)\n", s.n, s.m, s.nn, s.k, best, flop / best / 1e9,
           p.max_threads);
    fflush(stdout);
  }
  return 0;
}
