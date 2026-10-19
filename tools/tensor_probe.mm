// Probe: Metal 4 tensor ops (MetalPerformancePrimitives matmul2d, the M5's neural accelerators) against
// simdgroup_matrix MMAs on an 8-row x 64-column x 64-deep tile, operands in threadgroup memory (compute only).
//   tensor_probe [iters=256] [threadgroups=320]
// M5 Max (40 cores): 256 iterations x 1280 simdgroups: tensor 28.2 TFLOPS, simdgroup MMA 14.1 (the 8-row GEMMs'
// MMA side runs at the fp32 FMA rate); 1-4 iterations per simdgroup (a GEMM's per-call share): 7.4-17.8 vs
// 7.5-12.0 TFLOPS, i.e. matmul2d's per-call latency eats the gain at 8 rows. The 4-bit drafter GEMM built on it
// (bench_dfl BDFL_SRC4, uint4b codes staged per 64-group + fp32 scale / bias epilogue) ran 25-80% slower than
// dfl_qmm8 at 6144 x 2048 / 2048 x 4096 / 2048 x 6144.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Y[8][64] += X[8][64] . W[64][64]^T, `iters` times, per simdgroup (4 simdgroups per threadgroup, own tiles)
kernel void tp_tensor(device const half* X [[buffer(0)]], device const half* W [[buffer(1)]], device float* Y [[buffer(2)]],
                      constant uint& iters [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half xs[8 * 64];
  threadgroup half ws[1][64 * 64];
  for (uint i = tid; i < 8 * 64; i += 128) xs[i] = X[i];
  for (uint i = tid; i < 64 * 64; i += 128) ws[0][i] = W[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr auto desc = matmul2d_descriptor(8, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tX(xs, dextents<int32_t, 2>(64, 8));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(ws[0], dextents<int32_t, 2>(64, 64));
  auto c = op.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  for (uint16_t i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  for (uint it = 0; it < iters; ++it) op.run(tX, tW, c);
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y + (tg * 4 + sg) * 512, dextents<int32_t, 2>(64, 8));
  c.store(tY);
}

kernel void tp_mma(device const half* X [[buffer(0)]], device const half* W [[buffer(1)]], device float* Y [[buffer(2)]],
                   constant uint& iters [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half xs[8 * 64];
  threadgroup half ws[1][64 * 64];
  for (uint i = tid; i < 8 * 64; i += 128) xs[i] = X[i];
  for (uint i = tid; i < 64 * 64; i += 128) ws[0][i] = W[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 xa[8];
  for (int k = 0; k < 8; ++k) simdgroup_load(xa[k], xs + k * 8, 64);
  simdgroup_float8x8 acc[8];
  for (int n = 0; n < 8; ++n) acc[n] = simdgroup_float8x8(0);
  for (uint it = 0; it < iters; ++it)
    for (int n = 0; n < 8; ++n)
      for (int k = 0; k < 8; ++k) {
        simdgroup_half8x8 wb;
        simdgroup_load(wb, ws[0] + n * 8 * 64 + k * 8, 64, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc[n], xa[k], wb, acc[n]);
      }
  for (int n = 0; n < 8; ++n) simdgroup_store(acc[n], Y + (tg * 4 + sg) * 512 + n * 8, 64);
}
)MSL";

int main(int argc, char** argv) {
  @autoreleasepool {
    const uint32_t iters = argc > 1 ? uint32_t(atoi(argv[1])) : 256;
    const uint32_t ntg = argc > 2 ? uint32_t(atoi(argv[2])) : 320;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:o error:&err];
    if (!lib) {
      fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> X = [dev newBufferWithLength:8 * 64 * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> W = [dev newBufferWithLength:64 * 64 * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y0 = [dev newBufferWithLength:size_t(ntg) * 4 * 512 * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y1 = [dev newBufferWithLength:size_t(ntg) * 4 * 512 * 4 options:MTLResourceStorageModeShared];
    __fp16* x = (__fp16*)X.contents;
    __fp16* w = (__fp16*)W.contents;
    for (int i = 0; i < 8 * 64; ++i) x[i] = __fp16(((i * 37) % 17 - 8) / 8.0f);
    for (int i = 0; i < 64 * 64; ++i) w[i] = __fp16(((i * 53) % 13 - 6) / 16.0f);
    const char* names[2] = {"tp_tensor", "tp_mma"};
    id<MTLBuffer> outs[2] = {Y0, Y1};
    for (int rep = 0; rep < 3; ++rep)
      for (int k = 0; k < 2; ++k) {
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:names[k]]];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&err];
        if (!ps) {
          fprintf(stderr, "pipeline %s: %s\n", names[k], err.localizedDescription.UTF8String);
          return 1;
        }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setBuffer:X offset:0 atIndex:0];
        [e setBuffer:W offset:0 atIndex:1];
        [e setBuffer:outs[k] offset:0 atIndex:2];
        [e setBytes:&iters length:4 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
          fprintf(stderr, "%s: %s\n", names[k], cb.error.localizedDescription.UTF8String);
          return 1;
        }
        const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        const double flop = 2.0 * 8 * 64 * 64 * double(iters) * ntg * 4;
        if (rep == 2)
          printf("%-10s %8.3f ms  %8.1f GFLOP/s  (threads/tg max %lu)\n", names[k], ms, flop / (ms * 1e6),
                 (unsigned long)ps.maxTotalThreadsPerThreadgroup);
      }
    // compare the two outputs (and one CPU reference element)
    const float* a = (const float*)Y0.contents;
    const float* b = (const float*)Y1.contents;
    double md = 0;
    for (size_t i = 0; i < size_t(ntg) * 4 * 512; ++i) md = std::max(md, (double)std::fabs(a[i] - b[i]));
    double ref = 0;
    for (int k = 0; k < 64; ++k) ref += double(x[0 * 64 + k]) * double(w[0 * 64 + k]);
    printf("max |tensor - mma| %.3e; Y[0][0] tensor %.4f mma %.4f cpu %.4f\n", md, a[0], b[0], ref * iters);
  }
  return 0;
}
