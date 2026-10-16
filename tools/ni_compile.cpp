// Compiles the embedded Metal kernel library (and optionally creates pipelines for the given
// kernel names) without running anything on the GPU. Exit code 0 = all good.
//   ni_compile [kernel ...]
#include "metal/gpu.h"
#include <cstdio>
int main(int argc, char** argv) {
  ni::Gpu& g = ni::Gpu::get();
  g.compile();
  for (int i = 1; i < argc; ++i) {
    ni::Pipeline p = g.pipeline(argv[i]);
    printf("%-36s ok (max threads %u)\n", argv[i], p.max_threads);
  }
  printf("kernels compiled\n");
  return 0;
}
