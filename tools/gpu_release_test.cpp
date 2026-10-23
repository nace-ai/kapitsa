#include "bench/bench.h"
#include "metal/gpu.h"

#include <cstdio>
#include <vector>

int main() {
  ni::bench::GpuLock lock;
  if (!lock.acquire()) {
    std::perror("GPU lock");
    return 1;
  }
  auto& gpu = ni::Gpu::get();
  gpu.release_all_buffers();
  const size_t baseline = gpu.allocated_bytes();
  for (int cycle = 0; cycle < 4; ++cycle) {
    std::vector<ni::Buffer> buffers;
    for (int i = 0; i < 3; ++i) buffers.push_back(gpu.alloc(size_t(32) << 20));
    gpu.make_resident(buffers);
    if (gpu.allocated_bytes() < baseline + (size_t(96) << 20)) {
      std::fprintf(stderr, "GPU allocations were not counted\n");
      return 1;
    }
    gpu.release(buffers.front());
    gpu.release_all_buffers();
    const size_t remaining = gpu.allocated_bytes();
    if (remaining > baseline + (size_t(1) << 20)) {
      std::fprintf(stderr, "Cycle %d retained %zu bytes above baseline\n", cycle, remaining - baseline);
      return 1;
    }
  }
  std::puts("GPU buffers and residency allocations released across four cycles");
}
