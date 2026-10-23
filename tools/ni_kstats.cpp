// Compiles the embedded Metal kernel library (and optionally creates pipelines for the given
// kernel names) without running anything on the GPU. Exit code 0 = all good.
//   ni_kstats [--stats | --limits] [kernel ...]
// --stats prints each pipeline's temporary register count and stack spill bytes per thread, read
// from the AGX pipeline object's private state (verified on macOS 15.3 / M4 Pro; prints "n/a" when
// the layout differs). With --stats and no names, every kernel in the library is listed. Spill > 0
// in a GEMM or attention kernel usually means a register array indexed dynamically (a loop the
// compiler left rolled): qmm_b4_m32_n64 spills its 8 accumulators (2304 B) and runs at 2-3 TFLOP/s.
#include "metal/gpu.h"

#include <objc/runtime.h>

namespace ni { extern const char* const kTensorKernelSource; }

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// regs/spill sit at a fixed offset of the AGXG16X pipeline's C++ state; the ivar type encoding up to
// those fields must match the one the offset was derived from (length and FNV-1a hash).
static bool agx_stats(void* pso, uint32_t* regs, uint32_t* spill) {
  Class c = object_getClass((id)pso);
  Ivar iv = c ? class_getInstanceVariable(c, "_impl") : nullptr;
  const char* enc = iv ? ivar_getTypeEncoding(iv) : nullptr;
  const char* f = enc ? strstr(enc, "\"max_temporary_register_count\"I\"total_spill_buffer_bytes\"I") : nullptr;
  if (!f || f - enc != 1291) return false;
  uint32_t h = 2166136261u;
  for (const char* p = enc; p < f; ++p) h = (h ^ uint8_t(*p)) * 16777619u;
  if (h != 0x151954aeu) return false;
  const uint8_t* base = (const uint8_t*)pso + ivar_getOffset(iv);
  memcpy(regs, base + 176, 4);
  memcpy(spill, base + 180, 4);
  return *regs > 0 && *regs <= 256;
}

int main(int argc, char** argv) {
  ni::Gpu& g = ni::Gpu::get();
  g.compile();
  // The tensor_ops kernels (qmt_*, qattn_t_*: kernels/tensor/) are a Metal 4 library of their own, compiled where
  // the models use them (an Apple10 GPU on macOS 26); without it their pipelines fail to create.
  if (g.apple10())
    if (__builtin_available(macOS 26.0, *)) g.compile_extra("tensor", ni::kTensorKernelSource, true);
  bool stats = false, limits = false;
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--stats")) stats = true;
    else if (!strcmp(argv[i], "--limits")) limits = true;
    else names.push_back(argv[i]);
  }
  if (limits) {
    // --limits: every kernel's static threadgroup memory against 32 KB (Apple7-Apple10, the Metal feature set
    // tables: M5 is Apple10 with the same 32 KB) and this device's limit, and pipelines whose compiled maximum
    // threads per threadgroup is below 1024 (a dispatch above it fails; Encoder::dispatch checks)
    const size_t dev = g.max_threadgroup_memory();
    printf("%s: %d GPU cores (%s), max threadgroup memory %zu B, Apple10 %s\n", g.name().c_str(), g.core_count(),
           g.core_count_measured() ? "IORegistry" : "not measured", dev, g.apple10() ? "yes" : "no");
    int over = 0;
    for (auto& n : names.empty() ? g.function_names() : names) {
      ni::Pipeline p = g.pipeline(n, false);
      if (!p.pso) {
        printf("%-36s pipeline creation FAILED\n", n.c_str());
        ++over;
        continue;
      }
      // named kernels always print; the full list only those over a limit
      if (!names.empty() || p.tg_mem > 32768 || p.tg_mem > dev || p.max_threads < 1024)
        printf("%-36s threadgroup memory %6u B  max threads %4u%s\n", n.c_str(), p.tg_mem, p.max_threads,
               p.tg_mem > 32768 ? "  <-- over 32 KB" : "");
      over += p.tg_mem > 32768;
    }
    printf("%d kernel(s) over 32 KB of threadgroup memory\n", over);
    return over ? 1 : 0;
  }
  if (stats && names.empty()) names = g.function_names();
  int failed = 0;
  for (auto& n : names) {
    ni::Pipeline p = g.pipeline(n, !stats);
    uint32_t regs = 0, spill = 0;
    if (!p.pso) ++failed;
    else if (!stats) printf("%-36s ok (max threads %u)\n", n.c_str(), p.max_threads);
    else if (agx_stats(p.pso, &regs, &spill))
      printf("%-36s max threads %4u  regs %3u  spill %6u B%s\n", n.c_str(), p.max_threads, regs, spill,
             spill ? "  <-- spills" : "");
    else printf("%-36s max threads %4u  regs n/a\n", n.c_str(), p.max_threads);
  }
  printf("kernels compiled%s\n", failed ? " (some pipelines failed)" : "");
  return failed ? 1 : 0;
}
