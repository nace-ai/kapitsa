// Compiles the embedded Metal kernel library (and optionally creates pipelines for the given
// kernel names) without running anything on the GPU. Exit code 0 = all good.
//   ni_kstats [--stats] [kernel ...]
// --stats prints each pipeline's temporary register count and stack spill bytes per thread, read
// from the AGX pipeline object's private state (verified on macOS 15.3 / M4 Pro; prints "n/a" when
// the layout differs). With --stats and no names, every kernel in the library is listed. Spill > 0
// in a GEMM or attention kernel usually means a register array indexed dynamically (a loop the
// compiler left rolled): qmm_b4_m32_n64 spills its 8 accumulators (2304 B) and runs at 2-3 TFLOP/s.
#include "metal/gpu.h"

#include <objc/runtime.h>

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
  bool stats = false;
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--stats")) stats = true;
    else names.push_back(argv[i]);
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
