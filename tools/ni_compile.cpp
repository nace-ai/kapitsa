// Compiles the embedded Metal kernel library (and optionally creates pipelines for the given
// kernel names) without running anything on the GPU. Exit code 0 = all good.
//   ni_compile [kernel ...]
// NI_ARCHIVE_DIR=<dir>: also compiles each kernel to machine code (<dir>/<kernel>.bin, a Metal
// binary archive) and prints its code size, per-thread stack (private memory) and GPR count.
// Stack > 0 means arrays or spilled registers live in memory: v1-v10 of the prefill attention
// kept 6-13 KB of fragments there. The two numbers come from the undocumented compiler metadata
// (__GPU_METADATA) found by probing kernels with known register use; "?" if the layout differs.
#include "gpu_archive.h"
#include "metal/gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using Bytes = std::vector<uint8_t>;

int main(int argc, char** argv) {
  ni::Gpu& g = ni::Gpu::get();
  g.compile();
  const char* adir = getenv("NI_ARCHIVE_DIR");
  // --formats: also the weight-format library (wfmt.metal); --match <s>: every kernel whose name contains s
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--formats")) {
      g.compile_formats();
    } else if (!strcmp(argv[i], "--match") && i + 1 < argc) {
      const std::string sub = argv[++i];
      for (const std::string& n : g.function_names(true))
        if (n.find(sub) != std::string::npos) names.push_back(n);
    } else {
      names.push_back(argv[i]);
    }
  }
  int failed = 0;
  for (const std::string& name : names) {
    const char* nm = name.c_str();
    ni::Pipeline p = g.pipeline(name, /*required=*/false);
    if (!p.pso) {  // e.g. an experimental kernel past the threadgroup memory limit: report it, go on
      printf("%-36s FAILED\n", nm);
      ++failed;
      continue;
    }
    printf("%-36s ok (max threads %u, threadgroup memory %u B)", nm, p.max_threads, p.tg_mem);
    if (adir) {
      std::string path = std::string(adir) + "/" + name + ".bin";
      std::ifstream f;
      if (g.archive(name, path) && (f.open(path, std::ios::binary), f)) {
        Bytes b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        long text, stack, gprs;
        kernel_stats(b, &text, &stack, &gprs);
        if (stack >= 0) printf("  code %6ld B  stack %5ld B  gprs %3ld", text, stack, gprs);
        else printf("  code %6ld B  stack ? gprs ?", text);
      } else {
        printf("  (archive failed)");
      }
    }
    printf("\n");
  }
  printf("kernels compiled%s\n", failed ? " (some pipelines failed)" : "");
  return failed ? 1 : 0;
}
