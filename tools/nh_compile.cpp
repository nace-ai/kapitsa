// Compile check of the Nemotron-H kernels (src/kernels/nemotron): builds the library and every nh_* pipeline
// (no GPU work). nh_compile [kernel ...] limits it to those kernels.
#include "metal/gpu.h"
#include "util/log.h"

#include <cstdio>
#include <string>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

int main(int argc, char** argv) {
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) names.push_back(argv[i]);
  if (names.empty())  // every Nemotron-H kernel of the library
    for (const std::string& n : g.function_names())
      if (n.rfind("nh_", 0) == 0) names.push_back(n);
  int bad = 0;
  for (auto& n : names) {
    Pipeline p = g.pipeline(n, false);
    if (!p.pso) {
      ++bad;
      printf("%-28s FAILED\n", n.c_str());
      continue;
    }
    printf("%-28s max threads %4u, threadgroup memory %5u B\n", n.c_str(), p.max_threads, p.tg_mem);
  }
  printf("%d kernels failed\n", bad);
  return bad ? 1 : 0;
}
