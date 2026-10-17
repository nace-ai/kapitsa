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
  if (names.empty()) {
    const std::string src = kNhKernelSource;
    for (size_t p = 0; (p = src.find("void nh_", p)) != std::string::npos;) {
      const size_t e = src.find('(', p);
      const std::string n = src.substr(p + 5, e - p - 5);
      if (n.find('#') == std::string::npos &&  // a macro's name pattern, not a kernel
          (src.compare(p - 11, 10, "[[kernel]]") == 0 || src.rfind("max_total_threads_per_threadgroup", p) + 64 > p))
        names.push_back(n);
      p = e;
    }
    // macro-instantiated kernels
    for (const char* f : {"q8", "fp4", "iq2", "f32", "bf16"})
      for (const char* r : {"1", "2", "4", "8"})
        for (const char* e : {"", "_add", "_relu2"}) names.push_back(std::string("nh_mv_") + f + "_r" + r + e);
    for (const char* k : {"nh_gemm_q8", "nh_gemm_fp4", "nh_gemm_iq2", "nh_gemm_iq2_moe_up", "nh_gemm_iq2_moe_dn"})
      for (const char* m : {"_m16", "_m32", "_m64"}) names.push_back(std::string(k) + m);
    for (const char* k : {"nh_moe2_up_iq2_", "nh_moe2_dn_iq2_"})
      for (const char* v : {"h", "f", "a"}) names.push_back(std::string(k) + v);
    names.push_back("nh_gemm_f32");
    names.push_back("nh_gemm_bf16");
  }
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
