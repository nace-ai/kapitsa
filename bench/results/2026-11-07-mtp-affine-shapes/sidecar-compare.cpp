#include "bench/bench.h"
#include "model/qwen35.h"
#include "util/log.h"
#include <fstream>
#include <cstdio>
struct Probe : ni::Qwen35 {
  Probe() : ni::Qwen35("", 1, NoLoad{}) {}
  ni::QLinear linear(const ni::WeightStore& w, const std::string& n) { return own_qlin(w, n); }
};
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  ni::bench::GpuLock lock;
  if (!lock.acquire({}, 1000)) return 75;
  ni::FatalRecovery recover;
  try {
    ni::Gpu::AllocationScope allocations;
    ni::WeightStore weights;
    weights.open_dir(argv[1], {}, [](const std::string& f) { return f == "mtp.safetensors"; });
    Probe p;
    std::ofstream out(argv[2], std::ios::binary);
    size_t matrices = 0, bytes = 0;
    for (const auto& [name,t] : weights.all()) {
      if (!name.ends_with(".scales")) continue;
      const auto q = p.linear(weights, name.substr(0, name.size()-7));
      for (const auto* v : {q.w,q.s,q.b}) {
        const size_t size = size_t(v->numel()) * ni::dtype_size(v->dtype);
        out.write(reinterpret_cast<const char*>(v->buf.ptr()), size);bytes += size;
      }
      ++matrices;
      std::printf("%s N=%u K=%u bits=%d\n", name.c_str(), q.N, q.K, q.bits);
    }
    std::printf("matrices=%zu bytes=%zu\n",matrices,bytes);
    return matrices && out ? 0 : 1;
  } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
