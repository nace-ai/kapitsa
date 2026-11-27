#include "bench/bench.h"
#include "model/qwen35.h"
#include "util/log.h"
#include <cstdio>
#include <exception>
#include <string>

struct Probe : ni::Qwen35 {
  Probe() : ni::Qwen35("", 1, NoLoad{}) {}
  void load(const std::string& dir) { load_mtp(dir); }
  ni::QLinear linear(const ni::WeightStore& weights) { return own_qlin(weights, "mtp.fc"); }
};
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  ni::bench::GpuLock lock;
  if (!lock.acquire({}, 1000)) return 75;
  ni::FatalRecovery recover;
  try {
    Probe p;
    if (std::string(argv[1]) == "head") p.load(argv[2]);
    else {
      ni::WeightStore weights;
      weights.open_dir(argv[2]);
      const auto q = p.linear(weights);
      std::printf("ACCEPT N=%u K=%u bits=%d\n", q.N, q.K, q.bits);
    }
  } catch (const std::exception& e) {
    std::printf("REJECT %s\n", e.what());
    return 1;
  }
}
