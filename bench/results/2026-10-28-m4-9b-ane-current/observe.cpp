#include "bench/bench.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
  const int count = argc > 1 ? std::atoi(argv[1]) : 1;
  const int interval_ms = argc > 2 ? std::atoi(argv[2]) : 1000;
  for (int i = 0; count == 0 || i < count; ++i) {
    using namespace ni::bench;
    const auto v = vm();
    const auto p = power(false);
    ojson clients = ojson::array();
    for (const auto& [client, ns] : gpu_snapshot())
      clients.push_back({{"pid", client.first}, {"name", client.second}, {"ns", ns}});
    const double epoch = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const double mono = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const ojson row = {{"epoch", epoch}, {"mono", mono}, {"thermal", thermal_state()},
      {"power", p.source}, {"low_power", p.low_power_mode}, {"swap_mb", v.swap_used_mb},
      {"swapouts", v.swapouts}, {"swapins", v.swapins}, {"memory_level", v.memorystatus_level},
      {"gpu_memory_bytes", gpu_memory_in_use()}, {"clients", clients}};
    std::puts(row.dump().c_str());
    std::fflush(stdout);
    if (count == 0 || i + 1 < count) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
  }
}
