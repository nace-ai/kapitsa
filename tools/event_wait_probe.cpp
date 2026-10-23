// Probe: how long a command buffer may wait on a shared event that the CPU sets, which is why a model split across
// Macs waits for the network on the CPU and not on the GPU (src/model/remote.h). The command buffer waits for the
// event, then runs one 16-byte copy; the CPU sets the event after `seconds`. A command buffer that waited too long ends
// with a GPU error, which is fatal here with the driver's reason. It takes the GPU lock (NI_LOCK_HELD=1 when the caller
// holds it): from about 5 s on, the command buffer sits on the GPU until the driver kills it, which would disturb a
// measurement running beside it.
//   event_wait_probe [seconds=2]
// M4 Pro, macOS 27.0.1: 1 s and 3 s complete; 6 s and 12 s are killed with a GPU timeout.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "util/log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace ni;

int main(int argc, char** argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 2;
  bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD") && !lock.acquire()) fatal("cannot acquire GPU lock");
  Gpu& g = Gpu::get();
  void* ev = g.new_shared_event();
  Buffer b = g.alloc(64, "probe");
  auto e = g.begin();
  e->wait_event(ev, 1);
  e->use(g.pipeline("copy_u32"));
  e->buf(0, b);
  e->buf(1, b);
  const uint32_t n = 1;
  e->val(2, n);
  e->dispatch(1, 1, 1, 32);
  e->commit();
  const auto t0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(int(secs * 1000)));
  Gpu::signal_event_cpu(ev, 1);
  e->wait();
  printf("the GPU waited %.1f s for the CPU: ok\n",
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return 0;
}
