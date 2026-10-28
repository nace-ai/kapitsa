// CPU by default. --gpu checks real Metal failure/retry and allocation ownership,
// under the GPU lock; never run that mode automatically in CTest.
#include "util/load_threads.h"
#include "loader/safetensors.h"
#include "model/expert_store.h"
#include "bench/bench.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <sys/wait.h>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
using namespace ni;
template<class F> void fails(F f, const std::string& text) {
  bool caught = false;
  try { f(); }
  catch (const FatalError& e) { caught = std::string(e.what()).find(text) != std::string::npos; }
  CHECK(caught);
}
static int descriptors() {
  int n = 0;
  for (int fd = 0; fd < 1024; ++fd) n += fcntl(fd, F_GETFD) != -1;
  return n;
}
static void child_fails(const char* self, const char* mode, const char* path = "") {
  int pipefd[2]; CHECK(pipe(pipefd) == 0);
  const pid_t pid = fork(); CHECK(pid >= 0);
  if (!pid) {
    close(pipefd[0]); dup2(pipefd[1], STDERR_FILENO); close(pipefd[1]);
    unsetenv("NI_ABORT"); execl(self, self, mode, path, nullptr); _exit(127);
  }
  close(pipefd[1]);
  std::string text; char buf[512]; ssize_t n;
  while ((n = read(pipefd[0], buf, sizeof buf)) > 0) text.append(buf, size_t(n));
  close(pipefd[0]); int status = 0; CHECK(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
  CHECK(text.find("[kapitsa] fatal: ") != std::string::npos);
}
static void cpu(const char* self) {
  CHECK(!FatalRecovery::enabled());
  {
    FatalRecovery recover;
    const std::string long_message(4096, 'x');
    fails([&] { fatal("%s %d", long_message.c_str(), 42); }, long_message + " 42");
    std::atomic<int> finished{0};
    fails([&] {
      LoadThreads jobs;
      jobs.emplace_back([] { fatal("worker read failed"); });
      jobs.emplace_back([&] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); ++finished; });
      jobs.join();
    }, "worker read failed");
    CHECK(finished == 1);
    try {
      LoadThreads jobs;
      jobs.emplace_back([&] { ++finished; });
      throw std::bad_alloc();  // unwind while an already-started job still owns its references
    } catch (const std::bad_alloc&) {}
    CHECK(finished == 2);
  }
  CHECK(!FatalRecovery::enabled());
  child_fails(self, "--fatal");
  child_fails(self, "--unscoped-thread");
  char pattern[] = "/tmp/kapitsa-load-errors-XXXXXX";
  CHECK(mkdtemp(pattern)); const std::filesystem::path dir(pattern);
  // One header read successfully, then one broken shard: both descriptors must close.
  { std::ofstream f(dir / "a.safetensors", std::ios::binary); uint64_t n = 2; f.write(reinterpret_cast<char*>(&n), 8); f << "{}"; }
  std::ofstream(dir / "b.safetensors", std::ios::binary) << "broken";
  const int before = descriptors();
  {
    FatalRecovery recover;
    for (int i = 0; i < 32; ++i) {
      fails([&] { SafetensorsIndex index(dir); }, "incomplete");
      fails([&] { WeightStore weights; weights.open_dir(dir); }, "incomplete");
      const auto lay = q20::slot_layout(128, 128);
      q20::LayerSrc layer;
      for (int j = 0; j < 6; ++j) layer.p[j] = {0, 0, 0, j % 2 ? lay.ud - lay.gd : lay.uc};
      q20::ExpertCache::Memory mem; mem.per_chunk = 1;
      fails([&] {
        q20::ExpertCache cache({(dir / "a.safetensors").string(), (dir / "missing").string()},
                              {layer}, 1, lay, 1, mem, {}, {});
      }, "cannot open");
    }
  }
  CHECK(descriptors() == before);
  child_fails(self, "--bad-file", pattern);
  std::filesystem::remove_all(dir);
  std::cout << "PASS scoped errors, thread joining/isolation, loader FD cleanup, CLI exit 1\n";
}
static void gpu() {
  bench::GpuLock lock;
  CHECK(bench::GpuLock::held_by_caller() || lock.acquire());
  FatalRecovery recover;
  auto& g = Gpu::get();
  fails([&] { g.alloc(g.max_buffer_length() + 1); }, "maximum buffer length");
  fails([&] { g.compile_extra("load-error-test", "this is not Metal"); }, "Metal compile failed");
  g.compile_extra("load-error-test", R"(#include <metal_stdlib>
    kernel void load_error_write(device int* out [[buffer(0)]]) { out[0] = 17; })");
  fails([&] { g.pipeline("load_error_missing_kernel"); }, "not found");
  auto pipeline = g.pipeline("load_error_write");
  auto sentinel = g.alloc(16384);
  // The validation layer lazily allocates its own dispatch instrumentation.
  { auto e = g.begin(); e->use(pipeline); e->buf(0, sentinel); e->dispatch(1, 1, 1, 1); e->commit_and_wait(); }
  sentinel.as<int>()[0] = 123;
  const size_t before = g.allocated_bytes();
  bool freed = false, freed_early = false;
  {
    Gpu::AllocationScope outer;
    auto keep = g.alloc(16384);
    keep.as<int>()[0] = 456;
    fails([&] {
      Gpu::AllocationScope failed_load;
      auto early = g.alloc(16384); g.release(early);
      void* memory = nullptr; CHECK(posix_memalign(&memory, size_t(getpagesize()), 16384) == 0);
      auto owner = std::shared_ptr<void>(memory, [&](void* p) {
        freed = static_cast<int*>(p)[0] == 17; std::free(p);
      });
      auto b = g.wrap(memory, 16384, "load error backing", true, std::move(owner));
      struct ReleaseOnUnwind {
        Gpu& g; Buffer b; bool& freed; bool& early;
        ~ReleaseOnUnwind() { g.release(b); early = freed; }
      } release{g, b, freed, freed_early};
      auto e = g.begin(); e->use(pipeline); e->buf(0, b); e->dispatch(1, 1, 1, 1); e->commit();
      fatal("failure after allocation and GPU submission");
    }, "after allocation");
    CHECK(freed && !freed_early && keep.as<int>()[0] == 456);
  }
  CHECK(sentinel.as<int>()[0] == 123);
  const size_t after = g.allocated_bytes();
  std::cout << "Metal allocated bytes before/after rollback: " << before << '/' << after << '\n';
  CHECK(after == before);
  g.release(sentinel);
  std::cout << "PASS real Metal compile/pipeline retry and buffer rollback after GPU work\n";
}
int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--fatal") fatal("outside SDK recovery");
  if (argc > 1 && std::string(argv[1]) == "--unscoped-thread") {
    FatalRecovery recover;
    std::thread t([] { fatal("unrelated thread remains fatal"); }); t.join();
  }
  if (argc > 2 && std::string(argv[1]) == "--bad-file") { WeightStore ws; ws.open_dir(argv[2]); }
  try {
    if (argc > 1 && std::string(argv[1]) == "--gpu") gpu();
    else cpu(argv[0]);
  } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
