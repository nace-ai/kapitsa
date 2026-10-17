// Thin Metal wrapper usable from plain C++ (no Objective-C in this header).
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ni {

// A GPU-visible buffer (shared storage, so `ptr` is CPU-addressable on Apple silicon).
// `offset` lets one MTLBuffer back many tensors (e.g. a whole mmapped safetensors file).
struct Buffer {
  void* mtl = nullptr;     // id<MTLBuffer> (retained by the Gpu's registry)
  uint8_t* base = nullptr; // CPU pointer to the MTLBuffer start
  size_t offset = 0;       // byte offset of this view inside the MTLBuffer
  size_t size = 0;         // byte size of this view
  void* ptr() const { return base + offset; }
  template <class T> T* as() const { return reinterpret_cast<T*>(base + offset); }
  Buffer view(size_t off, size_t bytes) const { return Buffer{mtl, base, offset + off, bytes}; }
  explicit operator bool() const { return mtl != nullptr; }
};

struct Pipeline {
  void* pso = nullptr; // id<MTLComputePipelineState>
  const char* name = "";
  uint32_t max_threads = 0;
  uint32_t simd_width = 32;
  uint32_t tg_mem = 0;  // static threadgroup memory (bytes)
};

class Encoder;

class Gpu {
 public:
  static Gpu& get();
  std::string name() const;
  size_t recommended_working_set() const;
  // GPU cores: the AGXAccelerator IORegistry entry's "gpu-core-count" (NI_GPU_CORES=<n> overrides: tests of
  // the core-scaled dispatch choices); without it a minimum per chip tier from the device name.
  int core_count() const;
  bool core_count_measured() const;  // read from the IORegistry (not the override or the name fallback)
  size_t max_threadgroup_memory() const;  // MTLDevice maxThreadgroupMemoryLength
  bool apple10() const;                   // MTLGPUFamilyApple10 (M5: Metal 4 neural accelerators)
  size_t allocated_bytes() const;  // MTLDevice currentAllocatedSize
  static const char* thermal_state();  // NSProcessInfo thermalState: nominal, fair, serious, critical

  // tracked=false skips Metal's hazard tracking: only for buffers the GPU never writes after
  // upload (weights). Everything the GPU writes stays tracked so consecutive command buffers
  // are ordered.
  Buffer alloc(size_t bytes, const char* label = nullptr, bool tracked = true);
  // Wrap existing page-aligned memory (e.g. mmap) without copying. tracked=true keeps Metal's
  // hazard tracking (memory the GPU writes, e.g. IOSurfaces shared with the Neural Engine).
  Buffer wrap(void* ptr, size_t bytes, const char* label = nullptr, bool tracked = false);
  // A new MTLSharedEvent (retained; id<MTLSharedEvent>), for ordering GPU work with the ANE.
  void* new_shared_event();
  void release(Buffer& b);

  // Compiles the embedded kernel library on first use. Extra source can be appended
  // (tests); `defines` become preprocessor macros.
  void compile(const std::string& extra_source = "");
  // Compiles `fn` for this GPU into a binary archive at `path` (machine code; no GPU work);
  // ni_compile reads code size, stack (spills) and GPR count from it.
  bool archive(const std::string& fn, const std::string& path);
  // required=false returns an empty Pipeline (pso null) instead of exiting when creation fails
  Pipeline pipeline(const std::string& fn, bool required = true);
  std::vector<std::string> function_names();  // kernels in the compiled library

  // Command buffer that holds one compute encoder. Commit is non-blocking.
  // concurrent=true lets independent dispatches overlap; call Encoder::barrier() between
  // dependent ones.
  std::unique_ptr<Encoder> begin(bool concurrent = false);
  // Keeps every resource resident (avoids per-dispatch residency checks).
  void make_resident(const std::vector<Buffer>& bufs);

  void* device() const { return device_; }
  void* queue() const { return queue_; }

 private:
  Gpu();
  void* device_ = nullptr;
  void* queue_ = nullptr;
  void* library_ = nullptr;
  void* residency_ = nullptr;
  struct Impl;
  Impl* impl_;
};

class Encoder {
 public:
  void use(const Pipeline& p);
  void buf(int idx, const Buffer& b, size_t extra_off = 0);
  void bytes(int idx, const void* data, size_t len);
  template <class T> void val(int idx, const T& v) { bytes(idx, &v, sizeof(T)); }
  void tgmem(int idx, size_t len);
  // threadgroups x/y/z, threads per threadgroup x/y/z
  void dispatch(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty = 1, uint32_t tz = 1);
  // Memory barrier between dependent dispatches in a concurrent encoder.
  void barrier();
  void commit();             // non-blocking
  void wait();               // waits for this command buffer (and the ones signal_event committed)
  // Shared-event steps (ANE split): signal_event ends the current command buffer with a signal of
  // `value` and commits it (so the signal is never held back behind later dispatches), then
  // continues in a new command buffer; if the signalling buffer fails, the CPU sets the event to
  // `value` so a waiter on it does not hang. wait_event makes the GPU wait until the event reaches
  // `value` before the work encoded after it. `event` is an id<MTLSharedEvent> from new_shared_event().
  void signal_event(void* event, uint64_t value);
  void wait_event(void* event, uint64_t value);
  void commit_and_wait() { commit(); wait(); }
  double gpu_ms() const;     // GPU time of the finished command buffer
  // GPU start / end of the finished command buffer on the host clock (mach time, ms), and that
  // clock now: for commit -> start and end -> wake-up latencies
  double gpu_start_host_ms() const;
  double gpu_end_host_ms() const;
  static double host_ms();
  ~Encoder();

  // Profiling (NI_PROFILE=1): every dispatch runs in its own command buffer so its GPU time can
  // be attributed to the kernel. Much slower; for analysis only.
  static bool profiling();
  static void profile_report(int steps);

 private:
  friend class Gpu;
  void open_encoder();
  void* cb_ = nullptr;
  void* enc_ = nullptr;
  std::vector<void*> prev_;   // command buffers committed by signal_event (retained)
  const char* cur_name_ = "";
  uint32_t cur_max_threads_ = 0;  // the pipeline's maxTotalThreadsPerThreadgroup (dispatch checks it)
  bool concurrent_ = false;
};

} // namespace ni
