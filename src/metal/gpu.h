// Thin Metal wrapper usable from plain C++ (no Objective-C in this header).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
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
  size_t max_buffer_length() const;       // MTLDevice maxBufferLength
  bool apple10() const;                   // MTLGPUFamilyApple10 (M5: Metal 4 neural accelerators)
  int apple_family() const;               // the newest MTLGPUFamilyApple<n>: 7 on the M1, 8 M2, 9 M3 and M4, 10 M5
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
  // Sets the event's value from the CPU (a GPU wait_event on it proceeds once it reaches the waited value)
  static void signal_event_cpu(void* event, uint64_t value);
  // The event's value as the CPU sees it now (MTLSharedEvent signaledValue; a few ns, for spinning on)
  static uint64_t event_value(void* event);
  void release(Buffer& b);
  // Keeps the GPU from going idle for about the next `ms`: a tiny ALU kernel (one 32-thread threadgroup) on a queue of
  // its own, at most two command buffers of about ms / 2 in flight. A GPU parked on an unresolved event wait resumes
  // ~150 us late on an M5 Pro, ~60 us while something else keeps it busy (tools/ni_handshake.mm, warm mode); the
  // heater takes no measurable time from other work. Cheap when enough is in flight (an atomic load); thread-safe.
  // ms <= 0: only prepares it (compiles the kernel and times it once: ~10 ms, at load rather than at a first token).
  // On by default on Apple10 (M5) only; NI_GPU_HEAT=0 off, =1 on (any chip); heat_on() says whether it is on.
  // heat_inflight(): its command buffers not yet completed (each is bounded, so they drain within ~ms of the last
  // call: nothing keeps running after a pass ends or fails).
  void keep_warm(double ms);
  static bool heat_on();
  static int heat_inflight();
  static uint64_t heat_started();  // its command buffers committed so far

  // Compiles the embedded kernel library on first use. Extra source can be appended (tests).
  void compile(const std::string& extra_source = "");
  // Compiles the weight-format kernels (src/kernels/wfmt.metal instantiations: bf16, 3/5/6-bit, ...) as a
  // second library; pipeline() does it on the first kernel the base library lacks.
  void compile_formats();
  // Compiles another embedded kernel source as a library of its own, once per key (the Nemotron-H kernels:
  // a model family's kernels, compiled only when such a model loads). pipeline() looks in these libraries
  // after the base library and before the weight-format one.
  // metal4: compiled as Metal 4 (tensor_ops; macOS 26) instead of Metal 3.2.
  void compile_extra(const std::string& key, const char* source, bool metal4 = false);
  // Compiles `fn` for this GPU into a binary archive at `path` (machine code; no GPU work);
  // ni_compile reads code size, stack (spills) and GPR count from it.
  bool archive(const std::string& fn, const std::string& path);
  // required=false returns an empty Pipeline (pso null) instead of exiting when creation fails
  Pipeline pipeline(const std::string& fn, bool required = true);
  // `p`'s kernel compiled for `threads` threads per threadgroup (MTLComputePipelineDescriptor's
  // maxTotalThreadsPerThreadgroup: the compiler fits them, spilling registers if it must). For a GPU whose compiler
  // gives the kernel fewer by default than it is dispatched with (Encoder::dispatch asks); cached.
  Pipeline pipeline_threads(const Pipeline& p, uint32_t threads);
  // kernels in the library and the extra libraries (+ the format kernels)
  std::vector<std::string> function_names(bool formats = false);

  // Command buffer that holds one compute encoder. Commit is non-blocking.
  // concurrent=true lets independent dispatches overlap; call Encoder::barrier() between
  // dependent ones.
  std::unique_ptr<Encoder> begin(bool concurrent = false);
  // A concurrent encoder that keeps serial order on its own: a barrier before every dispatch, except inside a
  // group (Encoder::group_begin / group_end), whose dispatches may run together.
  std::unique_ptr<Encoder> begin_grouped();
  // Keeps these buffers resident (wired) between command buffers too: they join the residency set, which a heartbeat
  // keeps requested while the GPU is in use and memory is not under pressure (NI_RESIDENCY; gpu.mm).
  void make_resident(const std::vector<Buffer>& bufs);

  void* device() const { return device_; }
  void* queue() const { return queue_; }

  struct Impl;

 private:
  Gpu();
  void* device_ = nullptr;
  void* queue_ = nullptr;
  void* library_ = nullptr;
  void* residency_ = nullptr;
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
  // Threadgroup counts read by the GPU from `args` (3 x uint32 at byte offset `off`), e.g. written by
  // an earlier dispatch of the same command buffer.
  void dispatch_indirect(const Buffer& args, size_t off, uint32_t tx, uint32_t ty = 1, uint32_t tz = 1);
  // Memory barrier between dependent dispatches in a concurrent encoder.
  void barrier();
  // With begin_grouped(): the dispatches between group_begin() and group_end() are independent of each other (none
  // reads what another writes) and may overlap on the GPU. No effect on other encoders.
  void group_begin() { in_group_ = true; group_first_ = true; }
  void group_end() { in_group_ = false; }
  bool grouped() const { return auto_barrier_; }
  void commit();             // non-blocking
  // `f` runs (on a Metal thread) once the GPU has finished this command buffer; set before commit(). For progress
  // displays: no wait, nothing on the GPU's path.
  void on_completed(std::function<void()> f);
  void wait();               // waits for this command buffer (and the ones signal_event committed)
  // Waits at most timeout_ms for this command buffer to finish (polling); false if it is still running.
  bool wait_for(double timeout_ms);
  // Shared-event steps (ANE split): signal_event ends the current command buffer with a signal of
  // `value` and commits it (so the signal is never held back behind later dispatches), then
  // continues in a new command buffer; if the signalling buffer fails, the CPU sets the event to
  // `value` so a waiter on it does not hang. wait_event makes the GPU wait until the event reaches
  // `value` before the work encoded after it. `event` is an id<MTLSharedEvent> from new_shared_event().
  void signal_event(void* event, uint64_t value);
  void wait_event(void* event, uint64_t value);
  // A signal of `value` once the work encoded so far has completed, inside this command buffer (no commit): for a
  // handshake with a CPU thread within one pass (the CPU spins on event_value, then lets a later wait_event go on)
  void signal_event_inline(void* event, uint64_t value);
  // Commits the work so far and continues in a new command buffer (same queue, so in order): long passes are split
  // into short command buffers, which keeps the GPU responsive to the desktop's own work.
  void split();
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
  // NI_PROFILE: a prefix for the names of the dispatches recorded from here on (one phase's kernels apart from
  // another's, e.g. a drafting pass's "d:"); nullptr or "" for none
  static void profile_tag(const char* tag);

 private:
  friend class Gpu;
  void open_encoder();
  void* cb_ = nullptr;
  void* enc_ = nullptr;
  std::vector<void*> prev_;   // command buffers committed by signal_event / split (retained)
  const char* cur_name_ = "";
  void* cur_pso_ = nullptr;
  uint32_t cur_max_threads_ = 0;  // the pipeline's maxTotalThreadsPerThreadgroup (dispatch checks it)
  bool concurrent_ = false;
  bool auto_barrier_ = false, in_group_ = false, group_first_ = false, dispatched_ = false;
  void pre_dispatch();
  void fit_threads(uint32_t threads);
};

} // namespace ni
