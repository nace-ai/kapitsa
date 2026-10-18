// Client of Apple's private AppleNeuralEngine.framework (the Neural Engine, "ANE"), usable from
// plain C++. Used by the opt-in GPU + ANE split of the dense prefill FFN (NI_ANE=1, ffn_split.h).
//
// How it works: the _ANEClient/_ANEModel/_ANERequest selectors, each checked against its type encoding
// before use; MIL text programs whose weights are IOSurface-backed tensor_buffer inputs; IOSurfaces shared
// zero-copy with Metal; evaluations ordered with the GPU through one MTLSharedEvent, with a
// watchdog that releases the GPU if the ANE stops answering.
#pragma once
#include "metal/gpu.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ni::ane {

// Empty when the private client resolves with the expected selectors and type encodings
// (checked once per process); otherwise the reason it cannot be used.
const std::string& unavailable();

// An IOSurface of `rows` x `width` int8 or fp16 elements (rows padded to 64 bytes, size rounded
// to 16 KB pages), wrapped zero-copy as a hazard-tracked Metal buffer.
struct Surface {
  void* ref = nullptr;  // IOSurfaceRef (owned)
  Buffer buf;
  uint32_t rows = 0, width = 0, elem = 1, stride = 0;  // stride in bytes
  size_t bytes = 0;
  bool fp16 = false;
  uint8_t* ptr() const { return buf.as<uint8_t>(); }
  std::string mil_type() const { return fp16 ? "fp16" : "int8"; }
  // "[plane, plane, stride, 1]" in elements, for a tensor of r rows
  std::string strides(uint32_t r) const;
  // MIL tensor_buffer type of the leading w elements of the first r rows (same strides)
  std::string buffer_type(uint32_t r, uint32_t w) const;
};
Surface make_surface(uint32_t rows, uint32_t width, bool fp16);  // throws std::runtime_error
// The same rows/width/stride/bytes as make_surface, with no IOSurface or Metal buffer: enough to
// generate MIL that addresses such a surface (FfnSplit compile-only mode).
Surface surface_layout(uint32_t rows, uint32_t width, bool fp16);
void free_surface(Surface& s);

// MIL blob storage v2 holding one fp16 constant at offset 64 (data at 128).
std::vector<uint8_t> fp16_blob(const _Float16* v, size_t n);
uint64_t fnv1a(const void* p, size_t n, uint64_t h = 14695981039346656037ULL);

// A compiled and loaded MIL program (one or more functions = procedures).
class Program {
 public:
  // Writes model.mil/weights.bin under $(DARWIN_USER_CACHE_DIR)/<tag>/<fnv>/, compiles unless the
  // ANE daemon still holds a compilation under that key, loads. Throws std::runtime_error.
  Program(const std::string& mil, const std::vector<uint8_t>& blob, const std::string& tag,
          double limit_s = 180.0);
  ~Program();
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  int proc(const std::string& function) const;  // -1 if absent
  const std::vector<std::string>& inputs(int proc) const;

  struct Binding;  // a procedure bound to fixed surfaces
  std::shared_ptr<Binding> bind(int proc, const std::vector<const Surface*>& in, const Surface& out) const;
  // Synchronous evaluation (no events); returns its wall time in ms. Throws on failure.
  double evaluate(const Binding& b) const;
  // Asynchronous evaluation: starts once `wait_event` (id<MTLSharedEvent>) reaches `wait`,
  // raises `signal_event` to `signal` when done; `done(ok)` reports from a service thread.
  // Throws if it cannot start.
  void enqueue(const Binding& b, void* wait_event, uint64_t wait, void* signal_event, uint64_t signal,
               std::function<void(bool)> done) const;

  bool was_cached = false;  // the daemon held a compilation (no compile)
  double compile_ms = 0, load_ms = 0;
  std::string dir;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// GPU <-> ANE handoff (two events). The GPU signals `ready`
// values on one MTLSharedEvent and the ANE signals `done` values on another, each increasing in
// its own order (next_ready() / next_done()), so jobs of two row halves can interleave (the
// wavefront schedule) without a value ever going back. queue() registers an ANE job that waits
// for `ready` and signals `done`; once the ready event reaches `ready` a watchdog of `bound_s`
// starts. A job that fails, times out or cannot start retires the handoff: both events are raised
// to the highest values handed out, so every pending wait passes, and later values are signalled
// at once.
class Handoff {
 public:
  explicit Handoff(double bound_s = 2.0);
  ~Handoff();
  void* ready_event() const;  // id<MTLSharedEvent>: GPU -> ANE
  void* done_event() const;   // id<MTLSharedEvent>: ANE -> GPU
  uint64_t next_ready();
  uint64_t next_done();
  using Report = std::function<void(bool)>;
  using Start = std::function<void(void* ready_event, uint64_t wait, void* done_event, uint64_t signal, Report report)>;
  void queue(uint64_t ready, uint64_t done, const Start& start);
  // Waits (at most one bound) for the jobs queued since the last finish(). False if any failed
  // or did not report. ane_ms: sum over jobs of (report - event reached ready).
  bool finish(double* ane_ms = nullptr);
  void retire(const std::string& why);
  bool retired() const;
  std::string reason() const;

 private:
  struct State;
  std::shared_ptr<State> st_;
};

}  // namespace ni::ane
