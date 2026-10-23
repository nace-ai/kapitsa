#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/sysctl.h>
#include <unistd.h>
#include "metal/gpu.h"
#include <IOKit/IOKitLib.h>
#include <mach/mach_time.h>
#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Residency sets and MTLCompileOptions.mathMode come with the macOS 15 SDK. The Command Line Tools for macOS 14.4
// and earlier ship the macOS 14 SDK, which lacks them: such a build goes without residency sets and sets fast math
// with fastMathEnabled, as every build does on macOS 14.
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
#define NI_SDK_MACOS15 1
#else
#define NI_SDK_MACOS15 0
#endif

namespace ni {
extern const char* const kKernelSource;

struct Gpu::Impl {
  std::mutex mu;
  std::unordered_map<std::string, Pipeline> pipelines;
  // pipeline_threads: (default pipeline, threads per threadgroup) -> the pipeline compiled for that many
  std::map<std::pair<void*, uint32_t>, Pipeline> wide;
  NSMutableArray* retained = [NSMutableArray new]; // keeps MTLBuffers alive
  // The kernels for weight formats without specialized kernels (bf16, 3/5/6-bit, 4-bit group 32, parts of
  // 8-bit; src/kernels/wfmt.metal): the same source compiled with NI_WF_INST, on the first lookup of a kernel
  // the base library lacks, so 4-bit models never pay its compile (~2 s uncached).
  id<MTLLibrary> wf = nil;
  std::mutex wf_mu;
  // compile_extra's libraries, in the order compiled
  std::vector<std::pair<std::string, id<MTLLibrary>>> extra;
  std::mutex rs_mu;  // every call on the residency set (make_resident, release, the heartbeat)
};

static inline id<MTLDevice> dev(void* p) { return (__bridge id<MTLDevice>)p; }

// ---- Residency ----
// make_resident() puts what the GPU reads on every pass (weights, converted tensors, the drafter's) in one
// MTLResidencySet. Metal lets a set go about 2 s after the process's last command buffer, requested or not (M4 Pro,
// macOS 27: a requested 8 GB set was unwired 2-3 s after one command buffer that did not even bind it), and a command
// buffer of a queue that holds the set waits until all of it is wired again: with the set on the queue, the 35B's
// first prefill pass of a request sent 10 s after the one before started 0.39-0.50 s after its commit.
// NI_RESIDENCY (read once) says how the set is used:
//   wired (default): the set is not on the command queue. A heartbeat requests it every 500 ms while command buffers
//     are committed and for NI_RESIDENCY_KEEP_S seconds after the last one (default 600; 0 = for good), so it stays
//     wired between requests; the first commit after that restarts the heartbeat. Under memory pressure (macOS's
//     warning or critical level) there is no heartbeat, so macOS can take the memory of an idle model back, as before;
//     the first commit once the pressure is back to normal restarts it. Metal makes resident what each command buffer
//     binds, which is everything a kernel reads: Kapitsa binds every buffer to its dispatches (Encoder::buf,
//     dispatch_indirect's arguments), with no GPU addresses, argument buffers or useResource.
//   queue: the set on the command queue, requested only at make_resident() (Kapitsa before).
//   0: no residency set.
namespace {
enum class Residency { Wired, Queue, Off };

Residency residency_mode() {
  static const Residency m = [] {
    const char* v = getenv("NI_RESIDENCY");
    if (!v || !*v || !strcmp(v, "wired")) return Residency::Wired;
    if (!strcmp(v, "queue")) return Residency::Queue;
    if (!strcmp(v, "0") || !strcmp(v, "off")) return Residency::Off;
    fatal("NI_RESIDENCY=%s: wired, queue or 0", v);
  }();
  return m;
}

constexpr uint64_t kBeatNs = 500 * NSEC_PER_MSEC;

// NI_RESIDENCY=wired's heartbeat (made once, at the first make_resident; never freed, as the Gpu)
struct Keeper {
  void* set = nullptr;       // id<MTLResidencySet> (Gpu::residency_)
  std::mutex* mu = nullptr;  // Gpu::Impl::rs_mu
  dispatch_queue_t queue = nil;
  dispatch_source_t beat = nil;
  dispatch_source_t pressure = nil;  // macOS's memory pressure level changes
  void* device = nullptr;            // id<MTLDevice>, for kick()
  id<MTLCommandQueue> kick_queue = nil;
  id<MTLBuffer> kick_buf = nil;
  uint64_t keep = 0;  // mach ticks after the last commit that the heartbeat goes on (0: for good)
  std::atomic<uint64_t> last{0};  // mach time of the last commit
  std::atomic<bool> beating{false};
  unsigned long level = DISPATCH_MEMORYPRESSURE_NORMAL;  // (on `queue`) the last level seen

  // macOS's memory pressure level now (kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical).
  // NI_RESIDENCY_PRESSURE_TEST=<file> (diagnostic, read once): the level that file names instead, "warn", "critical"
  // or "normal" (anything else), for testing the warning path without real pressure; it is read at each beat and at a
  // commit while the heartbeat is off, the source's events are ignored.
  static const char* test_file() {
    static const char* f = getenv("NI_RESIDENCY_PRESSURE_TEST") && *getenv("NI_RESIDENCY_PRESSURE_TEST")
                               ? getenv("NI_RESIDENCY_PRESSURE_TEST")
                               : nullptr;
    return f;
  }
  static unsigned long level_now() {
    if (const char* path = test_file()) {
      char s[16] = {0};
      if (FILE* f = fopen(path, "r")) {
        fread(s, 1, sizeof s - 1, f);
        fclose(f);
      }
      return !strncmp(s, "critical", 8) ? DISPATCH_MEMORYPRESSURE_CRITICAL
             : !strncmp(s, "warn", 4)   ? DISPATCH_MEMORYPRESSURE_WARN
                                        : DISPATCH_MEMORYPRESSURE_NORMAL;
    }
    int lv = 1;
    size_t len = sizeof lv;
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &lv, &len, nullptr, 0)) lv = 1;
    return lv >= 4  ? DISPATCH_MEMORYPRESSURE_CRITICAL
           : lv > 1 ? DISPATCH_MEMORYPRESSURE_WARN
                    : DISPATCH_MEMORYPRESSURE_NORMAL;
  }
  static bool pressed() { return level_now() != DISPATCH_MEMORYPRESSURE_NORMAL; }
  // (on `queue`) the heartbeat off; it starts again only at a commit that sees no pressure
  void stop() {
    beating.store(false);
    dispatch_source_set_timer(beat, DISPATCH_TIME_FOREVER, 0, 0);
  }
  // One small blit on a queue of its own: Metal lets the set go about 2 s after a GPU operation that no request
  // follows, while a requested set with no GPU work after it stays wired (an idle server's last command buffer is long
  // gone).
  void kick() {
    @autoreleasepool {
      if (!kick_queue) {
        kick_queue = [(__bridge id<MTLDevice>)device newCommandQueue];
        kick_buf = [(__bridge id<MTLDevice>)device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
      }
      id<MTLCommandBuffer> cb = [kick_queue commandBuffer];
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b fillBuffer:kick_buf range:NSMakeRange(0, 16) value:0];
      [b endEncoding];
      [cb commit];
    }
  }
  // (on `queue`) macOS's memory pressure level, from the source's events or read at a beat or a commit (so a change
  // the source does not report is not missed). A change is logged; under pressure the heartbeat stops at once and the
  // set is let go; back to normal, the next commit restarts it.
  void on_pressure(unsigned long now) {
    if (now == level) return;
    level = now;
    const bool p = now & (DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL);
    if (p) {
      stop();
      kick();
    }
    logi("memory pressure %s: %s", now & DISPATCH_MEMORYPRESSURE_CRITICAL ? "critical" : p ? "warning" : "normal",
         p ? "the weights are no longer kept wired while idle" : "the weights are kept wired again");
  }

  void request() {
#if NI_SDK_MACOS15
    if (@available(macOS 15.0, *)) {
      std::lock_guard<std::mutex> lk(*mu);
      [(__bridge id<MTLResidencySet>)set requestResidency];
    }
#endif
  }
  bool idle() const {
    const uint64_t l = last.load();  // (before the clock: a commit in between cannot make it later than `now`)
    return mach_absolute_time() > l + keep;
  }
  // (on `queue`) a beat: the set again, or the heartbeat stops under pressure or once `keep` has passed since the last
  // commit
  void tick() {
    on_pressure(level_now());
    if (level != DISPATCH_MEMORYPRESSURE_NORMAL) return stop();
    if (keep && idle()) {
      beating.store(false);
      // a commit that stored `last` before that store and still read `beating` as true is seen here
      bool f = false;
      if (idle() || !beating.compare_exchange_strong(f, true)) return stop();
    }
    request();
  }
  // every commit: marks the use; a stopped heartbeat starts again, the set requested at once (off the caller's
  // thread), unless memory is under pressure
  void use() {
    last.store(mach_absolute_time());
    if (beating.load() || pressed()) return;
    bool f = false;
    if (!beating.compare_exchange_strong(f, true)) return;
    dispatch_async(queue, ^{
      on_pressure(level_now());
      if (level != DISPATCH_MEMORYPRESSURE_NORMAL) return stop();
      request();
      dispatch_source_set_timer(beat, dispatch_time(DISPATCH_TIME_NOW, kBeatNs), kBeatNs, kBeatNs / 10);
    });
  }
};
std::atomic<Keeper*> g_keeper{nullptr};

// NI_CB_LAT=<file> (diagnostic, read once): a line per command buffer as it completes, "n commit start end": its
// number in commit order and the times in ms on the host clock (mach time, Encoder::host_ms). start - commit is its
// wait from the commit; start minus the later of its commit and the end of the one committed before it is the time it
// waited with nothing in front of it on the queue (for residency, another process's GPU work, ...).
FILE* cb_lat_file() {
  static FILE* f = [] {
    const char* p = getenv("NI_CB_LAT");
    FILE* r = p && *p ? fopen(p, "a") : nullptr;
    if (r) setvbuf(r, nullptr, _IOLBF, 0);
    return r;
  }();
  return f;
}
} // namespace

// Every command buffer is committed here.
static void commit_cb(id<MTLCommandBuffer> cb) {
  if (Keeper* k = g_keeper.load(std::memory_order_acquire)) k->use();
  if (FILE* f = cb_lat_file()) {
    static std::atomic<uint64_t> seq{0};
    const unsigned long long n = seq.fetch_add(1);
    const double t = Encoder::host_ms();
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
      fprintf(f, "%llu %.4f %.4f %.4f\n", n, t, b.GPUStartTime * 1e3, b.GPUEndTime * 1e3);
    }];
  }
  [cb commit];
}

Gpu& Gpu::get() {
  static Gpu* g = new Gpu();
  return *g;
}

Gpu::Gpu() : impl_(new Impl) {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  if (!d) fatal("no Metal device");
  device_ = (__bridge_retained void*)d;
  // The GPU + Neural Engine prefill splits end a command buffer at every event signal: a 2048-row chunk of the 27B
  // is 65 command buffers and one of Nemotron-H 121, with up to two chunks queued, so the queue must allow more than
  // Metal's default 64 in flight (fewer would deadlock on the waits). Nemotron-H splits by default, so always 512.
  id<MTLCommandQueue> q = [d newCommandQueueWithMaxCommandBufferCount:512];
  queue_ = (__bridge_retained void*)q;
}

std::string Gpu::name() const { return [[dev(device_) name] UTF8String]; }
size_t Gpu::recommended_working_set() const { return dev(device_).recommendedMaxWorkingSetSize; }
size_t Gpu::allocated_bytes() const { return dev(device_).currentAllocatedSize; }
size_t Gpu::max_threadgroup_memory() const { return dev(device_).maxThreadgroupMemoryLength; }
size_t Gpu::max_buffer_length() const { return dev(device_).maxBufferLength; }
bool Gpu::apple10() const {
  // MTLGPUFamilyApple10 = 1010 (macOS 26 SDK); the raw value keeps older SDKs building
  return [dev(device_) supportsFamily:(MTLGPUFamily)1010];
}
int Gpu::apple_family() const {
  static const int f = [this] {
    for (int n = 10; n > 7; --n)  // MTLGPUFamilyApple<n> = 1000 + n, by value as above
      if ([dev(device_) supportsFamily:(MTLGPUFamily)(1000 + n)]) return n;
    return 7;  // every Mac GPU Kapitsa runs on (M1) is Apple7
  }();
  return f;
}

static int ioreg_core_count() {
  int cores = 0;
  io_iterator_t it = 0;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS) return 0;
  for (io_object_t e; (e = IOIteratorNext(it)); IOObjectRelease(e)) {
    CFTypeRef v = IORegistryEntryCreateCFProperty(e, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    if (v) {
      if (CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &cores);
      CFRelease(v);
    }
    if (cores > 0) {
      IOObjectRelease(e);
      break;
    }
  }
  IOObjectRelease(it);
  return cores;
}

static int cores_cached(bool* measured) {
  static int src = 0;  // 1 measured, 2 override, 3 name fallback
  static const int n = [] {
    if (const char* v = getenv("NI_GPU_CORES"); v && atoi(v) > 0) {
      src = 2;
      return atoi(v);
    }
    if (int c = ioreg_core_count(); c > 0) {
      src = 1;
      return c;
    }
    // smallest GPU of the tier (M1-M5: 7-10 base, 14-20 Pro, 24-40 Max, 48-80 Ultra)
    src = 3;
    const std::string name = [MTLCreateSystemDefaultDevice() name].UTF8String;
    if (name.find("Ultra") != std::string::npos) return 48;
    if (name.find("Max") != std::string::npos) return 24;
    if (name.find("Pro") != std::string::npos) return 14;
    return 8;
  }();
  if (measured) *measured = src == 1;
  return n;
}
int Gpu::core_count() const { return cores_cached(nullptr); }
bool Gpu::core_count_measured() const {
  bool m = false;
  cores_cached(&m);
  return m;
}
const char* Gpu::thermal_state() {
  switch ([NSProcessInfo processInfo].thermalState) {
    case NSProcessInfoThermalStateNominal: return "nominal";
    case NSProcessInfoThermalStateFair: return "fair";
    case NSProcessInfoThermalStateSerious: return "serious";
    case NSProcessInfoThermalStateCritical: return "critical";
  }
  return "?";
}

// alloc/wrap run in their own autorelease pool: the C++ callers have none, and Metal keeps a new
// buffer in the current pool too (measured: a buffer allocated without a pool is never freed,
// even after the last reference is released), so release() could not return its memory.
Buffer Gpu::alloc(size_t bytes, const char* label, bool tracked) {
  if (bytes == 0) bytes = 16;
  @autoreleasepool {
    id<MTLBuffer> b = [dev(device_) newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared |
                                                        (tracked ? MTLResourceHazardTrackingModeTracked
                                                                 : MTLResourceHazardTrackingModeUntracked)];
    if (!b) fatal("Metal alloc of %zu bytes failed", bytes);
    if (label) b.label = [NSString stringWithUTF8String:label];
    static const bool alog = getenv("NI_ALLOC_LOG") && atoi(getenv("NI_ALLOC_LOG"));
    if (alog && bytes >= (size_t(4) << 20)) logi("alloc %8.1f MB %s", bytes / 1e6, label ? label : "?");
    std::lock_guard<std::mutex> lk(impl_->mu);
    [impl_->retained addObject:b];
    return Buffer{(__bridge void*)b, (uint8_t*)b.contents, 0, bytes};
  }
}

Buffer Gpu::wrap(void* ptr, size_t bytes, const char* label, bool tracked) {
  @autoreleasepool {
    id<MTLBuffer> b = [dev(device_) newBufferWithBytesNoCopy:ptr
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared |
                                                             (tracked ? MTLResourceHazardTrackingModeTracked
                                                                      : MTLResourceHazardTrackingModeUntracked)
                                                 deallocator:nil];
    if (!b) fatal("Metal no-copy wrap of %zu bytes failed (page aligned?)", bytes);
    if (label) b.label = [NSString stringWithUTF8String:label];
    std::lock_guard<std::mutex> lk(impl_->mu);
    [impl_->retained addObject:b];
    return Buffer{(__bridge void*)b, (uint8_t*)ptr, 0, bytes};
  }
}

void* Gpu::new_shared_event() {
  id<MTLSharedEvent> ev = [dev(device_) newSharedEvent];
  if (!ev) fatal("newSharedEvent failed");
  return (__bridge_retained void*)ev;
}

namespace {
// Gpu::keep_warm's state: its queue, kernel, rounds per ms (measured once) and command buffers in flight
struct Heater {
  std::mutex mu;
  id<MTLCommandQueue> q = nil;
  id<MTLComputePipelineState> pso = nil;
  id<MTLBuffer> sink = nil;
  double rounds_per_ms = 0;
  std::atomic<int> inflight{0};
  std::atomic<uint64_t> started{0};
  bool failed = false;
};
Heater& heater() {
  static Heater* h = new Heater;
  return *h;
}
const char* kHeatSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void ni_heat(device uint* sink [[buffer(0)]], constant uint& n [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  uint a = t;
  for (uint r = 0; r < n; ++r)
    for (uint i = 0; i < 256; ++i) a = a * 1664525u + 1013904223u;
  if (a == 0x9e3779b9u) sink[0] = t;
}
)MSL";
id<MTLCommandBuffer> heat_cb(Heater& h, uint32_t rounds) {
  id<MTLCommandBuffer> cb = [h.q commandBufferWithUnretainedReferences];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:h.pso];
  [e setBuffer:h.sink offset:0 atIndex:0];
  [e setBytes:&rounds length:sizeof rounds atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [e endEncoding];
  return cb;
}
}  // namespace

bool Gpu::heat_on() {
  static const bool on = [] {
    // Default: on for Apple10 (M5), where a real-pack A/B measured it (Flash-Next Q2_0, streamed experts: decode
    // 1.2-1.7x, logits bit for bit, CPU+GPU+DRAM energy about flat); off elsewhere (the M4 is not measured).
    // NI_GPU_HEAT=0 turns it off, NI_GPU_HEAT=1 on, on any chip.
    const char* v = getenv("NI_GPU_HEAT");
    if (v && *v) return atoi(v) > 0;
    return Gpu::get().apple10();
  }();
  return on;
}

int Gpu::heat_inflight() { return heater().inflight.load(); }
uint64_t Gpu::heat_started() { return heater().started.load(); }

void Gpu::keep_warm(double ms) {
  if (!heat_on()) return;
  Heater& h = heater();
  if (ms > 0 && h.inflight.load(std::memory_order_relaxed) >= 2) return;
  std::lock_guard<std::mutex> lk(h.mu);
  if (h.failed) return;
  @autoreleasepool {
    if (!h.q) {
      NSError* err = nil;
      id<MTLDevice> d = dev(device_);
      id<MTLLibrary> lib = [d newLibraryWithSource:@(kHeatSource) options:[MTLCompileOptions new] error:&err];
      id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"ni_heat"] : nil;
      h.pso = fn ? [d newComputePipelineStateWithFunction:fn error:&err] : nil;
      h.q = [d newCommandQueue];
      h.sink = [d newBufferWithLength:16 options:MTLResourceStorageModeShared];
      if (!h.pso || !h.q || !h.sink) {
        h.failed = true;
        logi("GPU heater: unavailable (%s)", err ? err.localizedDescription.UTF8String : "no queue");
        return;
      }
      // rounds per ms: the median of a few short runs
      std::vector<double> r;
      for (int i = 0; i < 5; ++i) {
        id<MTLCommandBuffer> cb = heat_cb(h, 500);
        [cb commit];
        [cb waitUntilCompleted];
        const double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        if (t > 0) r.push_back(500 / t);
      }
      std::sort(r.begin(), r.end());
      h.rounds_per_ms = r.empty() ? 0 : r[r.size() / 2];
      if (h.rounds_per_ms <= 0) {
        h.failed = true;
        return;
      }
      logi("GPU heater: %.0f rounds per ms", h.rounds_per_ms);
    }
    while (ms > 0 && h.inflight.load() < 2) {
      const uint32_t rounds = uint32_t(std::clamp(h.rounds_per_ms * ms / 2, 1.0, 1e8));
      id<MTLCommandBuffer> cb = heat_cb(h, rounds);
      Heater* hp = &h;
      [cb addCompletedHandler:^(id<MTLCommandBuffer>) { hp->inflight.fetch_sub(1); }];
      h.inflight.fetch_add(1);
      h.started.fetch_add(1);
      [cb commit];
    }
  }
}

void Gpu::release(Buffer& b) {
  if (!b.mtl) return;
  std::lock_guard<std::mutex> lk(impl_->mu);
  // The residency set holds its allocations strongly (measured: a released 256 MB buffer stays
  // allocated and resident until removed), so a buffer made resident must leave the set too.
#if NI_SDK_MACOS15
  if (residency_) {
    if (@available(macOS 15.0, *)) {
      std::lock_guard<std::mutex> rl(impl_->rs_mu);
      id<MTLResidencySet> rs = (__bridge id<MTLResidencySet>)residency_;
      id<MTLBuffer> mb = (__bridge id<MTLBuffer>)b.mtl;
      if ([rs containsAllocation:mb]) {
        [rs removeAllocation:mb];
        [rs commit];
      }
    }
  }
#endif
  [impl_->retained removeObjectIdenticalTo:(__bridge id)b.mtl];
  b = Buffer{};
}

static id<MTLLibrary> compile_library(id<MTLDevice> d, const std::string& src, bool wf, double* ms,
                                      bool metal32 = false, bool metal4 = false) {
  MTLCompileOptions* o = [MTLCompileOptions new];
  // extra libraries (Nemotron-H) use Metal 3.2 for device-scope fences (atomic_thread_fence with thread_scope_device)
  o.languageVersion = MTLLanguageVersion3_1;
  if (metal32) {
    // MTLLanguageVersion3_2 by value ((3 << 16) + 2): a build with the macOS 14 SDK gets Metal 3.2 on macOS 15 too
    if (@available(macOS 15.0, *)) o.languageVersion = (MTLLanguageVersion)((3 << 16) + 2);
  }
  // Metal 4 (macOS 26): the tensor types and Metal Performance Primitives' tensor_ops (matmul2d)
  if (metal4) {
    if (@available(macOS 26.0, *)) o.languageVersion = (MTLLanguageVersion)(4 << 16);
  }
  // NI_MATH_SAFE=1: no fast math, for bitwise A/B of kernel variants written to sum in the same
  // order (e.g. NI_GDN_PAR). Fast math reassociates each kernel's sums its own way, so two such
  // kernels only agree to the last bit or so; with safe math they must agree exactly.
  const bool safe = getenv("NI_MATH_SAFE") && atoi(getenv("NI_MATH_SAFE")) > 0;
#if NI_SDK_MACOS15
  if (@available(macOS 15.0, *)) {
    o.mathMode = safe ? MTLMathModeSafe : MTLMathModeFast;
  } else {
    o.fastMathEnabled = safe ? NO : YES;
  }
#else
  o.fastMathEnabled = safe ? NO : YES;
#endif
  if (wf) o.preprocessorMacros = @{@"NI_WF_INST" : @1};
  NSError* err = nil;
  const double t0 = now_ms();
  id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
  if (!lib) fatal("Metal compile failed: %s", err ? [[err description] UTF8String] : "?");
  *ms = now_ms() - t0;
  return lib;
}

void Gpu::compile(const std::string& extra) {
  if (library_) return;
  @autoreleasepool {
    if (getenv("NI_MATH_SAFE") && atoi(getenv("NI_MATH_SAFE")) > 0) logi("NI_MATH_SAFE: kernels compiled without fast math");
    double ms = 0;
    id<MTLLibrary> lib = compile_library(dev(device_), std::string(kKernelSource) + "\n" + extra, false, &ms);
    library_ = (__bridge_retained void*)lib;
    logi("compiled kernels in %.0f ms", ms);
  }
}

void Gpu::compile_formats() {
  compile();
  std::lock_guard<std::mutex> lk(impl_->wf_mu);
  if (impl_->wf) return;
  @autoreleasepool {
    double ms = 0;
    impl_->wf = compile_library(dev(device_), std::string(kKernelSource), true, &ms);
    logi("compiled the weight-format kernels in %.0f ms", ms);
  }
}

void Gpu::compile_extra(const std::string& key, const char* source, bool metal4) {
  compile();
  std::lock_guard<std::mutex> lk(impl_->wf_mu);
  for (auto& [k, lib] : impl_->extra)
    if (k == key) return;
  @autoreleasepool {
    double ms = 0;
    id<MTLLibrary> lib = compile_library(dev(device_), std::string(source), false, &ms, !metal4, metal4);
    impl_->extra.push_back({key, lib});
    logi("compiled the %s kernels in %.0f ms", key.c_str(), ms);
  }
}

// The function `fn` from the base library, else from an extra library, else from the weight-format library
// (compiled on first need).
static id<MTLFunction> find_function(Gpu& g, void* base, Gpu::Impl* impl, const std::string& fn) {
  NSString* n = [NSString stringWithUTF8String:fn.c_str()];
  id<MTLFunction> f = [(__bridge id<MTLLibrary>)base newFunctionWithName:n];
  if (f) return f;
  {
    std::lock_guard<std::mutex> lk(impl->wf_mu);
    for (auto& [k, lib] : impl->extra)
      if ((f = [lib newFunctionWithName:n])) return f;
  }
  g.compile_formats();
  return [impl->wf newFunctionWithName:n];
}

Pipeline Gpu::pipeline(const std::string& fn, bool required) {
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->pipelines.find(fn);
    if (it != impl_->pipelines.end()) return it->second;
  }
  compile();
  @autoreleasepool {
    id<MTLFunction> f = find_function(*this, library_, impl_, fn);
    if (!f && !required) return Pipeline{};
    if (!f) fatal("kernel %s not found", fn.c_str());
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev(device_) newComputePipelineStateWithFunction:f error:&err];
    if (!pso && !required) {
      fprintf(stderr, "pipeline %s: %s\n", fn.c_str(), [[err description] UTF8String]);
      return Pipeline{};
    }
    if (!pso) fatal("pipeline %s: %s", fn.c_str(), [[err description] UTF8String]);
    Pipeline p{(__bridge_retained void*)pso, strdup(fn.c_str()), (uint32_t)pso.maxTotalThreadsPerThreadgroup,
               (uint32_t)pso.threadExecutionWidth, (uint32_t)pso.staticThreadgroupMemoryLength};
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->pipelines[fn] = p;
    return p;
  }
}

Pipeline Gpu::pipeline_threads(const Pipeline& p, uint32_t threads) {
  const auto key = std::make_pair(p.pso, threads);
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->wide.find(key);
    if (it != impl_->wide.end()) return it->second;
  }
  @autoreleasepool {
    id<MTLFunction> f = find_function(*this, library_, impl_, p.name);
    MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
    d.computeFunction = f;
    d.maxTotalThreadsPerThreadgroup = threads;
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        f ? [dev(device_) newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&err]
          : nil;
    if (!pso || pso.maxTotalThreadsPerThreadgroup < threads)
      fatal("kernel %s: %u threads per threadgroup, this GPU allows %u%s%s", p.name, threads,
            pso ? (uint32_t)pso.maxTotalThreadsPerThreadgroup : p.max_threads, err ? ": " : "",
            err ? [[err description] UTF8String] : "");
    Pipeline w{(__bridge_retained void*)pso, p.name, (uint32_t)pso.maxTotalThreadsPerThreadgroup,
               (uint32_t)pso.threadExecutionWidth, (uint32_t)pso.staticThreadgroupMemoryLength};
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto [it, fresh] = impl_->wide.emplace(key, w);
    if (fresh) logi("kernel %s compiled for %u threads per threadgroup (this GPU's default: %u)", p.name, threads, p.max_threads);
    else CFRelease(w.pso);  // another thread made it first
    return it->second;
  }
}

bool Gpu::archive(const std::string& fn, const std::string& path) {
  compile();
  @autoreleasepool {
    id<MTLFunction> f = find_function(*this, library_, impl_, fn);
    if (!f) return false;
    NSError* err = nil;
    id<MTLBinaryArchive> ar = [dev(device_) newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
    MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
    d.computeFunction = f;
    if (!ar || ![ar addComputePipelineFunctionsWithDescriptor:d error:&err]) {
      logi("archive %s: %s", fn.c_str(), err ? [[err description] UTF8String] : "?");
      return false;
    }
    if (![ar serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&err]) {
      logi("archive %s: %s", fn.c_str(), err ? [[err description] UTF8String] : "?");
      return false;
    }
    return true;
  }
}

std::vector<std::string> Gpu::function_names(bool formats) {
  compile();
  std::vector<std::string> v;
  for (NSString* n in [(__bridge id<MTLLibrary>)library_ functionNames]) v.push_back([n UTF8String]);
  if (formats) {
    compile_formats();
    for (NSString* n in [impl_->wf functionNames]) v.push_back([n UTF8String]);
  }
  {
    std::lock_guard<std::mutex> lk(impl_->wf_mu);
    for (auto& [k, lib] : impl_->extra)
      for (NSString* n in [lib functionNames]) v.push_back([n UTF8String]);
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

std::unique_ptr<Encoder> Gpu::begin(bool concurrent) {
  auto e = std::unique_ptr<Encoder>(new Encoder());
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
  id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
  id<MTLComputeCommandEncoder> enc =
      [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
  e->cb_ = (__bridge_retained void*)cb;
  e->enc_ = (__bridge_retained void*)enc;
  e->concurrent_ = concurrent;
  return e;
}

std::unique_ptr<Encoder> Gpu::begin_grouped() {
  auto e = begin(true);
  e->auto_barrier_ = true;
  return e;
}

void Gpu::make_resident(const std::vector<Buffer>& bufs) {
#if !NI_SDK_MACOS15
  (void)bufs;
#else
  if (residency_mode() == Residency::Off) return;
  if (@available(macOS 15.0, *)) {
    Keeper* started = nullptr;
    @autoreleasepool {
      std::lock_guard<std::mutex> lk(impl_->rs_mu);
      if (!residency_) {
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.label = @"kapitsa";
        NSError* err = nil;
        id<MTLResidencySet> rs = [dev(device_) newResidencySetWithDescriptor:d error:&err];
        if (!rs) return;
        residency_ = (__bridge_retained void*)rs;
        if (residency_mode() == Residency::Queue) {
          [(__bridge id<MTLCommandQueue>)queue_ addResidencySet:rs];
        } else {
          Keeper* k = new Keeper();
          k->set = residency_;
          k->mu = &impl_->rs_mu;
          const char* ks = getenv("NI_RESIDENCY_KEEP_S");
          const double keep_s = ks && *ks ? atof(ks) : 600.0;
          mach_timebase_info_data_t tb;
          mach_timebase_info(&tb);
          k->keep = keep_s > 0 ? uint64_t(keep_s * 1e9 * tb.denom / tb.numer) : 0;
          k->queue = dispatch_queue_create("kapitsa.residency", DISPATCH_QUEUE_SERIAL);
          k->beat = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, k->queue);
          dispatch_source_set_event_handler(k->beat, ^{ k->tick(); });
          dispatch_source_set_timer(k->beat, DISPATCH_TIME_FOREVER, 0, 0);
          dispatch_activate(k->beat);
          k->device = device_;
          k->pressure = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                                               DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
                                                   DISPATCH_MEMORYPRESSURE_CRITICAL,
                                               k->queue);
          dispatch_source_set_event_handler(k->pressure, ^{
            if (!Keeper::test_file()) k->on_pressure(dispatch_source_get_data(k->pressure));
          });
          dispatch_activate(k->pressure);
          started = k;
        }
      }
      id<MTLResidencySet> rs = (__bridge id<MTLResidencySet>)residency_;
      for (auto& b : bufs)
        if (b.mtl) [rs addAllocation:(__bridge id<MTLBuffer>)b.mtl];
      [rs commit];
      [rs requestResidency];
    }
    if (started) {
      g_keeper.store(started, std::memory_order_release);
      started->use();  // the heartbeat from now on
    }
  }
#endif
}

// ---- Encoder ----
static inline id<MTLComputeCommandEncoder> E(void* p) { return (__bridge id<MTLComputeCommandEncoder>)p; }

void Encoder::use(const Pipeline& p) {
  [E(enc_) setComputePipelineState:(__bridge id<MTLComputePipelineState>)p.pso];
  cur_name_ = p.name;
  cur_pso_ = p.pso;
  cur_max_threads_ = p.max_threads;
}

// A GPU's compiler may give a kernel fewer threads per threadgroup than it is dispatched with (register pressure
// differs between GPU families: dfl_dist_rows gets 576 on the M1 Pro, 1024 on the M4 Pro), and a dispatch above that
// fails the command buffer. Such a kernel switches to its pipeline compiled for the dispatch's size; on GPUs whose
// pipelines allow every dispatch, nothing changes.
void Encoder::fit_threads(uint32_t threads) {
  if (!cur_max_threads_ || threads <= cur_max_threads_) return;
  Pipeline w = Gpu::get().pipeline_threads(Pipeline{cur_pso_, cur_name_, cur_max_threads_}, threads);
  [E(enc_) setComputePipelineState:(__bridge id<MTLComputePipelineState>)w.pso];
  cur_pso_ = w.pso;
  cur_max_threads_ = w.max_threads;
}

namespace {
struct ProfRec {
  std::string name;
  id<MTLCommandBuffer> cb;
};
std::vector<ProfRec>& prof_recs() {
  static std::vector<ProfRec> v;
  return v;
}
std::string& prof_tag() {
  static std::string t;
  return t;
}
} // namespace

void Encoder::profile_tag(const char* tag) { prof_tag() = tag ? tag : ""; }

bool Encoder::profiling() {
  static bool on = getenv("NI_PROFILE") && atoi(getenv("NI_PROFILE")) > 0;
  return on;
}

void Encoder::profile_report(int steps) {
  std::map<std::string, std::pair<double, int>> agg;
  double total = 0;
  for (auto& r : prof_recs()) {
    [r.cb waitUntilCompleted];
    double ms = (r.cb.GPUEndTime - r.cb.GPUStartTime) * 1000.0;
    agg[r.name].first += ms;
    agg[r.name].second += 1;
    total += ms;
  }
  std::vector<std::pair<std::string, std::pair<double, int>>> v(agg.begin(), agg.end());
  std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
  fprintf(stderr, "---- GPU profile per step (%d steps, %.3f ms/step total, isolated dispatches) ----\n", steps,
          total / steps);
  for (auto& [n, p] : v)
    fprintf(stderr, "  %-28s %8.3f ms/step  %5.1f%%  %6.1f calls/step  %7.2f us/call\n", n.c_str(), p.first / steps,
            100 * p.first / total, double(p.second) / steps, 1000 * p.first / p.second);
  prof_recs().clear();
}
void Encoder::buf(int idx, const Buffer& b, size_t extra) {
  [E(enc_) setBuffer:(__bridge id<MTLBuffer>)b.mtl offset:b.offset + extra atIndex:idx];
}
void Encoder::bytes(int idx, const void* data, size_t len) { [E(enc_) setBytes:data length:len atIndex:idx]; }
void Encoder::tgmem(int idx, size_t len) { [E(enc_) setThreadgroupMemoryLength:(len + 15) & ~size_t(15) atIndex:idx]; }
// NI_DUP=<kernel>[,<kernel>...] (timing aid, read once): every dispatch of these kernels runs twice in
// a row, so (time with) - (time without) is their in-pipeline cost, unlike NI_PROFILE's isolated
// command buffers. Only meaningful for kernels that do not read their own outputs (GEMMs without
// accumulation, top-k, ...): an accumulating kernel would change the results downstream.
static bool dup_kernel(const char* name) {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v;
    if (const char* s = getenv("NI_DUP")) {
      std::string all = s;
      size_t p = 0;
      while (p <= all.size()) {
        size_t q = all.find(',', p);
        if (q == std::string::npos) q = all.size();
        if (q > p) v.push_back(all.substr(p, q - p));
        p = q + 1;
      }
    }
    return v;
  }();
  if (names.empty()) return false;
  for (auto& n : names)
    if (n == name) return true;
  return false;
}

// grouped encoders: a barrier before each dispatch but the second and later of a group
void Encoder::pre_dispatch() {
  if (!auto_barrier_) return;
  if (dispatched_ && (!in_group_ || group_first_)) barrier();
  group_first_ = false;
  dispatched_ = true;
}

void Encoder::dispatch(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
  fit_threads(tx * ty * tz);
  pre_dispatch();
  [E(enc_) dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  if (dup_kernel(cur_name_))
    [E(enc_) dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  if (profiling()) {
    // close this dispatch into its own command buffer and continue in a fresh one
    [E(enc_) endEncoding];
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
    commit_cb(cb);
    {
      static const bool by_grid = atoi(getenv("NI_PROFILE")) > 1;  // NI_PROFILE=2: per kernel and grid
      std::string nm = prof_tag() + cur_name_;
      if (by_grid) nm += "/" + std::to_string(gx) + "x" + std::to_string(gy) + "x" + std::to_string(gz);
      prof_recs().push_back({nm, cb});
    }
    CFRelease(enc_);
    CFRelease(cb_);
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)Gpu::get().queue();
    id<MTLCommandBuffer> nb = [q commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> ne =
        [nb computeCommandEncoderWithDispatchType:concurrent_ ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    cb_ = (__bridge_retained void*)nb;
    enc_ = (__bridge_retained void*)ne;
    // the fresh encoder starts with no pipeline state: a kernel dispatched twice under one use() (a loop over tiles)
    // would otherwise dispatch it with none
    if (cur_pso_) [ne setComputePipelineState:(__bridge id<MTLComputePipelineState>)cur_pso_];
  }
}
void Encoder::dispatch_indirect(const Buffer& args, size_t off, uint32_t tx, uint32_t ty, uint32_t tz) {
  fit_threads(tx * ty * tz);
  pre_dispatch();
  [E(enc_) dispatchThreadgroupsWithIndirectBuffer:(__bridge id<MTLBuffer>)args.mtl
                             indirectBufferOffset:args.offset + off
                            threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  if (profiling()) {
    [E(enc_) endEncoding];
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
    commit_cb(cb);
    {
      static const bool by_grid = atoi(getenv("NI_PROFILE")) > 1;  // NI_PROFILE=2: per kernel and grid
      std::string nm = prof_tag() + cur_name_;
      if (by_grid) nm += "/ind";  // (an indirect grid is read on the GPU: all of a kernel's indirect dispatches in one row)
      prof_recs().push_back({nm, cb});
    }
    CFRelease(enc_);
    CFRelease(cb_);
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)Gpu::get().queue();
    id<MTLCommandBuffer> nb = [q commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> ne =
        [nb computeCommandEncoderWithDispatchType:concurrent_ ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    cb_ = (__bridge_retained void*)nb;
    enc_ = (__bridge_retained void*)ne;
    if (cur_pso_) [ne setComputePipelineState:(__bridge id<MTLComputePipelineState>)cur_pso_];
  }
}
void Encoder::barrier() { [E(enc_) memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
void Encoder::on_completed(std::function<void()> f) {
  [(__bridge id<MTLCommandBuffer>)cb_ addCompletedHandler:^(id<MTLCommandBuffer>) {
    f();
  }];
}

void Encoder::commit() {
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
    enc_ = nullptr;
  }
  commit_cb((__bridge id<MTLCommandBuffer>)cb_);
}
void Encoder::wait() {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
  [cb waitUntilCompleted];
  for (void* p : prev_) {
    id<MTLCommandBuffer> b = (__bridge id<MTLCommandBuffer>)p;
    [b waitUntilCompleted];
    if (b.status == MTLCommandBufferStatusError) fatal("GPU error: %s", [[b.error description] UTF8String]);
  }
  if (cb.status == MTLCommandBufferStatusError)
    fatal("GPU error: %s", [[cb.error description] UTF8String]);
  static const bool cbt = getenv("NI_CB_TRACE") && atoi(getenv("NI_CB_TRACE"));
  if (cbt && prev_.size() > 4) {
    // debug: GPU time of each command buffer of this encoder, the idle gaps between them
    std::vector<std::pair<double, double>> t;
    for (void* p : prev_) {
      id<MTLCommandBuffer> b = (__bridge id<MTLCommandBuffer>)p;
      t.push_back({b.GPUStartTime * 1e3, b.GPUEndTime * 1e3});
    }
    t.push_back({cb.GPUStartTime * 1e3, cb.GPUEndTime * 1e3});
    double busy = 0, gaps = 0, last_end = t[0].first;
    std::vector<std::pair<double, size_t>> big;
    for (size_t i = 0; i < t.size(); ++i) {
      const double st = std::max(t[i].first, last_end);
      if (t[i].first > last_end) {
        gaps += t[i].first - last_end;
        big.push_back({t[i].first - last_end, i});
      }
      if (t[i].second > st) busy += t[i].second - st;
      last_end = std::max(last_end, t[i].second);
    }
    std::sort(big.rbegin(), big.rend());
    std::string g;
    for (size_t i = 0; i < std::min<size_t>(6, big.size()); ++i)
      g += " " + std::to_string(int(big[i].first * 10) / 10.0).substr(0, 5) + "@" + std::to_string(big[i].second);
    logi("cb trace: %zu command buffers, span %.1f ms, GPU busy %.1f ms, idle gaps %.1f ms (largest:%s)", t.size(),
         last_end - t[0].first, busy, gaps, g.c_str());
  }
}
bool Encoder::wait_for(double timeout_ms) {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
  const double t0 = host_ms();
  while (cb.status < MTLCommandBufferStatusCompleted) {
    if (host_ms() - t0 > timeout_ms) return false;
    usleep(500);
  }
  return true;
}
double Encoder::gpu_start_host_ms() const {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
  id<MTLCommandBuffer> first = prev_.empty() ? cb : (__bridge id<MTLCommandBuffer>)prev_.front();
  return first.GPUStartTime * 1000.0;
}
double Encoder::gpu_end_host_ms() const { return ((__bridge id<MTLCommandBuffer>)cb_).GPUEndTime * 1000.0; }
double Encoder::host_ms() {
  static const double scale = [] {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return double(tb.numer) / double(tb.denom) / 1e6;
  }();
  return double(mach_absolute_time()) * scale;
}
double Encoder::gpu_ms() const {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
  id<MTLCommandBuffer> first = prev_.empty() ? cb : (__bridge id<MTLCommandBuffer>)prev_.front();
  return (cb.GPUEndTime - first.GPUStartTime) * 1000.0;
}
void Encoder::open_encoder() {
  id<MTLComputeCommandEncoder> ne = [(__bridge id<MTLCommandBuffer>)cb_
      computeCommandEncoderWithDispatchType:concurrent_ ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
  enc_ = (__bridge_retained void*)ne;
}
void Encoder::signal_event(void* event, uint64_t value) {
  id<MTLSharedEvent> ev = (__bridge id<MTLSharedEvent>)event;
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
    enc_ = nullptr;
  }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
  [cb encodeSignalEvent:ev value:value];
  [cb addCompletedHandler:^(id<MTLCommandBuffer> ended) {
    if (ended.status == MTLCommandBufferStatusError && ev.signaledValue < value) ev.signaledValue = value;
  }];
  commit_cb(cb);
  prev_.push_back(cb_);
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)Gpu::get().queue();
  cb_ = (__bridge_retained void*)[q commandBufferWithUnretainedReferences];
  open_encoder();
}
void Encoder::split() {
  if (profiling()) return;  // (every dispatch has its own command buffer then)
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
    enc_ = nullptr;
  }
  commit_cb((__bridge id<MTLCommandBuffer>)cb_);
  prev_.push_back(cb_);
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)Gpu::get().queue();
  cb_ = (__bridge_retained void*)[q commandBufferWithUnretainedReferences];
  open_encoder();
}
void Gpu::signal_event_cpu(void* event, uint64_t value) { ((__bridge id<MTLSharedEvent>)event).signaledValue = value; }
uint64_t Gpu::event_value(void* event) { return ((__bridge id<MTLSharedEvent>)event).signaledValue; }

void Encoder::signal_event_inline(void* event, uint64_t value) {
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
    enc_ = nullptr;
  }
  [(__bridge id<MTLCommandBuffer>)cb_ encodeSignalEvent:(__bridge id<MTLSharedEvent>)event value:value];
  open_encoder();
}

void Encoder::wait_event(void* event, uint64_t value) {
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
    enc_ = nullptr;
  }
  [(__bridge id<MTLCommandBuffer>)cb_ encodeWaitForEvent:(__bridge id<MTLSharedEvent>)event value:value];
  open_encoder();
}
Encoder::~Encoder() {
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
  }
  if (cb_) CFRelease(cb_);
  for (void* p : prev_) CFRelease(p);
}

} // namespace ni
