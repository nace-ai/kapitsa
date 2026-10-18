// Private AppleNeuralEngine client (see ane.h). Interface declarations and signature table adapted
// from Splash (incoai/splash, Apache-2.0, runtime/ane/Program.mm).
// Nace AI changes: NiAne protocols, local API resolution and service-queue/watchdog integration.
// License notice: THIRD_PARTY_NOTICES.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "ane/ane.h"
#include "util/log.h"

#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <sys/qos.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

// ---- the private interface, as protocols so calls type-check (resolved at run time) ----
@protocol NiAneModel
+ (id)modelAtURL:(NSURL*)url key:(NSString*)key;
- (NSDictionary*)modelAttributes;
@end
@protocol NiAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary*)options request:(id)request qos:(unsigned)qos
                    error:(NSError**)error;
@end
@protocol NiAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol NiAneRequest
+ (id)requestWithInputs:(NSArray*)inputs inputIndices:(NSArray*)inputIndices outputs:(NSArray*)outputs
          outputIndices:(NSArray*)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber*)procedure sharedEvents:(id)events transactionHandle:(NSNumber*)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError* error))handler;
@end
@protocol NiAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)symbol eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray*)signals waitEvents:(NSArray*)waits;
@end

namespace ni::ane {
namespace {

constexpr unsigned kQos = QOS_CLASS_DEFAULT;

struct Api {
  id<NiAneClient> client = nil;
  Class model = nil, surface = nil, request = nil, events = nil, signalEvent = nil, waitEvent = nil;
};

struct Sig {
  const char *cls, kind, *sel, *types;
};
// arm64 type encodings with digits stripped: a changed method is refused, never mis-called.
const Sig kSigs[] = {
    {"_ANEModel", '+', "modelAtURL:key:", "@@:@@"},
    {"_ANEModel", '-', "modelAttributes", "@@:"},
    {"_ANEClient", '+', "sharedConnection", "@@:"},
    {"_ANEClient", '-', "compileModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "compiledModelExistsFor:", "B@:@"},
    {"_ANEClient", '-', "purgeCompiledModel:", "v@:@"},
    {"_ANEClient", '-', "loadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "unloadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "evaluateWithModel:options:request:qos:error:", "B@:@@@I^@"},
    {"_ANEIOSurfaceObject", '+', "objectWithIOSurface:", "@@:^{__IOSurface=}"},
    {"_ANERequest", '+',
     "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:"
     "transactionHandle:",
     "@@:@@@@@@@@@"},
    {"_ANERequest", '-', "setCompletionHandler:", "v@:@?"},
    {"_ANESharedWaitEvent", '+', "waitEventWithValue:sharedEvent:eventType:", "@@:Q@Q"},
    {"_ANESharedSignalEvent", '+', "signalEventWithValue:symbolIndex:eventType:sharedEvent:", "@@:QIq@"},
    {"_ANESharedEvents", '+', "sharedEventsWithSignalEvents:waitEvents:", "@@:@@"},
};

std::string g_unavailable;

const Api* api() {
  static Api a;
  static bool ok = [] {
    @autoreleasepool {
      if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        const char* e = dlerror();
        g_unavailable = std::string("AppleNeuralEngine does not load: ") + (e ? e : "?");
        return false;
      }
      for (const Sig& s : kSigs) {
        Class c = NSClassFromString(@(s.cls));
        if (!c) {
          g_unavailable = std::string("AppleNeuralEngine lacks ") + s.cls;
          return false;
        }
        SEL sel = sel_registerName(s.sel);
        Method m = s.kind == '+' ? class_getClassMethod(c, sel) : class_getInstanceMethod(c, sel);
        if (!m) {
          g_unavailable = std::string("AppleNeuralEngine lacks ") + s.kind + "[" + s.cls + " " + s.sel + "]";
          return false;
        }
        std::string t = method_getTypeEncoding(m) ?: "";
        std::erase_if(t, [](char ch) { return ch >= '0' && ch <= '9'; });
        if (t != s.types) {
          g_unavailable = std::string("AppleNeuralEngine changed ") + s.kind + "[" + s.cls + " " + s.sel + "]: " + t;
          return false;
        }
      }
      a.model = NSClassFromString(@"_ANEModel");
      a.surface = NSClassFromString(@"_ANEIOSurfaceObject");
      a.request = NSClassFromString(@"_ANERequest");
      a.events = NSClassFromString(@"_ANESharedEvents");
      a.signalEvent = NSClassFromString(@"_ANESharedSignalEvent");
      a.waitEvent = NSClassFromString(@"_ANESharedWaitEvent");
      @try {
        a.client = [(Class<NiAneClient>)NSClassFromString(@"_ANEClient") sharedConnection];
      } @catch (NSException* ex) {
        g_unavailable = std::string("_ANEClient sharedConnection raised ") + ex.reason.UTF8String;
        return false;
      }
      if (!a.client) {
        g_unavailable = "no _ANEClient shared connection";
        return false;
      }
      return true;
    }
  }();
  return ok ? &a : nullptr;
}

std::string err_text(NSError* e) { return e ? std::string(e.description.UTF8String) : std::string("no error given"); }

// Runs `fn` on one serial queue (all compile/load/unload calls), waiting at most `limit_s`.
// Objective-C exceptions become std::runtime_error.
void on_service_queue(const char* what, double limit_s, const std::function<void()>& fn) {
  static dispatch_queue_t q = dispatch_queue_create("nace.ane.service", DISPATCH_QUEUE_SERIAL);
  struct Box {
    std::string error;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  };
  auto box = std::make_shared<Box>();
  auto f = std::make_shared<std::function<void()>>(fn);
  dispatch_async(q, ^{
    @autoreleasepool {
      @try {
        (*f)();
      } @catch (NSException* ex) {
        box->error = std::string(what) + " raised " + (ex.reason ? ex.reason.UTF8String : "?");
      } @catch (...) {
        box->error = std::string(what) + " failed";
      }
    }
    dispatch_semaphore_signal(box->sem);
  });
  if (dispatch_semaphore_wait(box->sem, dispatch_time(DISPATCH_TIME_NOW, int64_t(limit_s * 1e9))))
    throw std::runtime_error(std::string(what) + " did not finish within " + std::to_string(int(limit_s)) + " s");
  if (!box->error.empty()) throw std::runtime_error(box->error);
}

}  // namespace

const std::string& unavailable() {
  (void)api();
  return g_unavailable;
}

// ---- surfaces ----
std::string Surface::strides(uint32_t r) const {
  const uint64_t s = stride / elem, plane = uint64_t(r) * s;
  return "[" + std::to_string(plane) + ", " + std::to_string(plane) + ", " + std::to_string(s) + ", 1]";
}
std::string Surface::buffer_type(uint32_t r, uint32_t w) const {
  return "tensor_buffer<" + mil_type() + ", shape=[1, 1, " + std::to_string(r) + ", " + std::to_string(w) +
         "], strides=" + strides(rows) + ", interleave_factors=[1, 1, 1, 1]>";
}

Surface surface_layout(uint32_t rows, uint32_t width, bool fp16) {
  Surface s;
  s.rows = rows, s.width = width, s.fp16 = fp16, s.elem = fp16 ? 2 : 1;
  s.stride = (width * s.elem + 63) / 64 * 64;
  const size_t page = 16384;
  s.bytes = (size_t(s.stride) * rows + page - 1) / page * page;
  return s;
}

Surface make_surface(uint32_t rows, uint32_t width, bool fp16) {
  Surface s = surface_layout(rows, width, fp16);
  @autoreleasepool {
    NSDictionary* props = @{
      (id)kIOSurfaceWidth : @(width),
      (id)kIOSurfaceHeight : @(rows),
      (id)kIOSurfaceBytesPerElement : @(s.elem),
      (id)kIOSurfaceBytesPerRow : @(s.stride),
      (id)kIOSurfaceAllocSize : @(s.bytes),
      (id)kIOSurfacePixelFormat : @(fp16 ? 0x4c303068 : 0x4c303038),  // 'L00h' / 'L008'
    };
    IOSurfaceRef ref = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!ref) throw std::runtime_error("IOSurfaceCreate " + std::to_string(rows) + "x" + std::to_string(width) + " failed");
    if (IOSurfaceGetBytesPerRow(ref) != s.stride || IOSurfaceGetAllocSize(ref) < s.bytes) {
      CFRelease(ref);
      throw std::runtime_error("IOSurface took another row stride or size");
    }
    s.ref = ref;
    s.buf = Gpu::get().wrap(IOSurfaceGetBaseAddress(ref), s.bytes, "ane surface", /*tracked=*/true);
  }
  return s;
}

void free_surface(Surface& s) {
  Gpu::get().release(s.buf);
  if (s.ref) CFRelease((IOSurfaceRef)s.ref);
  s = Surface{};
}

std::vector<uint8_t> fp16_blob(const _Float16* v, size_t n) {
  std::vector<uint8_t> blob(128 + n * 2);
  auto put = [&](size_t off, auto x) { memcpy(blob.data() + off, &x, sizeof x); };
  put(0, uint32_t(1));            // blobs
  put(4, uint32_t(2));            // storage version
  put(64, uint32_t(0xdeadbeef));  // record sentinel
  put(68, uint32_t(1));           // fp16
  put(72, uint64_t(n * 2));
  put(80, uint64_t(128));
  memcpy(blob.data() + 128, v, n * 2);
  return blob;
}

uint64_t fnv1a(const void* p, size_t n, uint64_t h) {
  for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ULL;
  return h;
}

// ---- programs ----
struct Proc {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<uint32_t> in_sym;
  uint32_t out_sym = 0;
};

struct Program::Impl {
  const Api* api = nullptr;
  id model = nil;
  std::vector<Proc> procs;
  bool loaded = false;
};

struct Program::Binding {
  NSArray *ins = nil, *inIdx = nil, *outs = nil, *outIdx = nil;
  NSNumber* proc = nil;
};

Program::Program(const std::string& mil, const std::vector<uint8_t>& blob, const std::string& tag, double limit_s)
    : impl_(std::make_shared<Impl>()) {
  impl_->api = api();
  if (!impl_->api) throw std::runtime_error(g_unavailable);
  const uint64_t key = fnv1a(blob.data(), blob.size(), fnv1a(mil.data(), mil.size()));
  char hex[17];
  snprintf(hex, sizeof hex, "%016llx", (unsigned long long)key);
  char base[1024] = {0};
  if (!confstr(_CS_DARWIN_USER_CACHE_DIR, base, sizeof base)) throw std::runtime_error("no DARWIN_USER_CACHE_DIR");
  std::filesystem::path d = std::filesystem::path(base) / tag / hex;
  std::error_code ec;
  std::filesystem::create_directories(d, ec);
  if (ec) throw std::runtime_error("cannot create " + d.string());
  dir = d.string();
  auto write_if_changed = [&](const std::filesystem::path& p, const void* data, size_t n) {
    std::error_code e2;
    if (std::filesystem::exists(p, e2) && std::filesystem::file_size(p, e2) == n) {
      std::ifstream f(p, std::ios::binary);
      std::vector<char> old(n);
      f.read(old.data(), n);
      if (f && memcmp(old.data(), data, n) == 0) return;
    }
    std::filesystem::path tmp = p;
    tmp += ".tmp";
    {
      std::ofstream f(tmp, std::ios::binary);
      f.write((const char*)data, n);
      if (!f) throw std::runtime_error("cannot write " + tmp.string());
    }
    std::filesystem::rename(tmp, p);
  };
  write_if_changed(d / "model.mil", mil.data(), mil.size());
  write_if_changed(d / "weights.bin", blob.data(), blob.size());
  // The service call may outlive this constructor if it times out: it only touches shared state.
  struct Out {
    bool cached = false;
    double cms = 0, lms = 0;
  };
  auto out = std::make_shared<Out>();
  auto impl = impl_;
  const std::string dirs = dir, keys = hex;
  on_service_queue("ANE compile/load", limit_s, [impl, out, dirs, keys] {
    bool& cached = out->cached;
    double& cms = out->cms;
    double& lms = out->lms;
    const Api& a = *impl->api;
    impl->model = [(Class<NiAneModel>)a.model modelAtURL:[NSURL fileURLWithPath:@(dirs.c_str()) isDirectory:YES]
                                                     key:@(keys.c_str())];
    if (!impl->model) throw std::runtime_error("_ANEModel modelAtURL:key: returned nil");
    NSDictionary* opts = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
    NSError* err = nil;
    double t0 = now_ms();
    cached = [a.client compiledModelExistsFor:impl->model];
    if (!cached && ![a.client compileModel:impl->model options:opts qos:kQos error:&err])
      throw std::runtime_error("ANE compile failed: " + err_text(err));
    cms = now_ms() - t0;
    t0 = now_ms();
    if (![a.client loadModel:impl->model options:@{} qos:kQos error:&err]) {
      // the service may keep a compilation it cannot load: purge, compile again, load again
      logi("ANE load failed (%s); purging and recompiling", err_text(err).c_str());
      [a.client purgeCompiledModel:impl->model];
      err = nil;
      if (![a.client compileModel:impl->model options:opts qos:kQos error:&err])
        throw std::runtime_error("ANE recompile failed: " + err_text(err));
      if (![a.client loadModel:impl->model options:@{} qos:kQos error:&err])
        throw std::runtime_error("ANE load failed: " + err_text(err));
      cached = false;
    }
    lms = now_ms() - t0;
    impl->loaded = true;
    NSDictionary* desc = [impl->model modelAttributes][@"ANEFModelDescription"];
    NSArray* syms = desc[@"kANEFModelInputSymbolsArrayKey"];
    NSDictionary* names = desc[@"kANEFModelProcedureNameToIDMapKey"];
    NSArray* entries = desc[@"ANEFModelProcedures"];
    if (!syms || !names || !entries) throw std::runtime_error("ANE model does not describe its procedures");
    impl->procs.assign(entries.count, Proc{});
    for (NSString* n in names) {
      const unsigned pid = [names[n] unsignedIntValue];
      if (pid >= impl->procs.size()) throw std::runtime_error("ANE procedure id out of range");
      impl->procs[pid].name = n.UTF8String;
    }
    for (NSDictionary* e in entries) {
      const unsigned pid = [e[@"ANEFModelProcedureID"] unsignedIntValue];
      if (pid >= impl->procs.size()) throw std::runtime_error("ANE procedure id out of range");
      Proc& p = impl->procs[pid];
      NSArray* outs = e[@"ANEFModelOutputSymbolIndexArray"];
      if (outs.count != 1) throw std::runtime_error("ANE procedure with other than one output");
      p.out_sym = [outs[0] unsignedIntValue];
      for (NSNumber* s in e[@"ANEFModelInputSymbolIndexArray"]) {
        p.in_sym.push_back(s.unsignedIntValue);
        p.inputs.push_back([syms[s.unsignedIntValue] UTF8String]);
      }
    }
  });
  was_cached = out->cached;
  compile_ms = out->cms;
  load_ms = out->lms;
}

Program::~Program() {
  if (!impl_ || !impl_->loaded) return;
  auto impl = impl_;
  try {
    on_service_queue("ANE unload", 10.0, [impl] {
      NSError* err = nil;
      [impl->api->client unloadModel:impl->model options:@{} qos:kQos error:&err];
      impl->loaded = false;
    });
  } catch (const std::exception& e) {
    logi("ANE: %s", e.what());
  }
}

int Program::proc(const std::string& n) const {
  for (size_t i = 0; i < impl_->procs.size(); ++i)
    if (impl_->procs[i].name == n) return int(i);
  return -1;
}

const std::vector<std::string>& Program::inputs(int p) const { return impl_->procs.at(p).inputs; }

std::shared_ptr<Program::Binding> Program::bind(int proc, const std::vector<const Surface*>& in, const Surface& out) const {
  const Proc& p = impl_->procs.at(proc);
  if (in.size() != p.inputs.size()) throw std::runtime_error("ANE bind: input count mismatch");
  auto b = std::make_shared<Binding>();
  @autoreleasepool {
    const Api& a = *impl_->api;
    NSMutableArray *o = [NSMutableArray array], *ix = [NSMutableArray array];
    for (size_t i = 0; i < in.size(); ++i) {
      id obj = [(Class<NiAneSurface>)a.surface objectWithIOSurface:(IOSurfaceRef)in[i]->ref];
      if (!obj) throw std::runtime_error("_ANEIOSurfaceObject returned nil");
      [o addObject:obj];
      [ix addObject:@(p.in_sym[i])];
    }
    id oo = [(Class<NiAneSurface>)a.surface objectWithIOSurface:(IOSurfaceRef)out.ref];
    if (!oo) throw std::runtime_error("_ANEIOSurfaceObject returned nil");
    b->ins = o;
    b->inIdx = ix;
    b->outs = @[ oo ];
    b->outIdx = @[ @(p.out_sym) ];
    b->proc = @(proc);
  }
  return b;
}

static id make_request(const Api& a, const Program::Binding& b, id events) {
  return [(Class<NiAneRequest>)a.request requestWithInputs:b.ins
                                               inputIndices:b.inIdx
                                                    outputs:b.outs
                                              outputIndices:b.outIdx
                                              weightsBuffer:nil
                                                  perfStats:nil
                                             procedureIndex:b.proc
                                               sharedEvents:events
                                          transactionHandle:nil];
}

double Program::evaluate(const Binding& b) const {
  const Api& a = *impl_->api;
  @autoreleasepool {
    id req = nil;
    @try {
      req = make_request(a, b, nil);
    } @catch (NSException* ex) {
      throw std::runtime_error(std::string("ANE request raised ") + ex.reason.UTF8String);
    }
    if (!req) throw std::runtime_error("ANE request creation failed");
    NSError* err = nil;
    const double t0 = now_ms();
    BOOL ok = NO;
    @try {
      ok = [a.client evaluateWithModel:impl_->model options:@{} request:req qos:kQos error:&err];
    } @catch (NSException* ex) {
      throw std::runtime_error(std::string("ANE evaluate raised ") + ex.reason.UTF8String);
    }
    if (!ok) throw std::runtime_error("ANE evaluate failed: " + err_text(err));
    return now_ms() - t0;
  }
}

void Program::enqueue(const Binding& b, void* wait_event, uint64_t wait, void* signal_event, uint64_t signal,
                      std::function<void(bool)> done) const {
  const Api& a = *impl_->api;
  @autoreleasepool {
    id wev = (__bridge id)wait_event, sev = (__bridge id)signal_event;
    // the service shares the events by their Mach ports; an event wrapped by Metal's API
    // validation layer lacks one (an Objective-C exception otherwise)
    if (![wev respondsToSelector:NSSelectorFromString(@"eventPort")] ||
        ![sev respondsToSelector:NSSelectorFromString(@"eventPort")])
      throw std::runtime_error("the Metal shared event has no eventPort (Metal API validation on?)");
    id req = nil;
    @try {
      id sig = [(Class<NiAneEvents>)a.signalEvent signalEventWithValue:signal symbolIndex:0 eventType:0 sharedEvent:sev];
      id wt = [(Class<NiAneEvents>)a.waitEvent waitEventWithValue:wait sharedEvent:wev eventType:0];
      if (!sig || !wt) throw std::runtime_error("ANE event creation failed");
      id evs = [(Class<NiAneEvents>)a.events sharedEventsWithSignalEvents:@[ sig ] waitEvents:@[ wt ]];
      if (!evs) throw std::runtime_error("ANE event creation failed");
      req = make_request(a, b, evs);
      if (!req) throw std::runtime_error("ANE request creation failed");
      // with shared events the evaluation is asynchronous and needs a completion handler
      auto once = std::make_shared<std::once_flag>();
      auto cb = std::make_shared<std::function<void(bool)>>(std::move(done));
      [req setCompletionHandler:^(BOOL success, NSError*) {
        std::call_once(*once, [&] { (*cb)(success); });
      }];
      NSError* err = nil;
      if (![a.client evaluateWithModel:impl_->model options:@{} request:req qos:kQos error:&err])
        throw std::runtime_error("ANE evaluate (async) failed: " + err_text(err));
    } @catch (NSException* ex) {
      throw std::runtime_error(std::string("ANE enqueue raised ") + (ex.reason ? ex.reason.UTF8String : "?"));
    }
  }
}

// ---- handoff ----
namespace {
enum class Phase : uint8_t { Queued, Started, Succeeded, Failed };
struct Job {
  Phase phase = Phase::Queued;
  double started = 0, ran = 0;
};
}  // namespace

struct Handoff::State {
  std::mutex mu;
  std::condition_variable changed;
  id<MTLSharedEvent> ready = nil, done = nil;
  MTLSharedEventListener* listener = nil;
  double bound_s = 2.0;
  uint64_t ready_value = 0, done_value = 0;  // highest values handed out
  std::unordered_map<uint64_t, Job> jobs;
  uint64_t last_job = 0;
  int outstanding = 0;
  bool retired = false;
  std::string reason;

  void retire_locked(const std::string& why) {
    if (!retired) {
      retired = true;
      reason = why;
    }
    if (ready.signaledValue < ready_value) ready.signaledValue = ready_value;
    if (done.signaledValue < done_value) done.signaledValue = done_value;
    changed.notify_all();
  }
};

Handoff::Handoff(double bound_s) : st_(std::make_shared<State>()) {
  st_->bound_s = bound_s;
  st_->ready = (__bridge_transfer id<MTLSharedEvent>)Gpu::get().new_shared_event();
  st_->done = (__bridge_transfer id<MTLSharedEvent>)Gpu::get().new_shared_event();
  st_->ready_value = st_->ready.signaledValue;
  st_->done_value = st_->done.signaledValue;
  st_->listener = [[MTLSharedEventListener alloc]
      initWithDispatchQueue:dispatch_queue_create("nace.ane.handoff", DISPATCH_QUEUE_SERIAL)];
}

Handoff::~Handoff() {
  std::unique_lock lk(st_->mu);
  st_->retire_locked("the handoff ended");
  st_->changed.wait_for(lk, std::chrono::duration<double>(st_->bound_s), [&] { return st_->outstanding == 0; });
}

void* Handoff::ready_event() const { return (__bridge void*)st_->ready; }
void* Handoff::done_event() const { return (__bridge void*)st_->done; }

uint64_t Handoff::next_ready() {
  std::lock_guard lk(st_->mu);
  ++st_->ready_value;
  if (st_->retired && st_->ready.signaledValue < st_->ready_value) st_->ready.signaledValue = st_->ready_value;
  return st_->ready_value;
}

uint64_t Handoff::next_done() {
  std::lock_guard lk(st_->mu);
  ++st_->done_value;
  if (st_->retired && st_->done.signaledValue < st_->done_value) st_->done.signaledValue = st_->done_value;
  return st_->done_value;
}

void Handoff::queue(uint64_t ready, uint64_t done, const Start& start) {
  auto st = st_;
  uint64_t jid = 0;
  {
    std::lock_guard lk(st->mu);
    jid = ++st->last_job;
    if (st->retired) {
      st->jobs[jid].phase = Phase::Failed;
      return;
    }
    st->jobs[jid] = Job{};
    ++st->outstanding;
  }
  auto report = [st, jid](bool ok) {
    std::lock_guard lk(st->mu);
    --st->outstanding;
    st->changed.notify_all();
    auto it = st->jobs.find(jid);
    if (it == st->jobs.end()) return;
    Job& j = it->second;
    if (j.phase == Phase::Succeeded || j.phase == Phase::Failed) return;
    if (!ok) {
      j.phase = Phase::Failed;
      st->retire_locked("a Neural Engine evaluation failed");
      return;
    }
    if (j.phase == Phase::Started) j.ran = now_ms() - j.started;
    j.phase = Phase::Succeeded;
  };
  std::string failure;
  try {
    start((__bridge void*)st->ready, ready, (__bridge void*)st->done, done, report);
    const double bound = st->bound_s;
    [st->ready notifyListener:st->listener
                      atValue:ready
                        block:^(id<MTLSharedEvent>, uint64_t) {
                          {
                            std::lock_guard lk(st->mu);
                            auto it = st->jobs.find(jid);
                            if (it == st->jobs.end() || it->second.phase != Phase::Queued) return;
                            it->second.phase = Phase::Started;
                            it->second.started = now_ms();
                          }
                          dispatch_after(dispatch_time(DISPATCH_TIME_NOW, int64_t(bound * 1e9)),
                                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                                           std::lock_guard lk(st->mu);
                                           auto it = st->jobs.find(jid);
                                           if (it != st->jobs.end() && it->second.phase == Phase::Started) {
                                             it->second.phase = Phase::Failed;
                                             st->retire_locked("a Neural Engine evaluation did not complete within " +
                                                               std::to_string(bound) + " s");
                                           }
                                         });
                        }];
    return;
  } catch (const std::exception& e) {
    failure = std::string("a Neural Engine evaluation could not start: ") + e.what();
  }
  std::lock_guard lk(st->mu);
  --st->outstanding;
  st->jobs[jid].phase = Phase::Failed;
  st->retire_locked(failure);
}

bool Handoff::finish(double* ane_ms) {
  State& s = *st_;
  std::unique_lock lk(s.mu);
  auto settled = [&] {
    for (auto& [jid, j] : s.jobs) {
      if (j.phase == Phase::Failed) return true;
      if (j.phase != Phase::Succeeded) return false;
    }
    return true;
  };
  if (!s.changed.wait_for(lk, std::chrono::duration<double>(s.bound_s), settled))
    for (auto& [jid, j] : s.jobs)
      if (j.phase != Phase::Succeeded) {
        j.phase = Phase::Failed;
        s.retire_locked("a Neural Engine evaluation did not report within " + std::to_string(s.bound_s) +
                        " s of its command");
      }
  bool ok = true;
  double ms = 0;
  for (auto& [jid, j] : s.jobs) {
    if (j.phase != Phase::Succeeded) ok = false;
    ms += j.ran;
  }
  s.jobs.clear();
  if (ane_ms) *ane_ms = ms;
  return ok;
}

void Handoff::retire(const std::string& why) {
  std::lock_guard lk(st_->mu);
  st_->retire_locked(why);
}

bool Handoff::retired() const {
  std::lock_guard lk(st_->mu);
  return st_->retired;
}

std::string Handoff::reason() const {
  std::lock_guard lk(st_->mu);
  return st_->reason;
}

}  // namespace ni::ane
