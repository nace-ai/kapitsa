// ane_probe: phase-0 probe for a GPU + Neural Engine split of the dense prefill FFN
// Standalone: it does not touch the engine.
//
// It drives Apple's private AppleNeuralEngine.framework: the _ANEClient / _ANEModel /
// _ANERequest selectors, MIL text programs whose weights are IOSurface-backed tensor_buffer
// inputs (not constants), one function per row count in one program, W8A8 with a Hadamard
// rotation of the intermediate inside the program, and MTLSharedEvent ordering between the GPU
// and the ANE.
//
//   ane_probe mil   [--units U] [--rows R,..]   print the MIL program only
//   ane_probe sync  [--units U] [--rows R,R,..] [--iters N] [--wfp16] [--norot]
//       compile/load one program (a function per row count), time synchronous evaluations,
//       check the output against a CPU reference.
//   ane_probe event [--units U] [--rows R] [--iters N]
//       GPU -> ANE -> GPU round trips over one MTLSharedEvent (tiny GPU kernels): handoff cost.
//   ane_probe split [--units U] [--rows R] [--iters N] [--nostage]
//       the split layer pattern: the GPU packs, signals, runs its own channels' 4-bit GEMMs
//       (nace-infer's qmm_b4_m64_n64) and stages the next weight set while the ANE evaluates,
//       then waits and joins. Reports GPU part alone, ANE alone and the split.
//
// Shapes are the Qwen3.8-27B FFN: hidden 5120, intermediate 17408 = 34 units of 512 channels;
// --units is the ANE's share (default 25 on this M4 Pro).
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "metal/gpu.h"
#include "util/log.h"

#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <sys/qos.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ni;

// ---- the private interface (selectors as resolved at runtime) ----
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

namespace {

constexpr unsigned kQos = QOS_CLASS_DEFAULT;
constexpr uint32_t kHidden = 5120, kInter = 17408, kUnit = 512, kSeg = 2560, kMaxRows = 2048, kBlock = 512;

struct Api {
  id<NiAneClient> client = nil;
  Class model = nil, surface = nil, request = nil, events = nil, signalEvent = nil, waitEvent = nil;
};

struct Sig {
  const char *cls, kind, *sel, *types;
};
// Type encodings on arm64 (digits stripped), so a changed method is refused instead of mis-called.
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

Api resolve() {
  double t0 = now_ms();
  if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW))
    fatal("AppleNeuralEngine does not load: %s", dlerror());
  for (const Sig& s : kSigs) {
    Class c = NSClassFromString(@(s.cls));
    if (!c) fatal("AppleNeuralEngine lacks %s", s.cls);
    SEL sel = sel_registerName(s.sel);
    Method m = s.kind == '+' ? class_getClassMethod(c, sel) : class_getInstanceMethod(c, sel);
    if (!m) fatal("AppleNeuralEngine lacks %c[%s %s]", s.kind, s.cls, s.sel);
    std::string t = method_getTypeEncoding(m) ?: "";
    std::erase_if(t, [](char ch) { return ch >= '0' && ch <= '9'; });
    if (t != s.types) fatal("AppleNeuralEngine changed %c[%s %s]: %s", s.kind, s.cls, s.sel, t.c_str());
  }
  Api a;
  a.model = NSClassFromString(@"_ANEModel");
  a.surface = NSClassFromString(@"_ANEIOSurfaceObject");
  a.request = NSClassFromString(@"_ANERequest");
  a.events = NSClassFromString(@"_ANESharedEvents");
  a.signalEvent = NSClassFromString(@"_ANESharedSignalEvent");
  a.waitEvent = NSClassFromString(@"_ANESharedWaitEvent");
  a.client = [(Class<NiAneClient>)NSClassFromString(@"_ANEClient") sharedConnection];
  if (!a.client) fatal("no _ANEClient shared connection");
  logi("ANE interface resolved in %.1f ms (%zu methods checked)", now_ms() - t0, std::size(kSigs));
  return a;
}

// ---- IOSurface-backed tensors shared by the GPU (as MTLBuffers) and the ANE ----
struct Surface {
  IOSurfaceRef ref = nullptr;
  Buffer buf;               // the same memory as a Metal buffer
  uint32_t rows = 0, width = 0, elem = 1, stride = 0;  // stride in bytes, rows padded to 64 B
  bool fp16 = false;
  uint8_t* ptr() const { return (uint8_t*)IOSurfaceGetBaseAddress(ref); }
  std::string mil_type() const { return fp16 ? "fp16" : "int8"; }
  // tensor_buffer type of the leading `w` elements of each of the first `r` rows
  std::string buffer_type(uint32_t r, uint32_t w) const {
    uint64_t s = stride / elem, plane = uint64_t(rows) * s;
    return "tensor_buffer<" + mil_type() + ", shape=[1, 1, " + std::to_string(r) + ", " + std::to_string(w) +
           "], strides=[" + std::to_string(plane) + ", " + std::to_string(plane) + ", " + std::to_string(s) +
           ", 1], interleave_factors=[1, 1, 1, 1]>";
  }
};

Surface make_surface(uint32_t rows, uint32_t width, bool fp16) {
  Surface s;
  s.rows = rows, s.width = width, s.fp16 = fp16, s.elem = fp16 ? 2 : 1;
  s.stride = (width * s.elem + 63) / 64 * 64;
  const size_t page = 16384, bytes = (size_t(s.stride) * rows + page - 1) / page * page;
  NSDictionary* props = @{
    (id)kIOSurfaceWidth : @(width),
    (id)kIOSurfaceHeight : @(rows),
    (id)kIOSurfaceBytesPerElement : @(s.elem),
    (id)kIOSurfaceBytesPerRow : @(s.stride),
    (id)kIOSurfaceAllocSize : @(bytes),
    (id)kIOSurfacePixelFormat : @(fp16 ? 0x4c303068 : 0x4c303038),  // 'L00h' / 'L008'
  };
  s.ref = IOSurfaceCreate((__bridge CFDictionaryRef)props);
  if (!s.ref) fatal("IOSurfaceCreate %ux%u failed", rows, width);
  if (IOSurfaceGetBytesPerRow(s.ref) != s.stride || IOSurfaceGetAllocSize(s.ref) != bytes)
    fatal("IOSurface took another stride/size (%zu, %zu)", IOSurfaceGetBytesPerRow(s.ref),
          IOSurfaceGetAllocSize(s.ref));
  s.buf = Gpu::get().wrap(IOSurfaceGetBaseAddress(s.ref), bytes, "ane surface");
  return s;
}

// ---- the MIL program ----
std::string fp16lit(double v) {
  char t[48];
  snprintf(t, sizeof t, "fp16(%a)", v);
  return t;
}
std::string dims(uint64_t r, uint64_t w) { return "[1, 1, " + std::to_string(r) + ", " + std::to_string(w) + "]"; }
std::string tens(const char* ty, uint64_t r, uint64_t w) { return std::string("tensor<") + ty + ", " + dims(r, w) + ">"; }

struct Input {
  std::string name;
  const Surface* s[2];  // per weight set
  uint32_t rows, width;  // width 0 = the function's rows
};

struct Shape {
  uint32_t C = 0;                 // ANE channels
  std::vector<uint32_t> dseg;     // down's ANE input segments
  bool wfp16 = false, norot = false;
};

// The ANE's share of a SwiGLU FFN over `R` tokens (layout [1,1,channels,tokens], tokens innermost).
// int8 values are read as code/128 so fp16 sums cannot overflow; per-row weight scales and the
// per-token input scale carry the magnitude back. silu(g) = g/2 (1 + tanh(g/2)) (the
// ANE's sigmoid is much less accurate for small gate values). The intermediate is rotated in
// 512-blocks (a grouped 1x1 conv against a constant) and quantized per token; the output is
// down's partial before its token scale plus one last row holding that scale, applied by the
// GPU join in fp32.
std::string function(const Shape& sh, const std::vector<Input>& in, const Surface& out, uint32_t R) {
  const uint32_t C = sh.C, nseg = kHidden / kSeg;
  std::string par, b;
  auto line = [&](const std::string& t) { b += "        " + t + ";\n"; };
  auto f16 = [&](const std::string& n, uint64_t r, uint64_t w, const std::string& e) {
    line(tens("fp16", r, w) + " " + n + " = " + e);
  };
  const std::string unit = fp16lit(1.0 / 128);
  for (const Input& i : in) {
    const uint32_t w = i.width ? i.width : R;
    par += (par.empty() ? "" : ", ") + i.s[0]->buffer_type(i.rows, w) + " " + i.name;
    line(tens(i.s[0]->mil_type().c_str(), i.rows, w) + " " + i.name + "_t = tensor_buffer_to_tensor<ios17>(input = " +
         i.name + ")");
  }
  auto weights = [&](const std::string& w, uint64_t r, uint64_t c) -> std::string {
    if (sh.wfp16) return w + "_t";
    f16(w + "_d", r, c, "dequantize(input = " + w + "_t, scale = " + unit + ")");
    return w + "_d";
  };
  auto matmul = [&](const std::string& n, const std::string& w, const std::string& x, uint64_t r, uint64_t k) {
    const std::string wd = weights(w, r, k);
    line(tens("fp16", r, R) + " " + n + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + wd +
         ", y = " + x + ")");
  };
  auto sum = [&](const std::string& p, size_t n, uint64_t r) {
    std::string t = p + "0";
    for (size_t k = 1; k < n; ++k) {
      std::string nx = p + "_s" + std::to_string(k);
      f16(nx, r, R, "add(x = " + t + ", y = " + p + std::to_string(k) + ")");
      t = nx;
    }
    return t;
  };
  for (uint32_t k = 0; k < nseg; ++k)
    f16("x" + std::to_string(k) + "_d", kSeg, R, "dequantize(input = x" + std::to_string(k) + "_t, scale = " + unit + ")");
  for (const char* p : {"g", "u"}) {
    for (uint32_t k = 0; k < nseg; ++k)
      matmul(std::string(p) + "m" + std::to_string(k), std::string("w") + p + std::to_string(k),
             "x" + std::to_string(k) + "_d", C, kSeg);
    f16(std::string(p) + "s", C, R, "mul(x = " + sum(std::string(p) + "m", nseg, C) + ", y = s" + p + "_t)");
  }
  f16("gt", C, R, "mul(x = gs, y = tx_t)");
  f16("gh", C, R, "mul(x = gt, y = " + fp16lit(0.5) + ")");
  f16("th", C, R, "tanh(x = gh)");
  f16("tp", C, R, "add(x = th, y = " + fp16lit(1.0) + ")");
  f16("si", C, R, "mul(x = gh, y = tp)");
  f16("h", C, R, "mul(x = si, y = us)");
  const std::string c = std::to_string(C), r = std::to_string(R);
  std::string hr = "h";
  if (!sh.norot) {
    line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c + ", 1, " + r + "]))");
    line("tensor<fp16, [" + c + ", 512, 1, 1]> rot = const()[name = string(\"rot\"), val = tensor<fp16, [" + c +
         ", 512, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]");
    line("tensor<fp16, [1, " + c + ", 1, " + r + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
         std::to_string(C / kBlock) + "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), " +
         "strides = tensor<int32, [2]>([1, 1]), weight = rot, x = h4)");
    f16("hr", C, R, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dims(C, R) + "))");
    hr = "hr";
  }
  f16("ha", C, R, "abs(x = " + hr + ")");
  f16("pk", 1, R, "reduce_max(x = ha, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("fl", 1, R, "maximum(x = pk, y = " + fp16lit(0x1p-9) + ")");
  f16("iv", 1, R, "real_div(x = " + fp16lit(127.0) + ", y = fl)");
  f16("hs", C, R, "mul(x = " + hr + ", y = iv)");
  line(tens("int8", C, R) + " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", C, R, "dequantize(input = hq, scale = " + unit + ")");
  f16("hsc", 1, R, "mul(x = fl, y = " + fp16lit(128.0 / 127.0) + ")");
  uint32_t begin = 0;
  for (size_t i = 0; i < sh.dseg.size(); ++i) {
    const uint32_t w = sh.dseg[i];
    const std::string s = "hd" + std::to_string(i);
    f16(s, w, R, "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
                     ", 0]), size = tensor<int32, [4]>(" + dims(w, R) + "))");
    matmul("dm" + std::to_string(i), "wd" + std::to_string(i), s, kHidden, w);
    begin += w;
  }
  f16("ds", kHidden, R, "mul(x = " + sum("dm", sh.dseg.size(), kHidden) + ", y = sd_t)");
  f16("yt", kHidden + 1, R, "concat(axis = int32(2), interleave = bool(false), values = (ds, hsc))");
  line(out.buffer_type(kHidden + 1, R) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), " +
       "strides = tensor<int64, [4]>([" + std::to_string(uint64_t(out.rows) * out.stride / 2) + ", " +
       std::to_string(uint64_t(out.rows) * out.stride / 2) + ", " + std::to_string(out.stride / 2) + ", 1]))");
  return "    func ffn" + std::to_string(R) + "<ios18>(" + par + ") {\n" + b + "    } -> (y);\n";
}

// Hadamard signs shared by the rotation constant and the CPU reference
std::vector<float> signs512() {
  std::mt19937 g(20261016);
  std::vector<float> s(kBlock);
  for (float& v : s) v = (g() & 1) ? -1.f : 1.f;
  return s;
}
float had(uint32_t o, uint32_t i) { return (std::popcount(o & i) & 1) ? -1.f : 1.f; }

// weights.bin with one fp16 blob at offset 64 (MIL blob storage v2: count, version, then a record
// of sentinel 0xdeadbeef, dtype 1 = fp16, byte size, data offset).
std::vector<uint8_t> rotation_blob(uint32_t C, const std::vector<float>& sg) {
  std::vector<_Float16> v(size_t(C) * kBlock);
  const float n = 1.f / std::sqrt(float(kBlock));
  for (uint32_t ch = 0; ch < C; ++ch)
    for (uint32_t i = 0; i < kBlock; ++i) {
      const uint32_t o = ch % kBlock;
      v[size_t(ch) * kBlock + i] = _Float16(sg[o] * had(o, i) * n);
    }
  std::vector<uint8_t> blob(128 + v.size() * 2);
  auto put = [&](size_t off, auto x) { memcpy(blob.data() + off, &x, sizeof x); };
  put(0, uint32_t(1));
  put(4, uint32_t(2));
  put(64, uint32_t(0xdeadbeef));
  put(68, uint32_t(1));
  put(72, uint64_t(v.size() * 2));
  put(80, uint64_t(128));
  memcpy(blob.data() + 128, v.data(), v.size() * 2);
  return blob;
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 14695981039346656037ULL) {
  for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ULL;
  return h;
}

struct Proc {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<uint32_t> in_sym;
  uint32_t out_sym = 0;
};

struct Program {
  const Api& api;
  id model = nil;
  std::vector<Proc> procs;
  double compile_ms = 0, load_ms = 0;
  bool was_cached = false;

  Program(const Api& a, const std::string& mil, const std::vector<uint8_t>& blob) : api(a) {
    const uint64_t key = fnv(blob.data(), blob.size(), fnv(mil.data(), mil.size()));
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)key);
    char base[1024];
    confstr(_CS_DARWIN_USER_CACHE_DIR, base, sizeof base);
    std::filesystem::path dir = std::filesystem::path(base) / "nace-ane-probe" / hex;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "model.mil", std::ios::binary).write(mil.data(), mil.size());
    std::ofstream(dir / "weights.bin", std::ios::binary).write((const char*)blob.data(), blob.size());
    model = [(Class<NiAneModel>)api.model modelAtURL:[NSURL fileURLWithPath:@(dir.c_str()) isDirectory:YES]
                                                key:@(hex)];
    if (!model) fatal("_ANEModel modelAtURL:key: returned nil");
    NSError* err = nil;
    double t0 = now_ms();
    was_cached = [api.client compiledModelExistsFor:model];
    if (!was_cached) {
      NSDictionary* o = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
      if (![api.client compileModel:model options:o qos:kQos error:&err])
        fatal("ANE compile failed: %s", err ? err.description.UTF8String : "?");
    }
    compile_ms = now_ms() - t0;
    t0 = now_ms();
    if (![api.client loadModel:model options:@{} qos:kQos error:&err]) {
      logi("ANE load failed (%s); purging and recompiling", err ? err.description.UTF8String : "?");
      [api.client purgeCompiledModel:model];
      NSDictionary* o = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
      err = nil;
      if (![api.client compileModel:model options:o qos:kQos error:&err])
        fatal("ANE recompile failed: %s", err ? err.description.UTF8String : "?");
      if (![api.client loadModel:model options:@{} qos:kQos error:&err])
        fatal("ANE load failed: %s", err ? err.description.UTF8String : "?");
    }
    load_ms = now_ms() - t0;
    describe();
  }
  ~Program() {
    NSError* err = nil;
    [api.client unloadModel:model options:@{} qos:kQos error:&err];
  }
  void describe() {
    NSDictionary* d = [model modelAttributes][@"ANEFModelDescription"];
    NSArray* syms = d[@"kANEFModelInputSymbolsArrayKey"];
    NSDictionary* names = d[@"kANEFModelProcedureNameToIDMapKey"];
    NSArray* entries = d[@"ANEFModelProcedures"];
    if (!syms || !names || !entries) fatal("ANE model does not describe its procedures");
    procs.resize(entries.count);
    for (NSString* n in names) procs[[names[n] unsignedIntValue]].name = n.UTF8String;
    for (NSDictionary* e in entries) {
      Proc& p = procs[[e[@"ANEFModelProcedureID"] unsignedIntValue]];
      NSArray* outs = e[@"ANEFModelOutputSymbolIndexArray"];
      if (outs.count != 1) fatal("procedure with %zu outputs", (size_t)outs.count);
      p.out_sym = [outs[0] unsignedIntValue];
      for (NSNumber* s in e[@"ANEFModelInputSymbolIndexArray"]) {
        p.in_sym.push_back(s.unsignedIntValue);
        p.inputs.push_back([syms[s.unsignedIntValue] UTF8String]);
      }
    }
  }
  uint32_t proc(const std::string& n) const {
    for (size_t i = 0; i < procs.size(); ++i)
      if (procs[i].name == n) return uint32_t(i);
    fatal("no procedure %s", n.c_str());
  }
};

// a procedure bound to fixed surfaces (one weight set)
struct Binding {
  NSArray *ins = nil, *inIdx = nil, *outs = nil, *outIdx = nil;
  NSNumber* proc = nil;
};

Binding bind_proc(const Api& api, const Program& pg, uint32_t proc, const std::vector<Input>& inputs, const Surface& out,
             int set) {
  const Proc& p = pg.procs[proc];
  NSMutableArray *o = [NSMutableArray array], *ix = [NSMutableArray array];
  for (size_t i = 0; i < p.inputs.size(); ++i) {
    auto it = std::find_if(inputs.begin(), inputs.end(), [&](const Input& x) { return x.name == p.inputs[i]; });
    if (it == inputs.end()) fatal("procedure input %s unknown", p.inputs[i].c_str());
    [o addObject:[(Class<NiAneSurface>)api.surface objectWithIOSurface:it->s[set]->ref]];
    [ix addObject:@(p.in_sym[i])];
  }
  Binding b;
  b.ins = o, b.inIdx = ix;
  b.outs = @[ [(Class<NiAneSurface>)api.surface objectWithIOSurface:out.ref] ];
  b.outIdx = @[ @(p.out_sym) ];
  b.proc = @(proc);
  return b;
}

id make_request(const Api& api, const Binding& b, id events) {
  return [(Class<NiAneRequest>)api.request requestWithInputs:b.ins
                                                 inputIndices:b.inIdx
                                                      outputs:b.outs
                                                outputIndices:b.outIdx
                                                weightsBuffer:nil
                                                    perfStats:nil
                                               procedureIndex:b.proc
                                                 sharedEvents:events
                                            transactionHandle:nil];
}

// ---- all surfaces of the split ----
struct Split {
  Shape sh;
  std::vector<Surface> x;           // int8 [2560][2048] per hidden segment
  Surface tx, y;                    // fp16 [1][2048], fp16 [5121][2048]
  struct Set {
    std::vector<Surface> wg, wu, wd;
    Surface sg, su, sd;
  } set[2];
  std::vector<Input> inputs;
};

Split make_split(uint32_t units, bool wfp16, bool norot) {
  Split s;
  s.sh.C = units * kUnit;
  s.sh.wfp16 = wfp16, s.sh.norot = norot;
  for (uint32_t b = 0; b < s.sh.C; b += kSeg) s.sh.dseg.push_back(std::min(kSeg, s.sh.C - b));
  const uint32_t nseg = kHidden / kSeg;
  for (uint32_t k = 0; k < nseg; ++k) s.x.push_back(make_surface(kSeg, kMaxRows, false));
  s.tx = make_surface(1, kMaxRows, true);
  s.y = make_surface(kHidden + 1, kMaxRows, true);
  for (auto& w : s.set) {
    for (uint32_t k = 0; k < nseg; ++k) {
      w.wg.push_back(make_surface(s.sh.C, kSeg, wfp16));
      w.wu.push_back(make_surface(s.sh.C, kSeg, wfp16));
    }
    for (uint32_t d : s.sh.dseg) w.wd.push_back(make_surface(kHidden, d, wfp16));
    w.sg = make_surface(s.sh.C, 1, true);
    w.su = make_surface(s.sh.C, 1, true);
    w.sd = make_surface(kHidden, 1, true);
  }
  for (uint32_t k = 0; k < nseg; ++k) s.inputs.push_back({"x" + std::to_string(k), {&s.x[k], &s.x[k]}, kSeg, 0});
  s.inputs.push_back({"tx", {&s.tx, &s.tx}, 1, 0});
  for (uint32_t k = 0; k < nseg; ++k) {
    s.inputs.push_back({"wg" + std::to_string(k), {&s.set[0].wg[k], &s.set[1].wg[k]}, s.sh.C, kSeg});
    s.inputs.push_back({"wu" + std::to_string(k), {&s.set[0].wu[k], &s.set[1].wu[k]}, s.sh.C, kSeg});
  }
  s.inputs.push_back({"sg", {&s.set[0].sg, &s.set[1].sg}, s.sh.C, 1});
  s.inputs.push_back({"su", {&s.set[0].su, &s.set[1].su}, s.sh.C, 1});
  for (size_t i = 0; i < s.sh.dseg.size(); ++i)
    s.inputs.push_back({"wd" + std::to_string(i), {&s.set[0].wd[i], &s.set[1].wd[i]}, kHidden, s.sh.dseg[i]});
  s.inputs.push_back({"sd", {&s.set[0].sd, &s.set[1].sd}, kHidden, 1});
  return s;
}

void fill_random(Split& s) {
  std::mt19937 g(1);
  std::uniform_int_distribution<int> q(-127, 127);
  auto i8 = [&](Surface& f, uint32_t r, uint32_t w) {
    for (uint32_t i = 0; i < r; ++i) {
      int8_t* p = (int8_t*)(f.ptr() + size_t(i) * f.stride);
      for (uint32_t j = 0; j < w; ++j) p[j] = int8_t(q(g));
    }
  };
  auto h16 = [&](Surface& f, uint32_t r, uint32_t w, float v) {
    for (uint32_t i = 0; i < r; ++i) {
      _Float16* p = (_Float16*)(f.ptr() + size_t(i) * f.stride);
      for (uint32_t j = 0; j < w; ++j) p[j] = _Float16(v);
    }
  };
  auto w16 = [&](Surface& f, uint32_t r, uint32_t w) {  // fp16 weights = code / 128
    for (uint32_t i = 0; i < r; ++i) {
      _Float16* p = (_Float16*)(f.ptr() + size_t(i) * f.stride);
      for (uint32_t j = 0; j < w; ++j) p[j] = _Float16(q(g) / 128.f);
    }
  };
  for (auto& x : s.x) i8(x, kSeg, kMaxRows);
  h16(s.tx, 1, kMaxRows, 1.0f);
  // sum over 5120 of (w/128)(x/128) has std ~ sqrt(5120)/3 ~ 24: row scales bring gate/up to O(1)
  for (auto& w : s.set) {
    for (auto& m : w.wg) s.sh.wfp16 ? w16(m, s.sh.C, kSeg) : i8(m, s.sh.C, kSeg);
    for (auto& m : w.wu) s.sh.wfp16 ? w16(m, s.sh.C, kSeg) : i8(m, s.sh.C, kSeg);
    for (size_t i = 0; i < w.wd.size(); ++i) s.sh.wfp16 ? w16(w.wd[i], kHidden, s.sh.dseg[i]) : i8(w.wd[i], kHidden, s.sh.dseg[i]);
    h16(w.sg, s.sh.C, 1, 1.f / 24);
    h16(w.su, s.sh.C, 1, 1.f / 24);
    h16(w.sd, kHidden, 1, 1.f / 16);
  }
}

// CPU reference of the program's math for tokens `ts` and output channels [0, nout), from set 0,
// without the int8 quantization of the intermediate (so the difference measures that loss plus
// the ANE's fp16 arithmetic).
double check(const Split& s, uint32_t R, const std::vector<float>& sg) {
  const uint32_t C = s.sh.C, nseg = kHidden / kSeg, nout = 256;
  auto xi = [&](uint32_t k, uint32_t t) { return float(((int8_t*)(s.x[k / kSeg].ptr() + size_t(k % kSeg) * s.x[0].stride))[t]) / 128; };
  auto wv = [&](const std::vector<Surface>& m, uint32_t row, uint32_t col, uint32_t segw) -> float {
    const Surface& f = m[col / segw];
    const uint8_t* p = f.ptr() + size_t(row) * f.stride;
    return f.fp16 ? float(((const _Float16*)p)[col % segw]) : float(((const int8_t*)p)[col % segw]) / 128;
  };
  auto scal = [&](const Surface& f, uint32_t row) { return float(*(const _Float16*)(f.ptr() + size_t(row) * f.stride)); };
  const auto& w = s.set[0];
  double num = 0, den = 0;
  const float nrm = 1.f / std::sqrt(float(kBlock));
  std::vector<uint32_t> ts = {0, R / 2, R - 1};
  for (uint32_t t : ts) {
    std::vector<float> x(kHidden), h(C), hr(C);
    for (uint32_t k = 0; k < kHidden; ++k) x[k] = xi(k, t);
    const float tx = float(((const _Float16*)s.tx.ptr())[t]);
    for (uint32_t c = 0; c < C; ++c) {
      double gsum = 0, usum = 0;
      for (uint32_t k = 0; k < kHidden; ++k) {
        gsum += double(wv(w.wg, c, k, kSeg)) * x[k];
        usum += double(wv(w.wu, c, k, kSeg)) * x[k];
      }
      const double gt = gsum * scal(w.sg, c) * tx, us = usum * scal(w.su, c);
      h[c] = float(gt / (1 + std::exp(-gt)) * us);
    }
    if (!s.sh.norot) {
      for (uint32_t b = 0; b < C; b += kBlock)
        for (uint32_t o = 0; o < kBlock; ++o) {
          double a = 0;
          for (uint32_t i = 0; i < kBlock; ++i) a += double(sg[o]) * had(o, i) * nrm * h[b + i];
          hr[b + o] = float(a);
        }
    } else {
      hr = h;
    }
    for (uint32_t o = 0; o < nout; ++o) {
      double ref = 0;
      uint32_t col = 0;
      for (size_t i = 0; i < s.sh.dseg.size(); ++i)
        for (uint32_t j = 0; j < s.sh.dseg[i]; ++j, ++col) {
          const uint8_t* p = w.wd[i].ptr() + size_t(o) * w.wd[i].stride;
          const float wd = w.wd[i].fp16 ? float(((const _Float16*)p)[j]) : float(((const int8_t*)p)[j]) / 128;
          ref += double(wd) * hr[col];
        }
      ref *= scal(w.sd, o);
      const float yv = float(((const _Float16*)(s.y.ptr() + size_t(o) * s.y.stride))[t]) *
                       float(((const _Float16*)(s.y.ptr() + size_t(kHidden) * s.y.stride))[t]);
      num += (yv - ref) * (yv - ref);
      den += ref * ref;
    }
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

struct Args {
  std::string mode = "sync";
  uint32_t units = 25, iters = 10;
  std::vector<uint32_t> rows = {2048};
  bool wfp16 = false, norot = false, nostage = false;
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) fatal("%s needs a value", k.c_str()); return argv[++i]; };
    if (k == "--units") a.units = uint32_t(std::stoul(next()));
    else if (k == "--iters") a.iters = uint32_t(std::stoul(next()));
    else if (k == "--rows") {
      a.rows.clear();
      std::string v = next();
      for (size_t p = 0; p < v.size();) {
        size_t q = v.find(',', p);
        a.rows.push_back(uint32_t(std::stoul(v.substr(p, q - p))));
        p = q == std::string::npos ? v.size() : q + 1;
      }
    } else if (k == "--wfp16") a.wfp16 = true;
    else if (k == "--norot") a.norot = true;
    else if (k == "--nostage") a.nostage = true;
    else fatal("unknown option %s", k.c_str());
  }
  if (a.units < 1 || a.units > kInter / kUnit) fatal("--units in 1..34");
  for (uint32_t r : a.rows)
    if (r < 64 || r > kMaxRows || r % 64) fatal("--rows: multiples of 64 up to 2048");
  return a;
}

double tops(uint32_t C, uint32_t R, double ms) { return 3.0 * 2 * kHidden * double(C) * R / (ms * 1e-3) / 1e12; }

// ---- small GPU kernels for the event and split modes ----
const char* kProbeKernels = R"(
#include <metal_stdlib>
using namespace metal;
// stand-in for rotate+pack: copy a pre-made int8 tile into the ANE's input surface
kernel void probe_copy(device const uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
// join: out[t][o] += y[o][t] * y[hidden][t] (token scale 1), fp32
kernel void probe_join(device const half* y [[buffer(0)]], device float* out [[buffer(1)]],
                       constant uint4& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  // p = (hidden, stride halves, rows, 0); g.x = channel, g.y = token
  if (g.x >= p.x || g.y >= p.z) return;
  float v = float(y[g.x * p.y + g.y]) * float(y[p.x * p.y + g.y]);
  out[g.y * p.x + g.x] += v;
}
// stand-in for weight staging: read 4-bit words + bf16 scale/bias, write int8 rows (no rotation)
kernel void probe_stage(device const uint* w [[buffer(0)]], device const ushort* sb [[buffer(1)]],
                        device char4* dst [[buffer(2)]], constant uint4& p [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]]) {
  // p = (rows, words per row, dst stride in char4, 0); one thread per word (8 values -> 2 char4)
  if (g.y >= p.x || g.x >= p.y) return;
  uint word = w[g.y * p.y + g.x];
  float s = as_type<float>(uint(sb[(g.y * p.y + g.x) / 8]) << 16), b = 0.05f;
  char4 lo, hi;
  for (int j = 0; j < 4; ++j) {
    lo[j] = char(clamp(rint((s * float((word >> (4 * j)) & 15u) + b) * 127.f), -127.f, 127.f));
    hi[j] = char(clamp(rint((s * float((word >> (4 * j + 16)) & 15u) + b) * 127.f), -127.f, 127.f));
  }
  dst[g.y * p.z + 2 * g.x] = lo;
  dst[g.y * p.z + 2 * g.x + 1] = hi;
}
)";

struct QmmParams {
  uint32_t M, N, K, x_ld, y_ld, add, gs, w_expert_stride, s_expert_stride;
};

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    Args a = parse(argc, argv);
    Api api = resolve();
    const std::vector<float> sg = signs512();
    Split sp = make_split(a.units, a.wfp16, a.norot);
    fill_random(sp);
    std::vector<uint32_t> rows = (a.mode == "sync" || a.mode == "mil") ? a.rows : std::vector<uint32_t>{a.rows.front()};
    std::string mil = "program(1.3)\n{\n";
    for (uint32_t r : rows) mil += function(sp.sh, sp.inputs, sp.y, r);
    mil += "}\n";
    const std::vector<uint8_t> blob = rotation_blob(sp.sh.C, sg);
    if (a.mode == "mil") {  // print the program and stop (no ANE work)
      fwrite(mil.data(), 1, mil.size(), stdout);
      return 0;
    }
    double t0 = now_ms();
    Program pg(api, mil, blob);
    logi("program: %u ANE channels (%u/34 units), %zu function(s), MIL %zu B, blob %zu B; %s %.0f ms, load %.0f ms "
         "(total %.0f ms)", sp.sh.C, a.units, rows.size(), mil.size(), blob.size(),
         pg.was_cached ? "cached, lookup" : "compile", pg.compile_ms, pg.load_ms, now_ms() - t0);

    if (a.mode == "sync") {
      for (uint32_t R : rows) {
        const uint32_t proc = pg.proc("ffn" + std::to_string(R));
        Binding b = bind_proc(api, pg, proc, sp.inputs, sp.y, 0);
        std::vector<double> ms;
        // Without shared events the evaluation is expected to be synchronous; if a completion
        // handler fires after evaluate returns, wait for it instead (and say so).
        static bool handler_seen = false, told = false;
        for (uint32_t it = 0; it < a.iters + 1; ++it) {
          id req = make_request(api, b, nil);
          struct Fired { std::atomic<bool> on{false}; std::atomic<double> at{0}; };
          auto st = std::make_shared<Fired>();
          [req setCompletionHandler:^(BOOL, NSError*) {
            st->at.store(now_ms());
            st->on.store(true);
          }];
          NSError* err = nil;
          double s0 = now_ms();
          if (![api.client evaluateWithModel:pg.model options:@{} request:req qos:kQos error:&err])
            fatal("evaluate failed: %s", err ? err.description.UTF8String : "?");
          double s1 = now_ms();
          if (!st->on.load() && (handler_seen || it == 0)) {
            const double limit = s1 + 3000;
            while (!st->on.load() && now_ms() < limit) usleep(50);
          }
          if (st->on.load() && st->at.load() > s1) {
            handler_seen = true;
            if (!told) { told = true; printf("note: evaluate returned before its completion handler; timing to the handler\n"); }
            s1 = st->at.load();
          }
          ms.push_back(s1 - s0);
        }
        const double first = ms.front();
        ms.erase(ms.begin());
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        const double err = check(sp, R, sg);
        printf("sync rows %4u ch %5u: first %.1f ms, median %.2f ms (min %.2f max %.2f) = %.1f TOPS; rel err vs CPU "
               "%.4f\n", R, sp.sh.C, first, med, ms.front(), ms.back(), tops(sp.sh.C, R, med), err);
      }
      return 0;
    }

    // event / split: one function, GPU work ordered with the ANE by one shared event
    const uint32_t R = rows.front();
    const uint32_t proc = pg.proc("ffn" + std::to_string(R));
    Binding bnd[2] = {bind_proc(api, pg, proc, sp.inputs, sp.y, 0), bind_proc(api, pg, proc, sp.inputs, sp.y, 1)};
    Gpu& g = Gpu::get();
    id<MTLDevice> dev = (__bridge id<MTLDevice>)g.device();
    NSError* err = nil;
    MTLCompileOptions* co = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kProbeKernels) options:co error:&err];
    if (!lib) fatal("probe kernels: %s", err.description.UTF8String);
    auto pso = [&](NSString* n) {
      NSError* e = nil;
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e];
      if (!p) fatal("pipeline %s", n.UTF8String);
      return p;
    };
    id<MTLComputePipelineState> pcopy = pso(@"probe_copy"), pjoin = pso(@"probe_join"), pstage = pso(@"probe_stage");
    id<MTLCommandQueue> q = [dev newCommandQueueWithMaxCommandBufferCount:512];
    id<MTLSharedEvent> ev = [dev newSharedEvent];
    if (![ev respondsToSelector:NSSelectorFromString(@"eventPort")]) fatal("shared event has no eventPort (validation layer?)");
    // source tiles for the copies
    const size_t xbytes = size_t(kSeg) * sp.x[0].stride;
    std::vector<Buffer> xsrc;
    for (auto& x : sp.x) {
      Buffer b = g.alloc(xbytes);
      memcpy(b.ptr(), x.ptr(), xbytes);
      xsrc.push_back(b);
    }
    Buffer out = g.alloc(size_t(R) * kHidden * 4);
    // GPU part of the split: its channels' 4-bit GEMMs (gate, up, swiglu, down) at the same shape
    const uint32_t G = kInter - sp.sh.C;
    Buffer Xh = g.alloc(size_t(R) * kHidden * 2);
    Buffer Wg = g.alloc(size_t(G) * kHidden / 2), Wu = g.alloc(size_t(G) * kHidden / 2), Wd = g.alloc(size_t(kHidden) * G / 2);
    Buffer Sg = g.alloc(size_t(G) * kHidden / 64 * 2), Bg = g.alloc(size_t(G) * kHidden / 64 * 2);
    Buffer Sd = g.alloc(size_t(kHidden) * G / 64 * 2), Bd = g.alloc(size_t(kHidden) * G / 64 * 2);
    Buffer Hg = g.alloc(size_t(R) * G * 4), Hu = g.alloc(size_t(R) * G * 4), Hh = g.alloc(size_t(R) * G * 2);
    {
      std::mt19937 rr(3);
      for (Buffer* b : {&Wg, &Wu, &Wd})
        for (size_t i = 0; i < b->size / 4; ++i) b->as<uint32_t>()[i] = rr();
      for (Buffer* b : {&Sg, &Bg, &Sd, &Bd})
        for (size_t i = 0; i < b->size / 2; ++i) b->as<uint16_t>()[i] = 0x3c00 /* bf16 ~ 0.0078 */ - 0x0200;
      for (size_t i = 0; i < Xh.size / 2; ++i) Xh.as<uint16_t>()[i] = 0x2e66;  // half 0.1
    }
    // the ANE's next weights in 4-bit as the staging kernel's source (gate+up+down of C channels)
    Buffer Ws = g.alloc(size_t(sp.sh.C) * kHidden * 3 / 2), Ss = g.alloc(size_t(sp.sh.C) * kHidden * 3 / 64 * 2);
    for (size_t i = 0; i < Ss.size / 2; ++i) Ss.as<uint16_t>()[i] = 0x3c00;
    Pipeline qmm = g.pipeline("qmm_b4_m64_n64"), swi = g.pipeline("swiglu_half");
    auto enc_qmm = [&](id<MTLComputeCommandEncoder> e, Buffer x, Buffer w, Buffer s, Buffer b, Buffer y, uint32_t M,
                       uint32_t N, uint32_t K, uint32_t add) {
      [e setComputePipelineState:(__bridge id<MTLComputePipelineState>)qmm.pso];
      QmmParams p{M, N, K, K, N, add, 64, 0, 0};
      [e setBuffer:(__bridge id<MTLBuffer>)x.mtl offset:0 atIndex:0];
      [e setBuffer:(__bridge id<MTLBuffer>)w.mtl offset:0 atIndex:1];
      [e setBuffer:(__bridge id<MTLBuffer>)s.mtl offset:0 atIndex:2];
      [e setBuffer:(__bridge id<MTLBuffer>)b.mtl offset:0 atIndex:3];
      [e setBuffer:(__bridge id<MTLBuffer>)y.mtl offset:0 atIndex:4];
      [e setBytes:&p length:sizeof p atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    const bool split = a.mode == "split";
    auto gpu_part = [&](id<MTLComputeCommandEncoder> e, int nextSet) {
      if (!split) return;
      enc_qmm(e, Xh, Wg, Sg, Bg, Hg, R, G, kHidden, 0);
      enc_qmm(e, Xh, Wu, Sg, Bg, Hu, R, G, kHidden, 0);
      [e setComputePipelineState:(__bridge id<MTLComputePipelineState>)swi.pso];
      uint32_t n = R * G;
      [e setBuffer:(__bridge id<MTLBuffer>)Hg.mtl offset:0 atIndex:0];
      [e setBuffer:(__bridge id<MTLBuffer>)Hu.mtl offset:0 atIndex:1];
      [e setBuffer:(__bridge id<MTLBuffer>)Hh.mtl offset:0 atIndex:2];
      [e setBytes:&n length:4 atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      enc_qmm(e, Hh, Wd, Sd, Bd, out, R, kHidden, G, 1);
      if (a.nostage || nextSet < 0) return;
      // stage the next layer's ANE weights (gate, up over 5120 inputs; down over C inputs)
      size_t off = 0;  // the 4-bit source advances like the real gate, up and down slices
      auto stage = [&](const Surface& dst, uint32_t nrows, uint32_t width) {
        [e setComputePipelineState:pstage];
        const uint32_t wpr = width / 8;
        uint32_t p[4] = {nrows, wpr, dst.stride / 4, 0};
        [e setBuffer:(__bridge id<MTLBuffer>)Ws.mtl offset:off atIndex:0];
        off += size_t(nrows) * width / 2;
        [e setBuffer:(__bridge id<MTLBuffer>)Ss.mtl offset:0 atIndex:1];
        [e setBuffer:(__bridge id<MTLBuffer>)dst.buf.mtl offset:0 atIndex:2];
        [e setBytes:&p length:sizeof p atIndex:3];
        [e dispatchThreads:MTLSizeMake(wpr, nrows, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
      };
      const auto& W = sp.set[nextSet];
      for (uint32_t k = 0; k < W.wg.size(); ++k) {
        stage(W.wg[k], sp.sh.C, kSeg);
        stage(W.wu[k], sp.sh.C, kSeg);
      }
      for (size_t i = 0; i < W.wd.size(); ++i) stage(W.wd[i], kHidden, sp.sh.dseg[i]);
    };
    auto pack = [&](id<MTLComputeCommandEncoder> e) {
      [e setComputePipelineState:pcopy];
      for (size_t k = 0; k < sp.x.size(); ++k) {
        uint32_t n = uint32_t(xbytes / 16);
        [e setBuffer:(__bridge id<MTLBuffer>)xsrc[k].mtl offset:0 atIndex:0];
        [e setBuffer:(__bridge id<MTLBuffer>)sp.x[k].buf.mtl offset:0 atIndex:1];
        [e setBytes:&n length:4 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
    };
    auto join = [&](id<MTLComputeCommandEncoder> e) {
      [e setComputePipelineState:pjoin];
      uint32_t p[4] = {kHidden, sp.y.stride / 2, R, 0};
      [e setBuffer:(__bridge id<MTLBuffer>)sp.y.buf.mtl offset:0 atIndex:0];
      [e setBuffer:(__bridge id<MTLBuffer>)out.mtl offset:0 atIndex:1];
      [e setBytes:&p length:sizeof p atIndex:2];
      [e dispatchThreads:MTLSizeMake(kHidden, R, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
    };

    // (1) the GPU part alone (pack, own channels, staging, join), (2) the ANE alone (sync),
    // (3) the split: per iteration ready = 2i+1 (GPU), done = 2i+2 (ANE)
    auto gpu_alone = [&]() {
      double s0 = now_ms();
      id<MTLCommandBuffer> last = nil;
      for (uint32_t it = 0; it < a.iters; ++it) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        pack(e);
        gpu_part(e, int((it + 1) & 1));
        join(e);
        [e endEncoding];
        [cb commit];
        last = cb;
      }
      [last waitUntilCompleted];
      return (now_ms() - s0) / a.iters;
    };
    auto ane_alone = [&]() {
      std::vector<double> ms;
      for (uint32_t it = 0; it < a.iters; ++it) {
        id req = make_request(api, bnd[it & 1], nil);
        NSError* e = nil;
        double s0 = now_ms();
        if (![api.client evaluateWithModel:pg.model options:@{} request:req qos:kQos error:&e]) fatal("evaluate failed");
        ms.push_back(now_ms() - s0);
      }
      std::sort(ms.begin(), ms.end());
      return ms[ms.size() / 2];
    };
    std::atomic<int> reportedv{0}, failedv{0};
    std::atomic<int>* reported = &reportedv;
    std::atomic<int>* failed = &failedv;
    std::vector<double> reachv(a.iters, 0), donev(a.iters, 0);
    double* reach = reachv.data();
    double* done = donev.data();
    MTLSharedEventListener* listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("probe.events", DISPATCH_QUEUE_SERIAL)];
    uint64_t base = ev.signaledValue;
    auto both = [&]() {
      double s0 = now_ms();
      id<MTLCommandBuffer> last = nil;
      for (uint32_t it = 0; it < a.iters; ++it) {
        const uint64_t ready = base + 2 * it + 1, fin = base + 2 * it + 2;
        id sig = [(Class<NiAneEvents>)api.signalEvent signalEventWithValue:fin symbolIndex:0 eventType:0 sharedEvent:ev];
        id wt = [(Class<NiAneEvents>)api.waitEvent waitEventWithValue:ready sharedEvent:ev eventType:0];
        id evs = [(Class<NiAneEvents>)api.events sharedEventsWithSignalEvents:@[ sig ] waitEvents:@[ wt ]];
        id req = make_request(api, bnd[it & 1], evs);
        const uint32_t idx = it;
        [req setCompletionHandler:^(BOOL ok, NSError*) {
          done[idx] = now_ms();
          if (!ok) failed->fetch_add(1);
          reported->fetch_add(1);
        }];
        [ev notifyListener:listener atValue:ready block:^(id<MTLSharedEvent>, uint64_t) { reach[idx] = now_ms(); }];
        NSError* e = nil;
        if (![api.client evaluateWithModel:pg.model options:@{} request:req qos:kQos error:&e])
          fatal("async evaluate failed: %s", e ? e.description.UTF8String : "?");
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        pack(en);
        [en endEncoding];
        [cb encodeSignalEvent:ev value:ready];
        [cb commit];
        id<MTLCommandBuffer> cb2 = [q commandBuffer];
        en = [cb2 computeCommandEncoder];
        gpu_part(en, int((it + 1) & 1));
        [en endEncoding];
        [cb2 encodeWaitForEvent:ev value:fin];
        en = [cb2 computeCommandEncoder];
        join(en);
        [en endEncoding];
        [cb2 commit];
        last = cb2;
      }
      [last waitUntilCompleted];
      const double per = (now_ms() - s0) / a.iters;
      while (reported->load() < int(a.iters)) usleep(1000);
      if (last.status == MTLCommandBufferStatusError) fatal("GPU error: %s", last.error.description.UTF8String);
      std::vector<double> ane;
      for (uint32_t i = 0; i < a.iters; ++i) ane.push_back(done[i] - reach[i]);
      std::sort(ane.begin(), ane.end());
      return std::pair<double, double>{per, ane[ane.size() / 2]};
    };
    // warm up each path once
    (void)gpu_alone();
    (void)ane_alone();
    (void)both();
    base = ev.signaledValue;
    reported->store(0);
    const double gpu = gpu_alone(), ane = ane_alone();
    auto [splitms, aneIn] = both();
    base = ev.signaledValue;
    reported->store(0);
    const double gpu2 = gpu_alone();
    printf("%s rows %u: ANE %u ch (%u units), GPU %u ch | GPU part alone %.2f / %.2f ms, ANE alone (sync) %.2f ms "
           "(%.1f TOPS) | both %.2f ms per layer, ANE from GPU signal to report %.2f ms | failed %d\n",
           a.mode.c_str(), R, sp.sh.C, a.units, G, gpu, gpu2, ane, tops(sp.sh.C, R, ane), splitms, aneIn, failed->load());
    return 0;
  }
}
