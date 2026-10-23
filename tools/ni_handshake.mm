// The per-layer GPU <-> CPU handshake of streaming a MoE's routed experts from the SSD, on Metal: the host must see
// each layer's router choice and make sure those experts are in memory before the GPU computes them. Strata's Mac
// engine (github.com/niko1221/strata, metal/src/mtl_rt.mm run_graph_seg) encodes a decode pass as one command buffer of
// 48 segments, each ended by an event signal (the router is done) and a wait (the host made the experts resident), the
// host spin-reading signaledValue; on an M5 Max its streamed 2-token window took 38 ms against 17 ms resident.
// Synthetic layers of Qwen3.8-Flash-Next's shape with 2-bit experts: no model, nothing read from disk unless --miss.
// Raw Metal rather than ni::Gpu: Encoder::signal_event commits at every signal, and the point here is one command
// buffer with waits inside it.
//   ni_handshake [options]
//   --pre <MB>       per layer, before the router: mixer, hyper-connections, router, a 48th of the LM head (default 47)
//   --shared <MB>    per layer, the shared expert (default 2.8)
//   --expert <MB>    one routed expert (default 1.27: 3 x 2560 x 640 at IQ2_XXS)
//   --topk <k>       routed experts a layer (default 10; up to 32: the union over a verify window's rows)
//   --tiny           --pre 0.06 --shared 0.06 --expert 0.06: the latency alone
//   --layers <n>     (default 48)   --slots <n>  arena slots (default 1024)
//   --passes <n>     measured passes per mode (default 60), in --rounds <n> round-robin rounds (default 6), after
//                    --warm <n> passes per mode (default 10)
//   --turn-us <us>   the host's turn also busy-waits this long (a stand-in for LRU and read-ahead bookkeeping)
//   --miss <p>       each routed expert misses the arena with probability p: it is pread (F_NOCACHE) from --file into
//                    its slot before the host signals that layer
//   --ahead <f>      a share f of the misses was predicted (default 0): those are read --depth <d> layers ahead (default
//                    1) on the I/O threads, from the host's signal of layer l - d; the turn waits for what is still
//                    reading and reads the unpredicted misses itself, a layer's in parallel
//   --file <path>    any large file for the misses (random 16 KiB-aligned offsets)
//   --only <m,...>   only these modes
// Modes (a pass = all layers; medians over the measured passes):
//   base     one command buffer, a compute encoder per segment, no events: the GPU's own time
//   presig   base + a signal and a wait per layer, the waits' values set by the CPU before the commit: the GPU side alone
//   spin     Strata's handshake: the host spin-reads signaledValue, looks the experts up, sets the value
//   listen   the same through MTLSharedEventListener blocks on a dispatch queue
//   flag     the router kernel stores a ready word that the host spin-reads; the GPU waits on the event the host sets
//   overlap  spin, with the shared expert encoded after the signal and before the wait (it runs during the host's turn)
//   cbs      a command buffer per segment; the host waits for each (spinning on its status) before it commits the next
//   gspin    no events: each experts kernel spin-reads a go word the host stores (outside Metal's memory model; the
//            spin is bounded, and a pass whose GPU gave up waiting counts as failed)
// Every pass checks that the experts kernels read the slots the host wrote in that pass ("stale": a slot from before)
// and that the host read the ids the router wrote ("ids stale"). base and presig have no misses.
// Run it under the GPU lock: lockf -k -t 600 /tmp/kapitsa-gpu.lock build/ni_handshake
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/qos.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// ids, slots and seen hold kStride words a layer (the MSL below hard-codes the same 32)
constexpr int kStride = 32;

const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Streams n4 uint4 words of weights; the loads feed a value stored only on an impossible match.
kernel void hs_stream(device const uint4* w [[buffer(0)]], device uint* sink [[buffer(1)]],
                      constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]],
                      uint gsz [[threads_per_grid]]) {
  uint4 a = 0;
  for (uint i = gid; i < n4; i += gsz) a ^= w[i];
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x9e3779b9u) sink[0] = gid;
}

uint hs_hash(uint l, uint pass, uint t) {
  uint h = (l * 0x9E3779B1u) ^ (pass * 0x85EBCA77u) ^ (t * 0xC2B2AE3Du);
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h;
}

// The router's choice for layer p.x of pass p.y: p.z ids at ids[32 p.x ..]; then ready[p.x] = pass + 1.
kernel void hs_route(device uint* ids [[buffer(0)]], device atomic_uint* ready [[buffer(1)]],
                     constant uint4& p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  if (t < p.z) ids[32 * p.x + t] = hs_hash(p.x, p.y, t) & 511u;
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) atomic_store_explicit(&ready[p.x], p.y + 1, memory_order_relaxed);
}

// The routed experts of layer p.x: threadgroup row e reads slot slots[32 p.x + e] of the arena (p.y uint4 words a
// slot) and records the slot it read in seen[32 p.x + e].
kernel void hs_experts(device const uint4* arena [[buffer(0)]], device const uint* slots [[buffer(1)]],
                       device uint* seen [[buffer(2)]], device uint* sink [[buffer(3)]],
                       constant uint4& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint2 ntg [[threadgroups_per_grid]], uint lid [[thread_index_in_threadgroup]],
                       uint2 tpg [[threads_per_threadgroup]]) {
  const uint s = slots[32 * p.x + tg.y];
  if (tg.x == 0 && lid == 0) seen[32 * p.x + tg.y] = s;
  device const uint4* w = arena + ulong(s) * p.y;
  uint4 a = 0;
  for (uint i = tg.x * tpg.x + lid; i < p.y; i += ntg.x * tpg.x) a ^= w[i];
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x9e3779b9u) sink[1] = s;
}

// hs_experts after a spin on go[p.x] >= p.w (the host's store; atomic loads, so no cached copy is reread), bounded: a
// threadgroup that gives up sets *fail and reads its slot anyway.
kernel void hs_experts_spin(device const uint4* arena [[buffer(0)]], device atomic_uint* slots [[buffer(1)]],
                            device uint* seen [[buffer(2)]], device uint* sink [[buffer(3)]],
                            constant uint4& p [[buffer(4)]], device atomic_uint* go [[buffer(5)]],
                            device atomic_uint* fail [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint2 ntg [[threadgroups_per_grid]], uint lid [[thread_index_in_threadgroup]],
                            uint2 tpg [[threads_per_threadgroup]]) {
  threadgroup uint s_tg;
  if (lid == 0) {
    uint n = 0;
    while (atomic_load_explicit(&go[p.x], memory_order_relaxed) < p.w && ++n < (1u << 20)) {}
    if (n >= (1u << 20)) atomic_store_explicit(fail, 1u, memory_order_relaxed);
    s_tg = atomic_load_explicit(&slots[32 * p.x + tg.y], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint s = s_tg;
  if (tg.x == 0 && lid == 0) seen[32 * p.x + tg.y] = s;
  device const uint4* w = arena + ulong(s) * p.y;
  uint4 a = 0;
  for (uint i = tg.x * tpg.x + lid; i < p.y; i += ntg.x * tpg.x) a ^= w[i];
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x9e3779b9u) sink[1] = s;
}

// hs_experts_spin with MSL 3.2's coherent(device) loads instead of relaxed atomics (public API; whether the host's
// stores reach the GPU's coherence point is what gspin2 measures).
kernel void hs_experts_spin2(device const uint4* arena [[buffer(0)]], volatile coherent(device) device uint* slots [[buffer(1)]],
                             device uint* seen [[buffer(2)]], device uint* sink [[buffer(3)]],
                             constant uint4& p [[buffer(4)]], volatile coherent(device) device uint* go [[buffer(5)]],
                             device atomic_uint* fail [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint2 ntg [[threadgroups_per_grid]], uint lid [[thread_index_in_threadgroup]],
                             uint2 tpg [[threads_per_threadgroup]]) {
  threadgroup uint s_tg;
  if (lid == 0) {
    uint n = 0;
    while (go[p.x] < p.w && ++n < (1u << 20)) {}
    if (n >= (1u << 20)) atomic_store_explicit(fail, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    s_tg = slots[32 * p.x + tg.y];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint s = s_tg;
  if (tg.x == 0 && lid == 0) seen[32 * p.x + tg.y] = s;
  device const uint4* w = arena + ulong(s) * p.y;
  uint4 a = 0;
  for (uint i = tg.x * tpg.x + lid; i < p.y; i += ntg.x * tpg.x) a ^= w[i];
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x9e3779b9u) sink[1] = s;
}

// The warm mode's heater: one small threadgroup of ALU work on a second queue, until the host's stop word (seen or
// not: bounded by n rounds), so the GPU never idles during a pass's waits.
kernel void hs_heat(volatile coherent(device) device uint* stop [[buffer(0)]], device uint* sink [[buffer(1)]],
                    constant uint2& p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  uint a = t;
  for (uint r = 0; r < p.x; ++r) {
    for (uint i = 0; i < 256; ++i) a = a * 1664525u + 1013904223u;
    if (stop[0] > p.y) break;  // the host stores pass + 1 when pass p.y ends (0 at start: pass 0 heats too)
  }
  if (a == 0x9e3779b9u) sink[2] = t;
}
)MSL";

struct Opt {
  double pre_mb = 47, shared_mb = 2.8, expert_mb = 1.27, turn_us = 0, miss = 0, ahead = 0;
  int layers = 48, topk = 10, slots = 1024, passes = 60, rounds = 6, warm = 10, depth = 1;
  std::string file, only;
};

struct Stats {
  std::vector<double> wall_ms, gpu_ms, turn_us, gap_us;
  long stale = 0, ids_stale = 0, timeouts = 0, gfail = 0, misses = 0, ahead = 0, errors = 0;
  double miss_ms = 0;
};

struct Miss {
  uint32_t slot;
  off_t off;
  bool ahead;  // predicted: read ahead from the host's signal of layer l - depth
};

Opt o;
id<MTLDevice> dev;
id<MTLCommandQueue> q;
id<MTLComputePipelineState> p_stream, p_route, p_experts, p_experts_spin, p_experts_spin2, p_heat;
id<MTLCommandQueue> q_heat;
uint32_t heat_rounds = 4000;
id<MTLBuffer> b_heat;
bool use_spin2 = false;
id<MTLBuffer> b_pre, b_shared, b_arena, b_ids, b_slots, b_seen, b_ready, b_go, b_sink, b_fail;
id<MTLSharedEvent> ev, ev2;
MTLSharedEventListener* listener;
dispatch_queue_t io_q;
uint64_t ev_v = 0, ev2_v = 0;
uint32_t pre_n4 = 0, shared_n4 = 0, expert_n4 = 0, pass = 0;
size_t pre_bytes = 0, shared_bytes = 0, expert_bytes = 0;
int fd = -1;
off_t fsize = 0;
std::mt19937_64 rng(7);
std::vector<uint32_t> want;  // the slots the host wrote this pass, [layer * kStride + e]
std::vector<double> t_seen, t_go;
std::vector<std::vector<Miss>> plan;    // this pass's misses, by layer
std::vector<dispatch_group_t> groups;   // a layer's read-ahead
Stats* cur = nullptr;

double now_us() { return double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e3; }

double uniform() { return std::uniform_real_distribution<double>(0, 1)(rng); }

uint32_t route_id(uint32_t l, uint32_t ps, uint32_t t) {
  uint32_t h = (l * 0x9E3779B1u) ^ (ps * 0x85EBCA77u) ^ (t * 0xC2B2AE3Du);
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h & 511u;
}

// A slot per (layer, expert, pass): every pass writes values the previous one did not, so a stale read shows.
uint32_t slot_for(int l, uint32_t id) {
  const uint64_t x = (uint64_t(l) * 512 + id) * 0x9E3779B97F4A7C15ull + uint64_t(pass) * 0xD1B54A32D192ED03ull;
  return uint32_t((x >> 32) % uint64_t(o.slots));
}

void read_expert(const Miss& m) {
  uint8_t* dst = static_cast<uint8_t*>(b_arena.contents) + size_t(m.slot) * expert_bytes;
  if (pread(fd, dst, expert_bytes, m.off) != ssize_t(expert_bytes)) std::fprintf(stderr, "ni_handshake: short read\n");
}

// The pass's misses, drawn before it starts (the router's ids are a hash the host can compute); which of them the
// read-ahead predicted is drawn too.
void plan_misses() {
  for (int l = 0; l < o.layers; ++l) {
    plan[size_t(l)].clear();
    if (fd < 0 || o.miss <= 0) continue;
    for (int e = 0; e < o.topk; ++e)
      if (uniform() < o.miss) {
        const off_t off = off_t(rng() % uint64_t(fsize - off_t(expert_bytes))) & ~off_t(16383);
        plan[size_t(l)].push_back(Miss{slot_for(l, route_id(uint32_t(l), pass, uint32_t(e))), off, uniform() < o.ahead});
      }
  }
}

// The predicted misses of layer l start reading on the I/O threads.
void read_ahead(int l) {
  if (l < 0 || l >= o.layers) return;
  for (const Miss& m : plan[size_t(l)])
    if (m.ahead) {
      const Miss mc = m;
      dispatch_group_async(groups[size_t(l)], io_q, ^{ read_expert(mc); });
    }
}

id<MTLComputePipelineState> pipe(id<MTLLibrary> lib, const char* name) {
  NSError* err = nil;
  id<MTLFunction> f = [lib newFunctionWithName:@(name)];
  id<MTLComputePipelineState> p = f ? [dev newComputePipelineStateWithFunction:f error:&err] : nil;
  if (!p) {
    std::fprintf(stderr, "ni_handshake: pipeline %s: %s\n", name, err ? err.localizedDescription.UTF8String : "missing");
    std::exit(1);
  }
  return p;
}

id<MTLBuffer> buffer(size_t bytes, bool tracked) {
  const MTLResourceOptions opt =
      MTLResourceStorageModeShared | (tracked ? MTLResourceHazardTrackingModeTracked : MTLResourceHazardTrackingModeUntracked);
  id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(bytes, 16) options:opt];
  if (!b) {
    std::fprintf(stderr, "ni_handshake: cannot allocate %.2f GB\n", double(bytes) / 1e9);
    std::exit(1);
  }
  std::memset(b.contents, 0x5a, b.length);
  return b;
}

void enc_stream(id<MTLComputeCommandEncoder> e, id<MTLBuffer> w, size_t off, uint32_t n4) {
  [e setComputePipelineState:p_stream];
  [e setBuffer:w offset:off atIndex:0];
  [e setBuffer:b_sink offset:0 atIndex:1];
  [e setBytes:&n4 length:sizeof(n4) atIndex:2];
  const uint32_t tg = std::clamp<uint32_t>((n4 + 2047) / 2048, 1, 640);
  [e dispatchThreadgroups:MTLSizeMake(tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void enc_route(id<MTLComputeCommandEncoder> e, int l) {
  const uint32_t p[4] = {uint32_t(l), pass, uint32_t(o.topk), 0};
  [e setComputePipelineState:p_route];
  [e setBuffer:b_ids offset:0 atIndex:0];
  [e setBuffer:b_ready offset:0 atIndex:1];
  [e setBytes:p length:sizeof(p) atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void enc_experts(id<MTLComputeCommandEncoder> e, int l, bool spin) {
  const uint32_t p[4] = {uint32_t(l), expert_n4, uint32_t(o.topk), pass + 1};
  [e setComputePipelineState:spin ? (use_spin2 ? p_experts_spin2 : p_experts_spin) : p_experts];
  [e setBuffer:b_arena offset:0 atIndex:0];
  [e setBuffer:b_slots offset:0 atIndex:1];
  [e setBuffer:b_seen offset:0 atIndex:2];
  [e setBuffer:b_sink offset:0 atIndex:3];
  [e setBytes:p length:sizeof(p) atIndex:4];
  if (spin) {
    [e setBuffer:b_go offset:0 atIndex:5];
    [e setBuffer:b_fail offset:0 atIndex:6];
  }
  const uint32_t gx = std::clamp<uint32_t>((expert_n4 + 2047) / 2048, 1, 64);
  [e dispatchThreadgroups:MTLSizeMake(gx, o.topk, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Segment s of a pass (0 .. layers): the experts of layer s - 1 (and its shared expert unless it was overlapped with
// the host's turn), then the work before layer s's router and the router itself.
void enc_segment(id<MTLCommandBuffer> cb, int s, bool shared_here, bool spin) {
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  if (s > 0) {
    if (shared_here) enc_stream(e, b_shared, size_t(s - 1) * shared_bytes, shared_n4);
    enc_experts(e, s - 1, spin);
  }
  if (s < o.layers) {
    enc_stream(e, b_pre, size_t(s) * pre_bytes, pre_n4);
    enc_route(e, s);
  }
  [e endEncoding];
}

// The host's turn at layer l: read the router's ids and look the experts up; the layer's misses must be in their
// slots before it returns (wait for the read-ahead, read the rest); then write the slots the experts kernel reads.
void turn(int l) {
  const uint32_t* ids = static_cast<const uint32_t*>(b_ids.contents) + kStride * l;
  uint32_t* sl = static_cast<uint32_t*>(b_slots.contents) + kStride * l;
  const std::vector<Miss>& ms = plan[size_t(l)];
  if (!ms.empty()) {
    const double t0 = now_us();
    dispatch_group_wait(groups[size_t(l)], DISPATCH_TIME_FOREVER);
    std::vector<Miss> now;
    for (const Miss& m : ms)
      if (!m.ahead) now.push_back(m);
    if (!now.empty()) {
      const Miss* np = now.data();
      dispatch_apply(now.size(), io_q, ^(size_t i) { read_expert(np[i]); });
    }
    cur->miss_ms += (now_us() - t0) / 1e3;
    cur->misses += long(ms.size());
    cur->ahead += long(ms.size() - now.size());
  }
  for (int e = 0; e < o.topk; ++e) {
    const uint32_t id = ids[e];
    if (id != route_id(uint32_t(l), pass, uint32_t(e))) ++cur->ids_stale;
    const uint32_t s = slot_for(l, id);
    sl[e] = s;
    want[size_t(kStride * l + e)] = s;
  }
  if (o.turn_us > 0) {
    const double until = now_us() + o.turn_us;
    while (now_us() < until) {}
  }
}

// All of a pass's slots before its commit (base, presig): the host computes the router's ids itself.
void turns_ahead() {
  uint32_t* sl = static_cast<uint32_t*>(b_slots.contents);
  for (int l = 0; l < o.layers; ++l)
    for (int e = 0; e < o.topk; ++e) {
      const uint32_t s = slot_for(l, route_id(uint32_t(l), pass, uint32_t(e)));
      sl[kStride * l + e] = s;
      want[size_t(kStride * l + e)] = s;
    }
}

bool failed(id<MTLCommandBuffer> cb) {
  if (cb.status == MTLCommandBufferStatusError) {
    std::fprintf(stderr, "ni_handshake: command buffer failed: %s\n", cb.error.localizedDescription.UTF8String);
    return true;
  }
  return false;
}

void run_pass(const std::string& mode, Stats* st) {
  Stats scratch;
  cur = st ? st : &scratch;
  ++pass;
  const int L = o.layers;
  std::fill(t_seen.begin(), t_seen.end(), 0.0);
  std::fill(t_go.begin(), t_go.end(), 0.0);
  std::memset(b_fail.contents, 0, 4);
  const bool handshakes = mode != "base" && mode != "presig";
  if (handshakes) plan_misses();
  else
    for (auto& m : plan) m.clear();
  double gpu_s = 0, wall_us = 0;

  if (mode == "cbs") {
    const double t0 = now_us();
    for (int l = 0; l < o.depth; ++l) read_ahead(l);
    for (int s = 0; s <= L; ++s) {
      id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
      enc_segment(cb, s, true, false);
      [cb commit];
      if (s < L) {
        while (cb.status < MTLCommandBufferStatusCompleted) {}
        t_seen[size_t(s)] = now_us();
        turn(s);
        t_go[size_t(s)] = now_us();
        read_ahead(s + o.depth);
      } else {
        [cb waitUntilCompleted];
      }
      if (failed(cb)) ++cur->errors;
      gpu_s += cb.GPUEndTime - cb.GPUStartTime;
    }
    wall_us = now_us() - t0;
  } else {
    const bool overlap = mode == "overlap", flag = mode == "flag", presig = mode == "presig",
               gspin = mode == "gspin" || mode == "gspin2", warm = mode == "warm" || getenv("HS_HEAT");
    use_spin2 = mode == "gspin2";
    id<MTLCommandBuffer> hcb = nil;
    if (warm) {
      hcb = [q_heat commandBufferWithUnretainedReferences];
      id<MTLComputeCommandEncoder> he = [hcb computeCommandEncoder];
      const uint32_t hp[2] = {heat_rounds, pass};
      [he setComputePipelineState:p_heat];
      [he setBuffer:b_heat offset:0 atIndex:0];
      [he setBuffer:b_heat offset:16 atIndex:1];
      [he setBytes:hp length:sizeof(hp) atIndex:2];
      [he dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [he endEncoding];
      [hcb commit];
    }
    const bool events = mode != "base" && !gspin;
    const uint64_t b = ev_v, b2 = ev2_v;
    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
    for (int s = 0; s <= L; ++s) {
      enc_segment(cb, s, !overlap, gspin);
      if (s == L || !events) continue;
      if (!flag) [cb encodeSignalEvent:ev value:b + 2 * uint64_t(s) + 1];
      if (overlap) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        enc_stream(e, b_shared, size_t(s) * shared_bytes, shared_n4);
        [e endEncoding];
      }
      if (presig)
        [cb encodeWaitForEvent:ev2 value:b2 + uint64_t(s) + 1];
      else
        [cb encodeWaitForEvent:ev value:b + 2 * uint64_t(s) + 2];
    }
    if (!handshakes) turns_ahead();
    if (presig) ev2.signaledValue = b2 + uint64_t(L);
    if (mode == "listen") {
      double* ts = t_seen.data();
      double* tg = t_go.data();
      const int depth = o.depth;
      for (int s = 0; s < L; ++s)
        [ev notifyListener:listener
                   atValue:b + 2 * uint64_t(s) + 1
                     block:^(id<MTLSharedEvent> e, uint64_t v) {
                       const int l = int((v - b - 1) / 2);
                       ts[l] = now_us();
                       turn(l);
                       tg[l] = now_us();
                       e.signaledValue = v + 1;
                       read_ahead(l + depth);
                     }];
    }
    for (int l = 0; l < o.depth && handshakes; ++l) read_ahead(l);
    const double t0 = now_us();
    [cb commit];
    if (mode == "spin" || overlap || flag || gspin || mode == "warm") {
      const volatile uint32_t* ready = static_cast<const volatile uint32_t*>(b_ready.contents);
      uint32_t* go = static_cast<uint32_t*>(b_go.contents);
      for (int s = 0; s < L; ++s) {
        const double ts = now_us();
        bool timed_out = false;
        if (flag || gspin) {
          for (uint32_t n = 1; __atomic_load_n(&ready[s], __ATOMIC_ACQUIRE) != pass + 1; ++n)
            if ((n & 1023) == 0 && now_us() - ts > 200e3) {
              timed_out = true;
              break;
            }
        } else {
          const uint64_t w = b + 2 * uint64_t(s) + 1;
          while (ev.signaledValue < w) {
            if (cb.status >= MTLCommandBufferStatusCompleted) break;  // an error: reported below
            if (now_us() - ts > 2e6) {
              timed_out = true;
              break;
            }
          }
        }
        if (timed_out) ++cur->timeouts;
        t_seen[size_t(s)] = now_us();
        turn(s);
        t_go[size_t(s)] = now_us();
        if (gspin)
          __atomic_store_n(&go[s], pass + 1, __ATOMIC_RELEASE);
        else
          ev.signaledValue = b + 2 * uint64_t(s) + 2;
        read_ahead(s + o.depth);
      }
    }
    [cb waitUntilCompleted];
    wall_us = now_us() - t0;
    if (warm) {
      __atomic_store_n(static_cast<uint32_t*>(b_heat.contents), pass + 1, __ATOMIC_RELEASE);
      [hcb waitUntilCompleted];
    }
    if (failed(cb)) ++cur->errors;
    gpu_s = cb.GPUEndTime - cb.GPUStartTime;
    if (events) ev_v = b + 2 * uint64_t(L) + 2;
    if (presig) ev2_v = b2 + uint64_t(L);
    if (gspin && *static_cast<uint32_t*>(b_fail.contents)) ++cur->gfail;
  }
  for (auto& g : groups) dispatch_group_wait(g, DISPATCH_TIME_FOREVER);  // read-aheads past the last layer: none

  const uint32_t* seen = static_cast<const uint32_t*>(b_seen.contents);
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < o.topk; ++e)
      if (seen[kStride * l + e] != want[size_t(kStride * l + e)]) ++cur->stale;
  cur->wall_ms.push_back(wall_us / 1e3);
  cur->gpu_ms.push_back(gpu_s * 1e3);
  if (t_go[0] > 0) {
    for (int s = 0; s < L; ++s) cur->turn_us.push_back(t_go[size_t(s)] - t_seen[size_t(s)]);
    for (int s = 0; s + 1 < L; ++s) cur->gap_us.push_back(t_seen[size_t(s + 1)] - t_go[size_t(s)]);
  }
}

double quantile(std::vector<double> v, double f) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v[size_t(std::lround(f * double(v.size() - 1)))];
}

bool arg(int& i, int argc, char** argv, const char* name) {
  if (std::strcmp(argv[i], name) != 0) return false;
  if (i + 1 >= argc) {
    std::fprintf(stderr, "ni_handshake: %s needs a value\n", name);
    std::exit(2);
  }
  ++i;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    for (int i = 1; i < argc; ++i) {
      if (arg(i, argc, argv, "--pre")) o.pre_mb = std::atof(argv[i]);
      else if (arg(i, argc, argv, "--shared")) o.shared_mb = std::atof(argv[i]);
      else if (arg(i, argc, argv, "--expert")) o.expert_mb = std::atof(argv[i]);
      else if (arg(i, argc, argv, "--topk")) o.topk = std::clamp(std::atoi(argv[i]), 1, kStride);
      else if (arg(i, argc, argv, "--layers")) o.layers = std::max(1, std::atoi(argv[i]));
      else if (arg(i, argc, argv, "--slots")) o.slots = std::max(16, std::atoi(argv[i]));
      else if (arg(i, argc, argv, "--passes")) o.passes = std::max(1, std::atoi(argv[i]));
      else if (arg(i, argc, argv, "--rounds")) o.rounds = std::max(1, std::atoi(argv[i]));
      else if (arg(i, argc, argv, "--warm")) o.warm = std::max(0, std::atoi(argv[i]));
      else if (arg(i, argc, argv, "--turn-us")) o.turn_us = std::atof(argv[i]);
      else if (arg(i, argc, argv, "--miss")) o.miss = std::atof(argv[i]);
      else if (arg(i, argc, argv, "--ahead")) o.ahead = std::atof(argv[i]);
      else if (arg(i, argc, argv, "--depth")) o.depth = std::max(1, std::atoi(argv[i]));
      else if (arg(i, argc, argv, "--file")) o.file = argv[i];
      else if (arg(i, argc, argv, "--only")) o.only = argv[i];
      else if (std::strcmp(argv[i], "--tiny") == 0) o.pre_mb = o.shared_mb = o.expert_mb = 0.06;
      else {
        std::fprintf(stderr, "ni_handshake: unknown option %s (see the comment at the top of tools/ni_handshake.mm)\n",
                     argv[i]);
        return 2;
      }
    }
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    dev = MTLCreateSystemDefaultDevice();
    q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSource) options:[MTLCompileOptions new] error:&err];
    if (!lib) {
      std::fprintf(stderr, "ni_handshake: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    p_stream = pipe(lib, "hs_stream");
    p_route = pipe(lib, "hs_route");
    p_experts = pipe(lib, "hs_experts");
    p_experts_spin = pipe(lib, "hs_experts_spin");
    p_experts_spin2 = pipe(lib, "hs_experts_spin2");
    p_heat = pipe(lib, "hs_heat");
    q_heat = [dev newCommandQueue];
    b_heat = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    std::memset(b_heat.contents, 0, 64);
    if (const char* h = getenv("HS_HEAT_ROUNDS")) heat_rounds = uint32_t(atoi(h));

    const auto bytes = [](double mb) {
      const size_t b = std::max<size_t>(size_t(mb * 1e6), 16);
      return (b + 16383) / 16384 * 16384;  // page-aligned: the misses' preads land on page boundaries
    };
    pre_bytes = bytes(o.pre_mb);
    shared_bytes = bytes(o.shared_mb);
    expert_bytes = bytes(o.expert_mb);
    pre_n4 = uint32_t(pre_bytes / 16);
    shared_n4 = uint32_t(shared_bytes / 16);
    expert_n4 = uint32_t(expert_bytes / 16);
    b_pre = buffer(pre_bytes * size_t(o.layers), false);
    b_shared = buffer(shared_bytes * size_t(o.layers), false);
    b_arena = buffer(expert_bytes * size_t(o.slots), false);
    b_ids = buffer(4 * kStride * size_t(o.layers), true);
    b_slots = buffer(4 * kStride * size_t(o.layers), true);
    b_seen = buffer(4 * kStride * size_t(o.layers), true);
    b_ready = buffer(4 * size_t(o.layers), true);
    b_go = buffer(4 * size_t(o.layers) + 64, true);
    b_sink = buffer(64, true);
    b_fail = buffer(16, true);
    std::memset(b_ready.contents, 0, b_ready.length);
    std::memset(b_go.contents, 0, b_go.length);
    if (@available(macOS 15.0, *)) {
      MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
      d.initialCapacity = 16;
      id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:d error:&err];
      for (id<MTLBuffer> b : {b_pre, b_shared, b_arena, b_ids, b_slots, b_seen, b_ready, b_go, b_sink, b_fail})
        [rs addAllocation:b];
      [rs commit];
      [rs requestResidency];
      [q addResidencySet:rs];
    }
    ev = [dev newSharedEvent];
    ev2 = [dev newSharedEvent];
    listener = [[MTLSharedEventListener alloc]
        initWithDispatchQueue:dispatch_queue_create("ni_handshake.listener",
                                                    dispatch_queue_attr_make_with_qos_class(
                                                        DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0))];
    io_q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    if (!o.file.empty()) {
      fd = open(o.file.c_str(), O_RDONLY);
      struct stat sb;
      if (fd < 0 || fstat(fd, &sb) != 0 || sb.st_size < off_t(expert_bytes) * 4) {
        std::fprintf(stderr, "ni_handshake: cannot read %s\n", o.file.c_str());
        return 1;
      }
      fsize = sb.st_size;
      fcntl(fd, F_NOCACHE, 1);
    }
    want.assign(size_t(kStride * o.layers), 0);
    t_seen.assign(size_t(o.layers), 0.0);
    t_go.assign(size_t(o.layers), 0.0);
    plan.assign(size_t(o.layers), {});
    for (int l = 0; l < o.layers; ++l) groups.push_back(dispatch_group_create());

    std::vector<std::string> modes = {"base", "presig", "spin", "listen", "flag", "overlap", "cbs", "gspin", "gspin2", "warm"};
    if (!o.only.empty()) {
      std::vector<std::string> keep;
      for (const auto& m : modes)
        if (("," + o.only + ",").find("," + m + ",") != std::string::npos) keep.push_back(m);
      modes = keep;
    }

    const double per_pass_gb =
        double(o.layers) * double(pre_bytes + shared_bytes + size_t(o.topk) * expert_bytes) / 1e9;
    std::printf("%s, macOS %s; %d layers: %.2f MB before the router, %.2f MB shared expert, %d x %.2f MB experts "
                "(%.2f GB a pass), arena %d slots (%.2f GB)%s\n",
                dev.name.UTF8String, NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String, o.layers,
                pre_bytes / 1e6, shared_bytes / 1e6, o.topk, expert_bytes / 1e6, per_pass_gb, o.slots,
                double(expert_bytes) * o.slots / 1e9, o.turn_us > 0 ? ", host turn padded" : "");
    if (fd >= 0 && o.miss > 0)
      std::printf("misses: p = %.3f (%.1f a pass) from %s; %.0f%% predicted, read %d layer(s) ahead\n", o.miss,
                  o.miss * o.topk * o.layers, o.file.c_str(), o.ahead * 100, o.depth);
    {
      const int n = 200000;
      uint64_t acc = 0;
      const double t0 = now_us();
      for (int i = 0; i < n; ++i) acc += ev.signaledValue;
      std::printf("reading signaledValue: %.3f us a read%s\n", (now_us() - t0) / n, acc == 1 ? " " : "");
    }

    std::vector<Stats> st(modes.size());
    for (size_t m = 0; m < modes.size(); ++m)
      for (int i = 0; i < o.warm; ++i) run_pass(modes[m], nullptr);
    const int per_round = std::max(1, o.passes / o.rounds);
    for (int r = 0; r < o.rounds; ++r)
      for (size_t m = 0; m < modes.size(); ++m)
        for (int i = 0; i < per_round; ++i) run_pass(modes[m], &st[m]);

    double base_ms = NAN;
    for (size_t m = 0; m < modes.size(); ++m)
      if (modes[m] == "base") base_ms = quantile(st[m].wall_ms, 0.5);
    std::printf("\n%-8s %20s %8s %9s %10s %10s %10s  %s\n", "mode", "pass ms (p10-p90)", "GPU ms", "GB/s", "us/layer",
                "turn us", "gap us", "checks");
    for (size_t m = 0; m < modes.size(); ++m) {
      const Stats& s = st[m];
      const double med = quantile(s.wall_ms, 0.5), gpu = quantile(s.gpu_ms, 0.5);
      char extra[160] = "";
      if (s.misses)
        std::snprintf(extra, sizeof(extra), "; %.1f misses a pass (%.1f read ahead), %.2f ms a pass waiting for them",
                      double(s.misses) / double(s.wall_ms.size()), double(s.ahead) / double(s.wall_ms.size()),
                      s.miss_ms / double(s.wall_ms.size()));
      std::printf("%-8s %7.2f (%5.2f-%5.2f) %8.2f %9.1f %10.1f %10.1f %10.1f  stale %ld, ids stale %ld, timeouts %ld, "
                  "gpu gave up %ld, errors %ld%s\n",
                  modes[m].c_str(), med, quantile(s.wall_ms, 0.1), quantile(s.wall_ms, 0.9), gpu, per_pass_gb / med * 1e3,
                  (med - base_ms) / o.layers * 1e3, quantile(s.turn_us, 0.5), quantile(s.gap_us, 0.5), s.stale,
                  s.ids_stale, s.timeouts, s.gfail, s.errors, extra);
    }
    std::printf("\nus/layer: (pass - base pass) / layers. turn: the host's lookup (and misses). gap: from the host's "
                "signal to the next router seen by the host. %d measured passes a mode in %d rounds, after %d warm-up.\n",
                per_round * o.rounds, o.rounds, o.warm);
  }
  return 0;
}
