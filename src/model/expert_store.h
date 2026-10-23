// Streamed routed experts: Qwen3.8-Flash-Next's Q2_0 experts (src/model/moe_q20.h) on a Mac whose memory holds only
// part of them. An arena of slots in GPU memory holds experts, each one's six planes at the offsets of q20::slot_layout,
// and a slot table per layer (E words, chunk << 28 | page) says where; the expert kernels read through it
// (q20::Experts::arena), so resident and streamed experts run the same kernels and give the same outputs.
//
// A pass encodes a handshake in each layer's MoE between its routing and its expert kernels (ExpertStore::begin / end):
// the GPU signals an event once the routing ids are written and waits for the next value. The cache's turn thread spins
// on the event, reads the ids, loads the experts missing from the arena into slots (ARC replacement; a prompt's one-time
// sweep neither promotes experts nor adapts the order, so decoding's working set stays), writes the layer's slot table
// entries and lets the GPU go on. Reads are F_NOCACHE preads of the pack's safetensors files on a pool of reader
// threads: a piece aligned to 16 KiB pages goes straight into its slot, any other is read as the whole pages around it
// into a staging buffer and copied; a read the turn waits for jumps the queue. After the go the turn starts reads of the
// next layer's predicted experts (decode and verify rows: that layer's router on this layer's input; prefill chunks of
// 64+ rows: every absent expert of the next layer), so most misses are read before they are routed. A wrong prediction
// costs a read, never an output. When the arena holds every expert it is filled once and no handshakes are encoded.
//
// ExpertCache is the CPU side (replacement, reads, the turn) over any memory and any event, so tests drive it without a
// GPU (tools/fn_store_test.cpp); ExpertStore puts it on Metal buffers and an MTLSharedEvent.
// The design follows Splash 1.4.0's streamed experts (runtime/model/ExpertStore, SlotRecycler, SsdReaders; Apache-2.0).
#pragma once
#include "metal/gpu.h"
#include "model/moe_q20.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ni::q20 {

struct StoreStats {
  uint64_t handshakes = 0;     // turns (one per streamed layer per pass)
  uint64_t lookups = 0;        // distinct routed experts looked up
  uint64_t hits = 0;           // ... resident and loaded at the turn
  uint64_t late = 0;           // ... read ahead but still being read at the turn (waited for)
  uint64_t misses = 0;         // ... read at the turn
  uint64_t prefetches = 0;     // experts read ahead
  uint64_t prefetch_used = 0;  // ... routed before they were evicted
  uint64_t evictions = 0;
  uint64_t bytes = 0;          // bytes read from the SSD (pieces as read, widened to pages included)
  double stall_ms = 0;         // the turn thread's time from a ready signal to its go (the GPU waits that long)
  double read_wait_ms = 0;     // ... of which waiting for reads
};

// The automatic store size (NI_EXPERT_STORE unset), a pure function of its inputs (bytes):
//   all: every expert; least: three layers' experts; left: what the memory budget leaves the store;
//   used / keep: the process's GPU memory so far / the budget's reserve (KV, tools); ram: hw.memsize;
//   avail: the system's available memory now (0 = unknown); cap: the process's GPU memory as a share of the RAM
//   (0 = no cap).
// Steps: min(left, all); then no more than avail less max(4 GB, RAM/12) of headroom (another resident model shrinks
// the store rather than raising memory pressure); then, for a store short of all the experts, the RAM-share cap.
// Neither guard goes below least.
struct AutoStoreIn {
  size_t all = 0, least = 0, left = 0, used = 0, keep = 0, ram = 0, avail = 0;
  double cap = 0;
};
enum class AutoStoreWhy { Budget, All, Avail, Cap };
struct AutoStore {
  size_t bytes = 0;
  AutoStoreWhy why = AutoStoreWhy::Budget;
  size_t headroom = 0;  // the avail guard's
};
inline AutoStore auto_store_size(const AutoStoreIn& in) {
  AutoStore r;
  r.bytes = std::min(in.left, in.all);
  r.why = in.left >= in.all ? AutoStoreWhy::All : AutoStoreWhy::Budget;
  r.headroom = std::max<size_t>(size_t(4) << 30, in.ram / 12);
  if (in.avail) {
    const size_t lim = std::max(in.least, in.avail > r.headroom ? in.avail - r.headroom : 0);
    if (lim < r.bytes) r.bytes = lim, r.why = AutoStoreWhy::Avail;
  }
  if (r.bytes < in.all && in.cap > 0) {
    const size_t c = size_t(in.cap * double(in.ram));
    const size_t lim = std::max(in.least, c > in.used + in.keep ? c - in.used - in.keep : 0);
    if (lim < r.bytes) r.bytes = lim, r.why = AutoStoreWhy::Cap;
  }
  return r;
}
// The system's available memory now (free, inactive and purgeable pages; free_count already includes the speculative ones), 0 if unknown. Inactive pages
// include file cache, so the automatic store size depends on what the page cache held at launch: back-to-back launches
// can pick different sizes.
size_t available_memory();

// One plane of a layer's stacked experts in a pack file: expert e's bytes at off + e * stride, len of them
struct PlaneSrc {
  int file = -1;
  uint64_t off = 0, stride = 0;
  uint32_t len = 0;
};
// A layer's six planes in q20::Experts order: gate codes, gate d, up codes, up d, down codes, down d
struct LayerSrc {
  PlaneSrc p[6];
};

class ExpertCache {
 public:
  // The arena as the CPU sees it: slot s at chunk[s / per_chunk] + (s % per_chunk) * lay.bytes; the slot tables, one
  // per layer of E words (an absent expert's word is 0: never read, the turn loads what is routed)
  struct Memory {
    uint8_t* chunk[2] = {nullptr, nullptr};
    uint32_t per_chunk = 0;
    uint32_t* table = nullptr;
  };
  // The handshake event: its value as the CPU sees it now, and setting it (the GPU waits for go values)
  struct Sync {
    std::function<uint64_t()> value;
    std::function<void(uint64_t)> set;
  };
  struct Options {
    int readers = 16;     // reader threads (NI_EXPERT_READERS)
    bool direct = false;  // NI_EXPERT_IO=direct: unaligned pieces read straight into their slot too
    bool fill = true;     // fill the arena at start (NI_EXPERT_FILL=0: no)
  };
  // One layer's handshake: the routing ids (n uint32 expert ids of `layer`, repeats allowed) are readable once the
  // event reaches `ready`; the turn sets it to `go` when every routed expert is in its slot. pred / npred: ids predicted
  // for pred_layer, read ahead after the go (pred_all: every expert of pred_layer; pred_layer -1: none). scan: a prefill
  // chunk's lookups (no promotion, no adaptation).
  struct Req {
    int layer = 0;
    const uint32_t* ids = nullptr;
    uint32_t n = 0;
    const uint32_t* pred = nullptr;
    uint32_t npred = 0;
    int pred_layer = -1;
    bool pred_all = false, scan = false;
    uint64_t ready = 0, go = 0;
  };

  // nslots slots (at least 3 E, or every expert); the files' planes must match lay
  ExpertCache(const std::vector<std::string>& files, const std::vector<LayerSrc>& layers, uint32_t E, SlotLayout lay,
              uint32_t nslots, const Memory& mem, const Sync& sync, const Options& o);
  ~ExpertCache();
  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;

  uint32_t slots() const { return nslots_; }
  uint32_t layers() const { return L_; }
  uint32_t experts() const { return E_; }
  bool complete() const { return complete_; }  // every expert in the arena: nothing to serve
  void post(const Req& r);                      // in the order the GPU will reach them
  uint32_t word(int slot) const;                // a slot's table word
  int slot_of(int layer, uint32_t e) const;     // tests: the expert's slot, -1 if absent (only while nothing is served)
  StoreStats stats() const;
  void reset_stats();
  double fill_ms() const { return fill_ms_; }
  size_t filled() const { return filled_; }

 private:
  struct Job {
    int fd = -1;
    uint64_t off = 0;
    uint32_t len = 0;
    uint8_t* dst = nullptr;
    int key = -1;
  };
  struct List {
    int head = -1, tail = -1;  // head = most recently used
    int size = 0;
  };
  enum Where : uint8_t { kNone = 0, kT1, kT2, kB1, kB2 };

  void turn_loop();
  void reader_loop();
  void serve(const Req& r);
  void wait_gpu(uint64_t value);
  void lst_remove(List& L, int k);
  void lst_push(List& L, int k);
  List& list_of(Where w);
  int lru_free(const List& L) const;  // least recent key neither pinned nor being read, -1 if none
  void hit(int k, bool scan, bool first);  // first: k was read ahead and is routed for the first time
  int admit(int k, bool scan);        // a slot for absent key k (evicting if needed); k into T1 / T2
  void evict(int k);
  void trim_ghosts();
  void load(int k, bool urgent);      // reads of k's planes into its slot
  uint8_t* slot_ptr(int slot) const;

  uint32_t L_ = 0, E_ = 0, nslots_ = 0;
  SlotLayout lay_;
  Memory mem_;
  Sync sync_;
  bool complete_ = false, direct_ = false;
  std::vector<int> fds_;
  std::vector<LayerSrc> src_;

  // replacement state (the turn thread; the constructor before it starts)
  std::vector<int32_t> slot_of_, key_of_slot_, prv_, nxt_;
  std::vector<uint8_t> where_, routed_, prefetched_, mark_;
  std::vector<uint16_t> pin_;
  std::unique_ptr<std::atomic<int>[]> pending_;  // planes of a key still being read
  std::vector<int32_t> free_;
  List t1_, t2_, b1_, b2_;
  double p_ = 0;
  std::vector<int> active_;  // the keys of the last served request (in use on the GPU until the next ready)

  std::thread turn_;
  std::vector<std::thread> readers_;
  std::mutex req_mu_, io_mu_;
  std::condition_variable req_cv_, io_cv_;
  std::deque<Req> reqs_;
  std::deque<Job> urgent_, normal_;
  std::atomic<bool> stop_{false};

  mutable std::mutex st_mu_;
  StoreStats st_;
  std::atomic<uint64_t> bytes_{0};
  double fill_ms_ = 0;
  size_t filled_ = 0;
};

// The cache on Metal: the arena (one or two buffers: the kernels address two), the slot tables, the event, and the
// handshake encoded into a pass.
class ExpertStore {
 public:
  struct Options {
    size_t bytes = 0;  // arena bytes (slots of slot_layout().bytes)
    ExpertCache::Options cache;
  };
  ExpertStore(const std::vector<std::string>& files, const std::vector<LayerSrc>& layers, uint32_t E, SlotLayout lay,
              const Options& o);
  ~ExpertStore();
  ExpertStore(const ExpertStore&) = delete;
  ExpertStore& operator=(const ExpertStore&) = delete;

  bool complete() const { return cache_->complete(); }
  uint32_t slots() const { return cache_->slots(); }
  size_t arena_bytes() const { return size_t(cache_->slots()) * lay_.bytes; }
  Experts experts(int layer) const;   // layer's experts through its slot table
  std::vector<Buffer> buffers() const;  // the arena and the slot tables (residency)
  // A layer's handshake: begin() after the dispatches that write its routing (ids: n uint32 expert ids, in GPU memory),
  // end() before its expert kernels; work encoded between them does not need the experts and overlaps the turn. See
  // ExpertCache::Req for pred / pred_layer / pred_all / scan.
  void begin(Encoder& e, int layer, const Buffer& ids, uint32_t n, const Buffer* pred, uint32_t npred, int pred_layer,
             bool pred_all, bool scan);
  void end(Encoder& e);
  StoreStats stats() const { return cache_->stats(); }
  void reset_stats() { cache_->reset_stats(); }
  std::string describe() const;

 private:
  SlotLayout lay_;
  uint32_t E_ = 0;
  Buffer chunk_[2], table_;
  void* ev_ = nullptr;  // MTLSharedEvent
  uint64_t next_value_ = 0, pending_go_ = 0;
  std::unique_ptr<ExpertCache> cache_;
};

}  // namespace ni::q20
