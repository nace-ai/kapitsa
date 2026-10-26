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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ni::q20 {

// The seam-wrap decision as pure functions, so the engine and the CPU test drive one definition instead of the test
// re-deriving the rule: a prefill chunk wraps when it is not the prompt's last - the engine sets `sample` on exactly
// that chunk, and decode follows it with one row routing top-k experts, which a whole-layer read-ahead would only
// delay. At trunk layer `li` of a pass over this Mac's trunk layers [lo, hi) - store layer li - lo - the store layer
// to read ahead is the pass's next one; at the pass's last layer it is the store's layer 0 (the next chunk's first)
// only when the chunk wraps, and nothing (-1) otherwise. Passes that never run encode_prefill's gate (decode, verify,
// a split model's later stages: a Remote::Pass carries no chunk end) pass wraps=false and never wrap; a split model's
// first Mac does wrap - it runs the prompt's chunks, so its last stage layer ends a chunk like any other.
inline bool seam_wraps(bool sample) { return !sample; }
inline int seam_next_layer(int li, int lo, int hi, bool wraps) {
  return li + 1 < hi ? li + 1 - lo : wraps ? 0 : -1;
}

// The effective NI_PREFILL_GROUP of a model (layer-major prefill groups): a group pays the streamed store's reads
// alone, so one forms only while that store is incomplete. With every expert resident - no store, or a complete
// one - the ordinary chunk path is strictly ahead: it keeps its NI_HC_PRECOMB fold and carries no per-chunk slot
// buffers. That is runtime eligibility, not an off switch: the knob still asks for G, the model just has nothing
// for a group to save. A split's share and a layer-debugging model never group either.
inline int group_chunks(int g, bool split, bool debug, bool all_resident) {
  return g >= 2 && !split && !debug && !all_resident ? g : 1;
}

// The per-chunk slot bytes of G layer-major groups: the chunks' x_ / yb_ / injA_ / injM_ slots and the two sets
// of G n-gram stagings. One formula for the sides that must agree - the plan (plan() sets MemPlan::group_slots
// to it, kept OUT of `activations`), the allocation (Qwen4Exp::alloc4_group) and the store-world charge
// (group_world below: Qwen4Exp::init_store runs it live, ni::planned_group_world predicts it before a load).
// 0 unless G >= 2.
inline size_t group_slot_bytes(int g, size_t T, size_t D, size_t hc) {
  if (g < 2) return 0;
  const size_t SD = hc * D;
  return size_t(g) * (T * SD * 4 + T * D * 4 + 2 * T * hc * 4 + 2 * T * D * 2);
}

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
//   avail: the system's available memory now (0 = unknown); kv: the KV caches the process will allocate for its
//   conversations after the store is sized (auto_store_kv); cap: the process's GPU memory as a share of the RAM
//   (0 = no cap).
// Steps: min(left, all); then no more than avail less max(4 GB, RAM/12) of headroom and the KV caches to come
// (another resident model — or this process's own next conversation — shrinks the store rather than raising memory
// pressure); then, for a store short of all the experts, the RAM-share cap. Neither guard goes below least.
struct AutoStoreIn {
  size_t all = 0, least = 0, left = 0, used = 0, keep = 0, ram = 0, avail = 0, kv = 0;
  double cap = 0;
};
enum class AutoStoreWhy { Budget, All, Avail, Cap, Pinned };  // Pinned: NI_EXPERT_STORE set the size
struct AutoStore {
  size_t bytes = 0;
  AutoStoreWhy why = AutoStoreWhy::Budget;
  size_t headroom = 0;  // the avail guard's
};
// The available-memory guard's headroom: max(4 GB, RAM/12)
inline size_t auto_store_headroom(size_t ram) { return std::max<size_t>(size_t(4) << 30, ram / 12); }
inline AutoStore auto_store_size(const AutoStoreIn& in) {
  AutoStore r;
  r.bytes = std::min(in.left, in.all);
  r.why = in.left >= in.all ? AutoStoreWhy::All : AutoStoreWhy::Budget;
  r.headroom = auto_store_headroom(in.ram);
  if (in.avail) {
    const size_t res = r.headroom + in.kv;
    const size_t lim = std::max(in.least, in.avail > res ? in.avail - res : 0);
    if (lim < r.bytes) r.bytes = lim, r.why = AutoStoreWhy::Avail;
  }
  if (r.bytes < in.all && in.cap > 0) {
    const size_t c = size_t(in.cap * double(in.ram));
    const size_t lim = std::max(in.least, c > in.used + in.keep ? c - in.used - in.keep : 0);
    if (lim < r.bytes) r.bytes = lim, r.why = AutoStoreWhy::Cap;
  }
  return r;
}
// The NI_EXPERT_STORE pin, parsed exactly as the live load parses it (Qwen4Exp::init_store): "all" is every
// expert, a number is GB of store (atof, so garbage reads 0), "auto"/empty/unset is the automatic size
// (returns false, `bytes` untouched). Any other non-empty value is the number atof reads; a negative or NaN
// reads 0 like garbage - the sign is checked before the double->size_t conversion, which is undefined below
// zero and platform-split (arm64 saturates to 0, x86-64 to an indefinite huge value that `min` would read as
// a full pin). A value at or above `all` - including the inf atof reads from "inf"/"1e999", whose conversion
// is undefined on EVERY platform - is clamped to `all` by the same pre-conversion comparison, so the
// conversion itself only ever sees a finite value inside (0, all): defined everywhere, and the same clamp the
// load applies (init_store's `o.bytes = std::min(o.bytes, all)`), so `bytes` is always a valid store size.
// One parser, shared by the live load and every pre-load
// prediction (ni::planned_group_world), so the two never diverge on what the pin means.
inline bool store_pin_bytes(const char* es, size_t all, size_t& bytes) {
  if (!es || !*es || !strcmp(es, "auto")) return false;
  if (!strcmp(es, "all")) { bytes = all; return true; }
  const double gb = atof(es) * 1e9;
  bytes = !(gb > 0.0) ? 0 : gb >= double(all) ? all : size_t(gb);
  return true;
}
// The group slots' store-world choice (PR #628) - one decision, one place, asked by every side:
// Qwen4Exp::init_store at load time (live, authoritative) and ni::planned_group_world before one (the engine's
// fit checks, `kapitsa doctor`'s displayed plan). The store is sized first WITHOUT the charge (`bare`: the
// plan's activations carry none of it and nothing has allocated slots yet); complete without it, the world
// never groups and never pays - charge 0, the bare inputs and sizing stand, and every fit number stays the
// bare one (a reservation no load allocates must not shrink a context ladder - the full default context rides
// on this). Incomplete, the slots pay for themselves: the charge is the slot term and the world is the charged
// one (group_charged_in). Monotone in one direction - more charge shrinks the budget the store gets, which
// never moves the store toward complete - so the one correction is the fixed point: no oscillation, no
// circularity.
struct GroupWorld {
  AutoStoreIn in;     // the chosen world's sizing inputs (the bare ones when charge == 0)
  AutoStore st;       // ... and the store that world sizes (st.bytes >= in.all when complete)
  size_t charge = 0;  // the group-slot bytes this world pays (0: never groups, never pays)
};
// The charged world's inputs: the bare ones with the slot bytes taken out of what is left for the store and
// added to what the process already uses (the clamps Qwen4Exp::init_store ran before this existed).
inline AutoStoreIn group_charged_in(const AutoStoreIn& in, size_t slots) {
  AutoStoreIn c = in;
  c.left = in.left > slots ? in.left - slots : 0;
  c.used = in.used + slots;
  c.avail = in.avail > slots ? in.avail - slots : 0;
  return c;
}
inline GroupWorld group_world(size_t slots, const AutoStoreIn& bare, const size_t* pin = nullptr) {
  GroupWorld w;
  w.in = bare;
  if (pin) {
    // The NI_EXPERT_STORE pin is authoritative (Qwen4Exp::init_store sizes the store to it): the store is
    // the pin's bytes (the parser already clamped to all), never re-sized by budget/avail/cap. A pin short
    // of every expert stays incomplete, so the groups pay and alloc4_group allocates the slots on top (the
    // store's size is unchanged; the sizing inputs stay the bare ones - the pin overrides them); a pin that
    // holds every expert is complete, never groups and never pays. The pin's world is chunk-invariant
    // (complete or incomplete depends only on the pin against all); only the charge's magnitude scales with
    // the plan's rows. This is what makes a short pin reserve even where the automatic store (the `bare`
    // inputs below) would come out complete.
    w.st.bytes = *pin;
    w.st.headroom = auto_store_headroom(bare.ram);
    w.st.why = AutoStoreWhy::Pinned;
    if (slots && *pin < bare.all) w.charge = slots;
    return w;
  }
  w.st = auto_store_size(w.in);
  if (slots && w.st.bytes < w.in.all) {
    w.charge = slots;
    w.in = group_charged_in(bare, slots);
    w.st = auto_store_size(w.in);
  }
  return w;
}
// The automatic size's reason, for logs and JSON: "all", "budget", "available_memory", "ram_share_cap",
// "NI_EXPERT_STORE" (the pin)
inline const char* auto_store_why_name(AutoStoreWhy w) {
  switch (w) {
    case AutoStoreWhy::All: return "all";
    case AutoStoreWhy::Avail: return "available_memory";
    case AutoStoreWhy::Cap: return "ram_share_cap";
    case AutoStoreWhy::Pinned: return "NI_EXPERT_STORE";
    case AutoStoreWhy::Budget: break;
  }
  return "budget";
}
// The automatic size and its reason in one line (Qwen4Exp::init_store logs it after "expert store: ", and `kapitsa
// doctor` prints it before any load)
inline std::string auto_store_reason(const AutoStore& st, const AutoStoreIn& in) {
  char b[512];
  const double left = double(std::min(in.left, in.all)) / 1e9;
  switch (st.why) {
    case AutoStoreWhy::All:
      snprintf(b, sizeof b, "all %.1f GB of experts fit the %.1f GB the budget leaves: resident, no SSD reads",
               double(in.all) / 1e9, double(in.left) / 1e9);
      break;
    case AutoStoreWhy::Budget:
      snprintf(b, sizeof b,
               "%.1f GB of the %.1f GB of experts, what the budget leaves (%s; the process's GPU memory %.0f%% of the "
               "RAM)",
               double(st.bytes) / 1e9, double(in.all) / 1e9,
               in.cap > 0 ? "within the RAM-share cap" : "no RAM-share cap on this chip",
               in.ram ? 100.0 * double(in.used + st.bytes) / double(in.ram) : 0.0);
      break;
    case AutoStoreWhy::Avail: {
      char res[96];
      if (in.kv)
        snprintf(res, sizeof res, "%.1f GB headroom and %.1f GB of KV caches to come", double(st.headroom) / 1e9,
                 double(in.kv) / 1e9);
      else snprintf(res, sizeof res, "%.1f GB headroom", double(st.headroom) / 1e9);
      snprintf(b, sizeof b,
               "%.1f GB rather than the %.1f GB the budget leaves (the system's available memory %.1f GB less %s: "
               "other processes hold the rest)",
               double(st.bytes) / 1e9, left, double(in.avail) / 1e9, res);
      break;
    }
    case AutoStoreWhy::Cap:
      snprintf(b, sizeof b,
               "%.1f GB rather than the %.1f GB the budget leaves (the process's GPU memory within %.0f%% of the RAM: "
               "beyond ~45%%, every command buffer of a streamed store took seconds longer on a 48 GB M4 Pro; "
               "NI_EXPERT_CAP=0 lifts it)",
               double(st.bytes) / 1e9, left, in.cap * 100);
      break;
    case AutoStoreWhy::Pinned:
      if (st.bytes >= in.all)
        snprintf(b, sizeof b, "all %.1f GB of experts, the NI_EXPERT_STORE pin: resident, no SSD reads",
                 double(in.all) / 1e9);
      else
        snprintf(b, sizeof b, "%.1f GB of the %.1f GB of experts, the NI_EXPERT_STORE pin", st.bytes / 1e9,
                 double(in.all) / 1e9);
      break;
  }
  return b;
}
// The same in a short line (at most 100 characters for any numbers below 1000 GB): `kapitsa doctor`'s styled view, a
// table row of a terminal's width; its plain text and --json keep auto_store_reason's whole sentence
inline std::string auto_store_short(const AutoStore& st, const AutoStoreIn& in) {
  char b[160];
  const double gb = double(st.bytes) / 1e9, all = double(in.all) / 1e9, left = double(std::min(in.left, in.all)) / 1e9;
  switch (st.why) {
    case AutoStoreWhy::All: snprintf(b, sizeof b, "all %.1f GB of experts resident, no SSD reads", all); break;
    case AutoStoreWhy::Budget: snprintf(b, sizeof b, "%.1f of %.1f GB of experts, what the budget leaves", gb, all); break;
    case AutoStoreWhy::Avail: {
      char res[64];
      if (in.kv) snprintf(res, sizeof res, "%.1f headroom and %.1f KV", double(st.headroom) / 1e9, double(in.kv) / 1e9);
      else snprintf(res, sizeof res, "%.1f headroom", double(st.headroom) / 1e9);
      snprintf(b, sizeof b, "%.1f GB, not the %.1f GB the budget leaves: %.1f GB available less %s", gb, left,
               double(in.avail) / 1e9, res);
      break;
    }
    case AutoStoreWhy::Cap:
      snprintf(b, sizeof b, "%.1f GB, not the %.1f GB the budget leaves: the %.0f%% RAM-share cap (NI_EXPERT_CAP=0 lifts it)",
               gb, left, in.cap * 100);
      break;
    case AutoStoreWhy::Pinned:
      if (st.bytes >= in.all) snprintf(b, sizeof b, "all %.1f GB of experts, pinned resident", all);
      else snprintf(b, sizeof b, "%.1f of %.1f GB of experts, the pin", gb, all);
      break;
  }
  return b;
}
// The RAM-share cap of a store short of all the experts: 40% of the RAM before Apple10 (M5), none on it; NI_EXPERT_CAP=
// <share of the RAM, 0..1> sets it on any chip (0 = none; anything else is logged and ignored)
double auto_store_cap(bool apple10);
// The KV window the budget's keep assumes: flash_next_model_budget keeps one conversation's KV for min(context,
// kAutoKvCtxCap) tokens, and the avail guard reserves the same per-slot window for every conversation of the plan
inline constexpr int kAutoKvCtxCap = 32768;
// The avail guard's KV reserve: kv_per_pos times the positions the conversations to come will hold. The engine's start
// plan gives them: min(planned context, kAutoKvCtxCap) times the slots — every conversation's KV grows out of the same
// available memory the store is sized against. Without a plan (planned_ctx <= 0: tools that build the model directly)
// the default budget's keep window stands, 16384 positions of one conversation (Qwen4Exp::init_store).
// NI_EXPERT_KV_POS=<positions> overrides the planned count with an explicit total, 0 = no reserve (the pre-fix size);
// a bad value is logged and ignored, as NI_EXPERT_CAP's is
size_t auto_store_kv(size_t kv_per_pos, int planned_ctx, int slots);
// The system's available memory now (free, inactive and purgeable pages; free_count already includes the speculative ones), 0 if unknown. Inactive pages
// include file cache, so the automatic store size depends on what the page cache held at launch: back-to-back launches
// can pick different sizes.
size_t available_memory();
// The available memory for the automatic size, given time to settle. A process that has just exited holds its pages
// a while longer (a killed 6 GB process: ~0.7 s, then all at once, on a 48 GB M5 Pro), and one sample taken while a
// killed Flash-Next process was still going cut the next one's store to three layers' experts (1536 slots, 2.1 GB;
// decode 10.1 tok/s against ~50 at 4K, 25 Oct 2026). So while the guard would bind (the sample less its reserve — the
// headroom and the KV caches to come — short of want, the size before the guard), sample again every step_ms, keeping
// the largest sample, until flat_ms pass
// without it growing by grow bytes (memory another process holds for good: the wait is flat_ms), or wait_ms in all.
// sample(ms) waits ms, then reads the available memory (0 = unknown: no wait, the guard is off).
struct SettledAvail {
  size_t avail = 0, first = 0;
  int waited_ms = 0;
};
inline SettledAvail settle_available(size_t want, size_t reserve, int wait_ms, int step_ms, int flat_ms, size_t grow,
                                     const std::function<size_t(int)>& sample) {
  SettledAvail r;
  r.avail = r.first = sample(0);
  size_t mark = r.avail;  // the largest sample when the flat stretch began
  int since = 0;
  while (r.avail && r.avail < want + reserve && step_ms > 0 && r.waited_ms + step_ms <= wait_ms && since < flat_ms) {
    r.waited_ms += step_ms, since += step_ms;
    r.avail = std::max(r.avail, sample(step_ms));
    if (r.avail >= mark + grow) mark = r.avail, since = 0;
  }
  return r;
}

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
    std::function<void()> warm;  // after each go: keeps the GPU busy until its next wait (Gpu::keep_warm); optional
  };
  struct Options {
    int readers = 16;     // reader threads (NI_EXPERT_READERS)
    bool direct = false;  // NI_EXPERT_IO=direct: unaligned pieces read straight into their slot too
    bool fill = true;     // fill the arena at start (NI_EXPERT_FILL=0: no)
    bool scan_victim = true;  // a sweep evicts the layers it has passed (NI_SCAN_VICTIM=0: ARC's rule, as before)
  };
  // One layer's handshake: the routing ids (n uint32 expert ids of `layer`, repeats allowed) are readable once the
  // event reaches `ready`; the turn sets it to `go` when every routed expert is in its slot. pred / npred: ids predicted
  // for pred_layer, read ahead after the go (pred_all: every expert of pred_layer; pred_layer -1: none). scan: a prefill
  // chunk's lookups (no promotion, no adaptation). pred_floor: a sweep's eviction floor — with pred_floor >= 0 the
  // read-ahead prefers victims among the T1 keys of layers below pred_floor, not below the admitted layer, so a
  // layer-major visit's turns keep their own layer's experts (its sweep for the next layer must not evict what its
  // remaining chunks still route) while any lower layer's T1 key remains resident; when none does, ARC's
  // oldest-first rule still decides and the visit's own keys can be eaten. -1 (every chunk-major turn, decode,
  // fill): the admitted layer is the floor, as before.
  struct Req {
    int layer = 0;
    const uint32_t* ids = nullptr;
    uint32_t n = 0;
    const uint32_t* pred = nullptr;
    uint32_t npred = 0;
    int pred_layer = -1;
    int pred_floor = -1;
    bool pred_all = false, scan = false;
    bool warm = false;  // a decode / verify pass's turn: Sync::warm after the go (never a prefill chunk's)
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
  int lowest_layer() const { return low_layer_; }  // tests: the lowest layer any slot holds (scan_victim's floor)
  int lowest_t1_layer() const { return low_t1_; }  // tests: the lowest layer holding a T1 key, the sweep's victims
  int t1_slots(int layer) const;                   // tests: the per-layer T1 count admit/evict/hit keep
  int t1_slots_in_table(int layer) const;          // tests: the same, recomputed from the slots and their lists
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
  // a sweep's victim: a T1 slot below the floor (floor -1: below k's layer), rotating over the slots, -1 if none
  int scan_victim(int k, int floor_layer);
  void hit(int k, bool scan, bool first);  // first: k was read ahead and is routed for the first time
  int admit(int k, bool scan, int floor_layer = -1);  // a slot for absent key k (evicting if needed); k into T1 / T2
  void evict(int k);
  void trim_ghosts();
  void load(int k, bool urgent);      // reads of k's planes into its slot
  uint8_t* slot_ptr(int slot) const;
  void t1_add(int layer);   // k into T1 (admit): the layer's T1 count up, the floor possibly down
  void t1_drop(int layer);  // out of T1 (evict, or hit's promotion to T2): the floor up when the layer empties

  uint32_t L_ = 0, E_ = 0, nslots_ = 0;
  SlotLayout lay_;
  Memory mem_;
  Sync sync_;
  bool complete_ = false, direct_ = false, scan_victim_on_ = true;
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
  uint32_t scan_cursor_ = 0;  // a scan sweep's rotating slot for its victim (admit)
  std::vector<uint32_t> layer_res_;  // resident slots per layer (admit / evict): scan_victim's "has the sweep passed it"
  int low_layer_ = 0x7fffffff;       // the lowest layer with a resident slot: scan_victim returns -1 above it
  std::vector<uint32_t> layer_t1_;   // T1 slots per layer: a sweep's victims, so a lower layer's hot T2 keys are no lap
  int low_t1_ = 0x7fffffff;          // the lowest layer with a T1 slot: scan_victim returns -1 above it, a lap when below
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
  // ExpertCache::Req for pred / pred_layer / pred_all / pred_floor / scan.
  void begin(Encoder& e, int layer, const Buffer& ids, uint32_t n, const Buffer* pred, uint32_t npred, int pred_layer,
             bool pred_all, bool scan, bool warm = false, int pred_floor = -1);
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
