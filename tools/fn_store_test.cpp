// CPU test of the streamed-expert cache (src/model/expert_store.h): no GPU, no model. A synthetic pack of stacked Q2_0
// planes is written to a temporary folder (every byte a function of layer, plane, expert and position; planes at page-
// aligned and at odd offsets, so both the direct and the widened 16 KiB reads run), an arena in plain memory (two
// chunks), and a thread plays the GPU: it posts a pass's handshakes, then for each one writes the routing, signals the
// ready value, waits for the go and checks that every routed expert's table word points at a slot holding exactly
// that expert's bytes; it checks the same slots again just before its next ready (read-ahead must not have overwritten
// an expert in use). Cases: decode rows with next-layer predictions (good and wrong ones), verify rows with repeats,
// prefill chunks (scan lookups, every next-layer expert read ahead), a chunk seam's whole-layer wrap (the next
// chunk's layer 0 read ahead during the last layer's turn), a hot set that must survive a prompt's sweep (ARC),
// direct and widened reads, an arena of every expert (no handshakes), the statistics' sums, the layer-major
// prefill group's two pure sides: the walk's shapes at the merge gate's frozen lengths (the engine's own chunker,
// no model) and the group slots' eligibility and two-pass store-sizing decision, and the NI_EXPERT_STORE pin end
// to end: the environment variable through the live load's one parse into planned_group_world, the prediction
// `kapitsa doctor` and every pre-load fit number ask (no model, no load).
//   fn_store_test [--keep DIR]
#include "engine/engine.h"
#include "model/expert_store.h"
#include "util/log.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ni;
using namespace ni::q20;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                   \
  do {                                                  \
    if (!(c)) {                                         \
      ++g_fail;                                         \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                     \
      fprintf(stderr, "\n");                            \
    }                                                   \
  } while (0)

// D 512, I 256: code planes of 32 KiB per expert (two pages: read straight into the slot from page-aligned planes,
// widened to three pages from odd offsets), d planes of 4 KiB (read as they are); a slot is 7 pages
constexpr uint32_t D = 512, I = 256;

uint8_t byte_of(uint32_t l, uint32_t plane, uint32_t e, uint32_t i) {
  uint32_t h = l * 0x9E3779B1u ^ (plane + 1) * 0x85EBCA77u ^ (e + 1) * 0xC2B2AE3Du ^ i * 0x27D4EB2Fu;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return uint8_t(h);
}

struct Pack {
  std::vector<std::string> files;
  std::vector<LayerSrc> layers;
  SlotLayout lay;
  uint32_t E = 0;
};

// L layers in two files: in file 0 each plane starts on a page (aligned); in file 1 at an odd offset (widened reads)
Pack make_pack(const std::string& dir, uint32_t L, uint32_t E) {
  Pack p;
  p.E = E;
  p.lay = slot_layout(D, I);
  const uint32_t cb = p.lay.uc, db = p.lay.ud - p.lay.gd;
  const uint32_t len[6] = {cb, db, cb, db, cb, db};
  p.files = {dir + "/a.safetensors", dir + "/b.safetensors"};
  uint64_t end[2] = {16384, 16384 + 777};
  std::vector<std::vector<uint8_t>> img(2);
  for (uint32_t l = 0; l < L; ++l) {
    LayerSrc s;
    const int f = int(l % 2);
    for (uint32_t j = 0; j < 6; ++j) {
      PlaneSrc& pl = s.p[j];
      pl.file = f;
      if (f == 0) end[0] = (end[0] + 16383) / 16384 * 16384;
      else end[1] += 13;  // keep it odd
      pl.off = end[f];
      pl.stride = len[j];
      pl.len = len[j];
      end[f] += uint64_t(len[j]) * E;
      std::vector<uint8_t>& b = img[size_t(f)];
      if (b.size() < end[f]) b.resize(end[f]);
      for (uint32_t e = 0; e < E; ++e)
        for (uint32_t i = 0; i < len[j]; ++i) b[pl.off + uint64_t(e) * len[j] + i] = byte_of(l, j, e, i);
    }
    p.layers.push_back(s);
  }
  for (int f = 0; f < 2; ++f) {
    FILE* fp = fopen(p.files[size_t(f)].c_str(), "wb");
    if (!fp || fwrite(img[size_t(f)].data(), 1, img[size_t(f)].size(), fp) != img[size_t(f)].size()) fatal("write failed");
    fclose(fp);
  }
  return p;
}

// the arena in plain memory, two chunks
struct Arena {
  std::vector<uint8_t*> mem;
  ExpertCache::Memory m;
  std::vector<uint32_t> table;
  Arena(const SlotLayout& lay, uint32_t nslots, uint32_t L, uint32_t E) {
    const uint32_t per = (nslots + 1) / 2;
    for (int c = 0; c < 2; ++c) {
      void* p = nullptr;
      if (posix_memalign(&p, 16384, size_t(per) * lay.bytes)) fatal("out of memory");
      std::memset(p, 0xEE, size_t(per) * lay.bytes);
      mem.push_back(static_cast<uint8_t*>(p));
      m.chunk[c] = mem.back();
    }
    m.per_chunk = per;
    table.assign(size_t(L) * E, 0);
    m.table = table.data();
  }
  ~Arena() {
    for (uint8_t* p : mem) free(p);
  }
};

// the slot a table word names, as the kernels compute it
const uint8_t* slot_bytes(const Arena& a, const SlotLayout& lay, uint32_t w) {
  const uint32_t c = w >> 28, page = w & 0x0FFFFFFFu;
  return a.m.chunk[c] + size_t(page) * 16384;
}

bool expert_ok(const Arena& a, const SlotLayout& lay, uint32_t l, uint32_t e, uint32_t E) {
  const uint32_t w = a.m.table[size_t(l) * E + e];
  const uint8_t* s = slot_bytes(a, lay, w);
  const uint32_t at[6] = {lay.gc, lay.gd, lay.uc, lay.ud, lay.dc, lay.dd};
  const uint32_t cb = lay.uc, db = lay.ud - lay.gd;
  const uint32_t len[6] = {cb, db, cb, db, cb, db};
  for (uint32_t j = 0; j < 6; ++j)
    for (uint32_t i = 0; i < len[j]; i += 97)
      if (s[at[j] + i] != byte_of(l, j, e, i)) return false;
  for (uint32_t j = 0; j < 6; ++j)  // the last byte of each plane too
    if (s[at[j] + len[j] - 1] != byte_of(l, j, e, len[j] - 1)) return false;
  return true;
}

struct Event {
  std::atomic<uint64_t> v{0};
  ExpertCache::Sync sync() {
    return ExpertCache::Sync{[this] { return v.load(std::memory_order_acquire); },
                             [this](uint64_t x) { v.store(x, std::memory_order_release); }};
  }
};

// one handshake as the GPU sees it
struct Step {
  int layer = 0;
  std::vector<uint32_t> ids, pred;
  int pred_layer = -1;
  bool pred_all = false, scan = false;
  int pred_floor = -1;
};

// Posts the steps (as an encoder would, a pass at a time), then plays the GPU through them; returns the checks failed
int run_pass(ExpertCache& c, Event& ev, const Arena& a, const SlotLayout& lay, uint64_t& value,
             const std::vector<Step>& steps, std::mt19937& rng) {
  int bad = 0;
  std::vector<ExpertCache::Req> rs(steps.size());
  for (size_t i = 0; i < steps.size(); ++i) {
    ExpertCache::Req& r = rs[i];
    r.layer = steps[i].layer;
    r.ids = steps[i].ids.data();
    r.n = uint32_t(steps[i].ids.size());
    r.pred = steps[i].pred.empty() ? nullptr : steps[i].pred.data();
    r.npred = uint32_t(steps[i].pred.size());
    r.pred_layer = steps[i].pred_layer;
    r.pred_all = steps[i].pred_all;
    r.scan = steps[i].scan;
    r.pred_floor = steps[i].pred_floor;
    r.ready = ++value;
    r.go = ++value;
    c.post(r);
  }
  std::vector<std::pair<int, uint32_t>> in_use;
  for (size_t i = 0; i < steps.size(); ++i) {
    // the experts of the previous handshake are still intact (no read-ahead overwrote them before this ready)
    for (auto [l, e] : in_use)
      if (!expert_ok(a, lay, uint32_t(l), e, c.experts())) ++bad;
    in_use.clear();
    ev.v.store(rs[i].ready, std::memory_order_release);
    while (ev.v.load(std::memory_order_acquire) < rs[i].go) std::this_thread::yield();
    for (uint32_t e : steps[i].ids) {
      if (!expert_ok(a, lay, uint32_t(steps[i].layer), e, c.experts())) ++bad;
      in_use.push_back({steps[i].layer, e});
    }
    if (rng() % 4 == 0) std::this_thread::sleep_for(std::chrono::microseconds(rng() % 300));  // let read-ahead land
  }
  for (auto [l, e] : in_use)
    if (!expert_ok(a, lay, uint32_t(l), e, c.experts())) ++bad;
  return bad;
}

// Hard-sync of the turn thread (#512): serve() signals go before its read-ahead admit loop (expert_store.cpp), so
// run_pass can return while admits are still in flight, and a check that reads the replacement bookkeeping (slots,
// per-layer counts, residency) races the loop still writing it. An empty handshake drains it: the turn thread serves
// requests in order, so this go proves the previous request's serve() returned, admits included. Nothing moves
// between here and the next post, so a check after a quiesce reads one settled state on both of its sides.
void quiesce(ExpertCache& c, Event& ev, uint64_t& value) {
  ExpertCache::Req r;
  r.layer = 0;
  r.ready = ++value;
  r.go = ++value;
  c.post(r);
  ev.v.store(r.ready, std::memory_order_release);
  while (ev.v.load(std::memory_order_acquire) < r.go) std::this_thread::yield();
}

std::vector<uint32_t> topk(std::mt19937& rng, uint32_t E, uint32_t k, const std::vector<uint32_t>* hot = nullptr) {
  std::vector<uint32_t> v;
  while (v.size() < k) {
    const uint32_t e = hot && rng() % 8 ? (*hot)[rng() % hot->size()] : uint32_t(rng() % E);
    if (std::find(v.begin(), v.end(), e) == v.end()) v.push_back(e);
  }
  return v;
}

// cross_step: the window is one step of a pass whose read-ahead crosses the step boundary (the admit lands in the
// next step's window), so the routed-of-read-ahead invariant holds over the caller's whole accumulation, not here.
void check_sums(const StoreStats& s, const char* what, bool cross_step = false) {
  CHECK(s.hits + s.late + s.misses == s.lookups, "%s: hits %llu + late %llu + misses %llu != lookups %llu", what,
        (unsigned long long)s.hits, (unsigned long long)s.late, (unsigned long long)s.misses,
        (unsigned long long)s.lookups);
  if (!cross_step)
    CHECK(s.prefetch_used <= s.prefetches, "%s: %llu read-ahead experts routed of %llu read ahead", what,
          (unsigned long long)s.prefetch_used, (unsigned long long)s.prefetches);
}

// The settled available memory (q20::settle_available): a process that just exited frees its pages a while later
void case_settle() {
  const size_t GB = 1000000000ull, head = size_t(4) << 30, want = size_t(29.9 * GB), grow = size_t(500) << 20;
  int t = 0, calls = 0;
  auto run = [&](int wait, std::function<size_t(int)> f) {
    t = calls = 0;
    return settle_available(want, head, wait, 250, 3000, grow, [&](int ms) { t += ms, ++calls; return f(t); });
  };
  // freed all at once 2.5 s in (2 GB available before, 38 GB after): waits 2.5 s, then the whole budget, no more
  SettledAvail r = run(10000, [&](int at) { return at >= 2500 ? size_t(38 * GB) : size_t(2 * GB); });
  CHECK(r.first == size_t(2 * GB) && r.avail == size_t(38 * GB) && r.waited_ms == 2500 && calls == 11,
        "settle: freed %zu -> %zu after %d ms, %d samples", r.first, r.avail, r.waited_ms, calls);
  CHECK(auto_store_headroom(size_t(48) << 30) == head && auto_store_headroom(size_t(128) << 30) == (size_t(128) << 30) / 12,
        "settle: headroom");
  // freed a little at a time (1 GB a second up to 30 GB): follows it while it grows, then 3 s flat
  r = run(60000, [&](int at) { return std::min(size_t(30 * GB), size_t(2 * GB) + size_t(at / 1000) * GB); });
  CHECK(r.avail == size_t(30 * GB) && r.waited_ms == 28000 + 3000, "settle: gradual %zu after %d ms", r.avail, r.waited_ms);
  // ... but no longer than the whole wait
  r = run(10000, [&](int at) { return size_t(2 * GB) + size_t(at / 1000) * GB; });
  CHECK(r.avail == size_t(12 * GB) && r.waited_ms == 10000, "settle: capped %zu after %d ms", r.avail, r.waited_ms);
  // enough at once: one sample, no wait
  r = run(10000, [&](int) { return size_t(40 * GB); });
  CHECK(r.waited_ms == 0 && calls == 1 && r.avail == size_t(40 * GB), "settle: enough %d ms %d samples", r.waited_ms, calls);
  // another model resident (13 GB available, for good): 3 s flat, keeps 13 GB, so the guard still binds
  r = run(10000, [&](int) { return size_t(13 * GB); });
  CHECK(r.waited_ms == 3000 && calls == 13 && r.avail == size_t(13 * GB), "settle: resident %d ms %d samples", r.waited_ms,
        calls);
  // small wobbles (under 0.5 GB) do not restart the flat stretch; the largest sample is kept
  r = run(10000, [&](int at) { return size_t(13 * GB) + (at % 500 ? size_t(100) << 20 : 0); });
  CHECK(r.waited_ms == 3000 && r.avail == size_t(13 * GB) + (size_t(100) << 20), "settle: wobble %d ms %zu", r.waited_ms,
        r.avail);
  // unknown (0) or no wait allowed: one sample
  r = run(10000, [&](int) { return size_t(0); });
  CHECK(r.avail == 0 && r.waited_ms == 0 && calls == 1, "settle: unknown");
  r = run(0, [&](int) { return size_t(2 * GB); });
  CHECK(r.avail == size_t(2 * GB) && r.waited_ms == 0 && calls == 1, "settle: NI_EXPERT_SETTLE_MS=0");
}

// The automatic store size (q20::auto_store_size) with Flash-Next's numbers on a 48 GB Mac: 24576 experts of 1.39 MB
// (34.2 GB), the budget leaving 29.9 GB, 5.3 GB of GPU memory in use, a 1.5 GB reserve.
void case_auto_size() {
  const size_t GB = 1000000000ull, slot = 1392640, ram = size_t(48) << 30;
  AutoStoreIn in;
  in.all = 24576 * slot, in.least = 3 * 512 * slot, in.left = size_t(29.9 * GB), in.used = size_t(5.3 * GB);
  in.keep = size_t(1.5 * GB), in.ram = ram, in.avail = size_t(40 * GB), in.cap = 0;
  AutoStore r = auto_store_size(in);  // Apple10: no cap, nothing else resident -> what the budget leaves
  CHECK(r.why == AutoStoreWhy::Budget && r.bytes == in.left, "auto: M5 budget %zu why %d", r.bytes, int(r.why));
  CHECK(r.headroom == size_t(4) << 30, "auto: headroom %zu", r.headroom);
  in.cap = 0.40;  // before Apple10: 40% of 48 GiB = 20.6 GB less used and reserve -> 13.8 GB
  r = auto_store_size(in);
  CHECK(r.why == AutoStoreWhy::Cap && r.bytes == size_t(0.40 * double(ram)) - in.used - in.keep, "auto: cap %zu",
        r.bytes);
  in.cap = 0, in.avail = size_t(13 * GB);  // a 17.8 GB model resident: 13 GB available -> 8.7 GB
  r = auto_store_size(in);
  CHECK(r.why == AutoStoreWhy::Avail && r.bytes == in.avail - (size_t(4) << 30), "auto: avail %zu", r.bytes);
  in.avail = size_t(5 * GB);  // nearly nothing available: no less than three layers' experts
  r = auto_store_size(in);
  CHECK(r.why == AutoStoreWhy::Avail && r.bytes == in.least, "auto: least %zu", r.bytes);
  in.avail = 0;  // unknown: the guard is off
  r = auto_store_size(in);
  CHECK(r.why == AutoStoreWhy::Budget && r.bytes == in.left, "auto: avail unknown %zu", r.bytes);
  in.left = size_t(60 * GB), in.avail = size_t(70 * GB), in.cap = 0.40;  // a 96 GB-class budget: every expert,
  r = auto_store_size(in);                                               // the cap never applies to a full store
  CHECK(r.why == AutoStoreWhy::All && r.bytes == in.all, "auto: all %zu why %d", r.bytes, int(r.why));
  in.avail = size_t(30 * GB);  // ... unless memory is short: then the store streams, and the cap applies again
  r = auto_store_size(in);
  CHECK(r.bytes < in.all && r.why == AutoStoreWhy::Cap, "auto: all, short memory %zu why %d", r.bytes, int(r.why));
  in.ram = size_t(128) << 30, in.cap = 0.40;
  r = auto_store_size(in);
  CHECK(r.headroom == (size_t(128) << 30) / 12 && r.why == AutoStoreWhy::Avail, "auto: 128 GB headroom %zu why %d",
        r.headroom, int(r.why));
  // the reason line (the engine's log and kapitsa doctor print it) and its JSON name, one per reason
  in = AutoStoreIn{};
  in.all = 24576 * slot, in.least = 3 * 512 * slot, in.left = size_t(29.9 * GB), in.used = size_t(5.3 * GB);
  in.keep = size_t(1.5 * GB), in.ram = ram, in.avail = size_t(40 * GB), in.cap = 0;
  r = auto_store_size(in);
  std::string t = auto_store_reason(r, in);
  CHECK(t == "29.9 GB of the 34.2 GB of experts, what the budget leaves (no RAM-share cap on this chip; the process's "
             "GPU memory 68% of the RAM)",
        "auto: budget text '%s'", t.c_str());
  CHECK(!strcmp(auto_store_why_name(r.why), "budget"), "auto: budget name");
  in.cap = 0.40, r = auto_store_size(in), t = auto_store_reason(r, in);
  CHECK(t.rfind("13.8 GB rather than the 29.9 GB the budget leaves (the process's GPU memory within 40% of the RAM", 0) == 0 &&
            !strcmp(auto_store_why_name(r.why), "ram_share_cap"),
        "auto: cap text '%s'", t.c_str());
  in.cap = 0, in.avail = size_t(13 * GB), r = auto_store_size(in), t = auto_store_reason(r, in);
  CHECK(t == "8.7 GB rather than the 29.9 GB the budget leaves (the system's available memory 13.0 GB less 4.3 GB "
             "headroom: other processes hold the rest)" &&
            !strcmp(auto_store_why_name(r.why), "available_memory"),
        "auto: avail text '%s'", t.c_str());
  in.avail = 0, in.left = size_t(60 * GB), r = auto_store_size(in), t = auto_store_reason(r, in);
  CHECK(t == "all 34.2 GB of experts fit the 60.0 GB the budget leaves: resident, no SSD reads" &&
            !strcmp(auto_store_why_name(r.why), "all"),
        "auto: all text '%s'", t.c_str());
  // the short line (doctor's styled view, after "expert store: "): within 100 characters for every reason, even with
  // 999.9 GB figures and a 100% cap, so the row fits 120 columns unclipped
  t = auto_store_short(r, in);
  CHECK(t == "all 34.2 GB of experts resident, no SSD reads", "auto: all short '%s'", t.c_str());
  in.left = size_t(29.9 * GB), in.avail = size_t(13 * GB), r = auto_store_size(in), t = auto_store_short(r, in);
  CHECK(t == "8.7 GB, not the 29.9 GB the budget leaves: 13.0 GB available less 4.3 headroom", "auto: avail short '%s'",
        t.c_str());
  for (AutoStoreWhy w : {AutoStoreWhy::All, AutoStoreWhy::Budget, AutoStoreWhy::Avail, AutoStoreWhy::Cap,
                         AutoStoreWhy::Pinned}) {
    AutoStoreIn big;
    big.all = big.left = big.avail = big.ram = size_t(999.9 * GB), big.cap = 1.0;
    AutoStore b;
    b.bytes = b.headroom = size_t(999.9 * GB), b.why = w;
    const std::string sh = auto_store_short(b, big), lg = auto_store_reason(b, big);
    CHECK(sh.size() <= 100 && sh.size() < lg.size(), "auto: short %s %zu chars: '%s'", auto_store_why_name(w),
          sh.size(), sh.c_str());
  }
  printf("auto store size: ok\n");
}

void case_decode(const Pack& p, bool direct) {
  const uint32_t L = uint32_t(p.layers.size()), E = p.E, k = 8;
  const uint32_t nslots = 3 * E + 17;  // well below the L * E experts: evictions on most turns
  Arena a(p.lay, nslots, L, E);
  Event ev;
  ExpertCache::Options o;
  o.readers = 6;
  o.direct = direct;
  ExpertCache c(p.files, p.layers, E, p.lay, nslots, a.m, ev.sync(), o);
  CHECK(!c.complete(), "decode: complete");
  std::mt19937 rng(direct ? 7 : 3);
  uint64_t value = 0;
  int bad = 0;
  for (int pass = 0; pass < 60; ++pass) {
    // a pass: every layer's top-k, predicted from the layer before (6 of 8 right, or all wrong on odd passes)
    std::vector<Step> steps(L);
    std::vector<std::vector<uint32_t>> ids(L);
    for (uint32_t l = 0; l < L; ++l) ids[l] = topk(rng, E, k);
    for (uint32_t l = 0; l < L; ++l) {
      steps[l].layer = int(l);
      steps[l].ids = ids[l];
      if (l + 1 < L) {
        steps[l].pred_layer = int(l + 1);
        steps[l].pred = pass % 2 ? topk(rng, E, k) : ids[l + 1];
        if (pass % 2 == 0)
          for (int j = 0; j < 2; ++j) steps[l].pred[size_t(j)] = uint32_t(rng() % E);
      }
    }
    bad += run_pass(c, ev, a, p.lay, value, steps, rng);
  }
  const StoreStats s = c.stats();
  const char* mode = direct ? "all as they are" : "aligned direct, others widened";
  CHECK(bad == 0, "decode (%s reads): %d experts not in their slots", mode, bad);
  CHECK(s.handshakes == 60 * L, "decode: %llu handshakes", (unsigned long long)s.handshakes);
  CHECK(s.prefetches > 0 && s.prefetch_used > 0, "decode: no read-ahead used (%llu, %llu)",
        (unsigned long long)s.prefetches, (unsigned long long)s.prefetch_used);
  CHECK(s.evictions > 0, "decode: no evictions");
  check_sums(s, "decode");
  printf("decode, %s reads: %llu turns, %llu lookups: %llu hits, %llu read ahead (waited for), %llu misses; %llu read "
         "ahead, %llu of them routed; %llu evictions; %.1f MB read; stall %.3f ms a turn\n",
         mode, (unsigned long long)s.handshakes, (unsigned long long)s.lookups,
         (unsigned long long)s.hits, (unsigned long long)s.late, (unsigned long long)s.misses,
         (unsigned long long)s.prefetches, (unsigned long long)s.prefetch_used, (unsigned long long)s.evictions,
         s.bytes / 1e6, s.stall_ms / double(std::max<uint64_t>(1, s.handshakes)));
}

void case_verify_prefill(const Pack& p) {
  const uint32_t L = uint32_t(p.layers.size()), E = p.E, k = 8;
  const uint32_t nslots = 3 * E;  // the minimum: a prefill layer pins all E, its read-ahead takes E more
  Arena a(p.lay, nslots, L, E);
  Event ev;
  ExpertCache::Options o;
  o.readers = 4;
  ExpertCache c(p.files, p.layers, E, p.lay, nslots, a.m, ev.sync(), o);
  std::mt19937 rng(11);
  uint64_t value = 0;
  int bad = 0;
  for (int pass = 0; pass < 12; ++pass) {
    std::vector<Step> steps(L);
    for (uint32_t l = 0; l < L; ++l) {
      Step& s = steps[l];
      s.layer = int(l);
      if (pass % 3 == 2) {
        // a prefill chunk: 96 rows of top-k (most experts), every expert of the next layer read ahead
        for (int r = 0; r < 96; ++r)
          for (uint32_t e : topk(rng, E, k)) s.ids.push_back(e);
        s.scan = true;
        s.pred_all = l + 1 < L;
        s.pred_layer = l + 1 < L ? int(l + 1) : -1;
      } else {
        // a verify window of 4 rows sharing experts (repeats in the ids)
        const std::vector<uint32_t> pool = topk(rng, E, 12);
        for (int r = 0; r < 4; ++r)
          for (uint32_t e : topk(rng, E, k, &pool)) s.ids.push_back(e);
        if (l + 1 < L) {
          s.pred_layer = int(l + 1);
          for (int r = 0; r < 4; ++r)
            for (uint32_t e : topk(rng, E, k)) s.pred.push_back(e);
        }
      }
    }
    bad += run_pass(c, ev, a, p.lay, value, steps, rng);
  }
  const StoreStats s = c.stats();
  CHECK(bad == 0, "verify / prefill: %d experts not in their slots", bad);
  check_sums(s, "verify / prefill");
  printf("verify and prefill passes, %u slots: %llu turns, %llu lookups, %llu misses, %llu read ahead, %llu "
         "evictions\n",
         nslots, (unsigned long long)s.handshakes, (unsigned long long)s.lookups, (unsigned long long)s.misses,
         (unsigned long long)s.prefetches, (unsigned long long)s.evictions);
}

// ARC: decoding's hot set (routed again and again) outlives a prompt's sweep of every expert
void case_scan(const Pack& p) {
  const uint32_t L = uint32_t(p.layers.size()), E = p.E, k = 8;
  const uint32_t nslots = L * E / 2;
  Arena a(p.lay, nslots, L, E);
  Event ev;
  ExpertCache::Options o;
  o.readers = 4;
  o.fill = false;
  ExpertCache c(p.files, p.layers, E, p.lay, nslots, a.m, ev.sync(), o);
  std::mt19937 rng(5);
  uint64_t value = 0;
  std::vector<std::vector<uint32_t>> hot(L);
  for (uint32_t l = 0; l < L; ++l) hot[l] = topk(rng, E, 12);
  auto decode = [&](int passes) {
    for (int pass = 0; pass < passes; ++pass) {
      std::vector<Step> steps(L);
      for (uint32_t l = 0; l < L; ++l) {
        steps[l].layer = int(l);
        steps[l].ids = topk(rng, E, k, &hot[l]);
      }
      CHECK(run_pass(c, ev, a, p.lay, value, steps, rng) == 0, "scan case: an expert not in its slot");
    }
  };
  decode(40);
  // the sweep: two prefill chunks routing every expert of every layer
  for (int chunk = 0; chunk < 2; ++chunk) {
    std::vector<Step> steps(L);
    for (uint32_t l = 0; l < L; ++l) {
      steps[l].layer = int(l);
      steps[l].ids.resize(E);
      std::iota(steps[l].ids.begin(), steps[l].ids.end(), 0u);
      steps[l].scan = true;
    }
    CHECK(run_pass(c, ev, a, p.lay, value, steps, rng) == 0, "scan case: an expert not in its slot (sweep)");
  }
  int kept = 0, total = 0;
  for (uint32_t l = 0; l < L; ++l)
    for (uint32_t e : hot[l]) {
      ++total;
      kept += c.slot_of(int(l), e) >= 0;
    }
  CHECK(kept * 10 >= total * 9, "scan case: only %d of the %d hot experts kept through the sweep", kept, total);
  c.reset_stats();
  decode(10);
  const StoreStats s = c.stats();
  check_sums(s, "scan");
  CHECK(s.hits * 10 >= s.lookups * 7, "scan case: hit rate after the sweep %llu / %llu", (unsigned long long)s.hits,
        (unsigned long long)s.lookups);
  printf("hot set through a sweep of every expert: %d of %d kept; after it %llu of %llu lookups hit\n", kept, total,
         (unsigned long long)s.hits, (unsigned long long)s.lookups);
}

// A scan sweep's victim (admit / scan_victim): a prefill chunk routes every expert of its layer and has the whole next
// layer read ahead, so a victim taken in global insertion order (FIFO) is a layer the sweep still needs. With three
// layers' slots the chunk's own layer and the next one fit, so the next layer must be resident when the chunk ends, and
// the sweep must not overwrite experts in use (run_pass checks every routed expert's bytes).
void case_scan_victim(const Pack& p) {
  const uint32_t L = uint32_t(p.layers.size()), E = p.E;
  const uint32_t nslots = 3 * E;  // the minimum: a chunk pins E, its read-ahead takes E more
  Arena a(p.lay, nslots, L, E);
  Event ev;
  ExpertCache::Options o;
  o.readers = 4;
  o.fill = false;
  ExpertCache c(p.files, p.layers, E, p.lay, nslots, a.m, ev.sync(), o);
  std::mt19937 rng(17);
  uint64_t value = 0;
  int bad = 0;
  for (int pass = 0; pass < 3; ++pass) {
    for (uint32_t l = 0; l < L; ++l) {
      std::vector<Step> steps(1);
      steps[0].layer = int(l);
      steps[0].ids.resize(E);
      std::iota(steps[0].ids.begin(), steps[0].ids.end(), 0u);
      steps[0].scan = true;
      steps[0].pred_all = l + 1 < L;
      steps[0].pred_layer = l + 1 < L ? int(l + 1) : -1;
      bad += run_pass(c, ev, a, p.lay, value, steps, rng);
      quiesce(c, ev, value);  // #512: settle the read-ahead admits before reading the bookkeeping
      // the floor scan_victim reads: the lowest layer any slot holds, recomputed from the slots themselves
      int floor = 0x7fffffff;
      for (uint32_t ll = 0; ll < L; ++ll)
        for (uint32_t e = 0; e < E; ++e)
          if (c.slot_of(int(ll), e) >= 0 && int(ll) < floor) floor = int(ll);
      CHECK(c.lowest_layer() == floor, "scan victim: pass %d, chunk %u floor %d, slots say %d", pass, l, c.lowest_layer(),
            floor);
      // the T1 floor and the per-layer T1 counts are what make scan_victim skip a lap; check them against the slots
      int t1_floor = 0x7fffffff;
      for (uint32_t ll = 0; ll < L; ++ll) {
        const int n = c.t1_slots_in_table(int(ll));
        CHECK(c.t1_slots(int(ll)) == n, "scan victim: pass %d, chunk %u layer %u T1 %d counted, %d in the slots", pass, l,
              ll, c.t1_slots(int(ll)), n);
        if (n > 0 && int(ll) < t1_floor) t1_floor = int(ll);
      }
      CHECK(c.lowest_t1_layer() == t1_floor, "scan victim: pass %d, chunk %u T1 floor %d, slots say %d", pass, l,
            c.lowest_t1_layer(), t1_floor);
      if (l + 1 < L) {
        int kept = 0;
        for (uint32_t e = 0; e < E; ++e) kept += c.slot_of(int(l + 1), e) >= 0;
        CHECK(kept == int(E), "scan victim: pass %d, chunk %u left %d of %u of the next layer resident", pass, l, kept, E);
      }
    }
  }
  const StoreStats s = c.stats();
  CHECK(bad == 0, "scan victim: %d experts not in their slots", bad);
  CHECK(s.evictions > 0, "scan victim: no evictions");
  // The policy's own count, not just its bookkeeping: with the rule on the read-ahead stays resident and its next
  // layer's chunk consumes it as a hit (832 here), and the off arm below measures the same sweep without it. A case
  // that passes with scan_victim() returning -1 always is no test, so the two arms are compared.
  check_sums(s, "scan victim");
  printf("scan sweep, %u slots: %llu prefill chunks, %llu misses, %llu read ahead, %llu evictions, %.1f MB read\n",
         nslots, (unsigned long long)s.handshakes, (unsigned long long)s.misses, (unsigned long long)s.prefetches,
         (unsigned long long)s.evictions, s.bytes / 1e6);

  // The A/B's off arm (NI_SCAN_VICTIM=0): the same sweep with ARC's rule alone re-reads the next layer every chunk.
  ExpertCache::Options off = o;
  off.scan_victim = false;
  Arena a2(p.lay, nslots, L, E);
  Event ev2;
  ExpertCache d(p.files, p.layers, E, p.lay, nslots, a2.m, ev2.sync(), off);
  std::mt19937 rng2(17);
  uint64_t value2 = 0;
  int bad2 = 0;
  for (int pass = 0; pass < 3; ++pass)
    for (uint32_t l = 0; l < L; ++l) {
      std::vector<Step> steps(1);
      steps[0].layer = int(l);
      steps[0].ids.resize(E);
      std::iota(steps[0].ids.begin(), steps[0].ids.end(), 0u);
      steps[0].scan = true;
      steps[0].pred_all = l + 1 < L;
      steps[0].pred_layer = l + 1 < L ? int(l + 1) : -1;
      bad2 += run_pass(d, ev2, a2, p.lay, value2, steps, rng2);
    }
  quiesce(d, ev2, value2);  // #512: the last chunk's serve() updates the stats after its go; settle before reading
  const StoreStats t = d.stats();
  CHECK(bad2 == 0, "scan victim off: %d experts not in their slots", bad2);
  CHECK(t.prefetches > s.prefetches, "scan victim off: %llu read ahead, the rule's arm %llu: the switch does nothing",
        (unsigned long long)t.prefetches, (unsigned long long)s.prefetches);
  check_sums(t, "scan victim off");
  printf("scan sweep, %u slots, victim rule off: %llu read ahead, %.1f MB read\n", nslots,
         (unsigned long long)t.prefetches, t.bytes / 1e6);
}

// The chunk seam's wrap (q20_fetch: a chunk's last layer reads the store's layer 0 ahead unless the chunk is the
// prompt's last): a prediction never crosses a chunk boundary, the last layer
// passing pred_layer -1, and layer 0 is also the first victim of every sweep, so without the wrap each chunk's first
// turn re-reads its whole routed set with the GPU blocked. The wrap has the last layer read the store's whole layer 0
// ahead (pred_all, pred_layer 0), so the next chunk's turn 0 finds it resident. Both arms sweep the same pack over
// several chunks with the same seed and take pred_layer / pred_all from the engine's own decision - q20::seam_wraps
// and q20::seam_next_layer (expert_store.h), the pair encode_prefill and q20_fetch call - so an inverted rule fails
// here instead of only on a GPU: without the wrap every chunk misses its whole layer 0, with it only the first chunk
// (a cold store) does, and the wrap's extra read-ahead is exactly the seams it covers - every chunk but the prompt's
// last (the engine wraps on chunks - 1), a chunk's experts each.
void case_seam(const Pack& p) {
  const uint32_t L = uint32_t(p.layers.size()), E = p.E;
  const uint32_t nslots = 3 * E;  // case_scan_victim's minimum: a chunk pins E, its read-ahead takes E more
  const int chunks = 3;
  // the decision itself, pinned: the mid-pass successor ignores the wrap; a chunk but the prompt's last (its
  // `sample`) wraps to the store's layer 0; the prompt's last does not (decode follows it); a split model's later
  // stages never wrap (a Remote::Pass carries no chunk end) while its first Mac wraps like a whole model
  CHECK(q20::seam_next_layer(2, 0, int(L), true) == 3 && q20::seam_next_layer(2, 0, int(L), false) == 3,
        "seam rule: a mid-pass layer reads its successor ahead, wrapping or not");
  CHECK(q20::seam_wraps(false) && q20::seam_next_layer(int(L) - 1, 0, int(L), q20::seam_wraps(false)) == 0,
        "seam rule: a chunk but the prompt's last wraps to the store's layer 0");
  CHECK(!q20::seam_wraps(true) && q20::seam_next_layer(int(L) - 1, 0, int(L), q20::seam_wraps(true)) == -1,
        "seam rule: the prompt's last chunk does not wrap");
  CHECK(q20::seam_next_layer(int(L) - 1, int(L) - 2, int(L), false) == -1,
        "seam rule: a split model's later stage never wraps");
  CHECK(q20::seam_next_layer(3, 0, 4, true) == 0, "seam rule: a split model's head stage wraps like a whole model");
  auto arm = [&](bool wrap, const char* what) -> std::pair<uint64_t, uint64_t> {
    Arena a(p.lay, nslots, L, E);
    Event ev;
    ExpertCache::Options o;
    o.readers = 4;
    o.fill = false;
    ExpertCache c(p.files, p.layers, E, p.lay, nslots, a.m, ev.sync(), o);
    std::mt19937 rng(23);
    uint64_t value = 0, miss = 0, pf = 0, pu = 0;
    int bad = 0;
    for (int chunk = 0; chunk < chunks; ++chunk)
      for (uint32_t l = 0; l < L; ++l) {
        c.reset_stats();
        std::vector<Step> steps(1);
        steps[0].layer = int(l);
        steps[0].ids.resize(E);
        std::iota(steps[0].ids.begin(), steps[0].ids.end(), 0u);
        steps[0].scan = true;
        // the engine's own decision: this chunk wraps unless it is the prompt's last (the arm without the wrap
        // never does), and the store layer to read ahead is seam_next_layer's - not a schedule this test writes
        const bool wraps = wrap && q20::seam_wraps(chunk + 1 == chunks);
        const int nl = q20::seam_next_layer(int(l), 0, int(L), wraps);
        steps[0].pred_all = nl >= 0;
        steps[0].pred_layer = nl;
        bad += run_pass(c, ev, a, p.lay, value, steps, rng);
        quiesce(c, ev, value);  // #512: settle the read-ahead admits before reading the bookkeeping
        const StoreStats s = c.stats();
        check_sums(s, what, true);  // a step consumes the previous step's read-ahead: the invariant is the arm's
        if (l == 0)
          CHECK(s.misses == (wrap && chunk > 0 ? 0u : E), "%s: chunk %d's layer 0 missed %llu of %u experts", what,
                chunk, (unsigned long long)s.misses, E);
        miss += s.misses;
        pf += s.prefetches;
        pu += s.prefetch_used;
      }
    CHECK(pu <= pf, "%s: %llu read-ahead experts routed of %llu read ahead", what, (unsigned long long)pu,
          (unsigned long long)pf);
    CHECK(bad == 0, "%s: %d experts not in their slots", what, bad);
    return {miss, pf};
  };
  const auto plain = arm(false, "seam, no wrap");
  const auto wrapped = arm(true, "seam, wrap");
  CHECK(plain.first == uint64_t(chunks) * E, "seam, no wrap: %llu misses, %d chunks of %u expected",
        (unsigned long long)plain.first, chunks, E);
  CHECK(wrapped.first == E, "seam, wrap: %llu misses, the cold store's %u expected", (unsigned long long)wrapped.first,
        E);
  CHECK(wrapped.second == plain.second + uint64_t(chunks - 1) * E,
        "seam, wrap: %llu read ahead against %llu, %d seams of %u expected (the prompt's last chunk does not wrap)",
        (unsigned long long)wrapped.second, (unsigned long long)plain.second, chunks - 1, E);
  printf("chunk seam, %u slots: no wrap %llu misses / %llu read ahead; wrap %llu / %llu\n", nslots,
         (unsigned long long)plain.first, (unsigned long long)plain.second, (unsigned long long)wrapped.first,
         (unsigned long long)wrapped.second);
}

// The victim floor of a grouped visit (pred_floor, f75 #2): a layer-major prefill visit runs one layer over
// consecutive chunks, so turn (l, c0)'s sweep reads layer l+1 ahead while layer l's unrouted experts — the ones the
// visit's remaining turns still route — sit unpinned in T1. Without a floor they are ordinary victims: the rotating
// cursor eats them when it reaches their slots, and the ARC fallback (oldest first) can pick them too. With
// pred_floor = l the sweep's victims are the T1 keys of layers below l only, and the visit's own layer survives its
// own read-ahead. Two engineered schedules pin the difference down deterministically. "visit" (sweeping up): the
// visited layer's slots sit ahead of the cursor and below the layers the sweep may take, so the unfloored sweep eats
// them. "wrap" (sweeping layer 0 ahead with the visited layer as the floor — the admitted layer is below the floor):
// scan_victim's lap cannot run (its T1 floor is above the admitted layer), so the unfloored arm falls to ARC, whose
// oldest-first rule eats the visited layer's unpinned keys because priming made them the oldest T1. Every arm is the
// same schedule but for pred_floor; the unfloored arm must re-read or the floor is proven to do nothing. The default
// (-1, every other case's calls) leaves the shipped behaviour byte-identical.
void case_floor(const Pack& p) {
  const uint32_t L = uint32_t(p.layers.size()), E = p.E;
  const uint32_t nslots = 3 * E;  // the minimum: a turn pins E, its read-ahead takes E more
  const uint32_t half = E / 2;
  // One arm: turn A primes layer 2 whole (its misses take slots 0..E-1 in admit order); turn B primes layer b_layer
  // and sweeps layer b_sweep in, filling the store — every admit takes a free slot, so scan_victim never runs and the
  // cursor stays at 0. Turn C revisits layer 2's first half (pinning slots 0..half-1) and sweeps the target layer
  // with the arm's floor; turn D routes layer 2's second half, which the floored sweep kept resident and the
  // unfloored one evicted. Returns the C+D window's stats; kept_c / tgt_kept are the resident slots of layer 2 / the
  // target after C's quiesce.
  auto run_arm = [&](int b_layer, int b_sweep, int target, int floor, int& bad, uint32_t& kept_c, uint32_t& tgt_kept) {
    Arena a(p.lay, nslots, L, E);
    Event ev;
    ExpertCache::Options o;
    o.readers = 4;
    o.fill = false;
    ExpertCache c(p.files, p.layers, E, p.lay, nslots, a.m, ev.sync(), o);
    std::mt19937 rng(29);
    uint64_t value = 0;
    std::vector<uint32_t> all(E);
    std::iota(all.begin(), all.end(), 0u);
    auto turn = [&](int layer, std::vector<uint32_t> ids, int pred_layer, bool pred_all, int pf) {
      std::vector<Step> steps(1);
      steps[0].layer = layer;
      steps[0].ids = std::move(ids);
      steps[0].scan = true;
      steps[0].pred_layer = pred_layer;
      steps[0].pred_all = pred_all;
      steps[0].pred_floor = pf;
      bad += run_pass(c, ev, a, p.lay, value, steps, rng);
      quiesce(c, ev, value);  // #512: settle the sweep's admits before reading the bookkeeping
    };
    turn(2, all, -1, false, -1);            // A: layer 2 into slots 0..E-1
    turn(b_layer, all, b_sweep, true, -1);  // B: layer b_layer, then its sweep: the store is full
    c.reset_stats();                        // the arms are compared on the C+D window alone
    turn(2, std::vector<uint32_t>(all.begin(), all.begin() + half), target, true, floor);  // C: revisit, sweep
    kept_c = tgt_kept = 0;
    for (uint32_t e = 0; e < E; ++e) {
      kept_c += c.slot_of(2, e) >= 0;
      tgt_kept += c.slot_of(target, e) >= 0;
    }
    turn(2, std::vector<uint32_t>(all.begin() + half, all.end()), -1, false, -1);  // D: the second half
    return c.stats();
  };
  int bad = 0;
  uint32_t kept_on = 0, kept_off = 0, tgt_on = 0, tgt_off = 0;
  const StoreStats on = run_arm(0, 1, 3, 2, bad, kept_on, tgt_on);  // visit: floor = the visited layer
  const StoreStats off = run_arm(0, 1, 3, -1, bad, kept_off, tgt_off);
  CHECK(bad == 0, "pred floor: %d experts not in their slots", bad);
  CHECK(kept_on == E && kept_off == half, "pred floor: visit kept %u of %u of the visited layer, unfloored %u of %u",
        kept_on, E, kept_off, E);
  CHECK(tgt_on == E && tgt_off == E, "pred floor: visit left %u / %u of the target layer resident, unfloored %u",
        tgt_on, tgt_off, E);
  CHECK(on.misses == 0 && off.misses == half,
        "pred floor: visit re-read %llu experts, unfloored %llu: the floor must save the second half",
        (unsigned long long)on.misses, (unsigned long long)off.misses);
  CHECK(on.prefetches == E && off.prefetches == E, "pred floor: visit read ahead %llu, unfloored %llu, want %u each",
        (unsigned long long)on.prefetches, (unsigned long long)off.prefetches, E);
  check_sums(on, "pred floor");
  check_sums(off, "pred floor off");
  uint32_t wkept_on = 0, wkept_off = 0, wtgt_on = 0, wtgt_off = 0;
  const StoreStats won = run_arm(1, 4, 0, 2, bad, wkept_on, wtgt_on);  // wrap: admit layer 0 under floor 2
  const StoreStats woff = run_arm(1, 4, 0, -1, bad, wkept_off, wtgt_off);
  CHECK(bad == 0, "pred floor wrap: %d experts not in their slots", bad);
  CHECK(wkept_on == E && wkept_off == half, "pred floor wrap: kept %u of %u of the visited layer, unfloored %u of %u",
        wkept_on, E, wkept_off, E);
  CHECK(wtgt_on == E && wtgt_off == E, "pred floor wrap: left %u / %u of layer 0 resident, unfloored %u", wtgt_on,
        wtgt_off, E);
  CHECK(won.misses == 0 && woff.misses == half,
        "pred floor wrap: re-read %llu experts, unfloored %llu: ARC's oldest-first rule must eat the visited layer",
        (unsigned long long)won.misses, (unsigned long long)woff.misses);
  check_sums(won, "pred floor wrap");
  check_sums(woff, "pred floor wrap off");
  // nothing below the floor (floor 0: no layer is below it, so no T1 key ever qualifies): the ARC rule decides,
  // oldest T1 first, and the visit's own unpinned second half is eaten — the floor protects a visit only while a
  // lower layer's T1 key remains resident, as the header says
  uint32_t nkept = 0, ntgt = 0;
  const StoreStats none = run_arm(1, 4, 5, 0, bad, nkept, ntgt);
  CHECK(bad == 0, "pred floor none below: %d experts not in their slots", bad);
  CHECK(nkept == half && ntgt == E, "pred floor none below: kept %u of %u of the visited layer, target %u of %u",
        nkept, E, ntgt, E);
  CHECK(none.misses == half && none.prefetches == E,
        "pred floor none below: re-read %llu experts, read ahead %llu: ARC's rule must eat the visited layer",
        (unsigned long long)none.misses, (unsigned long long)none.prefetches);
  check_sums(none, "pred floor none below");
  printf("pred floor, %u slots: visit kept %u/%u (unfloored %u), wrap kept %u/%u (unfloored %u), none below kept "
         "%u/%u, unfloored re-reads %llu + %llu experts\n",
         nslots, kept_on, E, kept_off, wkept_on, E, wkept_off, nkept, E, (unsigned long long)off.misses,
         (unsigned long long)woff.misses);
}

void case_complete(const Pack& p) {
  const uint32_t L = uint32_t(p.layers.size()), E = p.E;
  Arena a(p.lay, L * E, L, E);
  Event ev;
  ExpertCache::Options o;
  ExpertCache c(p.files, p.layers, E, p.lay, L * E + 100, a.m, ev.sync(), o);
  CHECK(c.complete(), "complete: not complete");
  int bad = 0;
  for (uint32_t l = 0; l < L; ++l)
    for (uint32_t e = 0; e < E; ++e) bad += !expert_ok(a, p.lay, l, e, E);
  CHECK(bad == 0, "complete: %d experts wrong", bad);
  // nothing is served: post is a no-op
  ExpertCache::Req r;
  c.post(r);
  CHECK(c.stats().handshakes == 0, "complete: a handshake was served");
  printf("an arena of all %u experts: filled at start, every table word right, no handshakes\n", L * E);
}

// ---- the layer-major group walk's shapes at the merge gate's frozen lengths (PR #628) --------------------
// The leg table must be computable, not hand-written: the engine's own chunker decides a pass's chunk rows
// (prefill_chunk_rows, even-balanced - the legs pin --prefill-chunk 2048 and prefill_even_chunks on both arms),
// and engine.cpp's walk collects the next G chunks into a layer-major group while every one is plain: it breaks
// on the pass's last chunk (pos + Tc == n - that one samples and runs alone), on a split row count and on a save
// point inside or at a chunk's end, and under two plain chunks it yields nothing (nc < 2 -> 0) and the chunk
// path runs one chunk, the next pass retrying the walk from its end. The legs' cells are plain text with no
// split, no save point and no image or grammar rows, so those are the pinned conditions that make the other
// break rules inert; this replica keeps the two rules that can fire and calls the real chunker for every row
// count. A shape here that disagrees with the engine's log ("prefill group of ... rows" against the chunker's
// "prefill chunk of ...") means the pin or this replica is wrong.
void case_group_shape() {
  const int C = 2048;  // --prefill-chunk 2048, pinned on both gate arms
  struct Shape {
    std::vector<std::vector<int>> groups;  // the layer-major groups' chunk rows, in pass order
    std::vector<int> alone;                // the chunks that ran alone (the chunk path), in pass order
  };
  auto walk = [&](int64_t n, int g) {
    Shape sh;
    int64_t pos = 0;
    while (pos < n) {
      int Ts[8], nc = 0;
      int64_t q = pos;  // the walk's look-ahead cursor; it commits only when a group forms
      while (nc < g && q < n) {
        const int Tc = prefill_chunk_rows(C, q, n, true);
        if (q + Tc == n) break;  // the pass's last chunk: it samples and runs alone
        Ts[nc++] = Tc;
        q += Tc;
      }
      if (nc < 2) nc = 0;
      if (nc >= 2) {
        sh.groups.emplace_back(Ts, Ts + nc);
        pos = q;
      } else {
        const int T = prefill_chunk_rows(C, pos, n, true);
        sh.alone.push_back(T);
        pos += T;
      }
    }
    return sh;
  };
  auto show = [](const Shape& s) {
    std::string o;
    for (const std::vector<int>& gr : s.groups) {
      if (!o.empty()) o += " ";
      o += "group[";
      for (size_t i = 0; i < gr.size(); ++i) o += (i ? "," : "") + std::to_string(gr[i]);
      o += "]";
    }
    if (!s.alone.empty()) {
      if (!o.empty()) o += " ";
      o += "alone[";
      for (size_t i = 0; i < s.alone.size(); ++i) o += (i ? "," : "") + std::to_string(s.alone[i]);
      o += "]";
    }
    return o;
  };
  struct Cell {
    const char* name;
    int64_t n;
    int g;
    std::vector<std::vector<int>> groups;
    std::vector<int> alone;
  };
  // The gate's frozen lengths. The bench cells are not round numbers: pf16_seed33 is 16287 tokens, the frozen 4K
  // cell is 4132 - an nc=2 group, not the fall-through a 4096-token cell would be - and c64k / c128k are 65535
  // and 131071. EXACT4096 / EXACT6144 are exact-length prefix slices of pf16_seed33, pinned in the gate package.
  const std::vector<Cell> cells = {
      {"EXACT4096 fall-through (nc=1 -> 0)", 4096, 4, {}, {2048, 2048}},
      {"EXACT6144 smallest real group", 6144, 4, {{2048, 2048}}, {2048}},
      {"EXACT6144 G=2 diagnostic (1 group, 1 alone)", 6144, 2, {{2048, 2048}}, {2048}},
      {"pf16_seed33 (h16k)", 16287, 4, {{2036, 2036, 2036, 2036}, {2036, 2036, 2036}}, {2035}},
      {"c64k (h64k)", 65535, 4,
       {{2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048},
        {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048}},
       {2047}},
      {"c128k (h128k)", 131071, 4,
       {{2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048},
        {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048},
        {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048},
        {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048, 2048}, {2048, 2048, 2048}},
       {2047}},
      {"frozen 4K cell 4132 (nc=2, not the fall-through)", 4132, 4, {{1378, 1377}}, {1377}},
      {"6155 roundoff (the warning cell: nc=3)", 6155, 4, {{1539, 1539, 1539}}, {1538}},
  };
  for (const Cell& c : cells) {
    const Shape sh = walk(c.n, c.g);
    printf("group-shape %-48s n=%lld G=%d: %s\n", c.name, (long long)c.n, c.g, show(sh).c_str());
    CHECK(sh.groups == c.groups && sh.alone == c.alone, "group-shape %s: got %s", c.name, show(sh).c_str());
  }
}

// ---- the group's eligibility and its one accounting (the resident-store invariant, PR #628) --------------
// Runtime eligibility (q20::group_chunks) gates the three sides that must agree: the engine's walk,
// alloc4_group's allocation and the memory plan's MemPlan::group_slots. The slot term stays OUT of
// MemPlan::activations: whether it is charged at all is the store-world choice - q20::group_world, the one
// function Qwen4Exp::init_store runs live and ni::planned_group_world predicts before a load (the engine's fit
// checks, doctor.cpp). The store is sized WITHOUT the charge first; complete already, the group never forms,
// the bytes are never allocated and no fit number may carry them - that is what keeps a full default-context
// ladder from shrinking under a reservation no load pays. Incomplete, the slots pay for themselves and the
// chosen world carries the charge. The decision is monotone (more charge shrinks the store, it never completes
// it more), so the one correction is a fixed point - no oscillation, no circularity.
void case_group_plan() {
  CHECK(group_chunks(4, false, false, false) == 4, "group_chunks: an incomplete streamed store groups");
  CHECK(group_chunks(4, false, false, true) == 1, "group_chunks: an all-resident expert set never groups");
  CHECK(group_chunks(4, true, false, false) == 1, "group_chunks: a split model never groups");
  CHECK(group_chunks(4, false, true, false) == 1, "group_chunks: a layer-debugging model never groups");
  CHECK(group_chunks(1, false, false, false) == 1, "group_chunks: G=1 stays the chunk path");
  CHECK(group_chunks(8, false, false, false) == 8, "group_chunks: G=8 (the env's top) groups");
  // one formula for the plan, the allocation and the sizing charge: 0 below G=2, linear in G above; the terms are
  // alloc4_group's buffers per chunk - xg_ T*(hc*D) fp32, yg_ T*D fp32, injAg_ + injM_ 2*T*hc fp32, and the two
  // alternating sets of n-gram staging (ple_pfg_, 2G of them) T*D fp16
  const size_t T = 2048, D = 2048, hc = 4;
  CHECK(group_slot_bytes(0, T, D, hc) == 0 && group_slot_bytes(1, T, D, hc) == 0, "group_slot_bytes: 0 below G=2");
  CHECK(group_slot_bytes(2, T, D, hc) == 2 * (T * hc * D * 4 + T * D * 4 + 2 * T * hc * 4 + 2 * T * D * 2),
        "group_slot_bytes: alloc4_group's buffers, expanded");
  CHECK(group_slot_bytes(4, T, D, hc) == 2 * group_slot_bytes(2, T, D, hc), "group_slot_bytes: linear in G");
  // the store-world choice on the pure auto_store_size both the live (init_store) and the predicted
  // (planned_group_world) sides use
  const size_t gsb = group_slot_bytes(4, T, D, hc);
  const size_t G1 = size_t(1) << 30, G20 = size_t(20) << 30, G30 = size_t(30) << 30;
  const size_t G40 = size_t(40) << 30, G41 = size_t(41) << 30, G44 = size_t(44) << 30, G48 = size_t(48) << 30;
  // world A: the budget leaves the store short of all the experts -> the charge applies and the store stays
  // incomplete, so alloc4_group runs and prefill_group() answers G - the same predicate on every side
  AutoStoreIn a;
  a.all = G40, a.least = G1, a.left = G30, a.used = G20, a.ram = G48, a.avail = size_t(100) << 30;
  const GroupWorld A = group_world(gsb, a);
  CHECK(A.charge == gsb && A.st.bytes == G30 - gsb && A.st.bytes < G40,
        "group_world A: charged to %zu, still incomplete", A.st.bytes);
  CHECK(A.in.left == G30 - gsb && A.in.used == G20 + gsb && A.in.avail == (size_t(100) << 30) - gsb,
        "group_world A: the chosen inputs are the charged ones");
  // world B: every expert fits without the charge -> no charge, no slots, the bare world stands, and the chunk
  // path keeps its precomb fold. Every fit number that asks this world gets the bare sizing back - an unpaid
  // reservation must not shrink a default-context ladder. The order is the invariant: sizing WITH the charge
  // first would clamp this store below all on the avail guard and allocate slots that then make it incomplete -
  // a world that never settles.
  AutoStoreIn b = a;
  b.left = G41;
  b.avail = G44 + gsb / 2;
  const GroupWorld B = group_world(gsb, b);
  CHECK(B.charge == 0 && B.st.bytes == G40 && B.in.left == G41 && B.in.used == G20,
        "group_world B: complete without the charge, bare world, charge %zu", B.charge);
  AutoStoreIn bc = group_charged_in(b, gsb);
  CHECK(auto_store_size(bc).bytes < G40, "group_world B: charging first would have clamped it below all");
  // no term (G=1, or the split/all-resident worlds): the bare sizing, untouched
  const GroupWorld Z = group_world(0, a);
  CHECK(Z.charge == 0 && Z.st.bytes == auto_store_size(a).bytes && Z.in.left == a.left,
        "group_world: a zero term leaves the sizing alone");
  // the NI_EXPERT_STORE pin's world (A2): one parse (q20::store_pin_bytes) feeds group_world's pin branch, and
  // a pin decides its own world in the live load (init_store) and in every pre-load prediction
  // (planned_group_world) alike - chunk-invariant, only the charge's magnitude scales with the plan's rows
  // pinned-short: the store IS the pin's bytes (the parser clamped to all), the world stays incomplete so the
  // slots are charged, and the sizing inputs stay the bare ones - the pin overrides budget/avail/cap
  const size_t pin = G20;
  const GroupWorld P = group_world(gsb, a, &pin);
  CHECK(P.charge == gsb && P.st.bytes == pin && P.st.why == AutoStoreWhy::Pinned && P.in.left == a.left,
        "group_world pin: a short pin %zu charges %zu, why %d, sizing inputs stay bare", P.st.bytes, P.charge,
        int(P.st.why));
  // pinned-complete: a pin holding every expert is complete - never groups, never pays
  const size_t pin_all = a.all;
  const GroupWorld Q = group_world(gsb, a, &pin_all);
  CHECK(Q.charge == 0 && Q.st.bytes == a.all && Q.st.why == AutoStoreWhy::Pinned,
        "group_world pin: a full pin never charges, charge %zu", Q.charge);
  // the pin overrides the automatic world (the A2 mandatory fix): world B's automatic store is complete, yet a
  // short pin still reserves the slots - what keeps the pin and every fit number on one decision
  const GroupWorld R = group_world(gsb, b, &pin);
  CHECK(auto_store_size(b).bytes >= b.all && R.charge == gsb && R.st.bytes == pin,
        "group_world pin: a short pin reserves where the automatic store would be complete");
  printf("group plan: eligibility pinned, slot bytes one formula (%zu B at G=4, T=2048), store-world choice settled\n",
         gsb);
}

// The NI_EXPERT_STORE pin end to end, the seam the pure sides above stop short of: the environment variable
// through the one parse the live load shares (q20::store_pin_bytes) into the one prediction every pre-load fit
// number and `kapitsa doctor` ask (ni::planned_group_world; doctor.cpp publishes its st/in as the expert_store
// JSON: bytes, reason, group_slot_bytes, budget_leaves_bytes). No model, no load: header-only MemPlans of a
// streamed Q2_0 pack, the fields plan() fills. Qwen4Exp::init_store runs the identical expression live
// (qwen4_exp.cpp: getenv -> store_pin_bytes -> q20::group_world); that call site needs a model loaded on the
// GPU, so the CPU side pins the parser's own cases first, then the env-driven prediction through it.
void case_group_pin() {
  // the pin is process-wide: save whatever the case finds, restore it at the end
  const char* prior = getenv("NI_EXPERT_STORE");
  const std::string saved = prior ? prior : "";
  const auto set_es = [](const char* v) {
    if (v) setenv("NI_EXPERT_STORE", v, 1);
    else unsetenv("NI_EXPERT_STORE");
  };
  const size_t GB = size_t(1e9);  // the parser's GB: atof(es) * 1e9, exact at these values
  const size_t gsb = group_slot_bytes(4, 2048, 2048, 4);  // case_group_plan's G=4 shape
  const size_t budget = 20 * GB;
  const int slots = 2, ctx = 8192;

  // the parser alone, every case a parse regression could hide in: no pin, a pin, an exact size, a clamp, 0,
  // the values below zero and the infinities (the double->size_t conversion is undefined for all of them:
  // every clamp happens before it, not after)
  size_t pb = ~size_t(0);
  CHECK(!store_pin_bytes(nullptr, 4 * GB, pb) && !store_pin_bytes("", 4 * GB, pb) && !store_pin_bytes("auto", 4 * GB, pb),
        "store_pin_bytes: unset, the empty string and \"auto\" are no pin");
  CHECK(store_pin_bytes("all", 4 * GB, pb) && pb == 4 * GB, "store_pin_bytes: \"all\" pins the whole store (%zu)", pb);
  CHECK(store_pin_bytes("0.5", 4 * GB, pb) && pb == 5 * GB / 10, "store_pin_bytes: \"0.5\" is exact (%zu)", pb);
  CHECK(store_pin_bytes("1e6", 4 * GB, pb) && pb == 4 * GB, "store_pin_bytes: \"1e6\" clamps to all (%zu)", pb);
  CHECK(store_pin_bytes("abc", 4 * GB, pb) && pb == 0, "store_pin_bytes: garbage reads 0, pinned (%zu)", pb);
  CHECK(store_pin_bytes("-1", 4 * GB, pb) && pb == 0, "store_pin_bytes: a negative pin clamps to 0 (%zu)", pb);
  CHECK(store_pin_bytes("-0.5", 4 * GB, pb) && pb == 0, "store_pin_bytes: a negative fraction clamps to 0 (%zu)", pb);
  CHECK(store_pin_bytes("nan", 4 * GB, pb) && pb == 0, "store_pin_bytes: NaN clamps to 0 (%zu)", pb);
  CHECK(store_pin_bytes("inf", 4 * GB, pb) && pb == 4 * GB, "store_pin_bytes: \"inf\" clamps to all, never an undefined conversion (%zu)", pb);
  CHECK(store_pin_bytes("1e999", 4 * GB, pb) && pb == 4 * GB, "store_pin_bytes: \"1e999\" (atof overflows to inf) clamps to all (%zu)", pb);

  // fixture C: a 3-layer pack - the smallest store (3 layers') IS the whole store, so no live sample of the
  // available memory, no RAM-share cap and no budget can cut the automatic world below all: it is complete on
  // any machine, and a short pin must reserve the slots anyway (the A2 invariant, through the env seam)
  MemPlan c;
  c.store_all = 3 * GB, c.store_least = 3 * GB;
  c.streamed = c.store_all;
  c.weights = GB + c.store_least;  // the trunk (1 GB) plus the smallest store, as plan() accounts it
  c.mtp = 200000000, c.activations = 100000000;
  c.group_slots = gsb;
  c.slot_state = 50000000, c.kv_per_pos = 10000;
  set_es(nullptr);
  const GroupWorld U = planned_group_world(c, slots, ctx, budget);
  CHECK(U.st.bytes == c.store_all && U.st.why == AutoStoreWhy::All && U.charge == 0 && U.in.left > U.in.all,
        "group pin env: unset, the automatic world is complete and pays nothing (bytes %zu, why %d, charge %zu)",
        U.st.bytes, int(U.st.why), U.charge);
  set_es("2");  // a short pin: 2 GB of the 3 GB store
  const GroupWorld P = planned_group_world(c, slots, ctx, budget);
  CHECK(P.st.bytes == 2 * GB && P.st.why == AutoStoreWhy::Pinned && P.charge == gsb && P.in.left == U.in.left,
        "group pin env: a short pin reserves where the automatic store was complete (pin bytes %zu, charge %zu)",
        P.st.bytes, P.charge);
  CHECK(std::string(auto_store_why_name(P.st.why)) == "NI_EXPERT_STORE" &&
            auto_store_reason(P.st, P.in).find("NI_EXPERT_STORE pin") != std::string::npos,
        "group pin env: doctor's reason value and sentence name the pin");

  // fixture P: realistic proportions - a 4-layer pack, the smallest store 3 layers'; the pin branch ignores
  // budget/avail/cap, so every pin assertion is exact on any machine; only the "auto" row is an invariant (a
  // live avail sample moves by definition)
  MemPlan p = c;
  p.store_all = 4 * GB, p.store_least = 3 * GB;
  p.streamed = p.store_all;
  set_es("3.5");  // between the least and all: a realistic short pin
  const GroupWorld Pp = planned_group_world(p, slots, ctx, budget);
  CHECK(Pp.st.bytes == 35 * GB / 10 && Pp.st.why == AutoStoreWhy::Pinned && Pp.charge == gsb,
        "group pin env: the pin's bytes are exact at the parser's scale (bytes %zu, charge %zu)", Pp.st.bytes,
        Pp.charge);
  set_es("all");
  const GroupWorld A = planned_group_world(p, slots, ctx, budget);
  CHECK(A.st.bytes == p.store_all && A.st.why == AutoStoreWhy::Pinned && A.charge == 0 &&
            auto_store_reason(A.st, A.in).find("pin: resident") != std::string::npos,
        "group pin env: a full pin is resident and never pays (charge %zu)", A.charge);
  set_es("9");  // above the whole store: the parser clamps to all
  const GroupWorld O = planned_group_world(p, slots, ctx, budget);
  CHECK(O.st.bytes == p.store_all && O.st.why == AutoStoreWhy::Pinned && O.charge == 0,
        "group pin env: an over pin clamps to all (bytes %zu)", O.st.bytes);
  set_es("xyz");  // garbage reads as 0: fail-safe to the incomplete side, reservation charged
  const GroupWorld G = planned_group_world(p, slots, ctx, budget);
  CHECK(G.st.bytes == 0 && G.st.why == AutoStoreWhy::Pinned && G.charge == gsb,
        "group pin env: garbage pins 0 bytes and still reserves (charge %zu)", G.charge);
  set_es("-1");  // negative: the conversion below zero is undefined and platform-split (arm64 saturates to 0,
                 // x86-64 reads an indefinite huge value as a full pin); the parser's clamp makes it the
                 // garbage fail-safe everywhere: pins 0, still reserves
  const GroupWorld N = planned_group_world(p, slots, ctx, budget);
  CHECK(N.st.bytes == 0 && N.st.why == AutoStoreWhy::Pinned && N.charge == gsb,
        "group pin env: a negative pin clamps to 0 and still reserves (charge %zu)", N.charge);
  set_es("auto");  // the parser's explicit off-pin: the automatic world, never Pinned
  const GroupWorld V = planned_group_world(p, slots, ctx, budget);
  CHECK(V.st.why != AutoStoreWhy::Pinned && V.st.bytes >= p.store_least && V.st.bytes <= p.store_all &&
            V.charge == (V.st.bytes < p.store_all ? gsb : 0),
        "group pin env: \"auto\" is the automatic world (bytes %zu, why %d, charge %zu)", V.st.bytes, int(V.st.why),
        V.charge);
  set_es("3.5");
  // context: the pin's world is chunk-invariant - the store's bytes and charge identical at 4K and at 128K -
  // while the world's budget_leaves tracks the context exactly: flash_next_model_budget holds
  // slots*slot_state + kv_per_pos*min(ctx, 32768), so the 128K world leaves less by the KV term between 4K
  // and the 32K clamp, and one slot less leaves one slot_state more
  const GroupWorld W4 = planned_group_world(p, slots, 4096, budget);
  const GroupWorld W13 = planned_group_world(p, slots, 131072, budget);
  CHECK(W4.st.bytes == Pp.st.bytes && W4.st.bytes == W13.st.bytes && W4.charge == W13.charge && W4.charge == gsb,
        "group pin env: the pin is the same world at 4K, 8K and 128K (charge %zu vs %zu)", W4.charge, W13.charge);
  CHECK(W4.in.left - W13.in.left == p.kv_per_pos * (32768 - 4096),
        "group pin env: the budget's leaves track the context's KV reserve exactly (%zu vs %zu)", W4.in.left,
        W13.in.left);
  const GroupWorld S1 = planned_group_world(p, 1, ctx, budget);
  CHECK(S1.in.left - Pp.in.left == p.slot_state && S1.charge == gsb,
        "group pin env: one slot fewer: one slot_state more left, same charge (leaves %zu vs %zu, charge %zu vs %zu)",
        S1.in.left, Pp.in.left, S1.charge, Pp.charge);
  // restore the environment: the pin is process-wide and the other cases must see what this one found
  if (prior) setenv("NI_EXPERT_STORE", saved.c_str(), 1);
  else unsetenv("NI_EXPERT_STORE");
  printf("group pin: env -> one parse -> one world: bytes exact, a short pin reserves %zu B, the context moves "
         "the budget's leaves, never the pin\n",
         gsb);
}

}  // namespace

int main(int argc, char** argv) {
  std::string keep;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--keep") && i + 1 < argc) keep = argv[++i];
  std::string dir = keep;
  if (dir.empty()) {
    const char* t = getenv("TMPDIR");
    std::string tmpl = std::string(t && *t ? t : "/tmp") + "/fn_store_test.XXXXXX";
    std::vector<char> b(tmpl.begin(), tmpl.end());
    b.push_back('\0');
    if (!mkdtemp(b.data())) fatal("mkdtemp failed");
    dir = b.data();
  } else {
    mkdir(dir.c_str(), 0755);
  }
  const Pack p = make_pack(dir, 6, 64);
  case_decode(p, false);
  case_decode(p, true);
  case_verify_prefill(p);
  case_scan(p);
  case_scan_victim(p);
  case_seam(p);
  case_floor(p);
  case_complete(p);
  case_settle();
  case_auto_size();
  case_group_shape();
  case_group_plan();
  case_group_pin();
  if (keep.empty()) {
    for (const std::string& f : p.files) unlink(f.c_str());
    rmdir(dir.c_str());
  }
  if (g_fail) {
    fprintf(stderr, "fn_store_test: %d checks failed\n", g_fail);
    return 1;
  }
  printf("fn_store_test: all checks passed\n");
  return 0;
}
