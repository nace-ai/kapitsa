// CPU test of the streamed-expert cache (src/model/expert_store.h): no GPU, no model. A synthetic pack of stacked Q2_0
// planes is written to a temporary folder (every byte a function of layer, plane, expert and position; planes at page-
// aligned and at odd offsets, so both the direct and the widened 16 KiB reads run), an arena in plain memory (two
// chunks), and a thread plays the GPU: it posts a pass's handshakes, then for each one writes the routing, signals the
// ready value, waits for the go and checks that every routed expert's table word points at a slot holding exactly
// that expert's bytes; it checks the same slots again just before its next ready (read-ahead must not have overwritten
// an expert in use). Cases: decode rows with next-layer predictions (good and wrong ones), verify rows with repeats,
// prefill chunks (scan lookups, every next-layer expert read ahead), a hot set that must survive a prompt's sweep (ARC),
// direct and widened reads, an arena of every expert (no handshakes), and the statistics' sums.
//   fn_store_test [--keep DIR]
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

void check_sums(const StoreStats& s, const char* what) {
  CHECK(s.hits + s.late + s.misses == s.lookups, "%s: hits %llu + late %llu + misses %llu != lookups %llu", what,
        (unsigned long long)s.hits, (unsigned long long)s.late, (unsigned long long)s.misses,
        (unsigned long long)s.lookups);
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
  for (AutoStoreWhy w : {AutoStoreWhy::All, AutoStoreWhy::Budget, AutoStoreWhy::Avail, AutoStoreWhy::Cap}) {
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
  case_complete(p);
  case_settle();
  case_auto_size();
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
