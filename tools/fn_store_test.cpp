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
  case_complete(p);
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
