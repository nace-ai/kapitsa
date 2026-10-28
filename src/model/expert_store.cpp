#include "model/expert_store.h"
#ifdef NI_EXPERT_CACHE_TEST
#include "model/expert_store_test.h"
#endif

#include "util/log.h"

#include <CoreFoundation/CoreFoundation.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <sys/qos.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace ni::q20 {

namespace {

constexpr uint32_t kPage = 16384;

inline void relax() {
#if defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

bool valid_extent(const PlaneSrc& p, uint32_t E, uint64_t bytes) {
  const uint64_t bound = std::min(bytes, uint64_t(std::numeric_limits<off_t>::max()));
  return p.off <= bound && p.len <= bound - p.off &&
         (E == 1 || p.stride <= (bound - p.off - p.len) / (E - 1));
}

// n bytes at off into dst; a short read at the end of the file is allowed down to `need` bytes
void pread_all(int fd, uint8_t* dst, size_t n, uint64_t off, size_t need) {
  size_t done = 0;
  while (done < n) {
    const ssize_t r = pread(fd, dst + done, n - done, off_t(off + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      fatal("expert store: read failed (%s)", strerror(errno));
    }
    if (r == 0) break;
    done += size_t(r);
  }
  if (done < need) fatal("expert store: short read (%zu of %zu bytes at %llu)", done, need, (unsigned long long)off);
}

}  // namespace

size_t available_memory() {
  vm_statistics64_data_t v;
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n) != KERN_SUCCESS) return 0;
  const uint64_t pages = uint64_t(v.free_count) + v.inactive_count + v.purgeable_count;
  return size_t(pages * vm_kernel_page_size);
}

double auto_store_cap(bool apple10) {
  double cap = apple10 ? 0.0 : 0.40;
  const char* cv = getenv("NI_EXPERT_CAP");
  if (cv && *cv) {
    char* end = nullptr;
    const double x = strtod(cv, &end);
    if (end != cv && !*end && x >= 0 && x <= 1) cap = x;
    else logi("NI_EXPERT_CAP=%s is not a share of the RAM between 0 and 1, ignored: using %.2f", cv, cap);
  }
  return cap;
}

size_t auto_store_kv(size_t kv_per_pos, int planned_ctx, int slots) {
  // the plan's positions: min(planned context, kAutoKvCtxCap) per conversation times the slots; without a plan the
  // default budget's keep window (16384, one conversation)
  long long pos = 16384;
  if (planned_ctx > 0) {
    const int c = planned_ctx < kAutoKvCtxCap ? planned_ctx : kAutoKvCtxCap;
    pos = static_cast<long long>(c) * (slots > 1 ? slots : 1);
  }
  const char* kv = getenv("NI_EXPERT_KV_POS");
  if (kv && *kv) {
    char* end = nullptr;
    const long long x = strtoll(kv, &end, 10);
    if (end != kv && !*end && x >= 0) pos = x;
    else logi("NI_EXPERT_KV_POS=%s is not a position count of 0 or more, ignored: using %lld", kv, pos);
  }
  return kv_per_pos * size_t(pos);
}

// ---- ExpertCache ----

ExpertCache::ExpertCache(const std::vector<std::string>& files, const std::vector<LayerSrc>& layers, uint32_t E,
                         SlotLayout lay, uint32_t nslots, const Memory& mem, const Sync& sync, const Options& o)
    : L_(uint32_t(layers.size())), E_(E), nslots_(nslots), lay_(lay), mem_(mem), sync_(sync), direct_(o.direct),
      scan_victim_on_(o.scan_victim), src_(layers) {
  try {
    if (!L_ || !E_) fatal("expert store: no layers");
    if (lay_.bytes % kPage) fatal("expert store: slots of %u bytes (not whole 16 KiB pages)", lay_.bytes);
    // the planes must be the slot layout's: codes cb = uc, d db = ud - gd
    const uint32_t cb = lay_.uc, db = lay_.ud - lay_.gd;
    const uint32_t want[6] = {cb, db, cb, db, cb, db};
    for (const LayerSrc& s : src_)
      for (int j = 0; j < 6; ++j)
        if (s.p[j].len != want[j] || s.p[j].file < 0 || s.p[j].file >= int(files.size()))
          fatal("expert store: a layer's planes do not match the slot layout");
    const size_t keys = size_t(L_) * E_;
    if (nslots_ > keys) nslots_ = uint32_t(keys);
    if (nslots_ < keys && nslots_ < 3 * E_)
      fatal("expert store: %u slots; at least %u (3 layers' experts) or all %zu are needed", nslots_, 3 * E_, keys);
    if (!mem_.per_chunk || nslots_ > 2 * size_t(mem_.per_chunk)) fatal("expert store: the arena has no room for its slots");
    complete_ = nslots_ == keys;
    std::vector<uint64_t> sizes;
    for (const std::string& f : files) {
      Fd opened(open(f.c_str(), O_RDONLY | O_NONBLOCK));
      const int fd = opened.get();
      if (fd < 0) fatal("expert store: cannot open %s", f.c_str());
      struct stat st;
      if (fstat(fd, &st)) fatal("expert store: cannot stat %s", f.c_str());
      if (!S_ISREG(st.st_mode)) fatal("expert store: not a regular file: %s", f.c_str());
      if (st.st_size < 0) fatal("expert store: invalid extent in %s", f.c_str());
      sizes.push_back(uint64_t(st.st_size));
      fcntl(fd, F_NOCACHE, 1);  // one pass from the SSD: no second copy in the page cache
      fds_.push_back(std::move(opened));
    }
    for (uint32_t l = 0; l < L_; ++l)
      for (int j = 0; j < 6; ++j) {
        const PlaneSrc& p = src_[l].p[j];
        if (!valid_extent(p, E_, sizes[p.file]))
          fatal("expert store: invalid extent at layer %u plane %d in %s", l, j, files[p.file].c_str());
      }
    std::memset(mem_.table, 0, keys * 4);
    slot_of_.assign(keys, -1);
    key_of_slot_.assign(nslots_, -1);
    prv_.assign(keys, -1);
    nxt_.assign(keys, -1);
    where_.assign(keys, kNone);
    routed_.assign(keys, 0);
    prefetched_.assign(keys, 0);
    pin_.assign(keys, 0);
    mark_.assign(E_, 0);
    pending_.reset(new std::atomic<int>[keys]);
    for (size_t k = 0; k < keys; ++k) pending_[k].store(0, std::memory_order_relaxed);
    free_.resize(nslots_);
    layer_res_.assign(L_, 0);
    layer_t1_.assign(L_, 0);
    for (uint32_t i = 0; i < nslots_; ++i) free_[i] = int32_t(nslots_ - 1 - i);  // slot 0 first

    const int nr = std::max(1, std::min(o.readers, 64));
    for (int i = 0; i < nr; ++i) {
#ifdef NI_EXPERT_CACHE_TEST
      ExpertCacheTestAccess::observe(*this, ExpertCacheTestAccess::Point::BeforeWorkerStart);
#endif
      readers_.emplace_back([this] { reader_loop(); });
    }

    // fill: every expert, or the same share of each layer (experts 0, 1, ... of every layer in turn)
    const double t0 = now_ms();
    if (complete_ || o.fill) {
      for (uint32_t i = 0; i < nslots_; ++i) {
        const int k = complete_ ? int(i) : int((i % L_) * E_ + i / L_);
        admit(k, true);
        load(k, false);
        ++filled_;
      }
      for (size_t k = 0; k < keys; ++k)
        while (pending_[k].load(std::memory_order_acquire) > 0) usleep(500);
    }
    fill_ms_ = now_ms() - t0;
    if (!complete_) {
#ifdef NI_EXPERT_CACHE_TEST
      ExpertCacheTestAccess::observe(*this, ExpertCacheTestAccess::Point::BeforeWorkerStart);
#endif
      turn_ = std::thread([this] { turn_loop(); });
    }
  } catch (...) { stop_workers(); throw; }
}

ExpertCache::~ExpertCache() { stop_workers(); }

void ExpertCache::stop_workers() {
  {
    std::lock_guard<std::mutex> lk(req_mu_);
    turn_stop_ = true;
  }
  req_cv_.notify_all();
  if (turn_.joinable()) turn_.join();
  {
    std::lock_guard<std::mutex> lk(io_mu_);
    io_closed_ = true;
  }
#ifdef NI_EXPERT_CACHE_TEST
  ExpertCacheTestAccess::observe(*this, ExpertCacheTestAccess::Point::ReadersClosed);
#endif
  io_cv_.notify_all();
  for (std::thread& t : readers_) if (t.joinable()) t.join();
#ifdef NI_EXPERT_CACHE_TEST
  ExpertCacheTestAccess::observe(*this, ExpertCacheTestAccess::Point::ReadersJoined);
#endif
}

uint32_t ExpertCache::word(int slot) const {
  const uint32_t c = uint32_t(slot) / mem_.per_chunk, i = uint32_t(slot) % mem_.per_chunk;
  return (c << 28) | (i * (lay_.bytes / kPage));
}

uint8_t* ExpertCache::slot_ptr(int slot) const {
  const uint32_t c = uint32_t(slot) / mem_.per_chunk, i = uint32_t(slot) % mem_.per_chunk;
  return mem_.chunk[c] + size_t(i) * lay_.bytes;
}

int ExpertCache::slot_of(int layer, uint32_t e) const {
  const size_t k = size_t(layer) * E_ + e;
  return k < slot_of_.size() ? slot_of_[k] : -1;
}

void ExpertCache::post(const Req& r) {
  if (complete_) return;
  if (r.layer < 0 || uint32_t(r.layer) >= L_) fatal("expert store: layer %d", r.layer);
  {
    std::lock_guard<std::mutex> lk(req_mu_);
    reqs_.push_back(r);
  }
  req_cv_.notify_one();
}

void ExpertCache::wait_gpu(uint64_t value) {
  // spin while a decode layer's turn is near (they come ~0.1-0.5 ms apart), then back off (a prefill layer takes ~0.1 s)
  const double t0 = Encoder::host_ms();
  while (sync_.value() < value) {
    if (turn_stop_) return;
    const double dt = Encoder::host_ms() - t0;
    if (dt < 2.0) relax();
    else usleep(dt < 50.0 ? 20 : 200);
  }
}

void ExpertCache::turn_loop() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  for (;;) {
    Req r;
    {
      std::unique_lock<std::mutex> lk(req_mu_);
      req_cv_.wait(lk, [&] { return turn_stop_ || !reqs_.empty(); });
      if (turn_stop_) return;
      r = reqs_.front();
      reqs_.pop_front();
    }
    wait_gpu(r.ready);
    if (turn_stop_) return;
    serve(r);
  }
}

void ExpertCache::serve(const Req& r) {
  const double t_ready = Encoder::host_ms();
  // the GPU signalled this ready after the previous layer's expert kernels: those experts are no longer in use
  for (int k : active_) --pin_[size_t(k)];
  active_.clear();
  std::vector<int> keys;
  keys.reserve(std::min<uint32_t>(r.n, E_));
  for (uint32_t i = 0; i < r.n; ++i) {
    const uint32_t e = r.ids[i];
    if (e >= E_ || mark_[e]) continue;
    mark_[e] = 1;
    keys.push_back(int(uint32_t(r.layer) * E_ + e));
  }
  for (int k : keys) mark_[size_t(k) % E_] = 0;
  uint64_t hits = 0, late = 0, misses = 0, used = 0;
  // the present ones (resident, or being read ahead) are pinned first, so a miss never evicts an expert routed here
  std::vector<int> miss;
  for (int k : keys) {
    const Where w = Where(where_[size_t(k)]);
    if (w == kT1 || w == kT2) {
      ++pin_[size_t(k)];
      if (pending_[size_t(k)].load(std::memory_order_relaxed) > 0) ++late;
      else ++hits;
      const bool ahead = prefetched_[size_t(k)] != 0;
      prefetched_[size_t(k)] = 0;
      used += ahead;
      hit(k, r.scan, ahead);
    } else {
      miss.push_back(k);
    }
  }
  for (int k : miss) {
    admit(k, r.scan);
    if (!r.scan) routed_[size_t(k)] = 1;
    ++pin_[size_t(k)];
    load(k, true);
    ++misses;
  }
  const double t_rd = Encoder::host_ms();
  for (int k : keys)
    while (pending_[size_t(k)].load(std::memory_order_acquire) > 0) relax();
  const double t_loaded = Encoder::host_ms();
  // the predictions are copied before the go (the GPU rewrites them in its next layer)
  std::vector<uint32_t> pred;
  if (r.pred_layer >= 0 && !r.pred_all && r.pred) pred.assign(r.pred, r.pred + r.npred);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  sync_.set(r.go);
  const double t_go = Encoder::host_ms();
  if (r.warm && sync_.warm) sync_.warm();
  active_ = keys;
  // read ahead (a read-ahead expert enters the order as recent, unrouted: its first routed use counts as its first)
  uint64_t pf = 0;
  if (r.pred_layer >= 0 && uint32_t(r.pred_layer) < L_) {
    auto ahead = [&](uint32_t e) {
      const int k = int(uint32_t(r.pred_layer) * E_ + e);
      const Where w = Where(where_[size_t(k)]);
      if (w == kT1 || w == kT2) return;
      admit(k, true, r.pred_floor);
      prefetched_[size_t(k)] = 1;
      load(k, false);
      ++pf;
    };
    if (r.pred_all) {
      for (uint32_t e = 0; e < E_; ++e) ahead(e);
    } else {
      for (uint32_t e : pred)
        if (e < E_ && !mark_[e]) {
          mark_[e] = 1;
          ahead(e);
        }
      for (uint32_t e : pred)
        if (e < E_) mark_[e] = 0;
    }
  }
  std::lock_guard<std::mutex> lk(st_mu_);
  ++st_.handshakes;
  st_.lookups += keys.size();
  st_.hits += hits;
  st_.late += late;
  st_.misses += misses;
  st_.prefetches += pf;
  st_.prefetch_used += used;
  st_.stall_ms += t_go - t_ready;
  st_.read_wait_ms += t_loaded - t_rd;
}

// ---- replacement: ARC (Megiddo and Modha) over (layer, expert) keys; T1 / T2 in the arena, B1 / B2 their ghosts ----

ExpertCache::List& ExpertCache::list_of(Where w) { return w == kT1 ? t1_ : w == kT2 ? t2_ : w == kB1 ? b1_ : b2_; }

void ExpertCache::lst_remove(List& L, int k) {
  const int p = prv_[size_t(k)], n = nxt_[size_t(k)];
  if (p >= 0) nxt_[size_t(p)] = n;
  else L.head = n;
  if (n >= 0) prv_[size_t(n)] = p;
  else L.tail = p;
  prv_[size_t(k)] = nxt_[size_t(k)] = -1;
  --L.size;
}

void ExpertCache::lst_push(List& L, int k) {
  prv_[size_t(k)] = -1;
  nxt_[size_t(k)] = L.head;
  if (L.head >= 0) prv_[size_t(L.head)] = k;
  L.head = k;
  if (L.tail < 0) L.tail = k;
  ++L.size;
}

int ExpertCache::lru_free(const List& L) const {
  for (int k = L.tail; k >= 0; k = prv_[size_t(k)])
    if (pin_[size_t(k)] == 0 && pending_[size_t(k)].load(std::memory_order_relaxed) == 0) return k;
  return -1;
}

void ExpertCache::t1_add(int layer) {
  ++layer_t1_[size_t(layer)];
  if (layer < low_t1_) low_t1_ = layer;
}

void ExpertCache::t1_drop(int layer) {
  if (--layer_t1_[size_t(layer)] == 0 && low_t1_ == layer)
    while (low_t1_ < int(L_) && layer_t1_[size_t(low_t1_)] == 0) ++low_t1_;
}

int ExpertCache::t1_slots(int layer) const { return layer < int(L_) ? int(layer_t1_[size_t(layer)]) : 0; }

int ExpertCache::t1_slots_in_table(int layer) const {
  int n = 0;
  for (uint32_t slot = 0; slot < nslots_; ++slot) {
    const int v = key_of_slot_[size_t(slot)];
    if (v >= 0 && v / int(E_) == layer && Where(where_[size_t(v)]) == kT1) ++n;
  }
  return n;
}

void ExpertCache::hit(int k, bool scan, bool first) {
  if (scan) return;  // a prompt's sweep leaves the order as it is
  routed_[size_t(k)] = 1;
  const Where w = Where(where_[size_t(k)]);
  lst_remove(list_of(w), k);
  // a read-ahead expert's first routed use (first) is its first access (recency); later ones are its frequency
  if (first && w == kT1) {
    lst_push(t1_, k);
  } else {
    lst_push(t2_, k);
    where_[size_t(k)] = kT2;
    if (w == kT1) t1_drop(k / int(E_));  // its frequency moves it out of the sweep's victims
  }
}

// A scan sweep's victim: hit() records nothing while scan is set, so T1/T2 are insertion order and the policy is FIFO
// over every key: a turn's sweep then re-reads the whole store. Evicting a slot whose layer the sweep has already
// passed instead keeps the layer being read resident: a rotating cursor over the slots, at most one lap, and the ARC
// rule picks the victim when no slot qualifies. Decode never sets scan: its replacement is unchanged.
int ExpertCache::scan_victim(int k, int floor_layer) {
  const int layer = k / int(E_);
  // The floor: a layer-major visit's sweep passes the layer the visit is running, so the visit's own layer's keys —
  // the ones its remaining chunks still route, unpinned between its turns — are not victims of the read-ahead for
  // the next layer while any lower layer's T1 key remains; when none does, the ARC rule below decides and the
  // visit's own keys can be eaten. -1 (every chunk-major turn, decode, fill): the admitted key's layer is the floor.
  const int limit = floor_layer >= 0 ? floor_layer : layer;
  // A T1 slot below the floor is the victim the sweep wants, and the per-layer counts say whether one exists: with a
  // lower layer holding only T2 keys (decoding's hot experts) or slots above, no scan can find one, and the ARC rule
  // below is the answer without a lap. A lap here would cost nslots slot reads per admit, a whole store per turn.
  if (low_t1_ >= limit) return -1;
  for (uint32_t i = 0; i < nslots_; ++i) {
    const int slot = int(scan_cursor_);
    scan_cursor_ = scan_cursor_ + 1 == nslots_ ? 0 : scan_cursor_ + 1;
    const int v = key_of_slot_[size_t(slot)];
    if (v < 0 || v / int(E_) >= limit) continue;
    if (Where(where_[size_t(v)]) != kT1) continue;  // T2 holds decoding's hot experts: a sweep leaves them (ARC's rule)
    if (pin_[size_t(v)] || pending_[size_t(v)].load(std::memory_order_relaxed)) continue;
    return v;
  }
  return -1;
}

int ExpertCache::admit(int k, bool scan, int floor_layer) {
  const Where w = Where(where_[size_t(k)]);
  const double c = double(nslots_);
  if (w == kB1 || w == kB2) {
    if (!scan) {  // a ghost hit: the list it fell out of should have been longer
      if (w == kB1) p_ = std::min(c, p_ + std::max(double(b2_.size) / std::max(1, b1_.size), 1.0));
      else p_ = std::max(0.0, p_ - std::max(double(b1_.size) / std::max(1, b2_.size), 1.0));
    }
    lst_remove(list_of(w), k);
  }
  int slot;
  if (!free_.empty()) {
    slot = free_.back();
    free_.pop_back();
  } else {
    int v = scan && scan_victim_on_ ? scan_victim(k, floor_layer) : -1;
    if (v < 0) {
      const bool t1_first = scan || (t1_.size >= 1 && ((w == kB2 && double(t1_.size) >= p_) || double(t1_.size) > p_));
      v = lru_free(t1_first ? t1_ : t2_);
      if (v < 0) v = lru_free(t1_first ? t2_ : t1_);
      if (v < 0) fatal("expert store: all %u slots hold experts in use", nslots_);
    }
    slot = slot_of_[size_t(v)];
    evict(v);
  }
  slot_of_[size_t(k)] = slot;
  key_of_slot_[size_t(slot)] = k;
  const int kl = k / int(E_);
  ++layer_res_[size_t(kl)];
  if (kl < low_layer_) low_layer_ = kl;
  const bool freq = !scan && (w == kB1 || w == kB2);
  lst_push(freq ? t2_ : t1_, k);
  where_[size_t(k)] = freq ? kT2 : kT1;
  if (!freq) t1_add(kl);
  // its table entry: the GPU reads it only once k's layer is routed, and that turn waits for k's read first
  mem_.table[k] = word(slot);
  trim_ghosts();
  return slot;
}

void ExpertCache::evict(int v) {
  const Where w = Where(where_[size_t(v)]);
  lst_remove(list_of(w), v);
  key_of_slot_[size_t(slot_of_[size_t(v)])] = -1;
  slot_of_[size_t(v)] = -1;
  const int vl = v / int(E_);
  if (--layer_res_[size_t(vl)] == 0 && low_layer_ == vl)
    while (low_layer_ < int(L_) && layer_res_[size_t(low_layer_)] == 0) ++low_layer_;
  if (w == kT1) t1_drop(vl);
  mem_.table[v] = 0;
  prefetched_[size_t(v)] = 0;
  if (routed_[size_t(v)]) {  // remembered as a ghost: only experts decoding routed
    lst_push(w == kT1 ? b1_ : b2_, v);
    where_[size_t(v)] = w == kT1 ? kB1 : kB2;
  } else {
    where_[size_t(v)] = kNone;
  }
  std::lock_guard<std::mutex> lk(st_mu_);
  ++st_.evictions;
}

void ExpertCache::trim_ghosts() {
  const int c = int(nslots_);
  auto drop = [&](List& L) {
    const int k = L.tail;
    lst_remove(L, k);
    where_[size_t(k)] = kNone;
    routed_[size_t(k)] = 0;
  };
  while (b1_.size > 0 && t1_.size + b1_.size > c) drop(b1_);
  while (b2_.size > 0 && t1_.size + t2_.size + b1_.size + b2_.size > 2 * c) drop(b2_);
}

// ---- reads ----

void ExpertCache::load(int k, bool urgent) {
  const uint32_t l = uint32_t(k) / E_, e = uint32_t(k) % E_;
  const LayerSrc& s = src_[l];
  uint8_t* base = slot_ptr(slot_of_[size_t(k)]);
  const uint32_t at[6] = {lay_.gc, lay_.gd, lay_.uc, lay_.ud, lay_.dc, lay_.dd};
  pending_[size_t(k)].store(6, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(io_mu_);
    for (int j = 0; j < 6; ++j) {
      const PlaneSrc& p = s.p[j];
      (urgent ? urgent_ : normal_).push_back(Job{fds_[size_t(p.file)].get(), p.off + uint64_t(e) * p.stride, p.len, base + at[j], k});
    }
  }
  io_cv_.notify_all();
}

void ExpertCache::reader_loop() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  // widened reads: the whole pages around a piece into page-aligned staging (F_NOCACHE reads of whole pages: Splash
  // measured 9.3 vs 8.1 GB/s at under half the CPU)
  const size_t cap = size_t(std::max(lay_.uc, lay_.ud - lay_.gd)) + 2 * kPage;
  void* mem = nullptr;
  if (posix_memalign(&mem, kPage, cap)) fatal("expert store: out of memory");
  uint8_t* stage = static_cast<uint8_t*>(mem);
  for (;;) {
    Job j;
    {
      std::unique_lock<std::mutex> lk(io_mu_);
      io_cv_.wait(lk, [&] { return io_closed_ || !urgent_.empty() || !normal_.empty(); });
      if (io_closed_ && urgent_.empty() && normal_.empty()) break;
      std::deque<Job>& q = urgent_.empty() ? normal_ : urgent_;
      j = q.front();
      q.pop_front();
    }
#ifdef NI_EXPERT_CACHE_TEST
    ExpertCacheTestAccess::observe(*this, ExpertCacheTestAccess::Point::BeforeRead);
#endif
    // aligned pieces straight into the slot; others widened to their pages unless that reads half as much again
    // (small pieces: tests' shapes; Flash-Next's are 400 KB and 50 KB)
    const bool aligned = j.off % kPage == 0 && j.len % kPage == 0 && uintptr_t(j.dst) % kPage == 0;
    const uint64_t a = j.off / kPage * kPage, b = (j.off + j.len + kPage - 1) / kPage * kPage;
    if (aligned || direct_ || (b - a) * 2 > uint64_t(j.len) * 3) {
      pread_all(j.fd, j.dst, j.len, j.off, j.len);
      bytes_ += j.len;
    } else {
      pread_all(j.fd, stage, size_t(b - a), a, size_t(j.off + j.len - a));
      std::memcpy(j.dst, stage + (j.off - a), j.len);
      bytes_ += b - a;
    }
    pending_[size_t(j.key)].fetch_sub(1, std::memory_order_release);
#ifdef NI_EXPERT_CACHE_TEST
    ExpertCacheTestAccess::observe(*this, ExpertCacheTestAccess::Point::ReadDone);
#endif
  }
  free(mem);
}

StoreStats ExpertCache::stats() const {
  std::lock_guard<std::mutex> lk(st_mu_);
  StoreStats s = st_;
  s.bytes = bytes_.load();
  return s;
}

void ExpertCache::reset_stats() {
  std::lock_guard<std::mutex> lk(st_mu_);
  st_ = StoreStats{};
  bytes_ = 0;
}

// ---- ExpertStore ----

// the heater's reach past each go or encoded wait: decode layers come ~0.3 ms apart (a prefill layer takes longer, but
// its waits are a small share of it)
constexpr double kHeatMs = 4.0;

ExpertStore::ExpertStore(const std::vector<std::string>& files, const std::vector<LayerSrc>& layers, uint32_t E,
                         SlotLayout lay, const Options& o)
    : lay_(lay), E_(E) {
  Gpu& g = Gpu::get();
  const size_t keys = layers.size() * E;
  size_t n = std::min(keys, o.bytes / lay_.bytes);
  // at most two buffers (the kernels address chunk << 28 | page)
  const uint32_t per = uint32_t(std::min<size_t>((g.max_buffer_length() - (size_t(1) << 20)) / lay_.bytes, size_t(1) << 26));
  if (n > size_t(2) * per) {
    logi("expert store: %zu slots exceed two Metal buffers; %u kept", n, 2 * per);
    n = size_t(2) * per;
  }
  const size_t c0 = std::min<size_t>(n, per), c1 = n - c0;
  chunk_[0] = g.alloc(std::max<size_t>(c0, 1) * lay_.bytes, "expert_arena0", /*tracked=*/false);
  chunk_[1] = c1 ? g.alloc(c1 * lay_.bytes, "expert_arena1", /*tracked=*/false) : chunk_[0];
  table_ = g.alloc(keys * 4, "expert_slots", /*tracked=*/false);
  ev_.reset(g.new_shared_event(), [](void* p) { CFRelease(p); });
  next_value_ = Gpu::event_value(ev_.get());
  ExpertCache::Memory mem;
  mem.chunk[0] = chunk_[0].as<uint8_t>();
  mem.chunk[1] = chunk_[1].as<uint8_t>();
  mem.per_chunk = per;
  mem.table = table_.as<uint32_t>();
  void* ev = ev_.get();
  ExpertCache::Sync sync{[ev] { return Gpu::event_value(ev); }, [ev](uint64_t v) { Gpu::signal_event_cpu(ev, v); },
                         Gpu::heat_on() ? std::function<void()>([] { Gpu::get().keep_warm(kHeatMs); }) : nullptr};
  cache_ = std::make_unique<ExpertCache>(files, layers, E, lay, uint32_t(n), mem, sync, o.cache);
  const StoreStats st = cache_->stats();
  logi("expert store: %u slots of %.2f MB (%.2f GB, %.1f%% of the %zu experts) in %s (up to %.2f GB each); %zu read "
       "at load in %.0f ms (%.2f GB/s)%s",
       cache_->slots(), lay_.bytes / 1e6, arena_bytes() / 1e9, 100.0 * cache_->slots() / double(keys), keys,
       c1 ? "two buffers" : "one buffer", double(per) * lay_.bytes / 1e9, cache_->filled(), cache_->fill_ms(),
       cache_->fill_ms() > 0 ? st.bytes / 1e6 / cache_->fill_ms() : 0.0,
       cache_->complete() ? " (all of them: no handshakes)" : "");
  cache_->reset_stats();
  if (!cache_->complete() && Gpu::heat_on()) g.keep_warm(0);
}

ExpertStore::~ExpertStore() {
  cache_.reset();
  if (Gpu::heat_on()) {
    const double t0 = Encoder::host_ms();
    while (Gpu::heat_inflight() > 0 && Encoder::host_ms() - t0 < 100) usleep(500);
    logi("GPU heater: %llu command buffers run, %d in flight at exit (after %.1f ms)",
         (unsigned long long)Gpu::heat_started(), Gpu::heat_inflight(), Encoder::host_ms() - t0);
  }
  Gpu& g = Gpu::get();
  if (chunk_[1].mtl && chunk_[1].mtl != chunk_[0].mtl) g.release(chunk_[1]);
  g.release(chunk_[0]);
  g.release(table_);
}

Experts ExpertStore::experts(int layer) const {
  Experts x;
  x.arena = true;
  x.a0 = chunk_[0];
  x.a1 = chunk_[1];
  x.slots = table_.view(size_t(layer) * E_ * 4, size_t(E_) * 4);
  x.lay = lay_;
  return x;
}

std::vector<Buffer> ExpertStore::buffers() const {
  std::vector<Buffer> v{chunk_[0], table_};
  if (chunk_[1].mtl != chunk_[0].mtl) v.push_back(chunk_[1]);
  return v;
}

void ExpertStore::begin(Encoder& e, int layer, const Buffer& ids, uint32_t n, const Buffer* pred, uint32_t npred,
                        int pred_layer, bool pred_all, bool scan, bool warm, int pred_floor) {
  if (cache_->complete()) return;
  ExpertCache::Req r;
  r.layer = layer;
  r.ids = ids.as<uint32_t>();
  r.n = n;
  r.pred = pred ? pred->as<uint32_t>() : nullptr;
  r.npred = pred ? npred : 0;
  r.pred_layer = pred_layer;
  r.pred_all = pred_all;
  r.scan = scan;
  r.pred_floor = pred_floor;
  // Heater (Gpu::heat_on: M5 by default), decode / verify only: the pass's first wait comes before any go
  if (warm && Gpu::heat_on()) Gpu::get().keep_warm(kHeatMs);
  r.warm = warm;
  r.ready = ++next_value_;
  r.go = ++next_value_;
  e.signal_event_inline(ev_.get(), r.ready);
  cache_->post(r);
  pending_go_ = r.go;
}

void ExpertStore::end(Encoder& e) {
  if (cache_->complete()) return;
  e.wait_event(ev_.get(), pending_go_);
}

std::string ExpertStore::describe() const {
  const size_t keys = size_t(cache_->layers()) * E_;
  char b[200];
  snprintf(b, sizeof b, "%u of %zu experts in memory (%.1f GB, %.0f%%)%s", cache_->slots(), keys, arena_bytes() / 1e9,
           100.0 * cache_->slots() / double(keys), cache_->complete() ? "" : ", the rest read from the SSD");
  return b;
}

}  // namespace ni::q20
