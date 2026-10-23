// The SSD prefix cache in the engine (the store: src/engine/prefix_cache.h). A slot's conversation becomes an entry:
// its tokens, its KV rows in blocks of kBlockRows positions, and the states it resumes from (the live state when a
// follow-up continues it as it is, its turn-boundary checkpoints), each a recurrent state with the drafter's context
// ring (DFlash2) as the slot had it; with the MTP head, the head's KV rows and the draft rows it gathered too. A
// restored state is the slot's, byte for byte, so a request resumed from it decodes as it would from the slot.
//
// Written: after a turn, once the server has been idle for NI_DISK_CACHE_IDLE_MS (10 s), by a thread of its own that
// copies a chunk at a time under the GPU turn while no request runs, and stops when one arrives (nothing is written
// between decode cycles; what it wrote stays for the next try); when a request is about to replace a conversation the
// SSD lacks (another conversation, or memory for one), what it lacks is copied at once and written in the background;
// when the server stops (save_disk_cache). Read: when a prompt matches a stored state deeper than its slot holds,
// before the prefill, which then starts there.
#include "engine/engine.h"
#include "engine/disk_wait.h"
#include "util/log.h"

#include <sys/stat.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <numeric>
#include <sstream>
#include <thread>

namespace ni {

namespace dc {

// The states' format, part of every namespace's digest. States on the SSD outlive the binary that wrote them: bump it
// whenever what a state's bytes mean changes without the model's disk_format() saying so (the order or precision of the
// KV rows, the recurrent state's layout, the drafter ring, what a snapshot holds). Older namespaces are then never read,
// and go once unused for 14 days.
constexpr int kStateFormat = 1;
constexpr int kBlockRows = 512;  // KV rows per block file: 10 MB on the 35B, 32 MB on the 27B, 18 MB on the 4B

double env_num(const char* name, double def) {
  const char* v = getenv(name);
  return v && *v ? atof(v) : def;
}

// rows [0, at) keep the requests that computed them
void cut_fills(std::vector<std::pair<int32_t, uint64_t>>& f, int at) {
  while (!f.empty() && (f.size() == 1 ? 0 : f[f.size() - 2].first) >= at) f.pop_back();
  if (!f.empty() && f.back().first > at) f.back().first = at;
}

// A payload as byte ranges of host memory (Metal's shared buffers are host memory on Apple silicon). A null range is
// skipped on the way in (rows past a cache's capacity) and reads as zeros on the way out.
struct Gather {
  std::vector<std::pair<uint8_t*, size_t>> seg;
  uint64_t bytes = 0;
  void add(void* p, size_t n) {
    if (n) seg.push_back({static_cast<uint8_t*>(p), n}), bytes += n;
  }
  template <class F> void each(uint64_t off, size_t n, F f) const {
    uint64_t at = 0;
    for (const auto& [p, len] : seg) {
      const uint64_t a = std::max(at, off), b = std::min(at + len, off + n);
      if (a < b) f(p ? p + (a - at) : nullptr, size_t(a - off), size_t(b - a));
      at += len;
      if (at >= off + n) break;
    }
  }
  void out(uint64_t off, uint8_t* dst, size_t n) const {
    each(off, n, [&](uint8_t* p, size_t d, size_t k) {
      if (p) memcpy(dst + d, p, k);
      else memset(dst + d, 0, k);
    });
  }
  void in(uint64_t off, const uint8_t* src, size_t n) const {
    each(off, n, [&](uint8_t* p, size_t d, size_t k) {
      if (p) memcpy(p, src + d, k);
    });
  }
};

// rows [a, b) of every attention cache ([heads][cap][row]); rows at or past `keep` are left out (null ranges)
Gather kv_rows(const std::vector<KvRows>& t, int cap, int a, int b, int keep) {
  Gather g;
  const int in = std::clamp(keep - a, 0, b - a);
  for (const KvRows& r : t)
    for (int h = 0; h < r.heads; ++h) {
      g.add(r.buf.as<uint8_t>() + (size_t(h) * size_t(cap) + size_t(a)) * r.row_bytes, size_t(in) * r.row_bytes);
      g.add(nullptr, size_t(b - a - in) * r.row_bytes);
    }
  return g;
}

uint64_t kv_layout(const std::vector<KvRows>& t) {
  pcache::Hasher h;
  h.u64(t.size());
  for (const KvRows& r : t) h.u64(uint64_t(r.heads)).u64(r.row_bytes);
  return h.done().lo;
}

// a piece's header data: what it holds, checked before its bytes go anywhere
struct KvMeta {
  uint32_t a = 0, b = 0;  // rows [a, b)
  uint64_t layout = 0;
};
struct StMeta {
  uint64_t layout = 0;  // the state buffers' sizes
};
template <class T> std::string meta_of(const T& m) { return std::string(reinterpret_cast<const char*>(&m), sizeof m); }
template <class T> bool meta_is(const std::string& s, const T& m) {
  return s.size() == sizeof m && memcmp(s.data(), &m, sizeof m) == 0;
}

// A folder's files as "name size mtime" lines (not recursive), and config.json's CRC: the model the states belong to
std::string folder_id(const std::string& dir) {
  std::vector<std::string> lines;
  std::error_code ec;
  for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
    std::error_code e2;
    struct stat st;
    if (!de.is_regular_file(e2) || stat(de.path().c_str(), &st) != 0) continue;
    char b[600];
    snprintf(b, sizeof b, "  %s %lld %lld.%09ld\n", de.path().filename().c_str(), (long long)st.st_size,
             (long long)st.st_mtimespec.tv_sec, (long)st.st_mtimespec.tv_nsec);
    lines.push_back(b);
  }
  std::sort(lines.begin(), lines.end());
  std::string s;
  for (const std::string& l : lines) s += l;
  std::ifstream f(dir + "/config.json");
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string c = ss.str();
  return s + "  config.json CRC-32C " + std::to_string(pcache::crc32c(0, c.data(), c.size())) + "\n";
}

// The images among an entry's first N rows: per image its span (pos, n, grid t, h, w: int32) and its digest
std::string images_of(const std::vector<vision::ImageSpan>& spans, const std::vector<vision::Digest>& digests, int N) {
  std::string s;
  for (size_t k = 0; k < spans.size() && k < digests.size(); ++k) {
    const vision::ImageSpan& sp = spans[k];
    if (sp.pos >= N) continue;
    const int32_t v[5] = {sp.pos, sp.n, sp.grid_t, sp.grid_h, sp.grid_w};
    s.append(reinterpret_cast<const char*>(v), sizeof v);
    s.append(reinterpret_cast<const char*>(digests[k].data()), digests[k].size());
  }
  return s;
}

void images_from(const std::string& s, std::vector<vision::ImageSpan>* spans, std::vector<vision::Digest>* digests) {
  constexpr size_t kRec = 5 * 4 + std::tuple_size<vision::Digest>::value;
  for (size_t at = 0; at + kRec <= s.size(); at += kRec) {
    int32_t v[5];
    memcpy(v, s.data() + at, sizeof v);
    vision::ImageSpan sp;
    sp.pos = v[0], sp.n = v[1], sp.grid_t = v[2], sp.grid_h = v[3], sp.grid_w = v[4];
    vision::Digest dg;
    memcpy(dg.data(), s.data() + at + sizeof v, dg.size());
    spans->push_back(sp);
    digests->push_back(dg);
  }
}

uint64_t ram_bytes() {
  uint64_t m = 0;
  size_t n = sizeof m;
  return sysctlbyname("hw.memsize", &m, &n, nullptr, 0) == 0 ? m : 0;
}

}  // namespace dc

using dc::Gather;
using dc::kBlockRows;

struct Engine::DiskPlan {
  int slot = 0;
  uint64_t version = 0, requests = 0;
  pcache::Entry entry;
  std::vector<int> src;  // per entry.snaps[i]: -1 the live state, else the checkpoint it is
  std::vector<std::pair<pcache::Kind, pcache::Key>> pieces() const {
    std::vector<std::pair<pcache::Kind, pcache::Key>> p;
    for (const pcache::Key& b : entry.blocks) p.push_back({pcache::kKv, b});
    for (const pcache::Snap& s : entry.snaps) p.push_back({pcache::kState, s.state});
    return p;
  }
};

struct Engine::Disk {
  pcache::Store store;
  std::string root, ns, about;
  pcache::Key ns_key;
  uint64_t quota = 0;
  bool mtp = false;        // MTP decoding: the MTP head's KV rows and its gathered draft rows belong to a state
  int min_pos = 1024;      // states nearer the start than this are not saved (quick to prefill again)
  double idle_ms = 10000;  // write-behind: after this long without requests
  uint64_t capture_max = 0;  // most bytes copied at once when a request replaces a conversation the SSD lacks
  std::atomic<bool> ready{false}, quit{false};
  std::atomic<double> quiet_since{0};
  std::mutex mu;
  std::condition_variable cv;
  bool stopped = false;
  std::thread thread;
  struct Piece {
    pcache::Kind kind;
    pcache::Key key;
    std::string meta;
    Gather g;                         // where its bytes are in the slot
    std::unique_ptr<uint8_t[]> data;  // or a copy of them
  };
  // what a request left behind: the copies of the pieces the SSD lacked, written by the thread
  struct Captured {
    std::unique_ptr<DiskPlan> plan;
    std::unique_ptr<pcache::Store::Job> job;
    std::vector<Piece> pieces;
    double at = 0;
  };
  std::deque<std::unique_ptr<Captured>> captured;
  // the tokens of every capture not yet on the SSD (queued or being written), so a request whose prompt continues one
  // waits for it rather than reading its prompt again (a short side request takes a slot and the conversation's next
  // request follows within the time its copy takes to write)
  struct Unwritten {
    std::vector<int32_t> tokens;
    uint64_t bytes = 0;
  };
  std::list<Unwritten> unwritten;
  // what the wait is weighed against: how fast captures reach the SSD (bytes/ms, smoothed over writes of 8 MB or more)
  // and how fast the engine reads uncached prompt rows (tokens/ms, smoothed over prefills of 256 rows or more); 0 until
  // measured
  std::atomic<double> write_bpms{0}, prefill_tpms{0};
  static void smooth(std::atomic<double>& v, double x) {
    const double o = v.load();
    v = o > 0 ? 0.7 * o + 0.3 * x : x;
  }
  std::atomic<bool> no_room_said{false};

  // whether the cache has room at all (a volume with less free space than the floor leaves it none: said once)
  bool room() {
    const uint64_t lim = store.limit();
    if (lim == 0 && !no_room_said)
      logi("SSD prefix cache: no room: the disk has less free space than the 10 GiB the cache leaves free; nothing is "
           "saved until there is");
    no_room_said = lim == 0;
    return lim > 0;
  }

  // the pieces of plan p the store lacks, with where their bytes are in the slot (valid while its version is p's)
  std::vector<Piece> missing(const Engine& en, const DiskPlan& p, const Slot& slot) {
    std::vector<Piece> out;
    const Seq& s = *slot.seq;
    std::vector<KvRows> rows;
    en.model_->disk_rows(s, mtp, rows);
    const uint64_t lay = dc::kv_layout(rows);
    const int N = int(p.entry.tokens.size());
    for (size_t k = 0; k < p.entry.blocks.size(); ++k) {
      if (store.has(pcache::kKv, p.entry.blocks[k])) continue;
      const int a = int(k) * kBlockRows, b = std::min(N, a + kBlockRows);
      out.push_back({pcache::kKv, p.entry.blocks[k], dc::meta_of(dc::KvMeta{uint32_t(a), uint32_t(b), lay}),
                     dc::kv_rows(rows, s.cap, a, b, b), nullptr});
    }
    for (size_t i = 0; i < p.entry.snaps.size(); ++i) {
      const pcache::Snap& sn = p.entry.snaps[i];
      if (store.has(pcache::kState, sn.state)) continue;
      const int c = p.src[i];
      std::vector<Buffer> bufs;
      en.model_->disk_state(s, c < 0 ? nullptr : &slot.ckpts[size_t(c)], bufs);
      if (en.dfl_) {
        const DFlashSeq& r = c < 0 ? *slot.dseq : *slot.dckpts[size_t(c)];
        bufs.push_back(r.ring_k), bufs.push_back(r.ring_v);
      }
      Gather g;
      pcache::Hasher h;
      for (const Buffer& b : bufs) g.add(b.ptr(), b.size), h.u64(b.size);
      out.push_back({pcache::kState, sn.state, dc::meta_of(dc::StMeta{h.done().lo}), std::move(g), nullptr});
    }
    return out;
  }

  bool write_captured(Captured& c) {
    const double w0 = now_ms();
    uint64_t bytes = 0;
    for (Piece& pc : c.pieces) {
      const uint8_t* src = pc.data.get();
      if (!store.write(*c.job, pc.kind, pc.key, pc.meta, pc.g.bytes, [&](uint64_t off, uint8_t* dst, size_t n) {
            memcpy(dst, src + off, n);
            return true;
          })) {
        logi("SSD cache: the conversation slot %d held was not saved (no room within the cache's limit, or a write "
             "failed)", c.plan->slot);
        return false;
      }
      bytes += pc.g.bytes;
      pc.data.reset();
    }
    if (!store.put(*c.job, c.plan->entry)) return false;
    if (bytes >= (8u << 20)) smooth(write_bpms, double(bytes) / std::max(now_ms() - w0, 1.0));
    logi("SSD cache: saved the conversation slot %d held (%zu tokens, %.0f MB written) in %.0f ms", c.plan->slot,
         c.plan->entry.tokens.size(), bytes / 1e6, now_ms() - c.at);
    return true;
  }
};

void Engine::DiskFree::operator()(Disk* d) const { delete d; }

int Engine::reuse_start(const Slot& slot, size_t common, int n) {
  if (common == slot.tokens.size() && int(common) < n) return int(common);
  int best = 0;
  for (const TurnCheckpoint& c : slot.ckpts)
    if (c.pos > 0 && c.pos <= int(common) && c.pos < n) best = std::max(best, c.pos);
  return best;
}

void Engine::disk_filled(Slot& slot, int start, int end) {
  dc::cut_fills(slot.fills, start);
  if (end > start) {
    uint64_t id = 0;
    arc4random_buf(&id, sizeof id);
    slot.fills.push_back({int32_t(end), id});
  }
  if (disk_) {
    disk_->quiet_since = now_ms();
    disk_->cv.notify_all();
  }
}

void Engine::disk_start() {
  std::string want = opts_.disk_cache;
  if (want.empty()) {
    const char* v = getenv("NI_DISK_CACHE");
    want = v && *v ? v : "0";
  }
  uint64_t quota = 0;
  if (!pcache::parse_size(want, &quota)) {
    logi("SSD prefix cache: off (--disk-cache takes auto, 0 or a size such as 16G, not \"%s\")", want.c_str());
    return;
  }
  if (quota == 0) return;
  const std::string fmt = model_->disk_format();
  if (fmt.empty()) {
    logi("SSD prefix cache: off: not available for this model's states yet (the Qwen3.5-family models and Dory are)");
    return;
  }
  std::unique_ptr<Disk, DiskFree> d(new Disk());
  d->quota = quota;
  d->mtp = spec_;
  d->min_pos = std::max(64, int(dc::env_num("NI_DISK_CACHE_MIN", 1024)));
  d->idle_ms = std::max(0.0, dc::env_num("NI_DISK_CACHE_IDLE_MS", 10000));
  d->capture_max = std::min<uint64_t>(pcache::kGiB, dc::ram_bytes() / 32);
  const char* root = getenv("NI_DISK_CACHE_DIR");
  d->root = root && *root ? root : hub::home_dir() + "/prefix-cache";
  const std::string mode = dfl_ ? "DFlash2 drafter (" + std::to_string(dfl_->bits()) + "-bit), its context ring"
                           : spec_ ? std::string("MTP head: its KV rows and gathered draft rows")
                                   : std::string("no drafter");
  d->about = "Kapitsa SSD prefix cache, state format " + std::to_string(dc::kStateFormat) +
             " (src/engine/engine_disk.cpp)\nmodel: " + opts_.model_dir + "\n" +
             dc::folder_id(opts_.model_dir) + "states: " + fmt + "\ndecoding: " + mode + "\n" +
             (dfl_ ? "drafter: " + dfl_dir_ + "\n" + dc::folder_id(dfl_dir_) : std::string()) +
             "KV rows per block: " + std::to_string(kBlockRows) + "\n";
  d->ns_key = pcache::Hasher().str(d->about).done();
  d->ns = d->ns_key.hex();
  disk_ = std::move(d);
  disk_->thread = std::thread([this] { disk_loop(); });
}

// ---- what a slot's state is on the SSD ----

std::unique_ptr<Engine::DiskPlan> Engine::disk_plan(const Slot& slot) const {
  const Disk& d = *disk_;
  if (!slot.seq) return nullptr;
  const int live = int(slot.tokens.size());
  struct C {
    int pos, src;
    uint32_t flags;
  };
  std::vector<C> cs;
  if (slot.live_ok && live >= d.min_pos) cs.push_back({live, -1, pcache::Snap::kLive});
  for (size_t i = 0; i < slot.ckpts.size(); ++i) {
    const TurnCheckpoint& c = slot.ckpts[i];
    // the prefix checkpoint has no drafter ring (dckpts[i] is null): a restore from it could not rebuild the ring
    if (dfl_ && c.prefix) continue;
    std::vector<Buffer> bufs;  // (a checkpoint without its whole state, as Dory's rings when memory was short: none)
    model_->disk_state(*slot.seq, &c, bufs);
    const bool whole = std::all_of(bufs.begin(), bufs.end(), [](const Buffer& b) { return b.mtl != nullptr; });
    if (c.pos >= d.min_pos && c.pos <= live && !(slot.live_ok && c.pos == live) && whole)
      cs.push_back({c.pos, int(i), c.query ? pcache::Snap::kQuery : 0u});
  }
  if (cs.empty()) return nullptr;
  std::sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.pos < b.pos; });
  const int N = cs.back().pos;
  if (N > slot.seq->cap || slot.fills.empty() || slot.fills.back().first < N) return nullptr;
  auto p = std::make_unique<DiskPlan>();
  p->slot = int(&slot - slots_.data());
  p->version = slot.version;
  pcache::Entry& e = p->entry;
  e.block_rows = kBlockRows;
  e.tokens.assign(slot.tokens.begin(), slot.tokens.begin() + N);
  e.fills = slot.fills;
  dc::cut_fills(e.fills, N);
  // keys: rows [a, b) after `prev` (the key of rows [0, a)), named by their tokens and the requests that computed them
  auto chain = [&](const pcache::Key& prev, int a, int b) {
    pcache::Hasher h;
    h.key(prev).u64(uint64_t(a)).u64(uint64_t(b)).i32s(e.tokens.data() + a, size_t(b - a));
    int lo = 0;
    for (const auto& [end, id] : e.fills) {
      const int s0 = std::max(lo, a), s1 = std::min(int(end), b);
      if (s0 < s1) h.u64(uint64_t(s0)).u64(uint64_t(s1)).u64(id);
      lo = end;
    }
    return h.done();
  };
  std::vector<pcache::Key> full;  // full[k]: rows [0, (k + 1) * kBlockRows)
  for (int a = 0; a < N; a += kBlockRows) {
    const int b = std::min(N, a + kBlockRows);
    const pcache::Key k = chain(a == 0 ? d.ns_key : full.back(), a, b);
    e.blocks.push_back(k);
    if (b - a == kBlockRows) full.push_back(k);
  }
  for (const C& c : cs) {
    pcache::Snap sn;
    sn.pos = c.pos;
    sn.flags = c.flags;
    if (dfl_) {
      const DFlashSeq& r = c.src < 0 ? *slot.dseq : *slot.dckpts[size_t(c.src)];
      sn.dctx = r.ctx, sn.dvalid = r.valid_lo;
    }
    const int fb = c.pos / kBlockRows;
    const pcache::Key at = chain(fb == 0 ? d.ns_key : full[size_t(fb - 1)], fb * kBlockRows, c.pos);
    sn.state = pcache::Hasher().key(at).u64(sn.flags).u64(uint32_t(sn.dctx)).u64(uint32_t(sn.dvalid)).done();
    e.snaps.push_back(sn);
    p->src.push_back(c.src);
  }
  if (d.mtp && q35_) e.vocab = q35_->draft_vocab_ids(*slot.seq, &e.vocab_scanned);
  e.tag = dory_ ? uint32_t(slot.loops) : 0;  // (a looped model's KV holds the loop count that computed it)
  e.attach = dc::images_of(slot.img_spans, slot.img_digests, N);
  return p;
}

// ---- writes ----

bool Engine::disk_save_slot(int i, bool held) {
  Disk& d = *disk_;
  Slot& slot = slots_[size_t(i)];
  const double t0 = now_ms();
  // the GPU turn for a chunk: held throughout, or taken only while the engine is idle (no request in flight)
  auto take = [&] {
    if (held) return true;
    if (d.quit || !turn_.try_lock()) return false;
    if (active_ > 0) {
      turn_.unlock();
      return false;
    }
    return true;
  };
  auto give_back = [&] {
    if (!held) turn_.unlock();
  };
  auto done = [&](uint64_t version) {  // this version is saved, or has nothing the SSD can take
    if (!take()) return;
    if (slot.version == version) slot.disk_version = version;
    give_back();
  };
  if (!held) d.store.measure();
  if (!take()) return false;
  if (slot.busy || slot.version == slot.disk_version) {
    give_back();
    return true;
  }
  std::unique_ptr<DiskPlan> p = d.room() ? disk_plan(slot) : nullptr;
  if (!p) {
    slot.disk_version = slot.version;
    give_back();
    return true;
  }
  p->requests = requests_.load();
  std::vector<Disk::Piece> todo = d.missing(*this, *p, slot);
  give_back();
  auto job = d.store.begin(p->pieces());
  uint64_t bytes = 0;
  int nkv = 0, nst = 0;
  for (Disk::Piece& pc : todo) {
    bool gave_up = false;
    const Gather& g = pc.g;
    const bool ok = d.store.write(*job, pc.kind, pc.key, pc.meta, g.bytes, [&](uint64_t off, uint8_t* dst, size_t n) {
      if (!take()) return gave_up = true, false;
      const bool same = slot.version == p->version && (held || requests_.load() == p->requests);
      if (same) g.out(off, dst, n);
      give_back();
      gave_up = !same;
      return same;
    });
    if (!ok) {
      if (gave_up) return false;  // a request came: later (what was written stays)
      logi("SSD cache: slot %d was not saved (no room within the cache's limit of %s, or a write failed)", i,
           pcache::gib(d.store.limit()).c_str());
      done(p->version);
      return true;
    }
    bytes += g.bytes;
    ++(pc.kind == pcache::kKv ? nkv : nst);
  }
  if (!d.store.put(*job, p->entry)) {
    logi("SSD cache: slot %d was not saved (its entry could not be written)", i);
    done(p->version);
    return true;
  }
  done(p->version);
  logi("SSD cache: saved slot %d (%zu tokens, %zu restore point%s; %d of %zu KV blocks and %d state%s new: %.0f MB) in "
       "%.0f ms", i, p->entry.tokens.size(), p->entry.snaps.size(), p->entry.snaps.size() == 1 ? "" : "s", nkv,
       p->entry.blocks.size(), nst, nst == 1 ? "" : "s", bytes / 1e6, now_ms() - t0);
  return true;
}

void Engine::disk_leaving(Slot& slot, int start) {
  if (!disk_ || !disk_->ready || slot.version == slot.disk_version) return;
  Disk& d = *disk_;
  if (!d.room()) return;
  // what is left behind: the newest checkpoint (the live state when no checkpoint is saved). A follow-up resumes from
  // the checkpoint or the live state: the live state it leaves (a reply as generated, which the next prompt renders
  // otherwise) is not resumed from.
  int deep = 0;
  for (const TurnCheckpoint& c : slot.ckpts)
    if (c.pos >= d.min_pos && c.pos <= int(slot.tokens.size())) deep = std::max(deep, c.pos);
  if (deep == 0 && slot.live_ok && int(slot.tokens.size()) >= d.min_pos) deep = int(slot.tokens.size());
  if (start >= deep) return;
  std::unique_ptr<DiskPlan> p = disk_plan(slot);
  if (!p) return;
  const double t0 = now_ms();
  auto c = std::make_unique<Disk::Captured>();
  c->pieces = d.missing(*this, *p, slot);
  uint64_t bytes = 0;
  for (const Disk::Piece& pc : c->pieces) bytes += pc.g.bytes;
  if (bytes > d.capture_max) {
    // too much to hold a copy of in memory (a 27B conversation of 16k tokens is 1.4-1.6 GB): written to the SSD straight
    // from the slot, the request that replaces it waiting for the write (no more than 3 s, and less than reading the
    // rows again would take: the same weighing as a wait for a capture), or else the conversation is replaced unsaved
    // (only when the slot is the request's own: a slot dropped for memory is copied with the GPU turn held, and a
    // write of seconds would stall every other stream)
    const int lost = int(p->entry.tokens.size()) - start;
    const disk_wait::Decision dec = disk_wait::decide(double(bytes), lost, d.write_bpms, d.prefill_tpms);
    if (!slot.busy || !dec.wait) {
      if (!slot.busy)
        logi("SSD cache: slot %d's conversation (%zu tokens) is replaced before it was saved: the %.0f MB the SSD lacks "
             "of it exceed the %.0f MB copied at once (a slot dropped for memory is not written with the GPU turn held)",
             p->slot, p->entry.tokens.size(), bytes / 1e6, d.capture_max / 1e6);
      else
        logi("SSD cache: slot %d's conversation (%zu tokens) is replaced before it was saved: the %.0f MB the SSD lacks "
             "of it exceed the %.0f MB copied at once, and writing them (about %.0f ms) would take longer than reading "
             "its %d rows again (about %.0f ms, at most 3 s)", p->slot, p->entry.tokens.size(), bytes / 1e6,
             d.capture_max / 1e6, dec.write_ms, lost, dec.prefill_ms ? dec.prefill_ms : disk_wait::kMaxWaitMs);
      return;
    }
    auto job = d.store.begin(p->pieces());
    const double w0 = now_ms();
    turn_.unlock();  // (the request's own slot is its alone: others take the GPU turn meanwhile)
    bool ok = true;
    for (Disk::Piece& pc : c->pieces) {
      ok = d.store.write(*job, pc.kind, pc.key, pc.meta, pc.g.bytes, [&](uint64_t off, uint8_t* dst, size_t n) {
        pc.g.out(off, dst, n);
        return true;
      });
      if (!ok) break;
    }
    ok = ok && d.store.put(*job, p->entry);
    const double w1 = now_ms();  // (the write alone: the write speed is not the time spent waiting for the turn)
    turn_.lock();
    if (!ok) {
      logi("SSD cache: slot %d's conversation (%zu tokens) could not be written before it is replaced (no room within "
           "the cache's limit, or a write failed)", p->slot, p->entry.tokens.size());
      return;
    }
    if (bytes >= (8u << 20)) Disk::smooth(d.write_bpms, double(bytes) / std::max(w1 - w0, 1.0));
    slot.disk_version = slot.version;
    logi("SSD cache: slot %d's conversation (%zu tokens, %.0f MB) was too large to copy in memory: written to the SSD "
         "straight from the slot before it is replaced, in %.0f ms (the request waited %.0f ms)", p->slot,
         p->entry.tokens.size(), bytes / 1e6, w1 - w0, now_ms() - t0);
    return;
  }
  c->job = d.store.begin(p->pieces());
  // the copy: a request's own slot is its alone (busy), so the GPU turn goes to other requests meanwhile; a slot
  // dropped for memory is copied with the turn held (nothing may pick it before it is dropped)
  if (slot.busy) turn_.unlock();
  for (Disk::Piece& pc : c->pieces) {
    pc.data.reset(new uint8_t[pc.g.bytes]);
    pc.g.out(0, pc.data.get(), pc.g.bytes);
  }
  if (slot.busy) turn_.lock();
  c->at = t0;
  logd("SSD cache: slot %d's conversation (%zu tokens) copied for the SSD before it is replaced (%.0f MB in %.1f ms)",
       p->slot, p->entry.tokens.size(), bytes / 1e6, now_ms() - t0);
  c->plan = std::move(p);
  slot.disk_version = slot.version;
  {
    std::lock_guard<std::mutex> lk(d.mu);
    d.unwritten.push_back({c->plan->entry.tokens, bytes});
    d.captured.push_back(std::move(c));
  }
  d.cv.notify_all();
}

void Engine::disk_loop() {
  Disk& d = *disk_;
  {
    pcache::Store::Options o;
    o.root = d.root;
    o.ns = d.ns;
    o.about = d.about;
    o.quota = d.quota;
    o.cancel = &d.quit;
    std::string why;
    if (!d.store.open(o, &why)) {
      logi("SSD prefix cache: off (%s)", why.c_str());
      return;
    }
    const pcache::Store::Stats st = d.store.stats();
    logi("SSD prefix cache: %s, up to %s (%s; --disk-cache 0 turns it off): %zu conversation%s taken back (%s)",
         hub::tilde(d.store.dir()).c_str(), pcache::gib(st.limit).c_str(),
         d.quota == pcache::kAuto ? "auto: a quarter of the free space beyond 10 GiB, at most 32 GiB"
                                  : ("--disk-cache " + pcache::gib(d.quota) + ", within a quarter of the free space beyond 10 GiB").c_str(),
         st.entries, st.entries == 1 ? "" : "s", pcache::gib(st.held).c_str());
    d.ready = true;
  }
  double measured = now_ms(), touched = now_ms(), quiet = now_ms();
  uint64_t seen = requests_.load();
  for (;;) {
    std::unique_ptr<Disk::Captured> c;
    {
      std::unique_lock<std::mutex> lk(d.mu);
      d.cv.wait_for(lk, std::chrono::milliseconds(500), [&] { return d.quit.load() || !d.captured.empty(); });
      if (d.quit) return;
      if (!d.captured.empty()) {
        c = std::move(d.captured.front());
        d.captured.pop_front();
      }
    }
    if (c) {
      d.write_captured(*c);
      {
        std::lock_guard<std::mutex> lk(d.mu);
        d.unwritten.pop_front();  // (written in the order queued)
      }
      d.cv.notify_all();
      continue;
    }
    const double now = now_ms();
    if (now - measured > 60000) d.store.measure(), measured = now;
    if (now - touched > 3600e3) d.store.touch(), touched = now;
    if (const uint64_t rq = requests_.load(); rq != seen) seen = rq, quiet = now;
    if (now - std::max(quiet, d.quiet_since.load()) < d.idle_ms) continue;
    // write-behind: an idle slot that changed since it was saved, the most recently used first
    int pick = -1;
    if (turn_.try_lock()) {
      if (active_ == 0)
        for (size_t i = 0; i < slots_.size(); ++i) {
          const Slot& s = slots_[i];
          if (!s.busy && s.version != s.disk_version && (pick < 0 || s.last_used > slots_[size_t(pick)].last_used))
            pick = int(i);
        }
      turn_.unlock();
    }
    if (pick >= 0 && !disk_save_slot(pick, false)) quiet = now_ms();  // a request came: after the next quiet spell
  }
}

void Engine::save_disk_cache(double budget_ms) {
  if (!disk_) return;
  Disk& d = *disk_;
  {
    std::lock_guard<std::mutex> lk(d.mu);
    if (d.stopped) return;
    d.stopped = true;
  }
  d.quit = true;
  d.cv.notify_all();
  if (d.thread.joinable()) d.thread.join();  // a save in progress gives up at its next chunk; copies are written below
  if (!d.ready) return;
  const double t0 = now_ms();
  const uint64_t w0 = d.store.stats().written;
  int saved = 0;
  for (;;) {
    std::unique_ptr<Disk::Captured> c;
    {
      std::lock_guard<std::mutex> lk(d.mu);
      if (d.captured.empty()) break;
      c = std::move(d.captured.front());
      d.captured.pop_front();
    }
    saved += d.write_captured(*c);
  }
  {
    std::lock_guard<TurnLock> lk(turn_);  // nothing changes while it writes
    std::vector<int> order(slots_.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return slots_[size_t(a)].last_used > slots_[size_t(b)].last_used; });
    for (int i : order) {
      const Slot& s = slots_[size_t(i)];
      if (s.busy || s.version == s.disk_version) continue;
      if (now_ms() - t0 > budget_ms) {
        logi("SSD prefix cache: out of time (%.0f ms) as the server stops: other conversations are not saved", budget_ms);
        break;
      }
      const uint64_t w = d.store.stats().written;
      disk_save_slot(i, true);
      saved += d.store.stats().written > w;
    }
  }
  const uint64_t w = d.store.stats().written - w0;
  if (saved) logi("SSD prefix cache: %d conversation%s saved as the server stops (%.0f MB in %.0f ms)", saved,
                  saved == 1 ? "" : "s", w / 1e6, now_ms() - t0);
}

// ---- reads ----

void Engine::disk_note_prefill(int rows, double ms) {
  if (disk_ && rows >= 256) Disk::smooth(disk_->prefill_tpms, rows / std::max(ms, 1.0));
}

pcache::Hit Engine::disk_lookup(const std::vector<int>& keys, int n, int ram, uint32_t tag, const vision::Prompt* mm) {
  if (!disk_ || !disk_->ready) return {};
  // a prompt with images: every image in the reused prefix the same image as stored (span and digest), as in RAM
  pcache::Store::Reusable same_images;
  if (mm)
    same_images = [mm](const pcache::Entry& e, size_t l) {
      std::vector<vision::ImageSpan> spans;
      std::vector<vision::Digest> digests;
      dc::images_from(e.attach, &spans, &digests);
      return std::min(l, vision::common_prefix(e.tokens, spans, digests, *mm));
    };
  // a capture of the conversation this prompt continues that is still being written: wait for it, with the GPU turn
  // given up as a read does, when that is quicker than reading the rows again. The wait is the bytes queued up to and
  // including that capture at the measured write speed (a state too large for capture_max is never queued), against
  // the rows the SSD copy would save at the measured prefill speed; it is cut off at that time and at 3 s. Before
  // either is measured, a copy that saves fewer than 2048 rows is not waited for.
  {
    Disk& d = *disk_;
    std::unique_lock<std::mutex> lk(d.mu);
    auto continues = [&](double* ahead_bytes, int* saved_rows) {
      uint64_t ahead = 0;
      for (const Disk::Unwritten& u : d.unwritten) {
        ahead += u.bytes;
        size_t l = 0;
        while (l < u.tokens.size() && l < keys.size() && u.tokens[l] == keys[l]) ++l;
        if (int(l) > ram && int(l) >= d.min_pos) {
          if (ahead_bytes) *ahead_bytes = double(ahead);
          if (saved_rows) *saved_rows = int(l) - ram;
          return true;
        }
      }
      return false;
    };
    double ahead = 0;
    int rows = 0;
    if (continues(&ahead, &rows)) {
      const disk_wait::Decision dec = disk_wait::decide(ahead, rows, d.write_bpms, d.prefill_tpms);
      const double limit = dec.limit_ms;
      const bool wait = dec.wait;
      if (!wait && dec.write_ms > 0)
        logi("SSD cache: the request does not wait for the copy of its conversation (%.0f MB queued, about %.0f ms; "
             "its %d rows read again in about %.0f ms)", ahead / 1e6, dec.write_ms, rows, dec.prefill_ms);
      if (wait) {
        const double t0 = now_ms();
        turn_.unlock();
        d.cv.wait_for(lk, std::chrono::milliseconds(int64_t(limit)),
                      [&] { return d.quit.load() || !continues(nullptr, nullptr); });
        lk.unlock();
        turn_.lock();
        logi("SSD cache: the request waited %.0f ms for the copy of its conversation to be written (limit %.0f ms, %d "
             "rows saved)", now_ms() - t0, limit, rows);
      }
    }
  }
  const pcache::Hit h = disk_->store.lookup(keys.data(), int(keys.size()), n, tag, same_images);
  return h && h.pos > ram ? h : pcache::Hit{};
}

bool Engine::disk_restore(Slot& slot, const pcache::Hit& hit) {
  Disk& d = *disk_;
  pcache::Entry e;
  if (!d.store.pin(hit.entry, &e)) return false;
  if (hit.snap >= int(e.snaps.size())) {
    d.store.unpin(hit.entry, false);
    return false;
  }
  const pcache::Snap sn = e.snaps[size_t(hit.snap)];
  const int P = sn.pos, N = int(e.tokens.size());
  Seq& s = *slot.seq;
  std::vector<KvRows> rows;
  model_->disk_rows(s, d.mtp, rows);
  const uint64_t lay = dc::kv_layout(rows);
  const int nb = (P + kBlockRows - 1) / kBlockRows;
  std::vector<Gather> blocks;
  for (int k = 0; k < nb; ++k) {
    const int a = k * kBlockRows, b = std::min(N, a + kBlockRows);
    blocks.push_back(dc::kv_rows(rows, s.cap, a, b, s.cap));
  }
  std::vector<Buffer> bufs;
  model_->disk_state(s, nullptr, bufs);
  if (dfl_) bufs.push_back(slot.dseq->ring_k), bufs.push_back(slot.dseq->ring_v);
  Gather st;
  pcache::Hasher h;
  for (const Buffer& b : bufs) st.add(b.ptr(), b.size), h.u64(b.size);
  const uint64_t st_lay = h.done().lo;
  s.n_pend = 0;
  s.pend_tree = false;
  const double t0 = now_ms();
  const uint64_t r0 = d.store.stats().read;
  // the reads take no GPU turn: the slot is this request's alone, and other requests go on meanwhile. The pieces (the
  // state first, the largest, then the KV blocks) are read by up to 4 threads at once: the SSD serves parallel reads
  // faster than one.
  turn_.unlock();
  std::atomic<int> next{0};
  std::atomic<bool> ok{true};
  auto reader = [&] {
    for (int i; ok && (i = next++) <= nb;) {
      const int k = i - 1;  // (item 0: the state)
      std::string meta;
      uint64_t got = 0;
      bool r;
      if (k >= 0) {
        const int a = k * kBlockRows, b = std::min(N, a + kBlockRows);
        const Gather& g = blocks[size_t(k)];
        r = d.store.read(pcache::kKv, e.blocks[size_t(k)], &meta, [&](uint64_t off, const uint8_t* src, size_t n) {
              if (off == 0 && !dc::meta_is(meta, dc::KvMeta{uint32_t(a), uint32_t(b), lay})) return false;
              g.in(off, src, n);
              got += n;
              return true;
            }) && got == g.bytes;
      } else {
        r = d.store.read(pcache::kState, sn.state, &meta, [&](uint64_t off, const uint8_t* src, size_t n) {
              if (off == 0 && !dc::meta_is(meta, dc::StMeta{st_lay})) return false;
              st.in(off, src, n);
              got += n;
              return true;
            }) && got == st.bytes;
      }
      if (!r) ok = false;
    }
  };
  {
    std::vector<std::thread> more;
    for (int i = 1; i < std::min(4, nb + 1); ++i) more.emplace_back(reader);
    reader();
    for (std::thread& t : more) t.join();
  }
  turn_.lock();
  d.store.unpin(hit.entry, ok);
  const int idx = int(&slot - slots_.data());
  if (!ok) {
    logi("SSD cache: a stored conversation could not be read into slot %d; its prompt is read from the start", idx);
    return false;
  }
  slot.tokens.assign(e.tokens.begin(), e.tokens.begin() + P);
  slot.fills = e.fills;
  dc::cut_fills(slot.fills, P);
  slot.img_spans.clear();
  slot.img_digests.clear();
  {  // the images in the restored rows
    std::vector<vision::ImageSpan> spans;
    std::vector<vision::Digest> digests;
    dc::images_from(e.attach, &spans, &digests);
    for (size_t k = 0; k < spans.size(); ++k)
      if (spans[k].pos + spans[k].n <= P) slot.img_spans.push_back(spans[k]), slot.img_digests.push_back(digests[k]);
  }
  for (TurnCheckpoint& c : slot.ckpts) c.pos = -1;
  if (dfl_) slot.dseq->ctx = sn.dctx, slot.dseq->valid_lo = sn.dvalid;
  if (d.mtp && q35_) q35_->draft_vocab_restore(s, e.vocab, e.vocab_scanned);
  slot.turns = std::max(slot.turns, 2);
  slot.live_ok = false;
  logi("SSD cache: read %d tokens of a stored conversation into slot %d (%.0f MB in %.0f ms)", P, idx,
       (d.store.stats().read - r0) / 1e6, now_ms() - t0);
  return true;
}

}  // namespace ni
