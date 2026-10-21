#include "engine/prefix_cache.h"

#include "util/log.h"

#include <arm_acle.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>

namespace ni::pcache {

namespace {

constexpr char kMagic[8] = {'K', 'P', 'C', 'A', 'C', 'H', 'E', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeader = 4096;            // a file's header; the payload starts at this (aligned) offset
constexpr size_t kChunk = size_t(8) << 20;  // bytes per read / write call
constexpr size_t kAlign = size_t(16) << 10; // the transfer buffer's alignment (F_NOCACHE moves whole pages)

struct Header {
  char magic[8];
  uint32_t version, kind;
  uint64_t ns_hi, ns_lo, key_hi, key_lo;
  uint64_t payload;      // bytes after the header
  uint32_t payload_crc;  // CRC-32C of the payload
  uint32_t meta_bytes;   // kind-specific data right after this struct
  uint32_t header_crc;   // CRC-32C of the 4 KiB header with this field 0
  uint32_t pad;
};
constexpr size_t kMetaMax = kHeader - sizeof(Header);

double unix_now() {
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return double(t.tv_sec) + t.tv_nsec * 1e-9;
}

double mtime_of(const struct stat& st) { return double(st.st_mtimespec.tv_sec) + st.st_mtimespec.tv_nsec * 1e-9; }

bool is_ns_name(const std::string& s) {
  if (s.size() != 32) return false;
  for (char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

// A page-aligned transfer buffer
struct Buf {
  uint8_t* p = nullptr;
  explicit Buf(size_t n) {
    void* v = nullptr;
    if (posix_memalign(&v, kAlign, (n + kAlign - 1) / kAlign * kAlign) == 0) p = static_cast<uint8_t*>(v);
  }
  ~Buf() { free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

bool pread_all(int fd, uint8_t* p, size_t n, uint64_t off) {
  while (n > 0) {
    const ssize_t r = pread(fd, p, n, off_t(off));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    p += r, n -= size_t(r), off += uint64_t(r);
  }
  return true;
}

int pwrite_all(int fd, const uint8_t* p, size_t n, uint64_t off) {  // 0, or errno
  while (n > 0) {
    const ssize_t r = pwrite(fd, p, n, off_t(off));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return r < 0 ? errno : EIO;
    p += r, n -= size_t(r), off += uint64_t(r);
  }
  return 0;
}

const char* kind_dir(Kind k) { return k == kKv ? "/kv/" : k == kState ? "/st/" : "/e/"; }

// little-endian record encoding (Apple silicon is little-endian)
struct Writer {
  std::string b;
  template <class T> void put(const T& v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
  template <class T> void arr(const std::vector<T>& v) {
    put(uint32_t(v.size()));
    b.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  }
};
struct Reader {
  const uint8_t* p;
  size_t n, at = 0;
  bool ok = true;
  template <class T> T get() {
    T v{};
    if (at + sizeof v > n) {
      ok = false;
      return v;
    }
    memcpy(&v, p + at, sizeof v);
    at += sizeof v;
    return v;
  }
  template <class T> void arr(std::vector<T>& v, size_t max) {
    const uint32_t k = get<uint32_t>();
    if (!ok || k > max || at + size_t(k) * sizeof(T) > n) {
      ok = false;
      return;
    }
    v.resize(k);
    memcpy(v.data(), p + at, size_t(k) * sizeof(T));
    at += size_t(k) * sizeof(T);
  }
};

std::string encode(const Entry& e) {
  Writer w;
  w.put(e.block_rows);
  w.arr(e.tokens);
  w.put(uint32_t(e.fills.size()));
  for (const auto& [end, id] : e.fills) w.put(end), w.put(id);
  w.put(uint32_t(e.blocks.size()));
  for (const Key& k : e.blocks) w.put(k.hi), w.put(k.lo);
  w.put(uint32_t(e.snaps.size()));
  for (const Snap& s : e.snaps) w.put(s.pos), w.put(s.flags), w.put(s.dctx), w.put(s.dvalid), w.put(s.state.hi), w.put(s.state.lo);
  w.arr(e.vocab);
  w.put(e.vocab_scanned);
  w.put(e.tag);
  w.arr(std::vector<char>(e.attach.begin(), e.attach.end()));
  return w.b;
}

bool decode(const uint8_t* p, size_t n, Entry* e) {
  Reader r{p, n};
  e->block_rows = r.get<uint32_t>();
  r.arr(e->tokens, size_t(1) << 28);
  const uint32_t nf = r.get<uint32_t>();
  if (!r.ok || nf > (1u << 24)) return false;
  e->fills.resize(nf);
  for (auto& [end, id] : e->fills) end = r.get<int32_t>(), id = r.get<uint64_t>();
  const uint32_t nb = r.get<uint32_t>();
  if (!r.ok || nb > (1u << 24)) return false;
  e->blocks.resize(nb);
  for (Key& k : e->blocks) k.hi = r.get<uint64_t>(), k.lo = r.get<uint64_t>();
  const uint32_t ns = r.get<uint32_t>();
  if (!r.ok || ns > 64) return false;
  e->snaps.resize(ns);
  for (Snap& s : e->snaps) {
    s.pos = r.get<int32_t>(), s.flags = r.get<uint32_t>(), s.dctx = r.get<int32_t>(), s.dvalid = r.get<int32_t>();
    s.state.hi = r.get<uint64_t>(), s.state.lo = r.get<uint64_t>();
  }
  r.arr(e->vocab, size_t(1) << 22);
  e->vocab_scanned = r.get<int32_t>();
  e->tag = r.get<uint32_t>();
  std::vector<char> attach;
  r.arr(attach, size_t(1) << 24);
  e->attach.assign(attach.begin(), attach.end());
  if (!r.ok || r.at != n || e->block_rows == 0) return false;
  // consistent: the blocks cover the tokens, every restore point lies within them
  const size_t N = e->tokens.size();
  if (e->blocks.size() != (N + e->block_rows - 1) / e->block_rows) return false;
  for (const Snap& s : e->snaps)
    if (s.pos <= 0 || size_t(s.pos) > N) return false;
  return true;
}

// Time Machine skips the directory (as `tmutil addexclusion` marks one): a cache is rebuilt, not restored
void exclude_from_backups(const std::string& dir) {
  static const uint8_t plist[] = {0x62, 0x70, 0x6c, 0x69, 0x73, 0x74, 0x30, 0x30, 0x5f, 0x10, 0x11, 0x63, 0x6f, 0x6d, 0x2e,
                                  0x61, 0x70, 0x70, 0x6c, 0x65, 0x2e, 0x62, 0x61, 0x63, 0x6b, 0x75, 0x70, 0x64, 0x08, 0x00,
                                  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                                  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                  0x1c};  // bplist00 of "com.apple.backupd"
  static const char* name = "com.apple.metadata:com_apple_backup_excludeItem";
  if (getxattr(dir.c_str(), name, nullptr, 0, 0, 0) > 0) return;
  setxattr(dir.c_str(), name, plist, sizeof plist, 0, 0);
}

bool make_dirs(const std::string& path, mode_t mode) {
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  struct stat st;
  if (stat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
  chmod(path.c_str(), mode);
  return true;
}

uint64_t file_size(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 ? uint64_t(st.st_size) : 0;
}

}  // namespace

// ---- keys, hashes, CRC ----

std::string Key::hex() const {
  char b[33];
  snprintf(b, sizeof b, "%016llx%016llx", (unsigned long long)hi, (unsigned long long)lo);
  return b;
}

bool Key::parse(const std::string& s, Key* k) {
  if (!is_ns_name(s)) return false;
  k->hi = strtoull(s.substr(0, 16).c_str(), nullptr, 16);
  k->lo = strtoull(s.substr(16).c_str(), nullptr, 16);
  return true;
}

static inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
static inline uint64_t fmix(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return k;
}

Hasher& Hasher::u64(uint64_t v) {
  a_ = rotl(a_ ^ (v * 0x87c37b91114253d5ull), 31) * 0x4cf5ad432745937full + 0x52dce729;
  b_ = rotl(b_ + (v ^ 0x38495ab5), 27) * 0x9E3779B97F4A7C15ull ^ a_;
  ++n_;
  return *this;
}

Hasher& Hasher::i32s(const int32_t* p, size_t n) {
  u64(n);
  size_t i = 0;
  for (; i + 2 <= n; i += 2) u64(uint64_t(uint32_t(p[i])) | uint64_t(uint32_t(p[i + 1])) << 32);
  if (i < n) u64(uint64_t(uint32_t(p[i])) | (uint64_t(1) << 63));
  return *this;
}

Hasher& Hasher::bytes(const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  u64(n);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w;
    memcpy(&w, b + i, 8);
    u64(w);
  }
  if (i < n) {
    uint64_t w = 0;
    memcpy(&w, b + i, n - i);
    u64(w ^ (uint64_t(n - i) << 56));
  }
  return *this;
}

Key Hasher::done() const {
  Key k;
  k.hi = fmix(a_ ^ rotl(b_, 17) ^ (n_ * 0x9E3779B97F4A7C15ull));
  k.lo = fmix(b_ + rotl(a_, 41) + n_);
  if (!k) k.lo = 1;  // never the empty key
  return k;
}

uint32_t crc32c(uint32_t crc, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  crc = ~crc;
  while (n && (reinterpret_cast<uintptr_t>(p) & 7)) crc = __crc32cb(crc, *p++), --n;
  for (; n >= 32; p += 32, n -= 32) {
    uint64_t w[4];
    memcpy(w, p, 32);
    crc = __crc32cd(crc, w[0]);
    crc = __crc32cd(crc, w[1]);
    crc = __crc32cd(crc, w[2]);
    crc = __crc32cd(crc, w[3]);
  }
  for (; n >= 8; p += 8, n -= 8) {
    uint64_t w;
    memcpy(&w, p, 8);
    crc = __crc32cd(crc, w);
  }
  while (n--) crc = __crc32cb(crc, *p++);
  return ~crc;
}

// ---- sizes and space ----

bool parse_size(const std::string& in, uint64_t* out) {
  std::string s;
  for (char c : in)
    if (!isspace(static_cast<unsigned char>(c))) s += char(tolower(static_cast<unsigned char>(c)));
  if (s == "auto") return *out = kAuto, true;
  if (s == "0" || s == "off" || s == "none" || s == "no") return *out = 0, true;
  if (s.empty() || !(isdigit(static_cast<unsigned char>(s[0])) || s[0] == '.')) return false;
  char* end = nullptr;
  const double v = strtod(s.c_str(), &end);
  std::string unit = end ? end : "";
  if (!std::isfinite(v) || v < 0) return false;
  if (unit.size() >= 2 && unit.compare(unit.size() - 2, 2, "ib") == 0) unit.resize(unit.size() - 2);
  else if (!unit.empty() && unit.back() == 'b') unit.pop_back();
  double mult = 1;
  if (unit == "k") mult = 1024.0;
  else if (unit == "m") mult = 1024.0 * 1024;
  else if (unit == "g") mult = 1024.0 * 1024 * 1024;
  else if (unit == "t") mult = 1024.0 * 1024 * 1024 * 1024;
  else if (!unit.empty()) return false;
  const double b = v * mult;
  if (b >= 9.0e18) return false;
  *out = uint64_t(b);
  return true;
}

std::string gib(uint64_t bytes) {
  char b[32];
  if (bytes >= kGiB / 10) snprintf(b, sizeof b, "%.1f GiB", double(bytes) / double(kGiB));
  else snprintf(b, sizeof b, "%.0f MiB", double(bytes) / double(1 << 20));
  return b;
}

uint64_t space_limit(uint64_t quota, uint64_t avail, uint64_t held, uint64_t volume) {
  const uint64_t floor = volume > 0 ? std::min(kFloor, volume / 20) : kFloor;
  const uint64_t room = avail + held > floor ? avail + held - floor : 0;
  return std::min(quota, room / 4);
}

uint64_t dir_bytes(const std::string& dir) {
  uint64_t n = 0;
  std::error_code ec;
  for (auto it = std::filesystem::recursive_directory_iterator(dir, ec); !ec && it != std::filesystem::end(it);
       it.increment(ec)) {
    std::error_code e2;
    if (it->is_regular_file(e2)) n += it->file_size(e2);
  }
  return n;
}

// ---- the store ----

Store::~Store() {
  if (lock_fd_ >= 0) close(lock_fd_);
}

std::string Store::path_of(Kind kind, const Key& key) const { return dir_ + kind_dir(kind) + key.hex(); }

bool Store::open(const Options& o, std::string* why) {
  o_ = o;
  if (!is_ns_name(o.ns)) return *why = "bad namespace name", false;
  if (!make_dirs(o.root, 0700)) return *why = "cannot create " + o.root, false;
  exclude_from_backups(o.root);
  dir_ = o.root + "/" + o.ns;
  // the namespace's lock: one server at a time (another waits, as a restarting server waits for the old one's last
  // writes). A namespace removed between our open() and flock() (by a server that found it unused) is created again.
  const double t0 = now_ms();
  bool said = false;
  for (;;) {
    if (!make_dirs(dir_, 0700)) return *why = "cannot create " + dir_, false;
    const std::string lp = dir_ + "/lock";
    const int fd = ::open(lp.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return *why = "cannot open " + lp + ": " + strerror(errno), false;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
      close(fd);
      if (errno != EWOULDBLOCK) return *why = "cannot lock " + lp + ": " + strerror(errno), false;
      if (now_ms() - t0 > o.lock_wait_s * 1000 || (o.cancel && o.cancel->load()))
        return *why = "another Kapitsa server is using " + dir_, false;
      if (!said) logi("SSD prefix cache: another server holds %s; waiting up to %.0f s for it", dir_.c_str(), o.lock_wait_s);
      said = true;
      usleep(100 * 1000);
      continue;
    }
    struct stat a, b;
    if (fstat(fd, &a) == 0 && stat(lp.c_str(), &b) == 0 && a.st_ino == b.st_ino && a.st_dev == b.st_dev) {
      lock_fd_ = fd;
      break;
    }
    close(fd);  // removed meanwhile: again
  }
  touch();
  if (!o.about.empty()) {
    const std::string ap = dir_ + "/about.txt";
    if (file_size(ap) != o.about.size())
      if (FILE* f = fopen(ap.c_str(), "w")) fputs(o.about.c_str(), f), fclose(f);
  }
  for (Kind k : {kKv, kState, kEntry})
    if (!make_dirs(dir_ + kind_dir(k), 0700)) return *why = "cannot create " + dir_ + kind_dir(k), false;

  // the other namespaces: those nobody used for stale_days go, the rest count against the limit
  {
    DIR* d = opendir(o.root.c_str());
    for (dirent* de; d && (de = readdir(d));) {
      const std::string name = de->d_name;
      if (!is_ns_name(name) || name == o.ns) continue;
      const std::string p = o.root + "/" + name;
      struct stat st;
      if (stat(p.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;
      const int fd = ::open((p + "/lock").c_str(), O_RDWR | O_CLOEXEC);
      if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) != 0) {  // in use by another server: its own to count
        close(fd);
        continue;
      }
      struct stat lt;
      const double used = fd >= 0 && fstat(fd, &lt) == 0 ? mtime_of(lt) : mtime_of(st);
      if (unix_now() - used > o.stale_days * 86400) {
        const uint64_t bytes = dir_bytes(p);
        std::error_code ec;
        std::filesystem::remove_all(p, ec);
        logi("SSD prefix cache: removed %s (%s), unused for %.0f days", p.c_str(), gib(bytes).c_str(),
             (unix_now() - used) / 86400);
      } else {
        others_.push_back({p, used, dir_bytes(p)});
        others_bytes_ += others_.back().bytes;
      }
      if (fd >= 0) close(fd);
    }
    if (d) closedir(d);
    std::sort(others_.begin(), others_.end(), [](const Other& a, const Other& b) { return a.used < b.used; });
  }

  // this namespace's files: leftover temporary files go; pieces by name and size; entries read whole and checked
  for (Kind k : {kKv, kState, kEntry}) {
    const std::string sub = dir_ + kind_dir(k);
    DIR* d = opendir(sub.c_str());
    for (dirent* de; d && (de = readdir(d));) {
      const std::string name = de->d_name;
      if (name == "." || name == "..") continue;
      const std::string p = sub + name;
      if (name.find(".tmp.") != std::string::npos) {
        unlink(p.c_str());
        continue;
      }
      Key key;
      if (!Key::parse(name, &key)) continue;  // not ours
      if (k == kEntry) {
        Entry e;
        struct stat st;
        if (stat(p.c_str(), &st) != 0 || !load_entry(p, &e)) {
          unlink(p.c_str());
          continue;
        }
        e.key = key;
        e.bytes = uint64_t(st.st_size);
        e.used = mtime_of(st);
        entries_.emplace(key, std::move(e));
      } else {
        const uint64_t n = file_size(p);
        if (n < kHeader) {
          unlink(p.c_str());
          continue;
        }
        (k == kKv ? kv_ : st_)[key] = Piece{n, 0};
      }
    }
    if (d) closedir(d);
  }
  // entries whose pieces are all there; pieces no entry needs go
  for (auto it = entries_.begin(); it != entries_.end();) {
    bool whole = true;
    for (const Key& b : it->second.blocks) whole = whole && kv_.count(b);
    for (const Snap& s : it->second.snaps) whole = whole && st_.count(s.state);
    if (!whole) {
      unlink(path_of(kEntry, it->first).c_str());
      it = entries_.erase(it);
      continue;
    }
    for (const Key& b : it->second.blocks) ++kv_[b].refs;
    for (const Snap& s : it->second.snaps) ++st_[s.state].refs;
    ++it;
  }
  for (auto* m : {&kv_, &st_})
    for (auto it = m->begin(); it != m->end();) {
      if (it->second.refs == 0) {
        unlink(path_of(m == &kv_ ? kKv : kState, it->first).c_str());
        it = m->erase(it);
      } else {
        held_ += it->second.bytes;
        ++it;
      }
    }
  for (const auto& [k, e] : entries_) held_ += e.bytes;
  measure();
  return true;
}

bool Store::load_entry(const std::string& path, Entry* e) const {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st;
  bool ok = fstat(fd, &st) == 0 && size_t(st.st_size) >= kHeader && size_t(st.st_size) < (size_t(1) << 31);
  std::vector<uint8_t> b(ok ? size_t(st.st_size) : 0);
  ok = ok && pread_all(fd, b.data(), b.size(), 0);
  close(fd);
  if (!ok) return false;
  Header h;
  memcpy(&h, b.data(), sizeof h);
  const uint32_t hc = h.header_crc;
  memset(b.data() + offsetof(Header, header_crc), 0, 4);
  if (memcmp(h.magic, kMagic, 8) != 0 || h.version != kVersion || h.kind != kEntry || crc32c(0, b.data(), kHeader) != hc ||
      h.payload != b.size() - kHeader || crc32c(0, b.data() + kHeader, b.size() - kHeader) != h.payload_crc)
    return false;
  Key ns;
  Key::parse(o_.ns, &ns);
  if (h.ns_hi != ns.hi || h.ns_lo != ns.lo) return false;
  return decode(b.data() + kHeader, b.size() - kHeader, e);
}

void Store::touch() {
  if (lock_fd_ >= 0) futimes(lock_fd_, nullptr);
}

// ---- space ----

void Store::set_test_space(uint64_t avail, uint64_t volume) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    o_.test_avail = avail, o_.test_volume = volume;
  }
  measure();
}

void Store::measure() {
  struct statfs sf;
  uint64_t avail = 0, volume = 0;
  bool local = true;
  {
    std::lock_guard<std::mutex> lk(mu_);
    avail = o_.test_avail, volume = o_.test_volume;
  }
  if (!volume) {
    if (statfs(dir_.c_str(), &sf) != 0) return;  // not measured: the limit stays
    avail = uint64_t(sf.f_bavail) * sf.f_bsize, volume = uint64_t(sf.f_blocks) * sf.f_bsize;
    local = (sf.f_flags & MNT_LOCAL) != 0;
  }
  std::lock_guard<std::mutex> lk(mu_);
  avail_ = avail, volume_ = volume, local_ = local;
  const uint64_t quota = o_.quota == kAuto ? kAutoQuota : o_.quota;
  // off a local volume the free space is not followed: a size given is the limit, auto keeps nothing
  limit_ = local ? space_limit(quota, avail, held_ + others_bytes_, volume) : o_.quota == kAuto ? 0 : quota;
  fit();
}

uint64_t Store::limit() const {
  std::lock_guard<std::mutex> lk(mu_);
  return limit_;
}

uint64_t Store::held() const {
  std::lock_guard<std::mutex> lk(mu_);
  return held_;
}

Store::Stats Store::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  Stats s;
  s.entries = entries_.size();
  s.pieces = kv_.size() + st_.size();
  s.held = held_, s.limit = limit_, s.others = others_bytes_;
  s.written = written_, s.read = read_;
  return s;
}

void Store::remove_others(uint64_t limit) {
  while (!others_.empty() && held_ + others_bytes_ > limit) {
    const Other o = others_.front();
    others_.erase(others_.begin());
    others_bytes_ -= o.bytes;
    // nobody may hold it now: one that started using it since is its server's to count
    const int fd = ::open((o.path + "/lock").c_str(), O_RDWR | O_CLOEXEC);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) != 0) {
      close(fd);
      continue;
    }
    std::error_code ec;
    std::filesystem::remove_all(o.path, ec);
    if (fd >= 0) close(fd);
    logi("SSD prefix cache: removed %s (%s), the oldest cache of another model, to keep within %s", o.path.c_str(),
         gib(o.bytes).c_str(), gib(limit).c_str());
  }
}

bool Store::make_room(uint64_t bytes) {
  auto fits = [&] { return held_ + others_bytes_ + bytes <= limit_; };
  if (fits()) return true;
  if (bytes > limit_) return false;
  remove_others(limit_ > bytes ? limit_ - bytes : 0);
  // pieces nothing needs (a write given up), then whole entries, least recently used first
  for (auto* m : {&kv_, &st_})
    for (auto it = m->begin(); it != m->end() && !fits();) {
      if (it->second.refs > 0) {
        ++it;
        continue;
      }
      unlink(path_of(m == &kv_ ? kKv : kState, it->first).c_str());
      held_ -= std::min(held_, it->second.bytes);
      it = m->erase(it);
    }
  while (!fits()) {
    auto lru = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
      if (it->second.pins == 0 && (lru == entries_.end() || it->second.used < lru->second.used)) lru = it;
    if (lru == entries_.end()) return false;
    drop_entry(lru);
  }
  return true;
}

void Store::fit() { make_room(0); }

void Store::drop_entry(std::map<Key, Entry>::iterator it) {
  for (const Key& b : it->second.blocks) unref(kKv, b);
  for (const Snap& s : it->second.snaps) unref(kState, s.state);
  unlink(path_of(kEntry, it->first).c_str());
  held_ -= std::min(held_, it->second.bytes);
  entries_.erase(it);
}

// one fewer user of the piece; an entry's last one deletes it (a job's leaves it for a later write to reuse, until
// room is needed)
void Store::unref(Kind kind, const Key& key) {
  auto& m = kind == kKv ? kv_ : st_;
  auto it = m.find(key);
  if (it == m.end()) return;
  if (--it->second.refs <= 0) {
    it->second.refs = 0;
    erase_piece(kind, key);
  }
}

void Store::erase_piece(Kind kind, const Key& key) {
  auto& m = kind == kKv ? kv_ : st_;
  auto it = m.find(key);
  if (it == m.end()) return;
  unlink(path_of(kind, key).c_str());
  held_ -= std::min(held_, it->second.bytes);
  m.erase(it);
}

void Store::drop_damaged(Kind kind, const Key& key) {
  std::lock_guard<std::mutex> lk(mu_);
  if (kind == kEntry) {
    auto it = entries_.find(key);
    if (it != entries_.end() && it->second.pins == 0) drop_entry(it);
    return;
  }
  for (auto it = entries_.begin(); it != entries_.end();) {
    bool uses = false;
    for (const Key& b : it->second.blocks) uses = uses || (kind == kKv && b == key);
    for (const Snap& s : it->second.snaps) uses = uses || (kind == kState && s.state == key);
    if (uses && it->second.pins == 0) {
      auto next = std::next(it);
      drop_entry(it);
      it = next;
    } else {
      ++it;
    }
  }
  // (pinned users keep the piece's record until unpinned; its file goes now)
  unlink(path_of(kind, key).c_str());
  auto& m = kind == kKv ? kv_ : st_;
  if (auto it = m.find(key); it != m.end() && it->second.refs == 0) {
    held_ -= std::min(held_, it->second.bytes);
    m.erase(it);
  }
}

// ---- reads ----

Hit Store::lookup(const int* keys, int n, int below, uint32_t tag, const Reusable& reusable) const {
  std::lock_guard<std::mutex> lk(mu_);
  Hit best;
  double best_used = -1;
  for (const auto& [k, e] : entries_) {
    if (e.tag != tag) continue;
    const size_t m = std::min(e.tokens.size(), size_t(std::max(n, 0)));
    size_t l = 0;
    while (l < m && e.tokens[l] == keys[l]) ++l;
    if (reusable && l > 0) l = std::min(l, reusable(e, l));
    for (int i = 0; i < int(e.snaps.size()); ++i) {
      const int pos = e.snaps[size_t(i)].pos;
      if (size_t(pos) > l || pos >= below) continue;
      if (pos > best.pos || (pos == best.pos && e.used > best_used)) best = Hit{k, i, pos}, best_used = e.used;
    }
  }
  return best;
}

bool Store::pin(const Key& entry, Entry* out) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = entries_.find(entry);
  if (it == entries_.end()) return false;
  ++it->second.pins;
  *out = it->second;
  return true;
}

void Store::unpin(const Key& entry, bool used) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = entries_.find(entry);
  if (it == entries_.end()) return;
  if (it->second.pins > 0) --it->second.pins;
  if (used) {
    it->second.used = unix_now();
    utimes(path_of(kEntry, entry).c_str(), nullptr);  // the order of use survives a restart
  }
}

bool Store::has(Kind kind, const Key& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  return kind == kEntry ? entries_.count(key) > 0 : (kind == kKv ? kv_ : st_).count(key) > 0;
}

bool Store::read(Kind kind, const Key& key, std::string* meta, const Take& take) {
  const std::string p = path_of(kind, key);
  const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    logi("SSD prefix cache: %s is missing; the entries that need it are dropped", p.c_str());
    drop_damaged(kind, key);
    return false;
  }
  fcntl(fd, F_NOCACHE, 1);
  Buf buf(std::max(kChunk, kHeader));
  bool ok = buf.p && pread_all(fd, buf.p, kHeader, 0);
  Header h{};
  if (ok) {
    memcpy(&h, buf.p, sizeof h);
    const uint32_t hc = h.header_crc;
    memset(buf.p + offsetof(Header, header_crc), 0, 4);
    Key ns;
    Key::parse(o_.ns, &ns);
    ok = memcmp(h.magic, kMagic, 8) == 0 && h.version == kVersion && h.kind == kind && h.key_hi == key.hi &&
         h.key_lo == key.lo && h.ns_hi == ns.hi && h.ns_lo == ns.lo && h.meta_bytes <= kMetaMax &&
         crc32c(0, buf.p, kHeader) == hc;
    if (ok && meta) meta->assign(reinterpret_cast<const char*>(buf.p) + sizeof(Header), h.meta_bytes);
  }
  uint32_t crc = 0;
  bool refused = false;
  for (uint64_t off = 0; ok && off < h.payload; ) {
    const size_t n = size_t(std::min<uint64_t>(kChunk, h.payload - off));
    ok = pread_all(fd, buf.p, n, kHeader + off);
    if (!ok) break;
    crc = crc32c(crc, buf.p, n);
    if (!take(off, buf.p, n)) {
      refused = true;
      break;
    }
    off += n;
  }
  close(fd);
  if (refused) return false;
  if (!ok || crc != h.payload_crc) {
    logi("SSD prefix cache: %s is damaged (%s); the entries that need it are dropped", p.c_str(),
         ok ? "its CRC does not match" : "its header or length is wrong");
    drop_damaged(kind, key);
    return false;
  }
  std::lock_guard<std::mutex> lk(mu_);
  read_ += kHeader + h.payload;
  return true;
}

// ---- writes ----

Store::Job::~Job() {
  std::lock_guard<std::mutex> lk(st_->mu_);
  for (const auto& [kind, key] : held_) {
    auto& m = kind == kKv ? st_->kv_ : st_->st_;
    if (auto it = m.find(key); it != m.end() && it->second.refs > 0) --it->second.refs;  // kept: a later write reuses it
  }
}

std::unique_ptr<Store::Job> Store::begin(const std::vector<std::pair<Kind, Key>>& pieces) {
  std::unique_ptr<Job> j(new Job(this));
  std::lock_guard<std::mutex> lk(mu_);
  for (const auto& [kind, key] : pieces) {
    if (kind == kEntry) continue;
    auto& m = kind == kKv ? kv_ : st_;
    if (auto it = m.find(key); it != m.end()) {
      ++it->second.refs;
      j->held_.push_back({kind, key});
    }
  }
  return j;
}

bool Store::write_file(Kind kind, const Key& key, const std::string& meta, uint64_t bytes, const Give& give,
                       uint64_t* file_bytes) {
  const std::string p = path_of(kind, key);
  std::string tmp;
  {
    std::lock_guard<std::mutex> lk(mu_);
    tmp = p + ".tmp." + std::to_string(getpid()) + "." + std::to_string(++tmp_seq_);
  }
  if (meta.size() > kMetaMax) return false;
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  fcntl(fd, F_NOCACHE, 1);
  Buf buf(std::max(kChunk, kHeader));
  int err = buf.p ? 0 : ENOMEM;
  uint32_t crc = 0;
  bool given = true;
  for (uint64_t off = 0; !err && off < bytes;) {
    const size_t n = size_t(std::min<uint64_t>(kChunk, bytes - off));
    if (!(given = give(off, buf.p, n))) break;
    crc = crc32c(crc, buf.p, n);
    err = pwrite_all(fd, buf.p, n, kHeader + off);
    off += n;
  }
  if (!err && given) {
    memset(buf.p, 0, kHeader);
    Header h{};
    memcpy(h.magic, kMagic, 8);
    h.version = kVersion;
    h.kind = kind;
    Key ns;
    Key::parse(o_.ns, &ns);
    h.ns_hi = ns.hi, h.ns_lo = ns.lo, h.key_hi = key.hi, h.key_lo = key.lo;
    h.payload = bytes;
    h.payload_crc = crc;
    h.meta_bytes = uint32_t(meta.size());
    memcpy(buf.p, &h, sizeof h);
    memcpy(buf.p + sizeof h, meta.data(), meta.size());
    const uint32_t hc = crc32c(0, buf.p, kHeader);
    memcpy(buf.p + offsetof(Header, header_crc), &hc, 4);
    err = pwrite_all(fd, buf.p, kHeader, 0);
  }
  if (close(fd) != 0 && !err) err = errno;
  if (!err && given && rename(tmp.c_str(), p.c_str()) != 0) err = errno;
  if (err || !given) {
    unlink(tmp.c_str());
    if (err == ENOSPC || err == EDQUOT) {
      if (!full_logged_)
        logi("SSD prefix cache: the disk is full (%s); the cache shrinks to fit what it has room for",
             dir_.c_str());
      full_logged_ = true;
      measure();
    } else if (err) {
      logi("SSD prefix cache: writing %s failed: %s", p.c_str(), strerror(err));
    }
    return false;
  }
  *file_bytes = kHeader + bytes;
  return true;
}

bool Store::write(Job& job, Kind kind, const Key& key, const std::string& meta, uint64_t bytes, const Give& give) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto& m = kind == kKv ? kv_ : st_;
    if (auto it = m.find(key); it != m.end()) {
      ++it->second.refs;
      job.held_.push_back({kind, key});
      return true;
    }
    if (!make_room(kHeader + bytes)) return false;
  }
  uint64_t fb = 0;
  if (!write_file(kind, key, meta, bytes, give, &fb)) return false;
  std::lock_guard<std::mutex> lk(mu_);
  auto& m = kind == kKv ? kv_ : st_;
  auto [it, fresh] = m.emplace(key, Piece{fb, 0});
  if (fresh) held_ += fb;
  ++it->second.refs;
  job.held_.push_back({kind, key});
  written_ += fb;
  return true;
}

bool Store::put(Job& job, Entry e) {
  (void)job;  // (its pieces stay held until it ends: the entry takes its own references here)
  const std::string payload = encode(e);
  if (!e.key) e.key = Hasher().str(payload).done();
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (const Key& b : e.blocks)
      if (!kv_.count(b)) return false;
    for (const Snap& s : e.snaps)
      if (!st_.count(s.state)) return false;
    if (auto it = entries_.find(e.key); it != entries_.end()) {  // the same conversation, saved before
      it->second.used = unix_now();
      utimes(path_of(kEntry, e.key).c_str(), nullptr);
      return true;
    }
    if (!make_room(kHeader + payload.size())) return false;
  }
  uint64_t fb = 0;
  const bool ok = write_file(kEntry, e.key, "", payload.size(),
                             [&](uint64_t off, uint8_t* dst, size_t n) {
                               memcpy(dst, payload.data() + off, n);
                               return true;
                             },
                             &fb);
  if (!ok) return false;
  std::lock_guard<std::mutex> lk(mu_);
  for (const Key& b : e.blocks) ++kv_[b].refs;
  for (const Snap& s : e.snaps) ++st_[s.state].refs;
  e.bytes = fb;
  e.used = unix_now();
  e.pins = 0;
  held_ += fb;
  written_ += fb;
  // entries whose every restore point this one has are no longer needed
  std::set<Key> mine;
  for (const Snap& s : e.snaps) mine.insert(s.state);
  for (auto it = entries_.begin(); it != entries_.end();) {
    bool covered = it->second.pins == 0 && !it->second.snaps.empty();
    for (const Snap& s : it->second.snaps) covered = covered && mine.count(s.state);
    if (covered) {
      auto next = std::next(it);
      drop_entry(it);
      it = next;
    } else {
      ++it;
    }
  }
  const Key k = e.key;
  entries_.emplace(k, std::move(e));
  return true;
}

void Store::clear() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto it = entries_.begin(); it != entries_.end();) {
    auto next = std::next(it);
    drop_entry(it);
    it = next;
  }
  for (auto* m : {&kv_, &st_})
    for (auto it = m->begin(); it != m->end();) {
      unlink(path_of(m == &kv_ ? kKv : kState, it->first).c_str());
      held_ -= std::min(held_, it->second.bytes);
      it = m->erase(it);
    }
}

}  // namespace ni::pcache
