// SSD prefix cache: conversations' KV caches and recurrent states as files, so that a request whose prompt starts with
// a stored conversation reads it back instead of prefilling it again, also after the server restarts. This is the
// store: files, their index, the space they may take. The engine decides what to save and when, and restores slots
// from it (src/engine/engine_disk.cpp). CPU only (tools/prefix_cache_test checks it without a model).
//
// Under the root (KAPITSA_HOME/prefix-cache), one directory per namespace, named by 32 hex digits that digest the
// model's files, the drafter and the state layouts, so no model reads another's cache:
//   <ns>/lock       held (flock) by the server using the namespace; its mtime says when the namespace was last used
//   <ns>/about.txt  what the namespace holds, for people
//   <ns>/kv/<key>   the KV rows of one block of positions (every attention cache), keyed by how they were computed:
//                   the tokens up to the block's end and the requests that computed them
//   <ns>/st/<key>   a state at a position: the recurrent state and the drafter's context ring
//   <ns>/e/<key>    an entry: one conversation's tokens, its KV blocks and the states it can resume from
// A conversation's next turns share the blocks of the turns before (a write adds the new blocks and states only).
// Every file is a 4 KiB header (magic, kind, namespace, key, sizes, CRC-32C of the payload and of the header) and the
// payload, written to a temporary file that is then renamed, read and written with F_NOCACHE.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace ni::pcache {

// 128-bit content key (file names: 32 hex digits)
struct Key {
  uint64_t hi = 0, lo = 0;
  bool operator==(const Key& o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(const Key& o) const { return !(*this == o); }
  bool operator<(const Key& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
  explicit operator bool() const { return hi || lo; }
  std::string hex() const;
  static bool parse(const std::string& s, Key* k);
};

// Incremental 128-bit hash (two independent 64-bit lanes); for content keys, not security
class Hasher {
 public:
  Hasher& u64(uint64_t v);
  Hasher& i32s(const int32_t* p, size_t n);
  Hasher& bytes(const void* p, size_t n);
  Hasher& str(const std::string& s) { return u64(s.size()).bytes(s.data(), s.size()); }
  Hasher& key(const Key& k) { return u64(k.hi).u64(k.lo); }
  Key done() const;

 private:
  uint64_t a_ = 0x9E3779B97F4A7C15ull, b_ = 0xC2B2AE3D27D4EB4Full, n_ = 0;
};

// CRC-32C (Castagnoli), hardware instructions on Apple silicon; crc32c(0, p, n) for a whole buffer
uint32_t crc32c(uint32_t crc, const void* p, size_t n);

constexpr uint64_t kGiB = uint64_t(1) << 30;
constexpr uint64_t kAuto = ~uint64_t(0);        // the quota "auto": kAutoQuota, within the volume's free space
constexpr uint64_t kAutoQuota = 32 * kGiB;
constexpr uint64_t kFloor = 10 * kGiB;          // free space the cache never counts on
// --disk-cache's value: "auto", "0" / "off" / "none" (0), or a size: bytes, or K, M, G, T of them (powers of 1024;
// "16G", "512M", "1.5T", "4GiB"). False when it is none of those.
bool parse_size(const std::string& s, uint64_t* bytes);
std::string gib(uint64_t bytes);  // "6.5 GiB" ("80 MiB" below 0.1 GiB)
// The cache's limit: min(quota, max(0, avail + held - floor) / 4), floor = min(10 GiB, volume / 20). `held` is what the
// cache's own files take, so its writes do not shrink its limit; other files that fill the volume do, by a quarter.
uint64_t space_limit(uint64_t quota, uint64_t avail, uint64_t held, uint64_t volume);

enum Kind : uint32_t { kKv = 1, kState = 2, kEntry = 3 };

// A restore point of an entry: the state at `pos` (after tokens [0, pos))
struct Snap {
  int32_t pos = 0;
  uint32_t flags = 0;
  int32_t dctx = 0, dvalid = 0;  // the drafter ring's committed positions and its lowest valid one (DFlash2)
  Key state;
  static constexpr uint32_t kLive = 1, kQuery = 2;  // the slot's live state; a checkpoint at a user query
};

struct Entry {
  Key key;
  uint32_t block_rows = 0;
  std::vector<int32_t> tokens;                       // [0, N)
  std::vector<std::pair<int32_t, uint64_t>> fills;   // rows [previous end, end) were computed by request `id`
  std::vector<Key> blocks;                           // block b: KV rows [b * block_rows, min((b + 1) * block_rows, N))
  std::vector<Snap> snaps;
  std::vector<uint32_t> vocab;                       // the MTP head's gathered draft rows (Qwen35), in order
  int32_t vocab_scanned = 0;
  uint32_t tag = 0;                                  // what else its states depend on (Dory: the loop count)
  std::string attach;                                // the engine's own record (the images among the tokens)
  // index only (not in the file)
  double used = 0;      // last write or restore (Unix time), for least-recently-used eviction
  uint64_t bytes = 0;   // the record's file
  int pins = 0;
};

// The deepest restore point that matches a prompt
struct Hit {
  Key entry;
  int snap = -1;
  int pos = 0;
  explicit operator bool() const { return snap >= 0; }
};

class Store {
 public:
  struct Options {
    std::string root;          // the namespaces' parent directory (created; kept out of Time Machine)
    std::string ns;            // 32 hex digits
    std::string about;         // <ns>/about.txt
    uint64_t quota = kAuto;
    double stale_days = 14;    // namespaces under root nobody used for this long are removed at open
    double lock_wait_s = 30;   // another process holding the namespace: wait this long for it, then give up
    const std::atomic<bool>* cancel = nullptr;  // set: stop waiting for the lock
    uint64_t test_avail = 0, test_volume = 0;  // tests: the volume's free space and size as given (0: measured)
  };
  Store() = default;
  ~Store();
  Store(const Store&) = delete;
  Store& operator=(const Store&) = delete;

  // Takes the namespace (creating it), removes namespaces nobody used for stale_days, leftover temporary files, and
  // entries whose files are missing or damaged, then fits the limit. False with *why when the cache cannot be used.
  bool open(const Options& o, std::string* why);
  bool is_open() const { return lock_fd_ >= 0; }
  const std::string& dir() const { return dir_; }

  // ---- reads ----
  // The deepest restore point at a position < below of an entry with this tag whose tokens match keys[0, pos); the most
  // recently used entry on a tie. reusable(e, l), if given, cuts an entry's matching prefix of l rows further (the
  // engine's check of the images in it).
  using Reusable = std::function<size_t(const Entry& e, size_t l)>;
  Hit lookup(const int* keys, int n, int below, uint32_t tag = 0, const Reusable& reusable = nullptr) const;
  // A copy of the entry's record, pinned: its files stay until unpin (used: it was restored, for the LRU order)
  bool pin(const Key& entry, Entry* out);
  void unpin(const Key& entry, bool used);
  // Reads a piece: its header data (meta) and its payload in chunks to take(off, src, n). False when the piece is
  // missing, damaged (its CRC; it is then deleted with the entries that need it) or take refused a chunk.
  using Take = std::function<bool(uint64_t off, const uint8_t* src, size_t n)>;
  bool read(Kind kind, const Key& key, std::string* meta, const Take& take);

  // ---- writes ----
  bool has(Kind kind, const Key& key) const;
  // One entry's write: it holds the pieces it names (and those written through it) against eviction until it ends
  class Job {
   public:
    ~Job();
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;

   private:
    friend class Store;
    explicit Job(Store* s) : st_(s) {}
    Store* st_;
    std::vector<std::pair<Kind, Key>> held_;
  };
  std::unique_ptr<Job> begin(const std::vector<std::pair<Kind, Key>>& pieces);
  // Writes a piece of `bytes` from give(off, dst, n) (false: abort) unless it exists. False when it could not be
  // written: no room within the limit, an I/O error, or give refused.
  using Give = std::function<bool(uint64_t off, uint8_t* dst, size_t n)>;
  bool write(Job& job, Kind kind, const Key& key, const std::string& meta, uint64_t bytes, const Give& give);
  // Writes the entry's record (its pieces must exist) and indexes it. Entries whose restore points it has all
  // (by key) are removed. e.key is set from its content when empty.
  bool put(Job& job, Entry e);

  // ---- space ----
  // Measures the volume (free space, size): the limit follows it, and entries go (least recently used first) until the
  // cache fits. The engine calls it at open, before writes and every minute.
  void measure();
  uint64_t limit() const;
  uint64_t held() const;  // bytes of the namespace's files
  // the namespace's last use (its lock file's mtime): at open and every hour while it is open
  void touch();
  struct Stats {
    size_t entries = 0, pieces = 0;
    uint64_t held = 0, limit = 0, others = 0;  // others: namespaces nobody holds, counted against the limit
    uint64_t written = 0, read = 0;            // bytes since open
  };
  Stats stats() const;
  // removes every entry and piece (tests)
  void clear();
  // tests: the volume's free space and size from now on, as if measured (then measure())
  void set_test_space(uint64_t avail, uint64_t volume);

 private:
  struct Piece {
    uint64_t bytes = 0;
    int refs = 0;  // entries and jobs that need it
  };
  std::string path_of(Kind kind, const Key& key) const;
  bool write_file(Kind kind, const Key& key, const std::string& meta, uint64_t bytes, const Give& give,
                  uint64_t* file_bytes);
  // makes room for `bytes` more: orphan pieces first, then the least recently used entries (not pinned) and the
  // pieces only they needed. Locked.
  bool make_room(uint64_t bytes);
  void fit();  // within the limit (locked)
  void drop_entry(std::map<Key, Entry>::iterator it);  // locked
  void unref(Kind kind, const Key& key);             // locked
  void erase_piece(Kind kind, const Key& key);       // locked
  void drop_damaged(Kind kind, const Key& key);
  bool load_entry(const std::string& path, Entry* e) const;
  void remove_others(uint64_t limit);  // other namespaces nobody holds, oldest first, while root holds more than limit

  Options o_;
  std::string dir_;
  int lock_fd_ = -1;
  mutable std::mutex mu_;
  std::map<Key, Piece> kv_, st_;
  std::map<Key, Entry> entries_;
  uint64_t held_ = 0, limit_ = 0, avail_ = 0, volume_ = 0;
  bool local_ = true;
  struct Other {
    std::string path;
    double used = 0;
    uint64_t bytes = 0;
  };
  std::vector<Other> others_;  // oldest first
  uint64_t others_bytes_ = 0;
  uint64_t written_ = 0, read_ = 0;
  uint64_t tmp_seq_ = 0;
  bool full_logged_ = false;
};

// Bytes of a directory's files (recursive)
uint64_t dir_bytes(const std::string& dir);

}  // namespace ni::pcache
