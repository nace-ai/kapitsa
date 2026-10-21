// The SSD prefix cache's store (src/engine/prefix_cache.h), CPU only, no model: CRC-32C and keys; --disk-cache's sizes
// and the space rule; pieces and entries written and read back byte for byte, and taken back by the next open (a
// restart); a damaged piece (its CRC) and torn writes (a leftover temporary file, truncated files, a damaged record);
// the quota (least recently used entries go first, pieces shared by entries stay while one needs them, an entry whose
// restore points a newer one has goes, a pinned entry stays, a piece larger than the limit is refused); the volume
// filling up; namespaces (one unused for 14 days removed at open, another counted against the limit and removed before
// this one's entries, one another process holds left alone, other folders untouched); the namespace's lock.
// Works in a temporary folder it removes.
//   prefix_cache_test        exit status 1 if any check fails
#include "engine/prefix_cache.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace ni;
using namespace ni::pcache;
namespace fs = std::filesystem;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("\n     " + detail).c_str());
  ++(ok ? passes : fails);
}

constexpr uint64_t kMiB = uint64_t(1) << 20;

std::vector<uint8_t> pattern(uint64_t seed, size_t n) {
  std::vector<uint8_t> v(n);
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (size_t i = 0; i < n; ++i) {
    x ^= x << 13, x ^= x >> 7, x ^= x << 17;
    v[i] = uint8_t(x);
  }
  return v;
}

bool write_piece(Store& st, Store::Job& job, Kind k, const Key& key, const std::vector<uint8_t>& data,
                 const std::string& meta = "meta") {
  return st.write(job, k, key, meta, data.size(), [&](uint64_t off, uint8_t* dst, size_t n) {
    memcpy(dst, data.data() + off, n);
    return true;
  });
}

std::vector<uint8_t> read_piece(Store& st, Kind k, const Key& key, bool* ok, std::string* meta = nullptr) {
  std::vector<uint8_t> out;
  *ok = st.read(k, key, meta, [&](uint64_t off, const uint8_t* src, size_t n) {
    if (out.size() < off + n) out.resize(off + n);
    memcpy(out.data() + off, src, n);
    return true;
  });
  return out;
}

// A conversation: tokens base + i (i < n), KV blocks of `rows` tokens of block_bytes each, a state at every position
// in `pos` (state_bytes each); `id` keys its pieces (another id: other pieces, the same: the same ones)
struct Conv {
  Entry e;
  std::vector<std::vector<uint8_t>> blocks, states;
};

Conv make_conv(uint64_t id, int n, int base, std::vector<int> pos, size_t block_bytes, size_t state_bytes,
               int rows = 16, uint64_t state_id = 0) {
  Conv c;
  c.e.block_rows = uint32_t(rows);
  for (int i = 0; i < n; ++i) c.e.tokens.push_back(base + i);
  c.e.fills = {{n, id}};
  for (int b = 0; b * rows < n; ++b) {
    c.e.blocks.push_back(Hasher().u64(id).u64(uint64_t(b)).done());
    c.blocks.push_back(pattern(id * 1000 + uint64_t(b), block_bytes));
  }
  for (int p : pos) {
    Snap s;
    s.pos = p;
    s.flags = Snap::kQuery;
    s.dctx = p + 1, s.dvalid = 3;
    s.state = Hasher().u64(state_id ? state_id : id).u64(7).u64(uint64_t(p)).done();
    c.e.snaps.push_back(s);
    c.states.push_back(pattern((state_id ? state_id : id) * 77 + uint64_t(p), state_bytes));
  }
  c.e.vocab = {5, 9, 2};
  c.e.vocab_scanned = n;
  return c;
}

bool save_conv(Store& st, const Conv& c) {
  std::vector<std::pair<Kind, Key>> pieces;
  for (const Key& k : c.e.blocks) pieces.push_back({kKv, k});
  for (const Snap& s : c.e.snaps) pieces.push_back({kState, s.state});
  auto job = st.begin(pieces);
  for (size_t b = 0; b < c.blocks.size(); ++b)
    if (!write_piece(st, *job, kKv, c.e.blocks[b], c.blocks[b])) return false;
  for (size_t i = 0; i < c.states.size(); ++i)
    if (!write_piece(st, *job, kState, c.e.snaps[i].state, c.states[i], "state")) return false;
  return st.put(*job, c.e);
}

int hit_pos(Store& st, const std::vector<int>& prompt, int below = -1) {
  const Hit h = st.lookup(prompt.data(), int(prompt.size()), below < 0 ? int(prompt.size()) : below);
  return h ? h.pos : 0;
}

std::vector<int> tokens_of(const Conv& c, int n, int diverge = -1) {
  std::vector<int> t(c.e.tokens.begin(), c.e.tokens.begin() + std::min<size_t>(size_t(n), c.e.tokens.size()));
  while (int(t.size()) < n) t.push_back(900000 + int(t.size()));
  if (diverge >= 0 && diverge < n) t[size_t(diverge)] = -7;
  return t;
}

Store::Options opts(const std::string& root, const std::string& ns, uint64_t quota) {
  Store::Options o;
  o.root = root;
  o.ns = ns;
  o.about = "test namespace\n";
  o.quota = quota;
  o.lock_wait_s = 0.3;
  o.test_avail = 1000 * kGiB;  // a roomy volume unless a test says otherwise
  o.test_volume = 2000 * kGiB;
  return o;
}

std::string ns_name(uint64_t i) { return Hasher().u64(i).done().hex(); }

void set_mtime(const std::string& p, double days_ago) {
  timeval tv[2];
  gettimeofday(&tv[0], nullptr);
  tv[0].tv_sec -= time_t(days_ago * 86400);
  tv[1] = tv[0];
  utimes(p.c_str(), tv);
}

void flip_byte(const std::string& p, off_t at) {
  const int fd = open(p.c_str(), O_RDWR);
  uint8_t b = 0;
  pread(fd, &b, 1, at);
  b ^= 0x5a;
  pwrite(fd, &b, 1, at);
  close(fd);
}

// ---- tests ----

void crc_and_keys() {
  const char* s = "123456789";
  check(crc32c(0, s, 9) == 0xE3069283u, "CRC-32C of \"123456789\" is 0xE3069283");
  const std::vector<uint8_t> v = pattern(3, 100003);
  check(crc32c(crc32c(0, v.data(), 50001), v.data() + 50001, v.size() - 50001) == crc32c(0, v.data(), v.size()),
        "CRC-32C continues across calls");
  const int32_t a[3] = {1, 2, 3}, b[3] = {1, 2, 4};
  check(Hasher().i32s(a, 3).done() == Hasher().i32s(a, 3).done() && Hasher().i32s(a, 3).done() != Hasher().i32s(b, 3).done() &&
            Hasher().i32s(a, 2).done() != Hasher().i32s(a, 3).done(),
        "keys: the same tokens give the same key, others another");
  const Key k = Hasher().str("x").done();
  Key k2;
  check(Key::parse(k.hex(), &k2) && k2 == k && k.hex().size() == 32, "keys: 32 hex digits and back");
}

void sizes() {
  uint64_t b = 1;
  check(parse_size("auto", &b) && b == kAuto, "--disk-cache auto");
  check(parse_size("0", &b) && b == 0 && parse_size("off", &b) && b == 0, "--disk-cache 0 / off");
  check(parse_size("16G", &b) && b == 16 * kGiB && parse_size("4GiB", &b) && b == 4 * kGiB &&
            parse_size("512m", &b) && b == 512 * kMiB && parse_size("1.5T", &b) && b == 1536 * kGiB &&
            parse_size("100", &b) && b == 100,
        "--disk-cache sizes: 16G, 4GiB, 512m, 1.5T, bytes");
  check(!parse_size("abc", &b) && !parse_size("-1G", &b) && !parse_size("16X", &b) && !parse_size("", &b),
        "--disk-cache refuses what is not a size");
  check(space_limit(kAutoQuota, 28 * kGiB, 0, 460 * kGiB) == 18 * kGiB / 4,
        "auto: a quarter of the free space beyond the 10 GiB floor (28 GiB free: 4.5 GiB)");
  check(space_limit(kAutoQuota, 9 * kGiB, 0, 460 * kGiB) == 0, "auto: nothing below the floor");
  check(space_limit(kAutoQuota, 28 * kGiB, 2 * kGiB, 460 * kGiB) == 20 * kGiB / 4,
        "the cache's own files count as free space (its writes do not shrink its limit)");
  check(space_limit(kAutoQuota, 1000 * kGiB, 0, 2000 * kGiB) == kAutoQuota, "auto: at most 32 GiB");
  check(space_limit(2 * kGiB, 28 * kGiB, 0, 460 * kGiB) == 2 * kGiB, "a size caps it");
  check(space_limit(kAutoQuota, 25 * kGiB, 0, 100 * kGiB) == 20 * kGiB / 4, "a small volume keeps 5% of itself free");
}

void write_read_restart(const std::string& root) {
  const std::string ns = ns_name(1);
  Conv c = make_conv(11, 100, 1000, {48, 100}, 40000, 70000);
  {
    Store st;
    std::string why;
    check(st.open(opts(root, ns, 64 * kMiB), &why), "open a namespace", why);
    check(save_conv(st, c), "write an entry: 7 KV blocks and 2 states");
    const auto full = tokens_of(c, 130);
    check(hit_pos(st, full) == 100, "lookup: a prompt that extends the conversation hits its deepest state (100)");
    check(hit_pos(st, tokens_of(c, 130, 70)) == 48, "lookup: one that diverges at 70 hits the state at 48");
    check(hit_pos(st, tokens_of(c, 130, 10)) == 0, "lookup: one that diverges at 10 misses");
    check(hit_pos(st, full, 100) == 48, "lookup: only states before the prompt's last token (below 100: 48)");
    check(!st.lookup(full.data(), int(full.size()), int(full.size()), 3),
          "lookup: an entry is found for its own tag only (Dory: the loop count)");
    Conv t = make_conv(12, 80, 7000, {64}, 1000, 2000);
    t.e.tag = 2;
    const auto pt = tokens_of(t, 90);
    check(save_conv(st, t) && st.lookup(pt.data(), 90, 90, 2).pos == 64 && !st.lookup(pt.data(), 90, 90, 0),
          "... a tagged entry: found with its tag, not without");
    bool ok = true, all = true;
    std::string meta;
    for (size_t i = 0; i < c.blocks.size(); ++i) {
      bool r = false;
      all = all && read_piece(st, kKv, c.e.blocks[i], &r, &meta) == c.blocks[i] && r && meta == "meta";
    }
    for (size_t i = 0; i < c.states.size(); ++i) {
      bool r = false;
      all = all && read_piece(st, kState, c.e.snaps[i].state, &r, &meta) == c.states[i] && r && meta == "state";
    }
    check(ok && all, "read back: every piece's bytes and header data as written");
    Entry e;
    check(st.pin(st.lookup(full.data(), int(full.size()), int(full.size())).entry, &e) && e.tokens == c.e.tokens &&
              e.vocab == c.e.vocab && e.vocab_scanned == 100 && e.snaps.size() == 2 && e.snaps[1].dctx == 101 &&
              e.fills == c.e.fills,
          "the entry's record: tokens, fills, restore points, draft rows");
    st.unpin(e.key, true);
    const auto s = st.stats();
    check(s.entries == 2 && s.pieces == 15 && s.held == dir_bytes(st.dir() + "/kv") + dir_bytes(st.dir() + "/st") +
                                                           dir_bytes(st.dir() + "/e"),
          "held: the bytes of the namespace's files", std::to_string(s.held));
  }
  {
    Store st;  // the server restarts
    std::string why;
    check(st.open(opts(root, ns, 64 * kMiB), &why), "open again (a restart)", why);
    const auto pt = tokens_of(make_conv(12, 80, 7000, {64}, 1, 1), 90);
    check(st.stats().entries == 2 && hit_pos(st, tokens_of(c, 130)) == 100 && st.lookup(pt.data(), 90, 90, 2).pos == 64,
          "the entries are taken back, with their tags");
    bool all = true;
    for (size_t i = 0; i < c.blocks.size(); ++i) {
      bool r = false;
      all = all && read_piece(st, kKv, c.e.blocks[i], &r) == c.blocks[i] && r;
    }
    check(all, "its pieces read back as written after the restart");
    st.clear();
  }
}

void damage(const std::string& root) {
  const std::string ns = ns_name(2);
  Conv c = make_conv(21, 64, 2000, {64}, 30000, 50000);
  Conv d = make_conv(22, 64, 3000, {64}, 30000, 50000);
  Conv f = make_conv(23, 64, 4000, {64}, 30000, 50000);
  std::string dir;
  {
    Store st;
    std::string why;
    st.open(opts(root, ns, 64 * kMiB), &why);
    dir = st.dir();
    check(save_conv(st, c) && save_conv(st, d) && save_conv(st, f), "three entries written");
    flip_byte(dir + "/kv/" + c.e.blocks[2].hex(), 4096 + 777);  // one payload byte of one block
    bool r = true;
    read_piece(st, kKv, c.e.blocks[2], &r);
    check(!r && hit_pos(st, tokens_of(c, 80)) == 0 && !fs::exists(dir + "/kv/" + c.e.blocks[2].hex()),
          "a damaged piece fails its CRC: the entry that needs it is dropped with it");
    check(hit_pos(st, tokens_of(d, 80)) == 64, "the other entries stay");
  }
  // torn writes: a temporary file left by a crash, a state cut short, a block cut inside its payload, a damaged record
  const std::string tmp = dir + "/kv/" + d.e.blocks[0].hex() + ".tmp.999.1";
  if (FILE* t = fopen(tmp.c_str(), "w")) fputs("partial", t), fclose(t);
  truncate((dir + "/st/" + d.e.snaps[0].state.hex()).c_str(), 100);
  truncate((dir + "/kv/" + f.e.blocks[1].hex()).c_str(), 4096 + 1000);
  {
    Store st;
    std::string why;
    st.open(opts(root, ns, 64 * kMiB), &why);
    check(!fs::exists(tmp), "a leftover temporary file is removed at open");
    check(hit_pos(st, tokens_of(d, 80)) == 0, "an entry whose state was cut short is dropped at open");
    check(hit_pos(st, tokens_of(f, 80)) == 64, "a block cut inside its payload passes the open (its size is a header's)");
    bool r = true;
    read_piece(st, kKv, f.e.blocks[1], &r);
    check(!r && hit_pos(st, tokens_of(f, 80)) == 0, "... and fails when read: its entry is dropped");
    check(st.stats().pieces == 0 && st.stats().entries == 0, "no piece is left that no entry needs");
    Conv g = make_conv(24, 64, 5000, {64}, 30000, 50000);
    save_conv(st, g);
  }
  {
    std::string rec;
    for (const auto& de : fs::directory_iterator(dir + "/e")) rec = de.path();
    flip_byte(rec, 4096 + 5);
    Store st;
    std::string why;
    st.open(opts(root, ns, 64 * kMiB), &why);
    check(st.stats().entries == 0 && st.stats().pieces == 0, "a damaged entry record is dropped at open, with its pieces");
  }
}

void quota(const std::string& root) {
  const std::string ns = ns_name(3);
  Store st;
  std::string why;
  // one conversation: 4 blocks of 60000 + a state of 100000 (+ 4 KiB headers, a small record): ~0.37 MB
  check(st.open(opts(root, ns, 1 * kMiB), &why) && st.limit() == kMiB, "a quota of 1 MiB", why);
  Conv a = make_conv(31, 64, 100, {64}, 60000, 100000), b = make_conv(32, 64, 200, {64}, 60000, 100000),
       c = make_conv(33, 64, 300, {64}, 60000, 100000);
  check(save_conv(st, a) && save_conv(st, b), "two conversations fit");
  usleep(2000);
  Entry e;
  st.pin(st.lookup(tokens_of(a, 70).data(), 70, 70).entry, &e);  // a was used: b is now the least recently used
  st.unpin(e.key, true);
  check(save_conv(st, c), "a third one is written");
  check(hit_pos(st, tokens_of(b, 70)) == 0 && hit_pos(st, tokens_of(a, 70)) == 64 && hit_pos(st, tokens_of(c, 70)) == 64,
        "the least recently used entry went to make room", std::to_string(st.held()));
  check(st.held() <= st.limit(), "within the limit");
  // a later turn of c: the same first blocks (the same keys), a deeper state; c's entry has a restore point it lacks
  Conv c2 = make_conv(33, 64, 300, {64}, 60000, 100000);
  c2.e.snaps[0].state = Hasher().u64(99).done();
  c2.states[0] = pattern(99, 100000);
  const uint64_t before = st.stats().written;
  check(save_conv(st, c2), "a conversation sharing c's blocks");
  check(st.stats().written - before < 200000, "only its new state was written (the shared blocks were not)",
        std::to_string(st.stats().written - before));
  // a pinned entry stays; the others go first
  Entry pinned;
  st.pin(st.lookup(tokens_of(a, 70).data(), 70, 70).entry, &pinned);
  const uint64_t h0 = st.held();
  st.set_test_space(10 * kGiB + 2 * kMiB, 2000 * kGiB);  // the volume filled: 2 MiB free beyond the floor
  check(st.limit() == std::min(kMiB, (2 * kMiB + h0) / 4),
        "the volume fills: the limit shrinks to a quarter of what is left beyond the floor (its own files counted)");
  check(hit_pos(st, tokens_of(a, 70)) == 64 && st.held() <= st.limit() + 400000,
        "entries go to fit, least recently used first; a pinned one stays");
  st.unpin(pinned.key, false);
  st.measure();
  check(st.held() <= st.limit(), "unpinned, it goes too when it does not fit", std::to_string(st.held()));
  // the shared blocks: c2 holds them after c went
  st.set_test_space(1000 * kGiB, 2000 * kGiB);
  check(save_conv(st, c) && save_conv(st, c2), "c and c2 again");
  std::vector<int> pc = tokens_of(c, 70);
  // supersede: an entry with every restore point of another replaces it
  Conv c3 = c2;
  c3.e.snaps.push_back(c.e.snaps[0]);
  c3.states.push_back(c.states[0]);
  std::swap(c3.e.snaps[0], c3.e.snaps[1]);
  std::swap(c3.states[0], c3.states[1]);
  const size_t n0 = st.stats().entries;
  check(save_conv(st, c3) && st.stats().entries == n0 - 1, "an entry with all of two others' restore points replaces them",
        std::to_string(st.stats().entries) + " of " + std::to_string(n0));
  check(hit_pos(st, pc) == 64, "its restore points are all still found");
  // a piece larger than the limit is refused
  auto job = st.begin({});
  check(!write_piece(st, *job, kState, Hasher().u64(5).done(), pattern(5, 2 * kMiB)), "a piece larger than the limit is refused");
  // a write given up (no entry): its pieces stay for a later write, until room is needed or the next open
  check(write_piece(st, *job, kKv, Hasher().u64(6).done(), pattern(6, 1000)) && st.has(kKv, Hasher().u64(6).done()),
        "a piece written by a job");
  job.reset();
  check(st.has(kKv, Hasher().u64(6).done()), "kept after its job ends without an entry");
  st.clear();
}

// Threads at once, as the engine's are (its writer thread, requests reading what they pinned, lookups) while the volume's
// free space moves: no read of a pinned entry fails, and the files and the index agree afterwards and after a reopen.
void concurrency(const std::string& root) {
  const std::string ns = ns_name(4);
  std::atomic<int> read_fail{0}, reads{0}, writes{0};
  std::atomic<bool> stop{false};
  {
    Store st;
    std::string why;
    st.open(opts(root, ns, 3 * kMiB), &why);
    std::vector<std::thread> ts;
    for (int w = 0; w < 3; ++w)  // writers: conversations, some sharing their first blocks (the same ids)
      ts.emplace_back([&, w] {
        for (int i = 0; i < 40; ++i) {
          Conv c = make_conv(uint64_t(500 + (w * 40 + i) % 23), 64, 1000 * (w * 40 + i), {32, 64}, 20000, 30000);
          c.e.snaps[1].state = Hasher().u64(uint64_t(w * 1000 + i)).done();  // a state of its own
          writes += save_conv(st, c);
        }
      });
    for (int r = 0; r < 3; ++r)  // readers: an entry found, pinned, its pieces read, unpinned
      ts.emplace_back([&, r] {
        for (int i = 0; !stop; ++i) {
          Conv c = make_conv(uint64_t(500 + (r * 7 + i) % 23), 64, 0, {32, 64}, 1, 1);
          std::vector<int> p;
          for (int b = 0; b < 3; ++b)
            for (int t = 0; t < 64; ++t) p.push_back(1000 * (b * 40 + (r + i) % 40) + t);
          const Hit h = st.lookup(p.data(), 64, 64);
          Entry e;
          if (!h || !st.pin(h.entry, &e)) continue;
          for (const Key& k : e.blocks) {
            bool ok = false;
            read_piece(st, kKv, k, &ok);
            ++reads;
            read_fail += !ok;
          }
          st.unpin(e.key, true);
        }
      });
    ts.emplace_back([&] {  // the volume fills and empties
      for (int i = 0; !stop; ++i) {
        st.set_test_space(10 * kGiB + (i % 2 ? 4 : 12) * kMiB, 2000 * kGiB);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
    });
    for (int w = 0; w < 3; ++w) ts[size_t(w)].join();
    stop = true;
    for (size_t i = 3; i < ts.size(); ++i) ts[i].join();
    st.set_test_space(1000 * kGiB, 2000 * kGiB);
    const auto s = st.stats();
    check(writes > 0 && read_fail == 0, "threads at once: no read of a pinned entry fails",
          std::to_string(read_fail) + " of " + std::to_string(reads) + " reads failed, " + std::to_string(writes) +
              " conversations written");
    check(s.held == dir_bytes(st.dir() + "/kv") + dir_bytes(st.dir() + "/st") + dir_bytes(st.dir() + "/e"),
          "... and the index counts the bytes the files take", std::to_string(s.held));
  }
  Store st;
  std::string why;
  st.open(opts(root, ns, 3 * kMiB), &why);
  const size_t n = st.stats().entries;
  check(n > 0 && st.stats().held == dir_bytes(st.dir() + "/kv") + dir_bytes(st.dir() + "/st") + dir_bytes(st.dir() + "/e"),
        "... and a reopen takes back whole entries only", std::to_string(n) + " entries");
  st.clear();
}

void namespaces(const std::string& root) {
  const std::string r = root + "/ns-root";
  fs::create_directories(r);
  auto make_ns = [&](const std::string& name, double days, size_t bytes) {
    fs::create_directories(r + "/" + name + "/kv");
    const std::string lock = r + "/" + name + "/lock";
    if (FILE* f = fopen(lock.c_str(), "w")) fclose(f);
    if (FILE* f = fopen((r + "/" + name + "/kv/data").c_str(), "w")) {
      std::vector<uint8_t> z(bytes, 1);
      fwrite(z.data(), 1, z.size(), f);
      fclose(f);
    }
    set_mtime(lock, days);
  };
  const std::string old = ns_name(40), recent = ns_name(41), held = ns_name(42), mine = ns_name(43);
  make_ns(old, 15, 1000);
  make_ns(recent, 1, 3 * kMiB);
  make_ns(held, 30, 1000);
  fs::create_directories(r + "/not-a-cache");
  set_mtime(r + "/not-a-cache", 100);
  const int hfd = open((r + "/" + held + "/lock").c_str(), O_RDWR);
  flock(hfd, LOCK_EX | LOCK_NB);  // another server holds it
  {
    Store st;
    std::string why;
    check(st.open(opts(r, mine, 64 * kMiB), &why), "open beside other namespaces", why);
    check(!fs::exists(r + "/" + old), "a namespace unused for 14 days is removed at open");
    check(fs::exists(r + "/" + recent) && st.stats().others >= 3 * kMiB, "one used yesterday stays, counted against the limit");
    check(fs::exists(r + "/" + held), "one another server holds is left alone, however old");
    check(fs::exists(r + "/not-a-cache"), "a folder that is not a namespace is left alone");
    // the root's limit holds this namespace and the others nobody holds: the others go first, the oldest first
    Conv a = make_conv(51, 64, 100, {64}, 60000, 100000);
    check(save_conv(st, a), "an entry in the new namespace");
    st.set_test_space(10 * kGiB + 8 * kMiB, 2000 * kGiB);  // a 2 MiB limit: the other namespace's 3 MiB do not fit
    check(!fs::exists(r + "/" + recent) && hit_pos(st, tokens_of(a, 70)) == 64,
          "over the limit: another model's cache nobody holds goes before this one's entries");
    st.clear();
  }
  close(hfd);
  {
    Store a, b;
    std::string why;
    check(a.open(opts(r, mine, 64 * kMiB), &why), "the namespace's lock: the first server takes it", why);
    check(!b.open(opts(r, mine, 64 * kMiB), &why) && why.find("another") != std::string::npos,
          "a second one waits for it (here 0.3 s), then gives up", why);
  }
  {
    Store b;
    std::string why;
    check(b.open(opts(r, mine, 64 * kMiB), &why), "once the first lets it go, it opens", why);
  }
  const std::string root_attr = "com.apple.metadata:com_apple_backup_excludeItem";
  check(getxattr(r.c_str(), root_attr.c_str(), nullptr, 0, 0, 0) > 0, "the root is kept out of Time Machine");
}

}  // namespace

int main() {
  const char* base = getenv("TMPDIR");
  std::string t = std::string(base && *base ? base : "/tmp");
  if (t.back() != '/') t += '/';
  t += "kapitsa-pcache-test-XXXXXX";
  std::vector<char> buf(t.begin(), t.end());
  buf.push_back(0);
  if (!mkdtemp(buf.data())) {
    printf("FAIL cannot create a temporary folder\n");
    return 1;
  }
  const std::string root = buf.data();
  crc_and_keys();
  sizes();
  write_read_restart(root);
  damage(root);
  quota(root);
  concurrency(root);
  namespaces(root);
  std::error_code ec;
  fs::remove_all(root, ec);
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
