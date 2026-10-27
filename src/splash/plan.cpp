#include "splash/plan.h"

#include <algorithm>

namespace ni::splash {

namespace {

constexpr uint64_t kGiB = 1ull << 30;
// Splash's auto plan has no fixed threshold (its refusal is dynamic): one start was refused at 35.2 GiB under a system pressure
// warning, and one started at 33.0 GiB. 38 is a margin over the refusal seen, not a measured need; docs/splash-backend.md.
constexpr uint64_t kAutoPlanNeeds = 38 * kGiB;
constexpr uint64_t kHeadroom = 14 * kGiB;       // 2 GiB protected for macOS, the rest for everything else running
constexpr uint64_t kMinCache = 8 * kGiB;

// argparse takes `--flag value` and `--flag=value`: both mean the user gave it
bool has(const std::vector<std::string>& v, const std::string& flag) {
  return std::any_of(v.begin(), v.end(), [&](const std::string& a) { return a == flag || a.compare(0, flag.size() + 1, flag + "=") == 0; });
}

// BLAKE2b (RFC 7693) with a 16-byte digest and the 16-byte personalization "splash-salt", as frontend.py's _cache_salt hashes
// a client's cache_salt, so one salt means one cache key here and in plain Splash (and tenants cannot craft collisions).
struct Blake2b {
  uint64_t h[8];
  uint8_t buf[128];
  size_t n = 0;
  uint64_t total = 0;
  static constexpr uint64_t IV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                                     0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  Blake2b(const char* person, size_t plen) {
    uint8_t pb[64] = {};
    pb[0] = 16, pb[2] = 1, pb[3] = 1;  // digest length, key length 0, fanout 1, depth 1
    for (size_t i = 0; i < plen && i < 16; i++) pb[48 + i] = uint8_t(person[i]);
    for (int i = 0; i < 8; i++) {
      uint64_t w = 0;
      for (int j = 7; j >= 0; j--) w = (w << 8) | pb[i * 8 + j];
      h[i] = IV[i] ^ w;
    }
  }
  static uint64_t rotr(uint64_t x, int r) { return (x >> r) | (x << (64 - r)); }
  void compress(bool last) {
    static const uint8_t sg[10][16] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
                                       {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
                                       {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
                                       {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
                                       {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}};
    uint64_t m[16], v[16];
    for (int i = 0; i < 16; i++) {
      m[i] = 0;
      for (int j = 7; j >= 0; j--) m[i] = (m[i] << 8) | buf[i * 8 + j];
    }
    for (int i = 0; i < 8; i++) v[i] = h[i], v[i + 8] = IV[i];
    v[12] ^= total;
    if (last) v[14] = ~v[14];
    auto g = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
      v[a] += v[b] + x; v[d] = rotr(v[d] ^ v[a], 32);
      v[c] += v[d];     v[b] = rotr(v[b] ^ v[c], 24);
      v[a] += v[b] + y; v[d] = rotr(v[d] ^ v[a], 16);
      v[c] += v[d];     v[b] = rotr(v[b] ^ v[c], 63);
    };
    for (int r = 0; r < 12; r++) {
      const uint8_t* s_ = sg[r % 10];
      g(0, 4, 8, 12, m[s_[0]], m[s_[1]]);   g(1, 5, 9, 13, m[s_[2]], m[s_[3]]);
      g(2, 6, 10, 14, m[s_[4]], m[s_[5]]);  g(3, 7, 11, 15, m[s_[6]], m[s_[7]]);
      g(0, 5, 10, 15, m[s_[8]], m[s_[9]]);  g(1, 6, 11, 12, m[s_[10]], m[s_[11]]);
      g(2, 7, 8, 13, m[s_[12]], m[s_[13]]); g(3, 4, 9, 14, m[s_[14]], m[s_[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
  }
  void update(const std::string& in) {
    for (unsigned char c : in) {
      if (n == 128) {  // a full block that is not the last one
        total += 128;
        compress(false);
        n = 0;
      }
      buf[n++] = c;
    }
  }
  Salt finish() {
    total += n;
    std::fill(buf + n, buf + 128, uint8_t(0));
    compress(true);
    return {h[0], h[1]};  // the first 16 digest bytes, little-endian words: low half, then high half
  }
};

}  // namespace

std::vector<std::string> default_engine_args(const std::vector<std::string>& user, uint64_t available_bytes,
                                             bool persistent_cache, const std::string& cache_dir,
                                             std::string* note) {
  std::vector<std::string> out;
  std::string n;
  auto say = [&](const std::string& s) { n += (n.empty() ? "" : "; ") + s; };
  if (!has(user, "--idle-release")) {
    out.insert(out.end(), {"--idle-release", "off"});
    say("--idle-release off");
  }
  if (!has(user, "--expert-memory") && available_bytes > 0 && available_bytes < kAutoPlanNeeds) {
    if (available_bytes >= kHeadroom + kMinCache) {
      const uint64_t gib = (available_bytes - kHeadroom) / kGiB;
      out.insert(out.end(), {"--expert-memory", std::to_string(gib * kGiB)});
      say("--expert-memory " + std::to_string(gib) + "G (" + std::to_string(available_bytes / kGiB) + " GiB free+reclaimable)");
    } else {
      say("only " + std::to_string(available_bytes / kGiB) + " GiB free+reclaimable: expert memory left to Splash");
    }
  }
  if (persistent_cache && !has(user, "--persistent-cache")) {
    out.insert(out.end(), {"--persistent-cache", "on"});
    say("--persistent-cache on");
    if (!cache_dir.empty() && !has(user, "--cache-dir")) {
      out.insert(out.end(), {"--cache-dir", cache_dir});
      say("--cache-dir " + cache_dir);
    }
  }
  if (note) *note = n;
  return out;
}

Salt client_salt(const std::string& salt) {
  Blake2b b("splash-salt", 11);
  b.update(salt);
  return b.finish();
}

Salt conversation_salt(const nlohmann::ordered_json& messages) {
  Salt s{0xcbf29ce484222325ull, 0x84222325cbf29ce4ull};
  auto mix = [&](const std::string& t) {
    for (unsigned char c : t) {
      s.lo = (s.lo ^ c) * 0x100000001b3ull;
      s.hi = (s.hi ^ (c + 0x9eu)) * 0x100000001b3ull;
    }
  };
  for (const auto& m : messages) {
    mix(m.dump());
    if (m.value("role", "") == "user") break;
  }
  return s;
}

}  // namespace ni::splash
