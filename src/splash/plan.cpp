#include "splash/plan.h"

#include <algorithm>

namespace ni::splash {

namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kAutoPlanNeeds = 38 * kGiB;  // Splash's auto plan (about 35.8 GiB) and a margin over it
constexpr uint64_t kHeadroom = 14 * kGiB;       // 2 GiB protected for macOS, the rest for everything else running
constexpr uint64_t kMinCache = 8 * kGiB;

// argparse takes `--flag value` and `--flag=value`: both mean the user gave it
bool has(const std::vector<std::string>& v, const std::string& flag) {
  return std::any_of(v.begin(), v.end(), [&](const std::string& a) { return a == flag || a.compare(0, flag.size() + 1, flag + "=") == 0; });
}

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
