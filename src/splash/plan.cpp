#include "splash/plan.h"

#include <algorithm>

namespace ni::splash {

namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kAutoPlanNeeds = 38 * kGiB;  // Splash's auto plan (about 35.8 GiB) and a margin over it
constexpr uint64_t kHeadroom = 14 * kGiB;       // 2 GiB protected for macOS, the rest for everything else running
constexpr uint64_t kMinCache = 8 * kGiB;

bool has(const std::vector<std::string>& v, const char* flag) { return std::find(v.begin(), v.end(), flag) != v.end(); }

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

}  // namespace ni::splash
