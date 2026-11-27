#include "engine/profile.h"

#include "util/log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace ni {

Profile profile(Profile fallback) {
  static const std::optional<Profile> override = []() -> std::optional<Profile> {
    const char* v = getenv("KAPITSA_PROFILE");
    if (!v || !*v) return std::nullopt;
    if (!strcmp(v, "precise")) return Profile::kPrecise;
    if (!strcmp(v, "fast")) return Profile::kFast;
    fprintf(stderr, "[kapitsa] KAPITSA_PROFILE=%s: not fast or precise; running precise\n", v);
    return Profile::kPrecise;
  }();
  return override.value_or(fallback);
}

bool profile_switch(const char* env, bool fast_here, const char* what, Profile fallback) {
  if (const char* v = getenv(env); v && *v) return strcmp(v, "0") != 0;
  if (profile(fallback) != Profile::kFast || !fast_here) return false;
  logi("profile fast: %s (%s=0 or KAPITSA_PROFILE=precise turns it off)", what, env);
  return true;
}

}  // namespace ni
