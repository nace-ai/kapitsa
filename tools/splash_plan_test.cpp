// splash::default_engine_args: what Kapitsa adds to `splash serve-native` and when (src/splash/plan.h).
#include "splash/plan.h"

#include <cstdio>
#include <cstdlib>

using ni::splash::default_engine_args;
using V = std::vector<std::string>;
static const uint64_t G = 1ull << 30;
static int failures = 0;
static void check(bool c, const char* what) {
  if (!c) { ++failures; std::printf("FAIL: %s\n", what); }
}

int main() {
  std::string note;
  // plenty of memory: only the idle release
  check(default_engine_args({}, 40 * G, false, "", &note) == V({"--idle-release", "off"}), "40 GiB: idle release only");
  // 30.4 GiB (the measured case): 16 GiB cache
  const uint64_t a = uint64_t(30.4 * G);
  check(default_engine_args({}, a, false, "", &note) == V({"--idle-release", "off", "--expert-memory", std::to_string(16 * G)}), "30.4 GiB -> 16G");
  check(note.find("--expert-memory 16G") != std::string::npos, "note names the cache");
  // 33 GiB (this Mac's best): 19 GiB
  check(default_engine_args({}, 33 * G, false, "", nullptr)[3] == std::to_string(19 * G), "33 GiB -> 19G");
  // 22 GiB is the least that gets a cache of 8; below it Splash decides
  check(default_engine_args({}, 22 * G, false, "", nullptr)[3] == std::to_string(8 * G), "22 GiB -> 8G");
  check(default_engine_args({}, 21 * G, false, "", &note) == V({"--idle-release", "off"}) && note.find("left to Splash") != std::string::npos, "21 GiB: left to Splash");
  // unknown memory (0): nothing about memory
  check(default_engine_args({}, 0, false, "", nullptr) == V({"--idle-release", "off"}), "unknown memory");
  // the user's arguments win and are not doubled
  check(default_engine_args({"--idle-release", "30m", "--expert-memory", "off"}, a, true, "", nullptr) == V({"--persistent-cache", "on"}), "user's flags kept");
  check(default_engine_args({"--persistent-cache", "on"}, 40 * G, true, "", nullptr) == V({"--idle-release", "off"}), "persistent cache not doubled");
  check(default_engine_args({}, 40 * G, true, "/c/m1", nullptr) == V({"--idle-release", "off", "--persistent-cache", "on", "--cache-dir", "/c/m1"}), "cache dir with persistent cache");
  check(default_engine_args({"--cache-dir", "/mine"}, 40 * G, true, "/c/m1", nullptr) == V({"--idle-release", "off", "--persistent-cache", "on"}), "user's cache dir kept");
  check(default_engine_args({}, 40 * G, false, "/c/m1", nullptr) == V({"--idle-release", "off"}), "no cache dir without persistent cache");
  if (failures) return 1;
  std::puts("splash_plan_test: ok");
  return 0;
}
