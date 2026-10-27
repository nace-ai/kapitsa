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
  // `--flag=value` is the user's flag too: not doubled
  check(default_engine_args({"--idle-release=30m", "--expert-memory=off"}, a, true, "", nullptr) == V({"--persistent-cache", "on"}), "--flag=value kept");
  check(default_engine_args({"--persistent-cache=on", "--cache-dir=/mine"}, 40 * G, true, "/c/m1", nullptr) == V({"--idle-release", "off"}), "--persistent-cache=on and --cache-dir=x kept");
  check(default_engine_args({"--idle-release-x", "1"}, 40 * G, false, "", nullptr) == V({"--idle-release", "off"}), "a longer flag is not the flag");
  // the conversation salt
  using J = nlohmann::ordered_json;
  const J sys = {{"role", "system"}, {"content", "be brief"}}, u1 = {{"role", "user"}, {"content", "first"}},
          as = {{"role", "assistant"}, {"content", "ok"}}, u2 = {{"role", "user"}, {"content", "more"}}, other = {{"role", "user"}, {"content", "another"}};
  auto salt = [](std::initializer_list<J> m) { return ni::splash::conversation_salt(J(m)); };
  auto same = [](ni::splash::Salt x, ni::splash::Salt y) { return x.lo == y.lo && x.hi == y.hi; };
  check(same(salt({sys, u1}), salt({sys, u1, as, u2})), "salt: later turns keep the conversation's salt");
  check(same(salt({sys, u1, as, u2}), salt({sys, u1, as, u2, as, other})), "salt: turn 3 too");
  check(!same(salt({sys, u1}), salt({sys, other})), "salt: another first user message, another salt");
  check(!same(salt({sys, u1}), salt({u1})), "salt: another system prompt, another salt");
  check(salt({sys, u1}).lo != 0 && salt({sys, u1}).hi != 0 && salt({sys, u1}).lo != salt({sys, u1}).hi, "salt: 128 distinct bits, not zero");
  check(!same(salt({sys}), salt({sys, u1})) && same(salt({sys}), salt({sys})), "salt: a system-only history is stable and not a conversation's");
  // a client's cache_salt: Python hashlib.blake2b(digest_size=16, person=b"splash-salt") of the same string, as Splash's frontend computes it
  auto cs = [](const std::string& t) { return ni::splash::client_salt(t); };
  check(cs("tenant-a").lo == 4268102603047165630ull && cs("tenant-a").hi == 14543633796230781686ull, "client salt: short string");
  check(cs("x").lo != cs("y").lo && !same(cs("tenant-a"), cs("tenant-b")), "client salt: another string, another salt");
  check(cs(std::string(300, 'x')).lo == 16612783538970333422ull && cs(std::string(300, 'x')).hi == 11575204618970738073ull, "client salt: 300 bytes");
  check(cs(std::string(128, 'a')).lo == 12653317393471799425ull && cs(std::string(128, 'a')).hi == 16940112665974098766ull, "client salt: exactly one block");
  check(cs(std::string(256, 'a')).lo == 10747077782074702016ull && cs(std::string(256, 'a')).hi == 17198004416459853536ull, "client salt: exactly two blocks");
  if (failures) return 1;
  std::puts("splash_plan_test: ok");
  return 0;
}
