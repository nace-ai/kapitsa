// splash::default_engine_args: what Kapitsa adds to `splash serve-native` and when (src/splash/plan.h).
#include "splash/plan.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>

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
  check(default_engine_args({}, 40 * G, 0, false, "", &note) == V({"--idle-release", "off"}), "40 GiB: idle release only");
  // 30.4 GiB (the measured case): 16 GiB cache
  const uint64_t a = uint64_t(30.4 * G);
  check(default_engine_args({}, a, 0, false, "", &note) == V({"--idle-release", "off", "--expert-memory", std::to_string(16 * G)}), "30.4 GiB -> 16G");
  check(note.find("--expert-memory 16G") != std::string::npos, "note names the cache");
  // 33 GiB (this Mac's best): 19 GiB
  check(default_engine_args({}, 33 * G, 0, false, "", nullptr)[3] == std::to_string(19 * G), "33 GiB -> 19G");
  // 22 GiB is the least that gets a cache of 8; below it Splash decides
  check(default_engine_args({}, 22 * G, 0, false, "", nullptr)[3] == std::to_string(8 * G), "22 GiB -> 8G");
  check(default_engine_args({}, 21 * G, 0, false, "", &note) == V({"--idle-release", "off"}) && note.find("left to the engine") != std::string::npos, "21 GiB: left to the engine");
  // unknown memory (0): nothing about memory
  check(default_engine_args({}, 0, 0, false, "", nullptr) == V({"--idle-release", "off"}), "unknown memory");
  // the user's arguments win and are not doubled
  check(default_engine_args({"--idle-release", "30m", "--expert-memory", "off"}, a, 0, true, "", nullptr) == V({"--persistent-cache", "on"}), "user's flags kept");
  check(default_engine_args({"--persistent-cache", "on"}, 40 * G, 0, true, "", nullptr) == V({"--idle-release", "off"}), "persistent cache not doubled");
  check(default_engine_args({}, 40 * G, 0, true, "/c/m1", nullptr) == V({"--idle-release", "off", "--persistent-cache", "on", "--cache-dir", "/c/m1"}), "cache dir with persistent cache");
  check(default_engine_args({"--cache-dir", "/mine"}, 40 * G, 0, true, "/c/m1", nullptr) == V({"--idle-release", "off", "--persistent-cache", "on"}), "user's cache dir kept");
  check(default_engine_args({}, 40 * G, 0, false, "/c/m1", nullptr) == V({"--idle-release", "off"}), "no cache dir without persistent cache");
  // `--flag=value` is the user's flag too: not doubled
  check(default_engine_args({"--idle-release=30m", "--expert-memory=off"}, a, 0, true, "", nullptr) == V({"--persistent-cache", "on"}), "--flag=value kept");
  check(default_engine_args({"--persistent-cache=on", "--cache-dir=/mine"}, 40 * G, 0, true, "/c/m1", nullptr) == V({"--idle-release", "off"}), "--persistent-cache=on and --cache-dir=x kept");
  check(default_engine_args({"--idle-release-x", "1"}, 40 * G, 0, false, "", nullptr) == V({"--idle-release", "off"}), "a longer flag is not the flag");
  // the model's weights decide whether the cache is forced (sizes from Splash's resolved model.json, M5 Pro 48 GB, Oct 2026)
  const uint64_t flash_next = uint64_t(62.98 * G), a3b = uint64_t(19.98 * G), dense27 = uint64_t(18.53 * G);
  // Flash-Next (63 GiB of weights) never fits on this Mac: the same arguments as with an unknown size, at every memory
  for (uint64_t m = 0; m <= 48 * G; m += G / 4)
    check(default_engine_args({}, m, flash_next, false, "", nullptr) == default_engine_args({}, m, 0, false, "", nullptr), "Flash-Next: unchanged");
  check(default_engine_args({}, a, flash_next, false, "", nullptr)[3] == std::to_string(16 * G), "Flash-Next at 30.4 GiB -> 16G");
  // the 35B-A3B (19.98 GiB) fits from 22 GiB free: no forced cache, so no forced streaming of its experts
  for (uint64_t m = 22 * G; m < 38 * G; m += G / 4)
    check(default_engine_args({}, m, a3b, false, "", nullptr) == V({"--idle-release", "off"}), "35B-A3B at 22-38 GiB: left to the engine");
  check((default_engine_args({}, 25 * G, a3b, false, "", &note), note.find("weights (19 GiB) fit in 25 GiB") != std::string::npos), "note says the weights fit");
  check(default_engine_args({}, 21 * G, a3b, false, "", nullptr) == V({"--idle-release", "off"}), "35B-A3B at 21 GiB: left to the engine");
  // outside the 22-38 GiB band all three get what the rule before the weights gave, byte for byte
  for (uint64_t w : {flash_next, a3b, dense27}) {
    for (uint64_t m = 0; m < 22 * G; m += G / 4)
      check(default_engine_args({}, m, w, false, "", nullptr) == default_engine_args({}, m, 0, false, "", nullptr), "below 22 GiB: unchanged");
    for (uint64_t m = 38 * G; m <= 128 * G; m += G / 4)
      check(default_engine_args({}, m, w, true, "/c/m", nullptr) == default_engine_args({}, m, 0, true, "/c/m", nullptr), "from 38 GiB: unchanged");
  }
  // the dense 27B (no routed experts; Splash ignores --expert-memory there) is never given one
  for (uint64_t m = 0; m <= 48 * G; m += G / 4)
    check(default_engine_args({}, m, dense27, false, "", nullptr) == V({"--idle-release", "off"}), "27B dense: no --expert-memory");
  // the user's own --expert-memory still wins
  check(default_engine_args({"--expert-memory", "12G"}, 25 * G, a3b, false, "", nullptr) == V({"--idle-release", "off"}), "user's --expert-memory kept");
  // the weights' bytes from model.json: .safetensors and .gguf of the target and the draft, nothing else
  const nlohmann::json mj = {{"files", {{"target/model-00001-of-00002.safetensors", {{"bytes", 3 * G}}},
                                        {"target/model-00002-of-00002.safetensors", {{"bytes", 2 * G}}},
                                        {"draft/model.safetensors", {{"bytes", G}}},
                                        {"target/x.gguf", {{"bytes", 7 * G}}},
                                        {"target/config.json", {{"bytes", 4096}}},
                                        {"tokenizer/tokenizer.json", {{"bytes", 11 * G}}}}}};
  check(ni::splash::model_weight_bytes(mj) == 13 * G, "weight bytes: weights only");
  // the draft counts only when the resolved folder has one (serve-native then always loads it); without one, the target only
  const nlohmann::json no_draft = {{"files", {{"target/model.safetensors", {{"bytes", 5 * G}}}, {"target/config.json", {{"bytes", 4096}}}}}};
  check(ni::splash::model_weight_bytes(no_draft) == 5 * G, "weight bytes: no draft, target only");
  nlohmann::json mj_no_draft = mj;
  mj_no_draft["files"].erase("draft/model.safetensors");
  check(ni::splash::model_weight_bytes(mj) - ni::splash::model_weight_bytes(mj_no_draft) == G, "weight bytes: the draft's 1 GiB counts when listed");
  check(ni::splash::model_weight_bytes(nlohmann::json::object()) == 0 && ni::splash::model_weight_bytes(nlohmann::json()) == 0, "weight bytes: none listed");
  check(ni::splash::model_weight_bytes({{"files", {{"a.safetensors", {{"bytes", "x"}}}}}}) == 0, "weight bytes: a bad entry counts nothing");
  // model.json read from the resolved folder: missing or not JSON is 0, the rule before the weights
  {
    char tmpl[] = "/tmp/splash_plan_test.XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    check(ni::splash::model_weight_bytes_in(dir) == 0, "model.json missing: 0");
    check(ni::splash::model_weight_bytes_in(dir + "/nope") == 0, "folder missing: 0");
    { std::ofstream(dir + "/model.json") << "{\"files\": {\"a.safetensors\": {\"bytes\": 5"; }
    check(ni::splash::model_weight_bytes_in(dir) == 0, "model.json truncated: 0");
    check(default_engine_args({}, a, ni::splash::model_weight_bytes_in(dir), false, "", nullptr) ==
              V({"--idle-release", "off", "--expert-memory", std::to_string(16 * G)}), "unreadable model.json: the cache as before");
    { std::ofstream(dir + "/model.json") << mj.dump(); }
    check(ni::splash::model_weight_bytes_in(dir) == 13 * G, "model.json read");
    std::remove((dir + "/model.json").c_str());
    rmdir(dir.c_str());
  }
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
