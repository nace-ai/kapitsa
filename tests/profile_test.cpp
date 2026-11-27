#include "engine/device_profile.h"
#include "engine/profile.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "profile_test:%d: %s\n", __LINE__, #condition); std::abort(); \
} } while (false)

static int child(const std::string& setting, bool inkling_first) {
  if (setting == "unset") unsetenv("KAPITSA_PROFILE");
  else setenv("KAPITSA_PROFILE", setting == "empty" ? "" : setting.c_str(), 1);
  const bool automatic = setting == "unset" || setting == "empty";
  const auto device = ni::resolve_device({.apple_family = 10, .cores = 20});
  const auto inkling = ni::default_profile(device, {.inkling2 = true});
  const auto other = ni::default_profile(device, {});
  const auto expect = [&](ni::Profile fallback) {
    return automatic ? fallback : setting == "fast" ? ni::Profile::kFast : ni::Profile::kPrecise;
  };
  const auto first = inkling_first ? inkling : other;
  const auto second = inkling_first ? other : inkling;
  CHECK(ni::profile(first) == expect(first));
  CHECK(ni::profile(second) == expect(second));
  CHECK(ni::profile() == expect(ni::Profile::kPrecise));

  std::atomic<bool> ok{true};
  std::vector<std::thread> readers;
  for (int i = 0; i < 8; ++i) readers.emplace_back([&, i] {
    for (int n = 0; n < 1000; ++n) {
      const auto fallback = (n + i) % 2 ? inkling : other;
      if (ni::profile(fallback) != expect(fallback)) ok = false;
    }
  });
  for (auto& t : readers) t.join();
  CHECK(ok);

  for (const char* env : {"NI_INK2_TENSOR", "NI_INK2_Q8"}) {
    unsetenv(env);
    CHECK(ni::profile_switch(env, true, "test member", inkling) == (expect(inkling) == ni::Profile::kFast));
    CHECK(!ni::profile_switch(env, false, "test member", inkling));
    setenv(env, "", 1);
    CHECK(ni::profile_switch(env, true, "test member", inkling) == (expect(inkling) == ni::Profile::kFast));
    setenv(env, "0", 1);
    CHECK(!ni::profile_switch(env, true, "test member", inkling));
    setenv(env, "1", 1);
    CHECK(ni::profile_switch(env, false, "test member", ni::Profile::kPrecise));
    setenv(env, "yes", 1);
    CHECK(ni::profile_switch(env, false, "test member", ni::Profile::kPrecise));
    unsetenv(env);
  }
  // Only the explicit override is cached. Changing the environment later cannot leak a new selection.
  setenv("KAPITSA_PROFILE", setting == "fast" ? "precise" : "fast", 1);
  CHECK(ni::profile(inkling) == expect(inkling));
  CHECK(ni::profile(other) == expect(other));
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "--child") return child(argv[2], std::string(argv[3]) == "inkling");
  for (const char* env : {"unset", "empty", "precise", "fast", "invalid"}) {
    for (const char* first : {"inkling", "other"}) {
      const pid_t pid = fork();
      CHECK(pid >= 0);
      if (pid == 0) {
        execl(argv[0], argv[0], "--child", env, first, static_cast<char*>(nullptr));
        _exit(127);
      }
      int status = 0;
      CHECK(waitpid(pid, &status, 0) == pid);
      CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
  }
  std::puts("profile_test: 10 fresh-process cases PASS; both model orders, concurrent reads and switch overrides");
}
