// hub.h's offline visits (issue #1187): while one is open, no fetch is made, and a worker thread started inside one is
// inside it too. Before this, set_offline(true) was one process-wide flag, so `kapitsa doctor`'s plans, which run on
// threads of their own, fetched every header they needed and a plain `kapitsa doctor` meant a multi-GB download.
// Needs no GPU and loads no model: a visit, a plan_dir on a throwaway KAPITSA_HOME, and the flag itself.
#include "hub/hub.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

namespace fs = std::filesystem;
namespace hub = ni::hub;

static int failures = 0;
static void check(bool c, const std::string& what) {
  if (!c) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

// Whether a thread started now (as plan_dir's workers are) is refused a fetch
static bool worker_refused() {
  bool refused = false;
  std::thread t([&] { refused = hub::fetch_refused(); });
  t.join();
  return refused;
}

int main() {
  const fs::path home = fs::temp_directory_path() / ("k1187-hub-" + std::to_string(getpid()));
  fs::remove_all(home);
  fs::create_directories(home / "plans");
  setenv("KAPITSA_HOME", home.c_str(), 1);

  check(!hub::fetch_refused(), "a fetch outside a visit goes ahead");
  check(!worker_refused(), "a worker outside a visit goes ahead");

  hub::set_offline(true);
  check(hub::fetch_refused(), "inside a visit, a fetch is refused");
  check(worker_refused(), "a worker thread started inside a visit is inside it too");

  hub::set_offline(true);  // nested: its end leaves the outer visit open
  hub::set_offline(false);
  check(hub::fetch_refused(), "with the outer visit still open, a fetch is refused");
  check(hub::fetch_refused(), "a nested visit's end leaves the outer one open");
  check(worker_refused(), "the outer visit still covers a worker thread");

  hub::set_offline(false);
  check(!hub::fetch_refused(), "the outer visit's end lets a fetch go ahead");

  hub::set_offline(true);
  hub::set_offline(true);
  hub::leave_offline();  // every visit on this thread ends at once
  check(!hub::fetch_refused(), "leave_offline ends a nested visit and the outer one with it");

  // a catalog model that is not on this Mac: refused, and counted, rather than fetched
  hub::set_offline(true);
  const hub::ModelRef r = hub::resolve("qwen3.5-4b:4bit");
  const bool refused = hub::plan_dir(r).empty();
  check(refused, "plan_dir fetches nothing inside a visit: the plan is empty");
  check(hub::offline_misses() > 0, "plan_dir counts the plan it could not make offline");
  hub::leave_offline();
  check(!hub::fetch_refused(), "leave_offline ends the visit plan_dir was in");

  fs::remove_all(home);
  if (failures == 0) std::printf("PASS: hub offline visits\n");
  return failures == 0 ? 0 : 1;
}
