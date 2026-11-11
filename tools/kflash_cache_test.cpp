// KFlash budget cache (src/model/kflash_budget.cpp): a cache file loads only into the build that wrote it. A file an
// earlier build wrote, or one without a build, starts the budget fresh (kapitsa#1150: a calibration learned on a
// broken verify kept a fixed build's trees shaped by it across restarts). CPU only: KAPITSA_HOME is a temporary folder.
#include "model/kflash_budget.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using ni::KfBudget;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++fails;
}

static void write_cache(const std::string& dir, const std::string& key, const char* build, float a1) {
  nlohmann::json j;
  j["cost_of"] = "cycle GPU ms";
  if (build) j["build"] = build;
  std::vector<double> cost(33, 0.0);
  for (int n = 1; n <= 8; ++n) cost[size_t(n)] = 30.0 + n;
  j["cost"] = cost;
  std::vector<float> a(33, 0.0f), b(33, 1.0f);
  a[1] = a1;
  b[1] = 1.5f;
  j["alpha"] = a;
  j["beta"] = b;
  std::ofstream(dir + "/kflash/" + key + ".json") << j.dump() << "\n";
}

int main() {
  char tmpl[] = "/tmp/kflash_cache_test.XXXXXX";
  const char* home = mkdtemp(tmpl);
  if (!home) {
    perror("mkdtemp");
    return 2;
  }
  setenv("KAPITSA_HOME", home, 1);
  const std::string dir = home;
  mkdir((dir + "/kflash").c_str(), 0755);
  const std::string key = "model-Drafter-Chip";

  write_cache(dir, key, KfBudget::build(), 2.5f);
  {
    KfBudget b(key);
    check(b.alpha(1) == 2.5f && b.beta(1) == 1.5f, "a cache this build wrote is loaded");
  }
  write_cache(dir, key, "0.2.33 (f55b55b5)", 2.5f);
  {
    KfBudget b(key);
    check(b.alpha(1) == 0.0f && b.beta(1) == 1.0f, "a cache another build wrote is not loaded (calibration fresh)");
    check(b.cost(1) == 0.0, "a cache another build wrote is not loaded (cost table fresh)");
  }
  write_cache(dir, key, nullptr, 2.5f);
  {
    KfBudget b(key);
    check(b.alpha(1) == 0.0f && b.beta(1) == 1.0f, "a cache without a build (written before the check) is not loaded");
  }
  {
    std::ifstream f(dir + "/kflash/" + key + ".json");
    check(bool(f), "the earlier build's file stays until this build saves over it");
  }
  std::string rm = "rm -rf '" + dir + "'";
  if (system(rm.c_str()) != 0) fprintf(stderr, "could not remove %s\n", dir.c_str());
  printf("%s\n", fails ? "kflash_cache_test: FAILED" : "kflash_cache_test: all passed");
  return fails ? 1 : 0;
}
