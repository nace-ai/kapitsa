// KFlash budget cache (src/model/kflash_budget.cpp): a cache file loads only into the build that wrote it. A file an
// earlier build wrote, or one without a build, starts the budget fresh (kapitsa#1150: a calibration learned on a
// broken verify kept a fixed build's trees shaped by it across restarts). KfBudget's and KfBatch's saves write the
// build and load back. CPU only: KAPITSA_HOME is a temporary folder.
#include "model/kflash_budget.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using ni::KfBatch;
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
  {
    const std::string path = dir + "/kflash/" + key + ".json";
    auto j = nlohmann::json::parse(std::ifstream(path));
    j["wide"] = true;
    j["cost_narrow8"] = 12.0;
    j["beta"][2] = "damaged";
    std::ofstream(path) << j.dump();
    KfBudget b(key);
    check(b.alpha(1) == 0.0f && b.beta(1) == 1.0f, "a bad later depth discards the earlier calibration");
    check(b.cost(1) == 0.0 && b.cost_narrow() == 0.0, "a bad depth discards the cached row and narrow costs");
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
  auto file_build = [&](const std::string& k) {
    try {
      return nlohmann::json::parse(std::ifstream(dir + "/kflash/" + k + ".json")).value("build", "");
    } catch (...) {
      return std::string();
    }
  };
  // KfBudget::save, then a new budget of the same key: the file names this build and loads back
  {
    double c8 = 0;
    {
      KfBudget b(key);
      b.set_rows(1u << 7, 8);
      std::vector<int32_t> tree(ni::kf::kWords, 0);
      tree[ni::kf::kN] = 8;
      for (int i = 0; i < 40; ++i) b.observe(tree.data(), 30.0);
      c8 = b.cost(8);
      b.save();
    }
    check(file_build(key) == KfBudget::build(), "KfBudget::save writes this build into the file");
    KfBudget b(key);
    check(c8 > 0 && b.cost(8) == c8, "KfBudget::save, then load: the cost table comes back");
  }
  // KfBatch::save, then a new batch budget of the same key; then the same file from another build
  {
    const std::string bkey = "model-Drafter-Chip-batch";
    KfBatch::Member m[2];
    for (auto& g : m)
      for (float& lc : g.lc) lc = -0.5f;
    const int k[2] = {0, 0}, acc[2] = {0, 0};
    double c2 = 0;
    {
      KfBatch kb(bkey);
      for (int i = 0; i < 40; ++i) kb.observe(m, 2, k, acc, 20.0, nullptr);
      c2 = kb.cost(2, 2);
      kb.save();
    }
    check(file_build(bkey) == KfBudget::build(), "KfBatch::save writes this build into the file");
    {
      KfBatch kb(bkey);
      check(c2 > 0 && kb.cost(2, 2) == c2, "KfBatch::save, then load: the cost table comes back");
    }
    const std::string bf = dir + "/kflash/" + bkey + ".json";
    nlohmann::json j = nlohmann::json::parse(std::ifstream(bf));
    auto bad = j;
    bad["alpha"][1] = 2.5f;
    bad["beta"][2] = "damaged";
    std::ofstream(bf) << bad.dump();
    {
      KfBatch kb(bkey);
      float alpha[2], beta[2];
      kb.calibration(alpha, beta, 2);
      check(alpha[1] == 0.0f && beta[1] == 1.0f && kb.cost(2, 2) == 0.0,
            "a bad batch depth discards the calibration and member costs together");
    }
    bad = j;
    bad["cost"][3] = "damaged";
    std::ofstream(bf) << bad.dump();
    check(KfBatch(bkey).cost(2, 2) == 0.0, "a bad member cost row discards earlier rows");
    j["build"] = "0.2.33 (f55b55b5)";
    std::ofstream(bf) << j.dump() << "\n";
    KfBatch kb(bkey);
    check(kb.cost(2, 2) != c2 && kb.cycles() == 0, "a KfBatch file another build wrote is not loaded");
  }
  std::string rm = "rm -rf '" + dir + "'";
  if (system(rm.c_str()) != 0) fprintf(stderr, "could not remove %s\n", dir.c_str());
  printf("%s\n", fails ? "kflash_cache_test: FAILED" : "kflash_cache_test: all passed");
  return fails ? 1 : 0;
}
