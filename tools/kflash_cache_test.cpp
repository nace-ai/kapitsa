// KFlash budget cache (src/model/kflash_budget.cpp): a cache file loads only into the build that wrote it. A file an
// earlier build wrote, or one without a build, starts the budget fresh (kapitsa#1150: a calibration learned on a
// broken verify kept a fixed build's trees shaped by it across restarts). KfBudget's and KfBatch's saves write the
// build and load back. CPU only: KAPITSA_HOME is a temporary folder.
#include "model/kflash_budget.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
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

static ni::KfParams plan(KfBudget& b, bool greedy_moe = true) {
  ni::KfParams p{};
  b.write(p, greedy_moe);
  return p;
}

static void observe(KfBudget& b, int n, double ms) {
  std::array<int32_t, ni::kf::kWords> tree{};
  tree[ni::kf::kObs] = n;
  b.observe(tree.data(), ms);
}

// NI_KFLASH_FORCE is read once per process. Fork before the parent writes its first plan.
static bool test_forced(const char* setting, int first, int second) {
  const pid_t pid = fork();
  if (pid == 0) {
    setenv("NI_KFLASH_FORCE", setting, 1);
    KfBudget b;
    const auto p = plan(b);
    observe(b, first, 20);
    const auto q = plan(b);
    _exit(p.force_n == first && q.force_n == second && q.cost[8] == 1.0f ? 0 : 1);
  }
  int status = 0;
  return pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static void test_greedy_costs(const std::string& dir) {
  check(test_forced("3", 3, 3), "a fixed verify count takes precedence over greedy exploration");
  check(test_forced("-1", 1, 2), "cycling every allowed count takes precedence over greedy exploration");
  unsetenv("NI_KFLASH_FORCE");

  KfBudget b("partial");
  check(plan(b, false).force_n == 1 && plan(b).force_n == 8, "greedy and sampled start at their own row counts");
  observe(b, 8, 26);
  const auto upper = plan(b);
  check(upper.force_n == 1 && upper.cost[5] == 26, "one measured count sets costs until a second is observed");
  observe(b, 1, 12);
  const auto lower = plan(b);
  check(lower.force_n == 4 && lower.cost[4] == 18 && lower.cost[6] == 22, "interpolate between the two endpoints");
  observe(b, 4, 21);
  const auto mid = plan(b);
  check(!mid.force_n && mid.cost[2] == 15 && mid.cost[6] == 23.5, "a third measurement splits the estimate");
  check(b.cost(2) == 0 && plan(b, false).cost[2] == 1.0f, "sampled planning only sees measured costs");

  for (int i = 3; i < 32; ++i) observe(b, 4, 21);
  b.save();
  const auto j = nlohmann::json::parse(std::ifstream(dir + "/kflash/partial.json"));
  check(j.at("cost").at(2) == 0 && j.at("cost").at(1) == 12, "only actual measurements reach the cache");
  KfBudget partial("partial");
  check(plan(partial).force_n == 8, "a partial cache first requests its largest measured count");
  observe(partial, 8, 36);
  const auto rebased = plan(partial);
  check(!rebased.force_n && rebased.cost[2] == 25 && partial.cost(2) == 0, "partial cache estimates use the new base");
  bool missing_only = true;
  int probes = 0;
  for (int i = 1; i < 64; ++i) {
    const int n = plan(partial).force_n;
    if (n) {
      ++probes;
      missing_only = missing_only && i % 8 == 0 && partial.cost(n) == 0;
    }
    observe(partial, n ? n : 4, 20);
  }
  check(missing_only && probes == 5, "one missing count per eighth cycle eventually fills a partial table");

  write_cache(dir, "complete", KfBudget::build(), 0);
  KfBudget complete("complete");
  check(plan(complete).force_n == 8, "a complete cache requests its reference count");
  observe(complete, 8, 48);
  const auto warm = plan(complete);
  check(!warm.force_n && warm.cost[4] == 44 && warm.cost[4] == complete.cost(4), "a complete cache uses measured costs");
  KfBudget clipped;
  int late = 0;
  for (int i = 0; i < 64; ++i) {
    if (plan(clipped).force_n && i >= 6) ++late;
    observe(clipped, 6, 20);  // the tree actually verified six rows, whatever the budget requested
  }
  check(late == 7, "clipped anchors release the planner after the initial six cycles");
  KfBudget restricted;
  restricted.set_rows(0x88u, 8);
  check(plan(restricted).force_n == 4, "a restricted mask uses its original exhaustive rule");
  KfBudget wide;
  wide.set_rows((1u << 7) | (1u << 15), 16);
  wide.set_wide(true);
  check(plan(wide).cap == 8 && plan(wide).force_n == 0, "wide starts at the existing narrow capacity");
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

  test_greedy_costs(dir);
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
  // Fixed-point formatting of a finite double can take over 300 characters per row.
  {
    nlohmann::json j;
    j["build"] = KfBudget::build();
    j["alpha"] = j["beta"] = std::vector<float>(9, 1.0f);
    const std::vector<double> costs(9, 1e300);
    j["cost_of"] = "cycle GPU ms";
    j["cost"] = costs;
    std::ofstream(dir + "/kflash/" + key + ".json") << j.dump();
    KfBudget single(key);
    single.set_rows(0xff, 8);
    check(single.summary().ends_with(" (1.00, 1.00)"), "large row costs retain the single budget calibration summary");

    j["cost_of"] = "batched cycle GPU ms";
    j["cost"] = std::vector<std::vector<double>>(KfBatch::kMembers + 1, costs);
    std::ofstream(dir + "/kflash/long-batch.json") << j.dump();
    KfBatch batch("long-batch");
    KfBatch::Member members[2]{};
    const int zeros[2]{};
    batch.observe(members, 2, zeros, zeros, 1e300, nullptr);
    check(batch.summary().ends_with(" (1.00, 1.00)"), "large row costs retain the batch calibration summary");
  }
  std::string rm = "rm -rf '" + dir + "'";
  if (system(rm.c_str()) != 0) fprintf(stderr, "could not remove %s\n", dir.c_str());
  printf("%s\n", fails ? "kflash_cache_test: FAILED" : "kflash_cache_test: all passed");
  return fails ? 1 : 0;
}
