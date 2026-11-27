// The drafter's optional load-time allocations, without a model or GPU. --gpu MODEL additionally compares the
// real loaded drafter's Metal allocation delta against the planner; use the project GPU lock for that mode.
#include "model/dflash2.h"
#include "bench/bench.h"
#include "util/log.h"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace ni;

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--gpu") {
    bench::GpuLock lock;
    if (!bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 0)) fatal("GPU lock busy");
    Qwen35 target(argv[2], 2048);
    const auto dir = DFlash2::find(argv[2], target.config());
    if (dir.empty()) fatal("no drafter");
    size_t prefill = 0;
    const auto plan = DFlash2::plan_bytes(dir, target.config(), Qwen35::plan(argv[2], 2048).head_row_bytes,
                                        nullptr, &prefill, DFlash2::kB, DFlash2::lanes_default(argv[2], target.config()));
    auto& g = Gpu::get();
    const auto before = g.allocated_bytes();
    DFlash2 drafter(target, dir);
    const auto actual = g.allocated_bytes() - before, planned = plan - prefill - (size_t(4) << 20);
    // Metal rounds each buffer to pages. The existing plan also has small legacy scratch approximations.
    const size_t allowance = 8u << 20;
    printf("DRAFT_PLAN={\"allocated\":%zu,\"planned\":%zu,\"page_allowance\":%zu,\"passed\":%s}\n",
           actual, planned, allowance, actual <= planned + allowance ? "true" : "false");
    return actual <= planned + allowance ? 0 : 1;
  }
  if (argc != 1) { fprintf(stderr, "usage: dflash_plan_test [--gpu MODEL]\n"); return 2; }
  char tmp[] = "/tmp/kapitsa-draft-plan-XXXXXX";
  if (!mkdtemp(tmp)) return 2;
  // All device-dependent defaults are overridden before calling the planner: no Metal device is created.
  setenv("NI_KFLASH_V8", "1", 1);
  setenv("NI_V8_PACKED_COEFF_NF1", "0", 1);
  setenv("NI_KFLASH_HEAD", "98304", 1);
  setenv("NI_KFLASH_BITS", "4", 1);
  setenv("NI_KFLASH_LOWRANK", "0", 1);
  Qwen35Config target;
  target.hidden = 2560; target.dense_I = 9216; target.layers = 32; target.group_size = 64;
  nlohmann::json config{{"hidden_size", 2560}, {"intermediate_size", 9216}, {"num_hidden_layers", 6},
                        {"vocab_size", 248320}, {"architectures", {"DFlashDraftModel"}}};
  auto save = [&] { std::ofstream(std::string(tmp) + "/config.json") << config; };
  save();
  auto plan = [&](int rows = 8) { return DFlash2::plan_bytes(tmp, target, 1440, nullptr, nullptr, rows); };
  int fails = 0;
  auto check = [&](bool ok, const char* why) { if (!ok) { ++fails; printf("FAIL %s\n", why); } };
  const auto plain = plan();
  setenv("NI_KFLASH_LOWRANK", "768", 1);
  check(plan() == plain + 44016000, "rank768 reserves both Q4 factors, permutation and three scratch buffers");
  for (const char* bad : {"-1", "0", "63", "2560", "4096"}) {
    setenv("NI_KFLASH_LOWRANK", bad, 1);
    check(plan() == plain, "ineligible rank reserves no factors");
  }
  setenv("NI_KFLASH_LOWRANK", "0", 1);
  const auto wide = plan(16);
  setenv("NI_KFLASH_LOWRANK", "768", 1);
  check(plan(16) == wide, "16-row drafter has no v8 low-rank path");
  setenv("NI_KFLASH_BITS", "8", 1);
  setenv("NI_KFLASH_LOWRANK", "0", 1);
  const auto eight = plan();
  setenv("NI_KFLASH_LOWRANK", "768", 1);
  check(plan() == eight + 82747776, "8-bit drafter factors reserve their own format");
  setenv("NI_KFLASH_HEAD", "0", 1);
  const auto full = plan();
  setenv("NI_KFLASH_LOWRANK", "0", 1);
  check(plan() == full, "full head has no low-rank factors");
  // The online selector update's fp32 masters of the two codebooks (2 x vocab x rank 256 x 4 bytes = 485 MB at 248320
  // ids) are in the plan when the update runs: by NI_KFLASH_SELUPD, by default with lanes, and not when it is 0 (#362)
  setenv("NI_KFLASH_LOWRANK", "0", 1);
  setenv("NI_KFLASH_BITS", "4", 1);
  setenv("NI_KFLASH_HEAD", "98304", 1);
  config["architectures"] = {"DFlash2DraftModel"};
  save();
  const size_t masters = size_t(2) * 248320 * 256 * 4;
  unsetenv("NI_KFLASH_SELUPD");
  const auto no_lanes = plan();
  const auto with_lanes = DFlash2::plan_bytes(tmp, target, 1440, nullptr, nullptr, 8, true);
  check(with_lanes - no_lanes == masters, "lanes default: the plan takes the 2 fp32 codebook masters (485 MB)");
  setenv("NI_KFLASH_SELUPD", "0", 1);
  check(DFlash2::plan_bytes(tmp, target, 1440, nullptr, nullptr, 8, true) == no_lanes, "NI_KFLASH_SELUPD=0: no masters in the plan");
  setenv("NI_KFLASH_SELUPD", "0.003", 1);
  check(plan() - no_lanes == masters, "NI_KFLASH_SELUPD=<lr> without lanes: the masters are in the plan");
  setenv("NI_KFLASH_SELUPD", "0.003", 1);
  setenv("NI_KFLASH_TREE", "0", 1);
  check(plan() == no_lanes, "NI_KFLASH_TREE=0: the update cannot run, no masters in the plan");
  unsetenv("NI_KFLASH_TREE");
  setenv("NI_KFLASH_VERIFY", "3", 1);
  check(plan() == no_lanes, "NI_KFLASH_VERIFY=3: the verify chains, the update cannot run, no masters in the plan");
  setenv("NI_KFLASH_VERIFY", "7", 1);
  check(plan() - no_lanes == masters, "NI_KFLASH_VERIFY=7: the full-depth trees run, the masters are in the plan");
  unsetenv("NI_KFLASH_VERIFY");
  unsetenv("NI_KFLASH_SELUPD");
  unsetenv("NI_KFLASH_SELUPD");
  std::filesystem::remove_all(tmp);
  printf("dflash_plan_test: %s\n", fails ? "FAILED" : "ALL PASSED");
  return fails ? 1 : 0;
}
