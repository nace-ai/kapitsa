// Dump a real DFlash2 pass; --from compares every saved byte and returns failure on any mismatch.
// qmm_draft_check MODEL PROMPT_IDS NEW_OUTPUT [--from REFERENCE_OUTPUT]
// Runs under the GPU lock. These are correctness runs, never performance measurements.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;
using json = nlohmann::json;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if ((argc != 4 && argc != 6) || (argc == 6 && std::string(argv[4]) != "--from")) {
    fprintf(stderr, "usage: qmm_draft_check MODEL PROMPT_IDS NEW_OUTPUT [--from REFERENCE_OUTPUT]\n");
    return 2;
  }
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const fs::path out = argv[3], ref = argc == 6 ? argv[5] : "";
  std::ifstream input(argv[2]);
  std::string raw{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  std::replace(raw.begin(), raw.end(), ',', ' ');
  std::istringstream tokens(raw);
  std::vector<int> ids;
  for (int t; tokens >> t;) ids.push_back(t);
  if (ids.empty() || ids.size() > 8192 || !tokens.eof()) fatal("invalid prompt ids");
  if (fs::exists(out) || !fs::create_directories(out)) fatal("output must be a new directory");
  json reference;
  if (!ref.empty()) {
    std::ifstream f(ref / "meta.json");
    if (!f) fatal("missing reference manifest");
    reference = json::parse(f);
    if (reference.at("prompt_ids") != ids) fatal("reference prompt differs");
  }
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 0)) fatal("GPU lock busy");
  Gpu& g = Gpu::get();
  Qwen35 target(argv[1], 2048);
  const std::string drafter = DFlash2::find(argv[1], target.config());
  if (drafter.empty()) fatal("no local DFlash2 drafter");
  DFlash2 draft(target, drafter, 4);
  if (draft.block() != 8) fatal("check requires an 8-row drafting block");
  const int n = int(ids.size());
  for (int t : ids) if (t < 0 || t >= target.config().vocab) fatal("prompt token outside vocabulary");
  auto seq = target.new_seq(n + 64);
  auto ds = draft.new_seq();
  std::copy(ids.begin(), ids.end(), seq->tokens());
  SamplingParams sp;
  sp.temperature = 0;
  sp.seed = 42;
  draft.begin_prefill(n);
  for (int pos = 0; pos < n; pos += target.prefill_chunk()) {
    const int count = std::min(target.prefill_chunk(), n - pos);
    auto e = g.begin();
    target.encode_prefill(*e, *seq, pos, count, sp, 0, pos + count == n);
    e->commit_and_wait();
    if (!target.ane_finish_chunk()) fatal("prefill failed");
  }
  draft.end_prefill();
  {
    auto e = g.begin();
    draft.encode_prefill_commit(*e, *ds);
    e->commit_and_wait();
  }
  const std::vector<int> anchors{seq->tokens()[n], ids.back(), ids[ids.size() / 2]};
  json meta{{"prompt_ids", ids}, {"anchors", anchors}, {"drafter", drafter}, {"gpu", g.name()},
            {"drafter_v1", draft.config().v1}, {"head_rows", draft.head_rows()}, {"files", json::object()}};
  if (!ref.empty() && (reference.at("anchors") != anchors || reference.at("head_rows") != draft.head_rows() ||
                       reference.at("drafter") != drafter)) fatal("reference input or drafter differs");
  size_t checks = 0, mismatches = 0, bytes = 0;
  auto save = [&](const std::string& name, const void* data, size_t size) {
    if (!size) fatal("empty result: %s", name.c_str());
    std::ofstream f(out / (name + ".bin"), std::ios::binary);
    if (!f.write(static_cast<const char*>(data), std::streamsize(size))) fatal("cannot save result");
    meta["files"][name] = size;
    bytes += size;
    if (ref.empty()) return;
    std::ifstream rf(ref / (name + ".bin"), std::ios::binary);
    const std::vector<char> expected{std::istreambuf_iterator<char>(rf), std::istreambuf_iterator<char>()};
    ++checks;
    if (reference.at("files").value(name, size_t(0)) != size || expected.size() != size ||
        std::memcmp(data, expected.data(), size) != 0) {
      ++mismatches;
      fprintf(stderr, "DIFFERENT: %s (%zu bytes)\n", name.c_str(), size);
    }
  };
  // Normal passes exercise fused attention. Debug passes additionally expose all per-layer intermediates;
  // the debug flag uses the existing unfused attention path, identically for baseline and candidate.
  for (int debug = 0; debug < 2; ++debug) {
    draft.set_debug(debug != 0);
    for (size_t a = 0; a < anchors.size(); ++a) {
      seq->tokens()[n] = anchors[a];
      auto e = g.begin();
      draft.encode_draft(*e, *seq, *ds, n, sp);
      e->commit_and_wait();
      const std::string prefix = "debug" + std::to_string(debug) + "-anchor" + std::to_string(a) + "-";
      // Logit row 0 is not a proposal. The seven active rows cover the complete reduced vocabulary.
      save(prefix + "logits", draft.logits() + draft.head_rows(), size_t(7) * draft.head_rows() * 4);
      save(prefix + "hidden", draft.final_hidden(), size_t(8) * draft.config().H * 4);
      if (!draft.config().v1) save(prefix + "selector", draft.sel_hidden(), size_t(8) * DFlash2::kRank * 4);
      save(prefix + "candidate_ids", draft.cand_ids(), size_t(7) * 16 * 4);
      save(prefix + "candidate_values", draft.cand_vals(), size_t(7) * 16 * 4);
      save(prefix + "chain", seq->tokens() + n + 1, 7 * sizeof(int));
      if (debug) {
        for (const auto& [name, buf] : draft.debug_bufs()) {
          // v1 has no dynamic convolution. The legacy debug hooks still copy that unused scratch.
          if (draft.config().v1 && (name.starts_with("D_") || name.starts_with("D2_"))) continue;
          save(prefix + name, buf.ptr(), buf.size);
        }
      }
    }
  }
  if (!ref.empty() && reference.at("files").size() != meta["files"].size()) ++mismatches;
  meta["compared_files"] = checks;
  meta["compared_bytes"] = ref.empty() ? 0 : bytes;
  meta["mismatches"] = mismatches;
  std::ofstream manifest(out / "meta.json");
  if (!(manifest << meta.dump(2) << '\n')) fatal("cannot save manifest");
  if (!ref.empty())
    printf("AKO_CHECK={\"passed\":%s,\"cases\":%zu,\"bytes\":%zu,\"mismatches\":%zu}\n",
           !mismatches && checks ? "true" : "false", checks, bytes, mismatches);
  else
    printf("Saved %zu files / %zu bytes to %s\n", meta["files"].size(), bytes, out.c_str());
  return mismatches || (!ref.empty() && !checks) ? 1 : 0;
}
