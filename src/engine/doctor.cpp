// `kapitsa doctor`: this Mac (GPU, cores, memory budget, Neural Engine) and, per model, whether and how it
// runs here: memory at the requested context and slots, the Neural Engine split, rough speeds.
#include "engine/doctor.h"
#include "ane/ffn_split.h"
#include "engine/engine.h"
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"

#include <nlohmann/json.hpp>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/sysctl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ni {
namespace {

std::string sysctl_str(const char* name) {
  char b[256] = {0};
  size_t n = sizeof b;
  return sysctlbyname(name, b, &n, nullptr, 0) ? std::string() : std::string(b);
}

uint64_t ram_bytes() {
  uint64_t m = 0;
  size_t n = sizeof m;
  sysctlbyname("hw.memsize", &m, &n, nullptr, 0);
  return m;
}

// Memory bandwidth (GB/s) by chip, from Apple's specifications; 0 = unknown. Max chips by GPU cores.
double bandwidth_gbs(const std::string& chip, int cores) {
  struct B {
    const char* name;
    double gbs;
  };
  static const B table[] = {{"Apple M1 Ultra", 800}, {"Apple M1 Max", 400}, {"Apple M1 Pro", 200}, {"Apple M1", 68},
                            {"Apple M2 Ultra", 800}, {"Apple M2 Max", 400}, {"Apple M2 Pro", 200}, {"Apple M2", 100},
                            {"Apple M3 Ultra", 819}, {"Apple M3 Max", 0},   {"Apple M3 Pro", 150}, {"Apple M3", 100},
                            {"Apple M4 Max", 0},     {"Apple M4 Pro", 273}, {"Apple M4", 120},     {"Apple M5", 153}};
  if (chip == "Apple M3 Max") return cores >= 40 ? 400 : 300;
  if (chip == "Apple M4 Max") return cores >= 40 ? 546 : 410;
  for (const B& b : table)
    if (chip == b.name) return b.gbs;
  return 0;
}

bool is_model_dir(const std::string& d) {
  std::ifstream f(d + "/config.json");
  if (!f) return false;
  try {
    nlohmann::json j = nlohmann::json::parse(f);
    const nlohmann::json t = j.contains("text_config") ? j["text_config"] : j;
    return t.contains("linear_num_value_heads") && t.contains("num_hidden_layers");  // Qwen3.5-family hybrid
  } catch (...) {
    return false;
  }
}

std::vector<std::string> scan(const std::string& root) {
  std::vector<std::string> out;
  if (DIR* d = opendir(root.c_str())) {
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] == '.') continue;
      const std::string p = root + "/" + e->d_name;
      if (is_model_dir(p)) out.push_back(p);
    }
    closedir(d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

const char* yesno(bool b) { return b ? "yes" : "no"; }

}  // namespace

int run_doctor(int argc, char** argv) {
  std::vector<std::string> models, roots;
  int ctx = 65536, slots = 2, chunk = 0;  // chunk 0: as the engine picks it (auto_prefill_chunk)
  bool as_json = false;
  double budget_gb = 0;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") models.push_back(next());
    else if (a == "--models-dir") roots.push_back(next());
    else if (a == "--ctx") ctx = std::stoi(next());
    else if (a == "--slots") slots = std::stoi(next());
    else if (a == "--prefill-chunk") chunk = std::stoi(next());
    else if (a == "--mem-budget") budget_gb = std::stod(next());
    else if (a == "--json") as_json = true;
    else {
      fprintf(stderr,
              "usage: kapitsa doctor [--model DIR]... [--models-dir DIR]... [--ctx 65536] [--slots 2] [--prefill-chunk 2048]\n"
              "                      [--mem-budget GB] [--json]\n");
      return 2;
    }
  }
  if (models.empty()) {
    if (roots.empty()) {
      const char* h = getenv("HOME");
      const std::string home = h ? h : "";
      roots = {home + "/.cache/nace-infer-models", home + "/.cache/nace-agent-models", home + "/.nace/models"};
    }
    for (const std::string& r : roots)
      for (const std::string& m : scan(r)) models.push_back(m);
  }

  Gpu& g = Gpu::get();
  const std::string chip = g.name();
  const int cores = g.core_count();
  const size_t rws = g.recommended_working_set();
  const size_t budget = budget_gb > 0 ? size_t(budget_gb * 1e9) : default_mem_budget();
  const std::string ane_why = ane::unavailable();
  const double bw = bandwidth_gbs(chip, cores);
  nlohmann::ordered_json J;
  J["chip"] = chip;
  J["gpu_cores"] = cores;
  J["gpu_cores_source"] = g.core_count_measured() ? "IORegistry" : "fallback";
  J["ram_bytes"] = ram_bytes();
  J["macos"] = sysctl_str("kern.osproductversion") + " (" + sysctl_str("kern.osversion") + ")";
  J["recommended_working_set_bytes"] = rws;
  J["memory_budget_bytes"] = budget;
  J["max_threadgroup_memory"] = g.max_threadgroup_memory();
  J["apple10"] = g.apple10();
  J["neural_engine"] = ane_why.empty() ? "available" : ane_why;
  J["bandwidth_gbs"] = bw;
  if (!as_json) {
    printf("Kapitsa doctor\n");
    printf("  Mac:     %s, %d-core GPU (%s), %.0f GB RAM, macOS %s\n", chip.c_str(), cores,
           g.core_count_measured() ? "IORegistry" : "not measured: the tier's smallest", ram_bytes() / 1073741824.0,
           J["macos"].get<std::string>().c_str());
    printf("  Memory:  Metal recommended working set %.1f GB; Kapitsa's budget %.1f GB%s\n", rws / 1e9, budget / 1e9,
           budget_gb > 0 ? " (--mem-budget)" : getenv("NI_MEM_BUDGET") ? " (NI_MEM_BUDGET)" : " (4% / 512 MB below it)");
    printf("  GPU:     threadgroup memory %zu B, Metal 4 neural accelerators (Apple10): %s%s\n", g.max_threadgroup_memory(),
           yesno(g.apple10()), bw > 0 ? ("; memory bandwidth " + std::to_string(int(bw)) + " GB/s").c_str() : "");
    printf("  Neural Engine: %s\n", ane_why.empty() ? "available" : ("unavailable (" + ane_why + ")").c_str());
    printf("\nModels (--ctx %d, --slots %d%s):\n", ctx, slots,
           chunk ? (", --prefill-chunk " + std::to_string(chunk)).c_str() : "");
    if (models.empty()) printf("  none found (--model DIR)\n");
  }
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();
  for (const std::string& dir : models) {
    Qwen35::MemPlan m;
    const int ch = chunk > 0 ? chunk : auto_prefill_chunk(dir, ctx, slots, budget);
    try {
      m = Qwen35::plan(dir, ch);
    } catch (const std::exception& e) {
      if (!as_json) printf("  %s: cannot read (%s)\n", dir.c_str(), e.what());
      continue;
    }
    // Neural Engine split (dense FFN): calibration of this Mac or the chip's default share, as the engine decides
    const Qwen35::AnePlan ap = Qwen35::ane_plan(dir, m);
    const uint32_t units = ap.units, total = ap.total;
    const size_t ane_bytes = ap.bytes;
    const std::string src = ap.source;
    // a DFlash2 drafter next to the model or in the drafter cache (the engine's default when it finds one)
    std::string dfl;
    size_t dfl_bytes = 0;
    try {
      dfl = DFlash2::find(dir, Qwen35::read_config(dir));
    } catch (...) {
    }
    if (!dfl.empty()) {
      size_t per = 0;
      dfl_bytes = DFlash2::plan_bytes(dfl, &per) + 3 * per * size_t(slots);  // live ring + one per checkpoint
    }
    const size_t fixed = m.fixed(slots) + ane_bytes;
    const size_t start_need = fixed + m.kv_per_pos * size_t(std::min(ctx, 8192));
    const size_t full_need = fixed + m.kv_per_pos * size_t(ctx) * size_t(slots);
    const size_t room = budget > fixed ? budget - fixed : 0;
    const size_t tokens = room / std::max<size_t>(m.kv_per_pos, 1);
    const double dec = bw > 0 && m.decode_bytes ? 0.75 * bw * 1e9 / double(m.decode_bytes) : 0;
    // prefill: ~0.33 TFLOP/s of fp16 MMA per GPU core (M4 Pro: 6-7 TFLOP/s on 20), 2 flops per weight
    const double params = double(m.decode_bytes) / 0.5625;
    const double pf = 0.33e12 * cores / (2 * params);
    nlohmann::ordered_json o;
    o["dir"] = dir;
    o["dense"] = m.dense;
    o["weights_bytes"] = m.weights;
    o["mtp_bytes"] = m.mtp;
    o["prefill_chunk"] = ch;
    o["activation_bytes"] = m.activations;
    o["slot_state_bytes"] = m.slot_state;
    o["kv_bytes_per_token"] = m.kv_per_pos;
    o["ane_units"] = units;
    o["ane_units_total"] = total;
    o["ane_bytes"] = ane_bytes;
    o["starts"] = start_need <= budget;
    o["fits_full_context"] = full_need <= budget;
    o["kv_tokens_in_budget"] = tokens;
    o["dflash_drafter"] = dfl;
    o["dflash_bytes"] = dfl_bytes;
    o["kv_tokens_in_budget_with_dflash"] = budget > fixed + dfl_bytes ? (budget - fixed - dfl_bytes) / std::max<size_t>(m.kv_per_pos, 1) : 0;
    o["decode_tps_estimate"] = std::round(dec);
    o["prefill_tps_estimate_gpu"] = std::round(pf);
    arr.push_back(o);
    if (as_json) continue;
    std::string name = dir.substr(dir.find_last_of('/') + 1);
    printf("  %s (%s, %d layers, hidden %d)\n", name.c_str(), m.dense ? "dense" : "MoE", m.layers, m.hidden);
    printf("    memory: weights %.2f GB, MTP head %.2f, %d-row prefill buffers %.2f, %d slot%s' state %.2f, KV %.0f KB/token%s\n",
           m.weights / 1e9, m.mtp / 1e9, ch, m.activations / 1e9, slots, slots > 1 ? "s" : "",
           slots * m.slot_state / 1e9, m.kv_per_pos / 1024.0,
           ane_bytes ? (", Neural Engine split " + std::to_string(int(ane_bytes / 1e6)) + " MB").c_str() : "");
    if (start_need > budget)
      printf("    fits: NO: needs %.1f GB with an 8k-token conversation, the budget is %.1f GB\n", start_need / 1e9,
             budget / 1e9);
    else
      printf("    fits: yes; %d x %d tokens %s (%.1f GB); room for %zu tokens of KV cache in all%s\n", slots, ctx,
             full_need <= budget ? "fit" : "do not all fit at once", full_need / 1e9, tokens,
             full_need <= budget ? "" : " (the KV cache grows with each conversation; idle slots give theirs up)");
    if (!dfl.empty())
      printf("    DFlash2 drafter (used by default): %s, %.2f GB with %d slot%s; with it room for %zu tokens of KV cache\n",
             dfl.c_str(), dfl_bytes / 1e9, slots, slots > 1 ? "s" : "",
             budget > fixed + dfl_bytes ? (budget - fixed - dfl_bytes) / std::max<size_t>(m.kv_per_pos, 1) : size_t(0));
    if (m.dense)
      printf("    prefill: Neural Engine split %s\n",
             units ? (std::to_string(units) + "/" + std::to_string(total) + " units (" + src + ")").c_str()
                   : ("off: " + src).c_str());
    if (dec > 0)
      printf("    rough speed: plain decode ~%.0f tok/s (%.2f GB read per token), GPU prefill ~%.0f tok/s; MTP / Neural "
             "Engine add their measured factors (DEVELOPMENT.md)\n",
             dec, m.decode_bytes / 1e9, pf);
  }
  if (as_json) {
    J["context"] = ctx;
    J["slots"] = slots;
    J["models"] = arr;
    printf("%s\n", J.dump(2).c_str());
  }
  return 0;
}

}  // namespace ni
