// `kapitsa doctor`: this Mac (GPU, cores, memory budget, Neural Engine) and, per model, whether and how it
// runs here: memory at the requested context and slots, the Neural Engine split, rough speeds.
#include "engine/doctor.h"
#include "vision/encoder.h"
#include "ane/ffn_split.h"
#include "engine/engine.h"
#include "hub/hub.h"
#include "metal/gpu.h"
#include "model/dory.h"
#include "model/nemotron_h.h"
#include "model/nh_drafter.h"
#include "model/qwen35.h"
#include "util/term.h"

#include <nlohmann/json.hpp>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace ni {
namespace {

// z-lab's DFlash (v1) drafters run on the DFlash2 path; the label says which one this is
bool dflash_v1(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  const nlohmann::json c = nlohmann::json::parse(f, nullptr, false);
  return c.is_object() && c.value("architectures", nlohmann::json::array()) == nlohmann::json::array({"DFlashDraftModel"});
}

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

// ---- the styled view (a terminal): the same plan as the plain lines, in two tables

// One model's plan, for the table
struct Row {
  std::string name, where, error, decoding, formats, image;
  std::string note;  // a model's own line (Dory: its loops and the KV of each mode)
  bool planned = false, dense = false, starts = false, all_fit = false;
  int layers = 0, slots = 0;
  size_t full = 0, start = 0, weights = 0, kv = 0, drafter = 0, tokens = 0, image_bytes = 0;
  int default_ctx = 0, ctx_limit = 0;  // the context kapitsa serve takes by default here; the model's own limit
  double dec = 0, pf = 0;
};
// One catalog variant: fits (1), too big (0, `need`), not planned (-1); the one --model <name> takes
struct Variant {
  std::string model, alias, name, need;
  int fits = 0;
  bool chosen = false, starts = false;
};

std::string gb(double b) {
  char s[32];
  snprintf(s, sizeof s, "%.1f GB", b / 1e9);
  return s;
}

void print_tty(const nlohmann::ordered_json& J, size_t budget, const std::string& budget_note, int ctx, int slots,
               const std::vector<Row>& rows, const std::vector<Variant>& vars) {
  using namespace term;
  const int W = std::min(columns(), 100);
  std::vector<std::string> o;
  const std::string dot = std::string("  ") + kDot + "  ";
  o.push_back("");
  o.push_back("  " + bold("This Mac"));
  std::string mac = J["chip"].get<std::string>() + dot + std::to_string(J["gpu_cores"].get<int>()) + "-core GPU";
  if (J["bandwidth_gbs"].get<double>() > 0) mac += dot + std::to_string(int(J["bandwidth_gbs"].get<double>())) + " GB/s";
  char ram[32];
  snprintf(ram, sizeof ram, "%.0f GB RAM", J["ram_bytes"].get<double>() / 1073741824.0);
  std::string macos = J["macos"].get<std::string>();
  if (const size_t p = macos.find(" ("); p != std::string::npos) macos.resize(p);
  mac += dot + ram + dot + "macOS " + macos;
  o.push_back("    " + clip(mac, W - 4));
  o.push_back("    " + bold(gb(double(budget))) + " of GPU memory for Kapitsa" +
              dim(clip("  (" + (budget_note.empty() ? "" : budget_note + "; ") + "Metal recommends " +
                           gb(J["recommended_working_set_bytes"].get<double>()) + ")",
                       W - 42)));
  const std::string ane = J["neural_engine"].get<std::string>();
  o.push_back("    " + std::string(ane == "available" ? accent(kOk) + " Neural Engine" : dim(std::string(kFail) + " Neural Engine: " + ane)) +
              dot + (J["apple10"].get<bool>() ? accent(kOk) + " Metal 4 neural accelerators" : dim("no Metal 4 neural accelerators")));
  o.push_back("");

  // the models: fits, memory at the full context, decoding; then a dim line of details
  {
    const std::string title = "  " + bold("Models") + dim("  --context " + context(ctx) + "  --slots " +
                                                          (slots > 0 ? std::to_string(slots) : std::string("auto")));
    const std::string legend = accent(kOk) + dim(" fits  ") + warn("◐") + dim(" starts  ") + red(kFail) + dim(" too big");
    const int gap = std::min(W, 80) - width(title) - width(legend);
    o.push_back(gap >= 3 ? title + std::string(size_t(gap), ' ') + legend : title);
    int nw = 0;
    for (const auto& r : rows) nw = std::max(nw, std::min(30, width(r.name) + (r.planned ? 7 : 0)));
    for (const auto& r : rows) {
      std::string name = r.name + (r.planned ? dim(" (plan)") : "");
      if (!r.error.empty()) {
        o.push_back("    " + red(kFail) + " " + left(clip(name, 30), nw) + "   " + dim("cannot read: " + r.error));
        continue;
      }
      const std::string mark = r.all_fit ? accent(kOk) : r.starts ? warn("◐") : red(kFail);
      const std::string mem = r.starts ? right(gb(double(r.full)), 7) + dim(" of " + gb(double(budget)))
                                       : red("needs " + gb(double(r.start)));
      std::string l = "    " + mark + " " + left(clip(name, 30), nw) + "   " + left(dim(r.dense ? "dense" : "MoE"), 5) + "   " +
                      left(mem, 18) + "   " + r.decoding;
      o.push_back(clip(l, W));
      const std::string sep = std::string(" ") + kDot + " ";
      std::string d = r.where.empty() ? "" : r.where + sep;
      d += std::to_string(r.slots) + " slots" + sep + "weights " + gb(double(r.weights));
      char kv[48];
      snprintf(kv, sizeof kv, "KV %.0f KB/token", r.kv / 1024.0);
      d += sep + kv;
      if (r.starts) d += sep + "room for " + count(double(r.tokens)) + " tokens";
      if (r.default_ctx > 0)
        d += sep + "serve's context " + context(r.default_ctx) + (r.ctx_limit > r.default_ctx ? " (max " + context(r.ctx_limit) + ")" : "");
      if (r.image_bytes) d += sep + "images +" + gb(double(r.image_bytes));
      o.push_back("      " + dim(clip(d, std::max(W, std::min(columns(), 140)) - 8)));
      if (!r.note.empty()) o.push_back("      " + dim(clip(r.note, std::max(W, std::min(columns(), 140)) - 8)));
    }
    if (rows.empty()) o.push_back("    " + dim("none found (--model DIR)"));
    o.push_back("");
  }

  // every catalog variant on this Mac
  {
    const std::string title = "  " + bold("Catalog variants");
    const std::string legend = dim("← what --model <name> takes");
    const int gap = std::min(W, 80) - width(title) - width(legend);
    o.push_back(gap >= 3 ? title + std::string(size_t(gap), ' ') + legend : title);
    // columns: each model's default first, then the other precisions by name, aligned across models
    std::vector<std::string> models, names;
    for (const auto& v : vars) {
      if (std::find(models.begin(), models.end(), v.model) == models.end()) models.push_back(v.model);
      const hub::CatalogModel* c = hub::find_model(v.model);
      const bool dflt = c && c->default_variant == v.name;
      if (!dflt && std::find(names.begin(), names.end(), v.name) == names.end()) names.push_back(v.name);
    }
    std::vector<std::vector<std::string>> grid(models.size(), std::vector<std::string>(names.size() + 1));
    std::vector<int> cw(names.size() + 1, 0);
    std::vector<std::string> aliases(models.size());
    int aw = 0;
    for (const auto& v : vars) {
      const size_t mi = size_t(std::find(models.begin(), models.end(), v.model) - models.begin());
      const hub::CatalogModel* c = hub::find_model(v.model);
      const bool dflt = c && c->default_variant == v.name;
      const size_t k = dflt ? 0 : size_t(std::find(names.begin(), names.end(), v.name) - names.begin()) + 1;
      std::string cell = v.name + " " +
                         (v.fits == 1 ? accent(kOk) : v.fits < 0 ? dim("?") : v.starts ? warn("◐") : red(kFail) + dim(" " + v.need)) +
                         (v.chosen ? accent(" ←") : "");
      grid[mi][k] = cell;
      cw[k] = std::max(cw[k], width(cell));
      aliases[mi] = v.alias;
      aw = std::max(aw, width(v.alias));
    }
    for (size_t mi = 0; mi < models.size(); ++mi) {
      std::string l = "    " + left(bold(aliases[mi]), aw) + "   ";
      const int start = width(l);
      for (size_t k = 0; k < grid[mi].size(); ++k) {
        if (cw[k] == 0) continue;
        if (width(l) + cw[k] > W && width(l) > start) {
          while (!l.empty() && l.back() == ' ') l.pop_back();
          o.push_back(l), l = std::string(size_t(start), ' ');
        }
        l += left(grid[mi][k], cw[k]) + "   ";
      }
      while (!l.empty() && l.back() == ' ') l.pop_back();
      o.push_back(l);
    }
    o.push_back("    " + accent(kOk) + dim(" fits with its drafter   ") + warn("◐") + dim(" starts, without   ") + red(kFail) +
                dim(" too big   ? offline"));
  }
  o.push_back("");
  std::string out;
  for (const auto& x : o) out += x + "\n";
  fputs(out.c_str(), stdout);
}

}  // namespace

int run_doctor(int argc, char** argv) {
  std::vector<std::string> models, roots;
  int ctx = 65536, slots_opt = 0, chunk = 0;  // slots 0: per model as the engine picks them (auto_slots); chunk 0: as
                                               // the engine picks it (auto_prefill_chunk)
  std::string dflash;                      // as the server's --dflash ("": NI_DFLASH or auto)
  bool as_json = false;
  double budget_gb = 0;
  std::vector<Row> tty_rows;  // the styled view (a terminal)
  std::vector<Variant> tty_vars;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") models.push_back(next());
    else if (a == "--models-dir") roots.push_back(next());
    else if (a == "--context" || a == "--ctx") {
      std::string why;
      ctx = parse_context(next(), &why);
      if (ctx == kContextMax) why = "--context max has no one size here (every model is planned at one context): give it in tokens (--context 256k)";
      if (ctx <= 0) {
        if (term::fancy()) term::error(why, "kapitsa doctor --help shows how to use it.");
        else fprintf(stderr, "kapitsa doctor: %s\n", why.c_str());
        return 2;
      }
    }
    else if (a == "--slots") slots_opt = std::stoi(next());
    else if (a == "--prefill-chunk") chunk = std::stoi(next());
    else if (a == "--mem-budget") budget_gb = std::stod(next());
    else if (a == "--dflash") dflash = next();
    else if (a == "--json") as_json = true;
    else {
      const char* usage =
          "usage: kapitsa doctor [--model MODEL]... [--models-dir DIR]... [--context 64k] [--slots N] [--prefill-chunk 2048]\n"
          "                      [--dflash auto|none|DIR] [--mem-budget GB] [--json]\n";
      if (!term::fancy()) fputs(usage, stderr);
      else if (a == "--help" || a == "-h")
        term::help("kapitsa doctor [--model MODEL]... [--models-dir DIR]... [--context 64k] [--slots N] [--prefill-chunk 2048] "
                   "[--dflash auto|none|DIR] [--mem-budget GB] [--json]: This Mac's GPU and memory, and which models fit "
                   "it at the context and slots given (what kapitsa serve would hold; without --slots, the slots each "
                   "model would get), with the context kapitsa serve takes by default here.\n");
      else
        term::error("unknown option " + a, "kapitsa doctor --help shows how to use it.");
      return 2;
    }
  }
  // in a terminal: a spinner while the models are planned (a few seconds)
  std::unique_ptr<term::Spinner> spin;
  if (!as_json && term::fancy()) spin = std::make_unique<term::Spinner>("Planning the models on this Mac");
  // --model takes what `kapitsa serve --model` takes: a name or alias, org/name, or a folder; a catalog model that is
  // not downloaded is planned from its headers (hub::plan_dir, fetched once)
  for (std::string& m : models) {
    try {
      const hub::ModelRef r = hub::resolve(m);
      const std::string d = hub::plan_dir(r);
      m = d.empty() ? r.dir : d;
    } catch (const hub::Error&) {
    }
  }
  if (models.empty()) {
    if (roots.empty()) {
      // the catalog's models first (downloaded or not: what fits this Mac before a download), then the store and the
      // folders other tools keep models in
      // each model's variant for this Mac: the default where it fits, else a fallback (pick_variant)
      EngineOptions eo;
      eo.context = ctx, eo.slots = slots_opt, eo.prefill_chunk = chunk, eo.dflash = dflash;
      eo.mem_budget_gb = budget_gb;
      const size_t b = budget_gb > 0 ? size_t(budget_gb * 1e9) : default_mem_budget();
      for (const auto& c : hub::catalog())
        if (const std::string d = hub::plan_dir(pick_variant(c, eo, b)); !d.empty()) models.push_back(d);
      const char* h = getenv("HOME");
      const std::string home = h ? h : "";
      roots = {hub::models_dir(), home + "/.cache/nace-infer-models", home + "/.cache/nace-agent-models",
               home + "/.nace/models"};
    }
    for (const std::string& r : roots)
      for (const std::string& m : scan(r))
        if (std::find(models.begin(), models.end(), m) == models.end()) models.push_back(m);
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
  const bool tty = !as_json && term::fancy();
  if (!as_json && !tty) {
    printf("Kapitsa doctor\n");
    printf("  Mac:     %s, %d-core GPU (%s), %.0f GB RAM, macOS %s\n", chip.c_str(), cores,
           g.core_count_measured() ? "IORegistry" : "not measured: the tier's smallest", ram_bytes() / 1073741824.0,
           J["macos"].get<std::string>().c_str());
    printf("  Memory:  Metal recommended working set %.1f GB; Kapitsa's budget %.1f GB%s\n", rws / 1e9, budget / 1e9,
           budget_gb > 0 ? " (--mem-budget)" : getenv("NI_MEM_BUDGET") ? " (NI_MEM_BUDGET)" : " (4% / 512 MB below it)");
    printf("  GPU:     threadgroup memory %zu B, Metal 4 neural accelerators (Apple10): %s%s\n", g.max_threadgroup_memory(),
           yesno(g.apple10()), bw > 0 ? ("; memory bandwidth " + std::to_string(int(bw)) + " GB/s").c_str() : "");
    printf("  Neural Engine: %s\n", ane_why.empty() ? "available" : ("unavailable (" + ane_why + ")").c_str());
    printf("\nModels (--context %s, --slots %s%s):\n", term::context(ctx).c_str(),
           slots_opt > 0 ? std::to_string(slots_opt).c_str() : "as each model's memory allows",
           chunk ? (", --prefill-chunk " + std::to_string(chunk)).c_str() : "");
    if (models.empty()) printf("  none found (--model DIR)\n");
  }
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();
  for (const std::string& dir : models) {
    // the engine's own decisions before it loads: the drafter (auto leaves it out where it costs context) and the
    // prefill chunk
    MemPlan m;
    LoadPlan lp;
    const bool nemotron = NemotronH::is_model(dir);
    const bool dory = Dory::is_model(dir);
    int slots = slots_opt;  // as the engine would start it: the ones asked for, else auto_slots
    int default_ctx = 0;    // what kapitsa serve takes here without --context
    try {
      {
        EngineOptions dco;
        dco.slots = slots_opt, dco.prefill_chunk = chunk, dco.dflash = dflash;
        default_ctx = default_context(dir, dco, budget);
      }
      if (slots <= 0) {
        EngineOptions so;
        so.context = ctx, so.prefill_chunk = chunk, so.dflash = dflash;
        slots = auto_slots(dir, so, budget);
      }
      lp = plan_load(dir, dflash, false, ctx, slots, budget, chunk);
      m = model_plan(dir, lp.prefill_chunk);
    } catch (const std::exception& e) {
      if (tty) tty_rows.push_back(Row{.name = dir.substr(dir.find_last_of('/') + 1), .error = e.what()});
      else if (!as_json) printf("  %s: cannot read (%s)\n", dir.c_str(), e.what());
      continue;
    }
    // Nemotron-H's block drafters: LiLiCorr (Bolt, Super-VL) or DSpark (Lightning)
    const bool dspark = nemotron && NhDrafter::kind(lp.dflash.empty() ? lp.dflash_left_out : lp.dflash) == "dspark";
    const std::string drafter_dir = lp.dflash.empty() ? lp.dflash_left_out : lp.dflash;
    const char* drafter = nemotron ? (dspark ? "DSpark drafter" : "LiLiCorr drafter")
                          : !drafter_dir.empty() && dflash_v1(drafter_dir) ? "DFlash drafter" : "DFlash2 drafter";
    const int ch = lp.prefill_chunk;
    // a dense Nemotron-H model (Nemotron Nano) without a drafter: prompt lookup, as the engine decides it (by default
    // with BF16 weights, not NVFP4; NI_NH_NGRAM sets the drafts per cycle)
    int lookup = 0;
    if (nemotron && m.dense && lp.dflash.empty()) {
      const char* v = getenv("NI_NH_NGRAM");
      lookup = std::min(v && *v ? atoi(v) : m.formats.find("NVFP4") == std::string::npos ? 4 : 0, NemotronH::kMaxRows - 1);
    }
    // Neural Engine split: the Qwen dense FFN's (calibration of this Mac or the chip's default share, as the engine
    // decides); Nemotron-H's Mamba projections and shared expert (nemotron_ane.h: the bytes its plan holds, as
    // conversation_fits counts them); none for Dory
    const Qwen35::AnePlan ap = dory       ? Qwen35::AnePlan{0, 0, "not used for this architecture", 0}
                               : nemotron ? Qwen35::AnePlan{0, 0, m.ane ? "Mamba projections and shared expert" : "off", m.ane}
                                          : Qwen35::ane_plan(dir, m);
    // Dory: the loops it runs and the KV per token of each mode (the full-attention layers'; the sliding-window
    // layers keep rings of window + 8 positions per conversation)
    std::string dory_note;
    nlohmann::ordered_json dory_j;
    if (dory) {
      const DoryConfig dc = Dory::read_config(dir);
      const int loops = Dory::serving_loops() > 0 ? Dory::serving_loops() : dc.loops;
      char b[400];
      snprintf(b, sizeof b, "loops: %d (requests take 1-%d) %s KV: %s; per token: last-loop %.0f KB, per-loop %.0f KB "
               "(%d loops) / %.0f KB (4); sliding-window rings %.0f MB per conversation",
               loops, Dory::kMaxLoops, term::kDot, loop_kv_name(Dory::serving_kv()),
               Dory::kv_per_pos(dc, LoopKv::last_loop, loops) / 1024.0,
               Dory::kv_per_pos(dc, LoopKv::per_loop, loops) / 1024.0, loops,
               Dory::kv_per_pos(dc, LoopKv::per_loop, 4) / 1024.0, Dory::ring_bytes(dc, Dory::serving_kv(), loops) / 1e6);
      dory_note = b;
      dory_j = {{"loops", loops},
                {"loop_kv", loop_kv_name(Dory::serving_kv())},
                {"kv_bytes_per_token", {{"last_loop", Dory::kv_per_pos(dc, LoopKv::last_loop, loops)},
                                        {"per_loop", Dory::kv_per_pos(dc, LoopKv::per_loop, loops)},
                                        {"per_loop_4", Dory::kv_per_pos(dc, LoopKv::per_loop, 4)}}},
                {"ring_bytes_per_conversation", Dory::ring_bytes(dc, Dory::serving_kv(), loops)}};
    }
    const uint32_t units = ap.units, total = ap.total;
    const size_t ane_bytes = ap.bytes;
    const std::string src = ap.source;
    // a DFlash2 drafter next to the model or in the drafter cache counts when the engine loads it
    const std::string dfl = lp.dflash.empty() ? lp.dflash_left_out : lp.dflash;
    const size_t fixed = m.fixed(slots) + ane_bytes + lp.dflash_bytes;
    const size_t start_need = fixed + m.kv_per_pos * size_t(std::min(ctx, 8192));
    const size_t full_need = fixed + m.kv_per_pos * size_t(ctx) * size_t(slots);
    const size_t room = budget > fixed ? budget - fixed : 0;
    const size_t tokens = room / std::max<size_t>(m.kv_per_pos, 1);
    const double dec = bw > 0 && m.decode_bytes ? 0.75 * bw * 1e9 / double(m.decode_bytes) : 0;
    // prefill: ~0.33 TFLOP/s of fp16 MMA per GPU core (M4 Pro: 6-7 TFLOP/s on 20), 2 flops per weight
    const double params = double(m.decode_params);
    const double pf = 0.33e12 * cores / (2 * params);
    nlohmann::ordered_json o;
    o["dir"] = dir;
    o["dense"] = m.dense;
    o["weight_formats"] = m.formats;
    o["weights_bytes"] = m.weights;
    o["mtp_bytes"] = m.mtp;
    o["prefill_chunk"] = ch;
    o["slots"] = slots;
    o["activation_bytes"] = m.activations;
    o["slot_state_bytes"] = m.slot_state;
    o["kv_bytes_per_token"] = m.kv_per_pos;
    o["ane_units"] = units;
    o["ane_units_total"] = total;
    o["ane_bytes"] = ane_bytes;
    o["starts"] = start_need <= budget;
    o["fits_full_context"] = full_need <= budget;
    o["kv_tokens_in_budget"] = tokens;
    o["context_limit"] = model_context_limit(dir);
    o["default_context"] = default_ctx;
    o["dflash_drafter"] = dfl;
    o["dflash_used"] = !lp.dflash.empty();
    o["dflash_bytes"] = lp.dflash_found_bytes;
    o["decoding"] = !lp.dflash.empty() ? (dspark ? "dspark" : nemotron ? "lilicorr" : "dflash2")
                    : m.mtp            ? "mtp"
                    : lookup > 0       ? "prompt_lookup"
                                       : "plain";
    const auto enc = vision::open_encoder(dir);  // images: the encoder's weights load with the first image
    o["image_encoder_bytes"] = enc ? enc->weight_bytes() : 0;
    if (dory) o["dory"] = dory_j;
    o["decode_tps_estimate"] = std::round(dec);
    o["prefill_tps_estimate_gpu"] = std::round(pf);
    arr.push_back(o);
    if (tty) {
      Row r;
      r.name = dir.substr(dir.find_last_of('/') + 1);
      r.planned = dir.rfind(hub::home_dir() + "/plans/", 0) == 0;
      if (!r.planned && dir.rfind(hub::models_dir() + "/", 0) != 0) r.where = hub::tilde(dir.substr(0, dir.find_last_of('/')));
      r.dense = m.dense, r.layers = m.layers, r.formats = m.formats, r.slots = slots;
      r.starts = start_need <= budget, r.all_fit = full_need <= budget;
      r.full = full_need, r.start = start_need, r.weights = m.weights, r.kv = m.kv_per_pos, r.tokens = tokens;
      r.drafter = lp.dflash_bytes;
      r.decoding = !lp.dflash.empty() ? std::string(drafter)
                   : !lp.dflash_left_out.empty() ? std::string(m.mtp ? "MTP" : "plain") + term::dim(" (drafter left out)")
                   : m.mtp ? "MTP head" : lookup > 0 ? "prompt lookup" : "plain";
      r.image_bytes = enc ? enc->weight_bytes() : 0;
      r.dec = dec, r.pf = pf;
      r.note = dory_note;
      r.default_ctx = default_ctx, r.ctx_limit = model_context_limit(dir);
      tty_rows.push_back(r);
      continue;
    }
    if (as_json) continue;
    std::string name = dir.substr(dir.find_last_of('/') + 1);
    if (dir.rfind(hub::home_dir() + "/plans/", 0) == 0) name += ", not downloaded: planned from its headers";
    printf("  %s (%s, %d layers, hidden %d; weights %s)\n", name.c_str(), m.dense ? "dense" : "MoE", m.layers, m.hidden,
           m.formats.c_str());
    printf("    memory: weights %.2f GB, MTP head %.2f, %d-row prefill buffers %.2f, %d %s state %.2f, KV %.0f KB/token%s%s\n",
           m.weights / 1e9, m.mtp / 1e9, ch, m.activations / 1e9, slots, slots > 1 ? "slots'" : "slot's",
           slots * m.slot_state / 1e9, m.kv_per_pos / 1024.0,
           ane_bytes ? (", Neural Engine split " + std::to_string(int(ane_bytes / 1e6)) + " MB").c_str() : "",
           lp.dflash_bytes ? (std::string(", ") + drafter + " " + std::to_string(int(lp.dflash_bytes / 1e6)) + " MB").c_str() : "");
    if (m.streamed)
      printf("    routed experts: %.1f GB streamed from the SSD through a store of what the budget leaves, the "
             "process's GPU memory within 40%% of the RAM (the weights count its smallest size)\n",
             m.streamed / 1e9);
    if (start_need > budget)
      printf("    fits: NO: needs %.1f GB with an 8k-token conversation, the budget is %.1f GB\n", start_need / 1e9,
             budget / 1e9);
    else
      printf("    fits: yes; %d x %d tokens %s (%.1f GB); room for %zu tokens of KV cache in all%s\n", slots, ctx,
             full_need <= budget ? "fit" : "do not all fit at once", full_need / 1e9, tokens,
             full_need <= budget ? "" : " (the KV cache grows with each conversation; idle slots give theirs up)");
    {
      const int limit = model_context_limit(dir);
      printf("    context: kapitsa serve takes %s here without --context%s\n", term::context(default_ctx).c_str(),
             limit > 0 ? ("; the model's limit is " + term::context(limit)).c_str() : "");
    }
    if (!lp.dflash.empty())
      printf("    decoding: %s %s (%.2f GB with %d slot%s, counted above)\n", drafter, lp.dflash.c_str(),
             lp.dflash_bytes / 1e9, slots, slots > 1 ? "s" : "");
    else if (!lp.dflash_left_out.empty())
      printf("    decoding: %s; the %s %s is left out here: its %.2f GB would keep a %d-token conversation "
             "from fitting\n",
             m.mtp ? "MTP" : "plain", drafter, lp.dflash_left_out.c_str(), lp.dflash_found_bytes / 1e9, ctx);
    else
      printf("    decoding: %s%s\n",
             m.mtp ? "MTP" : lookup > 0 ? ("prompt lookup, up to " + std::to_string(lookup) + " drafts per cycle").c_str() : "plain",
             lp.dflash_off || (nemotron && m.dense) ? "" : (std::string(" (no ") + drafter + " found)").c_str());
    if (enc)
      printf("    images: %s, +%.2f GB when the first image arrives (not counted above)\n", enc->name().c_str(),
             enc->weight_bytes() / 1e9);
    if (dory) printf("    %s\n", dory_note.c_str());
    if (nemotron)
      printf("    prefill: Neural Engine split %s\n",
             ane_bytes ? ("on: " + src + ", " + std::to_string(int(ane_bytes / 1e6)) + " MB").c_str() : "off");
    if (m.dense && !dory && !nemotron)
      printf("    prefill: Neural Engine split %s\n",
             units ? (std::to_string(units) + "/" + std::to_string(total) + " units (" + src + ")").c_str()
                   : ("off: " + src).c_str());
    if (dec > 0)
      printf("    rough speed: plain decode ~%.0f tok/s (%.2f GB read per token), GPU prefill ~%.0f tok/s; %s add their "
             "measured factors (docs/benchmarks.md)\n",
             dec, m.decode_bytes / 1e9, pf,
             nemotron ? (m.dense ? "prompt lookup's accepted drafts" : "the block drafter and the Neural Engine split")
             : dory   ? "loops other than the default"
                      : "MTP / Neural Engine");
  }
  // every catalog variant: fits (one --ctx conversation beside it, its drafter kept) or what it needs
  EngineOptions eo;
  eo.context = ctx, eo.slots = slots_opt, eo.prefill_chunk = chunk, eo.dflash = dflash;
  const int slots = fit_slots(slots_opt);  // a variant fits with the slots asked for, else with two (auto's fewest)
  nlohmann::ordered_json cv = nlohmann::ordered_json::array();
  if (!as_json && !tty) printf("\nCatalog variants on this Mac (--context %s, --slots %d; `--model <name>` takes the first that fits of its default and fallbacks):\n", term::context(ctx).c_str(), slots);
  for (const auto& c : hub::catalog()) {
    bool fits = false;
    const hub::ModelRef pick = pick_variant(c, eo, budget, &fits);
    std::string line;
    for (const auto& v : c.variants) {
      const hub::ModelRef r = hub::resolve(c.name + ":" + v.name);
      const int f = variant_fits(r, eo, budget);
      std::string what = f == 1 ? "fits" : f == 0 ? "too big" : "not planned (offline)";
      if (f == 0) {
        const std::string d = hub::plan_dir(r);
        // whether it starts: with the slots the engine would take (one where two cannot start: auto_slots)
        const int s = slots_opt > 0 ? slots : std::min(slots, auto_slots(d, eo, budget));
        const LoadPlan lp = plan_load(d, dflash, false, ctx, s, budget, chunk);
        const StartNeed need = start_need(d, lp, lp.prefill_chunk, ctx, s);
        char b[64];
        if (need.bytes > budget) snprintf(b, sizeof b, "needs %.1f GB", need.bytes / 1e9);
        else snprintf(b, sizeof b, "starts, but not with a full --ctx conversation and its drafter");
        what = b;
      }
      const bool chosen = pick.variant == &v;
      line += (line.empty() ? "" : ", ") + v.name + " " + what + (chosen ? " <- --model " + hub::short_name(c) : "");
      cv.push_back({{"model", c.name}, {"variant", v.name}, {"fits", f == 1}, {"planned", f >= 0}, {"chosen", chosen},
                    {"default", v.name == c.default_variant}});
      if (tty) {
        Variant tv{.model = c.name, .alias = hub::short_name(c), .name = v.name, .fits = f, .chosen = chosen};
        if (f == 0) {
          tv.starts = what.rfind("starts", 0) == 0;
          if (what.rfind("needs ", 0) == 0) tv.need = what.substr(6);
        }
        tty_vars.push_back(tv);
      }
    }
    if (!as_json && !tty) printf("  %s: %s\n", c.name.c_str(), line.c_str());
  }
  spin.reset();
  if (tty)
    print_tty(J, budget, budget_gb > 0 ? "--mem-budget" : getenv("NI_MEM_BUDGET") ? "NI_MEM_BUDGET" : "",
              ctx, slots_opt, tty_rows, tty_vars);
  if (as_json) {
    J["context"] = ctx;
    J["slots"] = slots_opt;  // 0: each model's own (its "slots")
    J["models"] = arr;
    J["catalog_variants"] = cv;
    printf("%s\n", J.dump(2).c_str());
  }
  return 0;
}

}  // namespace ni
