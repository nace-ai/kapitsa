// The result file of `kapitsa bench` (bench/community/schema.json, version kSchema) and the texts made from it: its
// name, the pull request's title and body, and the formats the screens share (bench.h).
//
// The file is small on purpose (about 2 KB): the person submitting it reads all of it before it is sent, and a
// maintainer reads it as a pull request's file. So it keeps what the community table and its checks need: the medians and each run's rate, the prompts' sizes and times, the follow-ups'
// times, the peak memory, the tool check, and the conditions in brief (the phases other apps disturbed, the swap, the
// thermal state, the power), with the harness's flags.
#include "bench/bench.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

namespace ni::bench {

// ---------------------------------------------------------------- a folder's precision

// A folder's precision from its config.json and weights ("4-bit", "bf16").
//
// config.json's top-level quantization is the default a per-module entry overrides, and a pack can use several: the
// Q2_0 Flash-Next pack (tools/qwen4_exp_q20.py) keeps its 144 routed-expert modules in Q2_0 (2-bit) and 91% of its
// weight bytes there, under a top-level "bits": 4 that names only its dense modules. So the share is counted in weight
// bytes, as the engine's own formats_summary does, and the dominant format names the precision; a mixture is spelled
// out in `quant` rather than passed off as the default.
std::string folder_precision(const std::string& dir, std::string* quant) {
  std::string cfg;
  {
    std::ifstream in(dir + "/config.json");
    std::stringstream ss;
    ss << in.rdbuf();
    cfg = ss.str();
  }
  const ojson c = ojson::parse(cfg, nullptr, false);
  ojson q;
  if (!c.is_discarded() && c.is_object())
    q = c.contains("quantization") ? c["quantization"] : c.contains("quantization_config") ? c["quantization_config"] : ojson();
  if (!q.is_object() || !q.contains("bits") || !q["bits"].is_number()) {
    *quant = "bf16";
    return "bf16";
  }
  const auto label = [](const ojson& e) {
    const int bits = e.value("bits", 4);
    if (e.value("mode", "affine") == "q2_0") return std::string("Q2_0 2-bit");
    return std::to_string(bits) + "-bit group " + std::to_string(e.value("group_size", 64));
  };
  // the per-module overrides, longest name first so a module's own entry wins over a prefix of it
  std::vector<std::pair<std::string, std::string>> overrides;
  for (auto& [k, v] : q.items())
    if (v.is_object() && v.contains("bits")) overrides.emplace_back(k, label(v));
  std::sort(overrides.begin(), overrides.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  const std::string dflt = label(q);
  std::map<std::string, size_t> bytes;
  if (DIR* d = opendir(dir.c_str())) {
    std::vector<std::string> files;
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) files.push_back(n);
    }
    closedir(d);
    for (const std::string& f : files) {
      std::string hdr;
      {
        // the header only: a shard is gigabytes, and its 8-byte length says how far to read
        std::ifstream in(dir + "/" + f, std::ios::binary);
        uint64_t n = 0;
        if (!in.read(reinterpret_cast<char*>(&n), sizeof n) || n > (64u << 20)) continue;
        hdr.resize(size_t(n));
        if (!in.read(hdr.data(), std::streamsize(n))) continue;
      }
      const ojson h = ojson::parse(hdr, nullptr, false);
      if (h.is_discarded() || !h.is_object()) continue;
      for (auto& [name, v] : h.items()) {
        if (name == "__metadata__" || !v.contains("data_offsets")) continue;
        std::string mod = name;
        for (const char* tail : {".weight", ".scales", ".biases"})
          if (mod.size() > strlen(tail) && mod.compare(mod.size() - strlen(tail), strlen(tail), tail) == 0) {
            mod.resize(mod.size() - strlen(tail));
            break;
          }
        std::string fmt = dflt;
        for (const auto& [k, l] : overrides)
          if (mod.rfind(k, 0) == 0) {
            fmt = l;
            break;
          }
        bytes[fmt] += v["data_offsets"][1].get<size_t>() - v["data_offsets"][0].get<size_t>();
      }
    }
  }
  if (bytes.empty()) {  // the weights are not readable here: the config's default is all that is known
    *quant = dflt;
    const int bits = q["bits"];
    return std::to_string(bits) + "-bit";
  }
  size_t tot = 0, top = 0;
  std::string best;
  for (auto& [k, v] : bytes) {
    tot += v;
    if (v > top) top = v, best = k;
  }
  std::string r;
  for (auto& [k, v] : bytes)
    r += (r.empty() ? "" : ", ") + k + " " + std::to_string(size_t(100.0 * double(v) / double(std::max<size_t>(tot, 1)) + 0.5)) + "%";
  *quant = r;
  // the precision, short (the catalog's own style): the dominant format's bit count, "(mixed)" when more than one
  if (best == "Q2_0 2-bit") return bytes.size() == 1 ? "2-bit (Q2_0)" : "2-bit (mixed)";
  const std::string bits = best.substr(0, best.find('-')) + "-bit";
  return bytes.size() == 1 ? bits : bits + " (mixed)";
}

// ---------------------------------------------------------------- formats

std::string thousands(long v) {
  std::string s = std::to_string(std::labs(v)), out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (i && (s.size() - i) % 3 == 0) out += ',';
    out += s[i];
  }
  return (v < 0 ? "-" : "") + out;
}

std::string fmt_tps(double v, bool integer) {
  if (!std::isfinite(v)) return "–";
  if (integer) return thousands(std::lround(v)) + " tok/s";
  char b[32];
  snprintf(b, sizeof b, "%.1f tok/s", v);
  return b;
}

std::string fmt_secs(double v) {
  if (!std::isfinite(v)) return "–";
  char b[32];
  snprintf(b, sizeof b, "%.2f s", v);
  return b;
}

std::string fmt_gb(double bytes) {
  char b[32];
  snprintf(b, sizeof b, "%.1f GB", bytes / 1073741824.0);  // GB of 2^30 bytes, as docs/benchmarks.md
  return b;
}

std::string fmt_temp(double t) {
  char b[32];
  snprintf(b, sizeof b, "%.1f", t);
  return b;
}

std::string model_display(const std::string& title) {
  // "Qwen3.6-35B-A3B" -> "Qwen3.6 35B-A3B", as the README names the Qwen models
  std::string t = title;
  if (t.rfind("Qwen", 0) == 0)
    if (const size_t d = t.find('-'); d != std::string::npos) t[d] = ' ';
  return t;
}

namespace {

std::string str_or(const ojson& o, const char* k, const std::string& def = "") {
  return o.is_object() && o.contains(k) && o[k].is_string() ? o[k].get<std::string>() : def;
}

bool is_num(const ojson& o, const char* k) { return o.is_object() && o.contains(k) && o[k].is_number(); }

ojson decode_block(const ojson& runs, const ojson& summary) {
  ojson tps = ojson::array();
  for (const auto& r : runs)
    if (r["invalid"].empty() && is_num(r, "decode_tps")) tps.push_back(py_round(r["decode_tps"], 2));
  return {{"median_tps", is_num(summary, "median_tps") ? summary["median_tps"] : ojson(nullptr)}, {"runs_tps", tps}};
}

std::string mac_short(const ojson& r) {
  const ojson& m = r["mac"];
  return mac_display(str_or(m, "name"), m.value("year", 0), str_or(m, "chip"), str_or(m, "identifier"));
}

std::string slug(const std::string& s) {
  std::string out;
  for (char ch : s) {
    const char c = char(tolower(uint8_t(ch)));
    if (isalnum(uint8_t(c)) || c == '.') out += c;
    else if (!out.empty() && out.back() != '-') out += '-';
  }
  while (!out.empty() && out.back() == '-') out.pop_back();
  return out;
}

int ram_gb(const ojson& r) { return r["mac"].value("memory_gb", 0); }

std::string py_list(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& x : v) s += (s.empty() ? "'" : ", '") + x + "'";
  return "[" + s + "]";
}

}  // namespace

// ---------------------------------------------------------------- the result

std::string decoding_of(const std::string& detail) {
  for (const char* n : {"KFlash", "prompt-lookup", "DFlash2", "LiLiCorr", "DSpark", "MTP", "plain"})
    if (detail.rfind(n, 0) == 0) return n;
  return "unknown";
}

bool decoding_has_drafter(const std::string& d) { return d == "KFlash" || d == "DFlash2" || d == "LiLiCorr" || d == "DSpark"; }

ojson make_result(const RunInputs& in) {
  const ojson& raw = in.raw;
  ojson flags = ojson::array(), invalid = ojson::array();
  // bench.py's roll-up: each phase's invalid measurements and warnings
  for (const auto& [name, key] : {std::pair<const char*, const char*>{"decode", "decode_summary"}, {"decode_greedy", "decode_greedy_summary"}}) {
    if (!raw.contains(key)) continue;
    for (const auto& x : raw[key]["invalid"]) invalid.push_back(std::string(name) + ": " + x.get<std::string>());
    for (const auto& x : raw[key]["flags"]) flags.push_back(std::string(name) + ": " + x.get<std::string>());
  }
  if (raw.contains("prefill"))
    for (const auto& p : raw["prefill"]) {
      const std::string sz = std::to_string(p["target"].get<int>());
      for (const auto& x : p["invalid"]) invalid.push_back("prefill_" + sz + ": " + x.get<std::string>());
      for (const auto& x : p["flags"]) flags.push_back("prefill_" + sz + ": " + x.get<std::string>());
      if (p.contains("cached")) {
        for (const auto& x : p["cached"]["invalid"]) invalid.push_back("cached_" + sz + ": " + x.get<std::string>());
        for (const auto& x : p["cached"]["flags"]) flags.push_back("cached_" + sz + ": " + x.get<std::string>());
      }
    }

  if (in.external_server) invalid.push_back("server: a server kapitsa bench did not start (KAPITSA_BENCH_SERVER), so its settings are unknown");
  // conditions per timed phase: other apps on the GPU (compare_engines.py's contention summary), swap, thermal state
  std::vector<std::string> busy, hot;
  double t_first = INFINITY, t_last = -INFINITY;
  long swapouts = 0;
  if (raw.contains("phases"))
    for (const auto& [name, span] : raw["phases"].items()) {
      if (name == "warmup") continue;
      const double t0 = span[0], t1 = span[1];
      const ojson c = summarize_samples(in.samples, t0, t1);
      if (c.value("flagged", false)) busy.push_back(name);
      if (c.contains("thermal") && thermal_rank(c["thermal"]) >= 2) hot.push_back(name);
      t_first = std::min(t_first, t0), t_last = std::max(t_last, t1);
      const ojson mem = raw.contains("memory") && raw["memory"].contains(name) ? raw["memory"][name] : ojson::object();
      if (is_num(mem, "swapouts_delta")) {
        const long n = mem["swapouts_delta"].get<long>();
        swapouts += std::max(0L, n);
        if (n > 1000) flags.push_back(name + ": " + std::to_string(n) + " pages swapped out during the phase");
      }
    }
  const ojson overall = std::isfinite(t_first) ? summarize_samples(in.samples, t_first, t_last) : ojson{{"samples", 0}, {"flagged", false}};
  if (!busy.empty() || overall.value("flagged", false)) {
    // the harness's words, so that its tables mark these runs alike
    std::vector<std::string> top;
    for (const auto& t : overall["top"]) top.push_back(t);
    char b[48];
    snprintf(b, sizeof b, "%g", overall.value("mean_foreign_gpu_pct", 0.0));
    flags.push_back("CONTENTION: sustained foreign GPU load > 5% in " + py_list(busy.empty() ? std::vector<std::string>{"bench"} : busy) +
                    " (mean " + b + "%, top " + py_list(top) + ")");
  }
  bool battery = in.power_start.source == "battery" || in.power_end.source == "battery";
  bool low = in.power_start.low_power_mode || in.power_end.low_power_mode || in.power_start.power_mode == "low";
  std::string worst = in.thermal_start;
  for (const auto& s : in.samples) {
    battery = battery || s.power_source == "battery";
    low = low || s.low_power;
    if (thermal_rank(s.thermal) > thermal_rank(worst)) worst = s.thermal;
  }
  if (battery) flags.push_back("power: on battery (a Mac on battery may run slower)");
  if (low) flags.push_back("power: Low Power Mode (the GPU runs slower)");
  if (!hot.empty()) {
    std::string ph;
    for (const auto& p : hot) ph += (ph.empty() ? "" : ", ") + p;
    flags.push_back("thermal: " + worst + " during " + ph + " (macOS slows the chip down)");
  }

  // memory: the server's peak (phys_footprint_peak: the kernel's lifetime peak, so the largest is the last)
  uint64_t peak = 0;
  int64_t mapped = -1;
  if (raw.contains("memory"))
    for (const auto& [name, m] : raw["memory"].items()) {
      if (is_num(m, "phys_footprint_peak")) peak = std::max<uint64_t>(peak, m["phys_footprint_peak"].get<uint64_t>());
      if (is_num(m, "mapped_file_clean") && m["mapped_file_clean"].get<int64_t>() >= 0)
        mapped = std::max<int64_t>(mapped, m["mapped_file_clean"].get<int64_t>());
    }

  ojson prefill = ojson::array();
  if (raw.contains("prefill"))
    for (const auto& p : raw["prefill"]) {
      ojson turns = ojson::array();
      const ojson cached = p.contains("cached") ? p["cached"] : ojson::object();
      if (cached.contains("samples"))
        for (const auto& s : cached["samples"])
          if (is_num(s, "ttft_s")) turns.push_back(py_round(s["ttft_s"], 4));
      ojson entry = {{"size", p["target"]},
                     {"prompt_tokens", is_num(p, "prompt_tokens") ? p["prompt_tokens"] : ojson(nullptr)},
                     {"ttft_s", is_num(p, "ttft_s") ? ojson(py_round(p["ttft_s"], 4)) : ojson(nullptr)},
                     {"tps", is_num(p, "prefill_tps") ? ojson(py_round(p["prefill_tps"], 1)) : ojson(nullptr)},
                     {"followup_s", turns},
                     {"followup_median_s", is_num(cached, "median_ttft_s") ? cached["median_ttft_s"] : ojson(nullptr)}};
      // additive: the rate the follow-ups' uncached suffix was read at (absent from results made before it)
      if (is_num(cached, "median_suffix_prefill_tps")) entry["followup_suffix_tps"] = cached["median_suffix_prefill_tps"];
      prefill.push_back(entry);
    }

  ojson results = {{"decode", raw.contains("decode") ? decode_block(raw["decode"], raw["decode_summary"]) : ojson(nullptr)},
                   {"decode_greedy", raw.contains("decode_greedy") ? decode_block(raw["decode_greedy"], raw["decode_greedy_summary"]) : ojson(nullptr)},
                   {"prefill", prefill},
                   {"tool_calls", raw.value("tools_pass", false) ? "pass" : "fail"},
                   {"peak_memory_gb", peak ? ojson(py_round(double(peak) / 1073741824.0, 2)) : ojson(nullptr)}};
  if (mapped >= int64_t(0.05 * 1073741824.0)) results["mapped_gb"] = py_round(double(mapped) / 1073741824.0, 2);
  if (!raw.value("tools_pass", false) && raw.contains("tools")) {
    ojson failed = ojson::array();
    for (const auto& t : raw["tools"]) {
      const std::string th = t.value("thinking", false) ? "thinking on" : "thinking off";
      for (const char* part : {"stream", "nonstream", "round_trip"})
        if (!(t.contains(part) && t[part].value("ok", false))) failed.push_back(th + ": " + part);
    }
    results["tool_calls_failed"] = failed;
  }

  const Mac& m = in.mac;
  ojson mac = {{"identifier", m.identifier},
               {"name", m.name.empty() ? ojson(nullptr) : ojson(m.name)},
               {"year", m.year > 0 ? ojson(m.year) : ojson(nullptr)},
               {"chip", m.chip},
               {"cpu_cores", m.cpu_cores},
               {"gpu_cores", m.gpu_cores > 0 ? ojson(m.gpu_cores) : ojson(nullptr)},
               {"memory_gb", int(std::lround(double(m.memory_bytes) / 1073741824.0))},
               {"macos", m.macos},
               {"macos_build", m.macos_build}};
  const ModelInfo& mi = in.model;
  ojson model = {{"name", mi.name},
                 {"title", mi.title},
                 {"variant", mi.variant.empty() ? ojson(nullptr) : ojson(mi.variant)},
                 {"precision", mi.precision},
                 {"catalog", mi.catalog},
                 {"revision", mi.revision.empty() ? ojson(nullptr) : ojson(mi.revision.substr(0, 7))},
                 {"decoding", mi.decoding},
                 {"decoding_detail", mi.decoding_detail},
                 {"drafter", mi.drafter.empty() ? ojson(nullptr) : ojson(mi.drafter)}};
  const Settings& s = in.settings;
  ojson protocol = {{"corpus_sha256", sha256_hex(corpus())},
                    {"decode_tokens", s.decode_tokens},
                    {"decode_runs", s.decode_runs},
                    {"greedy_runs", s.greedy_runs},
                    {"sizes", s.sizes},
                    {"temperature", s.temperature}};
  if (raw.contains("context")) {
    protocol["context_window"] = raw["context"]["window_tokens"];
    for (const auto& x : raw["context"]["invalid"]) invalid.push_back("context: " + x.get<std::string>());
    for (const auto& x : raw["context"]["flags"]) flags.push_back("context: " + x.get<std::string>());
  }
  if (in.prefill_chunk) protocol["prefill_chunk"] = in.prefill_chunk;
  ojson users = ojson::array();
  for (const auto& u : overall.value("gpu_users", ojson::array())) users.push_back(u["name"]);
  ojson conditions = {{"power_source", in.power_start.source.empty() ? ojson(nullptr) : ojson(in.power_start.source)},
                      {"power_mode", in.power_start.power_mode.empty() ? ojson(nullptr) : ojson(in.power_start.power_mode)},
                      {"low_power_mode", low},
                      {"thermal_state", worst.empty() ? ojson(nullptr) : ojson(worst)},
                      {"swapouts", swapouts},
                      {"gpu_quiet_wait_s", py_round(in.quiet_wait_s, 1)},
                      {"gpu_other_mean_pct", overall.value("mean_foreign_gpu_pct", 0.0)},
                      {"gpu_other_max_pct", overall.value("max_foreign_gpu_pct", 0.0)},
                      {"gpu_ambient_mean_pct", overall.value("mean_ambient_gpu_pct", 0.0)},
                      {"gpu_users", users},
                      {"contention", busy}};
  ojson result = {{"schema", kSchema},
          {"kind", "kapitsa-bench"},
          {"mode", raw.contains("context") ? (in.quick ? "context-quick" : "context")
                   : in.prefill_chunk ? (in.quick ? "prefill-quick" : "prefill")
                   : (in.quick ? "quick" : "full")},
          {"time_utc", in.time_utc},
          {"kapitsa", {{"version", in.version}, {"commit", in.commit.empty() ? ojson(nullptr) : ojson(in.commit)}}},
          {"mac", mac},
          {"model", model},
          {"protocol", protocol},
          {"results", results},
          {"conditions", conditions},
          {"flags", flags},
          {"invalid", invalid}};
  if (raw.contains("context")) result["context"] = raw["context"];
  return result;
}

// Keep display precision and grouping consistent without rounding the saved measurements.
std::vector<std::pair<std::string, std::string>> context_rows(const ojson& r) {
  std::vector<std::pair<std::string, std::string>> rows;
  if (!r.contains("context")) return rows;
  const auto& cx = r["context"];
  auto number = [](double v, int places) {
    char b[64];
    snprintf(b, sizeof b, "%.*f", places, v);
    return std::string(b);
  };
  auto has = [&](const char* key) { return cx.contains(key) && cx[key].is_object(); };
  auto range = [&](const ojson& v, int places, const std::string& unit = "") {
    return number(v["median"], places) + unit + " [" + number(v["min"], places) + "–" + number(v["max"], places) + "]";
  };
  const std::string dot = "  ·  ";
  rows.emplace_back("Context", std::to_string(cx["window_tokens"].get<int>() / 1024) + "K; three repetitions, median [min–max]");
  if (cx.contains("prefill_prompt_tokens") && cx.contains("decode_prompt_tokens"))
    rows.emplace_back("Prompt tokens", thousands(cx["prefill_prompt_tokens"].get<long>()) + " reading; " +
                      thousands(cx["decode_prompt_tokens"].get<long>()) + " writing; " + thousands(cx.value("reply_reserve_tokens", 2048)) + " reserved");
  if (has("cached_tokens")) {
    const auto& v = cx["cached_tokens"];
    std::string s = thousands(std::lround(v["median"].get<double>())) + " [" + thousands(std::lround(v["min"].get<double>())) +
                    "–" + thousands(std::lround(v["max"].get<double>())) + "]";
    if (has("cache_reuse_pct")) s += dot + number(cx["cache_reuse_pct"]["median"], 1) + "% reuse";
    rows.emplace_back("Cached tokens", s);
  } else if (has("cache_reuse_pct")) rows.emplace_back("Cache reuse", number(cx["cache_reuse_pct"]["median"], 1) + "%");
  rows.emplace_back("", "");
  for (const auto& [key, label, rate, suffix] : {
       std::tuple{"uncached_ttft_s", "Cold TTFT", "cold_input_effective_tps", " input"},
       std::tuple{"cached_ttft_s", "Cached TTFT", "new_input_effective_tps", " new input"}}) {
    if (!has(key)) continue;
    std::string s = range(cx[key], 3, " s");
    if (has(rate)) s += dot + fmt_tps(cx[rate]["median"], true) + suffix;
    rows.emplace_back(label, s);
  }
  if (has("ttft_speedup")) {
    const auto& v = cx["ttft_speedup"];
    rows.emplace_back("Cache speedup", number(v["median"], 1) + "x [" + number(v["min"], 1) + "x–" + number(v["max"], 1) + "x]");
  }
  rows.emplace_back("", "");
  for (const auto& [key, label] : {std::pair{"decode_summary", "Ctx writing"}, {"decode_greedy_summary", "Ctx greedy"}}) {
    if (!has(key) || !cx[key].contains("median_tps")) continue;
    const auto& v = cx[key];
    std::string s = number(v["median_tps"], 1) + " tok/s [" + number(v["min_tps"], 1) + "–" + number(v["max_tps"], 1) + "]";
    if (key == std::string("decode_summary")) s += dot + "T = " + number(cx.value("temperature", 0.6), 1);
    rows.emplace_back(label, s);
  }
  if (cx.contains("diagnostics"))
    for (const auto& [key, label] : {std::pair{"decode", "Draft yield"}, {"decode_greedy", "Greedy yield"}}) {
      const auto& all = cx["diagnostics"];
      if (all.contains(key) && all[key].value("tokens_per_cycle", ojson()).is_object()) {
        const auto& d = all[key];
        std::string value = range(d["tokens_per_cycle"], 2, " tok/cycle");
        if (d.value("accepted_draft_tokens_per_cycle", ojson()).is_object())
          value += dot + number(d["accepted_draft_tokens_per_cycle"]["median"], 2) + " accepted drafts/cycle";
        rows.emplace_back(label, value);
      }
    }
  if (cx.contains("decoding_detail") && cx["decoding_detail"] != r["model"].value("decoding_detail", std::string()))
    rows.emplace_back("Ctx decoding", cx["decoding_detail"].get<std::string>());
  if (cx.contains("peak_memory_gb") && cx["peak_memory_gb"].is_number())
    rows.emplace_back("Ctx memory", number(cx["peak_memory_gb"], 2) + " GB peak");
  return rows;
}

// ---------------------------------------------------------------- names and texts

std::string result_file_name(const ojson& r) {
  const ojson& m = r["model"];
  std::string model = slug(str_or(m, "name"));
  if (!str_or(m, "variant").empty()) model += "-" + slug(str_or(m, "variant"));
  return "kapitsa-bench-" + str_or(r, "time_utc").substr(0, 10) + "-" + slug(chip_short(str_or(r["mac"], "chip"))) + "-" + model +
         (r["protocol"].contains("context_window") ? "-ctx" + std::to_string(r["protocol"]["context_window"].get<int>()) : "") +
         (str_or(r, "mode").find("quick") != std::string::npos ? "-quick" : "") + ".json";
}

std::string result_mac_line(const ojson& r) {
  return mac_short(r) + " · " + chip_short(str_or(r["mac"], "chip")) + " · " + std::to_string(ram_gb(r)) + " GB";
}

std::string result_title(const ojson& r) {
  const ojson& m = r["model"];
  std::string t = "Benchmark: " + result_mac_line(r) + " · " + model_display(str_or(m, "title", str_or(m, "name"))) + " " +
                  str_or(m, "precision");
  if (r["protocol"].contains("context_window")) t += " · " + std::to_string(r["protocol"]["context_window"].get<int>()) + "-token context";
  if (str_or(r, "mode").find("quick") != std::string::npos) t += " · quick";
  return t;
}

double writing_temperature(const ojson& r) {
  if (r.contains("protocol") && is_num(r["protocol"], "temperature")) return r["protocol"]["temperature"].get<double>();
  return r.contains("model") && str_or(r["model"], "name") == "flint" ? 1.0 : 0.6;
}

std::string result_markdown(const ojson& r) {
  const ojson& res = r["results"];
  const ojson& m = r["model"];
  const ojson& mac = r["mac"];
  const ojson& c = r["conditions"];
  std::set<std::string> busy;
  for (const auto& p : c.value("contention", ojson::array())) busy.insert(p.get<std::string>());
  const char* mark = " ⚠";
  auto cell_decode = [&](const char* k) -> std::string {
    const ojson& d = res[k];
    if (!d.is_object() || !d["median_tps"].is_number()) return "–";
    double lo = INFINITY, hi = -INFINITY;
    for (const auto& x : d["runs_tps"]) lo = std::min(lo, x.get<double>()), hi = std::max(hi, x.get<double>());
    char b[96];
    snprintf(b, sizeof b, " (%.1f–%.1f, median of %zu)", lo, hi, d["runs_tps"].size());
    return fmt_tps(d["median_tps"]) + (busy.count(k) ? mark : "") + b;
  };
  std::string mac_col = result_mac_line(r);
  if (mac["gpu_cores"].is_number())
    mac_col = mac_short(r) + " · " + chip_short(str_or(mac, "chip")) + " " + std::to_string(mac["gpu_cores"].get<int>()) +
              "-core GPU · " + std::to_string(ram_gb(r)) + " GB";
  const std::string model_col = model_display(str_or(m, "title", str_or(m, "name"))) + " · " + str_or(m, "precision");
  std::string s = "| " + mac_col + " | " + model_col + " |\n| --- | ---: |\n";
  if (r["protocol"].contains("prefill_chunk"))
    s += "| Prefill chunk (requested) | " + thousands(r["protocol"]["prefill_chunk"].get<int>()) + " tokens |\n";
  s += "| Writing, T = " + fmt_temp(writing_temperature(r)) + " | " + cell_decode("decode") + " |\n";
  s += "| Writing, greedy | " + cell_decode("decode_greedy") + " |\n";
  for (const auto& p : res["prefill"]) {
    const int size = p["size"];
    const std::string k = std::to_string(size / 1000) + "K", ph = "prefill_" + std::to_string(size);
    const std::string mk = busy.count(ph) ? mark : "";
    s += "| Reading a " + k + "-token prompt | " +
         (p["tps"].is_number() ? fmt_tps(p["tps"], true) + mk + " (" + thousands(p["prompt_tokens"].get<long>()) + " tokens in " +
                                     fmt_secs(p["ttft_s"]) + ")"
                               : std::string("–")) +
         " |\n";
    s += "| Follow-up at " + k + " | " + (p["followup_median_s"].is_number() ? fmt_secs(p["followup_median_s"]) + mk : std::string("–")) +
         " |\n";
  }
  if (res["peak_memory_gb"].is_number()) {
    char b[48];
    snprintf(b, sizeof b, "%.1f GB", res["peak_memory_gb"].get<double>());
    std::string mem = b;
    if (res.contains("mapped_gb")) snprintf(b, sizeof b, " + %.1f GB mapped", res["mapped_gb"].get<double>()), mem += b;
    s += "| Peak memory | " + mem + " |\n";
  }
  s += "| Tool calls | " + str_or(res, "tool_calls", "–") + (busy.count("tools") ? mark : "") + " |\n\n";
  if (!busy.empty()) s += "⚠ other apps used the GPU during this measurement (contention flagged): it may be low.\n\n";
  if (r.contains("context")) {
    s += "Additional context test:\n\n";
    s += "| Context metric | Result |\n| --- | ---: |\n";
    for (const auto& [label, value] : context_rows(r))
      if (!label.empty()) s += "| " + label + " | " + value + " |\n";
    s += "\nInput rates include first-token latency; cached rate counts only new tokens. Standard results above use the original 32K server.\n\n";
  }
  const std::string src = str_or(c, "power_source");
  std::string cond = "Kapitsa " + str_or(r["kapitsa"], "version") +
                     (r["kapitsa"]["commit"].is_string() ? " (" + r["kapitsa"]["commit"].get<std::string>() + ")" : "") + " · " +
                     (m["drafter"].is_string() && str_or(m, "decoding") != "KFlash" ? str_or(m, "decoding") + " drafter"
                                                                                       : str_or(m, "decoding") + " decoding") +
                     " · macOS " + str_or(mac, "macos") + " (" + str_or(mac, "macos_build") + ") · " +
                     (src == "ac" ? "on AC power" : src == "battery" ? "on battery" : src == "ups" ? "on UPS power" : "power source unknown") +
                     (str_or(c, "power_mode") == "high"  ? " · High Power mode"
                      : str_or(c, "power_mode") == "low" ? " · Low Power Mode"
                      : str_or(c, "power_mode") == "automatic" ? " · Automatic power mode" : "") +
                     (str_or(c, "thermal_state").empty() ? "" : " · thermal state " + str_or(c, "thermal_state")) + " · " +
                     (r["protocol"].contains("context_window") ? "standard benchmark followed by context workload at " + std::to_string(r["protocol"]["context_window"].get<int>()) + " tokens" :
                     r["protocol"].contains("prefill_chunk") ? std::string("standard workload with an explicit prefill chunk of ") + std::to_string(r["protocol"]["prefill_chunk"].get<int>()) + " tokens" :
                     str_or(r, "mode").find("quick") != std::string::npos ? "quick run (3 runs per writing test, the 4K prompt only)" : "full run");
  s += cond + "\n\n";
  if (r["flags"].empty()) {
    s += "Flags: none.\n";
  } else {
    s += "Flags:\n";
    for (const auto& f : r["flags"]) s += "- " + f.get<std::string>() + "\n";
  }
  s += "\nMeasured with `kapitsa bench`: the protocol of [bench/harness](https://github.com/" + target_repo() +
       "/tree/main/bench/harness), as a client of the local server.\n";
  return s;
}

std::string submit_problem(const ojson& r) {
  if (r.value("kind", "") != "kapitsa-bench" || r.value("schema", 0) != kSchema) return "This is not a kapitsa bench result.";
  if (r.contains("protocol") && r["protocol"].contains("context_window")) return "Custom-context runs cannot be submitted as standard community benchmarks.";
  if (r.contains("protocol") && r["protocol"].contains("prefill_chunk"))
    return "Custom prefill chunks cannot be submitted as standard community benchmarks.";
  const ojson& m = r["model"];
  if (!m.value("catalog", false) || !m["variant"].is_string())
    return "Only results of the catalog's models can be submitted (their weights are pinned, so results compare): "
           "kapitsa bench --model 35b, 27b, 4b or bolt, at any precision.";
  if (!r["invalid"].empty())
    return "This run has invalid measurements (" + r["invalid"][0].get<std::string>() + "), so it can't go in the table.";
  if (const std::string c = str_or(r["protocol"], "corpus_sha256"); c != kCorpusSha256 && c != kCorpusSha256V1)
    return "This run did not use the frozen prompt corpus.";
  return "";
}

}  // namespace ni::bench
