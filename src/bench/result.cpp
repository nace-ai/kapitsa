// The result file of `kapitsa bench` (bench/community/schema.json, version kSchema) and the texts made from it: its
// name, the pull request's title and body, and the formats the screens share (bench.h).
//
// The file is small on purpose (about 2 KB): the person submitting it reads all of it before it is sent, and a
// maintainer reads it as a pull request's file. So it keeps what the community table and its checks need: the medians and each run's rate, the prompts' sizes and times, the follow-ups'
// times, the peak memory, the tool check, and the conditions in brief (the phases other apps disturbed, the swap, the
// thermal state, the power), with the harness's flags.
#include "bench/bench.h"

#include <cmath>
#include <cstdio>

namespace ni::bench {

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
      prefill.push_back({{"size", p["target"]},
                         {"prompt_tokens", is_num(p, "prompt_tokens") ? p["prompt_tokens"] : ojson(nullptr)},
                         {"ttft_s", is_num(p, "ttft_s") ? ojson(py_round(p["ttft_s"], 4)) : ojson(nullptr)},
                         {"tps", is_num(p, "prefill_tps") ? ojson(py_round(p["prefill_tps"], 1)) : ojson(nullptr)},
                         {"followup_s", turns},
                         {"followup_median_s", is_num(cached, "median_ttft_s") ? cached["median_ttft_s"] : ojson(nullptr)}});
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
                    {"sizes", s.sizes}};
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
  return {{"schema", kSchema},
          {"kind", "kapitsa-bench"},
          {"mode", in.quick ? "quick" : "full"},
          {"time_utc", in.time_utc},
          {"kapitsa", {{"version", in.version}, {"commit", in.commit.empty() ? ojson(nullptr) : ojson(in.commit)}}},
          {"mac", mac},
          {"model", model},
          {"protocol", protocol},
          {"results", results},
          {"conditions", conditions},
          {"flags", flags},
          {"invalid", invalid}};
}

// ---------------------------------------------------------------- names and texts

std::string result_file_name(const ojson& r) {
  const ojson& m = r["model"];
  std::string model = slug(str_or(m, "name"));
  if (!str_or(m, "variant").empty()) model += "-" + slug(str_or(m, "variant"));
  return "kapitsa-bench-" + str_or(r, "time_utc").substr(0, 10) + "-" + slug(chip_short(str_or(r["mac"], "chip"))) + "-" + model +
         (str_or(r, "mode") == "quick" ? "-quick" : "") + ".json";
}

std::string result_mac_line(const ojson& r) {
  return mac_short(r) + " · " + chip_short(str_or(r["mac"], "chip")) + " · " + std::to_string(ram_gb(r)) + " GB";
}

std::string result_title(const ojson& r) {
  const ojson& m = r["model"];
  std::string t = "Benchmark: " + result_mac_line(r) + " · " + model_display(str_or(m, "title", str_or(m, "name"))) + " " +
                  str_or(m, "precision");
  if (str_or(r, "mode") == "quick") t += " · quick";
  return t;
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
  s += "| Writing, T = 0.6 | " + cell_decode("decode") + " |\n";
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
  const std::string src = str_or(c, "power_source");
  std::string cond = "Kapitsa " + str_or(r["kapitsa"], "version") +
                     (r["kapitsa"]["commit"].is_string() ? " (" + r["kapitsa"]["commit"].get<std::string>() + ")" : "") + " · " +
                     (m["drafter"].is_string() ? str_or(m, "decoding") + " drafter" : str_or(m, "decoding") + " decoding") +
                     " · macOS " + str_or(mac, "macos") + " (" + str_or(mac, "macos_build") + ") · " +
                     (src == "ac" ? "on AC power" : src == "battery" ? "on battery" : src == "ups" ? "on UPS power" : "power source unknown") +
                     (str_or(c, "power_mode") == "high"  ? " · High Power mode"
                      : str_or(c, "power_mode") == "low" ? " · Low Power Mode"
                      : str_or(c, "power_mode") == "automatic" ? " · Automatic power mode" : "") +
                     (str_or(c, "thermal_state").empty() ? "" : " · thermal state " + str_or(c, "thermal_state")) + " · " +
                     (str_or(r, "mode") == "quick" ? "quick run (3 runs per writing test, the 4K prompt only)" : "full run");
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
