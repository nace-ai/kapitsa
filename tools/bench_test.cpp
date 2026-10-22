// `kapitsa bench`'s core on the CPU, without a model (tests/test_bench.py drives it):
//   bench_test self                    the Mac table, the names and formats, the embedded corpus (exit 1 on a failure)
//   bench_test protocol CASES.json     the protocol's functions on the cases' inputs, as JSON (compared with bench.py's)
//   bench_test result [quick]          a result file from a made-up run, as JSON (checked against schema.json)
//   bench_test plan RESULT.json LOGIN  the pull request's plan for a result, as JSON
//   bench_test machine                 this Mac and its conditions now, as the command records them
//   bench_test run PORT                the protocol against a server on 127.0.0.1:PORT, with fixed nonces
//   bench_test lock [HOLD_S]           takes the GPU lock (BENCH_GPU_LOCK), holds it HOLD_S seconds
#include "bench/bench.h"
#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.0.0"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <fstream>
#include <sstream>

using namespace ni::bench;

namespace {

int fails = 0, checks = 0;

void check(bool ok, const std::string& name, const std::string& got = "") {
  ++checks;
  if (!ok) ++fails;
  if (!ok) printf("FAIL %s%s\n", name.c_str(), got.empty() ? "" : ("  (got: " + got + ")").c_str());
}

std::string read(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int self_test() {
  // the corpus
  check(sha256_hex(corpus()) == kCorpusSha256, "the embedded corpus is the frozen one", sha256_hex(corpus()));
  check(corpus().size() == 145652, "the corpus has 145,652 bytes", std::to_string(corpus().size()));
  auto chars = [](const std::string& x) {
    size_t n = 0;
    for (unsigned char c : x) n += (c & 0xC0) != 0x80;
    return n;
  };
  check(chars(build_context(4000)) == 17000 && chars(build_context(16000)) == 68000, "contexts of int(tokens * 4.25) characters");
  check(chars(build_context(2700, 100000)) == 11475, "the warm-up context: 11,475 characters from character 100,000");
  check(build_context(16000).size() > 68000, "the 16K context holds non-ASCII characters (more bytes than characters)");
  bool threw = false;
  try {
    build_context(40000);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "a context past the corpus' end is refused");
  // the Mac table: every entry named and dated, and this one's known models
  const ojson& t = mac_models();
  check(t.contains("models") && t["models"].size() >= 58, "the Mac table is embedded");
  std::set<std::string> seen;
  for (const auto& m : t["models"]) {
    const std::string id = m.value("identifier", "");
    check(!id.empty() && !m.value("name", std::string()).empty(), "entry has an identifier and a name: " + id);
    check(m.value("year", 0) >= 2020 && m.value("year", 0) <= 2030, "entry's year is plausible: " + id);
    check(seen.insert(id).second, "identifier listed once: " + id);
    std::string n;
    int y = 0;
    check(mac_model(id, &n, &y) && n == m["name"].get<std::string>() && y == m["year"].get<int>(), "lookup of " + id);
  }
  struct Case {
    const char* id;
    const char* name;
    int year;
  };
  for (const Case& c : {Case{"Mac16,7", "MacBook Pro (16-inch, 2024)", 2024}, Case{"Mac16,8", "MacBook Pro (14-inch, 2024)", 2024},
                        Case{"Mac16,6", "MacBook Pro (14-inch, 2024)", 2024}, Case{"Mac17,6", "MacBook Pro (16-inch, M5 Pro or M5 Max)", 2026},
                        Case{"Mac17,2", "MacBook Pro (14-inch, M5)", 2025}, Case{"MacBookAir10,1", "MacBook Air (M1, 2020)", 2020},
                        Case{"Mac15,3", "MacBook Pro (14-inch, Nov 2023)", 2023}, Case{"Mac14,8", "Mac Pro (2023)", 2023},
                        Case{"Mac13,2", "Mac Studio (2022)", 2022}, Case{"iMac21,2", "iMac (24-inch, M1, 2021)", 2021},
                        Case{"Macmini9,1", "Mac mini (M1, 2020)", 2020}, Case{"Mac16,11", "Mac mini (2024)", 2024}}) {
    std::string n;
    int y = 0;
    check(mac_model(c.id, &n, &y) && n == c.name && y == c.year, std::string("Apple's name for ") + c.id, n);
  }
  std::string n;
  int y = 0;
  check(!mac_model("MacBookPro16,1", &n, &y), "an Intel Mac is not in the table");
  check(!mac_model("Mac99,1", &n, &y), "an unknown identifier is not in the table");
  // display names: the year where Apple's name has none, the chip where it names two
  struct Disp {
    const char* name;
    int year;
    const char* chip;
    const char* want;
  };
  for (const Disp& d : {Disp{"MacBook Pro (16-inch, 2024)", 2024, "Apple M4 Pro", "MacBook Pro (16-inch, 2024)"},
                        Disp{"MacBook Pro (14-inch, M5)", 2025, "Apple M5", "MacBook Pro (14-inch, 2025)"},
                        Disp{"MacBook Pro (16-inch, M5 Pro or M5 Max)", 2026, "Apple M5 Max", "MacBook Pro (16-inch, 2026)"},
                        Disp{"Mac mini (M6)", 2026, "Apple M6", "Mac mini (2026)"},
                        Disp{"Mac mini (M5 Pro)", 2026, "Apple M5 Pro", "Mac mini (2026)"},
                        Disp{"MacBook Air (13-inch, M5)", 2026, "Apple M5", "MacBook Air (13-inch, 2026)"},
                        Disp{"MacBook Pro (14-inch, Nov 2023)", 2023, "Apple M3", "MacBook Pro (14-inch, Nov 2023)"},
                        Disp{"MacBook Air (15-inch, M4, 2025)", 2025, "Apple M4", "MacBook Air (15-inch, M4, 2025)"},
                        Disp{"iMac (24-inch, 2024, Four ports)", 2024, "Apple M4", "iMac (24-inch, 2024, Four ports)"},
                        Disp{"Mac Studio (M5 Ultra)", 2026, "Apple M5 Ultra", "Mac Studio (2026)"},
                        Disp{"", 0, "Apple M9", "Mac99,1"}}) {
    const std::string got = mac_display(d.name, d.year, d.chip, "Mac99,1");
    check(got == d.want, std::string("display of ") + (*d.name ? d.name : "an unknown Mac"), got);
  }
  check(chip_short("Apple M4 Pro") == "M4 Pro", "chip_short");
  // formats and Python's rounding
  check(py_round(2.675, 2) == 2.67, "round(2.675, 2) is 2.67 (binary value below the tie)");
  check(py_round(0.125, 2) == 0.12, "round(0.125, 2) is 0.12 (tie to even)");
  check(py_round(153.7449, 1) == 153.7, "round(153.7449, 1)");
  check(median({3, 1, 2}) == 2 && median({4, 1, 3, 2}) == 2.5, "median");
  check(thousands(1169) == "1,169" && thousands(16286) == "16,286" && thousands(999) == "999", "thousands");
  check(fmt_tps(153.72) == "153.7 tok/s" && fmt_tps(1169.4, true) == "1,169 tok/s", "fmt_tps", fmt_tps(1169.4, true));
  check(fmt_secs(0.0603) == "0.06 s" && fmt_gb(22678000000.0) == "21.1 GB", "fmt_secs / fmt_gb", fmt_gb(22678000000.0));
  check(model_display("Qwen3.6-35B-A3B") == "Qwen3.6 35B-A3B" && model_display("Nemotron 3.5 Bolt 74B-A9B") == "Nemotron 3.5 Bolt 74B-A9B",
        "model_display");
  check(shell_quote("abc") == "abc" && shell_quote("a b") == "'a b'" && shell_quote("it's") == "'it'\\''s'", "shell_quote");
  // the reference prompt counts
  check(reference_prompt_tokens("qwen3.5").at(4000) == 4133 && reference_prompt_tokens("qwen3.5").at(16000) == 16286,
        "Qwen's reference prompt tokens");
  check(reference_prompt_tokens("nemotron-h").at(16000) == 16514, "Nemotron's reference prompt tokens");
  check(reference_prompt_tokens("flint").at(4000) == 3874 && reference_prompt_tokens("flint").at(16000) == 15066,
        "Flint's reference prompt tokens (configs/flint.json's)");
  // nonces: the shapes bench.py makes
  Nonces nn;
  const std::string r = nn.run(), h64 = nn.hex(64), h48 = nn.hex(48);
  check(r.size() == 8 && r.rfind("0.", 0) == 0, "decode nonce like 0.123456", r);
  check(h64.size() == 16 && h48.size() == 12 && h64.find_first_not_of("0123456789abcdef") == std::string::npos, "hex nonces", h64);
  printf("%s: %d checks, %d failed\n", fails ? "FAIL" : "ok", checks, fails);
  return fails ? 1 : 0;
}

// The protocol's functions on given inputs: the cases come from tests/test_bench.py, which runs bench.py on the same
int protocol(const std::string& path) {
  const ojson cases = ojson::parse(read(path));
  ojson out = ojson::object();
  Settings s;
  s.model = cases.value("model", std::string("qwen3.6-35b-a3b"));
  // prompts with fixed nonces, and the request bodies
  ojson prompts = ojson::object();
  for (const auto& c : cases["contexts"]) {
    const int size = c["size"], offset = c.value("offset", 0);
    const std::string ctx = build_context(size, offset);
    prompts["context_" + std::to_string(size) + "_" + std::to_string(offset)] = {{"sha256", sha256_hex(ctx)}, {"bytes", ctx.size()}};
  }
  const std::string nonce = cases.value("nonce", std::string("session 0123456789abcdef"));
  const std::string run = cases.value("run", std::string("0.123456"));
  prompts["prefill_messages_4000"] = prefill_messages(build_context(4000), nonce, kPrefillQuestion);
  prompts["warm_messages"] = prefill_messages(build_context(kWarmTokens, kWarmOffsetChars), "warmup 0123456789abcdef", kWarmQuestion);
  prompts["decode_messages"] = decode_messages(run);
  prompts["tools_messages"] = tools_messages();
  prompts["tools"] = tools();
  prompts["followups"] = ojson::array({kFollowups[0], kFollowups[1], kFollowups[2]});
  prompts["tool_result"] = kToolResult;
  prompts["body_decode"] = body_of(s, decode_messages(run), s.decode_tokens);
  prompts["body_greedy"] = body_of(s, decode_messages(run), s.decode_tokens, true);
  prompts["body_prefill"] = body_of(s, prefill_messages("CTX", nonce, kPrefillQuestion), 256);
  prompts["body_tools_off"] = body_of(s, tools_messages(), 512, false, false, {{"tools", tools()}, {"tool_choice", "auto"}});
  prompts["body_tools_on"] = body_of(s, tools_messages(), 4096, false, true, {{"tools", tools()}, {"tool_choice", "auto"}});
  out["prompts"] = prompts;
  // rates and checks
  ojson rates = ojson::array();
  for (const auto& c : cases["rates"]) {
    std::vector<double> t;
    for (const auto& x : c["t"]) t.push_back(x);
    std::vector<int> ch;
    for (const auto& x : c["c"]) ch.push_back(x);
    const Rate r = decode_rates(t, ch, c["ct"], c.value("head", 16), c.value("tail", 4));
    rates.push_back({{"mid", r.mid ? ojson(*r.mid) : ojson(nullptr)}, {"e2e", r.e2e ? ojson(*r.e2e) : ojson(nullptr)}, {"info", r.info}});
  }
  out["rates"] = rates;
  ojson d4 = ojson::array();
  for (const auto& c : cases["distinct4"]) {
    const auto v = distinct4(c.get<std::string>());
    d4.push_back(v ? ojson(*v) : ojson(nullptr));
  }
  out["distinct4"] = d4;
  ojson leaks = ojson::array();
  for (const auto& c : cases["leaks"]) {
    Stream r;
    r.content = c.value("content", "");
    r.reasoning = c.value("reasoning", "");
    leaks.push_back(thinking_leak(r));
  }
  out["leaks"] = leaks;
  ojson rounds = ojson::array();
  for (const auto& c : cases["rounds"]) rounds.push_back(py_round(c[0].get<double>(), c[1].get<int>()));
  out["rounds"] = rounds;
  ojson sums = ojson::array();
  for (const auto& c : cases["summaries"]) sums.push_back(summary(c));
  out["summaries"] = sums;
  printf("%s\n", out.dump().c_str());
  return 0;
}

// A made-up run: what make_result does with it is what tests/test_bench.py checks against the schema
int result(bool quick) {
  RunInputs in;
  in.mac.identifier = "Mac16,7";
  mac_model(in.mac.identifier, &in.mac.name, &in.mac.year);
  in.mac.chip = "Apple M4 Pro";
  in.mac.cpu_cores = 14, in.mac.cpu_performance_cores = 10, in.mac.cpu_efficiency_cores = 4, in.mac.gpu_cores = 20;
  in.mac.memory_bytes = 51539607552ull;
  in.mac.macos = "27.0.1", in.mac.macos_build = "26A434";
  in.model = {.name = "qwen3.6-35b-a3b", .title = "Qwen3.6-35B-A3B", .variant = "4bit", .precision = "4-bit",
              .quant = "MLX affine 4-bit, group 64", .catalog = true, .repo = "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
              .revision = "95460516fbfe34c48f577f67c9ed191f5d61fbaf", .decoding = "DFlash2",
              .decoding_detail = "DFlash2 speculative decoding (verify 7 of 7)", .drafter = "incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6"};
  in.quick = quick;
  if (quick) in.settings.decode_runs = 3, in.settings.greedy_runs = 3, in.settings.sizes = {4000};
  in.settings.model = "qwen3.6-35b-a3b";
  in.settings.ref_prompt_tokens = reference_prompt_tokens("qwen3.5");
  in.version = KAPITSA_VERSION, in.commit = "940ee7d";
  in.time_utc = "2026-10-18T09:12:44Z";
  in.power_start = {.source = "ac", .battery_percent = 84, .low_power_mode = false, .power_mode = "high"};
  in.power_end = in.power_start;
  in.vm_start.swap_used_mb = 0, in.vm_start.swapouts = 100, in.vm_start.memorystatus_level = 70;
  in.vm_end = in.vm_start;
  in.thermal_start = "nominal";
  in.quiet_wait_s = 16.2;
  ojson& raw = in.raw;
  double t = 1000;
  auto runs = [&](int n, double base, bool greedy) {
    ojson a = ojson::array();
    for (int i = 0; i < n; ++i)
      a.push_back({{"greedy", greedy}, {"invalid", ojson::array()}, {"flags", ojson::array()}, {"completion_tokens", 512},
                   {"decode_tps", base + i * 3.1}, {"decode_tps_e2e", base + i * 3.1 - 2}, {"finish", "length"}, {"distinct4", 0.91}});
    return a;
  };
  raw["warmup"] = {{"ok", true}};
  raw["phases"]["warmup"] = {t, t + 4};
  raw["decode"] = runs(in.settings.decode_runs, 147.5, false);
  raw["decode_summary"] = summary(raw["decode"]);
  raw["phases"]["decode"] = {t + 20, t + 40};
  raw["decode_greedy"] = runs(in.settings.greedy_runs, 172.2, true);
  raw["decode_greedy_summary"] = summary(raw["decode_greedy"]);
  raw["phases"]["decode_greedy"] = {t + 40, t + 56};
  raw["prefill"] = ojson::array();
  for (int size : in.settings.sizes) {
    const int pt = size == 4000 ? 4133 : 16286;
    const double ttft = size == 4000 ? 3.5355 : 16.517;
    raw["prefill"].push_back({{"target", size}, {"invalid", ojson::array()}, {"flags", ojson::array()}, {"prompt_tokens", pt},
                              {"expected_prompt_tokens", pt}, {"ttft_s", ttft}, {"prefill_tps", pt / ttft}, {"cold_cached_tokens", 0},
                              {"finish", "stop"},
                              {"cached", {{"samples", ojson::array({{{"ttft_s", 0.061}, {"prompt_tokens", pt + 40}, {"cached_tokens", pt + 30}},
                                                                    {{"ttft_s", 0.0603}, {"prompt_tokens", pt + 80}, {"cached_tokens", pt + 70}},
                                                                    {{"ttft_s", 0.0598}, {"prompt_tokens", pt + 120}, {"cached_tokens", pt + 110}}})},
                                          {"invalid", ojson::array()}, {"flags", ojson::array()}, {"median_ttft_s", 0.0603}}}});
    raw["phases"]["prefill_" + std::to_string(size)] = {t + 60, t + 70 + ttft};
    t += 30;
  }
  raw["tools"] = ojson::array({{{"thinking", false}, {"stream", {{"ok", true}}}, {"nonstream", {{"ok", true}}}, {"round_trip", {{"ok", true}}}},
                               {{"thinking", true}, {"stream", {{"ok", true}}}, {"nonstream", {{"ok", true}}}, {"round_trip", {{"ok", true}}}}});
  raw["tools_pass"] = tools_pass(raw["tools"]);
  raw["phases"]["tools"] = {t + 100, t + 110};
  for (const auto& [name, span] : raw["phases"].items())
    raw["memory"][name] = {{"phys_footprint", 21000000000ull}, {"phys_footprint_peak", 22678000000ull}, {"mapped_file_clean", 2506752},
                           {"swap_used_mb", 0}, {"memorystatus_level", 70}, {"swapouts_delta", 0}};
  // the GPU: quiet but for one busy stretch in the 4K prefill (flagged), and a little ambient load
  for (double x = 1000; x < t + 112; x += 2) {
    Sample sm;
    sm.t = x;
    const bool busy = x >= 1060 && x <= 1068;
    sm.foreign = busy ? 12.5 : 0.4;
    sm.top = busy ? "Google Chrome" : "-";
    sm.top_pct = sm.foreign;
    sm.ambient = 3.0;
    if (busy) sm.users = {{"Google Chrome", 12.5}};
    sm.thermal = "nominal";
    sm.power_source = "ac";
    in.samples.push_back(sm);
  }
  printf("%s", make_result(in).dump(2).c_str());
  printf("\n");
  return 0;
}

int plan(const std::string& path, const std::string& login) {
  const std::string content = read(path);
  const ojson r = ojson::parse(content);
  const PrPlan p = pr_plan(target_repo(), login, r, content);
  printf("%s\n", ojson{{"repo", p.repo}, {"fork", p.fork}, {"branch", p.branch}, {"path", p.path}, {"title", p.title},
                       {"body", p.body}, {"commands", p.commands},
                       {"submit_problem", submit_problem(r)}, {"file_name", result_file_name(r)}}
                     .dump()
                     .c_str());
  return 0;
}

// The protocol against a server (tests/test_bench.py's scripted one), with fixed nonces: warm-up, one decode at
// T = 0.6 and one greedy, the 4K prefill with its follow-ups, the tool checks
int run_protocol(int port) {
  Settings s;
  s.port = port;
  s.model = "qwen3.6-35b-a3b";
  s.turn_gap = 0.2;
  s.ref_prompt_tokens = reference_prompt_tokens("qwen3.5");
  Nonces n;
  n.fixed = true;
  ojson out = {{"warmup", warmup(s, n)}, {"decode", t_decode(s, false, n)}, {"greedy", t_decode(s, true, n)},
               {"prefill", t_prefill(s, 4000, n)}, {"tools_off", t_tools(s, false)}, {"tools_on", t_tools(s, true)}};
  printf("%s\n", out.dump().c_str());
  return 0;
}

// The GPU lock (BENCH_GPU_LOCK names a test file): takes it, says how long it waited and who held it, holds it for
// `hold` seconds, releases it
int lock_test(double hold) {
  const auto t0 = std::chrono::steady_clock::now();
  std::string holders;
  bool waited = false;
  GpuLock l;
  const bool ok = l.acquire([&](const std::string& who) { holders = who, waited = true; });
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("%s\n", ojson{{"ok", ok}, {"path", GpuLock::path()}, {"waited", waited}, {"wait_s", secs}, {"holders", holders}}.dump().c_str());
  fflush(stdout);
  if (ok && hold > 0) std::this_thread::sleep_for(std::chrono::milliseconds(long(hold * 1000)));
  l.release();
  return ok ? 0 : 1;
}

int machine() {
  const Mac m = this_mac();
  const Power p = power();
  const Vm v = vm();
  const GpuSnap g = gpu_snapshot();
  printf("%s\n", ojson{{"identifier", m.identifier}, {"name", m.name}, {"year", m.year},
                       {"display", mac_display(m.name, m.year, m.chip, m.identifier)}, {"chip", m.chip}, {"cpu_cores", m.cpu_cores},
                       {"cpu_performance_cores", m.cpu_performance_cores}, {"cpu_efficiency_cores", m.cpu_efficiency_cores},
                       {"gpu_cores", m.gpu_cores}, {"memory_bytes", m.memory_bytes}, {"macos", m.macos}, {"macos_build", m.macos_build},
                       {"power_source", p.source}, {"battery_percent", p.battery_percent}, {"low_power_mode", p.low_power_mode},
                       {"power_mode", p.power_mode}, {"thermal", thermal_state()}, {"swap_used_mb", v.swap_used_mb},
                       {"swapouts", v.swapouts}, {"memorystatus_level", v.memorystatus_level}, {"gpu_clients", g.size()},
                       {"gpu_memory_in_use", gpu_memory_in_use()}, {"ancestors", ancestors().size()}, {"ambient", ambient_names()}}
                     .dump(2)
                     .c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "self";
  if (cmd == "self") return self_test();
  if (cmd == "protocol" && argc > 2) return protocol(argv[2]);
  if (cmd == "result") return result(argc > 2 && std::string(argv[2]) == "quick");
  if (cmd == "plan" && argc > 3) return plan(argv[2], argv[3]);
  if (cmd == "machine") return machine();
  if (cmd == "run" && argc > 2) return run_protocol(atoi(argv[2]));
  if (cmd == "lock") return lock_test(argc > 2 ? atof(argv[2]) : 0);
  fprintf(stderr, "usage: bench_test self | protocol CASES.json | result [quick] | plan RESULT.json LOGIN | machine | run PORT | lock [HOLD_S]\n");
  return 2;
}
