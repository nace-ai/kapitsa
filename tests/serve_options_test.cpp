// kapitsa serve's one validator (src/server/service_config.cpp): the command line's flags and --config's JSON give the
// same Options, defaults stay EngineOptions' own, and what cannot run is refused with a message naming the field.
// No model, no GPU.
#include "server/service.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <iterator>
#include <unistd.h>
#include <string>
#include <vector>

using nlohmann::json;
using ni::server::Options;
using ni::server::Source;

static int failures = 0;
#define CHECK(condition, what)                                                                                         \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, std::string(what).c_str(), #condition);       \
      ++failures;                                                                                                      \
    }                                                                                                                  \
  } while (false)

// every field serve reads, side by side
static json fields(const Options& o) {
  const ni::EngineOptions& e = o.engine;
  return {{"model", o.model}, {"host", o.host}, {"port", o.port}, {"context", o.context}, {"api_key", o.api_key},
          {"profile", o.profile}, {"timing", o.timing}, {"idle_unload_min", o.idle_unload_min}, {"split", o.split},
          {"peers", o.peers}, {"engine_splash", o.engine_splash}, {"check_only", o.check_only},
          {"splash_kflash", o.splash_kflash}, {"priv", o.priv}, {"local", o.local}, {"allow_switch", o.allow_switch},
          {"verbose", o.verbose}, {"yes", o.yes}, {"model_id", e.model_id}, {"slots", e.slots},
          {"spec_depth", e.spec_depth}, {"prefill_chunk", e.prefill_chunk}, {"dflash", e.dflash},
          {"mem_budget_gb", e.mem_budget_gb}, {"image_max_pixels", e.image_max_pixels}, {"loops", e.loops},
          {"loop_kv", e.loop_kv}, {"disk_cache", e.disk_cache}, {"splash_binary", e.splash_binary},
          {"splash_model", e.splash_model}, {"splash_args", e.splash_args},
          {"splash_persistent_cache", e.splash_persistent_cache},
          {"splash_conversation_salt", e.splash_conversation_salt}, {"context_engine", e.context},
          {"model_dir", e.model_dir}};
}

static Options from_flags(const std::vector<std::string>& args, bool positional = true) {
  return ni::server::parse_options(*ni::server::flags_to_json(args, positional), Source::flags);
}

// the error a configuration gets, or "" when it is taken
static std::string refusal(const json& j, Source source = Source::json) {
  try {
    ni::server::parse_options(j, source);
    return "";
  } catch (const std::invalid_argument& e) { return e.what(); }
}
static std::string flag_refusal(const std::vector<std::string>& args) {
  try {
    from_flags(args);
    return "";
  } catch (const std::invalid_argument& e) { return e.what(); }
}
static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

int main(int argc, char** argv) {
  // --check: each stdin line is a config; prints "ok" or the validator's refusal (tests/test_serve_schema.py runs the
  // schema on the same lines)
  if (argc > 1 && std::string(argv[1]) == "--check") {
    for (std::string line; std::getline(std::cin, line);) {
      try {
        ni::server::parse_options(json::parse(line), Source::json);
        std::printf("ok\n");
      } catch (const std::invalid_argument& e) { std::printf("refused: %s\n", e.what()); }
    }
    return 0;
  }
  // --reexec N: reads --config - as serve does, then execs itself N more times, as restart_server() does; the last
  // prints what it read
  if (argc > 2 && std::string(argv[1]) == "--reexec") {
    const json j = ni::server::read_config("-");
    const int n = std::atoi(argv[2]);
    if (n <= 0) return std::printf("%s\n", j.dump().c_str()) < 0;
    const std::string left = std::to_string(n - 1);
    char* av[] = {argv[0], argv[1], const_cast<char*>(left.c_str()), nullptr};
    execv(argv[0], av);
    std::perror("execv");
    return 1;
  }
  // defaults: nothing given is EngineOptions' defaults and serve's own, by both routes
  {
    const ni::EngineOptions d;
    const json none = fields(ni::server::parse_options(json::object()));
    CHECK(none == fields(from_flags({})), "no flags and an empty config agree");
    CHECK(none == fields(Options{}), "an empty config is Options' defaults");
    CHECK(none["slots"] == d.slots && none["spec_depth"] == d.spec_depth && none["prefill_chunk"] == d.prefill_chunk &&
              none["mem_budget_gb"] == d.mem_budget_gb && none["image_max_pixels"] == d.image_max_pixels &&
              none["dflash"] == d.dflash && none["disk_cache"] == d.disk_cache && none["loops"] == d.loops,
          "engine defaults untouched");
    CHECK(none["port"] == 8000 && none["timing"] == "both" && none["idle_unload_min"] == -1 &&
              none["splash_kflash"] == "auto" && none["context"] == 0,
          "serve defaults");
  }

  // flags and JSON: the same settings give the same Options
  struct Same {
    std::vector<std::string> flags;
    json config;
  };
  const std::vector<Same> same = {
      {{"--model", "4b", "--port", "8123", "--host", "127.0.0.1", "--ctx", "128k", "--slots", "2", "--spec", "3",
        "--dflash", "none", "--prefill-chunk", "1024", "--api-key", "k", "--allow-switch", "--idle-unload", "0.5",
        "--mem-budget", "20.5", "--image-max-pixels", "0", "--timing", "ttft", "--disk-cache", "16G", "--loops", "3",
        "--loop-kv", "per-loop", "--split", "a:1,b:2", "--profile", "fast", "--private", "--local", "--peer", "m1",
        "--peer", "m2:9000", "--model-id", "id", "--verbose", "--yes"},
       {{"model", "4b"}, {"port", 8123}, {"host", "127.0.0.1"}, {"context", "128k"}, {"slots", 2}, {"spec", 3},
        {"dflash", "none"}, {"prefill_chunk", 1024}, {"api_key", "k"}, {"allow_switch", true}, {"idle_unload", 0.5},
        {"mem_budget", 20.5}, {"image_max_pixels", 0}, {"timing", "ttft"}, {"disk_cache", "16G"}, {"loops", 3},
        {"loop_kv", "per-loop"}, {"split", {"a:1", "b:2"}}, {"profile", "fast"}, {"private", true}, {"local", true},
        {"peer", {"m1", "m2:9000"}}, {"model_id", "id"}, {"verbose", true}, {"yes", true}}},
      {{"35b", "--context", "max", "--idle-unload", "off"}, {{"model", "35b"}, {"context", "max"}, {"idle_unload", 0}}},
      {{"-m", "x", "--context", "131072", "--idle-unload", "auto"},
       {{"model", "x"}, {"context", 131072}, {"idle_unload", "auto"}}},
      {{"--engine", "splash", "--splash-model", "/m", "--splash-binary", "/b", "--splash-arg", "--verify-rows",
        "--splash-arg", "2", "--splash-kflash", "off", "--splash-persistent-cache", "--splash-conversation-salt",
        "--check"},
       {{"engine", "splash"}, {"splash_model", "/m"}, {"splash_binary", "/b"}, {"splash_arg", {"--verify-rows", "2"}},
        {"splash_kflash", "off"}, {"splash_persistent_cache", true}, {"splash_conversation_salt", true},
        {"check", true}}},
      {{"--port", "1", "--port", "2"}, {{"port", 2}}},  // a later flag wins
  };
  for (const Same& s : same) {
    const json via_flags = fields(from_flags(s.flags)), via_json = fields(ni::server::parse_options(s.config));
    CHECK(via_flags == via_json, s.flags[0] + ": flags " + via_flags.dump() + " vs json " + via_json.dump());
  }
  CHECK(from_flags({"--context", "max"}).context == ni::kContextMax, "max is kContextMax");
  CHECK(from_flags({"--mem-budget", "20"}).engine.mem_budget_gb == 20, "mem budget in GB");

  // refused, with the field named as its source spells it
  struct Bad {
    json config;
    const char* says;
  };
  const std::vector<Bad> bad = {
      {{{"prot", 8000}}, "unknown field \"prot\""},
      {{{"port", 0}}, "port must be 1-65535"},
      {{{"port", 65536}}, "port must be 1-65535"},
      {{{"port", "8000"}}, "port needs a whole number"},
      {{{"port", 8000.5}}, "port needs a whole number"},
      {{{"slots", -1}}, "slots needs a whole number"},
      {{{"slots", 3000000000LL}}, "slots needs a whole number"},
      {{{"spec", 18446744073709551615ULL}}, "spec needs a whole number"},
      {{{"loops", 5}}, "loops: 1 to 4"},
      {{{"loops", 0}}, "loops: 1 to 4"},
      {{{"disk_cache", "lots"}}, "disk_cache takes auto, 0 (off) or a size"},
      {{{"disk_cache", "16Q"}}, "disk_cache takes auto, 0 (off) or a size"},
      {{{"disk_cache", "16"}}, "a size with its unit such as 16G, not \"16\""},
      {{{"context", "131072"}}, "context takes tokens (131072), k or m of them (128k, 1m) or max, not \"131072\""},
      {{{"disk_cache", 16}}, "disk_cache must be a string"},
      {{{"context", "12q"}}, "context takes tokens"},
      {{{"context", 0}}, "context takes tokens"},
      {{{"context", true}}, "context takes tokens"},
      {{{"idle_unload", 0.1}}, "idle_unload takes auto"},
      {{{"idle_unload", "soon"}}, "idle_unload takes auto"},
      {{{"mem_budget", 0}}, "mem_budget needs a size in GB"},
      {{{"mem_budget", "20"}}, "mem_budget needs a size in GB"},
      {{{"timing", "TTFT"}}, "timing takes ttft, prefill or both"},
      {{{"profile", "turbo"}}, "profile takes fast or precise"},
      {{{"engine", "mlx"}}, "engine takes native or splash"},
      {{{"loop_kv", "every"}}, "loop_kv: last-loop or per-loop"},
      {{{"split", json::array()}}, "split takes the stages' host:port"},
      {{{"split", "a:1"}}, "split must be a list of strings"},
      {{{"verbose", "yes"}}, "verbose must be true or false"},
      {{{"check", true}}, "check needs engine splash"},
      {{{"engine", "splash"}, {"model", "4b"}}, "engine splash serves the splash_model folder"},
      {{{"engine", "splash"}, {"splash_kflash", "maybe"}}, "splash_kflash must be auto, on or off"},
      {{{"splash_model", "/m"}}, "need engine splash"},
      {{{"allow_switch", true}, {"host", "0.0.0.0"}}, "allow_switch on a network address needs api_key"},
      {json::array(), "must be a JSON object"},
  };
  for (const Bad& b : bad) {
    const std::string why = refusal(b.config);
    CHECK(has(why, b.says), b.config.dump() + " -> \"" + why + "\", wanted \"" + b.says + "\"");
  }
  // taken: the edges of what is allowed
  for (const json& ok : {json{{"port", 1}}, json{{"port", 65535}}, json{{"loops", 4}}, json{{"disk_cache", "0"}},
                         json{{"disk_cache", "auto"}}, json{{"idle_unload", 10080}}, json{{"allow_switch", true}},
                         json{{"allow_switch", true}, {"host", "::1"}},
                         json{{"allow_switch", true}, {"host", "0.0.0.0"}, {"api_key", "k"}}})
    CHECK(refusal(ok).empty(), ok.dump() + " -> " + refusal(ok));

  // the command line keeps its messages (tests/test_server.py checks them through the binary too)
  struct FlagBad {
    std::vector<std::string> args;
    const char* says;
  };
  const std::vector<FlagBad> flag_bad = {
      {{"--port", "0"}, "--port must be 1-65535"},
      {{"--slots", "two"}, "--slots needs a whole number, not \"two\""},
      {{"--context"}, "--context needs a value"},
      {{"--ctx", "0"}, "--context takes tokens"},
      {{"--not-a-serve-option"}, "unknown option --not-a-serve-option"},
      {{"--prefill_chunk", "8"}, "unknown option --prefill_chunk"},
      {{"--private", "folder"}, "unknown option folder"},
      {{"--disk-cache", "lots"}, "--disk-cache takes auto, 0 (off) or a size such as 16G, not \"lots\""},
      {{"--loops", "9"}, "--loops: 1 to 4"},
      {{"--mem-budget", "0"}, "--mem-budget needs a size in GB, not \"0\""},
      {{"--split", ","}, "--split takes the stages' host:port, comma-separated"},
      {{"--allow-switch", "--host", "0.0.0.0"}, "--allow-switch on a network address needs --api-key"},
      {{"--splash-kflash", "on"}, "--splash-binary, --splash-model, --splash-arg"},
  };
  for (const FlagBad& b : flag_bad) {
    const std::string why = flag_refusal(b.args);
    CHECK(has(why, b.says), b.args[0] + " -> \"" + why + "\", wanted \"" + b.says + "\"");
  }
  CHECK(!ni::server::flags_to_json({"--port", "1", "--help"}, true), "--help asks for help");
  CHECK(from_flags({"35b"}).model == "35b", "a positional model");
  CHECK((from_flags({"--split", "a:1", "--split", "b:2,c:3"}).split == std::vector<std::string>{"a:1", "b:2", "c:3"}),
        "--split given twice adds both flags' stages");

  // --config -: the JSON stays on fd 0 for a server that execs itself again (restart_server)
  {
    const std::string text = "{\"port\": 8123, \"api_key\": \"k\"}";
    const int saved = dup(STDIN_FILENO);
    ni::server::keep_stdin(text);
    char tmpl[] = "/tmp/serve_options_test-XXXXXX";
    const int out = mkstemp(tmpl);
    const std::string cmd = std::string("/bin/cat > ") + tmpl;  // an exec'd child reads what the restarted server would
    CHECK(out >= 0 && std::system(cmd.c_str()) == 0, "a child reads the kept stdin");
    std::ifstream f(tmpl);
    const std::string got((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(got == text, "the kept stdin holds the same JSON, from its start: " + got);
    if (out >= 0) close(out);
    unlink(tmpl);
    dup2(saved, STDIN_FILENO);
    close(saved);
  }
  try {
    ni::server::flags_to_json({"35b"}, false);
    CHECK(false, "a positional model where none is taken");
  } catch (const std::invalid_argument& e) { CHECK(has(e.what(), "unknown option 35b"), e.what()); }

  // the schema in docs/ names the same fields as the validator
  if (argc > 1) {
    std::ifstream f(argv[1]);
    const json schema = json::parse(f, nullptr, false);
    CHECK(!schema.is_discarded() && schema.value("additionalProperties", true) == false, "schema is strict");
    std::set<std::string> in_schema, in_code(ni::server::config_fields().begin(), ni::server::config_fields().end());
    if (!schema.is_discarded())
      for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) in_schema.insert(it.key());
    CHECK(in_schema == in_code, "docs/serve-config.schema.json and config_fields() list the same fields");
  } else CHECK(false, "pass docs/serve-config.schema.json");

  if (failures) {
    std::fprintf(stderr, "%d failed\n", failures);
    return 1;
  }
  std::printf("serve_options_test: ok\n");
  return 0;
}
