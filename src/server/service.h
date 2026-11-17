// `kapitsa serve`'s settings as one typed value, and the one validator every way of giving them goes through: the
// command line's flags (converted to the same JSON first) and `kapitsa serve --config -|FILE` (the JSON itself). The
// schema of that JSON is docs/serve-config.schema.json; serve_options_test checks the two name the same fields.
#pragma once
#include "engine/engine.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ni::server {

struct Options {
  // model_id, slots, spec_depth, prefill_chunk, dflash, mem_budget_gb, image_max_pixels, loops, loop_kv, disk_cache
  // and the splash_* fields, with EngineOptions' own defaults; the server fills in the rest (model_dir, context)
  EngineOptions engine;
  std::string model;                   // a catalog name, a folder or a GGUF file; "" picks the best for this Mac
  std::string host;                    // "": every address, or 127.0.0.1 with private
  int port = 8000;
  int context = 0;                     // 0 none asked (default_context), kContextMax, or tokens
  std::string api_key;
  std::string profile;                 // "", fast or precise (KAPITSA_PROFILE)
  std::string timing = "both";         // ttft, prefill or both
  double idle_unload_min = -1;         // -1 auto, 0 off, else minutes
  std::vector<std::string> split;      // the stages' host:port, in the order of their layers
  std::vector<std::string> peers;      // other Macs' servers where Bonjour does not reach
  bool engine_splash = false, check_only = false;
  std::string splash_kflash = "auto";  // auto, on or off
  bool priv = false, local = false, allow_switch = false, verbose = false, yes = false;
  nlohmann::json given = nlohmann::json::object();  // the validated JSON without api_key: the control socket's `config`
};

// The JSON fields the validator takes, in the schema's order: each is a serve flag's name with '_' for '-'
const std::vector<std::string>& config_fields();

// Where a configuration came from: errors name the flag (--disk-cache) or the JSON field (disk_cache)
enum class Source { flags, json };

// The one validator: unknown fields, wrong types, ranges, sizes and combinations that cannot run are refused with
// std::invalid_argument, whose message names the field as `source` spells it.
Options parse_options(const nlohmann::json& config, Source source = Source::json);

// The command line's arguments as that JSON, in order (a later flag wins; --peer and --splash-arg add to a list). Only
// the flags' syntax is checked here (a value given, a whole number where one is due); everything else is
// parse_options'. nullopt: --help was asked for. `positional_model`: a first argument that is not a flag is the model
// (`kapitsa serve 35b`).
std::optional<nlohmann::json> flags_to_json(const std::vector<std::string>& args, bool positional_model);

// Serves with these options until stopped (main.cpp: everything `kapitsa serve` did after reading its flags)
int run_service(Options options);

// --config -: puts the JSON read from stdin back on fd 0, as an unlinked 0600 file at its start, so a server that
// starts itself again (pairing, a split: restart_server() re-execs `serve --config -`) reads the same JSON. The key in
// it never goes in argv.
void keep_stdin(const std::string& text);

// --config's JSON: "-" reads stdin (and keeps it, keep_stdin), else the file at that path. Its contents are never
// echoed (they may hold the API key).
nlohmann::json read_config(const std::string& from);

}  // namespace ni::server
