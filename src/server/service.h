#pragma once
#include "engine/engine.h"

namespace ni {
int run_cli(int argc, char** argv);
namespace server {
// Both frontends enter the same native server with typed options. The runtime
// entry point accepts JSON, never a reconstructed CLI command.
struct Options {
  EngineOptions engine;
  int context_req = 0, port = 8000;
  std::string host, api_key, model, instance_id, name;
  std::vector<std::string> split;
  bool verbose = false, yes = false, allow_switch = false, priv = false, local = false;
  bool engine_splash = false, check_only = false, headless = false, managed = false;
  std::string splash_kflash = "auto", timing = "both";
};

Options native_options(const ojson& config);
int run_service(Options options);
}  // namespace server
}  // namespace ni
