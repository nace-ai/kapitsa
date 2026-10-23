// What the command line's files share (main.cpp: serve, pull, list, rm; launch.cpp: the coding agents; bench_cmd.cpp:
// kapitsa bench; update_cmd.cpp: kapitsa update and the automatic update).
#pragma once
#include "engine/engine.h"
#include "hub/hub.h"

#include <string>

namespace ni {

// This build's version (CMakeLists.txt's KAPITSA_VERSION), as `kapitsa --version` prints it: the one copy the update
// code compares (main.cpp)
extern const char* const kVersion;

// Prints "kapitsa: <msg>" and where the usage is, and exits with 2
[[noreturn]] void usage_error(const std::string& msg, const char* help = "kapitsa --help");
// Prints "kapitsa: <msg>" and returns `code`
int fail(const std::string& msg, int code = 1);
int to_int(const std::string& opt, const std::string& v, const char* help);
std::string expand_home(const std::string& p);
double ram_gb();
// The best catalog model for this Mac (what `kapitsa serve` without --model starts); false if none can be planned
bool best_model(const EngineOptions& opt, size_t budget, hub::ModelRef* out, std::string* why);
// --model <name> without a variant: the one for this Mac; anything else as resolved
hub::ModelRef for_this_mac(hub::ModelRef ref, const EngineOptions& opt, size_t budget);
// Why the server could not listen on host:port, or ""
std::string port_problem(const std::string& host, int port);

// The executable `name` on the PATH, or ""
std::string find_in_path(const std::string& name);

// kapitsa claude|codex|opencode: starts the coding agent against a Kapitsa server (one already running on the port,
// or one it starts and stops); argv[first] is the first argument after the agent's name
int launch_agent(const std::string& name, int argc, char** argv, int first);

// kapitsa bench: benchmarks Kapitsa on this Mac with the harness's protocol, saves the result and submits it when
// asked (bench_cmd.cpp); argv[first] is the first argument after "bench"
int run_bench(int argc, char** argv, int first);

// kapitsa update: updates this kapitsa to the latest release (update_cmd.cpp); argv[first] is the first argument after
// "update"
int run_update(int argc, char** argv, int first);

// kapitsa cluster (kapitsa macs): the Macs on this network that run Kapitsa (cluster_cmd.cpp); argv[first] is the
// first argument after the command
int run_cluster(int argc, char** argv, int first);
// kapitsa stage: holds a run of Qwen3.8-Flash-Next's layers for another Mac's `kapitsa serve --split` (stage_cmd.cpp,
// src/cluster/pipe.h); argv[first] is the first argument after "stage"
int run_stage(int argc, char** argv, int first);
// Before a command runs: the line about the last automatic update (in a terminal), and once a day the next check,
// started in the background (update_cmd.cpp). Leaves help, the version, kapitsa update and kapitsa bench alone.
void auto_update(int argc, char** argv);

}  // namespace ni
