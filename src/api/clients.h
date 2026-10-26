// The coding agents `kapitsa claude|codex|opencode|pi` starts against a Kapitsa server: the command line, the
// environment and the files that point each one at the server, its model and its context window.
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ni::api {

struct AgentServer {
  std::string base_url;  // http://127.0.0.1:8000
  std::string model;     // the model id the server reports
  int context = 0;       // its context window (tokens)
  std::string api_key;   // the server's --api-key, or "" (the clients still need some key)
  // The other models it loads when a request names one (a server started with --allow-switch), each with the context
  // it would run with: Pi lists them beside `model`, and picking one there switches the server to it
  std::vector<std::pair<std::string, int>> switch_to;
};

struct AgentLaunch {
  std::vector<std::string> argv;                         // argv[0]: the client's executable
  std::vector<std::pair<std::string, std::string>> set;  // environment variables to set
  std::vector<std::string> unset;                        // and to remove
  std::vector<std::string> notes;                        // lines to show before it starts
  std::vector<std::pair<std::string, std::string>> files;  // files it reads, written before it starts (path, content)
};

bool known_agent(const std::string& name);
// How to install a client that is not on the PATH
std::string agent_install_hint(const std::string& name);
// The launch of `name` (its executable at `path`) with the user's arguments. `env`: the current environment (an
// opencode configuration in OPENCODE_CONFIG_CONTENT is kept, with the server added); `major`: the client's major
// version, -1 when unknown; `dir`: a folder of Kapitsa's for the files a client reads (Pi's provider). Throws
// std::runtime_error when OPENCODE_CONFIG_CONTENT is not a JSON object.
AgentLaunch agent_launch(const std::string& name, const std::string& path, const AgentServer& s,
                         const std::vector<std::string>& args, const std::map<std::string, std::string>& env, int major,
                         const std::string& dir = "");

}  // namespace ni::api
