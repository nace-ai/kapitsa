#include "api/clients.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>

namespace ni::api {

using ojson = nlohmann::ordered_json;

bool known_agent(const std::string& name) {
  return name == "claude" || name == "codex" || name == "opencode" || name == "pi";
}

std::string agent_install_hint(const std::string& name) {
  if (name == "claude")
    return "Claude Code is not installed (no `claude` on the PATH). Install it:\n"
           "  curl -fsSL https://claude.ai/install.sh | bash      (or: npm install -g @anthropic-ai/claude-code)";
  if (name == "codex")
    return "Codex is not installed (no `codex` on the PATH). Install it:\n"
           "  npm install -g @openai/codex      (or: brew install --cask codex)";
  if (name == "opencode")
    return "opencode is not installed (no `opencode` on the PATH). Install it:\n"
           "  curl -fsSL https://opencode.ai/install | bash      (or: npm install -g opencode-ai)";
  if (name == "pi")
    return "Pi is not installed (no `pi` on the PATH). Install it:\n"
           "  npm install -g @earendil-works/pi-coding-agent";
  return "";
}

namespace {

std::string toml_string(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

AgentLaunch claude(const std::string& path, const AgentServer& s, const std::vector<std::string>& args,
                   const std::map<std::string, std::string>& env) {
  AgentLaunch l;
  const std::string n = std::to_string(s.context);
  l.set = {{"ANTHROPIC_BASE_URL", s.base_url},
           {"KAPITSA_API_KEY", s.api_key.empty() ? "kapitsa" : s.api_key},
           {"ANTHROPIC_MODEL", s.model},
           {"ANTHROPIC_DEFAULT_OPUS_MODEL", s.model},
           {"ANTHROPIC_DEFAULT_SONNET_MODEL", s.model},
           {"ANTHROPIC_DEFAULT_HAIKU_MODEL", s.model},
           {"ANTHROPIC_DEFAULT_FABLE_MODEL", s.model},
           {"ANTHROPIC_SMALL_FAST_MODEL", s.model},
           // the window Claude Code tracks and compacts at: the server's context
           {"CLAUDE_CODE_MAX_CONTEXT_TOKENS", n},
           {"CLAUDE_CODE_AUTO_COMPACT_WINDOW", n},
           // the direct endpoint, even in a shell set up for a cloud provider
           {"CLAUDE_CODE_USE_BEDROCK", "0"},
           {"CLAUDE_CODE_USE_VERTEX", "0"},
           {"CLAUDE_CODE_USE_FOUNDRY", "0"}};
  // its replies ask for 32,000 tokens; a smaller window keeps room for the conversation
  if (s.context < 128000 && !env.count("CLAUDE_CODE_MAX_OUTPUT_TOKENS"))
    l.set.push_back({"CLAUDE_CODE_MAX_OUTPUT_TOKENS", std::to_string(std::max(4096, s.context / 4))});
  // The server's key through an apiKeyHelper (settings layered over the user's own, for this run only): Claude Code
  // then sends it, and only it, as x-api-key and as the bearer token. Otherwise it also sends the user's saved Console
  // key or claude.ai login to the local server. ANTHROPIC_AUTH_TOKEN beside a helper would add a "Both ... set" warning;
  // ANTHROPIC_API_KEY would ask whether to use a custom key.
  l.unset = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"};
  l.argv = {path, "--settings", R"({"apiKeyHelper":"printf '%s' \"$KAPITSA_API_KEY\""})", "--model", s.model,
            "--disallowedTools", "WebSearch"};
  l.argv.insert(l.argv.end(), args.begin(), args.end());
  l.notes = {"WebSearch is off: Claude Code runs it on Anthropic's servers. Other tools and MCP servers work as usual."};
  return l;
}

AgentLaunch codex(const std::string& path, const AgentServer& s, const std::vector<std::string>& args) {
  AgentLaunch l;
  l.set = {{"KAPITSA_API_KEY", s.api_key.empty() ? "kapitsa" : s.api_key}};
  const std::vector<std::string> cfg = {
      "model=" + toml_string(s.model),
      "model_provider=\"kapitsa\"",
      "model_providers.kapitsa={name=\"Kapitsa\",base_url=" + toml_string(s.base_url + "/v1") +
          ",env_key=\"KAPITSA_API_KEY\",wire_api=\"responses\"}",
      "model_context_window=" + std::to_string(s.context),
      // compact before the window is full: the rest is room for the reply and the compaction itself
      "model_auto_compact_token_limit=" + std::to_string(s.context / 10 * 9),
      "web_search=\"disabled\"",
      // Qwen's reasoning comes as reasoning text, which Codex shows only with this
      "show_raw_agent_reasoning=true"};
  l.argv = {path};
  for (const auto& c : cfg) l.argv.insert(l.argv.end(), {"-c", c});
  // the user's config overrides go with ours, before any subcommand: a subcommand's -c list replaces the root's
  std::vector<std::string> overrides, rest;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--") {
      rest.insert(rest.end(), args.begin() + long(i), args.end());
      break;
    }
    if ((a == "-c" || a == "--config") && i + 1 < args.size()) overrides.insert(overrides.end(), {"-c", args[++i]});
    else if (a.rfind("--config=", 0) == 0) overrides.insert(overrides.end(), {"-c", a.substr(9)});
    else if (a.size() > 2 && a.rfind("-c", 0) == 0 && a[2] != '-') overrides.insert(overrides.end(), {"-c", a.substr(a[2] == '=' ? 3 : 2)});
    else rest.push_back(a);
  }
  l.argv.insert(l.argv.end(), overrides.begin(), overrides.end());
  l.argv.insert(l.argv.end(), rest.begin(), rest.end());
  l.notes = {"Web search is off: Codex runs it on OpenAI's servers. Other tools and MCP servers work as usual."};
  return l;
}

ojson object_at(const ojson& j, const char* key) {
  if (!j.contains(key)) return ojson::object();
  if (!j[key].is_object()) throw std::runtime_error(std::string("OPENCODE_CONFIG_CONTENT: \"") + key + "\" must be a JSON object");
  return j[key];
}

AgentLaunch opencode(const std::string& path, const AgentServer& s, const std::vector<std::string>& args,
                     const std::map<std::string, std::string>& env, int major) {
  AgentLaunch l;
  ojson cfg = ojson::object();
  if (auto it = env.find("OPENCODE_CONFIG_CONTENT"); it != env.end() && !it->second.empty()) {
    cfg = ojson::parse(it->second, nullptr, false);
    if (!cfg.is_object()) throw std::runtime_error("OPENCODE_CONFIG_CONTENT must be a JSON object");
  }
  const std::string served = "kapitsa/" + s.model;
  cfg["model"] = served;
  cfg["small_model"] = served;
  // a model set per agent (in the user's config) outranks the top-level one: the built-in agents use this one too
  ojson agents = object_at(cfg, "agent");
  for (const char* a : {"build", "plan", "general", "explore", "title", "compaction"}) {
    ojson one = object_at(agents, a);
    one["model"] = served;
    agents[a] = one;
  }
  cfg["agent"] = agents;
  ojson providers = object_at(cfg, "provider");
  const int out = std::min(32768, std::max(1, s.context / 4));
  ojson variants = ojson::object();
  for (const char* e : {"none", "low", "medium", "high"}) variants[e] = {{"reasoningEffort", e}};
  providers["kapitsa"] = {
      {"npm", "@ai-sdk/openai-compatible"},
      {"name", "Kapitsa"},
      // no timeouts: the server sends its headers at once and a chunk every 15 s of a wait or a long prefill, which
      // resets opencode's chunkTimeout (5 min in 1.19, 15 in 2.x); its whole-request timeout has no default
      {"options", {{"baseURL", s.base_url + "/v1"}, {"apiKey", s.api_key.empty() ? "kapitsa" : s.api_key}}},
      {"models",
       {{s.model,
         {{"name", s.model},
          {"reasoning", true},  // reasoning_content streams as thinking and goes back with the turn
          {"tool_call", true},
          {"variants", variants},
          // the window holds the reply too; an input budget leaves opencode its compaction reserve
          {"limit", {{"context", s.context}, {"input", std::max(1, s.context - out)}, {"output", out}}}}}}}};
  cfg["provider"] = providers;
  l.set = {{"OPENCODE_CONFIG_CONTENT", cfg.dump()}};
  l.argv = {path};
  l.argv.insert(l.argv.end(), args.begin(), args.end());
  // opencode 2 reads its configuration in a background service that never sees this environment; a private server
  // (--standalone, a flag of the command, so after any subcommand and before "--") reads it. Not when the user picked
  // a server.
  bool chosen = false;
  for (const auto& a : args) {
    if (a == "--") break;
    chosen = chosen || a == "--standalone" || a == "--server" || a.rfind("--server=", 0) == 0;
  }
  if (major >= 2 && !chosen) {
    auto dd = std::find(l.argv.begin() + 1, l.argv.end(), std::string("--"));
    l.argv.insert(dd, "--standalone");
  }
  return l;
}

// Pi's provider for the server: an extension that `kapitsa pi` loads for that run (pi -e), so nothing in ~/.pi
// changes. It reads the server from the environment, so the file stays the same and runs on other ports do not race.
// Its models: the one served and, on a server `kapitsa pi` started (--allow-switch), every other downloaded one, each
// with its context; picking one in Pi switches the server to it (src/server/model_manager.h, switch_to).
// Anthropic Messages: each turn's reasoning goes back exactly (in our signature; a turn cut short keeps its thinking,
// with an empty signature), so every request of a tool loop continues the prompt the server has cached. Pi's thinking
// levels go as efforts (adaptive thinking), which the server maps for the models that have levels.
const char* const kPiExtension = R"JS(// Kapitsa as a provider for Pi, written by `kapitsa pi`, which loads it for one run (pi -e): the server it started
// or found, from the environment it gives Pi (KAPITSA_PI_BASE_URL, KAPITSA_PI_MODEL, KAPITSA_PI_CONTEXT, KAPITSA_API_KEY;
// KAPITSA_PI_MODELS: every model the server serves or switches to, [{"id", "context"}], the served one first).
export default function (pi) {
  const env = process.env;
  if (!env.KAPITSA_PI_BASE_URL || !env.KAPITSA_PI_MODEL) return;
  let list = [{ id: env.KAPITSA_PI_MODEL, context: env.KAPITSA_PI_CONTEXT }];
  try {
    const all = JSON.parse(env.KAPITSA_PI_MODELS || "[]");
    if (Array.isArray(all) && all.length) list = all;
  } catch {}
  pi.registerProvider("kapitsa", {
    name: "Kapitsa",
    baseUrl: env.KAPITSA_PI_BASE_URL,
    api: "anthropic-messages",
    apiKey: "$KAPITSA_API_KEY",
    models: list.map((m) => {
      const context = Number(m.context) || 32768;
      return {
        id: m.id,
        name: m.id + " (Kapitsa)",
        reasoning: true,
        input: ["text", "image"],
        contextWindow: context,
        maxTokens: Math.min(32768, Math.max(1024, Math.floor(context / 4))),
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: { forceAdaptiveThinking: true, allowEmptySignature: true },
      };
    }),
  });
}
)JS";

AgentLaunch pi(const std::string& path, const AgentServer& s, const std::vector<std::string>& args, const std::string& dir) {
  AgentLaunch l;
  const std::string ext = (dir.empty() ? std::string(".") : dir) + "/pi-kapitsa.js";
  l.files = {{ext, kPiExtension}};
  l.set = {{"KAPITSA_API_KEY", s.api_key.empty() ? "kapitsa" : s.api_key},
           {"KAPITSA_PI_BASE_URL", s.base_url},
           {"KAPITSA_PI_MODEL", s.model},
           {"KAPITSA_PI_CONTEXT", std::to_string(s.context)}};
  ojson models = ojson::array({{{"id", s.model}, {"context", s.context}}});
  for (const auto& [id, context] : s.switch_to) models.push_back({{"id", id}, {"context", context}});
  l.set.push_back({"KAPITSA_PI_MODELS", models.dump()});
  if (!s.switch_to.empty())
    l.notes.push_back("Pi lists the other downloaded models too: picking one (/model) loads it in place of " + s.model +
                      ", which takes a moment.");
  l.argv = {path, "-e", ext, "--provider", "kapitsa", "--model", s.model};
  l.argv.insert(l.argv.end(), args.begin(), args.end());
  return l;
}

}  // namespace

AgentLaunch agent_launch(const std::string& name, const std::string& path, const AgentServer& s,
                         const std::vector<std::string>& args, const std::map<std::string, std::string>& env, int major,
                         const std::string& dir) {
  if (name == "claude") return claude(path, s, args, env);
  if (name == "codex") return codex(path, s, args);
  if (name == "opencode") return opencode(path, s, args, env, major);
  if (name == "pi") return pi(path, s, args, dir);
  throw std::runtime_error("unknown coding agent: " + name);
}

}  // namespace ni::api
