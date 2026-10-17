// kapitsa: Kapitsa's command line, and its OpenAI-compatible server (chat completions, streaming, tools) with a
// chat page at /.
//   kapitsa serve [--model NAME | org/name[@rev] | DIR] [--model-id ID] [--host 127.0.0.1] [--port 8000]
//                 [--ctx 32768] [--slots 2] [--spec N] [--dflash auto|none|<drafter dir>] [--prefill-chunk 2048]
//                 [--api-key K] [--mem-budget GB]    no --model: the best catalog model for this Mac's memory
//   kapitsa doctor [--model MODEL]... [--ctx N] [--slots N] [--dflash auto|none|<dir>] [--json]   this Mac, what fits
//   kapitsa pull MODEL... [--from DIR]... [--dry-run [--json]]
//   kapitsa list [--json]
//   kapitsa rm MODEL...
//   kapitsa mtp-sidecar SRC OUT [--revision REV] [--hidden post_norm|pre_norm]   (tools/mtp_sidecar.py, same bytes)
// ("serve" is optional for options-only command lines; build/nace-infer-server is a symlink kept for old scripts.)
#include <cstring>
#include "engine/doctor.h"
#include "engine/engine.h"
#include "hub/hub.h"
#include "metal/gpu.h"
#include "util/log.h"

#include <httplib/httplib.h>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <thread>
#include <csignal>
#include <ctime>

#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.1.0"
#endif
#ifndef KAPITSA_GIT
#define KAPITSA_GIT ""
#endif

namespace ni {
extern const char* const kChatPage;  // src/server/chat.html, embedded at build time
}

using namespace ni;
using json = nlohmann::ordered_json;

static json normalize_messages(const json& in) {
  json out = json::array();
  for (auto m : in) {
    if (m.value("role", "") == "developer") m["role"] = "system";
    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
      for (auto& tc : m["tool_calls"]) {
        if (tc.contains("function") && tc["function"].contains("arguments") && tc["function"]["arguments"].is_string()) {
          try {
            tc["function"]["arguments"] = json::parse(tc["function"]["arguments"].get<std::string>());
          } catch (...) {
          }
        }
      }
    }
    if (m.contains("content") && m["content"].is_null()) m["content"] = "";
    out.push_back(m);
  }
  return out;
}

static ChatRequest parse_request(const json& b) {
  ChatRequest r;
  r.messages = normalize_messages(b.at("messages"));
  if (b.contains("tools") && b["tools"].is_array()) r.tools = b["tools"];
  if (b.contains("tool_choice") && b["tool_choice"].is_string() && b["tool_choice"] == "none") r.tools = json::array();
  r.max_tokens = b.value("max_completion_tokens", b.value("max_tokens", 16384));
  if (b.contains("temperature") && b["temperature"].is_number()) r.temperature = b["temperature"];
  if (b.contains("top_p") && b["top_p"].is_number()) r.top_p = b["top_p"];
  if (b.contains("top_k") && b["top_k"].is_number()) r.top_k = b["top_k"];
  if (b.contains("seed") && b["seed"].is_number()) r.seed = uint32_t(b["seed"].get<int64_t>());
  if (b.contains("stop")) {
    if (b["stop"].is_string()) r.stop.push_back(b["stop"]);
    else if (b["stop"].is_array())
      for (auto& s : b["stop"]) r.stop.push_back(s);
  }
  if (b.contains("chat_template_kwargs") && b["chat_template_kwargs"].is_object())
    r.template_kwargs = b["chat_template_kwargs"];
  if (b.contains("reasoning_effort") && b["reasoning_effort"].is_string()) {
    std::string e = b["reasoning_effort"];
    if (e == "none") r.template_kwargs["enable_thinking"] = false;
    else if (!r.template_kwargs.contains("reasoning_effort")) r.template_kwargs["reasoning_effort"] = e;
  }
  return r;
}

static json usage_json(const ChatResult& res) {
  return json{{"prompt_tokens", res.prompt_tokens},
              {"completion_tokens", res.completion_tokens},
              {"total_tokens", res.prompt_tokens + res.completion_tokens},
              {"prompt_tokens_details", {{"cached_tokens", res.cached_tokens}}}};
}

static json tool_calls_json(const std::vector<ToolCall>& tcs, size_t index0, bool with_index) {
  json a = json::array();
  for (size_t i = 0; i < tcs.size(); ++i) {
    json t{{"id", tcs[i].id}, {"type", "function"}, {"function", {{"name", tcs[i].name}, {"arguments", tcs[i].arguments}}}};
    if (with_index) t["index"] = index0 + i;
    a.push_back(t);
  }
  return a;
}

// ---------------------------------------------------------------- command line

[[noreturn]] static void usage_error(const std::string& msg, const char* help = "kapitsa --help") {
  fprintf(stderr, "kapitsa: %s\n(%s shows how to use it.)\n", msg.c_str(), help);
  exit(2);
}

static int fail(const std::string& msg, int code = 1) {
  fprintf(stderr, "kapitsa: %s\n", msg.c_str());
  return code;
}

static int to_int(const std::string& opt, const std::string& v, const char* help) {
  char* end = nullptr;
  errno = 0;
  const long x = strtol(v.c_str(), &end, 10);
  if (v.empty() || *end || errno || x < 0 || x > INT32_MAX) usage_error(opt + " needs a whole number, not \"" + v + "\"", help);
  return int(x);
}

static std::string expand_home(const std::string& p) {
  const char* h = getenv("HOME");
  if (h && (p == "~" || p.rfind("~/", 0) == 0)) return h + p.substr(1);
  return p;
}

static double ram_gb() {
  uint64_t m = 0;
  size_t n = sizeof m;
  sysctlbyname("hw.memsize", &m, &n, nullptr, 0);
  return double(m) / 1073741824.0;
}

static double to_gb(const std::string& opt, const std::string& v, const char* help) {
  char* end = nullptr;
  const double x = strtod(v.c_str(), &end);
  if (v.empty() || *end || !(x > 0)) usage_error(opt + " needs a size in GB, not \"" + v + "\"", help);
  return x;
}

// The best catalog model for this Mac: the first (the catalog is best first) with a variant whose memory plan, the
// engine's own, leaves room for one conversation of --ctx tokens within the budget and keeps its drafter (each model
// takes its default variant, else the first of its fallbacks that fits: pick_variant). A model not downloaded yet is
// planned from its headers (hub::plan_dir: a few hundred KB, fetched once); one that cannot be planned (offline)
// cannot be downloaded either and is skipped. False if none can be planned.
static bool best_model(const EngineOptions& opt, size_t budget, hub::ModelRef* out, std::string* why) {
  bool have = false;
  for (const auto& m : hub::catalog()) {
    bool fits = false;
    hub::ModelRef r = pick_variant(m, opt, budget, &fits);
    if (fits) {
      *out = r;
      return true;
    }
    if (variant_fits(r, opt, budget) == 0) *out = r, have = true;  // too big: the smallest such is the last resort
  }
  if (!have && why) *why = "Can't reach huggingface.co to see which model fits this Mac (offline?), and none is "
                           "downloaded. Connect to the internet, or pass --model with a model folder.";
  return have;
}

// --model <name> without a variant: the one for this Mac (pick_variant); anything else as resolved.
static hub::ModelRef for_this_mac(hub::ModelRef ref, const EngineOptions& opt, size_t budget) {
  if (ref.kind != hub::ModelRef::kCatalog || ref.variant_given) return ref;
  return pick_variant(*ref.model, opt, budget);
}

static std::string models_table() {
  std::string s;
  char line[256];
  for (const auto& m : hub::catalog()) {
    const hub::Variant* d = m.variant();
    snprintf(line, sizeof line, "  %-16s %-6s %5.1f GB  %s\n", m.name.c_str(), ("(" + hub::short_name(m) + ")").c_str(),
             d ? d->download_gb : 0.0, m.about.c_str());
    s += line;
    std::string more;
    for (const auto& v : m.variants)
      if (&v != d) {
        snprintf(line, sizeof line, "%s:%s %.1f GB", more.empty() ? "" : ", ", v.name.c_str(), v.download_gb);
        more += line;
      }
    if (d) s += "                           " + d->name + " (" + d->quant + ")";
    if (!m.fallbacks.empty()) {
      std::string fb;
      for (const auto& f : m.fallbacks) fb += (fb.empty() ? "" : ", then ") + f;
      s += "; smaller Macs: " + fb;
    }
    if (d) s += "\n";
    if (!more.empty()) s += "                           also " + more + "\n";
  }
  return s;
}

static void print_help() {
  printf("Kapitsa %s: fast local Qwen on Apple silicon.\n\n"
         "  kapitsa serve                 start the best model for this Mac (downloads it the first time)\n"
         "  kapitsa serve --model 27b     start a given model\n\n"
         "Then open http://127.0.0.1:8000 to chat, or point any OpenAI-compatible app at\n"
         "http://127.0.0.1:8000/v1.\n\n"
         "Models (with the files that speed them up; <model>:<variant> picks another precision):\n%s"
         "  or a Hugging Face repo of a Qwen3.5-family model in MLX format, bf16 or 3-8-bit (org/name[@revision]),\n"
         "  or a model folder.\n\n"
         "Commands:\n"
         "  serve        start the server: chat page at /, OpenAI API at /v1 (kapitsa serve --help)\n"
         "  pull MODEL   download a model without starting it (kapitsa pull --help)\n"
         "  list         downloaded models, their size and what they include\n"
         "  rm MODEL     delete a downloaded model\n"
         "  doctor       this Mac's GPU and memory, and which models fit it (kapitsa doctor --help)\n\n"
         "Models are stored in %s (KAPITSA_HOME moves them). HF_TOKEN is used when set.\n",
         KAPITSA_VERSION, models_table().c_str(), hub::tilde(hub::models_dir()).c_str());
}

static void print_serve_help() {
  printf("kapitsa serve [--model MODEL] [options]\n\n"
         "  --model MODEL          35b, 27b, 4b (full names: kapitsa --help; <model>:<variant> for another\n"
         "                         precision), org/name[@revision], or a model folder. Default: the best model\n"
         "                         for this Mac. Downloaded on first use.\n"
         "  --host HOST            listen address (default 127.0.0.1; 0.0.0.0 serves your network)\n"
         "  --port PORT            default 8000\n"
         "  --api-key KEY          require \"Authorization: Bearer KEY\" (the chat page asks for it once)\n"
         "  --model-id ID          the model's name in the API (default: the model's name)\n"
         "  --ctx TOKENS           context per conversation (default 32768)\n"
         "  --slots N              conversations kept ready for instant follow-ups (default 2)\n"
         "  --dflash auto|none|DIR DFlash2 drafter: auto (default) takes the one downloaded with the model;\n"
         "                         none decodes with the model's MTP head\n"
         "  --spec N               MTP draft depth without a drafter (default per model; 0 = no speculation)\n"
         "  --mem-budget GB        GPU memory Kapitsa may hold (default: what macOS recommends for this GPU, a\n"
         "                         little less; kapitsa doctor shows it)\n"
         "  --prefill-chunk N      prompt tokens per prefill pass (default 2048, 1024 when memory is short)\n\n"
         "Endpoints: GET / (chat page), POST /v1/chat/completions, GET /v1/models, GET /health.\n"
         "Tuning switches (NI_* variables): DEVELOPMENT.md.\n");
}

static void print_pull_help() {
  printf("kapitsa pull MODEL... [--from DIR]...\n\n"
         "Downloads a model with the files that speed it up (MTP head, DFlash2 drafter) into\n"
         "%s/<name>-<variant>, verifies every file, and continues an interrupted download.\n\n"
         "  --from DIR   take files that are already on this Mac from DIR (a folder from `hf download --local-dir`,\n"
         "               another app's models): only exact matches of the pinned checksums, as APFS clones\n"
         "  --dry-run    show what MODEL resolves to, without downloading (--json for programs)\n\n"
         "Files already in the Hugging Face cache (~/.cache/huggingface/hub) are reused the same way.\n",
         hub::tilde(hub::models_dir()).c_str());
}

// Why the server could not listen on host:port, or "" (a test bind, as the server binds: SO_REUSEADDR only).
static std::string port_problem(const std::string& host, int port) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  const std::string p = std::to_string(port);
  if (getaddrinfo(host.c_str(), p.c_str(), &hints, &res) != 0 || !res) return "--host " + host + " is not an address of this Mac.";
  std::string why;
  const int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd >= 0) {
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(fd, res->ai_addr, res->ai_addrlen) != 0) {
      if (errno == EADDRINUSE)
        why = "Port " + p + " is already in use (another kapitsa or another server). Stop it, or pass --port " +
              std::to_string(port + 1) + ".";
      else if (errno == EACCES)
        why = "Not allowed to listen on port " + p + ". Pass a port above 1024, e.g. --port 8000.";
      else if (errno == EADDRNOTAVAIL)
        why = "--host " + host + " is not an address of this Mac.";
      else
        why = "Can't listen on " + host + ":" + p + " (" + strerror(errno) + ").";
    }
    close(fd);
  }
  freeaddrinfo(res);
  return why;
}

static int serve(int argc, char** argv, int first) {
  EngineOptions opt;
  std::string host = "127.0.0.1", api_key, model;
  int port = 8000;
  const char* help = "kapitsa serve --help";
  for (int i = first; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage_error(a + " needs a value", help);
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      print_serve_help();
      return 0;
    } else if (a == "--model" || a == "-m") model = next();
    else if (a == "--model-id") opt.model_id = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = to_int(a, next(), help);
    else if (a == "--ctx") opt.context = to_int(a, next(), help);
    else if (a == "--slots") opt.slots = to_int(a, next(), help);
    else if (a == "--spec") opt.spec_depth = to_int(a, next(), help);
    else if (a == "--dflash") opt.dflash = next();
    else if (a == "--prefill-chunk") opt.prefill_chunk = to_int(a, next(), help);
    else if (a == "--api-key") api_key = next();
    else if (a == "--mem-budget") opt.mem_budget_gb = to_gb(a, next(), help);
    else if (i == first && first == 2 && !a.empty() && a[0] != '-') model = a;  // kapitsa serve 35b
    else usage_error("unknown option " + a, help);
  }
  if (port < 1 || port > 65535) usage_error("--port must be 1-65535", help);

  // the model: a name, a repo or a folder; downloaded (or completed) first if needed
  const size_t budget = opt.mem_budget_gb > 0 ? size_t(opt.mem_budget_gb * 1e9) : default_mem_budget();
  hub::ModelRef ref;
  try {
    if (model.empty()) {
      std::string why;
      if (!best_model(opt, budget, &ref, &why)) return fail(why);
      fprintf(stderr, "No --model given: %s is the best fit for this Mac (%s, %.0f GB). --model picks another.\n",
              ref.name.c_str(), Gpu::get().name().c_str(), ram_gb());
    } else {
      const hub::ModelRef named = hub::resolve(expand_home(model));
      ref = named;
      if (named.kind == hub::ModelRef::kCatalog && !named.variant_given) {
        bool fits = false;
        ref = pick_variant(*named.model, opt, budget, &fits);
        if (fits && ref.variant != named.variant)
          fprintf(stderr, "%s on this Mac (%.0f GB): %s, the largest of its variants that fits (the default, %s, does "
                  "not). --model %s:<variant> picks another.\n", model.c_str(), ram_gb(), ref.name.c_str(),
                  named.variant->name.c_str(), hub::short_name(*ref.model).c_str());
      }
    }
  } catch (const hub::Error& e) {
    return fail(e.what(), 2);
  }
  if (ref.kind == hub::ModelRef::kCatalog && !hub::is_current(ref)) {
    // before a long download: the engine's own start check on the model's headers; a model that would not start
    // here is refused, with the one that fits
    if (const std::string dir = hub::plan_dir(ref); !dir.empty()) {
      const LoadPlan lp = plan_load(dir, opt.dflash, opt.spec_depth == 0, opt.context, opt.slots, budget, opt.prefill_chunk);
      const StartNeed need = start_need(dir, lp, lp.prefill_chunk, opt.context, opt.slots);
      if (need.bytes > budget) {
        // a variant of the same model that fits, else the best model that does
        bool fits = false;
        hub::ModelRef alt = pick_variant(*ref.model, opt, budget, &fits);
        if (!fits && !(best_model(opt, budget, &alt, nullptr) && variant_fits(alt, opt, budget) == 1)) alt = {};
        const bool other = alt.variant && alt.variant != ref.variant;
        char msg[768];
        if (need.weights > budget)  // nothing but another variant helps
          snprintf(msg, sizeof msg, "%s does not fit this Mac: its weights alone (%.1f GB, %s) exceed the %.1f GB of GPU "
                   "memory Kapitsa may use here (%.0f GB of RAM).%s%s", ref.name.c_str(), need.weights / 1e9,
                   need.formats.c_str(), budget / 1e9, ram_gb(), other ? " This one fits: kapitsa serve --model " : "",
                   other ? hub::short_name(*alt.model, *alt.variant).c_str() : "");
        else
          snprintf(msg, sizeof msg, "%s needs %.1f GB of GPU memory with %s, and this Mac (%.0f GB) allows %.1f GB, so it "
                   "would not start.%s%s", ref.name.c_str(), need.bytes / 1e9, need.parts.c_str(), ram_gb(),
                   budget / 1e9, other ? " This one fits: kapitsa serve --model " : "",
                   other ? hub::short_name(*alt.model, *alt.variant).c_str() : "");
        return fail(msg);
      }
    }
  }
  if (ref.kind == hub::ModelRef::kLocal) {
    struct stat st;
    if (stat((ref.dir + "/config.json").c_str(), &st) != 0)
      return fail(ref.dir + " has no config.json. --model takes a model folder (config.json, *.safetensors, "
                  "tokenizer.json), a model name (kapitsa --help) or a Hugging Face repo.", 2);
  }
  // fail now, not after loading the model, when the port is taken
  if (const std::string why = port_problem(host, port); !why.empty()) return fail(why);
  try {
    opt.model_dir = hub::ensure(ref, hub::PullOptions{});
  } catch (const hub::Stopped& e) {
    fprintf(stderr, "\n%s\n", e.what());
    return 130;
  } catch (const hub::Error& e) {
    return fail(e.what());
  }
  if (opt.model_id.empty()) opt.model_id = ref.id;
  // the server compiles the Neural Engine program in the background (no 40 s start after the ANE
  // daemon evicted it); ni_run and the benchmarks build it synchronously
  setenv("NI_ANE_ASYNC", "1", 0);
  Engine engine(opt);

  httplib::Server srv;
  srv.set_payload_max_length(64u << 20);
  // Stream events go out as written: with Nagle's algorithm a small SSE chunk waits for the ACK of
  // the previous one (the client's delayed ACK), which can hold the first token of a fast request.
  srv.set_tcp_nodelay(!getenv("NI_TCP_NODELAY") || env_on("NI_TCP_NODELAY"));
  srv.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
    // the chat page needs no key (it asks for one and sends it with its API calls), nor does /health
    if (api_key.empty() || req.path == "/health" || (req.method == "GET" && (req.path == "/" || req.path == "/favicon.ico")))
      return httplib::Server::HandlerResponse::Unhandled;
    if (req.get_header_value("Authorization") == "Bearer " + api_key) return httplib::Server::HandlerResponse::Unhandled;
    res.status = 401;
    res.set_content(R"({"error":{"message":"invalid api key","type":"auth_error"}})", "application/json");
    return httplib::Server::HandlerResponse::Handled;
  });
  srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
  });
  // the chat page: one embedded file, no requests anywhere but this server's API
  srv.Get("/", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Content-Security-Policy",
                   "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; "
                   "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'");
    res.set_header("X-Content-Type-Options", "nosniff");
    res.set_header("Referrer-Policy", "no-referrer");
    res.set_header("Cache-Control", "no-cache");
    res.set_content(kChatPage, "text/html; charset=utf-8");
  });
  srv.Get("/favicon.ico", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
  srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    json j{{"object", "list"},
           {"data", json::array({json{{"id", engine.model_id()},
                                      {"object", "model"},
                                      {"owned_by", "nace"},
                                      {"context_length", engine.context_length()},
                                      // tokens one conversation can hold within this Mac's memory budget
                                      {"context_fit", engine.context_fit()}}})}};
    res.set_content(j.dump(), "application/json");
  });
  srv.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
    json body;
    ChatRequest r;
    try {
      body = json::parse(req.body);
      r = parse_request(body);
    } catch (const std::exception& ex) {
      res.status = 400;
      res.set_content(json{{"error", {{"message", std::string("bad request: ") + ex.what()}, {"type", "invalid_request_error"}}}}.dump(),
                      "application/json");
      return;
    }
    const std::string id = "chatcmpl-" + std::to_string(std::time(nullptr)) + std::to_string(rand() % 100000);
    const long created = long(std::time(nullptr));
    const std::string model = engine.model_id();
    if (!body.value("stream", false)) {
      ChatResult out = engine.chat(r, [](const ChatDelta&) { return true; });
      std::thread([&engine, r, out] { engine.prefill_history(r, out); }).detach();
      if (out.finish_reason == "error") {
        res.status = 400;
        res.set_content(json{{"error", {{"message", out.error}, {"type", "invalid_request_error"}}}}.dump(), "application/json");
        return;
      }
      json msg{{"role", "assistant"}, {"content", out.content}};
      if (!out.reasoning.empty()) msg["reasoning_content"] = out.reasoning;
      if (!out.tool_calls.empty()) msg["tool_calls"] = tool_calls_json(out.tool_calls, 0, false);
      json j{{"id", id},
             {"object", "chat.completion"},
             {"created", created},
             {"model", model},
             {"choices", json::array({json{{"index", 0}, {"message", msg}, {"finish_reason", out.finish_reason}}})},
             {"usage", usage_json(out)}};
      res.set_content(j.dump(), "application/json");
      return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream", [=, &engine](size_t, httplib::DataSink& sink) {
      auto send = [&](const json& delta, const json& finish, const json* usage) {
        json c{{"id", id},
               {"object", "chat.completion.chunk"},
               {"created", created},
               {"model", model},
               {"choices", json::array({json{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
        if (usage) c["usage"] = *usage;
        std::string s = "data: " + c.dump() + "\n\n";
        return sink.write(s.data(), s.size());
      };
      if (!send(json{{"role", "assistant"}, {"content", ""}}, nullptr, nullptr)) return false;
      size_t ntool = 0;
      ChatResult out = engine.chat(r, [&](const ChatDelta& d) {
        bool ok = true;
        if (!d.reasoning.empty()) ok = ok && send(json{{"reasoning_content", d.reasoning}}, nullptr, nullptr);
        if (!d.content.empty()) ok = ok && send(json{{"content", d.content}}, nullptr, nullptr);
        if (!d.tool_calls.empty()) {
          ok = ok && send(json{{"tool_calls", tool_calls_json(d.tool_calls, ntool, true)}}, nullptr, nullptr);
          ntool += d.tool_calls.size();
        }
        return ok;
      });
      if (out.finish_reason == "error") {
        std::string s = "data: " + json{{"error", {{"message", out.error}, {"type", "invalid_request_error"}}}}.dump() + "\n\n";
        sink.write(s.data(), s.size());
      } else if (out.finish_reason != "cancelled") {
        json u = usage_json(out);
        send(json::object(), out.finish_reason, &u);
      }
      std::string done = "data: [DONE]\n\n";
      sink.write(done.data(), done.size());
      sink.done();
      // the reply is out: prefill the history as the next turn will render it (NI_HISTORY_PREFILL)
      std::thread([&engine, r, out] { engine.prefill_history(r, out); }).detach();
      return true;
    });
  });

  // A busy port is an error: httplib's default SO_REUSEPORT would let a second server bind it too.
  srv.set_socket_options([](socket_t sock) {
    int one = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  });
  if (!srv.bind_to_port(host, port))
    return fail("Can't listen on " + host + ":" + std::to_string(port) + ": " + port_problem(host, port) +
                " Pass another --port.");
  logi("listening on http://%s:%d", host.c_str(), port);
  const std::string shown = host == "0.0.0.0" || host == "::" ? "127.0.0.1" : host;
  const std::string base =
      "http://" + (shown.find(':') != std::string::npos ? "[" + shown + "]" : shown) + ":" + std::to_string(port);
  printf("Ready: chat at %s, API at %s/v1 (model \"%s\"%s). Ctrl+C stops.\n", base.c_str(), base.c_str(),
         engine.model_id().c_str(), api_key.empty() ? "" : ", API key required");
  fflush(stdout);
  srv.listen_after_bind();
  return 0;
}

static void print_plan(const hub::ModelRef& r, bool as_json) {
  static const char* kinds[] = {"catalog", "repo", "local"};
  const bool local = r.kind == hub::ModelRef::kLocal;
  json j{{"model", r.name}, {"kind", kinds[r.kind]}, {"id", r.id}, {"dir", r.dir}};
  if (!local) {
    j["repo"] = r.repo;
    j["revision"] = r.revision.empty() ? "main" : r.revision;
    j["downloaded"] = hub::is_current(r);
  }
  json comp = json::array();
  if (r.variant) {
    j["variant"] = r.variant->name;
    j["quant"] = r.variant->quant;
    for (const auto& c : r.variant->companions)
      comp.push_back(json{{"kind", c.kind}, {"repo", c.repo}, {"revision", c.revision}, {"into", c.subdir.empty() ? "." : c.subdir}});
  }
  j["companions"] = comp;
  if (as_json) {
    printf("%s\n", j.dump().c_str());
    return;
  }
  printf("%s\n", r.name.c_str());
  printf("  folder:     %s%s\n", hub::tilde(r.dir).c_str(),
         local ? " (yours, used as is)" : hub::is_current(r) ? " (downloaded)" : " (not downloaded yet)");
  if (!local) printf("  weights:    %s @ %s\n", r.repo.c_str(), r.revision.empty() ? "main" : r.revision.c_str());
  if (r.variant) printf("  variant:    %s (%s)%s\n", r.variant->name.c_str(), r.variant->quant.c_str(),
                        r.variant->name == r.model->default_variant ? ", the default" : "");
  if (r.variant)
    for (const auto& c : r.variant->companions)
      printf("  %-11s %s @ %s%s\n", (c.kind == "dflash2" ? "drafter:" : "MTP head:"), c.repo.c_str(), c.revision.c_str(),
             c.kind == "mtp-sidecar" ? " (its mtp.* tensors, made into mtp.safetensors)" : "");
  printf("  API name:   %s\n", r.id.c_str());
}

static int pull(int argc, char** argv) {
  std::vector<std::string> models;
  hub::PullOptions po;
  bool dry = false, as_json = false;
  const char* help = "kapitsa pull --help";
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      print_pull_help();
      return 0;
    } else if (a == "--from") {
      if (i + 1 >= argc) usage_error("--from needs a folder", help);
      po.from.push_back(expand_home(argv[++i]));
    } else if (a == "--dry-run") {
      dry = true;
    } else if (a == "--json") {
      as_json = true;
    } else if (!a.empty() && a[0] == '-') {
      usage_error("unknown option " + a, help);
    } else {
      models.push_back(a);
    }
  }
  if (models.empty()) usage_error("which model? e.g. kapitsa pull 35b", help);
  EngineOptions eo;  // serve's defaults: one 32k conversation, 2 slots
  for (const auto& name : models) {
    hub::ModelRef ref;
    try {
      ref = for_this_mac(hub::resolve(expand_home(name)), eo, default_mem_budget());
    } catch (const hub::Error& e) {
      return fail(e.what(), 2);
    }
    if (dry) {
      print_plan(ref, as_json);
      continue;
    }
    if (ref.kind == hub::ModelRef::kLocal) {
      printf("%s is a folder on this Mac: nothing to download.\n", ref.dir.c_str());
      continue;
    }
    const bool had = hub::is_current(ref);
    try {
      hub::ensure(ref, po);
    } catch (const hub::Stopped& e) {
      fprintf(stderr, "\n%s\n", e.what());
      return 130;
    } catch (const hub::Error& e) {
      return fail(e.what());
    }
    printf("%s %s in %s. Start it: kapitsa serve --model %s\n", ref.name.c_str(),
           had ? "is already downloaded" : "is ready", hub::tilde(ref.dir).c_str(),
           ref.model ? hub::short_name(*ref.model, *ref.variant).c_str() : ref.name.c_str());
  }
  return 0;
}

static int list(int argc, char** argv) {
  bool as_json = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") as_json = true;
    else if (a == "--help" || a == "-h") {
      printf("kapitsa list [--json]: the models in %s, their size on disk and what they include.\n",
             hub::tilde(hub::models_dir()).c_str());
      return 0;
    } else usage_error("unknown option " + a);
  }
  const auto v = hub::installed();
  if (as_json) {
    json a = json::array();
    for (const auto& m : v)
      a.push_back(json{{"name", m.name}, {"variant", m.variant}, {"dir", m.dir}, {"repo", m.repo}, {"commit", m.commit},
                       {"complete", m.complete}, {"bytes", m.bytes}, {"includes", m.extras}});
    printf("%s\n", a.dump(2).c_str());
    return 0;
  }
  if (v.empty()) {
    printf("No models downloaded yet (%s).\n", hub::tilde(hub::models_dir()).c_str());
  } else {
    printf("Models in %s:\n", hub::tilde(hub::models_dir()).c_str());
    for (const auto& m : v) {
      std::string what = "weights";
      for (const auto& e : m.extras) what += " + " + e;
      if (!m.complete) what = "incomplete: `kapitsa pull " + m.name + "` continues it";
      printf("  %-18s %9s  %s\n", m.name.c_str(), hub::fmt_bytes(m.bytes).c_str(), what.c_str());
    }
  }
  std::string more;
  for (const auto& c : hub::catalog())
    for (const auto& var : c.variants) {
      bool have = false;
      for (const auto& m : v) have = have || (m.name == hub::full_name(c, var) && m.complete);
      if (have) continue;
      char b[200];
      snprintf(b, sizeof b, "%s%s (%s, %.1f GB)", more.empty() ? "" : ", ", hub::full_name(c, var).c_str(),
               hub::short_name(c, var).c_str(), var.download_gb);
      more += b;
    }
  if (!more.empty())
    printf("\n%s: %s.\n`kapitsa pull MODEL` downloads one; `kapitsa serve --model MODEL` downloads and starts it.\n",
           v.empty() ? "Available" : "Also available", more.c_str());
  return 0;
}

static int rm(int argc, char** argv) {
  if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
    printf("kapitsa rm MODEL...: deletes downloaded models from %s.\n", hub::tilde(hub::models_dir()).c_str());
    return argc < 3 ? 2 : 0;
  }
  for (int i = 2; i < argc; ++i) {
    hub::ModelRef ref;
    try {
      ref = hub::resolve(argv[i]);
    } catch (const hub::Error& e) {
      return fail(e.what(), 2);
    }
    struct stat st;
    if (ref.kind == hub::ModelRef::kCatalog && !ref.variant_given && stat(ref.dir.c_str(), &st) != 0) {
      // no variant named and the default is not here: the model's one downloaded variant
      std::vector<hub::ModelRef> have;
      for (const auto& v : ref.model->variants) {
        hub::ModelRef r = hub::resolve(ref.model->name + ":" + v.name);
        if (stat(r.dir.c_str(), &st) == 0) have.push_back(r);
      }
      if (have.size() == 1) ref = have[0];
      else if (have.size() > 1) {
        std::string names;
        for (const auto& r : have) names += (names.empty() ? "" : ", ") + hub::short_name(*r.model, *r.variant);
        return fail("Several variants of " + ref.model->name + " are downloaded: " + names + ". Name one.", 2);
      }
    }
    if (ref.kind == hub::ModelRef::kLocal)
      return fail(ref.dir + " is a folder of yours, not a downloaded model: kapitsa rm leaves it alone.", 2);
    if (stat(ref.dir.c_str(), &st) != 0) {
      printf("%s is not downloaded.\n", ref.name.c_str());
      continue;
    }
    const std::string root = hub::models_dir() + "/";
    if (ref.dir.compare(0, root.size(), root) != 0 || ref.dir.size() <= root.size())
      return fail("refusing to delete " + ref.dir + " (not in " + root + ")");
    uint64_t bytes = 0;
    for (const auto& m : hub::installed())
      if (m.dir == ref.dir) bytes = m.bytes;
    std::error_code ec;
    std::filesystem::remove_all(ref.dir, ec);
    if (ec) return fail("Can't delete " + hub::tilde(ref.dir) + ": " + ec.message());
    printf("Deleted %s (%s).\n", ref.name.c_str(), hub::fmt_bytes(bytes).c_str());
  }
  return 0;
}

static int mtp_sidecar(int argc, char** argv) {
  std::string src, out, rev, hidden = "post_norm";
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--revision" && i + 1 < argc) rev = argv[++i];
    else if (a == "--hidden" && i + 1 < argc) hidden = argv[++i];
    else if (!a.empty() && a[0] != '-' && src.empty()) src = a;
    else if (!a.empty() && a[0] != '-' && out.empty()) out = a;
    else src.clear(), out.clear(), i = argc;
  }
  if (src.empty() || out.empty() || (hidden != "post_norm" && hidden != "pre_norm")) {
    fprintf(stderr,
            "usage: kapitsa mtp-sidecar <hf repo | local dir of HF safetensors> <out mtp.safetensors> [--revision REV]\n"
            "                           [--hidden post_norm|pre_norm]\n"
            "Builds Kapitsa's MTP sidecar from an official Qwen3.5 checkpoint's mtp.* tensors, byte for byte what\n"
            "tools/mtp_sidecar.py writes (a repo is read at REV, default main, by range requests).\n");
    return 2;
  }
  try {
    hub::build_mtp_sidecar(src, rev, out, hidden);
  } catch (const hub::Stopped& e) {
    fprintf(stderr, "\n%s\n", e.what());
    return 130;
  } catch (const hub::Error& e) {
    return fail(e.what());
  }
  return 0;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd.empty() || cmd == "--help" || cmd == "-h" || cmd == "help") {
    if (argc > 2 && !strcmp(argv[2], "serve")) print_serve_help();
    else if (argc > 2 && !strcmp(argv[2], "pull")) print_pull_help();
    else print_help();
    return 0;
  }
  if (cmd == "--version" || cmd == "-v" || cmd == "version") {
    printf("kapitsa %s%s%s%s\n", KAPITSA_VERSION, *KAPITSA_GIT ? " (" : "", KAPITSA_GIT, *KAPITSA_GIT ? ")" : "");
    return 0;
  }
  if (cmd == "serve") return serve(argc, argv, 2);
  if (cmd == "pull") return pull(argc, argv);
  if (cmd == "list" || cmd == "ls") return list(argc, argv);
  if (cmd == "rm") return rm(argc, argv);
  if (cmd == "mtp-sidecar") return mtp_sidecar(argc, argv);
  if (cmd == "doctor") return ni::run_doctor(argc - 2, argv + 2);
  if (cmd[0] == '-') return serve(argc, argv, 1);  // options only: `kapitsa --model DIR ...` (older scripts)
  usage_error("unknown command \"" + cmd + "\"");
}
