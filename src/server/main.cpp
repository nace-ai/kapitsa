// kapitsa: OpenAI-compatible HTTP server (chat completions, streaming, tools).
//   kapitsa serve --model <dir> [--model-id ID] [--host 127.0.0.1] [--port 8000] [--ctx 32768] [--slots 2]
//                 [--spec N] [--prefill-chunk 2048] [--api-key K] [--mem-budget GB]
//   kapitsa doctor [--model DIR]... [--ctx N] [--slots N] [--json]
// ("serve" is optional; build/nace-infer-server is a symlink kept for older scripts.)
#include <cstring>
#include "engine/doctor.h"
#include "engine/engine.h"
#include "util/log.h"

#include <httplib/httplib.h>

#include <atomic>
#include <thread>
#include <csignal>
#include <ctime>

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

int main(int argc, char** argv) {
  EngineOptions opt;
  std::string host = "127.0.0.1", api_key;
  int port = 8000;
  int first = 1;
  if (argc > 1 && !strcmp(argv[1], "serve")) first = 2;
  if (argc > 1 && !strcmp(argv[1], "doctor")) return ni::run_doctor(argc - 2, argv + 2);
  if (argc > 1 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h") || !strcmp(argv[1], "help"))) {
    printf("Kapitsa: fast Qwen inference on Apple silicon\n"
           "usage: kapitsa serve --model <dir> [--model-id ID] [--host 127.0.0.1] [--port 8000] [--ctx 32768]\n"
           "                     [--slots 2] [--spec N] [--prefill-chunk 2048] [--api-key KEY] [--mem-budget GB]\n"
           "       kapitsa doctor [--model DIR]... [--ctx 65536] [--slots 2] [--json]   this Mac, and which models fit\n");
    return 0;
  }
  for (int i = first; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--model") opt.model_dir = next();
    else if (a == "--model-id") opt.model_id = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--ctx") opt.context = std::stoi(next());
    else if (a == "--slots") opt.slots = std::stoi(next());
    else if (a == "--spec") opt.spec_depth = std::stoi(next());
    else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
    else if (a == "--api-key") api_key = next();
    else if (a == "--mem-budget") opt.mem_budget_gb = std::stod(next());
    else fatal("unknown option %s", a.c_str());
  }
  if (opt.model_dir.empty()) fatal("--model is required");
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
    if (api_key.empty() || req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;
    if (req.get_header_value("Authorization") == "Bearer " + api_key) return httplib::Server::HandlerResponse::Unhandled;
    res.status = 401;
    res.set_content(R"({"error":{"message":"invalid api key","type":"auth_error"}})", "application/json");
    return httplib::Server::HandlerResponse::Handled;
  });
  srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
  });
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

  logi("listening on http://%s:%d", host.c_str(), port);
  if (!srv.listen(host, port)) fatal("cannot listen on %s:%d", host.c_str(), port);
  return 0;
}
