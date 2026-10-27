#include "api/anthropic.h"

#include "util/log.h"

#include <mutex>
#include <set>

namespace ni::api {

namespace {

[[noreturn]] void bad(const std::string& msg) { throw ApiError(400, "invalid_request_error", msg); }

std::string str(const ojson& j, const char* key) {
  if (!j.is_object()) return "";
  auto it = j.find(key);
  return it != j.end() && it->is_string() ? it->get<std::string>() : "";
}

// Requests repeat the same tools and blocks every turn: each kind of message is logged once
void log_once(const std::string& key, const std::string& msg) {
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> lk(mu);
  if (seen.insert(key).second) logi("%s", msg.c_str());
}

// Content in order: runs of text joined, images as image_url parts (the engine decides what to do with them).
// All text: one string.
struct Parts {
  ojson parts = ojson::array();
  bool media = false;
  void text(const std::string& t) {
    if (t.empty()) return;
    if (!parts.empty() && parts.back()["type"] == "text") {
      std::string s = parts.back()["text"];
      join_text(s, t);
      parts.back()["text"] = s;
    } else {
      parts.push_back({{"type", "text"}, {"text", t}});
    }
  }
  void image(const std::string& url) {
    parts.push_back({{"type", "image_url"}, {"image_url", {{"url", url}}}});
    media = true;
  }
  bool empty() const { return parts.empty(); }
  ojson content() const {
    if (media) return parts;
    return parts.empty() ? ojson("") : parts.back()["text"];
  }
};

// An image block's source as a URL: base64 data as a data: URL, a url source as is
bool image_url(const ojson& block, std::string* url) {
  const ojson src = block.is_object() && block.contains("source") ? block["source"] : ojson();
  const std::string type = str(src, "type");
  if (type == "base64") {
    *url = "data:" + (str(src, "media_type").empty() ? std::string("image/png") : str(src, "media_type")) + ";base64," +
           str(src, "data");
    return true;
  }
  if (type == "url") *url = str(src, "url");
  return type == "url" && !url->empty();
}

void add_image(Parts& p, const ojson& block) {
  std::string url;
  if (image_url(block, &url)) p.image(url);
  else p.text("[image omitted: this server can't read this image source]");
}

// A document: a plain-text one as its text; a PDF (or a file reference) as a short note
void add_document(Parts& p, const ojson& block) {
  const std::string title = str(block, "title");
  const ojson src = block.contains("source") ? block["source"] : ojson();
  const std::string type = str(src, "type");
  if (type == "text") {
    p.text(title.empty() ? str(src, "data") : title + "\n\n" + str(src, "data"));
    return;
  }
  if (type == "content" && src.contains("content")) {
    if (!title.empty()) p.text(title);
    if (src["content"].is_string()) p.text(src["content"].get<std::string>());
    else if (src["content"].is_array())
      for (const auto& b : src["content"]) {
        if (str(b, "type") == "image") add_image(p, b);
        else p.text(str(b, "text"));
      }
    return;
  }
  log_once("document " + type, "anthropic: " + (type.empty() ? std::string("a") : "a " + type) +
                                   " document left out of a request: this server reads only text documents");
  p.text("[document omitted" + (title.empty() ? "" : ": " + title) + ": this server can't read PDF documents yet]");
}

// A tool_result block as a tool message ("Error: " first when is_error)
ojson tool_message(const ojson& block) {
  Parts p;
  const ojson c = block.contains("content") ? block["content"] : ojson();
  if (c.is_string()) {
    p.text(c.get<std::string>());
  } else if (c.is_array()) {
    for (const auto& b : c) {
      const std::string type = str(b, "type");
      if (type == "text") p.text(str(b, "text"));
      else if (type == "image") add_image(p, b);
      else if (type == "document") add_document(p, b);
      else if (type == "tool_reference") p.text("Tool available: " + str(b, "tool_name"));
      else if (type == "search_result") {
        p.text(str(b, "title").empty() ? str(b, "source") : str(b, "title") + " (" + str(b, "source") + ")");
        if (b.contains("content") && b["content"].is_array())
          for (const auto& t : b["content"]) p.text(str(t, "text"));
      } else if (!str(b, "text").empty()) p.text(str(b, "text"));
      else log_once("tool_result " + type, "anthropic: a " + type + " block in a tool result left out");
    }
  }
  ojson content = p.content();
  if (block.contains("is_error") && block["is_error"].is_boolean() && block["is_error"].get<bool>()) {
    if (content.is_string()) content = "Error: " + content.get<std::string>();
    else content.insert(content.begin(), ojson{{"type", "text"}, {"text", "Error: "}});
  }
  return {{"role", "tool"}, {"tool_call_id", str(block, "tool_use_id")}, {"content", content}};
}

// Anthropic tools with an input schema as OpenAI functions; server tools (no input_schema) are left out
ojson function_of(const ojson& t) {
  ojson f = ojson::object();
  f["name"] = str(t, "name");
  if (t.contains("description") && t["description"].is_string()) f["description"] = t["description"];
  f["parameters"] = t["input_schema"];
  return {{"type", "function"}, {"function", f}};
}

struct Prompt {
  std::string system;               // the leading system text
  bool leading = true;              // no user or assistant message yet
  ojson messages = ojson::array();  // after the leading system message
  ojson tools = ojson::array();
};

void add_tool(Prompt& pr, const ojson& t) {
  const std::string name = str(t, "name");
  if (!t.is_object() || name.empty()) return;
  if (!t.contains("input_schema") || !t["input_schema"].is_object()) {
    log_once("tool " + name, "anthropic: tool " + name + " (" + (str(t, "type").empty() ? "no type" : str(t, "type")) +
                                 ") left out: a server tool without an input_schema");
    return;
  }
  for (auto& have : pr.tools)
    if (have["function"]["name"] == name) {
      have = function_of(t);
      return;
    }
  pr.tools.push_back(function_of(t));
}

// System text (a string or text blocks) without Claude Code's billing header block, which changes with every
// request and would end the cached prefix at the system prompt. Tool changes in it apply to the tool list.
std::string system_text(const ojson& c, Prompt& pr) {
  std::string out;
  if (c.is_string()) {
    if (c.get<std::string>().rfind("x-anthropic-billing-header", 0) != 0) out = c.get<std::string>();
    return out;
  }
  if (!c.is_array()) return out;
  for (const auto& b : c) {
    const std::string type = str(b, "type");
    if (type == "text") {
      const std::string t = str(b, "text");
      if (t.rfind("x-anthropic-billing-header", 0) != 0) join_text(out, t);
    } else if (type == "tool_addition" || type == "tool_removal") {
      // mid-conversation tool changes: a definition joins the tool list (replacing one of its name), a removal
      // takes one out (the template declares one tool list, at the start)
      const ojson tool = b.contains("tool") ? b["tool"] : ojson();
      if (type == "tool_addition" && str(tool, "type") == "tool_definition" && tool.contains("definition")) {
        add_tool(pr, tool["definition"]);
      } else if (type == "tool_removal") {
        const std::string name = str(tool, "name");
        for (size_t i = 0; i < pr.tools.size(); ++i)
          if (pr.tools[i]["function"]["name"] == name) {
            pr.tools.erase(pr.tools.begin() + long(i));
            break;
          }
      }
    } else if (!type.empty()) {
      log_once("system " + type, "anthropic: a " + type + " block in a system message left out");
    }
  }
  return out;
}

void add_message(const ojson& m, Prompt& pr) {
  if (!m.is_object()) bad("messages: each message must be an object");
  const std::string role = str(m, "role");
  const ojson c = m.contains("content") && !m["content"].is_null() ? m["content"] : ojson("");
  if (!c.is_string() && !c.is_array()) bad("messages: content must be a string or an array of content blocks");
  if (role == "system") {  // a system message in the conversation (mid-conversation context)
    const std::string t = system_text(c, pr);
    if (pr.leading) join_text(pr.system, t);
    else if (!t.empty()) pr.messages.push_back({{"role", "system"}, {"content", t}});
    return;
  }
  if (role != "user" && role != "assistant") bad("messages: role must be \"user\" or \"assistant\", not \"" + role + "\"");
  pr.leading = false;
  if (c.is_string()) {
    pr.messages.push_back({{"role", role}, {"content", c}});
    return;
  }
  if (role == "user") {
    // tool results first (each a tool message), then the rest of the turn as one user message
    Parts p;
    bool results = false;
    for (const auto& b : c) {
      const std::string type = str(b, "type");
      if (type == "tool_result") {
        pr.messages.push_back(tool_message(b));
        results = true;
      } else if (type == "text") p.text(str(b, "text"));
      else if (type == "image") add_image(p, b);
      else if (type == "document") add_document(p, b);
      else if (type == "tool_reference") p.text("Tool available: " + str(b, "tool_name"));
      else if (type == "thinking" || type == "redacted_thinking") continue;
      else if (!str(b, "text").empty()) p.text(str(b, "text"));
      else log_once("user " + type, "anthropic: a " + type + " block in a user message left out");
    }
    if (!p.empty() || !results) pr.messages.push_back({{"role", "user"}, {"content", p.content()}});
    return;
  }
  // assistant: text, the reasoning of its thinking blocks (our signature carries it exactly), tool calls
  std::string text, reasoning;
  bool thought = false;
  ojson calls = ojson::array();
  for (const auto& b : c) {
    const std::string type = str(b, "type");
    if (type == "text") {
      join_text(text, str(b, "text"));
    } else if (type == "thinking") {
      std::string r;
      if (!unpack_reasoning(str(b, "signature"), &r)) r = str(b, "thinking");
      reasoning += r;
      thought = true;
    } else if (type == "tool_use") {
      calls.push_back({{"id", str(b, "id")},
                       {"type", "function"},
                       {"function", {{"name", str(b, "name")}, {"arguments", arguments_object(b.contains("input") ? b["input"] : ojson())}}}});
    } else if (type != "redacted_thinking") {
      log_once("assistant " + type, "anthropic: a " + type + " block in an assistant message left out");
    }
  }
  ojson msg = {{"role", "assistant"}, {"content", text}};
  if (thought) msg["reasoning_content"] = reasoning;
  if (!calls.empty()) msg["tool_calls"] = calls;
  pr.messages.push_back(msg);
}

std::string tool_use_id(const std::string& id) { return id.rfind("call_", 0) == 0 ? "toolu_" + id.substr(5) : id; }

ojson tool_input(const std::string& arguments) {
  ojson j = ojson::parse(arguments, nullptr, false);
  return j.is_object() ? j : ojson::object();
}

}  // namespace

ChatRequest messages_prompt(const ojson& body, bool* show_thinking) {
  if (!body.is_object()) bad("the request body must be a JSON object");
  if (!body.contains("messages") || !body["messages"].is_array()) bad("messages: Field required (an array of messages)");
  Prompt pr;
  if (body.contains("tools") && body["tools"].is_array())
    for (const auto& t : body["tools"]) add_tool(pr, t);
  if (body.contains("system")) pr.system = system_text(body["system"], pr);
  for (const auto& m : body["messages"]) add_message(m, pr);
  ChatRequest r;
  if (!pr.system.empty()) r.messages.push_back({{"role", "system"}, {"content", pr.system}});
  for (auto& m : pr.messages) r.messages.push_back(std::move(m));
  r.tools = pr.tools;
  if (body.contains("tool_choice") && str(body["tool_choice"], "type") == "none") r.tools = ojson::array();

  // Thinking: on for "enabled" and "adaptive" (Claude Code's main loop), off for "disabled" and when absent, as
  // Anthropic's API does: Claude Code leaves it out of its short side requests (titles, classifiers), which then
  // answer at once instead of thinking first.
  const ojson th = body.contains("thinking") ? body["thinking"] : ojson();
  const std::string type = str(th, "type");
  const bool think = type == "enabled" || type == "adaptive";
  r.template_kwargs["enable_thinking"] = think;
  if (show_thinking) *show_thinking = str(th, "display") != "omitted";
  if (think && body.contains("output_config")) apply_effort(str(body["output_config"], "effort"), r.template_kwargs);
  // looped models (Engine::request_loops): thinking off (disabled, absent) 1 loop, a budget by its size, adaptive or
  // enabled without a budget the server's default; output_config.effort or a "loops" field first
  if (!think) r.thinking_budget = 0;
  else if (th.contains("budget_tokens") && th["budget_tokens"].is_number_integer())
    r.thinking_budget = int(std::clamp<int64_t>(th["budget_tokens"].get<int64_t>(), 0, INT32_MAX));
  if (think && body.contains("output_config")) r.effort = str(body["output_config"], "effort");
  r.loops = request_loops_field(body);
  // context_management keeping every turn's thinking (Claude Code: clear_thinking keep "all"): every assistant turn
  // is rendered with its reasoning (Qwen3.6's preserve_thinking), so the prompt of the next request continues what the
  // engine generated whatever comes after it
  if (body.contains("context_management") && body["context_management"].contains("edits") &&
      body["context_management"]["edits"].is_array())
    for (const auto& e : body["context_management"]["edits"])
      if (str(e, "type").rfind("clear_thinking", 0) == 0 && e.contains("keep") && e["keep"] == "all")
        r.template_kwargs["preserve_thinking"] = true;
  if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
    for (auto& [k, v] : body["chat_template_kwargs"].items()) r.template_kwargs[k] = v;
  return r;
}

MessagesRequest messages_request(const ojson& body) {
  MessagesRequest q;
  q.chat = messages_prompt(body, &q.show_thinking);
  ChatRequest& r = q.chat;
  // Structured output (src/engine/grammar.h): tool_choice any (one call or more) or tool (that one's), with
  // disable_parallel_tool_use (one call); strict tools ("strict": true); output_format / output_config.format, a JSON
  // schema whose objects take no keys they do not declare (Anthropic's rule) unless they say otherwise
  {
    OutputSpec spec;
    if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
      const ojson& tc = body["tool_choice"];
      if (!tc.is_object()) bad("tool_choice must be an object");
      const std::string type = str(tc, "type");
      if (type == "any") {
        spec.tools = OutputSpec::Tools::kRequired;
      } else if (type == "tool") {
        if (str(tc, "name").empty()) bad("tool_choice.name: Field required for a tool choice of type \"tool\"");
        spec.tools = OutputSpec::Tools::kNamed;
        spec.tool = str(tc, "name");
      } else if (type != "auto" && type != "none") {
        bad("tool_choice.type must be \"auto\", \"any\", \"tool\" or \"none\"");
      }
      if (tc.contains("disable_parallel_tool_use") && tc["disable_parallel_tool_use"].is_boolean())
        spec.parallel = !tc["disable_parallel_tool_use"].get<bool>();
    }
    if (!r.tools.empty() && body.contains("tools") && body["tools"].is_array())
      for (const auto& t : body["tools"])
        if (t.is_object() && t.contains("strict") && t["strict"].is_boolean() && t["strict"].get<bool>() &&
            !str(t, "name").empty())
          spec.strict_tools.push_back(str(t, "name"));
    const ojson* of = nullptr;
    if (body.contains("output_format") && !body["output_format"].is_null()) of = &body["output_format"];
    else if (body.contains("output_config") && body["output_config"].is_object() && body["output_config"].contains("format") &&
             !body["output_config"]["format"].is_null())
      of = &body["output_config"]["format"];
    if (of) {
      if (!of->is_object() || str(*of, "type") != "json_schema") bad("output_format.type must be \"json_schema\"");
      spec.strict = true;
      schema_format(*of, spec, "output_format");
    }
    r.grammar = request_grammar(spec, r.tools, "output_format");
  }
  if (!r.messages.empty() && r.messages.back()["role"] == "assistant")
    bad("a final assistant message (a prefill of the reply) is not supported; end the messages with a user turn");
  if (body.contains("max_tokens")) {
    if (!body["max_tokens"].is_number_integer() || body["max_tokens"].get<int64_t>() < 1)
      bad("max_tokens: must be a positive integer");
    r.max_tokens = int(std::min<int64_t>(body["max_tokens"].get<int64_t>(), INT32_MAX));
  } else {
    r.max_tokens = 32768;
  }
  if (body.contains("temperature") && body["temperature"].is_number()) r.temperature = body["temperature"];
  if (body.contains("top_p") && body["top_p"].is_number()) r.top_p = body["top_p"];
  if (body.contains("top_k") && body["top_k"].is_number()) r.top_k = body["top_k"];
  if (body.contains("stop_sequences") && body["stop_sequences"].is_array())
    for (const auto& s : body["stop_sequences"])
      if (s.is_string() && !s.get<std::string>().empty()) r.stop.push_back(s);
  q.stream = body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();
  return q;
}

ojson messages_error(const std::string& type, const std::string& message) {
  return {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
}

std::pair<int, ojson> messages_failure(const ChatResult& res) {
  if (res.busy) return {503, messages_error("overloaded_error", res.error)};
  if (res.prompt_limit > 0)
    return {400, messages_error("invalid_request_error", "prompt is too long: " + std::to_string(res.prompt_tokens) +
                                                             " tokens > " + std::to_string(res.prompt_limit) + " maximum")};
  if (res.invalid_request || res.error.rfind("chat template", 0) == 0)
    return {400, messages_error("invalid_request_error", res.error)};
  return {500, messages_error("api_error", res.error.empty() ? "the request failed" : res.error)};
}

std::pair<std::string, ojson> messages_stop(const ChatResult& res) {
  if (res.finish_reason == "tool_calls") return {"tool_use", nullptr};
  if (res.finish_reason == "length") return {res.context_full ? "model_context_window_exceeded" : "max_tokens", nullptr};
  if (!res.stop_sequence.empty()) return {"stop_sequence", res.stop_sequence};
  return {"end_turn", nullptr};
}

ojson messages_usage(const ChatResult& res) {
  ojson u = {{"input_tokens", res.prompt_tokens - res.cached_tokens},
             {"cache_creation_input_tokens", 0},
             {"cache_read_input_tokens", res.cached_tokens},
             {"output_tokens", res.completion_tokens}};
  if (res.loops > 0) u["kapitsa"] = kapitsa_usage(res);
  return u;
}

ojson messages_response(const std::string& id, const std::string& model, const ChatResult& res, bool show_thinking) {
  ojson content = ojson::array();
  if (!res.reasoning.empty())
    content.push_back({{"type", "thinking"},
                       {"thinking", show_thinking ? res.reasoning : ""},
                       {"signature", pack_reasoning(res.reasoning)}});
  if (!res.content.empty()) content.push_back({{"type", "text"}, {"text", res.content}});
  for (const auto& tc : res.tool_calls)
    content.push_back({{"type", "tool_use"}, {"id", tool_use_id(tc.id)}, {"name", tc.name}, {"input", tool_input(tc.arguments)}});
  if (res.content.empty() && res.tool_calls.empty()) content.push_back({{"type", "text"}, {"text", ""}});
  auto [reason, seq] = messages_stop(res);
  return {{"id", id},
          {"type", "message"},
          {"role", "assistant"},
          {"model", model},
          {"content", content},
          {"stop_reason", reason},
          {"stop_sequence", seq},
          {"usage", messages_usage(res)}};
}

std::string MessagesStream::begin(const ChatResult& prompt) {
  ChatResult u = prompt;
  u.completion_tokens = 0;
  const ojson msg = {{"id", id_},
                     {"type", "message"},
                     {"role", "assistant"},
                     {"model", model_},
                     {"content", ojson::array()},
                     {"stop_reason", nullptr},
                     {"stop_sequence", nullptr},
                     {"usage", messages_usage(u)}};
  return sse("message_start", {{"type", "message_start"}, {"message", msg}}) + keepalive();
}

std::string MessagesStream::open(Kind k) {
  std::string s = close();
  index_ = blocks_++;
  open_ = k;
  const ojson block = k == kThinking ? ojson{{"type", "thinking"}, {"thinking", ""}, {"signature", ""}}
                                     : ojson{{"type", "text"}, {"text", ""}};
  return s + sse("content_block_start", {{"type", "content_block_start"}, {"index", index_}, {"content_block", block}});
}

std::string MessagesStream::close() {
  if (open_ == kNone) return "";
  std::string s;
  if (open_ == kThinking)
    s = sse("content_block_delta", {{"type", "content_block_delta"},
                                    {"index", index_},
                                    {"delta", {{"type", "signature_delta"}, {"signature", pack_reasoning(reasoning_)}}}});
  s += sse("content_block_stop", {{"type", "content_block_stop"}, {"index", index_}});
  open_ = kNone;
  reasoning_.clear();
  return s;
}

std::string MessagesStream::delta(const ChatDelta& d) {
  std::string s;
  if (!d.reasoning.empty()) {
    if (open_ != kThinking) s += open(kThinking);
    reasoning_ += d.reasoning;
    if (show_)
      s += sse("content_block_delta", {{"type", "content_block_delta"},
                                       {"index", index_},
                                       {"delta", {{"type", "thinking_delta"}, {"thinking", d.reasoning}}}});
  }
  if (!d.content.empty()) {
    if (open_ != kText) s += open(kText);
    s += sse("content_block_delta",
             {{"type", "content_block_delta"}, {"index", index_}, {"delta", {{"type", "text_delta"}, {"text", d.content}}}});
  }
  for (const auto& tc : d.tool_calls) {
    // a tool call arrives complete: one block with its whole input in one input_json_delta
    s += close();
    const int i = blocks_++;
    s += sse("content_block_start",
             {{"type", "content_block_start"},
              {"index", i},
              {"content_block", {{"type", "tool_use"}, {"id", tool_use_id(tc.id)}, {"name", tc.name}, {"input", ojson::object()}}}});
    s += sse("content_block_delta", {{"type", "content_block_delta"},
                                     {"index", i},
                                     {"delta", {{"type", "input_json_delta"}, {"partial_json", tc.arguments}}}});
    s += sse("content_block_stop", {{"type", "content_block_stop"}, {"index", i}});
  }
  return s;
}

std::string MessagesStream::keepalive() { return sse("ping", {{"type", "ping"}}); }

std::string MessagesStream::end(const ChatResult& res) {
  std::string s = close();
  if (res.finish_reason == "error")
    return s + sse("error", messages_error("api_error", res.error.empty() ? "the request failed" : res.error));
  auto [reason, seq] = messages_stop(res);
  s += sse("message_delta", {{"type", "message_delta"},
                             {"delta", {{"stop_reason", reason}, {"stop_sequence", seq}}},
                             {"usage", messages_usage(res)}});
  return s + sse("message_stop", {{"type", "message_stop"}});
}

}  // namespace ni::api
