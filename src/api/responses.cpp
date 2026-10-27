#include "api/responses.h"

#include "engine/tool_args.h"
#include "util/log.h"

namespace ni::api {

namespace {

[[noreturn]] void bad(const std::string& msg, const std::string& param = "") {
  throw ApiError(400, "invalid_request_error", msg, "", param);
}

std::string str(const ojson& j, const char* key) {
  if (!j.is_object()) return "";
  auto it = j.find(key);
  return it != j.end() && it->is_string() ? it->get<std::string>() : "";
}

void log_once(const std::string& key, const std::string& msg) {
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> lk(mu);
  if (seen.insert(key).second) logi("%s", msg.c_str());
}

// separate messages joined into one: a blank line between them
void join_block(std::string& acc, const std::string& piece) {
  if (piece.empty()) return;
  if (!acc.empty()) acc += acc.size() >= 2 && acc.compare(acc.size() - 2, 2, "\n\n") == 0 ? "" : acc.back() == '\n' ? "\n" : "\n\n";
  acc += piece;
}

// The text of content: a string, or the text of its text-like parts
std::string text_of(const ojson& c) {
  if (c.is_string()) return c.get<std::string>();
  std::string s;
  if (c.is_array())
    for (const auto& p : c) {
      const std::string t = str(p, "type");
      if (t == "input_text" || t == "output_text" || t == "text" || t == "reasoning_text" || t == "summary_text")
        join_text(s, str(p, "text"));
      else if (t == "refusal") join_text(s, str(p, "refusal"));
    }
  return s;
}

// User or tool-output content: a string when it is all text, else parts with images as image_url parts (the engine
// decides what to do with them); files become a short note
ojson user_content(const ojson& c) {
  if (c.is_string()) return c;
  if (!c.is_array()) return "";
  ojson parts = ojson::array();
  bool media = false;
  auto text = [&](const std::string& t) {
    if (t.empty()) return;
    if (!parts.empty() && parts.back()["type"] == "text") {
      std::string s = parts.back()["text"];
      join_text(s, t);
      parts.back()["text"] = s;
    } else {
      parts.push_back({{"type", "text"}, {"text", t}});
    }
  };
  for (const auto& p : c) {
    const std::string t = str(p, "type");
    if (t == "input_text" || t == "output_text" || t == "text") {
      text(str(p, "text"));
    } else if (t == "input_image") {
      std::string url = p.contains("image_url") && p["image_url"].is_object() ? str(p["image_url"], "url") : str(p, "image_url");
      if (url.empty() && !str(p, "file_id").empty()) url = "file_id:" + str(p, "file_id");
      if (url.empty()) {
        text("[image omitted: no image_url or file_id]");
        continue;
      }
      parts.push_back({{"type", "image_url"}, {"image_url", {{"url", url}}}});
      media = true;
    } else if (t == "input_file") {
      const std::string name = str(p, "filename");
      log_once("input_file", "responses: an input_file left out of a request: this server can't read files yet");
      text("[file omitted" + (name.empty() ? "" : ": " + name) + ": this server can't read files yet]");
    } else {
      log_once("part " + t, "responses: a " + t + " content part left out");
    }
  }
  if (media) return parts;
  return parts.empty() ? ojson("") : parts.back()["text"];
}

// The prompt's name for a namespaced tool
std::string alias(const std::string& ns, const std::string& name) { return ns + "__" + name; }

// The response an output item of ours came from: its id is <kind>_<response>_<index> (see ResponsesStream)
std::string response_of(const ojson& it) {
  const std::string id = str(it, "id");
  const size_t a = id.find('_'), b = id.rfind('_');
  if (a == std::string::npos || b <= a + 1 || b + 1 >= id.size()) return "";
  const std::string kind = id.substr(0, a), from = id.substr(a + 1, b - a - 1);
  if (kind != "rs" && kind != "msg" && kind != "fc" && kind != "ctc") return "";
  for (char c : from)
    if (!isalnum(uint8_t(c))) return "";
  for (size_t i = b + 1; i < id.size(); ++i)
    if (!isdigit(uint8_t(id[i]))) return "";
  return from;
}

struct Conv {
  std::string system;
  bool leading = true;
  ojson messages = ojson::array();
  ojson pending;                    // the assistant turn being put together
  std::string from;                 // the response it came from, if known
  std::vector<ojson> outputs;       // tool outputs since its calls: they follow it, once it is complete
  void flush() {
    if (!pending.is_null()) messages.push_back(pending);
    for (auto& o : outputs) messages.push_back(std::move(o));
    pending = nullptr;
    from.clear();
    outputs.clear();
  }
  // the turn item `it` belongs to: the pending one, unless tool outputs came after that one and `it` is not from the
  // same response (a client that sends each call followed by its output, as OpenAI's examples do, keeps a reply with
  // several calls one assistant turn, as it was generated)
  ojson& turn(const ojson& it) {
    const std::string r = response_of(it);
    if (!outputs.empty() && (r.empty() || r != from)) flush();
    if (pending.is_null()) pending = {{"role", "assistant"}, {"content", ""}};
    if (from.empty()) from = r;
    return pending;
  }
  void call(const ojson& it, const std::string& name, const ojson& args) {
    ojson& a = turn(it);
    if (!a.contains("tool_calls")) a["tool_calls"] = ojson::array();
    a["tool_calls"].push_back({{"id", str(it, "call_id")}, {"type", "function"}, {"function", {{"name", name}, {"arguments", args}}}});
  }
};

void add_item(const ojson& it, Conv& cv) {
  if (!it.is_object()) bad("input: each item must be an object", "input");
  std::string type = str(it, "type");
  if (type.empty() && it.contains("role")) type = "message";
  if (type == "message") {
    std::string role = str(it, "role");
    const ojson c = it.contains("content") ? it["content"] : ojson("");
    if (role == "system" || role == "developer") {
      const std::string t = text_of(c);
      cv.flush();
      if (cv.leading) join_block(cv.system, t);
      else if (!t.empty()) cv.messages.push_back({{"role", "system"}, {"content", t}});
      return;
    }
    cv.leading = false;
    if (role == "user") {
      cv.flush();
      cv.messages.push_back({{"role", "user"}, {"content", user_content(c)}});
    } else if (role == "assistant") {
      ojson& a = cv.turn(it);
      std::string s = a["content"];
      join_text(s, text_of(c));
      a["content"] = s;
    } else {
      bad("input: unsupported message role \"" + role + "\"", "input");
    }
    return;
  }
  cv.leading = false;
  if (type == "reasoning") {
    // the reasoning of the turn it starts: our encrypted_content carries it exactly, else its reasoning text, else its
    // summary
    std::string r;
    if (!unpack_reasoning(str(it, "encrypted_content"), &r)) {
      if (it.contains("content")) r = text_of(it["content"]);
      if (r.empty() && it.contains("summary")) r = text_of(it["summary"]);
    }
    // a turn already holding text or calls is complete: reasoning starts the next one
    if (!cv.pending.is_null() && (!cv.pending["content"].get<std::string>().empty() || cv.pending.contains("tool_calls")))
      cv.flush();
    ojson& a = cv.turn(it);
    a["reasoning_content"] = (a.contains("reasoning_content") ? a["reasoning_content"].get<std::string>() : "") + r;
  } else if (type == "function_call") {
    std::string name = str(it, "name");
    if (!str(it, "namespace").empty()) name = alias(str(it, "namespace"), name);
    cv.call(it, name, arguments_object(it.contains("arguments") ? it["arguments"] : ojson()));
    if (it.contains("arguments") && it["arguments"].is_string())  // the spelling of each value, as the client sent it
      tool_args::note_raw(cv.pending["tool_calls"].back()["function"], it["arguments"].get<std::string>());
  } else if (type == "custom_tool_call") {
    cv.call(it, str(it, "name"), ojson{{"input", str(it, "input")}});
  } else if (type == "function_call_output" || type == "custom_tool_call_output") {
    ojson out = it.contains("output") ? it["output"] : ojson("");
    if (out.is_object() && out.contains("content")) out = out["content"];
    ojson msg = {{"role", "tool"}, {"tool_call_id", str(it, "call_id")}, {"content", user_content(out)}};
    if (cv.pending.is_null()) cv.messages.push_back(std::move(msg));
    else cv.outputs.push_back(std::move(msg));
  } else {
    log_once("item " + type, "responses: " + (type.empty() ? std::string("an untyped") : "a " + type) +
                                 " input item left out");
  }
}

ojson function_tool(const std::string& name, const ojson& t) {
  ojson f = ojson::object();
  f["name"] = name;
  if (t.contains("description") && t["description"].is_string()) f["description"] = t["description"];
  f["parameters"] = t.contains("parameters") && t["parameters"].is_object()
                        ? t["parameters"]
                        : ojson{{"type", "object"}, {"properties", ojson::object()}};
  return {{"type", "function"}, {"function", f}};
}

// A freeform tool as a function with one string parameter, `input` (its grammar, if any, in the description)
ojson custom_tool(const ojson& t) {
  std::string about = "The input to the tool, as free text";
  const ojson fmt = t.contains("format") ? t["format"] : ojson();
  if (str(fmt, "type") == "grammar" && !str(fmt, "definition").empty())
    about += ". It must match this " + (str(fmt, "syntax").empty() ? std::string("") : str(fmt, "syntax") + " ") +
             "grammar:\n" + str(fmt, "definition");
  ojson f = ojson::object();
  f["name"] = str(t, "name");
  if (t.contains("description") && t["description"].is_string()) f["description"] = t["description"];
  f["parameters"] = {{"type", "object"},
                     {"properties", {{"input", {{"type", "string"}, {"description", about}}}}},
                     {"required", ojson::array({"input"})}};
  return {{"type", "function"}, {"function", f}};
}

}  // namespace

void ResponseStore::put(const std::string& id, ojson items) {
  std::lock_guard<std::mutex> lk(mu_);
  if (items_.count(id) == 0) order_.push_back(id);
  items_[id] = std::move(items);
  while (order_.size() > cap_) {
    items_.erase(order_.front());
    order_.pop_front();
  }
}

bool ResponseStore::get(const std::string& id, ojson* items) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = items_.find(id);
  if (it == items_.end()) return false;
  *items = it->second;
  return true;
}

ResponsesRequest responses_request(const ojson& body, const ResponseStore& store) {
  if (!body.is_object()) bad("the request body must be a JSON object");
  ResponsesRequest q;
  // the conversation: the stored one this request continues, then its input
  const std::string prev = str(body, "previous_response_id");
  if (!prev.empty() && !store.get(prev, &q.items))
    throw ApiError(400, "invalid_request_error", "Previous response with id '" + prev + "' not found.",
                   "previous_response_not_found", "previous_response_id");
  if (!body.contains("input")) bad("input: Field required", "input");
  const ojson& input = body["input"];
  if (input.is_string()) q.items.push_back({{"type", "message"}, {"role", "user"}, {"content", input}});
  else if (input.is_array())
    for (const auto& it : input) q.items.push_back(it);
  else bad("input: must be a string or an array of items", "input");

  Conv cv;
  if (body.contains("instructions") && body["instructions"].is_string()) cv.system = body["instructions"];
  for (const auto& it : q.items) add_item(it, cv);
  cv.flush();
  ChatRequest& r = q.chat;
  if (!cv.system.empty()) r.messages.push_back({{"role", "system"}, {"content", cv.system}});
  for (auto& m : cv.messages) r.messages.push_back(std::move(m));

  if (body.contains("tools") && body["tools"].is_array())
    for (const auto& t : body["tools"]) {
      const std::string type = str(t, "type"), name = str(t, "name");
      if (type == "function" && !name.empty()) {
        r.tools.push_back(function_tool(name, t));
      } else if (type == "custom" && !name.empty()) {
        r.tools.push_back(custom_tool(t));
        q.custom.insert(name);
      } else if (type == "namespace" && t.contains("tools") && t["tools"].is_array()) {
        for (const auto& c : t["tools"])
          if (str(c, "type") == "function" && !str(c, "name").empty()) {
            const std::string a = alias(name, str(c, "name"));
            r.tools.push_back(function_tool(a, c));
            q.namespaced[a] = {name, str(c, "name")};
          }
      } else {
        log_once("tool " + type + name, "responses: tool " + (name.empty() ? type : name + " (" + type + ")") +
                                            " left out: a built-in tool this server does not run");
      }
    }
  if (body.contains("tool_choice") && body["tool_choice"].is_string() && body["tool_choice"] == "none")
    r.tools = ojson::array();
  // Structured output (src/engine/grammar.h): text.format, tool_choice required, a named function or custom tool, or
  // allowed_tools, parallel_tool_calls, and strict function tools ("strict": true; one that leaves it out is not strict,
  // where OpenAI defaults it to true: coding agents send their tools without it and stay unconstrained)
  {
    OutputSpec spec;
    const ojson text = body.contains("text") && body["text"].is_object() ? body["text"] : ojson::object();
    if (text.contains("format") && !text["format"].is_null()) {
      const ojson& f = text["format"];
      if (!f.is_object()) bad("text.format must be an object", "text.format");
      const std::string type = str(f, "type");
      if (type == "json_object") spec.format = OutputSpec::Format::kJsonObject;
      else if (type == "json_schema") schema_format(f, spec, "text.format");
      else if (type != "text") bad("text.format.type must be \"text\", \"json_object\" or \"json_schema\"", "text.format.type");
    }
    // a function's name as the prompt declares it (a namespace's functions go by namespace__name)
    auto prompt_name = [&](const ojson& c) {
      return str(c, "namespace").empty() ? str(c, "name") : alias(str(c, "namespace"), str(c, "name"));
    };
    if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
      const ojson& tc = body["tool_choice"];
      const std::string type = tc.is_string() ? tc.get<std::string>() : str(tc, "type");
      if (tc.is_string()) {
        if (type == "required") spec.tools = OutputSpec::Tools::kRequired;
        else if (type != "auto" && type != "none") bad("tool_choice must be \"none\", \"auto\", \"required\" or a tool", "tool_choice");
      } else if ((type == "function" || type == "custom") && !str(tc, "name").empty()) {
        spec.tools = OutputSpec::Tools::kNamed;
        spec.tool = prompt_name(tc);
      } else if (type == "allowed_tools") {
        if (!tc.contains("tools") || !tc["tools"].is_array() || tc["tools"].empty())
          bad("tool_choice.tools must list the allowed tools", "tool_choice.tools");
        for (const ojson& c : tc["tools"])
          if (!str(c, "name").empty()) spec.allowed.push_back(prompt_name(c));
        const std::string mode = str(tc, "mode");
        if (mode == "required") spec.tools = OutputSpec::Tools::kRequired;
        else if (!mode.empty() && mode != "auto") bad("tool_choice.mode must be \"auto\" or \"required\"", "tool_choice.mode");
      } else if (tc.is_object() && !type.empty()) {
        bad("tool_choice \"" + type + "\": a built-in tool this server does not run", "tool_choice");
      } else {
        bad("tool_choice must be \"none\", \"auto\", \"required\" or a tool", "tool_choice");
      }
    }
    if (body.contains("parallel_tool_calls") && body["parallel_tool_calls"].is_boolean())
      spec.parallel = body["parallel_tool_calls"].get<bool>();
    if (!r.tools.empty() && body.contains("tools") && body["tools"].is_array())
      for (const auto& t : body["tools"]) {
        auto strict = [](const ojson& x) { return x.contains("strict") && x["strict"].is_boolean() && x["strict"].get<bool>(); };
        if (str(t, "type") == "function" && strict(t)) spec.strict_tools.push_back(str(t, "name"));
        if (str(t, "type") == "namespace" && t.contains("tools") && t["tools"].is_array())
          for (const auto& c : t["tools"])
            if (str(c, "type") == "function" && strict(c)) spec.strict_tools.push_back(alias(str(t, "name"), str(c, "name")));
      }
    r.grammar = request_grammar(spec, r.tools, "text.format");
  }

  r.max_tokens = 32768;
  if (body.contains("max_output_tokens") && !body["max_output_tokens"].is_null()) {
    if (!body["max_output_tokens"].is_number_integer() || body["max_output_tokens"].get<int64_t>() < 1)
      bad("max_output_tokens: must be a positive integer", "max_output_tokens");
    r.max_tokens = int(std::min<int64_t>(body["max_output_tokens"].get<int64_t>(), INT32_MAX));
  }
  if (body.contains("temperature") && body["temperature"].is_number()) r.temperature = body["temperature"];
  if (body.contains("top_p") && body["top_p"].is_number()) r.top_p = body["top_p"];
  if (body.contains("top_k") && body["top_k"].is_number()) r.top_k = body["top_k"];
  const ojson reasoning = body.contains("reasoning") && body["reasoning"].is_object() ? body["reasoning"] : ojson::object();
  apply_effort(str(reasoning, "effort"), r.template_kwargs);
  r.effort = str(reasoning, "effort");   // looped models: the loop count (Engine::request_loops)
  r.loops = request_loops_field(body);
  if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
    for (auto& [k, v] : body["chat_template_kwargs"].items()) r.template_kwargs[k] = v;

  q.stream = body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();
  q.store = !(body.contains("store") && body["store"].is_boolean() && !body["store"].get<bool>());
  auto get = [&](const char* k, ojson def) { return body.contains(k) && !body[k].is_null() ? body[k] : def; };
  q.echo = {{"instructions", get("instructions", nullptr)},
            {"max_output_tokens", get("max_output_tokens", nullptr)},
            {"parallel_tool_calls", get("parallel_tool_calls", true)},
            {"previous_response_id", prev.empty() ? ojson() : ojson(prev)},
            {"reasoning", {{"effort", reasoning.contains("effort") ? reasoning["effort"] : ojson()},
                           {"summary", reasoning.contains("summary") ? reasoning["summary"] : ojson()}}},
            {"temperature", get("temperature", 1.0)},
            {"text", get("text", ojson{{"format", {{"type", "text"}}}})},
            {"tool_choice", get("tool_choice", "auto")},
            {"tools", get("tools", ojson::array())},
            {"top_p", get("top_p", 1.0)},
            {"truncation", get("truncation", "disabled")},
            {"metadata", get("metadata", ojson::object())},
            {"user", get("user", nullptr)}};
  return q;
}

ojson responses_error(const std::string& message, const std::string& type, const std::string& code,
                      const std::string& param) {
  return {{"error",
           {{"message", message}, {"type", type}, {"param", param.empty() ? ojson() : ojson(param)},
            {"code", code.empty() ? ojson() : ojson(code)}}}};
}

std::pair<int, ojson> responses_failure(const ChatResult& res) {
  if (res.busy) return {503, responses_error(res.error, "server_error", "server_error")};
  if (res.prompt_limit > 0)
    return {400, responses_error("Your input exceeds the context window of this model (" + std::to_string(res.prompt_tokens) +
                                     " tokens > " + std::to_string(res.prompt_limit) +
                                     " maximum). Please adjust your input and try again.",
                                 "invalid_request_error", "context_length_exceeded", "input")};
  if (res.invalid_request || res.error.rfind("chat template", 0) == 0)
    return {400, responses_error(res.error, "invalid_request_error")};
  return {500, responses_error(res.error.empty() ? "the request failed" : res.error, "server_error", "server_error")};
}

std::string ResponsesStream::event(const std::string& type, ojson data) {
  ojson e = {{"type", type}, {"sequence_number", seq_++}};
  for (auto& [k, v] : data.items()) e[k] = v;
  return sse(type, e);
}

ojson ResponsesStream::response(const std::string& status, const ojson& usage, const ojson& error,
                                const ojson& incomplete) const {
  ojson r = {{"id", id_},
             {"object", "response"},
             {"created_at", created_},
             {"status", status},
             {"background", false},
             {"error", error},
             {"incomplete_details", incomplete},
             {"instructions", echo_.value("instructions", ojson())},
             {"max_output_tokens", echo_.value("max_output_tokens", ojson())},
             {"model", model_},
             {"output", status == "in_progress" ? ojson::array() : output_},
             {"parallel_tool_calls", echo_.value("parallel_tool_calls", ojson(true))},
             {"previous_response_id", echo_.value("previous_response_id", ojson())},
             {"reasoning", echo_.value("reasoning", ojson::object())},
             {"store", store_},
             {"temperature", echo_.value("temperature", ojson(1.0))},
             {"text", echo_.value("text", ojson::object())},
             {"tool_choice", echo_.value("tool_choice", ojson("auto"))},
             {"tools", echo_.value("tools", ojson::array())},
             {"top_p", echo_.value("top_p", ojson(1.0))},
             {"truncation", echo_.value("truncation", ojson("disabled"))},
             {"usage", usage},
             {"user", echo_.value("user", ojson())},
             {"metadata", echo_.value("metadata", ojson::object())}};
  return r;
}

std::string ResponsesStream::begin(const ChatResult&) {
  const ojson r = response("in_progress", nullptr, nullptr, nullptr);
  return event("response.created", {{"response", r}}) + event("response.in_progress", {{"response", r}});
}

// <kind>_<the response id's random part>_<output index>: a client that sends the items back in another order still
// shows which reply each came from (Conv::turn)
std::string ResponsesStream::item_id(const char* kind) const {
  return std::string(kind) + "_" + (id_.size() > 5 ? id_.substr(5) : id_) + "_" + std::to_string(output_.size());
}

std::string ResponsesStream::open(Kind k) {
  std::string s = close();
  open_ = k;
  text_.clear();
  const int index = int(output_.size());
  if (k == kReasoning) {
    item_id_ = item_id("rs");
    s += event("response.output_item.added",
               {{"output_index", index},
                {"item", {{"id", item_id_}, {"type", "reasoning"}, {"summary", ojson::array()}, {"content", ojson::array()},
                          {"status", "in_progress"}}}});
    s += event("response.content_part.added", {{"item_id", item_id_}, {"output_index", index}, {"content_index", 0},
                                               {"part", {{"type", "reasoning_text"}, {"text", ""}}}});
  } else {
    item_id_ = item_id("msg");
    s += event("response.output_item.added",
               {{"output_index", index},
                {"item", {{"id", item_id_}, {"type", "message"}, {"status", "in_progress"}, {"role", "assistant"},
                          {"content", ojson::array()}}}});
    s += event("response.content_part.added",
               {{"item_id", item_id_}, {"output_index", index}, {"content_index", 0},
                {"part", {{"type", "output_text"}, {"text", ""}, {"annotations", ojson::array()}, {"logprobs", ojson::array()}}}});
  }
  return s;
}

std::string ResponsesStream::close(bool cut) {
  if (open_ == kNone) return "";
  std::string s;
  const int index = int(output_.size());
  const std::string status = cut ? "incomplete" : "completed";
  ojson item;
  if (open_ == kReasoning) {
    const ojson part = {{"type", "reasoning_text"}, {"text", text_}};
    s += event("response.reasoning_text.done", {{"item_id", item_id_}, {"output_index", index}, {"content_index", 0}, {"text", text_}});
    s += event("response.content_part.done", {{"item_id", item_id_}, {"output_index", index}, {"content_index", 0}, {"part", part}});
    item = {{"id", item_id_}, {"type", "reasoning"}, {"summary", ojson::array()}, {"content", ojson::array({part})},
            {"encrypted_content", pack_reasoning(text_)}, {"status", status}};
  } else {
    const ojson part = {{"type", "output_text"}, {"text", text_}, {"annotations", ojson::array()}, {"logprobs", ojson::array()}};
    s += event("response.output_text.done",
               {{"item_id", item_id_}, {"output_index", index}, {"content_index", 0}, {"text", text_}, {"logprobs", ojson::array()}});
    s += event("response.content_part.done", {{"item_id", item_id_}, {"output_index", index}, {"content_index", 0}, {"part", part}});
    item = {{"id", item_id_}, {"type", "message"}, {"status", status}, {"role", "assistant"}, {"content", ojson::array({part})}};
  }
  s += event("response.output_item.done", {{"output_index", index}, {"item", item}});
  output_.push_back(item);
  open_ = kNone;
  text_.clear();
  return s;
}

std::string ResponsesStream::delta(const ChatDelta& d) {
  std::string s;
  if (!d.reasoning.empty()) {
    if (open_ != kReasoning) s += open(kReasoning);
    text_ += d.reasoning;
    s += event("response.reasoning_text.delta", {{"item_id", item_id_}, {"output_index", int(output_.size())},
                                                 {"content_index", 0}, {"delta", d.reasoning}});
  }
  if (!d.content.empty()) {
    if (open_ != kMessage) s += open(kMessage);
    text_ += d.content;
    s += event("response.output_text.delta", {{"item_id", item_id_}, {"output_index", int(output_.size())},
                                              {"content_index", 0}, {"delta", d.content}, {"logprobs", ojson::array()}});
  }
  for (const auto& tc : d.tool_calls) {
    s += close();
    const int index = int(output_.size());
    ojson item;
    if (custom_.count(tc.name)) {
      // a freeform tool: its input is the call's one string argument
      const ojson args = ojson::parse(tc.arguments, nullptr, false);
      std::string input = tc.arguments;
      if (args.is_object() && args.contains("input") && args["input"].is_string()) input = args["input"];
      else if (args.is_object() && args.size() == 1 && args.begin()->is_string()) input = args.begin()->get<std::string>();
      const std::string id = item_id("ctc");
      item = {{"id", id}, {"type", "custom_tool_call"}, {"call_id", tc.id}, {"name", tc.name}, {"input", ""}};
      s += event("response.output_item.added", {{"output_index", index}, {"item", item}});
      s += event("response.custom_tool_call_input.delta", {{"item_id", id}, {"output_index", index}, {"delta", input}});
      s += event("response.custom_tool_call_input.done", {{"item_id", id}, {"output_index", index}, {"input", input}});
      item["input"] = input;
    } else {
      std::string name = tc.name, ns;
      if (auto it = namespaced_.find(tc.name); it != namespaced_.end()) ns = it->second.first, name = it->second.second;
      const std::string id = item_id("fc");
      item = {{"id", id}, {"type", "function_call"}, {"status", "in_progress"}, {"arguments", ""}, {"call_id", tc.id}, {"name", name}};
      if (!ns.empty()) item["namespace"] = ns;
      s += event("response.output_item.added", {{"output_index", index}, {"item", item}});
      s += event("response.function_call_arguments.delta", {{"item_id", id}, {"output_index", index}, {"delta", tc.arguments}});
      s += event("response.function_call_arguments.done",
                 {{"item_id", id}, {"output_index", index}, {"name", name}, {"arguments", tc.arguments}});
      item["arguments"] = tc.arguments;
      item["status"] = "completed";
    }
    s += event("response.output_item.done", {{"output_index", index}, {"item", item}});
    output_.push_back(item);
  }
  return s;
}

std::string ResponsesStream::keepalive() {
  // a lifecycle event, which the clients' idle timers count (Codex's, 300 s, skips SSE comments), also once output has
  // started: the clients leave its empty output alone (Codex logs it as unhandled; the openai SDKs' stream state and
  // the AI SDK skip it)
  return event("response.in_progress", {{"response", response("in_progress", nullptr, nullptr, nullptr)}});
}

std::string ResponsesStream::end(const ChatResult& res) {
  if (res.finish_reason == "error") {
    std::string s = close();
    final_ = response("failed", nullptr, {{"code", "server_error"}, {"message", res.error}}, nullptr);
    return s + event("response.failed", {{"response", final_}});
  }
  const bool cut = res.finish_reason == "length";
  std::string s = close(cut);
  ojson usage = {{"input_tokens", res.prompt_tokens},
                 {"input_tokens_details", {{"cached_tokens", res.cached_tokens},
                                           {"cache_write_tokens", res.prompt_tokens - res.cached_tokens}}},
                 {"output_tokens", res.completion_tokens},
                 {"output_tokens_details", {{"reasoning_tokens", res.reasoning_tokens}}},
                 {"total_tokens", res.prompt_tokens + res.completion_tokens}};
  if (res.loops > 0) usage["kapitsa"] = kapitsa_usage(res);
  final_ = response(cut ? "incomplete" : "completed", usage, nullptr,
                    cut ? ojson{{"reason", "max_output_tokens"}} : ojson());
  return s + event(cut ? "response.incomplete" : "response.completed", {{"response", final_}});
}

ojson responses_response(const ResponsesRequest& rq, const std::string& id, const std::string& model, long created,
                         const ChatResult& res, ojson* output) {
  ResponsesStream w(rq, id, model, created);
  w.begin(res);
  ChatDelta d;
  d.reasoning = res.reasoning;
  d.content = res.content;
  d.tool_calls = res.tool_calls;
  w.delta(d);
  w.end(res);
  if (output) *output = w.output();
  return w.final_response();
}

}  // namespace ni::api
