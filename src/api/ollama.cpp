// Ollama-compatible API: stream writers, request parsers, response builders, model name mapping.
// Streams newline-delimited JSON (each line: one JSON object + '\n'), not SSE.
#include "api/ollama.h"
#include "api/chat_completions.h"

#include <climits>
#include <cmath>
#include <ctime>

namespace ni::api {

std::string ollama_now() {
  time_t t = std::time(nullptr);
  struct tm tm {};
  gmtime_r(&t, &tm);
  char buf[64];
  strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", &tm);
  return std::string(buf) + ".000000000Z";
}

void ollama_eval_stats(ojson& j, const ChatResult& res) {
  int64_t prefill_ns = int64_t(res.prefill_ms * 1e6), decode_ns = int64_t(res.decode_ms * 1e6);
  j["total_duration"] = prefill_ns + decode_ns;
  j["load_duration"] = 0;
  j["prompt_eval_count"] = res.prompt_tokens;
  if (res.cached_tokens > 0) j["prompt_eval_cached_count"] = res.cached_tokens;
  j["prompt_eval_duration"] = prefill_ns;
  j["eval_count"] = res.completion_tokens;
  j["eval_duration"] = decode_ns;
}

// Resolve an Ollama-style model name through the catalog; unknown names pass through unchanged.
std::string ollama_resolve_model(const std::string& name) {
  if (name.empty()) return name;
  try { return hub::resolve(name).id; } catch (...) {}
  return name;
}

static void parse_options(const ojson& opts, ChatRequest& r) {
  if (!opts.is_object()) return;
  if (opts.contains("temperature") && opts["temperature"].is_number()) r.temperature = opts["temperature"];
  if (opts.contains("top_p") && opts["top_p"].is_number()) r.top_p = opts["top_p"];
  if (opts.contains("top_k") && opts["top_k"].is_number()) r.top_k = opts["top_k"];
  if (opts.contains("seed") && opts["seed"].is_number()) {
    r.seed = uint32_t(opts["seed"].get<int64_t>());
    // Splash backend seed (same logic as the OpenAI path in parse_request)
    const ojson& s = opts["seed"];
    if (s.is_number_unsigned()) r.splash_seed = s.get<uint64_t>();
    else if (s.is_number_integer() && s.get<int64_t>() != -1) r.splash_seed = uint64_t(s.get<int64_t>());
    else if (!s.is_number_integer()) r.splash_seed_invalid = true;
  }
  if (opts.contains("num_predict") && opts["num_predict"].is_number()) {
    // Ollama: -1 means infinite, -2 means fill context. Both leave the server's default (the engine stops
    // at the context limit regardless). Positive values are clamped to int range before casting.
    double v = opts["num_predict"].get<double>();
    if (v >= 0) {
      r.max_tokens = (v > 1e9) ? int(1e9) : int(v);
    }
    // v < 0: leave r.max_tokens at the server default
  }
  if (opts.contains("presence_penalty") && opts["presence_penalty"].is_number()) r.presence_penalty = opts["presence_penalty"];
  if (opts.contains("frequency_penalty") && opts["frequency_penalty"].is_number()) r.frequency_penalty = opts["frequency_penalty"];
  if (opts.contains("repeat_penalty") && opts["repeat_penalty"].is_number()) r.repetition_penalty = opts["repeat_penalty"];
  if (opts.contains("min_p") && opts["min_p"].is_number()) r.min_p = opts["min_p"];
  if (opts.contains("stop")) {
    if (opts["stop"].is_string()) r.stop.push_back(opts["stop"]);
    else if (opts["stop"].is_array())
      for (const auto& s : opts["stop"]) if (s.is_string()) r.stop.push_back(s);
  }
}

static std::shared_ptr<const Grammar> parse_ollama_format(const ojson& body, const ojson& tools) {
  if (!body.contains("format") || body["format"].is_null()) return nullptr;
  OutputSpec spec;
  const ojson& fmt = body["format"];
  if (fmt.is_string() && fmt.get<std::string>() == "json") {
    spec.format = OutputSpec::Format::kJsonObject;
  } else if (fmt.is_object()) {
    // JSON schema directly in the format field
    spec.schema = fmt;
    spec.format = OutputSpec::Format::kJsonSchema;
    spec.strict = true;
  } else {
    return nullptr;
  }
  return request_grammar(spec, tools);
}

static void parse_think(const ojson& body, ojson& kwargs) {
  if (!body.contains("think")) return;
  const ojson& think = body["think"];
  if (think.is_boolean()) {
    kwargs["enable_thinking"] = think.get<bool>();
  } else if (think.is_string()) {
    kwargs["enable_thinking"] = true;
    const std::string level = think.get<std::string>();
    if (level == "low" || level == "medium" || level == "high" || level == "max")
      kwargs["reasoning_effort"] = level;
  }
}

static ojson normalize_ollama_messages(const ojson& messages) {
  ojson out = ojson::array();
  for (const auto& m : messages) {
    if (!m.is_object()) continue;
    ojson msg = ojson::object();
    msg["role"] = (m.contains("role") && m["role"].is_string()) ? m["role"].get<std::string>() : std::string("user");
    // Handle images: Ollama puts them as a list of base64 strings in the message
    if (m.contains("images") && m["images"].is_array() && !m["images"].empty()) {
      ojson parts = ojson::array();
      if (m.contains("content") && m["content"].is_string() && !m["content"].get<std::string>().empty())
        parts.push_back(ojson{{"type", "text"}, {"text", m["content"]}});
      for (const auto& img : m["images"]) {
        if (img.is_string()) {
          parts.push_back(ojson{{"type", "image_url"},
                                {"image_url", {{"url", "data:image/png;base64," + img.get<std::string>()}}}});
        }
      }
      msg["content"] = parts;
    } else {
      // content may be null, absent, or non-string: treat all as ""
      msg["content"] = (m.contains("content") && m["content"].is_string()) ? m["content"].get<std::string>() : std::string();
    }
    // Handle tool_calls from assistant messages
    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
      ojson tcs = ojson::array();
      for (const auto& tc : m["tool_calls"]) {
        if (tc.contains("function") && tc["function"].is_object()) {
          ojson call{{"id", new_id("call_")}, {"type", "function"},
                     {"function", {{"name", tc["function"].value("name", "")},
                                   {"arguments", tc["function"].contains("arguments") ? tc["function"]["arguments"].dump() : "{}"}}}};
          tcs.push_back(call);
        }
      }
      if (!tcs.empty()) msg["tool_calls"] = tcs;
    }
    out.push_back(msg);
  }
  return out;
}

ChatRequest ollama_chat_request(const ojson& body, const ChatRequest& defaults) {
  ChatRequest r = defaults;
  if (body.contains("messages") && body["messages"].is_array())
    r.messages = normalize_ollama_messages(body["messages"]);
  if (body.contains("tools") && body["tools"].is_array()) r.tools = body["tools"];
  if (body.contains("options") && body["options"].is_object()) parse_options(body["options"], r);
  parse_think(body, r.template_kwargs);
  r.grammar = parse_ollama_format(body, r.tools);
  return r;
}

ChatRequest ollama_generate_request(const ojson& body, const ChatRequest& defaults) {
  ChatRequest r = defaults;
  // Build a single user message from the prompt (and optional system)
  ojson messages = ojson::array();
  if (body.contains("system") && body["system"].is_string() && !body["system"].get<std::string>().empty())
    messages.push_back(ojson{{"role", "system"}, {"content", body["system"]}});

  ojson user_msg{{"role", "user"}};
  // Guard: prompt may be null, absent, or non-string
  std::string prompt;
  if (body.contains("prompt") && body["prompt"].is_string()) prompt = body["prompt"].get<std::string>();

  // Handle images in generate
  if (body.contains("images") && body["images"].is_array() && !body["images"].empty()) {
    ojson parts = ojson::array();
    if (!prompt.empty()) parts.push_back(ojson{{"type", "text"}, {"text", prompt}});
    for (const auto& img : body["images"]) {
      if (img.is_string())
        parts.push_back(ojson{{"type", "image_url"},
                              {"image_url", {{"url", "data:image/png;base64," + img.get<std::string>()}}}});
    }
    user_msg["content"] = parts;
  } else {
    user_msg["content"] = prompt;
  }
  messages.push_back(user_msg);
  r.messages = messages;

  if (body.contains("options") && body["options"].is_object()) parse_options(body["options"], r);
  parse_think(body, r.template_kwargs);
  r.grammar = parse_ollama_format(body, ojson::array());
  return r;
}

// /api/chat stream: each line is a JSON object + '\n'. Reasoning in message.thinking.
std::string OllamaChatStream::begin(const ChatResult&) {
  ojson j{{"model", model_},
          {"created_at", ollama_now()},
          {"message", {{"role", "assistant"}, {"content", ""}}},
          {"done", false}};
  return dump(j) + "\n";
}

std::string OllamaChatStream::delta(const ChatDelta& d) {
  std::string out;
  if (!d.reasoning.empty()) {
    ojson j{{"model", model_},
            {"created_at", ollama_now()},
            {"message", {{"role", "assistant"}, {"content", ""}, {"thinking", d.reasoning}}},
            {"done", false}};
    out += dump(j) + "\n";
  }
  if (!d.content.empty()) {
    ojson j{{"model", model_},
            {"created_at", ollama_now()},
            {"message", {{"role", "assistant"}, {"content", d.content}}},
            {"done", false}};
    out += dump(j) + "\n";
  }
  if (!d.tool_calls.empty()) {
    ojson tcs = ojson::array();
    for (const auto& tc : d.tool_calls) {
      ojson args;
      try { args = ojson::parse(tc.arguments); } catch (...) { args = ojson::object(); }
      tcs.push_back(ojson{{"function", {{"name", tc.name}, {"arguments", args}}}});
    }
    ojson j{{"model", model_},
            {"created_at", ollama_now()},
            {"message", {{"role", "assistant"}, {"content", ""}, {"tool_calls", tcs}}},
            {"done", false}};
    out += dump(j) + "\n";
  }
  return out;
}

std::string OllamaChatStream::keepalive() {
  ojson j{{"model", model_},
          {"created_at", ollama_now()},
          {"message", {{"role", "assistant"}, {"content", ""}}},
          {"done", false}};
  return dump(j) + "\n";
}

std::string OllamaChatStream::end(const ChatResult& res) {
  ojson j{{"model", model_},
          {"created_at", ollama_now()},
          {"message", {{"role", "assistant"}, {"content", ""}}},
          {"done_reason", res.finish_reason == "stop" || res.finish_reason == "tool_calls" ? "stop" : res.finish_reason},
          {"done", true}};
  if (res.finish_reason == "error") j["error"] = res.error.empty() ? "request failed" : res.error;
  ollama_eval_stats(j, res);
  return dump(j) + "\n";
}

// /api/generate stream: reasoning in a separate "thinking" field.
std::string OllamaGenerateStream::begin(const ChatResult&) {
  ojson j{{"model", model_},
          {"created_at", ollama_now()},
          {"response", ""},
          {"done", false}};
  return dump(j) + "\n";
}

std::string OllamaGenerateStream::delta(const ChatDelta& d) {
  std::string out;
  if (!d.reasoning.empty()) {
    ojson j{{"model", model_},
            {"created_at", ollama_now()},
            {"response", ""},
            {"thinking", d.reasoning},
            {"done", false}};
    out += dump(j) + "\n";
  }
  if (!d.content.empty()) {
    ojson j{{"model", model_},
            {"created_at", ollama_now()},
            {"response", d.content},
            {"done", false}};
    out += dump(j) + "\n";
  }
  return out;
}

std::string OllamaGenerateStream::keepalive() {
  ojson j{{"model", model_},
          {"created_at", ollama_now()},
          {"response", ""},
          {"done", false}};
  return dump(j) + "\n";
}

std::string OllamaGenerateStream::end(const ChatResult& res) {
  ojson j{{"model", model_},
          {"created_at", ollama_now()},
          {"response", ""},
          {"done_reason", res.finish_reason == "stop" || res.finish_reason == "tool_calls" ? "stop" : res.finish_reason},
          {"done", true}};
  if (res.finish_reason == "error") j["error"] = res.error.empty() ? "request failed" : res.error;
  ollama_eval_stats(j, res);
  return dump(j) + "\n";
}

// Family from name (the catalog has no architecture field).
static std::string model_family(const std::string& name) {
  if (name.find("nemotron") != std::string::npos || name.find("dory") != std::string::npos) return "mamba2";
  if (name.find("flint") != std::string::npos) return "qwen3_moe";
  if (name.find("clover") != std::string::npos) return "qwen3_moe";
  if (name.find("qwen3") != std::string::npos) return "qwen3";
  return "";
}

static ojson model_details(const std::string& name, const std::string& quant) {
  std::string family = model_family(name);
  ojson d{{"parent_model", ""}, {"format", "safetensors"}, {"parameter_size", ""},
          {"quantization_level", quant}};
  if (!family.empty()) {
    d["family"] = family;
    d["families"] = ojson::array({family});
  }
  return d;
}

// Capabilities: the ollama library and Open WebUI read these to decide whether to send tools / thinking / vision.
static ojson model_capabilities(const std::string& name) {
  ojson caps = ojson::array({"completion"});
  if (name.find("qwen3") != std::string::npos || name.find("flint") != std::string::npos ||
      name.find("clover") != std::string::npos || name.find("nemotron") != std::string::npos)
    caps.push_back("tools");
  if (name.find("qwen3") != std::string::npos || name.find("flint") != std::string::npos ||
      name.find("clover") != std::string::npos)
    caps.push_back("thinking");
  // Vision: every catalog model reads images except these text-only ones
  bool text_only = name.find("lightning") != std::string::npos || name.find("dory") != std::string::npos ||
                   name.find("nano-4b") != std::string::npos || name.find("nano-9b") != std::string::npos ||
                   name.find("flint") != std::string::npos || name.find("clover") != std::string::npos;
  if (!text_only) caps.push_back("vision");
  return caps;
}

ojson ollama_tags_response(const std::string& model_id, const std::string& model_name, time_t started) {
  ojson models = ojson::array();

  // The running model
  if (!model_id.empty()) {
    const hub::CatalogModel* cat = hub::find_model(model_id);
    std::string quant;
    if (cat) {
      const hub::Variant* v = cat->variant();
      if (v) quant = v->quant;
    }
    ojson m;
    m["name"] = model_id;
    m["model"] = model_id;
    m["modified_at"] = ollama_now();
    m["size"] = 0;
    m["digest"] = "";
    m["details"] = model_details(model_id, quant);
    models.push_back(m);
  }

  // Also list installed models (skip the running one to avoid duplicates).
  // The running model's folder name is model_name with ':' replaced by '-' (e.g. "qwen3.6-35b-a3b:4bit" ->
  // "qwen3.6-35b-a3b-4bit"), which matches the installed list's name.
  std::string active_folder = model_name;
  for (char& c : active_folder) if (c == ':') c = '-';
  try {
    const auto installed = hub::installed();
    for (const auto& inst : installed) {
      if (!inst.complete) continue;
      if (inst.name == model_id || inst.name == model_name || inst.name == active_folder) continue;

      ojson m;
      m["name"] = inst.name;
      m["model"] = inst.name;
      m["modified_at"] = ollama_now();
      m["size"] = int64_t(inst.bytes);
      m["digest"] = "";
      m["details"] = model_details(inst.name, inst.variant);
      models.push_back(m);
    }
  } catch (...) {}

  return ojson{{"models", models}};
}

ojson ollama_show_response(const std::string& model_id, const std::string& model_name) {
  const hub::CatalogModel* cat = hub::find_model(model_id.empty() ? model_name : model_id);
  std::string quant;
  if (cat) {
    const hub::Variant* v = cat->variant();
    if (v) quant = v->quant;
  }
  ojson j;
  j["modelfile"] = "# Kapitsa model\nFROM " + (model_id.empty() ? model_name : model_id);
  j["parameters"] = "";
  j["template"] = "";
  j["details"] = model_details(model_id.empty() ? model_name : model_id, quant);
  j["model_info"] = ojson::object();
  j["capabilities"] = model_capabilities(model_id.empty() ? model_name : model_id);
  return j;
}

ojson ollama_ps_response(const std::string& model_id, const std::string& model_name, time_t started) {
  ojson models = ojson::array();
  if (!model_id.empty()) {
    const hub::CatalogModel* cat = hub::find_model(model_id);
    std::string quant;
    if (cat) {
      const hub::Variant* v = cat->variant();
      if (v) quant = v->quant;
    }
    ojson m;
    m["name"] = model_id;
    m["model"] = model_id;
    m["size"] = 0;
    m["digest"] = "";
    m["details"] = model_details(model_id, quant);

    // Kapitsa keeps models loaded (no keep_alive timeout); expires_at is omitted
    m["size_vram"] = 0;
    models.push_back(m);
  }
  return ojson{{"models", models}};
}

}  // namespace ni::api
