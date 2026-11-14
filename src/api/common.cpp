#include "api/common.h"
#include "model/dory.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <mutex>
#include <random>

namespace ni::api {

std::string dump(const ojson& j) { return j.dump(-1, ' ', false, ojson::error_handler_t::replace); }

std::string sse(const std::string& event, const ojson& data) { return "event: " + event + "\ndata: " + dump(data) + "\n\n"; }

std::string new_id(const std::string& prefix) {
  static std::mutex mu;
  static std::mt19937_64 rng(std::random_device{}());
  static const char* al = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::string s = prefix;
  std::lock_guard<std::mutex> lk(mu);
  for (int i = 0; i < 24; ++i) s += al[rng() % 62];
  return s;
}

static const char* kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const std::string& in) {
  std::string out;
  out.reserve((in.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < in.size(); i += 3) {
    const uint32_t v = uint8_t(in[i]) << 16 | uint8_t(in[i + 1]) << 8 | uint8_t(in[i + 2]);
    out += kB64[v >> 18], out += kB64[v >> 12 & 63], out += kB64[v >> 6 & 63], out += kB64[v & 63];
  }
  if (i + 1 == in.size()) {
    const uint32_t v = uint8_t(in[i]) << 16;
    out += kB64[v >> 18], out += kB64[v >> 12 & 63], out += "==";
  } else if (i + 2 == in.size()) {
    const uint32_t v = uint8_t(in[i]) << 16 | uint8_t(in[i + 1]) << 8;
    out += kB64[v >> 18], out += kB64[v >> 12 & 63], out += kB64[v >> 6 & 63], out += '=';
  }
  return out;
}

bool base64_decode(const std::string& in, std::string* out) {
  static int8_t rev[256];
  static std::once_flag once;
  std::call_once(once, [] {
    for (int c = 0; c < 256; ++c) rev[c] = -1;
    for (int c = 0; c < 64; ++c) rev[uint8_t(kB64[c])] = int8_t(c);
  });
  out->clear();
  uint32_t acc = 0;
  int bits = 0;
  size_t n = in.size();
  while (n > 0 && in[n - 1] == '=') --n;
  for (size_t i = 0; i < n; ++i) {
    const int v = rev[uint8_t(in[i])];
    if (v < 0) return false;
    acc = acc << 6 | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      *out += char(acc >> bits & 0xff);
    }
  }
  return true;
}

static const std::string kReasoningTag = "kapitsa.r1.";

std::string pack_reasoning(const std::string& text) { return kReasoningTag + base64_encode(text); }

bool unpack_reasoning(const std::string& blob, std::string* text) {
  if (blob.compare(0, kReasoningTag.size(), kReasoningTag) != 0) return false;
  return base64_decode(blob.substr(kReasoningTag.size()), text);
}

bool apply_effort(const std::string& effort, ojson& kwargs) {
  if (effort == "none" || effort == "minimal") kwargs["enable_thinking"] = false;
  // Anything above minimal asks for thinking, and the rungs that scale it pass reasoning_effort too: no template in
  // the catalog reads reasoning_effort on its own (Qwen3.5/3.6/3.8 and Flint both key off enable_thinking), so low and
  // medium without it were a no-op — a client that asked for less reasoning got no reasoning at all.
  else if (effort == "low" || effort == "medium") {
    kwargs["reasoning_effort"] = effort;
    kwargs["enable_thinking"] = true;
  }
  // high and up ask for thinking explicitly: a template whose default is no thinking (Flint's, so that a client which
  // sends nothing — Finch's "off" omits the field — gets a direct answer) still reasons when the request asks it to.
  else if (effort == "high" || effort == "xhigh" || effort == "max") kwargs["enable_thinking"] = true;
  else return false;
  return true;
}

int request_loops_field(const ojson& body) {
  if (!body.is_object() || !body.contains("loops") || body["loops"].is_null()) return 0;
  const ojson& v = body["loops"];
  if (!v.is_number_integer() || v.get<int64_t>() < 1 || v.get<int64_t>() > Dory::kMaxLoops)
    throw ApiError(400, "invalid_request_error", "loops: a whole number from 1 to " + std::to_string(Dory::kMaxLoops), "",
                   "loops");
  return int(v.get<int64_t>());
}

void request_think_field(const ojson& body, ChatRequest& r) {
  if (!body.is_object() || !body.contains("thinking_control") || body["thinking_control"].is_null()) return;
  const ojson& c = body["thinking_control"];
  auto bad = [](const std::string& m, const std::string& param) {
    throw ApiError(400, "invalid_request_error", "thinking_control." + m, "", "thinking_control." + param);
  };
  if (!c.is_object()) throw ApiError(400, "invalid_request_error", "thinking_control: an object", "", "thinking_control");
  if (c.contains("hard_cap") && !c["hard_cap"].is_null()) {
    if (!c["hard_cap"].is_number_integer() || c["hard_cap"].get<int64_t>() < 0)
      bad("hard_cap: a whole number of tokens, 0 or more", "hard_cap");
    r.think_cap = int(std::min<int64_t>(c["hard_cap"].get<int64_t>(), INT32_MAX));
  }
  if (c.contains("close_text") && !c["close_text"].is_null()) {
    if (!c["close_text"].is_string()) bad("close_text: a string", "close_text");
    r.think_close_text = c["close_text"].get<std::string>();
  }
  if (c.contains("end_on_tool_call") && !c["end_on_tool_call"].is_null()) {
    if (!c["end_on_tool_call"].is_boolean()) bad("end_on_tool_call: a boolean", "end_on_tool_call");
    r.think_end_on_tool_call = c["end_on_tool_call"].get<bool>() ? 1 : 0;
  }
}

ojson kapitsa_usage(const ChatResult& res) {
  ojson u;
  if (res.loops > 0) u["loops"] = res.loops;
  // (reasoning_tokens: the capped block's, the first, its closing </think> included as the cap counts it: at most
  // hard_cap, forced or not, src/engine/think_control.h. The API's own reasoning count covers every block.)
  if (res.think_cap >= 0)
    u["thinking"] = {{"hard_cap", res.think_cap}, {"reasoning_tokens", res.think_tokens}, {"forced", res.think_forced}};
  return u;
}

void join_text(std::string& acc, const std::string& piece) {
  if (piece.empty()) return;
  if (!acc.empty() && acc.back() != '\n' && piece.front() != '\n') acc += '\n';
  acc += piece;
}

ojson arguments_object(const ojson& args) {
  if (args.is_object()) return args;
  if (args.is_string()) {
    const std::string s = args.get<std::string>();
    if (s.find_first_not_of(" \n\r\t") == std::string::npos) return ojson::object();
    ojson j = ojson::parse(s, nullptr, false);
    if (j.is_object()) return j;
    return ojson{{"arguments", s}};
  }
  if (args.is_null()) return ojson::object();
  return ojson{{"arguments", dump(args)}};
}

void schema_format(const ojson& f, OutputSpec& spec, const std::string& param) {
  spec.format = OutputSpec::Format::kJsonSchema;
  if (!f.contains("schema") || f["schema"].is_null()) {
    // no schema: any JSON object, as json_object
    spec.format = OutputSpec::Format::kJsonObject;
  } else if (!f["schema"].is_object() && !f["schema"].is_boolean()) {
    throw ApiError(400, "invalid_request_error", param + ".schema must be a JSON Schema object", "", param + ".schema");
  } else {
    spec.schema = f["schema"];
  }
  if (f.contains("strict") && !f["strict"].is_null()) {
    if (!f["strict"].is_boolean())
      throw ApiError(400, "invalid_request_error", param + ".strict must be a boolean", "", param + ".strict");
    spec.strict = f["strict"].get<bool>();
  }
}

std::shared_ptr<const Grammar> request_grammar(const OutputSpec& spec, const ojson& tools, const std::string& format_param) {
  try {
    return output_grammar(spec, tools);
  } catch (const GrammarError& e) {
    const std::string m = e.what();
    const std::string param = m.rfind("response format", 0) == 0 ? format_param
                              : m.find("tool_choice") != std::string::npos ? "tool_choice"
                                                                          : "tools";
    throw ApiError(400, "invalid_request_error", m, "", param);
  }
}

}  // namespace ni::api
