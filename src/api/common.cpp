#include "api/common.h"
#include "model/dory.h"

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
  else if (effort == "low" || effort == "medium") kwargs["reasoning_effort"] = effort;
  else return effort == "high" || effort == "xhigh" || effort == "max";  // the template's default
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

ojson kapitsa_usage(const ChatResult& res) { return res.loops > 0 ? ojson{{"loops", res.loops}} : ojson(); }

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
