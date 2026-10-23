#pragma once
// A tool call's arguments as text, with each value's own spelling.
//
// The Qwen3.5 and Nemotron templates write a scalar argument through `args_value | string` (a mapping or a list through
// `| tojson`), which spells a JSON `true` as Python's `True`, a list as `[1, 2]` with the template's spacing and a float
// as std::to_string does. The model wrote what it wrote (`true`, `[1,2]`, `2.50`): once the call is parsed into a JSON
// value that spelling is lost, the next prompt differs from the generated text inside the call, and the cache restarts at
// the last checkpoint. So the spelling travels with the call:
//   * the output parser writes the arguments object with each non-string value's text as the model wrote it
//     (object_text), so a client that sends the arguments string back sends the model's own spelling;
//   * the API layers that receive an arguments string keep each top-level value's text in the call's
//     "arguments_raw" object next to the parsed "arguments" (note_raw), and render_chat writes that text where the
//     template would spell the value itself, when it still parses to the value in "arguments".
// A client that sends a parsed object (Anthropic's tool_use input) has no spelling to give: booleans and null go in as
// JSON's `true`, `false` and `null` (a model's `True` is no JSON, it stays a string and comes back as it was), lists and
// objects as the template writes them.
#include "engine/grammar.h"  // ojson
#include <string>
#include <utility>
#include <vector>

namespace ni {

namespace tool_args {

inline bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

// The end of the JSON value that starts at s[i] (no leading whitespace): one past its last character, or npos.
inline size_t value_end(const std::string& s, size_t i) {
  if (i >= s.size()) return std::string::npos;
  if (s[i] == '"') {
    for (size_t j = i + 1; j < s.size(); ++j) {
      if (s[j] == '\\') ++j;
      else if (s[j] == '"') return j + 1;
    }
    return std::string::npos;
  }
  if (s[i] == '{' || s[i] == '[') {
    int depth = 0;
    for (size_t j = i; j < s.size(); ++j) {
      const char c = s[j];
      if (c == '"') {
        const size_t e = value_end(s, j);
        if (e == std::string::npos) return e;
        j = e - 1;
      } else if (c == '{' || c == '[') {
        ++depth;
      } else if (c == '}' || c == ']') {
        if (--depth == 0) return j + 1;
      }
    }
    return std::string::npos;
  }
  size_t j = i;
  while (j < s.size() && s[j] != ',' && s[j] != '}' && s[j] != ']' && !is_ws(s[j])) ++j;
  return j;
}

// The top-level values of a JSON object's text as written: key, text (trimmed). Empty when `text` is no object. A key
// that repeats keeps its last text (as parsing does).
inline std::vector<std::pair<std::string, std::string>> raw_values(const std::string& text) {
  std::vector<std::pair<std::string, std::string>> out;
  const ojson whole = ojson::parse(text, nullptr, false);
  if (!whole.is_object()) return out;
  size_t i = text.find('{');
  if (i == std::string::npos) return out;
  ++i;
  while (true) {
    while (i < text.size() && is_ws(text[i])) ++i;
    if (i >= text.size() || text[i] == '}') break;
    if (text[i] == ',') {
      ++i;
      continue;
    }
    const size_t ke = value_end(text, i);
    if (text[i] != '"' || ke == std::string::npos) return {};
    const ojson key = ojson::parse(text.substr(i, ke - i), nullptr, false);
    if (!key.is_string()) return {};
    i = ke;
    while (i < text.size() && is_ws(text[i])) ++i;
    if (i >= text.size() || text[i] != ':') return {};
    ++i;
    while (i < text.size() && is_ws(text[i])) ++i;
    const size_t ve = value_end(text, i);
    if (ve == std::string::npos) return {};
    std::string raw = text.substr(i, ve - i);
    const std::string k = key.get<std::string>();
    bool found = false;
    for (auto& kv : out)
      if (kv.first == k) {
        kv.second = raw;
        found = true;
      }
    if (!found) out.emplace_back(k, std::move(raw));
    i = ve;
  }
  return out;
}

// An arguments object as JSON text, each key in `raw` (a value's text as written, valid JSON that parses to the object's
// value) written with that text, the others as nlohmann writes them.
inline std::string object_text(const ojson& args, const std::vector<std::pair<std::string, std::string>>& raw) {
  std::string out = "{";
  bool first = true;
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (!first) out += ',';
    first = false;
    out += ojson(it.key()).dump();
    out += ':';
    const std::string* r = nullptr;
    for (auto& kv : raw)
      if (kv.first == it.key()) r = &kv.second;
    if (r && !it.value().is_string() && ojson::parse(*r, nullptr, false) == it.value()) out += *r;
    else out += it.value().dump();
  }
  return out + "}";
}

// The spelling of the arguments string `text` goes into `fn` (a tool call's "function" object) as "arguments_raw": an
// object of key -> text for each value that is neither a string nor an integer spelled as JSON's dump writes it (so
// nothing is added for the common case of strings and integers).
inline void note_raw(ojson& fn, const std::string& text) {
  if (!fn.is_object() || !fn.contains("arguments") || !fn["arguments"].is_object()) return;
  ojson raw = ojson::object();
  for (auto& [k, v] : raw_values(text)) {
    if (!fn["arguments"].contains(k)) continue;
    const ojson& val = fn["arguments"][k];
    if (val.is_string() || (val.is_number_integer() && val.dump() == v)) continue;
    raw[k] = v;
  }
  if (!raw.empty()) fn["arguments_raw"] = raw;
}

}  // namespace tool_args

}  // namespace ni
