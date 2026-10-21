#include "engine/grammar.h"

#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <algorithm>
#include <bitset>
#include <climits>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

namespace ni {

namespace gram {

// Symbols the matcher reads: bytes 0-255, then the added tokens as atoms
constexpr int kSymOpen = 256, kSymClose = 257, kSymOther = 258, kSymEos = 259;
constexpr int kMaxDepth = 128;          // frames a parse may hold (nesting)
constexpr int kMaxThreads = 32;         // parses kept at once (anyOf)
constexpr int kWsMax = 48, kNlMax = 2;  // whitespace between two JSON tokens: bytes, newlines
constexpr int kIntDigits = 19, kNumDigits = 20, kFracDigits = 24, kExpDigits = 3;
constexpr int kMaxNodes = 200000;
constexpr int kMaxSchemaDepth = 64;
constexpr uint32_t kOff = 0xffffffffu;  // a key or a parameter name off the declared names' trie
constexpr uint32_t kAdd = 0xffffffffu;  // the property being written is an additional one (Frame::a)
constexpr uint16_t kAddN = 0xffff;      // the same for a tool parameter (Frame::n)

// ---- lexical states of free text: a JSON string's content, a raw tool argument, plain text ----
enum Lexer : int { kLexNone = 0, kLexStr = 1, kLexRaw = 2, kLexFree = 3, kLexName = 4 };
constexpr int kLexDead = -1, kLexExit = -2;

// UTF-8 well-formedness: 0 at a character boundary; 1 one continuation byte left; 2-4 two left (the first A0-BF after
// E0, 80-9F after ED, else 80-BF); 5-7 three left (the first 90-BF after F0, 80-8F after F4, else 80-BF). -1: invalid.
inline int utf8_step(int u, int c) {
  switch (u) {
    case 0:
      if (c < 0x80) return 0;
      if (c >= 0xC2 && c <= 0xDF) return 1;
      if (c == 0xE0) return 2;
      if (c == 0xED) return 3;
      if (c >= 0xE1 && c <= 0xEF) return 4;
      if (c == 0xF0) return 5;
      if (c == 0xF4) return 6;
      if (c >= 0xF1 && c <= 0xF3) return 7;
      return -1;
    case 1: return c >= 0x80 && c <= 0xBF ? 0 : -1;
    case 2: return c >= 0xA0 && c <= 0xBF ? 1 : -1;
    case 3: return c >= 0x80 && c <= 0x9F ? 1 : -1;
    case 4: return c >= 0x80 && c <= 0xBF ? 1 : -1;
    case 5: return c >= 0x90 && c <= 0xBF ? 4 : -1;
    case 6: return c >= 0x80 && c <= 0x8F ? 4 : -1;
    case 7: return c >= 0x80 && c <= 0xBF ? 4 : -1;
  }
  return -1;
}

inline bool is_hex(int c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

// A JSON string's content after its opening quote: sigma = utf8 state + 8 * escape state (0 none, 1 after '\', 2-5
// the \u escape's 4..1 hex digits to come). kLexExit at the closing quote.
inline int lex_str(int s, int c) {
  const int u = s & 7, e = s >> 3;
  if (e == 0) {
    if (u != 0) {
      const int r = utf8_step(u, c);
      return r < 0 ? kLexDead : r;
    }
    if (c == '"') return kLexExit;
    if (c == '\\') return 8;
    if (c < 0x20) return kLexDead;
    const int r = utf8_step(0, c);
    return r < 0 ? kLexDead : r;
  }
  if (e == 1) {
    if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') return 0;
    return c == 'u' ? 16 : kLexDead;
  }
  if (!is_hex(c)) return kLexDead;
  return e == 5 ? 0 : 8 * (e + 1);
}

// A raw tool argument (Qwen's XML): any well-formed UTF-8 that does not hold "</parameter>", ended by
// "\n</parameter>" (kLexExit at its '>'). sigma = utf8 state | last byte '\n' << 3 | the open match began after a
// '\n' << 4 | the open match's length of "</parameter>" << 5.
constexpr char kParamEnd[] = "</parameter>";
inline int lex_raw(int s, int c) {
  const int nu = utf8_step(s & 7, c);
  if (nu < 0) return kLexDead;
  const int p = (s >> 3) & 1;
  int m = (s >> 4) & 1, k = s >> 5;
  if (k > 0 && c == kParamEnd[k]) {
    if (++k == 12) return m ? kLexExit : kLexDead;
  } else if (c == '<') {
    k = 1;
    m = p;
  } else {
    k = 0;
  }
  return nu | (c == '\n') << 3 | m << 4 | k << 5;
}

// A tool parameter's name past the declared ones: well-formed UTF-8 without '<' or a line break, ended by '>'
inline int lex_name(int s, int c) {
  if (s == 0 && c == '>') return kLexExit;
  if (c == '<' || c == '\n') return kLexDead;
  const int r = utf8_step(s, c);
  return r < 0 ? kLexDead : r;
}

inline int lex(int lexer, int s, int c) {
  switch (lexer) {
    case kLexStr: return lex_str(s, c);
    case kLexRaw: return lex_raw(s, c);
    case kLexFree: return 0;
    case kLexName: return lex_name(s, c);
  }
  return kLexDead;
}

// A set of byte strings (enum values, property names, tool names), as a trie
struct LitTrie {
  struct Node {
    std::vector<std::pair<uint8_t, int>> next;
    int val = -1;  // >= 0: a string ends here (its value)
  };
  std::vector<Node> n{Node{}};
  int child(int s, int c) const {
    if (c < 0 || c > 255) return -1;
    for (const auto& [b, t] : n[size_t(s)].next)
      if (b == c) return t;
    return -1;
  }
  void add(const std::string& s, int val) {
    int at = 0;
    for (const unsigned char b : s) {
      int c = child(at, b);
      if (c < 0) {
        c = int(n.size());
        n[size_t(at)].next.push_back({b, c});
        n.emplace_back();
      }
      at = c;
    }
    if (n[size_t(at)].val < 0) n[size_t(at)].val = val;
  }
  bool empty() const {
    for (const Node& x : n)
      if (x.val >= 0) return false;
    return true;
  }
  // the values ending in each node's subtree (the first 64; `big`: one past them), for names: a key that can only
  // become a name already written is given up at once instead of at its closing quote
  std::vector<uint64_t> sub;
  std::vector<char> big;
  void finish() {
    sub.assign(n.size(), 0);
    big.assign(n.size(), 0);
    for (size_t i = n.size(); i-- > 0;) {  // children are created after their parent
      if (n[i].val >= 0) (n[i].val < 64 ? sub[i] |= uint64_t(1) << n[i].val : big[i] = 1);
      for (const auto& [b, c] : n[i].next) sub[i] |= sub[size_t(c)], big[i] |= big[size_t(c)];
    }
  }
  // a name not in `seen` (and allowed) can still be completed from node s
  bool open(int s, uint64_t seen, uint64_t allowed) const { return big[size_t(s)] || (sub[size_t(s)] & ~seen & allowed); }
};

enum class NK : uint8_t { Any, Never, Null, Bool, Int, Num, Str, Lit, Obj, Arr, Alt };

struct GNode {
  NK kind = NK::Any;
  // Int: bounds; Num: no minus sign
  bool lo_set = false, hi_set = false, nonneg = false;
  int64_t lo = 0, hi = 0;
  int lit = -1;  // Lit (and the Bool and Null nodes): the trie of its texts
  // Obj, and a tool's parameters: the declared properties, required ones first
  std::vector<std::string> names;
  std::vector<int> vals;
  std::vector<uint8_t> raw;  // tool parameters: 1 written as raw text (declared "type": "string"), 2 one of raw values
  int nreq = 0;
  int additional = -1;  // other keys' schema, -1: none
  int keys = -1;        // trie of the names (JSON string bodies; for tool parameters raw names + '>'), val = index
  uint64_t allowed = 0; // the properties (of the first 64) that may be written: their schema allows a value
  // Arr
  std::vector<int> prefix;
  int items = -1;  // items past `prefix`, -1: none
  int64_t min_items = 0, max_items = INT64_MAX;
  // Str: the characters a string holds (minLength, maxLength; -1: no most)
  int64_t min_len = 0, max_len = -1;
  bool limited() const { return min_len > 0 || max_len >= 0; }
  // Alt
  std::vector<int> alts;
  std::bitset<256> first;  // bytes a value can start with
};

}  // namespace gram

using namespace gram;

class Grammar {
 public:
  enum class Top : uint8_t { kJson, kCalls, kJsonOrCalls, kTextOrCalls };
  // well-known nodes, first in every grammar
  static constexpr int kAny = 0, kNever = 1, kAnyObj = 2, kAnyArr = 3, kAnyStr = 4, kAnyNum = 5, kAnyBool = 6,
                       kAnyNull = 7, kAnyInt = 8;
  Top top = Top::kJson;
  int root = -1;  // the JSON value (kJson, kJsonOrCalls)
  struct Tool {
    std::string name;
    int params = -1;  // the parameters' node (an Obj), -1: the call's body is free
  };
  std::vector<Tool> tools;
  int names = -1;  // trie of the tools' names + '>'
  bool parallel = true;
  std::vector<GNode> nodes;
  std::vector<LitTrie> tries;

  Grammar() {
    auto lit = [&](std::initializer_list<const char*> texts) {
      LitTrie t;
      int k = 0;
      for (const char* s : texts) t.add(s, k++);
      tries.push_back(std::move(t));
      return int(tries.size()) - 1;
    };
    nodes.resize(9);
    nodes[kAny].kind = NK::Any;
    nodes[kNever].kind = NK::Never;
    nodes[kAnyObj].kind = NK::Obj;
    nodes[kAnyObj].additional = kAny;
    nodes[kAnyArr].kind = NK::Arr;
    nodes[kAnyArr].items = kAny;
    nodes[kAnyStr].kind = NK::Str;
    nodes[kAnyNum].kind = NK::Num;
    nodes[kAnyBool].kind = NK::Bool;
    nodes[kAnyBool].lit = lit({"true", "false"});
    nodes[kAnyNull].kind = NK::Null;
    nodes[kAnyNull].lit = lit({"null"});
    nodes[kAnyInt].kind = NK::Int;
  }
};

namespace gram {

// ---- JSON Schema -> nodes ----

std::string json_text(const ojson& v) { return v.dump(); }  // canonical: compact, UTF-8 kept

class Compiler {
 public:
  Compiler(Grammar& g, const ojson& doc, bool strict) : g_(g), doc_(doc), strict_(strict) {}

  int value(const ojson& s, int depth) {
    if (depth > kMaxSchemaDepth) throw GrammarError("the schema nests deeper than 64 levels");
    if (int(g_.nodes.size()) > kMaxNodes) throw GrammarError("the schema is too large");
    if (s.is_boolean()) return s.get<bool>() ? Grammar::kAny : Grammar::kNever;
    if (!s.is_object()) throw GrammarError("a schema must be an object or a boolean");
    if (s.contains("$ref")) {
      if (!s["$ref"].is_string()) throw GrammarError("$ref must be a string");
      return ref(s["$ref"].get<std::string>(), depth);
    }
    if (s.contains("allOf")) {
      const ojson& all = s["allOf"];
      if (!all.is_array() || all.empty()) throw GrammarError("allOf must be a non-empty array");
      ojson m = s;
      m.erase("allOf");
      for (const ojson& p : all) m = merge(m, deref(p, depth));
      return value(m, depth + 1);
    }
    if (s.contains("anyOf") && s.contains("oneOf")) throw GrammarError("a schema with both anyOf and oneOf is not supported");
    // oneOf is read as anyOf: a value matching two alternatives is not ruled out
    for (const char* k : {"anyOf", "oneOf"})
      if (s.contains(k)) {
        const ojson& alts = s[k];
        if (!alts.is_array() || alts.empty()) throw GrammarError(std::string(k) + " must be a non-empty array");
        ojson base = s;
        base.erase(k);
        const bool plain = !constrains(base);
        GNode n;
        n.kind = NK::Alt;
        for (const ojson& a : alts)
          n.alts.push_back(plain ? value(a, depth + 1) : value(merge(base, deref(a, depth)), depth + 1));
        return n.alts.size() == 1 ? n.alts[0] : add(std::move(n));
      }
    if (s.contains("const")) return lit({json_text(s["const"])});
    if (s.contains("enum")) {
      const ojson& e = s["enum"];
      if (!e.is_array() || e.empty()) throw GrammarError("enum must be a non-empty array");
      std::vector<std::string> texts;
      const std::vector<std::string> types = type_list(s);
      for (const ojson& v : e)
        if (types.empty() || type_matches(v, types)) texts.push_back(json_text(v));
      return texts.empty() ? Grammar::kNever : lit(texts);
    }
    std::vector<std::string> types = type_list(s);
    if (types.empty()) types = inferred(s);
    if (s.contains("nullable") && s["nullable"].is_boolean() && s["nullable"].get<bool>() && !types.empty() &&
        std::find(types.begin(), types.end(), "null") == types.end())
      types.push_back("null");
    if (types.empty()) return Grammar::kAny;
    if (types.size() == 1) return typed(types[0], s, depth);
    GNode n;
    n.kind = NK::Alt;
    for (const std::string& t : types) n.alts.push_back(typed(t, s, depth));
    return add(std::move(n));
  }

  // A tool's parameters: an object whose top-level properties are the call's XML parameters
  int params(const ojson& s0, int depth) {
    const ojson s = deref(s0, depth);
    if (s.is_boolean()) return object(ojson::object(), depth, true);
    if (!s.is_object()) throw GrammarError("a tool's parameters must be a JSON Schema object");
    ojson m = s;
    if (m.contains("allOf")) {
      const ojson all = m["allOf"];
      m.erase("allOf");
      if (all.is_array())
        for (const ojson& p : all) m = merge(m, deref(p, depth));
    }
    return object(m, depth, true);
  }

 private:
  int add(GNode n) {
    g_.nodes.push_back(std::move(n));
    return int(g_.nodes.size()) - 1;
  }

  int lit(const std::vector<std::string>& texts) {
    LitTrie t;
    for (size_t i = 0; i < texts.size(); ++i) t.add(texts[i], int(i));
    g_.tries.push_back(std::move(t));
    GNode n;
    n.kind = NK::Lit;
    n.lit = int(g_.tries.size()) - 1;
    return add(std::move(n));
  }

  // keywords that constrain a value (a schema without any is "anything")
  static bool constrains(const ojson& s) {
    static const char* const keys[] = {"type", "properties", "required", "additionalProperties", "items", "prefixItems",
                                       "enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                                       "minItems", "maxItems", "$ref", "allOf", "anyOf", "oneOf", "nullable"};
    for (const char* k : keys)
      if (s.contains(k)) return true;
    return false;
  }

  static std::vector<std::string> type_list(const ojson& s) {
    std::vector<std::string> t;
    if (!s.contains("type")) return t;
    const ojson& v = s["type"];
    if (v.is_string()) t.push_back(v.get<std::string>());
    else if (v.is_array())
      for (const ojson& x : v) {
        if (!x.is_string()) throw GrammarError("type must be a string or an array of strings");
        if (std::find(t.begin(), t.end(), x.get<std::string>()) == t.end()) t.push_back(x.get<std::string>());
      }
    else throw GrammarError("type must be a string or an array of strings");
    if (v.is_array() && t.empty()) throw GrammarError("type must name at least one type");
    return t;
  }

  static bool type_matches(const ojson& v, const std::vector<std::string>& types) {
    for (const std::string& t : types) {
      if ((t == "null" && v.is_null()) || (t == "boolean" && v.is_boolean()) || (t == "string" && v.is_string()) ||
          (t == "object" && v.is_object()) || (t == "array" && v.is_array()) || (t == "number" && v.is_number()) ||
          (t == "integer" &&
           (v.is_number_integer() || (v.is_number_float() && std::floor(v.get<double>()) == v.get<double>()))))
        return true;
    }
    return false;
  }

  static std::vector<std::string> inferred(const ojson& s) {
    for (const char* k : {"properties", "required", "additionalProperties", "patternProperties", "minProperties",
                          "maxProperties"})
      if (s.contains(k)) return {"object"};
    for (const char* k : {"items", "prefixItems", "minItems", "maxItems"})
      if (s.contains(k)) return {"array"};
    for (const char* k : {"minLength", "maxLength", "pattern", "format"})
      if (s.contains(k)) return {"string"};
    for (const char* k : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"})
      if (s.contains(k)) return {"number"};
    return {};
  }

  const ojson& target(const std::string& r) const {
    if (r == "#") return doc_;
    if (r.rfind("#/", 0) != 0)
      throw GrammarError("$ref \"" + r + "\": only references within the schema (#/...) are supported");
    // a JSON pointer: '/'-separated tokens, ~1 for '/' and ~0 for '~'
    const ojson* at = &doc_;
    for (size_t p = 2; p <= r.size();) {
      size_t q = r.find('/', p);
      if (q == std::string::npos) q = r.size();
      std::string key;
      for (size_t k = p; k < q; ++k)
        if (r[k] == '~' && k + 1 < q && (r[k + 1] == '0' || r[k + 1] == '1')) key += r[++k] == '0' ? '~' : '/';
        else key += r[k];
      if (at->is_object() && at->contains(key)) {
        at = &(*at)[key];
      } else if (at->is_array() && !key.empty() && key.find_first_not_of("0123456789") == std::string::npos &&
                 std::stoul(key) < at->size()) {
        at = &(*at)[std::stoul(key)];
      } else {
        throw GrammarError("$ref \"" + r + "\" does not resolve within the schema");
      }
      p = q + 1;
    }
    return *at;
  }

  int ref(const std::string& r, int depth) {
    if (auto it = refs_.find(r); it != refs_.end()) return it->second;
    const ojson& t = target(r);
    GNode hold;  // a placeholder while the target compiles (a recursive schema refers to it)
    hold.kind = NK::Alt;
    const int at = add(std::move(hold));
    refs_[r] = at;
    const int n = value(t, depth + 1);
    g_.nodes[size_t(at)] = g_.nodes[size_t(n)];
    return at;
  }

  // a subschema with its $ref resolved (for merging)
  ojson deref(const ojson& s, int depth) const {
    const ojson* p = &s;
    for (int k = 0; p->is_object() && p->contains("$ref") && k < 32; ++k) {
      if (!(*p)["$ref"].is_string()) throw GrammarError("$ref must be a string");
      p = &target((*p)["$ref"].get<std::string>());
    }
    (void)depth;
    return *p;
  }

  // Both schemas' constraints in one (allOf, and an anyOf's alternatives under keywords beside it): properties of both
  // (one in both takes both schemas), the union of the required ones, the stricter bounds, the types both allow
  static ojson merge(const ojson& a, const ojson& b) {
    if (b.is_boolean()) return b.get<bool>() ? a : ojson(false);
    if (a.is_boolean()) return a.get<bool>() ? b : ojson(false);
    if (!a.is_object() || !b.is_object()) throw GrammarError("allOf members must be schemas");
    ojson m = a;
    for (const auto& [k, v] : b.items()) {
      if (!m.contains(k)) {
        m[k] = v;
        continue;
      }
      ojson& o = m[k];
      if (k == "properties" && o.is_object() && v.is_object()) {
        for (const auto& [pk, pv] : v.items())
          o[pk] = o.contains(pk) ? ojson{{"allOf", ojson::array({o[pk], pv})}} : pv;
      } else if (k == "required" && o.is_array() && v.is_array()) {
        for (const ojson& r : v)
          if (std::find(o.begin(), o.end(), r) == o.end()) o.push_back(r);
      } else if (k == "type") {
        std::vector<std::string> x = type_list(m), y = type_list(b), both;
        for (const std::string& t : x)
          if (std::find(y.begin(), y.end(), t) != y.end() ||
              (t == "integer" && std::find(y.begin(), y.end(), "number") != y.end()))
            both.push_back(t);
        for (const std::string& t : y)
          if (t == "integer" && std::find(x.begin(), x.end(), "number") != x.end() &&
              std::find(both.begin(), both.end(), "integer") == both.end())
            both.push_back(t);
        if (both.empty()) return ojson(false);
        o = both.size() == 1 ? ojson(both[0]) : ojson(both);
      } else if (k == "additionalProperties" || k == "items") {
        if ((o.is_boolean() && !o.get<bool>()) || (v.is_boolean() && !v.get<bool>())) o = false;
        else if (o.is_boolean()) o = v;
        else if (!v.is_boolean()) o = ojson{{"allOf", ojson::array({o, v})}};
      } else if (k == "minimum" || k == "exclusiveMinimum" || k == "minItems" || k == "minLength") {
        if (o.is_number() && v.is_number()) o = std::max(o.get<double>(), v.get<double>());
      } else if (k == "maximum" || k == "exclusiveMaximum" || k == "maxItems" || k == "maxLength") {
        if (o.is_number() && v.is_number()) o = std::min(o.get<double>(), v.get<double>());
      } else if (k == "enum" && o.is_array() && v.is_array()) {
        ojson both = ojson::array();
        for (const ojson& x : o)
          if (std::find(v.begin(), v.end(), x) != v.end()) both.push_back(x);
        if (both.empty()) return ojson(false);
        o = both;
      } else {
        o = v;
      }
    }
    return m;
  }

  int typed(const std::string& t, const ojson& s, int depth) {
    if (t == "null") return Grammar::kAnyNull;
    if (t == "boolean") return Grammar::kAnyBool;
    if (t == "string") {  // (pattern, format: not enforced)
      int64_t lo = 0, hi = -1;
      if (s.contains("minLength") && s["minLength"].is_number())
        lo = std::max<int64_t>(0, int64_t(s["minLength"].get<double>()));
      if (s.contains("maxLength") && s["maxLength"].is_number())
        hi = std::max<int64_t>(0, int64_t(s["maxLength"].get<double>()));
      if (lo == 0 && hi < 0) return Grammar::kAnyStr;
      if (hi >= 0 && lo > hi) return Grammar::kNever;
      GNode n;
      n.kind = NK::Str;
      n.min_len = lo;
      n.max_len = hi;
      return add(std::move(n));
    }
    if (t == "object") return object(s, depth, false);
    if (t == "array") return array(s, depth);
    if (t == "integer" || t == "number") {
      GNode n;
      n.kind = t == "integer" ? NK::Int : NK::Num;
      double lo = -INFINITY, hi = INFINITY;
      auto num = [&](const char* k, double& out) {
        if (s.contains(k) && s[k].is_number()) out = s[k].get<double>();
      };
      num("minimum", lo);
      num("maximum", hi);
      // exclusive bounds: a number (draft 6 on), or a boolean on minimum / maximum (draft 4)
      if (s.contains("exclusiveMinimum")) {
        const ojson& e = s["exclusiveMinimum"];
        if (e.is_number()) lo = std::max(lo, std::nextafter(e.get<double>(), INFINITY));
        else if (e.is_boolean() && e.get<bool>() && std::isfinite(lo)) lo = std::nextafter(lo, INFINITY);
      }
      if (s.contains("exclusiveMaximum")) {
        const ojson& e = s["exclusiveMaximum"];
        if (e.is_number()) hi = std::min(hi, std::nextafter(e.get<double>(), -INFINITY));
        else if (e.is_boolean() && e.get<bool>() && std::isfinite(hi)) hi = std::nextafter(hi, -INFINITY);
      }
      if (n.kind == NK::Num) {
        if (lo > hi) return Grammar::kNever;
        if (lo >= 0) {  // (a negative number is the one other bound a byte can rule out early)
          n.nonneg = true;
          return add(std::move(n));
        }
        return Grammar::kAnyNum;
      }
      if (std::isfinite(lo)) {
        const double c = std::ceil(lo);
        n.lo_set = c > -9.2e18;
        n.lo = n.lo_set ? int64_t(std::min(c, 9.2e18)) : 0;
        if (c > 9.2e18) return Grammar::kNever;
      }
      if (std::isfinite(hi)) {
        const double f = std::floor(hi);
        n.hi_set = f < 9.2e18;
        n.hi = n.hi_set ? int64_t(std::max(f, -9.2e18)) : 0;
        if (f < -9.2e18) return Grammar::kNever;
      }
      if (n.lo_set && n.hi_set && n.lo > n.hi) return Grammar::kNever;
      if (!n.lo_set && !n.hi_set) return Grammar::kAnyInt;
      return add(std::move(n));
    }
    throw GrammarError("unknown type \"" + t + "\"");
  }

  int object(const ojson& s, int depth, bool tool) {
    GNode n;
    n.kind = NK::Obj;
    std::vector<std::string> names;
    std::vector<int> vals;
    std::vector<uint8_t> raw;
    std::vector<bool> req;
    if (s.contains("properties")) {
      if (!s["properties"].is_object()) throw GrammarError("properties must be an object");
      for (const auto& [k, v] : s["properties"].items()) {
        if (tool && (k.empty() || k.find_first_of("<>\n") != std::string::npos))
          throw GrammarError("tool parameter name \"" + k + "\" cannot be written in a call (it holds <, > or a line break)");
        names.push_back(k);
        uint8_t r = 0;
        int node;
        // a tool parameter whose schema says "type": "string" is written as raw text (the output parser reads it so);
        // any other as JSON
        if (tool && v.is_object() && v.contains("type") && v["type"].is_string() && v["type"].get<std::string>() == "string") {
          r = 1;
          node = Grammar::kAnyStr;
          const ojson* vals_of =
              v.contains("const") ? nullptr : v.contains("enum") && v["enum"].is_array() ? &v["enum"] : nullptr;
          std::vector<std::string> texts;
          if (v.contains("const") && v["const"].is_string()) texts.push_back(v["const"].get<std::string>());
          if (vals_of)
            for (const ojson& e : *vals_of)
              if (e.is_string()) texts.push_back(e.get<std::string>());
          if (v.contains("const") || vals_of) {
            for (auto it = texts.begin(); it != texts.end();)  // a value holding "</parameter>" cannot be written raw
              it = it->find(kParamEnd) != std::string::npos ? texts.erase(it) : it + 1;
            node = texts.empty() ? Grammar::kNever : lit(texts);
            r = texts.empty() ? 1 : 2;
          }
        } else {
          node = value(v, depth + 1);
        }
        vals.push_back(node);
        raw.push_back(r);
        req.push_back(false);
      }
    }
    if (s.contains("additionalProperties")) {
      const ojson& ap = s["additionalProperties"];
      if (ap.is_boolean()) n.additional = ap.get<bool>() ? Grammar::kAny : -1;
      else if (ap.is_object()) n.additional = value(ap, depth + 1);
      else throw GrammarError("additionalProperties must be a boolean or a schema");
    } else if (s.contains("unevaluatedProperties") && s["unevaluatedProperties"].is_boolean() &&
               !s["unevaluatedProperties"].get<bool>()) {
      n.additional = -1;
    } else if (s.contains("patternProperties")) {
      n.additional = -1;  // keys the patterns would allow cannot be checked: only the declared ones are written
    } else {
      n.additional = strict_ ? -1 : Grammar::kAny;
    }
    if (s.contains("required")) {
      if (!s["required"].is_array()) throw GrammarError("required must be an array of property names");
      for (const ojson& r : s["required"]) {
        if (!r.is_string()) throw GrammarError("required must be an array of property names");
        const std::string k = r.get<std::string>();
        auto it = std::find(names.begin(), names.end(), k);
        if (it == names.end()) {  // required but not declared: a property of the additional properties' schema
          if (tool && (k.empty() || k.find_first_of("<>\n") != std::string::npos))
            throw GrammarError("tool parameter name \"" + k + "\" cannot be written in a call (it holds <, > or a line break)");
          names.push_back(k);
          vals.push_back(n.additional >= 0 ? n.additional : Grammar::kNever);
          raw.push_back(tool ? 1 : 0);
          req.push_back(true);
        } else {
          req[size_t(it - names.begin())] = true;
        }
      }
    }
    // required properties first (the matcher counts the first 64 properties it sees)
    for (int pass = 0; pass < 2; ++pass)
      for (size_t i = 0; i < names.size(); ++i)
        if (req[i] == (pass == 0)) {
          n.names.push_back(names[i]);
          n.vals.push_back(vals[i]);
          n.raw.push_back(raw[i]);
          if (pass == 0) ++n.nreq;
        }
    if (!n.names.empty()) {
      LitTrie t;
      for (size_t i = 0; i < n.names.size(); ++i) {
        const std::string body = tool ? n.names[i] + ">" : [&] {
          const std::string q = json_text(ojson(n.names[i]));
          return q.substr(1, q.size() - 2);
        }();
        t.add(body, int(i));
      }
      g_.tries.push_back(std::move(t));
      n.keys = int(g_.tries.size()) - 1;
    }
    if (!tool && n.names.empty() && n.additional == Grammar::kAny) return Grammar::kAnyObj;
    return add(std::move(n));
  }

  int array(const ojson& s, int depth) {
    GNode n;
    n.kind = NK::Arr;
    n.items = Grammar::kAny;
    if (s.contains("prefixItems")) {
      if (!s["prefixItems"].is_array()) throw GrammarError("prefixItems must be an array of schemas");
      for (const ojson& p : s["prefixItems"]) n.prefix.push_back(value(p, depth + 1));
    }
    if (s.contains("items")) {
      const ojson& it = s["items"];
      if (it.is_array()) {  // draft 4 tuples: items as a list, additionalItems past it
        for (const ojson& p : it) n.prefix.push_back(value(p, depth + 1));
        if (s.contains("additionalItems")) {
          const ojson& ai = s["additionalItems"];
          n.items = ai.is_boolean() ? (ai.get<bool>() ? Grammar::kAny : -1) : value(ai, depth + 1);
        }
      } else if (it.is_boolean()) {
        n.items = it.get<bool>() ? Grammar::kAny : -1;
      } else {
        n.items = value(it, depth + 1);
      }
    }
    if (s.contains("minItems") && s["minItems"].is_number())
      n.min_items = std::max<int64_t>(0, int64_t(s["minItems"].get<double>()));
    if (s.contains("maxItems") && s["maxItems"].is_number())
      n.max_items = std::max<int64_t>(0, int64_t(s["maxItems"].get<double>()));
    if (n.prefix.empty() && n.items == Grammar::kAny && n.min_items == 0 && n.max_items == INT64_MAX) return Grammar::kAnyArr;
    return add(std::move(n));
  }

  Grammar& g_;
  const ojson& doc_;
  bool strict_;
  std::map<std::string, int> refs_;
};

// Values a node allows exist (least fixpoint: a recursive schema that needs an infinite value has none); the slots
// holding a node with none are closed (an optional property's key can no longer be finished, an alternative dropped,
// items past them ended), so that a reply never reaches a position where no token is allowed. The bytes each node's
// values start with.
void finish_nodes(Grammar& g) {
  const size_t N = g.nodes.size();
  std::vector<char> sat(N, 0);
  auto item = [&](const GNode& n, int64_t i) { return i < int64_t(n.prefix.size()) ? n.prefix[size_t(i)] : n.items; };
  for (bool changed = true; changed;) {
    changed = false;
    for (size_t i = 0; i < N; ++i) {
      if (sat[i]) continue;
      const GNode& n = g.nodes[i];
      bool s = false;
      switch (n.kind) {
        case NK::Any: case NK::Null: case NK::Bool: case NK::Num: case NK::Str: s = true; break;
        case NK::Int: s = !(n.lo_set && n.hi_set && n.lo > n.hi); break;
        case NK::Never: s = false; break;
        case NK::Lit: s = !g.tries[size_t(n.lit)].empty(); break;
        case NK::Obj: {
          s = true;
          for (int p = 0; p < n.nreq && s; ++p) s = sat[size_t(n.vals[size_t(p)])];
          break;
        }
        case NK::Arr: {
          s = n.min_items <= n.max_items;
          const int64_t need = std::min<int64_t>(n.min_items, int64_t(n.prefix.size()) + 1);
          for (int64_t k = 0; k < need && s; ++k) {
            const int it = item(n, k);
            s = it >= 0 && sat[size_t(it)];
          }
          if (s && n.min_items > int64_t(n.prefix.size()) && n.items < 0) s = false;
          break;
        }
        case NK::Alt:
          for (int a : n.alts) s = s || sat[size_t(a)];
          break;
      }
      if (s) sat[i] = 1, changed = true;
    }
  }
  for (size_t i = 0; i < N; ++i) {
    GNode& n = g.nodes[i];
    if (n.kind == NK::Alt) {
      std::vector<int> keep;
      for (int a : n.alts)
        if (sat[size_t(a)]) keep.push_back(a);
      n.alts = keep;
    } else if (n.kind == NK::Obj) {
      for (int& v : n.vals)
        if (!sat[size_t(v)]) v = Grammar::kNever;
      if (n.additional >= 0 && !sat[size_t(n.additional)]) n.additional = -1;
    } else if (n.kind == NK::Arr) {
      for (size_t k = 0; k < n.prefix.size(); ++k)
        if (!sat[size_t(n.prefix[k])]) {
          n.prefix.resize(k);
          n.items = -1;
          break;
        }
      if (n.items >= 0 && !sat[size_t(n.items)]) n.items = -1;
      n.max_items = std::min<int64_t>(n.max_items, n.items < 0 ? int64_t(n.prefix.size()) : INT64_MAX);
    }
    if (!sat[i] && i != size_t(Grammar::kNever)) n.kind = NK::Never;
    if (n.kind == NK::Obj)
      for (size_t p = 0; p < n.vals.size() && p < 64; ++p)
        if (n.vals[p] != Grammar::kNever) n.allowed |= uint64_t(1) << p;
  }
  for (LitTrie& t : g.tries) t.finish();
  // first bytes (a fixpoint over alternatives)
  for (size_t i = 0; i < N; ++i) {
    GNode& n = g.nodes[i];
    std::bitset<256> f;
    switch (n.kind) {
      case NK::Any: for (const char c : std::string("{[\"-0123456789tfn")) f.set(uint8_t(c)); break;
      case NK::Null: f.set('n'); break;
      case NK::Bool: f.set('t'), f.set('f'); break;
      case NK::Int: case NK::Num: for (const char c : std::string("-0123456789")) f.set(uint8_t(c)); break;
      case NK::Str: f.set('"'); break;
      case NK::Obj: f.set('{'); break;
      case NK::Arr: f.set('['); break;
      case NK::Lit: for (const auto& [b, t] : g.tries[size_t(n.lit)].n[0].next) f.set(b); break;
      default: break;
    }
    n.first = f;
  }
  for (bool changed = true; changed;) {
    changed = false;
    for (GNode& n : g.nodes)
      if (n.kind == NK::Alt)
        for (int a : n.alts) {
          const std::bitset<256> u = n.first | g.nodes[size_t(a)].first;
          if (u != n.first) n.first = u, changed = true;
        }
  }
}

}  // namespace gram

std::shared_ptr<const Grammar> json_grammar(const ojson& schema, bool strict) {
  auto g = std::make_shared<Grammar>();
  g->top = Grammar::Top::kJson;
  g->root = Compiler(*g, schema, strict).value(schema, 0);
  finish_nodes(*g);
  if (g->nodes[size_t(g->root)].kind == NK::Never) throw GrammarError("the schema allows no value");
  return g;
}

std::shared_ptr<const Grammar> output_grammar(const OutputSpec& spec, const ojson& tools) {
  using F = OutputSpec::Format;
  using T = OutputSpec::Tools;
  const bool fmt = spec.format != F::kText;
  struct Offered {
    std::string name;
    const ojson* params;
    bool strict;
  };
  std::vector<Offered> offered;
  if (tools.is_array())
    for (const ojson& t : tools) {
      if (!t.is_object()) continue;
      const ojson& f = t.contains("function") && t["function"].is_object() ? t["function"] : t;
      if (!f.contains("name") || !f["name"].is_string()) continue;
      const std::string name = f["name"].get<std::string>();
      if (!spec.allowed.empty() && std::find(spec.allowed.begin(), spec.allowed.end(), name) == spec.allowed.end())
        continue;
      const bool strict = std::find(spec.strict_tools.begin(), spec.strict_tools.end(), name) != spec.strict_tools.end();
      offered.push_back({name, f.contains("parameters") ? &f["parameters"] : nullptr, strict});
    }
  auto callable = [](const std::string& n) { return !n.empty() && n.find_first_of("<>\n") == std::string::npos; };
  auto g = std::make_shared<Grammar>();
  g->parallel = spec.parallel;
  std::vector<std::pair<const Offered*, bool>> calls;  // the tools a call may name, and whether its arguments follow the schema
  if (spec.tools == T::kNamed) {
    const Offered* o = nullptr;
    for (const Offered& x : offered)
      if (x.name == spec.tool) o = &x;
    if (!o) throw GrammarError("tool_choice names \"" + spec.tool + "\", which is not one of the request's tools");
    if (!callable(o->name))
      throw GrammarError("tool \"" + o->name + "\" has a name a call cannot write (<, > or a line break)");
    calls.push_back({o, true});
    g->top = Grammar::Top::kCalls;
  } else if (spec.tools == T::kRequired) {
    for (const Offered& x : offered)
      if (callable(x.name)) calls.push_back({&x, true});
    if (calls.empty()) throw GrammarError("tool_choice \"required\" needs at least one tool");
    g->top = Grammar::Top::kCalls;
  } else {
    bool any_strict = false;
    for (const Offered& x : offered) any_strict = any_strict || x.strict;
    if (!fmt && !any_strict && spec.allowed.empty()) return nullptr;
    for (const Offered& x : offered)
      if (callable(x.name)) calls.push_back({&x, x.strict});
    if (calls.empty() && !fmt) return nullptr;  // (text, and no tool to call)
    g->top = calls.empty() ? Grammar::Top::kJson : fmt ? Grammar::Top::kJsonOrCalls : Grammar::Top::kTextOrCalls;
  }
  if (g->top == Grammar::Top::kJson || g->top == Grammar::Top::kJsonOrCalls) {
    if (spec.format == F::kJsonObject) {
      g->root = Grammar::kAnyObj;
    } else {
      try {
        g->root = Compiler(*g, spec.schema, spec.strict).value(spec.schema, 0);
      } catch (const GrammarError& e) {
        throw GrammarError(std::string("response format: ") + e.what());
      }
    }
  }
  LitTrie names;
  for (const auto& [o, constrained] : calls) {
    Grammar::Tool t;
    t.name = o->name;
    if (constrained) {
      static const ojson none = ojson::object();
      const ojson& p = o->params ? *o->params : none;
      try {
        t.params = Compiler(*g, p, o->strict).params(p, 0);
      } catch (const GrammarError& e) {
        // a strict tool's schema must be held; another's (forced by tool_choice) keeps a free call body rather than
        // failing the request (an agent's tool list may hold a schema this subset cannot)
        if (o->strict) throw GrammarError("tool \"" + o->name + "\": " + e.what());
        logi("grammar: tool %s: %s; its arguments are not held to its schema", o->name.c_str(), e.what());
      }
    }
    names.add(t.name + ">", int(g->tools.size()));
    g->tools.push_back(std::move(t));
  }
  if (!g->tools.empty()) {
    g->tries.push_back(std::move(names));
    g->names = int(g->tries.size()) - 1;
  }
  finish_nodes(*g);
  if (g->root >= 0 && g->nodes[size_t(g->root)].kind == NK::Never)
    throw GrammarError("response format: the schema allows no value");
  for (Grammar::Tool& t : g->tools)
    if (t.params >= 0 && g->nodes[size_t(t.params)].kind == NK::Never)
      throw GrammarError("tool \"" + t.name + "\": its parameters' schema allows no arguments");
  return g;
}

bool grammar_calls(const Grammar& g) { return !g.tools.empty(); }

std::string grammar_summary(const Grammar& g) {
  static const char* tops[] = {"JSON", "tool calls", "JSON or tool calls", "text or tool calls"};
  std::string s = tops[int(g.top)];
  if (!g.tools.empty()) {
    s += " (";
    for (size_t i = 0; i < g.tools.size(); ++i) {
      if (i) s += ", ";
      s += g.tools[i].name + (g.tools[i].params >= 0 ? "" : " free");
    }
    s += g.parallel ? ")" : "; one call)";
  }
  s += ", " + std::to_string(g.nodes.size()) + " nodes";
  return s;
}

namespace gram {

// ---- the matcher: parses as stacks of frames, fed one symbol at a time ----

using Frame = GrammarState::Frame;
using Thread = std::vector<Frame>;

enum FK : uint8_t { F_PROG, F_OBJ, F_KEY, F_ARR, F_STR, F_NUM, F_LIT, F_CALL, F_PARAMS, F_RAW };
enum : uint8_t { P_START, P_DONE };
enum : uint8_t { O_OPEN, O_FIRST, O_COLON, O_VALUE, O_AFTER, O_NEXT };
enum : uint8_t { A_OPEN, A_FIRST, A_AFTER, A_ITEM };
enum : uint8_t { N_START, N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };
enum : uint8_t { C_LIT, C_NAME, C_NL, C_BODY, C_ATOM, C_FREE };
enum : uint8_t { X_NEXT, X_LT, X_PLIT, X_PNAME, X_PFREE, X_NL, X_VAL, X_JEND, X_FLIT };

constexpr char kFnOpen[] = "\n<function=";      // after <tool_call>
constexpr char kParamOpen[] = "<parameter=";
constexpr char kParamClose[] = "\n</parameter>\n";  // after a JSON argument
constexpr char kFnClose[] = "</function>";

// Parses, with their storage kept for reuse (no allocation once warm)
struct TSet {
  std::vector<Thread> t;
  int n = 0;
  void clear() { n = 0; }
  int add(const Thread& src) {  // (src may be one of t's own: copied before t grows)
    if (n == int(t.size())) t.push_back(Thread(src));
    else t[size_t(n)].assign(src.begin(), src.end());
    return n++;
  }
  void compact() {
    int w = 0;
    for (int r = 0; r < n; ++r)
      if (!t[size_t(r)].empty()) {
        if (w != r) std::swap(t[size_t(w)], t[size_t(r)]);
        ++w;
      }
    n = w;
  }
};

inline bool is_ws(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// One whitespace byte of a gap between JSON tokens; false past the gap's bounds
inline bool take_ws(Frame& f, int c) {
  int bytes = f.n & 0xff, nl = f.n >> 8;
  if (++bytes > kWsMax) return false;
  if (c == '\n' && ++nl > kNlMax) return false;
  f.n = uint16_t(bytes | nl << 8);
  return true;
}

inline uint64_t req_mask(const GNode& n) {
  const int r = std::min(n.nreq, 64);
  return r == 64 ? ~uint64_t(0) : (uint64_t(1) << r) - 1;
}
inline bool required_ok(const GNode& n, uint64_t seen) { return (seen & req_mask(n)) == req_mask(n); }
inline bool more_keys(const GNode& n, uint64_t seen) {
  if (n.additional >= 0 || n.names.size() > 64) return true;
  for (size_t p = 0; p < n.names.size(); ++p)
    if (!(seen >> p & 1) && n.vals[p] != Grammar::kNever) return true;
  return false;
}

// Integers within bounds: whether digits appended to magnitude v (k digits so far; zero: the number is "0") can still
// give a value in [lo, hi]
bool int_feasible(const GNode& n, bool neg, uint64_t v, int k, bool zero) {
  const __int128 lo = n.lo_set ? __int128(n.lo) : -(__int128(1) << 100), hi = n.hi_set ? __int128(n.hi) : (__int128(1) << 100);
  auto meets = [&](__int128 a, __int128 b) { return a <= hi && b >= lo; };  // [a, b] meets [lo, hi]
  if (zero) return meets(0, 0);
  if (k == 0) return neg ? lo <= 0 : hi >= 0;
  __int128 p = 1;
  for (int j = 0; k + j <= kIntDigits; ++j, p *= 10) {
    const __int128 a = __int128(v) * p, b = a + p - 1;
    if (neg ? meets(-b, -a) : meets(a, b)) return true;
  }
  return false;
}
inline bool int_in(const GNode& n, bool neg, uint64_t v) {
  const __int128 x = neg ? -__int128(v) : __int128(v);
  return (!n.lo_set || x >= n.lo) && (!n.hi_set || x <= n.hi);
}
inline bool num_complete(const GNode& n, const Frame& f) {
  if (!(f.st == N_ZERO || f.st == N_INT || f.st == N_FRAC || f.st == N_EXP)) return false;
  return n.kind != NK::Int || int_in(n, f.a == 1, f.st == N_ZERO ? 0 : f.x);
}

void feed(const Grammar& g, TSet& out, int i, int sym);

// Pushes the frame of a value of `node` that starts with `sym` onto parse i and feeds it sym (alternatives that can
// start with it: one parse each)
void begin_value(const Grammar& g, TSet& out, int i, int node, int sym, int alt_depth = 0) {
  const GNode& nd = g.nodes[size_t(node)];
  if (sym > 255 || !nd.first[size_t(sym)] || alt_depth > 16) {
    out.t[size_t(i)].clear();
    return;
  }
  if (nd.kind == NK::Alt) {
    int c[64], nc = 0;
    for (int a : nd.alts)
      if (g.nodes[size_t(a)].first[size_t(sym)] && nc < 64) c[nc++] = a;
    if (nc == 0) {
      out.t[size_t(i)].clear();
      return;
    }
    for (int k = 1; k < nc; ++k) {
      const int j = out.add(out.t[size_t(i)]);
      begin_value(g, out, j, c[k], sym, alt_depth + 1);
    }
    begin_value(g, out, i, c[0], sym, alt_depth + 1);
    return;
  }
  if (nd.kind == NK::Any) {
    const int to = sym == '{' ? Grammar::kAnyObj : sym == '[' ? Grammar::kAnyArr : sym == '"' ? Grammar::kAnyStr
                 : sym == 't' || sym == 'f' ? Grammar::kAnyBool : sym == 'n' ? Grammar::kAnyNull : Grammar::kAnyNum;
    begin_value(g, out, i, to, sym, alt_depth + 1);
    return;
  }
  Thread& w = out.t[size_t(i)];
  if (int(w.size()) >= kMaxDepth) {
    w.clear();
    return;
  }
  Frame f;
  f.node = uint32_t(node);
  switch (nd.kind) {
    case NK::Null: case NK::Bool: case NK::Lit: f.kind = F_LIT; break;
    case NK::Int: case NK::Num: f.kind = F_NUM, f.st = N_START; break;
    case NK::Str: f.kind = F_STR; break;
    case NK::Obj: f.kind = F_OBJ, f.st = O_OPEN; break;
    case NK::Arr: f.kind = F_ARR, f.st = A_OPEN; break;
    default: w.clear(); return;
  }
  w.push_back(f);
  feed(g, out, i, sym);
}

void push_key(const GNode& nd, Thread& w, uint32_t obj) {
  Frame k;
  k.kind = F_KEY;
  k.node = obj;
  k.a = 0;
  k.x = nd.keys >= 0 ? 0 : kOff;
  w.push_back(k);
}

void feed(const Grammar& g, TSet& out, int i, int sym) {
  for (;;) {
    Thread& w = out.t[size_t(i)];
    if (w.empty()) return;
    Frame& f = w.back();
    const GNode& nd = g.nodes[f.kind == F_CALL ? 0 : size_t(f.node)];  // (a call frame holds its tool's index)
    auto dead = [&] { out.t[size_t(i)].clear(); };
    // a whitespace byte in a gap: true when taken (or the parse died: too much)
    auto gap = [&]() -> bool {
      if (sym > 255 || !is_ws(sym)) return false;
      if (!take_ws(f, sym)) dead();
      return true;
    };
    switch (f.kind) {
      case F_PROG: {
        using Top = Grammar::Top;
        if (f.st == P_START) {
          if (g.top == Top::kTextOrCalls) {  // free text, then calls
            if (sym == kSymOpen) {
              f.st = P_DONE;
              f.n = 0;
              Frame c;
              c.kind = F_CALL;
              c.st = C_LIT;
              w.push_back(c);
              return;
            }
            if (sym > 255) dead();
            return;
          }
          if (gap()) return;
          if (sym == kSymOpen && g.top != Top::kJson) {
            f.st = P_DONE;
            f.n = 0;
            Frame c;
            c.kind = F_CALL;
            c.st = C_LIT;
            w.push_back(c);
            return;
          }
          if (g.top == Top::kJson || g.top == Top::kJsonOrCalls) {
            f.st = P_DONE;
            f.x = 1;  // the reply is the JSON
            begin_value(g, out, i, g.root, sym);
            return;
          }
          dead();
          return;
        }
        // after the value, or a call
        if (f.x == 1 || !g.parallel) {
          dead();
          return;
        }
        if (gap()) return;
        if (sym == kSymOpen) {
          f.n = 0;
          Frame c;
          c.kind = F_CALL;
          c.st = C_LIT;
          w.push_back(c);
          return;
        }
        dead();
        return;
      }
      case F_CALL: {
        switch (f.st) {
          case C_LIT:
            if (sym == kFnOpen[f.a]) {
              if (++f.a == sizeof kFnOpen - 1) f.st = C_NAME, f.a = 0;
              return;
            }
            dead();
            return;
          case C_NAME: {
            const LitTrie& t = g.tries[size_t(g.names)];
            const int c = t.child(int(f.a), sym);
            if (c < 0) {
              dead();
              return;
            }
            f.a = uint32_t(c);
            if (t.n[size_t(c)].val >= 0) f.node = uint32_t(t.n[size_t(c)].val), f.st = C_NL;
            return;
          }
          case C_NL: {
            if (sym != '\n') {
              dead();
              return;
            }
            const int p = g.tools[f.node].params;
            if (p < 0) {
              f.st = C_FREE;
              return;
            }
            f.st = C_BODY;
            Frame x;
            x.kind = F_PARAMS;
            x.node = uint32_t(p);
            x.st = X_NEXT;
            w.push_back(x);
            return;
          }
          case C_BODY:
            if (sym == '\n') f.st = C_ATOM;
            else dead();
            return;
          case C_ATOM:
            if (sym == kSymClose) w.pop_back();
            else dead();
            return;
          case C_FREE:
            if (sym == kSymClose) w.pop_back();
            else if (sym > 255) dead();
            return;
        }
        dead();
        return;
      }
      case F_PARAMS: {
        switch (f.st) {
          case X_NEXT:
            if (sym == '<') f.st = X_LT;
            else dead();
            return;
          case X_LT:
            if (sym == 'p' && more_keys(nd, f.x)) f.st = X_PLIT, f.a = 2;
            else if (sym == '/' && required_ok(nd, f.x)) f.st = X_FLIT, f.a = 2;
            else dead();
            return;
          case X_PLIT:
            if (sym != kParamOpen[f.a]) {
              dead();
              return;
            }
            if (++f.a == sizeof kParamOpen - 1) f.st = X_PNAME, f.a = nd.keys >= 0 ? 0 : kOff;
            return;
          case X_PNAME: {
            if (f.a != kOff) {
              const LitTrie& t = g.tries[size_t(nd.keys)];
              const int c = t.child(int(f.a), sym);
              if (c >= 0) {
                if (nd.additional < 0 && !t.open(c, f.x, nd.allowed)) {  // only names written already
                  dead();
                  return;
                }
                f.a = uint32_t(c);
                const int p = t.n[size_t(c)].val;
                if (p >= 0) {  // a declared name, '>' included
                  if ((p < 64 && (f.x >> p & 1)) || nd.vals[size_t(p)] == Grammar::kNever) dead();
                  else f.n = uint16_t(p), f.st = X_NL;
                }
                return;
              }
              // off the declared names: another parameter's name
              if (nd.additional < 0) {
                dead();
                return;
              }
              if (sym == '>') {
                if (f.a == 0) dead();  // (an empty name)
                else f.n = kAddN, f.st = X_NL;
                return;
              }
              const int r = sym > 255 ? kLexDead : lex_name(0, sym);
              if (r < 0) dead();
              else f.st = X_PFREE, f.a = uint32_t(r);
              return;
            }
            // no declared names
            const int r = sym > 255 || sym == '>' ? kLexDead : lex_name(0, sym);
            if (r < 0) dead();
            else f.st = X_PFREE, f.a = uint32_t(r);
            return;
          }
          case X_PFREE: {  // (f.a: the name's UTF-8 state)
            const int r = sym > 255 ? kLexDead : lex_name(int(f.a), sym);
            if (r == kLexExit) f.n = kAddN, f.st = X_NL;
            else if (r < 0) dead();
            else f.a = uint32_t(r);
            return;
          }
          case X_NL:
            if (sym == '\n') f.st = X_VAL;
            else dead();
            return;
          case X_VAL: {
            if (sym > 255) {
              dead();
              return;
            }
            const int p = f.n == kAddN ? -1 : int(f.n);
            if (p >= 0 && p < 64) f.x |= uint64_t(1) << p;
            const uint8_t raw = p < 0 ? 1 : nd.raw[size_t(p)];
            if (raw == 1) {  // raw text (another parameter's value too: the output parser reads it as a string)
              f.st = X_NEXT;
              Frame r;
              r.kind = F_RAW;
              r.a = 1 << 3;  // after the line break that follows the name: "</parameter>" at once is an empty value
              w.push_back(r);
              continue;      // the raw text takes sym
            }
            f.st = X_JEND;
            f.a = 0;
            if (raw == 2) {
              Frame l;
              l.kind = F_LIT;
              l.node = uint32_t(nd.vals[size_t(p)]);
              w.push_back(l);
              continue;
            }
            begin_value(g, out, i, nd.vals[size_t(p)], sym);
            return;
          }
          case X_JEND:
            if (sym != kParamClose[f.a]) {
              dead();
              return;
            }
            if (++f.a == sizeof kParamClose - 1) f.st = X_NEXT;
            return;
          case X_FLIT:
            if (sym != kFnClose[f.a]) {
              dead();
              return;
            }
            if (++f.a == sizeof kFnClose - 1) w.pop_back();
            return;
        }
        dead();
        return;
      }
      case F_RAW: {
        if (f.st == 1) {
          if (sym == '\n') w.pop_back();
          else dead();
          return;
        }
        if (sym > 255) {
          dead();
          return;
        }
        const int r = lex_raw(int(f.a), sym);
        if (r == kLexDead) dead();
        else if (r == kLexExit) f.st = 1;
        else f.a = uint32_t(r);
        return;
      }
      case F_LIT: {
        const LitTrie& t = g.tries[size_t(nd.lit)];
        const int c = t.child(int(f.a), sym);
        const bool can_end = t.n[f.a].val >= 0;
        if (c >= 0) {
          if (can_end) {  // the value may also end here, the symbol going to what follows it
            const int j = out.add(out.t[size_t(i)]);
            out.t[size_t(j)].pop_back();
            feed(g, out, j, sym);
          }
          Thread& w2 = out.t[size_t(i)];
          w2.back().a = uint32_t(c);
          if (t.n[size_t(c)].val >= 0 && t.n[size_t(c)].next.empty()) w2.pop_back();  // complete: nothing can follow
          return;
        }
        if (can_end) {
          w.pop_back();
          continue;
        }
        dead();
        return;
      }
      case F_STR: {
        if (f.st == 0) {
          if (sym == '"') f.st = 1, f.a = 0, f.x = 0;
          else dead();
          return;
        }
        if (sym > 255) {
          dead();
          return;
        }
        const int r = lex_str(int(f.a), sym);
        if (r == kLexDead) {
          dead();
        } else if (r == kLexExit) {
          if (int64_t(f.x) < nd.min_len) dead();
          else w.pop_back();
        } else {
          // a limited string counts its characters (x): one starts at each byte read at a boundary (f.a 0: no
          // escape, no character open)
          if (nd.limited() && f.a == 0 && ++f.x > uint64_t(nd.max_len) && nd.max_len >= 0) {
            dead();
            return;
          }
          f.a = uint32_t(r);
        }
        return;
      }
      case F_KEY: {
        if (sym > 255) {
          dead();
          return;
        }
        const int r = lex_str(int(f.a), sym);
        if (r == kLexDead) {
          dead();
          return;
        }
        const GNode& on = g.nodes[size_t(f.node)];
        if (r == kLexExit) {
          Frame& parent = w[w.size() - 2];
          const LitTrie* t = on.keys >= 0 ? &g.tries[size_t(on.keys)] : nullptr;
          const int p = t && f.x != kOff ? t->n[size_t(f.x)].val : -1;
          if (p >= 0) {
            if ((p < 64 && (parent.x >> p & 1)) || on.vals[size_t(p)] == Grammar::kNever) {
              dead();
              return;
            }
            parent.a = uint32_t(p);
          } else {
            if (on.additional < 0) {
              dead();
              return;
            }
            parent.a = kAdd;
          }
          w.pop_back();
          return;
        }
        f.a = uint32_t(r);
        if (f.x != kOff) {
          const LitTrie& t = g.tries[size_t(on.keys)];
          const int c = t.child(int(f.x), sym);
          f.x = c < 0 ? kOff : uint64_t(c);
          // a closed object: the key must still be able to become a name not written yet
          if (on.additional < 0 && (c < 0 || !t.open(c, w[w.size() - 2].x, on.allowed))) dead();
        }
        return;
      }
      case F_OBJ: {
        switch (f.st) {
          case O_OPEN:
            if (sym == '{') f.st = O_FIRST, f.n = 0;
            else dead();
            return;
          case O_FIRST:
          case O_NEXT:
            if (gap()) return;
            if (sym == '"' && more_keys(nd, f.x)) {
              f.st = O_COLON;
              f.n = 0;
              push_key(nd, w, f.node);
              return;
            }
            if (sym == '}' && f.st == O_FIRST && required_ok(nd, f.x)) {
              w.pop_back();
              return;
            }
            dead();
            return;
          case O_COLON:
            if (gap()) return;
            if (sym == ':') f.st = O_VALUE, f.n = 0;
            else dead();
            return;
          case O_VALUE: {
            if (gap()) return;
            const int p = f.a == kAdd ? -1 : int(f.a);
            if (p >= 0 && p < 64) f.x |= uint64_t(1) << p;
            f.st = O_AFTER;
            f.n = 0;
            begin_value(g, out, i, p < 0 ? nd.additional : nd.vals[size_t(p)], sym);
            return;
          }
          case O_AFTER:
            if (gap()) return;
            if (sym == ',' && more_keys(nd, f.x)) f.st = O_NEXT, f.n = 0;
            else if (sym == '}' && required_ok(nd, f.x)) w.pop_back();
            else dead();
            return;
        }
        dead();
        return;
      }
      case F_ARR: {
        auto item = [&](int64_t k) { return k < int64_t(nd.prefix.size()) ? nd.prefix[size_t(k)] : nd.items; };
        switch (f.st) {
          case A_OPEN:
            if (sym == '[') f.st = A_FIRST, f.n = 0;
            else dead();
            return;
          case A_FIRST:
          case A_ITEM: {
            if (gap()) return;
            if (sym == ']' && f.st == A_FIRST) {
              if (int64_t(f.a) >= nd.min_items) w.pop_back();
              else dead();
              return;
            }
            const int s = item(f.a);
            if (s < 0 || int64_t(f.a) >= nd.max_items || sym > 255) {
              dead();
              return;
            }
            ++f.a;
            f.st = A_AFTER;
            f.n = 0;
            begin_value(g, out, i, s, sym);
            return;
          }
          case A_AFTER:
            if (gap()) return;
            if (sym == ',' && int64_t(f.a) < nd.max_items && item(f.a) >= 0) f.st = A_ITEM, f.n = 0;
            else if (sym == ']' && int64_t(f.a) >= nd.min_items) w.pop_back();
            else dead();
            return;
        }
        dead();
        return;
      }
      case F_NUM: {
        const bool is_int = nd.kind == NK::Int;
        const bool bounded = is_int && (nd.lo_set || nd.hi_set);
        const bool digit = sym >= '0' && sym <= '9';
        auto end_here = [&]() -> bool {  // the number ends before sym: pop it and feed sym to what follows
          if (!num_complete(nd, f)) return false;
          w.pop_back();
          return true;
        };
        switch (f.st) {
          case N_START:
          case N_MINUS: {
            const bool neg = f.st == N_MINUS;
            if (sym == '-' && !neg) {
              if (nd.nonneg || (bounded && !int_feasible(nd, true, 0, 0, false))) dead();
              else f.st = N_MINUS, f.a = 1;
              return;
            }
            if (sym == '0') {
              if (bounded && !int_feasible(nd, neg, 0, 1, true)) dead();
              else f.st = N_ZERO;
              return;
            }
            if (digit) {
              const uint64_t v = uint64_t(sym - '0');
              if (bounded && !int_feasible(nd, neg, v, 1, false)) dead();
              else f.st = N_INT, f.x = v, f.n = 1;
              return;
            }
            dead();
            return;
          }
          case N_ZERO:
          case N_INT:
            if (digit && f.st == N_INT) {
              if ((f.n & 0xff) >= (is_int ? kIntDigits : kNumDigits)) {
                dead();
                return;
              }
              const uint64_t v = f.x <= (UINT64_MAX - 9) / 10 ? f.x * 10 + uint64_t(sym - '0') : UINT64_MAX;
              if (bounded && !int_feasible(nd, f.a == 1, v, (f.n & 0xff) + 1, false)) {
                dead();
                return;
              }
              f.x = v;
              ++f.n;
              return;
            }
            if (digit) {  // (no digit after a leading 0)
              dead();
              return;
            }
            if (!is_int && sym == '.') {
              f.st = N_DOT;
              return;
            }
            if (!is_int && (sym == 'e' || sym == 'E')) {
              f.st = N_E;
              f.n &= 0xff;
              return;
            }
            if (end_here()) continue;
            dead();
            return;
          case N_DOT:
            if (digit) f.st = N_FRAC, f.n = uint16_t((f.n & 0xff) | 1 << 8);
            else dead();
            return;
          case N_FRAC:
            if (digit) {
              if ((f.n >> 8) >= kFracDigits) dead();
              else f.n = uint16_t(f.n + (1 << 8));
              return;
            }
            if (sym == 'e' || sym == 'E') {
              f.st = N_E;
              f.n &= 0xff;
              return;
            }
            if (end_here()) continue;
            dead();
            return;
          case N_E:
            if (sym == '+' || sym == '-') f.st = N_ESIGN;
            else if (digit) f.st = N_EXP, f.n = uint16_t((f.n & 0xff) | 1 << 8);
            else dead();
            return;
          case N_ESIGN:
            if (digit) f.st = N_EXP, f.n = uint16_t((f.n & 0xff) | 1 << 8);
            else dead();
            return;
          case N_EXP:
            if (digit) {
              if ((f.n >> 8) >= kExpDigits) dead();
              else f.n = uint16_t(f.n + (1 << 8));
              return;
            }
            if (end_here()) continue;
            dead();
            return;
        }
        dead();
        return;
      }
    }
    dead();
    return;
  }
}

// The parse may end here (the reply complete)
bool can_end(const Grammar& g, const Thread& w) {
  for (size_t k = w.size(); k-- > 0;) {
    const Frame& f = w[k];
    const GNode& nd = g.nodes[f.kind == F_CALL ? 0 : size_t(f.node)];
    switch (f.kind) {
      case F_PROG: return f.st == P_DONE || (f.st == P_START && g.top == Grammar::Top::kTextOrCalls);
      case F_NUM:
        if (!num_complete(nd, f)) return false;
        continue;
      case F_LIT:
        if (g.tries[size_t(nd.lit)].n[f.a].val < 0) return false;
        continue;
      default: return false;
    }
  }
  return false;
}

// The parse's top frame reads free text from a lexical state that is all it holds: the tokens that stay inside the
// text then depend on (lexer, sigma) alone
// (a limited string: lo the characters it still needs, hi the most it may still take; else 0 and -1)
bool pure_text(const Grammar& g, const Thread& w, int* lexer, int* sigma, int64_t* lo, int64_t* hi) {
  *lo = 0, *hi = -1;
  if (w.empty()) return false;
  const Frame& f = w.back();
  switch (f.kind) {
    case F_STR: {
      if (f.st != 1) return false;
      *lexer = kLexStr, *sigma = int(f.a);
      const GNode& nd = g.nodes[size_t(f.node)];
      if (nd.limited()) {
        *lo = std::max<int64_t>(0, nd.min_len - int64_t(f.x));
        *hi = nd.max_len >= 0 ? nd.max_len - int64_t(f.x) : -1;
      }
      return true;
    }
    case F_KEY:
      if (f.x != kOff) return false;
      *lexer = kLexStr, *sigma = int(f.a);
      return true;
    case F_RAW:
      if (f.st != 0) return false;
      *lexer = kLexRaw, *sigma = int(f.a);
      return true;
    case F_PROG:
      if (f.st != P_START || g.top != Grammar::Top::kTextOrCalls) return false;
      *lexer = kLexFree, *sigma = 0;
      return true;
    case F_CALL:
      if (f.st != C_FREE) return false;
      *lexer = kLexFree, *sigma = 0;
      return true;
    case F_PARAMS:
      if (f.st != X_PFREE) return false;
      *lexer = kLexName, *sigma = int(f.a);
      return true;
    default: return false;
  }
}

// The parse after its free text ends (the byte that ends it taken)
void leave_text(const Grammar& g, Thread& w) {
  Frame& f = w.back();
  switch (f.kind) {
    case F_STR: w.pop_back(); return;
    case F_KEY: {  // off the declared names: another property
      if (g.nodes[size_t(f.node)].additional < 0) {
        w.clear();
        return;
      }
      w[w.size() - 2].a = kAdd;
      w.pop_back();
      return;
    }
    case F_RAW: f.st = 1; return;
    case F_PARAMS: f.n = kAddN, f.st = X_NL; return;  // (a parameter name: at its '>')
    default: w.clear(); return;  // (free text never ends on a byte)
  }
}

}  // namespace gram

// ---- the vocabulary ----

struct TokenTable::Bulk {
  std::vector<uint32_t> bits;  // the tokens that stay inside the text: all mask words, and/or
  std::vector<int32_t> stay;   // their ids (kept for a JSON string's content, with the characters each adds)
  std::vector<uint16_t> stay_n;
  // a JSON string's content with `bits`: le[t] the tokens adding at most t characters (t <= kLe), and those adding more
  static constexpr int kLe = 16;
  std::vector<std::vector<uint32_t>> le;
  std::vector<std::pair<uint16_t, int32_t>> longer;
  // those that leave it: the bytes after the end of the text, in a trie (preorder; node 0 the empty rest), each node's
  // tokens toks[off[i], off[i + 1]) (with the characters before the end, a JSON string's)
  std::vector<uint8_t> b, d;
  std::vector<uint32_t> e, off;
  std::vector<int32_t> toks;
  std::vector<uint16_t> toks_n;
  int max_depth = 0;
};

TokenTable::TokenTable(const Tokenizer& tok, int model_vocab, const std::vector<int>& eos, int tc_open, int tc_close)
    : tc_open_(tc_open), tc_close_(tc_close) {
  const double t0 = now_ms();
  ntok_ = tok.vocab_size();
  vocab_ = std::max(model_vocab, ntok_);
  bytes_.resize(size_t(ntok_));
  sym_.assign(size_t(ntok_), -1);
  for (int id : eos)
    if (id >= 0 && id < ntok_) eos_.push_back(id);
  std::vector<int> text;
  for (int id = 0; id < ntok_; ++id) {
    if (std::find(eos_.begin(), eos_.end(), id) != eos_.end()) sym_[size_t(id)] = kSymEos;
    else if (id == tc_open) sym_[size_t(id)] = kSymOpen;
    else if (id == tc_close) sym_[size_t(id)] = kSymClose;
    else if (tok.is_added(id) || tok.token_bytes(id).empty()) sym_[size_t(id)] = kSymOther;
    if (sym_[size_t(id)] >= 0) continue;
    bytes_[size_t(id)] = tok.token_bytes(id);
    text.push_back(id);
  }
  std::sort(text.begin(), text.end(), [&](int a, int b) { return bytes_[size_t(a)] < bytes_[size_t(b)]; });
  // the trie in preorder from the sorted strings (a string's prefixes come before it)
  byte_.push_back(0), depth_.push_back(0), tok_.push_back(-1), end_.push_back(0);
  std::vector<uint32_t> path{0};
  const std::string* prev = nullptr;
  int dups = 0;
  for (int id : text) {
    const std::string& s = bytes_[size_t(id)];
    size_t c = 0;
    if (prev)
      while (c < prev->size() && c < s.size() && (*prev)[c] == s[c]) ++c;
    while (path.size() - 1 > c) {
      end_[path.back()] = uint32_t(byte_.size());
      path.pop_back();
    }
    for (size_t k = c; k < s.size(); ++k) {
      path.push_back(uint32_t(byte_.size()));
      byte_.push_back(uint8_t(s[k])), depth_.push_back(uint8_t(std::min<size_t>(k + 1, 255)));
      tok_.push_back(-1), end_.push_back(0);
    }
    if (tok_[path.back()] >= 0) ++dups;  // (two ids with the same bytes: the first is the one the trie gives)
    else tok_[path.back()] = id;
    max_depth_ = std::max(max_depth_, int(s.size()));
    prev = &s;
  }
  while (!path.empty()) {
    end_[path.back()] = uint32_t(byte_.size());
    path.pop_back();
  }
  if (max_depth_ > 255) fatal("grammar: a token of %d bytes", max_depth_);
  logi("grammar: %d tokens in a %zu-node trie%s (%.0f ms)", int(text.size()), byte_.size(),
       dups ? (", " + std::to_string(dups) + " with the bytes of another").c_str() : "", now_ms() - t0);
}

TokenTable::~TokenTable() = default;

void TokenTable::warm() const {
  bulk(0, kLexStr, 0);       // inside a JSON string, at a character boundary
  bulk(0, kLexRaw, 1 << 3);  // a raw tool argument's start (after the line break that follows its name)
  bulk(0, kLexRaw, 0);       // inside one
}

const TokenTable::Bulk& TokenTable::bulk(uint32_t node, int lexer, int sigma) const {
  const uint64_t key = uint64_t(node) << 20 | uint64_t(lexer) << 16 | uint64_t(sigma);
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (auto it = bulk_.find(key); it != bulk_.end()) return *it->second;
  }
  auto b = std::make_unique<Bulk>();
  const int base = depth_[node];
  struct Exit {
    std::string rest;
    int id, chars;
    bool operator<(const Exit& o) const { return rest < o.rest || (rest == o.rest && id < o.id); }
  };
  std::vector<Exit> exits;
  for (uint32_t i = node + 1; i < end_[node]; ++i) {
    const int id = tok_[i];
    if (id < 0) continue;
    const std::string& s = bytes_[size_t(id)];
    int st = sigma, chars = 0;
    size_t k = size_t(base);
    bool alive = true, left = false;
    for (; k < s.size(); ++k) {
      const int r = lex(lexer, st, uint8_t(s[k]));
      if (r == kLexDead) {
        alive = false;
        break;
      }
      if (r == kLexExit) {
        left = true;
        break;
      }
      if (st == 0) ++chars;  // (a JSON string's: a character starts here)
      st = r;
    }
    if (!alive) continue;
    if (left) {
      exits.push_back({s.substr(k + 1), id, chars});
    } else {
      b->stay.push_back(id);
      b->stay_n.push_back(uint16_t(chars));
    }
  }
  if (b->stay.size() > 4096) {
    b->bits.assign(size_t(words()), 0);
    for (int id : b->stay) b->bits[size_t(id) >> 5] |= 1u << (id & 31);
    if (lexer == kLexStr) {  // (for strings with a length limit)
      b->le.assign(Bulk::kLe + 1, std::vector<uint32_t>(size_t(words()), 0));
      for (size_t k = 0; k < b->stay.size(); ++k) {
        const int id = b->stay[k], c = b->stay_n[k];
        if (c > Bulk::kLe) b->longer.push_back({uint16_t(c), id});
        else
          for (int t = c; t <= Bulk::kLe; ++t) b->le[size_t(t)][size_t(id) >> 5] |= 1u << (id & 31);
      }
    }
    std::vector<int32_t>().swap(b->stay), std::vector<uint16_t>().swap(b->stay_n);
  }
  std::sort(exits.begin(), exits.end());
  // exit trie: node 0 = the empty rest
  b->b.push_back(0), b->d.push_back(0), b->e.push_back(0), b->off.push_back(0);
  std::vector<std::vector<std::pair<int32_t, uint16_t>>> at(1);
  std::vector<uint32_t> path{0};
  const std::string* prev = nullptr;
  for (const auto& [rest, id, chars] : exits) {
    size_t c = 0;
    if (prev)
      while (c < prev->size() && c < rest.size() && (*prev)[c] == rest[c]) ++c;
    while (path.size() - 1 > c) {
      b->e[path.back()] = uint32_t(b->b.size());
      path.pop_back();
    }
    for (size_t k = c; k < rest.size(); ++k) {
      path.push_back(uint32_t(b->b.size()));
      b->b.push_back(uint8_t(rest[k])), b->d.push_back(uint8_t(k + 1)), b->e.push_back(0), b->off.push_back(0);
      at.emplace_back();
    }
    at[path.back()].push_back({id, uint16_t(chars)});
    b->max_depth = std::max(b->max_depth, int(rest.size()));
    prev = &rest;
  }
  while (!path.empty()) {
    b->e[path.back()] = uint32_t(b->b.size());
    path.pop_back();
  }
  b->off.resize(b->b.size() + 1);
  uint32_t o = 0;
  for (size_t i = 0; i < at.size(); ++i) {
    b->off[i] = o;
    for (const auto& [id, chars] : at[i]) b->toks.push_back(id), b->toks_n.push_back(chars);
    o += uint32_t(at[i].size());
  }
  b->off[at.size()] = o;
  std::lock_guard<std::mutex> lk(mu_);
  auto [it, fresh] = bulk_.emplace(key, std::move(b));
  return *it->second;
}

// ---- the matcher ----

GrammarMatcher::GrammarMatcher(std::shared_ptr<const Grammar> g, std::shared_ptr<const TokenTable> table)
    : g_(std::move(g)), t_(std::move(table)) {}

GrammarState GrammarMatcher::start() const {
  GrammarState s;
  Frame f;
  f.kind = F_PROG;
  f.st = P_START;
  s.threads.push_back({f});
  return s;
}

namespace {

// the parses of `s` after symbol sym (deduplicated)
void step_state(const Grammar& g, GrammarState& s, int sym) {
  thread_local TSet out;
  out.clear();
  for (const Thread& th : s.threads) {
    const int j = out.add(th);
    feed(g, out, j, sym);
  }
  out.compact();
  s.threads.clear();
  for (int k = 0; k < out.n && int(s.threads.size()) < kMaxThreads; ++k) {
    const Thread& th = out.t[size_t(k)];
    if (std::find(s.threads.begin(), s.threads.end(), th) == s.threads.end()) s.threads.push_back(th);
  }
}

inline void set_bit(uint32_t* w, int id) { w[id >> 5] |= 1u << (id & 31); }

// scratch parse sets per trie depth, one array per walk level (the vocabulary's trie, an exit trie)
struct Scratch {
  std::vector<TSet> s;
  void need(size_t n) {
    if (s.size() < n) s.resize(n);
  }
};

}  // namespace

bool GrammarMatcher::advance_text(GrammarState& s, const std::string& text) const {
  for (const unsigned char c : text) {
    step_state(*g_, s, c);
    if (s.dead()) return false;
  }
  return true;
}
bool GrammarMatcher::advance_tool_open(GrammarState& s) const {
  step_state(*g_, s, kSymOpen);
  return !s.dead();
}
bool GrammarMatcher::advance_tool_close(GrammarState& s) const {
  step_state(*g_, s, kSymClose);
  return !s.dead();
}

bool GrammarMatcher::advance(GrammarState& s, int token) const {
  if (!t_ || token < 0 || token >= t_->ntok_) {
    s.threads.clear();
    return false;
  }
  const int sym = t_->sym_[size_t(token)];
  if (sym == kSymEos) {
    if (!accepting(s)) s.threads.clear();
    return !s.dead();
  }
  if (sym >= 0) {
    step_state(*g_, s, sym);
    return !s.dead();
  }
  return advance_text(s, t_->bytes_[size_t(token)]);
}

bool GrammarMatcher::accepting(const GrammarState& s) const {
  for (const Thread& th : s.threads)
    if (can_end(*g_, th)) return true;
  return false;
}

void GrammarMatcher::apply_bulk(uint32_t node, const Thread& th, int lexer, int sigma, uint32_t* words, int64_t lo,
                                int64_t hi) const {
  const TokenTable::Bulk& b = t_->bulk(node, lexer, sigma);
  // a limited string (lo characters still needed, at most hi more): a token that stays inside it adds at most hi
  // characters, one that ends it adds lo to hi before its quote
  const bool limited = lo > 0 || hi >= 0;
  auto fits = [&](uint16_t c, bool ends) { return (!ends || c >= lo) && (hi < 0 || c <= hi); };
  if (limited && hi >= 0 && !b.le.empty()) {
    const std::vector<uint32_t>& le = b.le[size_t(std::min<int64_t>(hi, TokenTable::Bulk::kLe))];
    for (size_t k = 0; k < le.size(); ++k) words[k] |= le[k];
    for (const auto& [c, id] : b.longer)
      if (c <= hi) set_bit(words, id);
  } else if (limited && hi >= 0) {
    for (size_t k = 0; k < b.stay.size(); ++k)
      if (fits(b.stay_n[k], false)) set_bit(words, b.stay[k]);
  } else if (!b.bits.empty()) {
    for (size_t k = 0; k < b.bits.size(); ++k) words[k] |= b.bits[k];
  } else {
    for (int id : b.stay) set_bit(words, id);
  }
  if (b.toks.empty()) return;
  thread_local Scratch sc;
  sc.need(size_t(b.max_depth) + 2);
  TSet& s0 = sc.s[0];
  s0.clear();
  const int j = s0.add(th);
  leave_text(*g_, s0.t[size_t(j)]);
  s0.compact();
  if (s0.n == 0) return;
  for (uint32_t k = b.off[0]; k < b.off[1]; ++k)  // the text ends with the token
    if (!limited || fits(b.toks_n[k], true)) set_bit(words, b.toks[k]);
  for (uint32_t i = 1; i < b.b.size();) {
    const int d = b.d[i];
    TSet& prev = sc.s[size_t(d - 1)];
    TSet& cur = sc.s[size_t(d)];
    cur.clear();
    for (int k = 0; k < prev.n; ++k) {
      const int x = cur.add(prev.t[size_t(k)]);
      feed(*g_, cur, x, b.b[i]);
    }
    cur.compact();
    if (cur.n == 0) {
      i = b.e[i];
      continue;
    }
    for (uint32_t k = b.off[i]; k < b.off[i + 1]; ++k)
      if (!limited || fits(b.toks_n[k], true)) set_bit(words, b.toks[k]);
    ++i;
  }
}

void GrammarMatcher::walk_thread(const Thread& th, uint32_t* words) const {
  const TokenTable& T = *t_;
  thread_local Scratch sc;
  sc.need(size_t(T.max_depth_) + 2);
  TSet& s0 = sc.s[0];
  s0.clear();
  s0.add(th);
  for (uint32_t i = 1; i < T.byte_.size();) {
    const int d = T.depth_[i];
    TSet& prev = sc.s[size_t(d - 1)];
    TSet& cur = sc.s[size_t(d)];
    cur.clear();
    for (int k = 0; k < prev.n; ++k) {
      const int x = cur.add(prev.t[size_t(k)]);
      feed(*g_, cur, x, T.byte_[i]);
    }
    cur.compact();
    if (cur.n == 0) {
      i = T.end_[i];
      continue;
    }
    if (T.tok_[i] >= 0) set_bit(words, T.tok_[i]);
    int lexer, sigma;
    int64_t lo, hi;
    if (cur.n == 1 && pure_text(*g_, cur.t[0], &lexer, &sigma, &lo, &hi)) {  // the rest of every token below: free text first
      apply_bulk(i, cur.t[0], lexer, sigma, words, lo, hi);
      i = T.end_[i];
      continue;
    }
    ++i;
  }
}

void GrammarMatcher::fill_mask(const GrammarState& s, uint32_t* words) const {
  const TokenTable& T = *t_;
  memset(words, 0, size_t(T.words()) * 4);
  for (const Thread& th : s.threads) {
    int lexer, sigma;
    int64_t lo, hi;
    if (pure_text(*g_, th, &lexer, &sigma, &lo, &hi)) apply_bulk(0, th, lexer, sigma, words, lo, hi);
    else walk_thread(th, words);
    for (const auto& [id, sym] : {std::pair<int, int>{T.tc_open_, kSymOpen}, {T.tc_close_, kSymClose}}) {
      if (id < 0) continue;
      GrammarState one;
      one.threads.push_back(th);
      step_state(*g_, one, sym);
      if (!one.dead()) set_bit(words, id);
    }
    if (can_end(*g_, th))
      for (int id : T.eos_) set_bit(words, id);
  }
}

}  // namespace ni
