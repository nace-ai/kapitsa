// Structured output's grammar and mask engine (src/engine/grammar.h), CPU only, no model loaded:
//  - JSON Schema grammars: texts each schema must accept, leave open (a valid prefix) or reject, at the byte level;
//    schemas it must refuse;
//  - tool-call grammars (Qwen's XML calls; <tool_call> and </tool_call> as tokens): tool_choice required, named, auto
//    with strict tools, a format beside tools;
//  - with a tokenizer (a model folder): every mask along valid replies (their tokens as the tokenizer encodes them) and
//    along random walks through the grammars, against brute force (each token fed to a copy of the state); the mask's
//    time per token over the whole vocabulary.
//   grammar_test [--quick] [MODEL_DIR]   (default: ~/.cache/kapitsa/models/qwen3.5-4b-4bit when it exists; without a
//   tokenizer the mask checks are skipped)
//   exit status 1 if any check fails
#include "engine/grammar.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {

int fails = 0, passes = 0;
std::string g_hist;  // the current random walk's text (failure details)

// a test's name on one line: line breaks, control bytes and bytes past ASCII escaped
std::string esc(const std::string& s) {
  std::string o;
  for (const unsigned char c : s) {
    if (c == '\n') o += "\\n";
    else if (c == '\\') o += "\\\\";
    else if (c >= 0x20 && c < 0x7f) o += char(c);
    else {
      char b[8];
      snprintf(b, sizeof b, "\\x%02x", c);
      o += b;
    }
  }
  return o;
}

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", esc(name).c_str(), ok || detail.empty() ? "" : ("\n     " + esc(detail)).c_str());
  ok ? ++passes : ++fails;
}

ojson J(const char* s) { return ojson::parse(s); }

double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A text with "<tool_call>" / "</tool_call>" standing for those tokens, fed to the matcher
enum class Fed { kAccepted, kOpen, kRejected };
Fed feed_text(const GrammarMatcher& m, const std::string& text) {
  GrammarState s = m.start();
  static const std::string open = "<tool_call>", close = "</tool_call>";
  for (size_t p = 0; p < text.size();) {
    bool ok;
    if (text.compare(p, open.size(), open) == 0) ok = m.advance_tool_open(s), p += open.size();
    else if (text.compare(p, close.size(), close) == 0) ok = m.advance_tool_close(s), p += close.size();
    else ok = m.advance_text(s, std::string(1, text[p])), ++p;
    if (!ok) return Fed::kRejected;
  }
  return m.accepting(s) ? Fed::kAccepted : Fed::kOpen;
}

const char* fed_name(Fed f) { return f == Fed::kAccepted ? "accepted" : f == Fed::kOpen ? "open" : "rejected"; }

struct Case {
  const char* name;
  const char* schema;
  bool strict;
  std::vector<std::string> accept, open, reject;
};

void schema_cases(std::vector<std::pair<std::string, std::shared_ptr<const Grammar>>>* grammars,
                  std::vector<std::pair<std::string, std::string>>* instances) {
  const std::vector<Case> cases = {
      {"object, required, closed",
       R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],
           "additionalProperties":false})",
       false,
       {R"({"name":"Bob","age":42})", R"({ "age" : -3 ,"name": "x" })", R"({"name":"a\"b\\é\n","age":0})",
        "{\n  \"name\": \"Zo\xc3\xab\",\n  \"age\": 7\n}", R"({"name":"","age":-9223372036854775807})"},
       {R"({"name":"Bob")", R"({"name":"Bob","age":4)", R"({)", R"({"na)", R"({"name":"Bob\)"},
       {R"({"name":"Bob"})", R"({"name":"Bob","age":42,"x":1})", R"({"name":"Bob","name":"x","age":1})",
        R"({"name":1,"age":1})", R"({"name":"a","age":4.5})", R"({"name":"a","age":042})",
        R"({"name":"a","age":1,})", R"({"name":"a","age":1} )", R"([])", R"({"name":"a)" "\x01" R"(","age":1})",
        "{\"name\":\"\xff\",\"age\":1}", "{\"name\":\"\xc3\",\"age\":1}", R"({"name":"\q","age":1})",
        R"({"name":"\u12G4","age":1})", "{" + std::string(49, ' ') + "\"name\":\"a\",\"age\":1}",
        "{\n\n\n\"name\":\"a\",\"age\":1}", R"({"age":1,"name":"a","age":2})"}},
      {"enum, const",
       R"({"type":"object","properties":{"color":{"enum":["red","green","blue"]},"k":{"const":5},
           "mode":{"type":"string","enum":["a","b",1]}},"required":["color"]})",
       false,
       {R"({"color":"red"})", R"({"k":5,"color":"blue"})", R"({"color":"green","mode":"b","extra":[1,{"z":null}]})"},
       {R"({"color":"gr)", R"({"color":"green")"},
       {R"({"color":"reds"})", R"({"color":"Red"})", R"({"color":"red","k":6})", R"({"color":"red","k":5.0})",
        R"({"color":"red","mode":1})", R"({"k":5})"}},
      {"array of numbers, 1-3 items",
       R"({"type":"array","items":{"type":"number"},"minItems":1,"maxItems":3})",
       false,
       {"[1]", "[1.5, -2e10, 3E+2]", "[ 0 ]", "[-0.0]", "[1e-7,2]"},
       {"[1,", "[", "[1.", "[1e"},
       {"[]", "[1,2,3,4]", "[1,]", "[01]", "[1.]", "[.5]", "[+1]", "[1e]", "[\"a\"]", "[1 2]", "[NaN]"}},
      {"nested, anyOf with null",
       R"({"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}},
           "meta":{"anyOf":[{"type":"null"},{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}]}},
           "required":["tags","meta"]})",
       false,
       {R"({"tags":[],"meta":null})", R"({"tags":["a","b"],"meta":{"id":3}})",
        R"({"meta":{"id":-1,"x":true},"tags":["é"]})"},
       {R"({"tags":["a")", R"({"tags":[],"meta":{)"},
       {R"({"tags":[],"meta":{}})", R"({"tags":[1],"meta":null})", R"({"tags":[],"meta":nul})",
        R"({"tags":[],"meta":"x"})"}},
      {"recursive $ref",
       R"({"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},
           "kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"],"additionalProperties":false}},
           "$ref":"#/$defs/node"})",
       false,
       {R"({"v":1})", R"({"v":1,"kids":[{"v":2},{"v":3,"kids":[{"v":4,"kids":[]}]}]})"},
       {R"({"v":1,"kids":[{"v":2)"},
       {R"({"v":1,"kids":[{}]})", R"({"v":1,"kids":[{"v":"2"}]})", R"({"v":1,"other":2})"}},
      {"integer bounds",
       R"({"type":"integer","minimum":-5,"maximum":120})",
       false,
       {"0", "-5", "120", "7", "-0", "99"},
       {"-"},
       {"121", "-6", "1000", "-50", "007", "1.0", "1e2", "130"}},
      {"integer exclusive bounds",
       R"({"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":10})",
       false,
       {"1", "9"},
       {},
       {"0", "10", "-1", "-0"}},
      {"non-negative number",
       R"({"type":"number","minimum":0})",
       false,
       {"0", "3.25", "1e3"},
       {},
       {"-1", "-0.5"}},
      {"string or null",
       R"({"type":["string","null"]})",
       false,
       {"null", R"("x")", R"("")"},
       {R"("x)", "nu"},
       {"1", "{}", "nulll", "true"}},
      {"booleans and null",
       R"({"type":"object","properties":{"ok":{"type":"boolean"},"n":{"type":"null"}},"required":["ok","n"]})",
       false,
       {R"({"ok":true,"n":null})", R"({"n":null,"ok":false})"},
       {R"({"ok":tr)"},
       {R"({"ok":"true","n":null})", R"({"ok":1,"n":null})", R"({"ok":true,"n":0})"}},
      {"open object (additional properties)",
       R"({"type":"object","properties":{"a":{"type":"integer"}}})",
       false,
       {R"({"a":1})", R"({"b":"x","a":2,"c":[1,{"d":false}]})", "{}", R"({"ab":1.5})", R"({"":0})"},
       {R"({"a")"},
       {R"({"a":"x"})", R"({"a":1,"a":2})"}},
      {"the same, strict: closed",
       R"({"type":"object","properties":{"a":{"type":"integer"}}})",
       true,
       {R"({"a":1})", "{}"},
       {},
       {R"({"b":1})", R"({"a":1,"b":2})", R"({"ab":1})"}},
      {"oneOf by const",
       R"({"oneOf":[{"type":"object","properties":{"kind":{"const":"circle"},"r":{"type":"number"}},"required":["kind","r"],
                     "additionalProperties":false},
                    {"type":"object","properties":{"kind":{"const":"rect"},"w":{"type":"number"},"h":{"type":"number"}},
                     "required":["kind","w","h"],"additionalProperties":false}]})",
       false,
       {R"({"kind":"circle","r":2})", R"({"w":1,"kind":"rect","h":2})"},
       {R"({"kind":"rect","w":1)"},
       {R"({"kind":"circle","w":2})", R"({"kind":"rect","w":1})", R"({"kind":"square","r":1})"}},
      {"tuple (prefixItems)",
       R"({"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}],"items":false})",
       false,
       {R"(["a",1])", R"(["a"])", "[]"},
       {},
       {R"(["a",1,2])", R"([1,"a"])", R"(["a","b"])"}},
      {"allOf",
       R"({"allOf":[{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},
                    {"properties":{"b":{"type":"integer","minimum":0}},"required":["b"]}],"additionalProperties":false})",
       false,
       {R"({"a":"x","b":0})", R"({"b":3,"a":""})"},
       {},
       {R"({"a":"x"})", R"({"a":"x","b":-1})", R"({"a":"x","b":1,"c":1})"}},
      {"string lengths",
       R"({"type":"object","properties":{"code":{"type":"string","minLength":2,"maxLength":5},
           "note":{"type":"string","maxLength":0}},"required":["code"],"additionalProperties":false})",
       false,
       {R"({"code":"ab"})", R"({"code":"abcde"})", R"({"code":"éé"})", "{\"code\":\"\xc3\xa9\\u00e9x\"}",
        R"({"code":"\n\t"})", R"({"code":"xyz","note":""})", "{\"code\":\"\xf0\x9f\x98\x80\xf0\x9f\x98\x80\"}"},
       {R"({"code":"a)", R"({"code":"abcde)"},
       {R"({"code":"a"})", R"({"code":"abcdef"})", R"({"code":""})", R"({"code":"ab","note":"x"})",
        "{\"code\":\"\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\"}"}},
      {"any JSON value",
       "{}",
       false,
       {"1", R"("s")", R"([1,"2",{"3":[null,true,false]}])", R"({"a":{"b":{}}})", "-0.5e+3"},
       {"[", R"({"a")"},
       {"", "{,}", "[1,]", "tru", "{'a':1}", "1 2"}},
      {"keys with escapes and non-ASCII",
       R"({"type":"object","properties":{"café":{"type":"string"},"a\"b":{"type":"integer"}},
           "required":["café","a\"b"],"additionalProperties":false})",
       false,
       {"{\"caf\xc3\xa9\":\"x\",\"a\\\"b\":1}"},
       {},
       {"{\"cafe\":\"x\",\"a\\\"b\":1}"}},
  };
  for (const Case& c : cases) {
    std::shared_ptr<const Grammar> g;
    try {
      g = json_grammar(ojson::parse(c.schema), c.strict);
    } catch (const std::exception& e) {
      check(false, std::string("schema compiles: ") + c.name, e.what());
      continue;
    }
    GrammarMatcher m(g, nullptr);
    for (const std::string& t : c.accept) {
      const Fed f = feed_text(m, t);
      check(f == Fed::kAccepted, std::string(c.name) + ": accepts " + t, fed_name(f));
      instances->push_back({c.name, t});
    }
    for (const std::string& t : c.open) {
      const Fed f = feed_text(m, t);
      check(f == Fed::kOpen, std::string(c.name) + ": leaves open " + t, fed_name(f));
    }
    for (const std::string& t : c.reject) {
      const Fed f = feed_text(m, t);
      check(f != Fed::kAccepted, std::string(c.name) + ": rejects " + t, fed_name(f));
    }
    grammars->push_back({c.name, g});
  }
  // any JSON object (response_format json_object)
  OutputSpec obj;
  obj.format = OutputSpec::Format::kJsonObject;
  auto go = output_grammar(obj, ojson::array());
  GrammarMatcher mo(go, nullptr);
  for (const char* t : {R"({})", R"({"a":[1,2,{"b":null}],"c":"d"})", "\n{\"x\": 1}"}) {
    check(feed_text(mo, t) == Fed::kAccepted, std::string("json_object: accepts ") + t);
    instances->push_back({"json_object", t});
  }
  for (const char* t : {"[]", "1", R"("x")", "{} ", "{}{}"})
    check(feed_text(mo, t) != Fed::kAccepted, std::string("json_object: rejects ") + t);
  grammars->push_back({"json_object", go});
  check(output_grammar(OutputSpec{}, ojson::array()) == nullptr, "no format, no tools: no grammar");
}

void refused() {
  const char* bad[] = {
      R"({"$ref":"http://example.com/s.json"})",
      R"({"$ref":"#/$defs/missing"})",
      R"({"type":"widget"})",
      R"({"anyOf":[]})",
      R"({"type":"object","properties":{"a":false},"required":["a"]})",
      R"({"type":"integer","minimum":5,"maximum":2})",
      R"({"type":"array","minItems":3,"maxItems":1})",
      R"({"enum":[]})",
      R"({"type":"object","properties":{"a":{"type":"string"}},"required":["b"],"additionalProperties":false})",
      R"({"anyOf":[{"type":"string"}],"oneOf":[{"type":"string"}]})",
      R"({"$defs":{"a":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"})",
      R"(5)",
  };
  for (const char* s : bad) {
    bool threw = false;
    std::string why;
    try {
      json_grammar(ojson::parse(s));
    } catch (const GrammarError& e) {
      threw = true;
      why = e.what();
    }
    check(threw, std::string("refuses ") + s + (threw ? " (" + why + ")" : ""));
  }
  std::string deep = "{\"type\":\"string\"}";
  for (int i = 0; i < 70; ++i) deep = "{\"type\":\"array\",\"items\":" + deep + "}";
  bool threw = false;
  try {
    json_grammar(ojson::parse(deep));
  } catch (const GrammarError&) {
    threw = true;
  }
  check(threw, "refuses a schema nested 70 levels deep");
}

const char* kTools = R"([
  {"type":"function","function":{"name":"get_weather","description":"Weather for a place",
    "parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["c","f"]},
      "days":{"type":"integer","minimum":1,"maximum":14},"detail":{"type":"object","properties":{"hourly":{"type":"boolean"}}}},
      "required":["location"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"run_shell","parameters":{"type":"object","properties":{"command":{"type":"string"}},
      "required":["command"]}}},
  {"type":"function","function":{"name":"noop","parameters":{"type":"object","properties":{}}}}
])";

void tool_cases(std::vector<std::pair<std::string, std::shared_ptr<const Grammar>>>* grammars,
                std::vector<std::pair<std::string, std::string>>* instances) {
  const ojson tools = ojson::parse(kTools);
  auto call = [](const std::string& name, const std::vector<std::pair<std::string, std::string>>& args) {
    std::string s = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& [k, v] : args) s += "<parameter=" + k + ">\n" + v + "\n</parameter>\n";
    return s + "</function>\n</tool_call>";
  };
  const std::string weather = call("get_weather", {{"location", "Paris, France"}, {"days", "3"}});
  const std::string weather2 = call("get_weather", {{"unit", "c"}, {"location", "Oslo"}, {"detail", R"({"hourly":true})"}});
  const std::string shell = call("run_shell", {{"command", "ls -la\n echo </param> done"}});
  {
    OutputSpec s;
    s.tools = OutputSpec::Tools::kRequired;
    auto g = output_grammar(s, tools);
    GrammarMatcher m(g, nullptr);
    for (const std::string& t : {weather, weather2, shell, "\n" + weather, weather + "\n" + shell,
                                 call("noop", {}), call("run_shell", {{"command", ""}}),
                                 std::string("<tool_call>\n<function=run_shell>\n<parameter=command>\n</parameter>\n</function>\n</tool_call>"),
                                 call("run_shell", {{"command", "x"}, {"cwd", "/tmp"}})}) {
      const Fed f = feed_text(m, t);
      check(f == Fed::kAccepted, "required: accepts " + t, fed_name(f));
      instances->push_back({"tool_choice required", t});
    }
    for (const std::string& t : {std::string("Sure.") + weather, call("get_wether", {{"location", "x"}}),
                                 call("get_weather", {{"days", "3"}}), call("get_weather", {{"location", "x"}, {"days", "3.5"}}),
                                 call("get_weather", {{"location", "x"}, {"days", "15"}}),
                                 call("get_weather", {{"location", "x"}, {"unit", "k"}}),
                                 call("get_weather", {{"location", "x"}, {"zone", "y"}}),
                                 call("get_weather", {{"location", "x"}, {"location", "y"}}),
                                 call("run_shell", {{"command", "a</parameter>b"}}),
                                 call("get_weather", {{"location", "x"}, {"detail", R"({"hourly":"yes"})"}}),
                                 weather + "Done.", std::string("")}) {
      const Fed f = feed_text(m, t);
      check(f != Fed::kAccepted, "required: rejects " + t, fed_name(f));
    }
    check(feed_text(m, "<tool_call>\n<function=get_weather>\n<parameter=location>\nPar") == Fed::kOpen,
          "required: a call in progress is open");
    grammars->push_back({"tool_choice required", g});
  }
  {
    OutputSpec s;
    s.tools = OutputSpec::Tools::kNamed;
    s.tool = "run_shell";
    s.parallel = false;
    auto g = output_grammar(s, tools);
    GrammarMatcher m(g, nullptr);
    check(feed_text(m, shell) == Fed::kAccepted, "named: accepts its call");
    check(feed_text(m, weather) == Fed::kRejected, "named: rejects another tool");
    check(feed_text(m, shell + "\n" + shell) == Fed::kRejected, "named, parallel_tool_calls false: rejects a second call");
    grammars->push_back({"tool_choice named", g});
    s.tool = "missing";
    bool threw = false;
    try {
      output_grammar(s, tools);
    } catch (const GrammarError&) {
      threw = true;
    }
    check(threw, "named: refuses a tool the request does not offer");
    OutputSpec r;
    r.tools = OutputSpec::Tools::kRequired;
    threw = false;
    try {
      output_grammar(r, ojson::array());
    } catch (const GrammarError&) {
      threw = true;
    }
    check(threw, "required: refuses a request without tools");
  }
  {
    OutputSpec s;
    s.strict_tools = {"get_weather"};
    auto g = output_grammar(s, tools);
    GrammarMatcher m(g, nullptr);
    for (const std::string& t : {std::string("Hello, no call needed."), std::string(""), "Let me check.\n\n" + weather,
                                 call("run_shell", {{"command", "anything"}, {"free", "form"}})})
      check(feed_text(m, t) == Fed::kAccepted, "auto + strict tool: accepts " + t);
    check(feed_text(m, call("get_weather", {{"days", "2"}})) != Fed::kAccepted,
          "auto + strict tool: rejects the strict tool's call without its required argument");
    instances->push_back({"auto + strict tool", "Let me check.\n\n" + weather});
    grammars->push_back({"auto + strict tool", g});
  }
  {  // more tools than the grammar has nodes: calls to tools without parameters (free bodies), one strict
    ojson many = ojson::array();
    for (int k = 0; k < 40; ++k)
      many.push_back({{"type", "function"}, {"function", {{"name", "tool" + std::to_string(k)}}}});
    many.push_back(tools[0]);
    OutputSpec s;
    s.strict_tools = {"get_weather"};
    auto g = output_grammar(s, many);
    GrammarMatcher m(g, nullptr);
    check(feed_text(m, "On it.\n" + call("tool37", {{"x", "1"}})) == Fed::kAccepted, "40 tools: a free call to the 38th");
    check(feed_text(m, call("tool40", {})) == Fed::kRejected, "40 tools: a tool not offered is rejected");
    check(feed_text(m, call("get_weather", {{"location", "x"}, {"days", "0"}})) == Fed::kRejected,
          "40 tools: the strict one's arguments are held (days below its minimum)");
  }
  {
    OutputSpec s;
    s.format = OutputSpec::Format::kJsonSchema;
    s.schema = J(R"({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]})");
    auto g = output_grammar(s, tools);
    GrammarMatcher m(g, nullptr);
    check(feed_text(m, R"({"answer":"42"})") == Fed::kAccepted, "format + tools: accepts the JSON");
    check(feed_text(m, shell) == Fed::kAccepted, "format + tools: accepts a call");
    check(feed_text(m, "The answer") == Fed::kRejected, "format + tools: rejects free text");
    grammars->push_back({"format + tools", g});
  }
}

// ---- masks against brute force, on a tokenizer ----

struct MaskStats {
  std::vector<double> us;
  long states = 0, compared = 0, bad = 0;
};

std::vector<uint32_t> brute(const GrammarMatcher& m, const GrammarState& s, int vocab, int ntok) {
  std::vector<uint32_t> w(size_t((vocab + 31) / 32), 0);
  for (int t = 0; t < ntok; ++t) {
    GrammarState c = s;
    if (m.advance(c, t)) w[size_t(t) >> 5] |= 1u << (t & 31);
  }
  return w;
}

bool bit(const std::vector<uint32_t>& w, int t) { return w[size_t(t) >> 5] >> (t & 31) & 1; }

std::string tok_str(const Tokenizer& tok, int t) {
  std::string s;
  for (unsigned char c : tok.token_bytes(t)) {
    if (c >= 0x20 && c < 0x7f && c != '\\') s += char(c);
    else {
      char b[8];
      snprintf(b, sizeof b, "\\x%02x", c);
      s += b;
    }
  }
  return std::to_string(t) + " '" + s + "'";
}

// the mask at s (timed), against brute force when `cmp`
void mask_check(const GrammarMatcher& m, const GrammarState& s, const Tokenizer& tok, const TokenTable& tt, bool cmp,
                MaskStats& st, const std::string& where) {
  std::vector<uint32_t> w(size_t(tt.words()));
  m.fill_mask(s, w.data());  // (the first fill of a lexical state builds its cache entry: timed below on a second)
  const double t0 = now_us();
  m.fill_mask(s, w.data());
  st.us.push_back(now_us() - t0);
  ++st.states;
  if (getenv("GRAMMAR_SLOW") && st.us.back() > atof(getenv("GRAMMAR_SLOW")))
    printf("slow mask: %.0f us at %s (after %s)\n", st.us.back(), where.c_str(), g_hist.c_str());
  if (!cmp) return;
  ++st.compared;
  const std::vector<uint32_t> b = brute(m, s, tt.vocab(), tok.vocab_size());
  if (b != w) {
    ++st.bad;
    std::string diff;
    int shown = 0;
    for (int t = 0; t < tt.vocab() && shown < 6; ++t)
      if (bit(b, t) != bit(w, t)) diff += " " + tok_str(tok, t) + (bit(b, t) ? " (brute only)" : " (mask only)"), ++shown;
    check(false, "mask equals brute force at " + where, diff);
  }
  bool any = false;
  for (uint32_t x : w) any = any || x;
  if (!any) check(false, "a live state allows some token: " + where, "after " + g_hist);
}

void mask_tests(const std::string& model_dir, bool quick,
                const std::vector<std::pair<std::string, std::shared_ptr<const Grammar>>>& grammars,
                const std::vector<std::pair<std::string, std::string>>& instances) {
  Tokenizer tok;
  tok.load(model_dir + "/tokenizer.json");
  int vocab = tok.vocab_size();
  {
    std::ifstream f(model_dir + "/config.json");
    if (f) {
      const ojson c = ojson::parse(f, nullptr, false);
      for (const ojson* o : {&c, c.contains("text_config") ? &c["text_config"] : nullptr})
        if (o && o->is_object() && o->contains("vocab_size") && (*o)["vocab_size"].is_number_integer())
          vocab = std::max(vocab, (*o)["vocab_size"].get<int>());
    }
  }
  std::vector<int> eos;
  for (const char* e : {"<|im_end|>", "<|endoftext|>"})
    if (tok.find(e) >= 0) eos.push_back(tok.find(e));
  const double t0 = now_us();
  auto tt = std::make_shared<TokenTable>(tok, vocab, eos, tok.find("<tool_call>"), tok.find("</tool_call>"));
  printf("token table: %d tokens (model vocabulary %d), %zu trie nodes, built in %.0f ms\n", tok.vocab_size(), vocab,
         tt->trie_nodes(), (now_us() - t0) / 1000);
  MaskStats st;
  // valid replies, token by token as the tokenizer encodes them: the mask allows each next token; every few
  // positions (every position with --quick off) against brute force
  for (const auto& [name, text] : instances) {
    std::shared_ptr<const Grammar> g;
    for (const auto& [n, gg] : grammars)
      if (n == name) g = gg;
    if (!g) continue;
    GrammarMatcher m(g, tt);
    GrammarState s = m.start();
    const std::vector<int> ids = tok.encode(text);
    bool ok = true;
    for (size_t k = 0; k <= ids.size() && ok; ++k) {
      const bool cmp = !quick || k % 4 == 0 || k == ids.size();
      mask_check(m, s, tok, *tt, cmp, st, name + " before token " + std::to_string(k) + " of " + text);
      std::vector<uint32_t> w(size_t(tt->words()));
      m.fill_mask(s, w.data());
      if (k == ids.size()) {
        for (int e : eos) ok = ok && bit(w, e);
        check(ok, name + ": the end of a complete reply is allowed: " + text);
        break;
      }
      ok = bit(w, ids[k]);
      if (!ok) check(false, name + ": the mask allows each token of a valid reply: " + text, "token " + tok_str(tok, ids[k]));
      ok = ok && m.advance(s, ids[k]);
    }
  }
  // random walks: a random allowed token at each step (now and then the end of the turn where allowed)
  std::mt19937 rng(1234);
  const int walks = quick ? 2 : 6, steps = quick ? 24 : 48;
  for (const auto& [name, g] : grammars) {
    GrammarMatcher m(g, tt);
    for (int wk = 0; wk < walks; ++wk) {
      GrammarState s = m.start();
      g_hist.clear();
      for (int k = 0; k < steps; ++k) {
        const bool cmp = !quick || k % 3 == 0;
        mask_check(m, s, tok, *tt, cmp, st, name + " walk " + std::to_string(wk) + " step " + std::to_string(k));
        std::vector<uint32_t> w(size_t(tt->words()));
        m.fill_mask(s, w.data());
        std::vector<int> allowed;
        for (int t = 0; t < tok.vocab_size(); ++t)
          if (bit(w, t) && std::find(eos.begin(), eos.end(), t) == eos.end()) allowed.push_back(t);
        if (allowed.empty()) break;
        // prefer tokens that hold structure (punctuation) half the time, so walks leave strings and go deep
        std::vector<int> punct;
        for (int t : allowed) {
          const std::string& b = tok.token_bytes(t);
          if (!b.empty() && b.find_first_of("\"{}[]:,<>\n0123456789") != std::string::npos) punct.push_back(t);
        }
        const std::vector<int>& pool = !punct.empty() && rng() % 2 ? punct : allowed;
        const int t = pool[rng() % pool.size()];
        g_hist += tok.token_bytes(t);
        if (!m.advance(s, t)) {
          check(false, name + ": a token the mask allows is accepted", tok_str(tok, t));
          break;
        }
      }
    }
  }
  check(st.bad == 0, "masks equal brute force at " + std::to_string(st.compared) + " states");
  std::vector<double> us = st.us;
  std::sort(us.begin(), us.end());
  double sum = 0;
  for (double x : us) sum += x;
  if (!us.empty())
    printf("mask time per token over %d tokens (%ld states, warm): mean %.1f us, median %.1f, p90 %.1f, p99 %.1f, max %.1f\n",
           vocab, st.states, sum / double(us.size()), us[us.size() / 2], us[us.size() * 9 / 10], us[us.size() * 99 / 100],
           us.back());
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  std::string model;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--quick") quick = true;
    else model = a;
  }
  if (model.empty()) {
    const char* h = getenv("HOME");
    const std::string d = std::string(h ? h : "") + "/.cache/kapitsa/models/qwen3.5-4b-4bit";
    struct stat st;
    if (stat((d + "/tokenizer.json").c_str(), &st) == 0) model = d;
  }
  std::vector<std::pair<std::string, std::shared_ptr<const Grammar>>> grammars;
  std::vector<std::pair<std::string, std::string>> instances;
  schema_cases(&grammars, &instances);
  refused();
  tool_cases(&grammars, &instances);
  if (!model.empty()) mask_tests(model, quick, grammars, instances);
  else printf("(no tokenizer: the mask checks are skipped; pass a model folder)\n");
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
