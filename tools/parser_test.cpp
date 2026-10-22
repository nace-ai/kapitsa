// OutputParser checks on real token ids (CPU only): reasoning / content / tool calls for the cases a tool call
// can take inside or after Qwen's think block, with NI_THINK_TOOLCALLS off and on; GLM-4.5's call markup (Flint's,
// CallFormat::Glm) as the tokenizer encodes it, a token per byte and cut at odd places.
//   parser_test <model_dir>   (reads <model_dir>/tokenizer.json; exit status 1 if any check fails)
#include "engine/output_parser.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace ni;

namespace {

struct Out {
  std::string reasoning, content;
  std::vector<ToolCall> calls;
  std::vector<ChatDelta> deltas;
};

const ojson kTools = ojson::parse(R"([
  {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "start_line": {"type": "integer"}}}}},
  {"type": "function", "function": {"name": "run_shell", "parameters": {"type": "object", "properties": {
    "command": {"type": "string"}}}}}])");

// for the GLM calls: every JSON type a parameter can have, an untyped one, a tool without parameters
const ojson kGlmTools = ojson::parse(R"([
  {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "start_line": {"type": "integer"}}}}},
  {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}, "limit": {"type": "number"}, "regex": {"type": "boolean"},
    "filters": {"type": "object"}, "paths": {"type": "array"}, "note": {}}}}},
  {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}])");

int fails = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("  " + detail).c_str());
  if (!ok) ++fails;
}

std::string calls_str(const std::vector<ToolCall>& v) {
  std::string s;
  for (auto& c : v) s += (s.empty() ? "" : ", ") + c.name + c.arguments;
  return "[" + s + "]";
}

// One generated turn: the ids the model would write after the generation prompt (in the think block when
// `in_think`), ending by itself when `ended`.
Out run_ids(const Tokenizer& tok, const std::vector<int>& ids, bool think_tools, bool in_think, bool ended,
            const ojson& tools, bool think_ws, CallFormat fmt) {
  OutputParser p(tok, tok.find("<think>"), tok.find("</think>"), tok.find("<tool_call>"), tok.find("</tool_call>"),
                 in_think, !tools.empty(), tools, think_tools, think_ws, {}, fmt);
  Out o;
  auto add = [&](const ChatDelta& d) {
    o.reasoning += d.reasoning;
    o.content += d.content;
    for (auto& c : d.tool_calls) o.calls.push_back(c);
    o.deltas.push_back(d);
  };
  for (int id : ids) add(p.push(id));
  add(p.finish(ended));
  return o;
}

// The same with `text` as the tokenizer encodes it, in Qwen's call markup (the parser's default).
Out run(const Tokenizer& tok, const std::string& text, bool think_tools, bool in_think = true, bool ended = true,
        const ojson& tools = kTools, bool think_ws = false) {
  return run_ids(tok, tok.encode(text), think_tools, in_think, ended, tools, think_ws, CallFormat::QwenXml);
}

// A turn in GLM-4.5's call markup with the GLM tools.
Out run_glm(const Tokenizer& tok, const std::vector<int>& ids, bool think_tools, bool in_think, bool ended = true) {
  return run_ids(tok, ids, think_tools, in_think, ended, kGlmTools, false, CallFormat::Glm);
}

// Each byte's single-byte token.
std::vector<int> byte_ids(const Tokenizer& tok) {
  std::vector<int> byte_id(256, -1);
  for (int id = 0; id < tok.vocab_size(); ++id)
    if (tok.token_bytes(id).size() == 1 && byte_id[uint8_t(tok.token_bytes(id)[0])] < 0)
      byte_id[uint8_t(tok.token_bytes(id)[0])] = id;
  return byte_id;
}

// `text` in other tokens than tok.encode gives it: the tags the parser reads by id as their tokens, the text between
// them cut every `cut` bytes and each piece encoded alone (1: a token per byte, cutting characters too; else cut at
// the next character's start), so that <arg_key> and the like come in pieces.
std::vector<int> cut_ids(const Tokenizer& tok, const std::string& text, size_t cut, const std::vector<int>& byte_id) {
  static const std::string tags[] = {"<think>", "</think>", "<tool_call>", "</tool_call>"};
  std::vector<int> ids;
  auto plain = [&](const std::string& s) {
    for (size_t i = 0, n; i < s.size(); i += n) {
      n = std::min(cut, s.size() - i);
      if (cut == 1) {
        ids.push_back(byte_id[uint8_t(s[i])]);
        continue;
      }
      while (i + n < s.size() && (uint8_t(s[i + n]) & 0xC0) == 0x80) ++n;
      for (int id : tok.encode(s.substr(i, n))) ids.push_back(id);
    }
  };
  size_t start = 0, i = 0;
  while (i < text.size()) {
    const std::string* tag = nullptr;
    for (auto& t : tags)
      if (text.compare(i, t.size(), t) == 0) tag = &t;
    if (!tag) {
      ++i;
      continue;
    }
    plain(text.substr(start, i - start));
    ids.push_back(tok.find(*tag));
    start = i += tag->size();
  }
  plain(text.substr(start));
  return ids;
}

bool same_result(const Out& a, const Out& b) {
  return a.reasoning == b.reasoning && a.content == b.content && calls_str(a.calls) == calls_str(b.calls);
}

bool same_stream(const Out& a, const Out& b) {
  if (a.deltas.size() != b.deltas.size()) return false;
  for (size_t i = 0; i < a.deltas.size(); ++i)
    if (a.deltas[i].reasoning != b.deltas[i].reasoning || a.deltas[i].content != b.deltas[i].content ||
        calls_str(a.deltas[i].tool_calls) != calls_str(b.deltas[i].tool_calls))
      return false;
  return true;
}

bool markup_streamed(const Out& o) {
  for (auto& d : o.deltas)
    if (d.reasoning.find("<tool_call>") != std::string::npos || d.reasoning.find("<function=") != std::string::npos)
      return true;
  return false;
}

std::string call(const std::string& fn, const std::string& params) {
  return "<tool_call>\n<function=" + fn + ">\n" + params + "</function>\n</tool_call>";
}
std::string param(const std::string& name, const std::string& value) {
  return "<parameter=" + name + ">\n" + value + "\n</parameter>\n";
}

// GLM-4.5's call markup, as Flint's template renders a call: a value is a string's text or the JSON of anything else
std::string glm_call(const std::string& fn, const std::string& args = "") {
  return "<tool_call>" + fn + args + "</tool_call>";
}
std::string glm_arg(const std::string& key, const std::string& value) {
  return "<arg_key>" + key + "</arg_key><arg_value>" + value + "</arg_value>";
}

std::string show(const Out& o) { return o.reasoning + "|" + o.content + calls_str(o.calls); }

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: parser_test <model_dir>\n");
    return 2;
  }
  Tokenizer tok;
  tok.load(std::string(argv[1]) + "/tokenizer.json");
  for (const char* t : {"<think>", "</think>", "<tool_call>", "</tool_call>"})
    if (tok.find(t) < 0) {
      fprintf(stderr, "tokenizer has no %s\n", t);
      return 2;
    }
  const std::string read_a = call("read_file", param("path", "a.txt"));
  const std::string read_b = call("read_file", param("path", "b.txt"));
  const std::string read_3 = call("read_file", param("path", "notes/step3.txt"));

  // Turns without a call inside the reasoning: the stream is identical with and without the switch.
  struct Same {
    const char* name;
    std::string text;
    bool in_think;
  };
  const std::vector<Same> same = {
      {"think, answer", "The user asks for 2+2.\n</think>\n\nThe answer is 4.", true},
      {"think, call after </think>", "I need the file.\n</think>\n\n" + read_a, true},
      {"think, text and call", "Read it.\n</think>\n\nReading the file.\n\n" + read_a, true},
      {"thinking off: call", "Reading.\n\n" + read_a, false},
      {"thinking off: answer", "Hello there.", false},
      {"unterminated call after </think>", "x\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na", true},
  };
  for (auto& c : same) {
    Out off = run(tok, c.text, false, c.in_think), on = run(tok, c.text, true, c.in_think);
    check(same_stream(off, on), std::string("unchanged stream: ") + c.name,
          "off " + off.reasoning + "|" + off.content + calls_str(off.calls) + " on " + on.reasoning + "|" + on.content +
              calls_str(on.calls));
  }

  // The case this is for: the reasoning ends with a call and the turn ends without </think>.
  {
    const std::string t = "The user wants me to read notes/step3.txt next. Let me do that.\n\n" + read_3;
    Out off = run(tok, t, false), on = run(tok, t, true);
    check(off.calls.empty() && off.reasoning.find("<tool_call>") != std::string::npos,
          "unclosed think, switch off: the call stays reasoning (old behaviour)", calls_str(off.calls));
    check(on.calls.size() == 1 && on.calls[0].name == "read_file" &&
              on.calls[0].arguments == R"({"path":"notes/step3.txt"})",
          "unclosed think, switch on: returned as a tool call", calls_str(on.calls));
    check(on.reasoning == "The user wants me to read notes/step3.txt next. Let me do that." && on.content.empty(),
          "unclosed think, switch on: reasoning without the markup", on.reasoning);
    check(!markup_streamed(on), "unclosed think, switch on: no call markup streamed as reasoning");
    Out on_ws = run(tok, t + "\n", true);
    check(on_ws.calls.size() == 1, "unclosed think + trailing newline: returned as a tool call", calls_str(on_ws.calls));
  }
  {
    const std::string t = "Two files.\n" + read_a + "\n" + read_b + "\n";
    Out on = run(tok, t, true);
    check(on.calls.size() == 2 && calls_str(on.calls) == R"([read_file{"path":"a.txt"}, read_file{"path":"b.txt"}])",
          "two calls at the end of an unclosed think block", calls_str(on.calls));
  }
  {
    const std::string t = "Lines 5 on.\n" + call("read_file", param("path", "a.txt") + param("start_line", "5"));
    Out on = run(tok, t, true);
    check(on.calls.size() == 1 && on.calls[0].arguments == R"({"path":"a.txt","start_line":5})",
          "integer argument typed from the tool schema", calls_str(on.calls));
  }
  {
    // closed think block, then nothing: the turn's only action is the call
    const std::string t = "I'll read it.\n" + read_a + "\n</think>\n\n";
    Out off = run(tok, t, false), on = run(tok, t, true);
    check(off.calls.empty(), "call then </think> then the end, switch off: no call", calls_str(off.calls));
    check(on.calls.size() == 1 && on.reasoning == "I'll read it." && on.content.empty(),
          "call then </think> then the end, switch on: returned as a tool call", on.reasoning + calls_str(on.calls));
  }

  // Calls drafted while thinking: the reasoning goes on, so the result is exactly the old one.
  struct Draft {
    const char* name;
    std::string text;
    bool ended;
  };
  const std::vector<Draft> drafts = {
      {"draft, more reasoning, real call", "Plan: " + read_a + " is the call. Checking once more.\n</think>\n\n" + read_b,
       true},
      {"draft, </think>, a different real call", "Draft:\n" + read_a + "\n</think>\n\n" + read_b, true},
      {"draft, </think>, an answer", "Maybe " + read_a + "\n</think>\n\nI will read the file now.", true},
      {"unknown function at the end", "Hmm.\n" + call("delete_everything", param("path", "/")), true},
      {"unterminated call at the end", "Reading\n<tool_call>\n<function=read_file>\n<parameter=path>\na.t", true},
      {"cut by max_tokens after a call", "Reading.\n" + read_a, false},
      {"<tool_call> mentioned, reasoning goes on", "The format is <tool_call> then XML. Fine.\n</think>\n\nOK.", true},
      {"non-ASCII reasoning after a draft", "Черновик " + read_a + " Привет, мир — дальше.\n</think>\n\nОтвет: да.", true},
      {"stray </tool_call> after a call", "x " + read_a + "</tool_call>", true},
  };
  for (auto& c : drafts) {
    Out off = run(tok, c.text, false, true, c.ended), on = run(tok, c.text, true, true, c.ended);
    check(same_result(off, on), std::string("same result as before: ") + c.name,
          "off " + off.reasoning + "|" + off.content + calls_str(off.calls) + " on " + on.reasoning + "|" + on.content +
              calls_str(on.calls));
  }
  {
    // no tools in the request: nothing is held
    const std::string t = "x\n" + read_a;
    Out off = run(tok, t, false, true, true, ojson::array()), on = run(tok, t, true, true, true, ojson::array());
    check(same_stream(off, on) && on.calls.empty(), "request without tools: unchanged");
  }
  {
    // think_ws (Nemotron 3.5): the whitespace before </think> stays in the reasoning; without it, it is dropped
    const std::string t = "Let me check.\n</think>\n" + call("read_file", param("path", "/etc/hosts"));
    Out ws = run(tok, t, true, true, true, kTools, true), no = run(tok, t, true);
    check(ws.reasoning == "Let me check.\n" && no.reasoning == "Let me check." && ws.content.empty() &&
              calls_str(ws.calls) == calls_str(no.calls) && ws.calls.size() == 1,
          "think_ws: the newline before </think> stays in the reasoning", ws.reasoning + "|" + no.reasoning);
  }
  // Every case once more fed token by token is what run() does; also check the streamed deltas add up.
  {
    const std::string t = "Draft " + read_a + " more thinking\n</think>\n\nDone.";
    Out on = run(tok, t, true);
    std::string r, c;
    for (auto& d : on.deltas) r += d.reasoning, c += d.content;
    check(r == on.reasoning && c == on.content, "deltas add up to the result");
  }

  // Text is always well-formed UTF-8 (JSON can carry nothing else): a character the generation cut off, or bytes the
  // model wrote ill-formed, come out as U+FFFD, one per maximal subpart (as Python's decode(errors="replace")).
  {
    const std::string R = "\xEF\xBF\xBD";
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"h\xC3\xA9llo \xF0\x9F\x91\x8B \xE4\xB8\x96", "h\xC3\xA9llo \xF0\x9F\x91\x8B \xE4\xB8\x96"},  // well-formed
        {"Hi! \xF0\x9F\x91", "Hi! " + R},                     // cut after 3 of 4 bytes
        {"a\xC3(b", "a" + R + "(b"},                          // lead byte without its continuation
        {"\xED\xA0\x80", R + R + R},                          // a surrogate's encoding
        {"\xC0\xAF", R + R},                                  // overlong
        {"\xF4\x90\x80\x80", R + R + R + R},                  // past U+10FFFF
        {"x\xFF\x80y", "x" + R + R + "y"},                    // bytes that never start a character
    };
    bool ok = true;
    std::string bad;
    for (auto& [in, want] : cases)
      if (utf8_valid(in) != want) ok = false, bad += " [" + in + "] -> [" + utf8_valid(in) + "]";
    check(ok, "utf8_valid: U+FFFD per maximal subpart, well-formed text unchanged", bad);
    // a reply cut by max_tokens inside an emoji (Qwen3.5-9B's "Say hi." at 4 tokens): the parser's text ends in U+FFFD
    const std::vector<int> byte_id = byte_ids(tok);
    OutputParser p(tok, tok.find("<think>"), tok.find("</think>"), tok.find("<tool_call>"), tok.find("</tool_call>"),
                   false, false, ojson::array(), false, false);
    std::string content;
    std::vector<int> ids = tok.encode("Hi!");
    for (int b : {0xF0, 0x9F, 0x91}) ids.push_back(byte_id[b]);
    for (int id : ids) content += p.push(id).content;
    content += p.finish(false).content;
    check(content == "Hi!" + R, "a reply cut inside a character ends in U+FFFD", content);
  }

  // GLM-4.5's calls (Flint's template; CallFormat::Glm): the values typed from the tool schema as Qwen's parameters
  // are, in each layout the model may write; a call that is no such markup stays the text written. Each turn gives the
  // same result fed a token per byte and cut every 2, 3, 5 or 7 bytes (<arg_key> and the rest in pieces), and the same
  // stream with NI_THINK_TOOLCALLS on when no call is inside the reasoning.
  const std::vector<int> byte_id = byte_ids(tok);
  auto same_cut = [&](const std::string& text, const Out& want, bool think_tools, bool in_think, bool ended) {
    for (size_t cut : {1, 2, 3, 5, 7})
      if (!same_result(want, run_glm(tok, cut_ids(tok, text, cut, byte_id), think_tools, in_think, ended)))
        return false;
    return true;
  };
  const std::string glm_a = glm_call("read_file", glm_arg("path", "a.txt"));
  const std::string glm_b = glm_call("read_file", glm_arg("path", "b.txt"));
  struct GlmCase {
    std::string name, text;
    bool in_think;
    std::string reasoning, content, calls;
  };
  std::vector<GlmCase> glm = {
      {"string, number, boolean, object and array arguments",
       glm_call("search", glm_arg("query", "say \"hi\" {x}") + glm_arg("limit", "2.5") + glm_arg("regex", "true") +
                              glm_arg("filters", R"({"lang": "en", "n": [1, 2]})") +
                              glm_arg("paths", R"(["a b", "c"])")),
       false, "", "",
       R"([search{"query":"say \"hi\" {x}","limit":2.5,"regex":true,"filters":{"lang":"en","n":[1,2]},)"
       R"("paths":["a b","c"]}])"},
      {"typing: a string as written, a typed value that is no JSON as text, an untyped one as JSON, an unknown one as "
       "text",
       glm_call("read_file", glm_arg("path", "123") + glm_arg("start_line", "five")) +
           glm_call("search", glm_arg("note", "[1, 2]") + glm_arg("other", "7")),
       false, "", "", R"([read_file{"path":"123","start_line":"five"}, search{"note":[1,2],"other":"7"}])"},
      {"two calls", glm_a + glm_call("read_file", glm_arg("path", "b.txt") + glm_arg("start_line", "5")), false, "", "",
       R"([read_file{"path":"a.txt"}, read_file{"path":"b.txt","start_line":5}])"},
      {"a call without arguments", glm_call("get_time"), false, "", "", "[get_time{}]"},
      {"GLM-4.5's newlines between the elements and the calls",
       "<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>a.txt</arg_value>\n<arg_key>start_line</arg_key>\n"
       "<arg_value>5</arg_value>\n</tool_call>\n<tool_call>get_time\n</tool_call>\n",
       false, "", "", R"([read_file{"path":"a.txt","start_line":5}, get_time{}])"},
      {"spaces around the name and a key; the value as written",
       "<tool_call> read_file <arg_key> path </arg_key> <arg_value>  two spaces\n</arg_value> </tool_call>", false, "",
       "", R"([read_file{"path":"  two spaces\n"}])"},
      {"content, then the call", "I'll read the file.\n\n" + glm_a, false, "", "I'll read the file.",
       R"([read_file{"path":"a.txt"}])"},
      {"reasoning (the model's own <think>), content, then calls",
       "<think>The user wants a.txt and the time.</think>Reading it.\n" + glm_a + glm_call("get_time"), false,
       "The user wants a.txt and the time.", "Reading it.", R"([read_file{"path":"a.txt"}, get_time{}])"},
      {"reasoning in the prompt's think block, then a call", "Plan: read a.txt.\n</think>\n\n" + glm_a, true,
       "Plan: read a.txt.", "", R"([read_file{"path":"a.txt"}])"},
      {"non-ASCII values", glm_call("search", glm_arg("query", "Привет, мир 👋 — да") + glm_arg("limit", "3")),
       false, "", "", R"([search{"query":"Привет, мир 👋 — да","limit":3}])"},
      {"a string value with the call's tags in it", glm_call("search", glm_arg("query", "<arg_key>k</arg_key> x")),
       false, "", "", R"([search{"query":"<arg_key>k</arg_key> x"}])"},
      {"a malformed call, then a good one", glm_call("read file") + "\n" + glm_call("get_time"), false, "",
       "<tool_call>read file</tool_call>", "[get_time{}]"},
  };
  // malformed: no call, the markup as written
  for (auto& [name, text] : std::vector<std::pair<std::string, std::string>>{
           {"a name with a space", glm_call("read file", glm_arg("path", "a.txt"))},
           {"no name", glm_call("", glm_arg("path", "a.txt"))},
           {"an empty call", glm_call("")},
           {"a value left open", "<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</tool_call>"},
           {"a key without a value", "<tool_call>read_file<arg_key>path</arg_key></tool_call>"},
           {"text between a key and its value",
            glm_call("read_file", "<arg_key>path</arg_key> is <arg_value>a</arg_value>")},
           {"text after a value", glm_call("read_file", glm_arg("path", "a.txt") + " and more")},
           {"a tag in a key", glm_call("read_file", "<arg_key>pa<arg_value>th</arg_key><arg_value>a</arg_value>")},
           {"Qwen's markup", call("read_file", param("path", "a.txt"))},
           {"unterminated at the end of the turn", "<tool_call>read_file<arg_key>path</arg_key><arg_value>a.t"},
       })
    glm.push_back({"malformed, " + name, text, false, "", text, "[]"});
  std::string varies;
  for (auto& c : glm) {
    const std::vector<int> ids = tok.encode(c.text);
    Out off = run_glm(tok, ids, false, c.in_think), on = run_glm(tok, ids, true, c.in_think);
    check(off.reasoning == c.reasoning && off.content == c.content && calls_str(off.calls) == c.calls, "GLM: " + c.name,
          show(off));
    if (!same_stream(off, on) || !same_cut(c.text, off, false, c.in_think, true) ||
        !same_cut(c.text, on, true, c.in_think, true))
      varies += " [" + c.name + "]";
  }
  check(varies.empty(), "GLM: each turn the same a token per byte, cut every 2, 3, 5, 7 bytes, NI_THINK_TOOLCALLS on",
        varies);
  {
    // the default stays Qwen's markup: a GLM call is text to it (as Qwen's markup is to the GLM parser, above)
    Out q = run(tok, glm_a, false, false);
    check(q.calls.empty() && q.content == glm_a, "GLM: the default call format reads no GLM call", show(q));
  }
  {
    // NI_THINK_TOOLCALLS: a GLM call at the end of a think block never closed is held and returned as Qwen's is
    const std::string t = "<think>I need a.txt.\n" + glm_a;
    Out off = run_glm(tok, tok.encode(t), false, false), on = run_glm(tok, tok.encode(t), true, false);
    check(off.calls.empty() && off.reasoning == "I need a.txt.\n" + glm_a,
          "GLM: unclosed think, switch off: the call stays reasoning", show(off));
    check(calls_str(on.calls) == R"([read_file{"path":"a.txt"}])" && on.reasoning == "I need a.txt." &&
              on.content.empty(),
          "GLM: unclosed think, switch on: returned as a tool call", show(on));
    bool streamed = false;
    for (auto& d : on.deltas) streamed |= d.reasoning.find("<tool_call>") != std::string::npos ||
                                          d.reasoning.find("<arg_key>") != std::string::npos;
    check(!streamed && same_cut(t, on, true, false, true),
          "GLM: unclosed think, switch on: no markup streamed; the same in other tokens");
  }
  {
    // ... and a GLM call drafted while thinking, an unknown function or a malformed call there stay what they were
    const std::vector<std::pair<std::string, std::string>> drafts = {
        {"a draft, more reasoning, a real call", "<think>Maybe " + glm_a + " - no, b.</think>" + glm_b},
        {"an unknown function at the end", "<think>Hmm.\n" + glm_call("delete_everything", glm_arg("path", "/"))},
        {"a malformed call at the end", "<think>So:\n" + glm_call("read file", glm_arg("path", "a.txt"))},
    };
    for (auto& [name, t] : drafts) {
      Out off = run_glm(tok, tok.encode(t), false, false), on = run_glm(tok, tok.encode(t), true, false);
      check(same_result(off, on) && same_cut(t, on, true, false, true), "GLM: same result as before: " + name,
            show(off) + " / " + show(on));
    }
    Out d = run_glm(tok, tok.encode(drafts[0].second), true, false);
    check(d.reasoning == "Maybe " + glm_a + " - no, b." && calls_str(d.calls) == R"([read_file{"path":"b.txt"}])",
          "GLM: a draft stays reasoning, the call after </think> is the call", show(d));
  }

  printf(fails ? "FAILED: %d\n" : "ALL PASSED\n", fails);
  return fails ? 1 : 0;
}
