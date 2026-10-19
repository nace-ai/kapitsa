// OutputParser checks on real token ids (CPU only): reasoning / content / tool calls for the cases a tool call
// can take inside or after Qwen's think block, with NI_THINK_TOOLCALLS off and on.
//   parser_test <model_dir>   (reads <model_dir>/tokenizer.json; exit status 1 if any check fails)
#include "engine/output_parser.h"
#include "tokenizer/tokenizer.h"

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

// One generated turn: `text` as the model would write it after the generation prompt (in the think block
// when `in_think`), ending by itself when `ended`.
Out run(const Tokenizer& tok, const std::string& text, bool think_tools, bool in_think = true, bool ended = true,
        const ojson& tools = kTools, bool think_ws = false) {
  OutputParser p(tok, tok.find("<think>"), tok.find("</think>"), tok.find("<tool_call>"), tok.find("</tool_call>"),
                 in_think, !tools.empty(), tools, think_tools, think_ws);
  Out o;
  auto add = [&](const ChatDelta& d) {
    o.reasoning += d.reasoning;
    o.content += d.content;
    for (auto& c : d.tool_calls) o.calls.push_back(c);
    o.deltas.push_back(d);
  };
  for (int id : tok.encode(text)) add(p.push(id));
  add(p.finish(ended));
  return o;
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
    std::vector<int> byte_id(256, -1);
    for (int id = 0; id < tok.vocab_size(); ++id)
      if (tok.token_bytes(id).size() == 1 && byte_id[uint8_t(tok.token_bytes(id)[0])] < 0)
        byte_id[uint8_t(tok.token_bytes(id)[0])] = id;
    OutputParser p(tok, tok.find("<think>"), tok.find("</think>"), tok.find("<tool_call>"), tok.find("</tool_call>"),
                   false, false, ojson::array(), false, false);
    std::string content;
    std::vector<int> ids = tok.encode("Hi!");
    for (int b : {0xF0, 0x9F, 0x91}) ids.push_back(byte_id[b]);
    for (int id : ids) content += p.push(id).content;
    content += p.finish(false).content;
    check(content == "Hi!" + R, "a reply cut inside a character ends in U+FFFD", content);
  }

  printf(fails ? "FAILED: %d\n" : "ALL PASSED\n", fails);
  return fails ? 1 : 0;
}
