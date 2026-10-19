// The dense Nemotron-H models' chat side on the CPU (a model folder's tokenizer and chat template; no weights needed):
// the thinking switch (enable_thinking, or Nemotron Nano 9B v2's "/think" / "/no_think" flags), the generation prompt's
// tokens (a follow-up's cache resumes before them: tests/test_server.py allows 8), and the output parser on model-like
// replies as the engine sets it up: reasoning and content, tool calls (the Nano 4B's <tool_call> XML, the 9B v2's
// <TOOLCALL> JSON written as text), and calls rendered back through the template.
//   nh_chat_check <model_dir>
#include "engine/engine.h"
#include "engine/output_parser.h"
#include "util/log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace ni;

namespace {

int g_fail = 0;
void check(const std::string& what, bool ok, const std::string& detail = "") {
  printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), ok || detail.empty() ? "" : "  ", ok ? "" : detail.c_str());
  if (!ok) ++g_fail;
}

std::string show(const std::string& s) {
  std::string o;
  for (char c : s) o += c == '\n' ? std::string("\\n") : std::string(1, c);
  return "\"" + o + "\"";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: nh_chat_check <model_dir>\n");
    return 1;
  }
  log_level() = 0;
  const std::string dir = argv[1];
  Tokenizer tok;
  tok.load(dir + "/tokenizer.json");
  const std::string src = chat_template_source(dir);
  if (src.empty()) fatal("no chat template in %s", dir.c_str());
  minja::chat_template tmpl(src, "", "<|im_end|>");
  const bool flags = think_flag_template(src);
  const TextTags tt = text_tags_for(tok, src);
  printf("%s: thinking by %s; tags as text: think %s, tool calls %s\n", dir.c_str(),
         flags ? "\"/think\" / \"/no_think\" flags" : "enable_thinking", tt.think ? "yes" : "no",
         tt.tc_open.empty() ? "no" : tt.tc_open.c_str());
  // the engine's render (Engine::render's flag mapping; the template's own kwargs otherwise)
  auto render = [&](const ChatRequest& r, bool gen) {
    return render_chat(tmpl, flags && r.template_kwargs.contains("enable_thinking") ? think_flag_request(r) : r, gen);
  };
  const ojson tools = ojson::parse(R"([{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
      "required": ["path"]}}}, {"type": "function", "function": {"name": "run_shell", "description": "Run a command.",
      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}])");

  // 1. the thinking switch: the generation prompt opens the reasoning, or closes it at once
  for (int mode = 0; mode < 3; ++mode) {  // absent, true, false
    ChatRequest r;
    r.messages = ojson::parse(R"([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi."}])");
    if (mode) r.template_kwargs["enable_thinking"] = mode == 1;
    const std::string p = render(r, true);
    const bool open = p.size() >= 8 && p.compare(p.size() - 8, 8, "<think>\n") == 0;
    const bool closed = p.size() >= 15 && p.compare(p.size() - 15, 15, "<think></think>") == 0;
    const char* m = mode == 0 ? "absent" : mode == 1 ? "true" : "false";
    check(std::string("enable_thinking ") + m + ": the prompt ends with " + (mode == 2 ? "<think></think>" : "<think>\\n"),
          mode == 2 ? closed : open, show(p.substr(p.size() > 60 ? p.size() - 60 : 0)));
    // (the flags themselves, not the "/think" of a "</think>")
    bool flag_text = p.find("/no_think") != std::string::npos;
    for (size_t at = p.find("/think"); at != std::string::npos; at = p.find("/think", at + 1))
      if (at == 0 || p[at - 1] != '<') flag_text = true;
    check(std::string("enable_thinking ") + m + ": no flag text in the prompt", !flag_text, show(p));
    // the generation prompt's tokens after the conversation's (the boundary a follow-up resumes at)
    const std::string base = render(r, false);
    const std::vector<int> a = tok.encode(base), b = tok.encode(p);
    const bool prefix = a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
    check(std::string("enable_thinking ") + m + ": the generation prompt is " + std::to_string(b.size() - a.size()) +
              " tokens after the conversation's (<= 8)",
          prefix && b.size() - a.size() <= 8, std::to_string(a.size()) + " / " + std::to_string(b.size()));
  }
  // the flag goes to the last user message, a content array too, and the last flag wins over the user's own
  if (flags) {
    ChatRequest r;
    r.messages = ojson::parse(R"([{"role": "system", "content": "Be brief. /think"},
                                  {"role": "user", "content": [{"type": "text", "text": "Hi."}]}])");
    r.template_kwargs["enable_thinking"] = false;
    const std::string p = render(r, true);
    check("a content array and the system's /think: enable_thinking false still closes the reasoning",
          p.size() >= 15 && p.compare(p.size() - 15, 15, "<think></think>") == 0, show(p));
  }

  // 2. the parser on replies, token by token, as the engine sets it up
  const int t_open = tok.find("<think>"), t_close = tok.find("</think>"), c_open = tok.find("<tool_call>"),
            c_close = tok.find("</tool_call>");
  auto parse = [&](const std::string& text, bool in_think, bool with_tools, ChatResult& out) {
    OutputParser p(tok, t_open, t_close, c_open, c_close, in_think, with_tools, tools, false, false, tt);
    for (int id : tok.encode(text)) {
      ChatDelta d = p.push(id);
      out.reasoning += d.reasoning;
      out.content += d.content;
      for (auto& c : d.tool_calls) out.tool_calls.push_back(c);
    }
    ChatDelta d = p.finish(true);
    out.reasoning += d.reasoning;
    out.content += d.content;
    for (auto& c : d.tool_calls) out.tool_calls.push_back(c);
  };
  {
    ChatResult r;
    parse("I multiply 17 by 23: 340 + 51.\n</think>\n\n391", true, false, r);
    check("thinking reply: reasoning apart from the answer",
          r.reasoning == "I multiply 17 by 23: 340 + 51." && r.content == "391", show(r.reasoning) + " / " + show(r.content));
  }
  const bool json = tt.json_calls;
  auto call_text = [&](const std::string& name, const std::string& args_json) {
    if (json) return std::string("<TOOLCALL>[{\"name\": \"") + name + "\", \"arguments\": " + args_json + "}]</TOOLCALL>";
    const ojson a = ojson::parse(args_json);
    std::string s = "<tool_call>\n<function=" + name + ">\n";
    for (auto& [k, v] : a.items()) s += "<parameter=" + k + ">\n" + (v.is_string() ? v.get<std::string>() : v.dump()) + "\n</parameter>\n";
    return s + "</function>\n</tool_call>";
  };
  {
    ChatResult r;
    parse(call_text("read_file", R"({"path": "/etc/hosts", "max_lines": 5})"), false, true, r);
    const bool ok = r.tool_calls.size() == 1 && r.tool_calls[0].name == "read_file" &&
                    ojson::parse(r.tool_calls[0].arguments) == ojson::parse(R"({"path": "/etc/hosts", "max_lines": 5})");
    check("tool call: one call, its arguments as JSON (an integer kept)", ok && r.content.empty(),
          (r.tool_calls.empty() ? std::string("no call") : r.tool_calls[0].name + " " + r.tool_calls[0].arguments) +
              " content " + show(r.content));
  }
  {
    ChatResult r;
    parse("Reading it.\n" + call_text("run_shell", R"({"command": "wc -l < /etc/hosts"})"), false, true, r);
    check("text, then a call: the text is the content", r.content == "Reading it." && r.tool_calls.size() == 1 &&
          r.tool_calls[0].name == "run_shell", show(r.content));
  }
  {
    ChatResult r;
    parse("Let me look at the file.\n</think>\n" + call_text("read_file", R"({"path": "/etc/hosts"})"), true, true, r);
    check("reasoning, then a call", r.reasoning == "Let me look at the file." && r.tool_calls.size() == 1,
          show(r.reasoning) + " / " + std::to_string(r.tool_calls.size()));
  }
  if (json) {
    ChatResult r;
    parse(R"(<TOOLCALL>[{"name": "read_file", "arguments": "{\"path\": \"/etc/hosts\"}"}, {"name": "run_shell", "arguments": {"command": "ls"}}]</TOOLCALL>)",
          false, true, r);
    check("two calls, one with its arguments as a JSON string", r.tool_calls.size() == 2 &&
          ojson::parse(r.tool_calls[0].arguments) == ojson::parse(R"({"path": "/etc/hosts"})") &&
          r.tool_calls[1].name == "run_shell", std::to_string(r.tool_calls.size()));
    ChatResult b;
    parse("<TOOLCALL>[{not json}]</TOOLCALL>", false, true, b);
    check("a call that is not JSON stays text", b.tool_calls.empty() && b.content == "<TOOLCALL>[{not json}]</TOOLCALL>",
          show(b.content));
    ChatResult c;
    parse("The tag starts with <TOOL", false, true, c);
    check("a reply cut inside a tag: the text as it is", c.tool_calls.empty() && c.content == "The tag starts with <TOOL",
          show(c.content));
    ChatResult d;
    parse("No tools here: <TOOLCALL>[]</TOOLCALL>", false, false, d);
    check("without tools in the request the tags are text", d.content == "No tools here: <TOOLCALL>[]</TOOLCALL>",
          show(d.content));
  }

  // 3. a tool loop rendered back: the call, its result, and the next prompt
  {
    ChatRequest r;
    r.tools = tools;
    r.template_kwargs["enable_thinking"] = false;
    r.messages = ojson::parse(R"([{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Read /etc/hosts."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
          "function": {"name": "read_file", "arguments": "{\"path\": \"/etc/hosts\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "127.0.0.1 localhost"}])");
    std::string p;
    try {
      p = render(r, true);
    } catch (const std::exception& e) {
      p = std::string("error: ") + e.what();
    }
    check("a tool loop renders: the call and its result in the next prompt",
          p.find("read_file") != std::string::npos && p.find("/etc/hosts") != std::string::npos &&
              p.find("127.0.0.1 localhost") != std::string::npos,
          show(p.substr(0, 400)));
  }
  printf("%s\n", g_fail ? ("FAILED: " + std::to_string(g_fail)).c_str() : "ALL PASSED");
  return g_fail ? 1 : 0;
}
