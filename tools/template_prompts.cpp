// Every prompt a model's chat template makes for one fixed set of requests: one row per template and request, so two
// builds of the template path (chat_template_source, its postfix rewrite, and render_chat's polyfills) can be shown to
// move no prompt on the templates already on this Mac. The dumps are compared with `cmp`, and every row the templates
// make is one row of the dump: a request a template raises on prints the raise where the prompt would be, so a raise
// that moves is compared too.
//
//   template_prompts <model_dir>... > dump.txt
//
// It reads the template file only (<dir>/chat_template.jinja, else tokenizer_config.json's chat_template: no weights,
// no tokenizer, no GPU), so a model folder answers without its weights, and the same folder answers the same rows in
// two builds. The rows are sorted by path, as the shell expands them.
//
//   template_prompts --assert-unchanged <model_dir>...
//
// prints one line per folder instead: whether chat_template_source returns the template exactly as the file writes it.
// That is the rewrite's other half, the one a dump cannot see: a template the rewrite leaves alone is a template whose
// rows cannot move, whatever the request.
#include "engine/engine.h"

#include <minja/chat-template.hpp>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace ni;

namespace {

// One request through every template: the shapes tools/api_test.cpp renders (its template checks, and the tool-argument
// and float checks): a tool loop, an image part (a note, and the placeholder with an image encoder), a system message
// after the first, reasoning a client sends back, template_kwargs, and text a template has to escape.
struct Fixture {
  const char* name;
  ChatRequest req;
  bool add_generation_prompt;
  bool images;
  bool native;
};

const char* kTools =
    R"([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}])";
const char* kImageParts =
    R"([{"type": "text", "text": "Look:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        {"type": "text", "text": "what?"}])";

std::vector<Fixture> fixtures() {
  std::vector<Fixture> out;
  // a tool loop: the call the assistant sent back, then the tool's result
  ChatRequest tool_loop;
  tool_loop.messages = ojson::parse(R"([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
      "function": {"name": "f", "arguments": {"x": 1, "s": "a b"}}}]},
    {"role": "tool", "tool_call_id": "1", "content": "R"}])");
  tool_loop.tools = ojson::parse(kTools);
  // a later system message is a system turn in place, and an image is a note (or the encoder's placeholder)
  ChatRequest mixed;
  mixed.messages = ojson::parse(std::string(R"([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
      "function": {"name": "f", "arguments": {}}}]},
    {"role": "tool", "tool_call_id": "1", "content": "R"},
    {"role": "system", "content": "Later."},
    {"role": "user", "content": )") + kImageParts + "}]");
  mixed.tools = ojson::parse(kTools);
  // history with reasoning a client sends back, and template_kwargs a client asks for
  ChatRequest history;
  history.messages = ojson::parse(R"([{"role": "user", "content": "A1"},
    {"role": "assistant", "content": "B1", "reasoning_content": "why B1"},
    {"role": "user", "content": "A2"}, {"role": "assistant", "content": "B2"}, {"role": "user", "content": "A3"}])");
  ChatRequest kwargs = tool_loop;
  kwargs.template_kwargs = ojson::parse(R"({"enable_thinking": false, "reasoning_effort": "low"})");
  ChatRequest dirty;
  dirty.messages = ojson::parse(
      R"([{"role": "system", "content": "S & <T> \"q\""}, {"role": "user", "content": "a < b && c > d, 100% {braces}, {{ braces }}, {% macro %}, back\\slash, new\nline, tab\there, \u00fcn\u00efc\u00f6d\u00e9, \u65e5\u672c\u8a9e, \ud83d\ude00 \ud83c\udf89, and the ones jinja reads: (f(x)).y"}])");
  ChatRequest plain;
  plain.messages = ojson::parse(R"([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}])");

  out.push_back({"plain", plain, true, false, false});
  out.push_back({"plain (no generation prompt)", plain, false, false, false});
  out.push_back({"tool loop", tool_loop, true, false, false});
  out.push_back({"later system and image", mixed, true, false, false});
  out.push_back({"later system and image, encoder", mixed, true, true, false});
  out.push_back({"history with reasoning", history, true, false, false});
  out.push_back({"template_kwargs", kwargs, true, false, false});
  out.push_back({"dirty text", dirty, true, false, false});
  out.push_back({"tool loop, native", tool_loop, true, false, true});
  return out;
}

// The template as the file writes it: what chat_template_source returns when it changes nothing (main's rule), read here
// with the same two places chat_template_source reads.
std::string template_file_text(const std::string& dir) {
  const std::string path = dir + "/chat_template.jinja";
  std::string text;
  if (FILE* f = fopen(path.c_str(), "rb")) {
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    fclose(f);
  }
  if (!text.empty()) return text;
  std::string cfg;
  if (FILE* f = fopen((dir + "/tokenizer_config.json").c_str(), "rb")) {
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) cfg.append(buf, n);
    fclose(f);
  }
  const auto parsed = nlohmann::json::parse(cfg, nullptr, false);
  if (!parsed.is_object() || !parsed.contains("chat_template") || !parsed["chat_template"].is_string()) return {};
  return parsed["chat_template"].get<std::string>();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: template_prompts <model_dir>...\n");
    return 2;
  }
  bool assert_unchanged = false;
  std::vector<std::string> dirs;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--assert-unchanged") {
      assert_unchanged = true;
    } else {
      dirs.push_back(a);
    }
  }
  if (dirs.empty()) {
    fprintf(stderr, "usage: template_prompts [--assert-unchanged] <model_dir>...\n");
    return 2;
  }
  std::sort(dirs.begin(), dirs.end());
  dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
  std::vector<Fixture> fx;
  try {
    fx = fixtures();
  } catch (const std::exception& e) {
    fprintf(stderr, "template_prompts: the fixtures do not parse: %s\n", e.what());
    return 2;
  }
  // Every template stays alive until the dump is written: render_chat asks string_content_template whether a template
  // reads a content list, and that answer is cached by the template's address (src/engine/engine.cpp). A template freed
  // after its rows and a later one built on the same address would be answered for the first one, and the dump would
  // depend on where the allocator put each template rather than on the template.
  std::vector<std::string> dir_of;
  std::vector<std::unique_ptr<minja::chat_template>> tmpl_of;
  std::vector<std::string> note_of;  // why a dir has no rows: "no chat template", or the parse raise
  for (const std::string& dir : dirs) {
    const std::string src = chat_template_source(dir);
    dir_of.push_back(dir);
    tmpl_of.emplace_back();
    if (src.empty()) {
      note_of.push_back("no chat template");
      continue;
    }
    try {
      tmpl_of.back() = std::make_unique<minja::chat_template>(src, "", "<|im_end|>");
      note_of.emplace_back();
    } catch (const std::exception& e) {
      note_of.push_back(std::string("the template does not parse: ") + e.what());
    }
  }
  if (assert_unchanged) {
    int changed = 0;
    for (const std::string& dir : dir_of) {
      const std::string src = chat_template_source(dir);
      const bool same = !src.empty() && src == template_file_text(dir);
      if (!same) ++changed;
      printf("%s %s\n", same ? "same" : "REWRITTEN", dir.c_str());
    }
    printf("### %zu template(s), %d rewritten\n", dir_of.size(), changed);
    return changed == 0 ? 0 : 1;
  }
  int rows = 0;
  for (size_t i = 0; i < dir_of.size(); ++i) {
    if (!tmpl_of[i]) {
      printf("### %s: %s\n", dir_of[i].c_str(), note_of[i].c_str());
      continue;
    }
    const std::string& dir = dir_of[i];
    const minja::chat_template& tmpl = *tmpl_of[i];
    for (const Fixture& f : fx) {
      std::string p;
      try {
        p = render_chat(tmpl, f.req, f.add_generation_prompt, f.images, f.native);
      } catch (const std::exception& e) {
        p = std::string("raised: ") + e.what();
      }
      printf("### %s | %s | %zu bytes\n", dir.c_str(), f.name, p.size());
      fwrite(p.data(), 1, p.size(), stdout);
      putchar('\n');
      ++rows;
    }
  }
  printf("### %d rows\n", rows);
  return 0;
}
