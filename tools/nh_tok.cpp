// A model folder's tokenizer and chat template (tokenizer.json, chat_template.jinja or tokenizer_config.json's
// chat_template), as the engine runs them: nh_tok <dir> [--file F | TEXT] prints the token ids (comma separated);
// nh_tok <dir> --decode 1,2,3 prints the text; nh_tok <dir> --render REQ.json [--ids] prints the prompt the chat
// template renders for {"messages": [...], "tools": [...], "chat_template_kwargs": {...}} (with --ids its token
// ids). For checks against transformers.
#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: nh_tok <dir> [--file F | --decode IDS | --render REQ.json [--ids] | TEXT]\n");
    return 1;
  }
  log_level() = 0;
  const std::string dir = argv[1];
  Tokenizer tok;
  tok.load(dir + "/tokenizer.json");
  auto template_source = [&]() -> std::string {
    std::ifstream jf(dir + "/chat_template.jinja");
    if (jf) return std::string(std::istreambuf_iterator<char>(jf), {});
    std::ifstream cf(dir + "/tokenizer_config.json");
    const auto c = nlohmann::json::parse(cf);
    return c.value("chat_template", "");
  };
  const std::string a = argv[2];
  if (a == "--decode" && argc > 3) {
    std::vector<int> ids;
    std::stringstream ss(argv[3]);
    std::string t;
    while (std::getline(ss, t, ',')) ids.push_back(std::stoi(t));
    fwrite(tok.decode(ids).data(), 1, tok.decode(ids).size(), stdout);
    return 0;
  }
  if (a == "--render" && argc > 3) {
    std::ifstream in(argv[3]);
    auto req = nlohmann::ordered_json::parse(in);
    minja::chat_template tmpl(template_source(), "", "<|im_end|>");
    minja::chat_template_inputs ci;
    ci.messages = req["messages"];
    ci.tools = req.contains("tools") ? req["tools"] : nlohmann::ordered_json();
    ci.add_generation_prompt = true;
    ci.extra_context = req.contains("chat_template_kwargs") ? req["chat_template_kwargs"] : nlohmann::ordered_json::object();
    minja::chat_template_options opt;
    const std::string out = tmpl.apply(ci, opt);
    if (argc > 4 && std::string(argv[4]) == "--ids") {
      const std::vector<int> ids = tok.encode(out);
      for (size_t i = 0; i < ids.size(); ++i) printf("%s%d", i ? "," : "", ids[i]);
      printf("\n");
    } else {
      fwrite(out.data(), 1, out.size(), stdout);
    }
    return 0;
  }
  std::string text = a;
  if (a == "--file" && argc > 3) {
    std::ifstream in(argv[3], std::ios::binary);
    text.assign(std::istreambuf_iterator<char>(in), {});
  }
  const std::vector<int> ids = tok.encode(text);
  for (size_t i = 0; i < ids.size(); ++i) printf("%s%d", i ? "," : "", ids[i]);
  printf("\n");
  return 0;
}
