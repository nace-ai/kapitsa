// Multi-turn engine check with prefix reuse and long prompts (DFlash2 ring rewind across checkpoint
// restores and the 2047-row window): three greedy turns through Engine::chat, printed with their cached
// prefix, speed and output text, so a run with --dflash and one without can be compared token for token.
//   engine_turns <model_dir> <text file> [--dflash <dir|auto>] [--spec D] [--lines N] [--max-tokens N]
// Turn 1: the first N lines of the text file (a long prompt) + a question. Turn 2: the conversation so
// far + a follow-up (reuses the turn-1 checkpoint: the re-rendered answer differs from the live tokens
// when the model reasoned). Turn 3: a different follow-up after turn 1 (restores the same checkpoint).
#include "engine/engine.h"
#include "util/log.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: engine_turns <model_dir> <text file> [--dflash dir|auto] [--lines N] [--max-tokens N]\n");
    return 1;
  }
  EngineOptions o;
  o.model_dir = argv[1];
  o.context = 16384;
  int lines = 320, max_tokens = 160;
  for (int i = 3; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--dflash")) o.dflash = argv[i + 1];
    if (!strcmp(argv[i], "--lines")) lines = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--max-tokens")) max_tokens = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--spec")) o.spec_depth = atoi(argv[i + 1]);
  }
  std::string text;
  {
    std::ifstream f(argv[2]);
    std::string l;
    for (int i = 0; i < lines && std::getline(f, l); ++i) text += l + "\n";
  }
  Engine eng(o);
  auto turn = [&](const ojson& msgs, const char* name) {
    ChatRequest r;
    r.messages = msgs;
    r.temperature = 0.0f;
    r.max_tokens = max_tokens;
    r.template_kwargs = ojson{{"enable_thinking", false}};
    ChatResult res = eng.chat(r, [](const ChatDelta&) { return true; });
    printf("== %s: prompt %d, cached %d, %d tokens, %.1f tok/s, finish %s\n%s\n", name, res.prompt_tokens,
           res.cached_tokens, res.completion_tokens,
           res.completion_tokens > 1 ? (res.completion_tokens - 1) / (res.decode_ms / 1000) : 0.0,
           res.finish_reason.c_str(), res.content.c_str());
    fflush(stdout);
    return res.content;
  };
  ojson msgs = ojson::array();
  msgs.push_back({{"role", "system"}, {"content", "You are a careful senior C++ reviewer."}});
  msgs.push_back({{"role", "user"},
                  {"content", "Here is a source file:\n\n```cpp\n" + text +
                                  "```\n\nExplain in detail what the first three functions do."}});
  const std::string a1 = turn(msgs, "turn 1");
  ojson m2 = msgs;
  m2.push_back({{"role", "assistant"}, {"content", a1}});
  m2.push_back({{"role", "user"}, {"content", "Now list three concrete bugs or risks you see in this file."}});
  turn(m2, "turn 2");
  ojson m3 = msgs;
  m3.push_back({{"role", "assistant"}, {"content", a1}});
  m3.push_back({{"role", "user"}, {"content", "Rewrite the second function in a more compact style."}});
  turn(m3, "turn 3");
  return 0;
}
