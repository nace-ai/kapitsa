// Structured output's request handling, CPU only (tests/test_structured.py --cli drives it): each line of stdin is
// {"api": "chat" | "responses" | "messages", "body": <a request body>}, and each line of stdout what the server makes of
// the body's structured output: {"ok": true, "grammar": <src/engine/grammar.h's summary, or null>} or the error the
// server answers with, {"status": 400, "type": ..., "param": ..., "message": ...}.
//   structured_check < requests.jsonl
#include "api/anthropic.h"
#include "api/chat_completions.h"
#include "api/responses.h"

#include <iostream>
#include <string>

using namespace ni;
using namespace ni::api;

int main() {
  ResponseStore store;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    ojson out;
    try {
      const ojson in = ojson::parse(line);
      const std::string api = in.value("api", "chat");
      const ojson& body = in["body"];
      std::shared_ptr<const Grammar> g;
      if (api == "chat") {
        // as the server's parse_request: tool_choice "none" takes the tools out of the prompt
        ojson tools = body.contains("tools") && body["tools"].is_array() ? body["tools"] : ojson::array();
        if (body.contains("tool_choice") && body["tool_choice"].is_string() && body["tool_choice"] == "none")
          tools = ojson::array();
        g = chat_grammar(body, tools);
      } else if (api == "responses") {
        g = responses_request(body, store).chat.grammar;
      } else if (api == "messages") {
        g = messages_request(body).chat.grammar;
      } else {
        throw std::runtime_error("unknown api " + api);
      }
      out = {{"ok", true}, {"grammar", g ? ojson(grammar_summary(*g)) : ojson()}};
    } catch (const ApiError& e) {
      out = {{"status", e.status}, {"type", e.type}, {"param", e.param}, {"message", e.what()}};
    } catch (const std::exception& e) {
      out = {{"status", 500}, {"message", std::string("structured_check: ") + e.what()}};
    }
    std::cout << out.dump() << "\n" << std::flush;
  }
  return 0;
}
