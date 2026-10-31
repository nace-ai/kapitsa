#include "systemone/systemone.h"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: systemone_codes_driver TOKENIZER_JSON\n";
    return 2;
  }
  ni::Tokenizer tok;
  try {
    tok.load(argv[1]);
  } catch (const std::exception& error) {
    std::cerr << "tokenizer load failed: " << error.what() << '\n';
    return 2;
  }

  using namespace ni::systemone;
  std::string line;
  while (std::getline(std::cin, line)) {
    ojson result;
    try {
      const auto envelope = ojson::parse(line);
      Limits limits;
      if (envelope.contains("limits")) {
        const auto& values = envelope.at("limits");
        limits.max_state = values.value("max_state", limits.max_state);
        limits.max_branch = values.value("max_branch", limits.max_branch);
        limits.max_packed = values.value("max_packed", limits.max_packed);
      }
      const auto enc = encode_max(tok, envelope.at("body"), limits);
      if (envelope.contains("context")) check_fits(enc, envelope.at("context").get<int>());
      result = {{"state", enc.state}, {"branches", ojson::array()}, {"meta", ojson::array()},
                {"total_tokens", enc.total_tokens()}, {"max_branch", enc.max_branch()}};
      for (const auto& branch : enc.branches) {
        result["branches"].push_back({{"tokens", branch.tokens}, {"decide", branch.group.decide},
                                       {"options", branch.group.options}});
      }
      for (const auto& meta : enc.meta) {
        result["meta"].push_back({{"id", meta.id}, {"type", meta.type}, {"keys", meta.keys},
                                   {"legend", meta.legend}});
      }
    } catch (const BadRequest& error) {
      result = {{"error", "BadRequest"}, {"message", error.what()}};
    } catch (const std::exception& error) {
      result = {{"error", "UnexpectedError"}, {"message", error.what()}};
    } catch (...) {
      result = {{"error", "UnexpectedError"}, {"message", "non-standard exception"}};
    }
    std::cout << result.dump() << '\n';
  }
  return std::cin.bad() || !std::cout ? 2 : 0;
}
