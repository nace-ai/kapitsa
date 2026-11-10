// CPU tokenizer measurement driver. Build with the command in README.md.
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  ni::Tokenizer tokenizer;
  tokenizer.load(argv[1]);
  auto cases = nlohmann::json::parse(std::ifstream(argv[2]));
  for (const auto& c : cases) {
    const auto text = c.at("text").get<std::string>();
    auto start = std::chrono::steady_clock::now();
    const auto ids = tokenizer.encode(text);
    const auto first = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const int repetitions = c.at("iterations");
    if (repetitions < 1) return 2;
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i)
      if (tokenizer.encode(text) != ids) return 1;
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
                    repetitions;
    std::cout << nlohmann::json{{"name", c.at("name")}, {"bytes", text.size()}, {"ids", ids},
                              {"first_ms", first}, {"warm_ms", ms}, {"iterations", repetitions}}.dump() << '\n';
  }
}
