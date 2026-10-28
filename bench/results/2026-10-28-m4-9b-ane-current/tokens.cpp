#include "tokenizer/tokenizer.h"
#include "bench/bench.h"
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  ni::Tokenizer tokenizer;
  tokenizer.load(std::string(argv[2]) + "/tokenizer.json");
  std::ifstream f(argv[3]);
  if (!f) return 3;
  if (std::string(argv[1]) == "decode") {
    const auto ids = nlohmann::json::parse(f).get<std::vector<int>>();
    std::cout << tokenizer.decode(ids);
  } else {
    const std::string text((std::istreambuf_iterator<char>(f)), {});
    const auto ids = tokenizer.encode(text);
    std::cout << nlohmann::json(ids).dump() << '\n';
  }
}
