#include "tokenizer/tokenizer.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
int main(int argc, char** argv) {
  if (argc != 4) return 2;
  ni::Tokenizer tok;
  tok.load(std::string(argv[1]) + "/tokenizer.json");
  std::ifstream in(argv[3]);
  if (!in) return 3;
  if (std::string(argv[2]) == "encode") {
    std::string text((std::istreambuf_iterator<char>(in)), {});
    auto ids = tok.encode(text);
    for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
    std::cout << '\n';
  } else {
    std::vector<int> ids;
    int n;
    while (in >> n) ids.push_back(n);
    std::cout << tok.decode(ids);
  }
}
