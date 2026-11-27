#include "tokenizer/tokenizer.h"
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  ni::Tokenizer t;
  t.load(argv[1]);
  std::vector<int> ids;
  for (int id; std::cin >> id;) ids.push_back(id);
  std::cout << t.decode(ids);
}
