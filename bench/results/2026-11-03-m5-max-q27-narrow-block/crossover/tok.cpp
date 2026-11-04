// tok: a text file (its first N bytes) as the chat template's user turn, tokenized with a model's tokenizer.json, as
// comma-separated ids for ni_run. usage: tok TOKENIZER_JSON TEXT_FILE N_BYTES QUESTION > ids.txt
#include "tokenizer/tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
int main(int argc, char** argv) {
  ni::Tokenizer t;
  t.load(argv[1]);
  std::ifstream f(argv[2]);
  std::stringstream ss;
  ss << f.rdbuf();
  std::string text = ss.str().substr(0, size_t(atol(argv[3])));
  const std::string p = "<|im_start|>user\n" + text + argv[4] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  const auto ids = t.encode(p);
  for (size_t i = 0; i < ids.size(); ++i) printf(i ? ",%d" : "%d", ids[i]);
  fprintf(stderr, "%zu ids\n", ids.size());
}
