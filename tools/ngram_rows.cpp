// Every row of an MLX affine n-gram table (src/model/ngram.h, NgramTable::Fmt::Affine with bits and group size read
// from the shapes, as EngramCpu opens Hy4-flash's Engram tables) dumped as float32 [rows, dim], for
// tools/ngram_affine_test.py to compare with mx.dequantize. No GPU, no lock.
//   ngram_rows <file.safetensors> <codes> <scales> <biases> <dim> <out.f32>
#include "model/ngram.h"
#include "util/log.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: ngram_rows <file.safetensors> <codes> <scales> <biases> <dim> <out.f32>\n");
    return 2;
  }
  const int dim = std::atoi(argv[5]);
  NgramTable t;
  t.open({NgramTable::Part{argv[1], argv[2], argv[3], argv[4]}}, NgramTable::Fmt::Affine, dim);
  FILE* f = std::fopen(argv[6], "wb");
  if (!f) fatal("cannot write %s", argv[6]);
  std::vector<float> row(static_cast<size_t>(dim));
  for (int64_t r = 0; r < t.rows(); ++r) {
    t.row(r, row.data());
    if (std::fwrite(row.data(), 4, row.size(), f) != row.size()) fatal("write failed");
  }
  std::fclose(f);
  std::printf("%lld rows of %d\n", (long long)t.rows(), dim);
  return 0;
}
