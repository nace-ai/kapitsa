// CPU check of the drafter's load-time quantizer (no GPU work): quantizes one bf16 tensor of a
// safetensors dir and writes the raw packed words / scales / biases for comparison with mx.quantize.
//   dflash_qtest <dir> <tensor> <bits> <even 0|1> <out_prefix>   -> <out>.w (u32), <out>.s, <out>.b (bf16)
#include "loader/safetensors.h"
#include "model/dflash2.h"
#include "util/log.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: dflash_qtest <dir> <tensor> <bits> <even 0|1> <out_prefix>\n");
    return 1;
  }
  const std::string name = argv[2];
  const int bits = atoi(argv[3]);
  const bool even = atoi(argv[4]) != 0;
  WeightStore ws;
  ws.open_dir(argv[1], [&](const std::string& n) { return n == name; });
  const Tensor& t = ws.get(name);
  const uint32_t N = uint32_t(t.shape[0]), K = uint32_t(t.shape[1]);
  std::vector<const uint16_t*> rows(N);
  for (uint32_t i = 0; i < N; ++i) rows[i] = t.buf.as<uint16_t>() + size_t(i) * K;
  std::vector<uint32_t> W(size_t(N) * K * bits / 32);
  std::vector<uint16_t> S(size_t(N) * K / 64), B(size_t(N) * K / 64);
  double t0 = now_ms();
  dfl_quantize_cpu(rows, K, bits, even, W.data(), S.data(), B.data());
  fprintf(stderr, "quantized [%u, %u] to %d bits in %.0f ms\n", N, K, bits, now_ms() - t0);
  const std::string o = argv[5];
  std::ofstream(o + ".w", std::ios::binary).write((const char*)W.data(), W.size() * 4);
  std::ofstream(o + ".s", std::ios::binary).write((const char*)S.data(), S.size() * 2);
  std::ofstream(o + ".b", std::ios::binary).write((const char*)B.data(), B.size() * 2);
  return 0;
}
