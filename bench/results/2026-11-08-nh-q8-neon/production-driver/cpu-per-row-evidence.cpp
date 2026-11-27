// Private per-row CPU quantizer evidence. No engine/model load or GPU; no loading/inference-speed claim.
#include <CommonCrypto/CommonDigest.h>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

void reference_q8(const float*, int64_t, uint8_t*, uint8_t*);
void candidate_q8(const float*, int64_t, uint8_t*, uint8_t*);
using Quantize = void (*)(const float*, int64_t, uint8_t*, uint8_t*);

std::string sha256(const void* data, size_t bytes) {
  CC_SHA256_CTX context;
  CC_SHA256_Init(&context);
  CC_SHA256_Update(&context, data, CC_LONG(bytes));
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &context);
  char hex[CC_SHA256_DIGEST_LENGTH * 2 + 1];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(hex + i * 2, 3, "%02x", digest[i]);
  return hex;
}

// Reset the quantizer's K and input/code/scale bases for every row, as the loader does.
template<Quantize function>
__attribute__((noinline)) void quantize_rows(const float* x, size_t rows, size_t K, uint8_t* codes, uint8_t* scales) {
  for (size_t row = 0; row < rows; ++row) {
    function(x + row * K, int64_t(K), codes + row * K, scales + row * (K / 16));
  }
}

int main(int argc, char** argv) {
  if (argc != 1 && argc != 3) { puts("usage: cpu-per-row-evidence [K ROWS]"); return 2; }
  const size_t K = argc == 3 ? strtoull(argv[1], nullptr, 10) : 3136;
  const size_t rows = argc == 3 ? strtoull(argv[2], nullptr, 10) : 1024;
  if (!K || K % 32 || K > 1048576 || !rows || rows > 65536 || K * rows > (1u << 26)) return 2;

  std::mt19937 rng(80016);
  std::vector<float> x(K * rows);
  std::vector<uint8_t> a(x.size()), b(x.size()), as(x.size() / 16), bs(x.size() / 16);
  for (float& v : x) {
    const float f = float(int(rng() % 65535) - 32767) / 8192;
    v = std::bit_cast<float>(std::bit_cast<uint32_t>(f) & 0xffff0000u);
  }
  for (int warm = 0; warm < 2; ++warm) {
    quantize_rows<reference_q8>(x.data(), rows, K, a.data(), as.data());
    quantize_rows<candidate_q8>(x.data(), rows, K, b.data(), bs.data());
  }
  if (a != b || as != bs) { puts("FAIL complete per-row warmup byte gate"); return 1; }
  printf("{\"mode\":\"warm_cpu_per_row_quantizer\",\"K_per_call\":%zu,\"rows_per_pass\":%zu,\"warmup_passes_per_arm\":2,\"input_kind\":\"preconverted_float32_from_synthetic_BF16_bits\",\"includes_BF16_conversion\":false,\"includes_file_IO\":false,\"includes_model_loading\":false,\"input_sha256\":\"%s\",\"code_sha256\":\"%s\",\"scale_sha256\":\"%s\",\"complete_byte_gate\":true}\n",
         K, rows, sha256(x.data(), x.size() * 4).c_str(), sha256(a.data(), a.size()).c_str(), sha256(as.data(), as.size()).c_str());
  volatile uint64_t consume = 0;
  constexpr int repetitions = 16;
  for (int leg = 0; leg < 8; ++leg) {
    const bool candidate = leg % 2;
    uint8_t* q = candidate ? b.data() : a.data();
    uint8_t* s = candidate ? bs.data() : as.data();
    const auto start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      if (candidate) quantize_rows<candidate_q8>(x.data(), rows, K, q, s);
      else quantize_rows<reference_q8>(x.data(), rows, K, q, s);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repetitions;
    if (a != b || as != bs) { puts("FAIL complete per-row measured byte gate"); return 1; }
    consume = consume + q[x.size() / 2];
    printf("{\"mode\":\"warm_cpu_per_row_quantizer\",\"leg\":%d,\"arm\":\"%c\",\"K_per_call\":%zu,\"rows_per_pass\":%zu,\"passes\":%d,\"quantizer_calls_per_leg\":%zu,\"ms_per_rows_pass\":%.9f,\"ns_per_row_call\":%.9f,\"complete_byte_gate\":true,\"code_sha256\":\"%s\",\"scale_sha256\":\"%s\"}\n",
           leg + 1, candidate ? 'B' : 'A', K, rows, repetitions, rows * repetitions, ms,
           ms * 1000000.0 / rows, sha256(q, x.size()).c_str(), sha256(s, x.size() / 16).c_str());
  }
  printf("{\"consume\":%llu}\n", (unsigned long long)consume);
}
