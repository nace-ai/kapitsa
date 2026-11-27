// Private exact-source CPU evidence driver. Does not load an engine or touch the GPU.
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

void reference_q8(const float*, int64_t, uint8_t*, uint8_t*);
void candidate_q8(const float*, int64_t, uint8_t*, uint8_t*);
struct Hash {
  CC_SHA256_CTX context;
  Hash() { CC_SHA256_Init(&context); }
  void add(const void* data, size_t n) { CC_SHA256_Update(&context, data, CC_LONG(n)); }
  std::string finish() {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(digest, &context);
    char hex[CC_SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(hex + i * 2, 3, "%02x", digest[i]);
    return hex;
  }
};

int real_head(int argc, char** argv) {
  if (argc != 6 || strcmp(argv[1], "--bf16")) { puts("usage: cpu-evidence --bf16 FILE OFFSET ROWS K"); return 2; }
  const uint64_t offset = strtoull(argv[3], nullptr, 10), rows = strtoull(argv[4], nullptr, 10);
  const size_t K = strtoull(argv[5], nullptr, 10);
  if (!K || K % 32 || !rows || K > 1048576) return 2;
  std::ifstream file(argv[2], std::ios::binary);
  file.seekg(offset);
  if (!file) return 2;
  std::vector<uint16_t> raw(K);
  std::vector<float> x(K);
  std::vector<uint8_t> a(K), b(K), as(K / 16), bs(K / 16);
  Hash input, aq, bq, asc, bsc;
  uint64_t nonfinite = 0;
  for (uint64_t row = 0; row < rows; ++row) {
    if (!file.read(reinterpret_cast<char*>(raw.data()), K * 2)) { printf("short BF16 row=%llu\n", (unsigned long long)row); return 1; }
    input.add(raw.data(), K * 2);
    for (size_t i = 0; i < K; ++i) {
      x[i] = std::bit_cast<float>(uint32_t(raw[i]) << 16);
      if (!std::isfinite(x[i])) ++nonfinite;
    }
    reference_q8(x.data(), K, a.data(), as.data());
    candidate_q8(x.data(), K, b.data(), bs.data());
    if (a != b || as != bs) { printf("BF16 byte gate failed row=%llu\n", (unsigned long long)row); return 1; }
    aq.add(a.data(), K); bq.add(b.data(), K); asc.add(as.data(), K / 16); bsc.add(bs.data(), K / 16);
  }
  printf("{\"mode\":\"real_bf16_head\",\"rows\":%llu,\"K\":%zu,\"nonfinite_input_values\":%llu,\"complete_code_and_scale_byte_gate\":true,\"input_sha256\":\"%s\",\"reference_code_sha256\":\"%s\",\"candidate_code_sha256\":\"%s\",\"reference_scale_sha256\":\"%s\",\"candidate_scale_sha256\":\"%s\"}\n",
         (unsigned long long)rows, K, (unsigned long long)nonfinite, input.finish().c_str(), aq.finish().c_str(), bq.finish().c_str(), asc.finish().c_str(), bsc.finish().c_str());
  return 0;
}

int main(int argc, char** argv) {
  const bool selected = argc == 4 && !strcmp(argv[1], "--cpu");
  if (argc > 1 && !selected) return real_head(argc, argv);
  const size_t rows = selected ? strtoull(argv[3], nullptr, 10) : 1024;
  std::vector<size_t> shapes{1536u, 2048u, 3072u, 3136u, 4096u};
  if (selected) shapes = {size_t(strtoull(argv[2], nullptr, 10))};
  if (!rows || rows > 1048576) return 2;
  std::mt19937 rng(80016);
  volatile uint64_t consume = 0;
  for (size_t K : shapes) {
    if (!K || K % 32 || K > 1048576 || K * rows < 32768) return 2;
    std::vector<float> x(K * rows);
    std::vector<uint8_t> a(x.size()), b(x.size()), as(x.size() / 16), bs(x.size() / 16);
    for (float& v : x) {
      const float f = float(int(rng() % 65535) - 32767) / 8192;
      v = std::bit_cast<float>(std::bit_cast<uint32_t>(f) & 0xffff0000u);
    }
    for (int warm = 0; warm < 2; ++warm) { reference_q8(x.data(), x.size(), a.data(), as.data()); candidate_q8(x.data(), x.size(), b.data(), bs.data()); }
    if (a != b || as != bs) { puts("warmup byte gate failed"); return 1; }
    Hash input, codes, scales;
    input.add(x.data(), x.size() * 4); codes.add(a.data(), a.size()); scales.add(as.data(), as.size());
    printf("{\"mode\":\"warm_cpu_quantizer\",\"rows\":%zu,\"K\":%zu,\"input_sha256\":\"%s\",\"code_sha256\":\"%s\",\"scale_sha256\":\"%s\"}\n", rows, K, input.finish().c_str(), codes.finish().c_str(), scales.finish().c_str());
    for (int leg = 0; leg < 8; ++leg) {
      const bool candidate = leg % 2;
      auto start = std::chrono::steady_clock::now();
      for (int repetition = 0; repetition < 16; ++repetition) {
        if (candidate) candidate_q8(x.data(), x.size(), b.data(), bs.data());
        else reference_q8(x.data(), x.size(), a.data(), as.data());
      }
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 16;
      if (a != b || as != bs) { puts("measured byte gate failed"); return 1; }
      consume = consume + (candidate ? b[16384] : a[16384]);
      printf("{\"K\":%zu,\"rows\":%zu,\"leg\":%d,\"arm\":\"%c\",\"repetitions\":16,\"ms_per_call\":%.9f,\"complete_byte_gate\":true}\n", K, rows, leg + 1, candidate ? 'B' : 'A', ms);
    }
  }
  printf("{\"consume\":%llu}\n", (unsigned long long)consume);
}
