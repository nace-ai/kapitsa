// Real BF16 safetensors rows through the unchanged reference and the candidate CPU quantizer.
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
namespace ni {
void reference(const std::vector<const uint16_t*>&, uint32_t, int, bool, uint32_t*, uint16_t*, uint16_t*);
void candidate(const std::vector<const uint16_t*>&, uint32_t, int, bool, uint32_t*, uint16_t*, uint16_t*);
}  // namespace ni
std::string digest(const std::vector<uint32_t>& w, const std::vector<uint16_t>& s,
                   const std::vector<uint16_t>& b) {
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  CC_SHA256_Update(&ctx, w.data(), CC_LONG(w.size() * 4));
  CC_SHA256_Update(&ctx, s.data(), CC_LONG(s.size() * 2));
  CC_SHA256_Update(&ctx, b.data(), CC_LONG(b.size() * 2));
  unsigned char out[32];
  CC_SHA256_Final(out, &ctx);
  char hex[65];
  for (int i = 0; i < 32; ++i) snprintf(hex + i * 2, 3, "%02x", out[i]);
  return hex;
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream f(argv[1], std::ios::binary);
  uint64_t len = 0;
  f.read((char*)&len, 8);
  std::string raw(len, '\0');
  f.read(raw.data(), len);
  auto meta = nlohmann::json::parse(raw);
  for (const char* name : {"fc.weight", "layers.0.mlp.down_proj.weight", "layers.0.mlp.gate_proj.weight"}) {
    auto t = meta.at(name);
    unsigned N = t["shape"][0], K = t["shape"][1];
    if (t["dtype"] != "BF16") return 3;
    std::vector<uint16_t> x(size_t(N) * K);
    f.seekg(8 + len + t["data_offsets"][0].get<size_t>());
    if (!f.read((char*)x.data(), x.size() * 2)) return 4;
    for (unsigned count : {1u, 8u, N}) {
      std::vector<const uint16_t*> rows(count);
      for (unsigned n = 0; n < count; ++n) rows[n] = x.data() + size_t(N - 1 - n) * K;
      for (int bits : {4, 8})
        for (bool even : {false, true}) {
          std::vector<uint32_t> a(size_t(count) * K * bits / 32), b(a.size());
          std::vector<uint16_t> as(size_t(count) * K / 64), ab(as.size()), bs(as.size()), bb(as.size());
          ni::reference(rows, K, bits, even, a.data(), as.data(), ab.data());
          ni::candidate(rows, K, bits, even, b.data(), bs.data(), bb.data());
          if (a != b || as != bs || ab != bb) {
            fprintf(stderr, "MISMATCH %s %u %d %d\n", name, count, bits, even);
            return 1;
          }
          printf("CHECK %s N%u K%u bits%d even%d W/S/B SHA256 %s\n", name, count, K, bits, even,
                 digest(a, as, ab).c_str());
          for (int pair = 0; pair < 4; ++pair)
            for (int cand : {0, 1}) {
              int reps = count > 100 ? 3 : 300;
              auto fn = cand ? ni::candidate : ni::reference;
              auto begin = std::chrono::steady_clock::now();
              for (int i = 0; i < reps; ++i) fn(rows, K, bits, even, b.data(), bs.data(), bb.data());
              double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                              .count() /
                          reps;
              printf("TIME %s N%u K%u bits%d even%d %c%d %.6f ms\n", name, count, K, bits, even,
                     cand ? 'B' : 'A', pair, ms);
            }
        }
    }
  }
}
