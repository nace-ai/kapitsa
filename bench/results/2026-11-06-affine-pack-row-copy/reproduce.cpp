#include "reference-pack.h"
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

ni::AffinePackedCoeff reference_pack(const uint32_t *, const uint16_t *,
                                     const uint16_t *, uint32_t, uint32_t);
ni::AffinePackedCoeff candidate_pack(const uint32_t *, const uint16_t *,
                                     const uint16_t *, uint32_t, uint32_t);
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
std::string digest(const void *p, size_t n) {
  unsigned char out[32];
  CC_SHA256(p, CC_LONG(n), out);
  char hex[65];
  for (int i = 0; i < 32; ++i)
    snprintf(hex + i * 2, 3, "%02x", out[i]);
  return hex;
}
int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    return 2;
  const bool timing = argc == 2;
  std::ifstream f(argv[1], std::ios::binary);
  uint64_t hlen = 0;
  f.read((char *)&hlen, 8);
  std::string text(hlen, '\0');
  f.read(text.data(), hlen);
  const auto header = json::parse(text);
  for (const std::string prefix :
       {"language_model.model.embed_tokens",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.model.layers.0.mlp.gate_proj"}) {
    const auto &ws = header.at(prefix + ".weight");
    const auto &ss = header.at(prefix + ".scales");
    const uint32_t N = ss.at("shape")[0], G = ss.at("shape")[1], K = G * 64;
    std::vector<uint32_t> w(size_t(N) * K / 8);
    std::vector<uint16_t> s(size_t(N) * G), b(s.size());
    auto read = [&](const std::string &suffix, void *dst, size_t n) {
      const auto &t = header.at(prefix + suffix);
      auto off = t.at("data_offsets");
      if (off[1].get<size_t>() - off[0].get<size_t>() != n)
        throw std::runtime_error("byte size mismatch");
      f.seekg(8 + hlen + off[0].get<size_t>());
      if (!f.read((char *)dst, n))
        throw std::runtime_error("short tensor read");
    };
    if (ws.at("dtype") != "U32" || ss.at("dtype") != "BF16" ||
        header.at(prefix + ".biases").at("dtype") != "BF16")
      return 3;
    read(".weight", w.data(), w.size() * 4);
    read(".scales", s.data(), s.size() * 2);
    read(".biases", b.data(), b.size() * 2);
    auto a = reference_pack(w.data(), s.data(), b.data(), N, G),
         c = candidate_pack(w.data(), s.data(), b.data(), N, G);
    if (a.words != c.words || a.headers != c.headers || a.blocks != c.blocks ||
        a.compressed_blocks != c.compressed_blocks)
      return 1;
    json r = {
        {"tensor", prefix},
        {"N", N},
        {"K", K},
        {"codes_sha256", digest(w.data(), w.size() * 4)},
        {"scales_sha256", digest(s.data(), s.size() * 2)},
        {"biases_sha256", digest(b.data(), b.size() * 2)},
        {"packed_words_sha256", digest(a.words.data(), a.words.size() * 4)},
        {"packed_headers_sha256",
         digest(a.headers.data(), a.headers.size() * 4)},
        {"packed_bytes_equal", true},
        {"compressed_blocks", a.compressed_blocks},
        {"blocks", a.blocks}};
    a = {};
    c = {};
    if (timing) {
      // Warm both implementations before interleaved measured passes.
      for (auto fn : {reference_pack, candidate_pack}) {
        auto q = fn(w.data(), s.data(), b.data(), N, G);
        if (q.blocks != r["blocks"])
          return 1;
      }
      for (int pair = 0; pair < 4; ++pair)
        for (int v : {0, 1}) {
          auto fn = v ? candidate_pack : reference_pack;
          auto t = Clock::now();
          auto q = fn(w.data(), s.data(), b.data(), N, G);
          double ms =
              std::chrono::duration<double, std::milli>(Clock::now() - t)
                  .count();
          r["pack_runs"].push_back({{"variant", v ? "candidate" : "reference"},
                                    {"pair", pair},
                                    {"ms", ms}});
          if (q.blocks != r["blocks"])
            return 1;
        }
    }
    printf("%s\n", r.dump().c_str());
    fflush(stdout);
  }
}
