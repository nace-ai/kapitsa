#include "model/dflash2.h"
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

namespace ni {
// This driver accepts only plain MLX affine tensors; IQ and EXL3 layouts are
// not part of the comparison.
size_t QLinear::row_bytes() const { return size_t(K) * bits / 8; }
void reference_row(const QLinear &, uint32_t, uint16_t *);
void candidate_row(const QLinear &, uint32_t, uint16_t *);
} // namespace ni
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
    const uint32_t N = ss.at("shape")[0], G = ss.at("shape")[1], K = G * 64,
                   bits = ws.at("shape")[1].get<uint32_t>() * 32 / K;
    std::vector<uint32_t> w(size_t(N) * K * bits / 32);
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
    json r = {{"tensor", prefix},
              {"N", N},
              {"K", K},
              {"bits", bits},
              {"codes_sha256", digest(w.data(), w.size() * 4)},
              {"scales_sha256", digest(s.data(), s.size() * 2)},
              {"biases_sha256", digest(b.data(), b.size() * 2)}};
    ni::Tensor tw, ts, tb;
    tw.buf.base = (uint8_t *)w.data();
    ts.buf.base = (uint8_t *)s.data();
    tb.buf.base = (uint8_t *)b.data();
    ni::QLinear H;
    H.w = &tw;
    H.s = &ts;
    H.b = &tb;
    H.N = N;
    H.K = K;
    H.gs = 64;
    H.bits = int(bits);
    std::vector<uint16_t> ar(size_t(N) * K), br(ar.size());
    for (uint32_t i = 0; i < N; ++i) {
      ni::reference_row(H, i, ar.data() + size_t(i) * K);
      ni::candidate_row(H, i, br.data() + size_t(i) * K);
    }
    if (ar != br)
      return 1;
    r["decoded_bytes_equal"] = true;
    r["decoded_sha256"] = digest(ar.data(), ar.size() * 2);
    ar.clear();
    ar.shrink_to_fit();
    if (timing) {
      for (int pair = 0; pair < 4; ++pair)
        for (int v : {0, 1}) {
          auto fn = v ? ni::candidate_row : ni::reference_row;
          auto t = Clock::now();
          for (uint32_t i = 0; i < N; ++i)
            fn(H, i, br.data() + size_t(i) * K);
          double ms =
              std::chrono::duration<double, std::milli>(Clock::now() - t)
                  .count();
          r["row_runs"].push_back({{"variant", v ? "candidate" : "reference"},
                                   {"pair", pair},
                                   {"ms", ms}});
        }
    }
    printf("%s\n", r.dump().c_str());
    fflush(stdout);
  }
}
