// Hy4-flash's Engram lookup on the CPU (src/model/hy4_engram.h) on a synthetic pack in a temporary folder: no model,
// no GPU. Checks
//   - the hashed n-gram ids against the reference's hash_ngrams and build_ngram_tokens (transformers'
//     modeling_hy_v4_flash.py, revision c2ec8505), written out here: the tokens before the sequence and the ids past
//     the lookup table read as the pad id, XOR of the compressed ids times their multipliers, modulo each head's size,
//     plus the head's offset in the table;
//   - the fp8 e4m3 round trip on values whose result is known by hand (ties to even, subnormals), then the head norm;
//   - the head norm's eps (FLT_EPSILON), on a row small enough for eps to set the output;
//   - rows(): the mask of pad, image and video tokens, masked rows zero, and every other row equal bit for bit to the
//     same position gathered alone (one row: one thread), so the compaction of live rows and the 8-thread split change
//     nothing;
//   - configs and packs the constructor refuses with a message, not a crash: a head size of 0 (a division by zero
//     in the hash) or below 0, and a tensor whose header entry has no dtype.
// Usage: hy4_engram_test
#include "model/hy4_engram.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ni;
using json = nlohmann::json;

namespace {

int g_fail = 0;
float bf16_round(float v) { return bf16_to_f32(f32_to_bf16(v)); }
void check(bool ok, const char* what) {
  std::printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}

struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

// drop_dtype: a tensor whose header entry is written without its dtype (a malformed header)
void write_safetensors(const std::string& path, const std::vector<Tensor>& ts, const std::string& drop_dtype = "") {
  json h = json::object();
  size_t off = 0;
  for (const Tensor& t : ts) {
    h[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {off, off + t.bytes.size()}}};
    if (t.name == drop_dtype) h[t.name].erase("dtype");
    off += t.bytes.size();
  }
  std::string hs = h.dump();
  hs.resize((hs.size() + 7) / 8 * 8, ' ');
  std::ofstream f(path, std::ios::binary);
  const uint64_t n = hs.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hs.data(), std::streamsize(hs.size()));
  for (const Tensor& t : ts) f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
}

template <class T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  memcpy(b.data(), v.data(), b.size());
  return b;
}

constexpr int kVocab = 128, kPad = 100, kImage = 101, kVideo = 102, kN = 3, kPer = 8, kDim = 96;

}  // namespace

int main() {
  // ---- the pack: layer 2's Engram tensors in one file ----
  char tmpl[] = "/tmp/hy4_engram_test.XXXXXX";
  if (!mkdtemp(tmpl)) {
    std::perror("mkdtemp");
    return 1;
  }
  const std::string dir = tmpl;
  std::mt19937_64 rng(20261119);
  std::vector<std::vector<int64_t>> sizes(kN - 1);
  const int64_t primes[] = {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
  int64_t rows = 0;
  for (int o = 0; o < kN - 1; ++o)
    for (int j = 0; j < kPer; ++j) rows += (sizes[size_t(o)].push_back(primes[o * kPer + j]), primes[o * kPer + j]);
  const int64_t mult[kN] = {1000003, 7777777, 123456789};
  std::vector<int32_t> lookup(kVocab);
  for (int i = 0; i < kVocab; ++i) lookup[size_t(i)] = (i * 37) % kVocab;  // a permutation, as the frozen map is
  const std::string pre = "model.layers.1.engram_layer.", emb = pre + "multi_head_embedding.embedding";
  std::vector<Tensor> ts;
  ts.push_back({pre + "lookup_table", "I32", {kVocab}, bytes_of(lookup)});
  std::vector<uint16_t> one(kDim, f32_to_bf16(1.0f));
  for (int h = 0; h < (kN - 1) * kPer; ++h)
    ts.push_back({pre + "lookup_per_head_norms." + std::to_string(h) + ".weight", "BF16", {kDim}, bytes_of(one)});
  std::vector<uint32_t> codes(size_t(rows) * kDim / 8);
  for (auto& c : codes) c = uint32_t(rng());
  std::vector<uint16_t> sc(size_t(rows) * kDim / 32), bi(sc.size());
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (size_t i = 0; i < sc.size(); ++i) {
    sc[i] = f32_to_bf16(0.01f + 0.02f * std::fabs(u(rng)));
    bi[i] = f32_to_bf16(0.1f * u(rng));
  }
  ts.push_back({emb + ".weight", "U32", {rows, kDim / 8}, bytes_of(codes)});
  ts.push_back({emb + ".scales", "BF16", {rows, kDim / 32}, bytes_of(sc)});
  ts.push_back({emb + ".biases", "BF16", {rows, kDim / 32}, bytes_of(bi)});
  write_safetensors(dir + "/model-engram.safetensors", ts);
  json wm = json::object();
  for (const Tensor& t : ts) wm[t.name] = "model-engram.safetensors";
  std::ofstream(dir + "/model.safetensors.index.json") << json{{"weight_map", wm}}.dump();
  const json tc = {{"engram_max_ngram_size", kN},
                   {"engram_n_head_per_ngram", kPer},
                   {"engram_n_embed_per_ngram", kPer * kDim},
                   {"engram_pad_id", kPad},
                   {"image_token_id", kImage},
                   {"video_token_id", kVideo},
                   {"engram_fp8_embedding", true},
                   {"engram_layer_head_vocab_sizes", {{"2", sizes}}},
                   {"engram_hash_multipliers", {{"2", {mult[0], mult[1], mult[2]}}}}};
  const hy4::EngramCpu e(dir, tc, 2);
  const int H = e.heads(), W = e.width();

  // ---- the ids: the reference's build_ngram_tokens + hash_ngrams, plus each head's offset ----
  const int T = 100;
  std::vector<int32_t> toks(static_cast<size_t>(T));
  for (int i = 0; i < T; ++i) toks[size_t(i)] = int32_t(rng() % (kVocab + 20));  // some past the lookup table
  for (int i : {0, 37, 50, 51, 99}) toks[size_t(i)] = kPad;
  toks[60] = kImage;
  toks[61] = kVideo;
  size_t id_bad = 0;
  for (int pos = 0; pos < T; ++pos) {
    int64_t mixed[kN];
    for (int k = 0; k < kN; ++k) {
      const int64_t t = pos < k ? kPad : toks[size_t(pos - k)];  // F.pad + the document-offset test
      const int64_t c = t < kVocab ? lookup[size_t(t)] : lookup[kPad];
      mixed[k] = c * mult[k];
    }
    std::vector<int64_t> want;
    int64_t base = 0;
    for (int n = 2; n <= kN; ++n) {
      int64_t mix = mixed[0];
      for (int o = 1; o < n; ++o) mix ^= mixed[o];
      for (int j = 0; j < kPer; ++j) {
        const int64_t m = sizes[size_t(n - 2)][size_t(j)];
        want.push_back(mix % m + base);
        base += m;
      }
    }
    std::vector<int64_t> got(static_cast<size_t>(H));
    e.ids(toks.data(), pos, got.data());
    for (int h = 0; h < H; ++h) id_bad += got[size_t(h)] != want[size_t(h)];
  }
  char nm[128];
  std::snprintf(nm, sizeof nm, "ids of %d positions x %d heads against the reference's hash (%zu differ)", T, H, id_bad);
  check(id_bad == 0, nm);

  // ---- the e4m3 round trip: amax 448 makes the scale 1, so each value lands on its e4m3 neighbour ----
  {
    const std::pair<float, float> cases[] = {
        {448.0f, 448.0f},     {1.0625f, 1.0f},          {1.1875f, 1.25f},         {-1.1875f, -1.25f},
        {3.5f, 3.5f},         {17.0f, 16.0f},           {300.0f, 288.0f},         {240.0f, 240.0f},
        {0x1p-10f, 0.0f},     {1.5f * 0x1p-9f, 0x1p-8f}, {3.5f * 0x1p-9f, 0x1p-7f}, {0x1p-6f, 0x1p-6f}};
    std::vector<float> x(kDim, 0.0f), q(kDim, 0.0f);
    size_t k = 0;
    for (const auto& [in, out] : cases) {
      x[k] = in;
      q[k++] = out;
    }
    e.finish_head(x.data(), 0);
    // the head's RMSNorm (gamma 1) on the quantized values: fp32 rstd from their mean square, rounded to bf16
    double ss = 0;
    for (float v : q) ss += double(v) * double(v);
    const float rstd = 1.0f / std::sqrt(float(ss / kDim) + FLT_EPSILON);
    size_t bad = 0;
    for (int i = 0; i < kDim; ++i) bad += x[size_t(i)] != bf16_round(rstd * q[size_t(i)]);
    std::snprintf(nm, sizeof nm, "fp8 e4m3 round trip of %zu hand-checked values, then the head norm (%zu differ)", k,
                  bad);
    check(bad == 0, nm);  // (eps is a 1e-9 part of this row's mean square: the next case pins it)
  }
  // ---- the head norm's eps: a row of 2^-14, whose mean square (3.7e-9) is 3% of FLT_EPSILON, so the output is
  // x / sqrt(x^2 + eps), about 0.17 (0.12 with eps doubled, 1 without it). The fp8 round trip and bf16 move x by
  // under 1%, hence the 3% bound ----
  {
    std::vector<float> x(kDim, 0x1p-14f);
    e.finish_head(x.data(), 0);
    const float v = 0x1p-14f, want = v / std::sqrt(v * v + FLT_EPSILON);
    float worst = 0;
    for (float y : x) worst = std::max(worst, std::fabs(y / want - 1.0f));
    std::snprintf(nm, sizeof nm, "head norm's eps: a row of 2^-14 within %.1f%% of x / sqrt(x^2 + eps) = %.3f",
                  100.0 * worst, want);
    check(worst < 0.03f, nm);
  }

  // ---- rows(): mask, masked rows zero, live rows equal to each position gathered alone ----
  std::vector<float> all(static_cast<size_t>(T) * W), one_row(static_cast<size_t>(W));
  std::vector<uint8_t> mask(static_cast<size_t>(T)), m1(1);
  e.rows(toks.data(), 0, T, all.data(), mask.data());
  size_t mask_bad = 0, zero_bad = 0, row_bad = 0, live = 0;
  for (int r = 0; r < T; ++r) {
    const int32_t t = toks[size_t(r)];
    const bool own = t != kPad && t != kImage && t != kVideo;
    mask_bad += bool(mask[size_t(r)]) != own;
    live += own;
    if (!own) {
      for (int i = 0; i < W; ++i) zero_bad += all[size_t(r) * W + size_t(i)] != 0.0f;
      continue;
    }
    e.rows(toks.data(), r, 1, one_row.data(), m1.data());
    row_bad += memcmp(one_row.data(), all.data() + size_t(r) * W, size_t(W) * 4) != 0;
  }
  std::snprintf(nm, sizeof nm, "rows(): mask of %d tokens (%zu wrong), %d masked rows zero (%zu values not)", T,
                mask_bad, T - int(live), zero_bad);
  check(mask_bad == 0 && zero_bad == 0, nm);
  std::snprintf(nm, sizeof nm, "rows(): %zu live rows of one call equal to each gathered alone (%zu differ)", live,
                row_bad);
  check(row_bad == 0 && live >= 64, nm);

  // ---- refused: each with its message ----
  {
    FatalRecovery recover;
    const auto refused = [&](const char* what, const json& cfg, const char* message) {
      std::string got = "(constructed)";
      try {
        const hy4::EngramCpu bad(dir, cfg, 2);
      } catch (const FatalError& x) {
        got = x.what();
      }
      char m[256];
      std::snprintf(m, sizeof m, "refused: %s (%s)", what, got.c_str());
      check(got.find(message) != std::string::npos, m);
    };
    json z = tc;
    z["engram_layer_head_vocab_sizes"]["2"][1][3] = 0;
    refused("a head size of 0", z, "not a positive size");
    z["engram_layer_head_vocab_sizes"]["2"][1][3] = -5;
    refused("a head size below 0", z, "not a positive size");
    write_safetensors(dir + "/model-engram.safetensors", ts, pre + "lookup_table");
    refused("a header entry without its dtype", tc, "is not a safetensors tensor");
  }

  std::filesystem::remove_all(dir);
  std::printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
