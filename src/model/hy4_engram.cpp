#include "model/hy4_engram.h"

#include "util/bf16.h"
#include "util/fd.h"
#include "util/load_threads.h"
#include "util/log.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <thread>

namespace ni::hy4 {

namespace {

using json = nlohmann::json;

float bf16_round(float v) { return bf16_to_f32(f32_to_bf16(v)); }

// float -> float8_e4m3fn -> float, round to nearest even, |v| <= 448 (the caller clamps): c10's conversion in range
float e4m3_round(float v) {
  const float a = std::fabs(v);
  float q;
  if (a < 0x1p-6f) {
    q = std::nearbyint(a * 0x1p9f) * 0x1p-9f;  // subnormals: steps of 2^-9
  } else {
    // A normal e4m3 value has three fraction bits. The fp32 exponent alone gives the power of two below a.
    const float power = std::bit_cast<float>(std::bit_cast<uint32_t>(a) & 0x7f800000u);
    const float step = power * 0.125f;
    q = std::nearbyint(a / step) * step;      // exact: a / step has at most 8 significant bits plus the fraction
  }
  return std::copysign(q, v);
}

// A small tensor of the pack, found through model.safetensors.index.json: its dtype and bytes
struct Small {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

std::string file_of(const json& weight_map, const std::string& dir, const std::string& name) {
  const auto it = weight_map.find(name);
  if (it == weight_map.end() || !it->is_string()) fatal("hy4 engram: %s/model.safetensors.index.json has no %s", dir.c_str(), name.c_str());
  return dir + "/" + it->get<std::string>();
}

Small read_small(const std::string& path, const std::string& name) {
  Fd opened(::open(path.c_str(), O_RDONLY));
  const int fd = opened.get();
  if (fd < 0) fatal("hy4 engram: cannot open %s", path.c_str());
  uint64_t n = 0;
  if (pread(fd, &n, 8, 0) != 8 || n < 2 || n > (512u << 20)) fatal("hy4 engram: bad header in %s", path.c_str());
  std::string h(size_t(n), '\0');
  if (pread(fd, h.data(), size_t(n), 8) != ssize_t(n)) fatal("hy4 engram: short header in %s", path.c_str());
  const json j = json::parse(h, nullptr, false);
  if (!j.is_object() || !j.contains(name)) fatal("hy4 engram: %s has no tensor %s", path.c_str(), name.c_str());
  const json& v = j[name];
  const auto has = [&](const char* k, bool arr) { return v.contains(k) && (arr ? v[k].is_array() : v[k].is_string()); };
  if (!v.is_object() || !has("dtype", false) || !has("shape", true) || !has("data_offsets", true) ||
      v["data_offsets"].size() != 2 || !v["data_offsets"][0].is_number_unsigned() ||
      !v["data_offsets"][1].is_number_unsigned())
    fatal("hy4 engram: %s's header entry in %s is not a safetensors tensor", name.c_str(), path.c_str());
  Small s;
  s.dtype = v["dtype"].get<std::string>();
  for (const auto& d : v["shape"]) {
    if (!d.is_number_integer()) fatal("hy4 engram: %s in %s has a shape that is not integers", name.c_str(), path.c_str());
    s.shape.push_back(d.get<int64_t>());
  }
  const uint64_t b = v["data_offsets"][0].get<uint64_t>(), e = v["data_offsets"][1].get<uint64_t>();
  if (e < b || e - b > (256u << 20)) fatal("hy4 engram: %s in %s is not a small tensor", name.c_str(), path.c_str());
  s.bytes.resize(size_t(e - b));
  if (pread(fd, s.bytes.data(), s.bytes.size(), off_t(8 + n + b)) != ssize_t(s.bytes.size()))
    fatal("hy4 engram: short read of %s in %s", name.c_str(), path.c_str());
  return s;
}

// a vector of floats from a BF16 / F32 tensor of `count` values
std::vector<float> floats_of(const Small& s, size_t count, const std::string& name) {
  std::vector<float> out(count);
  if (s.dtype == "BF16" && s.bytes.size() == count * 2) {
    for (size_t i = 0; i < count; ++i) {
      uint16_t u;
      memcpy(&u, s.bytes.data() + 2 * i, 2);
      out[i] = bf16_to_f32(u);
    }
  } else if (s.dtype == "F32" && s.bytes.size() == count * 4) {
    memcpy(out.data(), s.bytes.data(), count * 4);
  } else {
    fatal("hy4 engram: %s is %s of %zu bytes, expected %zu BF16 or F32 values", name.c_str(), s.dtype.c_str(),
          s.bytes.size(), count);
  }
  return out;
}

}  // namespace

EngramCpu::EngramCpu(const std::string& dir, const nlohmann::json& tc, int layer_id) {
  const std::string key = std::to_string(layer_id);
  ngram_ = tc.value("engram_max_ngram_size", 3);
  per_ngram_ = tc.value("engram_n_head_per_ngram", 8);
  pad_ = tc.value("engram_pad_id", 120002);
  // engram_mask's image and video ids live in the outer config; the text config does not repeat them
  image_ = tc.value("image_token_id", 120120);
  video_ = tc.value("video_token_id", 120124);
  if (ngram_ < 2 || ngram_ > 8) fatal("hy4 engram: engram_max_ngram_size %d unsupported", ngram_);
  if (tc.value("engram_n_embed_per_ngram", 768) != per_ngram_ * kHeadDim)
    fatal("hy4 engram: heads of %d values, expected %d", tc.value("engram_n_embed_per_ngram", 768) / per_ngram_, kHeadDim);
  if (!tc.value("engram_fp8_embedding", true)) fatal("hy4 engram: engram_fp8_embedding false is not run here");

  const json* sizes = nullptr;
  if (tc.contains("engram_layer_head_vocab_sizes") && tc["engram_layer_head_vocab_sizes"].contains(key))
    sizes = &tc["engram_layer_head_vocab_sizes"][key];
  if (!sizes || !sizes->is_array() || int(sizes->size()) != ngram_ - 1)
    fatal("hy4 engram: %s/config.json has no engram_layer_head_vocab_sizes for layer %s", dir.c_str(), key.c_str());
  int64_t rows = 0;
  for (const auto& order : *sizes) {
    if (!order.is_array() || int(order.size()) != per_ngram_)
      fatal("hy4 engram: %s/config.json: layer %s's engram_layer_head_vocab_sizes do not have %d heads per n-gram",
            dir.c_str(), key.c_str(), per_ngram_);
    for (const auto& v : order) {
      if (!v.is_number_integer() || v.get<int64_t>() <= 0)
        fatal("hy4 engram: %s/config.json: layer %s's engram_layer_head_vocab_sizes holds %s, not a positive size",
              dir.c_str(), key.c_str(), v.dump().c_str());
      mod_.push_back(v.get<int64_t>());
      off_.push_back(rows);
      rows += mod_.back();
    }
  }
  if (!tc.contains("engram_hash_multipliers") || !tc["engram_hash_multipliers"].contains(key) ||
      int(tc["engram_hash_multipliers"][key].size()) != ngram_)
    fatal("hy4 engram: %s/config.json's text_config has no engram_hash_multipliers[\"%s\"] (the pack's converter "
          "writes them)", dir.c_str(), key.c_str());
  for (int i = 0; i < ngram_; ++i) mult_[i] = tc["engram_hash_multipliers"][key][size_t(i)].get<int64_t>();
  if (tc.contains("engram_head_offsets") && tc["engram_head_offsets"].contains(key)) {
    const json& o = tc["engram_head_offsets"][key];
    if (o.size() != off_.size()) fatal("hy4 engram: engram_head_offsets[\"%s\"] has %zu values", key.c_str(), o.size());
    for (size_t i = 0; i < off_.size(); ++i)
      if (o[i].get<int64_t>() != off_[i]) fatal("hy4 engram: engram_head_offsets[\"%s\"] differ from the head sizes' cumsum", key.c_str());
  }

  const std::string idx_path = dir + "/model.safetensors.index.json";
  std::ifstream f(idx_path);
  if (!f) fatal("hy4 engram: missing %s", idx_path.c_str());
  const json idx = json::parse(f, nullptr, false);
  if (!idx.is_object() || !idx.contains("weight_map")) fatal("hy4 engram: bad %s", idx_path.c_str());
  const json& wm = idx["weight_map"];
  const std::string pre = "model.layers." + std::to_string(layer_id - 1) + ".engram_layer.";

  {  // lookup_table: I32 [table size]
    const std::string n = pre + "lookup_table";
    const Small s = read_small(file_of(wm, dir, n), n);
    if (s.dtype != "I32" || s.shape.size() != 1 || s.bytes.size() != size_t(s.shape[0]) * 4 || s.shape[0] <= pad_)
      fatal("hy4 engram: %s is not an I32 vector past the pad id", n.c_str());
    lookup_.resize(size_t(s.shape[0]));
    memcpy(lookup_.data(), s.bytes.data(), s.bytes.size());
    int32_t mx = 0;
    for (int32_t v : lookup_) {
      if (v < 0) fatal("hy4 engram: %s holds a negative id", n.c_str());
      mx = std::max(mx, v);
    }
    // the multipliers' bound (draw_hash_multipliers): a compressed id times a multiplier stays within int64
    const int64_t bound = INT64_MAX / (int64_t(mx) + 1) / 2 * 2 + 1;
    for (int i = 0; i < ngram_; ++i)
      if (mult_[i] <= 0 || mult_[i] % 2 == 0 || mult_[i] > bound)
        fatal("hy4 engram: multiplier %d of layer %s (%lld) is not an odd value under the lookup table's bound", i,
              key.c_str(), (long long)mult_[i]);
  }
  gamma_.resize(size_t(heads()) * kHeadDim);
  for (int h = 0; h < heads(); ++h) {
    const std::string n = pre + "lookup_per_head_norms." + std::to_string(h) + ".weight";
    const auto g = floats_of(read_small(file_of(wm, dir, n), n), kHeadDim, n);
    std::copy(g.begin(), g.end(), gamma_.begin() + long(h) * kHeadDim);
  }
  const std::string t = pre + "multi_head_embedding.embedding";
  table_.open({NgramTable::Part{file_of(wm, dir, t + ".weight"), t + ".weight", t + ".scales", t + ".biases"}},
              NgramTable::Fmt::Affine, kHeadDim);
  if (table_.rows() < rows)
    fatal("hy4 engram: the layer %s table has %lld rows, the heads need %lld", key.c_str(), (long long)table_.rows(),
          (long long)rows);
}

void EngramCpu::ids(const int32_t* toks, int pos, int64_t* out) const {
  // build_ngram_tokens: the token k back, or the pad id before the sequence; hash_ngrams: ids past the lookup table
  // read as the pad id's entry
  const int64_t size = int64_t(lookup_.size());
  int64_t mixed[8];
  for (int k = 0; k < ngram_; ++k) {
    const int64_t t = pos - k < 0 ? pad_ : toks[pos - k];
    const int64_t c = t >= 0 && t < size ? lookup_[size_t(t)] : lookup_[size_t(pad_)];
    mixed[k] = c * mult_[k];
  }
  int64_t x = mixed[0];
  for (int m = 2; m <= ngram_; ++m) {
    x ^= mixed[m - 1];
    for (int j = 0; j < per_ngram_; ++j) {
      const size_t hd = size_t((m - 2) * per_ngram_ + j);
      out[hd] = x % mod_[hd] + off_[hd];
    }
  }
}

void EngramCpu::finish_head(float* x, int h) const {
  constexpr float kMax = 448.0f;  // finfo(float8_e4m3fn).max
  // _fp8_round_trip on bf16, each op in fp32 and rounded to bf16 as CUDA's elementwise kernels do; `amax / max` is a
  // tensor by a Python scalar, which CUDA computes as a product with the scalar's fp32 reciprocal
  float amax = 0;
  for (int i = 0; i < kHeadDim; ++i) amax = std::max(amax, std::fabs(x[i]));
  const float scale = bf16_round(bf16_round(std::max(amax, 1e-12f)) * (1.0f / kMax));
  for (int i = 0; i < kHeadDim; ++i) {
    const float y = std::clamp(bf16_round(x[i] / scale), -kMax, kMax);
    const float q = bf16_round(e4m3_round(y) * scale);
    x[i] = bf16_round(x[i] + bf16_round(q - x[i]));  // x + (quantized - x).detach()
  }
  // nn.RMSNorm(96), eps None: torch >= 2.9 takes FLT_EPSILON for bf16 input, normalizes in fp32 and rounds
  // gamma * (rstd * x) to bf16 once
  double ss = 0;
  for (int i = 0; i < kHeadDim; ++i) ss += double(x[i]) * double(x[i]);
  const float ms = float(ss / kHeadDim);
  const float rstd = 1.0f / std::sqrt(ms + FLT_EPSILON);
  const float* g = gamma_.data() + size_t(h) * kHeadDim;
  for (int i = 0; i < kHeadDim; ++i) x[i] = bf16_round(g[i] * (rstd * x[i]));
}

void EngramCpu::rows(const int32_t* toks, int pos0, int R, float* out, uint8_t* mask) const {
  if (R <= 0) return;
  const int H = heads(), W = width();
  std::vector<int64_t> id(size_t(R) * size_t(H));
  std::vector<int> live;
  live.reserve(size_t(R));
  for (int r = 0; r < R; ++r) {
    mask[r] = owns_ngram(toks[pos0 + r]);
    if (mask[r]) {
      ids(toks, pos0 + r, id.data() + live.size() * size_t(H));
      live.push_back(r);
    }
  }
  // the live rows' table rows, packed at the front of out, then moved to their places (from the last: a row only
  // moves down) and the masked rows cleared
  if (!live.empty()) table_.gather(id.data(), int(live.size()), H, out);
  for (size_t k = live.size(); k-- > 0;)
    if (size_t(live[k]) != k) memmove(out + size_t(live[k]) * W, out + k * W, size_t(W) * 4);
  for (int r = 0; r < R; ++r)
    if (!mask[r]) memset(out + size_t(r) * W, 0, size_t(W) * 4);
  auto work = [&](size_t a, size_t b) {
    for (size_t k = a; k < b; ++k) {
      float* row = out + size_t(live[k]) * W;
      for (int i = 0; i < W; ++i) row[i] = bf16_round(row[i]);  // the embedding reads bf16
      for (int h = 0; h < H; ++h) finish_head(row + size_t(h) * kHeadDim, h);
    }
  };
  const size_t n = live.size();
  const unsigned nt = n >= 64 ? std::max(1u, std::min(8u, std::thread::hardware_concurrency())) : 1u;
  if (nt == 1) {
    work(0, n);
    return;
  }
  LoadThreads th;
  const size_t per = (n + nt - 1) / nt;
  for (unsigned t = 0; t < nt; ++t)
    if (t * per < n) th.emplace_back(work, t * per, std::min(n, (t + 1) * per));
  th.join();
}

}  // namespace ni::hy4
