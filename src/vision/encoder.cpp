// The parts of image input every model shares (see encoder.h).
#include "vision/encoder.h"
#include "util/log.h"
#include "vision/qwen_vit.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace ni::vision {

std::unique_ptr<ImageEncoder> open_encoder(const std::string& dir, long max_pixels) {
  QwenViTConfig qc;
  if (QwenViT::supports(dir, &qc)) {
    if (max_pixels > 0) qc.max_pixels = std::max(max_pixels, qc.min_pixels);
    return std::make_unique<QwenViT>(dir, qc);
  }
  // other models' encoders go here (each behind ImageEncoder)
  return nullptr;
}

// ---- Vision ----

Vision::Vision(std::unique_ptr<ImageEncoder> enc) : enc_(std::move(enc)) {
  const char* v = getenv("NI_IMAGE_CACHE_MB");
  cache_cap_ = size_t((v && *v ? atof(v) : 256.0) * 1048576.0);
}

Vision::~Vision() {
  for (Entry& e : cache_) Gpu::get().release(e.buf);
}

std::vector<Prepared> Vision::prepare(const std::vector<Source>& sources) {
  std::vector<Prepared> out;
  const FetchLimits lim = fetch_limits();
  for (const Source& s : sources) {
    // data: URLs (and inline bytes) are their content: remembered by a hash of it, so an image resent every turn is
    // neither decoded nor resized again (http(s) URLs are fetched each time: what they point to may change)
    const bool inline_data = !s.bytes.empty() || (s.url.size() > 5 && strncasecmp(s.url.c_str(), "data:", 5) == 0);
    Digest key{};
    if (inline_data) {
      key = s.bytes.empty() ? sha256(s.url.data(), s.url.size()) : sha256(s.bytes.data(), s.bytes.size());
      std::lock_guard<std::mutex> lk(meta_mu_);
      auto it = std::find_if(meta_.begin(), meta_.end(), [&](const Meta& m) { return m.key == key; });
      if (it != meta_.end()) {
        meta_.splice(meta_.begin(), meta_, it);
        out.push_back(meta_.front().p);
        continue;
      }
    }
    Prepared p = enc_->prepare(decode(fetch(s, lim)));
    if (inline_data) {
      std::vector<uint8_t> px = std::move(p.pixels);  // remembered without its pixels (no copy of them)
      Meta m{key, p};
      p.pixels = std::move(px);
      std::lock_guard<std::mutex> lk(meta_mu_);
      meta_.push_front(std::move(m));
      if (meta_.size() > 256) meta_.pop_back();
    }
    out.push_back(std::move(p));
  }
  return out;
}

void Vision::evict(size_t cap, bool keep_current) {
  for (auto it = cache_.end(); cache_bytes_ > cap && it != cache_.begin();) {
    --it;
    if (keep_current && it->used == request_) continue;
    cache_bytes_ -= it->buf.size;
    Gpu::get().release(it->buf);
    it = cache_.erase(it);
  }
}

Buffer Vision::embeddings(Prepared& p, const Source& src, const std::function<bool(size_t)>& make_room,
                          bool* encoded) {
  if (encoded) *encoded = false;
  for (auto it = cache_.begin(); it != cache_.end(); ++it)
    if (it->digest == p.digest) {
      it->used = request_;
      cache_.splice(cache_.begin(), cache_, it);
      return cache_.front().buf;
    }
  if (p.pixels.empty()) {  // metadata from the cache: the pixels again
    Prepared q = enc_->prepare(decode(fetch(src, fetch_limits())));
    if (q.digest != p.digest) throw Error("an image changed while it was being read");
    p = std::move(q);
  }
  const size_t out_bytes = size_t(p.tokens) * size_t(enc_->out_dim()) * 2;
  const size_t need = out_bytes + enc_->work_bytes(p) + (enc_->loaded() ? 0 : enc_->weight_bytes());
  if (!make_room(need)) {
    evict(0, true);  // embeddings of earlier requests first
    if (!make_room(need))
      throw Error("insufficient memory for the image: encoding it takes " + std::to_string(need / 1048576) +
                  " MB of GPU memory (the encoder's weights and scratch) beyond the KV cache Kapitsa holds within its "
                  "memory budget");
  }
  if (!enc_->loaded()) {
    const double t0 = now_ms();
    enc_->load();
    logi("vision: %s loaded on the first image: %.2f GB in %.0f ms", enc_->name().c_str(), enc_->weight_bytes() / 1e9,
         now_ms() - t0);
  }
  Buffer out = Gpu::get().alloc(out_bytes + 16, "image_embeddings");
  const double t0 = now_ms();
  enc_->encode(p, out);
  logi("vision: %dx%d image -> %d tokens (grid %dx%d) encoded in %.0f ms", p.width, p.height, p.tokens, p.grid_h,
       p.grid_w, now_ms() - t0);
  cache_.push_front(Entry{p.digest, out, request_});
  cache_bytes_ += out.size;
  if (encoded) *encoded = true;
  return out;
}

void Vision::end_request() {
  evict(cache_cap_, false);
  ++request_;
}

bool Vision::release(bool weights) {
  const size_t before = cache_bytes_;
  evict(0, true);
  weights = weights && enc_->loaded();
  if (weights) enc_->unload();
  if (weights || cache_bytes_ < before)
    logi("vision: released %s%.0f MB of cached image embeddings for the KV cache", weights ? "the encoder's weights and " : "",
         (before - cache_bytes_) / 1048576.0);
  return weights || cache_bytes_ < before;
}

// ---- the prompt side ----

static uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Key of row j of an image: negative (no token id is), from the image's digest and the row.
static int image_key(const Digest& d, int j) {
  uint64_t a, b;
  memcpy(&a, d.data(), 8);
  memcpy(&b, d.data() + 8, 8);
  const uint64_t h = mix64(a ^ mix64(b + uint64_t(j)));
  return int(0x80000000u | uint32_t(h >> 33));
}

Prompt expand(const std::vector<int>& ids, int placeholder, const std::vector<Prepared>& images, int* index) {
  Prompt pr;
  size_t k = 0;
  int shift = 0;
  size_t total = ids.size();
  for (const Prepared& p : images) total += size_t(std::max(0, p.tokens - 1));
  pr.ids.reserve(total);
  pr.keys.reserve(total);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != placeholder) {
      pr.ids.push_back(ids[i]);
      pr.keys.push_back(ids[i]);
      continue;
    }
    if (k >= images.size())
      throw Error("the prompt has more image placeholders than images (an <|image_pad|> written in the text?)");
    const Prepared& p = images[k++];
    pr.spans.push_back(ImageSpan{int(pr.ids.size()), p.tokens, p.grid_t, p.grid_h, p.grid_w});
    pr.digests.push_back(p.digest);
    for (int j = 0; j < p.tokens; ++j) {
      pr.ids.push_back(placeholder);
      pr.keys.push_back(image_key(p.digest, j));
    }
    if (index && int(i) < *index) shift += p.tokens - 1;
  }
  if (k != images.size())
    throw Error("the chat template rendered " + std::to_string(k) + " of the request's " + std::to_string(images.size()) +
                " images (images are taken in user, assistant and tool messages, not in system messages)");
  if (index && *index >= 0) *index += shift;
  return pr;
}

size_t common_prefix(const std::vector<int>& a_keys, const std::vector<ImageSpan>& a_spans,
                     const std::vector<Digest>& a_digests, const Prompt& b) {
  const size_t n = std::min(a_keys.size(), b.keys.size());
  size_t i = 0;
  while (i < n && a_keys[i] == b.keys[i]) ++i;
  for (size_t k = 0; k < b.spans.size() && size_t(b.spans[k].pos) < i; ++k) {
    bool same = false;
    for (size_t j = 0; j < a_spans.size() && !same; ++j)
      same = a_spans[j].pos == b.spans[k].pos && a_spans[j].n == b.spans[k].n && a_digests[j] == b.digests[k];
    if (!same) return size_t(b.spans[k].pos);
  }
  return i;
}

// ---- prefill rows from image embeddings ----

struct EmbedRowsParams {
  uint32_t D, src_ld, dst_row0, src_row0;
};

void enc_embed_runs(Encoder& e, const std::vector<EmbedRun>& runs, int pos0, int T, const Buffer& x, int D) {
  static Pipeline pl = Gpu::get().pipeline("embed_rows_bf16");
  for (const EmbedRun& r : runs) {
    const int lo = std::max(pos0, r.pos), hi = std::min(pos0 + T, r.pos + r.n);
    if (lo >= hi) continue;
    const EmbedRowsParams p{uint32_t(D), uint32_t(r.src_ld ? r.src_ld : D), uint32_t(lo - pos0),
                            uint32_t(r.row + (lo - r.pos))};
    e.use(pl);
    e.buf(0, r.src);
    e.buf(1, x);
    e.val(2, p);
    e.dispatch((uint32_t(D) / 4 + 63) / 64, uint32_t(hi - lo), 1, 64);
  }
}

}  // namespace ni::vision
