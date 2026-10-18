// Image encoders: a model's vision tower and the processor in front of it.
// Shared by every model (here): the encoder interface, loading it on the first image with the memory budget in
// mind, the caches (prepared images by source, embeddings by content), and the prompt side (a placeholder token per
// image expanded to the image's rows, with prefix-reuse keys that tell images apart by content).
// Per model: the processor and the tower behind ImageEncoder (QwenViT, qwen_vit.h, for the Qwen3.5 family; a new
// model's encoder is one more class and one more line in open_encoder), and the model's own handling of image rows:
// its prefill puts the embeddings in (vision::enc_embed_runs for Seq::embeds) and Model::set_rope_positions gives
// them positions (Qwen35: M-RoPE; the default does nothing, for attention without positions as in Nemotron-H).
#pragma once
#include "metal/gpu.h"
#include "vision/embed.h"
#include "vision/image.h"

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ni::vision {

// An image as its encoder takes it: the CPU part of the processor done, the GPU part (encode) to come.
struct Prepared {
  int tokens = 0;                          // embedding rows the image takes in the prompt
  int grid_t = 1, grid_h = 0, grid_w = 0;  // its token grid (positions use it): tokens = grid_t * grid_h * grid_w
  Digest digest{};                         // the picture and the shape the processor gave it: equal digests, equal rows
  int width = 0, height = 0;               // the encoder's input size (after resizing)
  std::vector<uint8_t> pixels;             // the encoder's input (QwenViT: resized RGB rows); empty when only the
                                           // metadata came from the cache (Vision::embeddings prepares it again)
};

class ImageEncoder {
 public:
  virtual ~ImageEncoder() = default;
  virtual std::string name() const = 0;
  virtual std::string describe() const { return name(); }  // for logs: its shape and the image sizes it takes
  virtual int placeholder() const = 0;    // the token the chat template writes for one image
  virtual int out_dim() const = 0;        // embedding width (the language model's hidden size)
  virtual Prepared prepare(const Image& im) const = 0;      // CPU only, thread-safe
  virtual size_t weight_bytes() const = 0;                  // GPU memory of the loaded weights
  virtual size_t work_bytes(const Prepared& p) const = 0;   // GPU scratch encode(p) allocates (and frees)
  virtual bool loaded() const = 0;
  virtual void load() = 0;
  virtual void unload() = 0;
  // p's embeddings into `out` (p.tokens rows of out_dim() bf16): its own short command buffers, waited for.
  virtual void encode(const Prepared& p, const Buffer& out) = 0;
};

// The encoder of the model in `dir`, or null when the model takes no images (no vision tower in its config or files).
// `max_pixels` > 0 caps the pixels an image keeps (encoders that resize: fewer tokens, faster), else the model's own.
std::unique_ptr<ImageEncoder> open_encoder(const std::string& dir, long max_pixels = 0);

// The engine's handle on an encoder: prepares a request's images (data: URLs remembered by content, so the screenshots
// an agent resends every turn are not decoded again), loads the encoder on the first image that needs encoding
// (text-only use never pays for it), keeps the embeddings of recent images (NI_IMAGE_CACHE_MB, default 256: an image
// sent again is not encoded again) and gives the memory back under pressure.
class Vision {
 public:
  explicit Vision(std::unique_ptr<ImageEncoder> enc);
  ~Vision();
  ImageEncoder& encoder() { return *enc_; }
  // The images of a request, prepared (thread-safe: called outside the engine lock). Throws Error.
  std::vector<Prepared> prepare(const std::vector<Source>& sources);
  // p's embeddings (bf16 rows), from the cache or encoded now (its source decoded again if p came from the cache).
  // `make_room(bytes)` must return true when `bytes` more may be allocated within the memory budget (it may free other
  // memory). Embeddings handed out stay valid until end_request(). *encoded: whether they were encoded now (else
  // they came from the cache). Throws Error.
  Buffer embeddings(Prepared& p, const Source& src, const std::function<bool(size_t)>& make_room,
                    bool* encoded = nullptr);
  // The request's prefill is done: the cache goes back to its size.
  void end_request();
  // Frees the cached embeddings of earlier requests, and with `weights` the encoder's weights too (the KV cache needs
  // the room; the next image loads them again). False if nothing was freed.
  bool release(bool weights);

 private:
  struct Entry {
    Digest digest;
    Buffer buf;
    uint64_t used = 0;  // request that last used it
  };
  void evict(size_t cap, bool keep_current);
  std::unique_ptr<ImageEncoder> enc_;
  std::list<Entry> cache_;  // embeddings, most recently used first
  size_t cache_bytes_ = 0, cache_cap_ = 0;
  uint64_t request_ = 1;
  struct Meta {
    Digest key;
    Prepared p;  // without pixels
  };
  std::mutex meta_mu_;
  std::list<Meta> meta_;  // prepared data: URL images by the URL's SHA-256, most recent first
};

// ---- the prompt side ----

// A prompt with its images in place: `ids` for the model (each placeholder expanded to its image's rows), `keys` for
// prefix reuse (image rows get ids derived from the image's digest, ids no token has), the images' spans and digests.
struct Prompt {
  std::vector<int> ids, keys;
  std::vector<ImageSpan> spans;
  std::vector<Digest> digests;  // per span
};
// Expands each placeholder token of `ids` (in order) to its image's rows. `index` (e.g. a turn boundary) moves with
// the expansion. Throws Error when the placeholders and the images do not pair up.
Prompt expand(const std::vector<int>& ids, int placeholder, const std::vector<Prepared>& images, int* index = nullptr);
// Rows two prompts share as a reusable prefix: equal keys, and every image in the prefix the same image (digest and
// span) in both: a different image never reuses another image's state, however its rows' keys compare.
size_t common_prefix(const std::vector<int>& a_keys, const std::vector<ImageSpan>& a_spans,
                     const std::vector<Digest>& a_digests, const Prompt& b);

}  // namespace ni::vision
