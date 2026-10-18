// The Qwen3.5-family image encoder (Qwen3.5-4B, Qwen3.6-35B-A3B, Qwen3.8-27B, ...): the Qwen2-VL image processor
// (smart_resize, bicubic resize, normalize, 16x16 patches, a still image repeated over the 2 temporal patches) and the
// ViT (blocks of LayerNorm + attention with 2D RoPE + GELU MLP, learned position embeddings interpolated to the grid,
// 2x2 patch merger to the language model's width: 24 blocks of width 1024 and head dimension 64 on the 4B, 27 of 1152
// and 72 on the 35B and 27B). Weights bf16, run on the prefill GEMMs; the small kernels in vit.metal.
#pragma once
#include "loader/safetensors.h"
#include "vision/encoder.h"
#include "vision/vit_common.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ni::vision {

struct QwenViTConfig {
  int depth = 27, hidden = 1152, inter = 4304, heads = 16, out_hidden = 2048;
  int patch = 16, temporal = 2, merge = 2, in_ch = 3, pos_embeds = 2304;
  bool gelu_tanh = true;          // the blocks' MLP activation (gelu_pytorch_tanh); the merger's is exact GELU
  int image_token = 248056;
  // processor
  long min_pixels = 65536, max_pixels = 16777216;
  float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
  double rescale = 1.0 / 255.0;
  bool do_rescale = true, do_normalize = true;
};

// The processor's resize target for an h x w image: multiples of `factor` (patch x merge), pixels within
// [min_pixels, max_pixels], aspect ratio kept (Qwen2-VL smart_resize, Python's rounding). Throws Error past 200:1.
std::pair<int, int> qwen_smart_resize(int h, int w, int factor, long min_pixels, long max_pixels);
// PIL's Image.resize(..., BICUBIC) of 8-bit RGB rows, to the bit (Pillow's fixed-point two-pass resampling).
std::vector<uint8_t> pil_resize_bicubic(const std::vector<uint8_t>& rgb, int w, int h, int ow, int oh);

class QwenViT : public ImageEncoder {
 public:
  QwenViT(const std::string& dir, const QwenViTConfig& c);
  ~QwenViT() override;
  static bool supports(const std::string& dir, QwenViTConfig* c);  // a Qwen3.5-family vision tower in dir's files

  std::string name() const override { return "Qwen3.5 ViT"; }
  std::string describe() const override;
  int placeholder() const override { return c_.image_token; }
  int out_dim() const override { return c_.out_hidden; }
  Prepared prepare(const Image& im) const override;
  size_t weight_bytes() const override;
  size_t work_bytes(const Prepared& p) const override;
  bool loaded() const override { return loaded_; }
  void load() override;
  void unload() override;
  void encode(const Prepared& p, const Buffer& out) override;

  const QwenViTConfig& config() const { return c_; }
  // The processor's output as the reference processors return it (pixel_values: [patches, C x T x P x P] in
  // (channel, frame, row, column) order, float32): for checks against them.
  std::vector<float> pixel_values(const Prepared& p) const override;
  // Tests: the hidden state after `layers` blocks (fp32 [patches, hidden]; layers = depth: before the merger) instead
  // of the embeddings, when set before encode() (then out must hold patches x hidden floats).
  void set_debug_stop(int layers) override { dbg_stop_ = layers; }
  int debug_rows(const Prepared& p) const override { return (p.height / c_.patch) * (p.width / c_.patch); }
  int debug_width() const override { return c_.hidden; }

 private:
  struct Block {
    Buffer n1w, n1b, n2w, n2b;
    VitLin qkv, proj, fc1, fc2;
  };

  std::string dir_, prefix_;
  QwenViTConfig c_;
  uint32_t inter_pad_ = 0;  // MLP width padded to a multiple of 64 (the 35B's and 27B's 4304 -> 4352; the 4B's 4096)
  VitWeights w_;            // the vision tensors: headers while loading, then our own buffers
  bool loaded_ = false;
  VitLin patch_;            // [hidden, T*P*P*C] in (frame, row, column, channel) order
  Buffer pos_;              // bf16 [pos_embeds, hidden]
  std::vector<Block> blocks_;
  Buffer mnw_, mnb_;        // merger norm
  VitLin m1_, m2_;
  int dbg_stop_ = -1;
};

}  // namespace ni::vision
