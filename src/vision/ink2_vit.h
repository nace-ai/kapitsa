// Thinking Machines Inkling-2 Vision Encoder (model_type "inkling2_vision"):
// Processor: pads arbitrary RGB images bottom-right with -1.0 (pre-rescale) to multiples of patch * merge (40),
// rescales 1/255, and normalizes with mean 0.5 / std 0.5 into 40x40x3 tiles.
// Tower: 23 ViT layers (hidden 1024, 16 heads x 64, GELU MLP inter 4096) with prepended CLS token, pre-LayerNorm
// before layer 0, 2D axial pairwise RoPE (1-indexed patch coordinates, width pairs first), and post-branch LayerScale,
// followed by dropping CLS and 2x2 merge projection (Linear 4096->4096 -> GELU -> Linear 4096->2304).
#pragma once
#include "vision/encoder.h"
#include "vision/vit_common.h"

#include <string>
#include <vector>

namespace ni::vision {

constexpr long kInk2DefaultMaxPixels = 1600000;

struct Ink2ViTConfig {
  int depth = 23, hidden = 1024, heads = 16, inter = 4096;
  int patch = 20, merge = 2, side = 32;
  int proj_hidden = 4096, out_hidden = 2304;
  bool proj_bias = false;
  float ln_eps = 1e-5f;
  float rope_theta = 10000.0f;
  int image_token = 200054;
  // processor
  long max_pixels = 0;  // optional server cap; 0 preserves native resolution (matches reference processor)
  double rescale = 1.0 / 255.0;
  double pad_value = -1.0;
  float mean[3] = {0.5f, 0.5f, 0.5f};
  float stdv[3] = {0.5f, 0.5f, 0.5f};
};

class Ink2ViT : public ImageEncoder {
 public:
  Ink2ViT(const std::string& dir, const Ink2ViTConfig& c);
  ~Ink2ViT() override;
  static bool supports(const std::string& dir, Ink2ViTConfig* c);

  std::string name() const override { return "Inkling-2 ViT"; }
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

  std::vector<float> pixel_values(const Prepared& p) const override;
  void set_debug_stop(int layers) override { dbg_stop_ = layers; }
  int debug_rows(const Prepared& p) const override {
    return 1 + p.tokens * c_.merge * c_.merge;
  }
  int debug_width() const override { return c_.hidden; }
  const Ink2ViTConfig& config() const { return c_; }

 private:
  struct Block {
    Buffer n1w, n1b, n2w, n2b, ls1, ls2;
    VitLin qkv, proj, fc1, fc2;
  };

  std::string dir_;
  Ink2ViTConfig c_;
  uint32_t patch_k_pad_ = 0;
  uint32_t inter_pad_ = 0;
  uint32_t proj_pad_ = 0;
  VitWeights w_;
  bool loaded_ = false;
  VitLin patch_;
  Buffer pos_, cls_, cls_pos_, ln_pre_w_, ln_pre_b_, zero_h4_;
  std::vector<Block> blocks_;
  VitLin m1_, m2_;
  int dbg_stop_ = -1;
};

}  // namespace ni::vision
