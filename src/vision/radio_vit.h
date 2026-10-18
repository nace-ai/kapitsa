// Nemotron 3.5 Bolt's image encoder (NVIDIA's NemotronH Omni): the processor, C-RADIO ViT-H/16 and the projector.
// Processor: one tile per image whose 16-pixel patch grid holds between 1024 and 13312 patches (aspect kept, each side
// even), resized as torch's antialiased bicubic interpolate in float32, normalized with CLIP's mean and std. Tower: 32
// blocks of LayerNorm + attention (16 heads of 80) + GELU MLP over 10 cls / register tokens and the patches, whose
// position embeddings come from a 128 x 128 table resized to the grid's longer side and cropped (CPE). Then the patches
// in 2x2 groups (pixel shuffle) through mlp1 (RMSNorm, linear, ReLU squared, linear) to the language model's width.
// Weights bf16 on the prefill GEMMs; the small kernels in vit.metal (radio_*).
#pragma once
#include "vision/encoder.h"
#include "vision/vit_common.h"

#include <string>
#include <utility>
#include <vector>

namespace ni::vision {

struct RadioViTConfig {
  int depth = 32, hidden = 1280, heads = 16, inter = 5120, patch = 16;
  int prefix = 10;          // cls + register tokens before the patches
  int side = 128;           // the position table: side x side
  int proj_hidden = 20480;  // mlp1's inner width
  int out_hidden = 4096;    // the language model's hidden size
  int image_token = 18;     // <image>
  std::string image_start = "<img>", image_end = "</img>";
  float ln_eps = 1e-6f, rms_eps = 1e-5f;
  // processor
  int min_patches = 1024, max_patches = 13312;
  long max_pixels = 0;      // a cap on the pixels (patches * 256) an image keeps; 0: none
  float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f}, stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
};

// The processor's patch grid (columns, rows; both even) for a w x h image and a budget of patches (NVIDIA's
// _compute_target_patches, Python's rounding).
std::pair<int, int> radio_target_patches(int w, int h, int patch, int budget, int min_patches);
// torch.nn.functional.interpolate(mode="bicubic", align_corners=False, antialias=True) of 8-bit RGB rows taken as
// float32 values 0-255, as torch computes it in float32 (width pass, then height; a side that keeps its size is not
// resampled): channel planes [3][oh][ow].
std::vector<float> torch_resize_bicubic_aa(const std::vector<uint8_t>& rgb, int w, int h, int ow, int oh);

class RadioViT : public ImageEncoder {
 public:
  RadioViT(const std::string& dir, const RadioViTConfig& c);
  ~RadioViT() override;
  static bool supports(const std::string& dir, RadioViTConfig* c);  // a NemotronH Omni RADIO tower in dir's files

  std::string name() const override { return "RADIO ViT-H/16"; }
  std::string describe() const override;
  int placeholder() const override { return c_.image_token; }
  std::string image_start() const override { return c_.image_start; }
  std::string image_end() const override { return c_.image_end; }
  int out_dim() const override { return c_.out_hidden; }
  Prepared prepare(const Image& im) const override;
  size_t weight_bytes() const override;
  size_t work_bytes(const Prepared& p) const override;
  bool loaded() const override { return loaded_; }
  void load() override;
  void unload() override;
  void encode(const Prepared& p, const Buffer& out) override;

  // the processor's pixel_values: float32 [3][height][width], normalized
  std::vector<float> pixel_values(const Prepared& p) const override;
  // tests: the hidden state of every row (cls / register tokens, then the patches) after `layers` blocks
  void set_debug_stop(int layers) override { dbg_stop_ = layers; }
  int debug_rows(const Prepared& p) const override {
    return c_.prefix + (p.height / c_.patch) * (p.width / c_.patch);
  }
  int debug_width() const override { return c_.hidden; }
  const RadioViTConfig& config() const { return c_; }

 private:
  struct Block {
    Buffer n1w, n1b, n2w, n2b;
    VitLin qkv, proj, fc1, fc2;
  };
  int patch_budget() const;

  std::string dir_;
  RadioViTConfig c_;
  VitWeights tw_, pw_;  // the tower's tensors (vision_model.radio_model.model.*), the projector's (mlp1.*)
  bool loaded_ = false;
  VitLin patch_;        // [hidden, 3 * patch * patch], (channel, row, column) order; no bias
  Buffer pos_, cls_;    // bf16 [side * side, hidden], [prefix, hidden]
  std::vector<Block> blocks_;
  Buffer mnorm_;        // mlp1's RMSNorm weight [4 * hidden]
  VitLin m1_, m2_;      // no biases (zeros)
  int dbg_stop_ = -1;
};

}  // namespace ni::vision
