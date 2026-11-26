// Inkling-2's image encoder (Thinking Machines' inkling2_vision, model.visual.* of an Inkling-2 checkpoint): the
// processor, the ViT and the merger.
// Processor: transparency composited onto the reference's 8-pixel gray checkerboard, no resizing (the image at its own
// resolution, within --image-max-pixels: an image whose token grid is larger is scaled down first), padded right and below to whole 40-pixel tokens (2 x 2 patches of 20 pixels) with the reference's fill (-1 on
// the 0-255 scale), normalized with mean and std 0.5. Tower: a class token in front of the patches (which keep their
// 2 x 2 token order), the patch embedding + a 32 x 32 position table resampled bilinearly to the patch grid, a
// LayerNorm, then 23 pre-LayerNorm blocks of attention (16 heads of 64, axial 2D RoPE over interleaved pairs, the class
// token at position (0, 0) and the patches at (row + 1, column + 1)) and a GELU MLP, each branch scaled by its learned
// layer scale (ls_1, ls_2). Merger: the 2 x 2 patches of a token side by side (4096 wide) through proj_in, GELU and
// proj_out to the language model's width (2304), one row per token, which Inkling2::enc_forward writes over the
// placeholder tokens' normalized embeddings (as the reference's masked_scatter after embed_norm).
// Weights bf16 on the prefill GEMMs; the small kernels in vit.metal (ivit_*).
#pragma once
#include "vision/encoder.h"
#include "vision/vit_common.h"

#include <string>
#include <utility>
#include <vector>

namespace ni::vision {

struct Ink2ViTConfig {
  int depth = 23, hidden = 1024, heads = 16, inter = 4096, patch = 20, merge = 2, in_ch = 3;
  int side = 32;            // the position table: side x side
  int proj_hidden = 4096;   // the merger's inner width
  bool proj_bias = false;   // the merger's linears have biases
  int out_hidden = 2304;    // the language model's hidden size
  int image_token = 200054; // <|unused_200054|>, the chat template's one token per image
  float ln_eps = 1e-5f;
  float theta = 10000.0f;   // the axial RoPE's base
  // processor
  long max_pixels = 0;      // Kapitsa's cap on the pixels an image keeps (scaled down to it); 0: none (the reference
                            // processor has none)
  float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
  double rescale = 1.0 / 255.0;
  bool do_rescale = true, do_normalize = true;
};

// The size a w x h image is scaled to so that its grid of `token`-pixel tokens (padding included) holds at most
// max_pixels / token^2 tokens (at least one): the largest such size with the aspect kept and neither side enlarged;
// w x h itself when it fits or max_pixels <= 0.
std::pair<int, int> ink2_target_size(int w, int h, int token, long max_pixels);

class Ink2ViT : public ImageEncoder {
 public:
  Ink2ViT(const std::string& dir, const Ink2ViTConfig& c);
  ~Ink2ViT() override;
  static bool supports(const std::string& dir, Ink2ViTConfig* c);  // an Inkling-2 vision tower in dir's files

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

  const Ink2ViTConfig& config() const { return c_; }
  // The processor's pixel_values as the reference returns them: float32 [tokens, merge * patch, merge * patch, 3]
  // (one 40 x 40 pixel block per token in raster order, (row, column, channel) within it, padding included)
  std::vector<float> pixel_values(const Prepared& p) const override;
  // Tests: the hidden state of every row (the class token, then the patches) after `layers` blocks (fp32; 0: after the
  // LayerNorm that follows the embeddings) instead of the embeddings, when set before encode()
  void set_debug_stop(int layers) override { dbg_stop_ = layers; }
  int debug_rows(const Prepared& p) const override { return 1 + patches(p); }
  int debug_width() const override { return c_.hidden; }

 private:
  struct Block {
    Buffer n1w, n1b, n2w, n2b, g1, g2;  // the two LayerNorms, the two layer scales
    VitLin qkv, proj, fc1, fc2;         // q, k and v fused into one [3 * hidden, hidden]
  };
  // the padded canvas of a prepared image, in tokens and in patches
  int grid_rows(const Prepared& p) const { return (p.height + c_.patch * c_.merge - 1) / (c_.patch * c_.merge); }
  int grid_cols(const Prepared& p) const { return (p.width + c_.patch * c_.merge - 1) / (c_.patch * c_.merge); }
  int patches(const Prepared& p) const { return grid_rows(p) * grid_cols(p) * c_.merge * c_.merge; }
  void build_lut(float* lut) const;  // [3][257]: a channel's value after rescale and normalize; [256] the padding

  std::string dir_;
  Ink2ViTConfig c_;
  uint32_t patch_k_ = 0;     // 3 * patch * patch padded to a multiple of 64 (the GEMMs' K step)
  VitWeights w_;             // the vision tensors (model.visual.*): headers while loading, then our own buffers
  bool loaded_ = false;
  VitLin patch_;             // [hidden, patch_k_], (row, column, channel) order; no bias
  Buffer pos_;               // bf16 [side * side, hidden]
  std::vector<float> cls_;   // class_embedding + cls_position_embedding, fp32 [hidden]
  Buffer lnw_, lnb_;         // ln_pre
  std::vector<Block> blocks_;
  VitLin m1_, m2_;           // proj_in [proj_hidden, merge^2 * hidden], proj_out [out_hidden, proj_hidden]
  int dbg_stop_ = -1;
};

}  // namespace ni::vision
