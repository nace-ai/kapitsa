// Hy4-flash's DFly2 drafter (model.dfly_layers.* in model-dfly.safetensors; as vLLM d2d72635's hy_v4 dfly.py): five
// MLA decoder layers over a block of 6 rows [t0, M, M, M, M, M] at positions a .. a+5, attending to the target's
// latent cache rows of layers 25/29/33/37/41 (every position < a: the target indexer's top-k is every cached position
// up to 2048 keys) plus the block causally, then a sequential head with an autoregressive hidden correction that
// drafts 5 tokens greedily through the target's embedding and LM head. No KV of its own: the block's latent rows are
// written into the target's cache at a .. a+5 (the verify overwrites them), as vLLM does.
// Kernels: src/kernels/hy4/hy4_dfly.metal, plus hy4_ops' norms, adds, SwiGLU and enc_attn_latent.
// Above 2048 cached positions it attends, as the reference does, to the target indexer's selection of layer L_i at
// position a - 1 (In::sel).
#pragma once
#include "loader/safetensors.h"
#include "metal/gpu.h"
#include "model/hy4_ops.h"
#include "model/qwen35.h"

#include <deque>
#include <string>
#include <vector>

namespace ni {

struct Hy4DflyConfig {
  std::vector<int> kv_layers;  // dfly_target_kv_layers: draft layer i reads the target cache of kv_layers[i]
  uint32_t mask_token = 120009;  // dfly_mask_token_id
  uint32_t D = 3072;             // hidden_size
  uint32_t H = 32;               // num_attention_heads (kv_lora 512, nope 192, rope 64, v 256: fixed)
  float eps = 1e-5f;             // rms_norm_eps
  float theta = 1e7f;            // the target's deepseek_sparse_attention rope_theta
  bool rope_bf16 = true;         // as the target's attention layers (hy4::AttnLayer::rope_bf16)
  // the pack's config.json: dfly_* keys at the root, the rest from text_config; fatal on a drafter this class cannot
  // run (context-local window, no hidden correction, a fusion other than slot0, other head dims)
  static Hy4DflyConfig from_json(const std::string& path);
};

class Hy4Dfly {
 public:
  static constexpr int kBlock = 6;  // rows: t0 and 5 mask rows
  static constexpr int kDraft = 5;  // drafted tokens a cycle

  // dir: the folder of model-dfly.safetensors. embed / lm_head: the target's (affine 4- or 8-bit group 64), alive as
  // long as this drafter. Allocates its weights (~1.7 GB, bf16) and activations.
  Hy4Dfly(const std::string& dir, const Hy4DflyConfig& cfg, const QLinear& embed, const QLinear& lm_head);
  ~Hy4Dfly();
  Hy4Dfly(const Hy4Dfly&) = delete;
  Hy4Dfly& operator=(const Hy4Dfly&) = delete;

  const Hy4DflyConfig& config() const { return cfg_; }
  static bool present(const std::string& dir);  // model-dfly.safetensors is in dir

  struct In {
    Buffer h;   // fp32 [D]: the target's final hidden (after hc_head and model.norm) at its last accepted row
    Buffer t0;  // uint32 [1]: the token sampled from that row (the block's first token)
    int a = 0;  // t0's position, the committed length: the block is a .. a+5
    // the target's latent caches of config().kv_layers, in that order: rows < a are read, rows a .. a+5 written;
    // linear caches (no ring) with cap >= a + 6
    std::vector<hy4::LatentCache*> caches;
    // per layer the target indexer's selection row [512] of position a - 1 (hy4::enc_index_pick), when a > kIndexDense;
    // empty: every position < a
    std::vector<const Buffer*> sel;
  };
  // Encodes one draft cycle into a serial encoder (Gpu::begin()): out uint32 [5] = the drafted tokens d1 .. d5 for
  // positions a+1 .. a+5, chosen by argmax on the GPU (nothing is read back). Never inside a grouped encoder.
  void draft(Encoder& e, const In& in, const Buffer& out);

  // Debug and check views of the last cycle (valid after its command buffer completed): z fp32 [6, D] (the final
  // norm's output), logits fp32 [5, V] (the head's step k at row k - 1, pre-temperature)
  const Buffer& z() const { return z_; }
  const Buffer& logits() const { return logits_; }
  uint32_t vocab() const { return V_; }

  std::vector<Buffer> buffers() const;  // weights and activations (make_resident)
  size_t weight_bytes() const { return weight_bytes_; }
  std::vector<std::string> kernel_names() const;

 private:
  struct Lin {  // a dense bf16 matrix [N, K]
    Buffer w;
    uint32_t N = 0, K = 0;
  };
  struct Layer {
    const Tensor *in_norm = nullptr, *post_norm = nullptr, *q_a_norm = nullptr, *kv_a_norm = nullptr,
                 *post_proj_norm = nullptr, *sinks = nullptr, *kv_b = nullptr;
    Lin q_a, q_b, kv_lat, kv_rope, gate, o, mg, mu, md;
    hy4::AttnLayer at;
  };
  Lin lin(const std::string& name, uint32_t r0 = 0, uint32_t n = 0);  // rows [r0, r0 + n) (n 0: all)
  Lin cols(const std::string& name, uint32_t c0, uint32_t n);         // a copy of columns [c0, c0 + n)
  const Tensor* vec(const std::string& name, uint32_t n);
  // y [R, N] = x [R, K] . W^T (+ resid [R, N] when resid); x_ld / y_ld / r_ld: row strides (0: K / N / N)
  void mv(Encoder& e, const Lin& L, const Buffer& x, int R, const Buffer& y, const Buffer* resid = nullptr,
          uint32_t x_ld = 0, uint32_t y_ld = 0, uint32_t r_ld = 0);
  void embed(Encoder& e, const Buffer& ids, uint32_t nids, int R, const Buffer& y);
  void head(Encoder& e, const Buffer& x, const Buffer& y);  // y [V] = lm_head x

  Hy4DflyConfig cfg_;
  const QLinear& embed_;
  const QLinear& lm_head_;
  uint32_t V_ = 0;
  WeightStore ws_;
  std::deque<Tensor> own_;
  std::vector<Buffer> own_bufs_;
  size_t weight_bytes_ = 0;
  std::vector<Layer> L_;
  const Tensor *enorm_ = nullptr, *hnorm_ = nullptr, *fgate_ = nullptr, *final_norm_ = nullptr;
  const Tensor *hc_hnorm_ = nullptr, *hc_enorm_ = nullptr;
  Lin eh_, hc_gh_, hc_ge_, hc_uh_, hc_ue_, hc_down_;
  // activations
  Buffer X_, u_, qa_, q_, lat_, kr_, gt_, att_, y_, g_, up_;
  Buffer cat_, f_, hb_, z_, hn_, Gh_, Uh_, e1_, en_, hg_, hu_, zc_, logits_, attn_scratch_;
};

}  // namespace ni
