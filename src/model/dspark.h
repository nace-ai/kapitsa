// DSpark block drafter for Nemotron 3.5 Lightning (NVIDIA's NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark, its
// safetensors as they are): 6 Qwen3-style layers over the target's hidden states (the outputs of 6 target layers,
// fused by fc into each layer's context K/V, sliding window 1024, attention sinks), one pass over the block [anchor,
// mask x 7] at positions P..P+7, causal within the block; the target's LM head gives each slot's logits, and the
// Markov head (logits_i += W2 W1[token_{i-1}], rank 512) corrects them left to right from the anchor. Kapitsa applies
// the correction to each slot's top-K candidates on the GPU (a greedy walk; the drafts only propose, the target
// verifies every token). Semantics: vLLM's / mlx-vlm's qwen3_dspark (the DFlash backbone with a causal block).
// The context side (capture hooks, ring, commits) is LiliCorr's (lilicorr.h) without its convolutions.
#pragma once
#include "model/lilicorr.h"
#include "model/nh_drafter.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

class DSpark : public NhDrafter {
 public:
  static constexpr int kMaxCand = 32;  // candidates per slot the Markov head rescores (NI_DSPARK_K, default 16)
  DSpark(NemotronH& target, const std::string& dir);
  ~DSpark() override;
  // a DSpark drafter folder for a target of this hidden size (config.json dspark_fixup_head_type markov, model.safetensors)
  static bool is_dspark(const std::string& dir, int hidden);
  static size_t plan_bytes(const std::string& dir, size_t* per_slot);
  const char* name() const override { return "DSpark"; }
  int window() const override { return window_; }

  std::unique_ptr<LiliSeq> new_seq() override;
  void encode_copy(Encoder& e, const LiliSeq& src, LiliSeq& dst) override;
  void begin_prefill(int n, int start) override;
  void encode_prefill_commit(Encoder& e, LiliSeq& d) override;
  void end_prefill() override;
  void arm_verify(int P) override;
  void disarm() override;
  void encode_commit(Encoder& e, LiliSeq& d, int m) override;
  void encode_draft(Encoder& e, Seq& s, LiliSeq& d, int P) override;

 private:
  struct DLayer {
    Buffer attn_norm, ffn_norm, qn, kn, sinks;
    NhMat q, k, v, o, gate, up, down;
  };
  NhMat mat(const std::string& name);                // NVFP4 as stored, BF16 as Q8
  std::vector<float> host(const std::string& name);  // a BF16 / F32 tensor as fp32 on the CPU
  Buffer upload(const void* p, size_t bytes, const char* label);
  Buffer vec(const std::string& name);
  void load();
  void alloc_buffers();
  void enc_commit_rows(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0);
  void lin(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int M, int epi, bool scaled = false);

  NemotronH& tg_;
  std::string path_;
  std::unique_ptr<SafetensorsIndex> st_;  // while loading
  int H_ = 0, NL_ = 0, I_ = 0, hq_ = 0, hkv_ = 0, window_ = 1024, mask_id_ = 0, rank_ = 0, K_ = 16, head_rows_ = 0;
  float eps_ = 1e-6f, theta_ = 10000.0f;
  std::vector<int> taps_;  // target layers whose inputs feed fc (an HF id + 1; the last layer's output: layers)
  std::vector<DLayer> dl_;
  NhMat fc_, w2_;
  Buffer enc_norm_, out_norm_, mask_row_, w1_;  // w1_: the Markov head's embedding, BF16 [V][rank]
  // dynamic verify depth (NI_DSPARK_DYN; 0 = every draft): per-row cost over a cycle's fixed cost. A verify row costs
  // Lightning ~6 of 128 experts more per layer: 0.2 measured best with 0.1-0.3 on the harness's coding prompt and an
  // essay (512 tokens, greedy: 207 / 154 tok/s against 155 / 119 verifying every draft)
  float dyn_ratio_ = 0.2f;
  std::vector<Buffer> owned_;
  size_t weight_bytes_ = 0;

  NhCapture hook_pf_, hook_v_;
  int pf_n_ = 0, pf_lo_ = 0;
  Buffer pfeat_, pfeat_h_, vfeat_;
  Buffer x_, n_, t2_, q_, k_, v_, ao_, gt_, upb_, hfin_, logits_, cv_, ci_, ms_, cand_, clp_, qlp_, picks_;
  Buffer cg_, cgh_, ck_, cv2_, rscale_, xh_;
};

}  // namespace ni
