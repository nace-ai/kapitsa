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
  // Q8 codes [rows][K] in place into nh_xq8_v8's order (each 16-element group: elements 0, 1, 8, 9, 2, 3, 10, 11, ...)
  static void repack_xq8(uint8_t* q, size_t rows, uint32_t K);
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
  void encode_draft_tree(Encoder& e, Seq& s, LiliSeq& d, int P, const Buffer& kfp) override;
  bool chains() const override { return chain_; }
  // the last drafting pass's candidates [7][K] (ids, log-probs), the Markov head's corrected log-probs [7][K] and
  // picks [7] (measurements; once its command buffer completed)
  int cand_k() const { return K_; }
  void debug_read(std::vector<int>& cand, std::vector<float>& clp, std::vector<float>& qlp, std::vector<int>& picks) const;
  void encode_draft_chained(Encoder& e, Seq& s, LiliSeq& d, int P) override;

 private:
  struct DLayer {
    Buffer attn_norm, ffn_norm, qn, kn, sinks;
    // qkv: q|k|v's rows in one matrix (Q8), kv (k|v, the commit's) a view of it; gu: gate|up's rows (NVFP4, the
    // tensor scales gs_g, gs_u applied by nh_dsp_swiglu_half)
    // (xq8_: qkv and o's codes repacked for nh_xq8_v8, kv_std k|v's in the Q8 layout for the other row counts)
    NhMat qkv, kv, kv_std, o, gu, down;
    float gs_g = 1.0f, gs_u = 1.0f;
  };
  NhMat mat(const std::string& name);                // NVFP4 as stored, BF16 as Q8
  // the matrices' rows in one (the same format and K); each one's tensor scale into gs (the result's: gs[0] if shared,
  // else 1)
  NhMat mat_cat(const std::vector<std::string>& names, std::vector<float>& gs, bool shared = false);
  static NhMat rows(const NhMat& m, uint32_t r0, uint32_t n);  // rows [r0, r0 + n) of m, a view
  std::vector<float> host(const std::string& name);  // a BF16 / F32 tensor as fp32 on the CPU
  Buffer upload(const void* p, size_t bytes, const char* label);
  Buffer vec(const std::string& name);
  void load();
  void alloc_buffers();
  const NhCapture& verify_hook() const override { return hook_v_; }
  void enc_commit_rows(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0) override;
  // the commit (chained: the ring takes the first m rows, m from cyc_) and the drafting pass (chained: the positions
  // from cyc_, written on the GPU by nh_dsp_cycle)
  void enc_commit(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0, bool chained);
  void enc_draft(Encoder& e, Seq& s, LiliSeq& d, int P, bool chained, bool walk);
  void lin(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int M, int epi, bool scaled = false);
  // the drafting pass up to each slot's top-K candidates (cand_, clp_)
  void encode_pass(Encoder& e, Seq& s, LiliSeq& d, int P);
  // y = W x for M rows of half X: a repacked Q8 matrix (xq8_) on nh_xq8_v8 (M <= 8), else enc_gemm
  void gemm_q8(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi);
  void norm_half(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& y, int R);  // RMSNorm rows into half

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
  // essay (512 tokens, greedy: 207 / 154 tok/s against 155 / 119 verifying every draft). On the M5 0.3: replaying
  // 4,227 full-depth T = 0.6 cycles (4 prompts x 6 seeds) with the verify's measured cost per depth (7.35 ms for the
  // anchor's row + ~1.0 per draft) puts 0.3 0.7% above 0.2 (greedy 1.2%), better on each prompt; the head's own
  // probabilities are overconfident, which a larger ratio offsets (calibrating them per slot gained < 0.5% held out)
  float dyn_ratio_ = 0.2f;
  int fixed_depth_ = -1;  // NI_DSPARK_DEPTH=d: verify d drafts every cycle (measurements)
  // the attention projections on nh_xq8_v8 (default on the M5, where it was measured; NI_DSPARK_XQ8=0/1)
  bool xq8_ = false;
  bool chain_ = true;  // double-buffered decode (NI_DSPARK_CHAIN=0: the drafting half after the host reads each verify)
  std::vector<Buffer> owned_;
  size_t weight_bytes_ = 0;

  NhCapture hook_pf_, hook_v_;
  int pf_n_ = 0, pf_lo_ = 0;
  Buffer pfeat_, pfeat_h_, vfeat_;
  Buffer x_, t2_, q_, ao_, gt_, logits_, cv_, ci_, ms_, cand_, clp_, qlp_, picks_;
  Buffer tlq_;  // KFlash: [S][K][K] log q of each candidate after each candidate of the slot before (nh_dsp_table)
  Buffer cg_, cgh_, ck_, rscale_, xh_;
  Buffer cyc_;   // a chained drafting half's parameters (nh_dsp_cycle)
  Buffer rope_;  // the block's RoPE table (nh_dsp_rope_tab)
  Buffer walk_d_;  // the Markov walk's dots [7][K][K] (nh_dsp_walk_dots)
};

}  // namespace ni
