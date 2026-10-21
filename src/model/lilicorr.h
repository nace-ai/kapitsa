// LiLiCorr block drafter for Nemotron 3.5 Bolt (NVIDIA's NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, its
// safetensors as they are): a 5-layer DFlash2-style drafter that reads the target's hidden states (the outputs of
// 6 target layers, fused into its per-layer context K/V, sliding window 1024, attention sinks) and drafts a block
// of 7 tokens in one pass (anchor + 7 mask rows, bidirectional within the block); its LiLiCorr head scores the
// transitions between the top-8 candidates of consecutive slots, and a greedy walk over that lattice picks the
// draft. Semantics: NVIDIA Model-Optimizer's modeling_lilicorr.py / hf_dflash.py. Drafts are verified exactly by
// the target (NemotronH::encode_verify + encode_accept).
#pragma once
#include "model/nemotron_h.h"
#include "model/nh_drafter.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

struct LiliConfig {
  int H = 0, NL = 0, I = 0, hq = 0, hkv = 0, hd = 0, block = 8, window = 1024, conv_k = 2, conv_gs = 16;
  int mask_id = 0;
  float eps = 1e-6f, theta = 10000.0f;
  std::vector<int> target_layers;  // target layers whose inputs feed fc, in order
  // LiLiCorr head
  int lh = 0, topk = 0, heads = 0, lnl = 0, factor = 0, mlp = 0;
  float vec_eps = 1e-4f, logit_scale = 8.0f;
};

// Per-sequence drafter state.
struct LiliSeq {
  Buffer ring_k, ring_v;  // half [NL][hkv][window][hd]; slot = position % window
  Buffer anchor;          // fp32 [kAnchors][H]: projected context features; slot = position % kAnchors
  int ctx = 0;            // committed context positions [0, ctx)
  int hi = 0;             // one past the highest position written since the reset: the ring holds [hi - window, hi)
  // Batched decode (NhDrafter::stage_init): the target captures of rows not yet committed, positions ctx ..
  // ctx + staged - 1, fp32 [8][n_slots * H]; the hook a batched verify writes them with (by row)
  Buffer stage;
  int staged = 0;
  NhCapture stage_hook;
};

class LiliCorr : public NhDrafter {
 public:
  static constexpr int kB = 8;         // block: anchor + 7 proposals
  static constexpr int kAnchors = 64;  // anchor slots: a restored checkpoint up to 63 positions back still has its own
  // draft head rows by default (the first ids: special tokens and the most frequent BPE tokens; NI_LILI_HEAD)
  static constexpr int kHeadRows = 131072;
  // dir: the drafter's folder (config.json, model.safetensors)
  LiliCorr(NemotronH& target, const std::string& dir);
  ~LiliCorr();
  const LiliConfig& config() const { return cfg_; }
  // the drafter of a Nemotron model folder: its dflash/ (the catalog's companion folder) or
  // lilicorr/ subfolder when that holds a LiLiCorr drafter of the model's hidden size; "" if none
  static std::string find(const std::string& model_dir);
  // memory: weights as loaded and buffers; *per_slot: one sequence's ring (a slot holds three: live + 2 snapshots)
  static size_t plan_bytes(const std::string& dir, size_t* per_slot);

  const char* name() const override { return "LiLiCorr"; }
  int window() const override { return cfg_.window; }
  std::unique_ptr<LiliSeq> new_seq() override;
  static void reset(LiliSeq& d) { d.ctx = d.hi = 0; }
  // prefix reuse: context back to pos (ring entries of positions >= pos are dead; the draft reads
  // [max(pos - window, hi - window), pos))
  static void rewind(LiliSeq& d, int pos) { d.ctx = pos; }
  void encode_copy(Encoder& e, const LiliSeq& src, LiliSeq& dst) override;

  // ---- context ----
  // Prompt rows [start, n): begin_prefill arms the target capture of the rows the window still sees,
  // [max(start, n - window), n); the target's prefill passes capture them; encode_prefill_commit projects them
  // into d (d.ctx = n); end_prefill disarms.
  void begin_prefill(int n, int start) override;
  void encode_prefill_commit(Encoder& e, LiliSeq& d) override;
  void end_prefill() override;
  // Decode: arm_verify(P) before the target's verify at P (captures its rows); encode_commit(e, d, m) commits verify
  // rows 0..m-1 at positions d.ctx .. d.ctx + m - 1.
  void arm_verify(int P) override;
  void disarm() override;
  void encode_commit(Encoder& e, LiliSeq& d, int m) override;

  // ---- drafting: block [tokens[P], mask x 7] at positions P..P+7 (d.ctx == P) -> tokens[P+1 .. P+7] ----
  void encode_draft(Encoder& e, Seq& s, LiliSeq& d, int P) override;
  void encode_draft_tree(Encoder& e, Seq& s, LiliSeq& d, int P, const Buffer& kfp) override;
  const int* cand_ids() const { return cand_.as<int>(); }  // [7][8] of the last pass (tests)
  void debug_feature_stats() const;                        // magnitudes of the last prompt's captured features

 private:
  struct DLayer {
    Buffer attn_norm, ffn_norm, qn, kn, sinks, ac_base, fc_base;
    NhMat ac_proj, fc_proj, q, k, v, o, gate, up, down;
  };
  struct HLayer {
    Buffer attn_norm, mlp_norm, in_b, out_b, mlp0_b, mlp2_b;
    NhMat in_proj, out_proj, mlp0, mlp2;
  };
  void load();
  NhMat mat(const std::string& name);                   // NVFP4 as stored, BF16 as Q8
  std::vector<float> host(const std::string& name);     // a BF16 / F32 tensor as fp32 on the CPU
  Buffer upload(const std::vector<float>& v, const char* label);
  Buffer vec(const std::string& name);                  // host() in a GPU buffer
  void alloc_buffers();
  const NhCapture& verify_hook() const override { return hook_v_; }
  // fc + norm of feature rows [0, R) of `feat` at positions pos0.. into the ring and the anchor slots
  void enc_commit_rows(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0) override;
  void enc_bias(Encoder& e, const Buffer& x, const Buffer& b, int rows, int D, int act);
  // y = W x for M fp32 rows: one row on the matvec, more on the GEMM (x converted to half; scaled: per-row scale
  // for inputs that may leave half's range)
  void lin(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int M, int epi, bool scaled = false);
  // the drafting pass up to the LiLiCorr head's vectors (cand_, clp_, inv_, outv_, aout_)
  void encode_pass(Encoder& e, Seq& s, LiliSeq& d, int P);
  // y = W xh for M <= 8 half rows: nh_xq8_v8 for a matrix repacked for it (xq8_), else enc_gemm
  void gemm(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi);
  bool is_xq8(const NhMat& W) const;
  void make_xq8();  // the drafting pass's Q8 matrices repacked for nh_xq8_v8 (k / v: copies, the commit's GEMMs
                    // of a prompt read the Q8 layout)

  NemotronH& tg_;
  std::string path_;
  int head_rows_ = 0;  // draft head: the first rows of the drafter's own LM head (own_head_) or the target's
  bool has_own_head_ = false;
  NhMat own_head_;
  // dynamic verify depth: per-row cost / fixed cost of a cycle (NI_LILI_DYN; 0 = every draft). 0.04 measured best on the
  // harness's code prompt, 512 tokens: greedy 53.5 tok/s against 50.7 (0) and 52.3 (0.08); T = 0.6 52.2 against 47.3.
  // Super-VL (a row adds up to 22 experts per layer), 512 tokens, two rounds: 0.04 best or within 1% everywhere (C++
  // greedy 89.0 / 92.8 against 84.1 / 89.1 at 0.08 and 82.5 / 85.9 at 0.15)
  float dyn_ratio_ = 0.04f;
  uint32_t dyn_mode_ = 0;   // its acceptance estimate (NI_LILI_DYN_P): 0 drafter marginal, 1 LiLiCorr walk, 2 both
  Buffer picks_;            // the walk's chosen candidate per slot
  Buffer tlq_;              // KFlash: [S][K][K] log q of each candidate after each candidate of the slot before
  std::unique_ptr<SafetensorsIndex> st_;  // while loading
  LiliConfig cfg_;
  std::vector<DLayer> dl_;
  // the M5: the drafting pass's Q8 products of <= 8 rows on nh_xq8_v8 (NI_LILI_XQ8=0: nh_v8_q8); kx_ / vx_ the
  // repacked copies of the layers' k / v
  bool xq8_ = false;
  std::vector<NhMat> kx_, vx_;
  std::vector<const void*> xq8_mats_;
  std::vector<HLayer> hl_;
  NhMat fc_, tok_proj_, pass_proj_, ctx_proj_, mlp3_, factor_in_, out_head_, in_head_, anchor_head_;
  Buffer enc_norm_, out_norm_, tok_b_, pass_b_, ctx_b_, feat_ln_w_, feat_ln_b_, mlp1_w_, mlp1_b_, mlp3_b_, slot_emb_,
      rank_emb_, attn_bias_, lout_norm_, anchor_norm_, factor_b_, out_b_, in_b_, anchor_b_;
  std::vector<Buffer> owned_;
  size_t weight_bytes_ = 0;

  NhCapture hook_pf_, hook_v_;
  int pf_n_ = 0, pf_lo_ = 0;
  Buffer pfeat_, pfeat_h_, vfeat_;
  // pass buffers
  Buffer btok_, x_, n_, dyn_, t1_, t2_, q_, k_, v_, ao_, gt_, upb_, hfin_, logits_, cv_, ci_, ms_, cand_, clp_;
  Buffer ce_, ts_, ps_, h1_, fs_, hx_, hxn_, qkv_, hao_, ho_, hm0_, hm2_, as_, an_, fi_, fct_, outv_, inv_, aout_, scores_;
  // commit buffers (up to window rows)
  Buffer cg_, cgh_, ck_, cv2_, rscale_, xh_;
};

}  // namespace ni
