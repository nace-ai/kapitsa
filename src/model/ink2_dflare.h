// Bundled DFlare / DSpark block drafter for Thinking Machines Inkling-2-Mini (mtp.safetensors).
#pragma once

#include "loader/safetensors.h"
#include "model/inkling2.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

struct Ink2DFlareSeq {
  std::vector<Buffer> kc, vc;  // per draft layer (3): half [ctx_cap, Hkv * D]
  Buffer rows_f;               // float [H]: last committed markov_head context mix
  int ctx_len = 0;             // committed context positions in ring (clamped to ctx_cap)
};

class Ink2DFlare {
 public:
  static constexpr int kMaxDrafts = 7;
  static constexpr int kCandK = 32;

  static bool is_available(const std::string& dir);
  Ink2DFlare(Inkling2& target, const std::string& dir);
  ~Ink2DFlare();

  int max_drafts() const { return block_size_ - 1; }
  int block_size() const { return block_size_; }
  int ctx_cap() const { return ctx_cap_; }

  std::unique_ptr<Ink2DFlareSeq> new_seq();
  void reset(Ink2DFlareSeq& ds);

  // Commit `n_rows` target taps ([4, taps_stride_rows, H] float) starting at sequence position `pos0`.
  // `taps_stride_rows` defaults to `n_rows` when 0 (pass the verify batch size T when committing a prefix).
  // Only the newest min(n_rows, ctx_cap_) rows are projected into the rolling ring.
  void encode_commit(Encoder& e, Ink2DFlareSeq& ds, const Buffer& taps, int n_rows, int pos0,
                     int taps_stride_rows = 0);

  // Draft `depth` tokens (1..7) into `s.tok[pos + 1 .. pos + depth]` conditioned on `s.tok[pos]`
  // (the anchor token at `pos`) and the committed context ring `[pos - ctx_len .. pos - 1]`.
  void encode_draft(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int depth);

 private:
  struct Q8Mat {
    Buffer q, s;
    uint32_t N = 0, K = 0;
  };
  struct Layer {
    Buffer dflare_norm, input_norm, post_norm;
    Buffer q_norm, k_norm, k_norm_ctx;
    Q8Mat q_proj, k_proj, v_proj, o_proj;
    Q8Mat k_proj_ctx, v_proj_ctx;
    Q8Mat gate_proj, up_proj, down_proj;
  };

  void load();
  void alloc_buffers();
  void enc_q8(Encoder& e, const Q8Mat& w, const Buffer& x, const Buffer& y, int M, bool add = false);
  void enc_rmsnorm(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& out, int T);

  Inkling2& tg_;
  std::string dir_;
  int H_ = 2304, NL_ = 3, hq_ = 24, hkv_ = 8, D_ = 128, I_ = 9216;
  int block_size_ = 8, ctx_cap_ = 1023, rank_ = 256;
  float eps_ = 1e-6f, theta_ = 1000000.0f;
  float alpha_[4][4] = {};  // softmax(fuse_weight)

  std::vector<Layer> layers_;
  Buffer mask_embed_, norm_w_;
  Buffer mh_dflare_norm_, mh_w1_, mh_rnn_w_, mh_rnn_b_;
  Buffer fold_q_, fold_s_, fold_b_;  // folded unembed @ hidden_out (Q8_0 [V, 256] + F32 [V])
  std::vector<Buffer> owned_bufs_;

  // Reusable scratch buffers
  Buffer ctx_h_, ctx_k_, ctx_v_;              // commit scratch (up to 64 rows per batch)
  Buffer bx_, bn_, bq_, bk_, bv_, batt_;      // block scratch (B <= 8)
  Buffer bgate_, bup_, bact_, draft_h_;       // MLP + final hidden (B <= 8)
  Buffer base_logits_, pick_v_, pick_i_;
  Buffer mh_state_, mh_rnn_in_, mh_latent_;
};

}  // namespace ni
