// Recursive Flint (FlintForCausalLM, model_type "flint"): a decoder-only MoE for coding agents, 39.8 B parameters, ~2.5 B
// active per token. 49 layers: the first two with a dense SwiGLU MLP, the others a MoE (256 routed experts, the top 6 by
// sigmoid score with their scores renormalized, plus an ungated shared expert). Attention: 32 query and 4 KV heads of 128,
// per-head q / k RMSNorm, a learned sink logit per head; sliding-window layers (513 positions, a row's own included) with
// RoPE, and full-attention layers without positional encoding (the first two, every fourth after them, the last).
// Checkpoints: MLX safetensors, affine 4-bit group-64 projections and experts, an 8-bit embedding, the router in bf16
// (Recursive's flint-int4); or a GGUF import (`kapitsa import`, src/gguf/import.cpp: unsloth's flint-UD-IQ2_XXS) with
// the routed experts in llama.cpp's IQ blocks as stored, run on src/kernels/wfmt_iq.metal's kernels, and the rest in
// 4-bit group 64. The math is Recursive's (rec's model.py, recursive-oss/rec).
// Sliding-window layers keep a ring of window + 8 positions (Dory's layout: a ring cannot go back further than its 8
// spare rows, so checkpoints hold copies of the rings); full-attention layers a cache that grows with the conversation.
//
// The same backend runs a dense Llama (LlamaForCausalLM, model_type "llama": Nanbeige4.3-33B) as a Flint without its
// extras (FlintConfig::llama): every layer full attention with RoPE (NeoX halves, as Flint's sliding-window layers) and
// a dense SwiGLU MLP, no q / k norms (flint_prep_nonorm) and no sinks (a sink of -inf: the softmax without one, exactly).
// Its checkpoints: MLX affine 4-bit or 8-bit group 64 (mlx_lm.convert), or the bf16 safetensors as published.
#pragma once
#include "loader/safetensors.h"
#include "model/model.h"

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ni {

struct FlintConfig {
  int hidden = 0, vocab = 0, layers = 0, hq = 0, hkv = 0, hd = 0;
  int ffn = 0;                          // the dense layers' intermediate size
  int moe_I = 0, shared_I = 0, experts = 0, top_k = 0;
  int window = 0;                       // sliding window, a row's own position included
  double theta = 1e4;
  float eps = 1e-6f;
  std::vector<char> kind;               // per layer: 'S' sliding-window attention, 'F' full attention
  std::vector<bool> rope;               // per layer: RoPE on q and k
  std::vector<bool> dense;              // per layer: a dense MLP (else the MoE)
  std::vector<int> eos;
  int bos = -1;
  int bits = 4, gs = 64;                // the projections' and experts' format
  int embed_bits = 8;
  std::map<std::string, uint32_t> iq;   // modules kept in llama.cpp's IQ blocks (GGUF imports): their GGUF type
  bool llama = false;                   // a dense Llama (model_type "llama"): no q / k norms, no sinks, no experts
};

// An MLX linear (or a stack of E of them): codes (uint32 words or bytes, the same little-endian stream), bf16 scale and
// bias per group of gs inputs; bits 16: bf16 weights (s, b null)
struct FlintMat {
  const Tensor* w = nullptr;
  const Tensor* s = nullptr;
  const Tensor* b = nullptr;
  uint32_t N = 0, K = 0, E = 1;
  int bits = 4, gs = 64;
  // routed experts in llama.cpp's IQ blocks: their GGUF type (gguf::iq_gpu_type's; s, b null), w the stacked [E, N, K]
  // matrix in the kernels' layout with its sections at these byte offsets (Q at 0)
  uint32_t iq = 0;
  uint32_t iq_x1 = 0, iq_x2 = 0, iq_x3 = 0, iq_d = 0;
  explicit operator bool() const { return w != nullptr; }
  size_t row_bytes() const;
};

struct FlintLayer {
  char kind = 'S';
  bool rope = true, dense = false;
  const Tensor *in_norm = nullptr, *post_norm = nullptr, *qn = nullptr, *kn = nullptr, *sinks = nullptr;
  FlintMat q, k, v, o;
  FlintMat mg, mu, md;      // dense MLP
  FlintMat router;          // bf16 [E, hidden]
  FlintMat eg, eu, ed;      // routed experts, stacked [E, N, K]
  FlintMat sg, su, sd;      // shared expert
};

class Flint : public Model {
 public:
  static constexpr int kMaxRows = 8;    // rows of a decode / verify pass
  static constexpr int kRingSpare = 8;  // a sliding-window ring's positions beyond the window

  explicit Flint(const std::string& path, int prefill_chunk = 2048);
  ~Flint() override;
  static bool is_model(const std::string& dir);  // config.json's model_type is "flint" (or "llama": a dense Llama)
  static FlintConfig read_config(const std::string& dir);
  // what in a dense Llama's config.json Kapitsa does not run ("" when it runs it): read_config's refusals, for
  // `kapitsa pull` to give before any download
  static std::string llama_unsupported(const std::string& config_json);
  // what in a dense Llama's chat template (chat_template.jinja, else tokenizer_config.json's chat_template) Kapitsa does
  // not run ("" when it is ChatML, as Nanbeige4.3's): the engine writes a Llama's turns in that markup
  static std::string llama_template_unsupported(const std::string& chat_template);
  static MemPlan plan(const std::string& path, int prefill_chunk = 2048);
  // KV cache bytes per position (full-attention layers) and per conversation (sliding-window rings)
  static size_t kv_per_pos(const FlintConfig& c);
  static size_t ring_bytes(const FlintConfig& c);

  const FlintConfig& config() const { return cfg_; }
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }
  std::string formats_summary() const;

  // ---- Model ----
  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override { return kv_per_pos(cfg_); }
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override;
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  int prefill_chunk() const override { return tmax_; }
  bool can_snapshot() const override { return false; }
  // SSD prefix cache: the full-attention caches as rows, the sliding-window rings whole (a checkpoint's: their copies)
  std::string disk_format() const override;
  void disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const override;
  void disk_state(const Seq& s, const Checkpoint* c, std::vector<Buffer>& out) const override;

  // ---- speculative decoding ----
  // Rows [pos, pos + R) (R <= kMaxRows) as one pass, each row seeing the rows before it as decoding them one by one
  // would: their next-token logits in logits() rows [0, R)
  void encode_verify(Encoder& e, Seq& s, int pos, int R);
  // After a verify: each row's token sampled with sp (step = its position, as plain decoding samples it), the drafts
  // tokens[pos + 1 .. pos + K] accepted while they agree, tokens[pos + 1 + a] = samples[a]; accepted() = a once the
  // command buffer has completed
  void encode_accept(Encoder& e, Seq& s, int pos, int R, int K, const SamplingParams& sp);
  int accepted() const { return acc_.as<int>()[0]; }

  // ---- tools ----
  float* logits() { return logits_.as<float>(); }
  // teacher-forced scoring: tokens [pos0, pos0 + T) as one pass (T <= kMaxRows: the decode rows' kernels, else the
  // prefill's), every row's next-token logits into out [T][vocab]
  void encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out);

 private:
  void load(int prefill_chunk);
  void alloc_buffers();
  void alloc_prefill();
  bool ensure_rings(Seq& s);
  int cache_rows(const Seq& s, int layer) const { return cfg_.kind[size_t(layer)] == 'S' ? ring_rows_ : s.cap; }

  // passes
  void enc_rows(Encoder& e, Seq& s, int pos0, int R);           // R <= kMaxRows rows; final hidden rows in x_
  void enc_layer_rows(Encoder& e, Seq& s, int li, int pos0, int R);
  void enc_prefill_rows(Encoder& e, Seq& s, int pos0, int T);   // GEMM pass; final hidden rows in px_
  void enc_layer_prefill(Encoder& e, Seq& s, int li, int pos0, int T);
  void enc_moe_rows(Encoder& e, const FlintLayer& L, int R);
  void enc_moe_prefill(Encoder& e, const FlintLayer& L, int T);
  // IQ experts (src/kernels/wfmt_iq.metal): loading, and the MoE passes after the router's top-k
  void load_iq_experts();
  void enc_moe_iq_rows(Encoder& e, const FlintLayer& L, int R);
  void enc_moe_iq_prefill(Encoder& e, const FlintLayer& L, int T);
  // SwiGLU of fp32 gate / up rows into half rows scaled where they would overflow (mode: where the scale goes, as
  // flint_swiglu_scaled's)
  void enc_swiglu(Encoder& e, const Buffer& g, const Buffer& u, const Buffer& h, int rows, uint32_t I, uint32_t mode,
                  const Buffer* sorted = nullptr);
  void enc_head(Encoder& e, const Buffer& x, int R, const Buffer& out, size_t ld);
  void enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                  uint32_t step);
  void enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out);
  // up to 4 matrices sharing x (R <= kMaxRows rows, 4/8-bit group 64), an RMSNorm of x folded in; gu: segments 0 / 1
  // are a SwiGLU gate / up (y = silu(gate) up into segment 0's y)
  struct Seg {
    const FlintMat* M;
    Buffer y;
    bool add = false;
  };
  void enc_mv(Encoder& e, const std::vector<Seg>& segs, const Buffer& x, int R, const Tensor* norm, uint32_t y_ld = 0,
              bool gu = false);
  // enc_mv's bf16 form (a dense Llama's bf16 checkpoint): wfmt.metal's one-row kernels, row by row
  void enc_mv_bf16(Encoder& e, const std::vector<Seg>& segs, const Buffer& x, int R, const Tensor* norm, uint32_t y_ld,
                   bool gu);
  // GEMM of M half rows: y [M][y_ld] (+)= xh [M][K] W^T
  void enc_gemm(Encoder& e, const FlintMat& W, const Buffer& xh, const Buffer& y, int M, bool add, uint32_t y_ld = 0);
  void enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T);

  std::string path_;
  FlintConfig cfg_;
  WeightStore ws_;
  std::vector<FlintLayer> layers_;
  FlintMat embed_, head_;
  std::deque<Tensor> iq_tensors_;  // the IQ experts in the kernels' layout (iq_bufs_), as the FlintMats point at them
  std::vector<Buffer> iq_bufs_;
  const Tensor* final_norm_ = nullptr;
  Buffer nosink_;                // a dense Llama's sinks: bf16 [hq] of -inf (each head's softmax without a sink)
  const Buffer& sinks(const FlintLayer& L) const { return L.sinks ? L.sinks->buf : nosink_; }
  Buffer inv_freq_;              // fp32 [hd / 2]: RoPE's inverse frequencies
  // (cos, sin) per frequency at one position, as flint_attn_d2 takes them inline (the last position asked for)
  const float* rope_cs(int pos);
  std::vector<float> rope_cs_;
  int rope_pos_ = -1;
  int ring_rows_ = 0;
  int tmax_ = 2048;
  size_t mem_budget_ = 0;
  bool qmt_ = false;             // dense 4-bit prefill GEMMs on Metal 4's tensor_ops (M5)
  bool split_ = false;           // the decode kernels' merges in a dispatch of their own (split_merges)

  // decode / verify activations (kMaxRows rows)
  Buffer x_, xn_, qkv_, q_, ao_, mh_, part_o_, part_ml_, rlog_, eidx_, ewts_, eh_;
  Buffer logits_, cand_v_, cand_i_, sprob_, samp_, acc_, live_all_;
  Buffer uq_;    // a few rows' distinct experts (moe4_unique: ids, row masks, slots, count)
  Buffer rcnt_;  // flint_router_topk's / flint_router_shared's arrival counter (zero between dispatches)
  Buffer acnt_;  // flint_attn_d2's arrival counters, one per KV head (zero between dispatches)
  // prefill activations (tmax_ rows), allocated on the first prefill of more than kMaxRows rows
  bool prefill_alloc_ = false;
  Buffer px_, pxh_, pqkv_, pq_, pck_, pcv_, pao_, phg_, phu_, phh_;
  Buffer prlog_, pidx_, pwts_, spair_, stok_, tiles_, ntiles_, ehh_, yd_, sgb_, sub_, shh_, ys_, pscale_;
};

}  // namespace ni
