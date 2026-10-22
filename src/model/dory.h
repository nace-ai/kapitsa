// NVIDIA Nemotron-Labs Dory (DoryForCausalLM, model_type "dory"): a looped (latent-recurrent) normalized GPT. 72
// physical layers, each an attention or an MLP sublayer: 16 input layers, a 40-layer recurrent block run `loops` times
// (the input re-injected each time), 16 output layers. Sliding-window (1024) and full attention 3:1, GQA 16 / 4 heads
// of 256, SwiGLU MLP, nGPT normalizations (justnorm: L2) and slerp-like residuals, an output gate on attention, and a
// per-vocabulary logit scale. NVIDIA's safetensors as they are: BF16, or NVFP4 (ModelOpt W4A16) linears with BF16
// embedding and LM head; the matrix kernels are Nemotron-H's (nemotron_h.h's formats and repacking).
//
// The math is NVIDIA's (vLLM PR #60149, vllm/model_executor/models/dory.py). Loops: 1-4 (the card recommends 2, its
// config's n_recurrent_loops), per conversation (Seq::loops). KV of the recurrent block's attention layers:
//  - last-loop (default; the card's recurrent_kv_cache_mode "last_loop"): one cache per layer, every loop writing
//    the step's keys over the previous loop's, so a step's tokens attend to their own loop's keys and earlier
//    tokens to their final loop's; the cache costs what a one-loop model's does at any loop count. A step's result
//    depends on how the tokens are grouped into steps (as in vLLM and llama.cpp): a prompt chunk's tokens see each
//    other's same-loop keys, decoded tokens one at a time see final-loop keys.
//  - per-loop: a cache per loop (what NVIDIA's vLLM and llama.cpp commands run; steps do not matter), loops x the
//    recurrent layers' KV.
// Sliding-window layers keep a ring of window + 8 positions; full-attention layers a cache that grows with the
// conversation. A ring cannot go back further than its 8 spare positions, so checkpoints (Model::save / restore,
// the engine's turn boundaries) hold copies of the rings.
#pragma once
#include "loader/safetensors.h"
#include "model/model.h"
#include "model/nemotron_h.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

enum class LoopKv : uint8_t { last_loop = 0, per_loop = 1 };
const char* loop_kv_name(LoopKv m);              // "last-loop" / "per-loop"
bool parse_loop_kv(const std::string& s, LoopKv* out);  // last-loop | last_loop | per-loop | per_loop

struct DoryConfig {
  int hidden = 0, vocab = 0, layers = 0, hq = 0, hkv = 0, hd = 0, ffn = 0;
  int n_in = 0, n_rec = 0, n_out = 0;
  int loops = 2;             // config.json's n_recurrent_loops (the card's recommended default)
  int window = 0;
  double theta_swa = 1e4, theta_full = 1e7;
  double init_std = 0.02, alpha_init = 0.05, gate_init = 0;  // gate_init: the output gate's logit scale init
  std::vector<char> kind;    // per physical layer: 'S' sliding-window attention, 'F' full attention, 'M' MLP
  std::vector<int> eos;
  int bos = -1;
  bool recurrent(int l) const { return l >= n_in && l < n_in + n_rec; }
  bool attention(int l) const { return kind[size_t(l)] != 'M'; }
};

struct DoryLayer {
  char kind = 'M';
  bool rec = false;   // in the recurrent block
  NhMat qkvg, o;       // attention: q | k | v | g rows as one matrix (one product), o
  NhMat gu, down;      // MLP: gate | up rows as one matrix, down
  NhFmt fmt_any() const { return kind == 'M' ? gu.fmt : qkvg.fmt; }
  float sq = 1, sk = 1, sv = 1, sg = 1;  // the joint matrices' parts' NVFP4 tensor scales (1 for BF16)
  float sgate = 1, sup = 1;
  Buffer alpha;       // fp32 [hidden]: |alpha x ngpt_alpha_init / initializer_range|
  Buffer sqk2;        // fp32 [hq x hd]: (sqk x sqrt(hd) / initializer_range)^2
  Buffer sg2;         // fp32 [hq x hd]: (s_gate x sqrt(gate_init) / initializer_range)^2
  Buffer sug, suu;    // fp32 [ffn]: suv_gate, suv_up x sqrt(hidden)
  int cache = -1;     // attention: the index of its (first) cache in Seq::kc / vc
};

class Dory : public Model {
 public:
  static constexpr int kMaxLoops = 4;
  static constexpr int kMaxRows = 8;   // rows of a decode / verify pass
  static constexpr int kRingSpare = 8;  // a sliding-window ring's positions beyond the window

  // path: the checkpoint folder; loops: the default loop count (0 = the config's); kv: the recurrent KV mode
  Dory(const std::string& path, int prefill_chunk = 2048, int loops = 0, LoopKv kv = LoopKv::last_loop);
  ~Dory() override;
  static bool is_model(const std::string& dir);  // config.json's model_type is "dory"
  static DoryConfig read_config(const std::string& dir);
  // The memory plan (headers only): kv_per_pos for `loops` (0: the config's) in mode kv
  static MemPlan plan(const std::string& path, int prefill_chunk, LoopKv kv, int loops);
  static MemPlan plan(const std::string& path, int prefill_chunk = 2048) {
    return plan(path, prefill_chunk, serving_kv(), serving_loops());
  }
  // What the server will run (--loops, --loop-kv), for the memory plans made before the model loads (plan_load,
  // auto_slots, doctor): plan() without them takes these
  static void set_serving(int loops, LoopKv kv);
  static int serving_loops();     // 0: the config's
  static LoopKv serving_kv();
  // KV cache bytes per position (full-attention layers) and per conversation (sliding-window rings), for doctor
  static size_t kv_per_pos(const DoryConfig& c, LoopKv kv, int loops);
  static size_t ring_bytes(const DoryConfig& c, LoopKv kv, int loops);

  const DoryConfig& config() const { return cfg_; }
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }
  int default_loops() const { return loops_; }
  LoopKv kv_mode() const { return kv_; }
  // the loop count a conversation runs: its own, else the default
  int loops_of(const Seq& s) const { return s.loops > 0 ? s.loops : loops_; }
  std::string formats_summary() const;

  // ---- Model ----
  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override;
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override;
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  int prefill_chunk() const override { return tmax_; }
  bool can_snapshot() const override { return false; }
  // SSD prefix cache: the full-attention caches of the loops s runs as rows, its sliding-window rings whole (a
  // checkpoint's: their copies); the engine keys the states by the loop count too
  std::string disk_format() const override;
  void disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const override;
  void disk_state(const Seq& s, const Checkpoint* c, std::vector<Buffer>& out) const override;

  // ---- speculative decoding (exact with per-loop KV, up to half-row rounding: the engine runs it only then) ----
  // Rows [pos, pos + R) (R <= kMaxRows) through the whole model as one step: with per-loop KV each row sees the
  // earlier rows' keys of its own loop, as decoding them one by one would. Next-token logits of each row in logits().
  void encode_verify(Encoder& e, Seq& s, int pos, int R);
  // Self-speculation: the draft at pos (row `row` of the coming verify) is the model with one loop: the input layers
  // and loop 0 (with per-loop KV the target's own values, kept for the verify), then the output layers (their KV is
  // overwritten by the verify) and the draft head; its greedy token into tokens[pos + 1].
  void encode_draft(Encoder& e, Seq& s, int pos, int row);
  // The verify of rows [pos, pos + R) whose first R - 1 were drafted: the last row's input layers and loop 0, then
  // loops 1.. and the output layers and head of all R rows (logits()).
  void encode_verify_drafted(Encoder& e, Seq& s, int pos, int R);
  // After a verify: each row's token sampled with sp (step = its position, as a plain decode samples it), the drafts
  // tokens[pos + 1 .. pos + K] accepted while they agree, tokens[pos + 1 + a] = samples[a]; accepted() = a once the
  // command buffer has completed.
  void encode_accept(Encoder& e, Seq& s, int pos, int R, int K, const SamplingParams& sp);
  int accepted() const { return acc_.as<int>()[0]; }

  // ---- tools ----
  // logits of rows [0, T) of the last pass (fp32 [T][vocab]): a decode step's row 0, a prefill's last row
  float* logits() { return logits_.as<float>(); }
  // teacher-forced scoring: tokens [pos0, pos0 + T) as one step, every row's next-token logits into out [T][vocab]
  void encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out);

 private:
  void load(int prefill_chunk);
  void alloc_buffers();
  void alloc_prefill();
  bool ensure_loop_caches(Seq& s, int loops);  // per-loop mode: the caches of loops [0, loops)
  int cache_index(const DoryLayer& L, int loop) const;
  int cache_rows(const Seq& s, int cache) const;  // a ring's rows, or the linear cap
  bool is_ring(int cache) const { return ring_[size_t(cache)]; }

  // passes
  void enc_rows(Encoder& e, Seq& s, int pos0, int R);           // R <= kMaxRows rows; final hidden rows in x_
  // the stages of a rows pass: input layers; the loops [l0, l1) of `loops` (h = x_ entering loop l0); output layers
  void enc_input_rows(Encoder& e, Seq& s, int pos0, int R);
  void enc_loops_rows(Encoder& e, Seq& s, int pos0, int R, int l0, int l1, int loops);
  void enc_output_rows(Encoder& e, Seq& s, int pos0, int R);
  void enc_copy_rows(Encoder& e, const Buffer& src, const Buffer& dst, int R);
  void enc_prefill_rows(Encoder& e, Seq& s, int pos0, int T);   // GEMM pass; final hidden rows in px_
  void enc_layer_rows(Encoder& e, Seq& s, const DoryLayer& L, int loop, int pos0, int R, int mode);
  void enc_layer_prefill(Encoder& e, Seq& s, const DoryLayer& L, int loop, int pos0, int T, int mode, bool last_loop);
  void enc_head(Encoder& e, const Buffer& x, int R, const Buffer& out, size_t ld);
  void enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                  uint32_t step);
  // matrix products: rows of fp32 x (<= 8) / GEMM of half rows
  void enc_mv(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int R, int epi);
  void enc_gemm(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi);
  void enc_slerp(Encoder& e, const Buffer& h, const Buffer& y, const Buffer& alpha, const Buffer& inp,
                 const Buffer* xh, int R, int mode);
  void enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out);

  std::string path_;
  std::unique_ptr<SafetensorsIndex> st_;
  DoryConfig cfg_;
  int loops_ = 2;
  LoopKv kv_ = LoopKv::last_loop;
  std::vector<DoryLayer> layers_;
  NhMat embed_, head_;
  Buffer logit_scale_;           // fp32 [vocab]: logit_scale / initializer_range
  Buffer inv_swa_, inv_full_;    // fp32 [hd / 2]: rope inverse frequencies
  std::vector<bool> ring_;       // per cache: a sliding-window ring
  std::vector<int> cache_loop_;  // per cache: its loop (per-loop mode's recurrent caches), else 0
  int ring_rows_ = 0;
  int n_caches_ = 0;
  int tmax_ = 2048;
  size_t mem_budget_ = 0;
  std::vector<Buffer> owned_;    // weight buffers
  void* embed_map_ = nullptr;
  size_t embed_map_len_ = 0;
  Buffer dummy_;                 // bound where a shared kernel declares a buffer Dory's formats never read

  // decode / verify activations (kMaxRows rows)
  Buffer x_, inp_, y_, q_, qkvg_, ao_, mgu_, mh_, part_o_, part_ml_;
  Buffer logits_, cand_v_, cand_i_, sprob_, samp_;
  Buffer st0_, inp0_, acc_, live_all_;  // self-speculation: drafted rows' state after loop 0 and input rows; acceptance
  Buffer x4part_, x4cnt_;               // NVFP4 9-32-row GEMMs: split-K partials and arrival counters
  Buffer xh_;                           // 2-8 rows as half (the 8-row MMA products of a verify pass)
  // NVFP4 codes in layout L2 (nh_x_fp4.metal) instead of L0, read by the L2 kernels (nh_x4_mv, nh_x4_v8, nh_x4_mm*,
  // nh_gemm_fp4l2_m*): on the M1's GPU (Apple7) their decode is 12-16% faster at Dory's shapes, with the same outputs
  // to the bit (the layouts' values differ by powers of 2, which the kernels undo exactly). NI_DORY_FP4=l0 / l2.
  bool fp4_l2_ = false;
  // M5's matrix units (Metal 4 tensor_ops, Apple10 and macOS 26) for the prefill GEMMs over 32 rows: the Nemotron-H
  // gemms of nemotron/nh_gemm.metal with the products on matmul2d (tensor/nh_gemm_t.metal). NI_DORY_TG=0 turns it off.
  bool tg_ = false;
  static const char* x4(const NhMat& W, const char* l2, const char* l0) { return W.l2 ? l2 : l0; }
  int attn_max_chunks_ = 64;
  // prefill activations (tmax_ rows; on first use)
  bool pf_alloc_ = false;
  Buffer px_, pxh_, pinp_, py_, pq_, pqkvg_, pck_, pcv_, pao_, paoh_, pgu_, pmh_;
};

}  // namespace ni
