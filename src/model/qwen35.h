// Qwen3.5 / 3.6 / 3.8 hybrid model (Gated DeltaNet + gated full attention, 3:1), dense or MoE,
// MLX affine-quantized safetensors.
#pragma once
#include <atomic>
#include <thread>
#include "loader/safetensors.h"
#include "metal/gpu.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace ni {

namespace ane {
class FfnSplit;
struct FfnConfig;
struct FfnLayer;
}

struct QLinear {
  const Tensor* w = nullptr;
  const Tensor* s = nullptr;
  const Tensor* b = nullptr;
  uint32_t N = 0, K = 0;
  int bits = 4, gs = 64;
  bool mtp = false;  // part of the MTP head (NI_QMM8=1 selects its 8-bit prefill GEMM)
  explicit operator bool() const { return w != nullptr; }
};

struct Qwen35Config {
  int hidden = 0, layers = 0, vocab = 0;
  std::vector<bool> is_linear;
  // Gated DeltaNet
  int lin_hk = 0, lin_hv = 0, lin_dk = 0, lin_dv = 0, conv_k = 4;
  // full attention
  int hq = 0, hkv = 0, head_dim = 0, rot_dims = 0;
  float theta = 1e7f, eps = 1e-6f;
  // MLP
  bool moe = false;
  int experts = 0, top_k = 0, moe_I = 0, shared_I = 0, dense_I = 0;
  bool norm_topk = true;
  std::vector<int> eos;
  int group_size = 64;
};

struct Layer {
  bool linear = false;
  const Tensor* in_norm = nullptr;
  const Tensor* post_norm = nullptr;
  // GDN
  QLinear qkv, z, a, b, out;
  const Tensor *conv_w = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gnorm = nullptr;
  // attention
  QLinear q, k, v, o;
  const Tensor *q_norm = nullptr, *k_norm = nullptr;
  // MoE
  QLinear router, shared_gate;
  QLinear eg, eu, ed;          // stacked experts [E, ...]
  QLinear sg, su, sd;          // shared expert
  // dense MLP
  QLinear mg, mu, md;
};

// One conversation's state: token buffer, recurrent (GDN) states, attention KV cache.
struct Seq {
  int cap = 0;                     // max positions
  Buffer tok;                      // int32 [cap + 8]
  std::vector<Buffer> conv, rec;   // per layer (linear layers only)
  std::vector<Buffer> kc, vc;      // per layer (attention layers only), half [Hkv, cap, D]
  // speculative decoding: rows accepted by the last verify, not yet folded into conv/rec
  std::vector<Buffer> pend_qkv, pend_k, pend_g, pend_d;  // per linear layer
  int n_pend = 0;
  Buffer mtp_kc, mtp_vc;           // MTP head's attention cache (positions = trunk positions)
  int* tokens() { return tok.as<int>(); }
};

// Snapshot of a Seq's recurrent state at a position (attention KV needs no copy: entries
// before `pos` stay valid as long as the token prefix matches).
struct Checkpoint {
  int pos = -1;
  std::vector<Buffer> conv, rec;
};

// Target hidden capture for a block drafter (DFlash2):
// after target layer l (mixer + FFN residual add), every row whose position p lies in
// [pos_base, pos_base + rows) is copied (fp32 -> bf16) to dst row p - pos_base, columns
// [slot[l] * H, (slot[l] + 1) * H). Unset (the default) adds no work anywhere.
struct CaptureHook {
  Buffer dst;              // bf16 [rows, ld]
  uint32_t ld = 0;         // elements per dst row (slots * H)
  int pos_base = 0, rows = 0;
  std::vector<int> slot;   // per target layer: capture slot, or -1
};

struct SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  int top_k = 20;
  uint32_t seed = 0;
};

// Timings of one split layer (ANE split benchmark / calibration), ms per layer.
struct AneLayerTimes {
  double gpu_only = 0;   // the FFN on the GPU alone (as without NI_ANE)
  double gpu_part = 0;   // the split's GPU part alone: pack, own channels, staging, join
  double ane = 0;        // the ANE alone (synchronous evaluation)
  double split = 0;      // both, concurrently
  double ane_in_split = 0;  // ANE time inside the split (event reached ready -> report)
};

// Runtime knobs of the 8-row verify kernels (src/model/qwen35_v8.cpp; docs/PERF.md "27B DFlash2 cycle"). Defaults
// are the measured-fastest settings; the environment (NI_V8_CM, NI_V8_PAIR, NI_V8_KPLAN, NI_V8_GUSPLIT,
// NI_V8_GDNSPLIT, NI_V8_PREPU) overrides them, and tools may change them between encodes: ni_run's NI_AB alternates
// two settings per decode cycle so both are timed under the same GPU conditions.
struct V8Knobs {
  int cm = 1;       // GEMM operands: 0 = codes as 1024 + q, 1 = codes as exact half subnormals q * 2^-24
  int pair = 1;     // 1: two 64-groups per inner step (scale / bias pairs in one load each)
  int kplan = 2;    // 1: 8 k-slices x 1 column block per threadgroup when K >= 4096; 2: also 16 k-slices for K >= 16384
  int gusplit = 1;  // 1: dense MLP gate / up as two segments of one GEMM (fp32) + swiglu_x8, instead of qmm8_gu1k2
  int gdnsplit = 256;  // GDN step over value-row blocks of this many threads + gdn_gnorm_x8 (0: one threadgroup per head)
  int prepu = 1;    // 1: x8_prep2's row norms with 4 independent load chains
  // timing only (wrong outputs): skip kernel groups of a dense verify / drafting pass. Verify: 1 GDN step, 2 attention
  // core, 4 GDN prep, 8 MLP down, 16 MLP gate/up, 32 GDN in_proj, 64 GDN out / attention o, 128 LM head + sampling,
  // 256 normed X8 preps, 512 attention q|k|v; drafter: 1024 MLP gate/up, 2048 MLP down, 4096 LM head,
  // 8192 conv / attention projections, 16384 ring commit. With NI_AB: the in-pipeline cost of those kernels.
  int skip = 0;
  int pad = 0;  // timing only: this many empty dispatches after every verify GEMM (the per-dispatch cost, with NI_AB)
};
V8Knobs& v8_knobs();
// "key=value,..." over a copy of `base` (keys: cm, pair, kplan, gusplit, gdnsplit, prepu, skip, pad); unknown keys are fatal
V8Knobs v8_knobs_parse(const char* spec, V8Knobs base);

class Qwen35 {
 public:
  explicit Qwen35(const std::string& model_dir, int prefill_chunk = 2048);
  ~Qwen35();
  const Qwen35Config& config() const { return cfg_; }

  // ---- GPU + Neural Engine split of the dense prefill FFN (opt-in NI_ANE=1; qwen35_ane.cpp) ----
  // Whether a prefill chunk of T rows would run the split (callers checkpoint the recurrent state
  // before such a chunk so it can be rerun on the GPU alone if the split fails).
  bool ane_splits(int T) const;
  // After the command of a chunk encoded with the split has completed: false if the split failed
  // (it is then off for the rest of the process) and the chunk must be rerun. True otherwise.
  bool ane_finish_chunk();
  ane::FfnSplit* ane_split() { return ane_.get(); }
  // Startup check: layers [0, n) on random input rows, GPU alone vs split. Returns
  // RMS(split - gpu) / RMS(gpu - gpu_part) (worst over the functions checked).
  double ane_verify(int n_layers, const std::vector<int>& rows);
  // Benchmark of split layers [l0, l0 + n) at T rows (real weights, random input); `reps` runs
  // of each mode, median per layer.
  AneLayerTimes ane_bench(int l0, int n, int T, int reps, bool stage = true);
  // Rebuilds the split with another share (calibration); false (and the reason logged) if it fails.
  bool ane_rebuild(uint32_t units, uint32_t min_rows, const std::vector<uint32_t>& fn_rows, int fault_layer = -1,
                   int lanes = 1);
  // Times one prefill chunk of T rows (random tokens, fresh state; split per the current setup or
  // the GPU alone), median of `reps` after a warm-up; ms.
  double ane_bench_chunk(int T, int reps, bool split);
  struct AneStats {
    int chunks = 0, evals = 0, reruns = 0;  // evals: ANE evaluations of chunks that held
    double ane_ms = 0;
  };
  const AneStats& ane_stats() const { return ane_stats_; }
  // NI_ANE_ASYNC (the server): finish the Neural Engine split once its background compile is done.
  // Call between requests (it runs the startup check through the model's buffers).
  void ane_poll();

  std::unique_ptr<Seq> new_seq(int cap);
  // Clears recurrent state (KV entries are overwritten by position, so they need no reset).
  void reset(Seq& s);
  Checkpoint new_checkpoint();
  // GPU copies, ordered with the surrounding work in `e`.
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos);
  void restore(Encoder& e, Seq& s, const Checkpoint& c);

  // Encodes one decode step for the token at tokens[pos]. With `sample`, computes logits and
  // writes the sampled id to tokens[pos+1]; without, only advances the state (prompt tokens).
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true);
  // Encodes T prompt tokens tokens[pos0 .. pos0+T) as one batch (T <= prefill_chunk).
  // With `sample`, also samples tokens[pos0+T] from the last position.
  // With `snap` (requires can_snapshot(), 0 < snap_row < T): also saves the recurrent state as it
  // stands after the first snap_row tokens into `snap` (pos = pos0 + snap_row), inside the pass.
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0);
  int prefill_chunk() const { return tmax_; }
  // NI_SHORTPF: checkpoints can be captured inside a prefill pass (encode_prefill's `snap`).
  bool can_snapshot() const;

  // ---- speculative decoding with the model's MTP head ----
  bool has_mtp() const { return mtp_loaded_; }
  // Measured best MTP draft depth for this model (M4 Pro): MoE 35B-A3B 1, dense 27B 2.
  int default_spec_depth() const { return cfg_.moe ? 1 : 2; }
  // Rows per verify (pending token + up to 7 drafts). R <= 4 runs the MTP-era small-batch kernels
  // (unchanged); R >= 5 (or NI_V8=1 for any R >= 2) the 8-row MMA path (qwen35_v8.cpp).
  static constexpr int kMaxVerify = 8;
  static constexpr int kMaxMtpRows = 4;  // MTP history rows per encode_mtp on the default path
  // MTP rows: pairs (hidden row h_row0 + r of `h`, token tokens[tok0 + r]) at MTP positions
  // mpos0 + r; they extend the MTP cache. The last row's post-norm hidden is left in mtp_post()
  // row R-1; with draft_out >= 0 the draft for the next token goes to tokens[draft_out].
  // h_trunk: `h` holds trunk hidden states (pre-norm); false for the MTP's own chained output.
  void encode_mtp(Encoder& e, Seq& s, int mpos0, int R, int tok0, const Buffer& h, int h_row0, int draft_out,
                  bool h_trunk = true);
  // Trunk forward of tokens[pos .. pos+R) (R <= kMaxVerify) without committing the recurrent
  // state: the rows go to the tape; samples() gets one sampled token per row and hidden_rows()
  // their pre-norm hidden states. commit_rows() then picks how many of them were accepted.
  // v8: take the 8-row path for any R >= 1 (a block drafter verifying fewer than 4 proposals).
  void encode_verify(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step, bool v8 = false);
  void commit_rows(Seq& s, int n) { s.n_pend = n; }
  // Folds pending rows into the recurrent state (before plain decode/prefill or a checkpoint).
  void encode_flush(Encoder& e, Seq& s);
  // MTP history for prompt rows [pos0, pos0 + R) using the trunk hidden states of the last
  // prefill chunk (row offset h_row0 in hidden_rows()); tokens[i + 1] must be known.
  void encode_mtp_prefill(Encoder& e, Seq& s, int pos0, int R, int h_row0);
  const int* samples() { return samp_.as<int>(); }
  // MoE models, after an 8-row verify: distinct routed experts of each layer (0 for dense layers)
  const uint32_t* verify8_experts() { return v8ustat_ ? v8ustat_.as<uint32_t>() : nullptr; }
  Buffer hidden_rows() { return x_; }
  Buffer mtp_post() { return mtp_post_; }
  float* logits() { return logits_.as<float>(); }
  // Debug: copy of the residual stream after the last step.
  const float* hidden() { return x_.as<float>(); }
  Buffer debug_buf(const std::string& name);

  // ---- block drafter support ----
  // Arms (or with nullptr disarms) the capture of target layer outputs in encode_step,
  // encode_prefill (every schedule: the plain layer loop, the NI_ANE split and its two-half wavefront,
  // in-pass checkpoints, short prefills) and encode_verify. The hook is read at encode time and must
  // outlive the encodes that use it. A prefill chunk re-encoded after a failed NI_ANE split captures
  // its rows again, so the hook must stay armed until the chunk has completed or been rerun.
  void set_capture(const CaptureHook* h) { cap_ = h; }
  const QLinear& embed_table() const { return embed_; }
  const QLinear& lm_head() const { return lm_head_; }
  // Drafter's dynamic verify depth (DFlash2, MoE only): while set, the 8-row verify routes experts only
  // for rows < min(R, rows[0]) (uint read by the GPU); later rows' outputs are garbage and must not be used.
  void set_verify_row_limit(const Buffer* rows) { v8_rlim_ = rows; }
  // Row-independent MoE numerics in the 8-row verify (every routed expert as an MMA tile, no scalar
  // passes for 1-2-row experts): a row's logits then depend only on the rows before it, not on how the
  // other rows route. A speculative sampler that needs p(. | prefix) exactly sets it (~3% slower verify).
  void set_verify_moe_exact(bool on) { v8_moe_exact_ = on; }
  // The 8-row verify GEMMs, for other 8-row passes (the DFlash2 drafter). Inputs are X8 tables
  // (verify8.metal layout: half operands in fragment order + per-64 group sums), R <= 8 live rows.
  struct Seg8 {
    const QLinear* L;
    Buffer y;
    uint32_t y_ld = 0;  // 0: N
    bool add = false;   // y += X W^T
  };
  // y_i[R, N_i] = X W_i^T for up to 4 matrices sharing K and bits (x2: two-term hi+lo inputs)
  void enc_qmm8(Encoder& e, const std::vector<Seg8>& segs, int R, const Buffer& table, const Buffer& sums,
                bool x2 = false);
  // h = half(silu(X Wg^T) * (X Wu^T)) written as the down projection's X8 table h + sums hs (4-bit, N % 64 == 0)
  void enc_gu8(Encoder& e, const QLinear& gate, const QLinear& up, const Buffer& table, const Buffer& sums,
               const Buffer& h, const Buffer& hs);
  // The last verify's sampler inputs and outputs, as GPU buffers (for an on-GPU acceptance step):
  // stage-1 top-k candidates per row (values / ids, verify_candidates() per row) and the sampled ids.
  Buffer verify_cand_vals() const { return cand_v_; }
  Buffer verify_cand_ids() const { return cand_i_; }
  Buffer verify_samples() const { return samp_; }
  // candidates per verify row for these sampling parameters (stage-1 chunks x top-k)
  uint32_t verify_candidates(const SamplingParams& sp) const;

 private:
  void load_config(const std::string& dir);
  QLinear qlin(const std::string& name, bool required = true);
  void alloc_state();
  void prewarm_pipelines();

  void enc_qmv(Encoder& e, const QLinear& L, const Buffer& x, const Buffer& y, const Buffer* resid,
               const Tensor* norm);
  struct Seg {
    const QLinear* L;
    Buffer y;
  };
  void enc_qmv_multi(Encoder& e, std::vector<Seg> segs, const Buffer& x, const Tensor* norm,
                     const Buffer* xn_out = nullptr);
  void enc_gdn(Encoder& e, Seq& s, int li, const Layer& L);
  void enc_attn(Encoder& e, Seq& s, int li, const Layer& L, int pos);
  void enc_moe(Encoder& e, const Layer& L);
  // prefill
  void enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T, int D);
  void enc_qmm(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add);
  // the large-prefill kernel choice (qmr, else the original qmm tiles), no short-prefill paths
  void enc_qmm_large(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add);
  // NI_QMM + NI_QMM_GU: h = half(silu(xh G^T) * (xh U^T)) in one dispatch; false when off/unsupported
  bool enc_qmm_gu(Encoder& e, const QLinear& G, const QLinear& U, const Buffer& xh, const Buffer& h, int M);
  void enc_qmm_moe(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, const Buffer& xrows,
                   const Buffer& yrows, int P, int bm);
  void enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp, uint32_t step);
  void enc_copy(Encoder& e, const Buffer& src, const Buffer& dst);
  void enc_moe_prefill(Encoder& e, const Layer& L, int T, const Buffer& x);
  void enc_attn_prefill(Encoder& e, const Buffer& kc, const Buffer& vc, int T, int pos0, int cap);
  bool enc_moe_prefill_fused(Encoder& e, const Layer& L, int T, const Buffer& x);
  // NI_GDN_PAR=1: GDN conv/prep, recurrence and gated norm of T rows; leaves the half rows in hh_
  void enc_gdn_prefill_par(Encoder& e, Seq& s, int li, const Layer& L, int T, Checkpoint* snap = nullptr,
                           int snap_row = 0);
  void enc_mlp(Encoder& e, const Layer& L);
  // NI_SHORTPF: register-streaming GEMMs for T <= 64 rows (qmms.metal)
  struct SegQ {
    const QLinear* L;
    Buffer y;
    bool add;
  };
  bool small_t(int T) const;
  bool qmms_ok(const QLinear& L) const;
  void enc_qmms(Encoder& e, std::vector<SegQ> segs, const Buffer& xh, int T);
  // projections sharing xh: one dispatch where the short-prefill kernels allow it
  void enc_proj(Encoder& e, const std::vector<SegQ>& segs, const Buffer& xh, int M);
  void enc_topk_rows(Encoder& e, uint32_t E, uint32_t k, bool norm, int T);
  void enc_qmr_short(Encoder& e, const std::vector<SegQ>& segs, const Buffer& xh, int M);
  bool enc_qmms_gu(Encoder& e, const QLinear& G, const QLinear& U, const Buffer& xh, const Buffer& h, int T);
  bool enc_moe_prefill_small(Encoder& e, const Layer& L, int T, const Buffer& x);
  void enc_mlp_prefill(Encoder& e, const Layer& L, int T, const Buffer& x);
  // ANE split (qwen35_ane.cpp)
  void init_ane();
  void ane_finish_init();  // builds the split (program from the daemon's cache when warm) + startup check
  ane::FfnConfig ane_cfg(uint32_t units, uint32_t min_rows, const std::vector<uint32_t>& fn_rows, int fault_layer,
                         int lanes) const;
  std::vector<ane::FfnLayer> ane_layers() const;
  struct AneInit {
    uint32_t units = 0, min_rows = 0;
    std::vector<uint32_t> fns;
    int fault = -1, lanes = 1;
    const char* source = "";
  } ane_init_;
  std::thread ane_thread_;                           // NI_ANE_ASYNC: background compile of the program
  std::shared_ptr<std::atomic<int>> ane_async_st_;   // 1 compiling, 2 compiled, 3 failed
  void ane_begin_chunk(int T);
  void enc_mixer_prefill(Encoder& e, Seq& s, int i, const Layer& L, int pos0, int T, const Buffer& x,
                         Checkpoint* snap = nullptr, int snap_row = 0);
  // NI_ANE wavefront: the first half's rows (0 = not used for a chunk of T rows)
  int ane_wave_rows(int T) const;
  void enc_prefill_wave(Encoder& e, Seq& s, int pos0, int T, int h);
  void enc_mlp_gpu_part(Encoder& e, const Layer& L, int T, uint32_t g, const Buffer& x);
  void enc_mlp_prefill_split(Encoder& e, int li, const Layer& L, int T, const Buffer& x);
  void enc_mlp_rows(Encoder& e, const Layer& L, int R, const Buffer& x);
  QLinear own_qlin(const class WeightStore& w, const std::string& name);
  // speculative decoding helpers (qwen35_spec.cpp)
  void load_mtp(const std::string& dir);
  QLinear quantize8(const Tensor& w);
  const Tensor* own_copy(const Tensor& t);
  // A float parameter the kernels read as bf16 (norms, conv, A_log, dt_bias): t itself when it is
  // bf16, else an owned bf16 copy (++lossy when a value is not a bf16 number)
  const Tensor* bf16_param(const Tensor* t, int& lossy);
  QLinear stack_experts(const class WeightStore& w, const std::string& prefix, const std::string& proj);
  void enc_rows_norm(Encoder& e, const Buffer& x, const Tensor* w, const Buffer& y, int R, int D);
  // block-drafter capture of rows [0, T) of x (default x_) at positions pos0.. after layer `layer` (no-op
  // unless armed); x must hold the layer's final output for those rows (mixer and FFN added)
  void enc_capture(Encoder& e, int layer, int pos0, int T, const Buffer* x = nullptr);
  void enc_qmvr(Encoder& e, const QLinear& L, const Buffer& x, const Buffer& y, int R, bool add,
                uint32_t x_ld = 0, uint32_t y_ld = 0);
  // fused (norm +) projections of R rows sharing the input; falls back to separate kernels
  void enc_qmvr_multi(Encoder& e, std::vector<Seg> segs, const Buffer& x, const Tensor* norm, int R,
                      const Buffer* xn_out = nullptr, std::vector<bool> add = {});
  void enc_attn_rows(Encoder& e, const Layer& L, const Buffer& kc, const Buffer& vc, int cap, int pos, int R,
                     const Buffer& x);
  void enc_moe_rows(Encoder& e, const Layer& L, int R, const Buffer& x, bool mtp_layer);
  // ---- 8-row verify path (qwen35_v8.cpp) ----
  void v8_alloc();
  void encode_verify8(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step);
  // x rows (fp32, x_ld floats apart) -> X8 table (+ group sums), optional RMSNorm, optional lo table
  void enc_x8(Encoder& e, const Buffer& x, int R, int K, const Tensor* norm, bool lo = false, int x_ld = 0,
              const Buffer* xn_out = nullptr);
  void enc_attn8(Encoder& e, const Layer& L, const Buffer& kc, const Buffer& vc, int cap, int pos, int R);
  void enc_mlp8(Encoder& e, const Layer& L, int R);
  void enc_moe8(Encoder& e, const Layer& L, int R, int li);

  std::string dir_;
  std::string prefix_;
  Qwen35Config cfg_;
  WeightStore ws_;
  std::vector<Layer> layers_;
  const Tensor* final_norm_ = nullptr;
  QLinear embed_, lm_head_;
  int tmax_;
  // MTP head
  bool mtp_loaded_ = false;
  bool mtp_post_norm_hidden_ = false;  // head expects the trunk hidden after the final norm
  Layer mtp_;
  QLinear mtp_fc_;
  const Tensor *mtp_norm_ = nullptr, *mtp_pre_e_ = nullptr, *mtp_pre_h_ = nullptr;
  QLinear draft_head_[2];   // lm_head rows [0, 32768) and the added/special tokens
  uint32_t draft_split_ = 0, draft_special0_ = 0;
  bool full_draft_ = false;  // draft over the whole vocabulary (dense default; NI_FULL_DRAFT=0/1 overrides)
  const CaptureHook* cap_ = nullptr;
  const Buffer* v8_rlim_ = nullptr;
  bool v8_moe_exact_ = false;
  std::deque<Tensor> owned_;
  std::vector<Buffer> owned_bufs_;

  std::unique_ptr<ane::FfnSplit> ane_;
  bool ane_chunk_ = false;  // the chunk being encoded runs the split
  bool ane_on_ = true;      // benchmarks: false runs the GPU alone
  AneStats ane_stats_;

  // state
  Buffer x_, xn_, logits_;
  Buffer qkv_, zb_, ab_, bb_, gq_, gk_, gv_, gg_, gbeta_, gout_;
  Buffer qg_, kr_, vr_, qo_, attn_out_, part_o_, part_ml_;
  Buffer rlog_, slog_, eidx_, ewts_, eh_;
  Buffer cand_v_, cand_i_, sprob_;
  // prefill-only
  Buffer mtp_e_, mtp_u_, mtp_x_, mtp_post_, xrn_, samp_, dlog_, ue_, uw_, un_, mh_;
  Buffer xh_, hh_, ridx_, rwts_, spair_, stok_, tiles_, ntiles_, hg_, hu_, hhalf_, yd_, sgb_, sub_, ys_;
  // 8-row verify (allocated on first use)
  bool v8_ready_ = false;
  Buffer v8g_, v8u_;  // split MLP: fp32 gate / up rows [8][I]
  Buffer v8cnt_;      // gdn_step_verify8s: per-head arrival counters (zero between dispatches)
  Buffer v8t_, v8s_, v8tl_, v8sl_, v8h_, v8hs_, v8tiles_, v8nt_, v8slot_, v8args_, v8out_, v8sh_, v8ustat_, v8sm_, v8hsc_, v8xn_;
};

} // namespace ni
