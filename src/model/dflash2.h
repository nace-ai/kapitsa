// DFlash2 block drafter: loads a DFlash2DraftModel
// checkpoint (bf16 safetensors), quantizes its projections to MLX affine 4-bit (or 8-bit) g64 at
// load, keeps the per-sequence context K/V ring fed from the target's captured hidden states, and
// encodes one drafting pass (anchor + 7 mask rows -> 7 proposals) into a command buffer.
// Opt-in: nothing here is constructed unless a drafter directory is given.
#pragma once
#include "model/kflash_budget.h"
#include "model/qwen35.h"

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ni {

struct DFlash2Config {
  int H = 0, NL = 0, I = 0, vocab = 0, mask_id = 0;
  int nq = 32, nkv = 8, hd = 128, block = 8, window = 2048, conv_gs = 16, rank = 256, topk = 16;
  float eps = 1e-6f, theta = 1e7f;
  std::vector<int> target_layers;  // target layer ids whose outputs feed fc, in slot order
  // the target's layer outputs reach the draft as their hyper-connection streams' mean (target_hidden_reduction
  // stream_mean: Qwen3.8-Flash-Next), else as they are
  bool stream_mean = false;
  // DFlash (v1) drafter (DFlashDraftModel, e.g. z-lab/Qwen3.5-4B-DFlash): the same context path and block layout
  // without the block convolutions and the candidate selector; it runs with the 8-row block and the 2048-position
  // context ring here (trained with 16-row blocks and a 4096-position window: the block size is a runtime choice,
  // the shorter window only matters past 2047 tokens of context)
  bool v1 = false;
  int trained_block = 0, trained_window = 0;
};

// Per-sequence drafter state: the context K/V ring of every draft layer.
struct DFlashSeq {
  Buffer ring_k, ring_v;  // bf16 [NL][8 kv heads][2048 slots][128]; slot = position mod 2048
  int ctx = 0;            // committed context positions 0 .. ctx-1 (the ring holds the last 2048)
  int valid_lo = 0;       // lowest position whose ring entry is valid (> 0 after a rewind past the window)
  // Batched decode (DFlash2::stage_init): the target captures of rows not yet in the ring, positions ctx ..
  // ctx + staged - 1, bf16 [8][n_slots * H]; the hook a batched verify writes them with (by row)
  Buffer stage;
  int staged = 0;
  CaptureHook stage_hook;
  // sampled requests in a batch: the walk of its last draft (candidate ids [7][16], q [7][16], choices [8]) and its
  // acceptance's result {a, token}, kept per sequence (bytes 0, 448, 896, 928; 944: a drafter state it may write);
  // every member under the batch budget (KfBatch): the walk's scores [7][16] (byte 960, encode_save_scores)
  Buffer walk;
  int accepted() const { return walk.as<int>()[232]; }
  const float* scores() const { return walk.as<float>() + 240; }
};

// 8-row affine-quantized GEMM (dfl_qmm8): Y_i[8, N_i] = oscale_i * X[8, K] . W_i^T for up to 4
// matrices sharing the X8 table (xh half [8][x_ld], xs float [K/64][8]). ks = 0 picks the K split;
// rf = 2 gives each simdgroup 16 weight rows (0: NI_DFLASH_RF, default 1).
struct DflSeg8 {
  const QLinear* L;
  Buffer y;
  uint32_t y_ld;
  float oscale;
};
void dfl_enc_qmm8(Encoder& e, const std::vector<DflSeg8>& segs, const Buffer& xh, const Buffer& xs, uint32_t K,
                  uint32_t x_ld, int ks = 0, int rf = 0);
// Gate/up + SwiGLU (dfl_qmm8_gu): hh half [8][N] = silu(X Wg^T) * (X Wu^T), hs float [N/64][8].
void dfl_enc_qmm8_gu(Encoder& e, const QLinear& g, const QLinear& u, const Buffer& xh, const Buffer& xs,
                     const Buffer& hh, const Buffer& hs, int ks = 0);

// MLX affine quantization (mx.quantize, group 64) of bf16 rows on the CPU: W uint32 [N][K*bits/32],
// S/B bf16 [N][K/64]. even=false reproduces MLX's Metal kernel (IEEE fp32, round half away from
// zero), even=true MLX's CPU backend (round half to even).
void dfl_quantize_cpu(const std::vector<const uint16_t*>& rows, uint32_t K, int bits, bool even, uint32_t* W,
                      uint16_t* S, uint16_t* B);
// One row of a linear in any format (bf16, or affine bits / group gs) as bf16 values (CPU).
void dfl_row_bf16(const QLinear& H, uint32_t row, uint16_t* out);

// Draft attention split plan for anchor position P: context [lo, P) cut into nsplit chunks of
// `chunk` positions (multiple of 32), at least min_chunk each (0: NI_DFLASH_ATTN_CHUNK, default 256),
// at most max_splits.
void dfl_attn_plan(int P, uint32_t max_splits, uint32_t min_chunk, uint32_t* lo, uint32_t* chunk, uint32_t* nsplit,
                   int valid_lo = 0);

class DFlash2 {
 public:
  static constexpr int kB = 8, kWin = 2048, kTopK = 16, kRank = 256, kCtxKeep = 2047;
  // The drafting block's rows: kB (8, the tile), or 16 (the anchor + 15 mask rows, two 8-row tiles; KFlash tree cycles
  // only). NI_DFLASH_BLOCK picks it (8, 16, or `auto` for the default); where that says nothing it is 8, and Engine asks
  // for 16 on a single-slot engine that takes the 16-row paths (the ctor's rows16). Fixed while the drafter loads, since
  // alloc_buffers() and the reduced head size the rows from it; per drafter, so a process that switches models
  // (ModelManager::switch_to) picks its block again.
  int block() const { return block_; }
  static constexpr uint32_t kGpuChunk = 256;  // context positions per attention split, GPU-driven passes
  // default reduced draft head rows: from a ranked id list (NI_DFLASH_HEAD_IDS), and in id order (the default)
  static constexpr int kHeadRows = 65536, kHeadRowsIdOrder = 98304;
  // bits: 4 or 8 (0: NI_DFLASH_BITS, default 4). rows16: draft a 16-row block where this target, the drafter's
  // projections and the knobs allow it and NI_DFLASH_BLOCK says nothing (Engine passes slots == 1: a 16-row block has
  // no batched cycle to lose, Engine::batch_rows).
  DFlash2(Qwen35& target, const std::string& dir, int bits = 0, bool rows16 = false);
  const DFlash2Config& config() const { return cfg_; }
  int bits() const { return bits_; }
  size_t resident_bytes() const { return resident_; }

  // A DFlash2 drafter directory for this target, or "" (no match): <model_dir>/dflash2 or /drafter,
  // then `*-DFlash2` directories next to the model, in <parent>/drafters and in
  // ~/.cache/nace-infer-models/drafters whose name (minus -DFlash2, case-insensitive) is a prefix of
  // the model directory's name up to a '-', and whose config.json is a DFlash2DraftModel with the
  // target's hidden size, vocabulary and layer count that passes the load-time checks (SPEC §1).
  // The longest name match wins.
  static std::string find(const std::string& model_dir, const Qwen35Config& target);
  // Memory a drafter takes, from its config.json and safetensors headers only (the engine's fit check): what the
  // constructor allocates (projections quantized to NI_DFLASH_BITS, default 4; norms, codebooks and conv kernels
  // as stored; buffers; the reduced draft head, rows of the target's LM head of head_row_bytes each), the target's
  // 8-row verify scratch and the prompt-capture buffers of the first prefill (*at_prefill, included); *per_slot:
  // one context ring (a slot holds three: the live state and its two checkpoints' snapshots).
  // The drafting block the engine will load for `target` with `slots` slots: NI_DFLASH_BLOCK if it names 8 or 16, else
  // 16 on a one-slot engine whose target has a wide tree verify, else 8 (the load-time checks repeat this in full).
  // block_env: NI_DFLASH_BLOCK's value to decide from (nullptr: the environment; "" or "auto": the default), so the
  // decision can be asked several times in one process (a --allow-switch server decides each model at its load).
  // tw_ok: whether this Mac has the matrix-unit drafting pass (dfl_tw_supported(): M5-class GPU, macOS 26): the
  // caller asks the GPU, so that the decision stays a pure function a CPU test can drive both ways.
  static int plan_block_rows(const Qwen35Config& target, int slots, const char* block_env, bool tw_ok);
  // Whether this Mac and these knobs run the drafting pass on the matrix units (dfl_tw_supported() and V8Knobs):
  // plan_block_rows takes the answer as an argument so that the decision itself is left with the environment alone.
  static bool tensor_path_ok();
  // block_rows: the drafting block the engine will load (plan_block_rows), whose rows the target's verify buffers and
  // every slot's verify tape are sized from.
  static size_t plan_bytes(const std::string& dir, const Qwen35Config& target, size_t head_row_bytes,
                           size_t* per_slot, size_t* at_prefill = nullptr, int block_rows = kB);
  // The target's tensors the drafter borrows have a supported format (4-bit embedding, 4/8-bit g64 LM head).
  static bool supports(const Qwen35& target);
  // The reduction a draft for this target reads its layer outputs with: "stream_mean" for a hyper-connected target
  // (Qwen3.8-Flash-Next's 4 streams), else "none" (dflash_config.target_hidden_reduction; a config without it takes it)
  static std::string target_reduction(const Qwen35Config& target);
  // A drafter config.json's text against a target: the failed checks (the load-time ones; empty: it loads)
  static std::vector<std::string> config_check(const std::string& config_json, const Qwen35Config& target);
  // Creates the drafter's and the 8-row verify's pipelines up front (in parallel, as Qwen35 does for its
  // prefill), so the first request does not create them inside its prompt commit and first cycles.
  // NI_PREWARM=0 skips it.
  static void prewarm_pipelines();

  std::unique_ptr<DFlashSeq> new_seq();
  // Prefix reuse: entries at positions >= pos no longer belong to the sequence (the target state was
  // restored to pos); afterwards d.ctx = pos. reset() empties the ring.
  static void rewind(DFlashSeq& d, int pos);
  static void reset(DFlashSeq& d) { d.ctx = d.valid_lo = 0; }
  // Ring snapshot (the engine keeps one per recurrent-state checkpoint): copies src's ring into dst on the GPU,
  // in order with the work in `e`, and its ctx / valid_lo. Restoring a checkpoint with its snapshot (then
  // rewind() to the checkpoint) gives the drafter the context it had there, however much was committed since.
  void encode_ring_copy(Encoder& e, const DFlashSeq& src, DFlashSeq& dst);

  // ---- context path (SPEC §6) ----
  // Prefill of prompt rows [start, n) (start > 0: a reused prefix, d.ctx == start): begin_prefill
  // arms the target capture of the rows that can still be visible, [max(start, n - 2047), n); run
  // the target's prefill chunks; encode_prefill_commit() projects them into the ring (d.ctx = n);
  // end_prefill() disarms the hook.
  void begin_prefill(int n, int start = 0);
  void encode_prefill_commit(Encoder& e, DFlashSeq& d);
  void end_prefill();
  // Host-side stand-in for the target captures of the prompt (tests): rows [0, rows) = positions
  // n - rows .. n - 1, bf16 [rows][n_slots * H]. Call after begin_prefill(n).
  void set_prompt_captures(const uint16_t* bf16_rows, int rows);
  // Decode: arm_verify(P) before the target's encode_verify at P captures its rows into capt();
  // encode_commit(e, d, m) adds verify rows 0..m-1 at positions d.ctx .. d.ctx + m - 1.
  void arm_verify(int P, int rows = kB);  // rows: the verify's (a wide tree verify: its capacity)
  void disarm();
  void encode_commit(Encoder& e, DFlashSeq& d, int m);
  // Batched decode: d's staging buffer and its capture hook (once; allocated on first use), and the commit of its
  // staged rows into the ring (d.ctx += d.staged, d.staged = 0)
  void stage_init(DFlashSeq& d);
  void encode_commit_staged(Encoder& e, DFlashSeq& d);
  // Batched decode at T > 0: d keeps the walk of its draft (encode_save_walk right after its encode_draft); after the
  // batched pass, the exact acceptance of its rows [r0, r0 + K + 1) (stage-1 candidates n per row), as encode_accept
  // does for a single verify: tokens[P + 1 + a] = the next token, d.accepted() = a once the command buffer completed
  void encode_save_walk(Encoder& e, DFlashSeq& d);
  void encode_accept_rows(Encoder& e, Seq& s, DFlashSeq& d, int P, int K, const SamplingParams& sp, int r0, uint32_t n);
  // Batched decode under the batch budget: d keeps the walk's scores of its draft (d.scores(), the budget's input; right
  // after its encode_draft)
  void encode_save_scores(Encoder& e, DFlashSeq& d);
  Buffer capt() const { return capt_; }  // bf16 [32][n_slots * H] (rows of the last verify)

  // ---- drafting pass (SPEC §4, §5, §7) ----
  // Block [tokens[P], M x 7] at positions P..P+7 (requires d.ctx == P); writes the proposals to
  // tokens[P+1..P+7]; candidates, walk and q stay in the buffers below. With
  // sp.temperature > 0 the walk samples (uniforms keyed by sp.seed and the position).
  void encode_draft(Encoder& e, Seq& s, DFlashSeq& d, int P, const SamplingParams& sp);

  // ---- decode cycle (SPEC §3, §8, §9) ----
  // Acceptance after the target's verify of [anchor, d_0 .. d_{K-1}] at P (K = proposals verified,
  // 1..7): greedy (T = 0: longest prefix equal to the verify argmax) or exact speculative sampling
  // with the walk's q (T > 0) on the GPU; writes the bonus / correction token to tokens[P+1+a].
  // accepted() = a once the command buffer has completed.
  void encode_accept(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp);
  // The target's sampler distribution of verify rows 0..R-1 (dist_ids() / dist_probs()), T > 0.
  void encode_dist(Encoder& e, int R, const SamplingParams& sp);
  const int* dist_ids() const { return dist_id_.as<int>(); }       // [8][32], -1 past the kept prefix
  const float* dist_probs() const { return dist_p_.as<float>(); }  // [8][32]
  // One whole cycle into `e`, no host round trip inside: ring commit of the previous cycle's m_prev
  // kept rows, the drafting pass from the anchor tokens[P], the target's verify of the first K
  // proposals with captures, the acceptance. Afterwards tokens[P+1 .. P+1+accepted()] are the next
  // tokens; the caller emits m <= accepted() + 1 of them, calls target.commit_rows(s, m) and passes
  // m as the next cycle's m_prev (P advances by m).
  void encode_cycle(Encoder& e, Seq& s, DFlashSeq& d, int P, int m_prev, int K, const SamplingParams& sp);
  // The same cycle as two command buffers committed back to back (no host round trip between them):
  // the small first one (ring commit + drafting pass, ~85 dispatches) starts on the GPU while the host
  // encodes the second (verify + acceptance, ~600 dispatches), which hides most of the encoding time.
  void encode_cycle_draft(Encoder& e, Seq& s, DFlashSeq& d, int P, int m_prev, const SamplingParams& sp);
  void encode_cycle_verify(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp);
  // Double-buffered loop: the drafting half of the NEXT cycle reads its anchor position and the rows to
  // commit from a GPU-side state that the current cycle's acceptance writes, so it can be committed right
  // behind the current verify, before the host has read the result (the GPU then never waits for the
  // host between cycles). gpu_state_reset(P) sets "anchor P, nothing to commit" (host write: only while
  // the GPU is idle, e.g. after the prefill). P_max bounds the anchor (capacity check). The caller keeps
  // d.ctx = the anchor position after each cycle (the GPU committed the accepted rows).
  void gpu_state_reset(int P);
  void encode_cycle_draft_gpu(Encoder& e, Seq& s, DFlashSeq& d, int P_max, const SamplingParams& sp);
  int accepted() const { return acc_.as<int>()[0]; }
  // proposals the last completed cycle verified (K, or the dynamic depth's choice; a tree: its live nodes, as its
  // acceptance recorded them: kN may already be the next cycle's)
  int verified(int K) const {
    const int* t = tree_buf_.as<int>();
    return tree_ ? (t[kf::kObs] > 0 ? t[kf::kObs] : t[kf::kN]) - 1
                 : dyn_tau_ > 0 ? std::min(K, int(kdyn_.as<uint32_t>()[0])) : K;
  }
  // KFlash on every target (NI_KFLASH=0 or NI_DFLASH_TREE=0 keeps the chain): the single-stream cycles (encode_cycle*)
  // verify a best-first tree over the lattice (the shared builder kf_build on this drafter's children tables,
  // dfl_kf_table) instead of the walk's chain, under the shared budget (KfBudget: the lattice's probabilities
  // calibrated against the verify's own acceptances, a cycle's GPU time per row count measured on this Mac), and
  // accept it with the shared kf_accept: greedy and sampled alike the emitted tokens are plain decoding's (at T > 0 its
  // sampler's draws with each position's uniform). The verify's rows: whole 8-row tiles on dense targets (8, 16 or 32
  // per cycle by the size rule where the target runs wide tree verifies, Qwen35::verify_wide_ok; NI_KFLASH_ROWS
  // restricts them; else 8), 1 .. 8 live rows on MoE targets (the experts route the live rows alone). encode_draft (the
  // batched path) keeps the walk: there the batch budget (batch_budget(), KfBatch) chooses each member's rows from its
  // chain under the same calibration and a cost measured the same way.
  bool tree() const { return tree_; }
  int tree_rows() const { return tree_rows_; }  // the widest tree verify (8, 16 or 32)
  void set_tree(bool on);
  const int* tree_desc() const { return tree_buf_.as<int>(); }  // the tree descriptor (kflash.h) of the last cycle
  KfBudget* budget() const { return kf_budget_.get(); }
  KfBatch* batch_budget() const { return kf_batch_.get(); }
  // after each tree cycle completed: the cycle's GPU ms for the budget, its drafting half's command buffer and its
  // verify's (the loop calls it after its wait)
  void observe(double ms);
  // proposals verified per cycle: NI_DFLASH_VERIFY (1..block - 1), default block() - 1
  int verify_depth() const { return vdepth_; }
  // dynamic verify depth threshold (dfl_depth; MoE targets only): 0.5 by default, NI_DFLASH_DYN=tau, 0 = off
  float dyn_tau() const { return dyn_tau_; }
  // sampled cycles verify with row-independent MoE numerics (Qwen35::set_verify_moe_exact)
  bool exact_moe() const { return exact_moe_; }

  // results of the last pass (host-visible after the command buffer completed)
  const int* cand_ids() const { return cand_i_.as<int>(); }       // [7][16]
  const float* cand_vals() const { return cand_v_.as<float>(); }  // [7][16]
  const float* qprob() const { return qprob_.as<float>(); }       // [7][16] (sampled walk)
  const int* walk_k() const { return walk_k_.as<int>(); }         // [7]
  const float* walk_scores() const { return walk_s_.as<float>(); }// [7][16]
  const float* edges() const { return edges_.as<float>(); }       // [7][16][16]
  const float* logits() const { return logits_.as<float>(); }     // [8][Vd] (row 0 unused; Vd = head_rows())
  uint32_t head_rows() const { return draft_vocab(); }             // draft logit columns (vocab or the reduced head)
  const uint32_t* head_ids() const { return head_n_ ? head_ids_.as<uint32_t>() : nullptr; }  // column -> token id
  const float* sel_hidden() const { return sel_.as<float>(); }    // [8][256]
  const float* final_hidden() const { return n_.as<float>(); }    // [8][H] after the last layer

  // ---- debugging / tests ----
  // h0 override: when set, the pass starts from these 8 fp32 rows instead of the embedding.
  void set_h0(const float* rows8xH);
  void clear_h0() { h0_set_ = false; }
  // Per-stage copies (cycle-0 style intermediates) into named buffers; see dflash2.cpp.
  void set_debug(bool on) { debug_ = on; }
  // draft attention kernel: 0 = default (simdgroup-matrix; NI_DFLASH_ATTN=scalar), 1 = scalar, 2 = mma
  void set_attn_variant(int v) { attn_variant_ = v; }
  // prompt context commit: rows <= threshold go through 8-row chunks (-1: NI_DFLASH_PF8, default 256)
  void set_pf8(int rows) { pf8_ = rows; }
  const std::map<std::string, Buffer>& debug_bufs() const { return dbg_; }
  // fp32 c(j) = RMSNorm(x_ctx W_fc^T; hidden_norm) of the last commit: prefill rows [0, rows) or
  // decode rows [0, m) (tests)
  Buffer ctx_c() const { return cdbg_; }
  // the prompt capture rows written by the target hook (bf16 [rows][n_slots * H])
  Buffer prompt_captures() const { return pcap_; }

 private:
  struct DLayer {
    const Tensor *in_w = nullptr, *post_w = nullptr, *qn = nullptr, *kn = nullptr;
    const Tensor *ac_base = nullptr, *mc_base = nullptr;
    QLinear ac, q, kv, o, mc, gate, up, down;  // kv: row view of kv_all_
  };

  void load_config(const std::string& dir);
  const Tensor* own(const Tensor& t, const char* label);
  QLinear quantize(const std::vector<const Tensor*>& parts, const char* label, const std::vector<uint32_t>& perm = {});
  QLinear row_view(const QLinear& src, uint32_t r0, uint32_t n);
  void alloc_buffers();
  // a 16-row block's drafting pass on the tensor path: not the v1 drafter, the matrix units on, 4-bit group-64
  // projections, the fused attention (the load-time gate below refuses NI_DFLASH_BLOCK=16 without it)
  bool rows16_tensor_ok() const;
  // and whether this target and these knobs allow the block at all: trees on, a depth-15 walk (NI_DFLASH_VERIFY >= 15),
  // the target's wide tree verify, and rows16_tensor_ok
  bool rows16_auto_ok() const;

  // dfl_qmm8, or the target's tensor-op GEMM qmm8t (src/kernels/tensor/verify8t.metal) on the X8 table's T table
  // when the target has it on (Qwen35::v8_tw) and every segment is 4-bit group 64 with oscale 1
  // tiles > 1 (16-row blocks): the same GEMM on each following 8-row tile of the X8 table (xh + t * 16 K halfs,
  // xs + t * K / 8 floats) into y + t * 8 y_ld floats (tensor path only)
  void enc_qmm8(Encoder& e, const std::vector<DflSeg8>& segs, const Buffer& xh, const Buffer& xs, uint32_t K,
                uint32_t x_ld, int tiles = 1);
  bool tw_for(const std::vector<DflSeg8>& segs, uint32_t K) const;
  void enc_qmm8_gu(Encoder& e, const QLinear& g, const QLinear& u, const Buffer& xh, const Buffer& xs) {
    dfl_enc_qmm8_gu(e, g, u, xh, xs, hh_, hs_);
  }
  void enc_qmm_rows(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M);  // prefill GEMM
  void enc_commit_kv(Encoder& e, DFlashSeq& d, const Buffer& kv, int rows, int pos0);
  // encode_commit's projections of m rows of `src` (bf16 [8][n_slots * H]; the verify captures capt_ by default);
  // gather: rows gst[4 + r] of src (a tree cycle's accepted path)
  void enc_commit_from(Encoder& e, DFlashSeq& d, int m, const Buffer& src, bool gather = false);
  void encode_draft_impl(Encoder& e, Seq& s, DFlashSeq& d, int P, const SamplingParams& sp, bool gpu,
                         bool tree = false);
  void dbg(Encoder& e, const std::string& name, const Buffer& src, size_t bytes);

  Qwen35& tg_;
  std::string dir_;
  DFlash2Config cfg_;
  int bits_ = 4;
  int block_ = kB;         // kB, or 16 (the ctor picks it before anything is sized)
  int vdepth_ = kB - 1;    // proposals verified per cycle: verify_depth()
  size_t resident_ = 0;
  std::vector<DLayer> layers_;
  QLinear fc_, sel_w_, kv_all_;
  const Tensor *hnorm_ = nullptr, *norm_ = nullptr, *cbA_ = nullptr, *cbB_ = nullptr, *kn_all_ = nullptr;
  std::deque<Tensor> owned_;
  std::vector<Buffer> owned_bufs_;
  Buffer inv_freq_;

  CaptureHook hook_pf_, hook_vf_;
  int pf_n_ = 0, pf_rows_ = 0;

  // pass buffers (8 rows)
  Buffer h_, n_, xh_, xs_, xh2_, xs2_, hh_, hs_, gu_g_, gu_u_, dyn_, q_, kv_, Q_, Kb_, Vb_, part_o_, part_ml_, u_;
  Buffer logits_, sel_, tk_v_, tk_i_, cand_v_, cand_i_, edges_, walk_s_, qprob_, walk_k_, rope_cs_;
  // acceptance: target distributions of the verify rows, result {a, token}
  Buffer dist_id_, dist_p_, dist_keep_, acc_, kdyn_, gst_;
  float dyn_tau_ = 0.0f;    // T = 0 (0: dynamic depth off)
  float dyn_tau_t_ = 0.0f;  // T > 0
  uint32_t dyn_rule_ = 0;   // dfl_depth rule (NI_DFLASH_DYN_RULE)
  bool exact_moe_ = true;  // MoE targets: row-independent expert numerics, tree verifies at every T and the chain's at T > 0 (NI_DFLASH_EXACT_MOE=0 off)
  // KFlash (see tree()): the tree descriptor (int32 [kf::kWords]), the lattice's children tables (clq float
  // [7][16][16], ctok int, the constant successors), the builder's live-row word, the budget
  bool tree_ = false;
  bool gst_tree_ = false;  // the drafter state's rows are a tree acceptance's path (gst[4 ..]): commits gather them
  int tree_rows_ = kB;     // the widest allowed verify (the budget's max rows)
  int tree_cap_ = kB;      // the verify capacity of the cycle being drafted (tree_rows_ unless the context is short)
  static std::string r_list(uint32_t allowed);
  Buffer tree_buf_, kf_clq_, kf_tok_, kf_next_, kf_live_;
  std::unique_ptr<KfBudget> kf_budget_;
  std::unique_ptr<KfBatch> kf_batch_;  // batched decode's rows (the same key + "-batch")
  // T > 0: the walk samples its chain at T_q = walk_tf_ * T (NI_DFLASH_QT; 0 = argmax chain with one-hot q)
  float walk_tf_ = 1.0f;
  // MLP and LM head on the target's 8-row verify GEMMs (Qwen35::enc_gu8 / enc_qmm8 with the verify's V8Knobs;
  // NI_DFLASH_V8, default on for hidden >= 4096, where they measured 6-16% faster than dfl_qmm8): inputs permuted
  // into their table
  bool v8_ = false;
  Buffer v8t_, v8h_, v8hs_;
  // Reduced draft head (NI_DFLASH_HEAD=n rows, 0 = the full vocabulary): the target lm_head rows of the n
  // highest-ranked token ids (a ranked id list from NI_DFLASH_HEAD_IDS, kHeadRows by default; else id order =
  // the tokenizer's BPE merge order, kHeadRowsIdOrder) plus every id >= the first end-of-sequence id
  // (added/special tokens), ascending; the drafter's top-16 then picks among those ids only (the verify always
  // uses the full head, so outputs are unchanged).
  QLinear head_sub_;
  Buffer head_ids_;
  uint32_t head_n_ = 0;  // 0: full head
  void build_head(int n, bool explicit_n);
  const QLinear& draft_head() const { return head_n_ ? head_sub_ : tg_.lm_head(); }
  uint32_t draft_vocab() const { return head_n_ ? head_n_ : uint32_t(cfg_.vocab); }
  void enc_to_v8(Encoder& e, const Buffer& xh, uint32_t K, const Buffer& table);
  void enc_qmm8_cp(Encoder& e, const QLinear& L, const Tensor* base, const Buffer& xh, const Buffer& xs);
  // context commit
  Buffer capt_, y8_, c8_, kv8_, cdbg_;
  Buffer pcap_, xph_, yph_, cph_, kvph_;
  int max_splits_ = 32;  // partial buffers (attention splits)
  // context splits of the draft attention (NI_DFLASH_ATTN_SPLITS, <= max_splits_)
  int attn_splits_ = std::clamp(getenv("NI_DFLASH_ATTN_SPLITS") ? atoi(getenv("NI_DFLASH_ATTN_SPLITS")) : 16, 1, 32);
  bool h0_set_ = false, debug_ = false;
  int attn_variant_ = 0, pf8_ = -1;
  std::map<std::string, Buffer> dbg_;
};

} // namespace ni
