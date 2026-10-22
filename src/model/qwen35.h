// Qwen3.5 / 3.6 / 3.8 hybrid model (Gated DeltaNet + gated full attention, 3:1), dense or MoE,
// MLX affine-quantized safetensors.
#pragma once
#include <atomic>
#include <thread>
#include "loader/safetensors.h"
#include "metal/gpu.h"
#include "model/model.h"
#include "model/moe_q20.h"
#include "vision/embed.h"

#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ni {

namespace ane {
class FfnSplit;
struct FfnConfig;
struct FfnLayer;
}

// A linear layer's weights. Formats (src/kernels/wfmt.metal): MLX affine `bits`-bit codes (2-8; uint32 words, a
// little-endian bit stream along each row) with bf16 scale and bias per group of `gs` inputs, or bits = 16: bf16
// weights [N, K] with no scales / biases (s and b then point at w, so buffer bindings stay valid; no kernel
// reads them). Routed experts may also be ggml's Q2_0 (q20; src/model/moe_q20.h): 2-bit codes in MLX's order with an
// fp16 d per 64 inputs in s and no biases (b points at s), weight = (code - 1) d.
struct QLinear {
  const Tensor* w = nullptr;
  const Tensor* s = nullptr;
  const Tensor* b = nullptr;
  uint32_t N = 0, K = 0;
  int bits = 4, gs = 64;
  bool q20 = false;   // Q2_0 (bits 2, gs 64): routed experts of Qwen3.8-Flash-Next only
  bool mtp = false;   // part of the MTP head (8-bit prefill GEMM on qmr8 since it was introduced)
  bool gate = false;  // the trunk's MoE router / shared-expert gate: their 8-bit prefill GEMMs keep the old kernels
  // Routed experts of a GGUF import kept in llama.cpp's blocks (src/kernels/wfmt_iq.metal): their GGUF type
  // (gguf::iq_gpu_type's; bits 1-3, gs 256, s and b point at w), 0 otherwise. w is the stacked [E, N, K] matrix in the
  // kernels' layout, its sections at these byte offsets (Q at 0).
  uint32_t iq = 0;
  uint32_t iq_x1 = 0, iq_x2 = 0, iq_x3 = 0, iq_d = 0;
  explicit operator bool() const { return w != nullptr; }
  bool dense() const { return bits == 16 && !iq; }
  bool q4() const { return bits == 4 && gs == 64 && !iq; }  // the specialized 4-bit kernels
  bool q8() const { return bits == 8 && gs == 64 && !iq; }  // the specialized 8-bit kernels
  // kernel-name suffix of the format: "f16" (bf16), "f<bits>g<gs>", "q20", or iq1s / iq2xxs / iq2s / iq3xxs / iq3s
  std::string fmt() const;
  std::string name() const;  // "4-bit group 64", "bf16", "IQ2_XXS", "Q2_0", ...
  bool same_format(const QLinear& o) const {
    return iq == o.iq && q20 == o.q20 && bits == o.bits && (bits == 16 || gs == o.gs);
  }
  size_t row_bytes() const;  // weight bytes per row (IQ: the blocks', d included)
  size_t group_row_bytes() const { return bits == 16 || iq ? 0 : size_t(K / gs) * 2; }  // scales (or biases) per row
};

// Formats with a full set of kernels: 4-bit and 8-bit group 64 (specialized), and the generic ones of wfmt.metal.
bool format_supported(int bits, int gs);
std::string format_name(int bits, int gs);  // "bf16", "4-bit group 64", ...
// The per-row MoE kernel (gate/up or down) for routed experts e and shared expert s, "" if none exists.
// wide: the experts' staged h exceeds 4608 floats (Qwen35::moe_wide), so the down kernel's _w form ("" if none).
std::string moe_rows_kernel(bool gate_up, const QLinear& e, const QLinear& s, bool wide = false);

struct Qwen35Config {
  int hidden = 0, layers = 0, vocab = 0;
  std::vector<bool> is_linear;
  // Gated DeltaNet
  int lin_hk = 0, lin_hv = 0, lin_dk = 0, lin_dv = 0, conv_k = 4;
  // full attention
  int hq = 0, hkv = 0, head_dim = 0, rot_dims = 0;
  float theta = 1e7f, eps = 1e-6f;
  int mrope[3] = {11, 11, 10};  // M-RoPE (images): rotary frequencies per (t, h, w) axis, interleaved
  // MLP
  bool moe = false;
  int experts = 0, top_k = 0, moe_I = 0, shared_I = 0, dense_I = 0;
  bool norm_topk = true;
  bool iq = false;  // routed experts in llama.cpp's IQ blocks (GGUF imports): their own MoE kernels (qwen35_iq.cpp)
  int hc = 1;       // residual streams: Qwen3.8-Flash-Next's hyper-connections keep 4 (text_config.hc_count)
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


// Timings of one split layer (ANE split benchmark / calibration), ms per layer.
struct AneLayerTimes {
  double gpu_only = 0;   // the FFN on the GPU alone (as without NI_ANE)
  double gpu_part = 0;   // the split's GPU part alone: pack, own channels, staging, join
  double ane = 0;        // the ANE alone (synchronous evaluation)
  double split = 0;      // both, concurrently
  double ane_in_split = 0;  // ANE time inside the split (event reached ready -> report)
};

// Runtime knobs of the 8-row verify kernels (src/model/qwen35_v8.cpp). Defaults
// are the measured-fastest settings; the environment (NI_V8_CM, NI_V8_PAIR, NI_V8_KPLAN, NI_V8_GUSPLIT,
// NI_V8_GDNSPLIT, NI_V8_PREPU) overrides them, and tools may change them between encodes: ni_run's NI_AB alternates
// two settings per decode cycle so both are timed under the same GPU conditions.
struct V8Knobs {
  int cm = 1;       // GEMM operands: 0 = codes as 1024 + q, 1 = codes as exact half subnormals q * 2^-24
  int pair = 1;     // 1: two 64-groups per inner step (scale / bias pairs in one load each)
  int kplan = 2;    // 1: 8 k-slices x 1 column block per threadgroup when K >= 4096; 2: also 16 k-slices for K >= 16384;
                    // 3 (the default on the M1): k-slices of >= 16 groups, at most 16384 simdgroups (v8_plan)
  int gusplit = 1;  // 1: dense MLP gate / up as two segments of one GEMM (fp32) + swiglu_x8, instead of qmm8_gu1k2
  int gdnsplit = 256;  // GDN step over value-row blocks of this many threads + gdn_gnorm_x8 (0: one threadgroup per head)
  int prepu = 1;    // 1: x8_prep2's row norms with 4 independent load chains
  // timing only (wrong outputs): skip kernel groups of a dense verify / drafting pass. Verify: 1 GDN step, 2 attention
  // core, 4 GDN prep, 8 MLP down, 16 MLP gate/up, 32 GDN in_proj, 64 GDN out / attention o, 128 LM head + sampling,
  // 256 normed X8 preps, 512 attention q|k|v; drafter: 1024 MLP gate/up, 2048 MLP down, 4096 LM head,
  // 8192 conv / attention projections, 16384 ring commit, 32768 attention, 262144 attention reduce. With NI_AB: the
  // in-pipeline cost of those kernels.
  int skip = 0;
  int pad = 0;  // timing only: this many empty dispatches after every verify GEMM (the per-dispatch cost, with NI_AB)
  int tw = 1;   // the dense 4-bit verify GEMMs on the GPU's matrix units (qmm8t; Apple10 = M5 and macOS 26, else off)
  int twm = 1;  // with tw: the dense MLP's gate|up and down GEMMs on qmm8t as well (NI_V8_TWM; 0 = the MMA kernels)
  // fused verify passes (verify8_fuse.metal, bitwise the passes they replace): 1 = MoE combine + the next norm's X8
  // table (and qmm8t's T table with tw), 4 = GDN prep with the rows' norms reduced together (the MoE top-k + plan
  // fusion is moe8_plan's own: tkp)
  int fuse = 5;
  int emu = -1;  // MMA-exact small-expert passes (NI_V8_MOE_EMU, see enc_moe8): -1 = the environment's, else 0-2
  int tkp = -1;  // NI_V8_MOE_TOPKPLAN: the router top-k inside moe8_plan (-1 = the environment's)
  int shs = -1;  // NI_V8_MOE_SHS: the shared expert as small tiles (with split 8; -1 = the environment's)
  int split = -1;  // NI_V8_MOE_SPLIT: experts with 3..split rows as 2-row small tiles (-1 = the environment's)
  int dfl = 0;  // drafter experiments (DFlash2::encode_draft_impl): variant bits, A/B'd per cycle with NI_AB
};
V8Knobs& v8_knobs();
// "key=value,..." over a copy of `base` (keys: cm, pair, kplan, gusplit, gdnsplit, prepu, skip, pad, tw, twm, fuse, emu, shs,
// split, tkp, dfl); unknown keys are fatal
V8Knobs v8_knobs_parse(const char* spec, V8Knobs base);

class Qwen35 : public Model {
 public:
  explicit Qwen35(const std::string& model_dir, int prefill_chunk = 2048);
  ~Qwen35() override;
  const Qwen35Config& config() const { return cfg_; }
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }

  // ---- GPU + Neural Engine split of the dense prefill FFN (opt-in NI_ANE=1; qwen35_ane.cpp) ----
  // Whether a prefill chunk of T rows would run the split (callers checkpoint the recurrent state
  // before such a chunk so it can be rerun on the GPU alone if the split fails).
  bool ane_splits(int T) const override;
  bool ane_split_on() const override { return ane_ && !cfg_.moe; }
  // After the command of a chunk encoded with the split has completed: false if the split failed
  // (it is then off for the rest of the process) and the chunk must be rerun. True otherwise.
  bool ane_finish_chunk() override;
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
  using AneStats = ni::AneStats;
  const AneStats& ane_stats() const override { return ane_stats_; }
  // NI_ANE_ASYNC (the server): finish the Neural Engine split once its background compile is done.
  // Call between requests (it runs the startup check through the model's buffers).
  void ane_poll() override;
  // Frees the split's memory for good (prefill on the GPU alone from then on): a conversation's KV cache
  // needs the room. False if there was no split.
  bool ane_release() override;

  // ---- memory ----
  // A sequence of up to `limit` positions. Its attention caches hold `initial` positions (default: all)
  // and grow with reserve(); 0 allocates them on first use.
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  // Makes s's attention caches hold `need` positions (<= s.limit): new buffers, a GPU copy of the old
  // positions, the old buffers released, in batches so the overlap stays small. Grows by at least a
  // quarter (in 1024-position steps) when the memory budget allows, else exactly. Synchronous: no
  // command buffer that uses s may be in flight. False (nothing changed) if the budget does not allow it.
  bool reserve(Seq& s, int need) override;
  // Frees s's attention caches (cap 0; positions must be prefilled again). Nothing using s in flight.
  void drop_kv(Seq& s) override;
  // Bytes of attention cache per position of a sequence (every attention layer and the MTP head, K + V)
  size_t kv_bytes_per_pos() const override;
  // Upper limit for Metal allocations (bytes) that reserve() and the Neural Engine split respect; 0 = none
  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  size_t mem_budget() const { return mem_budget_; }
  // What a model needs, from config.json and the safetensors headers only (no weights read): for fit
  // checks before loading and `kapitsa doctor`. Bytes.
  using MemPlan = ni::MemPlan;
  // rows16: the engine will draft with a 16-row block (one slot), whose verify rows the live state's tape is sized for
  static MemPlan plan(const std::string& model_dir, int prefill_chunk = 2048, bool rows16 = false);
  static Qwen35Config read_config(const std::string& model_dir);  // config.json only
  // The Neural Engine split this Mac would build for the model (dense FFN): its share in units of 512 channels
  // (0 = none), where it comes from, and its memory (surfaces + the engine's rerun checkpoint), as init_ane decides.
  struct AnePlan {
    uint32_t units = 0, total = 0;
    std::string source;  // "calibration" or "default for this chip"; why not, when units == 0
    size_t bytes = 0;
  };
  static AnePlan ane_plan(const std::string& model_dir, const MemPlan& m);
  // Images: the M-RoPE positions of a prompt of n rows whose images sit at `spans` (Qwen's get_rope_index: text rows
  // count up one per row, an image's rows take (t, h, w) = start + (frame, row, column) of its token grid, and the text
  // after it continues at start + max(grid_h, grid_w)), into s for every pass that follows (prefill, decode, verify,
  // MTP). No spans: positions = rows again. Call between requests (nothing in flight may read s).
  void set_rope_positions(Seq& s, int n, const std::vector<vision::ImageSpan>& spans) override;
  // The positions themselves: (t, h, w) of rows [0, end of the last image) into `table`; returns the delta of the
  // rows after it (position = row + delta)
  static int rope_positions(const std::vector<vision::ImageSpan>& spans, std::vector<int32_t>& table);
  // Clears recurrent state (KV entries are overwritten by position, so they need no reset).
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override;
  // GPU copies, ordered with the surrounding work in `e`.
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  // SSD prefix cache: the attention layers' K and V caches (then the MTP head's), the GDN layers' conv and recurrent
  // states
  std::string disk_format() const override;
  void disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const override;
  void disk_state(const Seq& s, const Checkpoint* c, std::vector<Buffer>& out) const override;
  // The ids the MTP draft head gathered for s (NI_DRAFT_CTX), in their order, and how far it scanned s's tokens; and
  // the same rows gathered again into another sequence (empty: none), as a restored state needs them for the same drafts
  std::vector<uint32_t> draft_vocab_ids(const Seq& s, int* scanned) const;
  void draft_vocab_restore(Seq& s, const std::vector<uint32_t>& ids, int scanned);
  bool copy_prefix(Encoder& e, const Seq& src, Seq& dst, int n) override;

  // Encodes one decode step for the token at tokens[pos]. With `sample`, computes logits and
  // writes the sampled id to tokens[pos+1]; without, only advances the state (prompt tokens).
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  // Encodes T prompt tokens tokens[pos0 .. pos0+T) as one batch (T <= prefill_chunk).
  // With `sample`, also samples tokens[pos0+T] from the last position.
  // With `snap` (requires can_snapshot(), 0 < snap_row < T): also saves the recurrent state as it
  // stands after the first snap_row tokens into `snap` (pos = pos0 + snap_row), inside the pass.
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  int prefill_chunk() const override { return tmax_; }
  // NI_SHORTPF: checkpoints can be captured inside a prefill pass (encode_prefill's `snap`).
  bool can_snapshot() const override;

  // ---- speculative decoding with the model's MTP head ----
  bool has_mtp() const { return mtp_loaded_; }
  // bits of the MTP head's bf16 linears as loaded (and planned): 4 on dense models (NI_MTP_BITS=8: 8), 8 on MoE ones
  static int mtp_head_bits(bool moe);
  // Measured best MTP draft depth (M4 Pro): MoE 35B-A3B 1 (its verify reads the union of the rows' experts), dense
  // models 4. Dense verifies of 3-8 rows run the 8-row MMA path (~1.1-1.25x a decode step) and a draft pass costs
  // ~0.5 ms (4B) to ~4 ms (27B); depth 4 is best at 4-bit and at bf16 alike (the bf16 step costs 3.3x, but the drafts'
  // acceptance falls off by depth 5). Greedy, code / prose, 2-3 alternations: 4B 4-bit depth 1 127.8 / 126.8, 2
  // 149.0 / 142.1, 3 153.7 / 147.8, 4 157.6 / 147.5, 5 155.6 / 142.4 tok/s (512 tokens); 4B bf16 depth 1 45.8 / 43.0, 2
  // 60.7 / 53.7, 3 65.9 / 56.2, 4 69.4 / 58.1, 5 69.7 / 57.6, 6 70.0 / 57.2 (256); 27B 4-bit depth 2 32.7 / 31.0, 4
  // 39.3 / 34.9, 6 40.0 / 35.1 (128 tokens; depth 2 on the small-batch kernels: 29.1).
  virtual int default_spec_depth() const { return cfg_.moe ? 1 : 4; }
  // Rows per verify (pending token + up to 7 drafts). R <= 2 (and R <= 4 on MoE models) runs the small-batch
  // kernels; more rows (NI_V8=1: from 2; NI_V8=0: from 5) the 8-row MMA path (qwen35_v8.cpp).
  static constexpr int kMaxVerify = 8;
  // KFlash tree verifies: up to 32 rows (whole 8-row tiles; verify_wide_ok, set_tree_rows)
  static constexpr int kMaxTreeRows = 32;
  static constexpr int kMaxMtpRows = 8;  // MTP rows per encode_mtp (history rows of a cycle: depth + 1); depth <= 7
  // MTP rows: pairs (hidden row h_row0 + r of `h`, token tokens[tok0 + r]) at MTP positions
  // mpos0 + r; they extend the MTP cache. The last row's post-norm hidden is left in mtp_post()
  // row R-1; with draft_out >= 0 the draft for the next token goes to tokens[draft_out].
  // h_trunk: `h` holds trunk hidden states (pre-norm); false for the MTP's own chained output.
  virtual void encode_mtp(Encoder& e, Seq& s, int mpos0, int R, int tok0, const Buffer& h, int h_row0, int draft_out,
                          bool h_trunk = true);
  // Trunk forward of tokens[pos .. pos+R) (R <= kMaxVerify) without committing the recurrent
  // state: the rows go to the tape; samples() gets one sampled token per row and hidden_rows()
  // their pre-norm hidden states. commit_rows() then picks how many of them were accepted.
  // v8: take the 8-row path for any R >= 1 (a block drafter verifying fewer than 4 proposals).
  virtual void encode_verify(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step,
                             bool v8 = false);
  // Batched decode: one 8-row verify pass over the rows of several sequences. Group g verifies tokens[pos .. pos + R)
  // of its sequence (folding its pending rows first, as encode_verify does); the projections, the MLP / MoE and the
  // LM head run once over all the rows (sum of R <= 8), the per-sequence parts (GDN conv, recurrence and tape;
  // attention with its KV cache) per group. Group g's rows are rows [r0, r0 + R) of samples() (r0 = the rows of the
  // groups before it), each sampled with its group's parameters (RNG step: the row's position). With `cap`, the
  // drafter's target layer outputs of the group's rows go to cap->dst rows [cap_row0, cap_row0 + R) (cap->ld,
  // cap->slot; its positions are not used). Afterwards commit_rows() per sequence, as after encode_verify.
  // split_after > 0: the pass's first split_after layers are committed as a command buffer of their own (Encoder::split)
  // as soon as they are encoded, so the GPU starts on them while the host encodes the rest (a pass encoded only once the
  // GPU is idle: the batch budget's, whose rows follow the drafting passes).
  struct VerifyGroup {
    Seq* s = nullptr;
    int pos = 0, R = 1;
    SamplingParams sp;
    const CaptureHook* cap = nullptr;
    int cap_row0 = 0;
    // the samples go into the sequence's own tokens (row j's to tokens[pos + 1 + j]) instead of samples(), where the
    // next pass can read them before the host has (pipelined plain rows)
    bool tok_out = false;
  };
  void encode_verify_multi(Encoder& e, const std::vector<VerifyGroup>& groups, int split_after = 0);
  void commit_rows(Seq& s, int n) { s.n_pend = n; }
  // A chain pass: plain rows [pos, pos + R) verified as the verify's rows (row t at position pos + t, its state its
  // predecessors' fold), for a cached turn's pending suffix. R <= 8 is the 8-row verify with no tree; R = 16, 24 or 32 a
  // wide verify over a chain tree descriptor (a target of verify_wide_ok with the capacity for it: pass_rows). Every row is a real
  // token, so commit_rows(s, R) after each pass folds them all; a pass streams the weights once, the point of it.
  void encode_verify_chain(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp);
  // rows is a whole number of 8-row tiles of the verify capacity (8, 16, 24 or 32): the wide path reads the tile
  // pair of an odd tile count one tile past its last, which is inside the capacity's tables
  bool verify_chain_ok(int rows) const {
    return rows <= 8 || (rows % 8 == 0 && rows <= verify_rows_max() && verify_wide_ok());
  }
  // The rows the verify buffers hold: the capacity set_tree_rows sizes (the drafter's tree width) or the wider chain
  // pass a cached suffix may run (pass_rows_, sized at load from vpf_pass_rows)
  int verify_rows_max() const { return tree_rows_ > pass_rows_ ? tree_rows_ : pass_rows_; }
  // Folds pending rows into the recurrent state (before plain decode/prefill or a checkpoint).
  virtual void encode_flush(Encoder& e, Seq& s);
  // MTP history for prompt rows [pos0, pos0 + R) using the trunk hidden states of the last
  // prefill chunk (row offset h_row0 in hidden_rows()); tokens[i + 1] must be known.
  virtual void encode_mtp_prefill(Encoder& e, Seq& s, int pos0, int R, int h_row0);
  const int* samples() { return samp_.as<int>(); }
  // MoE models, after an 8-row verify: distinct routed experts of each layer (0 for dense layers)
  const uint32_t* verify8_experts() { return v8ustat_ ? v8ustat_.as<uint32_t>() : nullptr; }
  Buffer hidden_rows() { return x_; }
  // bytes of one hidden_rows() row as encode_mtp reads it (Qwen3.8-Flash-Next's rows hold its 4 streams)
  virtual size_t hidden_row_bytes() const { return size_t(cfg_.hidden) * 4; }
  Buffer mtp_post() { return mtp_post_; }
  // GPU copy of src into dst (the smaller of the two sizes, a multiple of 16 bytes), in order with the work in `e`
  // (the engine keeps a sequence's MTP hidden rows aside while another request's passes reuse hidden_rows())
  void encode_copy(Encoder& e, const Buffer& src, const Buffer& dst) { enc_copy(e, src, dst); }
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
  const std::string& model_dir() const { return dir_; }
  const QLinear& lm_head() const { return lm_head_; }
  // Drafter's dynamic verify depth (DFlash2, MoE only): while set, the 8-row verify routes experts only
  // for rows < min(R, rows[0]) (uint read by the GPU); later rows' outputs are garbage and must not be used.
  void set_verify_row_limit(const Buffer* rows) { v8_rlim_ = rows; }
  // Row-independent MoE numerics in the 8-row verify (every routed expert as an MMA tile, no scalar
  // passes for 1-2-row experts): a row's logits then depend only on the rows before it, not on how the
  // other rows route. A speculative sampler that needs p(. | prefix) exactly sets it (~3% slower verify).
  void set_verify_moe_exact(bool on) { v8_moe_exact_ = on; }
  // KFlash tree verifies: while set, the 8-row verify's (or a wide one's: verify_wide_ok) rows are a tree (tree = the
  // KFlash tree descriptor, src/kernels/kflash.h: parents, depths, ancestor-or-self masks): row t's token takes position
  // pos + depth(t) (RoPE,
  // its sample's uniform), its attention sees the history and its ancestors' rows, its GDN conv window and recurrent
  // state are its ancestors' fold. The rows' K/V stay in cache rows pos + t and their tape in rows t until the
  // acceptance (kf_accept) writes the accepted path to s.pend_path, moves its K/V (encode_kv_tree_compact) and sets
  // s.pend_tree: the next pass (or encode_flush) folds tape rows pend_path[0 .. n_pend).
  void set_verify_tree(const Buffer* tree) { v8_tree_ = tree; }
  // the 8-row verify can run a tree (its attention on attn8m, the split GDN step)
  bool verify_tree_ok() const;
  // Tree verifies wider than 8 rows (16 or 32: 2 or 4 tiles of 8 rows; KFlash's size rule picks the rows per cycle):
  // encode_verify with a tree set and R = 16 or 32 encodes that capacity, and the GPU runs the tiles of the tree's live
  // rows (8 ceil(n / 8), n = the descriptor's kKfN, chosen by the drafting pass after the host encoded the verify):
  // every grid that scales with the rows is dispatched indirectly from args the verify's first kernel writes
  // (v8_tree_plan), the GEMMs run a column block's tiles side by side (one weight read from DRAM), the rows' attention,
  // GDN conv windows and recurrent state are their ancestors' as in the 8-row tree. Dense targets whose verify runs the
  // tree (verify_tree_ok), the split MLP and the X8 preps of x8_prep2 (widths multiple of 256); MoE targets keep 8.
  bool verify_wide_ok() const;
  // The KFlash tree row counts a dense target's verifies may pick, bit n-1 set for n = 8, 16, 32: 8 alone, or 8 with 16
  // and 32 where the target runs wide verifies (verify_wide_ok) and the drafting pass's knobs (NI_KFLASH,
  // NI_DFLASH_TREE, NI_KFLASH_ROWS) let them. DFlash2 picks its rows from it and plan() reserves the tape they need, so
  // the sizes agree; a bad NI_KFLASH_ROWS is fatal here (before either).
  static uint32_t kf_dense_rows(const Qwen35Config& c);
  // The rows a cached suffix's chain pass runs (NI_VPF_ROWS, else the measured default: 32 on a dense target at or
  // above the weight floor the NI_VPF default uses, 8 elsewhere): plan() reserves the tape for them and the model
  // sizes its verify buffers for them at load. Kept apart from kf_dense_rows, which is the drafter's tree width: the
  // pass is the engine's own, not a decode cycle's, so it may be wider than anything the drafter drafts.
  static int vpf_pass_rows(const Qwen35Config& c);
  static constexpr long long kPassWeightFloor = 3000000000LL;  // hidden * dense intermediate * layers
  // The widest tree verify to prepare for (8, 16 or 32 rows): the verify's model-wide buffers (X8 tables, logits,
  // candidates, samples) are sized for it here, and a sequence's verify tape is allocated at it in new_seq (the tape of
  // one created before this call grows on its first wide verify). Call while nothing is in flight (the drafter's
  // construction).
  void set_tree_rows(int rows, bool drafting = false);
  int tree_rows() const { return tree_rows_; }
  // The widest tree verify the drafter itself asked for (set_tree_rows with drafting): its drafting block, where the
  // target's own chain passes may run wider (pass_rows). NI_VPF's default is measured on a target whose drafter drafts
  // 16 rows, so the default keys on this and not on the capacity the pass raised.
  bool drafter_wide() const { return drafter_wide_; }
  // The rows one chain pass of a cached suffix runs (8, 16 or 32: vpf_pass_rows at load; a 24-row tail of a ladder
  // still runs inside the 32-row capacity).
  int pass_rows() const { return pass_rows_; }
  // After a tree verify at pos: the K/V rows of the accepted path (the descriptor's, kf_accept) move to positions
  // pos + i (every attention layer)
  void encode_kv_tree_compact(Encoder& e, Seq& s, int pos, const Buffer& tree);
  // dev (tools/bench_verify8): the last verify's MoE expert passes again with layer li's weights (timing only)
  void moe8_replay(Encoder& e, int li, int mask);
  // The 8-row verify GEMMs, for other 8-row passes (the DFlash2 drafter). Inputs are X8 tables
  // (verify8.metal layout: half operands in fragment order + per-64 group sums), R <= 8 live rows.
  struct Seg8 {
    const QLinear* L;
    Buffer y;
    uint32_t y_ld = 0;  // 0: N
    bool add = false;   // y += X W^T
  };
  // y_i[R, N_i] = X W_i^T for up to 4 matrices sharing K and bits (x2: two-term hi+lo inputs). tt: the same
  // inputs as a T table (enc_x8 with tw), for qmm8t on the GPU's matrix units (V8Knobs::tw; 4-bit group 64)
  void enc_qmm8(Encoder& e, const std::vector<Seg8>& segs, int R, const Buffer& table, const Buffer& sums,
                bool x2 = false, const Buffer* tt = nullptr);
  // h = half(silu(X Wg^T) * (X Wu^T)) written as the down projection's X8 table h + sums hs (4-bit, N % 64 == 0)
  // tt: the inputs' T table (qmm8t for gate|up); ht: also write h as a T table there (split path only). Returns
  // whether ht was written. R: the rows (a wide tree verify's capacity: its row tiles' tables)
  bool enc_gu8(Encoder& e, const QLinear& gate, const QLinear& up, const Buffer& table, const Buffer& sums,
               const Buffer& h, const Buffer& hs, const Buffer* tt = nullptr, const Buffer* ht = nullptr, int R = 8);
  // The last verify's sampler inputs and outputs, as GPU buffers (for an on-GPU acceptance step):
  // stage-1 top-k candidates per row (values / ids, verify_candidates() per row) and the sampled ids.
  Buffer verify_cand_vals() const { return cand_v_; }
  Buffer verify_cand_ids() const { return cand_i_; }
  Buffer verify_samples() const { return samp_; }
  // candidates per verify row for these sampling parameters (stage-1 chunks x top-k)
  uint32_t verify_candidates(const SamplingParams& sp) const;

  // Logits of rows [0, T) of the last prefill chunk (teacher-forced scoring, tools/ni_eval): final norm + LM head
  // on the prefill GEMM -> out [T, vocab] fp32.
  virtual void encode_logits_rows(Encoder& e, int T, const Buffer& out);
  // Token embedding rows tokens[tok_off .. tok_off + T) -> out [T, hidden] (fp32), any embedding format.
  void enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int T, const Buffer& out);
  // The weight formats this model runs (log line), e.g. "bf16" or "4-bit group 64 (8-bit: router, shared gate)"
  virtual std::string formats_summary() const;
  // The block drafter (DFlash2) binds to this layout (Qwen3.5-family trunks; not Qwen3.8-Flash-Next's)
  virtual bool block_drafter_ok() const { return true; }

 protected:
  // Subclasses (Qwen4Exp) that load the checkpoint themselves: members only, nothing read
  struct NoLoad {};
  Qwen35(const std::string& model_dir, int prefill_chunk, NoLoad);
  void check_shapes() const;
  bool moe_wide() const;
  void load_config(const std::string& dir);
  struct ActBuf {
    Buffer Qwen35::*member;
    size_t bytes;
    const char* label;
  };
  static std::vector<ActBuf> act_layout(const Qwen35Config& c, int T, int max_verify, bool mtp, size_t hgu_rows);
  size_t dense_hgu_rows() const;
  void ensure_hgu(size_t bytes);
  // q20_ok: the linear may be Q2_0 (routed experts of Qwen4Exp, whose MoE paths have the Q2_0 kernels)
  QLinear qlin(const std::string& name, bool required = true, bool q20_ok = false);
  // A layer's Q2_0 routed experts as stacked planes, its 4-bit shared expert, the MoE's shapes (src/model/moe_q20.h)
  q20::Experts q20_experts(const Layer& L) const;
  q20::Shared q20_shared(const Layer& L) const;
  q20::Shape q20_shape() const;
  // A layer's Q2_0 experts for a prefill chunk's MoE (T rows), once the dispatches writing its routing are encoded (ids:
  // n expert ids, the rows' top-k): resident experts are the stacked planes; Qwen4Exp's streamed ones encode the expert
  // store's handshake around `between` (work that does not need the experts, overlapping the store's turn) and come
  // through the arena's slot table. `between` is encoded either way.
  virtual q20::Experts q20_fetch(Encoder& e, const Layer& L, const Buffer& ids, uint32_t n, int T,
                                 const std::function<void()>& between);
  // the Q2_0 kernels' slot-table twins run (prewarm list)
  virtual bool q20_arena() const { return false; }
  std::string tname(const std::string& canonical) const;
  const Tensor* bf16_copy(const Tensor* t, float offset = 0.0f);
  void check_formats() const;
  void test_transforms();  // NI_DEQUANT / NI_REQUANT (tests: other formats from a 4-bit checkpoint)
  WeightStore::Filter transform_defer() const;  // the tensors test_transforms converts: read only then
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
  // NI_QMVF (default on): R <= 2 token rows of up to 4 matrices sharing x (4- or 8-bit group 64, one width) in one
  // qmvf.metal dispatch, an RMSNorm of x folded in (xn_out: also the normalized rows, [R, K]), add[i]: y += ...;
  // gu: segments 0 / 1 are gate / up and segs[0].y gets silu(gate) * up. x_ld / y_ld: 0 = K / each matrix's N.
  // b6: one-row decode matvecs, which may also take 6-bit group 64 (IQ models on the M1; enc_qmv_multi).
  // False (nothing encoded) when off or not applicable.
  bool enc_qmvf(Encoder& e, const std::vector<Seg>& segs, const Buffer& x, int R, const Tensor* norm,
                const Buffer* xn_out = nullptr, const std::vector<bool>& add = {}, uint32_t x_ld = 0,
                uint32_t y_ld = 0, bool gu = false, bool b6 = false);
  void enc_gdn(Encoder& e, Seq& s, int li, const Layer& L);
  void enc_attn(Encoder& e, Seq& s, int li, const Layer& L, int pos);
  void enc_moe(Encoder& e, const Layer& L);
  // prefill
  void enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T, int D);
  void enc_qmm(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add);
  // the large-prefill kernel choice (qmr, else the original qmm tiles), no short-prefill paths
  void enc_qmm_large(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add);
  // the prefill GEMM of wfmt.metal for formats without a specialized one (qrf_*)
  void enc_qrf(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add);
  // NI_QMM + NI_QMM_GU: h = half(silu(xh G^T) * (xh U^T)) in one dispatch; false when off/unsupported
  bool enc_qmm_gu(Encoder& e, const QLinear& G, const QLinear& U, const Buffer& xh, const Buffer& h, int M);
  void enc_qmm_moe(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, const Buffer& xrows,
                   const Buffer& yrows, int P, int bm);
  void enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp, uint32_t step);
  void enc_copy(Encoder& e, const Buffer& src, const Buffer& dst);
  // xh_ready: xh_ already holds the layer's half input rows (Qwen4Exp: the gated residual's mix), x only receives the
  // output (x += moe); else x is normed into xh_ first
  void enc_moe_prefill(Encoder& e, const Layer& L, int T, const Buffer& x, bool xh_ready = false);
  void enc_attn_prefill(Encoder& e, const Buffer& kc, const Buffer& vc, int T, int pos0, int cap);
  bool enc_moe_prefill_fused(Encoder& e, const Layer& L, int T, const Buffer& x, bool xh_ready = false);
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
  bool enc_moe_prefill_small(Encoder& e, const Layer& L, int T, const Buffer& x, bool xh_ready = false);
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
  // the same on the tensor_ops GEMMs (NI_QMT, 4-bit group 64); false: not applicable
  bool enc_mlp_gpu_part_qmt(Encoder& e, const Layer& L, int T, uint32_t g, const Buffer& x);
  static bool qmt_ready();  // NI_QMT's tensor_ops GEMMs are on
  // the M5 default share of the split (default_units' gpu_mma) applies to this model: measured on it (NI_ANE_MMA)
  static bool mma_share_measured(const Qwen35Config& c);
  void enc_mlp_prefill_split(Encoder& e, int li, const Layer& L, int T, const Buffer& x);
  void enc_mlp_rows(Encoder& e, const Layer& L, int R, const Buffer& x);
  QLinear own_qlin(const class WeightStore& w, const std::string& name);
  // speculative decoding helpers (qwen35_spec.cpp)
  void load_mtp(const std::string& dir);
  QLinear quantize8(const Tensor& w, int bits = 8);
  // rows [r0, r0 + n) of a linear in any format -> an owned 4-bit group-64 copy (CPU, MLX rounding)
  QLinear quantize_rows4(const QLinear& src, uint32_t r0, uint32_t n);
  const Tensor* own_copy(const Tensor& t);
  // A float parameter the kernels read as bf16 (norms, conv, A_log, dt_bias): t itself when it is
  // bf16, else an owned bf16 copy (++lossy when a value is not a bf16 number)
  const Tensor* bf16_param(const Tensor* t, int& lossy);
  QLinear stack_experts(const class WeightStore& w, const std::string& prefix, const std::string& proj);
  void enc_rows_norm(Encoder& e, const Buffer& x, const Tensor* w, const Buffer& y, int R, int D);
  // block-drafter capture of rows [0, T) of x (default x_) at positions pos0.. after layer `layer` (no-op
  // unless armed); x must hold the layer's final output for those rows (mixer and FFN added)
  void enc_capture(Encoder& e, int layer, int pos0, int T, const Buffer* x = nullptr);
  int v8_rows_min(bool mtp = false) const;
  void enc_qmvr(Encoder& e, const QLinear& L, const Buffer& x, const Buffer& y, int R, bool add,
                uint32_t x_ld = 0, uint32_t y_ld = 0);
  // fused (norm +) projections of R rows sharing the input; falls back to separate kernels
  void enc_qmvr_multi(Encoder& e, std::vector<Seg> segs, const Buffer& x, const Tensor* norm, int R,
                      const Buffer* xn_out = nullptr, std::vector<bool> add = {});
  void enc_attn_rows(Encoder& e, const Seq& s, const Layer& L, const Buffer& kc, const Buffer& vc, int cap, int pos,
                     int R, const Buffer& x);
  // attn_prep_m's arguments beyond attn_prep's (a sequence with images: M-RoPE positions, buffers 9 and 10)
  static bool mrope(const Seq& s) { return s.rope_n > 0 || s.rope_delta != 0; }
  void enc_mrope_args(Encoder& e, const Seq& s);
  // in: the rows' (already normalized) input instead of norm(x); it must be xrn_ (Qwen4Exp)
  void enc_moe_rows(Encoder& e, const Layer& L, int R, const Buffer& x, bool mtp_layer, const Buffer* in = nullptr);
  // ---- 8-row verify path (qwen35_v8.cpp) ----
  void v8_alloc();
  void encode_verify8(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step);
  // x rows (fp32, x_ld floats apart) -> X8 table (+ group sums), optional RMSNorm, optional lo table.
  // tw: also the T table v8tt_ of qmm8t when that path is on (v8_tw()); returns whether it was written.
  bool enc_x8(Encoder& e, const Buffer& x, int R, int K, const Tensor* norm, bool lo = false, int x_ld = 0,
              const Buffer* xn_out = nullptr, bool tw = false);
  // the 8-row GEMMs on the matrix units (qmm8t, src/kernels/tensor/verify8t.metal): an M5 and V8Knobs::tw
  bool v8_tw() const;
  // wide tree verifies (verify_wide_ok): the descriptor region and the plan kernel at the start of one, the growth of
  // a sequence's tape to kMaxTreeRows rows; v8_disp dispatches a grid that scales with the rows: flags kV8wX (x times
  // the live tiles), kV8wY (y times the tiles), kV8wYrows (y = the live rows), kV8wXrows (x = the live rows); outside a
  // wide verify (or under NI_PROFILE, from the tree the host read) the grid directly
  static constexpr uint32_t kV8wX = 1, kV8wY = 2, kV8wYrows = 4, kV8wXrows = 8, kV8wXpair = 16;
  void v8_wide_begin(Encoder& e, Seq& s, int R);
  void v8_wide_end();
  void v8_disp(Encoder& e, uint32_t x, uint32_t y, uint32_t z, uint32_t flags, uint32_t tx);
  void grow_tape(Encoder& e, Seq& s, int rows);
  // table_ready: the layer's input-norm table is already written (by the previous layer's enc_moe8)
  void enc_attn8(Encoder& e, const Seq& s, const Layer& L, const Buffer& kc, const Buffer& vc, int cap, int pos, int R,
                 bool table_ready = false);
  void enc_mlp8(Encoder& e, const Layer& L, int R);
  // next_norm: the norm of what follows (the next layer's input norm, or the final norm); when given, the combine may
  // also write that norm's X8 table (V8Knobs::fuse & 1; with v8_tw() also the T table). Returns whether it did.
  bool enc_moe8(Encoder& e, const Layer& L, int R, int li, const Tensor* next_norm = nullptr);
  // ---- routed experts in llama.cpp's IQ blocks (GGUF imports; qwen35_iq.cpp) ----
  void load_iq_experts();
  std::string iq_layer_check(const Layer& L) const;  // "" when the IQ MoE kernels take the layer
  void enc_moe_iq_rows(Encoder& e, const Layer& L, int R, const Buffer& x);
  void enc_moe8_iq(Encoder& e, const Layer& L, int R, int li);
  void enc_moe_prefill_iq(Encoder& e, const Layer& L, int T, const Buffer& x, bool xh_ready);
  std::unordered_map<const Tensor*, std::string> iq_src_;  // IQ experts' deferred tensors until load_iq_experts

  std::string dir_;
  std::string prefix_;
  bool hf_names_ = false;  // official Hugging Face tensor names (model.language_model.*)
  bool hf_norms_ = false;  // zero-centred RMSNorm weights (Hugging Face): read as 1 + w
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
  // draft_head_ (and the gathered DraftVocab rows) are a 4-bit group-64 copy of the output layer's rows: an output
  // layer in another format (bf16, 6- or 8-bit variants) drafts from it (NI_DRAFT_Q4=0: from its own rows)
  bool draft_q4_ = false;
  // NI_DRAFT_CTX (default on): the reduced draft head also scores every other id this conversation's tokens hold
  // (prompt and output; a coding conversation's identifiers), from output-layer rows gathered per Seq
  struct DraftVocab;
  DraftVocab* draft_vocab_update(Seq& s, int upto);
  void draft_vocab_rewind(Seq& s, int pos0);
  bool draft_vocab_add(DraftVocab& v, int id);  // gathers id's row (false: the set is full)
  uint32_t draft_split_ = 0, draft_special0_ = 0;
  bool full_draft_ = false;  // draft over the whole vocabulary (see load_mtp; NI_FULL_DRAFT=0/1 overrides)
  const CaptureHook* cap_ = nullptr;
  const Buffer* v8_rlim_ = nullptr;
  bool v8_moe_exact_ = false;
  const Buffer* v8_tree_ = nullptr;
  // wide tree verifies (qwen35_v8.cpp): the capacity the buffers hold, the one being encoded (0: none), its descriptor
  // region (a ring of kV8wRegions, so a verify still on the GPU keeps its own), descriptors so far; under NI_PROFILE the
  // live tiles the host read from the tree (-1: GPU-driven)
  int tree_rows_ = 8;
  // the rows a cached suffix's chain pass may run (vpf_pass_rows): wider than tree_rows_ where the drafter drafts a
  // 16-row block, so the suffix's rows stream the weights once per pass instead of once per 8-row tile
  int pass_rows_ = 8;
  bool drafter_wide_ = false;
  int v8w_R_ = 0, v8w_region_ = 0, v8w_n_ = 0, v8w_host_nt_ = -1;
  Buffer v8w_desc_, v8w_args_;
  // the chain tree descriptors of encode_verify_chain, one per row count it runs (index rows / 8), each a single path
  // built once: the passes of a request share them and the plan kernel copies them into its own region, so they are
  // never rewritten while a pass may still read them
  Buffer chain_tree_[kMaxTreeRows / 8 + 1];
  std::deque<Tensor> owned_;
  std::vector<Buffer> owned_bufs_;

  size_t mem_budget_ = 0;
  bool gdn_sigmoid_ = false;  // the gated norm's gate: sigmoid (Qwen3.8-Flash-Next), else silu
  // 3-row verify GEMVs on the fp32 kernels instead of qmvht_t3_r4's half products (Qwen4Exp: accuracy first)
  bool no_qmvh_ = false;
  virtual void grow_kv(Seq& s, int cap);
  void enc_kv_regrow(Encoder& e, const Buffer& src, const Buffer& dst, int old_cap, int new_cap);

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
  Buffer v8tt_;  // T table (qmm8t's input layout)
  Buffer v8t_, v8s_, v8tl_, v8sl_, v8h_, v8hs_, v8tiles_, v8nt_, v8slot_, v8args_, v8out_, v8sh_, v8ustat_, v8sm_, v8hsc_, v8xn_, v8hse_, v8sm2_, v8hsc2_, v8hse2_;
};

} // namespace ni
