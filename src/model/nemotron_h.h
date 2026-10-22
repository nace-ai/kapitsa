// NVIDIA Nemotron-H hybrid (Mamba-2 + attention + LatentMoE with per-layer expert counts and widths, or dense relu^2
// MLP layers), from NVIDIA's safetensors checkpoints as they are: Nemotron 3.5 Bolt 74B-A9B (IQ2_XXS experts, NVFP4
// Mamba projections, Q8_0 LM head / attention / latent projections as GGUF blocks) and Nemotron 3.5 Super-VL 120B-A12B
// (NVFP4 experts and shared expert, FP8 E4M3 Mamba projections / attention / LM head with one scale per tensor, BF16
// latent projections), BF16 embedding and router; the dense Nemotron Nano 4B (BF16) and Nano 9B v2 (NVFP4 with BF16
// attention, first and last layers and LM head). Weights go into GPU buffers at load with codes and scales apart
// (src/kernels/nemotron/nh_mv.metal). No vision tower.
#pragma once
#include "loader/safetensors.h"
#include "model/model.h"
#include "model/nemotron_ane.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace ni {

struct NhConfig {
  int hidden = 0, layers = 0, vocab = 0;
  std::vector<char> kind;  // per layer: 'M' Mamba-2, 'A' attention, 'E' LatentMoE, 'D' dense MLP ('-' in the pattern)
  // Mamba-2 (head_dim 64, or 80: the Nano 4B / 9B v2)
  int d_inner = 0, d_state = 0, n_group = 0, n_head = 0, head_dim = 0, conv_k = 4;
  int conv_ch() const { return d_inner + 2 * n_group * d_state; }
  int in_proj() const { return 2 * d_inner + 2 * n_group * d_state + n_head; }
  // attention
  int hq = 0, hkv = 0, hd = 0;
  // LatentMoE (per layer: experts, experts used, expert width)
  int latent = 0, shared_I = 0;
  float w_scale = 1.0f;
  bool w_norm = true;
  std::vector<int> n_exp, top_k, exp_I;
  int max_E = 0, max_k = 0, max_I = 0;
  // dense MLP layers: relu^2(x W_up^T) W_down^T, the shared expert's computation (per layer: the intermediate size)
  std::vector<int> mlp_I;
  int max_mlp_I = 0;
  int ffn_I() const { return std::max(shared_I, max_mlp_I); }  // the widest relu^2 intermediate of a row
  float eps = 1e-5f;
  std::vector<int> eos;
  int bos = -1;
};

enum class NhFmt : uint8_t { F32, Q8, FP4, IQ2, BF16, FP8 };

// A weight matrix [N outputs][K inputs], or E stacked expert matrices, in a repacked GPU layout.
struct NhMat {
  NhFmt fmt = NhFmt::F32;
  uint32_t N = 0, K = 0, E = 1;
  Buffer q, s;            // codes / values; scales (none for F32 / BF16 / FP8: s = q)
  float gs = 1.0f;        // per-tensor scale (NVFP4's weight_scale_2; FP8: weight_scale x 256, the kernels' FP8
                          // operands being the E4M3 values x 2^-8)
  Buffer gse;             // NVFP4 expert stacks: each expert's weight_scale_2 (float [E]; gs is then 1)
  size_t q_row = 0, s_row = 0;  // bytes per row
  bool l2 = false;              // NVFP4 codes in layout L2 (nh_x_fp4.metal), else L0
  size_t q_exp() const { return q_row * N; }
  size_t s_exp() const { return s_row * N; }
  explicit operator bool() const { return N != 0; }
};

// Repacking (CPU): bytes per repacked row (codes, scales) of K elements in format f; bytes per source row in the
// checkpoint; source rows into the repacked layout (NVFP4: the codes only, nibbles reordered: layout L0, or with
// fp4_l2 layout L2, nh_x_fp4.metal); fp32 rows quantized to the Q8 layout (Q8_0's rounding: d = max|x| / 127).
void nh_row_bytes(NhFmt f, int64_t K, size_t& qb, size_t& sb);
size_t nh_src_row_bytes(NhFmt f, int64_t K);
void nh_repack_rows(NhFmt f, int64_t K, const uint8_t* src, int64_t rows, uint8_t* q, uint8_t* s, bool fp4_l2 = false);
void nh_quant_q8(const float* x, int64_t K, uint8_t* q, uint8_t* s);
// The format and shape of a checkpoint matrix (scale: its weight_scale entry, NVFP4's block scales or FP8's tensor
// scale, else null); false if unknown.
bool nh_describe(const SafetensorsIndex::Entry& w, const SafetensorsIndex::Entry* scale, NhFmt& f, uint32_t& N,
                 uint32_t& K);

struct NhLayer {
  char kind = 'M';
  Buffer norm;  // f32 [hidden]: the block's pre-norm
  // Mamba-2
  NhMat in, out;
  Buffer conv_w, conv_b, A, D, dt_bias, gnorm;
  // attention
  NhMat q, k, v, o;
  // LatentMoE; up_s / down_s: its shared expert, or a dense MLP layer's ('D') up / down projections (the same product)
  NhMat router, lat_down, lat_up, up_e, down_e, up_s, down_s;
  Buffer exp_bias;
  int E = 0, k_used = 0, I = 0;
};

// Capture of target layer inputs for a block drafter: before layer l (slot[l] >= 0; l = layers: the trunk's output,
// before the final norm), the hidden rows at positions p in [pos_base, pos_base + rows) are copied (fp32) to feat row
// p - pos_base, columns [slot * H, (slot + 1) * H).
struct NhCapture {
  Buffer feat;
  uint32_t ld = 0;          // floats per feat row (n_slots * H)
  int pos_base = 0, rows = 0;
  std::vector<int> slot;    // per target layer
};

class NemotronH : public Model {
 public:
  // path: the checkpoint folder (config.json, *.safetensors)
  explicit NemotronH(const std::string& path, int prefill_chunk = 2048);
  ~NemotronH() override;
  const NhConfig& config() const { return cfg_; }
  const std::string& dir() const { return path_; }
  bool dense() const { return dense_; }  // no MoE layers (Nano 4B / 9B v2)
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }

  // a folder whose config.json is a Nemotron-H model (model_type nemotron_h / nemotron_h_puzzle, also as an Omni
  // checkpoint's llm_config)
  static bool is_model(const std::string& dir);
  static MemPlan plan(const std::string& path, int prefill_chunk = 2048);
  // the prefill chunk the engine tries before 2048 rows when it fits (0: none): 4096 for the MoE models on the M5
  // without the Neural Engine split (Lightning), whose prefill runs on the matrix units (NI_NH_TG) and streams every
  // routed expert once per chunk and layer
  static int wide_chunk(const std::string& path);
  static size_t ane_plan_bytes(const NhConfig& c, size_t state);

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
  // GPU + Neural Engine prefill split (NI_ANE; nemotron_ane.h)
  bool ane_splits(int T) const override;
  bool ane_split_on() const override { return ane_ != nullptr; }
  bool ane_finish_chunk() override;
  void ane_abort(const std::string& why) override;
  bool grouped_passes() const override { return group_; }
  // the routed experts are IQ2_XXS (Nemotron 3.5 Bolt's 24 GB checkpoint)
  bool iq2_experts() const;
  bool ane_release() override;
  void ane_poll() override;
  const AneStats& ane_stats() const override { return ane_stats_; }
  const std::vector<NhLayer>& layers() const { return layers_; }

  // ---- speculative decoding ----
  static constexpr int kMaxRows = 8;  // rows of a decode / verify pass
  // Trunk forward of tokens[pos .. pos+R) without committing the Mamba state: the rows' in_proj outputs stay in
  // the sequence's pending rows; logits() rows 0..R-1 hold their next-token logits. commit_rows() then says how
  // many of the rows were accepted, and the next pass (or encode_flush) folds them into the state.
  void encode_verify(Encoder& e, Seq& s, int pos, int R);
  // Batched decode: one pass over the rows of several sequences (sum of R <= kMaxRows). Group g verifies
  // tokens[pos .. pos + R) of its sequence as encode_verify does (its pending rows are committed first, its rows
  // become pending); the norms, projections, MoE and LM head run once over all the rows, the Mamba scans and the
  // attention with its KV cache per group. Then one token per row is sampled with its group's parameters (RNG step:
  // the row's position) into samples() rows [r0, r0 + R) (r0: the rows of the groups before it). With `cap`, the
  // drafter's target layer inputs of the group's rows go to cap->feat rows [cap_row0, cap_row0 + R).
  struct VerifyGroup {
    Seq* s = nullptr;
    int pos = 0, R = 1;
    SamplingParams sp;
    const NhCapture* cap = nullptr;
    int cap_row0 = 0;
    // the samples go into the sequence's own tokens (row j's to tokens[pos + 1 + j]) instead of samples(), where the
    // next pass can read them before the host has (pipelined plain rows)
    bool tok_out = false;
  };
  void encode_verify_multi(Encoder& e, const std::vector<VerifyGroup>& groups);
  void commit_rows(Seq& s, int n) { s.n_pend = n; }
  // KFlash tree verify (src/model/kflash_budget.h): while on, the verify block's rows form a tree (tree_buf(), written
  // on the GPU by the drafter's tree builder: rows, parents, depths, ancestor masks; the nodes' tokens in tokens[pos +
  // r]): row r attends to the history and its ancestors, and its Mamba-2 state is its root path's. encode_accept_tree
  // then samples row r at position pos + depth(r), accepts the longest root path whose tokens are the rows' own samples
  // (accepted() = a; tokens[pos + 1 ..] = the path's tokens and the last row's sample; the path's K/V moved into place)
  // and commit_rows_tree(s, n) makes the path's first n rows pending.
  void set_tree(bool on) { tree_on_ = on; }
  Buffer tree_buf() const { return tree_; }
  void encode_accept_tree(Encoder& e, Seq& s, int pos, const SamplingParams& sp);
  void commit_rows_tree(Seq& s, int n) {
    s.n_pend = n;
    s.pend_tree = n > 0;
  }
  void encode_flush(Encoder& e, Seq& s);
  // Logits of rows [0, T) of the last pass (fp32 [T][vocab], T <= kMaxRows) / of the last prefill chunk.
  float* logits() { return logits_.as<float>(); }
  Buffer logits_buf() const { return logits_; }
  // teacher-forced scoring (tools): logits of every row of a prefill chunk of T rows into out [T][vocab]
  void encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out);
  std::string formats_summary() const;
  // After encode_verify: one token per row sampled from its logits (sp; step = the row's position) into
  // samples() [R], then the acceptance of drafts tokens[pos + 1 .. pos + K]: a = longest prefix agreeing with the
  // samples, tokens[pos + 1 + a] = samples[a], accepted() = a once the command buffer completed.
  void encode_accept(Encoder& e, Seq& s, int pos, int R, int K, const SamplingParams& sp);
  // Dynamic verify depth: while on, verify / accept passes keep only the first live_rows()[0] rows (written on the
  // GPU by the drafter: the rows past it skip their routed experts and scan steps, and acceptance stops there)
  void set_dynamic_rows(bool on) { dyn_rows_ = on; }
  Buffer live_rows() const { return live_; }
  const int* samples() const { return samp_.as<int>(); }
  int accepted() const { return acc_.as<int>()[0]; }
  const Buffer& accepted_buf() const { return acc_; }  // (accepted() on the GPU: int [0])

  // ---- for the drafter ----
  void set_capture(const NhCapture* c) { cap_ = c; }
  const NhMat& embed_table() const { return embed_; }
  const NhMat& lm_head() const { return head_; }
  // y = W x for R <= 8 fp32 rows (epi: 0 store, 1 add, 2 relu^2); strides in floats (0: K / N)
  void enc_mv(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int R, int epi, uint32_t x_stride = 0,
              uint32_t y_stride = 0);
  // the same for any number of rows (groups of 8)
  void enc_mv_rows(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int M, int epi);
  // RMSNorm rows (eps: the model's when < 0)
  void enc_norm(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& y, int R, int D, float eps = -1.0f);
  void enc_gemm(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi, uint32_t y_ld = 0,
                uint32_t x_ld = 0);
  void enc_rows_half(Encoder& e, const Buffer& x, const Buffer& y, int R, int D);
  // RMSNorm of fp32 rows straight into half rows (nh_rows_half's norm), and into both (nh_rmsnorm_h): one pass where
  // enc_norm + enc_rows_half took two, with the same values
  void enc_norm_half(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& yh, int R, int D);
  void enc_norm2(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& y, const Buffer& yh, int R, int D);
  bool enc_qkv(Encoder& e, const NhLayer& L, int R);
  // (params: the kernel's {D, tok_off} read from that buffer instead, e.g. written on the GPU by a chained drafting half)
  void enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out, const Buffer* params = nullptr,
                 size_t params_off = 0);
  static Pipeline pipeline(const std::string& name);

 private:
  void load(int prefill_chunk);
  void alloc_buffers();
  // R rows (<= kMaxRows) at positions pos0.. through the trunk; final hidden rows (pre-norm) stay in x_.
  // commit: the Mamba state and conv ring advance (prefill, decode); else the rows become pending (verify).
  void enc_rows(Encoder& e, Seq& s, int pos0, int R, bool commit);
  void enc_head(Encoder& e, int R, const Buffer& out, size_t out_row_stride);
  // the draw's RNG step: step, plus (*dstep)[out_index] (int32 at byte offset dstep_off) when dstep is given
  void enc_sample(Encoder& e, const Buffer& logits_row, const Buffer& tokens, int out_index, const SamplingParams& sp,
                  uint32_t step, const Buffer* dstep = nullptr, size_t dstep_off = 0);
  void enc_capture(Encoder& e, int layer, const Buffer& x, int pos0, int R);
  // np pending rows zp (a verify block's accepted rows, at pos0 - np) are committed first when given
  // parts: 1 the scan, 2 the conv ring updates after it (a batched pass runs every sequence's scans, then their rings)
  void enc_ssm(Encoder& e, Seq& s, int li, const Buffer& zx, int T, int pos0, bool write, const Buffer& y,
               const Buffer* zp = nullptr, int np = 0, int parts = 3);
  void enc_attn(Encoder& e, Seq& s, int li, int pos0, int R);
  void enc_moe(Encoder& e, const NhLayer& L, int R);
  void enc_mlp(Encoder& e, const NhLayer& L, int R);
  // prefill of T rows (> kMaxRows) with GEMMs; the final hidden rows (pre-norm) stay in px_
  void enc_prefill_gemm(Encoder& e, Seq& s, int pos0, int T);
  void alloc_prefill(int T);

  std::string path_;
  std::unique_ptr<SafetensorsIndex> st_;
  std::unique_ptr<NhAne> ane_;
  bool ane_chunk_ = false;  // the chunk being encoded runs the split
  AneStats ane_stats_;
  bool group_ = false;  // NI_NH_GROUP
  bool v8_n1_ = false;   // NI_NH_V8_N1
  bool rsu_ = false;     // NI_NH_RSU
  bool sample_sg_ = false;  // NI_NH_SAMPLE_SG
  // a dynamic verify's routed-expert grids (nh_moe_pair_args into moe_args_; NI_NH_PAIR_IND): set by enc_rows for
  // its pass, the shapes they were made for
  Buffer moe_args_;
  bool pair_ind_ = false, pair_args_ = false, moe_lightning_ = false;
  uint32_t moe_I_ = 0, moe_k_ = 0, pair_thr_ = 0, pair_rows_ = 0, pair_k_ = 0, pair_gx_up_ = 0, pair_gx_dn_ = 0;
  Buffer sdv_;           // the shared expert's down product of a verify's rows (rsu_)
  void init_ane();
  void ane_build();
  void prewarm_pipelines();
  NhAneConfig ane_cfg_;
  std::shared_ptr<std::atomic<int>> ane_async_;  // background compile: 1 running, 2 done, 3 failed
  std::thread ane_thread_;
  NhConfig cfg_;
  std::vector<NhLayer> layers_;
  NhMat embed_, head_;
  Buffer out_norm_;
  int tmax_ = 2048;
  size_t mem_budget_ = 0;
  std::vector<Buffer> arena_;   // weight buffers
  bool map_embed_ = true;       // NI_NH_MAP_EMBED: the BF16 embedding mapped from its file (no copy)
  void* embed_map_ = nullptr;
  size_t embed_map_len_ = 0;
  size_t weight_bytes_ = 0;
  std::vector<int> mamba_idx_;  // layer -> index among Mamba layers (-1 otherwise)

  // activations (kMaxRows rows)
  Buffer x_, xn_, zx_, y_, yn_, q_, k_, v_, ao_, part_o_, part_ml_, rl_, eid_, ew_, lat_, h_, lo_, sh_;
  Buffer logits_, cand_v_, cand_i_, sprob_, samp_, acc_;
  Buffer tree_;            // KFlash: the verify block's tree and its acceptance (layout kKf*, nh_common.h)
  bool tree_on_ = false;
  Buffer xnh_, aoh_, loh_, shh_;  // half inputs of the multi-row GEMMs (passes of 2+ rows)
  Buffer part_;                   // routed experts' weighted partial outputs [rows * k][latent]
  Buffer iq2tab_;                 // IQ2_XXS lookup tables (constant address space in the kernels)
  Buffer live_, live_all_;        // live rows of a verify block (dynamic depth), and the no-limit constant
  bool dyn_rows_ = false;
  const Buffer& live() const { return dyn_rows_ ? live_ : live_all_; }
  bool gemm_rows_ = true;         // NI_NH_ROWS_GEMM: passes of 2+ rows (verify) run their dense matmuls as GEMMs
  bool v8_ = true;                // NI_NH_V8: GEMMs of <= 8 rows on the 8-row MMA kernels (nh_v8.metal)
  bool x4_ = true;  // NVFP4 x4 kernels (one row, 8-row verify, 9-32-row split-K); NI_NH_X4=0: the older ones
  bool x8_ = true;  // FP8 x8 kernels (one row, 8-row verify, 9-32-row split-K); NI_NH_X8=0: the generic ones
  // The M5's matrix units (Metal 4 tensor_ops, src/kernels/tensor/nh_v8t.metal) for the NVFP4 GEMMs of <= 8 rows:
  // Apple10 on macOS 26; NI_NH_V8T=0 keeps them on nh_x4_v8
  bool v8t_ = false;
  // M5 (Apple10, macOS 26): the prefill's tile GEMMs (> 32 rows) on the GPU's matrix units (nh_gemm_t_*, Metal 4
  // tensor_ops); NI_NH_TG=0: the simdgroup-MMA ones (nh_gemm_*)
  bool tg_ = false;
  Buffer pcv_;  // nh_ssm_conv's conv'd rows and dt of a prefill chunk (M5; pout_ when it is large enough)
  Buffer psb_;  // nh_ssd's per-chunk M and decays (M5; the rest of pout_ when it is large enough)
  Buffer x4part_, x4cnt_;  // split-K partials / arrival counters of the 9-32-row x4 / x8 kernels (serial calls share them)
  // A dense model (no MoE layers: Nemotron Nano 4B / 9B v2). Its products of K % 128 != 0 (hidden 3136, the 9B's MLP
  // width 15680) run on the tail variants of the 2-32-row kernels (nh_xb_v8_t, nh_x4_*_t, nh_x8_*_t), which the MoE
  // models' paths do not take.
  bool dense_ = false;
  // NVFP4 codes in layout L2 (nh_x_fp4.metal): the dense models on the M1's GPU family, as Dory (NI_NH_FP4=l0 / l2
  // overrides); every NVFP4 product then runs on the x4 kernels or nh_gemm_fp4l2_*
  bool fp4_l2_ = false;
  bool experts_l2_ = false;  // NVFP4 routed experts in layout L2 (NI_NH_FP4_EXPERTS); see NemotronH::load
  bool x2_ = true;  // IQ2_XXS x2 kernels (subnormal-half weights); NI_NH_X2=0: the older ones (fused decode excepted)
  Buffer x2sorted_, x2tiles_, x2args_, x2rank_;  // verify routed experts grouped by expert (nh_x2_bucket)
  bool merge_flush_ = true;       // NI_NH_MERGE_FLUSH: a verify's scans commit the previous block's accepted rows
  bool fuse_dec_ = true;          // NI_NH_FUSE: a token's MoE layer in 4 dispatches (nh_zdec.metal)
  bool route_bucket_ = true;      // NI_NH_ROUTE_BUCKET: an M5 verify's IQ2 routing and bucketing in one dispatch
  uint32_t x2_sgd_ = 1;           // NI_NH_X2_SGD: simdgroups per threadgroup of a verify's IQ2 experts' down projection
  uint32_t moe_thr_ = 256;        // NI_NH_MOE_THR: threads per threadgroup of a verify's NVFP4 routed experts
  Buffer route_ctr_, sd_;         // fused MoE: the router threadgroups' arrival counter, the shared expert's output
  Buffer moecnt_;                 // NVFP4 experts: arrival counters per (row, 32-row block) of nh_x4_moe_dn_sum
  int attn_max_chunks_ = 64;
  uint32_t attn_chunk_ = 256;  // keys per attention chunk (NI_NH_ATTN_CHUNK; M5: 64)
  const NhCapture* cap_ = nullptr;
  // prefill activations (tmax_ rows; allocated on first use)
  int pf_rows_ = 0;
  Buffer px_, pxn_, pxh_, pzx_, py_, pyh_, pq_, pk_, pv_, poh_, prl_, peid_, pew_, plat_, ph_, pout_, plo_, psh_;
  Buffer psorted_, ptiles_, pntiles_, prank_;
};

}  // namespace ni
