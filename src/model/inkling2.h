// Thinking Machines Inkling-2-Mini (thinkingmachines/Inkling-2-Mini, model_type "inkling2_loop"):
// Stage 1 text engine (24 unique transformer+MoE layers evaluated across a 48-visit recurrent loop schedule
// [0..5, 6..17, 6..17, 6..17, 18..23], 200,058 unpadded vocab out of 201,024 padded, AttRes across blocks +
// cross-pass MessageGate controllers + static/dynamic short convolutions + SWA/global sink attention with
// relative position bias + 128-expert top-8 SiTU MoE in NVFP4 L0-19 and MXFP8 L20-23 + final logit softcap 30).
// Vision tower (model.visual.*) is skipped in Stage 1.
#pragma once

#include "engine/profile.h"
#include "loader/safetensors.h"
#include "model/model.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace ni {

// The M1 stack's switches (inkling2.cpp): an env var when set, else m1_value on the fast profile on the legacy class
// (device profile row ink2_m1_stack), else off_value (the path main shipped)
bool ink2_m1_stack();
int ink2_knob(const char* env, int m1_value, int off_value);

// In-process A/B timer (lab8; ink2_bench BENCH_AB="VAR=val[,VAR=val..]"): while g_ink2_ab_phase is 1, the knobs read
// through ink2_ab(name, v) return the BENCH_AB value instead of v (the environment's), so one process alternates two
// configs step by step over the same KV state.
extern int g_ink2_ab_phase;
int ink2_ab(const char* name, int v);

struct Ink2Config {
  int hidden = 2304;
  int num_layers = 24;
  int vocab_padded = 201024;
  int vocab = 200058;  // unpadded_vocab_size (logits sliced/evaluated over [0, vocab))
  int hq = 24;
  int hkv = 8;
  int head_dim = 128;
  int d_rel = 16;
  int rel_extent = 1024;
  int swa = 1024;
  int block_sz = 6;
  int core_start = 6;
  int core_end = 18;
  int num_passes = 3;
  int n_experts = 128;
  int top_k = 8;
  int moe_I = 1152;
  int gate_rank = 64;
  float rms_eps = 1e-6f;
  float qk_clip = 22.627416997969522f;
  float rel_clip = 8.0f;
  float log_floor = 8192.0f;
  float log_alpha = 0.1f;
  float route_scale = 8.0f;
  float logit_softcap = 30.0f;
  std::vector<bool> is_local;
  std::vector<int> schedule;  // length 48
  std::vector<int> eos;
};

struct Ink2Ctrl {
  const Tensor* down = nullptr;    // [64, 4608]
  const Tensor* forget = nullptr;  // [2304, 64]
  const Tensor* gain = nullptr;    // [2304, 64]
};

struct Ink2Blocks {
  Buffer data;
  uint32_t type = 0, row_bytes = 0;
};

struct Ink2Mat {
  const Tensor* raw = nullptr;  // BF16 or F32 when not Q8-packed (or kept beside the Q8 copy: NI_INK2_Q8_BF16)
  Buffer q, s;                  // Q8_0: int8 [N, K], half [N, K/32]
  Ink2Blocks gguf;
  uint32_t N = 0, K = 0;
  bool is_q8() const { return bool(q); }
};

// The fused qkvrsg rows' Q8_0 copy (int8 codes + half block-32 scales; the segments' #1381 packing, end to end)
struct Ink2Q8 {
  Buffer q, s;
  uint32_t N = 0, K = 0;
  explicit operator bool() const { return bool(q); }
};

struct Ink2Layer {
  bool is_swa = true;
  bool has_ctrl = false;
  bool is_mxfp8 = false;
  // attention projections
  Ink2Mat wq_du, wk_dv, wv_dv, wr_du, gate_attn, wo_ud;
  const Tensor *attn_sink_proj = nullptr;
  const Tensor *q_scale = nullptr, *rel_scale = nullptr, *rel_proj = nullptr;
  // norms
  const Tensor *attn_norm = nullptr;  // SWA only
  const Tensor *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;  // Global only
  const Tensor *mlp_norm = nullptr;
  // short convolutions
  const Tensor *k_sconv = nullptr, *v_sconv = nullptr;  // SWA static [1024, 1, 4]
  const Tensor *k_static = nullptr, *k_proj = nullptr;  // Global dynamic [4, 1024], [32, 2304]
  const Tensor *v_static = nullptr, *v_proj = nullptr;
  const Tensor *attn_sconv = nullptr, *mlp_sconv = nullptr;  // static [2304, 1, 4]
  // cross-pass controllers (core layers 6..17)
  Ink2Ctrl ctrl_attn, ctrl_k, ctrl_v, ctrl_mlp;
  // MoE
  const Tensor *router_w = nullptr, *router_bias = nullptr;
  const Tensor *w13 = nullptr, *w13_sc = nullptr, *w2 = nullptr, *w2_sc = nullptr;
  Ink2Blocks gguf_gu, gguf_down;
  Buffer w13_outer, w2_outer;  // FP32 [128] = amax[e] / outer_scale_denominator[0] (NVFP4 only)
  Buffer w13_rp, w2_rp;        // NVFP4 nibbles repacked for the lean decode (repack_moe; == w13/w2->buf when in place)
  Buffer qkvrsg_w;             // Fused BF16 [wq_du; wr_du; attn_sink_proj; gate_attn; wk_dv; wv_dv]
  Ink2Q8 qkvrsg_q8;            // its Q8 copy (decode / verify rows)
  uint32_t qkvrsg_offs[7] = {};
};

class Inkling2 : public Model {
 public:
  explicit Inkling2(const std::string& path, int prefill_chunk = 512);
  ~Inkling2() override;

  static bool is_model(const std::string& dir);
  // True when the precise profile runs with a fast-profile member that would be on here (its env switch unset): the engine
  // ready line then names `--profile fast`. False under the fast profile, and when the member is off on this Mac.
  static bool fast_profile_available();
  static Ink2Config read_config(const std::string& dir);
  static MemPlan plan(const std::string& dir, int prefill_chunk = 512);
  static size_t kv_per_pos(const Ink2Config& c);
  static size_t conv_bytes_per_seq(const Ink2Config& c);
  static size_t ring_bytes_per_seq(const Ink2Config& c, int tmax);

  Profile effective_profile() const { return profile_; }
  const Ink2Config& config() const { return cfg_; }
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }
  int prefill_chunk() const override { return tmax_; }
  bool can_snapshot() const override { return false; }
  bool grouped_passes() const override { return true; }

  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  size_t kv_grow_held(const Seq& s) const override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override { return kv_per_pos(cfg_); }
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override;
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  std::string disk_format() const override;
  void disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const override;
  void disk_state(const Seq& s, const Checkpoint* ck, std::vector<Buffer>& out) const override;
  void enc_kt_tile(Encoder& e, Seq& s, size_t v, int a, int b);  // kc rows [a, b) of visit v into s.kt[v]
  void disk_loaded(Seq& s) override { s.kt_stale = !s.kt.empty(); }

  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;

  // Speculative decoding hooks (the model half of #1383, its drafter left out): multi-row verify + conv rollback
  static constexpr int kMaxVerify = 8;
  void encode_verify(Encoder& e, Seq& s, int pos0, int T, const Buffer& out_logits, const Buffer* taps_out,
                     bool save_conv_snap = true);
  void encode_conv_rollback(Encoder& e, Seq& s, int keep_idx, int total_rows);

  // Diagnostic helpers (tools/ink2_check)
  float* logits() { return logits_.as<float>(); }
  Buffer logits_buf() const { return logits_; }
  void encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out);
  void encode_prefill_diag(Encoder& e, Seq& s, int pos0, int T, const Buffer& diag_rows);

 private:
  void load();
  void load_gguf();
  void alloc_buffers();
  bool ensure_rings(Seq& s);
  size_t kv_grow_unit() const;
  void enc_forward(Encoder& e, Seq& s, int pos0, int T, bool dump_logits, const Buffer* out_logits, bool need_all_logits,
                   const Buffer* diag_rows = nullptr, const Buffer* taps_out = nullptr, bool save_conv_snap = false);
  void enc_mm(Encoder& e, const Tensor& w, const Buffer& x, const Buffer& y, int M, uint32_t N_limit = 0, bool add = false,
              bool exact_f32 = false);
  void enc_mm(Encoder& e, const Ink2Mat& w, const Buffer& x, const Buffer& y, int M, uint32_t N_limit = 0, bool add = false,
              bool exact_f32 = false);
  void enc_norm(Encoder& e, const Buffer& x, const Tensor* w, const Buffer& out, int T, uint32_t dim);
  void enc_attn_res(Encoder& e, const Buffer* const* states, int M, const void* query_ptr, bool q_f32, uint32_t Q,
                    const Buffer& out, int T);
  // enc_attn_res + nn RMSNorms of its outputs in one kernel (src[j] = output query, w[j] weight, nout[j]); false = not
  // applicable (NI_INK2_NFOLD=0, shapes), caller encodes the two stages
  bool enc_attn_res_norm(Encoder& e, const Buffer* const* states, int M, const void* query_ptr, bool q_f32, uint32_t Q,
                         const Buffer& out, int T, int nn, const uint32_t* src, const Tensor* const* w,
                         const Buffer* const* nout);
  void enc_msg_gate(Encoder& e, const Ink2Ctrl& c, const Buffer& h, const Buffer& m, const Buffer& out, int T);
  void enc_msg_gate3(Encoder& e, const Ink2Layer& L, const Buffer* const* in_ptrs, const Buffer& m0, const Buffer& m1,
                     const Buffer& m2, int T);
  void enc_sample(Encoder& e, Seq& s, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                  uint32_t step);

  Profile profile_ = Profile::kPrecise;  // Resolved once during construction, after validating the checkpoint.
  std::string path_;
  Ink2Config cfg_;
  WeightStore ws_;
  std::vector<Ink2Layer> layers_;
  const Tensor *embed_ = nullptr, *embed_norm_ = nullptr, *final_norm_ = nullptr;
  Ink2Mat unembed_, gguf_embed_;
  std::deque<Tensor> gguf_tensors_;
  std::vector<const Tensor*> attn_queries_;  // [23] (layers 1..23 -> idx 0..22)
  std::vector<const Tensor*> mlp_queries_;   // [24] (layers 0..23)
  const Tensor* output_query_ = nullptr;
  Buffer msg_queries_;                       // FP32 [12][4][2304] interleaved (attn, k, v, mlp)
  std::vector<Buffer> owned_bufs_;

  int tmax_ = 512;
  int ring_rows_ = 0;
  size_t mem_budget_ = 0;
  bool tensor_pf_ = false;
  bool q8_dense_ = false;

  // Reusable activations sized for `tmax_` rows
  Buffer x_, x_pre_, inp_sa_, ffn_inp_, h_norm_;
  Buffer blocks_[6];        // block boundary states + bank state during middle cycle
  Buffer prelude_blocks_[4];
  Buffer ar_out_;           // attn_res output: up to 4 * tmax * H
  Buffer msg_out_;          // cross-pass messages: 4 * tmax * H
  Buffer mg_cat_, mg_lat_;  // split MessageGate staging buffers (T <= 16)
  Buffer mg_catb_, mg_latb_, mg_gb_;  // MessageGate as GEMMs (T > 16, NI_INK2_MSG_MM)
  // largest row count on the matvec kernels (the M > 1 matvec keeps its accumulators in memory: verify T=4 ran
  // bf16_mv4 at 218 us a call vs 57 at T=1); above it the GEMM path. NI_INK2_MV_ROWS (default 32; 1 measured worse at T=2: 102 -> 141 ms)
  static int mv_rows() {
    static const int v = getenv("NI_INK2_MV_ROWS") ? atoi(getenv("NI_INK2_MV_ROWS")) : 32;
    return v;
  }
  // compile-time-M matvec for 2..8 rows (ink2_bf16_mvm<M>, weights read once); NI_INK2_MVM=0 = old runtime-M kernels
  static bool mvm() {
    static const bool v = ink2_knob("NI_INK2_MVM", 1, 0) != 0;
    return v;
  }
  // Q8_0 dense decode/verify matvecs for rows <= 8 (attention projections, fused qkvrsg, unembed) on #1381's packing
  // (NI_INK2_Q8). NI_INK2_Q8M=0: #1381's own kernels for those rows. NI_INK2_Q8_BF16=1 keeps the BF16
  // copies beside the Q8 ones for the prefill GEMMs (the tensor-unit BF16 GEMMs; +1.04 GB), =0 frees them (#1381)
  bool q8m_ = false, q8_bf16_ = false;
  bool enc_q8(Encoder& e, const Ink2Mat& w, const Buffer& x, const Buffer& y, int M, uint32_t N, bool add);
  // lab10 NI_INK2_REPACK (1 in place, 2 = beside the original for BENCH_AB NI_INK2_LEAN) and the sign-0 scale check
  // behind NI_INK2_SCPAIR
  void repack_moe();
  uint32_t moe_dec(bool w2) const;  // DEC mask for the bt kernels this step (ink2.metal ink2_dot8d)
  bool moe_lean() const;
  int repack_ = 0;
  bool sc_pos_ = false;
  // small-T MessageGate as 2 kernels (norms folded into down, coalesced up_mix); NI_INK2_MSGF=0 = prep/down_s/up_mix_s
  static bool msg_fused() {
    static const bool v = ink2_knob("NI_INK2_MSGF", 1, 0) != 0;
    return v;
  }
  void enc_msg_gate_f(Encoder& e, const Ink2Ctrl& c, const Buffer& h, const Buffer& m, const Buffer& lat,
                      const Buffer& out, int T, int stage);
  // NVFP4 MoE decode kernels with a threadgroup byte -> float2 E2M1 LUT; NI_INK2_MOELUT=0 = constant-space lookups
  static bool moe_lut() {
    static const bool v = ink2_knob("NI_INK2_MOELUT", 1, 0) != 0;
    return v;
  }
  static bool msg_mm() {
    static const bool v = ink2_knob("NI_INK2_MSG_MM", 1, 0) != 0;
    return v;
  }
  Buffer n0_, n1_, n2_;     // normed inputs [tmax, H]
  Buffer q_, k_, v_, r_, sinks_, gate_attn_, att_out_, a_out_;
  Buffer dyn_, rb_, part_m_, part_l_, part_o_;
  Buffer attn_cnt_;  // [kMaxVerify][hkv] arrival counters of ink2_attn_decode_g3kw_r (zero between dispatches)
  Buffer h_mlp_, x_mlp_, rlog_, eid_, ew_, moe_act_, moe_out_;
  Buffer moe_sorted_, moe_rank_, moe_tiles_, moe_ntiles_, moe_tmp_, moe_act_h_;
  Buffer logits_, cand_v_, cand_i_, sprob_, samp_;
  Buffer conv_snap_;  // lazy [kMaxVerify, conv_bytes_per_seq] for speculative verify conv rollback
};

}  // namespace ni
