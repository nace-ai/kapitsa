// DFlash2 block drafter (research/inference/dflash2/SPEC.md): loads a DFlash2DraftModel
// checkpoint (bf16 safetensors), quantizes its projections to MLX affine 4-bit (or 8-bit) g64 at
// load, keeps the per-sequence context K/V ring fed from the target's captured hidden states, and
// encodes one drafting pass (anchor + 7 mask rows -> 7 proposals) into a command buffer.
// Opt-in: nothing here is constructed unless a drafter directory is given.
#pragma once
#include "model/qwen35.h"

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
};

// Per-sequence drafter state: the context K/V ring of every draft layer.
struct DFlashSeq {
  Buffer ring_k, ring_v;  // bf16 [NL][8 kv heads][2048 slots][128]; slot = position mod 2048
  int ctx = 0;            // committed context positions 0 .. ctx-1 (the ring holds the last 2048)
  int valid_lo = 0;       // lowest position whose ring entry is valid (> 0 after a rewind past the window)
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

// Draft attention split plan for anchor position P: context [lo, P) cut into nsplit chunks of
// `chunk` positions (multiple of 32), at least min_chunk each (0: NI_DFLASH_ATTN_CHUNK, default 256),
// at most max_splits.
void dfl_attn_plan(int P, uint32_t max_splits, uint32_t min_chunk, uint32_t* lo, uint32_t* chunk, uint32_t* nsplit,
                   int valid_lo = 0);

class DFlash2 {
 public:
  static constexpr int kB = 8, kWin = 2048, kTopK = 16, kRank = 256, kCtxKeep = 2047;
  // bits: 4 or 8 (0: NI_DFLASH_BITS, default 4).
  DFlash2(Qwen35& target, const std::string& dir, int bits = 0);
  const DFlash2Config& config() const { return cfg_; }
  int bits() const { return bits_; }
  size_t resident_bytes() const { return resident_; }

  // A DFlash2 drafter directory for this target, or "" (no match): <model_dir>/dflash2 or /drafter,
  // then `*-DFlash2` directories next to the model, in <parent>/drafters and in
  // ~/.cache/nace-infer-models/drafters whose name (minus -DFlash2, case-insensitive) is a prefix of
  // the model directory's name up to a '-', and whose config.json is a DFlash2DraftModel with the
  // target's hidden size, vocabulary and layer count. The longest name match wins.
  static std::string find(const std::string& model_dir, const Qwen35Config& target);

  std::unique_ptr<DFlashSeq> new_seq();
  // Prefix reuse: entries at positions >= pos no longer belong to the sequence (the target state was
  // restored to pos); afterwards d.ctx = pos. reset() empties the ring.
  static void rewind(DFlashSeq& d, int pos);
  static void reset(DFlashSeq& d) { d.ctx = d.valid_lo = 0; }

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
  void arm_verify(int P);
  void disarm();
  void encode_commit(Encoder& e, DFlashSeq& d, int m);
  Buffer capt() const { return capt_; }  // bf16 [8][n_slots * H]

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
  // One whole cycle into `e`, no host round trip inside: ring commit of the previous cycle's m_prev
  // kept rows, the drafting pass from the anchor tokens[P], the target's verify of the first K
  // proposals with captures, the acceptance. Afterwards tokens[P+1 .. P+1+accepted()] are the next
  // tokens; the caller emits m <= accepted() + 1 of them, calls target.commit_rows(s, m) and passes
  // m as the next cycle's m_prev (P advances by m).
  void encode_cycle(Encoder& e, Seq& s, DFlashSeq& d, int P, int m_prev, int K, const SamplingParams& sp);
  int accepted() const { return acc_.as<int>()[0]; }
  // proposals verified per cycle: NI_DFLASH_VERIFY (1..7), default 7
  static int verify_depth();

  // results of the last pass (host-visible after the command buffer completed)
  const int* cand_ids() const { return cand_i_.as<int>(); }       // [7][16]
  const float* cand_vals() const { return cand_v_.as<float>(); }  // [7][16]
  const float* qprob() const { return qprob_.as<float>(); }       // [7][16] (sampled walk)
  const int* walk_k() const { return walk_k_.as<int>(); }         // [7]
  const float* walk_scores() const { return walk_s_.as<float>(); }// [7][16]
  const float* edges() const { return edges_.as<float>(); }       // [7][16][16]
  const float* logits() const { return logits_.as<float>(); }     // [8][V] (row 0 unused)
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
  QLinear quantize(const std::vector<const Tensor*>& parts, const char* label);
  QLinear row_view(const QLinear& src, uint32_t r0, uint32_t n);
  void alloc_buffers();

  void enc_qmm8(Encoder& e, const std::vector<DflSeg8>& segs, const Buffer& xh, const Buffer& xs, uint32_t K,
                uint32_t x_ld) {
    dfl_enc_qmm8(e, segs, xh, xs, K, x_ld);
  }
  void enc_qmm8_gu(Encoder& e, const QLinear& g, const QLinear& u, const Buffer& xh, const Buffer& xs) {
    dfl_enc_qmm8_gu(e, g, u, xh, xs, hh_, hs_);
  }
  void enc_qmm_rows(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M);  // prefill GEMM
  void enc_commit_kv(Encoder& e, DFlashSeq& d, const Buffer& kv, int rows, int pos0);
  void dbg(Encoder& e, const std::string& name, const Buffer& src, size_t bytes);

  Qwen35& tg_;
  std::string dir_;
  DFlash2Config cfg_;
  int bits_ = 4;
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
  Buffer h_, n_, xh_, xs_, hh_, hs_, dyn_, q_, kv_, Q_, Kb_, Vb_, part_o_, part_ml_, u_;
  Buffer logits_, sel_, tk_v_, tk_i_, cand_v_, cand_i_, edges_, walk_s_, qprob_, walk_k_, rope_cs_;
  // acceptance: target distributions of the verify rows, result {a, token}
  Buffer dist_id_, dist_p_, dist_keep_, acc_;
  // context commit
  Buffer capt_, y8_, c8_, kv8_, cdbg_;
  Buffer pcap_, xph_, yph_, cph_, kvph_;
  int max_splits_ = 32;
  bool h0_set_ = false, debug_ = false;
  int attn_variant_ = 0, pf8_ = -1;
  std::map<std::string, Buffer> dbg_;
};

} // namespace ni
