// Qwen3.8-Flash-Next (Hugging Face model_type qwen4_exp): the Qwen3.5 hybrid (Gated DeltaNet + gated attention, 3:1,
// 512-expert MoE) with
//  - hyper-connections: the residual stream is 4 streams of the hidden size; each block's input is a gated mean of
//    the per-stream normalized streams (a low-rank mix replaces the input norms) and its output goes back into
//    every stream scaled by an injection weight; the final norm is such a mix too;
//  - a sigmoid output gate in the DeltaNet layers;
//  - a sparse-attention indexer: queries that see more than 2051 tokens attend only to the 512 blocks of 4 tokens
//    that score best against compressed index keys (plus the incomplete block); exactly dense below;
//  - a hashed n-gram embedding (PLE) added before layer 1: 16 rows per token from a 51 B-parameter table that stays
//    on the SSD (mapped, gathered on the CPU), gated by the streams and passed through a dilated causal conv;
//  - an MTP head (one such layer) fed by the pre-mix 4-stream state.
// The class reuses Qwen35's kernels and encoders (projections, DeltaNet, attention, MoE, sampling, KV growth,
// checkpoints, the verify tape) and replaces its layer loops.
#pragma once
#include "model/expert_store.h"
#include "model/ngram.h"
#include "model/qwen35.h"
#include "model/remote.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace ni {

struct Qwen4ExpConfig {
  int hc = 4;            // residual streams
  int hc_rank = 320;     // low-rank width of the mix
  // sparse-attention indexer
  int ix_heads = 4, ix_dim = 128, ix_budget = 2048, ix_ratio = 4;
  // n-gram embedding
  int ple_layer = -1;    // the decoder layer it is added before (-1: none)
  int ngram = 3, per_ngram = 8, seed = 1234, ple_conv = 4, eos = 248044;
  int64_t ngram_base = 20000000;
};

// A gated residual (hyper-connection) of one block: per-stream RMSNorm weights (fp32, 1 + w), the low-rank mix
// (down [rank, S*D], up [S*D, rank]) and the injection weights (bf16 [S, S*D]; absent on the final mix).
struct GatedRes {
  const Tensor* norm = nullptr;
  QLinear down, up, inj;
};

struct Qwen4Layer {
  GatedRes ga, gm;            // before the token mixer, before the MoE
  QLinear iqk;                // indexer: [iq heads | ik] projection
  const Tensor *iqn = nullptr, *ikn = nullptr;  // indexer q / block-key norms (fp32, 1 + w)
};

class Qwen4Exp : public Qwen35 {
 public:
  // A model split across Macs by layers (src/cluster/pipe.h): this process holds trunk layers [lo, hi). The head (lo
  // 0) also holds the embedding, the n-gram block (its layer below hi), the final mix, the LM head and the MTP head,
  // and `remote` runs layers [hi, layers) on the other Macs inside each pass (Remote); a stage (lo > 0) holds its
  // layers alone and runs the parts of passes a head sends it (encode_stage). hi -1: every layer from lo.
  struct Split {
    int lo = 0, hi = -1;
    std::shared_ptr<Remote> remote;
  };
  // The process's split, for the engine's Qwen4Exp and every plan(): set before either (the server's --split, `kapitsa
  // stage`); none by default
  static void set_split(const Split& s);
  static const Split& split();
  // mem_budget: the GPU memory the model may hold, its KV caches and slots' state set apart (0: the default budget as
  // default_mem_budget(), less a 16K-token cache); a pack with Q2_0 routed experts keeps as many of them in memory
  // as that leaves room for and streams the rest (NI_EXPERT_STORE=<GB>|all|off sets it). split: this model's, instead
  // of the process's (tools that hold a whole model beside a split one). planned_ctx / planned_slots: the engine's
  // start plan (its context and conversations); the store's avail guard reserves every slot's KV, up to the budget's
  // 32K keep window, against the available memory it sizes the store on (q20::auto_store_kv) — 0 (a tool without a
  // plan): the default budget's keep window
  explicit Qwen4Exp(const std::string& model_dir, int prefill_chunk = 2048, size_t mem_budget = 0,
                    const Split* split = nullptr, int planned_ctx = 0, int planned_slots = 0);
  ~Qwen4Exp() override;
  // config.json's (text_config's) model_type is qwen4_exp_text / qwen4_exp
  static bool is_model(const std::string& model_dir);
  // a Flash-Next folder whose routed experts are Q2_0 (tools/qwen4_exp_q20.py)
  static bool q20_pack(const std::string& model_dir);
  // What load adds to the RMSNorm weights, from the mean of the first attention layer's indexer layernorms: 1 for
  // Qwen's zero-centred weights (a mean of at most 0.25), 0 where a converter added the 1 (at least 0.75), -1 between
  static float norm_offset(double indexer_norm_mean);
  // Whether the prefill sparse attention runs on the matrix units (qattn_t_sparse_g12d2), the decision enc_attn_sparse
  // makes: NI_QSA_TATTN=1 (opt-in since #585), 12 query heads per KV head, head dim 256, an Apple10 GPU and macOS 26.
  // The model_dir form reads config.json (false for a folder that is not Flash-Next).
  static bool tattn(const Qwen35Config& c);
  static bool tattn(const std::string& model_dir);
  // the process's split's share of the model (split()), or `split`'s
  static MemPlan plan(const std::string& model_dir, int prefill_chunk = 2048, const Split* split = nullptr);
  // For placing a split (src/cluster/place.h), from the headers: each trunk layer's weight bytes as stored (its routed
  // experts included, and apart: what a Q2_0 pack's store may stream) and as a decode step reads them (the routed share
  // of its experts), whether it is a linear-attention layer, and the same for the head's own tensors (the embedding's
  // row aside: the LM head, the final mix, the MTP head unless NI_NO_MTP) into *head
  struct LayerBytes {
    double weights = 0, decode = 0, experts = 0;
    bool linear = false;
  };
  static std::vector<LayerBytes> layer_bytes(const std::string& model_dir, LayerBytes* head);
  // A Q2_0 pack's store: the bytes of one layer's experts in its slots (plan() keeps at least 3 layers' worth)
  static double store_layer_bytes(const std::string& model_dir);
  const Qwen4ExpConfig& config4() const { return c4_; }
  std::string lost() const override { return remote_ ? remote_->lost() : ""; }

  // ---- a stage of a split model ----
  // The stage's part of a pass its head sent: the rows' state before layer layer_lo() is in residual(), block_out()
  // and injection() (rows of hidden_row_bytes(), hidden * 4 and hc * 4 bytes); layers [layer_lo, layer_hi) run on it
  // and leave the state after them there (a prefill on the last stage also injects its last block, as the head's own
  // prefill does). snap: the stage's checkpoint of p.snap, or null.
  void encode_stage(Encoder& e, Seq& s, const Remote::Pass& p, Checkpoint* snap);
  Buffer block_out() const { return yb_; }
  Buffer injection() const { return injM_; }

  // ---- memory ----
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override;
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override;
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  void set_rope_positions(Seq& s, int n, const std::vector<vision::ImageSpan>& spans) override;
  // not on the SSD prefix cache yet: its indexer keys and n-gram window are state Qwen35's layout does not name
  std::string disk_format() const override { return ""; }
  bool copy_prefix(Encoder&, const Seq&, Seq&, int) override { return false; }  // the sparse indexer's keys too: not copied

  // ---- passes ----
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  // layer-major prefill groups (NI_PREFILL_GROUP; model.h): G chunks share the per-layer expert reads; the chunks
  // must be plain (engine.cpp only forms such groups: no mid-chunk checkpoint, no split rerun, no image/grammar
  // rows, not the pass's last chunk). 1 - and the chunk path stays - on a split or debug model, and while every
  // expert is resident (no store, or a complete one): a group pays the streamed store's reads alone, and an
  // all-resident model would lose the chunk path's NI_HC_PRECOMB fold and carry slot buffers for nothing
  // (q20::group_chunks; the same predicate decides alloc4_group and the plan's group_slots).
  int prefill_group() const override {
    return q20::group_chunks(prefill_group_, remote_ || lo_ != 0, dbg_layers_ >= 0, !store_ || store_->complete());
  }
  void encode_prefill_group(Encoder& e, Seq& s, int pos0, const int* Ts, int nc) override;
  void encode_logits_rows(Encoder& e, int T, const Buffer& out) override;
  // speculative decoding (qwen4_exp_spec.cpp)
  int default_spec_depth() const override { return 2; }
  size_t hidden_row_bytes() const override { return size_t(c4_.hc) * size_t(cfg_.hidden) * 4; }
  void encode_mtp(Encoder& e, Seq& s, int mpos0, int R, int tok0, const Buffer& h, int h_row0, int draft_out,
                  bool h_trunk = true) override;
  void encode_verify(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step,
                     bool v8 = false) override;
  void encode_flush(Encoder& e, Seq& s) override;
  void encode_mtp_prefill(Encoder& e, Seq& s, int pos0, int R, int h_row0) override;
  void encode_mtp_prefill_group(Encoder& e, Seq& s, int pos0, const int* Ts, int nc) override;
  std::string formats_summary() const override;
  bool block_drafter_ok() const override { return false; }

  // Debug (tools/fn_eval): the residual stream after the last pass, rows of hc * hidden floats
  Buffer residual() const { return x_; }
  // Tools: prefills stop after this many layers (the residual then holds their output; -1 = all)
  void set_debug_layers(int n) { dbg_layers_ = n; }
  // Teacher forcing: the token the last pass sampled is replaced on the CPU, so its n-gram rows are gathered anew
  void ple_forget() { pend_ = Pending{}; }
  // the last draft's logits (the draft head's rows; the full vocabulary with NI_FULL_DRAFT=1)
  Buffer draft_logits() const { return dlog_; }
  // the n-gram table, for tools
  const NgramTable& ngram_table() const { return table_; }
  const NgramHash& ngram_hash() const { return hash_; }
  // Q2_0 routed experts through the expert store (src/model/expert_store.h), nullptr when they are stacked planes
  q20::ExpertStore* expert_store() { return store_.get(); }

 protected:
  void grow_kv(Seq& s, int cap) override;

 private:
  void load(const std::string& dir);
  void load_mtp4(const std::string& dir);
  void alloc4();
  void alloc4_group();  // the group slot buffers; init_store calls it only while the store is incomplete
  const Tensor* norm32(const Tensor* t, float offset);  // fp32 copy of a norm weight, + offset
  GatedRes load_gr(const std::string& prefix, bool inject, const class WeightStore* w = nullptr);
  int attn_layers() const;

  // gated residual over R rows of X from row x_row0 (decode / verify / MTP: u fp32 rows) or T rows (prefill: u half
  // rows in xh_); X is x_ or the MTP head's state; the block output y is yb_
  void enc_gr_rows(Encoder& e, const GatedRes& g, int R, const Buffer& X, size_t x_row0, bool inject,
                   const Buffer& inj_prev, const Buffer& inj_out, const Buffer& u, uint32_t u_ld);
  void enc_gr_prefill(Encoder& e, const GatedRes& g, int T, const Buffer& X, bool inject, const Buffer& inj_prev,
                      const Buffer& inj_out);
  void enc_inject(Encoder& e, int R, const Buffer& X, const Buffer& inj);  // X += inj * y, y = 0 (yb_ rows)
  void enc_expand(Encoder& e, int R);                     // x_ rows = 4 copies of the yb_ rows, yb_ = 0
  // R rows (verify / MTP history) of projections sharing x (rows x_ld floats apart) on the row kernels
  // (zqwen4v.metal: a simdgroup per token row, the decode kernels' arithmetic per row): 4/8-bit segments in one
  // dispatch, other formats one dispatch per format; y rows y_ld (0: N) floats apart
  struct RowSeg {
    const QLinear* L;
    Buffer y;
    bool add = false;
    uint32_t y_ld = 0;
  };
  void enc_rows(Encoder& e, const std::vector<RowSeg>& segs, const Buffer& x, int R, uint32_t x_ld = 0,
                bool pair8 = false);
  // the MoE of R rows of xrn_ into yb_ (router on the row kernels, the 512-expert top-k per row, the experts)
  void enc_moe4_rows(Encoder& e, const Layer& L, int R, bool mtp_layer);
  // the same with EXL3 routed experts (src/model/moe_exl3.h): the shared expert on the row kernels, then the routed
  // ones
  void enc_moe_exl3_rows(Encoder& e, const Layer& L, int R);
  // streamed Q2_0 experts (src/model/expert_store.h)
  q20::Experts q20_fetch(Encoder& e, const Layer& L, const Buffer& ids, uint32_t n, int T,
                         const std::function<void()>& between) override;
  bool q20_arena() const override { return store_ != nullptr; }
  int layer_index(const Layer& L) const;  // its index in layers_, -1 for the MTP head's
  // the Q2_0 experts of decode / verify rows (routing in idx: R * k ids), through the store when there is one; with a
  // next layer to predict, its router ran on the same rows into pidx_
  q20::Experts q20_rows_fetch(Encoder& e, const Layer& L, int R, const Buffer& idx, bool pred);
  void init_store(const std::string& dir);
  size_t mem_budget_arg_ = 0;
  // the seam-wrap gate: q20_fetch reads the store's layer 0 ahead at a chunk's last layer only while this is set,
  // and encode_prefill sets it around enc_prefill_layers to q20::seam_wraps(sample) - every chunk but the prompt's
  // last (the engine sets `sample` on exactly that chunk); q20_fetch's choice itself is q20::seam_next_layer, the
  // pair fn_store_test's case_seam pins. Passes that do not run encode_prefill (decode, verify) and a split
  // model's tail (encode_stage: a Remote::Pass carries no chunk end) leave it false and never wrap. The engine's
  // background fills (the draft path's history prefill) pass sample=false on every chunk, so their last chunk
  // wraps too; decode follows only a draft's fill, where the sweep then costs one layer's read ahead of the first
  // draft token - bounded, read-ahead only, and off the prompt paths.
  bool seam_more_ = false;
  // layer-major prefill groups (NI_PREFILL_GROUP=G, 1..8; the shipped default 4; =1 keeps the chunks one at a
  // time, the merge gate's B arm and debugging alone; an all-resident expert set never groups, q20::group_chunks).
  // The G chunks' state
  // - the residual x_, the block output yb_, the injection weights injA_ / injM_ - lives in per-chunk slots (xg_
  // etc., alloc4_group, allocated only while the streamed store is incomplete) swapped into the members per turn,
  // because Metal binds buffers at encode time; group_swap /
  // group_restore do the swap. A turn's read-ahead follows the group's successor table, not the chunk path's: the
  // successor of turn (i, c) is (i, c + 1) while chunks remain - the same layer, resident - so an inner turn reads
  // nothing ahead (group_inner_); a visit's last turn admits layer i + 1's whole set for the whole visit; and only
  // the group's final turn wraps to layer 0 (seam_more_), for the chunk after the group. group_floor_ is the sweep
  // floor (#624) a turn passes to the store - its visit's layer index, so the sweep cannot evict keys the group's
  // later turns still need; -1 at the wrapping final turn and outside groups. in_group_ keeps NI_HC_PRECOMB off
  // inside a group (its pending combine folds into the next layer's gated residual, which crosses turns under
  // grouping). group_parity_ alternates the two sets of n-gram row staging (ple_pfg_): the host gathers group g + 1's
  // rows while the GPU runs g.
  int prefill_group_ = 1;
  int group_parity_ = 0;
  int group_floor_ = -1;
  bool group_inner_ = false;
  bool in_group_ = false;
  // the engine's start plan (context and slots; 0 = none, a tool that builds the model directly), which the
  // store's available-memory guard reserves KV for: min(context, q20::kAutoKvCtxCap) positions per slot
  int planned_ctx_ = 0, planned_slots_ = 0;
  // DFlash2 capture of layer `layer`'s output rows [pos0, pos0 + R): their 4 streams' mean (stream_mean)
  void enc_capture4(Encoder& e, int layer, int pos0, int R);
  // whether layer `layer`'s enc_capture4 skips (its early-returns exactly): the NI_HC_PRECOMB fold gate
  bool capture_noop(int layer, int pos0, int R) const;
  void enc_moe4_down(Encoder& e, const Layer& L, int R, const Buffer& idx, const Buffer& wts);
  // R rows (verify / MTP) of the token mixers on the row GEMVs; u in xrn_, output to yb_
  void enc_gdn4_rows(Encoder& e, Seq& s, int li, const Layer& L, int R);
  void enc_attn4_rows(Encoder& e, const Seq& s, const Layer& L, const Qwen4Layer& F, const Buffer& kc, const Buffer& vc,
                      const Buffer& ik, const Buffer& bk, int pos, int R);
  // the MTP layer over R rows of mtpX_ at positions mpos0.. (moe: also its MoE; the history passes skip it)
  void enc_mtp_layer_rows(Encoder& e, Seq& s, int mpos0, int R, bool moe);
  // the blocks; their outputs go to yb_
  void enc_gdn4(Encoder& e, Seq& s, int li, const Layer& L, const Buffer& u);
  void enc_attn4(Encoder& e, Seq& s, int li, const Layer& L, const Qwen4Layer& F, int pos, const Buffer& u);
  void enc_moe4(Encoder& e, const Layer& L, const Buffer& u);
  void enc_attn4_prefill(Encoder& e, Seq& s, int li, const Layer& L, const Qwen4Layer& F, int pos0, int T);
  void enc_qsa(Encoder& e, const Seq& s, const Qwen4Layer& F, const Buffer& iqk, uint32_t ld, int R, int pos0,
               const Buffer& ik, const Buffer& bk);
  // Sparse attention: a row at position pos selects blocks once it sees more than budget + ratio - 1 tokens
  bool qsa_sparse(int pos) const { return (pos + 1) / c4_.ix_ratio > c4_.ix_budget / c4_.ix_ratio; }
  // block scores and key lists (idx_ / cnt_ rows 0..R-1) of R rows at positions pos0.. whose index queries are iq_ rows
  // iq_row0..; the score scratch grows on demand (between passes)
  void regrow(Encoder& e, Buffer& b, size_t bytes, const char* label);  // (qscore_ / qmask_, see the .cpp)
  void enc_qsa_select(Encoder& e, const Buffer& bk, int R, int pos0, int iq_row0);
  // the same two as one qsa_all dispatch (prep + block + score + select, NI_QSA_FUSE): the same computation,
  // block keys and scores a few ulp off (see the kernel in qwen4.metal); only for the shapes its 1024-thread
  // threadgroups cover
  bool qsa_fusable(int R) const;
  void enc_qsa_fused(Encoder& e, const Seq& s, const Qwen4Layer& F, const Buffer& iqk, uint32_t ld, int R, int pos0,
                     const Buffer& ik, const Buffer& bk);
  // the same as bit rows (mask rows row0.. of qmask_, mwords words each) for the masked prefill attention
  void enc_qsa_mask(Encoder& e, const Buffer& bk, int R, int pos0, int row0, uint32_t mwords);
  // attention of R rows over their key lists (q rows qo_ from q_row0; output attn_out_ rows from q_row0)
  void enc_attn_sparse(Encoder& e, const Buffer& kc, const Buffer& vc, int cap, int R, bool partials, int q_row0);
  // the n-gram embedding block at layer c4_.ple_layer: rows staged in `rows` (fp32 for R rows, half for prefill)
  void enc_ple_rows(Encoder& e, Seq& s, int R, int pos0, const Buffer& rows);
  void enc_ple_prefill(Encoder& e, Seq& s, int T, int pos0, const Buffer& rows, Checkpoint* snap, int snap_row);
  // CPU: the n-gram rows of tokens tok[pos0 .. pos0 + R) into dst (fp32 or half rows of hidden)
  void gather_rows(const int* tok, int pos0, int R, void* dst, bool half_rows) const;
  // the decode staging slot for token `pos`: gathered now, or by the completion handler of the pass that samples it
  struct PleWait {
    Buffer rows;
    uint64_t ticket = 0;  // 0: ready now
  };
  PleWait ple_rows_for(Seq& s, int pos);
  void ple_after_sample(Encoder& e, Seq& s, int pos);  // the pass samples tokens[pos]: gather its rows when done

  // The trunk layers held here ([lo_, hi_)) of a decode step, a prefill chunk and a verify; the rows' state before
  // layer lo_ is in x_ (and the previous block's output in yb_ / injM_), after layer hi_ - 1 there too. pw / rows /
  // ticket: the n-gram rows (the head's)
  void enc_step_layers(Encoder& e, Seq& s, int pos, const PleWait& pw);
  void enc_prefill_layers(Encoder& e, Seq& s, int pos0, int T, const Buffer& rows, Checkpoint* snap, int snap_row);
  // one layer's prefill block over one chunk (the loop body above; the group path calls it per visit and chunk)
  void enc_prefill_layer(Encoder& e, Seq& s, int i, int pos0, int T, const Buffer& rows, Checkpoint* snap,
                         int snap_row);
  // a group's turn binds chunk c's slots to the members the encode functions read; save/restore brackets the group
  struct GroupSave {
    Buffer x, yb, injA, injM;
  };
  GroupSave group_save() const { return {x_, yb_, injA_, injM_}; }
  void group_swap(int c) {
    x_ = xg_[size_t(c)];
    yb_ = yg_[size_t(c)];
    injA_ = injAg_[size_t(c)];
    injM_ = injMg_[size_t(c)];
  }
  void group_restore(const GroupSave& sv) {
    x_ = sv.x;
    yb_ = sv.yb;
    injA_ = sv.injA;
    injM_ = sv.injM;
  }
  void enc_verify_layers(Encoder& e, Seq& s, int pos, int R, const Buffer& rows, uint64_t ticket);
  // The head of a split model: layers [hi_, layers) of the pass on the other Macs (remote_), its rows' state out of and
  // back into x_ / yb_ / injM_; it waits on the CPU (the pass so far committed, the rest encoded once the rows are
  // back); nothing without a split
  void enc_hop(Encoder& e, Remote::Kind k, Seq& s, int pos, int R, const Checkpoint* snap = nullptr, int snap_row = 0);
  std::shared_ptr<Remote> remote_;
  void* hop_ev_ = nullptr;  // MTLSharedEvent of the hops: the GPU's ready values
  uint64_t hop_v_ = 0;

  Qwen4ExpConfig c4_;
  int dbg_layers_ = -1;
  // streamed experts: the store, and the next layer's predicted routing (router logits, top-k ids and weights of up
  // to kMaxVerify rows)
  bool q20_pack_ = false, stream_ = false;
  // added to the zero-centred RMSNorm weights at load: 1 (Qwen's release, Nace AI's packs), 0 where a converter added
  // it already (Sushi's packs)
  float norm1_ = 1.0f;
  std::unique_ptr<q20::ExpertStore> store_;
  Buffer plog_, pidx_, pwts_;
  std::vector<Qwen4Layer> l4_;
  GatedRes mix_;              // the final mix (no injection)
  // PLE
  QLinear ple_key_, ple_val_;
  const Tensor *ple_nk_ = nullptr, *ple_nq_ = nullptr, *ple_nc_ = nullptr, *ple_conv_ = nullptr;
  NgramHash hash_;
  NgramTable table_;
  // MTP head
  Qwen4Layer mtp4_;
  GatedRes mtp_mix_;
  QLinear mtp_fce_, mtp_fch_;
  const Tensor *mtp_ne_ = nullptr, *mtp_nh_ = nullptr;  // pre_fc_norm_embedding / _hidden (fp32, 1 + w)
  // buffers (beside Qwen35's)
  Buffer yb_;                 // block outputs [T, hidden] (zeroed when injected)
  Buffer injA_, injM_;        // injection weights [T, hc] of the mixer's and the MoE's residual
  Buffer xn4_, xn4h_;         // normalized streams: fp32 [kMaxVerify, hc*hidden], half [T, hc*hidden]
  Buffer hch_, hchh_, hcm_;   // low-rank hidden fp32 [T, rank], half [T, rank]; mix logits fp32 [T, hc*hidden]
  Buffer iqk_, iq_;           // indexer projection [T, 640], queries [T, heads, dim]
  Buffer ple_k_, ple_v_;      // n-gram key [T, hc*hidden] (also gvn), value [T, hidden]
  Buffer ple_pf_[2];          // prefill staging (half [T, hidden]), alternating chunks
  // layer-major groups (NI_PREFILL_GROUP >= 2; alloc4_group, while the store is incomplete): the chunks' x_ / yb_ /
  // injA_ / injM_ slots, and two sets of
  // G n-gram stagings (the host gathers group g + 1 while the GPU runs g)
  std::vector<Buffer> xg_, yg_, injAg_, injMg_, ple_pfg_;
  Buffer ple_dec_[4];         // decode / verify staging (fp32 [kMaxVerify, hidden]), a ring
  Buffer mtpX_;               // the MTP head's 4-stream state [T, hc*hidden] (mtp_post_: its rows after the layer)
  Buffer qscore_, qidx_, qcnt_;  // sparse attention: block scores [rows, blocks], key lists [rows, i_ld], counts
  Buffer qmask_;                 // sparse prefill: key bit rows [T, ceil(keys / 32)]
  Buffer xr2_, hinv_;         // one-row gated residual (hc_down / hc_up): the injected row, its streams' inverse RMS
  Buffer hpart_;              // row gated residual: hc_down_sk's partial sums [S*D / 1024, kMaxVerify, rank]
  Buffer hx2_, hstat_;        // folded gated residual (NI_HC_FOLD): injected rows [kMaxVerify, S*D], per-chunk stats
  uint32_t qidx_ld_ = 0;
  int ple_pf_flip_ = 0, ple_dec_next_ = 0;
  void* ple_ev_ = nullptr;    // MTLSharedEvent: tickets of gathers done by completion handlers
  std::atomic<uint64_t> ple_ticket_{0};
  struct Pending {
    const Seq* s = nullptr;
    int pos = -1;
    uint64_t ticket = 0;
    int slot = 0;
  } pend_;
};

// A Qwen3.5-family model of either layout: Qwen4Exp for a qwen4_exp checkpoint, else Qwen35 (tools)
std::unique_ptr<Qwen35> load_qwen(const std::string& model_dir, int prefill_chunk = 2048);

}  // namespace ni
