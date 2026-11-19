// Tencent Hy4-flash (model_type hy_v4_flash): 42 layers of gated MLA attention with sinks (12 layers over the whole
// context below 2048 keys, 30 sliding-window layers of 512), 4 independent hyper-connection streams, a latent MoE of 512
// routed experts (top 16, MLX affine 3- or 4-bit, streamed from the SSD: src/model/moe_a4.h) plus a shared expert, a
// dense layer 0 and Engram hashed n-gram memories at 4 layers (src/model/hy4_engram.h). Text only. The reference
// semantics are transformers' modeling_hy_v4_flash.py; the blocks no other model has are in src/model/hy4_ops.h.
// Past 2048 keys the sparse layers attend to the 512 groups of 4 keys their compressed indexer selects.
// Not yet: the checkpoint's DFly2 drafter (its weights are not loaded), the SSD prefix cache.
#pragma once
#include "model/hy4_ops.h"
#include "model/qwen35.h"

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace ni {

namespace a4 {
class Store;
}
namespace hy4 {
class EngramCpu;
}

struct Hy4Layer {
  bool sliding = false;
  int engram = -1;  // index into Hy4Flash::eng_ (-1: no Engram memory before this layer)
  const Tensor *in_norm = nullptr, *post_norm = nullptr;
  hy4::Hc hca, hcm;
  // attention
  QLinear q_a, q_b, kv_a_lat, kv_a_rope, kv_b, gate, o;
  const Tensor *q_a_norm = nullptr, *kv_a_norm = nullptr, *post_proj_norm = nullptr;
  hy4::AttnLayer at;
  // compressed indexer (the full-context deepseek_sparse_attention layers): token keys and head weights from the
  // normed input, queries from q_resid (hy4_ops.h)
  bool ix = false;
  QLinear ix_tk, ix_hw, ix_q;
  hy4::IndexerWeights ixw;
  // dense MLP (layer 0)
  QLinear mg, mu, md;
  const Tensor* post_fc2_norm = nullptr;
  // MoE
  bool moe = false;
  QLinear router, sg, su, sd, lat_down, lat_up;
  const Tensor *router_bias = nullptr, *combine_norm = nullptr;
};

struct Hy4Engram {
  int layer = -1;  // the decoder layer it is added before (0-based)
  QLinear value, key[hy4::kStreams];
  hy4::EngramW w;
  std::unique_ptr<hy4::EngramCpu> cpu;
};

class Hy4Flash : public Qwen35 {
 public:
  static bool is_model(const std::string& dir);
  // What a run of this folder takes (the engine's memory plan): weights resident, the expert arena, activations
  static MemPlan plan(const std::string& dir, int chunk);
  Hy4Flash(const std::string& dir, int prefill_chunk);
  ~Hy4Flash() override;

  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override;
  size_t kv_unit_bytes_per_pos() const override;
  void grow_kv(Seq& s, int cap, size_t batch_max = size_t(256) << 20) override;
  void reset(Seq& s) override;
  // A checkpoint at pos: the sliding layers' rings (their rows of positions near pos are overwritten later); the linear
  // caches keep every row below pos, as rows past it are written again before any row reads them
  Checkpoint new_checkpoint() override;
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  bool can_snapshot() const override { return false; }
  std::string disk_format() const override { return ""; }
  bool copy_prefix(Encoder&, const Seq&, Seq&, int) override { return false; }

  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  void encode_logits_rows(Encoder& e, int T, const Buffer& out) override;
  std::string formats_summary() const override;
  bool block_drafter_ok() const override { return false; }
  // NI_HY4_DUMP_SEL=<file> (debug): every indexed layer's selection of a pass past kIndexDense keys is copied aside;
  // write_selection(), after the pass completed, appends them to the file: per layer and pass an int32 header
  // [layer, pos0, R], then uint32 [R][kIndexGroups] (row t's group ids ascending, min(G, 511) + 1 of them with
  // G = (pos0 + t) / 4, the last one G; the rest of the row unused). Nothing is copied when the variable is unset.
  void write_selection();

 private:
  void load(const std::string& dir);
  uint32_t ring_cap() const;
  hy4::LatentCache lat_of(const Seq& s, int i) const;
  hy4::IndexerCache ix_of(const Seq& s, int i) const;  // an indexed layer's (s.ixk / s.ixb), empty before grow_kv
  static constexpr size_t kIxRowBytes = 320;            // the indexer's bytes per position (hy4_ops.h, IndexerCache)
  uint32_t lat_cap_of(const Seq& s) const;
  uint32_t lat_ring_cap_ = 0;
  void alloc_hy();
  // y [R, L.N] fp32 = L . x (x fp32 [R, L.K], norm: an RMSNorm weight folded on x)
  void mm(Encoder& e, const QLinear& L, const Buffer& x, const Tensor* norm, const Buffer& y, int R);
  // the layers over R rows at positions pos0..: X_ holds the streams; rows: the Engram rows of every Engram layer
  void enc_layers(Encoder& e, Seq& s, int pos0, int R, bool prefill);
  void enc_attn_block(Encoder& e, Seq& s, int i, int pos0, int R, bool prefill);
  void enc_mlp_block(Encoder& e, int i, int R, bool prefill);
  void enc_engram_block(Encoder& e, int i, int R);
  void enc_head(Encoder& e, int R, size_t row0);  // hc_head over R rows from row0 -> u_ (fp32), pending flushed
  // Engram rows into dst: fp32 [R, 1536] per Engram layer, then uint32 [R] engram_mask (0: a pad, image or video
  // token, whose row gets no Engram output)
  void gather_engram(const int* toks, int pos0, int R, const Buffer& dst) const;
  // bytes of attention and indexer scratch one more context position adds (counted in kv_bytes_per_pos)
  size_t scratch_per_pos() const;
  void engram_after_sample(Encoder& e, Seq& s, int pos);

  hy4::HcConst hcc_;
  int hcc_window_ = 512;
  float engram_eps_ = 1.1920929e-07f;
  float route_scale_ = 3.998f;
  std::vector<Hy4Layer> hl_;
  std::vector<Hy4Engram> eng_;
  hy4::Hc hc_head_;
  std::deque<Tensor> split_;  // row views of split matrices (kv_a)
  std::unique_ptr<a4::Store> store_;
  std::vector<int> moe_layers_;
  // activations (rows: tmax_)
  Buffer X_, u_, xn_, y_, post_a_, post_m_, xh_;
  Buffer qa_, kvl_, kr_, gt_, q_, att_, attn_scratch_;
  // the indexer: raw projections, a call's selections [T][512], its scratch
  Buffer ix_tk_, ix_hw_, ix_q_, sel_, ix_scratch_;
  Buffer seldump_;                // NI_HY4_DUMP_SEL: [indexed layers][tmax][kIndexGroups] uint32
  std::vector<int> seldump_rec_;  // (layer, pos0, R) per selection copied since the last write_selection()
  bool seldump_full_ = false;
  Buffer dg_, du_, rl_, idx_, wts_, sgb_, sub_, shb_, ysb_, lat_, ylat_, moe_scratch_;
  Buffer eng_v_, eng_k_;
  // prefill Engram rows (gather_engram's layout for T rows), two of them: the host writes a chunk's rows while the
  // chunk before may still run, and the engine waits for the chunk before that one before encoding the next
  Buffer eng_pf_[2];
  int eng_pf_next_ = 0;
  Buffer eng_dec_[4];    // decode Engram rows (gather_engram's layout for 1 row), a ring
  const Buffer* pending_ = nullptr;  // the block output pending injection (y_), with its post weights
  const Buffer* pending_post_ = nullptr;
  Buffer eng_cur_;       // the rows this pass reads
  size_t eng_stride_ = 0;  // bytes between Engram layers' rows in eng_cur_
  // the async decode gather (Qwen4Exp's PLE pattern)
  struct Pending {
    const Seq* s = nullptr;
    int pos = -1;
    uint64_t ticket = 0;
    int slot = 0;
  };
  Pending pend_;
  int dec_next_ = 0;
  void* eng_ev_ = nullptr;
  std::atomic<uint64_t> eng_ticket_{0};
  uint64_t wait_ticket_ = 0;
};

}  // namespace ni
