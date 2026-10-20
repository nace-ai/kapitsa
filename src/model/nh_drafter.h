// Block drafters for Nemotron-H targets: a drafter reads the target's hidden states at a few layers (captured during
// the target's prefill and verify passes), keeps them as per-layer context K/V in a sliding-window ring, and drafts
// the 7 tokens after an anchor in one pass of an 8-row block; the target verifies them (NemotronH::encode_verify +
// encode_accept), so the output is the target's. Two heads on that backbone:
//   LiliCorr (lilicorr.h): NVIDIA's LiLiCorr drafters of Nemotron 3.5 Bolt and Super-VL (dynamic convolutions,
//     bidirectional block, a candidate-lattice head)
//   DSpark (dspark.h): NVIDIA's DSpark drafter of Nemotron 3.5 Lightning (a causal block, a low-rank Markov head over
//     the previous drafted token)
// The engine drives either through this interface; the per-sequence state is LiliSeq for both.
#pragma once
#include "model/nemotron_h.h"

#include <memory>
#include <string>

namespace ni {

struct LiliSeq;

class NhDrafter {
 public:
  static constexpr int kB = 8;  // block: anchor + 7 proposals
  virtual ~NhDrafter() = default;

  // the drafter of a Nemotron model folder (its dflash/, lilicorr/ or dspark/ subfolder holding one for the model's
  // hidden size); "" if none
  static std::string find(const std::string& model_dir);
  // "lilicorr" / "dspark" for a drafter folder, "" if neither
  static std::string kind(const std::string& dir);
  // memory: weights as loaded and buffers; *per_slot: one sequence's ring (a slot holds three: live + 2 snapshots)
  static size_t plan_bytes(const std::string& dir, size_t* per_slot);
  static std::unique_ptr<NhDrafter> open(NemotronH& target, const std::string& dir);
  // for logs: "LiLiCorr", "DSpark"
  virtual const char* name() const = 0;
  // the context window: the last rows a draft attends to (the prompt rows a prefill must capture)
  virtual int window() const = 0;

  virtual std::unique_ptr<LiliSeq> new_seq() = 0;
  static void reset(LiliSeq& d);
  // prefix reuse: context back to pos (ring entries of positions >= pos are dead)
  static void rewind(LiliSeq& d, int pos);
  virtual void encode_copy(Encoder& e, const LiliSeq& src, LiliSeq& dst) = 0;

  // ---- context ----
  // Prompt rows [start, n): begin_prefill arms the target capture of the rows the window still sees; the target's
  // prefill passes capture them; encode_prefill_commit projects them into d (d.ctx = n); end_prefill disarms.
  virtual void begin_prefill(int n, int start) = 0;
  virtual void encode_prefill_commit(Encoder& e, LiliSeq& d) = 0;
  virtual void end_prefill() = 0;
  // Decode: arm_verify(P) before the target's verify at P (captures its rows); encode_commit(e, d, m) commits verify
  // rows 0..m-1 at positions d.ctx .. d.ctx + m - 1.
  virtual void arm_verify(int P) = 0;
  virtual void disarm() = 0;
  virtual void encode_commit(Encoder& e, LiliSeq& d, int m) = 0;

  // ---- drafting: block [tokens[P], mask x 7] at positions P..P+7 (d.ctx == P) -> tokens[P+1 .. P+7], and the
  // coming verify's depth in the target's live rows ----
  virtual void encode_draft(Encoder& e, Seq& s, LiliSeq& d, int P) = 0;

  // ---- double-buffered decode (a drafter that chains()): the next cycle's drafting half, encoded and committed right
  // behind the verify at P (+ its acceptance), before the host knows the outcome. It commits the verify's rows
  // 0..a (a: the acceptance's count, read on the GPU) and drafts from the anchor P + a + 1 the acceptance wrote. Needs
  // d.ctx == P; once the host has read a, it keeps m <= a + 1 rows: d.ctx = P + m, d.hi = max(d.hi, P + a + 1) (the
  // ring holds all a + 1). Nothing else may use the drafter or the target's live rows in between. ----
  virtual bool chains() const { return false; }
  virtual void encode_draft_chained(Encoder& e, Seq& s, LiliSeq& d, int P);

  // ---- batched decode: one verify pass over several sequences' rows writes each one's captures into its own stage
  // (LiliSeq::stage, by row) instead of the shared verify capture. stage_init: d's stage and its hook (once);
  // encode_commit_staged: its staged rows into the ring (d.ctx += d.staged, d.staged = 0) ----
  void stage_init(LiliSeq& d);
  void encode_commit_staged(Encoder& e, LiliSeq& d);

 protected:
  // the verify capture hook (its row width and layer slots), and the commit of feature rows [0, R) of `feat` at
  // positions pos0.. into d's ring
  virtual const NhCapture& verify_hook() const = 0;
  virtual void enc_commit_rows(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0) = 0;
};

}  // namespace ni
