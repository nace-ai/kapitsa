// What the chat engine needs from a model, whatever its architecture: per-conversation state with growing KV
// caches and recurrent-state checkpoints, the prefill and decode passes. Qwen35 (the Qwen3.5 family: Gated
// DeltaNet + attention, MLX safetensors) and NemotronH (Mamba-2 + attention + LatentMoE, NVIDIA safetensors) implement it;
// their speculative decoding paths (MTP / DFlash2, LiLiCorr) are driven by the engine through the classes.
#pragma once
#include "metal/gpu.h"
#include "vision/embed.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

// One conversation's state: token buffer, recurrent states, attention KV cache.
struct Seq {
  // Positions the attention caches hold now, which is also their row stride (every kernel takes it as
  // max_seq). It grows on demand up to `limit` (Model::reserve); encoding past it is a fatal "context full".
  int cap = 0;
  int limit = 0;                   // the conversation's context: max positions
  int loops = 0;                   // looped models (Dory): the recurrent block's loop count (0: the model's default)
  Buffer tok;                      // int32 [limit + 8]
  std::vector<Buffer> conv, rec;   // per layer (recurrent layers only: GDN / Mamba-2)
  std::vector<Buffer> kc, vc;      // per layer (attention layers only), half [Hkv, cap, D]
  // speculative decoding: rows accepted by the last verify, not yet folded into conv/rec
  std::vector<Buffer> pend_qkv, pend_k, pend_g, pend_d;  // per recurrent layer
  int n_pend = 0;
  // KFlash tree verifies: the pending rows are the verify rows pend_path[0 .. n_pend) (int32, the accepted path the
  // tree acceptance writes), not rows 0 .. n_pend
  Buffer pend_path;
  bool pend_tree = false;
  int pend_pos = 0;                // position of the first pending row (Nemotron-H)
  int pend_half = 0;               // which half of pend_qkv holds the pending rows (Nemotron-H: 2 x 8 rows)
  Buffer mtp_kc, mtp_vc;           // MTP head's attention cache (positions = trunk positions)
  // Qwen3.8-Flash-Next: per attention layer (and the MTP head's) the sparse-attention indexer's raw index keys
  // (half [cap, 128]) and block keys (bf16 [cap / 4, 128]), grown with the KV cache; the n-gram embedding's conv
  // window (fp32 [32, 4 x hidden], position mod 32)
  std::vector<Buffer> ixk, ixb;
  Buffer mtp_ixk, mtp_ixb;
  Buffer pconv;
  std::shared_ptr<void> draft_vocab;  // Qwen35: the MTP draft head's rows for this conversation's tokens
  int* tokens() { return tok.as<int>(); }
  // Images (src/vision): prompt rows these runs cover take image embeddings in the prefill (the engine sets it around
  // a prompt's prefill; null: text only). Positions (Model::set_rope_positions): rows [0, rope_n) rotate by the
  // (t, h, w) M-RoPE positions in rope_pos (int32 x 3 per row), later rows by row + rope_delta; rope_n == 0 and
  // rope_delta == 0: position = row, the text-only path.
  const std::vector<vision::EmbedRun>* embeds = nullptr;
  Buffer rope_pos;
  int rope_n = 0, rope_delta = 0;
};

// Snapshot of a Seq's recurrent state at a position (attention KV needs no copy: entries
// before `pos` stay valid as long as the token prefix matches).
struct Checkpoint {
  int pos = -1;
  std::vector<Buffer> conv, rec;
  Buffer pconv;  // Qwen3.8-Flash-Next: the n-gram embedding's conv window
};

struct SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  int top_k = 20;
  uint32_t seed = 0;
};

// What a model needs, from its headers only (no weights read): for fit checks before loading and `kapitsa
// doctor`. Bytes.
struct MemPlan {
  size_t weights = 0;      // trunk (vision tower and MTP excluded), as loaded
  size_t mtp = 0;          // MTP head as loaded (bf16 linears become 4-bit), 0 without mtp.safetensors
  size_t activations = 0;  // prefill-chunk buffers (scale with the chunk)
  size_t slot_state = 0;   // per slot: recurrent state, its two checkpoints and the verify tape
  size_t kv_per_pos = 0;   // per slot and position
  size_t decode_bytes = 0; // weights one plain decode step reads (MoE: the routed share of the experts)
  size_t head_row_bytes = 0;  // one vocabulary row of the draft head (the output layer's, or its 8-bit copy)
  size_t decode_params = 0;   // weights one plain decode step multiplies (MoE: the routed experts only)
  int hidden = 0, layers = 0, ffn = 0;  // ffn: the dense FFN's intermediate size (0 for MoE)
  bool dense = false;
  bool ffn_ane = true;        // dense FFN weights in a format the Neural Engine split stages (any Kapitsa reads)
  std::string formats;        // "bf16", "4-bit group 64 + 80 modules 8-bit group 64", ...
  size_t ane = 0;             // a GPU + Neural Engine prefill split the model runs by default (Nemotron-H)
  size_t fixed(int slots) const { return weights + mtp + activations + size_t(slots) * slot_state; }
};

// Neural Engine split statistics (dense Qwen3.5 prefill)
struct AneStats {
  int chunks = 0, evals = 0, reruns = 0;  // evals: ANE evaluations of chunks that held
  double ane_ms = 0;
};

class Model {
 public:
  virtual ~Model() = default;
  virtual int vocab() const = 0;
  virtual const std::vector<int>& eos_ids() const = 0;  // the config's end-of-sequence ids

  // ---- memory ----
  virtual void set_mem_budget(size_t bytes) = 0;
  virtual std::unique_ptr<Seq> new_seq(int limit, int initial = -1) = 0;
  virtual bool reserve(Seq& s, int need) = 0;
  virtual void drop_kv(Seq& s) = 0;
  virtual size_t kv_bytes_per_pos() const = 0;
  virtual void reset(Seq& s) = 0;
  virtual Checkpoint new_checkpoint() = 0;
  virtual void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) = 0;
  virtual void restore(Encoder& e, Seq& s, const Checkpoint& c) = 0;

  // ---- passes ----
  virtual void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) = 0;
  virtual void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step,
                              bool sample, Checkpoint* snap = nullptr, int snap_row = 0) = 0;
  virtual int prefill_chunk() const = 0;
  virtual bool can_snapshot() const = 0;

  // ---- GPU + Neural Engine split of the dense prefill FFN (Qwen35); none elsewhere ----
  virtual bool ane_splits(int) const { return false; }
  virtual bool ane_finish_chunk() { return true; }
  // Gives up on the split of a chunk still running (a Neural Engine job that never reports): releases the GPU's
  // waits; ane_finish_chunk() then fails and the chunk reruns on the GPU alone
  virtual void ane_abort(const std::string& why) { (void)why; }
  // Decode and verify passes encoded into Gpu::begin_grouped() encoders (independent dispatches overlap)
  virtual bool grouped_passes() const { return false; }
  virtual void ane_poll() {}
  virtual bool ane_release() { return false; }
  virtual const AneStats& ane_stats() const {
    static const AneStats none;
    return none;
  }

  // ---- images ----
  // The positions of a prompt of n rows whose images sit at `spans`, kept in s for every pass that follows (prefill,
  // decode, verify, drafts). Qwen35: interleaved M-RoPE. Default: nothing (attention without positions, as
  // Nemotron-H's). Call between requests (nothing in flight may read s).
  virtual void set_rope_positions(Seq& s, int n, const std::vector<vision::ImageSpan>& spans) {
    (void)s, (void)n, (void)spans;
  }
};

}  // namespace ni
