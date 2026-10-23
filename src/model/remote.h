// The trunk layers other Macs hold, as a model split across Macs by layers sees them (Qwen4Exp's head; src/cluster/
// pipe.h implements it over TCP). Calls come in the order the model encodes its work, and the other Macs apply them in
// that order: what the model encodes into its command buffers here, they run between the same passes there.
#pragma once
#include "metal/gpu.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ni {

class Remote {
 public:
  virtual ~Remote() = default;
  enum class Kind : uint16_t { step = 1, prefill = 2, verify = 3 };
  struct Pass {
    Kind kind = Kind::step;
    int seq = -1;           // the sequence (Seq::rid)
    int pos = 0, rows = 1;  // the rows' first position and how many
    int n_pend = 0;         // the sequence's pending verify rows as the pass starts (its GDN layers fold them first)
    int snap = -1;          // prefill: a checkpoint (Checkpoint::rid) saved inside the pass after snap_row rows
    int snap_row = 0;
  };
  // A pass's rows. Once the GPU has set `event` to `ready`, the rows' state before the first remote layer (rows of x,
  // y and inj, x_bytes, y_bytes and inj_bytes each, from the model's buffers) goes to the other Macs. Their state after
  // the last layer comes back into the same buffers (a prefill's: x alone, its last block's injection applied), and
  // the CPU then sets `event` to `back`, which the GPU waits for. pass() never blocks.
  struct Rows {
    Buffer x, y, inj;
    size_t x_bytes = 0, y_bytes = 0, inj_bytes = 0;
    void* event = nullptr;
    uint64_t ready = 0, back = 0;
  };
  virtual void pass(const Pass& p, const Rows& r) = 0;
  // Sequences and checkpoints, by the ids the other Macs know them by
  virtual int new_seq(int limit) = 0;
  // Makes the other Macs' caches of seq hold at least `need` positions; waits for their answer (false: they lack the
  // memory)
  virtual bool grow(int seq, int need) = 0;
  virtual void drop_kv(int seq) = 0;
  virtual void reset(int seq) = 0;
  virtual int new_checkpoint() = 0;
  virtual void save(int seq, int ckpt, int pos) = 0;
  virtual void restore(int seq, int ckpt) = 0;
  virtual void flush(int seq, int n_pend) = 0;
  // images: the sequence's M-RoPE positions (rope_n rows of (t, h, w) int32, the rows after them at row + delta)
  virtual void rope(int seq, int rope_n, int rope_delta, const int32_t* table) = 0;
  // why the other Macs no longer answer ("" while they do)
  virtual std::string lost() const = 0;
};

}  // namespace ni
