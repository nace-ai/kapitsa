// The trunk layers other Macs hold, as a model split across Macs by layers sees them (Qwen4Exp's head; src/cluster/
// pipe.h implements it over TCP). Calls come in the order the model encodes its work, and the other Macs apply them in
// that order: what the model encodes into its command buffers here, they run between the same passes there.
//
// The contract an implementation keeps:
//  - Order: every call is applied after every pass() posted before it, even one whose GPU ready value has not been
//    reached yet (its command buffer not committed, or still running): a host operation (flush, save, restore, rope,
//    grow, reset, drop_kv) is queued behind the passes, never sent ahead of them.
//  - Release: a pass's event is always set to its back value, also when the other Macs are lost (then once the GPU has
//    reached ready, its rows garbage and lost() saying why); otherwise the GPU queue would wait on it for ever.
//  - pass() and every call but grow() return at once; grow() waits for the answer, and gives -1 once the Macs are lost.
#pragma once
#include "metal/gpu.h"
#include "model/model.h"

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
  // Makes the other Macs' caches of seq hold at least `need` positions; waits for their answer: the positions they
  // hold now (each rounds its growth up by its own rule; the least of theirs), or -1 when one lacks the memory
  virtual int grow(int seq, int need) = 0;
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

// A reserve() of `need` positions on the other Macs' layers of s: asked of them only past what they hold (s.rcap,
// which their answers set; not this Mac's cap, which grows by this Mac's rounding), so a failed growth is asked again
// at the next reserve. False when they cannot hold `need` (s.rcap unchanged).
inline bool remote_reserve(Remote& r, Seq& s, int need) {
  if (need <= s.rcap || need > s.limit) return true;
  const int c = r.grow(s.rid, need);
  if (c < need) return false;
  s.rcap = c;
  return true;
}

}  // namespace ni
