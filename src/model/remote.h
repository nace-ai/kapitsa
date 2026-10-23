// The trunk layers other Macs hold, as a model split across Macs by layers sees them (Qwen4Exp's head; src/cluster/
// pipe.h implements it over TCP). Calls come in the order the model encodes its work, and the other Macs apply them in
// that order: what the model encodes into its command buffers here, they run between the same passes there.
//
// The contract an implementation keeps:
//  - Order: every call is applied by the other Macs after every pass() made before it.
//  - pass() waits for the other Macs' result: it returns once their rows are back in the model's buffers, or once
//    they are lost (false; the rows are then garbage and lost() says why), never later than a timeout. The timeout
//    covers only the wait for the other Macs' answer, counted from the moment `ready` is reached: waiting for `ready`
//    (the local GPU still running the head's layers) has no deadline of its own, or one scaled to the GPU's progress,
//    so a slow local chunk is never reported as lost Macs (lost() is permanent). The GPU never
//    waits for the network: a command buffer that waits on an event for about 5 s is killed with a GPU timeout
//    (tools/event_wait_probe.cpp, macOS 27.0.1), and a Wi-Fi network can stall that long.
//  - grow() waits for the answer too (-1 once the Macs are lost); every other call returns at once.
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
  // A pass's rows: once the GPU has set `event` to `ready` (the work before the hop is done), the rows' state before the
  // first remote layer (rows of x, y and inj, x_bytes, y_bytes and inj_bytes each, from the model's buffers) goes to the
  // other Macs; their state after the last layer comes back into the same buffers (a prefill's: x alone, its last
  // block's injection applied)
  struct Rows {
    Buffer x, y, inj;
    size_t x_bytes = 0, y_bytes = 0, inj_bytes = 0;
    void* event = nullptr;
    uint64_t ready = 0;
  };
  // Waits for the other Macs' result (see the contract); false once they are lost
  virtual bool pass(const Pass& p, const Rows& r) = 0;
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
