#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// What a save attempt did, and what that means for Slot::disk_version. A slot's version is the one the SSD holds only
// when the write of it landed, or when the SSD takes nothing from it; a save refused for lack of room, or one that
// failed while writing, leaves the slot's version unsaved (#467: such a failure used to be recorded as a save, so
// disk_safe_to_replace let another conversation replace a slot whose state was never written, and that conversation's
// next request read its prompt again).
enum class SaveOutcome : uint8_t {
  kNoRoom,         // the cache is at its limit (or the volume has none): nothing was written
  kWriteFailed,    // a piece could not be written: the limit was reached mid-save, or the write failed
  kEntryFailed,    // the pieces were written but the conversation's entry could not be
  kNothingToTake,  // the SSD takes nothing from this slot (empty, or no conversation worth storing)
  kLanded,         // written and recorded: the slot's version is on the SSD
};

// Whether this outcome makes the slot's current version the one the SSD holds.
inline bool save_landed(SaveOutcome o) {
  return o == SaveOutcome::kLanded || o == SaveOutcome::kNothingToTake;
}

// What an outcome means to the code that asked: the slot's version becomes the saved one, and the next attempt waits
// (a save that could not be written is not worth copying again straight away). A cache with no room at all is a reason
// for every slot to wait; a write that failed is a reason for that slot alone, whose copy may be too large for the room.
struct SaveVerdict {
  bool save;        // the slot's version counts as on the SSD
  bool pace_retry;  // wait before the next attempt
  bool cache_full;  // the wait is not about this slot: no room for any
};
inline SaveVerdict save_verdict(SaveOutcome o) {
  return {save_landed(o), !save_landed(o), o == SaveOutcome::kNoRoom};
}

// Whether to leave a slot alone for now: waiting out a refused save on this slot, or waiting out a cache that had no room.
inline bool skip_retry(bool waiting, bool cache_full, bool same_version) {
  return waiting && (cache_full || same_version);
}

// Whether a save that landed ends the wait: only the cache-wide one (the room that came free may be this slot's too). A
// wait on one slot's own version ends when the slot changes or the clock runs out: another slot's save says nothing
// about the room this slot's pieces need.
inline bool clear_retry_on_save(bool cache_full) { return cache_full; }

// A slot as the write-behind sees it. The pick takes the one used longest ago among those that can be written: not busy,
// not on the SSD, not waiting out a refused save (one slot whose conversation never fits must not block the others).
struct WriteCandidate {
  bool busy = false, saved = false, waiting = false;
  double last_used = 0;
};
inline int pick_write(const std::vector<WriteCandidate>& v) {
  int best = -1;
  for (size_t i = 0; i < v.size(); ++i) {
    const WriteCandidate& s = v[i];
    if (s.busy || s.saved || s.waiting) continue;
    if (best < 0 || v[size_t(best)].last_used < s.last_used) best = int(i);
  }
  return best;
}

// Whether the conversation a slot holds may be replaced: it is on the SSD, there is no SSD, or it is shorter than the
// cache stores (quick to read back, not worth keeping).
inline bool safe_to_replace(bool have_disk, uint64_t version, uint64_t disk_version, size_t tokens, int min_pos) {
  if (!have_disk || version == disk_version) return true;
  return int(tokens) < min_pos;
}
