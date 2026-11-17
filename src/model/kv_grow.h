#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

// How far a sequence's attention cache can grow within the memory budget. Growing allocates each cache buffer at the
// new size, copies the old rows into it on the GPU and then releases the old buffer, in batches: while a batch's copy
// runs, its old buffers are held beside the new ones. So a grow needs room for the new size plus what one batch holds.
// A batch is at most 256 MB and at least one buffer; with less room it shrinks to fit, down to one buffer, so a
// growing conversation is served up to its room less one buffer's old rows (kapitsa#1300: counting a whole 256 MB
// batch refused qwen3.8-27b's growing conversations at ~10.5K of the 14.3K positions the budget held).
namespace ni::kvgrow {

inline constexpr size_t kBatch = size_t(256) << 20;  // a batch's old buffers, at most (bytes)

// The least a grow from `cap` positions holds beside the new size (bytes). per: bytes per position of every buffer
// that grows. unit: of the largest single buffer.
inline size_t held(int cap, size_t per, size_t unit) {
  return std::min(size_t(cap) * unit, size_t(cap) * per);
}

struct Plan {
  int target = 0;    // positions to grow to; 0: no room
  size_t batch = 0;  // the most bytes of new buffers one copy batch allocates (always at least one buffer)
};

// room: bytes the budget has left (SIZE_MAX: no budget). Grows by at least a quarter, in 1024-position steps, when
// the room allows, else to `need` rounded up to 256 positions; never past `limit`.
inline Plan plan(int cap, int need, int limit, size_t per, size_t unit, size_t room) {
  auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
  const size_t hold = held(cap, per, unit);
  auto fits = [&](int t) { return room >= hold && size_t(t - cap) * per <= room - hold; };
  int target = std::min(limit, round_up(std::max(need, cap + cap / 4), 1024));
  if (!fits(target)) target = std::min(limit, round_up(need, 256));
  if (!fits(target)) return {};
  // a batch's old buffers are no larger than its new ones, and one buffer's old rows fit (fits())
  return {target, std::min(kBatch, room - hold - size_t(target - cap) * per)};
}

// The most positions a sequence of `cap` positions can grow to with `room` bytes left, a grow holding `hold` bytes
// beside its new size (held(); cap if it cannot grow).
inline int fit(int cap, size_t per, size_t hold, size_t room) {
  return room > hold ? cap + int(std::min<size_t>((room - hold) / per, INT32_MAX - size_t(cap))) : cap;
}

}  // namespace ni::kvgrow
