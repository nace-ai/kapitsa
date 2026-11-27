// How far a conversation's attention cache grows within the memory budget (src/model/kv_grow.h). CPU only.
//
// The numbers are kapitsa#1300's: qwen3.8-27b 4-bit on a 24 GB Mac mini, 68 KB of attention cache per position
// (16 attention layers and the MTP head, 4 KV heads of 256, fp16: 2 KB per buffer per position), 16K context, a
// budget that left 14,348 positions for the conversation. An agent's next turn extends the last one, so its cache
// grows from what it holds; a growth that counts a whole 256 MB copy batch refused every turn past ~10.5K.
#include "model/kv_grow.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); \
} } while (false)

using namespace ni;

static constexpr size_t kPer = 69632;  // 68 KB: (16 + 1) layers x K, V x 4 heads x 256 x fp16
static constexpr size_t kUnit = 2048;  // one buffer: 4 heads x 256 x fp16
static constexpr int kLimit = 16384;

// What a grow holds at its peak beyond the new size: the old buffers of its largest copy batch (at least one buffer,
// never more than the whole old cache)
static size_t peak_held(int cap, const kvgrow::Plan& p) {
  const size_t old = size_t(cap) * kPer, one = size_t(cap) * kUnit;
  return std::min(old, std::max(one, p.batch));
}

// kapitsa#1300: the cache holds 10,240 positions and the room left takes 14,348 in all. The next turn's 10,238-token
// prompt needs 10,270 (+32), 10,496 once rounded up. It fits: growing holds one 20 MB buffer, not a 256 MB batch.
static void issue_1300() {
  const int cap = 10240;
  const size_t room = size_t(14348 - cap) * kPer;
  const kvgrow::Plan p = kvgrow::plan(cap, 10238 + 32, kLimit, kPer, kUnit, room);
  CHECK(p.target >= 10238 + 32);
  CHECK(size_t(p.target - cap) * kPer + peak_held(cap, p) <= room);
  // the most it can grow to now: within one buffer (+ rounding) of the 14,348 the room holds, far past 10,490
  const int fit = kvgrow::fit(cap, kPer, kvgrow::held(cap, kPer, kUnit), room);
  CHECK(fit > 14000 && fit <= 14348);
}

// An agent-style conversation from an empty cache: each turn's prompt is the last one plus ~600 tokens and asks for
// up to 1,024 more (no max_tokens: the engine reserves n + 1024 + 32, else n + 32), within a 14,348-position room.
// It is served past 13K (one buffer's old rows and the 256-position rounding short of that room), never holding more
// than the budget; counting a whole 256 MB batch stopped it at 10,200.
static void growing_conversation() {
  size_t room = size_t(14348) * kPer;
  int cap = 0, served = 0;
  for (int n = 600; n < kLimit; n += 600) {
    const int want = std::min(kLimit, n + 1024 + 32), least = std::min(kLimit, n + 32);
    // (Engine::chat: reserve(want) || reserve(least); a cache that holds the request already needs no grow)
    kvgrow::Plan p = want <= cap ? kvgrow::Plan{cap, 0} : kvgrow::plan(cap, want, kLimit, kPer, kUnit, room);
    if (!p.target) p = least <= cap ? kvgrow::Plan{cap, 0} : kvgrow::plan(cap, least, kLimit, kPer, kUnit, room);
    if (!p.target) break;
    CHECK(size_t(p.target - cap) * kPer + peak_held(cap, p) <= room);
    room -= size_t(p.target - cap) * kPer;
    cap = p.target;
    served = n;
  }
  std::printf("growing conversation: served up to %d tokens of %d positions' room (cache %d)\n", served, 14348, cap);
  CHECK(served >= 13000);
}

// With room to spare a grow is what it always was: a quarter more in 1024-position steps, copied in 256 MB batches
static void roomy() {
  const kvgrow::Plan p = kvgrow::plan(8192, 8200, kLimit, kPer, kUnit, SIZE_MAX);
  CHECK(p.target == 10240 && p.batch == kvgrow::kBatch);
  const kvgrow::Plan q = kvgrow::plan(8192, 8200, kLimit, kPer, kUnit, size_t(8) << 30);
  CHECK(q.target == 10240 && q.batch == kvgrow::kBatch);
  const kvgrow::Plan first = kvgrow::plan(0, 3000, kLimit, kPer, kUnit, size_t(1) << 30);  // nothing to copy
  CHECK(first.target == 3072);
  CHECK(kvgrow::plan(16000, 16384, kLimit, kPer, kUnit, SIZE_MAX).target == kLimit);  // never past the limit
}

// Too little room: no grow, and the fit says what the cache can reach; a grow never needs more than the room
static void tight() {
  CHECK(!kvgrow::plan(10240, 10270, kLimit, kPer, kUnit, 0).target);
  CHECK(!kvgrow::plan(10240, 10270, kLimit, kPer, kUnit, size_t(256) * kPer).target);  // the rows, not the copy
  CHECK(kvgrow::fit(10240, kPer, kvgrow::held(10240, kPer, kUnit), 0) == 10240);
  for (int cap = 0; cap <= 15360; cap += 1024)
    for (size_t room = 0; room < size_t(8000) * kPer; room += size_t(97) * kPer + 12345) {
      const kvgrow::Plan p = kvgrow::plan(cap, cap + 1, kLimit, kPer, kUnit, room);
      if (!p.target) continue;
      CHECK(p.target > cap && p.target <= kLimit);
      CHECK(size_t(p.target - cap) * kPer + peak_held(cap, p) <= room);
      CHECK(p.target <= kvgrow::fit(cap, kPer, kvgrow::held(cap, kPer, kUnit), room));
    }
}

// Inkling-2-Mini 1M context: 9 global layers (8 trunk + 1 MTP) x K, V x 8 heads x 128 x fp16 = 36 KB/pos,
// with each single buffer 2 KB/pos (2.0 GiB at 1,048,576 positions).
static void inkling2_1m() {
  constexpr size_t kInkPer = 36864;
  constexpr size_t kInkUnit = 2048;
  constexpr int kInkLimit = 1048576;
  CHECK(kvgrow::held(524288, kInkPer, kInkUnit) == size_t(1) << 30);
  CHECK(kvgrow::held(kInkLimit, kInkPer, kInkUnit) == size_t(2) << 30);

  const kvgrow::Plan p0 = kvgrow::plan(0, kInkLimit, kInkLimit, kInkPer, kInkUnit, size_t(36) << 30);
  CHECK(p0.target == kInkLimit);

  const int mid = 524288;
  const size_t need_mid = size_t(kInkLimit - mid) * kInkPer + kvgrow::held(mid, kInkPer, kInkUnit);
  CHECK(kvgrow::plan(mid, kInkLimit, kInkLimit, kInkPer, kInkUnit, need_mid).target == kInkLimit);
  CHECK(!kvgrow::plan(mid, kInkLimit, kInkLimit, kInkPer, kInkUnit, need_mid - 1).target);

  const int hi = 900000;
  const size_t need_hi = size_t(kInkLimit - hi) * kInkPer + kvgrow::held(hi, kInkPer, kInkUnit);
  CHECK(kvgrow::plan(hi, kInkLimit, kInkLimit, kInkPer, kInkUnit, need_hi).target == kInkLimit);
  CHECK(!kvgrow::plan(hi, kInkLimit, kInkLimit, kInkPer, kInkUnit, need_hi - 1).target);
}

int main() {
  issue_1300();
  growing_conversation();
  roomy();
  tight();
  inkling2_1m();
  std::printf("kv_grow_test: ok\n");
  return 0;
}
