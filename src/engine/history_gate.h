#pragma once
#include <cstdint>

// When Engine::prefill_history may take the GPU turn: the server has been quiet for `idle_ms` (no request started in
// that long, none running). Fed the request counter, the running count and the time (ms) at each poll.
struct QuietGate {
  double idle_ms = 250;
  uint64_t seen = 0;
  double quiet_since = 0;
  QuietGate(double idle, uint64_t requests, double now) : idle_ms(idle), seen(requests), quiet_since(now) {}
  bool ready(uint64_t requests, int active, double now) {
    if (requests != seen) seen = requests, quiet_since = now;
    return active == 0 && now - quiet_since >= idle_ms;
  }
};

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

// Which slot a history prefill takes when its conversation's slot was used by another request meanwhile (the other
// conversation's, with --slots 1): the one a new conversation would take (as Engine::pick_slot ranks them), so no
// continued conversation is evicted while an empty or one-off slot exists; least recently used within a rank. -1: none.
struct HistorySlotView {
  bool usable = true;    // not busy, and the history fits it
  bool empty = false;    // no tokens
  bool one_off = false;  // a short conversation never continued (a title or summary request)
  uint64_t last_used = 0;
};
// A continued conversation's slot is taken only when it is the only slot (--slots 1: both conversations share it, and the
// one it holds is on the SSD already when it is read back later); with more slots a history prefill that finds no empty or
// one-off slot gives up, as before.
inline int pick_history_slot(const std::vector<HistorySlotView>& v) {
  int pick = -1;
  auto rank = [&](const HistorySlotView& s) { return s.empty ? 0 : s.one_off ? 1 : 2; };
  for (size_t i = 0; i < v.size(); ++i) {
    if (!v[i].usable) continue;
    if (rank(v[i]) == 2 && v.size() > 1) continue;
    if (pick < 0 || rank(v[i]) < rank(v[size_t(pick)]) ||
        (rank(v[i]) == rank(v[size_t(pick)]) && v[i].last_used < v[size_t(pick)].last_used))
      pick = int(i);
  }
  return pick;
}

// A history prefill running between chunks gives the turn up when a request started since the work began: the rows
// done so far stay (a state of the rendered history, saved like any other) and the request does not wait for the rest.
inline bool history_should_yield(uint64_t requests_seen, uint64_t requests_now) { return requests_now != requests_seen; }

// What tells a conversation's requests apart: a hash of each message after the system prompt (role and content, not
// the tool arguments' spelling, which a client may change), so a system prompt that changes between requests (a date,
// a directory, a session id) does not make one conversation look like two.
template <class Messages>
inline std::vector<size_t> history_message_hashes(const Messages& messages) {
  std::vector<size_t> h;
  for (const auto& m : messages) {
    const std::string role = m.value("role", std::string());
    if (role == "system" || role == "developer") continue;
    h.push_back(std::hash<std::string>()(role + "\x1f" + (m.contains("content") ? m["content"].dump() : std::string())));
  }
  return h;
}
// A newer request of the same conversation sends the older one's messages back, so the older reply's history prefill
// (a state of the rendered history up to that reply) is obsolete: the next question's is the one worth having.
inline bool history_superseded(const std::vector<size_t>& older, const std::vector<size_t>& newer) {
  return !older.empty() && older.size() <= newer.size() && std::equal(older.begin(), older.end(), newer.begin());
}
