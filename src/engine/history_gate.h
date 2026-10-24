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
