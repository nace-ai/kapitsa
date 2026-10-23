#pragma once
// Whether a request waits for the SSD copy of its conversation that is still being written (Engine::disk_lookup), and
// for how long. Model-free so that tools/api_test.cpp can check it.
#include <algorithm>

namespace disk_wait {

struct Decision {
  bool wait = false;
  double limit_ms = 0;  // the longest wait
  double write_ms = 0;  // the estimate it was weighed with (0: not measured)
  double prefill_ms = 0;
};

constexpr double kMaxWaitMs = 3000;
constexpr int kUnmeasuredMinRows = 2048;  // before both speeds are measured: wait only for a copy that saves this many rows

// ahead_bytes: bytes still to write, up to and including the copy the prompt continues (the writer takes them in order);
// rows: how many rows of the prompt that copy saves over what the slot holds; write_bpms: bytes per ms the SSD takes
// captures at; prefill_tpms: rows per ms the engine reads a prompt at (each 0 until measured).
// It waits when the write is estimated to finish sooner than prefilling those rows, and never past kMaxWaitMs.
inline Decision decide(double ahead_bytes, int rows, double write_bpms, double prefill_tpms) {
  Decision d;
  d.limit_ms = kMaxWaitMs;
  if (rows <= 0) return d;
  if (write_bpms > 0 && prefill_tpms > 0) {
    d.write_ms = ahead_bytes / write_bpms;
    d.prefill_ms = rows / prefill_tpms;
    d.limit_ms = std::min(kMaxWaitMs, d.prefill_ms);
    d.wait = d.write_ms < d.limit_ms;
  } else {
    d.wait = rows >= kUnmeasuredMinRows;
  }
  return d;
}

}  // namespace disk_wait
