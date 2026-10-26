// NI_CYCLE_TRACE's arithmetic for ni_run's DFlash2 double-buffered decode cycles (tools/ni_run.cpp, mode 3),
// apart from the run loop so a CPU test can drive it with synthetic stamps (tools/ni_cycle_trace_test.cpp):
// no Encoder, no GPU, no model.
//
// Two clocks feed a cycle's stamps, and the difference matters:
//   - host stamps: Encoder::host_ms(), mach_absolute_time scaled to ms;
//   - GPU stamps: MTLCommandBuffer.GPUStartTime / GPUEndTime * 1e3 (src/metal/gpu.mm).
// The GPU stamps are Metal's own clock, not host_ms's: a difference between the two carries an offset (it can
// be arbitrarily large, and negative — an observed run printed a commit -> start of -3.171 ms). So every field
// here is a difference within one clock — the printed decompositions then hold exactly — except the one named
// _xclock, which is kept because the queue latency it shows has no single-clock stamp, and says so.
//
// The cycle (mode 3): iteration c verifies what the drafting half committed at the end of iteration c - 1
// drafted, then commits the drafting half for iteration c + 1. Two decompositions of the cycle's wall, both
// exact:
//   host clock: cycle = encode the verify + wait for it (the wake-up is a part of the wait)
//   GPU clock:  cycle = idle verify->draft + draft + idle draft->verify + verify
#pragma once

namespace ni {

struct CycleTrace {
  // One completed cycle's stamps, in ms; *_host from Encoder::host_ms, *_gpu from GPUStartTime / GPUEndTime:
  //   cycle_begin_host:    the top of the run loop's iteration
  //   verify_commit_host:  just after the verify's commit() returned
  //   verify_done_host:    inside the verify's on_completed handler, on Metal's completion thread (a host stamp
  //                        at the GPU's end — the wake-up's other endpoint, in the host clock)
  //   wait_back_host:      when wait() returned
  //   draft_start_gpu / draft_end_gpu:    the drafting half that fed this cycle's verify
  //   verify_start_gpu / verify_end_gpu:  this cycle's verify; a chained encoder (split, signal_event) spans
  //                                       its first buffer's start to its last one's end, its stalls between
  //                                       buffers included (an expert store's SSD reads, e.g.)
  //   prev_verify_end_gpu: the previous cycle's verify's GPU end (the double buffer's cycle boundary)
  struct Stamps {
    double cycle_begin_host = 0, verify_commit_host = 0, verify_done_host = 0, wait_back_host = 0;
    double draft_start_gpu = 0, draft_end_gpu = 0, verify_start_gpu = 0, verify_end_gpu = 0;
    double prev_verify_end_gpu = 0;
  };
  void add(const Stamps& s) {
    host_cycle_ += s.wait_back_host - s.cycle_begin_host;
    host_encode_ += s.verify_commit_host - s.cycle_begin_host;
    host_wait_ += s.wait_back_host - s.verify_commit_host;
    host_wakeup_ += s.wait_back_host - s.verify_done_host;
    gpu_cycle_ += s.verify_end_gpu - s.prev_verify_end_gpu;
    gpu_idle_verify_to_draft_ += s.draft_start_gpu - s.prev_verify_end_gpu;
    gpu_draft_ += s.draft_end_gpu - s.draft_start_gpu;
    gpu_idle_draft_to_verify_ += s.verify_start_gpu - s.draft_end_gpu;
    gpu_verify_ += s.verify_end_gpu - s.verify_start_gpu;
    commit_to_start_xclock_ += s.verify_start_gpu - s.verify_commit_host;
    ++n_;
  }
  struct Mean {
    int n = 0;
    // the host clock: cycle = encode + wait, the wake-up a part of the wait
    double host_cycle = 0, host_encode = 0, host_wait = 0, host_wakeup = 0;
    // the GPU clock: cycle = idle verify->draft + draft + idle draft->verify + verify
    double gpu_cycle = 0, gpu_idle_verify_to_draft = 0, gpu_draft = 0, gpu_idle_draft_to_verify = 0, gpu_verify = 0;
    // cross-clock: carries the two clocks' offset, so it can come out negative (the GPU may also start while
    // commit() is still on the stack, before verify_commit_host was stamped); a small negative is information,
    // a large one is the offset
    double commit_to_start_xclock = 0;
  };
  int n() const { return n_; }
  Mean mean() const {
    if (!n_) return {};
    const double n = double(n_);
    Mean m;
    m.n = n_;
    m.host_cycle = host_cycle_ / n;
    m.host_encode = host_encode_ / n;
    m.host_wait = host_wait_ / n;
    m.host_wakeup = host_wakeup_ / n;
    m.gpu_cycle = gpu_cycle_ / n;
    m.gpu_idle_verify_to_draft = gpu_idle_verify_to_draft_ / n;
    m.gpu_draft = gpu_draft_ / n;
    m.gpu_idle_draft_to_verify = gpu_idle_draft_to_verify_ / n;
    m.gpu_verify = gpu_verify_ / n;
    m.commit_to_start_xclock = commit_to_start_xclock_ / n;
    return m;
  }

 private:
  double host_cycle_ = 0, host_encode_ = 0, host_wait_ = 0, host_wakeup_ = 0;
  double gpu_cycle_ = 0, gpu_idle_verify_to_draft_ = 0, gpu_draft_ = 0, gpu_idle_draft_to_verify_ = 0;
  double gpu_verify_ = 0, commit_to_start_xclock_ = 0;
  int n_ = 0;
};

}  // namespace ni
