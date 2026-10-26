// NI_CYCLE_TRACE's arithmetic (tools/ni_cycle_trace.h), CPU only: no GPU, no Encoder, no model.
//   ni_cycle_trace_test   CPU: each field decomposes the cycle's wall within 2% on each of the two clocks; the
//                         host fields do not move when the GPU clock's offset does, and the GPU fields do not
//                         either (the regression the mixed-domain stamps caused: the old trace printed a
//                         commit -> start of -3.171 ms); means over a run of varying cycles; an empty trace is
//                         zeros; the one cross-clock field carries exactly the offset and says so
#include "ni_cycle_trace.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace ni;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

// within 2% of wall: the tolerance the cycle trace's decompositions are asserted at
bool near(double a, double b, double wall) { return std::fabs(a - b) <= 0.02 * wall; }
// equal up to the last ulps of ms-scale doubles
bool same(double a, double b) { return std::fabs(a - b) < 1e-9; }

// One synthetic cycle of `wall` ms, stamped the way mode 3 does: the host spends 6 ms encoding the verify, then
// waits; the GPU idles 0.5 ms after the last verify's end, drafts for 3 ms, and 0.2 ms later starts the verify,
// which ends at the cycle's boundary (so the GPU cycle is exactly wall). `off` shifts every GPU stamp: Metal's
// clock against host_ms — arbitrary, and observed to make differences like -3.171 ms in a live run.
CycleTrace::Stamps cycle(double begin_host, double wall, double off) {
  CycleTrace::Stamps s;
  s.cycle_begin_host = begin_host;
  s.verify_commit_host = begin_host + 6.0;
  s.verify_done_host = begin_host + wall - 0.2;  // the completion handler's stamp: 0.2 ms before wait() returns
  s.wait_back_host = begin_host + wall;
  s.prev_verify_end_gpu = begin_host + off;  // the cycle's boundary on the GPU clock
  s.draft_start_gpu = begin_host + 0.5 + off;
  s.draft_end_gpu = begin_host + 3.5 + off;
  s.verify_start_gpu = begin_host + 3.7 + off;
  s.verify_end_gpu = begin_host + wall + off;
  return s;
}

double host_sum(const CycleTrace::Mean& m) { return m.host_encode + m.host_wait; }
double gpu_sum(const CycleTrace::Mean& m) {
  return m.gpu_idle_verify_to_draft + m.gpu_draft + m.gpu_idle_draft_to_verify + m.gpu_verify;
}

}  // namespace

int main() {
  // one cycle at the observed live-run offset: both decompositions hold against the wall
  {
    const double wall = 12.0, off = -3.171;
    CycleTrace t;
    t.add(cycle(1000.0, wall, off));
    const CycleTrace::Mean m = t.mean();
    check(m.n == 1, "one cycle recorded");
    check(near(m.host_cycle, wall, wall), "host cycle equals the cycle wall (12 ms)");
    check(near(host_sum(m), m.host_cycle, wall), "host fields decompose the host cycle within 2%");
    check(near(host_sum(m), wall, wall), "sum of the host fields equals wall within 2%");
    check(near(m.gpu_cycle, wall, wall), "GPU cycle equals the cycle wall (12 ms)");
    check(near(gpu_sum(m), m.gpu_cycle, wall), "GPU fields decompose the GPU cycle within 2%");
    check(near(gpu_sum(m), wall, wall), "sum of the GPU fields equals wall within 2%");
    check(near(m.host_wakeup, 0.2, wall), "wake-up is the 0.2 ms after the completion stamp, host clock only");
    check(near(m.gpu_verify, wall - 3.7, wall), "the verify pass is wall - 3.7 ms of GPU time");
    // the cross-clock field: the true gap (the GPU starts 2.3 ms before the commit stamp: 3.7 - 6.0) plus the
    // offset — the one field allowed to carry it, and the print says so
    check(near(m.commit_to_start_xclock, -2.3 + off, wall), "commit -> GPU start carries the offset, as labeled");
  }
  // the mixed-domain regression: shifting the GPU clock moves nothing but the cross-clock field
  {
    CycleTrace a, b;
    a.add(cycle(1000.0, 12.0, 0.0));
    b.add(cycle(1000.0, 12.0, 250.0));  // what a drift between the two clocks looks like
    const CycleTrace::Mean x = a.mean(), y = b.mean();
    check(same(x.host_cycle, y.host_cycle) && same(x.host_encode, y.host_encode) && same(x.host_wait, y.host_wait) &&
              same(x.host_wakeup, y.host_wakeup),
          "host fields identical under a 250 ms GPU clock shift (the old tr[4]/tr[5] carried it)");
    check(same(x.gpu_cycle, y.gpu_cycle) && same(x.gpu_idle_verify_to_draft, y.gpu_idle_verify_to_draft) &&
              same(x.gpu_draft, y.gpu_draft) && same(x.gpu_idle_draft_to_verify, y.gpu_idle_draft_to_verify) &&
              same(x.gpu_verify, y.gpu_verify),
          "GPU fields identical under a 250 ms GPU clock shift");
    check(same(y.commit_to_start_xclock - x.commit_to_start_xclock, 250.0),
          "only the cross-clock field moves, by exactly the shift");
  }
  // means over a run of varying cycles: the means equal the mean wall
  {
    CycleTrace t;
    double begin = 500.0, wall_sum = 0;
    for (int i = 0; i < 10; ++i) {
      const double wall = 10.0 + 0.5 * i;
      t.add(cycle(begin, wall, -3.171));
      begin += wall;  // the next cycle begins at this one's boundary
      wall_sum += wall;
    }
    const CycleTrace::Mean m = t.mean();
    const double mean_wall = wall_sum / 10;
    check(m.n == 10, "ten cycles recorded");
    check(near(m.host_cycle, mean_wall, mean_wall), "mean host cycle equals the mean wall (12.25 ms)");
    check(near(m.gpu_cycle, mean_wall, mean_wall), "mean GPU cycle equals the mean wall (12.25 ms)");
    check(near(host_sum(m), mean_wall, mean_wall), "sum of the mean host fields equals the mean wall within 2%");
    check(near(gpu_sum(m), mean_wall, mean_wall), "sum of the mean GPU fields equals the mean wall within 2%");
    check(near(m.host_wakeup, 0.2, mean_wall), "mean wake-up is the 0.2 ms in every cycle");
  }
  // an empty trace: no division by zero, zeros everywhere
  {
    const CycleTrace::Mean m = CycleTrace{}.mean();
    check(m.n == 0 && m.host_cycle == 0 && m.gpu_cycle == 0 && m.commit_to_start_xclock == 0,
          "an empty trace means all zeros");
  }
  std::printf("%s\n", g_fail ? "FAILED" : "all passed");
  return g_fail ? 1 : 0;
}
