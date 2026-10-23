// KFlash around every decode: the one place that says which proposal source a model decodes with and what a cycle
// of it did, and the budget for the sources with no drafter model (prompt lookup).
//
// KFlash's rule is the same for every model: a cycle verifies the rows its budget picks (1 .. N) of whatever the
// source proposes, and 1 row is plain decoding. The drafter models (DFlash2, LiLiCorr, DSpark) and the MTP head have
// their budgets (kflash_budget.h and the loops in engine.cpp); a model with no source decodes 1 row per cycle, which
// is a plain step. This file holds
//  - KfWrap: the source's name and a request's cycle counts (the startup log's and the usage's "KFlash (<source>)");
//  - KfLookup: the budget of the prompt-lookup source (Nemotron Nano, Flint with --spec): how many of the proposed
//    tokens a cycle verifies, from the acceptance the verifies recorded and the model's cost of an extra row, so that
//    where a verify's extra rows cost more than the tokens they win, the cycle verifies the pending token alone.
// KfLookup only ever lowers the drafts a cycle verifies from the source's own count (never raises it), so a model
// whose drafts pay keeps the cycles it had. NI_KFLASH=0 or NI_KFLASH_WRAP=0 turns the wrapper off (the cycles are the
// source's own, as before; nothing is reported).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace ni {

// NI_KFLASH=0 and NI_KFLASH_WRAP=0 both turn the wrapper off (NI_KFLASH=0 also keeps the drafters' chains)
inline bool kflash_wrap_on() {
  static const bool on = [] {
    for (const char* name : {"NI_KFLASH", "NI_KFLASH_WRAP"}) {
      const char* v = getenv(name);
      if (v && *v && atoi(v) == 0) return false;
    }
    return true;
  }();
  return on;
}

// A request's cycles by one source
struct KfWrap {
  std::string source = "none";  // "DFlash2", "LiLiCorr", "MTP", "prompt lookup", ... or "none" (plain decoding)
  long cycles = 0, tokens = 0;  // decode cycles and the tokens they wrote
  void cycle(int m) { ++cycles, tokens += m; }
  double tokens_per_cycle() const { return cycles > 0 ? double(tokens) / double(cycles) : 0.0; }
};

// The budget of prompt lookup. A cycle with `avail` proposed tokens verifies d of them (0 .. avail) in a pass of d + 1
// rows. Verifying d gives 1 + sum over j <= d of (the chance that the first j proposals were accepted) tokens and
// costs 1 + r d steps (r: what each extra row costs in decode steps, a constant of the model: 0.6 on Flint, where a
// pass of 2 / 4 / 8 rows costs 1.6 / 2.8 / 5.2 steps), so the budget takes the d that gives the most tokens per step.
// The chances are the recorded acceptance by position (a decaying count, a prior of 0.5 worth 2 proposals). Nothing
// in the choice reads a clock, so the rows of a cycle, and with them the text of a pass of 2+ rows, are a function of
// the tokens: the same request decodes the same text under load. Until 16 proposals were verified it takes the
// source's own count. A d of 0 records nothing about the proposals, so one cycle in 64 with proposals verifies them
// all. observe() also keeps the measured cost of a one-row pass (c1) and of an extra row (s) in ms, for the log, to
// calibrate r.
class KfLookup {
 public:
  static constexpr int kMax = 8;  // proposals by position
  KfLookup(int k_max, double row_cost) : k_(std::min(k_max, kMax)), r_(row_cost) {}

  // drafts to verify now, of `avail` proposed (0: the pending token alone)
  int pick(int avail) {
    avail = std::min(avail, k_);
    if (avail <= 0) return 0;
    ++asked_;
    if (tried_ < 16) return avail;
    if (asked_ % 64 == 0) return avail;  // the probe: keeps the acceptance fresh when the budget stopped drafting
    int best = 0;
    double best_rate = 1.0;
    double e = 1, path = 1;
    for (int d = 1; d <= avail; ++d) {
      path *= p(d);
      e += path;
      const double rate = e / (1.0 + r_ * d);
      if (rate > best_rate * 1.01) best = d, best_rate = rate;  // (a 1% margin: ties keep the shorter pass)
    }
    return best;
  }

  // a cycle that verified `drafts` proposals took `ms` and accepted `accepted` of them
  void observe(int drafts, int accepted, double ms) {
    if (!(ms > 0)) return;
    if (drafts == 0) {
      c1_ = c1_n_ == 0 ? ms : c1_ + kA * (ms - c1_);
      ++c1_n_;
      return;
    }
    tried_ += drafts;
    for (int j = 1; j <= drafts && j <= kMax; ++j) {
      if (j > accepted + 1) break;  // (a proposal after the first rejected one was not tested)
      hit_[j] = hit_[j] * kD + (j <= accepted ? 1.0 : 0.0);
      try_[j] = try_[j] * kD + 1.0;
    }
    if (c1_n_ > 0) {  // the extra rows' cost per row
      const double extra = std::max(0.0, (ms - c1_) / drafts);
      s_ = s_n_ == 0 ? extra : s_ + kA * (extra - s_);
      ++s_n_;
    }
  }

  double c1() const { return c1_; }  // measured, ms: a pass of one row
  double s() const { return s_; }    // measured, ms: each extra row of a pass
  double row_cost() const { return r_; }
  // the chance that proposal j (1-based) is accepted when the ones before it were
  double p(int j) const {
    j = std::min(std::max(j, 1), kMax);
    return (hit_[j] + 1.0) / (try_[j] + 2.0);
  }

 private:
  static constexpr double kA = 0.1;   // the cost means' weight for a new cycle
  static constexpr double kD = 0.98;  // the acceptance counts' decay per proposal
  int k_;
  double r_;
  double hit_[kMax + 1] = {}, try_[kMax + 1] = {};
  double c1_ = 0, s_ = 0;
  long c1_n_ = 0, s_n_ = 0, tried_ = 0, asked_ = 0;
};

}  // namespace ni
