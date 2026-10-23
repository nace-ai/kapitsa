// KFlash around every decode: the one place that says which proposal source a model decodes with and what a cycle
// of it did, and the budget for the sources with no drafter model (prompt lookup).
//
// KFlash's rule is the same for every model: a cycle verifies the rows its budget picks (1 .. N) of whatever the
// source proposes, and 1 row is plain decoding. The drafter models (DFlash2, LiLiCorr, DSpark) and the MTP head have
// their budgets (kflash_budget.h and the loops in engine.cpp); a model with no source decodes 1 row per cycle, which
// is a plain step. This file holds
//  - KfWrap: what the engine builds per request and every decode loop goes through: plan(avail) the rows of the next
//    cycle, cycle(m, rows) what it wrote; its counts are the usage's "KFlash (<source>)";
//  - KfLookup: the budget of the prompt-lookup source (Nemotron Nano, Flint with --spec): how many of the proposed
//    tokens a cycle verifies, from the acceptance the verifies recorded and the model's cost of an extra row, so that
//    where a verify's extra rows cost more than the tokens they win, the cycle verifies the pending token alone.
// KfLookup only ever lowers the drafts a cycle verifies from the source's own count (never raises it), and is off
// unless NI_KFLASH_BUDGET=1. NI_KFLASH=0 or NI_KFLASH_WRAP=0 turns the wrapper off: nothing is reported and the budget is
// off too (every proposal is verified, as before KFlash).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
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

// The prompt-lookup budget is opt-in (NI_KFLASH_BUDGET=1) and needs the wrapper on: a pass of 2+ rows rounds differently
// from decode steps, so a budget that picks other rows than the source's own would change a default model's text.
inline bool kflash_budget_on() {
  static const bool on = [] {
    const char* v = getenv("NI_KFLASH_BUDGET");
    return kflash_wrap_on() && v && *v && atoi(v) != 0;
  }();
  return on;
}

// The budget of prompt lookup. A cycle with `avail` proposed tokens verifies d of them (0 .. avail) in a pass of d + 1
// rows. Verifying d gives 1 + sum over j <= d of (the chance that the first j proposals were accepted) tokens and
// costs 1 + r d steps (r: what each extra row costs in decode steps, a constant of the model: 0.6 on Flint, where a
// pass of 2 / 4 / 8 rows costs 1.6 / 2.8 / 5.2 steps), so the budget takes the d that gives the most tokens per step.
// The chances are the recorded acceptance by position (a decaying count, a prior of 0.5 worth 2 proposals). Nothing
// in the choice reads a clock, so the rows of a cycle, and with them the text of a pass of 2+ rows, are a function of
// the tokens: the same request decodes the same text under load. Until 16 proposals were verified it takes the
// source's own count. A d of 0 records nothing about the proposals, so one cycle in 16 with proposals verifies them
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
    if (asked_ % kProbe == 0) return avail;  // the probe: keeps the acceptance fresh when the budget stopped drafting
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
  static constexpr long kProbe = 16;  // one cycle in 16 with proposals verifies them all
  static constexpr double kA = 0.1;   // the cost means' weight for a new cycle
  static constexpr double kD = 0.95;  // the acceptance counts' decay per proposal
  int k_;
  double r_;
  double hit_[kMax + 1] = {}, try_[kMax + 1] = {};
  double c1_ = 0, s_ = 0;
  long c1_n_ = 0, s_n_ = 0, tried_ = 0, asked_ = 0;
};

// A request's KFlash: the engine builds one per request and every decode loop goes through it. plan(avail) is how many
// rows the next cycle verifies given the `avail` tokens its source proposed (the pending token's row plus the drafts
// worth their rows); cycle(m, rows) records what the cycle wrote. A source with no proposals (a plain model: avail 0)
// gets 1 row, which the loop runs as its existing step, so a model that decodes plain is bit-identical to before. The
// drafter models and the MTP head keep their own budgets (kflash_budget.h) and report their cycles here.
// `lookup` is the prompt-lookup budget, built per request when it is asked for (NI_KFLASH_BUDGET=1: kflash_budget_on()),
// so what a request decodes does not depend on the requests before it. Without it plan() is the source's own count: every
// proposal is verified, as before KFlash.
struct KfWrap {
  std::string source = "none";  // "DFlash2", "LiLiCorr", "MTP", "prompt lookup", ... or "none" (plain decoding)
  std::optional<KfLookup> lookup;
  long cycles = 0, tokens = 0, rows = 0;  // decode cycles, the tokens they wrote, the rows the plan()-driven ones (plain, prompt lookup) verified
  long planned = 0;                        // plan() calls: 0 for a loop that keeps its own budget (drafters, MTP, Dory), whose rows are not counted here

  int plan(int avail) {
    ++planned;
    return (lookup ? lookup->pick(avail) : std::max(avail, 0)) + 1;
  }
  void observe(int drafts, int accepted, double ms) {
    if (lookup) lookup->observe(drafts, accepted, ms);
  }
  void cycle(int m, int r = 1) { ++cycles, tokens += m, rows += r; }
  double tokens_per_cycle() const { return cycles > 0 ? double(tokens) / double(cycles) : 0.0; }
};

}  // namespace ni
