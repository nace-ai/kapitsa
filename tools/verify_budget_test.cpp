// The learned MTP verify budget's models (src/model/verify_budget.h) on synthetic cycles, no model, no GPU.
//   verify_budget_test
// 1. The cost seed's shape: its knots, interpolation between them, clamping outside, and costs() before any
//    cycle (the seed) and after cycles (the exponentially weighted line fit, nondecreasing, and a stall
//    clamped to twice the fit's expectation so it does not price the next cycles' rows).
// 2. rows(): the explore cycles verify every row; otherwise the choice is in 1..maxRows, nondecreasing in
//    the drafts' confidences, and a confidence that is not a probability prices its proposal at the
//    intercept alone (the same rows as confidence 1/2, whose logit is zero).
// 3. observe()'s labels: a cycle labels the proposals it checked, to the first rejection (that one
//    negative); a cycle that accepted every proposal it checked labels no further one (its next proposal
//    was never checked); a confidence that is not a probability is skipped; the ring keeps the last
//    kLabels, oldest first.
// 4. fit(): the prior alone reproduces it (its MAP with no labels), and labels drawn from a known model
//    recover it (damped Newton over the log-likelihood); refit happens on the kRefitCycles-th cycle and
//    moves the model against labels the prior contradicts.
#include "model/verify_budget.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// A cycle's facts for observe(): rows verified, proposals accepted, and the drafts' confidences.
void cycle(VerifyBudget& b, int rows, int accepted, double ms, const float* conf) { b.observe(conf, rows, accepted, ms); }

const float kHalf[VerifyBudget::kPositions] = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};

}  // namespace

int main() {
  // 1. The cost seed and costs().
  check(std::fabs(VerifyBudget::seed(1) - 0.60) < 1e-9 && std::fabs(VerifyBudget::seed(2) - 0.69) < 1e-9 &&
            std::fabs(VerifyBudget::seed(4) - 0.85) < 1e-9 && std::fabs(VerifyBudget::seed(8) - 1.0) < 1e-9,
        "seed: the knots (1, 2, 4, 8 rows)");
  check(std::fabs(VerifyBudget::seed(3) - 0.5 * (0.69 + 0.85)) < 1e-9 && VerifyBudget::seed(0) == VerifyBudget::seed(1) &&
            VerifyBudget::seed(99) == VerifyBudget::seed(8),
        "seed: interpolation between the knots, clamping outside");
  {
    VerifyBudget b;
    const auto cost = b.costs();
    bool ok = cost[0] == 0.0F;
    for (int r = 1; r <= VerifyBudget::kBlockRows; ++r) ok = ok && std::fabs(cost[r] - float(VerifyBudget::seed(r))) < 1e-6F;
    check(ok, "costs: before any cycle, the seed's shape");
  }
  {
    VerifyBudget b;
    // A line through (2 rows, 1 ms) and (6 rows, 3 ms): the slope about 0.5 ms a row, the prior's shape
    // (about 0.13) worth five cycles against the fit's own spread.
    for (int i = 0; i < 200; ++i) {
      cycle(b, 2, 1, 1.0, kHalf);
      cycle(b, 6, 3, 3.0, kHalf);
    }
    const auto cost = b.costs();
    bool ok = cost[0] == 0.0F;
    for (int r = 2; r <= VerifyBudget::kBlockRows; ++r) ok = ok && cost[r] >= cost[r - 1];
    const double slope = (double(cost[6]) - double(cost[2])) / 4.0;
    check(ok && slope > 0.42 && slope < 0.56, "costs: the weighted line fit of the cycles' rows and times");
  }
  {
    VerifyBudget b;
    for (int i = 0; i < 4; ++i) cycle(b, 4, 2, 2.0, kHalf);
    const float before = b.costs()[4];
    cycle(b, 4, 2, 10000.0, kHalf);  // a stall: clamped to at most twice the fit's expectation
    const float after = b.costs()[4];
    check(after < 10.0F && after >= before, "costs: a stalled cycle is clamped, not priced");
  }

  // 2. rows().
  {
    VerifyBudget b;
    check(b.rows(kHalf, 1) == 1, "rows: maxRows 1 verifies the pending row alone");
    int r = 0;
    bool bounded = true, monotone = true;
    for (int i = 1; i <= 99; ++i) {
      const float c = float(i) / 100.0F;
      const float conf[VerifyBudget::kPositions] = {c, c, c, c, c, c, c};
      const int now = b.rows(conf, VerifyBudget::kBlockRows);
      bounded = bounded && now >= 1 && now <= VerifyBudget::kBlockRows;
      monotone = monotone && now >= r;
      r = now;
    }
    check(bounded, "rows: the choice stays in 1..maxRows");
    check(monotone, "rows: nondecreasing in the drafts' confidences");
    const float nan = std::nanf(""), over[VerifyBudget::kPositions] = {2.0F, -1.0F, nan, 0.5F, nan, 3.0F, 0.5F};
    check(b.rows(over, VerifyBudget::kBlockRows) >= 1 && b.rows(over, VerifyBudget::kBlockRows) <= VerifyBudget::kBlockRows &&
              b.rows(over, 3) == b.rows(kHalf, 3),
          "rows: a confidence that is not a probability prices its proposal at the intercept alone");
    for (int i = 0; i < VerifyBudget::kExploreCycles - 1; ++i) cycle(b, 1, 0, 1.0, kHalf);
    check(b.explores() && b.rows(kHalf, VerifyBudget::kBlockRows) == VerifyBudget::kBlockRows,
          "rows: every kExploreCycles-th cycle verifies every row");
    cycle(b, 1, 0, 1.0, kHalf);
    check(!b.explores(), "rows: the explore cycle passes");
  }

  // 3. observe()'s labels.
  {
    VerifyBudget b;
    cycle(b, 4, 2, 1.0, kHalf);  // proposals 0..2 checked, the third rejected
    auto labels = b.labels();
    check(labels.size() == 3 && labels[0].position == 0 && labels[1].position == 1 && labels[2].position == 2 &&
              labels[0].accepted && labels[1].accepted && !labels[2].accepted,
          "labels: a rejection labels the checked proposals, itself negative");
    VerifyBudget c;
    cycle(c, 4, 3, 1.0, kHalf);  // every checked proposal accepted; proposal 3 was never checked
    labels = c.labels();
    check(labels.size() == 3 && labels[0].accepted && labels[1].accepted && labels[2].accepted,
          "labels: an all-accepted cycle labels no unchecked proposal");
    VerifyBudget d;
    cycle(d, 1, 0, 1.0, kHalf);
    check(d.labels().empty(), "labels: a cycle of the pending row alone labels nothing");
    VerifyBudget e;
    cycle(e, 2, 0, 1.0, kHalf);
    check(e.labels().size() == 1 && !e.labels()[0].accepted, "labels: the only checked proposal, rejected");
    VerifyBudget f;
    const float conf[VerifyBudget::kPositions] = {0.5F, std::nanf(""), 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
    cycle(f, 3, 2, 1.0, conf);  // proposals 0..1 checked and accepted; the second's confidence is not a probability
    check(f.labels().size() == 1 && f.labels()[0].position == 0 && f.labels()[0].accepted,
          "labels: a confidence that is not a probability is skipped");
    VerifyBudget g;
    for (int i = 0; i < VerifyBudget::kLabels + 10; ++i) {
      const float one[VerifyBudget::kPositions] = {float(sigmoid(0.0005 * i)), 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
      cycle(g, 2, 0, 1.0, one);
    }
    labels = g.labels();
    bool ordered = labels.size() == size_t(VerifyBudget::kLabels);
    for (size_t i = 1; ordered && i < labels.size(); ++i) ordered = labels[i].feature > labels[i - 1].feature;
    check(ordered && std::fabs(double(labels[0].feature) - 0.005) < 1e-5,
          "labels: the ring keeps the last kLabels, oldest first");
  }

  // 4. fit().
  {
    check(VerifyBudget::fit({}, VerifyBudget::kPrior) == VerifyBudget::kPrior, "fit: the prior alone reproduces it");
    VerifyBudget::Acceptance away{2.0F, {0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F}};
    const auto pulled = VerifyBudget::fit({}, away);
    check(std::fabs(double(pulled.slope) - 0.7) < 1e-6 && std::fabs(double(pulled.intercepts[0]) - 0.0) < 1e-6,
          "fit: with no labels the MAP is the prior");
    // Labels from a known model: the recovery.
    const double trueSlope = 1.3;
    const std::array<double, VerifyBudget::kPositions> trueIntercepts{0.5, 0.1, -0.3, -0.6, -0.9, -1.2, -1.5};
    std::vector<VerifyBudget::Label> labels;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> features(1.0, 1.5);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int i = 0; i < 20000; ++i) {
      const int position = i % VerifyBudget::kPositions;
      const double x = features(rng);
      labels.push_back({position, float(x), uniform(rng) < sigmoid(trueSlope * x + trueIntercepts[position])});
    }
    const auto fitted = VerifyBudget::fit(labels, VerifyBudget::kPrior);
    check(fitted == VerifyBudget::fit(labels, VerifyBudget::kPrior), "fit: deterministic");
    bool close = std::fabs(double(fitted.slope) - trueSlope) < 0.08;
    for (int p = 0; p < VerifyBudget::kPositions; ++p)
      close = close && std::fabs(double(fitted.intercepts[p]) - trueIntercepts[p]) < 0.2;
    check(close, "fit: labels from a known model recover it");
    VerifyBudget b;
    const float sure[VerifyBudget::kPositions] = {0.99F, 0.99F, 0.99F, 0.99F, 0.99F, 0.99F, 0.99F};
    for (int i = 0; i < VerifyBudget::kRefitCycles - 1; ++i) cycle(b, 2, 0, 1.0, sure);
    check(b.acceptance() == VerifyBudget::kPrior, "fit: before the kRefitCycles-th cycle the prior stands");
    cycle(b, 2, 0, 1.0, sure);  // every sure proposal rejected: the model must move against the prior
    check(b.acceptance() != VerifyBudget::kPrior && b.acceptance().intercepts[0] < -1.0F &&
              b.acceptance().slope <= VerifyBudget::kPrior.slope,
          "fit: the kRefitCycles-th cycle refits against labels the prior contradicts");
  }

  std::printf("%s (%d checks failed)\n", g_fail ? "FAILED" : "all passed", g_fail);
  return g_fail ? 1 : 0;
}
