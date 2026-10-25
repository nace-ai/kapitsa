// What a MoE target's MTP decode cycles choose their verify rows by, learned online on the engine thread:
// an acceptance model of the draft's confidences and a fit of what a cycle costs by its rows. The host
// selects between the cycle's draft commit and its verify encode (Kapitsa's MTP cycle is synchronous, so
// no GPU-side selection kernel is needed; Splash picks in its draft_verify_rows kernel).
//
// The acceptance model: once the proposals before it are accepted, a proposal is accepted with probability
// p, where logit(p) = slope * logit(confidence) + intercepts[position], the confidence being the largest
// candidate probability at its draft position. Each completed cycle labels the proposals it verified, up to
// the first the target rejected, but for those whose confidence is not a probability; the model is refit
// every kRefitCycles cycles over the last kLabels labels by damped Newton steps (IRLS) on their
// log-likelihood with a Gaussian prior around kPrior, which also stands before any label.
//
// The costs: a cycle that verifies R rows takes about a + b * R milliseconds, fit to the wall times of the
// latest cycles (which with streamed experts include the store's reads of its misses) by least squares that
// weigh each cycle kDecay times the next. The latest cycles stand for what the machine does now: what a row
// costs varies with the load and the context, so old cycles must not price it. The slope shrinks toward the
// seed's, the shape the measured MoE targets share, at the cycles' level, as much as kPriorCycles cycles
// over a row spread would; before any cycle, the seed itself is the cost. A cycle counts at most twice and
// at least half the time the fit gives it, so that a stall does not price the rows of the next thirty
// cycles. Every kExploreCycles-th cycle verifies every row, which keeps the rows of the fit spread.
//
// Ported from Splash's runtime/model/VerifyBudget.{hpp,cpp} (Apache-2.0, (c) incoai; NOTICE). Changes: one
// lane (Kapitsa's MTP cycles decode a single stream, so the lane and constrained dimensions and the
// cross-lane ranking are gone), rows() is the host-side selection replacing Splash's draft_verify_rows
// kernel (its single-lane case: the survival probabilities are nonincreasing, so the ranking is the
// identity), and the Lane struct folds into observe()'s arguments (a Kapitsa cycle always labels the
// proposals it verified).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ni {

class VerifyBudget final {
 public:
  static constexpr int kPositions = 7;  // proposals a cycle drafts (Qwen35::kMaxMtpRows - 1)
  static constexpr int kBlockRows = 8;  // the pending row plus the proposals (Qwen35::kMaxMtpRows)
  static constexpr int kRefitCycles = 64;
  static constexpr int kLabels = 4096;
  static constexpr int kExploreCycles = 32;
  static constexpr double kDecay = 0.97;
  static constexpr double kPriorCycles = 5.0;
  // The prior's precision on the slope and on each intercept: about twenty labels' worth each (a label
  // tells the slope about its logit squared, some 2.5, times what it tells its intercept).
  static constexpr double kSlopePrecision = 50.0;
  static constexpr double kInterceptPrecision = 4.0;
  [[nodiscard]] static constexpr double priorPrecision(int parameter) noexcept {
    return parameter ? kInterceptPrecision : kSlopePrecision;
  }

  struct Acceptance final {
    float slope = 0.0F;
    std::array<float, kPositions> intercepts{};
    bool operator==(const Acceptance&) const = default;
  };
  // Splash's fit of the measured targets' logged cycles (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B: slopes
  // 0.65-0.71), taken as the prior; not re-measured under Kapitsa.
  static constexpr Acceptance kPrior{0.7F, {0.0F, -0.35F, -0.8F, -0.75F, -0.75F, -0.85F, -0.45F}};

  // A label: a proposal at `position`, whose confidence had logit `feature`, that the target accepted or
  // rejected.
  struct Label final {
    int position = 0;
    float feature = 0.0F;
    bool accepted = false;
  };

  [[nodiscard]] const Acceptance& acceptance() const noexcept { return acceptance_; }
  // The verify rows of the next cycle, 1 (the pending row alone) to `maxRows`, maximizing the expected
  // tokens a millisecond buys; `maxRows` on every kExploreCycles-th cycle. `conf` holds kPositions draft
  // confidences; one that is not a probability prices its proposal at the intercept alone.
  [[nodiscard]] int rows(const float* conf, int maxRows) const;
  // Whether the next cycle verifies every row.
  [[nodiscard]] bool explores() const noexcept { return costCycles_ % kExploreCycles == kExploreCycles - 1; }
  // costs()[R] for R rows, R from 1 to kBlockRows, nondecreasing in R; costs()[0] is zero.
  [[nodiscard]] std::array<float, kBlockRows + 1> costs() const;
  // A completed cycle and its commands' wall time: it verified `verifyRows` rows and the target accepted
  // `accepted` proposals before the first rejection.
  void observe(const float* conf, int verifyRows, int accepted, double milliseconds);

  // The labels kept, oldest first.
  [[nodiscard]] std::vector<Label> labels() const;
  // The maximum a posteriori model of `labels` under the prior, by damped Newton steps from `start`.
  [[nodiscard]] static Acceptance fit(const std::vector<Label>& labels, const Acceptance& start);
  // The shape of a cycle's cost by its rows (a cycle of every row costs 1).
  [[nodiscard]] static double seed(int rows);

 private:
  // The weighted sums of the latest cycles' rows R and times t.
  struct Cost final {
    double weight = 0.0;
    double rows = 0.0;
    double rowsSquared = 0.0;
    double time = 0.0;
    double rowsTime = 0.0;
  };

  // The fit's intercept and slope; none before any cycle.
  [[nodiscard]] std::optional<std::pair<double, double>> line() const;

  Acceptance acceptance_ = kPrior;
  std::vector<Label> labels_;
  size_t next_ = 0;
  uint64_t cycles_ = 0;
  Cost cost_{};
  uint64_t costCycles_ = 0;
};

}  // namespace ni
