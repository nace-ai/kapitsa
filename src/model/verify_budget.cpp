// The verify budget's models (src/model/verify_budget.h), ported from Splash's
// runtime/model/VerifyBudget.cpp (Apache-2.0, (c) incoai; NOTICE). The changes the header lists: the lane
// and constrained dimensions are gone, rows() replaces Splash's draft_verify_rows selection kernel with its
// single-lane case, and observe() takes one cycle's facts instead of a span of lanes.
#include "model/verify_budget.h"

#include <algorithm>
#include <cmath>

namespace ni {

namespace {

constexpr int kParameters = 1 + VerifyBudget::kPositions;
constexpr int kIterations = 16;

// A cycle's cost at 1, 2, 4 and 8 rows, one lane (every row costs 1): the mean shape of
// Qwen3.8-Flash-Next's cycles (resident and with 12 GB of experts) and Qwen3.6-35B-A3B's (resident and
// with 4 GB).
constexpr std::array<double, 4> kSeedShape{0.60, 0.69, 0.85, 1.0};

using Parameters = std::array<double, kParameters>;

double logit(double p) {
  p = std::clamp(p, 1.0e-6, 1.0 - 1.0e-6);
  return std::log(p / (1.0 - p));
}

Parameters parametersOf(const VerifyBudget::Acceptance& acceptance) {
  Parameters parameters{acceptance.slope};
  for (int position = 0; position < VerifyBudget::kPositions; ++position)
    parameters[1 + position] = acceptance.intercepts[position];
  return parameters;
}

// The negative log posterior, up to a constant.
double objective(const std::vector<VerifyBudget::Label>& labels, const Parameters& theta, const Parameters& prior) {
  double value = 0.0;
  for (const VerifyBudget::Label& label : labels) {
    const double z = theta[0] * label.feature + theta[1 + label.position];
    // log(1 + e^-margin), the label's negative log-likelihood, stably.
    const double margin = label.accepted ? z : -z;
    value += margin > 0.0 ? std::log1p(std::exp(-margin)) : -margin + std::log1p(std::exp(margin));
  }
  for (int i = 0; i < kParameters; ++i)
    value += 0.5 * VerifyBudget::priorPrecision(i) * (theta[i] - prior[i]) * (theta[i] - prior[i]);
  return value;
}

// x with h x = g, h positive definite: Gaussian elimination.
Parameters solve(std::array<Parameters, kParameters> h, Parameters g) {
  for (int column = 0; column < kParameters; ++column) {
    int pivot = column;
    for (int row = column + 1; row < kParameters; ++row)
      if (std::fabs(h[row][column]) > std::fabs(h[pivot][column])) pivot = row;
    std::swap(h[column], h[pivot]);
    std::swap(g[column], g[pivot]);
    for (int row = column + 1; row < kParameters; ++row) {
      const double factor = h[row][column] / h[column][column];
      for (int k = column; k < kParameters; ++k) h[row][k] -= factor * h[column][k];
      g[row] -= factor * g[column];
    }
  }
  Parameters x{};
  for (int row = kParameters; row-- > 0;) {
    double sum = g[row];
    for (int k = row + 1; k < kParameters; ++k) sum -= h[row][k] * x[k];
    x[row] = sum / h[row][row];
  }
  return x;
}

}  // namespace

double VerifyBudget::seed(int rows) {
  const std::array<double, 4> knots{1.0, 2.0, 4.0, 8.0};
  const double r = double(std::clamp(rows, 1, kBlockRows));
  int upper = 1;
  while (upper < 3 && r > knots[upper]) ++upper;
  const double t = (r - knots[upper - 1]) / (knots[upper] - knots[upper - 1]);
  return kSeedShape[upper - 1] + t * (kSeedShape[upper] - kSeedShape[upper - 1]);
}

std::optional<std::pair<double, double>> VerifyBudget::line() const {
  if (cost_.weight <= 0.0) return std::nullopt;
  const double meanRows = cost_.rows / cost_.weight, meanTime = cost_.time / cost_.weight;
  const double spread = std::max(cost_.rowsSquared / cost_.weight - meanRows * meanRows, 0.0);
  const double covariance = cost_.rowsTime / cost_.weight - meanRows * meanTime;
  // The seed's slope at the cycles' level, worth kPriorCycles cycles whose rows spread by a row.
  const double seedSlope = (seed(kBlockRows) - seed(1)) / double(kBlockRows - 1);
  const double priorSlope = seedSlope * meanTime / seed(int(std::lround(meanRows)));
  const double priorSpread = 1.0;
  const double slope = std::max((cost_.weight * covariance + kPriorCycles * priorSpread * priorSlope) /
                                    (cost_.weight * spread + kPriorCycles * priorSpread),
                                0.0);
  return std::pair{meanTime - slope * meanRows, slope};
}

std::array<float, VerifyBudget::kBlockRows + 1> VerifyBudget::costs() const {
  std::array<float, kBlockRows + 1> result{};
  const auto fit = line();
  if (!fit) {
    for (int rows = 1; rows <= kBlockRows; ++rows) result[rows] = float(seed(rows));
    return result;
  }
  // Nondecreasing and positive: a negative intercept leaves the fewest rows a twentieth of the mean.
  for (int rows = 1; rows <= kBlockRows; ++rows)
    result[rows] = float(std::max(fit->first + fit->second * rows, 0.05 * cost_.time / cost_.weight));
  return result;
}

int VerifyBudget::rows(const float* conf, int maxRows) const {
  const int most = std::clamp(maxRows, 1, kBlockRows);
  if (most <= 1 || explores()) return most;
  // Splash's draft_verify_rows kernel for one lane: the proposals in order (their survival probabilities
  // are nonincreasing, so its ranking is the identity), the prefix that maximizes the expected tokens the
  // rows' cost buys. Row 0 is the pending row and proposal j is checked against row j + 1; survival_j
  // enters the sum only once the cycle verifies j + 2 rows, because the walk's cap (a < R - 1) pays its
  // match only at that row depth.
  const std::array<float, kBlockRows + 1> cost = costs();
  const Acceptance& a = acceptance_;
  float expected = 1.0F, survival = 1.0F;  // the pending row is a token already
  float best = expected / cost[1];
  int taken = 0;
  for (int j = 0; j + 2 <= most; ++j) {
    const double c = conf[j];
    const bool probability = std::isfinite(c) && c >= 0.0F && c <= 1.0F;
    const double z = a.slope * (probability ? logit(c) : 0.0) + a.intercepts[j];
    survival *= float(1.0 / (1.0 + std::exp(-z)));
    expected += survival;
    const float value = expected / cost[j + 2];
    if (value > best) {
      best = value;
      taken = j + 1;
    }
  }
  return 1 + taken;
}

void VerifyBudget::observe(const float* conf, int verifyRows, int accepted, double milliseconds) {
  if (verifyRows < 1 || verifyRows > kBlockRows || accepted < 0 || accepted > kPositions) return;
  ++costCycles_;
  if (std::isfinite(milliseconds) && milliseconds > 0.0) {
    if (const auto fit = line()) {
      const double expected = std::max(fit->first + fit->second * verifyRows, 1.0e-3);
      milliseconds = std::clamp(milliseconds, 0.5 * expected, 2.0 * expected);
    }
    cost_.weight = kDecay * cost_.weight + 1.0;
    cost_.rows = kDecay * cost_.rows + verifyRows;
    cost_.rowsSquared = kDecay * cost_.rowsSquared + double(verifyRows) * verifyRows;
    cost_.time = kDecay * cost_.time + milliseconds;
    cost_.rowsTime = kDecay * cost_.rowsTime + verifyRows * milliseconds;
  }
  // Proposal j is checked against the target's row j + 1, and the engine's walk stops at a < r - 1: the
  // last verified row's sample is committed as the cycle's final token without a comparison, so a cycle
  // that verified r rows walked proposals 0..r-2 and labels them to the first rejection: the loop is
  // inclusive and the bound holds it at that rejection (the negative label). A cycle whose whole walk
  // accepted labels no further one — its next proposal was never checked, so there is nothing to label.
  // (Splash's walk is uncapped and compares proposal r - 1 against fresh row r - 1, so its verifyRows - 1
  // bound labels only checked proposals; the port censors its unchecked last proposal instead.)
  const int last = std::min({accepted, verifyRows - 2, kPositions - 1});
  for (int position = 0; position <= last; ++position) {
    const float confidence = conf[position];
    if (!(confidence >= 0.0F && confidence <= 1.0F)) continue;
    const Label label{position, float(logit(confidence)), position < accepted};
    if (labels_.size() < size_t(kLabels)) {
      labels_.push_back(label);
    } else {
      labels_[next_] = label;
      next_ = (next_ + 1) % size_t(kLabels);
    }
  }
  if (++cycles_ % kRefitCycles == 0 && !labels_.empty()) acceptance_ = fit(labels(), acceptance_);
}

std::vector<VerifyBudget::Label> VerifyBudget::labels() const {
  std::vector<Label> ordered(labels_.begin() + ptrdiff_t(next_), labels_.end());
  ordered.insert(ordered.end(), labels_.begin(), labels_.begin() + ptrdiff_t(next_));
  return ordered;
}

VerifyBudget::Acceptance VerifyBudget::fit(const std::vector<Label>& labels, const Acceptance& start) {
  const Parameters prior = parametersOf(kPrior);
  Parameters theta = parametersOf(start);
  double current = objective(labels, theta, prior);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    Parameters gradient{};
    std::array<Parameters, kParameters> hessian{};
    for (const Label& label : labels) {
      if (label.position >= kPositions || !std::isfinite(label.feature)) continue;
      const int b = 1 + label.position;
      const double p = 1.0 / (1.0 + std::exp(-(theta[0] * label.feature + theta[b])));
      const double residual = p - (label.accepted ? 1.0 : 0.0);
      const double weight = p * (1.0 - p);
      gradient[0] += residual * label.feature;
      gradient[b] += residual;
      hessian[0][0] += weight * label.feature * label.feature;
      hessian[0][b] += weight * label.feature;
      hessian[b][0] += weight * label.feature;
      hessian[b][b] += weight;
    }
    for (int i = 0; i < kParameters; ++i) {
      gradient[i] += priorPrecision(i) * (theta[i] - prior[i]);
      hessian[i][i] += priorPrecision(i);
    }
    const Parameters step = solve(hessian, gradient);
    // Damped: the Newton step, halved until the objective does not grow.
    Parameters next{};
    double value = current, scale = 1.0;
    bool moved = false;
    for (int halving = 0; halving < 30 && !moved; ++halving, scale *= 0.5) {
      for (int i = 0; i < kParameters; ++i) next[i] = theta[i] - scale * step[i];
      value = objective(labels, next, prior);
      moved = value <= current;
    }
    if (!moved) break;
    double largest = 0.0;
    for (int i = 0; i < kParameters; ++i) largest = std::max(largest, std::fabs(next[i] - theta[i]));
    theta = next;
    current = value;
    if (largest < 1.0e-9) break;
  }
  Acceptance result{float(theta[0]), {}};
  for (int position = 0; position < kPositions; ++position) result.intercepts[position] = float(theta[1 + position]);
  return result;
}

}  // namespace ni
