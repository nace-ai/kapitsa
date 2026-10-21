#include "model/kflash_budget.h"

#include "hub/hub.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace ni {

namespace {
// What the cache's cost table holds, saved with it: a table measured as anything else (the verify alone, or the
// host's clock, as the server and the Nemotron loop measured before) is measured again; the calibration is kept.
constexpr const char* kCostOf = "cycle GPU ms";
}  // namespace

KfBudget::KfBudget(std::string key) : key_(std::move(key)) {
  for (int d = 0; d < kD; ++d) {
    alpha_[d] = 0.0f;
    beta_[d] = 1.0f;
  }
  load();
}

KfBudget::~KfBudget() { save(); }

void KfBudget::set_rows(uint32_t allowed, int max_rows) {
  max_rows_ = std::clamp(max_rows, 1, kf::kMaxRows);
  allowed_ = allowed & ((max_rows_ == 32 ? 0u : 1u << max_rows_) - 1u);
  if (!allowed_) allowed_ = 1u << (max_rows_ - 1);
}

float KfBudget::logit_q(float lq) {
  const double q = std::min(std::exp(double(lq)), 0.999999);
  return float(double(lq) - std::log1p(-q));
}

// L2-regularized logistic regression of y on (1, x) toward the identity (alpha 0, beta 1): Newton steps from the
// current values. The prior weighs about as much as four records; beta stays positive (the ranking of siblings is
// the path model's).
void KfBudget::fit(const std::vector<float>& x, const std::vector<uint8_t>& y, float& alpha, float& beta, int iters) {
  constexpr double lam = 1.0;
  double a = alpha, b = beta;
  for (int it = 0; it < iters; ++it) {
    double ga = lam * a, gb = lam * (b - 1.0), haa = lam, hab = 0.0, hbb = lam;
    for (size_t i = 0; i < x.size(); ++i) {
      const double xi = x[i], z = a + b * xi, p = 1.0 / (1.0 + std::exp(-z)), w = p * (1.0 - p), r = p - y[i];
      ga += r;
      gb += r * xi;
      haa += w;
      hab += w * xi;
      hbb += w * xi * xi;
    }
    const double det = haa * hbb - hab * hab;
    if (!(det > 1e-12)) break;
    const double da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det;
    a = std::clamp(a - da, -8.0, 8.0);
    b = std::clamp(b - db, 0.1, 4.0);
    if (std::fabs(da) + std::fabs(db) < 1e-4) break;
  }
  alpha = float(a);
  beta = float(b);
}

void KfBudget::observe(const int32_t* t, double ms) {
  std::lock_guard<std::mutex> lk(mu_);
  // the rows the acceptance walked (kObs: kN may already be the next cycle's choice), else kN (a descriptor no
  // acceptance wrote)
  const int n = std::clamp(int(t[kf::kObs] > 0 ? t[kf::kObs] : t[kf::kN]), 1, kf::kMaxRows);
  ++cycles_;
  ++ncycles_[n];
  rows_sum_ += n;
  if (t[kf::kObs] > 0) {  // the tokens the builder expected (E(n)) against the cycle's: its accepted drafts + 1
    float e;
    memcpy(&e, &t[kf::kObs + 1], 4);
    if (e >= 1.0f && e <= float(kf::kMaxRows) + 1.0f) {
      epred_ += e;
      etok_ += double(std::clamp(int(t[kf::kA]), 0, kf::kMaxRows) + 1);
      ++ne_;
    }
  }
  // the costs: this process's first cycle sets the base (with a cached table, from a count it holds: write forces one
  // first; until then the counts measured keep absolute offsets); a count's first cycle sets its offset; later cycles
  // move the base by half of what they cost above the table and their count's offset by a running mean's step (its
  // last ~32 cycles), so a change of every count's cost (the context's length) goes to the base (all but ~6%)
  if (!based_ && (!loaded_ || (ncost_[n] > 0 && !((pre_ >> n) & 1u)))) {
    base_ = ncost_[n] > 0 ? ms - off_[n] : ms;
    if (ncost_[n] == 0) off_[n] = 0.0, ncost_[n] = 1;
    for (int m = 1; m < kD; ++m)
      if ((pre_ >> m) & 1u) off_[m] -= base_;
    pre_ = 0;
    based_ = true;
  } else if (!based_) {  // a count the cache does not hold, before the base: its offset absolute for now
    off_[n] = ncost_[n] == 0 ? ms : off_[n] + (ms - off_[n]) / double(std::min(ncost_[n] + 1, 32));
    ++ncost_[n];
    pre_ |= uint64_t(1) << n;
  } else if (ncost_[n] == 0) {
    off_[n] = ms - base_;
    ncost_[n] = 1;
  } else {
    const double e = ms - (base_ + off_[n]);
    base_ += 0.5 * e;
    off_[n] += e / double(std::min(ncost_[n] + 1, 32));
    ++ncost_[n];
  }
  // the rate: the recent cycles' tokens per ms (sums decaying by 1/64 a cycle)
  rtok_ = rtok_ * (63.0 / 64.0) + double(std::clamp(int(t[kf::kA]), 0, kf::kMaxRows) + 1);
  rms_ = rms_ * (63.0 / 64.0) + ms;
  ++nrate_;
  const int nrec = std::clamp(int(t[kf::kNrec]), 0, kf::kMaxRows);
  for (int i = 0; i < nrec; ++i) {
    const int d = t[kf::kRecDep + i];
    if (d < 1 || d > kf::kMaxRows) continue;
    float lq;
    memcpy(&lq, &t[kf::kRecLq + i], 4);
    const float x = logit_q(lq);
    const uint8_t y = t[kf::kRecHit + i] != 0;
    if (x_[d].size() < size_t(kWin)) {
      x_[d].push_back(x);
      y_[d].push_back(y);
    } else {
      x_[d][head_[d]] = x;
      y_[d][head_[d]] = y;
      head_[d] = (head_[d] + 1) % size_t(kWin);
    }
  }
  if (++since_fit_ >= 16) {
    since_fit_ = 0;
    for (int d = 1; d < kD; ++d)
      if (x_[d].size() >= 32) fit(x_[d], y_[d], alpha_[d], beta_[d], 4);
  }
}

void KfBudget::write(KfParams& p) const {
  std::lock_guard<std::mutex> lk(mu_);
  for (int d = 0; d < kD; ++d) {
    p.alpha[d] = alpha_[d];
    p.beta[d] = beta_[d];
  }
  // NI_KFLASH_FORCE=n (tests): every cycle verifies n rows (-1: the allowed counts in turn, cycle by cycle)
  static const int forced = getenv("NI_KFLASH_FORCE") ? std::clamp(atoi(getenv("NI_KFLASH_FORCE")), -1, max_rows_) : 0;
  int force = std::max(forced, 0);
  if (forced < 0)
    for (int i = 0; i < max_rows_ && !force; ++i) {
      const int n = 1 + (cycles_ + i) % max_rows_;
      if ((allowed_ >> (n - 1)) & 1u) force = n;
    }
  // with a cached table and a choice of counts, this process's first cycles verify a count it holds (they set the
  // base); then every allowed count kExplore times
  if (!force && !based_ && loaded_ && __builtin_popcount(allowed_) > 1)
    for (int n = max_rows_; n >= 1 && !force; --n)
      if (((allowed_ >> (n - 1)) & 1u) && ncost_[n] > 0 && !((pre_ >> n) & 1u)) force = n;
  for (int n = 1; n <= max_rows_ && !force; ++n)
    if (((allowed_ >> (n - 1)) & 1u) && ncost_[n] < kExplore) force = n;
  p.force_n = force;
  p.allowed = allowed_;
  p.max_rows = max_rows_;
  p.rate = nrate_ >= 16 && rms_ > 0 ? float(rtok_ / rms_) : 0.0f;
  p.cost[0] = 1.0f;
  for (int n = 1; n < kD; ++n) p.cost[n] = ncost_[n] > 0 ? float(std::max(base_ + off_[n], 0.01)) : 1.0f;
}

std::string KfBudget::summary() const {
  std::lock_guard<std::mutex> lk(mu_);
  char b[1024];
  int o = snprintf(b, sizeof b, "KFlash: %d cycles, %.2f rows per cycle; cycles by rows", cycles_,
                   cycles_ ? rows_sum_ / cycles_ : 0.0);
  for (int n = 1; n < kD && o < int(sizeof b) - 16; ++n)
    if (ncycles_[n]) o += snprintf(b + o, sizeof b - o, " %d:%d", n, ncycles_[n]);
  if (ne_) o += snprintf(b + o, sizeof b - o, "; tokens per cycle expected %.3f, got %.3f", epred_ / ne_, etok_ / ne_);
  if (rms_ > 0) o += snprintf(b + o, sizeof b - o, "; rate %.4f tok/ms", rtok_ / rms_);
  o += snprintf(b + o, sizeof b - o, "; cost ms by rows");
  for (int n = 1; n <= max_rows_ && o < int(sizeof b) - 16; ++n)
    if ((allowed_ >> (n - 1)) & 1u)
      o += snprintf(b + o, sizeof b - o, " %d:%.2f", n, ncost_[n] ? base_ + off_[n] : 0.0);
  o += snprintf(b + o, sizeof b - o, "; calibration (alpha, beta) by depth");
  for (int d = 1; d <= std::min(max_rows_, 8) && o < int(sizeof b) - 24; ++d)
    o += snprintf(b + o, sizeof b - o, " (%.2f, %.2f)", alpha_[d], beta_[d]);
  return b;
}

void KfBudget::load() {
  if (key_.empty()) return;
  std::ifstream f(hub::home_dir() + "/kflash/" + key_ + ".json");
  if (!f) return;
  try {
    const nlohmann::json j = nlohmann::json::parse(f);
    const auto& c = j.at("cost");
    const auto& a = j.at("alpha");
    const auto& bb = j.at("beta");
    if (j.value("cost_of", "") == kCostOf)  // (offsets above base 0: the costs at the save)
      for (int n = 1; n < kD && n < int(c.size()); ++n)
        if (c[size_t(n)].get<double>() > 0) off_[n] = c[size_t(n)].get<double>(), ncost_[n] = kExplore, loaded_ = true;
    for (int d = 1; d < kD && d < int(a.size()) && d < int(bb.size()); ++d) {
      alpha_[d] = a[size_t(d)].get<float>();
      beta_[d] = bb[size_t(d)].get<float>();
    }
  } catch (...) {
    logi("KFlash: ignoring an unreadable cache %s", key_.c_str());
  }
}

void KfBudget::save(int fresh) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key_.empty() || cycles_ < 32 || cycles_ - saved_ < std::max(fresh, 1)) return;
  const std::string dir = hub::home_dir() + "/kflash";
  mkdir(dir.c_str(), 0755);
  nlohmann::json j;
  j["cost_of"] = kCostOf;
  std::vector<double> cost(kD, 0.0);  // the costs now
  for (int n = 1; n < kD; ++n)
    if (ncost_[n] && !((pre_ >> n) & 1u)) cost[size_t(n)] = std::max(base_ + off_[n], 0.01);
  j["cost"] = cost;
  j["alpha"] = std::vector<float>(alpha_, alpha_ + kD);
  j["beta"] = std::vector<float>(beta_, beta_ + kD);
  // written whole beside it, then renamed over it: a process ended mid-write (kill -9 after a stop signal), or another
  // tool saving the same key at once, leaves one whole file
  const std::string path = dir + "/" + key_ + ".json", tmp = path + "." + std::to_string(getpid()) + ".tmp";
  {
    std::ofstream f(tmp);
    f << j.dump() << "\n";
    f.close();
    if (!f) {
      unlink(tmp.c_str());
      logi("KFlash: could not write the budget's cache %s", tmp.c_str());
      return;
    }
  }
  if (rename(tmp.c_str(), path.c_str()) != 0) {
    unlink(tmp.c_str());
    logi("KFlash: could not write the budget's cache %s", path.c_str());
    return;
  }
  saved_ = cycles_;
  logi("KFlash: budget saved (%d cycles) to %s", cycles_, path.c_str());
}

}  // namespace ni
