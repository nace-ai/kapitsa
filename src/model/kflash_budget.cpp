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
constexpr const char* kBatchCostOf = "batched cycle GPU ms";  // KfBatch's: the members' drafting passes and the verify

// A cache file written whole beside its place, then renamed over it: a process ended mid-write (kill -9 after a stop
// signal), or another tool saving the same key at once, leaves one whole file
bool write_whole(const std::string& path, const nlohmann::json& j) {
  const std::string tmp = path + "." + std::to_string(getpid()) + ".tmp";
  {
    std::ofstream f(tmp);
    f << j.dump() << "\n";
    f.close();
    if (!f) {
      unlink(tmp.c_str());
      logi("KFlash: could not write the budget's cache %s", tmp.c_str());
      return false;
    }
  }
  if (rename(tmp.c_str(), path.c_str()) != 0) {
    unlink(tmp.c_str());
    logi("KFlash: could not write the budget's cache %s", path.c_str());
    return false;
  }
  return true;
}
}  // namespace

// The costs: this process's first cycle sets the base (with a cached table, from a count it holds: force() asks for one
// first; until then the counts measured keep absolute offsets); a count's first cycle sets its offset; later cycles move
// the base by half of what they cost above the table and their count's offset by a running mean's step (its last ~32
// cycles), so a change of every count's cost (the context's length) goes to the base (all but ~6%)
void KfCosts::observe(int n, double ms) {
  if (!based && (!loaded || (ncost[n] > 0 && !((pre >> n) & 1u)))) {
    base = ncost[n] > 0 ? ms - off[n] : ms;
    if (ncost[n] == 0) off[n] = 0.0, ncost[n] = 1;
    for (int m = 1; m < kD; ++m)
      if ((pre >> m) & 1u) off[m] -= base;
    pre = 0;
    based = true;
  } else if (!based) {  // a count the cache does not hold, before the base: its offset absolute for now
    off[n] = ncost[n] == 0 ? ms : off[n] + (ms - off[n]) / double(std::min(ncost[n] + 1, 32));
    ++ncost[n];
    pre |= uint64_t(1) << n;
  } else if (ncost[n] == 0) {
    off[n] = ms - base;
    ncost[n] = 1;
  } else {
    const double e = ms - (base + off[n]);
    base += 0.5 * e;
    off[n] += e / double(std::min(ncost[n] + 1, 32));
    ++ncost[n];
  }
}

int KfCosts::force(uint32_t allowed, int max_rows, int explore) const {
  int f = 0;
  // with a cached table and a choice of counts, this process's first cycles verify a count it holds (they set the
  // base); then every allowed count `explore` times
  if (!based && loaded && __builtin_popcount(allowed) > 1)
    for (int n = max_rows; n >= 1 && !f; --n)
      if (((allowed >> (n - 1)) & 1u) && ncost[n] > 0 && !((pre >> n) & 1u)) f = n;
  for (int n = 1; n <= max_rows && !f; ++n)
    if (((allowed >> (n - 1)) & 1u) && ncost[n] < explore) f = n;
  return f;
}

std::vector<double> KfCosts::table() const {
  std::vector<double> cost(kD, 0.0);  // the costs now
  for (int n = 1; n < kD; ++n)
    if (ncost[n] && !((pre >> n) & 1u)) cost[size_t(n)] = std::max(base + off[n], 0.01);
  return cost;
}

void KfCosts::load(const std::vector<double>& c, int explore) {  // (offsets above base 0: the costs at the save)
  for (int n = 1; n < kD && n < int(c.size()); ++n)
    if (c[size_t(n)] > 0) off[n] = c[size_t(n)], ncost[n] = explore, loaded = true;
}

KfBudget::KfBudget(std::string key) : key_(std::move(key)) {
  for (int d = 0; d < kD; ++d) {
    alpha_[d] = 0.0f;
    beta_[d] = 1.0f;
  }
  load();
}

KfBudget::~KfBudget() { save(); }

void KfBudget::set_wide(bool on) {
  std::lock_guard<std::mutex> lk(mu_);
  wide_ = on;
  if (!on) return;
  for (int n = 1; n < kD; ++n)  // the cache's costs (loaded as offsets above base 0), or the cost table of an earlier call
    if (!c_.based && c_.ncost[n] > 0 && c_.off[n] > 0 && cost_[n] <= 0) cost_[n] = c_.off[n];
  if (!wide_saved_ && cost_[8] > 0 && ncost8n_ == 0) {  // a cache of 8-row verifies only: that is the base's cost
    cost8n_ = cost_[8];
    ncost8n_ = 2;
    cost_[8] = 0;
    c_.ncost[8] = 0;
  }
  // the log model from the cached costs (ratios to the 8-row verify's); this process's narrow cycles set the base
  if (cost8n_ > 0) {
    lb_ = std::log(cost8n_);
    for (int n = 1; n < kD; ++n) lq_[n] = cost_[n] > 0 ? std::log(cost_[n]) - lb_ : 0.0;
  }
}

void KfBudget::pin_wide() {
  std::lock_guard<std::mutex> lk(mu_);
  pinned_ = true;
  wide_pays_ = true;
}

void KfBudget::wide_costs() {
  cost8n_ = std::exp(lb_);
  for (int n = 1; n < kD; ++n) cost_[n] = c_.ncost[n] > 0 ? std::exp(lb_ + lq_[n]) : 0.0;
}

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

// Wide verifies: the expected tokens per ms of the per-cycle choice among the allowed counts at their (wide) costs
// against 8 rows at the narrow verify's cost, over the recent cycles' trees.
void KfBudget::eval_wide() {
  if (pinned_) {  // the capacity is not a decision (pin_wide)
    wide_pays_ = true;
    return;
  }
  const size_t nc = ehist_.size() / 3;
  if (!nc || cost8n_ <= 0) {
    wide_pays_ = true;
    return;
  }
  double e8 = 0, ew = 0, tw = 0;
  for (size_t c = 0; c < nc; ++c) {
    const float* e = &ehist_[3 * c];
    e8 += e[0];
    double best = -1, be = 0, bt = 1;
    const int counts[3] = {8, 16, 32};
    for (int i = 0; i < 3; ++i) {
      const int n = counts[i];
      if (n > max_rows_ || !((allowed_ >> (n - 1)) & 1u) || cost_[n] <= 0 || e[i] <= 0) continue;
      if (e[i] / cost_[n] > best) best = e[i] / cost_[n], be = e[i], bt = cost_[n];
    }
    if (best < 0) be = e[0], bt = cost8n_;
    ew += be;
    tw += bt;
  }
  wide_pays_ = ew / tw > e8 / (double(nc) * cost8n_);
}

void KfBudget::observe(const int32_t* t, double ms) {
  std::lock_guard<std::mutex> lk(mu_);
  // kf_accept's copies (kObs: the next cycle's builder may already have rewritten the descriptor's own words); a
  // descriptor no kf_accept wrote (tests): its own words
  const bool obs = t[kf::kObs] > 0;
  const int n = std::clamp(int(obs ? t[kf::kObs] : t[kf::kN]), 1, kf::kMaxRows);
  const int cap0 = obs ? t[kf::kObs + 2] : t[kf::kCap], cap = cap0 > 0 ? cap0 : max_rows_;
  const int nodes = obs ? t[kf::kObs + 3] : t[kf::kNodes];
  const bool wide = wide_ && max_rows_ > 8;
  const bool narrow = wide && cap <= 8;  // a wide build verified by the 8-row verify
  ++cycles_;
  ++ncycles_[n];
  rows_sum_ += n;
  if (obs) {  // the tokens the builder expected (E(n)) against the cycle's: its accepted drafts + 1
    float e;
    memcpy(&e, &t[kf::kObs + 1], 4);
    if (e >= 1.0f && e <= float(kf::kMaxRows) + 1.0f) {
      epred_ += e;
      etok_ += double(std::clamp(int(t[kf::kA]), 0, kf::kMaxRows) + 1);
      ++ne_;
    }
  }
  if (wide) {
    // wide verifies, log cost = lb_ + lq_[n]: a narrow cycle moves the base (this process's first one sets it); a
    // wide one fits its count's ratio (its first measurement exactly) and nudges the base by a quarter of that step,
    // so the base follows the context in wide stretches too
    const double y = std::log(std::max(ms, 1e-3));
    if (narrow) {
      lb_ += (y - lb_) / double(std::min(nb_ + 1, 16));
      ++nb_;
      ++ncost8n_;
      last8n_ = cycles_;
      ++narrow_cycles_;
      if (refresh_ < 0) refresh_ = 0;
    } else {
      if (c_.ncost[n] == 0 || (nb_ == 0 && ncost8n_ == 0)) {
        if (ncost8n_ == 0 && nb_ == 0 && n == 8) lb_ = y, lq_[n] = 0.0;  // no 8-row verify yet: the base is wide 8 rows
        else lq_[n] = y - lb_;
      } else {
        const double e = y - (lb_ + lq_[n]), eta = 1.0 / double(std::min(c_.ncost[n] + 1, 16));
        lq_[n] += eta * e;
        lb_ += 0.25 * eta * e;
      }
      ++c_.ncost[n];
      last_[n] = cycles_;
      if (n == refresh_) refresh_ = 0;
    }
    wide_costs();
  } else {
    c_.observe(n, ms);  // the costs (KfCosts::observe)
  }
  // the rate: the recent cycles' tokens per ms (sums decaying by 1/64 a cycle)
  rtok_ = rtok_ * (63.0 / 64.0) + double(std::clamp(int(t[kf::kA]), 0, kf::kMaxRows) + 1);
  rms_ = rms_ * (63.0 / 64.0) + ms;
  ++nrate_;
  if (wide && cycles_ % kRefresh == 0) {  // the count measured longest ago
    int stale = 0, at = cycles_;
    for (int m = 1; m <= max_rows_; ++m)
      if (((allowed_ >> (m - 1)) & 1u) && last_[m] < at) stale = m, at = last_[m];
    if (last8n_ < at) stale = -1, at = last8n_;
    if (stale && cycles_ - at >= kRefresh) refresh_ = stale;
  }
  // wide verifies: the tree's E(8), E(16), E(32) (a build of max_rows nodes), the capacity decision every 32 cycles
  if (wide && nodes >= std::min(max_rows_, 32)) {  // a whole build (capacity 8 builds 8 nodes mostly)
    float e[3] = {0, 0, 0};
    const int counts[3] = {8, 16, 32};
    for (int i = 0; i < 3; ++i)
      if (obs) memcpy(&e[i], &t[kf::kObs + 4 + i], 4);
      else if (counts[i] <= nodes) memcpy(&e[i], &t[kf::kE + counts[i]], 4);
    if (ehist_.size() < size_t(3 * kEWin)) {
      ehist_.insert(ehist_.end(), e, e + 3);
    } else {
      memcpy(&ehist_[3 * ehead_], e, sizeof e);
      ehead_ = (ehead_ + 1) % size_t(kEWin);
    }
    if (++since_eval_ >= 32) {
      since_eval_ = 0;
      eval_wide();
    }
  }
  const int nrec = std::clamp(int(t[kf::kNrec]), 0, kf::kMaxRows);
  for (int i = 0; i < nrec; ++i) {
    const int d = t[kf::kRecDep + i];
    if (d < 1 || d > kf::kMaxRows) continue;
    float lq;
    memcpy(&lq, &t[kf::kRecLq + i], 4);
    add_record(d, lq, t[kf::kRecHit + i] != 0);
  }
  refit_tick();
}

void KfBudget::add_record(int d, float lq, bool hit) {
  const float x = logit_q(lq);
  const uint8_t y = hit ? 1 : 0;
  if (x_[d].size() < size_t(kWin)) {
    x_[d].push_back(x);
    y_[d].push_back(y);
  } else {
    x_[d][head_[d]] = x;
    y_[d][head_[d]] = y;
    head_[d] = (head_[d] + 1) % size_t(kWin);
  }
}

void KfBudget::refit_tick() {
  if (++since_fit_ >= 16) {
    since_fit_ = 0;
    for (int d = 1; d < kD; ++d)
      if (x_[d].size() >= 32) fit(x_[d], y_[d], alpha_[d], beta_[d], 4);
  }
}

void KfBudget::records(const int* dep, const float* lq, const uint8_t* hit, int n) {
  std::lock_guard<std::mutex> lk(mu_);
  for (int i = 0; i < n; ++i)
    if (dep[i] >= 1 && dep[i] <= kf::kMaxRows) add_record(dep[i], lq[i], hit[i] != 0);
  refit_tick();
}

void KfBudget::calibration(float* alpha, float* beta, int n) const {
  std::lock_guard<std::mutex> lk(mu_);
  for (int d = 0; d < n && d < kD; ++d) {
    alpha[d] = alpha_[d];
    beta[d] = beta_[d];
  }
}

void KfBudget::write(KfParams& p) {
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
  const bool wide = wide_ && max_rows_ > 8;
  bool narrow = false;
  if (!forced) {
    if (wide) {
      // exploration: every allowed row count measured twice before the table decides, the narrow 8 rows too (the
      // base first: this process's 8-row verify cost), and the refreshes; a pinned capacity (pin_wide) runs wide from
      // the first cycle and has no narrow cost to learn, so its base starts at that cycle's own cost
      narrow = !pinned_ && nb_ == 0;
      if (!narrow)
        for (int n = 1; n <= max_rows_; ++n)
          if (((allowed_ >> (n - 1)) & 1u) && c_.ncost[n] < 2) {
            force = n;
            break;
          }
      if (!force && !pinned_ && ncost8n_ < 2) narrow = true;
      if (!force && !narrow) {
        if (refresh_ > 0) force = refresh_;
        else if (refresh_ < 0 && !pinned_) narrow = true;
      }
      // capacity 8 unless the per-cycle choice among the wider counts pays (eval_wide)
      if (!force && !narrow && !wide_pays_) narrow = true;
    } else {
      // with a cached table and a choice of counts, this process's first cycles verify a count it holds (they set
      // the base); then every allowed count kExplore times (KfCosts::force)
      if (!force) force = c_.force(allowed_, max_rows_, kExplore);
    }
  }
  p.force_n = force;
  p.allowed = allowed_;
  p.max_rows = max_rows_;
  p.cap = 0;
  p.rate = nrate_ >= 16 && rms_ > 0 ? float(rtok_ / rms_) : 0.0f;
  if (narrow) {  // the verify runs 8 rows; one cycle in kStatsEvery still builds max_rows nodes (their E(n) feed
                 // eval_wide), the others 8 (a 32-node build costs the 4B's drafting pass ~0.04 ms)
    p.cap = 8;
    p.allowed = allowed_ & 0xffu ? allowed_ & 0xffu : 1u << 7;
    p.force_n = 0;
    if (narrow_writes_++ % kStatsEvery) p.max_rows = 8;
  }
  p.cost[0] = 1.0f;
  for (int n = 1; n < kD; ++n) {
    const double c = cur_cost(n);
    p.cost[n] = c > 0 ? float(std::max(c, 0.01)) : 1.0f;
  }
}

std::string KfBudget::summary() const {
  std::lock_guard<std::mutex> lk(mu_);
  char b[1024];
  const bool wide = wide_ && max_rows_ > 8;
  int o = snprintf(b, sizeof b, "KFlash: %d cycles, %.2f rows per cycle; cycles by rows", cycles_,
                   cycles_ ? rows_sum_ / cycles_ : 0.0);
  for (int n = 1; n < kD && o < int(sizeof b) - 16; ++n)
    if (ncycles_[n]) o += snprintf(b + o, sizeof b - o, " %d:%d", n, ncycles_[n]);
  if (wide) o += snprintf(b + o, sizeof b - o, "; %d at capacity 8", narrow_cycles_);
  if (ne_) o += snprintf(b + o, sizeof b - o, "; tokens per cycle expected %.3f, got %.3f", epred_ / ne_, etok_ / ne_);
  if (rms_ > 0) o += snprintf(b + o, sizeof b - o, "; rate %.4f tok/ms", rtok_ / rms_);
  o += snprintf(b + o, sizeof b - o, "; cost ms by rows");
  for (int n = 1; n <= max_rows_ && o < int(sizeof b) - 16; ++n)
    if ((allowed_ >> (n - 1)) & 1u) o += snprintf(b + o, sizeof b - o, " %d:%.2f", n, cur_cost(n));
  if (wide) o += snprintf(b + o, sizeof b - o, " (8 at capacity 8: %.2f)", cost8n_);
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
    if (j.value("cost_of", "") == kCostOf) {
      c_.load(c.get<std::vector<double>>(), kExplore);
      // wide verifies' extras (a cache a run without them wrote: its 8 rows are the 8-row verify's cost, set_wide)
      wide_saved_ = j.value("wide", false);
      if (wide_saved_) {
        if (j.value("cost_narrow8", 0.0) > 0) cost8n_ = j["cost_narrow8"].get<double>(), ncost8n_ = 2;
        wide_pays_ = j.value("wide_pays", true);  // the last decision, until this process's trees decide again
      }
    }
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
  const bool wide = wide_ && max_rows_ > 8;
  if (wide) {
    std::vector<double> cost(kD, 0.0);  // the costs now
    for (int n = 1; n < kD; ++n)
      if (cost_[n] > 0) cost[size_t(n)] = std::max(cost_[n], 0.01);
    j["cost"] = cost;
    j["wide"] = true;
    j["cost_narrow8"] = cost8n_;
    j["wide_pays"] = wide_pays_;
  } else {
    j["cost"] = c_.table();  // the costs now
  }
  j["alpha"] = std::vector<float>(alpha_, alpha_ + kD);
  j["beta"] = std::vector<float>(beta_, beta_ + kD);
  const std::string path = dir + "/" + key_ + ".json";
  if (!write_whole(path, j)) return;
  saved_ = cycles_;
  logi("KFlash: budget saved (%d cycles) to %s", cycles_, path.c_str());
}

// ---- KfBatch ----

KfBatch::KfBatch(std::string key, int floor) : key_(std::move(key)), floor_(std::clamp(floor, 0, kDrafts)) {
  for (int d = 0; d <= kDrafts; ++d) {
    alpha_[d] = 0.0f;
    beta_[d] = 1.0f;
  }
  load();
}

KfBatch::~KfBatch() { save(); }

int KfBatch::allocate(const float* surv, int M, const int* kmax, const double* cost, double rate, int floor,
                      int force_n, int* k) {
  M = std::clamp(M, 1, kRows);
  int n = M;
  double E = M;  // the expected tokens of the rows taken: each member's anchor emits one, each draft its survival
  // the floor: as many drafts as every member can have within the rows (and its cap)
  const int fl = std::clamp(std::min(floor, (kRows - M) / M), 0, kDrafts);
  for (int g = 0; g < M; ++g) {
    k[g] = std::min(fl, std::clamp(kmax[g], 0, kDrafts));
    n += k[g];
    for (int p = 0; p < k[g]; ++p) E += surv[g * kDrafts + p];
  }
  struct Cand {
    float s;
    int p, g;
  };
  Cand c[kRows * kDrafts];
  int nc = 0;
  for (int g = 0; g < M; ++g)
    for (int p = k[g]; p < std::clamp(kmax[g], 0, kDrafts); ++p) {
      const float v = surv[g * kDrafts + p];
      c[nc++] = {v >= 0.0f ? v : 0.0f, p, g};  // (NaN: worth nothing)
    }
  std::sort(c, c + nc, [](const Cand& a, const Cand& b) {
    return a.s > b.s || (a.s == b.s && (a.p < b.p || (a.p == b.p && a.g < b.g)));
  });
  for (int i = 0; i < nc && n < kRows; ++i) {
    const Cand& e = c[i];
    if (e.p != k[e.g]) continue;  // (never: a member's survivals do not rise along its chain)
    if (force_n > 0) {
      if (n >= force_n) break;
    } else {
      // the row pays at the long-run rate (rate 0: the expected tokens per ms grow with it)
      const bool pays = rate > 0 ? double(e.s) >= rate * (cost[n + 1] - cost[n])
                                 : (E + double(e.s)) * cost[n] > E * cost[n + 1];
      if (!pays) break;
    }
    ++k[e.g];
    ++n;
    E += double(e.s);
  }
  return n;
}

void KfBatch::cost_line(const double* cost, int lo, int hi, double* out) {
  int ns = 0, xs[kRows + 1];
  for (int n = lo; n <= hi; ++n)
    if (cost[n] > 0) xs[ns++] = n;
  if (ns == 0) return;
  double b = 0.0;
  if (ns >= 2) {
    std::vector<double> sl;
    for (int i = 0; i < ns; ++i)
      for (int j = i + 1; j < ns; ++j) sl.push_back((cost[xs[j]] - cost[xs[i]]) / double(xs[j] - xs[i]));
    std::sort(sl.begin(), sl.end());
    const size_t h = sl.size() / 2;
    b = std::max(0.0, sl.size() % 2 ? sl[h] : 0.5 * (sl[h - 1] + sl[h]));
  }
  std::vector<double> ic;
  for (int i = 0; i < ns; ++i) ic.push_back(cost[xs[i]] - b * xs[i]);
  std::sort(ic.begin(), ic.end());
  const size_t h = ic.size() / 2;
  const double a = ic.size() % 2 ? ic[h] : 0.5 * (ic[h - 1] + ic[h]);
  for (int n = lo; n <= hi; ++n) out[n] = std::max(a + b * n, 0.01);
}

int KfBatch::plan(Member* m, int M, const KfBudget* greedy, int* k) {
  if (M < 1 || M > kMembers) {  // (not the budget's: plain rows)
    for (int g = 0; g < M; ++g) k[g] = 0;
    return M;
  }
  float ga[kDrafts + 1], gb[kDrafts + 1];
  for (int d = 0; d <= kDrafts; ++d) ga[d] = 0.0f, gb[d] = 1.0f;
  if (greedy) greedy->calibration(ga, gb, kDrafts + 1);
  std::lock_guard<std::mutex> lk(mu_);
  float surv[kMembers * kDrafts];
  int kmax[kMembers];
  for (int g = 0; g < M; ++g) {
    double sv = 1.0;
    for (int p = 0; p < kDrafts; ++p) {
      const int d = p + 1;
      const float a = m[g].sampled ? alpha_[d] : ga[d], b = m[g].sampled ? beta_[d] : gb[d];
      const double z = double(a) + double(b) * double(KfBudget::logit_q(std::min(m[g].lc[p], 0.0f)));
      sv *= 1.0 / (1.0 + std::exp(-z));
      m[g].surv[p] = surv[g * kDrafts + p] = float(sv);
    }
    kmax[g] = std::clamp(m[g].kmax, 0, kDrafts);
  }
  // the counts it may choose: from every member's anchor and floor up to the verify's rows
  const int nmin = M + M * std::clamp(std::min(floor_, (kRows - M) / M), 0, kDrafts);
  uint32_t allowed = 0;
  for (int n = nmin; n <= kRows; ++n) allowed |= 1u << (n - 1);
  // NI_BATCH_KF_FORCE=n (tests): every cycle n rows in all (-1: the allowed counts in turn, cycle by cycle)
  static const int forced = getenv("NI_BATCH_KF_FORCE") ? std::clamp(atoi(getenv("NI_BATCH_KF_FORCE")), -1, kRows) : 0;
  int force = forced > 0 ? std::max(forced, nmin) : 0;
  if (forced < 0) force = nmin + cycles_ % (kRows - nmin + 1);
  if (!force) force = c_[M].force(allowed, kRows, kExplore);
  if (!force && mcyc_[M] > 0 && mcyc_[M] % kRefresh == 0)  // the count measured longest ago
    for (int n = nmin; n <= kRows; ++n)
      if (!force || seen_[M][n] < seen_[M][force]) force = n;
  double meas[kRows + 1], cost[kRows + 1];
  for (int n = 0; n <= kRows; ++n) meas[n] = n >= nmin ? c_[M].at(n) : 0.0, cost[n] = std::max(meas[n], 0.01);
  cost_line(meas, nmin, kRows, cost);
  const double rate = nrate_[M] >= 16 && rms_[M] > 0 ? rtok_[M] / rms_[M] : 0.0;
  return allocate(surv, M, kmax, cost, rate, floor_, force, k);
}

void KfBatch::observe(const Member* m, int M, const int* k, const int* a, double ms, KfBudget* greedy) {
  if (M < 1 || M > kMembers) return;
  int gd[kMembers * kDrafts];
  float glq[kMembers * kDrafts];
  uint8_t gh[kMembers * kDrafts];
  int ng = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    int n = M, tok = 0;
    double e = M;
    bool srec = false;
    for (int g = 0; g < M; ++g) {
      const int kg = std::clamp(k[g], 0, kDrafts), ag = std::clamp(a[g], 0, kg);
      n += kg;
      tok += ag + 1;
      for (int p = 0; p < kg; ++p) e += m[g].surv[p];
      // the records: every verified draft whose drafts before it were accepted, and whether it was (none of a masked
      // member's: the grammar's masks decided its acceptances, which would bias the drafter's calibration)
      for (int p = 0; p < (m[g].masked ? 0 : std::min(ag + 1, kg)); ++p) {
        const int d = p + 1;
        const bool hit = p < ag;
        if (m[g].sampled) {
          const float x = KfBudget::logit_q(std::min(m[g].lc[p], 0.0f));
          if (x_[d].size() < size_t(kWin)) {
            x_[d].push_back(x);
            y_[d].push_back(hit);
          } else {
            x_[d][head_[d]] = x;
            y_[d][head_[d]] = hit;
            head_[d] = (head_[d] + 1) % size_t(kWin);
          }
          srec = true;
        } else {
          gd[ng] = d;
          glq[ng] = m[g].lc[p];
          gh[ng] = hit;
          ++ng;
        }
      }
    }
    n = std::min(n, kRows);
    ++cycles_;
    ++ncyc_[M][n];
    // a count not measured in the last kRefresh cycles of M members (the refresh's, or one the rule left for long): its
    // old cost is stale (measured while the GPU did other work, or by another process), this cycle's sets it afresh
    if (mcyc_[M] - seen_[M][n] >= kRefresh) c_[M].forget(n);
    seen_[M][n] = ++mcyc_[M];
    epred_[M] += e;
    etok_[M] += tok;
    c_[M].observe(n, ms);
    // the rate: the recent cycles' tokens per ms at M members (sums decaying by 1/64 a cycle)
    rtok_[M] = rtok_[M] * (63.0 / 64.0) + double(tok);
    rms_[M] = rms_[M] * (63.0 / 64.0) + ms;
    ++nrate_[M];
    if (srec && ++since_fit_ >= 16) {
      since_fit_ = 0;
      for (int d = 1; d <= kDrafts; ++d)
        if (x_[d].size() >= 32) KfBudget::fit(x_[d], y_[d], alpha_[d], beta_[d], 4);
    }
  }
  if (greedy && ng) greedy->records(gd, glq, gh, ng);
}

double KfBatch::cost(int M, int N) const {
  std::lock_guard<std::mutex> lk(mu_);
  return M >= 1 && M <= kMembers ? c_[M].at(N) : 0.0;
}

std::string KfBatch::summary() const {
  std::lock_guard<std::mutex> lk(mu_);
  char b[2048];
  int o = snprintf(b, sizeof b, "KFlash batch: %d cycles", cycles_);
  for (int M = 2; M <= kMembers && o < int(sizeof b) - 160; ++M) {
    long cyc = 0, rows = 0;
    for (int n = 1; n <= kRows; ++n) cyc += ncyc_[M][n], rows += ncyc_[M][n] * n;
    if (!cyc) continue;
    o += snprintf(b + o, sizeof b - o, "; %d members: %.2f rows per cycle (by rows", M, double(rows) / double(cyc));
    for (int n = 1; n <= kRows; ++n)
      if (ncyc_[M][n]) o += snprintf(b + o, sizeof b - o, " %d:%ld", n, ncyc_[M][n]);
    o += snprintf(b + o, sizeof b - o, "), tokens per cycle expected %.3f, got %.3f", epred_[M] / double(cyc),
                  etok_[M] / double(cyc));
    if (rms_[M] > 0) o += snprintf(b + o, sizeof b - o, ", rate %.4f tok/ms", rtok_[M] / rms_[M]);
    o += snprintf(b + o, sizeof b - o, ", cost ms by rows");
    for (int n = M; n <= kRows && o < int(sizeof b) - 16; ++n)
      if (c_[M].ncost[n]) o += snprintf(b + o, sizeof b - o, " %d:%.2f", n, c_[M].at(n));
  }
  o += snprintf(b + o, sizeof b - o, "; sampled calibration (alpha, beta) by depth");
  for (int d = 1; d <= kDrafts && o < int(sizeof b) - 24; ++d)
    o += snprintf(b + o, sizeof b - o, " (%.2f, %.2f)", alpha_[d], beta_[d]);
  return b;
}

void KfBatch::load() {
  if (key_.empty()) return;
  std::ifstream f(hub::home_dir() + "/kflash/" + key_ + ".json");
  if (!f) return;
  try {
    const nlohmann::json j = nlohmann::json::parse(f);
    if (j.value("cost_of", "") == kBatchCostOf) {
      const auto& c = j.at("cost");
      for (int M = 2; M <= kMembers && M < int(c.size()); ++M) c_[M].load(c[size_t(M)].get<std::vector<double>>(), kExplore);
    }
    const auto& a = j.at("alpha");
    const auto& bb = j.at("beta");
    for (int d = 1; d <= kDrafts && d < int(a.size()) && d < int(bb.size()); ++d) {
      alpha_[d] = a[size_t(d)].get<float>();
      beta_[d] = bb[size_t(d)].get<float>();
    }
  } catch (...) {
    logi("KFlash: ignoring an unreadable cache %s", key_.c_str());
  }
}

void KfBatch::save(int fresh) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key_.empty() || cycles_ < 32 || cycles_ - saved_ < std::max(fresh, 1)) return;
  const std::string dir = hub::home_dir() + "/kflash";
  mkdir(dir.c_str(), 0755);
  nlohmann::json j;
  j["cost_of"] = kBatchCostOf;
  std::vector<std::vector<double>> cost(kMembers + 1);  // by members (2 .. kMembers), each by rows
  for (int M = 2; M <= kMembers; ++M) cost[size_t(M)] = c_[M].table();
  j["cost"] = cost;
  j["alpha"] = std::vector<float>(alpha_, alpha_ + kDrafts + 1);
  j["beta"] = std::vector<float>(beta_, beta_ + kDrafts + 1);
  const std::string path = dir + "/" + key_ + ".json";
  if (!write_whole(path, j)) return;
  saved_ = cycles_;
  logi("KFlash: batch budget saved (%d cycles) to %s", cycles_, path.c_str());
}

}  // namespace ni
