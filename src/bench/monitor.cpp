// GPU contention while `kapitsa bench` runs (bench.h): bench/harness/gpu_hogs.py's quiet wait, contention monitor and
// per-phase summaries, with the same thresholds, plus who used the GPU by app name, the thermal state and the power
// source at each sample.
#include "bench/bench.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace ni::bench {

namespace {

constexpr size_t kCommLen = 16;  // the IORegistry names a GPU client's process by its first 16 characters
std::string comm(const std::string& s) { return s.substr(0, kCommLen); }

}  // namespace

std::vector<std::string> ambient_names() {
  // gpu_hogs.py AMBIENT: macOS's photo and media analysis (busy for hours after an OS update), the same for every
  // engine; BENCH_GPU_AMBIENT adds names, as for the harness
  std::vector<std::string> v = {"mediaanalysisd", "photoanalysisd", "VisualIntelligenceCore"};
  if (const char* e = getenv("BENCH_GPU_AMBIENT")) {
    std::string s = e;
    for (size_t a = 0; a <= s.size();) {
      size_t b = s.find(',', a);
      if (b == std::string::npos) b = s.size();
      std::string x = s.substr(a, b - a);
      while (!x.empty() && isspace(uint8_t(x.front()))) x.erase(x.begin());
      while (!x.empty() && isspace(uint8_t(x.back()))) x.pop_back();
      if (!x.empty() && std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
      a = b + 1;
    }
  }
  return v;
}

bool is_ambient(const std::string& name) {
  static const std::vector<std::string> names = ambient_names();
  for (const auto& a : names)
    if (comm(a) == comm(name)) return true;
  return false;
}

bool is_ignored(const std::string& name) { return comm(name) == "WindowServer"; }

std::vector<UseRow> diff_rows(const GpuSnap& a, const GpuSnap& b, double secs, const std::set<int>& exclude) {
  std::vector<UseRow> rows;
  for (const auto& [k, t] : b) {
    const auto it = a.find(k);
    const double d = double(t) - double(it == a.end() ? 0 : it->second);
    const double pct = d / (secs * 1e9) * 100;
    if (pct > 0.5 && !exclude.count(k.first) && !is_ignored(k.second) && !is_ambient(k.second))
      rows.push_back({pct, k.first, k.second});
  }
  // Python sorts (pct, (pid, name)) tuples, highest first
  std::sort(rows.begin(), rows.end(), [](const UseRow& x, const UseRow& y) {
    if (x.pct != y.pct) return x.pct > y.pct;
    if (x.pid != y.pid) return x.pid > y.pid;
    return x.name > y.name;
  });
  return rows;
}

double Monitor::now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

Monitor::Monitor(std::function<std::set<int>()> own, std::set<int> ambient_pids, double interval)
    : own_(std::move(own)), ambient_pids_(std::move(ambient_pids)), interval_(interval) {}

Monitor::~Monitor() { stop(); }

void Monitor::start() {
  stop_ = false;
  th_ = std::thread([this] { run(); });
}

void Monitor::stop() {
  stop_ = true;
  if (th_.joinable()) th_.join();
}

std::vector<Sample> Monitor::samples() {
  std::lock_guard<std::mutex> lk(mu_);
  return samples_;
}

void Monitor::run() {
  GpuSnap a = gpu_snapshot();
  double ta = now();
  std::map<int, std::string> names;  // pid -> app name, resolved once
  while (!stop_) {
    // wait `interval` (gpu_hogs: stop_ev.wait(interval)), in short steps so stop() returns quickly
    for (double waited = 0; waited < interval_ && !stop_; waited += 0.05)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (stop_) break;
    const GpuSnap b = gpu_snapshot();
    const double tb = now(), dt = tb - ta;
    std::set<int> exclude = own_ ? own_() : std::set<int>{};
    exclude.insert(ambient_pids_.begin(), ambient_pids_.end());
    const auto rows = diff_rows(a, b, dt, exclude);
    Sample s;
    s.t = tb;
    for (const auto& r : rows) s.foreign += r.pct;
    auto app = [&](int pid, const std::string& c) {
      auto it = names.find(pid);
      if (it == names.end()) it = names.emplace(pid, app_name(pid, c)).first;
      return it->second;
    };
    s.top = rows.empty() ? "-" : app(rows[0].pid, rows[0].name);
    s.top_pct = rows.empty() ? 0 : rows[0].pct;
    // ambient: the named processes and this process's terminal; a client closed meanwhile loses its time: clamped
    for (const auto& [k, t] : b) {
      if (!is_ambient(k.second) && !ambient_pids_.count(k.first)) continue;
      const auto it = a.find(k);
      const double d = double(t) - double(it == a.end() ? 0 : it->second);
      s.ambient += std::max(0.0, d) / (dt * 1e9) * 100;
    }
    std::map<std::string, double> by_app;
    for (const auto& r : rows) by_app[app(r.pid, r.name)] += r.pct;
    for (const auto& [n, p] : by_app) s.users.push_back({n, p});
    std::sort(s.users.begin(), s.users.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
    s.thermal = thermal_state();
    const Power p = power(false);
    s.power_source = p.source;
    s.low_power = p.low_power_mode;
    {
      std::lock_guard<std::mutex> lk(mu_);
      samples_.push_back(std::move(s));
    }
    a = b;
    ta = tb;
  }
}

ojson summarize_samples(const std::vector<Sample>& all, double t0, double t1) {
  std::vector<const Sample*> s;
  for (const auto& x : all)
    if (x.t >= t0 && x.t <= t1 + 2.5) s.push_back(&x);
  if (s.empty()) return {{"samples", 0}, {"flagged", false}};
  double sum = 0, mx = 0, amb = 0;
  int run = 0, best = 0, busy = 0, therm = -1;
  std::map<std::string, int> top;
  std::map<std::string, double> users;
  bool battery = false, low = false;
  std::string thermal;
  for (const Sample* x : s) {
    sum += x->foreign, mx = std::max(mx, x->foreign), amb += x->ambient;
    run = x->foreign > kForeignMaxPct ? run + 1 : 0;
    best = std::max(best, run);
    if (x->foreign > kForeignMaxPct) ++busy, ++top[x->top];
    for (const auto& [n, p] : x->users) users[n] += p;
    if (thermal_rank(x->thermal) > therm) therm = thermal_rank(x->thermal), thermal = x->thermal;
    battery = battery || x->power_source == "battery";
    low = low || x->low_power;
  }
  const double n = double(s.size()), mean = sum / n;
  std::vector<std::pair<std::string, int>> tv(top.begin(), top.end());
  std::stable_sort(tv.begin(), tv.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  ojson topn = ojson::array();
  for (size_t i = 0; i < tv.size() && i < 5; ++i) topn.push_back(tv[i].first);
  // the other apps on the GPU over the phase, by their mean share (those of at least 0.5%)
  std::vector<std::pair<std::string, double>> uv(users.begin(), users.end());
  std::sort(uv.begin(), uv.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  ojson who = ojson::array();
  for (const auto& [name, total] : uv)
    if (total / n >= 0.5 && who.size() < 3) who.push_back({{"name", name}, {"mean_pct", py_round(total / n, 2)}});
  ojson o = {{"samples", s.size()},
             {"mean_foreign_gpu_pct", py_round(mean, 2)},
             {"max_foreign_gpu_pct", py_round(mx, 2)},
             {"mean_ambient_gpu_pct", py_round(amb / n, 2)},
             {"busy_samples", busy},
             {"longest_busy_run", best},
             {"top", topn},
             {"gpu_users", who},
             {"flagged", mean > kForeignMaxPct || best >= kSustainSamples}};
  if (!thermal.empty()) o["thermal"] = thermal;
  o["on_battery"] = battery;
  o["low_power_mode"] = low;
  return o;
}

bool wait_quiet(double maxpct, double quiet_s, double timeout_s, const std::function<std::set<int>()>& own,
                const std::set<int>& ambient_pids, double* waited, std::vector<Busy>* busy_out,
                const std::function<bool(const std::vector<Busy>&, double)>& tick) {
  const double iv = 2.0, t0 = Monitor::now();
  double ok_since = -1;
  std::vector<Busy> last;
  std::map<int, std::string> names;
  for (;;) {
    const GpuSnap a = gpu_snapshot();
    std::this_thread::sleep_for(std::chrono::milliseconds(int(iv * 1000)));
    const GpuSnap b = gpu_snapshot();
    const std::set<int> mine = own ? own() : std::set<int>{};
    const auto rows = diff_rows(a, b, iv, ambient_pids);  // the server's tree included: it must be idle too
    std::vector<Busy> busy;
    double own_pct = 0;
    for (const auto& r : rows) {
      if (mine.count(r.pid)) {
        own_pct += r.pct;
        continue;
      }
      if (r.pct > maxpct) {
        auto it = names.find(r.pid);
        if (it == names.end()) it = names.emplace(r.pid, app_name(r.pid, r.name)).first;
        busy.push_back({r.pct, it->second});
      }
    }
    if (own_pct > maxpct) busy.push_back({own_pct, "the server (still busy after its warm-up)"});
    const double now = Monitor::now();
    if (!busy.empty()) ok_since = -1, last = busy;
    else if (ok_since < 0) ok_since = now;
    *waited = now - t0;
    if (ok_since >= 0 && now - ok_since >= quiet_s) {
      busy_out->clear();
      return true;
    }
    if (now - t0 > timeout_s) {
      last.resize(std::min<size_t>(last.size(), 3));
      *busy_out = last;
      return false;
    }
    if (tick && !tick(busy, now - t0)) {
      *busy_out = last;
      return false;
    }
  }
}

}  // namespace ni::bench
