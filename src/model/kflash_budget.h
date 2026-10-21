// KFlash (Kapitsa's drafter generation after DFlash2: best-first trees over a drafter's candidates, verified exactly)
// on the host: the tree descriptor's layout and the builder's parameters (src/kernels/kflash.h, shared with the GPU),
// and the budget that sets them between cycles, the same for every backend and with no per-model constants:
//  - calibration: the path model's probability q of a node is mapped to the probability that the target accepts it,
//    p = sigmoid(alpha_d + beta_d logit(q)) per depth d, fitted online (logistic regression, Newton steps on the last
//    records) from the acceptance's records: every verified node whose parent was accepted, and whether it was.
//    Before any records, alpha 0 and beta 1: p = q.
//  - cost: a cycle's GPU time with n verify rows, measured on the Mac it runs on: its drafting half's command buffer
//    and its verify's, in ms, whichever tool runs the cycles (ni_run, the server), so the cache means one thing
//    whoever wrote it. The first cycles try every allowed n three times (unless the cache already holds a table for
//    this model, drafter and chip). A cost is a base that every cycle moves (the context's length: the 35B's cycles
//    cost ~7 ms more at 21k tokens than at 100, whatever n) plus n's own offset, a running mean of what its cycles
//    cost above the base; each process sets the base by its first cycle (a count the cache holds).
// The builder takes the allowed n that maximizes E(n) - rate cost[n], E(n) the tokens a verify of n rows is expected
// to emit (1 + the summed calibrated path probabilities of the first n - 1 nodes) and rate the tokens per ms the
// recent cycles gave (their long-run rate, which a row must beat to pay: the base cancels); before 16 cycles of this
// process, E(n) / cost[n]. Which n a backend allows is what its verify kernels run at a cost of their own: whole
// 8-row tiles on the dense Qwen models (one tile today), every row count on MoE targets, whose rows cost experts.
// The cache file (KAPITSA_HOME/kflash/<key>.json) keeps the table and the calibration: written when the budget ends
// and on save() (the engine saves between requests, the server when it is stopped).
#pragma once
#include "kernels/kflash.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ni {

class KfBudget {
 public:
  // key: the cache file's name (model, drafter, chip); "" keeps nothing
  explicit KfBudget(std::string key = "");
  ~KfBudget();  // save()
  // the row counts the backend's verify runs (bit n - 1) and its capacity; default: 1 .. 8 rows, 8
  void set_rows(uint32_t allowed, int max_rows);
  // after a cycle: the tree descriptor as its acceptance left it (rows, records) and the cycle's GPU time (ms): its
  // drafting half's command buffer and its verify's, Encoder::gpu_ms() of each
  void observe(const int32_t* tree, double ms);
  // before the next cycle's drafting pass
  void write(KfParams& p) const;
  // the table and the calibration to the cache file, once 32 cycles were observed here and `fresh` of them since the
  // last save. From any thread (the server saves on its stop signal while a request may be in its cycles).
  void save(int fresh = 1);
  static constexpr int kSaveEvery = 256;  // the engine's save after a request: a server killed hard loses less
  // for logs: rows chosen on average and the cycles by rows, the calibration, the cost table
  std::string summary() const;
  int cycles() const { return cycles_; }
  // n rows' cost now (ms; 0: not measured)
  double cost(int n) const { return n >= 1 && n <= kf::kMaxRows && ncost_[n] ? base_ + off_[n] : 0.0; }
  static constexpr int kExplore = 3;  // cycles per allowed count before the rule decides
  float alpha(int d) const { return alpha_[d]; }
  float beta(int d) const { return beta_[d]; }

  // (tests) the calibration fit on given records
  static void fit(const std::vector<float>& x, const std::vector<uint8_t>& y, float& alpha, float& beta, int iters);
  static float logit_q(float lq);

 private:
  void load();
  std::string key_;
  mutable std::mutex mu_;  // observe, write, summary and save (the cycles' thread and a server's stop)
  static constexpr int kD = kf::kMaxRows + 1;  // depths and row counts: index 1 .. 32
  static constexpr int kWin = 2048;            // records kept per depth
  std::vector<float> x_[kD];                   // logit(q) per depth (ring)
  std::vector<uint8_t> y_[kD];
  size_t head_[kD] = {};
  float alpha_[kD], beta_[kD];
  double base_ = 0;        // the costs' base (ms) and the counts' offsets above it: cost(n) = base_ + off_[n]
  double off_[kD] = {};    // (loaded: the cache's costs, base 0 at its save)
  bool based_ = false;     // this process set the base (its first cycle; with a cached table: a count it holds)
  bool loaded_ = false;    // the cache held a cost table
  uint64_t pre_ = 0;       // counts this process measured before its base (offsets absolute until then)
  int ncost_[kD] = {};     // cycles that measured n (loaded: kExplore)
  double rtok_ = 0, rms_ = 0;  // the recent cycles' tokens and ms (decaying sums): the rate
  int nrate_ = 0;
  int ncycles_[kD] = {};  // cycles by rows (the summary)
  double epred_ = 0, etok_ = 0;  // the summary: E(n) the builder expected, the tokens the cycles gave (cycles with E)
  int ne_ = 0;
  uint32_t allowed_ = 0xffu;
  int max_rows_ = 8;
  int cycles_ = 0, since_fit_ = 0, force_ = 0, saved_ = 0;  // saved_: cycles_ at the last save
  double rows_sum_ = 0;
};

}  // namespace ni
