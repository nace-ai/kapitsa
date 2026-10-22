// KFlash (Kapitsa's drafter generation after DFlash2: best-first trees over a drafter's candidates, verified exactly)
// on the host: the tree descriptor's layout and the builder's parameters (src/kernels/kflash.h, shared with the GPU),
// and the budget that sets them between cycles, the same for every backend and with no per-model constants:
//  - calibration: the path model's probability q of a node is mapped to the probability that the target accepts it,
//    p = sigmoid(alpha_d + beta_d logit(q)) per depth d, fitted online (logistic regression, Newton steps on the last
//    records) from the acceptance's records: every verified node whose parent was accepted, and whether it was.
//    Before any records, alpha 0 and beta 1: p = q.
//  - cost: a cycle's GPU time with n verify rows, measured on the Mac it runs on: its drafting half's command buffer
//    and its verify's, in ms, whichever tool runs the cycles (ni_run, the server), so the cache means one thing
//    whoever wrote it. The first cycles try every allowed n a few times (unless the cache already holds a table for
//    this model, drafter and chip). A cost is a base that every cycle moves (the context's length: the 35B's cycles
//    cost ~7 ms more at 21k tokens than at 100, whatever n) plus n's own offset, a running mean of what its cycles
//    cost above the base; each process sets the base by its first cycle (a count the cache holds). Wide verifies
//    (below) model the drift as a ratio: log cost(n) = log base + log ratio(n), the base an 8-row verify's cost now
//    and the ratios each count's own, fitted online.
// The builder takes the allowed n that maximizes E(n) - rate cost[n], E(n) the tokens a verify of n rows is expected
// to emit (1 + the summed calibrated path probabilities of the first n - 1 nodes) and rate the tokens per ms the
// recent cycles gave (their long-run rate, which a row must beat to pay: the base cancels); before 16 cycles of this
// process, E(n) / cost[n]. Which n a backend allows is what its verify kernels run at a cost of their own: whole
// 8-row tiles on the dense Qwen models (8 | 16 | 32 rows), every row count on MoE targets, whose rows cost experts.
// Wide verifies (set_wide: Qwen's 16 / 32 rows, one verify of capacity max_rows whose live tiles the GPU picks): the
// capacity is a choice too. A wide verify of 8 live rows costs a little more than the 8-row verify (its grids are
// GPU-driven), so each cycle verifies at capacity 8 (the rule then has 8 rows only) unless the recent cycles' trees
// (the builder still builds max_rows nodes and records E(8), E(16), E(32)) say the per-cycle choice among all counts
// gives more tokens per ms than 8 rows at the 8-row verify's cost; both costs are measured, like every count's.
// The cache file (KAPITSA_HOME/kflash/<key>.json) keeps the table and the calibration: written when the budget ends
// and on save() (the engine saves between requests, the server when it is stopped).
// KfBatch (below) is the same budget for batched decode: the rows one verify pass gives the members of a batch.
#pragma once
#include "kernels/kflash.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ni {

// A cycle's cost by the rows it verifies (KfBudget's and KfBatch's tables): a base that every cycle moves (the
// context's length) plus each count's offset above it, a running mean of what its cycles cost above the base. A
// process sets the base by its first cycle; with a table loaded from the cache, by its first cycle of a count the cache
// holds (until then the counts it measures keep absolute offsets, `pre`).
struct KfCosts {
  static constexpr int kD = kf::kMaxRows + 1;  // row counts: index 1 .. 32
  double base = 0;
  double off[kD] = {};
  bool based = false, loaded = false;
  uint64_t pre = 0;
  int ncost[kD] = {};  // cycles that measured n (loaded: the exploration's count)
  void observe(int n, double ms);
  // n rows' cost now (0: not measured)
  double at(int n) const { return n >= 1 && n < kD && ncost[n] ? base + off[n] : 0.0; }
  // the count the next cycle must verify (0: none): with a loaded table and a choice of counts, before this process's
  // base, a count the table holds (the largest); then every allowed count (bit n - 1) `explore` times
  int force(uint32_t allowed, int max_rows, int explore) const;
  // the cache's table (n: the cost now; 0: not measured, or not since this process's base), and one read back
  std::vector<double> table() const;
  void load(const std::vector<double>& c, int explore);
  // n's cost forgotten (after the base: its next cycle sets it afresh)
  void forget(int n) {
    if (based && n >= 1 && n < kD) ncost[n] = 0;
  }
};

class KfBudget {
 public:
  // key: the cache file's name (model, drafter, chip); "" keeps nothing
  explicit KfBudget(std::string key = "");
  ~KfBudget();  // save()
  // the row counts the backend's verify runs (bit n - 1) and its capacity; default: 1 .. 8 rows, 8
  void set_rows(uint32_t allowed, int max_rows);
  // the counts past 8 run as one wide verify of capacity max_rows (see above): the capacity is chosen per cycle
  void set_wide(bool on);
  // every cycle verifies at capacity max_rows: for a drafter whose block is wider than the 8-row tile, which drafts
  // the whole lattice per weight read, a capacity 8 cycle verifies half of what was drafted (measured on a 16-row
  // block: 69.3 tok/s on narrow cycles against 90.6 on wide ones, so the fallback is not a choice worth having there)
  void pin_wide();
  // after a cycle: the tree descriptor as its acceptance left it (rows, records) and the cycle's GPU time (ms): its
  // drafting half's command buffer and its verify's, Encoder::gpu_ms() of each
  void observe(const int32_t* tree, double ms);
  // before the next cycle's drafting pass (p.cap: the capacity the verify must be encoded with)
  void write(KfParams& p);
  // the table and the calibration to the cache file, once 32 cycles were observed here and `fresh` of them since the
  // last save. From any thread (the server saves on its stop signal while a request may be in its cycles).
  void save(int fresh = 1);
  static constexpr int kSaveEvery = 256;  // the engine's save after a request: a server killed hard loses less
  // for logs: rows chosen on average and the cycles by rows, the calibration, the cost table
  std::string summary() const;
  int cycles() const { return cycles_; }
  // n rows' cost now (ms; 0: not measured)
  double cost(int n) const { return n >= 1 && n <= kf::kMaxRows ? cur_cost(n) : 0.0; }
  static constexpr int kExplore = 3;  // cycles per allowed count before the rule decides (wide verifies: 2)
  double cost_narrow() const { return cost8n_; }  // an 8-row cycle at capacity 8 (wide verifies)
  float alpha(int d) const { return alpha_[d]; }
  float beta(int d) const { return beta_[d]; }
  // Calibration records from outside a tree cycle (batched decode's greedy chains: a chain's node is a tree's node of
  // the same lattice, its q the same): depth, the path model's log q, accepted; counted toward the refits as a cycle's
  // records are. And the calibration of depths 1 .. n - 1 as it stands (index d).
  void records(const int* dep, const float* lq, const uint8_t* hit, int n);
  void calibration(float* alpha, float* beta, int n) const;

  // (tests) the calibration fit on given records
  static void fit(const std::vector<float>& x, const std::vector<uint8_t>& y, float& alpha, float& beta, int iters);
  static float logit_q(float lq);

 private:
  void load();
  void add_record(int d, float lq, bool hit);  // (under mu_)
  void refit_tick();                           // (under mu_) every 16 cycles' records: the calibration's Newton steps
  std::string key_;
  mutable std::mutex mu_;  // observe, write, summary and save (the cycles' thread and a server's stop)
  static constexpr int kD = kf::kMaxRows + 1;  // depths and row counts: index 1 .. 32
  static constexpr int kWin = 2048;            // records kept per depth
  std::vector<float> x_[kD];                   // logit(q) per depth (ring)
  std::vector<uint8_t> y_[kD];
  size_t head_[kD] = {};
  float alpha_[kD], beta_[kD];
  KfCosts c_;  // the cost table (loaded: the cache's costs, base 0 at its save)
  double rtok_ = 0, rms_ = 0;  // the recent cycles' tokens and ms (decaying sums): the rate
  int nrate_ = 0;
  int ncycles_[kD] = {};  // cycles by rows (the summary)
  double epred_ = 0, etok_ = 0;  // the summary: E(n) the builder expected, the tokens the cycles gave (cycles with E)
  int ne_ = 0;
  // wide verifies: log cost(n) = lb_ + lq_[n] (lb_: the 8-row verify's log cost now, set by this process's first
  // narrow cycle and moved by every cycle; lq_[n]: count n's log ratio to it, c_.ncost[n] its measurements) and the
  // narrow 8 rows' (count, last measured, measurements this process); cost_[n] / cost8n_ are their current values;
  // the capacity decision, the recent trees' E(8), E(16), E(32) (a ring), cycles run narrow
  double cost_[kD] = {};  // the counts' costs now (exp(lb_ + lq_[n]))
  double cur_cost(int n) const { return wide_ && max_rows_ > 8 ? cost_[n] : c_.at(n); }
  int last_[kD] = {};  // the cycle that last measured each row count
  int refresh_ = 0;    // a row count the next cycles verify to re-measure it (0: none; -1: the narrow 8 rows)
  bool wide_ = false, wide_pays_ = true, wide_saved_ = false;  // wide_saved_: a wide run wrote the cache
  bool pinned_ = false;  // wide_pays_ is not a decision (pin_wide): every cycle verifies at capacity max_rows
  double cost8n_ = 0, lb_ = 0, lq_[kD] = {};
  int ncost8n_ = 0, last8n_ = 0, nb_ = 0, narrow_cycles_ = 0, since_eval_ = 0, narrow_writes_ = 0;
  static constexpr int kStatsEvery = 8;  // capacity 8: one cycle in this many builds max_rows nodes (the E statistics)
  void wide_costs();  // cost_[n], cost8n_ from lb_, lq_
  static constexpr int kEWin = 256;
  static constexpr int kRefresh = 2048;  // cycles between re-measurements of the stalest allowed row count
  std::vector<float> ehist_;  // 3 per cycle
  size_t ehead_ = 0;
  void eval_wide();
  uint32_t allowed_ = 0xffu;
  int max_rows_ = 8;
  int cycles_ = 0, since_fit_ = 0, force_ = 0, saved_ = 0;  // saved_: cycles_ at the last save
  double rows_sum_ = 0;
};

// KFlash's budget for batched decode (Engine::batch_cycle, DFlash2): after the members' drafting passes, how many rows
// the batched verify checks in all and which member gets them, by KfBudget's rule (decided on the host: the batched
// verify's per-member dispatches take their rows when encoded).
//  - survival of a member's draft p: the product over j <= p of sigmoid(alpha_d + beta_d logit(c_j)), d = j + 1, c_j the
//    largest of the 16 candidates' probabilities under the walk's scores at j. A greedy walk drafts that candidate (c_j
//    is its node's q, as in a tree: KfBudget's calibration, fed these chains' records too); a sampled member keeps its
//    draws by exact speculative sampling, another event: this budget's own calibration. c_j does not move with the draw
//    at j, and each draft is decided from the drafts ranked before it alone, so the sampling stays exact.
//  - cost: a batched cycle's GPU time (drafting passes and verify) by members M and rows N (KfCosts per M; every N
//    explored kExplore times, cached; every kRefresh-th cycle verifies the count measured longest ago, which then takes
//    its new cost afresh), read as a line in N (cost_line) so that a count measured while the GPU was busy elsewhere
//    does not stop the rows above it.
//  - rule (allocate): every member verifies its anchor (and `floor` drafts), then drafts in decreasing survival (ties:
//    the earlier position, then the earlier member, so a member's drafts are a prefix of its chain), each while its
//    survival >= rate x (cost(N + 1) - cost(N)) (rate: the recent cycles' tokens per ms at M; before 16 of them, while
//    the tokens per ms grow), up to kRows rows. With a convex cost that is the best allocation of them all.
// Cache: KAPITSA_HOME/kflash/<key>.json (the costs by M, the sampled calibration).
class KfBatch {
 public:
  static constexpr int kMembers = 4;  // members it allocates rows for (2 .. kMembers)
  static constexpr int kRows = 8;     // rows the batched verify runs (Qwen35::encode_verify_multi)
  static constexpr int kDrafts = 7;   // a member's drafts per cycle (the walk's chain)
  static constexpr int kExplore = 3;  // cycles per allowed N and M before the rule decides
  static constexpr int kRefresh = 64; // every kRefresh-th cycle of M members verifies the count measured longest ago
  static constexpr int kSaveEvery = 256;
  // key: the cache file's name; "" keeps nothing. floor: drafts every member verifies (within the 8 rows)
  explicit KfBatch(std::string key = "", int floor = 0);
  ~KfBatch();  // save()
  struct Member {
    float lc[kDrafts] = {};    // log c_p, p = 0 .. 6
    bool sampled = false;      // T > 0: exact speculative sampling (this budget's calibration)
    bool masked = false;       // its rows sample under a grammar's masks (structured output): no calibration records
    int kmax = kDrafts;        // drafts it may verify
    float surv[kDrafts] = {};  // (plan) its survivals
  };
  // The drafts each of M members verifies this cycle (k[g] in 0 .. kmax; its rows k[g] + 1); greedy members' survivals
  // from `greedy`'s calibration (the drafter's KfBudget; null: the identity). Returns the rows in all.
  int plan(Member* m, int M, const KfBudget* greedy, int* k);
  // After the cycle: each member's verified drafts k[g] and accepted drafts a[g] (<= k[g]), the cycle's GPU ms (its
  // drafting passes' command buffer and the verify's). The greedy members' records go to `greedy`; a masked member's
  // acceptances (a grammar's, not the drafter's) are no records, but its tokens and rows count in the rate and cost.
  void observe(const Member* m, int M, const int* k, const int* a, double ms, KfBudget* greedy);
  void save(int fresh = 1);
  // for logs: cycles by members and rows, tokens per cycle expected and got, rates, costs, the sampled calibration
  std::string summary() const;
  int cycles() const { return cycles_; }
  double cost(int M, int N) const;
  // The rule (see above), for M members' survivals surv[g * kDrafts + p] (non-increasing in p) with caps kmax[g], the
  // costs of N rows cost[N] (N = 0 .. kRows; only M .. kRows read), rate (0: the ratio rule), force_n (> 0: exactly that
  // many rows, the drafts in the same order, as many as there are). Writes k[g]; returns the rows in all.
  static int allocate(const float* surv, int M, const int* kmax, const double* cost, double rate, int floor,
                      int force_n, int* k);
  // The costs as a line in N over [lo, hi] (out[N]; N outside it untouched): through the measured counts (cost[N] > 0),
  // the slope the median of their pairwise slopes (not below 0), the intercept the median of what each count leaves.
  // Fewer than two counts measured: their cost, flat.
  static void cost_line(const double* cost, int lo, int hi, double* out);

 private:
  void load();
  std::string key_;
  int floor_ = 0;
  mutable std::mutex mu_;
  static constexpr int kWin = 2048;  // records kept per depth
  KfCosts c_[kMembers + 1];          // by M
  double rtok_[kMembers + 1] = {}, rms_[kMembers + 1] = {};  // the recent cycles' tokens and ms by M (decaying sums)
  int nrate_[kMembers + 1] = {};
  std::vector<float> x_[kDrafts + 1];  // the sampled members' records: logit(c) by depth (ring), accepted
  std::vector<uint8_t> y_[kDrafts + 1];
  size_t head_[kDrafts + 1] = {};
  float alpha_[kDrafts + 1], beta_[kDrafts + 1];
  int cycles_ = 0, saved_ = 0, since_fit_ = 0;
  long mcyc_[kMembers + 1] = {};             // cycles by M (this process)
  long seen_[kMembers + 1][kRows + 1] = {};  // mcyc_ when N was last measured at M (the refresh's choice)
  long ncyc_[kMembers + 1][kRows + 1] = {};  // the summary: cycles by M and N
  double epred_[kMembers + 1] = {}, etok_[kMembers + 1] = {};  // expected and got tokens by M
};

}  // namespace ni
