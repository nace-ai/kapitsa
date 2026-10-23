// Where a split model's layers go (src/cluster/pipe.h). Each Mac holds a run of the layers in memory, and the first Mac
// (the head) holds the first run and the head's own tensors (the LM head, the final mix, the MTP head). A token's
// pass costs each Mac its layers' decode bytes over its memory bandwidth, one Mac after another, plus a network round
// trip around the ring; place() picks the runs that fit every Mac's memory with the shortest pass. Pure: no model, no
// network (tools/split_place_test checks it).
#pragma once
#include <string>
#include <vector>

namespace ni::cluster {

struct Member {
  std::string name;
  double mem = 0;  // bytes this Mac can give the model's weights (its budget less its buffers and caches)
  double gbs = 0;  // its memory bandwidth, GB/s (0 unknown: taken as the slowest known member's)
};

struct LayerCost {
  double weights = 0, decode = 0;  // bytes: held in memory, read by a decode step
};

struct Placement {
  bool fits = false;    // false: no runs fit every Mac's memory (lo is then empty)
  std::vector<int> lo;  // member i holds layers [lo[i], lo[i + 1]); lo has members + 1 entries, the last the layers
  double ms = 0;        // a decode step's weight reads: every member's share over its bandwidth, network excluded
  std::vector<double> held;  // bytes each member holds
};

// layers: each trunk layer's cost; head: the head's own tensors; min_head: layers the head must hold (the n-gram
// block's: Flash-Next's layers 1 and 2); members in the order of the ring, the head first. Every member holds at least
// one layer.
Placement place(const std::vector<LayerCost>& layers, const LayerCost& head, int min_head,
                const std::vector<Member>& members);

// The ms a decode step's weights take on one Mac that held them all (its bandwidth), for comparing a split with it
double whole_ms(const std::vector<LayerCost>& layers, const LayerCost& head, double gbs);

// The Flash-Next model in model_dir over `members` (budgets in mem): each layer counts its weights and its share of
// what the conversations need (two slots' states, two 32K-token conversations' caches), each Mac's budget less its
// prefill buffers for `prefill_chunk` rows (what a stage checks before it loads its share); the head holds the n-gram
// block's layers. From the headers alone; call it before the process's split is set (Qwen4Exp::set_split), which
// would plan only a share.
Placement plan_split(const std::string& model_dir, std::vector<Member> members, int prefill_chunk);

}  // namespace ni::cluster
