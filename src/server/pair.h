// Several Macs in `kapitsa serve` (docs/usage.md, "Several Macs"): Macs that would each stream Qwen3.8-Flash-Next's
// experts from the SSD pair on their own. The server that started first becomes the head of a split (src/cluster/
// pipe.h); the others hold the rest of the layers as its stages and send their own requests to the head (route.h). They
// pair only when they hold the model in memory between them, over round trips of at most cluster::kPairRttMs, and part
// again when the split runs slower than the head did alone. A Mac that comes later joins the split: the head places
// the layers again over every Mac when a token's pass would be a tenth faster with it, else keeps it as a spare.
#pragma once
#include "cluster/discovery.h"
#include "cluster/pipe.h"
#include "engine/engine.h"
#include "hub/hub.h"

#include <functional>
#include <memory>
#include <string>

namespace ni {

class ModelManager;

// Whether `kapitsa serve` pairs for `ref`: a Qwen3.8-Flash-Next pack that this Mac would stream (runs_whole_here
// false). NI_CLUSTER_PAIR=1 pairs any Flash-Next folder, whatever this Mac holds, and waits 3 s instead of 60 before
// pairing again after a stage was lost, 5 s instead of 30 before a stage whose head went away runs alone, and gives a
// stage that joins a split layers whenever the Macs hold the model with it (tests: several servers on one Mac).
bool pairs_for(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget);

// Whether `group` (cluster::join_group's, the budgets they give) and this Mac (its budget) hold the model of
// `model_dir` in memory between them
bool group_fits(const std::string& model_dir, const std::vector<cluster::MacInfo>& group, size_t budget,
                int prefill_chunk);

// The least RAM (of the sizes Macs come in, GB) of a Mac that would pair with this one for `model_dir`, its budget
// taken as the same share of its RAM as this Mac's; 0 when none would
int pair_ram_gb(const std::string& model_dir, size_t budget, int prefill_chunk);

// A server that is the stage of another's split (main.cpp's serve_routed): what its stage loads, and whose it is
struct PairStage {
  std::string model_dir, head_id;
  double mem = 0;  // its GPU budget (bytes)
  double gbs = 0;  // its memory bandwidth (GB/s, 0 unknown)
  int chunk = 0;   // the most rows a pass may bring
};

// The head's side, on a thread of its own while the server runs, every 2 s. Alone: the stages paired with it make the
// split once they and this Mac hold the model in memory (the model loads again with this Mac's share); a split of the
// model, or another server of it that started first, makes this one give way (give_way: it starts again as that one's
// stage) when they all hold it in memory. Split: when a stage is lost, or the split decodes slower than this Mac did
// alone, the model loads again whole (streamed); when a stage joins and a token's pass would be a tenth faster with
// it, the model loads again whole and then splits over every stage (those given other layers start again first).
struct PairHead {
  std::shared_ptr<ModelManager> mgr;
  std::shared_ptr<cluster::Browser> browser;
  std::function<cluster::MacInfo()> me;  // this server as the other Macs see it
  cluster::HeadInfo head;                // this Mac for the placement
  std::function<void(const std::string& line, bool warning)> say;
  std::function<void()> give_way;
  std::string not_with;  // a head this server left (alone_from): not its stage again for 10 minutes
};
void watch_pairs(PairHead h);
// Whether watch_pairs has the model split now
bool split_now();

// The server's command line, kept for restart_server
void keep_args(int argc, char** argv);
// Starts this server again in place, with the same command line: as the stage of a server that started first, or,
// with `alone_from` (the id of a head that went away), on its own and never again that head's stage
[[noreturn]] void restart_server(const std::string& alone_from = "");
// NI_CLUSTER_ALONE, read and cleared: the head this server is not to pair with ("" none)
std::string alone_from();

}  // namespace ni
