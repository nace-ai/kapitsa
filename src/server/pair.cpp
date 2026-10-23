// Several Macs in `kapitsa serve`: two Macs that would each stream Flash-Next pair on their own (pair.h).
#include "server/pair.h"

#include "cluster/place.h"
#include "engine/doctor.h"
#include "metal/gpu.h"
#include "model/qwen4_exp.h"
#include "server/cli.h"
#include "server/model_manager.h"
#include "server/route.h"
#include "server/ui.h"
#include "update/update.h"
#include "util/log.h"
#include "util/term.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <set>
#include <thread>

namespace ni {

namespace {

std::vector<std::string> g_args;  // the server's command line (restart_server)
std::atomic<bool> g_split{false};  // watch_pairs has split the model

std::string names_of(const std::vector<cluster::MacInfo>& macs) {
  std::string s;
  for (size_t i = 0; i < macs.size(); ++i) s += (i == 0 ? "" : i + 1 == macs.size() ? " and " : ", ") + macs[i].name;
  return s;
}

}  // namespace

bool pairs_for(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget) {
  try {
    const std::string dir = hub::plan_dir(ref);
    if (dir.empty() || !Qwen4Exp::is_model(dir)) return false;
    return env_on("NI_CLUSTER_PAIR") || !runs_whole_here(ref, opt, budget);
  } catch (const std::exception&) {
    return false;
  }
}

bool pair_fits(const std::string& model_dir, const cluster::MacInfo& head, size_t budget, int prefill_chunk) {
  try {
    const std::vector<cluster::Member> m = {{head.name, head.budget_gb * 1e9, 0}, {"this Mac", double(budget), 0}};
    return cluster::plan_split(model_dir, m, prefill_chunk, false).fits;
  } catch (const std::exception&) {
    return false;
  }
}

int pair_ram_gb(const std::string& model_dir, size_t budget, int prefill_chunk) {
  const double share = double(budget) / (ram_gb() * 1e9);
  try {
    for (int gb : {16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512}) {
      const std::vector<cluster::Member> m = {{"this Mac", double(budget), 0}, {"another", gb * 1e9 * share, 0}};
      if (cluster::plan_split(model_dir, m, prefill_chunk, false).fits) return gb;
    }
  } catch (const std::exception&) {
  }
  return 0;
}

void watch_pairs(PairHead h) {
  std::thread([h = std::move(h)] {
    const EngineOptions alone_opt = h.mgr->options();
    const std::string model = cluster::model_base(h.me().model);
    std::shared_ptr<cluster::Link> link;  // the split, while paired
    std::vector<cluster::MacInfo> stages;  // its stages, as they were found
    std::set<std::string> refused;        // stages that would not pair, or ran the split slower than this Mac alone
    double alone = 0, last = ui::recent_tok_s();  // the decode rate alone; the latest reply's
    int slow = 0;                                 // replies in a row slower on the split than alone
    auto next = std::chrono::steady_clock::now();  // no pairing before (after a stage was lost)
    const auto not_until = std::chrono::steady_clock::now() + std::chrono::minutes(10);  // (h.not_with)
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      const auto eng = h.mgr->current_engine();
      if (!eng || h.mgr->switching()) continue;
      const double rate = ui::recent_tok_s();
      const bool fresh = rate != last;  // a reply ended since the last look
      last = rate;
      if (link) {
        const std::string why = eng->lost();
        if (why.empty() && fresh && alone > 0) slow = rate < 0.8 * alone ? slow + 1 : 0;
        if (why.empty() && slow < 2) continue;
        if (!why.empty()) {
          h.say(names_of(stages) + " left the split: " + why + ". " + model + " runs on this Mac alone again.", true);
          next = std::chrono::steady_clock::now() + std::chrono::seconds(env_on("NI_CLUSTER_PAIR") ? 3 : 60);
        } else {
          char r[128];
          snprintf(r, sizeof r, " (%.1f tok/s, %.1f alone)", rate, alone);
          h.say("Paired with " + names_of(stages) + ", " + model + " decoded slower than on this Mac alone" + r +
                    ": it runs here alone again.",
                true);
          for (const auto& s : stages) refused.insert(s.id);
        }
        std::string err;
        h.mgr->reload(alone_opt, [] { Qwen4Exp::set_split({}); }, nullptr, &err);
        if (!err.empty()) h.say("Loading " + model + " again failed: " + err, true);
        link.reset();
        stages.clear();
        g_split = false;
        slow = 0;
        last = ui::recent_tok_s();
        continue;
      }
      if (fresh && rate > 0) alone = alone > 0 ? 0.7 * alone + 0.3 * rate : rate;
      const cluster::MacInfo me = h.me();
      const std::vector<cluster::MacInfo> peers = h.browser->peers();
      // a stage paired with this server: the split
      std::vector<cluster::MacInfo> found;
      for (const auto& s : cluster::stages_for(peers, me.model, me.id))
        if (!refused.count(s.id)) found.push_back(s);
      if (!found.empty() && std::chrono::steady_clock::now() >= next) {
        std::vector<std::string> addrs;
        for (const auto& s : found) addrs.push_back(s.host + ":" + std::to_string(s.stage_port));
        h.say("Pairing with " + names_of(found) + ", which runs " + model + " too: placing its layers...", false);
        std::string why;
        auto l = cluster::Link::connect(addrs, alone_opt.model_dir, h.head, &why);
        if (!l) {
          h.say("Could not pair with " + names_of(found) + ": " + why, true);
          for (const auto& s : found) refused.insert(s.id);
          continue;
        }
        EngineOptions o = alone_opt;
        if (o.prefill_chunk > l->rows() || (o.prefill_chunk <= 0 && l->rows() < prefill_chunk_room(2048)))
          o.prefill_chunk = l->rows();
        std::string err;
        if (!h.mgr->reload(o, [l] { Qwen4Exp::set_split({0, l->first(), l}); }, [] { Qwen4Exp::set_split({}); }, &err)) {
          h.say("Could not pair with " + names_of(found) + ": " + err, true);
          for (const auto& s : found) refused.insert(s.id);
          continue;
        }
        link = l;
        stages = found;
        g_split = true;
        slow = 0;
        last = ui::recent_tok_s();
        std::string where;
        for (const cluster::StageInfo& s : l->stages())
          where += ", " + std::to_string(s.lo + 1) + "-" + std::to_string(s.hi) + " on " + s.name;
        char rtt[64];
        snprintf(rtt, sizeof rtt, "; a pass's round trip %.1f ms", l->rtt_ms());
        h.say("Paired with " + names_of(found) + ": " + model + " runs in memory across the Macs (layers 1-" +
                  std::to_string(l->first()) + " here" + where + rtt + ").",
              false);
        continue;
      }
      // another server of the model that runs it alone and started first: this one becomes its stage
      std::vector<cluster::MacInfo> others;
      const bool shun = !h.not_with.empty() && std::chrono::steady_clock::now() < not_until;
      for (const auto& p : peers)
        if (p.id != me.id && !(shun && p.id == h.not_with)) others.push_back(p);
      if (h.mgr->in_flight() > 0 || !cluster::gives_way(me, others)) continue;
      const auto to = cluster::pair_with(others, me.model);
      if (!to || !pair_fits(alone_opt.model_dir, *to, size_t(h.head.mem), alone_opt.prefill_chunk > 0
                                                                               ? alone_opt.prefill_chunk
                                                                               : 2048))
        continue;
      h.say(to->name + " runs " + model + " too and started first: this server starts again as its stage, and the two "
            "hold the model in memory between them.",
            false);
      h.give_way();
    }
  }).detach();
}

bool split_now() { return g_split; }

void keep_args(int argc, char** argv) { g_args.assign(argv, argv + argc); }

void restart_server(const std::string& alone_from) {
  if (alone_from.empty()) unsetenv("NI_CLUSTER_ALONE");
  else setenv("NI_CLUSTER_ALONE", alone_from.c_str(), 1);
  logi("cluster: starting the server again%s", alone_from.empty() ? "" : " alone");
  term::end_status();
  fflush(stdout);
  fflush(stderr);
  // nothing open goes on to the new image but the terminal: the listening sockets are bound again there
  for (int fd = 3, n = getdtablesize(); fd < n; ++fd) fcntl(fd, F_SETFD, FD_CLOEXEC);
  std::vector<char*> av;
  for (auto& a : g_args) av.push_back(a.data());
  av.push_back(nullptr);
  const std::string self = update::self_path();
  execv(self.c_str(), av.data());
  fatal("could not start the server again (%s): %s", self.c_str(), strerror(errno));
}

std::string alone_from() {
  const char* v = getenv("NI_CLUSTER_ALONE");
  const std::string s = v ? v : "";
  unsetenv("NI_CLUSTER_ALONE");
  return s;
}

}  // namespace ni
