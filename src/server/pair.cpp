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
#include <map>
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

bool group_fits(const std::string& model_dir, const std::vector<cluster::MacInfo>& group, size_t budget,
                int prefill_chunk) {
  try {
    std::vector<cluster::Member> m;
    for (const auto& g : group) m.push_back({g.name, g.budget_gb * 1e9, 0});
    m.push_back({"this Mac", double(budget), 0});
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
    using clock = std::chrono::steady_clock;
    const EngineOptions alone_opt = h.mgr->options();
    const std::string model = cluster::model_base(h.me().model), dir = alone_opt.model_dir;
    std::shared_ptr<cluster::Link> link;  // the split, while paired
    std::vector<cluster::MacInfo> stages;  // its stages, as they were found
    std::set<std::string> refused;        // stages that would not pair, or ran the split slower than this Mac alone
    std::set<std::string> spare;          // stages that joined the split and would not make a token faster
    std::map<std::string, int> fails;     // failed attempts to pair, by stage (refused at the third)
    double alone = 0, last = ui::recent_tok_s();  // the decode rate alone; the latest reply's
    int slow = 0;                                 // replies in a row slower on the split than alone
    auto next = clock::now();                     // no pairing before (after a stage was lost)
    size_t want = 0;                              // stages to wait for (some start again to load their layers)...
    auto want_until = clock::now();               // ...until then
    std::string waiting;                          // the stages said to wait for the others
    const auto not_until = clock::now() + std::chrono::minutes(10);  // (h.not_with)
    const auto placed = [&](const std::vector<cluster::Member>& m) {
      return cluster::plan_split(dir, m, h.head.chunk, false);
    };
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      const auto eng = h.mgr->current_engine();
      if (!eng || h.mgr->switching()) continue;
      const double rate = ui::recent_tok_s();
      const bool fresh = rate != last;  // a reply ended since the last look
      last = rate;
      const cluster::MacInfo me = h.me();
      const std::vector<cluster::MacInfo> peers = h.browser->peers();
      if (link) {
        const std::string why = eng->lost();
        if (why.empty() && fresh && alone > 0) slow = rate < 0.8 * alone ? slow + 1 : 0;
        // a stage that joined since: the split again over every stage when a token's pass would be a tenth faster
        bool regrow = false;
        if (why.empty() && slow < 2 && h.mgr->in_flight() == 0) {
          std::vector<cluster::MacInfo> extra;
          for (const auto& s : cluster::stages_for(peers, me.model, me.id)) {
            bool in = false;
            for (const auto& t : stages) in = in || t.id == s.id;
            if (!in && !refused.count(s.id) && !spare.count(s.id)) extra.push_back(s);
          }
          if (!extra.empty()) {
            std::vector<cluster::Member> now = {{h.head.name, h.head.mem, h.head.gbs}};
            for (const cluster::StageInfo& s : link->stages()) now.push_back({s.name, s.mem, s.gbs});
            std::vector<cluster::Member> all = now;
            for (const auto& s : extra) {
              cluster::StageInfo si;
              std::string qwhy;
              if (cluster::Link::query(s.host + ":" + std::to_string(s.stage_port), dir, h.head, &si, &qwhy))
                all.push_back({si.name, si.mem, si.gbs});
              else if (++fails[s.id] >= 3)
                refused.insert(s.id);
            }
            if (all.size() > now.size()) {
              const cluster::Placement pn = placed(now), pa = placed(all);
              // the ring's round trip, a hop more for each Mac more
              const double tn = pn.ms + link->rtt_ms(), ta = pa.ms + link->rtt_ms() / now.size() * all.size();
              char t[160];
              snprintf(t, sizeof t, "a token's pass over %zu Macs about %.0f ms, over %zu %.0f ms", all.size(), ta,
                       now.size(), tn);
              if (pa.fits && (ta < 0.9 * tn || env_on("NI_CLUSTER_PAIR"))) {
                h.say(names_of(extra) + " joined the split (" + t + "): " + model + " loads again here, then its layers "
                      "are placed over every Mac.",
                      false);
                regrow = true;
              } else {
                h.say(names_of(extra) + " joined the split as a spare (" + t + "): it sends its requests here, and "
                      "takes layers if a Mac leaves.",
                      false);
                for (const auto& s : extra) spare.insert(s.id);
              }
            }
          }
        }
        if (why.empty() && slow < 2 && !regrow) continue;
        if (!why.empty()) {
          h.say(names_of(stages) + " left the split: " + why + ". " + model + " runs on this Mac alone again.", true);
          next = clock::now() + std::chrono::seconds(env_on("NI_CLUSTER_PAIR") ? 3 : 60);
        } else if (!regrow) {
          char r[128];
          snprintf(r, sizeof r, " (%.1f tok/s, %.1f alone)", rate, alone);
          h.say("Paired with " + names_of(stages) + ", " + model + " decoded slower than on this Mac alone" + r +
                    ": it runs here alone again.",
                true);
          for (const auto& s : stages) refused.insert(s.id);
        } else {  // every stage, the spares too, once they are all back (those given other layers start again)
          size_t n = 0;
          for (const auto& s : cluster::stages_for(peers, me.model, me.id)) n += !refused.count(s.id);
          want = n, want_until = clock::now() + std::chrono::seconds(60), next = clock::now();
        }
        std::string err;
        h.mgr->reload(alone_opt, [] { Qwen4Exp::set_split({}); }, nullptr, &err);
        if (!err.empty()) h.say("Loading " + model + " again failed: " + err, true);
        link.reset();
        stages.clear();
        spare.clear();
        g_split = false;
        slow = 0;
        last = ui::recent_tok_s();
        continue;
      }
      if (fresh && rate > 0) alone = alone > 0 ? 0.7 * alone + 0.3 * rate : rate;
      // the stages paired with this server: the split, once they and this Mac hold the model in memory
      std::vector<cluster::MacInfo> found;
      for (const auto& s : cluster::stages_for(peers, me.model, me.id))
        if (!refused.count(s.id)) found.push_back(s);
      const auto now = clock::now();
      if (!found.empty() && now >= next && (found.size() >= want || now >= want_until)) {
        std::vector<cluster::Member> m = {{h.head.name, h.head.mem, h.head.gbs}};
        for (const auto& s : found) m.push_back({s.name, s.budget_gb * 1e9, 0});
        if (!placed(m).fits) {
          if (waiting != names_of(found))
            h.say(names_of(found) + (found.size() > 1 ? " wait" : " waits") + " to pair: with this Mac they do not "
                  "hold " + model + " in memory yet, and another Mac that runs it would make it.",
                  false);
          waiting = names_of(found);
          continue;
        }
        waiting.clear();
        std::vector<std::string> addrs;
        for (const auto& s : found) addrs.push_back(s.host + ":" + std::to_string(s.stage_port));
        h.say("Pairing with " + names_of(found) + ", which " + (found.size() > 1 ? "run " : "runs ") + model +
                  " too: placing its layers...",
              false);
        std::string why;
        auto l = cluster::Link::connect(addrs, dir, h.head, &why);
        if (!l && why.find(cluster::kStageStartsAgain) != std::string::npos) {
          // a stage loads its new layers in a fresh process (a new id): pairing again once every stage is back
          h.say(why + ": pairing again once it is back.", false);
          size_t n = 0;
          for (const auto& s : found) n += !refused.count(s.id);
          want = n, want_until = now + std::chrono::seconds(60), next = now + std::chrono::seconds(3);
          continue;
        }
        if (!l) {
          h.say("Could not pair with " + names_of(found) + ": " + why, true);
          for (const auto& s : found)
            if (++fails[s.id] >= 3) refused.insert(s.id);
          next = now + std::chrono::seconds(10);
          continue;
        }
        EngineOptions o = alone_opt;
        o.prefill_chunk_cap = int(l->rows());  // a pass wider than a stage's chunk breaks the link
        if (o.prefill_chunk > l->rows()) o.prefill_chunk = int(l->rows());
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
        want = 0;
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
      if (!found.empty()) continue;  // (stages that wait for this one)
      // a split of the model, or another server of it that runs it alone and started first: this one becomes its
      // stage when they all hold the model in memory
      std::vector<cluster::MacInfo> others;
      const bool shun = !h.not_with.empty() && now < not_until;
      for (const auto& p : peers)
        if (p.id != me.id && !(shun && p.id == h.not_with)) others.push_back(p);
      if (h.mgr->in_flight() > 0) continue;
      const std::vector<cluster::MacInfo> group = cluster::join_group(others, me);
      if (group.empty() || !group_fits(dir, group, size_t(h.head.mem), h.head.chunk)) continue;
      const cluster::MacInfo& to = group[0];
      h.say(cluster::splits(others, to)
                ? to.name + " runs " + model + " split across Macs: this server starts again as a stage of its split."
                : to.name + " runs " + model + " too and started first: this server starts again as its stage, and " +
                      (group.size() > 1 ? "the Macs" : "the two") + " hold the model in memory between them.",
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
