// kapitsa cluster (also: kapitsa macs): the Macs on this network that run Kapitsa, found over Bonjour and read from
// each server's GET /cluster (src/cluster/): this Mac first, then the others, with the model each serves, how it runs
// it, its recent speed and the link to it.
#include "cluster/discovery.h"
#include "server/cli.h"
#include "server/ui.h"
#include "util/term.h"

#include <algorithm>
#include <csignal>
#include <set>
#include <thread>
#include <tuple>

namespace ni {

namespace {

const char* const kHelp =
    "kapitsa cluster [--json] [--watch] [--peer HOST[:PORT]]\n\n"
    "The Macs on this network that run Kapitsa, found in about 2 seconds: this Mac first, then the others, each with\n"
    "its chip and memory, the model it serves, how it runs it (in memory, streaming experts from the SSD, or as a proxy\n"
    "to another Mac), its recent decode speed, and the link to it with the round trip.\n\n"
    "  --json    the same as JSON: this Mac, and each server as its GET /cluster describes it, with the link\n"
    "  --watch   keep it up to date until Ctrl+C\n"
    "  --peer    another Mac's server by address, where the network blocks Bonjour between Macs (office Wi-Fi);\n"
    "            also KAPITSA_PEERS (comma-separated) and ~/.cache/kapitsa/peers (one per line)\n\n"
    "Every kapitsa serve is visible to the other Macs on its network, with no password (--private hides it). A Mac that\n"
    "would not hold a model whole sends its requests to a Mac listed here that runs it whole (--local runs it on the Mac\n"
    "itself). kapitsa macs is the same command.\n";

volatile sig_atomic_t g_stop = 0;
void on_stop(int) { g_stop = 1; }

// This Mac's servers first (by port), then the others by name: the table's order
std::vector<cluster::MacInfo> ordered(std::vector<cluster::MacInfo> v) {
  std::stable_sort(v.begin(), v.end(), [](const cluster::MacInfo& a, const cluster::MacInfo& b) {
    return std::make_tuple(!a.this_mac(), a.this_mac() ? std::string() : a.name, a.port) <
           std::make_tuple(!b.this_mac(), b.this_mac() ? std::string() : b.name, b.port);
  });
  return v;
}

std::string as_json(const cluster::MacInfo& here, const std::vector<cluster::MacInfo>& peers, const std::string& error) {
  ojson servers = ojson::array();
  for (const auto& p : ordered(peers)) servers.push_back(cluster::to_json(p));
  ojson j{{"this_mac", {{"name", here.name}, {"chip", here.chip}, {"ram_gb", here.ram_gb}}}, {"servers", servers}};
  if (!error.empty()) j["error"] = error;
  return j.dump();
}

// One look: the title, the table, and what to do when no other Mac is there
std::vector<std::string> screen(const cluster::MacInfo& here, const std::vector<cluster::MacInfo>& peers,
                                const std::string& error, int width) {
  using namespace term;
  std::set<std::string> others;
  for (const auto& p : peers)
    if (!p.this_mac()) others.insert(p.name);
  std::vector<std::string> o = {"", "  " + bold("Macs running Kapitsa on this network") +
                                        dim(others.empty() ? "" : "  " + std::to_string(others.size() + 1) + " Macs"), ""};
  for (const auto& l : cluster::format_table(here, peers, width)) o.push_back(l);
  o.push_back("");
  auto say = [&](const std::string& s, Style st) {
    const auto ls = wrap(s, std::max(30, width - 4));
    for (size_t i = 0; i < ls.size(); ++i) o.push_back("  " + st(std::string(i ? " " : kArrow) + " " + ls[i]));
  };
  if (!error.empty()) say("Bonjour: " + error + ".", warn);
  if (others.empty()) {
    say("No other Mac on this network runs Kapitsa. On another Mac: kapitsa serve, on the same Wi-Fi or Ethernet as "
        "this one (every server is visible unless started with --private).", dim);
    say("macOS asks once whether the terminal app may find devices on the local network: if it was refused, allow it in "
        "System Settings > Privacy & Security > Local Network, on each Mac.", dim);
  } else {
    say("A Mac that would not hold a model whole sends its requests to one that runs it whole: kapitsa serve --model "
        "NAME on it (--local keeps the model there).", dim);
  }
  o.push_back("");
  return o;
}

}  // namespace

int run_cluster(int argc, char** argv, int first) {
  bool json = false, watch = false;
  for (int i = first; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      if (term::fancy()) ui::help(kHelp);
      else fputs(kHelp, stdout);
      return 0;
    }
    if (a == "--json") json = true;
    else if (a == "--watch" || a == "-w") watch = true;
    else if (a == "--peer" && i + 1 < argc) cluster::add_peer(argv[++i]);
    else usage_error("unknown option " + a, "kapitsa cluster --help");
  }
  const bool fancy = term::fancy() && !json;
  const cluster::MacInfo here = cluster::this_mac();
  cluster::Browser browser;
  {
    std::unique_ptr<term::Spinner> sp;
    if (fancy) sp = std::make_unique<term::Spinner>("Looking for Macs running Kapitsa on this network");
    browser.wait(2);
    if (sp) sp->done();
  }
  auto look = [&] { return screen(here, browser.peers(), browser.error(), std::max(40, term::columns() - 1)); };
  if (!watch) {
    if (json) printf("%s\n", as_json(here, browser.peers(), browser.error()).c_str());
    else for (const auto& l : look()) printf("%s\n", l.c_str());
    return 0;
  }
  signal(SIGINT, on_stop);
  signal(SIGTERM, on_stop);
  // in a terminal the table redraws in place (and stays on screen after Ctrl+C); elsewhere a new one every 2 s
  const bool live = fancy && term::live();
  uint64_t seen = ~uint64_t(0);
  auto last = std::chrono::steady_clock::now() - std::chrono::hours(1);
  while (!g_stop) {
    const auto now = std::chrono::steady_clock::now();
    if ((live && browser.changes() != seen) || now - last >= std::chrono::seconds(2)) {
      seen = browser.changes(), last = now;
      if (live) term::status(look());
      else if (json) printf("%s\n", as_json(here, browser.peers(), browser.error()).c_str());
      else for (const auto& l : look()) printf("%s\n", l.c_str());
      fflush(stdout);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  if (live) term::keep_status();
  return 0;
}

}  // namespace ni
