// Kapitsa on several Macs: what a Mac running `kapitsa serve` tells the others on its network (MacInfo: in short in
// its Bonjour TXT record, in full at GET /cluster), and what is decided from it: where a Mac that would not hold a
// model whole sends its requests (pick_server), and how `kapitsa cluster` shows the Macs. No network here
// (discovery.h finds the Macs), so tools/cluster_test checks all of it offline.
#pragma once
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ni::cluster {

// What MacInfo, the TXT record and GET /cluster mean; a server of another protocol is left alone
constexpr int kProto = 1;
// The Bonjour service type every `kapitsa serve` registers (unless --private)
constexpr const char* kServiceType = "_kapitsa._tcp";

// One Kapitsa server on the network, as it describes itself, and (rtt_ms, link) as the Mac that found it measured it
struct MacInfo {
  int proto = kProto;
  std::string id;          // random hex, new for each server process
  std::string name;        // the Mac's computer name: "Studio"
  std::string chip;        // "Apple M3 Ultra"
  double ram_gb = 0;
  double budget_gb = 0;    // the GPU memory Kapitsa may hold there
  double free_gb = 0;      // of the budget, what its model does not use
  std::string model;       // "qwen3.8-flash-next:4bit" (catalog name:variant), else the model's id; "" none
  bool whole = false;      // the model runs fully in memory: no routed experts streamed from the SSD, not a proxy
  double tok_s = 0;        // its recent decode rate, 0 when none yet
  std::string version;     // Kapitsa's
  int port = 0;
  std::string host;        // where to reach it (IPv4): its address, 127.0.0.1 for a server on this Mac
  std::string routes_to;   // a server that sends its requests to another Mac: that Mac's name ("" otherwise)
  double since = 0;        // when the server started (Unix seconds): of two that would pair, the later is the stage
  // a server paired as the stage of another Mac's split (pair_with): its stage's port and that server's id
  int stage_port = 0;
  std::string stage_of;
  // measured by the Mac that found it (Browser)
  double rtt_ms = 0;       // median round trip of GET /health; 0 not measured, -1 not answering
  std::string link;        // "this Mac", "Wi-Fi", "Ethernet", "Thunderbolt" or "network"
  // fields this protocol does not name (a newer Kapitsa's), kept through from_json and to_json
  nlohmann::ordered_json extra = nlohmann::ordered_json::object();

  bool this_mac() const { return link == "this Mac"; }
};

// The Bonjour TXT record: length-prefixed "key=value" strings with short keys (proto, id, model, whole, ram, tps);
// decode_txt leaves what is missing or malformed at its default
std::string encode_txt(const MacInfo& m);
MacInfo decode_txt(const void* txt, size_t len);

// GET /cluster's JSON, and back (missing or mistyped fields stay at their defaults)
nlohmann::ordered_json to_json(const MacInfo& m);
MacInfo from_json(const nlohmann::ordered_json& j);
// GET /cluster's body as another Mac sent it (unauthenticated): read as from_json does, or nothing when it is past
// kMaxClusterBody, nested deeper than kMaxClusterDepth (from_json and dump recurse once a level), or not JSON
constexpr size_t kMaxClusterBody = 64 << 10;
constexpr int kMaxClusterDepth = 8;
std::optional<MacInfo> parse_cluster(const std::string& body);

// "qwen3.8-flash-next" and "4bit" of "Qwen3.8-Flash-Next:4bit" (lower case; "" without a variant)
std::string model_base(const std::string& model);
std::string model_variant(const std::string& model);

// Where requests for `model` go from a Mac that would not hold it whole (runs_whole_here false; when true, nowhere): a
// server that answers and runs the same model whole, not one that sends its own requests on. The same variant first,
// else the most precise one (bf16, then 8-bit, 6-bit, ...); among those the lowest round trip, then the fastest.
std::optional<MacInfo> pick_server(const std::vector<MacInfo>& peers, const std::string& model, bool runs_whole_here);

// Two Macs that would each stream a model's experts from the SSD pair (docs/usage.md, "Several Macs"): one is the head
// of a split of the model (src/cluster/pipe.h), the other its stage, which sends its own requests to the head. Of two
// that could, the server that started later becomes the stage. A pass of the split crosses the network once there and
// back, so a pair is made only over a short round trip: kPairRttMs (a token streamed from the SSD costs tens of ms more
// than one read from memory; the hop costs about the round trip).
constexpr double kPairRttMs = 15;

// For a server about to start `model`, which it would stream: the server whose stage it would become. One that runs
// the same model and variant (a split reads the same files on every Mac) by itself and streamed (not whole, not a
// proxy, not a stage), answering within max_rtt_ms; the earliest started.
std::optional<MacInfo> pair_with(const std::vector<MacInfo>& peers, const std::string& model,
                                 double max_rtt_ms = kPairRttMs);

// For a head (head_id) that runs `model` streamed: the stages paired with it, answering within max_rtt_ms, nearest first
std::vector<MacInfo> stages_for(const std::vector<MacInfo>& peers, const std::string& model, const std::string& head_id,
                                double max_rtt_ms = kPairRttMs);

// Whether `me`, a server that runs its model by itself and streamed, gives way to another that does too and started
// first (the same second: the smaller id), within max_rtt_ms: it then starts again as that one's stage
bool gives_way(const MacInfo& me, const std::vector<MacInfo>& peers, double max_rtt_ms = kPairRttMs);

// "Studio (M3 Ultra, 512 GB)"
std::string describe(const MacInfo& m);
// "Wi-Fi 4 ms", "this Mac", "Ethernet <1 ms", "not answering"
std::string link_text(const MacInfo& m);
// How it runs its model: "in memory", "streams from SSD", "proxy", "stage" ("" without a model)
std::string how_it_runs(const MacInfo& m);

// `kapitsa cluster`'s table (term.h styles, plain when not a terminal), at most `width` columns: a header, then the
// servers on this Mac (or, with none, `here` as not serving), then the others by name; one line each
std::vector<std::string> format_table(const MacInfo& here, const std::vector<MacInfo>& servers, int width);

}  // namespace ni::cluster
