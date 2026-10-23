// Finding the Macs on this network that run Kapitsa: Bonjour (DNS-SD through the system's dns_sd.h, in libSystem) for
// the servers, each server's GET /cluster for what it runs, GET /health for the round trip, and SystemConfiguration for
// the kind of link to it. macOS asks once whether the app that runs Kapitsa (the terminal) may use the local network
// (System Settings > Privacy & Security > Local Network); refused, Bonjour answers kDNSServiceErr_PolicyDenied, which
// error() reports.
#pragma once
#include "cluster/cluster.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ni::cluster {

// This Mac as a MacInfo starts: its computer name, chip, RAM, and link "this Mac"
MacInfo this_mac();
// A new server id: 16 random hex digits
std::string new_id();
// This Mac's IPv4 address on its network (the default route's interface, else a Wi-Fi, Ethernet or Thunderbolt one),
// "" with no network
std::string lan_address();
// The link from this Mac to `ipv4`, by the local interface the system routes it through: "this Mac" (one of its own
// addresses), "Wi-Fi", "Ethernet", "Thunderbolt" (the Thunderbolt Bridge or a Thunderbolt port) or "network"
std::string link_to(const std::string& ipv4);
// What a Bonjour error means, for a message
std::string bonjour_error(int code);
// Kapitsa servers named by address, for a network that does not carry Bonjour between Macs (many office Wi-Fi networks
// do not): "host[:port]" (port 8000 unless given) from KAPITSA_PEERS (comma-separated) and the lines of
// <Kapitsa's home>/peers (# starts a comment). Every Browser reads them as it reads the servers Bonjour finds.
std::vector<std::string> named_peers();
// Adds one to KAPITSA_PEERS for this process and the ones it starts (--peer)
void add_peer(const std::string& addr);

// Registers a server as `type` on every interface while it lives, under this Mac's computer name (with a number added
// on a conflict: two servers on one Mac). `info` is read at once and every few seconds after, on the advertiser's own
// thread; the TXT record (encode_txt) follows what it returns.
class Advertiser {
 public:
  Advertiser(int port, std::function<MacInfo()> info, const std::string& type = kServiceType);
  ~Advertiser();
  Advertiser(const Advertiser&) = delete;
  Advertiser& operator=(const Advertiser&) = delete;
  // Why the server is not visible: "" while it is, or until Bonjour has answered
  std::string error() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// Browses for `type` servers on threads of its own: resolves each, reads its GET /cluster, times its GET /health (the
// median of three) and works out the link; reads each again every few seconds; forgets those that leave. Thread-safe.
// The destructor does not wait for a read under way (its threads end on their own).
class Browser {
 public:
  explicit Browser(const std::string& type = kServiceType);
  ~Browser();
  Browser(const Browser&) = delete;
  Browser& operator=(const Browser&) = delete;
  // The servers found: those that answered (their GET /cluster, with host, port, rtt_ms and link) and those that do not
  // answer (what their TXT record says, rtt_ms -1)
  std::vector<MacInfo> peers() const;
  // Waits until `seconds` after the browser started, then up to a second more for servers found and not read yet
  void wait(double seconds) const;
  // Counts the changes to peers() (to know when to draw them again)
  uint64_t changes() const;
  // Why browsing fails ("" while it works)
  std::string error() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace ni::cluster
