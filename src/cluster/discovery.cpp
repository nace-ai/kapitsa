// Finding the Macs on this network that run Kapitsa (discovery.h).
//
// The browser runs two threads. The event thread owns the DNS-SD operations, all on one shared connection to
// mDNSResponder (kDNSServiceFlagsShareConnection): the browse, a resolve per server found (its host name, port and TXT
// record), then a GetAddrInfo per server for its IPv4 addresses, kept open so that a new address shows up. Callbacks
// only update the table and queue work, which the thread does once DNSServiceProcessResult returns (operations start
// and end outside the callbacks). The reader thread reads every server with addresses over HTTP, all at once, every few
// seconds. Bonjour's own goodbyes drop a server that stops; one that vanishes without one (its Mac asleep or gone) stops
// answering, and shows as such until Bonjour forgets it.
#include "cluster/discovery.h"

#include "util/log.h"

#include <httplib/httplib.h>

#include <SystemConfiguration/SystemConfiguration.h>
#include <arpa/inet.h>
#include <dns_sd.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

namespace ni::cluster {

using ojson = nlohmann::ordered_json;

namespace {

constexpr double kReadEvery = 4000;    // ms between two reads of a server
constexpr double kReadLimit = 5000;    // ms a read may take: a server slower than that counts as not answering
constexpr double kResolveWait = 5000;  // ms a resolve may take before it is tried again
constexpr double kTxtEvery = 5000;     // ms between two looks at what the advertised TXT record should say

std::string cf(CFStringRef s) {
  char b[512];
  return s && CFStringGetCString(s, b, sizeof b, kCFStringEncodingUTF8) ? b : "";
}

std::string sysctl_string(const char* name) {
  size_t n = 0;
  if (sysctlbyname(name, nullptr, &n, nullptr, 0) != 0 || n == 0) return "";
  std::string s(n, '\0');
  if (sysctlbyname(name, s.data(), &n, nullptr, 0) != 0) return "";
  s.resize(strnlen(s.c_str(), n));
  return s;
}

// This Mac's IPv4 addresses, each with its interface
std::map<std::string, std::string> own_addresses() {
  std::map<std::string, std::string> out;
  ifaddrs* ifs = nullptr;
  if (getifaddrs(&ifs) != 0) return out;
  for (ifaddrs* i = ifs; i; i = i->ifa_next) {
    if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET) continue;
    char b[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(i->ifa_addr)->sin_addr, b, sizeof b)) out[b] = i->ifa_name;
  }
  freeifaddrs(ifs);
  return out;
}

// The local address the system takes to reach `ipv4`: a UDP socket connected to it (which sends nothing)
std::string source_for(const std::string& ipv4) {
  sockaddr_in to{};
  to.sin_len = sizeof to;
  to.sin_family = AF_INET;
  to.sin_port = htons(9);
  if (inet_pton(AF_INET, ipv4.c_str(), &to.sin_addr) != 1) return "";
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) return "";
  std::string out;
  sockaddr_in from{};
  socklen_t n = sizeof from;
  char b[INET_ADDRSTRLEN];
  if (connect(fd, reinterpret_cast<sockaddr*>(&to), sizeof to) == 0 &&
      getsockname(fd, reinterpret_cast<sockaddr*>(&from), &n) == 0 && inet_ntop(AF_INET, &from.sin_addr, b, sizeof b))
    out = b;
  close(fd);
  return out;
}

// A network interface's kind by its BSD name ("en0"), as System Settings names it: the Thunderbolt Bridge and the
// Thunderbolt ports carry "Thunderbolt" in their names (the bridge's type has no public constant)
std::string interface_kind(const std::string& bsd) {
  if (bsd.rfind("lo", 0) == 0) return "this Mac";
  std::string kind = "network";
  CFArrayRef all = SCNetworkInterfaceCopyAll();
  if (!all) return kind;
  for (CFIndex k = 0; k < CFArrayGetCount(all); ++k) {
    auto i = static_cast<SCNetworkInterfaceRef>(const_cast<void*>(CFArrayGetValueAtIndex(all, k)));
    if (cf(SCNetworkInterfaceGetBSDName(i)) != bsd) continue;
    const CFStringRef type = SCNetworkInterfaceGetInterfaceType(i);
    if (cf(SCNetworkInterfaceGetLocalizedDisplayName(i)).find("Thunderbolt") != std::string::npos) kind = "Thunderbolt";
    else if (type && CFEqual(type, kSCNetworkInterfaceTypeIEEE80211)) kind = "Wi-Fi";
    else if (type && CFEqual(type, kSCNetworkInterfaceTypeEthernet)) kind = "Ethernet";
    break;
  }
  CFRelease(all);
  return kind;
}

// A pipe whose other end wakes the threads' polls when they should stop
struct Wake {
  int fd[2] = {-1, -1};
  Wake() {
    if (pipe(fd) != 0) fd[0] = fd[1] = -1;
  }
  ~Wake() {
    for (int f : fd)
      if (f >= 0) close(f);
  }
  void ring() {
    const char c = 1;
    if (fd[1] >= 0 && write(fd[1], &c, 1) < 0) {
    }
  }
  // up to `ms`: true when `other` is readable (and the pipe was not rung)
  bool wait(int other, int ms) const {
    pollfd p[2] = {{other, POLLIN, 0}, {fd[0], POLLIN, 0}};
    const int n = other >= 0 ? poll(p, 2, ms) : poll(p + 1, 1, ms);
    return n > 0 && other >= 0 && (p[0].revents & (POLLIN | POLLHUP | POLLERR)) && !(p[1].revents & POLLIN);
  }
};

// One server read over HTTP at the first of `addrs` that answers best: its GET /cluster, and the median of three
// GET /health round trips on the same connection. A server on this Mac is read at 127.0.0.1. What an unauthenticated
// peer sends is capped: /cluster's body at kMaxClusterBody (cluster.h's parse_cluster), each /health body at 4 KiB, and
// all of it at kReadLimit from the start (httplib's timeouts are per recv: a peer could drip it).
std::optional<MacInfo> read_server(const std::vector<std::string>& addrs, int port) {
  const double t0 = now_ms();
  const auto own = own_addresses();
  std::optional<MacInfo> best;
  for (const auto& a : addrs) {
    const bool here = own.count(a) > 0;
    for (const std::string& host : here ? std::vector<std::string>{"127.0.0.1", a} : std::vector<std::string>{a}) {
      httplib::Client c(host, port);
      c.set_connection_timeout(0, 800000);
      c.set_read_timeout(2, 0);
      c.set_write_timeout(2, 0);
      c.set_keep_alive(true);
      std::string body;
      const auto r = c.Get("/cluster", [&](const char* d, size_t n) {
        if (body.size() + n > kMaxClusterBody || now_ms() - t0 > kReadLimit) return false;
        body.append(d, n);
        return true;
      });
      if (!r || r->status != 200) continue;
      const std::optional<MacInfo> got = parse_cluster(body);
      if (!got || got->proto != kProto) break;
      MacInfo m = *got;
      std::vector<double> ms;
      for (int k = 0; k < 3; ++k) {
        const double t = now_ms();
        size_t got = 0;
        const auto h = c.Get("/health", [&](const char*, size_t n) {
          got += n;
          return got <= 4096 && now_ms() - t0 <= kReadLimit;
        });
        if (h && h->status == 200) ms.push_back(now_ms() - t);
      }
      if (ms.empty()) continue;
      std::sort(ms.begin(), ms.end());
      m.rtt_ms = std::max(0.01, ms[ms.size() / 2]);
      m.host = host;
      m.port = port;
      m.link = here ? "this Mac" : link_to(a);
      if (!best || m.rtt_ms < best->rtt_ms) best = m;
      break;
    }
  }
  return best;
}

}  // namespace

MacInfo this_mac() {
  MacInfo m;
  if (CFStringRef n = SCDynamicStoreCopyComputerName(nullptr, nullptr)) {
    m.name = cf(n);
    CFRelease(n);
  }
  if (m.name.empty()) {
    char h[256] = {};
    gethostname(h, sizeof h - 1);
    m.name = h;
  }
  m.chip = sysctl_string("machdep.cpu.brand_string");
  uint64_t mem = 0;
  size_t n = sizeof mem;
  sysctlbyname("hw.memsize", &mem, &n, nullptr, 0);
  m.ram_gb = std::round(double(mem) / 1073741824.0);
  m.link = "this Mac";
  return m;
}

std::string new_id() {
  uint8_t b[8];
  arc4random_buf(b, sizeof b);
  char s[17];
  for (int i = 0; i < 8; ++i) snprintf(s + 2 * i, 3, "%02x", b[i]);
  return s;
}

std::string lan_address() {
  const auto own = own_addresses();
  auto usable = [&](const std::string& a) {
    const auto it = own.find(a);
    return it != own.end() && it->second.rfind("lo", 0) != 0 && it->second.rfind("utun", 0) != 0;
  };
  if (const std::string via = source_for("192.0.2.1"); usable(via)) return via;  // TEST-NET-1: the default route
  std::string link_local;
  for (const auto& [a, ifn] : own)
    if (ifn.rfind("en", 0) == 0 || ifn.rfind("bridge", 0) == 0) {
      if (a.rfind("169.254.", 0) != 0) return a;
      link_local = a;
    }
  return link_local;
}

std::string link_to(const std::string& ipv4) {
  const auto own = own_addresses();
  if (own.count(ipv4)) return "this Mac";
  const auto it = own.find(source_for(ipv4));
  return it == own.end() ? "network" : interface_kind(it->second);
}

std::string bonjour_error(int code) {
  switch (code) {
    case kDNSServiceErr_PolicyDenied:
      return "macOS does not let the app that runs kapitsa use the local network (System Settings > Privacy & Security > "
             "Local Network)";
    case kDNSServiceErr_ServiceNotRunning: return "Bonjour (mDNSResponder) is not running";
    default: return "Bonjour error " + std::to_string(code);
  }
}

// ---------------------------------------------------------------- Advertiser

struct Advertiser::Impl {
  int port = 0;
  std::string type;
  std::function<MacInfo()> info;
  std::atomic<bool> stop{false};
  Wake wake;
  std::thread th;
  mutable std::mutex mu;
  std::string err;

  static void DNSSD_API registered(DNSServiceRef, DNSServiceFlags, DNSServiceErrorType e, const char* name, const char*,
                                   const char*, void* ctx) {
    auto* self = static_cast<Impl*>(ctx);
    std::lock_guard<std::mutex> lk(self->mu);
    self->err = e == kDNSServiceErr_NoError ? "" : bonjour_error(e);
    if (e == kDNSServiceErr_NoError) logi("cluster: visible on the network as \"%s\" (port %d)", name ? name : "", self->port);
    else logi("cluster: Bonjour registration failed, so the other Macs do not see this server: %s", self->err.c_str());
  }

  void run() {
    while (!stop) {
      std::string txt = encode_txt(info());
      DNSServiceRef ref = nullptr;
      const DNSServiceErrorType e =
          DNSServiceRegister(&ref, 0, kDNSServiceInterfaceIndexAny, nullptr, type.c_str(), nullptr, nullptr,
                             htons(uint16_t(port)), uint16_t(txt.size()), txt.data(), registered, this);
      if (e != kDNSServiceErr_NoError) {
        {
          std::lock_guard<std::mutex> lk(mu);
          err = bonjour_error(e);
        }
        wake.wait(-1, 5000);
        continue;
      }
      const int fd = DNSServiceRefSockFD(ref);
      double next = now_ms() + kTxtEvery;
      bool ok = true;
      while (!stop && ok) {
        if (wake.wait(fd, 1000) && !stop) ok = DNSServiceProcessResult(ref) == kDNSServiceErr_NoError;
        if (ok && !stop && now_ms() >= next) {
          next = now_ms() + kTxtEvery;
          const std::string t = encode_txt(info());
          if (t != txt && DNSServiceUpdateRecord(ref, nullptr, 0, uint16_t(t.size()), t.data(), 0) == kDNSServiceErr_NoError)
            txt = t;
        }
      }
      DNSServiceRefDeallocate(ref);  // deregisters: Bonjour says goodbye for it
      if (!stop) wake.wait(-1, 2000);  // the daemon went away: register again
    }
  }
};

Advertiser::Advertiser(int port, std::function<MacInfo()> info, const std::string& type) : impl_(std::make_shared<Impl>()) {
  impl_->port = port;
  impl_->type = type;
  impl_->info = std::move(info);
  impl_->th = std::thread([s = impl_] { s->run(); });
}

Advertiser::~Advertiser() {
  impl_->stop = true;
  impl_->wake.ring();
  if (impl_->th.joinable()) impl_->th.join();
}

std::string Advertiser::error() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->err;
}

// ---------------------------------------------------------------- Browser

struct Browser::Impl {
  std::string type;
  const double t0 = now_ms();
  std::atomic<bool> stop{false};
  Wake wake;

  struct Op {  // a DNS-SD operation on the shared connection, for one server
    Impl* impl = nullptr;
    std::string name;
    DNSServiceRef ref = nullptr;
    bool done = false;  // answered or no longer wanted: freed once the callbacks return
  };
  struct Entry {
    std::string domain;
    std::set<uint32_t> ifaces;  // the interfaces Bonjour sees it on
    bool resolved = false;
    std::string target;  // its host name ("Studio.local.")
    int port = 0;
    MacInfo txt;                  // what its TXT record says
    std::set<std::string> addrs;  // its IPv4 addresses
    Op *resolving = nullptr, *addressing = nullptr;
    double resolve_at = 0, retry_at = 0;
    MacInfo info;  // as read
    bool read = false, reading = false;
    int fails = 0;  // reads in a row that got no answer
    double read_at = 0, read_start = 0;
  };

  mutable std::mutex mu;
  mutable std::condition_variable cv;
  std::map<std::string, Entry> entries;  // by service name
  std::string err;
  uint64_t changes = 0;

  // the event thread's alone
  DNSServiceRef conn = nullptr;
  std::vector<std::unique_ptr<Op>> ops;
  std::vector<Op*> to_free;
  std::vector<std::string> to_resolve, to_address;

  void bump() {  // with mu held
    ++changes;
    cv.notify_all();
  }

  Op* new_op(const std::string& name) {
    ops.push_back(std::make_unique<Op>());
    Op* op = ops.back().get();
    op->impl = this;
    op->name = name;
    op->ref = conn;  // a copy of the shared connection, which the operation's call turns into its own
    return op;
  }
  void free_op(Op* op) {
    const auto it = std::find_if(ops.begin(), ops.end(), [&](const auto& o) { return o.get() == op; });
    if (it == ops.end()) return;
    if ((*it)->ref && (*it)->ref != conn) DNSServiceRefDeallocate((*it)->ref);
    ops.erase(it);
  }

  static void DNSSD_API browsed(DNSServiceRef, DNSServiceFlags flags, uint32_t iface, DNSServiceErrorType e,
                                const char* name, const char*, const char* domain, void* ctx) {
    Impl* self = static_cast<Op*>(ctx)->impl;
    std::lock_guard<std::mutex> lk(self->mu);
    if (e != kDNSServiceErr_NoError) {
      self->err = bonjour_error(e);
      return self->bump();
    }
    if (!name) return;
    if (flags & kDNSServiceFlagsAdd) {
      Entry& en = self->entries[name];
      en.domain = domain && *domain ? domain : "local.";
      en.ifaces.insert(iface);
      if (!en.resolved && !en.resolving) self->to_resolve.push_back(name);
      return;
    }
    const auto it = self->entries.find(name);
    if (it == self->entries.end()) return;
    it->second.ifaces.erase(iface);
    if (!it->second.ifaces.empty()) return;
    for (Op* op : {it->second.resolving, it->second.addressing})
      if (op) op->done = true, self->to_free.push_back(op);
    self->entries.erase(it);
    self->bump();
  }

  static void DNSSD_API resolved(DNSServiceRef, DNSServiceFlags, uint32_t, DNSServiceErrorType e, const char*,
                                 const char* target, uint16_t port, uint16_t txt_len, const unsigned char* txt, void* ctx) {
    auto* op = static_cast<Op*>(ctx);
    if (op->done) return;
    Impl* self = op->impl;
    op->done = true;  // one answer is enough: its address comes from GetAddrInfo
    self->to_free.push_back(op);
    std::lock_guard<std::mutex> lk(self->mu);
    const auto it = self->entries.find(op->name);
    if (it == self->entries.end()) return;
    Entry& en = it->second;
    if (en.resolving == op) en.resolving = nullptr;
    if (e != kDNSServiceErr_NoError || !target) {
      en.retry_at = now_ms() + 3000;
      return;
    }
    en.resolved = true;
    en.target = target;
    en.port = ntohs(port);
    en.txt = decode_txt(txt, txt_len);
    self->to_address.push_back(op->name);
    self->bump();
  }

  static void DNSSD_API addressed(DNSServiceRef, DNSServiceFlags flags, uint32_t, DNSServiceErrorType e, const char*,
                                  const sockaddr* a, uint32_t, void* ctx) {
    auto* op = static_cast<Op*>(ctx);
    if (op->done || e != kDNSServiceErr_NoError || !a || a->sa_family != AF_INET) return;
    char b[INET_ADDRSTRLEN];
    if (!inet_ntop(AF_INET, &reinterpret_cast<const sockaddr_in*>(a)->sin_addr, b, sizeof b)) return;
    Impl* self = op->impl;
    std::lock_guard<std::mutex> lk(self->mu);
    const auto it = self->entries.find(op->name);
    if (it == self->entries.end()) return;
    if (flags & kDNSServiceFlagsAdd) it->second.addrs.insert(b);
    else it->second.addrs.erase(b);
    self->bump();
  }

  bool open() {
    if (DNSServiceCreateConnection(&conn) != kDNSServiceErr_NoError) {
      conn = nullptr;
      std::lock_guard<std::mutex> lk(mu);
      err = bonjour_error(kDNSServiceErr_ServiceNotRunning);
      return false;
    }
    Op* op = new_op("");
    const DNSServiceErrorType e = DNSServiceBrowse(&op->ref, kDNSServiceFlagsShareConnection, kDNSServiceInterfaceIndexAny,
                                                   type.c_str(), nullptr, browsed, op);
    if (e != kDNSServiceErr_NoError) {
      op->ref = nullptr;
      close_all(false);
      std::lock_guard<std::mutex> lk(mu);
      err = bonjour_error(e);
      return false;
    }
    return true;
  }

  // Every operation, then the connection (subordinates first: dns_sd.h); `forget`: the servers too (the daemon went
  // away, and the browse that starts over finds them again)
  void close_all(bool forget) {
    for (auto& op : ops)
      if (op->ref && op->ref != conn) DNSServiceRefDeallocate(op->ref);
    ops.clear(), to_free.clear(), to_resolve.clear(), to_address.clear();
    if (conn) DNSServiceRefDeallocate(conn);
    conn = nullptr;
    std::lock_guard<std::mutex> lk(mu);
    for (auto& [name, en] : entries) en.resolving = en.addressing = nullptr;
    if (forget) entries.clear(), bump();
  }

  void start_resolve(const std::string& name) {
    std::string domain;
    {
      std::lock_guard<std::mutex> lk(mu);
      const auto it = entries.find(name);
      if (it == entries.end() || it->second.resolved || it->second.resolving) return;
      domain = it->second.domain;
    }
    Op* op = new_op(name);
    if (DNSServiceResolve(&op->ref, kDNSServiceFlagsShareConnection, kDNSServiceInterfaceIndexAny, name.c_str(),
                          type.c_str(), domain.c_str(), resolved, op) != kDNSServiceErr_NoError) {
      op->ref = nullptr;
      free_op(op);
      std::lock_guard<std::mutex> lk(mu);
      if (const auto it = entries.find(name); it != entries.end()) it->second.retry_at = now_ms() + 3000;
      return;
    }
    std::lock_guard<std::mutex> lk(mu);
    const auto it = entries.find(name);
    if (it == entries.end()) return (void)(op->done = true, to_free.push_back(op));
    it->second.resolving = op;
    it->second.resolve_at = now_ms();
  }

  void start_address(const std::string& name) {
    std::string target;
    {
      std::lock_guard<std::mutex> lk(mu);
      const auto it = entries.find(name);
      if (it == entries.end() || !it->second.resolved || it->second.addressing) return;
      target = it->second.target;
    }
    Op* op = new_op(name);
    if (DNSServiceGetAddrInfo(&op->ref, kDNSServiceFlagsShareConnection, kDNSServiceInterfaceIndexAny,
                              kDNSServiceProtocol_IPv4, target.c_str(), addressed, op) != kDNSServiceErr_NoError) {
      op->ref = nullptr;
      return free_op(op);
    }
    std::lock_guard<std::mutex> lk(mu);
    const auto it = entries.find(name);
    if (it == entries.end()) return (void)(op->done = true, to_free.push_back(op));
    it->second.addressing = op;
  }

  // After the callbacks: free what they finished, start what they asked for, retry resolves that got no answer
  void tend() {
    for (Op* op : std::exchange(to_free, {})) free_op(op);
    for (const auto& n : std::exchange(to_resolve, {})) start_resolve(n);
    for (const auto& n : std::exchange(to_address, {})) start_address(n);
    std::vector<std::string> again;
    {
      std::lock_guard<std::mutex> lk(mu);
      const double t = now_ms();
      for (auto& [name, en] : entries) {
        if (en.resolving && t - en.resolve_at > kResolveWait) {
          en.resolving->done = true, to_free.push_back(en.resolving);
          en.resolving = nullptr, en.retry_at = t;
        }
        if (!en.resolved && !en.resolving && en.retry_at > 0 && t >= en.retry_at) en.retry_at = 0, again.push_back(name);
      }
    }
    for (Op* op : std::exchange(to_free, {})) free_op(op);
    for (const auto& n : again) start_resolve(n);
  }

  void events() {
    while (!stop) {
      if (!conn && !open()) {
        wake.wait(-1, 3000);
        continue;
      }
      if (wake.wait(DNSServiceRefSockFD(conn), 250) && !stop) {
        const DNSServiceErrorType e = DNSServiceProcessResult(conn);
        if (e != kDNSServiceErr_NoError) {  // the daemon went away: start over
          {
            std::lock_guard<std::mutex> lk(mu);
            err = bonjour_error(e);
          }
          close_all(true);
          wake.wait(-1, 2000);
          continue;
        }
      }
      tend();
    }
    close_all(false);
  }

  // Reads each server that is due on a thread of its own, which puts the answer in when it comes, so a slow server
  // holds up no other. One still being read past kReadLimit counts as not answering; it is not read again until that
  // read ends.
  void reader(const std::shared_ptr<Impl>& self) {
    std::unique_lock<std::mutex> lk(mu);
    while (!stop) {
      const double t = now_ms();
      for (auto& [name, en] : entries) {
        if (en.reading) {
          if (t - en.read_start > kReadLimit && en.fails < 2) en.fails = 2, bump();
          continue;
        }
        if (!en.resolved || en.txt.proto != kProto || en.addrs.empty() || (en.read_at != 0 && t - en.read_at < kReadEvery))
          continue;
        en.reading = true;
        en.read_start = t;
        std::thread([self, name = name, addrs = std::vector<std::string>(en.addrs.begin(), en.addrs.end()),
                     port = en.port] {
          const std::optional<MacInfo> r = read_server(addrs, port);
          std::lock_guard<std::mutex> g(self->mu);
          const auto it = self->entries.find(name);
          if (it == self->entries.end()) return;
          Entry& e = it->second;
          e.reading = false;
          e.read_at = now_ms();
          if (r) e.info = *r, e.read = true, e.fails = 0;
          else ++e.fails;
          self->bump();
        }).detach();
      }
      cv.wait_for(lk, std::chrono::milliseconds(200));
    }
  }
};

Browser::Browser(const std::string& type) : impl_(std::make_shared<Impl>()) {
  impl_->type = type;
  std::thread([s = impl_] { s->events(); }).detach();
  std::thread([s = impl_] { s->reader(s); }).detach();
}

Browser::~Browser() {
  impl_->stop = true;
  impl_->wake.ring();
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->cv.notify_all();
}

std::vector<MacInfo> Browser::peers() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  std::vector<MacInfo> out;
  for (const auto& [name, en] : impl_->entries) {
    if (en.read && en.fails < 2) {
      out.push_back(en.info);
      continue;
    }
    if (!en.resolved || en.txt.proto != kProto || (!en.read && en.fails == 0)) continue;  // not read yet
    MacInfo m = en.read ? en.info : en.txt;  // it does not answer: what is known of it
    if (m.name.empty()) m.name = name;
    if (!en.addrs.empty()) m.host = *en.addrs.begin();
    m.port = en.port;
    m.rtt_ms = -1;
    out.push_back(m);
  }
  return out;
}

void Browser::wait(double seconds) const {
  Impl& s = *impl_;
  std::unique_lock<std::mutex> lk(s.mu);
  auto until = [&](double t, auto&& more) {
    while (!s.stop && now_ms() < t && more()) s.cv.wait_for(lk, std::chrono::microseconds(int64_t((t - now_ms()) * 1000) + 1));
  };
  until(s.t0 + seconds * 1000, [] { return true; });
  until(now_ms() + 1000, [&] {  // a server found and not read yet
    for (const auto& [name, en] : s.entries)
      if (!en.resolved || (!en.read && en.fails == 0 && en.txt.proto == kProto)) return true;
    return false;
  });
}

uint64_t Browser::changes() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->changes;
}

std::string Browser::error() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->err;
}

}  // namespace ni::cluster
