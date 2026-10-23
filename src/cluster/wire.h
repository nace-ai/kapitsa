// The split's wire (src/cluster/pipe.h): the messages' header and the socket helpers the head's Link (pipe.cpp) and a
// stage (stage.cpp) share. Internal to src/cluster.
#pragma once
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>

#include "cluster/pipe.h"

namespace ni::cluster::wire {

inline constexpr uint32_t kMagic = 0x5049504Bu;  // "KPIP"
inline constexpr int kPipeProto = 1;

// Messages. A pass's op is its Remote::Kind + 1.
enum Op : uint16_t {
  kHello = 1,
  kStep = 2,
  kPrefill = 3,
  kVerify = 4,
  kNewSeq = 10,
  kGrow = 11,
  kDropKv = 12,
  kReset = 13,
  kNewCkpt = 14,
  kSave = 15,
  kRestore = 16,
  kFlush = 17,
  kRope = 18,
  kPing = 19,
};
inline constexpr uint16_t kFail = 1;  // flags: a stage could not do it (grow: no memory; hello: "error" in its payload)

// seq, pos, rows, n_pend: a pass's (Remote::Pass); a, b: a pass's snap and snap_row, new_seq's limit, grow's need, a
// checkpoint's id, rope's rope_n and rope_delta
struct Header {
  uint32_t magic = kMagic;
  uint16_t op = 0, flags = 0;
  uint32_t id = 0;  // the head's message number: it comes back with the message
  int32_t seq = -1, pos = 0, rows = 0, n_pend = 0, a = 0, b = 0;
  uint32_t reserved = 0;
  uint64_t bytes = 0;  // the payload's
  uint32_t pad[4] = {};
};
static_assert(sizeof(Header) == 64, "the wire header is 64 bytes");

inline bool is_pass(uint16_t op) { return op == kStep || op == kPrefill || op == kVerify; }

// ---- sockets ----

inline bool write_all(int fd, iovec* iov, int n) {
  while (n > 0) {
    const ssize_t w = writev(fd, iov, std::min(n, IOV_MAX));
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    size_t left = size_t(w);
    while (n > 0 && left >= iov->iov_len) {
      left -= iov->iov_len;
      ++iov;
      --n;
    }
    if (n > 0) {
      iov->iov_base = static_cast<char*>(iov->iov_base) + left;
      iov->iov_len -= left;
    }
  }
  return true;
}

inline bool write_msg(int fd, Header h, const void* payload, size_t bytes) {
  h.bytes = bytes;
  iovec iov[2] = {{&h, sizeof h}, {const_cast<void*>(payload), bytes}};
  return write_all(fd, iov, bytes ? 2 : 1);
}

inline bool read_all(int fd, void* dst, size_t n) {
  auto* p = static_cast<uint8_t*>(dst);
  while (n) {
    const ssize_t r = read(fd, p, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    p += r;
    n -= size_t(r);
  }
  return true;
}

inline bool read_payload(int fd, const Header& h, std::string* out, size_t limit = size_t(64) << 20) {
  if (h.bytes > limit) return false;
  out->assign(size_t(h.bytes), '\0');
  return read_all(fd, out->data(), out->size());
}

// waits up to timeout_ms for fd to be readable
inline bool readable(int fd, int timeout_ms) {
  pollfd p{fd, POLLIN, 0};
  return poll(&p, 1, timeout_ms) == 1;
}

inline void tune(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);  // a decode step's 51 KB go at once
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
  // a Mac that went to sleep or off the network is noticed in ~11 s (5 s idle, then 3 probes 2 s apart), not 2 hours
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
  int idle = 5, intvl = 2, cnt = 3;
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof idle);
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt);
#ifdef TCP_RXT_CONNDROPTIME
  int drop = 15;  // ... and one that stops acknowledging what was sent in 15 s
  setsockopt(fd, IPPROTO_TCP, TCP_RXT_CONNDROPTIME, &drop, sizeof drop);
#endif
  int buf = 8 << 20;  // a prefill chunk's rows: tens of MB
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
}

// "host:port", "[v6]:port" or "host" (the default port)
inline bool split_addr(const std::string& a, std::string* host, std::string* port) {
  if (a.empty()) return false;
  if (a[0] == '[') {
    const size_t close = a.find(']');
    if (close == std::string::npos) return false;
    *host = a.substr(1, close - 1);
    *port = close + 2 <= a.size() && a[close + 1] == ':' ? a.substr(close + 2) : std::to_string(kStagePort);
  } else {
    const size_t colon = a.rfind(':');
    const bool one = colon != std::string::npos && a.find(':') == colon;
    *host = one ? a.substr(0, colon) : a;
    *port = one ? a.substr(colon + 1) : std::to_string(kStagePort);
  }
  return !host->empty() && !port->empty();
}

inline int connect_to(const std::string& addr, int timeout_ms, std::string* why) {
  std::string host, port;
  if (!split_addr(addr, &host, &port)) {
    *why = "not host:port: \"" + addr + "\"";
    return -1;
  }
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res); rc != 0) {
    *why = addr + ": " + gai_strerror(rc);
    return -1;
  }
  int fd = -1;
  std::string last = "no address";
  for (addrinfo* ai = res; ai; ai = ai->ai_next) {
    fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
    int rc = connect(fd, ai->ai_addr, ai->ai_addrlen);
    if (rc != 0 && errno == EINPROGRESS) {
      pollfd p{fd, POLLOUT, 0};
      rc = -1;
      if (poll(&p, 1, timeout_ms) == 1) {
        int err = 0;
        socklen_t len = sizeof err;
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        rc = err ? -1 : 0;
        errno = err;
      } else {
        errno = ETIMEDOUT;
      }
    }
    if (rc == 0) {
      fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
      tune(fd);
      break;
    }
    last = strerror(errno);
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) *why = addr + ": " + last;
  return fd;
}

// a listening socket on host:port (port 0: any), or -1 with *why
inline int listen_on(const std::string& host, int port, std::string* why) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* res = nullptr;
  const std::string p = std::to_string(port);
  if (const int rc = getaddrinfo(host.empty() ? nullptr : host.c_str(), p.c_str(), &hints, &res); rc != 0) {
    *why = host + ":" + p + ": " + gai_strerror(rc);
    return -1;
  }
  int fd = -1;
  for (addrinfo* ai = res; ai && fd < 0; ai = ai->ai_next) {
    fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(fd, ai->ai_addr, ai->ai_addrlen) != 0 || listen(fd, 4) != 0) {
      *why = "cannot listen on " + host + ":" + p + ": " + strerror(errno);
      close(fd);
      fd = -1;
    }
  }
  freeaddrinfo(res);
  return fd;
}

inline int bound_port(int fd) {
  sockaddr_storage a{};
  socklen_t len = sizeof a;
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) != 0) return 0;
  return a.ss_family == AF_INET6 ? ntohs(reinterpret_cast<sockaddr_in6*>(&a)->sin6_port)
                                 : ntohs(reinterpret_cast<sockaddr_in*>(&a)->sin_port);
}

// this side's address of a connected socket ("[v6]" for IPv6), as the Mac at its other end reaches this one
inline std::string local_addr(int fd) {
  sockaddr_storage a{};
  socklen_t len = sizeof a;
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) != 0) return "";
  char b[INET6_ADDRSTRLEN] = "";
  if (a.ss_family == AF_INET6) {
    inet_ntop(AF_INET6, &reinterpret_cast<sockaddr_in6*>(&a)->sin6_addr, b, sizeof b);
    return "[" + std::string(b) + "]";
  }
  inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(&a)->sin_addr, b, sizeof b);
  return b;
}

inline std::string peer_addr(const sockaddr_storage& a) {
  char b[INET6_ADDRSTRLEN] = "";
  if (a.ss_family == AF_INET6) inet_ntop(AF_INET6, &reinterpret_cast<const sockaddr_in6*>(&a)->sin6_addr, b, sizeof b);
  else inet_ntop(AF_INET, &reinterpret_cast<const sockaddr_in*>(&a)->sin_addr, b, sizeof b);
  return b;
}


}  // namespace ni::cluster::wire
