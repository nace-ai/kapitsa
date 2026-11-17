// #1288 regression, CPU only: a non-streaming request whose client hangs up stops generating. Uses the production
// ServingServer and wait_for_client over loopback with a fake decode; no model, no GPU.
#include "server/disconnect.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <thread>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace ni;
using namespace std::chrono_literals;
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " #x "\n"; std::abort(); } } while (false)

namespace {
constexpr int kMaxSteps = 20000;  // a full decode would run this long at 1 ms per step: about 20 s

struct Decode {
  std::atomic<int> steps{0};
  std::atomic<bool> stop_seen{false};
  std::atomic<int> finished{0};
  std::string reason;
};

// The fake decode: one step per millisecond until the stop token fires or the steps run out.
ChatResult fake_decode(Decode& d, int target, detail::StopToken stop) {
  ChatResult r;
  for (int i = 0; i < target; ++i) {
    if (stop.stop_requested()) {
      d.stop_seen = true;
      r.finish_reason = "cancelled";
      return r;
    }
    ++d.steps;
    std::this_thread::sleep_for(1ms);
  }
  r.finish_reason = "stop";
  return r;
}

// Bind a non-streaming route like run_service does: a native execution waited on through wait_for_client.
void serve_route(httplib::Server& srv, Decode& d, int target) {
  srv.Post("/run", [&d, target](const httplib::Request&, httplib::Response& res) {
    auto execution = detail::ExecutionFactory::bind(
        [&d, target](const ChatRequest&, detail::ChatObserver&, detail::StopToken stop) {
          return fake_decode(d, target, stop);
        },
        std::make_shared<detail::ChatObserver>());
    execution->start({});
    const ChatResult& out = server::wait_for_client(*execution);
    d.reason = out.finish_reason;
    d.finished = 1;
    res.set_content(out.finish_reason, "text/plain");
  });
}

int connect_client(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  CHECK(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
  const timeval timeout{5, 0};
  CHECK(::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout) == 0);
  return fd;
}

void send_all(int fd, const std::string& data) {
  size_t sent = 0;
  while (sent < data.size()) {
    const auto n = ::send(fd, data.data() + sent, data.size() - sent, 0);
    if (n < 0 && errno == EINTR) continue;
    CHECK(n > 0);
    sent += static_cast<size_t>(n);
  }
}

const std::string kRequest = "POST /run HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                             "Content-Length: 2\r\n\r\n{}";
const std::string kNextRequest = "GET /next HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";

// A client that sends a complete request and closes the socket while the decode is still running.
void send_and_hang_up(int port) {
  const int fd = connect_client(port);
  send_all(fd, kRequest);
  std::this_thread::sleep_for(100ms);  // let the server start decoding before the client leaves
  ::close(fd);
}

void wait_until(const std::function<bool()>& pred, std::chrono::seconds limit) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  while (!pred() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(10ms);
}
}  // namespace

// A dropped non-streaming client stops the decode well before its full length, and the route returns "cancelled".
static void dropped_client_stops_generation() {
  Decode d;
  server::ServingServer srv;
  serve_route(srv, d, kMaxSteps);
  const int port = srv.bind_to_any_port("127.0.0.1");
  CHECK(port > 0);
  std::thread serving([&] { srv.listen_after_bind(); });
  send_and_hang_up(port);
  wait_until([&] { return d.finished.load() == 1; }, 15s);
  CHECK(d.finished == 1);
  CHECK(d.stop_seen);
  CHECK(d.reason == "cancelled");
  CHECK(d.steps < kMaxSteps / 2);
  srv.stop();
  serving.join();
  std::cout << "dropped client: decode stopped after " << d.steps << " of " << kMaxSteps << " steps\n";
}

// A client that stays connected gets the full result: the watch does not cancel a live connection.
static void connected_client_gets_result() {
  Decode d;
  server::ServingServer srv;
  serve_route(srv, d, 50);
  const int port = srv.bind_to_any_port("127.0.0.1");
  CHECK(port > 0);
  std::thread serving([&] { srv.listen_after_bind(); });
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(5);
  cli.set_read_timeout(15);
  auto res = cli.Post("/run", "{}", "application/json");
  CHECK(res && res->status == 200);
  CHECK(res->body == "stop");
  CHECK(!d.stop_seen);
  srv.stop();
  serving.join();
}

// A reset must cancel even when a pipelined request leaves readable bytes ahead of the socket error.
static void reset_with_queued_request_stops_generation() {
  Decode d;
  server::ServingServer srv;
  serve_route(srv, d, kMaxSteps);
  const int port = srv.bind_to_any_port("127.0.0.1");
  CHECK(port > 0);
  std::thread serving([&] { srv.listen_after_bind(); });
  const int fd = connect_client(port);
  send_all(fd, kRequest);
  wait_until([&] { return d.steps.load() > 0; }, 5s);
  CHECK(d.steps > 0);
  // Send after the first handler starts so these bytes stay in the socket, beyond httplib's read buffer.
  send_all(fd, kNextRequest);
  const linger reset{1, 0};
  CHECK(::setsockopt(fd, SOL_SOCKET, SO_LINGER, &reset, sizeof reset) == 0);
  ::close(fd);
  wait_until([&] { return d.finished.load() == 1; }, 5s);
  CHECK(d.finished == 1);
  CHECK(d.stop_seen);
  CHECK(d.reason == "cancelled");
  CHECK(d.steps < kMaxSteps / 2);
  srv.stop();
  serving.join();
  std::cout << "reset with queued request: decode stopped after " << d.steps << " of " << kMaxSteps << " steps\n";
}

// Watching a healthy socket must neither cancel the decode nor consume the next request's bytes.
static void queued_request_on_live_connection_is_served() {
  Decode d;
  std::atomic<int> next_served{0};
  server::ServingServer srv;
  serve_route(srv, d, 750);  // keep the first request active across several watch intervals
  srv.Get("/next", [&](const httplib::Request&, httplib::Response& res) {
    ++next_served;
    res.set_content("next", "text/plain");
  });
  const int port = srv.bind_to_any_port("127.0.0.1");
  CHECK(port > 0);
  std::thread serving([&] { srv.listen_after_bind(); });
  const int fd = connect_client(port);
  send_all(fd, kRequest);
  wait_until([&] { return d.steps.load() > 0; }, 5s);
  CHECK(d.steps > 0);
  send_all(fd, kNextRequest);
  std::string response;
  char buffer[1024];
  for (;;) {
    const auto n = ::recv(fd, buffer, sizeof buffer, 0);
    if (n < 0 && errno == EINTR) continue;
    CHECK(n >= 0);
    if (n == 0) break;
    response.append(buffer, static_cast<size_t>(n));
  }
  ::close(fd);
  CHECK(!d.stop_seen);
  CHECK(d.steps == 750);
  CHECK(d.finished == 1);
  CHECK(d.reason == "stop");
  CHECK(next_served == 1);
  CHECK(response.starts_with("HTTP/1.1 200 OK\r\n"));
  CHECK(response.find("\r\n\r\nstopHTTP/1.1 200 OK\r\n") != std::string::npos);
  CHECK(response.ends_with("\r\n\r\nnext"));
  srv.stop();
  serving.join();
  std::cout << "live queued request: both responses returned without cancellation\n";
}

// Outside a served request there is no socket to watch; the wait returns the result without cancelling.
static void unserved_wait_returns_result() {
  Decode d;
  auto execution = detail::ExecutionFactory::bind(
      [&d](const ChatRequest&, detail::ChatObserver&, detail::StopToken stop) {
        return fake_decode(d, 20, stop);
      },
      std::make_shared<detail::ChatObserver>());
  execution->start({});
  CHECK(server::wait_for_client(*execution).finish_reason == "stop");
  CHECK(!d.stop_seen);
}

int main() {
  unserved_wait_returns_result();
  connected_client_gets_result();
  dropped_client_stops_generation();
  queued_request_on_live_connection_is_served();
  reset_with_queued_request_stops_generation();
  std::cout << "disconnect_test: all passed\n";
  return 0;
}
