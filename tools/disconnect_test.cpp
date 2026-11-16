// #1288 regression, CPU only: a non-streaming request whose client hangs up stops generating. Uses the production
// ServingServer and wait_for_client over loopback with a fake decode; no model, no GPU.
#include "server/disconnect.h"

#include <atomic>
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

// A client that sends a complete request and closes the socket while the decode is still running.
void send_and_hang_up(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  CHECK(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
  const std::string request = "POST /run HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                              "Content-Length: 2\r\nConnection: close\r\n\r\n{}";
  CHECK(::send(fd, request.data(), request.size(), 0) == static_cast<ssize_t>(request.size()));
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
  std::cout << "disconnect_test: all passed\n";
  return 0;
}
