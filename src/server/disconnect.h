// Non-streaming requests stop generating when the client hangs up (#1288). Header-only so the CPU test links the
// same code that kapitsa serve runs.
#pragma once
#include <httplib/httplib.h>

#include "engine/execution_session_internal.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace ni::server {

// #1288: ExecutionSession::wait() never looks at the client, so a non-streaming request whose client hangs up keeps
// generating to the end. The streaming path notices through its sink; these watch the connection the handler serves.
inline thread_local ::socket_t tl_serving_socket = INVALID_SOCKET;

// httplib's keep-alive loop as Server::process_and_close_socket runs it, plus the socket recorded for the handler
class ServingServer final : public httplib::Server {
protected:
  bool process_and_close_socket(::socket_t sock) override {
    tl_serving_socket = sock;
    std::string remote_addr;
    int remote_port = 0;
    httplib::detail::get_remote_ip_and_port(sock, remote_addr, remote_port);
    std::string local_addr;
    int local_port = 0;
    httplib::detail::get_local_ip_and_port(sock, local_addr, local_port);
    const bool ret = httplib::detail::process_server_socket(
        svr_sock_, sock, keep_alive_max_count_, keep_alive_timeout_sec_, read_timeout_sec_, read_timeout_usec_,
        write_timeout_sec_, write_timeout_usec_,
        [&](httplib::Stream& strm, bool close_connection, bool& connection_closed) {
          return process_request(strm, remote_addr, remote_port, local_addr, local_port, close_connection,
                                 connection_closed, nullptr);
        });
    tl_serving_socket = INVALID_SOCKET;
    httplib::detail::shutdown_socket(sock);
    httplib::detail::close_socket(sock);
    return ret;
  }
};

// Polls `sock` until the caller's wait ends; cancels the execution if the peer closes first.
class DisconnectWatch {
public:
  DisconnectWatch(::socket_t sock, ExecutionSession& execution)
      : thread_([this, sock, &execution] {
          std::unique_lock<std::mutex> lk(mu_);
          while (!cv_.wait_for(lk, std::chrono::milliseconds(250), [this] { return stop_; })) {
            if (!httplib::detail::is_socket_alive(sock)) {
              execution.cancel();
              return;
            }
          }
        }) {}
  ~DisconnectWatch() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  DisconnectWatch(const DisconnectWatch&) = delete;
  DisconnectWatch& operator=(const DisconnectWatch&) = delete;

private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::thread thread_;
};

// The non-streaming routes' wait: the result, cancelled early if the client left. Outside a served request there
// is no socket to watch, so it just waits (is_socket_alive on an invalid fd would cancel at once).
inline const ChatResult& wait_for_client(ExecutionSession& execution) {
  if (tl_serving_socket == INVALID_SOCKET) return execution.wait();
  DisconnectWatch watch(tl_serving_socket, execution);
  return execution.wait();
}

}  // namespace ni::server
