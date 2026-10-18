// HTTP GETs for model downloads, on NSURLSession (part of macOS: the system's proxy settings, trust store
// and HTTP/2 come with it; no library to ship). One HttpClient is one session with its own connections, so
// parallel downloads open parallel TCP connections (one connection per client: a single flow is capped
// well below the link speed on some networks).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ni::hub {

using Headers = std::vector<std::pair<std::string, std::string>>;

struct HttpResult {
  int status = 0;          // HTTP status, 0 when no response arrived
  std::string body;        // get(): the body; stream(): the body of an error response (first 4 KB)
  std::string error;       // transport failure (no response), empty otherwise
  int error_code = 0;      // NSURLError code of that failure
  bool offline = false;    // no network, or the host cannot be resolved / reached
  bool cancelled = false;  // the sink asked to stop
  bool range_ignored = false;  // a range was asked for and the server answered 200 with the whole file (not read)
  std::string url;             // the URL that answered, after redirects
  std::string refused;         // a redirect not followed (allow_redirects refused it): where it led
  bool ok() const { return error.empty() && !cancelled && !range_ignored && status >= 200 && status < 300; }
};

class HttpClient {
 public:
  // `idle_timeout_s`: a request fails when no data arrives for this long.
  explicit HttpClient(double idle_timeout_s = 60);
  ~HttpClient();
  HttpClient(const HttpClient&) = delete;
  HttpClient& operator=(const HttpClient&) = delete;

  // Whole body in memory (API JSON, small files, byte ranges of a few MB). Range [a, b) when b > a.
  HttpResult get(const std::string& url, const Headers& headers, uint64_t a = 0, uint64_t b = 0);
  // Body handed to `sink` as it arrives (on the session's queue); `sink` returns false to stop.
  // Range [a, b) when b > a. On a 2xx answer the body goes to the sink; otherwise into HttpResult::body.
  HttpResult stream(const std::string& url, const Headers& headers, uint64_t a, uint64_t b,
                    const std::function<bool(const uint8_t*, size_t)>& sink);
  // A HEAD request: the status and the URL that answered (HttpResult::url), no body.
  HttpResult head(const std::string& url, const Headers& headers);
  // Redirects are followed only to URLs `allow` accepts (by default, all): a refused one ends the request with its
  // 3xx answer, and HttpResult::refused says where it led.
  void allow_redirects(std::function<bool(const std::string& url)> allow);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ni::hub
