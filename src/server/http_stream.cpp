#include "server/http_stream.h"
#include "api/common.h"
#include <httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace ni {
namespace {
using json = nlohmann::ordered_json;
// HTTP framing only. Neither observer contains an execution handle, even indirectly.
struct StreamJob final : detail::ChatObserver {
  std::mutex mu;  // the queue, the framing flags and every call into the writer
  std::condition_variable cv;
  std::deque<std::string> out;
  enum class Terminal { Pending, Enqueued, Failed };
  Terminal terminal = Terminal::Pending;
  bool started = false, sent = false;  // sent: the first event is out
  const server::StreamMeter meter;
  const std::shared_ptr<api::StreamWriter> writer;
  const bool early;
  StreamJob(std::shared_ptr<api::StreamWriter> w, server::StreamMeter m, bool e)
      : meter(std::move(m)), writer(std::move(w)), early(e) {}
  void cancelled() noexcept override {
    std::lock_guard<std::mutex> lk(mu);
    cv.notify_all();
  }
  void open(const ChatResult& prompt) {  // mu held
    out.push_back(writer->begin(prompt));
    started = true;
    cv.notify_all();
  }
  bool accepted(const ChatResult& prompt, detail::StopToken stop) override {
    if (!early) return true;
    std::unique_lock<std::mutex> lk(mu);
    open(prompt);
    // A destroyed/abandoned provider cancels its session, waking this gate even without a first sink write.
    cv.wait(lk, [&] { return sent || stop.stop_requested(); });
    return !stop.stop_requested();
  }
  void prompt(const ChatResult& prompt) override {
    if (early) return;
    std::lock_guard<std::mutex> lk(mu);
    open(prompt);
  }
  bool delta(const ChatDelta& d) override {
    try { meter.token(); } catch (...) {}  // telemetry must not change framing or inference
    std::lock_guard<std::mutex> lk(mu);
    std::string s = writer->delta(d);
    if (!s.empty()) out.push_back(std::move(s));
    cv.notify_all();
    return true;
  }
  void finished(const ChatResult& result, detail::StopToken stop) override {
    std::lock_guard<std::mutex> lk(mu);
    struct Notify {
      StreamJob& job;
      ~Notify() {
        if (job.terminal == Terminal::Pending) job.terminal = Terminal::Failed;
        job.cv.notify_all();
      }
    } notify{*this};
    if (started && !stop.stop_requested() && result.finish_reason != "cancelled") {
      out.push_back(writer->end(result));
      terminal = Terminal::Enqueued;  // an empty end is valid; a writer exception never reaches here
    }
  }
};

struct StreamCompletion {
  const std::shared_ptr<ExecutionSession> execution;
  const server::StreamMeter meter;
  enum class Phase { Ready, Running, Finished };
  Phase phase = Phase::Ready;
  StreamCompletion(std::shared_ptr<ExecutionSession> e, server::StreamMeter m)
      : execution(std::move(e)), meter(std::move(m)) {}
  void start(const ChatRequest& request) {
    execution->start(request);
    phase = Phase::Running;
  }
  ~StreamCompletion() {
    if (phase == Phase::Running) {
      execution->cancel();
      finish();
    }
  }
  const ChatResult& finish() {
    const auto& result = execution->wait();
    if (phase != Phase::Finished) {
      phase = Phase::Finished;
      try { meter.done(result); } catch (...) {}
    }
    return result;
  }
};

}  // namespace

// Runs `r` and answers it as a stream of `w`'s events, or with failure(result) when it fails before the stream starts:
// once the prompt is matched against the slot's cache, or `early`, as soon as the request is accepted. Then calls
// after(result) (the reply is out) from the connection's thread.
void server::stream_reply(const ChatRequest& r, std::shared_ptr<api::StreamWriter> w, httplib::Response& res,
                          BindExecution bind, std::function<void(const ChatResult&, httplib::Response&)> failure,
                          std::function<void(const ChatResult&)> after, StreamMeter meter,
                          bool early, bool telemetry) {
  auto job = std::make_shared<StreamJob>(w, meter, early);
  auto execution = bind(job);
  auto completion = std::make_shared<StreamCompletion>(execution, meter);
  completion->start(r);
  {
    std::unique_lock<std::mutex> lk(job->mu);
    job->cv.wait(lk, [&] { return job->started || job->terminal != StreamJob::Terminal::Pending; });
  }
  if (!job->started) {
    const auto& result = completion->finish();
    failure(result, res);
    return;
  }
  res.set_header("Cache-Control", "no-cache");
  res.set_chunked_content_provider("text/event-stream", [completion, job, w, after, live = r.live, telemetry](size_t, httplib::DataSink& sink) {
    const auto& execution = completion->execution;
    // the client is gone: the engine stops at its next token, or gives the request up before it takes a slot
    auto gone = [&] {
      execution->cancel();
      completion->finish();
      return false;
    };
    auto last = std::chrono::steady_clock::now();  // the last write
    bool first = true;
    auto progress_at = std::chrono::steady_clock::time_point::min();
    auto terminal = StreamJob::Terminal::Pending;
    for (;;) {
      std::deque<std::string> batch;
      {
        std::unique_lock<std::mutex> lk(job->mu);
        // a second at a time while nothing comes, to see whether the client is still there
        job->cv.wait_for(lk, std::chrono::seconds(1), [&] { return !job->out.empty() || job->terminal != StreamJob::Terminal::Pending; });
        terminal = job->terminal;
        if (job->out.empty() && terminal == StreamJob::Terminal::Pending && std::chrono::steady_clock::now() - last >= std::chrono::seconds(15))
          job->out.push_back(w->keepalive());  // 15 s without an event
        batch.swap(job->out);
      }
      const bool done = terminal != StreamJob::Terminal::Pending;
      // Opt-in benchmark events sample existing atomics from the HTTP thread; no GPU polling or synchronization.
      if (telemetry && live) {
        const auto now = std::chrono::steady_clock::now();
        if (done || progress_at == std::chrono::steady_clock::time_point::min() ||
            now - progress_at >= std::chrono::seconds(1)) {
          const int prompt = live->prompt.load(std::memory_order_relaxed);
          const int cached = live->cached.load(std::memory_order_relaxed);
          json progress{{"phase", live->phase.load(std::memory_order_relaxed)}, {"prompt_tokens", prompt},
                        {"cached_tokens", cached}, {"processed_tokens", std::clamp(live->read.load(std::memory_order_relaxed), 0, prompt)},
                        {"completion_tokens", live->tokens.load(std::memory_order_relaxed)},
                        {"cycles", live->cycles.load(std::memory_order_relaxed)}, {"speculative", live->spec.load(std::memory_order_relaxed)},
                        {"final", done}};
          batch.push_front("event: kapitsa_progress\ndata: " + json{{"kapitsa_progress", progress}}.dump() + "\n\n");
          progress_at = now;
        }
      }
      if (batch.empty() && !sink.is_writable()) return gone();
      for (const auto& e : batch)
        if (!sink.write(e.data(), e.size())) return gone();
      if (!batch.empty()) last = std::chrono::steady_clock::now();
      if (!batch.empty() && first) {  // an early stream's engine goes on to a slot now
        first = false;
        std::lock_guard<std::mutex> lk(job->mu);
        job->sent = true;
        job->cv.notify_all();
      }
      if (done) break;
    }
    const auto& result = execution->wait();
    if (terminal != StreamJob::Terminal::Enqueued) {
      completion->finish();
      return false;
    }
    sink.done();
    completion->finish();
    after(result);
    return true;
  });
}

}  // namespace ni
