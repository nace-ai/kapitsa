// Socket-free regression of the production StreamJob and httplib provider, not a copied implementation.
#include "server/http_stream.h"
#include "api/common.h"
#include <httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace ni;
using namespace std::chrono_literals;
using Json = nlohmann::ordered_json;

enum class Mode { Success, EmptyEnd, EngineError, PreStreamError, BeginThrows, DeltaThrows, EndThrows,
                  EarlySuccess, EarlyFailedWrite, FailedWrite, IdleDisconnect, Abandoned, CompletedAbandoned, SinkThrows, Telemetry,
                  MeterThrows, AbandonedMeterThrows, TokenMeterThrows };
struct Observed {
  std::atomic<bool> work_exited{false}, accepted_returned{false}, saw_cancel{false}, token_on_worker{false};
  std::atomic<int> begins{0}, deltas{0}, ends{0}, tokens{0};
  int meters = 0, afters = 0, failures = 0, dones = 0, writes = 0;
  bool done_after_work = true, after_order = true, first_write_before_accept = true;
  std::string result_error, result_reason, wire;
  std::vector<std::string> errors;
  void check(bool ok, const std::string& why) { if (!ok) errors.push_back(why); }
};
struct Writer final : api::StreamWriter {
  Mode mode;
  std::shared_ptr<Observed> seen;
  Writer(Mode m, std::shared_ptr<Observed> s) : mode(m), seen(std::move(s)) {}
  std::string begin(const ChatResult&) override {
    ++seen->begins;
    if (mode == Mode::BeginThrows) throw std::runtime_error("begin broke");
    return "event: begin\ndata: {}\n\n";
  }
  std::string delta(const ChatDelta&) override {
    ++seen->deltas;
    if (mode == Mode::DeltaThrows) throw std::runtime_error("delta broke");
    return "event: delta\ndata: token\n\n";
  }
  std::string end(const ChatResult& result) override {
    ++seen->ends;
    if (mode == Mode::EndThrows) throw std::runtime_error("end broke");
    if (mode == Mode::EmptyEnd) return {};
    return "event: end\ndata: " + result.finish_reason + "\n\n";
  }
  std::string keepalive() override { return ": keepalive\n\n"; }
};

bool run(const char* name, Mode mode) {
  auto seen = std::make_shared<Observed>();
  const auto connection = std::this_thread::get_id();
  const bool abandoned = mode == Mode::Abandoned || mode == Mode::AbandonedMeterThrows;
  const bool early = mode == Mode::EarlySuccess || mode == Mode::EarlyFailedWrite || abandoned;
  const bool held = mode == Mode::EarlyFailedWrite || mode == Mode::FailedWrite ||
                    mode == Mode::IdleDisconnect || abandoned;
  const bool framing_failure = mode == Mode::DeltaThrows || mode == Mode::EndThrows;
  const bool pre_stream = mode == Mode::PreStreamError || mode == Mode::BeginThrows;
  std::weak_ptr<ExecutionSession> weak;
  bool provider_ok = false, chunk_ok = false;
  {
    httplib::Response response;
    ChatRequest request;
    if (mode == Mode::Telemetry) request.live = std::make_shared<LiveDecode>();
    server::stream_reply(request, std::make_shared<Writer>(mode, seen), response,
      [&](auto observer) {
        auto session = detail::ExecutionFactory::bind(
          [=](const ChatRequest&, detail::ChatObserver& events, detail::StopToken stop) {
            struct Exit { Observed& seen; ~Exit() { seen.work_exited = true; } } exit{*seen};
            ChatResult result;
            result.finish_reason = "stop";
            if (mode == Mode::PreStreamError) {
              result.finish_reason = "error";
              result.error = "refused before stream";
              return result;
            }
            bool accepted = events.accepted(result, stop);
            seen->accepted_returned = true;
            if (!accepted) {
              seen->saw_cancel = stop.stop_requested();
              result.finish_reason = "cancelled";
              return result;
            }
            events.prompt(result);
            if (held) {
              const auto deadline = std::chrono::steady_clock::now() + 5s;
              while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(1ms);
              seen->saw_cancel = stop.stop_requested();
              result.finish_reason = "cancelled";
            } else if (mode == Mode::EngineError) {
              result.finish_reason = "error";
              result.error = "engine failed with a valid error frame";
            } else if (mode != Mode::BeginThrows) {
              ChatDelta delta;
              delta.content = "token";
              events.delta(delta);
            }
            return result;
          }, std::move(observer));
        weak = session;
        return session;
      },
      [=](const ChatResult& result, httplib::Response& res) {
        ++seen->failures;
        seen->check(seen->work_exited, "pre-stream failure before work exit");
        seen->check(seen->meters == 1, "pre-stream failure before meter");
        res.status = 500;
        res.set_content(result.error, "application/json");
      },
      [=](const ChatResult&) {
        ++seen->afters;
        seen->after_order = seen->dones == 1 && seen->meters == 1 && seen->work_exited;
      },
      {[=] {
          ++seen->tokens;
          seen->token_on_worker = std::this_thread::get_id() != connection;
          if (mode == Mode::TokenMeterThrows) throw std::bad_alloc();
        },
       [=](const ChatResult& result) {
         ++seen->meters;
         seen->check(std::this_thread::get_id() == connection, "meter moved off connection thread");
         seen->check(seen->work_exited, "meter before work exit");
         seen->result_error = result.error;
         seen->result_reason = result.finish_reason;
          if (mode == Mode::MeterThrows || mode == Mode::AbandonedMeterThrows) throw std::bad_alloc();
       }}, early, mode == Mode::Telemetry);
    if (pre_stream) {
      seen->check(!response.content_provider_ && response.status == 500, "pre-stream failure installed provider");
      seen->check(seen->failures == 1 && seen->meters == 1 && seen->afters == 0, "pre-stream effects count");
      const auto wanted = mode == Mode::BeginThrows ? "begin broke" : "refused before stream";
      seen->check(response.body == wanted, "pre-stream error lost");
    } else if (abandoned || mode == Mode::CompletedAbandoned) {
      seen->check(bool(response.content_provider_), "no provider to abandon");
      if (early) seen->check(!seen->accepted_returned, "early acceptance escaped before provider");
      else {
        weak.lock()->wait();
        seen->check(seen->work_exited, "non-early worker did not complete before abandonment");
      }
    } else if (mode == Mode::SinkThrows) {
      httplib::DataSink sink;
      sink.write = [](const char*, size_t) -> bool { throw std::runtime_error("sink broke"); };
      sink.is_writable = [] { return true; };
      sink.done = [&] { ++seen->dones; };
      bool threw = false;
      try { response.content_provider_(0, 0, sink); }
      catch (const std::runtime_error& e) { threw = std::string(e.what()) == "sink broke"; }
      seen->check(threw, "sink exception lost");
    } else if (held) {
      httplib::DataSink sink;
      sink.write = [&](const char*, size_t) {
        ++seen->writes;
        if (early) seen->first_write_before_accept = !seen->accepted_returned;
        return mode == Mode::IdleDisconnect;  // idle case accepts begin, then closes while no events arrive
      };
      sink.is_writable = [] { return false; };
      sink.done = [&] { ++seen->dones; };
      provider_ok = response.content_provider_(0, 0, sink);
      seen->check(!provider_ok && seen->dones == 0 && seen->afters == 0, "disconnect completed HTTP");
      seen->check(seen->meters == 1 && seen->saw_cancel && seen->work_exited, "disconnect failed to cancel/join");
      seen->check(seen->writes == 1 && seen->first_write_before_accept, "first-write gate changed");
    } else {
      httplib::detail::BufferStream stream;
      httplib::detail::nocompressor compressor;
      httplib::Error error = httplib::Error::Unknown;
      httplib::ContentProvider provider = [&](size_t offset, size_t length, httplib::DataSink& sink) {
        auto done = sink.done;
        sink.done = [&] {
          ++seen->dones;
          seen->done_after_work = seen->work_exited.load();  // inspect inside done, not after provider returns
          done();
        };
        auto write = sink.write;
        sink.write = [&](const char* data, size_t size) {
          if (early && seen->writes == 0) seen->first_write_before_accept = !seen->accepted_returned;
          ++seen->writes;
          return write(data, size);
        };
        provider_ok = response.content_provider_(offset, length, sink);
        return provider_ok;
      };
      chunk_ok = httplib::detail::write_content_chunked(stream, provider, [] { return false; }, compressor, error);
      seen->wire = stream.get_buffer();
      const bool terminal = seen->wire.ends_with("0\r\n\r\n");
      if (framing_failure) {
        seen->check(!provider_ok, "provider returned success");
        seen->check(seen->dones == 0, "done called on failed framing");
        seen->check(!terminal, "terminal HTTP chunk on failed framing");
        seen->check(chunk_ok == provider_ok, "chunk writer disagrees with provider");
        const auto wanted = mode == Mode::DeltaThrows ? "delta broke" : "end broke";
        seen->check(seen->result_reason == "error" && seen->result_error == wanted, "writer exception result lost");
        seen->check(seen->ends == (mode == Mode::DeltaThrows ? 0 : 1), "unexpected end call count");
      } else {
        seen->check(provider_ok && chunk_ok && terminal && error == httplib::Error::Success, "valid frame aborted");
        seen->check(seen->dones == 1 && seen->ends == 1, "valid completion count");
        seen->check(seen->done_after_work, "done before work exit");
        seen->check(mode == Mode::EmptyEnd || seen->wire.find("event: end") != std::string::npos, "end frame missing");
        seen->check(seen->first_write_before_accept, "early acceptance escaped before first write");
        if (mode == Mode::EngineError) seen->check(seen->result_reason == "error", "engine error changed");
        if (mode == Mode::Telemetry) seen->check(seen->wire.find("event: kapitsa_progress") != std::string::npos, "telemetry missing");
      }
      seen->check(seen->meters == 1 && seen->afters == (provider_ok ? 1 : 0), "completion effects count");
      seen->check(seen->after_order, "after ran before done/join/meter");
      seen->check(seen->failures == 0, "stream invoked pre-stream failure");
      if (mode != Mode::EngineError) seen->check(seen->tokens == 1 && seen->token_on_worker, "token meter moved to dequeue");
    }
  }
  seen->check(weak.expired() && seen->work_exited, "execution retained after response destruction");
  seen->check(seen->meters == 1, "meter completion was not exactly once after response destruction");
  if (abandoned) seen->check(seen->saw_cancel, "abandoned early provider did not cancel");
  if (abandoned || mode == Mode::CompletedAbandoned || mode == Mode::SinkThrows)
    seen->check(seen->dones == 0 && seen->afters == 0, "abandoned/throwing provider ran success effects");
  Json row{{"case", name}, {"failures", seen->errors}, {"provider_ok", provider_ok}, {"chunk_ok", chunk_ok},
           {"sink_done", seen->dones}, {"after", seen->afters}, {"meter_done", seen->meters},
           {"work_exited", seen->work_exited.load()}, {"wire", seen->wire}};
  std::cout << row.dump() << '\n';
  return seen->errors.empty();
}
struct CopyFailure {
  std::shared_ptr<bool> bound;
  explicit CopyFailure(std::shared_ptr<bool> b) : bound(std::move(b)) {}
  CopyFailure(CopyFailure&&) noexcept = default;
  CopyFailure(const CopyFailure& other) : bound(other.bound) {
    if (*bound) throw std::bad_alloc();
  }
  void operator()(const ChatResult&) const {}
};

bool completion_setup_failure() {
  auto bound = std::make_shared<bool>(false);
  std::shared_ptr<ExecutionSession> execution;
  std::atomic<bool> ran = false;
  httplib::Response response;
  bool threw = false;
  try {
    server::stream_reply({}, std::make_shared<Writer>(Mode::Success, std::make_shared<Observed>()), response,
      [&](auto observer) {
        execution = detail::ExecutionFactory::bind(
          [&](const ChatRequest&, detail::ChatObserver&, detail::StopToken) {
            ran = true;
            return ChatResult{};
          }, std::move(observer));
        *bound = true;
        return execution;
      }, [](const ChatResult&, httplib::Response&) {}, [](const ChatResult&) {},
      {[] {}, CopyFailure(bound)});
  } catch (const std::bad_alloc&) { threw = true; }
  bool started = true;
  try { execution->wait(); }
  catch (const std::logic_error&) { started = false; }
  const bool ok = threw && !started && !ran && !response.content_provider_;
  std::cout << Json{{"case", "completion-setup-failure"}, {"passed", ok},
                    {"started", started}, {"ran", ran.load()}}.dump() << '\n';
  return ok;
}
}  // namespace

int main() {
  bool ok = completion_setup_failure();
  for (const auto& [name, mode] : std::vector<std::pair<const char*, Mode>>{
         {"success", Mode::Success}, {"empty-end", Mode::EmptyEnd}, {"framed-engine-error", Mode::EngineError},
         {"pre-stream-error", Mode::PreStreamError}, {"begin-throws", Mode::BeginThrows},
         {"delta-throws", Mode::DeltaThrows}, {"end-throws", Mode::EndThrows}, {"early-success", Mode::EarlySuccess},
         {"early-failed-first-write", Mode::EarlyFailedWrite}, {"failed-first-write", Mode::FailedWrite},
         {"idle-disconnect", Mode::IdleDisconnect}, {"abandoned-provider", Mode::Abandoned},
         {"completed-abandoned-provider", Mode::CompletedAbandoned}, {"sink-throws", Mode::SinkThrows}, {"telemetry", Mode::Telemetry},
         {"abandoned-throwing-meter", Mode::AbandonedMeterThrows}, {"throwing-meter", Mode::MeterThrows},
          {"throwing-token-meter", Mode::TokenMeterThrows}})
    ok = run(name, mode) && ok;
  return ok ? 0 : 1;
}
