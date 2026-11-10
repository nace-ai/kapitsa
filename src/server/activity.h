// What the server is doing, as events on the bus (src/server/events.h): requests as they start and finish, each
// active request's progress once per tick, and the log lines the ring holds. One fixed-rate tick samples every
// request, so every observer sees the same tokens per second however often it reads (docs/control-socket.md,
// "Rates").
#pragma once
#include "engine/chat.h"
#include "server/events.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ni::server {

class Activity {
 public:
  static constexpr int kTickMs = 500;

  // One request on the inference port, from its arrival to its end
  struct Request {
    std::string id, api, client;
    double t0 = 0;                                                   // arrival (steady ms)
    std::shared_ptr<LiveDecode> live = std::make_shared<LiveDecode>();  // ChatRequest::live
    // the last tick's sample (under Activity's lock)
    int tokens_at_tick = 0;
    double tick_at = 0;     // when that sample was taken (its start, before its first tick)
    nlohmann::json sample;  // the request_progress it gave
  };

  Activity(Events& bus, int tick_ms = kTickMs);
  int tick_ms() const { return tick_ms_; }

  // A request arrives: `api` openai, anthropic or ollama; `user_agent` its header (cut to 128 bytes). Publishes
  // request_started.
  std::shared_ptr<Request> start(const std::string& api, const std::string& user_agent);
  // It ends: publishes request_finished with the result's counts, times and rates. Once per request.
  void finish(const std::shared_ptr<Request>& r, const ChatResult& result);
  // status.requests: every active request as of the last tick
  nlohmann::json requests() const;
  // One tick: samples every active request (request_progress each) and turns the ring's log lines into log events.
  // `now_ms`: steady ms (a test passes its own clock).
  void tick(double now_ms);

  // Where log_sink's lines go before the tick reads them
  LogRing& logs() { return logs_; }

  // What the engine holds (Engine::View, read by the server): `engine` tells one loaded model from the next, so the
  // draft counters start again at a load
  struct EngineSample {
    const void* engine = nullptr;
    size_t budget = 0, allocated = 0, weights = 0, kv = 0;
    std::vector<int> slot_tokens;
    bool drafts = false;
  };
  // Asked every tick: true with `out` filled while an engine is there to show (the server: state ready, a native
  // engine), false otherwise. Each true tick publishes an engine event and is what engine() returns.
  void set_engine(std::function<bool(EngineSample& out)> sample);
  // status.engine as of the last tick, null when that tick had none
  nlohmann::json engine() const;

 private:
  Events& bus_;
  const int tick_ms_;
  LogRing logs_;
  mutable std::mutex mu_;
  std::vector<std::shared_ptr<Request>> active_;
  uint64_t next_id_ = 0;
  // the engine block (under mu_): the sampler, the last tick's block, and the draft counters of the requests that
  // finished on `drafts_engine_`
  std::function<bool(EngineSample&)> sample_engine_;
  nlohmann::json engine_;
  const void* drafts_engine_ = nullptr;
  long drafts_cycles_ = 0, drafts_tokens_ = 0;
  void tick_engine();
};

double steady_ms();
// A state event's `model`: {id, context_length, slots}, or null when there is none yet (an empty id, no context)
nlohmann::json model_json(const std::string& id, int context_length, int slots);
// Publishes a state event on the bus start_activity() made (nothing before it): `message` only when not empty
void publish_state(const char* state, const nlohmann::json& model, const std::string& message = "");
// The last state event's {state, model} ({starting, null} before the first), with `seq`, the bus's last event then: what
// GET /status reports
nlohmann::json last_state();

// `kapitsa serve`'s Activity and its bus, made once for the process (threads that log may outlive run_service, so
// they are never freed): a fresh run_id, log_sink fanned out to the bus (the sink set before still gets every line,
// stderr when there was none), and a thread that ticks every tick_ms. Later calls return the same one.
Activity& start_activity();
// It, or nullptr before start_activity()
Activity* activity();
// The bus start_activity() made, or nullptr
Events* events();

}  // namespace ni::server
