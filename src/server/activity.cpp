#include "server/activity.h"

#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>

namespace ni::server {

double steady_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

namespace {

int count(double v) { return v > 0 ? int(std::lround(v)) : 0; }
double rate(double tokens, double ms) { return tokens > 0 && ms > 0 ? tokens * 1000 / ms : 0; }

// LiveDecode::phase (0 waiting, 1 reading the prompt, 2 writing, 3 done) as request_progress's phase
const char* phase_name(int p) { return p <= 0 ? "waiting" : p == 1 ? "prefill" : "writing"; }

}  // namespace

Activity::Activity(Events& bus, int tick_ms) : bus_(bus), tick_ms_(std::clamp(tick_ms, 100, 10000)) {}

std::shared_ptr<Activity::Request> Activity::start(const std::string& api, const std::string& user_agent) {
  auto r = std::make_shared<Request>();
  r->api = api;
  r->client = user_agent.substr(0, 128);
  r->t0 = r->tick_at = steady_ms();
  {
    std::lock_guard<std::mutex> lk(mu_);
    r->id = "req-" + std::to_string(++next_id_);
    r->sample = {{"request_id", r->id}, {"api", r->api}, {"phase", "waiting"}, {"prompt_tokens", 0},
                 {"cached_tokens", 0}, {"prefilled_tokens", 0}, {"completion_tokens", 0}, {"elapsed_ms", 0},
                 {"tokens_per_second", 0}};
    if (!r->client.empty()) r->sample["client"] = r->client;
    active_.push_back(r);
    // Published under mu_, as finish() and tick() do: a request's events reach the bus as started, progress...,
    // finished (Events::publish takes only the bus's own mutex and never calls back here)
    nlohmann::json data = {{"request_id", r->id}, {"api", r->api}};
    if (!r->client.empty()) data["client"] = r->client;
    bus_.publish("request_started", data);
  }
  return r;
}

void Activity::finish(const std::shared_ptr<Request>& r, const ChatResult& res) {
  if (!r) return;
  static const char* const kReasons[] = {"stop", "length", "tool_calls", "cancelled", "error"};
  std::string reason = res.finish_reason;
  if (std::find(std::begin(kReasons), std::end(kReasons), reason) == std::end(kReasons))
    reason = res.error.empty() ? "stop" : "error";
  // the decode rate after the first token, as the request log's; the prefill rate of the uncached prompt tokens
  const int uncached = std::max(res.prompt_tokens - res.cached_tokens, 0);
  nlohmann::json data = {{"request_id", r->id},
                         {"finish_reason", reason},
                         {"prompt_tokens", std::max(res.prompt_tokens, 0)},
                         {"cached_tokens", std::max(res.cached_tokens, 0)},
                         {"completion_tokens", std::max(res.completion_tokens, 0)},
                         {"queue_ms", count(res.queue_ms)},
                         {"prefill_ms", count(res.prefill_ms)},
                         {"decode_ms", count(res.decode_ms)},
                         {"tokens_per_second", rate(res.completion_tokens - 1, res.decode_ms)},
                         {"prefill_tokens_per_second", rate(uncached, res.prefill_ms)}};
  if (reason == "error") data["error"] = (res.error.empty() ? std::string("error") : res.error).substr(0, 4096);
  std::lock_guard<std::mutex> lk(mu_);  // (see start())
  auto it = std::find(active_.begin(), active_.end(), r);
  if (it == active_.end()) return;  // finished already
  active_.erase(it);
  bus_.publish("request_finished", data);
}

nlohmann::json Activity::requests() const {
  nlohmann::json out = nlohmann::json::array();
  std::lock_guard<std::mutex> lk(mu_);
  for (const auto& r : active_) out.push_back(r->sample);
  return out;
}

void Activity::tick(double now_ms) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& r : active_) {
      const LiveDecode& l = *r->live;
      const int tokens = l.tokens.load(std::memory_order_relaxed);
      // LiveDecode::read counts the cached tokens too (it starts at `cached`); the spec's prefilled_tokens does not
      const int cached = std::max(l.cached.load(std::memory_order_relaxed), 0);
      const int prefilled = std::max(l.read.load(std::memory_order_relaxed) - cached, 0);
      const double since = now_ms - r->tick_at;
      nlohmann::json s = {{"request_id", r->id},
                          {"api", r->api},
                          {"phase", phase_name(l.phase.load(std::memory_order_relaxed))},
                          {"prompt_tokens", std::max(l.prompt.load(std::memory_order_relaxed), 0)},
                          {"cached_tokens", cached},
                          {"prefilled_tokens", prefilled},
                          {"completion_tokens", std::max(tokens, 0)},
                          {"elapsed_ms", count(now_ms - r->t0)},
                          {"tokens_per_second", rate(tokens - r->tokens_at_tick, since)}};
      if (!r->client.empty()) s["client"] = r->client;
      r->tokens_at_tick = tokens;
      r->tick_at = now_ms;
      r->sample = s;
      bus_.publish("request_progress", s);  // under mu_ (see start())
    }
  }
  uint64_t dropped = 0;
  const auto lines = logs_.drain(&dropped);
  if (dropped)  // the oldest went: say so before the lines that are left
    bus_.publish("log", {{"level", "warn"},
                         {"message", "warning: " + std::to_string(dropped) + " log lines dropped (more than the ring holds in one tick)"}});
  for (const auto& line : lines) bus_.publish("log", log_event(line));
}

namespace {

std::atomic<Activity*> g_activity{nullptr};
std::atomic<Events*> g_events{nullptr};
LogSink g_prev_sink = nullptr;
std::mutex g_state_mu;
nlohmann::json g_state = {{"state", "starting"}, {"model", nullptr}};

void bus_sink(int level, const char* msg) {
  if (Activity* a = g_activity.load(std::memory_order_acquire)) a->logs().push(level, msg);
  if (g_prev_sink) return g_prev_sink(level, msg);
  fprintf(stderr, level == 0 ? "[kapitsa] fatal: %s\n" : "[kapitsa] %s\n", msg);
}

std::string new_run_id() {
  std::random_device rd;
  char b[17];
  snprintf(b, sizeof b, "%08x%08x", rd(), rd());
  return b;
}

}  // namespace

Activity& start_activity() {
  static Activity* a = [] {
    auto* bus = new Events(new_run_id());
    auto* act = new Activity(*bus);
    g_events.store(bus, std::memory_order_release);
    g_activity.store(act, std::memory_order_release);
    g_prev_sink = log_sink();
    log_sink() = bus_sink;
    std::thread([act] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(act->tick_ms()));
        act->tick(steady_ms());
      }
    }).detach();
    return act;
  }();
  return *a;
}

nlohmann::json model_json(const std::string& id, int context_length, int slots) {
  if (id.empty() || context_length < 1 || slots < 1) return nullptr;
  return {{"id", id}, {"context_length", context_length}, {"slots", slots}};
}

void publish_state(const char* state, const nlohmann::json& model, const std::string& message) {
  Events* bus = g_events.load(std::memory_order_acquire);
  if (!bus) return;
  nlohmann::json d = {{"state", state}, {"model", model}};
  if (!message.empty()) d["message"] = message.substr(0, 4096);
  std::lock_guard<std::mutex> lk(g_state_mu);  // (the last state is the bus's last state event)
  bus->publish("state", d);
  g_state = {{"state", state}, {"model", model}};
}

nlohmann::json last_state() {
  std::lock_guard<std::mutex> lk(g_state_mu);
  return g_state;
}

Activity* activity() { return g_activity.load(std::memory_order_acquire); }
Events* events() { return g_events.load(std::memory_order_acquire); }

}  // namespace ni::server
