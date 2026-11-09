// The server's event bus (src/server/events.h) and what feeds it (src/server/activity.h): numbering, replay and the reset cases of a Last-Event-ID, slow
// readers closed at kMaxUnsent, the stream cap, and log lines through the ring. CPU only.
#include "server/activity.h"
#include "server/events.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ni::server;
using ni::ChatResult;
using nlohmann::json;
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " #x "\n"; std::abort(); } } while (false)

static const std::string kRun = "7f3a9c2e5b1d4e60";

static void numbering_and_live() {
  Events bus(kRun);
  CHECK(bus.publish("state", {{"state", "starting"}}) == 1);
  CHECK(bus.id(1) == kRun + ".1");
  Resume how;
  auto s = bus.open("", &how);
  CHECK(s && how == Resume::live && s->start() == 1);
  CHECK(!s->next(0));  // live: not the event before it opened
  bus.publish("log", {{"level", "info"}, {"message", "hi"}});
  auto e = s->next(1000);
  CHECK(e && e->seq == 2 && e->type == "log" && json::parse(e->data)["message"] == "hi");
}

static void replay_and_resets() {
  Events bus(kRun, 4);
  for (int i = 0; i < 10; ++i) bus.publish("log", {{"i", i}});  // keeps seq 7..10
  Resume how;
  auto s = bus.open(kRun + ".7", &how);
  CHECK(s && how == Resume::replay && s->start() == 7);
  for (uint64_t want : {8, 9, 10}) {
    auto e = s->next(0);
    CHECK(e && e->seq == want);
  }
  CHECK(!s->next(0));
  CHECK(bus.open(kRun + ".6", &how) && how == Resume::replay);  // 7.. is all there
  CHECK(bus.open(kRun + ".5", &how) && how == Resume::expired);
  CHECK(bus.open(kRun + ".10", &how) && how == Resume::replay);
  CHECK(bus.open(kRun + ".11", &how) && how == Resume::invalid_cursor);  // never existed
  CHECK(bus.open("0123456789abcdef.3", &how) && how == Resume::run_changed);
  for (const char* bad : {"x", "7f3a9c2e5b1d4e60", "7f3a9c2e5b1d4e60.", "7f3a9c2e5b1d4e60.0", "7f3a9c2e5b1d4e60.07",
                          "7F3A9C2E5B1D4E60.3", "abc.3", "7f3a9c2e5b1d4e60.3x"}) {
    CHECK(bus.open(bad, &how) && how == Resume::invalid_cursor);
  }
  CHECK(std::string(reset_reason(Resume::expired)) == "expired" && std::string(reset_reason(Resume::replay)).empty());
  auto r = bus.open(kRun + ".5", &how);
  CHECK(r->start() == 10);  // a reset continues from the newest event: the reset's id is status.last_event_id
}

static void slow_reader_closed() {
  Events bus(kRun);
  Resume how;
  auto slow = bus.open("", &how), fast = bus.open("", &how);
  const std::string chunk(64 * 1024, 'x');
  size_t got = 0;
  for (int i = 0; i < 20; ++i) {  // 20 x 64 KiB: more than 1 MiB for the reader that never reads
    bus.publish("log", {{"message", chunk}});
    while (fast->next(0)) ++got;
  }
  CHECK(got == 20 && !fast->closed());
  CHECK(slow->closed() && !slow->next(0));
}

static void stream_cap_and_close() {
  Events bus(kRun);
  bus.publish("state", json::object());
  Resume how;
  std::vector<std::shared_ptr<Events::Stream>> v;
  for (int i = 0; i < Events::kMaxStreams; ++i) v.push_back(bus.open("", &how));
  CHECK(v.back() && !bus.open("", &how));  // the 17th: 503 busy
  v.pop_back();
  CHECK(bus.open("", &how));  // a closed stream frees its place
  std::atomic<bool> woke{false};
  std::thread t([&] {
    CHECK(!v[0]->next(10000));
    woke = true;
  });
  bus.close();
  t.join();
  CHECK(woke && v[0]->closed() && !bus.open("", &how));
}

static void stream_outlives_bus() {
  std::shared_ptr<Events::Stream> s;
  {
    Events bus(kRun);
    Resume how;
    s = bus.open("", &how);
  }
  CHECK(s->closed() && !s->next(0));
}

static void logs() {
  LogRing ring(3);
  ring.push(1, "a");
  ring.push(1, "warning: b");
  ring.push(2, "c");
  ring.push(0, "d");  // the ring holds 3: "a" goes
  uint64_t dropped = 0;
  auto lines = ring.drain(&dropped);
  CHECK(dropped == 1 && lines.size() == 3 && lines[0].message == "warning: b" && lines[2].message == "d");
  CHECK(log_event(lines[0])["level"] == "warn" && log_event(lines[1])["level"] == "debug" &&
        log_event(lines[2])["level"] == "error");
  CHECK(ring.drain(&dropped).empty() && dropped == 0);
  const std::string big(LogRing::kMaxLine + 100, 'y');
  ring.push(1, big.c_str());
  CHECK(ring.drain(nullptr)[0].message.size() == LogRing::kMaxLine);
  // many threads at once: every line is kept or counted as dropped
  LogRing r2(64);
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t)
    ts.emplace_back([&] {
      for (int i = 0; i < 1000; ++i) r2.push(1, "line");
    });
  size_t kept = 0;
  uint64_t lost = 0;
  for (int i = 0; i < 100; ++i) {
    uint64_t d = 0;
    kept += r2.drain(&d).size();
    lost += d;
  }
  for (auto& t : ts) t.join();
  uint64_t d = 0;
  kept += r2.drain(&d).size();
  lost += d;
  CHECK(kept + lost == 8000);
}

// Every event the stream has now, as (type, data)
static std::vector<std::pair<std::string, json>> drain(Events::Stream& s) {
  std::vector<std::pair<std::string, json>> out;
  while (auto e = s.next(0)) out.emplace_back(e->type, json::parse(e->data));
  return out;
}

static void activity_events() {
  Events bus(kRun);
  Resume how;
  auto s = bus.open("", &how);
  Activity act(bus, 500);
  auto r = act.start("openai", "curl/8.7.1");
  auto ev = drain(*s);
  CHECK(ev.size() == 1 && ev[0].first == "request_started" && ev[0].second["request_id"] == "req-1" &&
        ev[0].second["client"] == "curl/8.7.1");
  CHECK(act.requests().size() == 1 && act.requests()[0]["phase"] == "waiting");
  // prefill, then 50 tokens in the first tick and 20 in the second: each tick's rate is its own tokens
  r->live->phase = 1, r->live->prompt = 3904, r->live->read = 2816;
  const double t = r->t0;
  act.tick(t + 500);
  ev = drain(*s);
  CHECK(ev.size() == 1 && ev[0].first == "request_progress" && ev[0].second["phase"] == "prefill" &&
        ev[0].second["prefilled_tokens"] == 2816 && ev[0].second["tokens_per_second"] == 0);
  r->live->phase = 2, r->live->read = 3904, r->live->tokens = 50;
  act.tick(t + 1000);
  CHECK(drain(*s)[0].second["tokens_per_second"] == 100.0);
  r->live->tokens = 70;
  act.tick(t + 1500);
  const json p = drain(*s)[0].second;
  CHECK(p["tokens_per_second"] == 40.0 && p["completion_tokens"] == 70 && p["elapsed_ms"] == 1500);
  CHECK(act.requests()[0] == p);  // status reports the last tick's sample, not a fresh read
  ChatResult res;
  res.finish_reason = "stop", res.prompt_tokens = 3904, res.cached_tokens = 0, res.completion_tokens = 71;
  res.prefill_ms = 1952, res.decode_ms = 1400, res.queue_ms = 3.4;
  act.finish(r, res);
  act.finish(r, res);  // once only
  ev = drain(*s);
  CHECK(ev.size() == 1 && ev[0].first == "request_finished" && ev[0].second["tokens_per_second"] == 50.0 &&
        ev[0].second["prefill_tokens_per_second"] == 2000.0 && ev[0].second["queue_ms"] == 3 &&
        !ev[0].second.contains("error"));
  CHECK(act.requests().empty());
  act.tick(t + 2000);
  CHECK(drain(*s).empty());  // nothing active, no log lines: a quiet tick publishes nothing
  // a failed request carries its error; an unknown reason with an error becomes error
  auto r2 = act.start("anthropic", "");
  ChatResult bad;
  bad.finish_reason = "", bad.error = "the prompt does not fit";
  act.finish(r2, bad);
  ev = drain(*s);
  CHECK(!ev[0].second.contains("client") && ev[1].second["finish_reason"] == "error" &&
        ev[1].second["error"] == "the prompt does not fit");
  // log lines reach the bus at the tick, a drop warning first
  LogRing& logs = act.logs();
  for (int i = 0; i < 300; ++i) logs.push(1, ("line " + std::to_string(i)).c_str());
  act.tick(t + 2500);
  ev = drain(*s);
  CHECK(ev.size() == 257 && ev[0].second["level"] == "warn" && ev[1].second["message"] == "line 44" &&
        ev.back().second["message"] == "line 299");
}

int main() {
  activity_events();
  numbering_and_live();
  replay_and_resets();
  slow_reader_closed();
  stream_cap_and_close();
  stream_outlives_bus();
  logs();
  std::cout << "events_test: ok\n";
}
