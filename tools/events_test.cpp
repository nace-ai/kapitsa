// The server's event bus (src/server/events.h) and what feeds it (src/server/activity.h): numbering, replay and the reset cases of a Last-Event-ID, slow
// readers closed at kMaxUnsent, the stream cap, and log lines through the ring. CPU only.
#include "server/activity.h"
#include "server/events.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <map>
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

// A cursor whose replay would pass kMaxReplay gets `expired`, not a stream closed at once (the client would reconnect
// with the same cursor and loop); a smaller replay still replays
static void big_replay_expires() {
  Events bus(kRun);
  const std::string line(4 * 1024, 'x');
  for (int i = 0; i < 300; ++i) bus.publish("log", {{"level", "info"}, {"message", line}});  // ~1.2 MiB kept
  Resume how;
  auto s = bus.open(kRun + ".1", &how);
  CHECK(s && how == Resume::expired && !s->closed() && !s->next(0));
  auto near = bus.open(kRun + ".250", &how);  // 50 lines, ~200 KiB
  CHECK(near && how == Resume::replay);
  int got = 0;
  while (near->next(0)) ++got;
  CHECK(got == 50 && !near->closed());
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
  {  // prefilled_tokens leaves the cached ones out: LiveDecode::read starts at `cached` (engine.cpp, the prefill start)
    auto c = act.start("openai", "");
    drain(*s);
    c->live->phase = 1, c->live->prompt = 2310, c->live->cached = 2048, c->live->read = 2048;
    act.tick(t + 1600);
    auto q = drain(*s);
    CHECK(q.size() == 2 && q[1].second["prefilled_tokens"] == 0 && q[1].second["cached_tokens"] == 2048);
    c->live->phase = 2, c->live->read = 2310;
    act.tick(t + 1700);
    q = drain(*s);
    CHECK(q.size() == 2 && q[1].second["prefilled_tokens"] == 262 && q[1].second["prompt_tokens"] == 2310);
    act.finish(c, ChatResult{});
    drain(*s);
  }
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

// The engine block: none while the sampler says so, a busy slot from its request's progress, the draft counters
// summed over finished and running requests and started again when another engine loads
static void engine_block() {
  Events bus(kRun);
  Activity act(bus, 500);
  Resume how;
  auto s = bus.open("", &how);
  int engine_events = 0;
  const auto count_engine = [&] { while (auto e = s->next(0)) engine_events += e->type == "engine"; };
  act.tick(steady_ms());
  CHECK(act.engine().is_null());
  int a = 0, b = 0;
  const void* which = &a;
  size_t budget = 40'000'000'000;
  bool ssd_on = false;
  act.set_engine([&](Activity::EngineSample& e) {
    e = {which, budget, 20'000'000'000, 17'000'000'000, 1'500'000'000, {0, 300}, true, 1, 700'000'000, 4, 9000};
    if (ssd_on) e.ssd = true, e.ssd_entries = 14, e.ssd_bytes = 9'100'000'000, e.ssd_limit = 32'000'000'000,
                e.ssd_read_bytes = 2'800'000'000, e.ssd_written_bytes = 6'300'000'000, e.ssd_hits = 5, e.ssd_hit_tokens = 96400;
    return true;
  });
  auto r = act.start("openai", "");
  r->live->slot = 1, r->live->phase = 2, r->live->read = 100, r->live->tokens = 5, r->live->spec = 2, r->live->cycles = 2;
  count_engine();
  engine_events = 0;
  act.tick(steady_ms());
  json e = act.engine();
  CHECK(e["memory"]["budget_bytes"] == budget && e["memory"]["kv_bytes"] == 1'500'000'000);
  CHECK(e["slots"][0] == json({{"index", 0}, {"busy", false}, {"tokens", 0}}));
  CHECK(e["slots"][1] == json({{"index", 1}, {"busy", true}, {"tokens", 105}, {"request_id", r->id}}));
  CHECK(e["drafts"] == json({{"cycles", 2}, {"tokens", 4}}));  // (the first token is the prefill's)
  // the RAM cache's block, no SSD one while the engine has none (not zeros)
  CHECK(e["cache"] == json({{"ram", {{"entries", 1}, {"bytes", 700'000'000}, {"hits", 4}, {"hit_tokens", 9000}}}}));
  act.finish(r, ChatResult{});
  act.tick(steady_ms());
  e = act.engine();
  CHECK(e["slots"][1] == json({{"index", 1}, {"busy", false}, {"tokens", 300}}) && e["drafts"]["cycles"] == 2);
  ssd_on = true;
  act.tick(steady_ms());
  CHECK(act.engine()["cache"]["ssd"] == json({{"entries", 14}, {"bytes", 9'100'000'000}, {"limit_bytes", 32'000'000'000},
                                              {"read_bytes", 2'800'000'000}, {"written_bytes", 6'300'000'000},
                                              {"hits", 5}, {"hit_tokens", 96400}}));
  ssd_on = false;
  which = &b;  // another model loaded: its counters start at 0
  act.tick(steady_ms());
  CHECK(act.engine()["drafts"] == json({{"cycles", 0}, {"tokens", 0}}));
  count_engine();
  CHECK(engine_events == 4);
  budget = 0;  // nothing to show
  act.tick(steady_ms());
  CHECK(act.engine().is_null());
  act.set_engine([](Activity::EngineSample&) { return false; });
  act.tick(steady_ms());
  count_engine();
  CHECK(act.engine().is_null() && engine_events == 4);
  // unloaded, then loaded again at the same address (a reload often gets the freed block back): the counters still
  // start at 0, and a request that finished while nothing was loaded does not count
  bool loaded = true;
  act.set_engine([&](Activity::EngineSample& e) {
    e = {which, 40'000'000'000, 20'000'000'000, 17'000'000'000, 1'500'000'000, {0, 300}, true};
    return loaded;
  });
  act.tick(steady_ms());
  const auto drafted = [&](int cycles) {
    auto q = act.start("openai", "");
    q->live->slot = 0, q->live->phase = 2, q->live->tokens = 2 * cycles + 1, q->live->spec = 2, q->live->cycles = cycles;
    act.finish(q, ChatResult{});
  };
  drafted(3);
  act.tick(steady_ms());
  CHECK(act.engine()["drafts"]["cycles"] == 3);
  loaded = false;
  act.tick(steady_ms());
  drafted(4);
  loaded = true;
  act.tick(steady_ms());
  CHECK(act.engine()["drafts"] == json({{"cycles", 0}, {"tokens", 0}}));
}

// --print: every kind of event the server publishes, one SSE frame per line as JSON, for tests/test_events_schema.py
static int print_frames() {
  Events bus(kRun);
  Resume how;
  auto s = bus.open("", &how);
  const json m = model_json("qwen3.6-35b-a3b", 262144, 4);
  CHECK(model_json("", 1, 1).is_null() && model_json("x", 0, 1).is_null());
  for (const char* st : {"starting", "loading"}) bus.publish("state", {{"state", st}, {"model", nullptr}});
  for (const char* st : {"loading", "ready", "switching", "released", "stopping"}) bus.publish("state", {{"state", st}, {"model", m}});
  bus.publish("state", {{"state", "released"}, {"model", m}, {"message", "loading the model again failed: out of memory"}});
  bus.publish("state", {{"state", "stopping"}, {"model", nullptr}});
  Activity act(bus, 500);
  auto r = act.start("openai", "curl/8.7.1");
  auto r2 = act.start("ollama", "");
  r->live->phase = 2, r->live->prompt = 100, r->live->cached = 40, r->live->read = 100, r->live->tokens = 12;
  act.tick(r->t0 + 500);
  ChatResult ok;
  ok.finish_reason = "length", ok.prompt_tokens = 100, ok.cached_tokens = 40, ok.completion_tokens = 12;
  ok.prefill_ms = 30, ok.decode_ms = 110;
  act.finish(r, ok);
  ChatResult bad;
  bad.finish_reason = "error", bad.error = "the prompt does not fit";
  act.finish(r2, bad);
  for (int lvl = 0; lvl < 4; ++lvl) act.logs().push(lvl % 3, lvl == 3 ? "warning: a warning" : "a line");
  auto r3 = act.start("anthropic", "");
  r3->live->slot = 1, r3->live->phase = 2, r3->live->read = 80, r3->live->tokens = 9, r3->live->spec = 2, r3->live->cycles = 3;
  act.set_engine([](Activity::EngineSample& e) {
    e = {&e, 40'000'000'000, 20'000'000'000, 17'000'000'000, 1'500'000'000, {120, 0}, true, 1, 700'000'000, 4, 9000,
         true, 14, 9'100'000'000, 32'000'000'000, 2'800'000'000, 6'300'000'000, 5, 96400};
    return true;
  });
  act.tick(r->t0 + 1000);
  while (auto e = s->next(0)) std::cout << json{{"id", bus.id(e->seq)}, {"type", e->type}, {"data", json::parse(e->data)}}.dump() << "\n";
  return 0;
}

// One thread ticks while another starts and finishes requests: each one's events must be started, progress*, finished
static void request_order() {
  Events bus(kRun);
  Activity act(bus, 100);
  Resume how;
  auto s = bus.open("", &how);
  std::atomic<bool> done{false};
  std::map<std::string, int> seen;  // 1 started, 2 finished
  int bad = 0;
  std::thread reader([&] {
    for (decltype(s->next(0)) e; (e = s->next(20)) || !done.load();) {
      if (!e) continue;
      int& st = seen[json::parse(e->data).value("request_id", "")];
      bad += e->type == "request_started" ? st != 0 : st != 1;
      if (e->type != "request_progress") st = e->type == "request_started" ? 1 : 2;
    }
  });
  std::thread ticker([&] { while (!done.load()) act.tick(steady_ms()); });
  for (int i = 0; i < 2000; ++i) act.finish(act.start("openai", ""), ChatResult{});
  done = true, ticker.join(), reader.join();
  for (const auto& [id, st] : seen) bad += st != 2;
  CHECK(!s->closed() && bad == 0 && seen.size() == 2000);
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--print") return print_frames();
  activity_events();
  numbering_and_live();
  replay_and_resets();
  slow_reader_closed();
  big_replay_expires();
  request_order();
  stream_cap_and_close();
  stream_outlives_bus();
  logs();
  engine_block();
  std::cout << "events_test: ok\n";
}
