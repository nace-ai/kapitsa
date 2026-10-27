#include "engine/execution_session_internal.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unistd.h>

static thread_local int allocation_budget = -1;
void* operator new(std::size_t n) {
  if (allocation_budget == 0) throw std::bad_alloc();
  if (allocation_budget > 0) --allocation_budget;
  if (auto* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }

using namespace ni;
using detail::ChatObserver;
using detail::StopToken;
using Factory = detail::ExecutionFactory;
static_assert(std::is_same_v<decltype(&ExecutionSession::start), void (ExecutionSession::*)(ChatRequest)>,
              "public start must not accept caller-supplied worker observers");
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " #x "\n"; std::abort(); } } while (false)

struct Gate {
  std::mutex mu;
  std::condition_variable cv;
  bool ready = false;
  void open() { std::lock_guard<std::mutex> lk(mu); ready = true; cv.notify_all(); }
  void wait() { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return ready; }); }
};

struct Probe : ChatObserver {
  std::function<bool(StopToken)> on_accept = [](auto) { return true; };
  std::function<void()> on_prompt = [] {}, on_cancel = [] {};
  std::function<bool()> on_delta = [] { return true; };
  std::function<void(const ChatResult&, StopToken)> on_finish = [](auto&, auto) {};
  std::atomic<int> accepted_count{0}, prompt_count{0}, delta_count{0}, cancelled_count{0}, finished_count{0};
  bool accepted(const ChatResult&, StopToken stop) override { ++accepted_count; return on_accept(stop); }
  void prompt(const ChatResult&) override { ++prompt_count; on_prompt(); }
  bool delta(const ChatDelta&) override { ++delta_count; return on_delta(); }
  void cancelled() noexcept override { ++cancelled_count; on_cancel(); }
  void finished(const ChatResult& r, StopToken stop) override { ++finished_count; on_finish(r, stop); }
};

static ChatResult result(std::string reason = "stop") { ChatResult r; r.finish_reason = std::move(reason); return r; }
template <class F> static void logic_error(F f) {
  bool threw = false;
  try { f(); } catch (const std::logic_error&) { threw = true; }
  CHECK(threw);
}

static void prestart() {
  int calls = 0;
  auto p = std::make_shared<Probe>();
  auto s = Factory::bind([&](auto&, auto&, auto) { ++calls; return result(); }, p);
  logic_error([&] { s->wait(); });
  s->cancel();
  s->cancel();
  s->start({});
  CHECK(s->wait().finish_reason == "cancelled");
  CHECK(&s->wait() == &s->wait());
  CHECK(calls == 0 && p->cancelled_count == 1 && p->finished_count == 1);
  logic_error([&] { s->start({}); });
}

static void never_started() {
  auto p = std::make_shared<Probe>();
  { auto s = Factory::bind([](auto&, auto&, auto) { return result(); }, p); }
  CHECK(p->accepted_count == 0 && p->prompt_count == 0 && p->delta_count == 0);
  CHECK(p->cancelled_count == 0 && p->finished_count == 0);
}

static void normal_completion() {
  auto p = std::make_shared<Probe>();
  {
    auto s = Factory::bind([](auto&, ChatObserver& events, StopToken stop) {
      CHECK(events.accepted({}, stop));
      events.prompt({});
      CHECK(events.delta({}));
      return result();
    }, p);
    s->start({});
    CHECK(s->wait().finish_reason == "stop");
    CHECK(p->accepted_count == 1 && p->prompt_count == 1 && p->delta_count == 1);
    CHECK(p->cancelled_count == 0 && p->finished_count == 1);
    s->cancel();
    s->cancel();
    CHECK(p->cancelled_count == 0 && p->finished_count == 1);
  }
  CHECK(p->cancelled_count == 0 && p->finished_count == 1);
}

static void accepted_gate(bool abandon) {
  Gate entered;
  std::mutex mu;
  std::condition_variable cv;
  auto p = std::make_shared<Probe>();
  p->on_cancel = [&] { std::lock_guard<std::mutex> lk(mu); cv.notify_all(); };
  p->on_accept = [&](StopToken stop) {
    std::unique_lock<std::mutex> lk(mu);
    entered.open();
    cv.wait(lk, [&] { return stop.stop_requested(); });
    return false;
  };
  std::atomic<bool> returned{false};
  auto s = Factory::bind([&](auto&, ChatObserver& events, StopToken stop) {
    CHECK(!events.accepted({}, stop));
    returned = true;
    return result("cancelled");
  }, p);
  p->on_finish = [&](auto&, auto) { CHECK(returned); };
  s->start({});
  entered.wait();
  if (abandon) s.reset();
  else {
    logic_error([&] { s->start({}); });
    s->cancel();
    CHECK(s->wait().finish_reason == "cancelled");
  }
  CHECK(returned && p->cancelled_count == 1 && p->finished_count == 1);
}

static void post_cancel() {
  Gate entered;
  auto p = std::make_shared<Probe>();
  p->on_finish = [&](auto&, auto stop) {
    CHECK(stop.stop_requested());
    CHECK(p->accepted_count == 1 && p->prompt_count == 1 && p->delta_count == 1);
    CHECK(p->cancelled_count == 1);
  };
  auto s = Factory::bind([&](auto&, ChatObserver& events, StopToken stop) {
    CHECK(events.accepted({}, stop));
    events.prompt({});
    CHECK(events.delta({}));
    entered.open();
    while (!stop.stop_requested()) std::this_thread::yield();
    CHECK(!events.accepted({}, stop));
    events.prompt({});
    CHECK(!events.delta({}));
    return result("cancelled");
  }, p);
  s->start({});
  entered.wait();
  s->cancel();
  s->cancel();
  CHECK(s->wait().finish_reason == "cancelled");
  CHECK(p->accepted_count == 1 && p->prompt_count == 1 && p->delta_count == 1);
  CHECK(p->cancelled_count == 1 && p->finished_count == 1);
}

static void terminal_order() {
  Gate entered, release;
  auto p = std::make_shared<Probe>();
  auto s = Factory::bind([](auto&, auto&, StopToken stop) {
    while (!stop.stop_requested()) std::this_thread::yield();
    return result("cancelled");
  }, p);
  auto* raw = s.get();
  p->on_cancel = [&] {
    logic_error([&] { raw->wait(); });
    entered.open();
    release.wait();
  };
  s->start({});
  std::thread canceller([&] { s->cancel(); });
  entered.wait();
  CHECK(p->cancelled_count == 1 && p->finished_count == 0);
  release.open();
  canceller.join();
  CHECK(s->wait().finish_reason == "cancelled");
  CHECK(p->cancelled_count == 1 && p->finished_count == 1);
}

static void drain_and_lifetime() {
  Gate callback_entered, callback_release, draining, drain_release, terminal_entered, terminal_release;
  std::atomic<bool> work_returned{false}, terminal_returned{false};
  auto owner = std::make_shared<int>(42);
  std::weak_ptr<int> owner_alive = owner;
  auto live = std::make_shared<LiveDecode>();
  std::weak_ptr<LiveDecode> buffer_alive = live;
  auto p = std::make_shared<Probe>();
  auto s = Factory::bind([&, owner](const ChatRequest& r, ChatObserver& events, StopToken stop) {
    CHECK(*owner == 42 && r.live && !buffer_alive.expired());
    events.delta({});
    draining.open();
    drain_release.wait();
    CHECK(stop.stop_requested() && !buffer_alive.expired());
    work_returned = true;
    return result("cancelled");
  }, p);
  owner.reset();
  p->on_delta = [&] { callback_entered.open(); callback_release.wait(); CHECK(!buffer_alive.expired()); return true; };
  p->on_cancel = [&] { CHECK(!owner_alive.expired() && !buffer_alive.expired()); };
  p->on_finish = [&](auto&, auto) {
    CHECK(work_returned && p->cancelled_count == 1 && !owner_alive.expired());
    terminal_entered.open();
    terminal_release.wait();
    terminal_returned = true;
  };
  ChatRequest request;
  request.live = std::move(live);
  s->start(std::move(request));
  callback_entered.wait();
  std::atomic<int> joined{0};
  auto waiter = [&] { CHECK(s->wait().finish_reason == "cancelled"); CHECK(terminal_returned); ++joined; };
  std::thread a(waiter), b(waiter), canceller([&] { for (int i = 0; i < 32; ++i) s->cancel(); });
  canceller.join();
  callback_release.open();
  draining.wait();
  CHECK(joined == 0 && p->finished_count == 0);
  drain_release.open();
  terminal_entered.wait();
  CHECK(joined == 0 && !buffer_alive.expired());
  terminal_release.open();
  a.join();
  b.join();
  CHECK(joined == 2 && p->finished_count == 1);
  CHECK(!buffer_alive.expired() && !owner_alive.expired());
  p.reset();
  s.reset();
  CHECK(buffer_alive.expired() && owner_alive.expired());
}

static void exceptions() {
  for (int which = 0; which < 6; ++which) {
    bool drained = false;
    auto p = std::make_shared<Probe>();
    if (which == 1) p->on_accept = [](auto) -> bool { throw std::runtime_error("accepted"); };
    if (which == 2) p->on_prompt = [] { throw std::runtime_error("prompt"); };
    if (which == 3) p->on_delta = []() -> bool { throw std::runtime_error("delta"); };
    p->on_finish = [&](auto&, auto) { CHECK(drained || which == 0 || which == 5); if (which == 4) throw std::runtime_error("terminal"); };
    auto s = Factory::bind([&](auto&, ChatObserver& events, StopToken stop) {
      if (which == 0) throw std::runtime_error("work");
      if (which == 5) throw 7;
      events.accepted({}, stop);
      events.prompt({});
      events.delta({});
      CHECK(stop.stop_requested() == (which >= 1 && which <= 3));
      drained = true;
      return result();
    }, p);
    s->start({});
    CHECK(s->wait().finish_reason == "error" && !s->wait().error.empty());
    CHECK(p->finished_count == 1);
  }
}

static void error_without_allocations() {
  const auto error = std::make_exception_ptr(std::runtime_error(std::string(1024, 'x')));
  for (int which = 0; which < 4; ++which) {
    auto fail = [error] { allocation_budget = 0; std::rethrow_exception(error); };
    auto p = std::make_shared<Probe>();
    if (which == 1) p->on_delta = [fail]() -> bool { fail(); return false; };
    p->on_finish = [&, fail](const ChatResult& r, auto) {
      if (which >= 2) fail();
      allocation_budget = -1;
      CHECK(r.finish_reason == "error" && !r.error.empty());
    };
    auto s = Factory::bind([which, fail](auto&, ChatObserver& events, auto) -> ChatResult {
      if (which == 0 || which == 3) fail();
      events.delta({});
      return result();
    }, p);
    s->start({});
    CHECK(s->wait().finish_reason == "error" && !s->wait().error.empty());
    CHECK(p->finished_count == 1);
  }
}

static void failed_launch() {
  bool called = false;
  auto p = std::make_shared<Probe>();
  p->on_finish = [](auto&, auto) { allocation_budget = -1; };
  auto s = Factory::bind([&](auto&, auto&, auto) { called = true; return result(); }, p);
  ChatRequest request;
  allocation_budget = 0;
  s->start(std::move(request));
  allocation_budget = -1;
  CHECK(s->wait().finish_reason == "error" && !s->wait().error.empty());
  CHECK(!called && p->finished_count == 1);
}

static void self_wait() {
  auto p = std::make_shared<Probe>();
  auto s = Factory::bind([](auto&, ChatObserver& events, auto) { events.delta({}); return result(); }, p);
  auto* raw = s.get();
  p->on_delta = [raw] { logic_error([&] { raw->wait(); }); raw->cancel(); return false; };
  p->on_cancel = [raw] { logic_error([&] { raw->wait(); }); };
  s->start({});
  s->wait();
  CHECK(p->finished_count == 1);
}

static void repeated_requests_and_startup_race() {
  for (int i = 0; i < 64; ++i) {
    Gate go;
    auto p = std::make_shared<Probe>();
    int order = 0;
    p->on_accept = [&](auto) { CHECK(order++ == 0); return true; };
    p->on_prompt = [&] { CHECK(order++ == 1); };
    p->on_delta = [&] { CHECK(order++ == 2); return true; };
    auto s = Factory::bind([&](const ChatRequest& r, ChatObserver& events, StopToken stop) {
      CHECK(r.seed == static_cast<unsigned>(i));
      ChatResult out = result();
      out.cached_tokens = 9;
      out.slot = 2;
      out.slot_gen = 7;
      if (!events.accepted(out, stop)) return result("cancelled");
      events.prompt(out);
      events.delta({});
      return out;
    }, p);
    ChatRequest r;
    r.seed = i;
    std::thread starter([&] { go.wait(); s->start(r); });
    std::thread cancel([&] { go.wait(); if (i % 2) s->cancel(); });
    go.open();
    starter.join();
    cancel.join();
    const auto& out = s->wait();
    CHECK(out.finish_reason == "stop" || out.finish_reason == "cancelled");
    if (!(i % 2)) CHECK(order == 3 && out.cached_tokens == 9 && out.slot == 2 && out.slot_gen == 7 && p->cancelled_count == 0);
    const auto cancellations = p->cancelled_count.load();
    s->cancel();
    CHECK(&s->wait() == &out && p->finished_count == 1 && p->cancelled_count == cancellations);
  }
}

// Name each sub-test on stderr as it starts, so a hang under the one alarm shows which one stalled.
#define RUN(test) do { std::cerr << "execution_session_test: " #test "\n"; test; } while (false)

int main() {
  alarm(30);
  RUN(prestart());
  RUN(never_started());
  RUN(normal_completion());
  RUN(accepted_gate(false));
  RUN(accepted_gate(true));
  RUN(post_cancel());
  RUN(terminal_order());
  RUN(drain_and_lifetime());
  RUN(exceptions());
  RUN(error_without_allocations());
  RUN(failed_launch());
  RUN(self_wait());
  RUN(repeated_requests_and_startup_race());
  std::cout << "execution_session_test: ok\n";
}
