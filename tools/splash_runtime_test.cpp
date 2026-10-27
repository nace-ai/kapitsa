// CPU test of src/splash/runtime.{h,cpp}: the Runtime spawns this very binary as a fake `splash serve-native` (argv[1] =
// "--fake MODE") that speaks the protocol over its stdin and stdout, and drives it through a request, a cancel, a status,
// a child that dies, one that sends garbage, one that never gets ready, a full request pool, a child that stops reading its
// input, a descendant that keeps its stdout, a Sink that throws, and a log callback that calls back in. No model, no GPU.
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "splash/runtime.h"

using namespace ni::splash;

// ---- the fake engine ----
static void put(std::vector<uint8_t>& b, uint64_t v, int bytes) {
  for (int i = 0; i < bytes; i++) b.push_back(uint8_t(v >> (8 * i)));
}
static void send_frame(uint16_t type, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> f = {'S', 'P', 'L', 'H'};
  put(f, 11, 2);
  put(f, type, 2);
  put(f, payload.size(), 8);
  f.insert(f.end(), payload.begin(), payload.end());
  size_t off = 0;
  while (off < f.size()) {
    ssize_t n = write(1, f.data() + off, f.size() - off);
    if (n <= 0) _exit(4);
    off += size_t(n);
  }
}
static void send_ready(uint32_t conc) {
  std::vector<uint8_t> p;
  put(p, conc, 4); put(p, 4096, 4); put(p, 0, 1);
  send_frame(0x0100, p);
}
static void send_start(uint64_t id) {
  std::vector<uint8_t> p;
  put(p, id, 8); put(p, 0, 4); put(p, 0, 4);
  send_frame(0x0101, p);
}
static void send_tokens(uint64_t id, uint32_t off, const std::vector<uint32_t>& t) {
  std::vector<uint8_t> p;
  put(p, id, 8); put(p, off, 4); put(p, t.size(), 4);
  for (uint32_t x : t) put(p, x, 4);
  send_frame(0x0102, p);
}
static void send_done(uint64_t id, uint8_t reason, uint32_t completion) {
  std::vector<uint8_t> p;
  put(p, id, 8); put(p, reason, 1); put(p, 0, 1); put(p, 5, 4); put(p, completion, 4);
  put(p, 1000, 8); put(p, 2000, 8); put(p, 3000, 8); put(p, 0, 4); put(p, 0, 4); put(p, 0, 4); put(p, 0, 4); put(p, 0, 4);
  send_frame(0x0104, p);
}
static bool read_all(void* b, size_t n) {
  auto* p = static_cast<uint8_t*>(b);
  while (n) {
    ssize_t r = read(0, p, n);
    if (r <= 0) return false;
    p += r; n -= size_t(r);
  }
  return true;
}
static uint64_t le(const uint8_t* p, int n) {
  uint64_t v = 0;
  for (int i = 0; i < n; i++) v |= uint64_t(p[i]) << (8 * i);
  return v;
}

static int fake(const std::string& mode) {
  if (mode == "noready") { sleep(60); return 0; }
  if (mode == "stall") { send_ready(2); sleep(60); return 0; }  // never reads its stdin
  if (mode == "grandchild" || mode == "orphan") {
    // A descendant keeps the stdout pipe open. "grandchild" stays in the group and ignores EOF; "orphan" leaves the group
    // and the session, and the child itself exits on EOF.
    send_ready(2);
    pid_t k = fork();
    if (k == 0) { if (mode == "orphan") setsid(); sleep(mode == "orphan" ? 6 : 60); _exit(0); }
    if (mode == "grandchild") { sleep(60); return 0; }
    uint8_t h[16];
    while (read_all(h, 16)) {}
    return 0;
  }
  if (mode == "truncate") {  // a frame cut short by the child's death
    send_ready(2);
    uint8_t h[16];
    if (read_all(h, 16)) { std::vector<uint8_t> pl(le(h + 8, 8)); read_all(pl.data(), pl.size()); (void)!write(1, "SPLH\x0b\0", 6); }
    _exit(0);
  }
  send_ready(mode == "pool1" ? 1 : 2);
  std::vector<uint64_t> hung;
  while (true) {
    uint8_t h[16];
    if (!read_all(h, 16)) return 0;  // EOF on stdin: drained, exit cleanly
    std::vector<uint8_t> payload(le(h + 8, 8));
    if (!payload.empty() && !read_all(payload.data(), payload.size())) return 0;
    uint16_t type = uint16_t(le(h + 6, 2));
    if (type == 0x0001) {
      uint64_t id = le(payload.data(), 8);
      uint32_t first = uint32_t(le(payload.data() + 114, 4));  // the prompt's first token picks the behaviour
      if (mode == "die") _exit(3);
      if (mode == "sigpipe") {  // SIGPIPE's disposition as the done's token count: 7 default, 9 ignored
        struct sigaction sa;
        sigaction(SIGPIPE, nullptr, &sa);
        send_start(id), send_done(id, 0, sa.sa_handler == SIG_DFL ? 7 : 9);
        continue;
      }
      if (mode == "garbage") { const char junk[32] = "this is not a frame at all....."; (void)!write(1, junk, 32); sleep(30); }
      if (mode == "stray") send_tokens(id + 100, 0, {1});  // an event for a request nobody made
      send_start(id);
      if (first == 999) continue;  // hangs until cancelled
      send_tokens(id, 0, {10, 11});
      send_tokens(id, 2, {12});
      send_done(id, 0, 3);
    } else if (type == 0x0002) {
      send_done(le(payload.data(), 8), 2, 0);
    } else if (type == 0x0004) {
      std::vector<uint8_t> p;
      put(p, le(payload.data(), 8), 8);
      for (char c : std::string("{\"ok\":true}")) p.push_back(uint8_t(c));
      send_frame(0x0107, p);
    }
  }
}

// ---- the test ----
static int failures = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } \
  } while (0)

struct Collect : Sink {
  std::mutex m;
  std::condition_variable cv;
  std::vector<uint32_t> tokens;
  bool started = false, done = false, errored = false;
  int terminals = 0;
  Done d;
  Error e;
  void on_start(const Start&) override { std::lock_guard<std::mutex> l(m); started = true; }
  void on_tokens(const Tokens& t) override { std::lock_guard<std::mutex> l(m); tokens.insert(tokens.end(), t.tokens.begin(), t.tokens.end()); }
  void on_done(const Done& x) override { std::lock_guard<std::mutex> l(m); done = true; d = x; terminals++; cv.notify_all(); }
  void on_error(const Error& x) override { std::lock_guard<std::mutex> l(m); errored = true; e = x; terminals++; cv.notify_all(); }
  bool wait(double s = 10) {
    std::unique_lock<std::mutex> l(m);
    return cv.wait_for(l, std::chrono::duration<double>(s), [&] { return terminals > 0; });
  }
  bool wait_started(double s = 10) {
    for (int i = 0; i < int(s * 100) && !([&] { std::lock_guard<std::mutex> l(m); return started; })(); i++) usleep(10000);
    std::lock_guard<std::mutex> l(m);
    return started;
  }
};

struct Thrower : Collect {
  void on_tokens(const Tokens&) override { throw std::runtime_error("a sink's bug"); }
  void on_done(const Done& x) override { Collect::on_done(x); throw std::runtime_error("a sink's bug"); }
};

static double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static RuntimeOptions opts(const char* self, const char* mode) {
  RuntimeOptions o;
  o.argv = {self, "--fake", mode};
  o.ready_timeout_s = 10;
  o.stop_grace_s = 2;
  return o;
}
static Request req(uint32_t first) {
  Request r;
  r.max_output_tokens = 16;
  r.prompt = {first, 2, 3, 4, 5};
  return r;
}

int main(int argc, char** argv) {
  if (argc == 3 && !std::strcmp(argv[1], "--fake")) return fake(argv[2]);
  const char* self = argv[0];
  alarm(180);  // a deadlock in the code under test is a failure, not a hung CI job

  {  // a request streams its tokens and ends with done; status answers; shutdown drains
    Runtime rt(opts(self, "echo"));
    rt.start();
    CHECK(rt.alive());
    CHECK(rt.ready().max_concurrent_requests == 2 && rt.ready().max_context_tokens == 4096);
    auto c = std::make_shared<Collect>();
    uint64_t id = rt.submit(req(1), c, 30);
    CHECK(id != 0);
    CHECK(c->wait());
    CHECK(c->started && c->done && !c->errored && c->terminals == 1);
    CHECK((c->tokens == std::vector<uint32_t>{10, 11, 12}));
    CHECK(c->d.completion_tokens == 3 && c->d.request_id == id);
    CHECK(rt.status() == "{\"ok\":true}");

    // a hung request ends cancelled, and the pool slot is free again
    auto h = std::make_shared<Collect>();
    uint64_t hid = rt.submit(req(999), h, 30);
    CHECK(h->wait_started());
    rt.cancel(hid);
    CHECK(h->wait());
    CHECK(h->done && h->d.reason == FinishReason::kCancelled && h->terminals == 1);
    rt.cancel(hid);  // already ended: no effect
    auto c2 = std::make_shared<Collect>();
    rt.submit(req(1), c2, 30);
    CHECK(c2->wait() && c2->done);

    // a request the engine would refuse is not sent and registers nothing
    Request bad = req(1);
    bad.max_output_tokens = 0;
    bool threw = false;
    try { rt.submit(bad, std::make_shared<Collect>(), 30); } catch (const ProtocolError&) { threw = true; }
    CHECK(threw);
    pid_t child = rt.pid();
    rt.shutdown();
    CHECK(!rt.alive());
    CHECK(kill(child, 0) != 0);  // reaped
    threw = false;
    try { rt.submit(req(1), std::make_shared<Collect>(), 30); } catch (const EngineUnhealthy&) { threw = true; }
    CHECK(threw);
    rt.shutdown();  // idempotent
  }

  {  // the pool: one slot, a second submit is refused while the first is held
    Runtime rt(opts(self, "pool1"));
    rt.start();
    auto h = std::make_shared<Collect>();
    uint64_t hid = rt.submit(req(999), h, 30);
    bool full = false;
    try { rt.submit(req(1), std::make_shared<Collect>(), 30); } catch (const PendingLimitExceeded&) { full = true; }
    CHECK(full);
    // shutdown with a request held ends it as an engine error, once
    rt.shutdown();
    CHECK(h->wait(1) && h->errored && h->terminals == 1 && h->e.failure == FailureClass::kEngineUnhealthy && h->e.request_id == hid);
  }

  {  // a child that dies with a request held: the request fails engine_unhealthy and the runtime is dead
    Runtime rt(opts(self, "die"));
    rt.start();
    auto c = std::make_shared<Collect>();
    rt.submit(req(1), c, 30);
    CHECK(c->wait() && c->errored && c->terminals == 1);
    CHECK(c->e.failure == FailureClass::kEngineUnhealthy && c->e.code == "engine_exited");
    CHECK(!rt.alive());
    bool threw = false;
    try { rt.submit(req(1), std::make_shared<Collect>(), 30); } catch (const EngineUnhealthy&) { threw = true; }
    CHECK(threw);
  }

  {  // the child's SIGPIPE is the default, though the server's is ignored
    Runtime rt(opts(self, "sigpipe")); rt.start();
    auto c = std::make_shared<Collect>(); rt.submit(req(1), c, 30);
    CHECK(c->wait(5) && c->done && c->d.completion_tokens == 7);
  }

  {  // a child that dies mid-frame: the request fails as truncated_frame, not as a clean exit
    Runtime rt(opts(self, "truncate")); rt.start();
    auto c = std::make_shared<Collect>(); rt.submit(req(1), c, 30);
    CHECK(c->wait(5) && c->errored && c->e.code == "truncated_frame");
  }

  {  // garbage on its stdout: the request fails with a protocol error and the child is killed, not waited for
    Runtime rt(opts(self, "garbage"));
    rt.start();
    auto c = std::make_shared<Collect>();
    rt.submit(req(1), c, 30);
    CHECK(c->wait(5) && c->errored && c->e.code == "protocol_error");
    CHECK(!rt.alive());
  }

  {  // never ready: start() throws after the timeout and nothing is left running
    RuntimeOptions o = opts(self, "noready");
    o.ready_timeout_s = 0.4;
    o.stop_grace_s = 0.3;
    Runtime rt(o);
    bool threw = false;
    auto t0 = std::chrono::steady_clock::now();
    try { rt.start(); } catch (const EngineUnhealthy&) { threw = true; }
    CHECK(threw);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(8));
    CHECK(!rt.alive());
  }

  {  // a binary that does not exist
    RuntimeOptions o;
    o.argv = {"/nonexistent/splash"};
    Runtime rt(o);
    bool threw = false;
    try { rt.start(); } catch (const EngineUnhealthy&) { threw = true; }
    CHECK(threw);
  }

  {  // eight submitters race for a one-slot pool: exactly one gets in
    Runtime rt(opts(self, "pool1"));
    rt.start();
    std::atomic<int> in{0}, full{0}, other{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 8; i++)
      ts.emplace_back([&] {
        try { rt.submit(req(999), std::make_shared<Collect>(), 30); in++; }
        catch (const PendingLimitExceeded&) { full++; }
        catch (...) { other++; }
      });
    for (auto& t : ts) t.join();
    CHECK(in == 1 && full == 7 && other == 0);
    rt.shutdown();
  }

  {  // a child that stops reading its stdin, with a request too large for the pipe: shutdown() still returns, and the child is gone
    RuntimeOptions o = opts(self, "stall");
    o.stop_grace_s = 0.3;
    Runtime rt(o);
    rt.start();
    pid_t child = rt.pid();
    Request big = req(1);
    big.prompt.assign(200000, 7);  // 800 KB: more than a pipe holds
    std::atomic<bool> threw{false};
    std::thread w([&] {
      try { rt.submit(big, std::make_shared<Collect>(), 30); } catch (const EngineUnhealthy&) { threw = true; }
    });
    usleep(300000);  // the writer is blocked in the pipe
    auto t0 = std::chrono::steady_clock::now();
    rt.shutdown();
    CHECK(since(t0) < 5);
    w.join();
    CHECK(threw);
    CHECK(kill(child, 0) != 0);
  }

  {  // cancel() from a Sink (the reader thread) or anywhere does not wait behind a request frame that is blocked in the pipe
    RuntimeOptions o = opts(self, "stall");
    o.stop_grace_s = 0.3;
    Runtime rt(o); rt.start();
    Request big = req(1);
    big.prompt.assign(200000, 7);
    std::thread w([&] { try { rt.submit(big, std::make_shared<Collect>(), 30); } catch (const EngineUnhealthy&) {} });
    usleep(300000);  // the writer holds the pipe; its request is id 1
    auto t0 = std::chrono::steady_clock::now();
    rt.cancel(1);
    CHECK(since(t0) < 0.2);
    rt.shutdown(), w.join();
  }

  {  // the same child, with the default shutdown path out of the picture: the frame times out and the child is killed
    RuntimeOptions o = opts(self, "stall");
    o.write_timeout_s = 0.5;
    Runtime rt(o);
    rt.start();
    Request big = req(1);
    big.prompt.assign(200000, 7);
    auto c = std::make_shared<Collect>();
    auto t0 = std::chrono::steady_clock::now();
    bool threw = false;
    try { rt.submit(big, c, 30); } catch (const EngineUnhealthy&) { threw = true; }
    CHECK(threw && since(t0) < 5);
    usleep(500000);
    CHECK(c->terminals <= 1);  // told by the throw, or by the reader's on_error when the child's death came first; never twice
    rt.shutdown();
    CHECK(!rt.alive());
  }

  {  // a descendant in the child's group holds its stdout and the child ignores EOF: SIGTERM to the group ends both
    RuntimeOptions o = opts(self, "grandchild");
    o.stop_grace_s = 0.3;
    Runtime rt(o);
    rt.start();
    auto t0 = std::chrono::steady_clock::now();
    rt.shutdown();
    CHECK(since(t0) < 5);
    CHECK(!rt.alive());
  }

  {  // a descendant outside the group holds its stdout after the child exits: shutdown() stops reading and returns
    RuntimeOptions o = opts(self, "orphan");
    o.stop_grace_s = 0.3;
    Runtime rt(o);
    rt.start();
    auto t0 = std::chrono::steady_clock::now();
    rt.shutdown();
    CHECK(since(t0) < 5);
    CHECK(!rt.alive());
  }

  {  // a Sink that throws does not end the process, and the runtime serves the next request
    Runtime rt(opts(self, "echo"));
    rt.start();
    auto t = std::make_shared<Thrower>();
    rt.submit(req(1), t, 30);
    CHECK(t->wait());
    CHECK(rt.alive());
    auto c = std::make_shared<Collect>();
    rt.submit(req(1), c, 30);
    CHECK(c->wait() && c->done);
    auto h = std::make_shared<Thrower>();  // and one that throws from on_error when the child dies
    rt.submit(req(999), h, 30);
    rt.shutdown();
    CHECK(h->wait(2) && h->errored);
  }

  {  // a log callback that calls alive() while the reader logs an unknown request does not deadlock
    RuntimeOptions o = opts(self, "stray");
    Runtime* rtp = nullptr;
    std::atomic<int> logs{0};
    o.log = [&](const std::string&) { if (rtp) { (void)rtp->alive(); logs++; } };
    Runtime rt(o);
    rtp = &rt;
    rt.start();
    auto c = std::make_shared<Collect>();
    rt.submit(req(1), c, 30);
    CHECK(c->wait() && c->done);
    CHECK(logs > 0);
    rt.shutdown();
  }

  {  // shutdown() after the child died on its own: nothing is signalled, nothing hangs
    Runtime rt(opts(self, "die"));
    rt.start();
    auto c = std::make_shared<Collect>();
    rt.submit(req(1), c, 30);
    CHECK(c->wait());
    CHECK(rt.pid() == -1);
    rt.shutdown();
  }

  if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
  std::printf("splash_runtime_test: ok\n");
  return 0;
}
