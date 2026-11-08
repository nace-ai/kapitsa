// CPU test of src/splash/runtime.{h,cpp}: the Runtime spawns this very binary as a fake `splash serve-native` (argv[1] =
// "--fake MODE") that speaks the protocol over its stdin and stdout, and drives it through a request, a cancel, a status,
// a child that dies, one that sends garbage, one that never gets ready, a full request pool, a child that stops reading its
// input, a descendant that keeps its stdout, a token mask asked and answered, a Sink that throws, and a log callback that calls back in. No model, no GPU.
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
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
  int events = -1;
  bool first_child = true;
  if (const char* path = std::getenv("RUNTIME_TEST_EVENTS")) {
    std::ifstream prior(path);
    first_child = prior.peek() == std::ifstream::traits_type::eof();
    events = open(path, O_WRONLY | O_APPEND);
    if (events >= 0) dprintf(events, "%d 0 0\n", getpid());
  }
  if (!first_child && mode == "recover_noready") { sleep(60); return 0; }
  if (!first_child && mode == "recover_die") return 3;
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
    if (events >= 0) dprintf(events, "%d %u %llu\n", getpid(), type, static_cast<unsigned long long>(le(payload.data(), 8)));
    if (type == 0x0001) {
      uint64_t id = le(payload.data(), 8);
      uint32_t first = uint32_t(le(payload.data() + 114, 4));  // the prompt's first token picks the behaviour
      if (mode == "die") _exit(3);
      if (mode == "fatal_cancel_code") {
        const std::string code = "cancel_timeout", message = "fatal child error";
        std::vector<uint8_t> error;
        put(error, 2, 1); put(error, 0, 1); put(error, 0, 4); put(error, 0, 8);
        put(error, code.size(), 4); put(error, message.size(), 4);
        error.insert(error.end(), code.begin(), code.end());
        error.insert(error.end(), message.begin(), message.end());
        send_frame(0x0105, error);
        continue;
      }
      if (mode == "sigpipe") {  // SIGPIPE's disposition as the done's token count: 7 default, 9 ignored
        struct sigaction sa;
        sigaction(SIGPIPE, nullptr, &sa);
        send_start(id), send_done(id, 0, sa.sa_handler == SIG_DFL ? 7 : 9);
        continue;
      }
      if (mode == "garbage") { const char junk[32] = "this is not a frame at all....."; (void)!write(1, junk, 32); sleep(30); }
      if (mode == "stray") send_tokens(id + 100, 0, {1});  // an event for a request nobody made
      send_start(id);
      if (first == 999 && mode == "block_after_start" && first_child) { sleep(60); return 0; }
      if (first == 999) continue;  // hangs until cancelled
      if (first == 777 || payload[9] == 1) {  // a token mask: ask for two rows after token 7, echo the answer as tokens
        std::vector<uint8_t> p;
        put(p, id, 8); put(p, 5, 8); put(p, 2, 4); put(p, 1, 4); put(p, 7, 4);
        send_frame(0x0103, p);
        if (first == 777) continue;  // asked on an unconstrained request: the client must fail the stream
        uint8_t mh[16];
        if (!read_all(mh, 16)) return 0;
        std::vector<uint8_t> mp(le(mh + 8, 8));
        if (!read_all(mp.data(), mp.size()) || le(mh + 6, 2) != 0x0003 || le(mp.data(), 8) != id || le(mp.data() + 8, 8) != 5) _exit(5);
        std::vector<uint32_t> words(le(mp.data() + 16, 4));
        for (size_t i = 0; i < words.size(); i++) words[i] = uint32_t(le(mp.data() + 20 + 4 * i, 4));
        send_tokens(id, 0, words);
        send_done(id, 0, uint32_t(words.size()));
        continue;
      }
      send_tokens(id, 0, {10, 11});
      send_tokens(id, 2, {12});
      send_done(id, 0, 3);
    } else if (type == 0x0002) {
      if (mode == "ignore_cancel" || mode == "recover_noready" || mode == "recover_die") continue;
      if (mode == "finish_cancel") {
        send_tokens(le(payload.data(), 8), 0, {10, 11, 12});
        send_done(le(payload.data(), 8), 0, 3);
      } else {
        send_done(le(payload.data(), 8), 2, 0);
      }
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

struct Masker : Collect {  // queues mask requests: the answer goes back from another thread, never from the reader
  std::vector<MaskRequest> asks;
  void on_mask_request(const MaskRequest& x) override { std::lock_guard<std::mutex> l(m); asks.push_back(x); cv.notify_all(); }
  bool wait_ask(double s = 10) {
    std::unique_lock<std::mutex> l(m);
    return cv.wait_for(l, std::chrono::duration<double>(s), [&] { return !asks.empty() || terminals > 0; }) && !asks.empty();
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

struct FakeEvents {
  std::string path;
  FakeEvents() {
    char name[] = "/tmp/native-recovery-events-XXXXXX";
    int fd = mkstemp(name);
    if (fd < 0) throw std::runtime_error("cannot create test event log");
    close(fd);
    path = name;
  }
  ~FakeEvents() { unlink(path.c_str()); }
  void attach(RuntimeOptions& o) const { o.env.push_back("RUNTIME_TEST_EVENTS=" + path); }
  struct Event { pid_t pid; unsigned type; uint64_t id; };
  std::vector<Event> read() const {
    std::ifstream in(path);
    std::vector<Event> out;
    Event x;
    while (in >> x.pid >> x.type >> x.id) out.push_back(x);
    return out;
  }
  int starts() const {
    int n = 0;
    for (auto x : read()) if (x.type == 0) n++;
    return n;
  }
};

static bool recovered(Runtime& rt, pid_t old, double seconds = 5) {
  auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < until) {
    if (rt.alive() && rt.pid() != old) return true;
    usleep(10000);
  }
  return false;
}

struct RemoveExecutable : Collect {
  std::string path;
  void on_error(const Error& e) override {
    unlink(path.c_str());  // fail the replacement exec before the reader starts it
    Collect::on_error(e);
  }
};

int main(int argc, char** argv) {
  if (argc == 3 && !std::strcmp(argv[1], "--fake")) return fake(argv[2]);
  const char* self = argv[0];
  alarm(180);  // a deadlock in the code under test is a failure, not a hung CI job

  {
    auto o = opts(self, "echo");
    o.cancel_timeout_s = 0;
    bool rejected = false;
    try { Runtime rt(o); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
  }
  {
    auto o = opts(self, "ignore_cancel");
    o.cancel_timeout_s = 0.4;
    FakeEvents events; events.attach(o);
    std::atomic<int> logged_starts{0};
    o.log = [&](const std::string& message) {
      if (message.find(" started") != std::string::npos && ++logged_starts == 2) throw std::runtime_error("logger failed");
    };
    Runtime rt(o);
    rt.start();
    const pid_t old = rt.pid();
    auto a = std::make_shared<Collect>(), b = std::make_shared<Collect>();
    auto ia = rt.submit(req(999), a, 30), ib = rt.submit(req(999), b, 30);
    CHECK(a->wait_started() && b->wait_started());
    rt.cancel(ia);
    usleep(250000);
    CHECK(!a->wait(0.01));
    rt.cancel(ia);
    CHECK(a->wait(0.35) && b->wait(5));
    CHECK(a->errored && b->errored && a->terminals == 1 && b->terminals == 1);
    CHECK(a->e.code == "cancel_timeout" && b->e.code == "cancel_timeout" && a->e.retryable && b->e.retryable);
    CHECK(recovered(rt, old));
    CHECK(kill(old, 0) != 0);
    auto fresh = std::make_shared<Collect>();
    const uint64_t fresh_id = rt.submit(req(1), fresh, 30);
    CHECK(fresh_id > ib && fresh->wait() && fresh->done && !fresh->errored && fresh->terminals == 1);
    CHECK(rt.status() == "{\"ok\":true}");
    rt.cancel(ia);  // a completed old request cannot reach the replacement
    rt.mask_response(ib, 5, {1});
    const pid_t replacement = rt.pid();
    rt.shutdown();
    CHECK(events.starts() == 2);
    int old_requests = 0, new_requests = 0;
    for (auto x : events.read()) {
      if (x.type == 1 && x.pid == old) old_requests++;
      if (x.type == 1 && x.pid == replacement) { new_requests++; CHECK(x.id == fresh_id); }
      if (x.pid == replacement && x.type == 2) CHECK(false);
    }
    CHECK(old_requests == 2 && new_requests == 1);
  }

  {  // a normal terminal after cancel removes the deadline without ending the runtime
    auto o = opts(self, "finish_cancel");
    o.cancel_timeout_s = 1;
    Runtime rt(o);
    rt.start();
    const pid_t child = rt.pid();
    auto active = std::make_shared<Collect>();
    auto id = rt.submit(req(999), active, 30);
    CHECK(active->wait_started());
    const auto cancelled_at = std::chrono::steady_clock::now();
    rt.cancel(id);
    CHECK(active->wait(0.75));
    CHECK(since(cancelled_at) < o.cancel_timeout_s);
    CHECK(active->done && !active->errored && active->d.reason == FinishReason::kStop);
    std::this_thread::sleep_until(cancelled_at + std::chrono::milliseconds(1400));
    CHECK(rt.alive() && rt.pid() == child);
    {
      std::lock_guard<std::mutex> l(active->m);
      CHECK(active->terminals == 1 && !active->errored);
    }
    auto healthy = std::make_shared<Collect>();
    rt.submit(req(1), healthy, 30);
    CHECK(healthy->wait() && healthy->done && !healthy->errored && healthy->terminals == 1);
    rt.shutdown();
  }

  {  // an old blocked writer and queued status/cancel frames never enter the new child
    FakeEvents events;
    auto o = opts(self, "block_after_start");
    events.attach(o);
    o.cancel_timeout_s = 0.5;
    o.stop_grace_s = 0.1;
    Runtime rt(o); rt.start();
    const pid_t old = rt.pid();
    auto active = std::make_shared<Collect>(), blocked = std::make_shared<Collect>();
    Request held = req(999); held.constraint = Constraint::kTokenMask;
    const uint64_t id = rt.submit(held, active, 30);
    CHECK(active->wait_started());
    Request big = req(1); big.prompt.assign(200000, 7);
    std::atomic<bool> writer_done{false};
    std::thread writer([&] { try { rt.submit(big, blocked, 30); } catch (const EngineUnhealthy&) {} writer_done = true; });
    usleep(150000);  // the large request is held behind the child's stopped reader
    std::string old_status;
    std::thread status([&] { old_status = rt.status(10); });
    std::thread mask([&] { try { rt.mask_response(id, 5, {1}); } catch (const EngineUnhealthy&) {} });
    usleep(50000);  // both old-generation control writers are behind the blocked request
    rt.cancel(id);
    CHECK(active->wait(5) && active->errored && active->e.code == "cancel_timeout");
    CHECK(blocked->wait(5) && blocked->errored && blocked->e.code == "cancel_timeout");
    writer.join(); status.join(); mask.join();
    CHECK(writer_done && old_status.empty());
    CHECK(recovered(rt, old));
    auto fresh = std::make_shared<Collect>();
    const uint64_t fresh_id = rt.submit(req(1), fresh, 30);
    CHECK(fresh->wait() && fresh->done && fresh->terminals == 1);
    const pid_t replacement = rt.pid();
    rt.shutdown();
    CHECK(active->terminals == 1 && blocked->terminals == 1 && events.starts() == 2);
    for (auto x : events.read()) if (x.pid == replacement && x.type != 0) CHECK(x.type == 1 && x.id == fresh_id);
  }

  {  // shutdown interrupts a replacement that has not sent READY and prevents another launch
    FakeEvents events;
    auto o = opts(self, "recover_noready"); events.attach(o);
    o.cancel_timeout_s = 0.2; o.stop_grace_s = 0.1;
    Runtime rt(o); rt.start();
    const pid_t old = rt.pid();
    auto active = std::make_shared<Collect>();
    const uint64_t id = rt.submit(req(999), active, 30);
    CHECK(active->wait_started()); rt.cancel(id);
    CHECK(active->wait(5) && active->e.code == "cancel_timeout");
    for (int i = 0; i < 500 && events.starts() < 2; i++) usleep(10000);
    CHECK(events.starts() == 2 && !rt.alive() && rt.pid() != old);
    const pid_t replacement = rt.pid();
    bool recovering_admission = false;
    try { rt.submit(req(1), std::make_shared<Collect>(), 30); }
    catch (const EngineUnhealthy& error) { recovering_admission = error.restarting; }
    CHECK(recovering_admission);
    const auto t0 = std::chrono::steady_clock::now();
    std::thread another_stop([&] { rt.shutdown(); });
    rt.shutdown(); another_stop.join();
    CHECK(since(t0) < 2 && !rt.alive() && rt.pid() == -1);
    CHECK(kill(old, 0) != 0 && kill(replacement, 0) != 0);
    CHECK(active->terminals == 1 && events.starts() == 2);
  }

  {  // a replacement that never becomes ready is stopped once rather than retried forever
    FakeEvents events;
    auto o = opts(self, "recover_noready"); events.attach(o);
    o.cancel_timeout_s = 0.2; o.ready_timeout_s = 0.3; o.stop_grace_s = 0.1;
    Runtime rt(o); rt.start();
    auto active = std::make_shared<Collect>();
    const uint64_t id = rt.submit(req(999), active, 30);
    CHECK(active->wait_started()); rt.cancel(id);
    CHECK(active->wait(5) && active->e.code == "cancel_timeout");
    for (int i = 0; i < 500 && (events.starts() < 2 || rt.pid() > 0); i++) usleep(10000);
    CHECK(events.starts() == 2 && rt.pid() == -1 && !rt.alive());
    rt.shutdown(); CHECK(active->terminals == 1 && events.starts() == 2);
  }

  {  // a replacement exec failure leaves the runtime dead without a retry loop
    FakeEvents events;
    char executable[] = "/tmp/native-recovery-child-XXXXXX";
    int fd = mkstemp(executable); CHECK(fd >= 0); close(fd);
    { std::ifstream in(self, std::ios::binary); std::ofstream out(executable, std::ios::binary); out << in.rdbuf(); }
    CHECK(chmod(executable, 0700) == 0);
    auto o = opts(executable, "ignore_cancel"); events.attach(o); o.cancel_timeout_s = 0.2;
    Runtime rt(o); rt.start();
    auto active = std::make_shared<RemoveExecutable>(); active->path = executable;
    const uint64_t id = rt.submit(req(999), active, 30);
    CHECK(active->wait_started()); rt.cancel(id);
    CHECK(active->wait(5) && active->e.code == "cancel_timeout");
    usleep(100000);
    CHECK(!rt.alive() && rt.pid() == -1 && active->terminals == 1 && events.starts() == 1);
    rt.shutdown(); unlink(executable);
  }

  {
    auto o = opts(self, "ignore_cancel");
    o.cancel_timeout_s = 0.4;
    Runtime rt(o);
    rt.start();
    auto active = std::make_shared<Collect>();
    auto id = rt.submit(req(999), active, 30);
    CHECK(active->wait_started());
    CHECK(!active->wait(0.7));
    CHECK(rt.alive());
    auto healthy = std::make_shared<Collect>();
    rt.submit(req(1), healthy, 30);
    CHECK(healthy->wait() && healthy->done && healthy->terminals == 1);
    rt.cancel(id);
    CHECK(active->wait(5) && active->errored && active->terminals == 1);
    rt.shutdown();
  }

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

  {  // a token-mask request: the engine asks, the caller answers off the reader thread, the engine gets the words
    Runtime rt(opts(self, "echo"));
    rt.start();
    auto c = std::make_shared<Masker>();
    Request r = req(1);
    r.constraint = Constraint::kTokenMask;
    uint64_t id = rt.submit(r, c, 30);
    CHECK(c->wait_ask());
    MaskRequest ask;
    { std::lock_guard<std::mutex> l(c->m); ask = c->asks.at(0); }
    CHECK(ask.request_id == id && ask.mask_request_id == 5 && ask.words_per_mask == 2);
    CHECK((ask.simulation_tokens == std::vector<uint32_t>{7}));
    rt.mask_response(id, 5, {0xFFFFFFFFu, 1u, 2u, 3u});
    CHECK(c->wait() && c->done && c->terminals == 1);
    CHECK((c->tokens == std::vector<uint32_t>{0xFFFFFFFFu, 1u, 2u, 3u}));
    rt.mask_response(id, 6, {1});  // the request is over: nothing is sent, and the engine stays healthy
    auto c2 = std::make_shared<Collect>();
    rt.submit(req(1), c2, 30);
    CHECK(c2->wait() && c2->done && rt.alive());
    bool threw = false;
    try { rt.mask_response(id, 6, {}); } catch (const ProtocolError&) { threw = true; }
    CHECK(threw);
  }

  {  // a mask request for a request submitted without a constraint is a protocol failure: the stream ends
    Runtime rt(opts(self, "echo"));
    rt.start();
    auto c = std::make_shared<Masker>();
    rt.submit(req(777), c, 30);
    CHECK(c->wait() && c->errored && c->terminals == 1);
    CHECK(c->asks.empty());
    CHECK(c->e.code == "protocol_error" && c->e.message.find("unconstrained") != std::string::npos);
    CHECK(!rt.alive());
  }

  {  // an engine-level error cannot request a relaunch by copying the timeout code
    FakeEvents events;
    auto o = opts(self, "fatal_cancel_code"); events.attach(o);
    Runtime rt(o); rt.start();
    auto active = std::make_shared<Collect>();
    rt.submit(req(1), active, 30);
    CHECK(active->wait(5) && active->errored && active->e.code == "cancel_timeout");
    usleep(100000);
    CHECK(!rt.alive() && rt.pid() == -1 && active->terminals == 1 && events.starts() == 1 && !active->e.retryable);
    bool permanent = false;
    try { rt.submit(req(1), std::make_shared<Collect>(), 30); }
    catch (const EngineUnhealthy& error) { permanent = !error.restarting; }
    CHECK(permanent);
    rt.shutdown(); CHECK(events.starts() == 1);
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
    auto failed = std::make_shared<Collect>();
    std::thread w([&] {
      try { rt.submit(big, failed, 30); } catch (const EngineUnhealthy&) { threw = true; }
    });
    usleep(300000);  // the writer is blocked in the pipe
    auto t0 = std::chrono::steady_clock::now();
    rt.shutdown();
    CHECK(since(t0) < 5);
    w.join();
    CHECK(threw || (failed->wait(1) && failed->errored));
    CHECK(failed->terminals == (threw ? 0 : 1));
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
    CHECK(since(t0) < 5);
    CHECK(threw || (c->wait(5) && c->errored));
    usleep(500000);
    CHECK(c->terminals == (threw ? 0 : 1));  // the throw or the terminal callback reports failure, never both
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
