#include "cluster/pipe.h"
#include "cluster/place.h"
#include "cluster/wire.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <dirent.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <thread>

namespace ni::cluster {

using json = nlohmann::json;
using namespace wire;

std::string model_digest(const std::string& dir) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a
  auto mix = [&](const std::string& s) {
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
  };
  {
    std::ifstream f(dir + "/config.json", std::ios::binary);
    mix(std::string(std::istreambuf_iterator<char>(f), {}));
  }
  std::vector<std::string> files;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      // the trunk's files: not the n-gram table or the MTP head, which only the head reads (a stage's folder may
      // lack them)
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0 && n != "ple.safetensors" &&
          n != "mtp.safetensors")
        files.push_back(n);
    }
    closedir(d);
  }
  std::sort(files.begin(), files.end());
  for (const std::string& f : files) {
    std::ifstream in(dir + "/" + f, std::ios::binary | std::ios::ate);
    mix(f + ":" + std::to_string(in ? int64_t(in.tellg()) : -1) + "\n");
  }
  char b[20];
  snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
  return b;
}

// ---- the head's side ----

struct Link::Impl {
  int up = -1, back = -1;  // to the first stage; the ring's way back (the same socket with one stage)
  struct Item {
    Header h;
    std::string payload;
    Remote::Rows rows;
    // the caller's wait: a pass's (0 once its rows are back) and grow's (the positions the stages hold); -1 when lost
    std::shared_ptr<std::promise<int>> answer;
    // a pass's: when its rows went out (ms; 0 before), where its timeout starts
    std::shared_ptr<std::atomic<double>> sent;
  };
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Item> queue;   // posted, to send
  std::deque<Item> flight;  // sent, waiting for the ring to bring them back
  bool stop = false;
  std::atomic<bool> broken{false};
  std::string why;  // (mu)
  uint32_t next_id = 1;
  std::atomic<int> seqs{0}, ckpts{0};
  std::thread sender, reader;

  void post(Item it) {
    {
      std::lock_guard<std::mutex> lk(mu);
      it.h.id = next_id++;
      queue.push_back(std::move(it));
    }
    cv.notify_one();
  }

  // A message whose answer will not come: its caller's wait ends with -1 (a pass's rows are garbage; lost() says why)
  static void release(Item& it) {
    if (it.answer) it.answer->set_value(-1);
    it.answer.reset();
  }

  void fail(const std::string& msg) {
    std::deque<Item> drop;
    bool closing = false;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (broken) return;  // (the first reason stays)
      why = msg;
      broken = true;
      closing = stop;
      drop.swap(flight);
    }
    if (!closing) logi("split: %s", msg.c_str());
    shutdown(up, SHUT_RDWR);
    if (back != up) shutdown(back, SHUT_RDWR);
    for (Item& it : drop) release(it);
    cv.notify_all();
  }

  // the GPU's ready value for a pass's rows (spinning briefly, then in 50 us naps); false once the split broke
  bool wait_ready(const Remote::Rows& r) {
    for (int spins = 0; Gpu::event_value(r.event) < r.ready; ++spins) {
      if (broken) return false;
      if (spins < 2000) std::this_thread::yield();
      else usleep(50);
    }
    return true;
  }

  void send_loop() {
    for (;;) {
      Item it;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return stop || !queue.empty(); });
        if (queue.empty()) return;
        it = std::move(queue.front());
        queue.pop_front();
      }
      const bool pass = is_pass(it.h.op);
      if (broken || (pass && !wait_ready(it.rows))) {
        release(it);
        continue;
      }
      Header h = it.h;
      iovec iov[4];
      int n = 0;
      iov[n++] = {&h, sizeof h};
      if (pass) {
        const size_t R = size_t(h.rows);
        h.bytes = R * (it.rows.x_bytes + it.rows.y_bytes + it.rows.inj_bytes);
        iov[n++] = {it.rows.x.ptr(), R * it.rows.x_bytes};
        iov[n++] = {it.rows.y.ptr(), R * it.rows.y_bytes};
        iov[n++] = {it.rows.inj.ptr(), R * it.rows.inj_bytes};
      } else {
        h.bytes = it.payload.size();
        if (!it.payload.empty()) iov[n++] = {it.payload.data(), it.payload.size()};
      }
      if (it.sent) it.sent->store(now_ms());
      bool late = false;
      {
        // in flight before it is sent: the answer may come back before write_all returns
        std::lock_guard<std::mutex> lk(mu);
        late = broken;
        if (!late) {
          Item f;
          f.h = it.h;
          f.rows = it.rows;
          f.answer = it.answer;
          flight.push_back(std::move(f));
        }
      }
      if (late) {
        release(it);
        continue;
      }
      if (!write_all(up, iov, n)) fail(std::string("sending to the first stage: ") + strerror(errno));
    }
  }

  void read_loop() {
    std::string skip;
    for (;;) {
      Header h;
      if (!read_all(back, &h, sizeof h)) {
        fail("the other Mac of the split closed the connection (stopped, asleep or off the network)");
        return;
      }
      Item it;  // (op 0: no message in flight matches this one)
      {
        std::lock_guard<std::mutex> lk(mu);
        if (broken) return;
        if (!flight.empty() && h.magic == kMagic && flight.front().h.id == h.id && flight.front().h.op == h.op) {
          it = std::move(flight.front());
          flight.pop_front();
        }
      }
      if (!it.h.op) {
        fail("an answer out of order from the split's stages");
        return;
      }
      if (is_pass(h.op)) {
        // the rows after the last layer: x, y and inj; a prefill's x alone (its injection applied there)
        const size_t R = size_t(it.h.rows);
        const bool pf = h.op == kPrefill;
        const uint64_t want = R * (it.rows.x_bytes + (pf ? 0 : it.rows.y_bytes + it.rows.inj_bytes));
        bool ok = h.bytes == want && read_all(back, it.rows.x.ptr(), R * it.rows.x_bytes);
        if (ok && !pf)
          ok = read_all(back, it.rows.y.ptr(), R * it.rows.y_bytes) &&
               read_all(back, it.rows.inj.ptr(), R * it.rows.inj_bytes);
        if (!ok) {
          release(it);
          fail("a pass's rows came back short");
          return;
        }
        if (it.answer) it.answer->set_value(0);
      } else {
        if (!read_payload(back, h, &skip)) {
          release(it);
          fail("a short answer from the split's stages");
          return;
        }
        if (it.answer) it.answer->set_value(h.flags & kFail ? -1 : h.b);
      }
    }
  }
};

namespace {
// A stage's answer to a query (a hello with "query": a connection of its own, closed after): what it is and holds
bool query_stage(const std::string& addr, const json& hello, StageInfo* out, std::string* why) {
  const int fd = connect_to(addr, 5000, why);
  if (fd < 0) return false;
  Header h;
  h.op = kHello;
  const std::string body = hello.dump();
  std::string reply;
  const bool ok = write_msg(fd, h, body.data(), body.size()) && readable(fd, 10000) && read_all(fd, &h, sizeof h) &&
                  h.magic == kMagic && h.op == kHello && read_payload(fd, h, &reply);
  close(fd);
  const json r = ok ? json::parse(reply, nullptr, false) : json();
  if (!r.is_object()) {
    *why = addr + ": no answer to the handshake (another program on that port, or an older Kapitsa)";
    return false;
  }
  if (h.flags & kFail) {
    *why = r.value("error", addr + " refused");
    return false;
  }
  out->addr = addr;
  out->name = r.value("name", addr);
  out->mem = r.value("mem", 0.0);
  out->gbs = r.value("gbs", 0.0);
  out->rows = r.value("rows", 0);
  out->fixed = r.value("fixed", false);
  out->lo = r.value("lo", 0);
  out->hi = r.value("hi", 0);
  return true;
}
}  // namespace

std::shared_ptr<Link> Link::connect(const std::vector<std::string>& addrs, const std::string& dir, const HeadInfo& me,
                                    std::string* why) {
  if (addrs.empty()) {
    *why = "no stages";
    return nullptr;
  }
  const Qwen35Config c = Qwen35::read_config(dir);
  const json root = json::parse(std::ifstream(dir + "/config.json"), nullptr, false);
  const json t = root.is_object() && root.contains("text_config") ? root["text_config"] : root;
  const int hc = t.is_object() ? t.value("hc_count", 4) : 4;
  json hello = {{"proto", kPipeProto},
                {"digest", model_digest(dir)},
                {"head", me.name},
                {"row_bytes", {size_t(hc) * c.hidden * 4, size_t(c.hidden) * 4, size_t(hc) * 4}}};
  // 1. each stage, on its own: what it is, and the layers it was started with, if any
  std::shared_ptr<Link> L(new Link());
  for (const std::string& a : addrs) {
    json q = hello;
    q["query"] = true;
    StageInfo si;
    if (!query_stage(a, q, &si, why)) return nullptr;
    L->stages_.push_back(si);
  }
  // 2. the layers: theirs when every stage was started with --layers, else the placement's (src/cluster/place.h)
  size_t fixed = 0;
  for (const StageInfo& s : L->stages_) fixed += s.fixed;
  if (fixed && fixed < addrs.size()) {
    *why = "some stages were started with --layers and some without: start all of them one way";
    return nullptr;
  }
  if (!fixed) {
    std::vector<Member> members = {{me.name, me.mem, me.gbs}};
    for (const StageInfo& s : L->stages_) members.push_back({s.name, s.mem, s.gbs});
    const Placement p = plan_split(dir, members, me.chunk, me.may_stream);
    if (!p.fits) {
      double mem = 0;
      for (const Member& m : members) mem += m.mem;
      char b[256];
      snprintf(b, sizeof b, "these %zu Macs' GPU memory (%.1f GB in all) does not hold the model's %d layers with "
               "their buffers and two 32K-token conversations: add a Mac, or use a smaller variant", members.size(),
               mem / 1e9, c.layers);
      *why = b;
      return nullptr;
    }
    for (size_t i = 0; i < L->stages_.size(); ++i) {
      L->stages_[i].lo = p.lo[i + 1];
      L->stages_[i].hi = p.lo[i + 2];
    }
    L->streams_ = p.streams;
  }
  // the layers line up: the stages' runs one after another, to the model's last layer
  bool ok = true;
  for (size_t i = 0; ok && i < L->stages_.size(); ++i)
    ok = L->stages_[i].lo < L->stages_[i].hi && (i == 0 || L->stages_[i].lo == L->stages_[i - 1].hi);
  if (!ok || L->stages_.back().hi != c.layers || L->first() < 1) {
    std::string runs;
    for (const StageInfo& s : L->stages_)
      runs += (runs.empty() ? "" : ", ") + s.name + " " + std::to_string(s.lo + 1) + "-" + std::to_string(s.hi);
    *why = "the stages' layers do not line up (" + runs + " of " + std::to_string(c.layers) +
           "): each stage starts where the one before it ends, and the last ends with the model";
    return nullptr;
  }
  // 3. the ring: the head to the first stage, each to the next (the chain), the last back to the head (its return
  // port with more than one stage, else the same socket). Each stage loads the layers given it, if it must.
  L->d_ = std::make_unique<Impl>();
  Impl& d = *L->d_;
  d.up = connect_to(addrs[0], 5000, why);
  if (d.up < 0) return nullptr;
  int lfd = -1;
  std::string ret;
  if (addrs.size() > 1) {
    lfd = listen_on("", 0, why);
    if (lfd < 0) return nullptr;
    ret = local_addr(d.up) + ":" + std::to_string(bound_port(lfd));
  }
  json assign = json::array();
  for (const StageInfo& s : L->stages_) assign.push_back({s.lo, s.hi});
  hello["chain"] = std::vector<std::string>(addrs.begin() + 1, addrs.end());
  hello["ret"] = ret;
  hello["assign"] = assign;
  hello["stages"] = json::array();
  Header h;
  h.op = kHello;
  const std::string body = hello.dump();
  if (!write_msg(d.up, h, body.data(), body.size())) {
    *why = addrs[0] + ": " + strerror(errno);
    return nullptr;
  }
  // (a stage loads the layers given it before it answers: up to minutes for a share of tens of GB)
  constexpr int kLoadMs = 15 * 60 * 1000;
  if (lfd >= 0) {
    if (!readable(lfd, kLoadMs)) {
      *why = "the last stage (" + addrs.back() + ") did not connect back";
      close(lfd);
      return nullptr;
    }
    d.back = accept(lfd, nullptr, nullptr);
    close(lfd);
    if (d.back < 0) {
      *why = std::string("accept: ") + strerror(errno);
      return nullptr;
    }
    tune(d.back);
  } else {
    d.back = d.up;
  }
  std::string reply;
  if (!readable(d.back, kLoadMs) || !read_all(d.back, &h, sizeof h) || h.magic != kMagic || h.op != kHello ||
      !read_payload(d.back, h, &reply)) {
    *why = "no answer from the stages";
    return nullptr;
  }
  const json r = json::parse(reply, nullptr, false);
  if (h.flags & kFail || !r.is_object()) {
    *why = r.is_object() ? r.value("error", std::string("the stages refused")) : "a garbled answer from the stages";
    return nullptr;
  }
  for (size_t i = 0; i < L->stages_.size() && i < r["stages"].size(); ++i)
    L->stages_[i].rows = r["stages"][i].value("rows", L->stages_[i].rows);
  // an empty message around the ring, five times
  std::vector<double> rt;
  for (int i = 0; i < 5; ++i) {
    Header p;
    p.op = kPing;
    p.id = 1u << 31 | uint32_t(i);
    const double t0 = now_ms();
    if (!write_msg(d.up, p, nullptr, 0) || !read_all(d.back, &p, sizeof p) || p.op != kPing) {
      *why = "the ring did not carry a ping";
      return nullptr;
    }
    rt.push_back(now_ms() - t0);
  }
  std::sort(rt.begin(), rt.end());
  L->rtt_ms_ = rt[2];
  d.sender = std::thread([&d] { d.send_loop(); });
  d.reader = std::thread([&d] { d.read_loop(); });
  return L;
}

Link::~Link() {
  if (!d_) return;
  {
    std::lock_guard<std::mutex> lk(d_->mu);
    d_->stop = true;
  }
  d_->cv.notify_all();
  if (d_->sender.joinable()) d_->sender.join();
  shutdown(d_->up, SHUT_RDWR);
  if (d_->back != d_->up) shutdown(d_->back, SHUT_RDWR);
  if (d_->reader.joinable()) d_->reader.join();
  close(d_->up);
  if (d_->back != d_->up) close(d_->back);
}

int Link::rows() const {
  int r = INT_MAX;
  for (const StageInfo& s : stages_) r = std::min(r, s.rows);
  return stages_.empty() ? 0 : r;
}

bool Link::pass(const Pass& p, const Rows& r) {
  if (d_->broken) return false;
  Impl::Item it;
  it.h.op = uint16_t(uint16_t(p.kind) + 1);
  it.h.seq = p.seq;
  it.h.pos = p.pos;
  it.h.rows = p.rows;
  it.h.n_pend = p.n_pend;
  it.h.a = p.snap;
  it.h.b = p.snap_row;
  it.rows = r;
  it.answer = std::make_shared<std::promise<int>>();
  it.sent = std::make_shared<std::atomic<double>>(0);
  std::future<int> f = it.answer->get_future();
  const std::shared_ptr<std::atomic<double>> sent = it.sent;
  d_->post(std::move(it));
  // the clock starts when the rows go out: the GPU's layers before the hop take as long as they take (remote.h). A pass
  // comes back in milliseconds, a long prompt's chunk over a slow network in tens of seconds; 5 minutes after the rows
  // went out the stages are taken as gone (a Mac asleep or off the network is noticed sooner, by TCP: tune())
  while (f.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
    const double t = sent->load();
    if (t > 0 && now_ms() - t > 5 * 60e3) {
      d_->fail("no answer to a pass in 5 minutes from the split's stages");
      break;
    }
  }
  return f.get() >= 0;  // (released with -1 by the failure)
}

namespace {
Link::Impl::Item control(uint16_t op, int seq, int a = 0, int b = 0, int pos = 0) {
  Link::Impl::Item it;
  it.h.op = op;
  it.h.seq = seq;
  it.h.a = a;
  it.h.b = b;
  it.h.pos = pos;
  return it;
}
}  // namespace

int Link::new_seq(int limit) {
  const int id = d_->seqs++;
  d_->post(control(kNewSeq, id, limit));
  return id;
}

int Link::grow(int seq, int need) {
  if (d_->broken) return -1;
  Impl::Item it = control(kGrow, seq, need, INT_MAX);  // (b: the least capacity of the stages, each taking its own)
  it.answer = std::make_shared<std::promise<int>>();
  std::future<int> f = it.answer->get_future();
  d_->post(std::move(it));
  return f.get();  // (released with -1 when the split breaks)
}

void Link::drop_kv(int seq) { d_->post(control(kDropKv, seq)); }
void Link::reset(int seq) { d_->post(control(kReset, seq)); }

int Link::new_checkpoint() {
  const int id = d_->ckpts++;
  d_->post(control(kNewCkpt, -1, id));
  return id;
}

void Link::save(int seq, int ckpt, int pos) { d_->post(control(kSave, seq, ckpt, 0, pos)); }
void Link::restore(int seq, int ckpt) { d_->post(control(kRestore, seq, ckpt)); }

void Link::flush(int seq, int n_pend) {
  Impl::Item it = control(kFlush, seq);
  it.h.n_pend = n_pend;
  d_->post(std::move(it));
}

void Link::rope(int seq, int rope_n, int rope_delta, const int32_t* table) {
  Impl::Item it = control(kRope, seq, rope_n, rope_delta);
  if (rope_n > 0) it.payload.assign(reinterpret_cast<const char*>(table), size_t(rope_n) * 12);
  d_->post(std::move(it));
}

std::string Link::lost() const {
  if (!d_->broken) return "";
  std::lock_guard<std::mutex> lk(d_->mu);
  return d_->why;
}

}  // namespace ni::cluster
