#include "server/events.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace ni::server {

const char* reset_reason(Resume r) {
  switch (r) {
    case Resume::expired: return "expired";
    case Resume::run_changed: return "run_changed";
    case Resume::invalid_cursor: return "invalid_cursor";
    case Resume::live:
    case Resume::replay: return "";
  }
  return "";
}

struct Events::Reader {
  std::deque<std::shared_ptr<const Event>> unsent;
  size_t bytes = 0;     // of unsent's data lines
  bool closed = false;  // fell kMaxUnsent behind, or the bus ended
};

struct Events::Core {
  size_t keep;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::shared_ptr<const Event>> ring;
  uint64_t seq = 0;
  bool closed = false;
  std::vector<std::shared_ptr<Reader>> readers;

  // Under mu: hands `e` to r, or closes r when that would put it more than kMaxUnsent behind
  static void give(Reader& r, const std::shared_ptr<const Event>& e) {
    if (r.closed) return;
    const size_t n = e->type.size() + e->data.size();
    if (r.bytes + n > kMaxUnsent) {
      r.closed = true;
      r.unsent.clear();
      r.bytes = 0;
      return;
    }
    r.unsent.push_back(e);
    r.bytes += n;
  }
};

Events::Events(std::string run_id, size_t keep) : run_id_(std::move(run_id)), core_(std::make_shared<Core>()) {
  core_->keep = std::max<size_t>(keep, 1);
}

Events::~Events() { close(); }

uint64_t Events::publish(const std::string& type, const nlohmann::json& data) {
  auto e = std::make_shared<Event>();
  e->type = type;
  e->data = data.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);  // outside the lock
  Core& c = *core_;
  uint64_t seq;
  {
    std::lock_guard<std::mutex> lk(c.mu);
    seq = e->seq = ++c.seq;
    std::shared_ptr<const Event> ce = std::move(e);
    c.ring.push_back(ce);
    if (c.ring.size() > c.keep) c.ring.pop_front();
    if (!c.closed)
      for (auto& r : c.readers) Core::give(*r, ce);
  }
  c.cv.notify_all();
  return seq;
}

uint64_t Events::last_seq() const {
  std::lock_guard<std::mutex> lk(core_->mu);
  return core_->seq;
}

namespace {

// `<run_id>.<seq>` with run_id hex (16 to 64 digits) and seq a positive decimal: the run and the seq, or false
bool parse_cursor(const std::string& s, std::string* run, uint64_t* seq) {
  const size_t dot = s.rfind('.');
  if (dot == std::string::npos || dot < 16 || dot > 64 || s.size() - dot - 1 < 1 || s.size() - dot - 1 > 19) return false;
  for (size_t i = 0; i < dot; ++i)
    if (!isxdigit(uint8_t(s[i])) || isupper(uint8_t(s[i]))) return false;
  uint64_t v = 0;
  for (size_t i = dot + 1; i < s.size(); ++i) {
    if (!isdigit(uint8_t(s[i]))) return false;
    v = v * 10 + uint64_t(s[i] - '0');
  }
  if (v == 0 || s[dot + 1] == '0') return false;
  *run = s.substr(0, dot);
  *seq = v;
  return true;
}

}  // namespace

std::shared_ptr<Events::Stream> Events::open(const std::string& cursor, Resume* how) {
  Core& c = *core_;
  std::string run;
  uint64_t after = 0;
  const bool parsed = !cursor.empty() && parse_cursor(cursor, &run, &after);
  std::lock_guard<std::mutex> lk(c.mu);
  if (c.closed || int(c.readers.size()) >= kMaxStreams) return nullptr;
  auto r = std::make_shared<Reader>();
  uint64_t start = c.seq;
  Resume h = Resume::live;
  if (!cursor.empty()) {
    if (!parsed) h = Resume::invalid_cursor;
    else if (run != run_id_) h = Resume::run_changed;
    else if (after > c.seq) h = Resume::invalid_cursor;
    else if (!c.ring.empty() && after + 1 < c.ring.front()->seq) h = Resume::expired;
    else {
      h = Resume::replay;
      start = after;
      for (const auto& e : c.ring)
        if (e->seq > after) Core::give(*r, e);
    }
  }
  c.readers.push_back(r);
  *how = h;
  return std::shared_ptr<Stream>(new Stream(core_, std::move(r), start));
}

void Events::close() {
  Core& c = *core_;
  {
    std::lock_guard<std::mutex> lk(c.mu);
    c.closed = true;
    for (auto& r : c.readers) {
      r->closed = true;
      r->unsent.clear();
      r->bytes = 0;
    }
  }
  c.cv.notify_all();
}

Events::Stream::~Stream() {
  std::lock_guard<std::mutex> lk(core_->mu);
  auto& v = core_->readers;
  v.erase(std::remove(v.begin(), v.end(), r_), v.end());
}

std::optional<Event> Events::Stream::next(int ms) {
  std::unique_lock<std::mutex> lk(core_->mu);
  core_->cv.wait_for(lk, std::chrono::milliseconds(std::max(ms, 0)), [&] { return r_->closed || !r_->unsent.empty(); });
  if (r_->closed || r_->unsent.empty()) return std::nullopt;
  Event e = *r_->unsent.front();
  r_->bytes -= e.type.size() + e.data.size();
  r_->unsent.pop_front();
  return e;
}

bool Events::Stream::closed() const {
  std::lock_guard<std::mutex> lk(core_->mu);
  return r_->closed;
}

LogRing::LogRing(size_t lines) : slots_(std::max<size_t>(lines, 1)) {
  for (auto& l : slots_) l.message.reserve(kMaxLine);
}

void LogRing::push(int level, const char* msg) {
  const size_t n = strnlen(msg, kMaxLine);
  std::lock_guard<std::mutex> lk(mu_);
  if (count_ == slots_.size()) {  // full: the oldest goes
    head_ = (head_ + 1) % slots_.size();
    --count_;
    ++dropped_;
  }
  Line& l = slots_[(head_ + count_) % slots_.size()];
  l.level = level;
  l.message.assign(msg, n);  // within the reserved capacity: no allocation
  ++count_;
}

std::vector<LogRing::Line> LogRing::drain(uint64_t* dropped) {
  std::vector<Line> out;
  std::lock_guard<std::mutex> lk(mu_);
  out.reserve(count_);
  for (size_t i = 0; i < count_; ++i) out.push_back(slots_[(head_ + i) % slots_.size()]);
  head_ = (head_ + count_) % slots_.size();
  count_ = 0;
  if (dropped) *dropped = dropped_;
  dropped_ = 0;
  return out;
}

nlohmann::json log_event(const LogRing::Line& line) {
  const char* level = line.level == 0 ? "error" : line.level >= 2 ? "debug" : "info";
  if (line.level == 1 && line.message.rfind("warning", 0) == 0) level = "warn";
  return {{"level", level}, {"message", line.message}};
}

}  // namespace ni::server
