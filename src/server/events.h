// The server's event bus (docs/control-socket.md, "Events"): every state change, request and log line the server
// emits, numbered `<run_id>.<seq>`, kept for replay and handed to every open stream. The bus never waits for a reader:
// a stream that falls kMaxUnsent bytes behind is closed, and its client reconnects with Last-Event-ID.
#pragma once
#include <nlohmann/json.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ni::server {

struct Event {
  uint64_t seq = 0;
  std::string type;  // state, request_started, request_progress, request_finished, engine, log
  std::string data;  // the SSE data line: the event's JSON
};

// How a stream starts from its Last-Event-ID
enum class Resume {
  live,            // no cursor: from the next event
  replay,          // every event after the cursor, then live
  expired,         // the cursor is older than the oldest event kept
  run_changed,     // the cursor is another run's
  invalid_cursor,  // it does not parse, or names an event that never existed
};
// The reset event's `reason` for a Resume that cannot be continued ("" for live and replay)
const char* reset_reason(Resume r);

class Events {
 public:
  static constexpr size_t kKeep = 1024;           // events kept for replay
  static constexpr size_t kMaxUnsent = 1u << 20;  // bytes a stream may fall behind before it is closed
  // A replay larger than this starts with an `expired` reset instead: replaying it would close the stream, or leave
  // so little room that the next events close it, and the client would reconnect with the same cursor forever
  static constexpr size_t kMaxReplay = kMaxUnsent / 2;
  static constexpr int kMaxStreams = 16;

  explicit Events(std::string run_id, size_t keep = kKeep);
  ~Events();
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;

  const std::string& run_id() const { return run_id_; }
  std::string id(uint64_t seq) const { return run_id_ + "." + std::to_string(seq); }
  // Adds an event and wakes the streams; never waits for one. Returns its seq (the first is 1).
  uint64_t publish(const std::string& type, const nlohmann::json& data);
  uint64_t last_seq() const;

  class Stream;
  // A stream from Last-Event-ID `cursor` ("": none). `how` says how it starts: for expired, run_changed and
  // invalid_cursor the caller sends one reset event first, whose id is the stream's start(). nullptr: kMaxStreams are
  // open (503 busy), or the bus has ended.
  std::shared_ptr<Stream> open(const std::string& cursor, Resume* how);
  // Ends every stream (the server stops); publish() still numbers events, but no stream gets them.
  void close();

 private:
  struct Reader;
  struct Core;  // shared with the streams, so a stream may outlive the bus
  const std::string run_id_;
  std::shared_ptr<Core> core_;
};

class Events::Stream {
 public:
  ~Stream();
  // The seq the stream continues after: the cursor's on replay, else the newest event's when it opened
  uint64_t start() const { return start_; }
  // The next event, waiting up to `ms`; nullopt on a timeout or once the stream is closed
  std::optional<Event> next(int ms);
  // It fell kMaxUnsent bytes behind, or the bus ended: the caller ends the response
  bool closed() const;

 private:
  friend class Events;
  Stream(std::shared_ptr<Core> core, std::shared_ptr<Reader> r, uint64_t start)
      : core_(std::move(core)), r_(std::move(r)), start_(start) {}
  std::shared_ptr<Core> core_;
  std::shared_ptr<Reader> r_;
  uint64_t start_;
};

// Log lines on their way to the bus, from any thread. push() copies the line into a fixed slot under a lock held only
// for that copy (no allocation once the slots are sized, no JSON, never a reader's wait), and drops the oldest line
// when every slot is full. The tick thread drains them into `log` events.
class LogRing {
 public:
  static constexpr size_t kMaxLine = 4096;  // a longer line is cut (the schema's maxLength)
  explicit LogRing(size_t lines = 256);
  void push(int level, const char* msg);  // level as log_sink's: 0 fatal, 1 logi, 2 logd
  struct Line {
    int level = 1;
    std::string message;
  };
  // The lines since the last drain, oldest first, and how many were dropped meanwhile
  std::vector<Line> drain(uint64_t* dropped);

 private:
  std::mutex mu_;
  std::vector<Line> slots_;
  size_t head_ = 0, count_ = 0;
  uint64_t dropped_ = 0;
};

// A log line as a `log` event's data: {"level": debug|info|warn|error, "message"}. A logi line that starts with
// "warning" is warn.
nlohmann::json log_event(const LogRing::Line& line);

}  // namespace ni::server
