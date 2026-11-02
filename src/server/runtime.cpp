#include "server/runtime.h"
#include <httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <uuid/uuid.h>

namespace ni::runtime {
namespace {
namespace fs = std::filesystem;
std::mutex observer_mutex;
std::weak_ptr<Instance> observer;

double clock_ms() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string dump(const json& j) {
  return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

void private_directory(const std::string& directory) {
  fs::create_directories(directory);
  struct stat st {};
  if (lstat(directory.c_str(), &st) || !S_ISDIR(st.st_mode) || st.st_uid != getuid() ||
      chmod(directory.c_str(), 0700))
    throw std::runtime_error("instance registry must be a directory owned by this user");
}

void write_record(const std::string& path, const json& record) {
  const std::string tmp = path + "." + record.at("run_id").get<std::string>() + ".tmp";
  const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) throw std::runtime_error("cannot create instance record");
  const std::string bytes = dump(record);
  size_t at = 0;
  while (at < bytes.size()) {
    const ssize_t n = write(fd, bytes.data() + at, bytes.size() - at);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    at += size_t(n);
  }
  const bool synced = fsync(fd) == 0;
  close(fd);
  if (at == bytes.size() && synced && rename(tmp.c_str(), path.c_str()) == 0) return;
  unlink(tmp.c_str());
  throw std::runtime_error("cannot publish instance record");
}

bool trusted_record(const json& r) {
  if (!r.is_object() || r.value("protocol", 0) != kProtocol ||
      !valid_id(r.value("instance_id", "")) || !valid_id(r.value("run_id", "")) ||
      !valid_id(r.value("token", ""))) return false;
  const std::string url = r.value("control_url", "");
  const std::string prefix = "http://127.0.0.1:";
  if (url.rfind(prefix, 0) != 0) return false;
  const std::string port = url.substr(prefix.size());
  return !port.empty() && port.size() <= 5 &&
         port.find_first_not_of("0123456789") == std::string::npos &&
         std::stoi(port) > 0 && std::stoi(port) <= 65535;
}
}  // namespace

bool valid_id(const std::string& id) {
  return !id.empty() && id.size() <= 64 &&
         id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") == std::string::npos;
}

std::string new_id() {
  uuid_t bytes;
  uuid_generate_random(bytes);
  char text[37];
  uuid_unparse_lower(bytes, text);
  return text;
}

json call(const json& record, const std::string& method, const std::string& path, const json& body) {
  if (!trusted_record(record)) throw std::runtime_error("invalid instance record");
  httplib::Client client(record.at("control_url").get<std::string>());
  client.set_connection_timeout(1);
  client.set_read_timeout(3);
  client.set_write_timeout(3);
  client.set_follow_location(false);
  const httplib::Headers headers{{"Authorization", "Bearer " + record.at("token").get<std::string>()},
                                {"X-Kapitsa-Run-ID", record.at("run_id").get<std::string>()}};
  const auto reply = method == "GET" ? client.Get(path, headers)
                                    : client.Post(path, headers, dump(body), "application/json");
  if (!reply) throw std::runtime_error("instance is not reachable");
  json j = json::parse(reply->body);
  if (reply->status != 200) throw std::runtime_error(j.value("error", "control request failed"));
  if (j.value("instance_id", "") != record.at("instance_id").get<std::string>() ||
      j.value("run_id", "") != record.at("run_id").get<std::string>() || j.value("protocol", 0) != kProtocol)
    throw std::runtime_error("instance identity changed; discover again");
  return j;
}

std::vector<json> discover(const std::string& directory) {
  std::vector<json> result;
  struct stat dir {};
  if (lstat(directory.c_str(), &dir) || !S_ISDIR(dir.st_mode) ||
      dir.st_uid != getuid() || (dir.st_mode & 0077)) return result;
  std::error_code ec;
  for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
    if (it->path().extension() != ".json") continue;
    struct stat st {};
    if (lstat(it->path().c_str(), &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid() ||
        (st.st_mode & 0077) || st.st_size > 16384) continue;
    try {
      std::ifstream input(it->path());
      json record = json::parse(input);
      if (trusted_record(record)) result.push_back(std::move(record));
    } catch (const std::exception&) {
      // A crashed writer or an old protocol must not hide the other instances.
    }
  }
  return result;
}

struct Instance::Impl {
  struct Request {
    std::string client;
    std::shared_ptr<LiveDecode> live;
    double started = clock_ms();
    std::deque<std::pair<double, int>> samples;
  };
  json record, state;
  std::string path;
  int lock_fd = -1;
  httplib::Server control;
  std::thread listener, stopper;
  std::mutex mutex;
  std::condition_variable cv;
  bool closing = false, stopping = false;
  uint64_t sequence = 0, request_sequence = 0;
  std::deque<json> events;
  std::map<std::string, Request> requests;
  std::function<json()> read_details;
  std::function<void()> stop;

  ~Impl() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      closing = true;
      cv.notify_all();
    }
    control.stop();
    if (listener.joinable()) listener.join();
    if (stopper.joinable()) stopper.join();
    if (lock_fd >= 0) {
      // Only the lock owner may remove its record. The lock file stays put:
      // unlinking it would let newcomers lock a different inode.
      if (!path.empty()) unlink(path.c_str());
      close(lock_fd);
    }
  }

  void emit(const std::string& type, const json& data) {
    events.push_back({{"seq", ++sequence}, {"type", type}, {"data", data}});
    if (events.size() > 256) events.pop_front();
  }

  json snapshot() {
    std::function<json()> read;
    {
      std::lock_guard<std::mutex> lk(mutex);
      read = read_details;
    }
    json extra = read ? read() : json::object();
    std::lock_guard<std::mutex> lk(mutex);
    json j = state;
    j.update(extra);
    if (state.value("state", "") == "stopping") j["state"] = "stopping";
    j["protocol"] = kProtocol;
    j["instance_id"] = record.at("instance_id");
    j["run_id"] = record.at("run_id");
    j["name"] = record.at("name");
    j["pid"] = record.at("pid");
    j["cursor"] = sequence;
    j["requests"] = json::array();
    const double now = clock_ms();
    for (auto& [id, r] : requests) {
      const int tokens = r.live->tokens.load();
      if (r.samples.empty() || now - r.samples.back().first >= 100) r.samples.emplace_back(now, tokens);
      while (r.samples.size() > 2 && now - r.samples[1].first > 1500) r.samples.pop_front();
      const double elapsed = now - r.samples.front().first;
      const int phase = r.live->phase.load();
      static const char* phases[] = {"waiting", "prefill", "writing", "done"};
      const int cycles = r.live->cycles.load();
      j["requests"].push_back({{"request_id", id}, {"client", r.client},
        {"phase", phases[std::clamp(phase, 0, 3)]}, {"prompt_tokens", r.live->prompt.load()},
        {"cached_tokens", r.live->cached.load()}, {"read_tokens", r.live->read.load()},
        {"completion_tokens", tokens}, {"elapsed_ms", now - r.started},
        {"tokens_per_second", elapsed > 0 ? 1000 * (tokens - r.samples.front().second) / elapsed : 0},
        {"tokens_per_step", cycles > 0 ? double(tokens) / cycles : 0}, {"spec", r.live->spec.load()}});
    }
    return j;
  }
};

Instance::Instance(const std::string& directory, const std::string& id, const std::string& name,
                   const std::string& api_url, const std::string& model, const std::string& version,
                   std::function<void()> stop) : impl_(std::make_unique<Impl>()) {
  if (!valid_id(id)) throw std::invalid_argument("instance id must be 1-64 letters, digits, '-' or '_'");
  private_directory(directory);
  auto& p = *impl_;
  const std::string lock = directory + "/" + id + ".lock";
  p.lock_fd = open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (p.lock_fd < 0) throw std::runtime_error("cannot open instance lock");
  struct stat st {};
  if (fstat(p.lock_fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid() || (st.st_mode & 0077))
    throw std::runtime_error("instance lock is not private");
  if (flock(p.lock_fd, LOCK_EX | LOCK_NB)) throw std::runtime_error("instance '" + id + "' is already running");
  // Set the path only after acquiring the lock: a failed second start must not
  // remove the first process's record during exception unwinding.
  p.path = directory + "/" + id + ".json";
  p.stop = std::move(stop);
  const std::string run = new_id(), token = new_id();
  p.control.set_pre_routing_handler([run, token](const auto& req, auto& res) {
    if (req.get_header_value("Authorization") == "Bearer " + token &&
        req.get_header_value("X-Kapitsa-Run-ID") == run)
      return httplib::Server::HandlerResponse::Unhandled;
    res.status = 401;
    res.set_content(R"({"error":"invalid instance credential or execution"})", "application/json");
    return httplib::Server::HandlerResponse::Handled;
  });
  p.control.set_payload_max_length(4096);
  p.control.new_task_queue = [] { return new httplib::ThreadPool(2); };
  p.control.set_read_timeout(3);
  p.control.set_write_timeout(3);
  p.control.Get("/v1/status", [&p](const auto&, auto& res) {
    res.set_content(dump(p.snapshot()), "application/json");
  });
  p.control.Get("/v1/events", [&p](const auto& req, auto& res) {
    uint64_t after = 0;
    try {
      const std::string cursor = req.get_param_value("after");
      if (cursor.empty() || cursor.size() > 20 || cursor.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("invalid cursor");
      after = std::stoull(cursor);
    } catch (const std::exception&) {
      res.status = 400;
      res.set_content(R"({"error":"after must be a nonnegative event cursor"})", "application/json");
      return;
    }
    json reply = p.snapshot();
    {
      std::lock_guard<std::mutex> lk(p.mutex);
      // Bound replay to the snapshot cursor: a concurrently arriving event
      // belongs to the next poll, never beyond the cursor returned to this one.
      const uint64_t through = reply.at("cursor").get<uint64_t>();
      reply["reset"] = after > through || (!p.events.empty() && after < p.events.front().at("seq").get<uint64_t>() - 1);
      reply["events"] = json::array();
      for (const auto& e : p.events)
        if (e.at("seq").get<uint64_t>() > after && e.at("seq").get<uint64_t>() <= through)
          reply["events"].push_back(e);
    }
    res.set_content(dump(reply), "application/json");
  });
  p.control.Post("/v1/stop", [&p](const auto&, auto& res) {
    json reply = p.snapshot();
    reply["state"] = "stopping";
    res.set_content(dump(reply), "application/json");
    std::lock_guard<std::mutex> lk(p.mutex);
    p.stopping = true;
    p.state["state"] = "stopping";
    p.emit("state", {{"state", "stopping"}});
    p.cv.notify_all();
  });
  const int port = p.control.bind_to_any_port("127.0.0.1");
  if (port < 0) throw std::runtime_error("cannot bind instance control endpoint");
  p.record = {{"protocol", kProtocol}, {"instance_id", id}, {"run_id", run}, {"name", name},
    {"pid", getpid()}, {"control_url", "http://127.0.0.1:" + std::to_string(port)}, {"token", token}};
  p.state = {{"state", "starting"}, {"api_url", api_url}, {"model", model}, {"version", version}};
  p.emit("state", {{"state", "starting"}});
  p.listener = std::thread([&p] { p.control.listen_after_bind(); });
  p.control.wait_until_ready();
  p.stopper = std::thread([&p] {
    std::unique_lock<std::mutex> lk(p.mutex);
    p.cv.wait(lk, [&p] { return p.closing || p.stopping; });
    if (p.closing) return;
    // Let the acknowledgment leave before stopping the runtime. The stop is
    // local to this instance, never a PID taken from a stale registry record.
    if (p.cv.wait_for(lk, std::chrono::milliseconds(200), [&p] { return p.closing; })) return;
    const auto stop = p.stop;
    lk.unlock();
    if (stop) stop();
  });
  write_record(p.path, p.record);
}

Instance::~Instance() = default;
const json& Instance::record() const { return impl_->record; }
json Instance::snapshot() { return impl_->snapshot(); }

void Instance::update(const json& fields) {
  std::lock_guard<std::mutex> lk(impl_->mutex);
  impl_->state.update(fields);
  impl_->emit("state", fields);
}

void Instance::details(std::function<json()> read) {
  std::lock_guard<std::mutex> lk(impl_->mutex);
  impl_->read_details = std::move(read);
}

void Instance::event(const std::string& type, const json& data) {
  std::lock_guard<std::mutex> lk(impl_->mutex);
  impl_->emit(type, data);
}

std::string Instance::begin_request(const std::string& client, std::shared_ptr<LiveDecode> live) {
  std::lock_guard<std::mutex> lk(impl_->mutex);
  const std::string id = std::to_string(++impl_->request_sequence);
  impl_->requests.emplace(id, Impl::Request{client.substr(0, 128), std::move(live)});
  impl_->emit("request_started", {{"request_id", id}, {"client", client.substr(0, 128)}});
  return id;
}

void Instance::finish_request(const std::string& id, const ChatResult& r) {
  std::lock_guard<std::mutex> lk(impl_->mutex);
  if (!impl_->requests.erase(id)) return;
  json summary{{"request_id", id}, {"finish_reason", r.finish_reason}, {"error", r.error},
    {"prompt_tokens", r.prompt_tokens}, {"cached_tokens", r.cached_tokens}, {"completion_tokens", r.completion_tokens},
    {"prefill_ms", r.prefill_ms}, {"decode_ms", r.decode_ms}, {"queue_ms", r.queue_ms},
    {"tokens_per_second", r.completion_tokens > 1 && r.decode_ms > 0 ? 1000 * (r.completion_tokens - 1) / r.decode_ms : 0},
    {"prefill_tokens_per_second", r.prefill_exact && r.prefill_ms > 0 ?
      1000 * (r.prompt_tokens - r.cached_tokens) / r.prefill_ms : 0}, {"kflash_source", r.kflash_source}};
  impl_->state["last_request"] = summary;
  impl_->emit("request_finished", summary);
}

void observe(const std::shared_ptr<Instance>& instance) {
  std::lock_guard<std::mutex> lk(observer_mutex);
  observer = instance;
}

std::shared_ptr<Instance> observed() {
  std::lock_guard<std::mutex> lk(observer_mutex);
  return observer.lock();
}

void log_event(int level, const char* message) {
  if (const auto instance = observed())
    instance->event("log", {{"level", level}, {"message", std::string(message).substr(0, 4096)}});
}
}  // namespace ni::runtime
