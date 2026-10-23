// Several Macs in `kapitsa serve` (route.h).
#include "server/route.h"

#include "api/anthropic.h"
#include "api/chat_completions.h"
#include "api/responses.h"
#include "util/log.h"

#include <sys/stat.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>

namespace ni {

std::string model_key(const hub::ModelRef& ref) {
  if (ref.kind == hub::ModelRef::kCatalog && ref.model && ref.variant) return hub::full_name(*ref.model, *ref.variant);
  if (ref.kind == hub::ModelRef::kLocal) {
    std::string base = ref.dir.substr(ref.dir.find_last_of('/') + 1);
    for (auto& c : base) c = char(tolower(uint8_t(c)));
    const hub::CatalogModel* best = nullptr;  // the longest catalog name it starts with
    for (const auto& m : hub::catalog())
      if (base.size() > m.name.size() + 1 && base.compare(0, m.name.size() + 1, m.name + "-") == 0 &&
          (!best || m.name.size() > best->name.size()))
        best = &m;
    if (best) return best->name + ":" + base.substr(best->name.size() + 1);
  }
  return ref.id;
}

bool runs_whole_here(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget) {
  struct stat st {};
  if (ref.kind == hub::ModelRef::kLocal && stat((ref.dir + "/config.json").c_str(), &st) != 0) return true;
  try {
    const std::string dir = hub::plan_dir(ref);
    if (dir.empty()) return false;
    EngineOptions po = opt;  // as serve's own check before a download
    po.context = context_within_limit(dir, opt.context);
    const int slots = po.slots > 0 ? po.slots : auto_slots(dir, po, budget);
    const LoadPlan lp = plan_load(dir, po.dflash, po.spec_depth == 0, po.context, slots, budget, po.prefill_chunk);
    const StartNeed need = start_need(dir, lp, lp.prefill_chunk, po.context, slots);
    if (need.bytes > budget) return false;
    // streamed experts (a Qwen3.8-Flash-Next Q2_0 pack): whole when the budget holds every one of them besides
    const MemPlan mp = model_plan(dir, lp.prefill_chunk);
    return mp.streamed == 0 || need.bytes + mp.streamed <= budget;
  } catch (const std::exception&) {
    return ref.kind == hub::ModelRef::kLocal;  // a folder that cannot be planned: the engine says why
  }
}

namespace {

// Not forwarded: what belongs to one connection, the lengths httplib sets, and httplib's own REMOTE_ADDR and the like
bool passes(const std::string& header) {
  std::string k = header;
  for (auto& c : k) c = char(tolower(uint8_t(c)));
  for (const char* skip : {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "upgrade", "te",
                           "trailer", "proxy-connection", "accept-encoding", "remote_addr", "remote_port", "local_addr",
                           "local_port"})
    if (k == skip) return false;
  return true;
}

// A reply's error in the format of the API at `path`
std::string error_body(const std::string& path, const std::string& msg) {
  if (path.rfind("/v1/messages", 0) == 0) return api::dump(api::messages_error("overloaded_error", msg));
  if (path.rfind("/v1/responses", 0) == 0) return api::dump(api::responses_error(msg, "server_error"));
  return api::dump(api::chat_error(msg, "server_error"));
}

// The last event of a stream cut short, in the format of the API at `path`
std::string error_event(const std::string& path, const std::string& msg) {
  if (path.rfind("/v1/messages", 0) == 0) return "event: error\ndata: " + api::dump(api::messages_error("api_error", msg)) + "\n\n";
  if (path.rfind("/v1/responses", 0) == 0)
    return "event: error\ndata: " + api::dump(ojson{{"type", "error"}, {"code", "server_error"}, {"message", msg}}) + "\n\n";
  return "data: " + api::dump(api::chat_error(msg, "server_error")) + "\n\ndata: [DONE]\n\n";
}

// One forwarded request: the reply as it comes back, handed from the client's thread to the connection's
struct Forward {
  std::mutex mu;
  std::condition_variable cv;
  bool head = false, done = false, cancelled = false;
  int status = 0;
  httplib::Headers headers;
  std::deque<std::string> chunks;
  std::string error;                  // no reply, or one cut short
  httplib::Client* client = nullptr;  // while the request is under way
  std::thread worker;
  void cancel() {  // the client is gone: the connection to the Mac closes, which stops its work on the reply
    std::lock_guard<std::mutex> lk(mu);
    cancelled = true;
    if (client) client->stop();
  }
  ~Forward() {
    cancel();
    if (worker.joinable()) worker.join();
  }
};

}  // namespace

Router::Router(std::shared_ptr<cluster::Browser> browser, std::string model, std::string head_id)
    : browser_(std::move(browser)), model_(std::move(model)), head_id_(std::move(head_id)) {}

std::optional<cluster::MacInfo> Router::peer() const {
  const std::vector<cluster::MacInfo> peers = browser_->peers();
  std::optional<cluster::MacInfo> p;
  for (const auto& m : peers)
    if (!head_id_.empty() && m.id == head_id_ && m.rtt_ms >= 0 && !m.host.empty() && m.port > 0) p = m;
  if (!p) p = cluster::pick_server(peers, model_, false);
  if (p) {
    std::lock_guard<std::mutex> lk(mu_);
    last_ = *p;
  }
  return p;
}

cluster::MacInfo Router::last() const {
  std::lock_guard<std::mutex> lk(mu_);
  return last_;
}

std::string Router::away() const {
  const cluster::MacInfo l = last();
  const std::string model = cluster::model_base(model_);
  if (l.name.empty())
    return "No Mac on this network serves " + model + " whole now. This server keeps looking and sends requests to one "
           "as soon as it appears; restarted with --local it runs the model on this Mac.";
  return model + " runs on " + cluster::describe(l) + ", which does not answer now (asleep, stopped or off the network). "
         "This server keeps looking and sends requests there again once it is back; restarted with --local it runs the "
         "model on this Mac.";
}

void Router::add_routes(httplib::Server& srv, Done done) {
  auto handler = [this, done](const httplib::Request& req, httplib::Response& res) { forward(req, res, done); };
  srv.Get(R"(/v1/.*)", handler);
  srv.Post(R"(/v1/.*)", handler);
}

void Router::forward(const httplib::Request& req, httplib::Response& res, const Done& done) {
  const double t0 = now_ms();
  const auto to = peer();
  if (!to) {
    res.status = 503;
    res.set_header("Retry-After", "10");
    res.set_content(error_body(req.path, away()), "application/json");
    return done(req, 503, (now_ms() - t0) / 1000, "");
  }
  const std::string mac = to->name;
  auto job = std::make_shared<Forward>();
  httplib::Request up;
  up.method = req.method;
  up.path = req.target.empty() ? req.path : req.target;  // as it came, query included
  for (const auto& [k, v] : req.headers)
    if (passes(k)) up.headers.emplace(k, v);
  up.body = req.body;
  Forward* raw = job.get();
  up.response_handler = [raw](const httplib::Response& r) {
    std::lock_guard<std::mutex> lk(raw->mu);
    raw->status = r.status, raw->headers = r.headers, raw->head = true;
    raw->cv.notify_all();
    return !raw->cancelled;
  };
  up.content_receiver = [raw](const char* d, size_t n, uint64_t, uint64_t) {
    std::lock_guard<std::mutex> lk(raw->mu);
    raw->chunks.emplace_back(d, n);
    raw->cv.notify_all();
    return !raw->cancelled;
  };
  // a stream sends something every 15 s once it has started, but an Anthropic or Responses one starts only when the
  // request has a slot there (a busy Mac queues it); a whole reply comes after its last token
  bool stream = false;
  if (req.body.find("\"stream\"") != std::string::npos) {
    const ojson b = ojson::parse(req.body, nullptr, false);
    stream = b.is_object() && b.contains("stream") && b["stream"].is_boolean() && b["stream"].get<bool>();
  }
  job->worker = std::thread([raw, up = std::move(up), host = to->host, port = to->port, stream] {
    httplib::Client c(host, port);
    c.set_url_encode(false);
    c.set_connection_timeout(3, 0);
    c.set_read_timeout(stream ? 600 : 3600, 0);
    c.set_write_timeout(60, 0);
    {
      std::lock_guard<std::mutex> lk(raw->mu);
      if (raw->cancelled) {
        raw->done = true;
        raw->cv.notify_all();
        return;
      }
      raw->client = &c;
    }
    const auto r = c.send(up);
    std::lock_guard<std::mutex> lk(raw->mu);
    raw->client = nullptr;
    if (!r) raw->error = httplib::to_string(r.error());
    else if (!raw->head) raw->status = r->status, raw->headers = r->headers, raw->head = true;  // no body (204)
    raw->done = true;
    raw->cv.notify_all();
  });
  std::unique_lock<std::mutex> lk(job->mu);
  job->cv.wait(lk, [&] { return job->head || job->done; });
  if (!job->head) {  // no reply at all: the Mac is away (Bonjour has not said so yet)
    const std::string why = job->error;
    lk.unlock();
    res.status = 503;
    res.set_header("Retry-After", "10");
    res.set_content(error_body(req.path, cluster::model_base(model_) + " runs on " + cluster::describe(*to) +
                                             ", which did not answer (" + why + "). This server keeps looking and sends "
                                             "requests there again once it answers; restarted with --local it runs the "
                                             "model on this Mac."),
                    "application/json");
    return done(req, 503, (now_ms() - t0) / 1000, mac);
  }
  std::string type = "application/json";
  for (const auto& [k, v] : job->headers) {
    std::string l = k;
    for (auto& c : l) c = char(tolower(uint8_t(c)));
    if (l == "content-type") type = v;
    else if (passes(k)) res.set_header(k, v);
  }
  res.status = job->status;
  if (type.find("text/event-stream") == std::string::npos) {  // a whole reply
    job->cv.wait(lk, [&] { return job->done; });
    std::string body;
    for (const auto& c : job->chunks) body += c;
    if (!job->error.empty()) {  // cut short
      res.status = 502;
      body = error_body(req.path, cluster::describe(*to) + " stopped answering in the middle of the reply (" + job->error + ").");
      type = "application/json";
    }
    const int status = res.status;
    lk.unlock();
    res.set_content(body, type);
    return done(req, status, (now_ms() - t0) / 1000, mac);
  }
  lk.unlock();
  const std::string path = req.path;
  const int status = job->status;
  httplib::Request brief;  // what the request log reads of it
  brief.method = req.method, brief.path = req.path;
  brief.headers.emplace("User-Agent", req.get_header_value("User-Agent"));
  // a stream: relayed as it comes; when the client goes, the connection to the Mac closes too
  res.set_chunked_content_provider(type, [job, path, status, t0, mac, done, req_copy = std::move(brief)](size_t, httplib::DataSink& sink) {
    for (;;) {
      std::deque<std::string> batch;
      bool fin;
      std::string err;
      {
        std::unique_lock<std::mutex> l2(job->mu);
        job->cv.wait_for(l2, std::chrono::seconds(1), [&] { return !job->chunks.empty() || job->done; });
        batch.swap(job->chunks);
        fin = job->done;
        err = job->error;
      }
      bool gone = batch.empty() && !fin && !sink.is_writable();
      for (const auto& b : batch)
        if (!gone && !sink.write(b.data(), b.size())) gone = true;
      if (gone) {
        job->cancel();
        done(req_copy, 499, (now_ms() - t0) / 1000, mac);  // (nginx's "client closed request")
        return false;
      }
      if (fin) {
        if (!err.empty()) {
          const std::string e = error_event(path, mac + " stopped answering in the middle of the reply (" + err + ").");
          sink.write(e.data(), e.size());
        }
        sink.done();
        done(req_copy, err.empty() ? status : 502, (now_ms() - t0) / 1000, mac);
        return true;
      }
    }
  });
}

}  // namespace ni
