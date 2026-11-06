#pragma once
#include "engine/engine.h"
#include "engine/history_gate.h"
#include "hub/hub.h"
#include "server/cli.h"
#include "server/ui.h"
#include "util/log.h"
#include "util/term.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace ni {

class ModelManager;

// RAII guard holding an Engine reference for an in-flight request
class RequestGuard : public std::enable_shared_from_this<RequestGuard> {
 public:
  ~RequestGuard();
  RequestGuard(const RequestGuard&) = delete;
  RequestGuard& operator=(const RequestGuard&) = delete;

  Engine& engine() const { return *eng_; }
  Engine* operator->() const { return eng_.get(); }
  Engine& operator*() const { return *eng_; }
  // An exported engine handle must keep its registration in the switch's request drain.
  std::shared_ptr<Engine> get() const { return {shared_from_this(), eng_.get()}; }

 private:
  friend class ModelManager;
  explicit RequestGuard(std::shared_ptr<Engine> eng) : eng_(std::move(eng)) {}
  std::shared_ptr<ModelManager> mgr_;
  std::shared_ptr<Engine> eng_;
};

class ModelManager : public std::enable_shared_from_this<ModelManager> {
 public:
  struct Error {
    enum class Code { Invalid, Busy };
    Code code = Code::Invalid;
    std::string message;
  };

  ModelManager() = default;
  ~ModelManager() {
    stop_idle_watch();
    stop_prefill();
  }

  void init(const EngineOptions& opt, const hub::ModelRef& ref, int context_req,
            size_t budget, bool tui, time_t started, const std::string& started_at) {
    base_opt_ = opt;
    opt_ = opt;
    ref_ = ref;
    context_req_ = context_req;
    budget_ = budget;
    tui_ = tui;
    started_ = started;
    started_at_ = started_at;

    if (tui_) ui::loading_begin(ref_.name);
    engine_ = load_engine(opt_);
    remember_info();
    if (tui_) ui::loading_end(ref_.name);
  }

  // --idle-unload: after `idle_s` seconds with no request the engine is released (its weights, KV cache and
  // activations: the whole GPU footprint) and loaded again by the next request. `only_pressed`: only while macOS's
  // memory pressure is at warning or critical (the default: an idle server then gives its memory back where it is
  // wanted). `veto` says when the engine must stay (a pair of Macs holding one split model). 0 seconds: never.
  void set_idle_unload(int idle_s, bool only_pressed, std::function<bool()> veto = nullptr) {
    stop_idle_watch();
    if (idle_s <= 0) return;
    idle_s_ = idle_s;
    only_pressed_ = only_pressed;
    veto_ = std::move(veto);
    touch();
    idle_stop_ = false;
    idle_th_ = std::thread([this] { idle_watch(); });
  }
  bool unloaded() const { return unloaded_.load(); }

  // --allow-switch: a request naming another model (or POST /v1/models/switch) may swap the served model. Off by
  // default: clients send their own model names (Claude Code, Cursor), which the server ignores, as it always did.
  void set_allow_switch(bool on) { allow_switch_ = on; }
  bool allow_switch() const { return allow_switch_; }
  // True while a switch/reload drains requests or loads the model: reject new requests instead of extending the drain.
  bool switching() const { return switching_.load(); }

  // Acquire active engine for a request, switching on demand if another valid model is requested and --allow-switch.
  std::shared_ptr<RequestGuard> acquire_request(const std::string& requested_model, Error* error_out = nullptr) {
    if (switching_.load()) {
      if (error_out) *error_out = {Error::Code::Busy, "another model load is in progress"};
      return nullptr;
    }
    {
      std::shared_lock<std::shared_mutex> lk(rw_mu_);
      if (switching_.load()) {
        if (error_out) *error_out = {Error::Code::Busy, "another model load is in progress"};
        return nullptr;
      }
      if (!allow_switch_ || is_current_model_unlocked(requested_model)) {
        if (!unloaded_.load()) return acquire_unlocked();
      }
    }
    // the engine was released while idle: this request loads it again (a model it names instead is a switch)
    if (unloaded_.load() && (!allow_switch_ || is_current_model_unlocked(requested_model))) {
      if (!load_back(error_out)) return nullptr;
      std::shared_lock<std::shared_mutex> lk(rw_mu_);
      return acquire_unlocked();
    }
    std::shared_ptr<RequestGuard> guard;
    switch_to_impl(requested_model, error_out, &guard);
    return guard;
  }

  // The active engine for a request that never switches models (POST /v1/draft): null when `requested_model` names
  // another one (with --allow-switch; without it the server serves its model whatever a request names)
  std::shared_ptr<RequestGuard> acquire_current(const std::string& requested_model, Error* error_out = nullptr) {
    if (switching_.load()) {
      if (error_out) *error_out = {Error::Code::Busy, "another model load is in progress"};
      return nullptr;
    }
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    if (switching_.load()) {
      if (error_out) *error_out = {Error::Code::Busy, "another model load is in progress"};
      return nullptr;
    }
    if (!allow_switch_ || is_current_model_unlocked(requested_model)) return acquire_unlocked();
    return nullptr;
  }

  // Explicitly switch the active model
  std::shared_ptr<Engine> switch_to(const std::string& target_model, Error* error_out = nullptr) {
    std::shared_ptr<RequestGuard> guard;
    switch_to_impl(target_model, error_out, &guard);
    return guard ? guard->get() : nullptr;
  }

 private:
  // An on-demand switch registers its request before releasing the selected engine's lock.
  std::shared_ptr<Engine> switch_to_impl(const std::string& target_model, Error* error_out,
                                         std::shared_ptr<RequestGuard>* request_out) {
    if (!allow_switch_) {
      if (error_out) error_out->message = "model switching is off (start the server with --allow-switch)";
      return nullptr;
    }
    if (target_model.empty()) {
      if (error_out) error_out->message = "model name cannot be empty";
      return nullptr;
    }
    if (switching_.exchange(true)) {
      if (error_out) *error_out = {Error::Code::Busy, "another model switch is in progress"};
      return nullptr;
    }
    struct Clear {
      std::atomic<bool>& f;
      ~Clear() { f = false; }
    } clear{switching_};

    std::unique_lock<std::shared_mutex> lk(rw_mu_);
    if (is_current_model_unlocked(target_model)) {
      if (unloaded_.load() && !load_back_locked(nullptr)) return nullptr;  // (released while idle: load it again)
      if (request_out) *request_out = acquire_unlocked();
      return engine_;
    }
    lk.unlock();  // resolution/download must not block the old model's request completion

    hub::ModelRef new_ref;
    try {
      const hub::ModelRef named = hub::resolve(expand_home(target_model));
      new_ref = named;
      if (named.kind == hub::ModelRef::kCatalog && !named.variant_given) {
        bool fits = false;
        new_ref = pick_variant(*named.model, base_opt_, budget_, &fits);
      }
    } catch (const std::exception& e) {
      if (error_out) error_out->message = e.what();
      return nullptr;
    }

    if (new_ref.kind == hub::ModelRef::kLocal) {
      struct stat st;
      if (stat((new_ref.dir + "/config.json").c_str(), &st) != 0) {
        if (error_out) error_out->message = new_ref.dir + " has no config.json";
        return nullptr;
      }
    }

    EngineOptions new_opt = base_opt_;
    try {
      hub::PullOptions po;
      new_opt.model_dir = hub::ensure(new_ref, po);
    } catch (const std::exception& e) {
      if (error_out) error_out->message = e.what();
      return nullptr;
    }

    const int limit = model_context_limit(new_opt.model_dir);
    if (context_req_ == kContextMax) {
      new_opt.context = limit > 0 ? limit : kDefaultContext;
    } else if (context_req_ > 0) {
      new_opt.context = (limit > 0 && context_req_ > limit) ? limit : context_req_;
    } else {
      ContextWhy why = ContextWhy::kDefault;
      new_opt.context = default_context(new_opt.model_dir, new_opt, budget_, &why);
    }
    new_opt.model_id = new_ref.id;

    // Another architecture (issue 447) is refused here, before the old engine is released and the new one is built:
    // a switch must not cost the running model its place, and the engine's constructor refuses it too, as a backstop
    // for every other caller (`hub::check_supported` only covers the Hub's repos, and this one may be any folder).
    if (const std::string why = unsupported_model(new_opt.model_dir); !why.empty()) {
      if (error_out) error_out->message = why;
      return nullptr;
    }

    lk.lock();
    wait_for_requests(lk);
    stop_prefill();
    {
      LoopKv kv = LoopKv::last_loop;
      if (!new_opt.loop_kv.empty()) parse_loop_kv(new_opt.loop_kv, &kv);
      Dory::set_serving(new_opt.loops, kv);
    }

    logi("switching model from %s to %s", ref_.name.c_str(), new_ref.name.c_str());
    if (tui_) ui::note("Switching model to " + term::bold(new_ref.name));
    else printf("Switching model to %s...\n", new_ref.name.c_str());
    fflush(stdout);

    // All engine handles and history workers are gone, so its allocation scope can finish before the next starts.
    engine_.reset();

    if (tui_) ui::loading_begin(new_ref.name);
    try {
      engine_ = load_engine(new_opt);
    } catch (const std::exception& e) {
      if (tui_) ui::loading_end(new_ref.name);
      if (error_out) error_out->message = std::string("Engine initialization failed: ") + e.what();
      // the old model comes back, so a failed switch (memory, a bad checkpoint) leaves the server serving
      logi("switch to %s failed (%s); reloading %s", new_ref.name.c_str(), e.what(), ref_.name.c_str());
      if (tui_) ui::loading_begin(ref_.name);
      try {
        engine_ = load_engine(opt_);
      } catch (const std::exception& e2) {
        logi("reloading %s failed too: %s", ref_.name.c_str(), e2.what());
        if (error_out) error_out->message += std::string("; reloading the previous model failed too: ") + e2.what();
      }
      if (tui_) ui::loading_end(ref_.name);
      return nullptr;
    }
    if (tui_) ui::loading_end(new_ref.name);

    ref_ = new_ref;
    opt_ = new_opt;
    unloaded_ = false;
    remember_info();
    touch();
    logi("switched model to %s (id: %s, context: %d)", ref_.name.c_str(), opt_.model_id.c_str(), opt_.context);
    if (request_out) *request_out = acquire_unlocked();
    return engine_;
  }

 public:
  // The same model loaded again with `opt` (several Macs: a pair's split made or undone, docs/usage.md). Requests in
  // flight finish first (those of a split whose other Macs are lost end at once); new ones get 503 while it loads.
  // `before` runs once the old engine is gone, before the new one loads (Qwen4Exp::set_split); when the new one fails,
  // `undo` runs and the model loads again as it was. False then, with *error_out.
  bool reload(const EngineOptions& opt, const std::function<void()>& before, const std::function<void()>& undo,
              std::string* error_out = nullptr) {
    if (switching_.exchange(true)) {
      if (error_out) *error_out = "another model load is in progress";
      return false;
    }
    struct Clear {
      std::atomic<bool>& f;
      ~Clear() { f = false; }
    } clear{switching_};
    std::unique_lock<std::shared_mutex> lk(rw_mu_);
    wait_for_requests(lk);
    stop_prefill();
    engine_.reset();
    if (before) before();
    if (tui_) ui::loading_begin(ref_.name);
    try {
      engine_ = load_engine(opt);
    } catch (const std::exception& e) {
      if (error_out) *error_out = e.what();
      logi("loading %s again failed (%s); loading it as it was", ref_.name.c_str(), e.what());
      if (undo) undo();
      try {
        engine_ = load_engine(opt_);
      } catch (const std::exception& e2) {
        logi("loading %s as it was failed too: %s", ref_.name.c_str(), e2.what());
        if (error_out) *error_out += std::string("; loading it as it was failed too: ") + e2.what();
      }
      if (tui_) ui::loading_end(ref_.name);
      return false;
    }
    if (tui_) ui::loading_end(ref_.name);
    opt_ = opt;
    unloaded_ = false;
    remember_info();
    touch();
    return true;
  }
  EngineOptions options() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return opt_;
  }
  // Requests in flight now
  int in_flight() const { return in_flight_.load(); }

  // On a stop signal: the serving engine's KFlash state and SSD prefix cache conversations. A switch in progress
  // (it holds the lock while a model loads) has no engine worth saving, so this does not wait for it.
  void save_state() {
    std::shared_lock<std::shared_mutex> lk(rw_mu_, std::try_to_lock);
    if (!lk.owns_lock() || !engine_) return;
    engine_->save_kflash();
    engine_->save_disk_cache();
  }

  void prefill_after(const ChatRequest& r, const ChatResult& out) {
    // Selection stays stable through registration; every path takes rw_mu_ before prefill_mu_.
    std::shared_lock<std::shared_mutex> rlk(rw_mu_);
    auto eng = engine_;
    if (!eng) return;
    eng->remember_reply(r, out);
    std::lock_guard<std::mutex> lk(prefill_mu_);
    // Several replies' history prefills can wait at once (two conversations on two slots finishing together):
    // each runs in its own thread, which waits for a quiet moment and takes the turn; the finished ones are
    // joined here, so this call does not wait for a thread that is still waiting. A newer request of the same
    // conversation (it sends the older request's messages back, after the system prompt, which may change) supersedes
    // the older reply's history prefill, which is told to stop; a stream of one-off requests is bounded by kMaxPending.
    // That bounds the threads (and the request copies they hold).
    std::vector<size_t> hashes = history_message_hashes(r.messages);
    for (auto& p : prefill_ths_)
      if (history_superseded(p.hashes, hashes)) p.stale->store(true);
    for (auto it = prefill_ths_.begin(); it != prefill_ths_.end();) {
      if (it->done->load()) {
        it->th.join();
        it = prefill_ths_.erase(it);
      } else {
        ++it;
      }
    }
    uint64_t epoch = model_epoch_.load();
    auto done = std::make_shared<std::atomic<bool>>(false);
    auto stale = std::make_shared<std::atomic<bool>>(false);
    std::thread th([this, eng, r, out, epoch, done, stale] {
      if (epoch == model_epoch_.load())
        eng->prefill_history(r, out, [this, epoch, stale] {
          if (epoch != model_epoch_.load()) return Engine::PrefillStop::kModelChanged;
          return stale->load() ? Engine::PrefillStop::kSuperseded : Engine::PrefillStop::kNone;
        });
      done->store(true);
    });
    prefill_ths_.push_back({std::move(th), done, stale, out.slot, std::move(hashes)});
    // (the oldest beyond the cap are told to stop too)
    size_t live = 0;
    for (auto& p : prefill_ths_) live += !p.stale->load() && !p.done->load();
    for (auto& p : prefill_ths_)
      if (live > kMaxPending && !p.stale->load() && !p.done->load()) p.stale->store(true), --live;
  }

  std::string current_model_id() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return info_.model_id;
  }
  int current_context_length() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return info_.context;
  }
  int current_context_fit() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return info_.fit;
  }
  int current_slots() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return info_.slots;
  }
  std::string current_name() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return ref_.name;
  }
  hub::ModelRef current_ref() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return ref_;
  }
  // The engine serving now, or null when released while idle or while a model loads. For the server's own looks
  // (the pair watcher every 2 s, what the cluster is told): not a request, so it does not count as use for
  // --idle-unload, does not wait for a load and does not load the engine again.
  std::shared_ptr<Engine> current_engine() {
    if (switching_.load()) return nullptr;
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    if (switching_.load()) return nullptr;
    return engine_;
  }

  nlohmann::ordered_json list_models_json() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    using json = nlohmann::ordered_json;
    json data = json::array();
    std::set<std::string> seen;

    // 1. Current active model is first
    if (engine_ || unloaded_.load()) {
      const std::string cur_id = info_.model_id;
      seen.insert(cur_id);
      seen.insert(ref_.name);
      if (ref_.model) {
        seen.insert(ref_.model->name);
        for (const auto& a : ref_.model->aliases) seen.insert(a);
      }
      json item{
        {"id", cur_id},
        {"object", "model"},
        {"type", "model"},
        {"display_name", ref_.name.empty() ? cur_id : ref_.name},
        {"created", long(started_)},
        {"created_at", started_at_},
        {"owned_by", "nace"},
        {"context_length", info_.context},
        {"context_fit", info_.fit},
        {"slots", info_.slots},
        {"active", true},
        {"downloaded", true}
      };
      if (ref_.variant) item["quant"] = ref_.variant->quant;
      data.push_back(item);
    }

    // 2. Downloaded / installed models
    try {
      const auto installed = hub::installed();
      for (const auto& m : installed) {
        if (!m.complete) continue;
        if (seen.count(m.name) || ((engine_ || unloaded_.load()) && m.name == info_.model_id)) continue;
        seen.insert(m.name);

        std::string disp = m.name;
        const hub::CatalogModel* cat = hub::find_model(m.name);
        if (cat) disp = cat->title;

        json item{
          {"id", m.name},
          {"object", "model"},
          {"type", "model"},
          {"display_name", disp},
          {"created", long(started_)},
          {"created_at", started_at_},
          {"owned_by", "nace"},
          {"bytes", m.bytes},
          {"active", false},
          {"downloaded", true}
        };
        if (!m.variant.empty()) item["variant"] = m.variant;
        data.push_back(item);
      }
    } catch (...) {}

    // 3. Catalog models
    try {
      for (const auto& c : hub::catalog()) {
        std::string short_n = hub::short_name(c);
        if (c.variant()) {
          std::string var_full = hub::full_name(c, *c.variant());
          if (!seen.count(var_full) && !seen.count(c.name) && !seen.count(short_n)) {
            seen.insert(var_full);
            seen.insert(c.name);
            seen.insert(short_n);
            data.push_back(json{
              {"id", var_full},
              {"object", "model"},
              {"type", "model"},
              {"display_name", c.title + " (" + c.variant()->name + ")"},
              {"short_name", short_n},
              {"about", c.about},
              {"created", long(started_)},
              {"created_at", started_at_},
              {"owned_by", "nace"},
              {"active", false},
              {"downloaded", false}
            });
          }
        }
      }
    } catch (...) {}

    std::string first_id = data.empty() ? "" : data[0]["id"].get<std::string>();
    std::string last_id = data.empty() ? "" : data.back()["id"].get<std::string>();
    return json{
      {"object", "list"},
      {"data", data},
      {"has_more", false},
      {"first_id", first_id},
      {"last_id", last_id}
    };
  }

  nlohmann::ordered_json get_model_json(const std::string& id) const {
    const auto list = list_models_json();
    if (list.contains("data") && list["data"].is_array()) {
      for (const auto& m : list["data"]) {
        if (m.value("id", "") == id || m.value("short_name", "") == id) {
          return m;
        }
      }
    }
    return nlohmann::ordered_json::object();
  }

 private:
  friend class RequestGuard;

  static std::shared_ptr<Engine> load_engine(const EngineOptions& opt) {
    // Like the SDK, keep one allocation scope around the entire engine lifetime, including failed construction.
    // Member order destroys the engine (joining its workers) before draining and releasing its Metal buffers.
    struct LoadedEngine {
      Gpu::AllocationScope allocations;
      Engine engine;
      explicit LoadedEngine(const EngineOptions& opt) : engine(opt) {}
    };
    auto loaded = std::make_shared<LoadedEngine>(opt);
    return {loaded, &loaded->engine};
  }

  // rw_mu_ protects selection and registration as one operation. Leave the guard unregistered until all allocation
  // succeeds: shared_ptr's control-block allocation may delete it here, and its destructor must not relock rw_mu_.
  std::shared_ptr<RequestGuard> acquire_unlocked() {
    if (!engine_) return nullptr;
    auto guard = std::shared_ptr<RequestGuard>(new RequestGuard(engine_));
    guard->mgr_ = shared_from_this();
    {
      std::lock_guard<std::mutex> lk(in_flight_mu_);
      in_flight_.fetch_add(1);
    }
    touch();
    return guard;
  }

  // Called after the switching gate and exclusive selection lock have excluded all new admissions.
  void wait_for_requests(std::unique_lock<std::shared_mutex>& selection) {
    selection.unlock();  // admitted requests still need selection for reply bookkeeping
    {
      std::unique_lock<std::mutex> lk(in_flight_mu_);
      in_flight_cv_.wait(lk, [&] { return in_flight_.load() == 0; });
    }
    selection.lock();
  }

  void remove_in_flight() {
    touch();  // (idle time counts from the end of the last request)
    std::lock_guard<std::mutex> lk(in_flight_mu_);
    if (in_flight_.fetch_sub(1) == 1) in_flight_cv_.notify_all();
  }

  // ---- --idle-unload ----
  static int64_t mono_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void touch() { last_use_ms_.store(mono_ms()); }
  // what the accessors answer while the engine is released
  void remember_info() {
    if (!engine_) return;
    info_ = {engine_->model_id(), engine_->context_length(), engine_->context_fit(), engine_->slots()};
  }

  void stop_idle_watch() {
    {
      std::lock_guard<std::mutex> lk(idle_mu_);
      idle_stop_ = true;
    }
    idle_cv_.notify_all();
    if (idle_th_.joinable()) idle_th_.join();
  }
  void idle_watch() {
    std::unique_lock<std::mutex> lk(idle_mu_);
    while (!idle_stop_) {
      idle_cv_.wait_for(lk, std::chrono::seconds(kIdleCheckS));
      if (idle_stop_) break;
      lk.unlock();
      try_unload();
      lk.lock();
    }
  }
  // Release the engine if the server has been idle for idle_s_ (and, by default, macOS is short of memory).
  // The checks are repeated under the exclusive lock: a request registers under the shared one, so none can slip in.
  void try_unload() {
    if (unloaded_.load() || switching_.load() || in_flight_.load() != 0) return;
    if (mono_ms() - last_use_ms_.load() < int64_t(idle_s_) * 1000) return;
    if (only_pressed_ && !Gpu::memory_pressed()) return;
    if (veto_ && veto_()) return;
    if (switching_.exchange(true)) return;
    struct Clear {
      std::atomic<bool>& f;
      ~Clear() { f = false; }
    } clear{switching_};
    std::unique_lock<std::shared_mutex> lk(rw_mu_);
    if (!engine_ || in_flight_.load() != 0 || mono_ms() - last_use_ms_.load() < int64_t(idle_s_) * 1000) return;
    const size_t before = Gpu::get().allocated_bytes();
    logi("idle for %d s%s: releasing %s from the GPU; the next request loads it again", idle_s_,
         only_pressed_ ? " with memory pressure" : "", ref_.name.c_str());
    stop_prefill();
    engine_->save_kflash();
    engine_->save_disk_cache();
    engine_.reset();
    unloaded_ = true;
    const size_t after = Gpu::get().allocated_bytes();
    logi("released %s: GPU memory %.1f -> %.1f GB", ref_.name.c_str(), before / 1e9, after / 1e9);
    if (tui_) ui::note(ref_.name + " released while idle; the next request loads it again");
  }
  // The request that finds the engine released loads it again; requests arriving meanwhile wait on the lock.
  bool load_back(Error* error_out) {
    std::unique_lock<std::shared_mutex> lk(rw_mu_);
    return load_back_locked(error_out);
  }
  // (rw_mu_ held exclusively)
  bool load_back_locked(Error* error_out) {
    if (!unloaded_.load()) return true;  // (another request or a switch did)
    const int64_t t0 = mono_ms();
    if (tui_) ui::loading_begin(ref_.name);
    try {
      engine_ = load_engine(opt_);
    } catch (const std::exception& e) {
      if (tui_) ui::loading_end(ref_.name);
      logi("loading %s again failed: %s", ref_.name.c_str(), e.what());
      if (error_out) *error_out = {Error::Code::Busy, std::string("loading the model again failed: ") + e.what()};
      return false;
    }
    if (tui_) ui::loading_end(ref_.name);
    unloaded_ = false;
    remember_info();
    touch();
    logi("loaded %s again in %.1f s for a request", ref_.name.c_str(), double(mono_ms() - t0) / 1e3);
    return true;
  }

  bool is_current_model_unlocked(const std::string& model_name) const {
    if (model_name.empty() || model_name == "default") return true;
    if (!engine_ && !unloaded_.load()) return false;
    if (model_name == info_.model_id) return true;
    if (model_name == ref_.name || model_name == ref_.id || model_name == ref_.dir) return true;
    if (ref_.model) {
      if (model_name == ref_.model->name) return true;
      for (const auto& a : ref_.model->aliases) {
        if (model_name == a) return true;
      }
      if (ref_.variant) {
        if (model_name == hub::full_name(*ref_.model, *ref_.variant)) return true;
        if (model_name == hub::short_name(*ref_.model, *ref_.variant)) return true;
      }
    }
    try {
      hub::ModelRef r = hub::resolve(expand_home(model_name));
      if (r.kind == hub::ModelRef::kCatalog && !r.variant_given && ref_.model && r.model == ref_.model) {
        return true;
      }
      if (!r.dir.empty() && r.dir == ref_.dir) return true;
    } catch (const std::exception& e) {
      logi("is_current_model_unlocked resolve(%s) exception: %s", model_name.c_str(), e.what());
      return false;
    } catch (...) {
      logi("is_current_model_unlocked resolve(%s) unknown exception", model_name.c_str());
      return false;
    }
    return false;
  }

  void stop_prefill() {
    std::lock_guard<std::mutex> lk(prefill_mu_);
    model_epoch_++;
    for (auto& p : prefill_ths_)
      if (p.th.joinable()) p.th.join();
    prefill_ths_.clear();
  }

  mutable std::shared_mutex rw_mu_;
  std::shared_ptr<Engine> engine_;
  hub::ModelRef ref_;
  EngineOptions opt_;
  EngineOptions base_opt_;
  int context_req_ = 0;
  size_t budget_ = 0;
  bool tui_ = false;
  time_t started_ = 0;
  std::string started_at_;

  bool allow_switch_ = false;
  std::atomic<bool> switching_{false};

  // --idle-unload
  struct Info {
    std::string model_id;
    int context = 0, fit = 0, slots = 0;
  };
  static constexpr int kIdleCheckS = 15;
  Info info_;  // the engine's model id, context and slots (under rw_mu_), kept while it is released
  std::atomic<bool> unloaded_{false};
  std::atomic<int64_t> last_use_ms_{0};
  int idle_s_ = 0;
  bool only_pressed_ = true;
  std::function<bool()> veto_;
  std::thread idle_th_;
  std::mutex idle_mu_;
  std::condition_variable idle_cv_;
  bool idle_stop_ = false;
  std::atomic<int> in_flight_{0};
  std::mutex in_flight_mu_;
  std::condition_variable in_flight_cv_;

  std::mutex prefill_mu_;
  struct PrefillThread {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
    std::shared_ptr<std::atomic<bool>> stale;  // a newer reply of the same conversation: this one can only give up
    int slot = -1;
    std::vector<size_t> hashes;  // the request's messages after the system prompt (history_message_hashes)
  };
  static constexpr size_t kMaxPending = 4;
  std::vector<PrefillThread> prefill_ths_;  // history prefills (under prefill_mu_)
  std::atomic<uint64_t> model_epoch_{0};
};

inline RequestGuard::~RequestGuard() {
  eng_.reset();  // A switch may release the old engine as soon as the count reaches zero.
  if (mgr_) mgr_->remove_in_flight();
}

}  // namespace ni
