#pragma once
#include "engine/engine.h"
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
class RequestGuard {
 public:
  RequestGuard(std::shared_ptr<ModelManager> mgr, std::shared_ptr<Engine> eng);
  ~RequestGuard();

  Engine& engine() const { return *eng_; }
  Engine* operator->() const { return eng_.get(); }
  Engine& operator*() const { return *eng_; }
  std::shared_ptr<Engine> get() const { return eng_; }

 private:
  std::shared_ptr<ModelManager> mgr_;
  std::shared_ptr<Engine> eng_;
};

class ModelManager : public std::enable_shared_from_this<ModelManager> {
 public:
  ModelManager() = default;
  ~ModelManager() {
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
    engine_ = std::make_shared<Engine>(opt_);
    if (tui_) ui::loading_end(ref_.name);
  }

  // --allow-switch: a request naming another model (or POST /v1/models/switch) may swap the served model. Off by
  // default: clients send their own model names (Claude Code, Cursor), which the server ignores, as it always did.
  void set_allow_switch(bool on) { allow_switch_ = on; }
  bool allow_switch() const { return allow_switch_; }
  // True while a switch loads the new model: requests get 503 + Retry-After instead of queueing behind it
  bool switching() const { return switching_.load(); }

  // Acquire active engine for a request, switching on demand if another valid model is requested and --allow-switch.
  std::shared_ptr<RequestGuard> acquire_request(const std::string& requested_model, std::string* error_out = nullptr) {
    std::shared_ptr<Engine> eng;
    {
      std::shared_lock<std::shared_mutex> lk(rw_mu_);
      if (!allow_switch_ || is_current_model_unlocked(requested_model)) {
        eng = engine_;
      }
    }
    if (!eng) {
      eng = switch_to(requested_model, error_out);
      if (!eng) return nullptr;
    }
    return std::make_shared<RequestGuard>(shared_from_this(), eng);
  }

  // The active engine for a request that never switches models (POST /v1/draft): null when `requested_model` names
  // another one (with --allow-switch; without it the server serves its model whatever a request names)
  std::shared_ptr<RequestGuard> acquire_current(const std::string& requested_model) {
    std::shared_ptr<Engine> eng;
    {
      std::shared_lock<std::shared_mutex> lk(rw_mu_);
      if (!allow_switch_ || is_current_model_unlocked(requested_model)) eng = engine_;
    }
    if (!eng) return nullptr;
    return std::make_shared<RequestGuard>(shared_from_this(), eng);
  }

  // Explicitly switch the active model
  std::shared_ptr<Engine> switch_to(const std::string& target_model, std::string* error_out = nullptr) {
    if (!allow_switch_) {
      if (error_out) *error_out = "model switching is off (start the server with --allow-switch)";
      return nullptr;
    }
    if (target_model.empty()) {
      if (error_out) *error_out = "model name cannot be empty";
      return nullptr;
    }
    if (switching_.exchange(true)) {
      if (error_out) *error_out = "another model switch is in progress";
      return nullptr;
    }
    struct Clear {
      std::atomic<bool>& f;
      ~Clear() { f = false; }
    } clear{switching_};

    std::unique_lock<std::shared_mutex> lk(rw_mu_);
    if (is_current_model_unlocked(target_model)) {
      return engine_;
    }

    hub::ModelRef new_ref;
    try {
      const hub::ModelRef named = hub::resolve(expand_home(target_model));
      new_ref = named;
      if (named.kind == hub::ModelRef::kCatalog && !named.variant_given) {
        bool fits = false;
        new_ref = pick_variant(*named.model, base_opt_, budget_, &fits);
      }
    } catch (const std::exception& e) {
      if (error_out) *error_out = e.what();
      return nullptr;
    }

    if (new_ref.kind == hub::ModelRef::kLocal) {
      struct stat st;
      if (stat((new_ref.dir + "/config.json").c_str(), &st) != 0) {
        if (error_out) *error_out = new_ref.dir + " has no config.json";
        return nullptr;
      }
    }

    EngineOptions new_opt = base_opt_;
    try {
      hub::PullOptions po;
      new_opt.model_dir = hub::ensure(new_ref, po);
    } catch (const std::exception& e) {
      if (error_out) *error_out = e.what();
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

    {
      LoopKv kv = LoopKv::last_loop;
      if (!new_opt.loop_kv.empty()) parse_loop_kv(new_opt.loop_kv, &kv);
      Dory::set_serving(new_opt.loops, kv);
    }

    // Wait for in-flight requests on old engine to finish
    in_flight_cv_.wait(lk, [&] { return in_flight_.load() == 0; });

    // Stop background prefill on old engine
    stop_prefill();

    logi("switching model from %s to %s", ref_.name.c_str(), new_ref.name.c_str());
    if (tui_) ui::note("Switching model to " + term::bold(new_ref.name));
    else printf("Switching model to %s...\n", new_ref.name.c_str());
    fflush(stdout);

    // Release old engine to free GPU memory
    engine_.reset();

    if (tui_) ui::loading_begin(new_ref.name);
    try {
      engine_ = std::make_shared<Engine>(new_opt);
    } catch (const std::exception& e) {
      if (tui_) ui::loading_end(new_ref.name);
      if (error_out) *error_out = std::string("Engine initialization failed: ") + e.what();
      // the old model comes back, so a failed switch (memory, a bad checkpoint) leaves the server serving
      logi("switch to %s failed (%s); reloading %s", new_ref.name.c_str(), e.what(), ref_.name.c_str());
      if (tui_) ui::loading_begin(ref_.name);
      try {
        engine_ = std::make_shared<Engine>(opt_);
      } catch (const std::exception& e2) {
        logi("reloading %s failed too: %s", ref_.name.c_str(), e2.what());
        if (error_out) *error_out += std::string("; reloading the previous model failed too: ") + e2.what();
      }
      if (tui_) ui::loading_end(ref_.name);
      return nullptr;
    }
    if (tui_) ui::loading_end(new_ref.name);

    ref_ = new_ref;
    opt_ = new_opt;
    logi("switched model to %s (id: %s, context: %d)", ref_.name.c_str(), opt_.model_id.c_str(), opt_.context);
    return engine_;
  }

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
    in_flight_cv_.wait(lk, [&] { return in_flight_.load() == 0; });
    stop_prefill();
    engine_.reset();
    if (before) before();
    if (tui_) ui::loading_begin(ref_.name);
    try {
      engine_ = std::make_shared<Engine>(opt);
    } catch (const std::exception& e) {
      if (error_out) *error_out = e.what();
      logi("loading %s again failed (%s); loading it as it was", ref_.name.c_str(), e.what());
      if (undo) undo();
      try {
        engine_ = std::make_shared<Engine>(opt_);
      } catch (const std::exception& e2) {
        logi("loading %s as it was failed too: %s", ref_.name.c_str(), e2.what());
        if (error_out) *error_out += std::string("; loading it as it was failed too: ") + e2.what();
      }
      if (tui_) ui::loading_end(ref_.name);
      return false;
    }
    if (tui_) ui::loading_end(ref_.name);
    opt_ = opt;
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
    {  // the reply is a conversation POST /v1/draft can read ahead in, at once (a stream's was, before its end)
      std::shared_lock<std::shared_mutex> rlk(rw_mu_);
      if (engine_) engine_->remember_reply(r, out);
    }
    std::lock_guard<std::mutex> lk(prefill_mu_);
    if (prefill_th_.joinable()) {
      prefill_th_.join();
    }
    std::shared_ptr<Engine> eng;
    {
      std::shared_lock<std::shared_mutex> rlk(rw_mu_);
      eng = engine_;
    }
    if (!eng) return;
    uint64_t epoch = model_epoch_.load();
    prefill_th_ = std::thread([this, eng, r, out, epoch] {
      if (epoch != model_epoch_.load()) return;
      eng->prefill_history(r, out);
    });
  }

  std::string current_model_id() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return engine_ ? engine_->model_id() : "";
  }
  int current_context_length() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return engine_ ? engine_->context_length() : 0;
  }
  int current_context_fit() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return engine_ ? engine_->context_fit() : 0;
  }
  int current_slots() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return engine_ ? engine_->slots() : 0;
  }
  std::string current_name() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return ref_.name;
  }
  hub::ModelRef current_ref() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return ref_;
  }
  std::shared_ptr<Engine> current_engine() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    return engine_;
  }

  nlohmann::ordered_json list_models_json() const {
    std::shared_lock<std::shared_mutex> lk(rw_mu_);
    using json = nlohmann::ordered_json;
    json data = json::array();
    std::set<std::string> seen;

    // 1. Current active model is first
    if (engine_) {
      const std::string cur_id = engine_->model_id();
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
        {"context_length", engine_->context_length()},
        {"context_fit", engine_->context_fit()},
        {"slots", engine_->slots()},
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
        if (seen.count(m.name) || (engine_ && m.name == engine_->model_id())) continue;
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

  void add_in_flight() { in_flight_.fetch_add(1); }
  void remove_in_flight() {
    if (in_flight_.fetch_sub(1) == 1) {
      in_flight_cv_.notify_all();
    }
  }

 private:
  bool is_current_model_unlocked(const std::string& model_name) const {
    if (model_name.empty() || model_name == "default") return true;
    if (!engine_) return false;
    if (model_name == engine_->model_id()) return true;
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
    if (prefill_th_.joinable()) {
      prefill_th_.join();
    }
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
  std::atomic<int> in_flight_{0};
  std::condition_variable_any in_flight_cv_;

  std::mutex prefill_mu_;
  std::thread prefill_th_;
  std::atomic<uint64_t> model_epoch_{0};
};

inline RequestGuard::RequestGuard(std::shared_ptr<ModelManager> mgr, std::shared_ptr<Engine> eng)
    : mgr_(std::move(mgr)), eng_(std::move(eng)) {
  if (mgr_) mgr_->add_in_flight();
}

inline RequestGuard::~RequestGuard() {
  if (mgr_) mgr_->remove_in_flight();
}

}  // namespace ni
