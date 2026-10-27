// Engine in Splash mode (EngineOptions::splash_model): the model runs in a Splash child process
// (src/splash/runtime.h) and this engine is its codec. It owns what turns a chat request into prompt tokens and the
// tokens back into reasoning, content and tool calls: the chat template, the tokenizer, the stop ids, OutputParser; both
// come from the Splash model folder, so the tokens sent are the ones plain Splash would send. Nothing here touches Metal
// or loads weights.
#include "engine/engine.h"
#include "engine/output_parser.h"
#include "model/expert_store.h"
#include "splash/plan.h"
#include "splash/runtime.h"
#include "util/json_file.h"
#include "util/log.h"
#include "vision/image.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>

namespace ni {

namespace {

// What a request is told when the engine child has exited: one sentence on every route, without the runtime's own
// "the splash engine is not running" prefix a second time.
std::string engine_down_message(std::string why) {
  for (const std::string lead : {"the splash engine is not running", ": "})
    if (why.rfind(lead, 0) == 0) why.erase(0, lead.size());
  return "the Splash engine has exited" + (why.empty() ? std::string() : " (" + why + ")") +
         "; restart kapitsa serve";
}

bool is_file(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// What the reader thread hands the worker: Sink callbacks only queue, the worker parses (as the native path does).
struct Relay : splash::Sink {
  std::mutex m;
  std::condition_variable cv;
  std::deque<uint32_t> tokens;
  bool started = false, done = false, errored = false;
  splash::Start start;
  splash::Done fin;
  splash::Error err;

  void on_start(const splash::Start& s) override {
    std::lock_guard<std::mutex> l(m);
    start = s;
    started = true;
    cv.notify_all();
  }
  void on_tokens(const splash::Tokens& t) override {
    std::lock_guard<std::mutex> l(m);
    tokens.insert(tokens.end(), t.tokens.begin(), t.tokens.end());
    cv.notify_all();
  }
  void on_done(const splash::Done& d) override {
    std::lock_guard<std::mutex> l(m);
    fin = d;
    done = true;
    cv.notify_all();
  }
  void on_error(const splash::Error& e) override {
    std::lock_guard<std::mutex> l(m);
    err = e;
    errored = true;
    cv.notify_all();
  }
};

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

ChatResult refuse(const std::string& why, bool invalid = true) {
  ChatResult res;
  res.finish_reason = "error";
  res.invalid_request = invalid;
  res.error = why;
  return res;
}

}  // namespace

void Engine::init_splash() {
  const std::string& dir = opts_.splash_model;
  if (opts_.splash_binary.empty()) throw std::runtime_error("splash: no Splash binary given (--splash-binary)");
  if (!is_file(dir + "/model.json") && !is_file(dir + "/config.json"))
    throw std::runtime_error("splash: " + dir + " is not a Splash model folder (no model.json)");
  // the tokenizer and template Splash itself uses: in its resolved folder's tokenizer/ (else beside model.json)
  const std::string tdir = is_file(dir + "/tokenizer/tokenizer.json") ? dir + "/tokenizer" : dir;
  if (!is_file(tdir + "/tokenizer.json")) throw std::runtime_error("splash: no tokenizer.json in " + tdir);
  if (opts_.model_id.empty()) {
    const size_t at = dir.find_last_of('/');
    opts_.model_id = at == std::string::npos || at + 1 == dir.size() ? dir : dir.substr(at + 1);
  }
  tok_.load(tdir + "/tokenizer.json");
  const std::string src = chat_template_source(tdir);
  if (src.empty()) throw std::runtime_error("splash: no chat template in " + tdir);
  tmpl_ = std::make_unique<minja::chat_template>(src, "", "<|im_end|>");
  think_open_ = tok_.find("<think>");
  think_close_ = tok_.find("</think>");
  tc_open_ = tok_.find("<tool_call>");
  tc_close_ = tok_.find("</tool_call>");
  // a reply ends at the tokenizer's eos token and at <|endoftext|>: the two Splash's own constraints name
  {
    auto tc = nlohmann::json::parse(read_file(tdir + "/tokenizer_config.json"), nullptr, false);
    std::string end = "<|im_end|>";
    if (!tc.is_discarded() && tc.contains("eos_token")) {
      const auto& e = tc["eos_token"];
      if (e.is_string()) end = e.get<std::string>();
      else if (e.is_object()) end = e.value("content", end);
    }
    for (const char* name : {end.c_str(), "<|endoftext|>"}) {
      const int id = tok_.find(name);
      if (id >= 0 && std::find(stop_ids_.begin(), stop_ids_.end(), id) == stop_ids_.end()) stop_ids_.push_back(id);
    }
  }
  // What a request that names no sampling gets is Splash's own default (frontend.py: temperature 1.0, top_p 0.95, top_k 20), not
  // Kapitsa's 0.6 for Qwen: plain Splash and this backend then answer the same request alike.
  defaults_.temperature = 1.0f;
  defaults_.top_p = 0.95f;
  defaults_.top_k = 20;
  splash::RuntimeOptions ro;
  ro.argv = {opts_.splash_binary, "serve-native", dir, opts_.context > 0 ? std::to_string(opts_.context) : "auto", "auto"};
  ro.argv.insert(ro.argv.end(), opts_.splash_args.begin(), opts_.splash_args.end());
  {
    std::string note;
    const char* home = getenv("HOME");
    // the real path of the model folder: a symlink or a trailing slash must not make the folder name another model's (or empty)
    char resolved[PATH_MAX];
    std::string trimmed = realpath(dir.c_str(), resolved) ? std::string(resolved) : dir;
    while (trimmed.size() > 1 && trimmed.back() == '/') trimmed.pop_back();
    const std::string folder = trimmed.substr(trimmed.find_last_of('/') == std::string::npos ? 0 : trimmed.find_last_of('/') + 1);
    // <name>-<hash of the real path>: two folders with the same name elsewhere are other models and keep apart
    char tag[17];
    snprintf(tag, sizeof tag, "%016llx", (unsigned long long)splash::client_salt(trimmed).lo);
    const std::string cache_dir = home && *home && !folder.empty() ? std::string(home) + "/Library/Caches/Kapitsa/splash-prefix-cache/" + folder + "-" + tag : "";
    const auto extra = splash::default_engine_args(opts_.splash_args, q20::available_memory(), opts_.splash_persistent_cache,
                                                   cache_dir, &note);
    ro.argv.insert(ro.argv.end(), extra.begin(), extra.end());
    if (!note.empty()) logi("splash: engine arguments added: %s", note.c_str());
  }
  ro.log = [](const std::string& s) { logi("splash: %s", s.c_str()); };
  splash_ = std::make_unique<splash::Runtime>(std::move(ro));
  splash_->start();  // READY: the child has loaded and calibrated the model
  splash_slots_ = std::max<int>(1, int(splash_->ready().max_concurrent_requests));
  opts_.context = int(splash_->ready().max_context_tokens);
  context_fit_ = opts_.context;
  logi("splash: engine ready (pid %d): context %d, %d concurrent request(s), tokenizer from %s", int(splash_->pid()),
       opts_.context, splash_slots_, tdir.c_str());
}

ChatResult Engine::chat_splash(const ChatRequest& r, const std::function<bool(const ChatDelta&)>& on_delta,
                               const std::function<void(const ChatResult&)>& on_prompt,
                               const std::function<bool()>& cancelled,
                               const std::function<bool(const ChatResult&)>& on_accepted) {
  if (!splash_) throw std::logic_error("Engine::chat_splash: this engine is not in Splash mode");
  ++requests_;
  // No reasoning is put back into assistant turns the client sent without it (the native path does, to extend the KV it
  // kept): Splash's engine caches a request's prompt blocks, not its output, so a longer prompt only costs prefill. Measured
  // on a 4K two-turn chat it was slower than sending the history as is, as plain Splash does; the runs are in docs/splash-backend.md.
  // what Splash's native protocol carries here is text with sampling: the rest is refused before anything is queued
  if (r.grammar) return refuse("structured output is not supported by the Splash engine");
  if (!vision::image_sources(r.messages).empty()) return refuse("image input is not supported by the Splash engine");
  if (r.loops > 0) return refuse("loop counts are not supported by the Splash engine");
  // Splash's own ranges (frontend.py SAMPLING_NUMBERS): a value outside them is a 400 naming the field, not a clamp
  auto bad = [&](const char* name, const char* want) { return refuse(std::string(name) + " must be " + want); };
  if (!(r.temperature >= 0 && r.temperature <= 2)) return bad("temperature", "a number in [0, 2]");
  if (!(r.top_p > 0 && r.top_p <= 1)) return bad("top_p", "a number in (0, 1]");
  if (!(r.presence_penalty >= -2 && r.presence_penalty <= 2)) return bad("presence_penalty", "a number in [-2, 2]");
  if (!(r.frequency_penalty >= -2 && r.frequency_penalty <= 2)) return bad("frequency_penalty", "a number in [-2, 2]");
  if (!(r.repetition_penalty > 0 && std::isfinite(r.repetition_penalty))) return bad("repetition_penalty", "a positive number");
  if (!(r.min_p >= 0 && r.min_p <= 1)) return bad("min_p", "a number in [0, 1]");
  if (!r.cache_salt.is_null() && !(r.cache_salt.is_string() && !r.cache_salt.get<std::string>().empty()))
    return bad("cache_salt", "a non-empty string");
  if (r.top_k < -1) return bad("top_k", "0 or -1 (disabled) or a positive integer");

  ChatResult res;
  std::string text, base;
  try {
    text = render(r, true);
    base = render(r, false);
  } catch (const std::exception& ex) {
    res.finish_reason = "error";
    res.error = std::string("chat template: ") + ex.what();
    return res;
  }
  int boundary = -1;
  const bool prefix_ok = text.compare(0, base.size(), base) == 0;
  const std::vector<int> prompt = tok_.encode(text, prefix_ok ? base.size() : std::string::npos, &boundary);
  const int n = int(prompt.size());
  const int ctx = int(splash_->ready().max_context_tokens);
  res.prompt_tokens = n;
  if (n + 1 >= ctx) {  // as the native path: the prompt must leave room for a token
    res.finish_reason = "error";
    res.prompt_limit = std::max(0, ctx - 2);
    res.error = "the prompt is " + std::to_string(n) + " tokens; the context holds " + std::to_string(ctx);
    return res;
  }
  // reasoning is on when the generation prompt opens the think block (as Engine::chat decides it)
  static const std::string kOpen = "<think>\n";
  const bool think = text.size() >= kOpen.size() && text.compare(text.size() - kOpen.size(), kOpen.size(), kOpen) == 0;

  const int want = std::max(1, r.max_tokens);
  const int room = ctx - n;  // positions left: Splash would clamp to it, and a length ending there is the context's
  const int max_new = std::min(want, room);
  splash::Request q;
  q.max_output_tokens = uint32_t(max_new);
  q.prompt.assign(prompt.begin(), prompt.end());
  // the sampler: the frontend's rules (a nonzero temperature below 0.01 is raised to it; top_k <= 0 is every token)
  q.sampling.temperature = r.temperature <= 0 ? 0.0f : std::max(r.temperature, 0.01f);
  q.sampling.top_p = r.top_p;
  q.sampling.top_k = r.top_k > 0 ? uint32_t(r.top_k) : 0;
  q.sampling.presence_penalty = r.presence_penalty;
  q.sampling.frequency_penalty = r.frequency_penalty;
  q.sampling.repetition_penalty = r.repetition_penalty;
  q.sampling.min_p = r.min_p;
  if (r.ignore_eos) q.flags |= splash::kIgnoreEndOfSequence;
  if (r.cache_salt.is_string()) {  // the client's own salt wins: it decides who shares cached prompts, as with plain Splash
    const auto salt = splash::client_salt(r.cache_salt.get<std::string>());
    q.cache_salt_lo = salt.lo;
    q.cache_salt_hi = salt.hi;
  } else if (opts_.splash_conversation_salt) {
    const auto salt = splash::conversation_salt(r.messages);
    q.cache_salt_lo = salt.lo;
    q.cache_salt_hi = salt.hi;
  }
  // Splash's rule (server/frontend.py:1153-1162): absent or -1 is random, any other integer is its 64 bits, 0 included
  q.seed = r.splash_seed ? *r.splash_seed : (uint64_t(std::random_device{}()) << 32 | std::random_device{}());
  if (boundary > 0 && boundary < n) q.generation_prompt_tokens = uint32_t(n - boundary);
  if (think && r.thinking_budget > 0 && r.thinking_budget < max_new && think_close_ >= 0) {
    splash::ThinkingBudget tb;
    tb.tokens = uint32_t(r.thinking_budget);
    tb.end = {uint32_t(think_close_)};
    tb.also_ends = uint32_t(!r.tools.empty() && tc_open_ >= 0 ? tc_open_ : think_close_);
    q.thinking = tb;
  }

  auto relay = std::make_shared<Relay>();
  uint64_t id = 0;
  const double t0 = now_ms();
  try {
    id = splash_->submit(std::move(q), relay, std::numeric_limits<double>::infinity());
  } catch (const splash::PendingLimitExceeded& ex) {
    res.finish_reason = "error";
    res.busy = true;  // a 503 with Retry-After, before a stream's first byte (on_accepted comes after the submit for that)
    res.error = std::string("the Splash engine is at its concurrent-request limit: ") + ex.what();
    return res;
  } catch (const splash::EngineUnhealthy& ex) {
    res.finish_reason = "error";
    res.engine_down = true;  // a 503 on every route; nothing relaunches the child, so no Retry-After
    res.error = engine_down_message(ex.what());
    return res;
  } catch (const splash::ProtocolError& ex) {
    return refuse(std::string("the Splash engine refused the request: ") + ex.what());
  }

  // after the submit, so a full pool answers before a stream has started; a client that is gone by now ends the request
  if (on_accepted && !on_accepted(res)) {
    splash_->cancel(id);
    res.finish_reason = "cancelled";
    return res;
  }

  OutputParser parser(tok_, think_open_, think_close_, tc_open_, tc_close_, think, !r.tools.empty(), r.tools, true, false,
                      text_tags_, CallFormat::QwenXml);
  std::string content_so_far, finish;
  int gen = 0;
  bool prompted = false, cancel_sent = false;
  // one token: the native handle's rules (eos, parser, stop strings); false once the reply is over
  auto handle = [&](int t) -> bool {
    ++gen;
    if (!r.ignore_eos && std::find(stop_ids_.begin(), stop_ids_.end(), t) != stop_ids_.end()) {
      finish = "stop";
      return false;
    }
    if (parser.in_think()) ++res.reasoning_tokens;
    ChatDelta d = parser.push(int(t));
    if (!d.content.empty()) {
      content_so_far += d.content;
      for (auto& st : r.stop) {
        const size_t at = st.empty() ? std::string::npos : content_so_far.find(st);
        if (at != std::string::npos) {
          const size_t over = content_so_far.size() - at;
          d.content.resize(d.content.size() > over ? d.content.size() - over : 0);
          finish = "stop";
          res.stop_sequence = st;
          break;
        }
      }
    }
    res.reasoning += d.reasoning;
    res.content += d.content;
    for (auto& tc : d.tool_calls) res.tool_calls.push_back(tc);
    if (!d.reasoning.empty() || !d.content.empty() || !d.tool_calls.empty())
      if (!on_delta(d)) finish = "cancelled";
    if (finish.empty() && gen >= want) finish = "length";
    return finish.empty();
  };

  double cancel_ms = 0;
  bool ended = false;  // the engine's own end (on_done / on_error) seen
  splash::Done fin;
  splash::Error err;
  bool errored = false;
  for (;;) {
    std::deque<uint32_t> batch;
    {
      std::unique_lock<std::mutex> l(relay->m);
      relay->cv.wait_for(l, std::chrono::milliseconds(20),
                         [&] { return !relay->tokens.empty() || relay->done || relay->errored || (relay->started && !prompted); });
      if (relay->started && !prompted) {
        prompted = true;
        res.cached_tokens = int(relay->start.matched_prompt_tokens);
        l.unlock();
        if (on_prompt) on_prompt(res);
        l.lock();
      }
      batch.swap(relay->tokens);
      if (relay->done) { ended = true; fin = relay->fin; }
      if (relay->errored) { ended = true; errored = true; err = relay->err; }
    }
    for (uint32_t t : batch) {
      if (!finish.empty()) break;  // tokens after the reply's end (the cancel is in flight) are dropped
      if (!handle(int(t))) break;
    }
    if (finish.empty() && cancelled && cancelled()) finish = "cancelled";
    // a cancelled reply waits for the engine's end, but not forever: a child that ignores the cancel is the runtime's
    // to supervise, and this request goes on without it (its late frames fall into a relay nobody reads)
    if (cancel_sent && !ended && now_ms() - cancel_ms > 30000) {
      logi("splash: the engine did not end request %llu 30 s after its cancel; the reply goes on without it", (unsigned long long)id);
      break;
    }
    // the reply is over here (stop string, max_tokens, client gone): stop the engine's work, then drain it to its end
    if (!finish.empty() && !ended && !cancel_sent) {
      cancel_sent = true;
      cancel_ms = now_ms();
      splash_->cancel(id);
    }
    if (ended && batch.empty()) {
      std::lock_guard<std::mutex> l(relay->m);
      if (relay->tokens.empty()) break;
    }
  }
  if (errored) {
    if (finish.empty() || finish == "length") {
      finish = "error";
      res.error = err.code.empty() ? err.message : err.code + ": " + err.message;
      if (err.failure == splash::FailureClass::kEngineUnhealthy) {  // the child died under the request (Runtime::fail_all)
        res.engine_down = true;
        res.error = engine_down_message(res.error);
      }
    }
  } else if (finish.empty()) {
    // the engine ended it: eos it did not send as a token (stop), the length it was asked, or a bound of its own
    if (fin.reason == splash::FinishReason::kStop) finish = "stop";
    else if (fin.reason == splash::FinishReason::kLength) {
      finish = "length";
      res.context_full = fin.bounded || (room < want && gen >= room);
    } else finish = "cancelled";
  }
  ChatDelta tail = parser.finish(finish == "stop");
  res.reasoning += tail.reasoning;
  res.content += tail.content;
  for (auto& tc : tail.tool_calls) res.tool_calls.push_back(tc);
  if (!tail.reasoning.empty() || !tail.content.empty() || !tail.tool_calls.empty()) on_delta(tail);
  if (finish == "stop" && !res.tool_calls.empty()) finish = "tool_calls";
  res.finish_reason = finish;
  res.completion_tokens = gen;
  if (!errored) {
    res.prefill_ms = double(fin.prefill_micros) / 1000.0;
    res.prefill_exact = fin.prefill_micros > 0;  // the engine's own clock for this request's prefill
    res.decode_ms = double(fin.decode_micros) / 1000.0;
  } else {
    res.decode_ms = now_ms() - t0;
  }
  return res;
}

}  // namespace ni
