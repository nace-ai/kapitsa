#include "engine/engine.h"
#include "util/log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

namespace ni {

static std::string read_file(const std::string& p) {
  std::ifstream f(p);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static std::string random_id(const char* prefix) {
  static std::mt19937_64 rng(std::random_device{}());
  static const char* al = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::string s = prefix;
  for (int i = 0; i < 24; ++i) s += al[rng() % 62];
  return s;
}

Engine::Engine(const EngineOptions& o) : opts_(o) {
  if (opts_.model_id.empty()) {
    std::string d = opts_.model_dir;
    while (!d.empty() && d.back() == '/') d.pop_back();
    opts_.model_id = d.substr(d.find_last_of('/') + 1);
  }
  model_ = std::make_unique<Qwen35>(opts_.model_dir, opts_.prefill_chunk);
  tok_.load(opts_.model_dir + "/tokenizer.json");
  std::string src = read_file(opts_.model_dir + "/chat_template.jinja");
  if (src.empty()) {
    auto cfg = nlohmann::json::parse(read_file(opts_.model_dir + "/tokenizer_config.json"));
    if (cfg.contains("chat_template") && cfg["chat_template"].is_string()) src = cfg["chat_template"];
  }
  if (src.empty()) fatal("no chat template in %s", opts_.model_dir.c_str());
  tmpl_ = std::make_unique<minja::chat_template>(src, "", "<|im_end|>");
  think_open_ = tok_.find("<think>");
  think_close_ = tok_.find("</think>");
  tc_open_ = tok_.find("<tool_call>");
  tc_close_ = tok_.find("</tool_call>");
  if (opts_.spec_depth < 0) opts_.spec_depth = model_->default_spec_depth();
  spec_ = opts_.spec_depth > 0 && model_->has_mtp();
  if (opts_.spec_depth > Qwen35::kMaxVerify - 1) opts_.spec_depth = Qwen35::kMaxVerify - 1;

  auto add_slot = [&](int cap) {
    Slot s;
    s.seq = model_->new_seq(cap);
    s.ckpts.push_back(model_->new_checkpoint());
    s.ckpts.push_back(model_->new_checkpoint());
    slots_.push_back(std::move(s));
  };
  for (int i = 0; i < std::max(1, opts_.slots); ++i) add_slot(opts_.context);
  logi("engine ready: %s, context %d, %d slots, %s", opts_.model_id.c_str(), opts_.context, std::max(1, opts_.slots),
       spec_ ? ("MTP speculative decoding depth " + std::to_string(opts_.spec_depth)).c_str() : "plain decoding");
}

Engine::~Engine() = default;

std::string Engine::render(const ChatRequest& r, bool add_generation_prompt) const {
  minja::chat_template_inputs in;
  in.messages = r.messages;
  in.tools = r.tools.empty() ? ojson() : r.tools;
  in.add_generation_prompt = add_generation_prompt;
  in.extra_context = r.template_kwargs;
  minja::chat_template_options opt;
  return tmpl_->apply(in, opt);
}

Engine::Slot& Engine::pick_slot(const std::vector<int>& prompt) {
  auto lcp = [&](const std::vector<int>& a) {
    size_t n = std::min(a.size(), prompt.size()), i = 0;
    while (i < n && a[i] == prompt[i]) ++i;
    return i;
  };
  // What a slot can actually reuse: its live state if the prompt extends it, else its newest
  // checkpoint inside the common prefix. A shared prefix without a usable state is worthless
  // and choosing that slot would only destroy its conversation.
  auto usable = [&](const Slot& s) -> int {
    size_t l = lcp(s.tokens);
    if (l == s.tokens.size() && l < prompt.size()) return int(l);
    int u = 0;
    for (auto& c : s.ckpts)
      if (c.pos > u && c.pos <= int(l) && c.pos < int(prompt.size())) u = c.pos;
    return u;
  };
  Slot* best = nullptr;
  int best_lcp = 0;
  for (auto& s : slots_) {
    if (int(prompt.size()) + 64 > s.seq->cap) continue;
    int u = usable(s);
    if (!best || u > best_lcp) {
      best = &s;
      best_lcp = u;
    }
  }
  if (!best) return slots_[0];
  if (best_lcp < 64) {
    // Nothing worth reusing: take the slot that is cheapest to rebuild later (fewest cached
    // tokens, so empty slots first), least recently used on ties.
    Slot* pick = nullptr;
    for (auto& s : slots_) {
      if (int(prompt.size()) + 64 > s.seq->cap) continue;
      if (!pick || s.tokens.size() < pick->tokens.size() ||
          (s.tokens.size() == pick->tokens.size() && s.last_used < pick->last_used))
        pick = &s;
    }
    if (pick) return *pick;
  }
  return *best;
}

namespace {

// Streaming parser for Qwen3.5+ output: <think>...</think> reasoning, content, <tool_call> XML.
class OutputParser {
 public:
  OutputParser(const Tokenizer& t, int think_open, int think_close, int tc_open, int tc_close, bool start_in_think,
               bool parse_tools, const ojson& tools)
      : dec_(t), think_open_(think_open), think_close_(think_close), tc_open_(tc_open), tc_close_(tc_close),
        in_think_(start_in_think), parse_tools_(parse_tools), tools_(tools) {}

  // Returns the delta produced by this token.
  ChatDelta push(int id) {
    ChatDelta d;
    if (id == think_open_) {
      flush_text(d);
      in_think_ = true;
      return d;
    }
    if (id == think_close_) {
      flush_text(d);
      held_.clear();  // drop whitespace before </think>
      in_think_ = false;
      skip_leading_ws_ = true;
      return d;
    }
    if (parse_tools_ && id == tc_open_ && !in_think_) {
      flush_text(d);
      held_.clear();
      in_tool_ = true;
      tool_buf_.clear();
      return d;
    }
    if (parse_tools_ && id == tc_close_ && in_tool_) {
      tool_buf_ += dec_.flush();
      in_tool_ = false;
      ToolCall tc;
      if (parse_tool(tool_buf_, tc)) d.tool_calls.push_back(tc);
      else emit(d, "<tool_call>" + tool_buf_ + "</tool_call>");
      skip_leading_ws_ = true;
      return d;
    }
    std::string s = dec_.push(id);
    if (in_tool_) {
      tool_buf_ += s;
      return d;
    }
    emit(d, s);
    return d;
  }

  ChatDelta finish() {
    ChatDelta d;
    std::string s = dec_.flush();
    if (in_tool_) {
      // unterminated tool call: surface as text
      emit(d, "<tool_call>" + tool_buf_ + s);
      in_tool_ = false;
    } else {
      emit(d, s);
    }
    held_.clear();  // trailing whitespace
    return d;
  }

  bool in_think() const { return in_think_; }

 private:
  // Holds back trailing whitespace so it can be dropped before </think>, <tool_call> or the end.
  void emit(ChatDelta& d, const std::string& s) {
    if (s.empty()) return;
    std::string t = held_ + s;
    held_.clear();
    if (skip_leading_ws_) {
      size_t k = t.find_first_not_of(" \n\r\t");
      if (k == std::string::npos) return;
      t = t.substr(k);
      skip_leading_ws_ = false;
    }
    size_t e = t.find_last_not_of(" \n\r\t");
    if (e == std::string::npos) {
      held_ = t;
      return;
    }
    held_ = t.substr(e + 1);
    t.resize(e + 1);
    (in_think_ ? d.reasoning : d.content) += t;
  }
  void flush_text(ChatDelta& d) { emit(d, dec_.flush()); }

  static std::string trim_nl(const std::string& v) {
    size_t a = 0, b = v.size();
    if (a < b && v[a] == '\n') ++a;
    if (b > a && v[b - 1] == '\n') --b;
    return v.substr(a, b - a);
  }

  std::string param_type(const std::string& fn, const std::string& param) const {
    for (auto& t : tools_) {
      const ojson& f = t.contains("function") ? t["function"] : t;
      if (f.value("name", "") != fn) continue;
      if (f.contains("parameters") && f["parameters"].contains("properties") &&
          f["parameters"]["properties"].contains(param)) {
        const ojson& p = f["parameters"]["properties"][param];
        if (p.contains("type") && p["type"].is_string()) return p["type"];
        return "any";
      }
    }
    return "string";
  }

  bool parse_tool(const std::string& body, ToolCall& tc) const {
    size_t f = body.find("<function=");
    if (f == std::string::npos) return false;
    size_t fe = body.find('>', f);
    if (fe == std::string::npos) return false;
    tc.name = body.substr(f + 10, fe - f - 10);
    ojson args = ojson::object();
    size_t p = fe + 1;
    while (true) {
      size_t ps = body.find("<parameter=", p);
      if (ps == std::string::npos) break;
      size_t pe = body.find('>', ps);
      if (pe == std::string::npos) break;
      std::string pname = body.substr(ps + 11, pe - ps - 11);
      size_t ve = body.find("</parameter>", pe);
      if (ve == std::string::npos) ve = body.find("</function>", pe);
      if (ve == std::string::npos) ve = body.size();
      std::string val = trim_nl(body.substr(pe + 1, ve - pe - 1));
      std::string type = param_type(tc.name, pname);
      if (type == "string") {
        args[pname] = val;
      } else {
        try {
          args[pname] = ojson::parse(val);
        } catch (...) {
          args[pname] = val;
        }
      }
      p = ve + 1;
    }
    tc.arguments = args.dump();
    tc.id = random_id("call_");
    return true;
  }

  StreamDecoder dec_;
  int think_open_, think_close_, tc_open_, tc_close_;
  bool in_think_;
  bool parse_tools_;
  const ojson& tools_;
  bool in_tool_ = false, skip_leading_ws_ = false;
  std::string tool_buf_, held_;
};

} // namespace

ChatResult Engine::chat(const ChatRequest& r, const std::function<bool(const ChatDelta&)>& on_delta) {
  ++requests_;
  std::lock_guard<std::mutex> lk(mu_);
  // NI_ANE_ASYNC: switch the prefill FFN split on once its program is compiled (under the engine lock:
  // it runs a startup check through the model's buffers, which the idle history prefill also uses)
  model_->ane_poll();
  const double t_req = now_ms();
  ChatResult res;
  Gpu& g = Gpu::get();
  std::string text, base;
  try {
    text = render(r, true);
    base = render(r, false);
  } catch (const std::exception& ex) {
    res.finish_reason = "error";
    res.error = std::string("chat template: ") + ex.what();
    return res;
  }
  // Checkpoint at the end of the conversation so far (before the assistant header): the next
  // turn re-renders this turn's answer without its reasoning, so it diverges right after it.
  // The generation prompt starts with an added token (<|im_start|>), so the boundary falls out of
  // the one encode; otherwise the conversation is encoded separately and compared.
  int boundary = -1;
  const bool prefix_ok = text.compare(0, base.size(), base) == 0;
  std::vector<int> prompt = tok_.encode(text, prefix_ok ? base.size() : std::string::npos, &boundary);
  if (boundary < 0 && prefix_ok) {
    std::vector<int> bt = tok_.encode(base);
    if (bt.size() < prompt.size() && std::equal(bt.begin(), bt.end(), prompt.begin())) boundary = int(bt.size());
  }
  if (boundary >= int(prompt.size())) boundary = -1;
  const int n = int(prompt.size());
  res.prompt_tokens = n;
  Slot& slot = pick_slot(prompt);
  Seq& s = *slot.seq;
  slot.last_used = ++clock_;
  ++slot.gen;
  if (n + 1 >= s.cap) {
    res.finish_reason = "error";
    res.error = "prompt (" + std::to_string(n) + " tokens) exceeds the context window (" + std::to_string(s.cap) + ")";
    return res;
  }

  // ---- prefix reuse ----
  size_t common = 0;
  while (common < slot.tokens.size() && common < prompt.size() && slot.tokens[common] == prompt[common]) ++common;
  int start = 0;
  auto e = g.begin();
  if (common == slot.tokens.size() && int(common) < n) {
    start = int(common);  // live state continues
  } else {
    const Checkpoint* best = nullptr;
    for (auto& c : slot.ckpts)
      if (c.pos > 0 && c.pos <= int(common) && c.pos < n && (!best || c.pos > best->pos)) best = &c;
    if (best) {
      model_->restore(*e, s, *best);
      start = best->pos;
    } else {
      model_->reset(s);
    }
  }
  for (auto& c : slot.ckpts)
    if (c.pos > start) c.pos = -1;
  res.cached_tokens = start;
  std::copy(prompt.begin(), prompt.end(), s.tokens());

  // ---- prefill ----
  const double tok_ms = now_ms() - t_req;  // template + tokenizer + slot pick
  SamplingParams sp;
  sp.temperature = r.temperature;
  sp.top_p = r.top_p;
  sp.top_k = r.top_k;
  sp.seed = r.seed ? r.seed : uint32_t(std::random_device{}());
  double t0 = now_ms();
  const int C = model_->prefill_chunk();
  int p = start;
  int last_chunk_row = 0;
  auto oldest_ckpt = [&]() {
    Checkpoint* c = &slot.ckpts[0];
    for (auto& k : slot.ckpts)
      if (k.pos < c->pos) c = &k;  // overwrite the oldest / invalid one
    return c;
  };
  // NI_SHORTPF: a checkpoint inside a chunk is captured by the pass itself (one weight pass for
  // the whole new suffix); otherwise the chunk is split so the checkpoint lands exactly.
  const bool in_pass = model_->can_snapshot();
  // NI_MTP_DEFER (default on): the last chunk's MTP history rows go into their own command buffer behind
  // the trunk's, so the first token (sampled by the trunk pass) is out before they run; the first
  // speculative cycle is queued after them. 0 keeps them in the trunk's command buffer.
  static const bool mtp_defer = [] {
    const char* v = getenv("NI_MTP_DEFER");
    return !(v && !strcmp(v, "0"));
  }();
  std::unique_ptr<Encoder> mtp_e;
  while (p < n) {
    int T = std::min(C, n - p);
    Checkpoint* snap = nullptr;
    if (boundary > p && boundary < p + T) {
      // NI_SHORTPF captures the checkpoint inside the pass; a chunk the NI_ANE split would run is cut at
      // the boundary instead (the split's two-half wavefront does not capture mid-chunk state).
      if (in_pass && !model_->ane_splits(T)) snap = oldest_ckpt();
      else T = boundary - p;
    }
    // NI_ANE=1: a chunk that runs the GPU + Neural Engine FFN split saves the recurrent state and
    // is waited for, so that it can run again on the GPU alone if the split fails.
    const bool split = !snap && model_->ane_splits(T);
    if (split) {
      if (!ane_ck_) ane_ck_ = std::make_unique<Checkpoint>(model_->new_checkpoint());
      model_->save(*e, s, *ane_ck_, p);
    }
    model_->encode_prefill(*e, s, p, T, sp, 0, p + T == n, snap, snap ? boundary - p : 0);
    // MTP history: pairs (hidden_i, token_{i+1}) for prompt rows whose next token is known
    const bool defer = spec_ && mtp_defer && !split && p + T == n;
    if (spec_ && !defer) model_->encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
    if (defer) {
      mtp_e = g.begin();
      model_->encode_mtp_prefill(*mtp_e, s, p, std::min(p + T, n - 1) - p, 0);
    }
    if (split) {
      e->commit_and_wait();
      if (!model_->ane_finish_chunk()) {
        e = g.begin();
        model_->restore(*e, s, *ane_ck_);
        model_->encode_prefill(*e, s, p, T, sp, 0, p + T == n);
        if (spec_) model_->encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
        e->commit_and_wait();
      }
      e = g.begin();
    }
    last_chunk_row = n - 1 - p;
    p += T;
    if (p == boundary) model_->save(*e, s, *oldest_ckpt(), boundary);
    e->commit();
    if (mtp_e) mtp_e->commit();  // queued behind the trunk; waited for by the first spec cycle's buffer
    if (p < n) e = g.begin();
  }
  const double t_enc = now_ms() - t0;  // until the last chunk's commit
  const double h_commit = Encoder::host_ms();
  e->wait();
  const double h_woke = Encoder::host_ms();
  res.prefill_ms = now_ms() - t0;
  const double gpu_last = e->gpu_ms();
  // last chunk: commit -> GPU start, GPU end -> this thread awake (host clock)
  const double lat_start = e->gpu_start_host_ms() - h_commit, lat_wake = h_woke - e->gpu_end_host_ms();
  slot.tokens = prompt;

  // ---- decode ----
  const bool think = !text.empty() && text.size() >= 8 && text.compare(text.size() - 8, 8, "<think>\n") == 0;
  OutputParser parser(tok_, think_open_, think_close_, tc_open_, tc_close_, think, !r.tools.empty(), r.tools);
  const auto& eos = model_->config().eos;
  double t1 = now_ms();
  int gen = 0;
  std::string content_so_far;
  std::string finish;
  // Handles one generated token; returns false when generation must stop (finish is set).
  auto handle = [&](int t) -> bool {
    ++gen;
    if (std::find(eos.begin(), eos.end(), t) != eos.end()) {
      finish = "stop";
      return false;
    }
    ChatDelta d = parser.push(t);
    if (!d.content.empty()) {
      content_so_far += d.content;
      for (auto& st : r.stop) {
        size_t at = st.empty() ? std::string::npos : content_so_far.find(st);
        if (at != std::string::npos) {
          size_t over = content_so_far.size() - at;
          d.content.resize(d.content.size() > over ? d.content.size() - over : 0);
          finish = "stop";
          break;
        }
      }
    }
    res.reasoning += d.reasoning;
    res.content += d.content;
    for (auto& tc : d.tool_calls) res.tool_calls.push_back(tc);
    if (!d.reasoning.empty() || !d.content.empty() || !d.tool_calls.empty())
      if (!on_delta(d)) finish = "cancelled";
    if (finish.empty() && gen >= r.max_tokens) finish = "length";
    return finish.empty();
  };
  int fed_until = n;  // exclusive: tokens whose effect is in the live state
  const int depth = spec_ ? opts_.spec_depth : 0;
  if (depth > 0) {
    // Speculative decoding: the MTP head drafts `depth` tokens, the trunk verifies them together
    // with the pending token, accepted rows are committed through the DeltaNet tape.
    int P = n, c = 1, h_row0 = last_chunk_row;
    bool go = handle(s.tokens()[n]);
    while (go) {
      if (P + depth + 2 >= s.cap) {
        finish = "length";
        break;
      }
      auto e2 = g.begin();
      model_->encode_mtp(*e2, s, P - c, c, P - c + 1, model_->hidden_rows(), h_row0, P + 1);
      for (int k = 2; k <= depth; ++k)
        model_->encode_mtp(*e2, s, P + k - 2, 1, P + k - 1, model_->mtp_post(), k == 2 ? c - 1 : 0, P + k, false);
      model_->encode_verify(*e2, s, P, depth + 1, sp, uint32_t(P));
      e2->commit_and_wait();
      const int* y = model_->samples();
      int* tk = s.tokens();
      int a = 0;
      while (a < depth && tk[P + 1 + a] == y[a]) ++a;
      tk[P + a + 1] = y[a];
      // emit up to a+1 tokens; m = how many were consumed before stopping
      int m = 0;
      while (m < a + 1) {
        ++m;
        if (!handle(tk[P + m])) {
          go = false;
          break;
        }
      }
      ++spec_cycles_;
      spec_accepted_ += a;
      model_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new pending token
      c = m;
      h_row0 = 0;
      P += m;
    }
    // finish: history rows for the last committed rows, then fold the tape into the state
    auto e3 = g.begin();
    if (c > 0 && P - c >= 0) model_->encode_mtp(*e3, s, P - c, c, P - c + 1, model_->hidden_rows(), h_row0, -1);
    model_->encode_flush(*e3, s);
    e3->commit_and_wait();
    fed_until = P;
  } else {
    std::unique_ptr<Encoder> next;  // step feeding the newest sampled token
    int pos = n;                    // position of the newest sampled token
    auto encode_at = [&](int q) {
      auto x = g.begin();
      model_->encode_step(*x, s, q, sp, uint32_t(q), true);
      x->commit();
      return x;
    };
    if (r.max_tokens > 1) next = encode_at(pos);
    while (true) {
      bool go = handle(s.tokens()[pos]);  // available: produced by the previous command buffer
      if (go && pos + 2 >= s.cap) {
        finish = "length";
        go = false;
      }
      if (!go) break;
      // the step feeding token `pos` is in flight; queue the one after it, then wait
      auto cur = std::move(next);
      next = encode_at(pos + 1);
      cur->wait();
      ++pos;
    }
    // Drain: the step feeding the last token (if encoded) completes, so the live state covers
    // tokens [0, pos] when it ran, else [0, pos).
    fed_until = pos;
    if (next) {
      next->wait();
      fed_until = pos + 1;
    }
  }
  res.slot = int(&slot - slots_.data());
  res.slot_gen = slot.gen;
  res.req_seq = requests_.load();
  ChatDelta tail = parser.finish();
  res.reasoning += tail.reasoning;
  res.content += tail.content;
  if (!tail.reasoning.empty() || !tail.content.empty()) on_delta(tail);
  slot.tokens.assign(s.tokens(), s.tokens() + fed_until);
  res.completion_tokens = gen;
  if (depth > 0 && spec_cycles_ > 0)
    logd("spec: %.2f accepted/drafted over %d cycles", double(spec_accepted_) / (spec_cycles_ * depth), spec_cycles_);
  res.decode_ms = now_ms() - t1;
  if (finish == "stop" && !res.tool_calls.empty()) finish = "tool_calls";
  res.finish_reason = finish;
  logi("chat: prompt %d (cached %d) in %.0f ms (+%.1f ms template/tokenize; last chunk: encoded by %.1f ms, GPU start "
       "+%.1f ms, GPU %.1f ms, wake +%.1f ms), %d tokens in %.0f ms (%.1f tok/s), %s", n, start, res.prefill_ms, tok_ms,
       t_enc, lat_start, gpu_last, lat_wake, gen, res.decode_ms, gen > 1 ? (gen - 1) / (res.decode_ms / 1000) : 0.0,
       finish.c_str());
  return res;
}

void Engine::prefill_history(const ChatRequest& r, const ChatResult& res) {
  static const bool on = [] {  // default on; NI_HISTORY_PREFILL=0 turns it off
    const char* v = getenv("NI_HISTORY_PREFILL");
    return !(v && !strcmp(v, "0"));
  }();
  static const int idle_ms = [] {
    const char* v = getenv("NI_HISTORY_IDLE_MS");
    return v && *v ? std::max(0, atoi(v)) : 250;
  }();
  // Only after a final answer ("stop"): the next message is then a user query, and the template renders
  // this turn (and any tool-call turns since the previous query) without the think block the live state
  // holds. After tool calls the next message is a tool response, which keeps the think block, so the
  // live state already matches it; a turn cut by max_tokens is rarely continued as it stands (benchmark
  // decode runs end this way).
  if (!on || res.slot < 0 || res.finish_reason != "stop") return;
  // only in an idle gap: a request that follows at once (a benchmark's next run, a fast tool) or another
  // conversation must not wait behind this work
  std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
  if (requests_.load() != res.req_seq) return;
  std::unique_lock<std::mutex> lk(mu_, std::try_to_lock);
  if (!lk.owns_lock() || requests_.load() != res.req_seq) return;
  Slot& slot = slots_[size_t(res.slot)];
  if (slot.gen != res.slot_gen) return;
  const double t0 = now_ms();
  // the answer as the client sends it back ("stop": no tool calls), then a placeholder user turn: the
  // template renders answers before the last user query without their think block, so render it that
  // way and keep the text up to the placeholder turn's start
  ChatRequest r2 = r;
  r2.messages.push_back({{"role", "assistant"}, {"content", res.content}});
  static const std::string marker = "\x01nace-history-marker\x01";
  r2.messages.push_back({{"role", "user"}, {"content", marker}});
  std::string text;
  try {
    text = render(r2, false);
  } catch (...) {
    return;
  }
  const size_t mpos = text.rfind(marker);
  const size_t cut = mpos == std::string::npos ? std::string::npos : text.rfind("<|im_start|>", mpos);
  if (cut == std::string::npos) return;
  text.resize(cut);
  std::vector<int> t2 = tok_.encode(text);
  // prefill all but the last token: the follow-up's own prefill then starts with it, so its MTP
  // history pass covers the pair (hidden, next token) of this turn's last row
  const int n = int(t2.size()) - 1;
  size_t common = 0;
  while (common < slot.tokens.size() && common < t2.size() && slot.tokens[common] == t2[common]) ++common;
  // nothing to add, or the live state matches it as far as it goes (the follow-up continues it)
  if (n <= 0 || int(common) >= n || common == slot.tokens.size()) return;
  const Checkpoint* best = nullptr;
  for (auto& c : slot.ckpts)
    if (c.pos > 0 && c.pos <= int(common) && c.pos < n && (!best || c.pos > best->pos)) best = &c;
  Seq& s = *slot.seq;
  if (!best || n + 64 >= s.cap) return;
  Gpu& g = Gpu::get();
  auto e = g.begin();
  model_->restore(*e, s, *best);
  const int start = best->pos;
  for (auto& c : slot.ckpts)
    if (c.pos > start) c.pos = -1;
  std::copy(t2.begin(), t2.end(), s.tokens());
  SamplingParams sp;
  const int C = model_->prefill_chunk();
  for (int p = start; p < n;) {
    const int T = std::min(C, n - p);
    model_->encode_prefill(*e, s, p, T, sp, 0, false);
    if (spec_) model_->encode_mtp_prefill(*e, s, p, T, 0);  // next tokens known up to t2[n]
    p += T;
    e->commit();
    if (p < n) e = g.begin();
  }
  e->wait();
  slot.tokens.assign(t2.begin(), t2.begin() + n);
  logi("history: prefilled %d tokens of the rendered reply (from %d) in %.0f ms", n - start, start, now_ms() - t0);
}

} // namespace ni
