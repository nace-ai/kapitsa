#include "engine/engine.h"
#include "engine/output_parser.h"
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

// Whether the conversation ends with a user query, by the Qwen3.5+ template's own test: a user message
// whose trimmed text is not wrapped in <tool_response> ... </tool_response>. Tool results (role "tool",
// rendered inside a user turn, or such a wrapped user message) are not queries.
static bool ends_with_query(const ojson& messages) {
  if (!messages.is_array() || messages.empty()) return false;
  const ojson& m = messages.back();
  if (!m.is_object() || m.value("role", "") != "user") return false;
  std::string text;
  if (m.contains("content") && m["content"].is_string()) {
    text = m["content"].get<std::string>();
  } else if (m.contains("content") && m["content"].is_array()) {
    for (auto& part : m["content"]) {
      if (!part.is_object() || !part.contains("text") || !part["text"].is_string()) return true;  // image, video
      text += part["text"].get<std::string>();
    }
  }
  const size_t a = text.find_first_not_of(" \n\r\t"), b = text.find_last_not_of(" \n\r\t");
  if (a == std::string::npos) return true;
  static const std::string open = "<tool_response>", close = "</tool_response>";
  const bool wrapped = b + 1 - a >= open.size() + close.size() && text.compare(a, open.size(), open) == 0 &&
                       text.compare(b + 1 - close.size(), close.size(), close) == 0;
  return !wrapped;
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
  // A reply ends at the config's eos ids or at the tokenizer's eos token: the small Qwen3.5 checkpoints
  // (Qwen/Qwen3.5-4B and its MLX conversions) list only <|endoftext|> in config.json and keep the chat
  // turn's end, <|im_end|>, in tokenizer_config.json, so their replies never stopped.
  stop_ids_ = model_->config().eos;
  {
    auto tc = nlohmann::json::parse(read_file(opts_.model_dir + "/tokenizer_config.json"), nullptr, false);
    std::string end = "<|im_end|>";
    if (!tc.is_discarded() && tc.contains("eos_token")) {
      const auto& e = tc["eos_token"];
      if (e.is_string()) end = e.get<std::string>();
      else if (e.is_object()) end = e.value("content", end);
    }
    const int id = tok_.find(end);
    if (id >= 0 && std::find(stop_ids_.begin(), stop_ids_.end(), id) == stop_ids_.end()) stop_ids_.push_back(id);
  }
  const bool spec_off = opts_.spec_depth == 0;  // --spec 0: no speculative decoding at all
  if (opts_.spec_depth < 0) opts_.spec_depth = model_->default_spec_depth();
  spec_ = opts_.spec_depth > 0 && model_->has_mtp();
  if (opts_.spec_depth > Qwen35::kMaxMtpRows - 1) opts_.spec_depth = Qwen35::kMaxMtpRows - 1;
  // DFlash2 block drafter (research/inference/dflash2/SPEC.md), default "auto": a drafter matching the model
  // (DFlash2::find: next to it or in ~/.cache/nace-infer-models/drafters) replaces MTP; without one MTP stays.
  // --dflash none / NI_DFLASH=0 keeps MTP, --dflash <dir> names the drafter, --spec 0 turns both off.
  std::string want = opts_.dflash;
  if (want.empty()) want = getenv("NI_DFLASH") && *getenv("NI_DFLASH") ? getenv("NI_DFLASH") : "auto";
  if (want == "1" || want == "on") want = "auto";
  const bool dfl_off = want == "none" || want == "0" || want == "off" || (want == "auto" && spec_off);
  if (!dfl_off) {
    const bool auto_dir = want == "auto";
    const std::string dir = auto_dir ? DFlash2::find(opts_.model_dir, model_->config()) : want;
    if (dir.empty()) {
      logi("DFlash2: no drafter found for %s (looked next to the model and in ~/.cache/nace-infer-models/drafters)",
           opts_.model_dir.c_str());
    } else if (auto_dir && !DFlash2::supports(*model_)) {
      logi("DFlash2: %s does not fit this model's embedding / LM head format; not used", dir.c_str());
    } else {
      dfl_ = std::make_unique<DFlash2>(*model_, dir);
      DFlash2::prewarm_pipelines();
      spec_ = false;
    }
  }

  auto add_slot = [&](int cap) {
    Slot s;
    s.seq = model_->new_seq(cap);
    if (dfl_) s.dseq = dfl_->new_seq();
    s.ckpts.push_back({model_->new_checkpoint()});
    s.ckpts.push_back({model_->new_checkpoint()});
    if (dfl_)
      for (size_t i = 0; i < s.ckpts.size(); ++i) s.dckpts.push_back(dfl_->new_seq());
    slots_.push_back(std::move(s));
  };
  for (int i = 0; i < std::max(1, opts_.slots); ++i) add_slot(opts_.context);
  const std::string mode = dfl_ ? "DFlash2 speculative decoding (verify " + std::to_string(DFlash2::verify_depth()) + " of 7)"
                         : spec_ ? "MTP speculative decoding depth " + std::to_string(opts_.spec_depth)
                                 : std::string("plain decoding");
  logi("engine ready: %s, context %d, %d slots, %s", opts_.model_id.c_str(), opts_.context, std::max(1, opts_.slots),
       mode.c_str());
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
    const TurnCheckpoint* best = nullptr;
    for (auto& c : slot.ckpts)
      if (c.pos > 0 && c.pos <= int(common) && c.pos < n && (!best || c.pos > best->pos)) best = &c;
    if (best) {
      model_->restore(*e, s, *best);
      start = best->pos;
      // the drafter ring as it was at that checkpoint (the live ring may have overwritten its window since)
      if (dfl_) dfl_->encode_ring_copy(*e, *slot.dckpts[size_t(best - slot.ckpts.data())], *slot.dseq);
    } else {
      model_->reset(s);
    }
    logd("checkpoints: prompt diverges at %zu of %d, %s", common, n,
         best ? (std::string("restored ") + (best->query ? "query" : "tool") + " checkpoint at " +
                 std::to_string(best->pos)).c_str()
              : "none usable, prefill from 0");
  }
  if (dfl_) {
    // the drafter ring keeps the entries of the reused prefix [0, start) that are still in its window
    if (start == 0) DFlash2::reset(*slot.dseq);
    else DFlash2::rewind(*slot.dseq, start);
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
  int saved_ckpt = -1;  // the checkpoint this prefill saves (DFlash2: its ring snapshot follows the prompt commit)
  // The checkpoint this request's turn boundary overwrites: an invalid one, else one saved at a tool
  // response, else the oldest (NI_QUERY_CKPT=0: the oldest of all, as before). The template renders the
  // assistant turns after the last user query with their reasoning and those before it without, so when
  // the next query arrives, the prompt diverges from the live state right after the previous query: only
  // that query's checkpoint can be restored. Every request of a tool loop saves a checkpoint at its tool
  // response, and with "oldest first" two tool calls evicted the query's, so the next query re-prefilled
  // the whole conversation. Kept this way: the newest query checkpoint and the newest one (the tool
  // response a tool loop's next request restores when it diverges).
  static const bool keep_query = [] {
    const char* v = getenv("NI_QUERY_CKPT");
    return !(v && v[0] == '0' && v[1] == 0);
  }();
  const bool query = ends_with_query(r.messages);
  auto victim = [&]() {
    auto rank = [&](const TurnCheckpoint& k) { return k.pos < 0 ? 0 : keep_query && k.query ? 2 : 1; };
    TurnCheckpoint* c = &slot.ckpts[0];
    for (auto& k : slot.ckpts)
      if (rank(k) < rank(*c) || (rank(k) == rank(*c) && k.pos < c->pos)) c = &k;
    logd("checkpoints: %s boundary at %d overwrites %s", query ? "query" : "tool", boundary,
         c->pos < 0 ? "an empty one" : (std::string(c->query ? "query" : "tool") + " at " + std::to_string(c->pos)).c_str());
    c->query = query;
    saved_ckpt = int(c - slot.ckpts.data());
    return c;
  };
  // NI_SHORTPF: a checkpoint inside a chunk is captured by the pass itself (one weight pass for
  // the whole new suffix); otherwise the chunk is split so the checkpoint lands exactly.
  const bool in_pass = model_->can_snapshot();
  // NI_MTP_DEFER (default on): the last chunk's MTP history rows go into their own command buffer behind
  // the trunk's, so the first token (sampled by the trunk pass) is out before they run; the first
  // speculative cycle is queued after them. 0 keeps them in the trunk's command buffer. With DFlash2 the
  // same switch defers the drafter's context commit of the prompt rows (the first drafting pass reads it).
  static const bool mtp_defer = [] {
    const char* v = getenv("NI_MTP_DEFER");
    return !(v && !strcmp(v, "0"));
  }();
  std::unique_ptr<Encoder> mtp_e;
  // DFlash2: every prefill pass below (each schedule, in-pass checkpoints, a chunk rerun after a failed
  // Neural Engine split) captures the target layers' outputs of the prompt rows the drafter can still see,
  // [max(start, n - 2047), n); the drafter projects them into its context ring after the last chunk.
  if (dfl_) dfl_->begin_prefill(n, start);
  double t_enc = 0, h_commit = 0;  // the last chunk's commit: ms since t0, host clock
  while (p < n) {
    int T = std::min(C, n - p);
    Checkpoint* snap = nullptr;
    if (boundary > p && boundary < p + T) {
      // NI_SHORTPF captures the checkpoint inside the pass; a chunk the NI_ANE split would run is cut at
      // the boundary instead (the split's two-half wavefront does not capture mid-chunk state).
      if (in_pass && !model_->ane_splits(T)) snap = victim();
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
        // rerun on the GPU alone; it captures the chunk's rows for the drafter again (the hook is armed)
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
    if (p == boundary) model_->save(*e, s, *victim(), boundary);
    // DFlash2: once every prompt row is captured (a rerun included), disarm the hook and project the rows into
    // the drafter's ring: behind the trunk's command buffer, encoded while the GPU runs it (the first token
    // does not wait for it; the first drafting pass is queued after it), or in it with NI_MTP_DEFER=0
    // The ring snapshot of the checkpoint saved by this prefill (at the turn boundary, near n) is taken right after
    // the commit: it then holds the window before the checkpoint for a later restore.
    const bool dfl_commit = dfl_ && p == n;
    auto commit_ring = [&](Encoder& en) {
      dfl_->encode_prefill_commit(en, *slot.dseq);
      if (saved_ckpt >= 0) dfl_->encode_ring_copy(en, *slot.dseq, *slot.dckpts[size_t(saved_ckpt)]);
    };
    if (dfl_commit) dfl_->end_prefill();
    if (dfl_commit && !mtp_defer) commit_ring(*e);
    e->commit();
    if (p == n) {
      t_enc = now_ms() - t0;  // until the last chunk's commit
      h_commit = Encoder::host_ms();
    }
    if (dfl_commit && mtp_defer) {
      mtp_e = g.begin();
      commit_ring(*mtp_e);
    }
    if (mtp_e) mtp_e->commit();  // queued behind the trunk; waited for by the first spec cycle's buffer
    if (p < n) e = g.begin();
  }
  e->wait();
  const double h_woke = Encoder::host_ms();
  res.prefill_ms = now_ms() - t0;
  const double gpu_last = e->gpu_ms();
  // last chunk: commit -> GPU start, GPU end -> this thread awake (host clock)
  const double lat_start = e->gpu_start_host_ms() - h_commit, lat_wake = h_woke - e->gpu_end_host_ms();
  slot.tokens = prompt;

  // ---- decode ----
  const bool think = !text.empty() && text.size() >= 8 && text.compare(text.size() - 8, 8, "<think>\n") == 0;
  // A tool call written inside the reasoning, with nothing after it but the end of the turn, is returned as a
  // tool call (Qwen3.6 sometimes stops without closing its think block); see OutputParser. NI_THINK_TOOLCALLS=0
  // leaves it in the reasoning, as before.
  static const bool think_tools = [] {
    const char* v = getenv("NI_THINK_TOOLCALLS");
    return !(v && v[0] == '0' && v[1] == 0);
  }();
  OutputParser parser(tok_, think_open_, think_close_, tc_open_, tc_close_, think, !r.tools.empty(), r.tools,
                      think_tools);
  const auto& eos = stop_ids_;
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
  int req_cycles = 0;  // DFlash2 cycles of this request
  const int depth = spec_ ? opts_.spec_depth : 0;
  if (dfl_) {
    // DFlash2, double-buffered: each cycle's verify (8-row target forward with captures + on-GPU
    // acceptance) is committed, then right behind it the NEXT cycle's drafting half (ring commit of the
    // rows this acceptance keeps + drafting pass from the anchor it picks; both read the GPU-side state
    // the acceptance writes), and only then does the host wait and read the accepted count. The GPU never
    // waits for the host between cycles; the host encodes each verify while the drafting half runs.
    DFlashSeq& d = *slot.dseq;
    const int K = DFlash2::verify_depth();
    int P = n, m = 0, a_last = 0;
    req_cycles = 0;
    bool go = handle(s.tokens()[n]);
    std::unique_ptr<Encoder> next_draft;  // committed drafting half of the coming cycle
    if (go && P + 8 + DFlash2::kB + 2 < s.cap) {
      // host write of the drafter state: nothing in flight reads it (the prefill is done; its deferred
      // prompt commit, if still running, is host-driven)
      dfl_->gpu_state_reset(P);
      next_draft = g.begin();
      dfl_->encode_cycle_draft_gpu(*next_draft, s, d, P, sp);
      next_draft->commit();
    }
    while (go) {
      if (!next_draft) {  // no room for another drafting pass
        finish = "length";
        break;
      }
      next_draft.reset();
      auto e2 = g.begin();
      dfl_->encode_cycle_verify(*e2, s, P, K, sp);
      e2->commit();
      if (P + 8 + 8 + DFlash2::kB + 2 < s.cap) {  // the next anchor is at most P + 8
        next_draft = g.begin();
        dfl_->encode_cycle_draft_gpu(*next_draft, s, d, P + 8, sp);
        next_draft->commit();
      }
      e2->wait();
      const int a = dfl_->accepted();
      a_last = a;
      const int* tk = s.tokens();
      m = 0;  // tokens consumed before stopping = rows committed
      while (m < a + 1) {
        ++m;
        if (!handle(tk[P + m])) {
          go = false;
          break;
        }
      }
      ++spec_cycles_;
      ++req_cycles;
      spec_accepted_ += m;
      model_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new anchor
      P += m;
      d.ctx = P;  // the committed drafting half (if any) puts these rows into the ring
    }
    // finish: the ring rows of the last cycle (unless the committed drafting half already wrote them),
    // then fold the tape into the recurrent state
    auto e3 = g.begin();
    if (!next_draft && m > 0) {
      d.ctx = P - m;
      dfl_->encode_commit(*e3, d, m);
    }
    // The committed drafting half wrote all a_last + 1 rows of the last acceptance (positions P - m ..
    // P - m + a_last) to the ring. When the host kept fewer (a stop token or the budget inside the
    // accepted chain), the rows past d.ctx overwrote the slots of the window's oldest positions, which a
    // later rewind (prefix reuse) would otherwise still treat as valid context.
    if (next_draft && m < a_last + 1) d.valid_lo = std::max(d.valid_lo, P - m + a_last + 1 - DFlash2::kWin);
    model_->encode_flush(*e3, s);
    e3->commit_and_wait();
    fed_until = P;
  } else if (depth > 0) {
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
  ChatDelta tail = parser.finish(finish == "stop");
  res.reasoning += tail.reasoning;
  res.content += tail.content;
  for (auto& tc : tail.tool_calls) res.tool_calls.push_back(tc);
  if (!tail.reasoning.empty() || !tail.content.empty() || !tail.tool_calls.empty()) on_delta(tail);
  if (parser.promoted() > 0)
    logi("chat: %d tool call(s) written inside the reasoning returned as tool calls", parser.promoted());
  slot.tokens.assign(s.tokens(), s.tokens() + fed_until);
  res.completion_tokens = gen;
  if (dfl_ && spec_cycles_ > 0)
    logd("dflash: %.2f tokens/cycle over %ld cycles", double(spec_accepted_) / spec_cycles_, spec_cycles_);
  else if (depth > 0 && spec_cycles_ > 0)
    logd("spec: %.2f accepted/drafted over %ld cycles", double(spec_accepted_) / (spec_cycles_ * depth), spec_cycles_);
  res.decode_ms = now_ms() - t1;
  if (finish == "stop" && !res.tool_calls.empty()) finish = "tool_calls";
  res.finish_reason = finish;
  char cyc[64] = "";
  if (dfl_ && req_cycles > 0) snprintf(cyc, sizeof cyc, ", %.2f tok/cycle", double(gen - 1) / req_cycles);
  logi("chat: prompt %d (cached %d) in %.0f ms (+%.1f ms template/tokenize; last chunk: encoded by %.1f ms, GPU start "
       "+%.1f ms, GPU %.1f ms, wake +%.1f ms), %d tokens in %.0f ms (%.1f tok/s), %s%s", n, start, res.prefill_ms, tok_ms,
       t_enc, lat_start, gpu_last, lat_wake, gen, res.decode_ms, gen > 1 ? (gen - 1) / (res.decode_ms / 1000) : 0.0,
       finish.c_str(), cyc);
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
  const TurnCheckpoint* best = nullptr;
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
  // DFlash2: the drafter's context ring goes back to the checkpoint with the target state, and the rows
  // prefilled here enter it like prompt rows (captured by every pass, projected after the last chunk)
  if (dfl_) {
    dfl_->encode_ring_copy(*e, *slot.dckpts[size_t(best - slot.ckpts.data())], *slot.dseq);
    DFlash2::rewind(*slot.dseq, start);
    dfl_->begin_prefill(n, start);
  }
  SamplingParams sp;
  const int C = model_->prefill_chunk();
  for (int p = start; p < n;) {
    const int T = std::min(C, n - p);
    // as in chat(): a chunk the NI_ANE split runs is checkpointed and waited for, and rerun on the GPU
    // alone if the split failed (the rerun captures its rows for the drafter again)
    const bool split = model_->ane_splits(T);
    if (split) {
      if (!ane_ck_) ane_ck_ = std::make_unique<Checkpoint>(model_->new_checkpoint());
      model_->save(*e, s, *ane_ck_, p);
    }
    model_->encode_prefill(*e, s, p, T, sp, 0, false);
    if (spec_) model_->encode_mtp_prefill(*e, s, p, T, 0);  // next tokens known up to t2[n]
    if (split) {
      e->commit_and_wait();
      if (!model_->ane_finish_chunk()) {
        e = g.begin();
        model_->restore(*e, s, *ane_ck_);
        model_->encode_prefill(*e, s, p, T, sp, 0, false);
        if (spec_) model_->encode_mtp_prefill(*e, s, p, T, 0);
        e->commit_and_wait();
      }
      e = g.begin();
    }
    p += T;
    if (dfl_ && p == n) {
      dfl_->end_prefill();
      dfl_->encode_prefill_commit(*e, *slot.dseq);
    }
    e->commit();
    if (p < n) e = g.begin();
  }
  e->wait();
  slot.tokens.assign(t2.begin(), t2.begin() + n);
  logi("history: prefilled %d tokens of the rendered reply (from %d) in %.0f ms", n - start, start, now_ms() - t0);
}

} // namespace ni
