#include "engine/engine.h"
#include "engine/output_parser.h"
#include "util/log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
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

// A message's role, or "" (a field of another type included)
static std::string role_of(const ojson& m) {
  if (!m.is_object()) return "";
  auto it = m.find("role");
  return it != m.end() && it->is_string() ? it->get<std::string>() : "";
}

// Whether the conversation ends with a user query, by the Qwen3.5+ template's own test: a user message
// whose trimmed text is not wrapped in <tool_response> ... </tool_response>. Tool results (role "tool",
// rendered inside a user turn, or such a wrapped user message) are not queries. System messages after it
// (an agent's context, rendered as system turns in place) do not change that.
static bool ends_with_query(const ojson& messages) {
  if (!messages.is_array() || messages.empty()) return false;
  size_t last = messages.size() - 1;
  while (last > 0 && role_of(messages[last]) == "system") --last;
  const ojson& m = messages[last];
  if (role_of(m) != "user") return false;
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

// Prompt lookup (n-gram drafts): the tokens that followed the latest earlier occurrence of the last n tokens of
// tokens[0, P] (n from `nmax` down to `nmin`), up to `k` of them, into out (empty: no match). The search looks back
// at most `window` positions.
static void ngram_drafts(const int* tok, int P, int k, std::vector<int>& out, int nmax = 4, int nmin = 2,
                         int window = 16384) {
  out.clear();
  for (int n = nmax; n >= nmin; --n) {
    if (P + 1 < n + 1) continue;
    const int* key = tok + P + 1 - n;  // the last n tokens, ending at P
    const int lo = std::max(0, P + 1 - window);
    for (int i = P - n; i >= lo; --i) {  // an earlier occurrence starting at i (ending before P)
      if (tok[i] != key[0] || !std::equal(key, key + n, tok + i)) continue;
      const int from = i + n;            // what followed it
      for (int j = 0; j < k && from + j <= P; ++j) out.push_back(tok[from + j]);
      if (!out.empty()) return;
    }
  }
}

// Dory's speculative drafts (per-loop KV): NI_DORY_SPEC = both (the default: prompt lookup, else self-speculation),
// ngram (prompt lookup only) or self
// Streamed experts (a Qwen3.8-Flash-Next Q2_0 pack, src/model/expert_store.h): the store's work since the last call,
// for a request's log line ("" when the model streams nothing); `tokens` > 0 adds the bytes read per token
static std::string store_note(Model* m, const char* what, int tokens) {
  auto* q4 = dynamic_cast<Qwen4Exp*>(m);
  q20::ExpertStore* st = q4 ? q4->expert_store() : nullptr;
  if (!st || st->complete()) return "";
  const q20::StoreStats s = st->stats();
  st->reset_stats();
  const double n = std::max<double>(1, double(s.lookups));
  char b[320];
  snprintf(b, sizeof b, "%s: %llu experts looked up in %llu turns, %.1f%% in memory, %.1f%% read ahead and waited for, "
           "%.1f%% read at the turn, %.2f GB read", what, (unsigned long long)s.lookups,
           (unsigned long long)s.handshakes, 100.0 * double(s.hits) / n, 100.0 * double(s.late) / n,
           100.0 * double(s.misses) / n, double(s.bytes) / 1e9);
  std::string out = b;
  if (tokens > 0) {
    snprintf(b, sizeof b, " (%.1f MB a token)", double(s.bytes) / 1e6 / tokens);
    out += b;
  }
  return out;
}

static std::string dory_spec_kind() {
  static const std::string k = [] {
    const char* v = getenv("NI_DORY_SPEC");
    const std::string s = v && *v ? v : "both";
    return s == "ngram" || s == "self" ? s : std::string("both");
  }();
  return k;
}

// ---- the GPU turn ----

void TurnLock::lock() {
  std::unique_lock<std::mutex> lk(m_);
  const uint64_t t = next_++;
  if (t != serving_) {
    waiting_.fetch_add(1, std::memory_order_relaxed);
    cv_.wait(lk, [&] { return serving_ == t; });
    waiting_.fetch_sub(1, std::memory_order_relaxed);
  }
  since_.store(now_ms(), std::memory_order_relaxed);
}

double TurnLock::held_ms() const { return now_ms() - since_.load(std::memory_order_relaxed); }

void TurnLock::unlock() {
  {
    std::lock_guard<std::mutex> lk(m_);
    ++serving_;
  }
  cv_.notify_all();
}

bool TurnLock::try_lock() {
  std::lock_guard<std::mutex> lk(m_);
  if (next_ != serving_) return false;  // held, or waited for
  ++next_;
  since_.store(now_ms(), std::memory_order_relaxed);
  return true;
}

uint64_t TurnLock::events() const {
  std::lock_guard<std::mutex> lk(m_);
  return events_;
}

void TurnLock::notify() {
  {
    std::lock_guard<std::mutex> lk(m_);
    ++events_;
  }
  ev_cv_.notify_all();
}

bool TurnLock::wait_event(uint64_t seen, const std::function<bool()>& stop) {
  unlock();
  bool ev = false;
  {
    std::unique_lock<std::mutex> lk(m_);
    auto changed = [&] { return events_ != seen; };
    if (!stop) {
      ev_cv_.wait(lk, changed);
      ev = true;
    } else {
      while (!(ev = ev_cv_.wait_for(lk, std::chrono::seconds(1), changed))) {
        lk.unlock();
        const bool s = stop();
        lk.lock();
        if (s) break;
      }
    }
  }
  lock();
  return ev;
}

bool Engine::contended(bool quantum_too) const {
  static const bool test = [] {
    const char* v = getenv("NI_PARK_TEST");
    return v && atoi(v) > 0;
  }();
  static const double quantum = [] {
    const char* v = getenv("NI_TURN_MS");
    return v && *v ? std::max(0.0, atof(v)) : 30.0;
  }();
  return test || (turn_.waiting() > 0 && (!quantum_too || turn_.held_ms() >= quantum));
}

void Engine::yield_turn(ChatResult& res) {
  ++res.yields;
  turn_.unlock();
  turn_.lock();
}

// Whether a chat template renders an assistant turn's reasoning with a newline of its own before </think> (Nemotron
// Nano 4B's: "<think>\n" + reasoning + "\n</think>\n"): the model's own newline there then belongs to the markup, not
// the reasoning, for the turn to render as it was generated (and the next request to reuse the cache through it)
static bool newline_before_think_close(const minja::chat_template& tmpl) {
  static const std::string needle = "\x01kapitsa-reasoning\x01";
  try {
    minja::chat_template_inputs in;
    in.messages = ojson::array({{{"role", "user"}, {"content", "u"}},
                                {{"role", "assistant"}, {"content", "c"}, {"reasoning_content", needle}}});
    in.add_generation_prompt = false;
    in.extra_context = {{"truncate_history_thinking", false}};
    minja::chat_template_options opt;
    const std::string out = tmpl.apply(in, opt);
    const size_t at = out.find(needle);
    return at != std::string::npos && out.compare(at + needle.size(), 9, "\n</think>") == 0;
  } catch (...) {
    return false;
  }
}

Engine::Engine(const EngineOptions& o) : opts_(o) {
  if (opts_.model_id.empty()) {
    std::string d = opts_.model_dir;
    while (!d.empty() && d.back() == '/') d.pop_back();
    opts_.model_id = d.substr(d.find_last_of('/') + 1);
  }
  // Memory: fail before reading any weights when the model, its buffers and the slots' state with a minimal
  // KV cache cannot fit the budget; the KV caches then grow as conversations fill (Qwen35::reserve).
  budget_ = opts_.mem_budget_gb > 0 ? size_t(opts_.mem_budget_gb * 1e9) : default_mem_budget();
  if (opts_.slots <= 0) {
    opts_.slots = auto_slots(opts_.model_dir, opts_, budget_);
    logi("slots: %d (%s; --slots sets how many requests run at once)", opts_.slots,
         opts_.slots == kAutoSlots ? "the default where memory allows"
                                   : "as many as this budget holds with one full conversation, the same drafter and prefill chunk");
  }
  // DFlash2 block drafter (src/model/dflash2.h), default "auto": a drafter matching the model
  // (DFlash2::find: next to it or in ~/.cache/nace-infer-models/drafters) replaces MTP; without one MTP stays.
  // --dflash none / NI_DFLASH=0 keeps MTP, --dflash <dir> names the drafter, --spec 0 turns both off. Found before
  // the model loads, so its memory counts in the fit check; auto leaves it out where it would cost context.
  const LoadPlan lp = plan_load(opts_.model_dir, opts_.dflash, opts_.spec_depth == 0, opts_.context, opts_.slots,
                                budget_, opts_.prefill_chunk);
  if (!lp.dflash_left_out.empty())
    logi("DFlash2: %s left out: its %.2f GB would keep a %d-token conversation from fitting the %.1f GB budget",
         lp.dflash_left_out.c_str(), lp.dflash_found_bytes / 1e9, opts_.context, budget_ / 1e9);
  {
    const int slots = std::max(1, opts_.slots);
    if (opts_.prefill_chunk <= 0) {
      opts_.prefill_chunk = lp.prefill_chunk;
      const int base = opts_.prefill_chunk > 2048 && opts_.prefill_chunk <= prefill_chunk_room(2048)
                           ? 2048  // only its tail room: the 2048-row chunk still fits
                           : opts_.prefill_chunk;
      if (base != 2048)
        logi("memory: %d-row prefill chunks (2048 would leave less than one %d-token conversation in the %.1f GB budget)",
             base, opts_.context, budget_ / 1e9);
    }
    const StartNeed need = start_need(opts_.model_dir, lp, opts_.prefill_chunk, opts_.context, slots);
    const size_t rws = Gpu::get().recommended_working_set();
    if (need.weights > budget_)  // nothing but a smaller variant helps: say so before the itemized message
      fatal("%s does not fit this Mac: its weights alone (%.1f GB, %s) exceed the GPU memory Kapitsa may use here "
            "(%.1f GB of Metal's %.1f GB working set). Use a quantized variant of the model (`kapitsa list` shows each "
            "model's variants; 8-bit and 6-bit weights are about 1/2 and 2/5 of bf16) or a Mac with more memory.",
            opts_.model_dir.c_str(), need.weights / 1e9, need.formats.c_str(), budget_ / 1e9, rws / 1e9);
    if (need.bytes > budget_)
      fatal("insufficient memory: %s needs %.1f GB of GPU memory with %s; this Mac's budget is %.1f GB (Metal's "
            "recommended working set %.1f GB). Fewer --slots, --prefill-chunk 1024, no drafter, a smaller model or "
            "--mem-budget can help.",
            opts_.model_dir.c_str(), need.bytes / 1e9, need.parts.c_str(), budget_ / 1e9, rws / 1e9);
  }
  // the publisher's sampling settings (generation_config.json; Nemotron 3.5 and Dory: temperature 1.0, top-p 0.95, no
  // top-k: the sampler's 32)
  auto publisher_sampling = [&] {
    auto gc = nlohmann::json::parse(read_file(opts_.model_dir + "/generation_config.json"), nullptr, false);
    if (!gc.is_discarded() && gc.is_object()) {
      if (gc.contains("temperature") && gc["temperature"].is_number()) defaults_.temperature = gc["temperature"];
      if (gc.contains("top_p") && gc["top_p"].is_number()) defaults_.top_p = gc["top_p"];
      defaults_.top_k = gc.contains("top_k") && gc["top_k"].is_number() ? gc["top_k"].get<int>() : 0;
    }
  };
  if (Qwen4Exp::is_model(opts_.model_dir)) {
    // Qwen3.8-Flash-Next; a Q2_0 pack's expert store takes what the budget leaves after the slots' state and one
    // conversation's KV cache (up to 32K tokens)
    const MemPlan mp = model_plan(opts_.model_dir, opts_.prefill_chunk);
    const int slots = std::max(1, opts_.slots);
    const size_t keep = size_t(slots) * mp.slot_state + mp.kv_per_pos * size_t(std::min(opts_.context, 32768));
    auto m = std::make_unique<Qwen4Exp>(opts_.model_dir, opts_.prefill_chunk, budget_ > keep ? budget_ - keep : 1);
    q35_ = m.get();
    model_ = std::move(m);
  } else if (Dory::is_model(opts_.model_dir)) {
    LoopKv kv = LoopKv::last_loop;
    if (!opts_.loop_kv.empty() && !parse_loop_kv(opts_.loop_kv, &kv))
      fatal("--loop-kv: last-loop or per-loop, not \"%s\"", opts_.loop_kv.c_str());
    if (opts_.loops < 0 || opts_.loops > Dory::kMaxLoops) fatal("--loops: 1 to %d", Dory::kMaxLoops);
    auto m = std::make_unique<Dory>(opts_.model_dir, opts_.prefill_chunk, opts_.loops, kv);
    dory_ = m.get();
    model_ = std::move(m);
    publisher_sampling();
  } else if (NemotronH::is_model(opts_.model_dir)) {
    auto m = std::make_unique<NemotronH>(opts_.model_dir, opts_.prefill_chunk);
    nh_ = m.get();
    model_ = std::move(m);
    publisher_sampling();
  } else if (Flint::is_model(opts_.model_dir)) {
    auto m = std::make_unique<Flint>(opts_.model_dir, opts_.prefill_chunk);
    flint_ = m.get();
    model_ = std::move(m);
    publisher_sampling();
  } else {
    auto m = std::make_unique<Qwen35>(opts_.model_dir, opts_.prefill_chunk);
    q35_ = m.get();
    model_ = std::move(m);
  }
  tok_.load(opts_.model_dir + "/tokenizer.json");
  model_->set_mem_budget(budget_);
  std::string src = chat_template_source(opts_.model_dir);
  if (src.empty()) fatal("no chat template in %s", opts_.model_dir.c_str());
  if (flint_) src = flint_template(src);
  tmpl_ = std::make_unique<minja::chat_template>(src, flint_ ? "<|begin_of_text|>" : "",
                                                 flint_ ? "<|end_of_text|>" : "<|im_end|>");
  think_open_ = tok_.find("<think>");
  think_close_ = tok_.find("</think>");
  tc_open_ = tok_.find("<tool_call>");
  tc_close_ = tok_.find("</tool_call>");
  if (nh_) {
    // Nemotron Nano 9B v2: its tokenizer has no <think> or tool-call tokens (the tags are text; its calls
    // <TOOLCALL>[{"name": ..., "arguments": ...}]</TOOLCALL>), and its template switches thinking with "/think" /
    // "/no_think" in a message instead of enable_thinking: the parser matches the tags in the text, render() turns
    // enable_thinking into the flag
    text_tags_ = text_tags_for(tok_, src);
    think_flags_ = think_flag_template(src);
    // the whitespace before </think> stays in the reasoning (OutputParser's think_ws) unless the template writes a
    // newline of its own there (Nemotron Nano 4B: "<think>\n" + reasoning + "\n</think>\n"), checked for the dense
    // models (the Nemotron 3.5 MoE models keep it as before)
    const auto& kind = nh_->config().kind;
    if (std::find(kind.begin(), kind.end(), 'E') == kind.end() && newline_before_think_close(*tmpl_)) think_ws_ = false;
  }
  // A reply ends at the config's eos ids or at the tokenizer's eos token: the small Qwen3.5 checkpoints
  // (Qwen/Qwen3.5-4B and its MLX conversions) list only <|endoftext|> in config.json and keep the chat
  // turn's end, <|im_end|>, in tokenizer_config.json, so their replies never stopped.
  stop_ids_ = model_->eos_ids();
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
  // Dory: speculative decoding, exact only with per-loop KV (with last-loop KV the rows of a verify pass would see
  // each other's same-loop keys, where decoding them one by one sees final-loop ones): prompt lookup, else drafts by
  // the model itself run with one loop (NI_DORY_SPEC). --spec N drafts per cycle (0 = plain decoding); by default 2
  // for a request of 2 loops and 3 for 3-4 (M4 Pro, 128-token greedy replies, against plain per-loop decoding:
  // 2 loops +17% on a short prompt, -2% after 2.9K tokens (3 drafts: +22% / -8%); 4 loops 3 drafts +57% / +19%)
  if (dory_ && dory_->kv_mode() == LoopKv::per_loop) {
    dory_spec_ = opts_.spec_depth < 0 ? 3 : std::min(opts_.spec_depth, Dory::kMaxRows - 1);
    dory_spec_auto_ = opts_.spec_depth < 0;
  }
  const int spec_asked = opts_.spec_depth;  // (-1: the model's default)
  if (opts_.spec_depth < 0) opts_.spec_depth = q35_ ? q35_->default_spec_depth() : 0;
  spec_ = q35_ && opts_.spec_depth > 0 && q35_->has_mtp();
  if (opts_.spec_depth > Qwen35::kMaxMtpRows - 1) opts_.spec_depth = Qwen35::kMaxMtpRows - 1;
  if (q35_ && q35_->block_drafter_ok() && !lp.dflash_off && lp.dflash_left_out.empty()) {
    const std::string& dir = lp.dflash;
    if (dir.empty()) {
      logi("DFlash2: no drafter found for %s (looked next to the model and in ~/.cache/nace-infer-models/drafters)",
           opts_.model_dir.c_str());
    } else if (lp.dflash_auto && !DFlash2::supports(*q35_)) {
      logi("DFlash2: %s does not fit this model's embedding / LM head format; not used", dir.c_str());
    } else {
      const size_t a0 = Gpu::get().allocated_bytes();
      // one slot: nothing decodes alongside a request, so a 16-row block's deeper lattice is free to take (with more
      // slots a 16-row block has no batched cycle at all, Engine::batch_rows, and requests take turns instead)
      const int block_rows = DFlash2::plan_block_rows(q35_->config(), opts_.slots, nullptr, DFlash2::tensor_path_ok());
      const bool rows16 = block_rows > DFlash2::kB;
      dfl_ = std::make_unique<DFlash2>(*q35_, dir, 0, rows16);
      dfl_dir_ = dir;
      DFlash2::prewarm_pipelines();
      spec_ = false;
      // the plan against what loading took (rings and the first prefill's buffers come later)
      size_t per_slot = 0, at_prefill = 0;
      const size_t planned =
          DFlash2::plan_bytes(dir, q35_->config(), Qwen35::plan(opts_.model_dir, 1024, rows16).head_row_bytes,
                              &per_slot, &at_prefill, block_rows);
      logi("DFlash2: %.3f GB allocated at load (planned %.3f GB); %.3f GB more for %d slots' rings and %.3f GB at the "
           "first prefill", (Gpu::get().allocated_bytes() - a0) / 1e9, (planned - at_prefill - (size_t(4) << 20)) / 1e9,
           3.0 * per_slot * std::max(1, opts_.slots) / 1e9, std::max(1, opts_.slots), at_prefill / 1e9);
    }
  }

  if (nh_ && !lp.dflash_off && lp.dflash_left_out.empty()) {
    if (lp.dflash.empty()) {
      logi("Nemotron-H: no block drafter found next to %s", opts_.model_dir.c_str());
    } else {
      lili_ = NhDrafter::open(*nh_, lp.dflash);
      // KFlash's one rule, as on the Qwen draft models: trees on every target (the budget picks 1 .. 8 live rows per
      // cycle; the verify runs those alone); NI_KFLASH=0 keeps the drafter's chains
      const char* kfv = getenv("NI_KFLASH");
      if (!(kfv && *kfv) || atoi(kfv) != 0) {
        // the budget's cache: this model, drafter and chip
        std::string key = opts_.model_dir.substr(opts_.model_dir.find_last_of('/') + 1) + "-" + lili_->name() + "-" +
                          Gpu::get().name();
        for (char& c : key)
          if (!isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '.') c = '_';
        kf_budget_ = std::make_unique<KfBudget>(key);
        kf_params_ = Gpu::get().alloc(sizeof(KfParams), "kf_params");
        logi("KFlash: tree drafting on %s (budget cache %s)", lili_->name(), key.c_str());
      }
    }
  }
  // A dense Nemotron-H model (Nemotron Nano) has no block drafter: prompt lookup drafts up to K tokens per cycle
  // (--spec K, else NI_NH_NGRAM=K), verified in one pass; the reply is the one plain decoding writes. By default 4
  // where the MLP layers are BF16 or FP8 (the Nano 4B: a verify pass's 2-8 rows read each weight once, as one row
  // does; on an M1 Pro decoding 4-8% faster in BF16; in FP8 with its 8-bit head 1% on the benchmark's code (4% with
  // the BF16 head), 1.4-2.3x on replies that repeat their prompt, 4-5% slower on free prose that drafts rarely match)
  // and off for NVFP4 (the Nano 9B v2: within 2% either way there, the M1's NVFP4 products of several rows being
  // bound by their arithmetic, not by the weights' bytes).
  if (nh_ && !lili_ && spec_asked != 0) {
    const auto& kind = nh_->config().kind;
    if (std::find(kind.begin(), kind.end(), 'E') == kind.end()) {
      bool fp4 = false;
      for (const NhLayer& L : nh_->layers()) fp4 = fp4 || (L.kind == 'D' && L.up_s.fmt == NhFmt::FP4);
      const char* v = getenv("NI_NH_NGRAM");
      const int k = spec_asked > 0 ? spec_asked : v && *v ? atoi(v) : fp4 ? 0 : 4;
      if (k > 0) nh_ngram_ = std::min(k, NemotronH::kMaxRows - 1);
    }
  }

  // Flint has no drafter: prompt lookup drafts up to K tokens per cycle when asked (--spec K, else NI_FLINT_NGRAM=K),
  // verified in one pass; the reply is the one plain decoding writes up to near-ties (a pass of 2+ rows sums in another
  // order) and to the rare row where a router score sits on bf16's rounding boundary: the router rounds its scores
  // before its top-6, so such a pass can pick another expert there than a decode step (docs/usage.md). Off by default: on an M4 Pro a pass of 2 / 4 rows costs 1.6 / 2.8 decode steps (each row brings its own 6
  // of 256 experts), and on the benchmark's coding reply a draft exists for 1 token in 4 (replayed: 4-16% slower than
  // plain decoding; it pays only on replies that copy their prompt).
  if (flint_ && spec_asked != 0) {
    const char* v = getenv("NI_FLINT_NGRAM");
    const int k = spec_asked > 0 ? spec_asked : v && *v ? atoi(v) : 0;
    if (k > 0) flint_ngram_ = std::min(k, Flint::kMaxRows - 1);
  }

  // NI_PREFIX_CKPT=0: no prefix checkpoint. Not with LiLiCorr / KFlash drafters (their context has no rewind to an
  // empty ring).
  {
    const char* v = getenv("NI_PREFIX_CKPT");
    prefix_ckpt_ = !(v && v[0] == '0' && v[1] == 0) && !lili_;
    { const char* c = getenv("NI_PREFIX_COPY"); prefix_copy_ = !(c && c[0] == '0' && c[1] == 0); }
  }
  auto add_slot = [&](int limit) {
    Slot s;
    s.seq = model_->new_seq(limit, 0);  // KV cache on first use, growing with the conversation
    if (dfl_) s.dseq = dfl_->new_seq();
    if (lili_) s.lseq = lili_->new_seq();
    s.ckpts.push_back({model_->new_checkpoint()});
    s.ckpts.push_back({model_->new_checkpoint()});
    if (prefix_ckpt_) s.ckpts.push_back({model_->new_checkpoint(), false, true});
    for (size_t i = 0; i < s.ckpts.size(); ++i) {  // (the prefix checkpoint has no drafter snapshot: null)
      if (dfl_) s.dckpts.push_back(s.ckpts[i].prefix ? nullptr : dfl_->new_seq());
      if (lili_) s.lckpts.push_back(s.ckpts[i].prefix ? nullptr : lili_->new_seq());
    }
    slots_.push_back(std::move(s));
  };
  for (int i = 0; i < std::max(1, opts_.slots); ++i) add_slot(opts_.context);
  Gpu& g = Gpu::get();
  const size_t used = g.allocated_bytes(), per = model_->kv_bytes_per_pos();
  const size_t room = budget_ > used ? budget_ - used : 0;
  context_fit_ = int(std::min<size_t>(size_t(opts_.context), room / per));
  fixed_bytes_ = used;
  logi("GPU: %s, %d cores%s", g.name().c_str(), g.core_count(), g.core_count_measured() ? "" : " (not measured: NI_GPU_CORES or the tier's smallest)");
  logi("memory: %.2f GB allocated (model, buffers, %d slots' state), budget %.1f GB (recommended working set %.1f GB): "
       "KV room %.2f GB = %zu tokens (%.0f KB per token), one conversation up to %d of %d tokens",
       used / 1e9, std::max(1, opts_.slots), budget_ / 1e9, g.recommended_working_set() / 1e9, room / 1e9, room / per,
       per / 1024.0, context_fit_, opts_.context);
  // images: the model's vision tower, loaded on the first image that needs encoding
  if (auto enc = vision::open_encoder(opts_.model_dir, opts_.image_max_pixels)) {
    const std::string m0 = enc->image_start(), m1 = enc->image_end();
    const int i0 = m0.empty() ? -2 : tok_.find(m0), i1 = m1.empty() ? -2 : tok_.find(m1);
    if (i0 >= 0) img_pre_ = {i0};
    if (i1 >= 0) img_post_ = {i1};
    if (i0 == -1 || i1 == -1) {
      logi("images: off (the tokenizer lacks the image marks %s %s)", enc->image_start().c_str(), enc->image_end().c_str());
    } else {
      logi("images: %s, %.2f GB loaded on the first image", enc->describe().c_str(), enc->weight_bytes() / 1e9);
      vision_ = std::make_unique<vision::Vision>(std::move(enc));
    }
  }
  // batched decode where it was measured: DFlash2 drafters (Qwen3.6 35B-A3B, Qwen3.8 27B) and LiLiCorr on Bolt's 24 GB
  // checkpoint; the 4B's DFlash (v1) drafter, Super-VL's LiLiCorr and Lightning's DSpark take turns until measured
  // (NI_BATCH=1 batches them too)
  batch_measured_ = (dfl_ && !dfl_->config().v1) ||
                    (lili_ && std::string(lili_->name()) == "LiLiCorr" && nh_->iq2_experts());
  const std::string mode = lili_ ? std::string(lili_->name()) + " speculative decoding (7-token blocks)"
                         : flint_ngram_ > 0 ? "prompt-lookup speculative decoding (up to " + std::to_string(flint_ngram_) +
                                                  " drafts per cycle)"
                         : nh_ngram_ > 0 ? "prompt-lookup speculative decoding (up to " + std::to_string(nh_ngram_) +
                                               " drafts per cycle)"
                         : dfl_ && dfl_->tree() ? std::string("KFlash speculative decoding (DFlash2 lattice, tree verify)")
                         : dfl_ ? "DFlash2 speculative decoding (verify " + std::to_string(dfl_->verify_depth()) + " of 7)"
                         : spec_ ? "MTP speculative decoding depth " + std::to_string(opts_.spec_depth)
                         : dory_spec_ > 0 ? "speculative decoding (" +
                                                    (dory_spec_kind() == "ngram"  ? std::string("prompt lookup")
                                                     : dory_spec_kind() == "self" ? std::string("1-loop drafts")
                                                                                  : std::string("prompt lookup, else 1-loop drafts")) +
                                                    ", " + (dory_spec_auto_ ? std::string("2-3") : std::to_string(dory_spec_)) +
                                                    " per cycle)"
                                 : std::string("plain decoding");
  const std::string looped = dory_ ? ", " + std::to_string(dory_->default_loops()) + " loops by default (requests take 1-" +
                                         std::to_string(Dory::kMaxLoops) + "), " + loop_kv_name(dory_->kv_mode()) +
                                         " KV"
                                   : std::string();
  logi("engine ready: %s, context %d, %d slots, %s%s%s", opts_.model_id.c_str(), opts_.context, std::max(1, opts_.slots),
       mode.c_str(), looped.c_str(), vision_ ? ", images on" : "");
  disk_start();  // the SSD prefix cache (engine_disk.cpp): its files open in the background
}

Engine::~Engine() { save_disk_cache(); }

void Engine::save_kflash() {
  if (kf_budget_) kf_budget_->save();
  if (dfl_ && dfl_->budget()) dfl_->budget()->save();
  if (dfl_ && dfl_->batch_budget()) dfl_->batch_budget()->save();
}

std::shared_ptr<const TokenTable> Engine::token_table() {
  std::lock_guard<std::mutex> lk(table_mu_);
  if (!table_) {
    auto t = std::make_shared<TokenTable>(tok_, model_->vocab(), stop_ids_, tc_open_, tc_close_);
    t->warm();
    table_ = t;
  }
  return table_;
}

size_t default_mem_budget() {
  if (const char* v = getenv("NI_MEM_BUDGET"); v && *v) return size_t(atof(v) * 1e9);
  const size_t rws = Gpu::get().recommended_working_set();
  return rws - std::max<size_t>(size_t(512) << 20, rws / 25);
}

MemPlan model_plan(const std::string& model_dir, int prefill_chunk) {
  // kept per process: a model's headers do not change while it is planned, and the largest take most of a second to
  // read (Super-VL's), which the picker, `kapitsa doctor` and default_context would otherwise pay on every plan.
  // The plan is the 8-row one: the key has no slot count, so a 16-row block's rows (a drafter's verify buffers, ~30 MB
  // on the 27B) are not in it. The picker is 0.07% of a 38 GB budget optimistic there; the engine's own load plans the
  // block it will really use (DFlash2::plan_block_rows) and logs that against what loading takes.
  static std::mutex mu;
  static std::map<std::string, MemPlan> plans;
  const std::string key = model_dir + "\n" + std::to_string(prefill_chunk) + "\n" + std::to_string(Dory::serving_loops()) +
                          loop_kv_name(Dory::serving_kv());  // Dory's plan counts the loops and KV it serves with
  {
    std::lock_guard<std::mutex> lk(mu);
    if (const auto it = plans.find(key); it != plans.end()) return it->second;
  }
  const MemPlan p = Dory::is_model(model_dir)       ? Dory::plan(model_dir, prefill_chunk)
                    : Flint::is_model(model_dir)    ? Flint::plan(model_dir, prefill_chunk)
                    : Qwen4Exp::is_model(model_dir) ? Qwen4Exp::plan(model_dir, prefill_chunk)
                    : NemotronH::is_model(model_dir) ? NemotronH::plan(model_dir, prefill_chunk)
                                                     : Qwen35::plan(model_dir, prefill_chunk);
  std::lock_guard<std::mutex> lk(mu);
  plans.emplace(key, p);
  return p;
}
static size_t model_ane_bytes(const std::string& model_dir, const MemPlan& p) {
  if (Dory::is_model(model_dir) || Flint::is_model(model_dir)) return 0;
  return NemotronH::is_model(model_dir) ? p.ane : Qwen35::ane_plan(model_dir, p).bytes;
}

// Tail room: a prompt a few rows past a chunk multiple should not pay a whole extra chunk of weight streaming for
// those rows (4K on the 35B: 2048+2048+34 -> 2065+2065). The buffers get 1/32 more rows and the chunk loop splits the
// prompt evenly, which only ever removes chunks. Off with NI_BAL=0 (the chunking before it).
bool prefill_even_chunks() {
  static const bool on = [] {
    const char* v = getenv("NI_BAL");
    return !(v && v[0] == '0' && v[1] == 0);
  }();
  return on;
}
int prefill_chunk_room(int c) { return c + std::max(8, c / 32); }
int prefill_chunk_rows(int chunk, int64_t p, int64_t n, bool even) {
  const int64_t left = n - p;
  int rows;
  if (!even || left <= chunk) {
    rows = int(std::min<int64_t>(chunk, left));
  } else {
    const int64_t k = (left + chunk - 1) / chunk;  // the chunks min(chunk, left) would take, without one more
    rows = int((left + k - 1) / k);
  }
  logd("prefill chunk of %d rows: %lld of %lld left, chunk %d%s", rows, (long long)left, (long long)n, chunk,
       even ? " even" : "");
  return rows;
}

int auto_prefill_chunk(const std::string& model_dir, int context, int slots, size_t budget) {
  // 1024-row buffers need ~45% less than 2048-row ones (35B: 0.75 -> 0.38 GB, 27B: 0.94 -> 0.52) for a few % of
  // prefill speed
  if (NemotronH::is_model(model_dir))
    if (const int w = NemotronH::wide_chunk(model_dir)) {
      const MemPlan pw = model_plan(model_dir, w);
      if (pw.fixed(std::max(1, slots)) + model_ane_bytes(model_dir, pw) + pw.kv_per_pos * size_t(context) <= budget) return w;
    }
  const MemPlan p = model_plan(model_dir, 2048);
  const size_t ane = model_ane_bytes(model_dir, p);
  // Tail room is for MoE prefill, which streams the expert weights once per chunk: a dense model's prefill is compute
  // bound, so a sliver of rows costs little, and a model whose prefill splits on the Neural Engine keeps the plain
  // chunk (the split runs chunks of at most 2048 rows, so room would push every chunk off it).
  const bool room = prefill_even_chunks() && !p.dense && ane == 0;
  // Single-stream 35B-A3B 4-bit on the 32-core M1 Max: larger prompt chunks
  // reduce expert-tile padding and weight rereads. Keep the existing chunk for
  // concurrent slots, other formats/devices, or when the extra buffers do not fit.
  if (slots == 1 && !p.dense && p.hidden == 2048 && p.layers == 40 &&
      p.formats.rfind("4-bit group 64", 0) == 0 && Gpu::get().name() == "Apple M1 Max" &&
      Gpu::get().core_count() == 32) {
    const MemPlan wide = model_plan(model_dir, room ? prefill_chunk_room(4096) : 4096);
    if (wide.fixed(1) + model_ane_bytes(model_dir, wide) + wide.kv_per_pos * size_t(context) <= budget)
      return room ? prefill_chunk_room(4096) : 4096;
  }
  if (p.fixed(std::max(1, slots)) + ane + p.kv_per_pos * size_t(context) > budget) return 1024;
  if (room) {
    const MemPlan r = model_plan(model_dir, prefill_chunk_room(2048));
    if (r.fixed(std::max(1, slots)) + model_ane_bytes(model_dir, r) + r.kv_per_pos * size_t(context) <= budget)
      return prefill_chunk_room(2048);
  }
  return 2048;
}

LoadPlan plan_load(const std::string& model_dir, const std::string& dflash, bool spec_off, int context, int slots,
                   size_t budget, int prefill_chunk) {
  LoadPlan lp;
  std::string want = dflash;
  if (want.empty()) want = getenv("NI_DFLASH") && *getenv("NI_DFLASH") ? getenv("NI_DFLASH") : "auto";
  if (want == "1" || want == "on") want = "auto";
  lp.dflash_auto = want == "auto";
  lp.dflash_off = want == "none" || want == "0" || want == "off" || (lp.dflash_auto && spec_off);
  slots = std::max(1, slots);
  if (Dory::is_model(model_dir) || Flint::is_model(model_dir)) {
    lp.dflash_off = true;  // no block drafter for Dory or Flint
  } else if (NemotronH::is_model(model_dir)) {
    // the block drafter (LiLiCorr, DSpark) in the model's dflash/ folder (auto), or the one named
    if (!lp.dflash_off) {
      lp.dflash = lp.dflash_auto ? NhDrafter::find(model_dir) : want;
      if (!lp.dflash.empty()) {
        size_t per_slot = 0;
        lp.dflash_bytes = lp.dflash_found_bytes = NhDrafter::plan_bytes(lp.dflash, &per_slot) + 3 * per_slot * size_t(slots);
        if (lp.dflash_auto &&
            !conversation_fits(model_dir, context, slots, budget > lp.dflash_bytes ? budget - lp.dflash_bytes : 0)) {
          lp.dflash_left_out = lp.dflash;
          lp.dflash.clear();
          lp.dflash_bytes = 0;
        }
      }
    }
  } else if (!lp.dflash_off && !Qwen4Exp::is_model(model_dir)) {  // (no block drafter binds to Qwen3.8-Flash-Next)
    lp.dflash = lp.dflash_auto ? DFlash2::find(model_dir, Qwen35::read_config(model_dir)) : want;
    if (!lp.dflash.empty()) {
      // auto: a drafter that would keep one conversation from its full context is left out (MTP or plain decoding
      // instead); the model with 1024-row prefill buffers, the slots' state, the Neural Engine split and one full
      // conversation come first
      const Qwen35Config dc = Qwen35::read_config(model_dir);
      const int block_rows = DFlash2::plan_block_rows(dc, slots, nullptr, DFlash2::tensor_path_ok());
      const Qwen35::MemPlan p = Qwen35::plan(model_dir, 1024, block_rows > DFlash2::kB);
      size_t per_slot = 0;  // the ring of the live state and of each of the slot's two checkpoints
      lp.dflash_bytes = lp.dflash_found_bytes =
          DFlash2::plan_bytes(lp.dflash, dc, p.head_row_bytes, &per_slot, nullptr, block_rows) +
          3 * per_slot * size_t(slots);
      if (lp.dflash_auto &&
          !conversation_fits(model_dir, context, slots, budget > lp.dflash_bytes ? budget - lp.dflash_bytes : 0)) {
        lp.dflash_left_out = lp.dflash;
        lp.dflash.clear();
        lp.dflash_bytes = 0;
      }
    }
  }
  lp.prefill_chunk = prefill_chunk > 0 ? prefill_chunk
                                       : auto_prefill_chunk(model_dir, context, slots,
                                                            budget > lp.dflash_bytes ? budget - lp.dflash_bytes : 0);
  return lp;
}

StartNeed start_need(const std::string& model_dir, const LoadPlan& lp, int prefill_chunk, int context, int slots) {
  slots = std::max(1, slots);
  const int min_ctx = std::min(context, 8192);
  const MemPlan mp = model_plan(model_dir, prefill_chunk);
  StartNeed n;
  n.bytes = mp.fixed(slots) + lp.dflash_bytes + mp.kv_per_pos * size_t(min_ctx);
  n.weights = mp.weights;
  n.formats = mp.formats;
  char b[400];
  snprintf(b, sizeof b,
           "%d slot%s (weights %.1f GB, MTP head %.1f, DFlash2 drafter %.1f, buffers for %d-token prefill chunks %.1f, "
           "conversation state %.1f, KV cache for %d tokens %.1f)",
           slots, slots > 1 ? "s" : "", mp.weights / 1e9, mp.mtp / 1e9, lp.dflash_bytes / 1e9, prefill_chunk,
           mp.activations / 1e9, slots * mp.slot_state / 1e9, min_ctx, mp.kv_per_pos * min_ctx / 1e9);
  n.parts = b;
  return n;
}

int auto_slots(const std::string& model_dir, const EngineOptions& opt, size_t budget) {
  try {
    const bool spec_off = opt.spec_depth == 0;
    const LoadPlan two = plan_load(model_dir, opt.dflash, spec_off, opt.context, 2, budget, opt.prefill_chunk);
    // the most slots up to kAutoSlots that cost nothing two slots have: the same decoding (a drafter two slots keep is
    // kept), the same prefill chunk, and one full --ctx conversation beside them
    for (int n = kAutoSlots; n > 2; --n) {
      const LoadPlan more = plan_load(model_dir, opt.dflash, spec_off, opt.context, n, budget, opt.prefill_chunk);
      if (more.dflash == two.dflash && more.prefill_chunk == two.prefill_chunk &&
          conversation_fits(model_dir, opt.context, n, budget > more.dflash_bytes ? budget - more.dflash_bytes : 0))
        return n;
    }
  } catch (const std::exception&) {  // headers that cannot be planned: the engine reports it when it loads them
  }
  return 2;
}

bool conversation_fits(const std::string& model_dir, int context, int slots, size_t budget) {
  const MemPlan p = model_plan(model_dir, 1024);
  return p.fixed(std::max(1, slots)) + model_ane_bytes(model_dir, p) + p.kv_per_pos * size_t(context) <= budget;
}

int parse_context(const std::string& s, std::string* why) {
  std::string v;
  for (const char c : s) v += char(tolower(static_cast<unsigned char>(c)));
  if (v == "max") return kContextMax;
  size_t i = 0;
  long long n = 0, unit = 1;
  while (i < v.size() && isdigit(static_cast<unsigned char>(v[i])) && n <= INT_MAX) n = n * 10 + (v[i++] - '0');
  if (i > 0 && i + 1 == v.size() && (v[i] == 'k' || v[i] == 'm')) unit = v[i++] == 'k' ? 1024 : 1048576;
  if (i == 0 || i != v.size() || n < 1 || n * unit > INT_MAX) {
    if (why) *why = "--context takes tokens (131072), k or m of them (128k, 1m) or max, not \"" + s + "\"";
    return 0;
  }
  return int(n * unit);
}

int model_context_limit(const std::string& model_dir) {
  std::ifstream f(model_dir + "/config.json");
  if (!f) return 0;
  const nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
  if (!j.is_object()) return 0;
  // the language model's config where it has one of its own (the Qwen3.5 family: text_config; Bolt: llm_config)
  static const nlohmann::json none;
  for (const char* k : {"text_config", "llm_config", ""}) {
    const nlohmann::json& c = !*k ? j : j.contains(k) ? j[k] : none;
    if (c.is_object() && c.contains("max_position_embeddings") && c["max_position_embeddings"].is_number_integer()) {
      const long long n = c["max_position_embeddings"].get<long long>();
      return n > 0 ? int(std::min<long long>(n, INT_MAX)) : 0;
    }
  }
  return 0;
}

int context_within_limit(const std::string& model_dir, int context) {
  const int limit = model_context_limit(model_dir);
  return limit > 0 ? std::min(context, limit) : context;
}

int default_context(const std::string& model_dir, const EngineOptions& opt, size_t budget, ContextWhy* why) {
  const int limit = model_context_limit(model_dir);
  const int top = limit > 0 ? std::min(kDefaultContext, limit) : kDefaultContext;
  const int floor = std::min(kMinDefaultContext, top);
  ContextWhy w = top < kDefaultContext ? ContextWhy::kModelLimit : ContextWhy::kDefault;
  int c = top;
  try {
    const int slots = fit_slots(opt.slots);
    const bool spec_off = opt.spec_depth == 0;
    // the drafter the model has at 32K: a longer context by default never costs it
    const std::string keep = plan_load(model_dir, opt.dflash, spec_off, floor, slots, budget, opt.prefill_chunk).dflash;
    auto keeps = [&](int ctx) {
      const LoadPlan lp = plan_load(model_dir, opt.dflash, spec_off, ctx, slots, budget, opt.prefill_chunk);
      return lp.dflash == keep &&
             conversation_fits(model_dir, ctx, slots, budget > lp.dflash_bytes ? budget - lp.dflash_bytes : 0);
    };
    if (!keeps(top)) {
      // the largest 8K step between that does (a longer context only takes more): found by halving, a few plans
      int lo = floor / 8192, hi = (top - 1) / 8192;
      while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (keeps(mid * 8192)) lo = mid;
        else hi = mid - 1;
      }
      c = std::max(floor, lo * 8192);
      w = keep.empty() ? ContextWhy::kMemory : ContextWhy::kMemoryWithDrafter;
    }
  } catch (const std::exception&) {  // headers that cannot be planned: the engine reports it when it loads them
    c = floor;
  }
  if (why) *why = w;
  return c;
}

int variant_fits(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget) {
  try {
    const std::string dir = hub::plan_dir(ref);
    if (dir.empty()) return -1;
    // a context past the model's own limit is planned at the limit, as the server would run it
    const int ctx = context_within_limit(dir, opt.context);
    if (!conversation_fits(dir, ctx, fit_slots(opt.slots), budget)) return 0;
    const LoadPlan lp = plan_load(dir, opt.dflash, opt.spec_depth == 0, ctx, fit_slots(opt.slots), budget, opt.prefill_chunk);
    return lp.dflash_left_out.empty() ? 1 : 0;
  } catch (const std::exception&) {  // headers that cannot be planned (a damaged download): unknown
    return -1;
  }
}

hub::ModelRef pick_variant(const hub::CatalogModel& m, const EngineOptions& opt, size_t budget, bool* fits) {
  std::vector<std::string> order{m.default_variant};
  order.insert(order.end(), m.fallbacks.begin(), m.fallbacks.end());
  hub::ModelRef smallest, downloaded, present;
  bool planned = false;
  for (const std::string& v : order) {
    hub::ModelRef r = hub::resolve(m.name + ":" + v);
    r.variant_given = false;
    const int f = variant_fits(r, opt, budget);
    if (f == 1) {
      if (fits) *fits = true;
      return r;
    }
    if (f == 0) smallest = r, planned = true;
    if (downloaded.dir.empty() && hub::is_current(r)) downloaded = r;
    struct stat st;
    if (present.dir.empty() && stat(r.dir.c_str(), &st) == 0) present = r;  // a download to finish
  }
  if (fits) *fits = false;
  if (planned) return smallest;
  if (!downloaded.dir.empty()) return downloaded;
  if (!present.dir.empty()) return present;
  return hub::resolve(m.name);
}

// Commits a chunk that runs a GPU + Neural Engine split and waits for it. A split job that never reports would keep
// the GPU waiting on its event (each job has a 2 s watchdog; this bounds the whole chunk as a last resort): the split
// is given up, the GPU's waits are released, and ane_finish_chunk() then reruns the chunk on the GPU alone.
void Engine::wait_split(Encoder& e) {
  e.commit();
  if (!e.wait_for(60000)) {
    logi("NI_ANE: a split prefill chunk has not finished in 60 s; giving up the Neural Engine split");
    model_->ane_abort("a split chunk did not finish within 60 s");
  }
  e.wait();
}

void Engine::drop_slot(Slot& slot) {
  model_->drop_kv(*slot.seq);
  model_->reset(*slot.seq);  // nothing in flight uses it: other slots are idle, this one's request has not started
  slot.tokens.clear();
  slot.img_spans.clear();
  slot.img_digests.clear();
  slot.turns = 0;
  slot.done_at = 0;
  for (auto& c : slot.ckpts) {
    c.pos = -1;
    c.query = false;
  }
  ++slot.gen;  // an idle history prefill for it gives up
  slot.fills.clear();
  slot.live_ok = false;
  slot.disk_version = ++slot.version;  // (nothing left to save)
}

bool Engine::reserve_kv(Slot& slot, int need, bool started) {
  Seq& s = *slot.seq;
  if (model_->reserve(s, need)) return true;
  if (vision_ && vision_->release(false) && model_->reserve(s, need)) return true;  // earlier requests' image embeddings
  // other slots' caches: idle ones (least recently used first), then those of requests waiting for memory before
  // their prefill, the newest first (a busy slot's cache is otherwise its request's)
  std::vector<Slot*> idle, waiting;
  for (auto& o : slots_) {
    if (&o == &slot || o.seq->cap == 0) continue;
    if (!o.busy) idle.push_back(&o);
    else if (o.mem_wait == Slot::MemWait::before_prefill && (started || o.arrival > slot.arrival)) waiting.push_back(&o);
  }
  std::sort(idle.begin(), idle.end(), [](const Slot* a, const Slot* b) { return a->last_used < b->last_used; });
  std::sort(waiting.begin(), waiting.end(), [](const Slot* a, const Slot* b) { return a->arrival > b->arrival; });
  for (Slot* o : idle) {
    logi("memory: dropping the KV cache of slot %d (%d tokens cached) for a %d-token context in slot %d",
         int(o - slots_.data()), int(o->tokens.size()), need, int(&slot - slots_.data()));
    if (!started) disk_leaving(*o, 0);  // to the SSD first (not between decode cycles: the write-behind has it there)
    drop_slot(*o);
    if (model_->reserve(s, need)) return true;
  }
  for (Slot* o : waiting) {
    logi("memory: dropping the KV cache of slot %d (%d tokens cached; its request waits for memory and will read its "
         "prompt from the start) for a %d-token context in slot %d", int(o - slots_.data()), int(o->tokens.size()), need,
         int(&slot - slots_.data()));
    drop_slot(*o);
    if (model_->reserve(s, need)) return true;
  }
  // last: the Neural Engine split's surfaces and its rerun checkpoint (the prefill then runs on the GPU alone)
  if (model_->ane_release()) {
    if (ane_ck_) {
      for (auto* v : {&ane_ck_->conv, &ane_ck_->rec})
        for (auto& b : *v) Gpu::get().release(b);
      ane_ck_.reset();
    }
    if (model_->reserve(s, need)) return true;
  }
  if (vision_ && vision_->release(true) && model_->reserve(s, need)) return true;  // the image encoder's weights
  return false;
}

bool Engine::reserve_kv_wait(Slot& slot, int need, const std::function<bool()>& cancelled, bool started) {
  if (reserve_kv(slot, need, started)) return true;
  // The memory is the other requests' (their KV caches). Wait while one of them runs (it ends, or takes the caches
  // of requests waiting before their prefill and goes on); before the prefill also while a started request waits
  // (one of those gives up when nothing runs) or an earlier one does (it comes first and may take this cache). So
  // something always runs or gives up, and a request before its prefill gives up only when its prompt does not fit
  // even alone: every other request in flight is a later one waiting, whose cache it took.
  bool logged = false;
  for (;;) {
    int running = 0, started_waiting = 0, earlier_waiting = 0;
    for (const Slot& o : slots_) {
      if (&o == &slot || !o.busy) continue;
      if (o.mem_wait == Slot::MemWait::none) ++running;
      else if (o.mem_wait == Slot::MemWait::growth) ++started_waiting;
      else if (o.arrival < slot.arrival) ++earlier_waiting;
    }
    if (!(running > 0 || (!started && (started_waiting > 0 || earlier_waiting > 0)))) return false;
    if (!logged)
      logi("memory: slot %d waits for a %d-token KV cache%s until other requests in flight end (%d running, %d waiting "
           "for memory)", int(&slot - slots_.data()), need, started ? " to go on" : " before its prefill", running,
           active_ - 1 - running);
    logged = true;
    if (cancelled && cancelled()) return false;
    slot.mem_wait = started ? Slot::MemWait::growth : Slot::MemWait::before_prefill;
    // nothing runs now: the waiting requests decide again (a started one takes this cache or gives up, an earlier
    // one takes it or finds its prompt too long), else none of them would see an event
    if (running == 0) turn_.notify();
    turn_.wait_event(turn_.events(), cancelled);
    slot.mem_wait = Slot::MemWait::none;
    if (reserve_kv(slot, need, started)) return true;
  }
}

void Engine::verify_masks(const GrammarMatcher& gm, const GrammarState& at, Seq& s, int P, int rows, const int* parent,
                          int live) {
  uint32_t* const mw = s.mask.as<uint32_t>();
  const size_t W = size_t(s.mask_words);
  thread_local std::vector<GrammarState> st;
  thread_local std::vector<char> ok;
  st.resize(size_t(rows));
  ok.assign(size_t(rows), 0);
  st[0] = at;
  ok[0] = 1;
  gm.fill_mask(at, mw);
  const int* tk = s.tokens();
  for (int r = 1; r < rows; ++r) {
    const int par = parent ? parent[r] : r - 1;
    if (r < live && par >= 0 && par < r && ok[size_t(par)]) {
      st[size_t(r)] = st[size_t(par)];
      ok[size_t(r)] = gm.advance(st[size_t(r)], tk[P + r]);
    }
    if (ok[size_t(r)]) gm.fill_mask(st[size_t(r)], mw + size_t(r) * W);
    else memset(mw + size_t(r) * W, 0xff, W * 4);
  }
}

// ---- batched decode ----

int Engine::batch_rows(int n) const {
  // NI_BATCH: 0 = requests take turns, 1 = batched with every drafter; unset: batched where it was measured
  static const int mode = [] {
    const char* v = getenv("NI_BATCH");
    return v && *v ? atoi(v) : -1;
  }();
  if (mode == 0 || n < 2 || n > 8 || (!dfl_ && !lili_) || (mode < 0 && !batch_measured_)) return 0;
  // a 16-row drafting block has no batched cycle: it drafts a depth-15 lattice with the pair kernel, and the batch loop
  // drafts 7-token chains (encode_draft with tree = false), so the requests take turns instead
  if (dfl_ && dfl_->block() > DFlash2::kB) return 0;
  // NI_BATCH_ROWS="n:rows,...": rows per member for n members (0: take turns)
  static const std::vector<int> env = [] {
    std::vector<int> t(9, -1);
    const char* v = getenv("NI_BATCH_ROWS");
    for (const char* p = v; p && *p;) {
      char* end = nullptr;
      const long k = strtol(p, &end, 10);
      if (end == p || *end != ':') break;
      const long r = strtol(end + 1, &end, 10);
      if (k >= 2 && k <= 8 && r >= 0 && r <= 8) t[size_t(k)] = int(r);
      p = *end == ',' ? end + 1 : end;
    }
    return t;
  }();
  int rows = env[size_t(n)];
  // KFlash's batch budget (DFlash2 drafters, 2 .. 4 members; NI_BATCH_KF=0 keeps the table below): each cycle, the rows
  // of one verify pass in all and which member gets them, from the members' calibrated drafts and this Mac's measured
  // cost of a batched cycle (KfBatch). Against the table on the M4 Pro (same streams, interleaved): 3 at once 35B
  // +16-20%, 27B +14-22% (it batched them instead of taking turns: 8 rows in all instead of 2 each); 2 and 4 at once
  // within -2 .. +4%
  static const bool kf = [] {
    const char* v = getenv("NI_BATCH_KF");
    return !(v && *v) || atoi(v) != 0;
  }();
  if (rows < 0 && kf && dfl_ && dfl_->batch_budget() && n <= KfBatch::kMembers) return kBatchBudget;
  if (rows < 0) {
    // As measured on the M4 Pro (MacBook Pro, 48 GB), 512-token greedy replies to code prompts, tok/s in all against
    // the requests taking turns (bench/batch/concurrency.py streams):
    // - Qwen3.6 35B-A3B 4-bit (DFlash2, MoE): 2 at once 4 rows each (the anchor + 3 drafts) 197 against 155 (2 rows
    //   each 153, plain 130); 3 at once 2 rows each 184 against 149 (plain 166); 4 at once 2 rows each 202 against 158
    //   (plain 199)
    // - Qwen3.8 27B 4-bit (DFlash2, dense): 2 at once 4 rows each 61-65 against 57-59 (2 rows each 41, plain 28); 3 at
    //   once 2 rows each 52-54 against 56: the requests take turns; 4 at once 2 rows each 63-65 against 55-56 (plain 51)
    // - Nemotron 3.5 Bolt, NVIDIA 24 GB (LiLiCorr): 2 at once 4 rows each 76.6 against 62.6 (2 rows each 55.5, plain
    //   47.2); 3 at once 2 rows each 64.4 against 62.8 (plain 62.5); 4 at once plain 74.6 against 61.7 (2 rows each 70.0)
    // More members (--slots above 4, not measured): plain rows.
    static constexpr int moe[5] = {0, 0, 4, 2, 2}, dense[5] = {0, 0, 4, 0, 2}, lili[5] = {0, 0, 4, 2, 1};
    const int* t = dfl_ ? (q35_->config().moe ? moe : dense) : lili;
    rows = n <= 4 ? t[n] : 1;
  }
  if (rows > 0 && n * rows > 8) rows = 8 / n;
  return rows;
}

bool Engine::batch_wanted() const {
  if (!dfl_ && !lili_) return false;
  int n = 0;  // requests decoding (not waiting for memory)
  for (const Slot& o : slots_) n += o.busy && o.decoding && o.mem_wait == Slot::MemWait::none;
  // join the running batch: it is to take every decoding request (its runner waits for them when fewer rows do not
  // pay, as 2 of 3 requests might not)
  if (batch_runner_) return batch_rows(std::min(std::max(n, int(batch_.size()) + 1), 8)) > 0;
  if (now_ms() < batch_backoff_) return false;
  return n >= 2 && batch_rows(std::min(n, 8)) > 0;
}

void Engine::batch_leave(BatchJob& j, BatchJob::State st) {
  batch_.erase(std::remove(batch_.begin(), batch_.end(), &j), batch_.end());
  // its staged rows into its drafter's ring: the ring holds every row before j.P again, as its own loop expects
  const bool staged = dfl_ ? j.slot->dseq->staged > 0 : lili_ && j.slot->lseq->staged > 0;
  if (staged) {
    auto e = Gpu::get().begin();
    if (dfl_) dfl_->encode_commit_staged(*e, *j.slot->dseq);
    else lili_->encode_commit_staged(*e, *j.slot->lseq);
    e->commit_and_wait();
  }
  if (st == BatchJob::State::done) j.slot->decoding = false;  // (no batch waits for it any more)
  {
    std::lock_guard<std::mutex> lk(bmu_);
    j.state = st;
  }
  bcv_.notify_all();
}

bool Engine::join_batch(BatchJob& job) {
  batch_.push_back(&job);
  turn_.notify();  // a runner waiting for its members
  if (!batch_runner_) {
    batch_runner_ = true;
    job.runner = true;
  }
  for (;;) {
    if (job.runner && job.state == BatchJob::State::active) run_batch(job);
    if (job.state != BatchJob::State::active) break;
    // a member: its cycles run in the runner's passes; it waits, without the turn, until it is done, returned or runs
    // the batch itself
    turn_.unlock();
    {
      std::unique_lock<std::mutex> lk(bmu_);
      bcv_.wait(lk, [&] { return job.state != BatchJob::State::active || job.runner; });
    }
    turn_.lock();
  }
  return job.state == BatchJob::State::done;
}

void Engine::run_batch(BatchJob& me) {
  double t_wait = -1;  // waiting for the members to come since
  while (me.state == BatchJob::State::active) {
    const int n = int(batch_.size());
    const int rows = n >= 2 ? batch_rows(n) : 0;
    if (rows == 0) {
      // the other decoding requests join at their next cycle boundary: wait for them (without the turn), up to a second
      int dec = 0;
      for (const Slot& o : slots_) dec += o.busy && o.decoding && o.mem_wait == Slot::MemWait::none;
      if (dec > n && batch_rows(std::min(dec, 8)) > 0 && (t_wait < 0 || now_ms() - t_wait < 1000)) {
        if (t_wait < 0) t_wait = now_ms();
        const double t0 = t_wait;
        turn_.wait_event(turn_.events(), [t0] { return now_ms() - t0 >= 1000; });
        continue;
      }
      if (t_wait >= 0 && now_ms() - t_wait >= 1000) batch_backoff_ = now_ms() + 2000;  // they did not come
      for (BatchJob* j : std::vector<BatchJob*>(batch_)) batch_leave(*j, BatchJob::State::returned);
      break;
    }
    t_wait = -1;
    // KV room for the cycle, as each member's own loop grows it; a member that cannot have it goes back to its loop
    // (which waits for the memory, or ends at the context's limit)
    for (BatchJob* j : std::vector<BatchJob*>(batch_)) {
      Seq& s = *j->slot->seq;
      const int need = std::min(s.limit, j->P + 8 + 8 + 8 + 3);
      if (j->P + 8 + 8 + 2 >= s.limit || (need > s.cap && !reserve_kv(*j->slot, need, true)))
        batch_leave(*j, BatchJob::State::returned);
    }
    if (me.state != BatchJob::State::active) break;
    const int m = int(batch_.size());
    if (m < 2 || batch_rows(m) == 0) continue;
    // plain rows run pipelined (NI_BATCH_PIPE=0: one pass at a time; also with NI_TRACE_TOKENS, which reads the
    // logits of each pass after it)
    static const bool pipe = [] {
      const char* v = getenv("NI_BATCH_PIPE");
      return !(v && v[0] == '0') && !(getenv("NI_TRACE_TOKENS") && *getenv("NI_TRACE_TOKENS"));
    }();
    if (batch_rows(m) == 1 && pipe) batch_plain_run();
    else batch_cycle(batch_rows(m));
    for (BatchJob* j : std::vector<BatchJob*>(batch_))
      if (j->stopped) batch_leave(*j, BatchJob::State::done);
    if (me.state != BatchJob::State::active) break;
    if (contended()) {  // between cycles: other requests' turns (prefill chunks, a member's end, a request joining)
      turn_.unlock();
      turn_.lock();
    }
  }
  // this member left: another one runs the batch, or the last one goes back to its own loop
  if (batch_.size() < 2) {
    batch_stats();
    if (dfl_ && dfl_->batch_budget() && dfl_->batch_budget()->cycles() > 0) {
      logi("%s", dfl_->batch_budget()->summary().c_str());
      dfl_->batch_budget()->save(KfBatch::kSaveEvery);
    }
  }
  batch_runner_ = false;
  if (batch_.size() >= 2) {
    BatchJob* next = batch_.front();
    {
      std::lock_guard<std::mutex> lk(bmu_);
      next->runner = true;
    }
    batch_runner_ = true;
    bcv_.notify_all();
  } else if (batch_.size() == 1) {
    batch_leave(*batch_.front(), BatchJob::State::returned);
  }
}

void Engine::batch_cycle(int rows) {
  Gpu& g = Gpu::get();
  const double t_cycle = now_ms();
  // KFlash's batch budget (DFlash2): every member drafts its chain, then the budget chooses the rows each verifies
  const bool kfb = rows == kBatchBudget && dfl_ && dfl_->batch_budget();
  if (rows == kBatchBudget && !kfb) rows = std::max(1, 8 / int(batch_.size()));  // (not reached: the budget is DFlash2's)
  const int k = kfb ? DFlash2::kB - 1 : rows - 1;  // drafted tokens per member
  const std::vector<BatchJob*> jobs = batch_;
  const size_t nj = jobs.size();
  std::vector<int> kv(nj, k);  // drafts each member verifies (its rows: kv + 1)
  // Two command buffers, as a single request's cycle: the first (each member's staged rows into its drafter's ring
  // before its draft or when its stage is full, then its draft) starts on the GPU while the host encodes the second,
  // one pass over every row (under the budget the second is encoded once the first is done: its rows follow the drafts)
  const SamplingParams greedy;
  auto e1 = g.begin();
  bool e1_work = false;
  // grouped: the members' per-sequence dispatches of a stage run side by side (encode_verify_multi groups them)
  auto e2 = g.begin_grouped();
  // DFlash2, sampled requests (T > 0): the walk samples their drafts and the exact speculative acceptance (as a single
  // request's cycle) keeps them; greedy requests keep a draft while it equals the verify's argmax
  const bool exact = [] {
    const char* v = getenv("NI_BATCH_EXACT");
    return !(v && v[0] == '0');
  }();
  std::vector<KfBatch::Member> mem(kfb ? nj : 0);
  // structured output: a masked member's rows (its own, kv[i] + 1) wait for their masks, which the host computes from
  // its drafts once the drafting command buffer is done, while the verify's layers run; one signal for all of them.
  // Flags as the verify was encoded (a member's </think> in this cycle turns its grammar on for the next).
  uint64_t pending = 0;
  std::vector<char> gmask(nj, 0);
  if (dfl_) {
    for (BatchJob* j : jobs) {
      Slot& sl = *j->slot;
      DFlashSeq& d = *sl.dseq;
      dfl_->stage_init(d);
      if (d.staged > 0 && (k > 0 || d.staged == dfl_->block())) dfl_->encode_commit_staged(*e1, d), e1_work = true;
      if (k > 0) {
        const bool sampled = exact && j->sp.temperature > 0;
        dfl_->encode_draft(*e1, *sl.seq, d, j->P, sampled ? j->sp : greedy);
        if (sampled) dfl_->encode_save_walk(*e1, d);
        if (kfb) dfl_->encode_save_scores(*e1, d);
        e1_work = true;
      }
    }
    if (kfb) {
      // the budget's input, once the drafting passes are done: per member and draft position the largest of the 16
      // candidates' probabilities under the walk's scores (a greedy walk's own draft; a sampled member's does not move
      // it), and which calibration it takes (exact speculative sampling: the budget's own)
      e1->commit();
      e1->wait();
      for (size_t i = 0; i < nj; ++i) {
        const float* S = jobs[i]->slot->dseq->scores();
        for (int p = 0; p < KfBatch::kDrafts; ++p) {
          float mx = -INFINITY;
          for (int c = 0; c < DFlash2::kTopK; ++c) mx = std::max(mx, S[p * DFlash2::kTopK + c]);
          double z = 0;
          for (int c = 0; c < DFlash2::kTopK; ++c) z += std::exp(double(S[p * DFlash2::kTopK + c] - mx));
          mem[i].lc[p] = std::isfinite(mx) && z >= 1.0 ? float(-std::log(z)) : -30.0f;
        }
        mem[i].sampled = exact && jobs[i]->sp.temperature > 0;
        mem[i].masked = jobs[i]->masked();
      }
      dfl_->batch_budget()->plan(mem.data(), int(nj), dfl_->budget(), kv.data());
    } else if (e1_work) {
      e1->commit();
    }
    std::vector<Qwen35::VerifyGroup> groups;
    for (size_t i = 0; i < nj; ++i) {
      Slot& sl = *jobs[i]->slot;
      groups.push_back(Qwen35::VerifyGroup{sl.seq.get(), jobs[i]->P, kv[i] + 1, jobs[i]->sp, &sl.dseq->stage_hook,
                                           k > 0 ? 0 : sl.dseq->staged});
      // a masked member's masks: its anchor row's (alone when the budget gave it no draft) and its drafts' rows
      if (jobs[i]->masked()) {
        gmask[i] = 1;
        if (!pending) {
          if (!mask_event_) mask_event_ = g.new_shared_event();
          pending = ++mask_value_;
        }
        sl.seq->mask_rows = kv[i] + 1;
        sl.seq->mask_event = mask_event_;
        sl.seq->mask_value = pending;
      }
    }
    // (under the budget the GPU is idle until the pass's first layer is committed: it goes at once)
    q35_->encode_verify_multi(*e2, groups, kfb ? 1 : 0);
    for (BatchJob* j : jobs) j->slot->seq->mask_rows = 0, j->slot->seq->mask_event = nullptr;
    if (k > 0 && exact) {  // the sampled members' acceptance (stage-1 candidates per row: as the pass made them)
      uint32_t Kmax = 1;
      for (const auto& gr : groups)
        if (gr.sp.temperature > 0) Kmax = std::max(Kmax, uint32_t(std::clamp(gr.sp.top_k <= 0 ? 32 : gr.sp.top_k, 1, 32)));
      const uint32_t n = (uint32_t(model_->vocab()) + 2047) / 2048 * Kmax;
      int r0 = 0;
      for (size_t i = 0; i < nj; ++i) {
        BatchJob* j = jobs[i];
        if (j->sp.temperature > 0 && kv[i] > 0)
          dfl_->encode_accept_rows(*e2, *j->slot->seq, *j->slot->dseq, j->P, kv[i], j->sp, r0, n);
        r0 += kv[i] + 1;
      }
    }
  } else {
    std::vector<NemotronH::VerifyGroup> groups;
    for (BatchJob* j : jobs) {
      Slot& sl = *j->slot;
      LiliSeq& l = *sl.lseq;
      lili_->stage_init(l);
      if (l.staged > 0 && (k > 0 || l.staged == NhDrafter::kB)) lili_->encode_commit_staged(*e1, l), e1_work = true;
      if (k > 0) lili_->encode_draft(*e1, *sl.seq, l, j->P), e1_work = true;
      groups.push_back(NemotronH::VerifyGroup{sl.seq.get(), j->P, rows, j->sp, &l.stage_hook, k > 0 ? 0 : l.staged});
    }
    if (e1_work) e1->commit();
    nh_->encode_verify_multi(*e2, groups);  // (no member is masked: a Nemotron-H reply under a grammar is not batched)
  }
  e2->commit();
  if (pending) {
    if (e1_work && !kfb) e1->wait();  // (under the budget it was waited for before the plan)
    for (size_t i = 0; i < nj; ++i)
      if (gmask[i])
        verify_masks(*jobs[i]->gm, *jobs[i]->gst, *jobs[i]->slot->seq, jobs[i]->P, kv[i] + 1, nullptr, kv[i] + 1);
    Gpu::signal_event_cpu(mask_event_, pending);
  }
  e2->wait();
  bstat_gpu_ += e2->gpu_ms() + (e1_work ? e1->gpu_ms() : 0.0);
  if (e1_work && k > 0) {  // the GPU's idle time between the drafting passes and the verify
    bstat_gap_ += std::max(0.0, e2->gpu_start_host_ms() - e1->gpu_end_host_ms());
    ++bstat_drafted_;
  }
  const int* samp = dfl_ ? q35_->samples() : nh_->samples();
  std::vector<int> acc(nj, 0);
  int r0 = 0;
  for (size_t i = 0; i < nj; ++i) {
    BatchJob* j = jobs[i];
    Seq& s = *j->slot->seq;
    int* tk = s.tokens();
    const int kj = kv[i];
    int a = 0;  // drafts kept: while each equals the sample of the row before it
    if (dfl_ && kj > 0 && exact && j->sp.temperature > 0) {
      a = j->slot->dseq->accepted();  // (the acceptance wrote the token after them)
    } else {
      while (a < kj && samp[r0 + a] == tk[j->P + 1 + a]) ++a;
      tk[j->P + 1 + a] = samp[r0 + a];  // the sample after them
    }
    acc[i] = a;
    int m = 0;  // rows fed: the anchor and the kept drafts, up to where the request stops
    bool go = true;
    while (m < a + 1) {
      ++m;
      if (!(*j->handle)(tk[j->P + m])) {
        go = false;
        break;
      }
    }
    if (dfl_) {
      q35_->commit_rows(s, m);
      DFlashSeq& d = *j->slot->dseq;
      d.staged = k > 0 ? m : d.staged + 1;  // their captures: positions P .. P + m - 1
    } else {
      nh_->commit_rows(s, m);
      LiliSeq& l = *j->slot->lseq;
      l.staged = k > 0 ? m : l.staged + 1;
    }
    j->P += m;
    ++j->cycles;
    j->tokens += m;
    if (k > 0) {
      ++spec_cycles_;
      spec_accepted_ += m;
    }
    (*j->cycle_done)(m);
    if (j->trace) {
      const float* lg = dfl_ ? q35_->logits() : nh_->logits();
      (*j->trace)(lg + size_t(r0) * size_t(model_->vocab()), m, tk + j->P - m + 1);
    }
    if (!go) j->stopped = true;
    r0 += kj + 1;
    bstat_rows_ += kj + 1;
  }
  // the budget reads the cycle's outcome (each member's verified and accepted drafts) and its GPU time: the drafting
  // passes' command buffer and the verify's, as a single request's KFlash cycle is measured
  if (kfb) dfl_->batch_budget()->observe(mem.data(), int(nj), kv.data(), acc.data(), e1->gpu_ms() + e2->gpu_ms(),
                                         dfl_->budget());
  ++bstat_cycles_;
  bstat_wall_ += now_ms() - t_cycle;
  if (bstat_cycles_ >= 200) batch_stats();
}

// Each row is its member's anchor, kept whatever its sample (commit_rows(s, 1) as each pass is encoded); the pass after
// a member's last token was planned before the host read that token: its row is dropped (commit_rows(s, 0)), as the
// staging of its capture. Returns after 8 passes (a drafter's stage holds 8 rows), a member's stop, or when another
// request waits for the GPU; nothing is in flight then.
void Engine::batch_plain_run() {
  Gpu& g = Gpu::get();
  const double t_run = now_ms();
  const std::vector<BatchJob*> jobs = batch_;
  const size_t n = jobs.size();
  {  // the stages take up to 8 rows: earlier rows into the rings first
    auto e1 = g.begin();
    bool any = false;
    for (BatchJob* j : jobs) {
      if (dfl_) {
        DFlashSeq& d = *j->slot->dseq;
        dfl_->stage_init(d);
        if (d.staged > 0) dfl_->encode_commit_staged(*e1, d), any = true;
      } else {
        LiliSeq& l = *j->slot->lseq;
        lili_->stage_init(l);
        if (l.staged > 0) lili_->encode_commit_staged(*e1, l), any = true;
      }
    }
    if (any) e1->commit();
  }
  std::vector<int> base(n), stop_at(n, -1);
  for (size_t i = 0; i < n; ++i) base[i] = jobs[i]->P;
  // structured output: a pass's masked rows wait for their masks, which the host computes once the pass before it has
  // given each masked member's token (its state after it), one signal per pass
  uint64_t pv[8] = {};
  auto masks = [&](int c) {
    if (!pv[c]) return;
    for (size_t i = 0; i < n; ++i) {
      if (!jobs[i]->masked()) continue;
      Seq& sq = *jobs[i]->slot->seq;
      if (stop_at[i] < 0) verify_masks(*jobs[i]->gm, *jobs[i]->gst, sq, base[i] + c, 1, nullptr, 1);
      else memset(sq.mask.ptr(), 0xff, size_t(sq.mask_words) * 4);  // (a stopped member's row: not used)
    }
    Gpu::signal_event_cpu(mask_event_, pv[c]);
  };
  auto encode = [&](int c) {  // pass c: row base + c of every member, its capture to stage row c
    for (size_t i = 0; i < n; ++i)
      if (jobs[i]->masked()) {
        if (!pv[c]) {
          if (!mask_event_) mask_event_ = g.new_shared_event();
          pv[c] = ++mask_value_;
        }
        Seq& sq = *jobs[i]->slot->seq;
        sq.mask_rows = 1;
        sq.mask_event = mask_event_;
        sq.mask_value = pv[c];
      }
    auto e = g.begin_grouped();
    if (dfl_) {
      std::vector<Qwen35::VerifyGroup> gr;
      for (size_t i = 0; i < n; ++i) {
        Slot& sl = *jobs[i]->slot;
        Qwen35::VerifyGroup v{sl.seq.get(), base[i] + c, 1, jobs[i]->sp, &sl.dseq->stage_hook, c};
        v.tok_out = true;
        gr.push_back(v);
      }
      q35_->encode_verify_multi(*e, gr);
      for (BatchJob* j : jobs) q35_->commit_rows(*j->slot->seq, 1);
    } else {
      std::vector<NemotronH::VerifyGroup> gr;
      for (size_t i = 0; i < n; ++i) {
        Slot& sl = *jobs[i]->slot;
        NemotronH::VerifyGroup v{sl.seq.get(), base[i] + c, 1, jobs[i]->sp, &sl.lseq->stage_hook, c};
        v.tok_out = true;
        gr.push_back(v);
      }
      nh_->encode_verify_multi(*e, gr);
      for (BatchJob* j : jobs) nh_->commit_rows(*j->slot->seq, 1);
    }
    for (BatchJob* j : jobs) j->slot->seq->mask_rows = 0, j->slot->seq->mask_event = nullptr;
    e->commit();
    return e;
  };
  std::unique_ptr<Encoder> cur = encode(0);
  masks(0);
  int encoded = 1, c = 0;
  bool stop = false;
  for (;;) {
    std::unique_ptr<Encoder> next;
    if (!stop && encoded < 8 && !contended()) next = encode(encoded++);
    cur->wait();
    bstat_gpu_ += cur->gpu_ms();
    for (size_t i = 0; i < n; ++i) {  // pass c's samples: each member's token after row base + c
      if (stop_at[i] >= 0) continue;
      BatchJob* j = jobs[i];
      const int P = base[i] + c;
      j->P = P + 1;
      ++j->cycles;
      ++j->tokens;
      if (dfl_) j->slot->dseq->staged = c + 1;
      else j->slot->lseq->staged = c + 1;
      (*j->cycle_done)(1);
      if (!(*j->handle)(j->slot->seq->tokens()[P + 1])) {
        j->stopped = true;
        stop_at[i] = c;
        stop = true;
      }
    }
    ++c;
    ++bstat_cycles_;
    bstat_rows_ += long(n);
    if (next) masks(c);  // pass c's (queued): the members' states after the tokens pass c - 1 gave
    cur = std::move(next);
    if (!cur) break;
  }
  for (size_t i = 0; i < n; ++i)  // a stopped member's row in the pass after its stop: not wanted
    if (stop_at[i] >= 0 && stop_at[i] < encoded - 1) {
      if (dfl_) q35_->commit_rows(*jobs[i]->slot->seq, 0);
      else nh_->commit_rows(*jobs[i]->slot->seq, 0);
    }
  bstat_wall_ += now_ms() - t_run;
  if (bstat_cycles_ >= 200) batch_stats();
}

void Engine::batch_stats() {  // (debug log) how the batch's time splits
  if (bstat_cycles_ <= 0) return;
  logd("batch: %ld cycles, %.1f rows each, %.2f ms per cycle (GPU %.2f; idle before the verify %.3f)", bstat_cycles_,
       double(bstat_rows_) / double(bstat_cycles_), bstat_wall_ / double(bstat_cycles_), bstat_gpu_ / double(bstat_cycles_),
       bstat_drafted_ ? bstat_gap_ / double(bstat_drafted_) : 0.0);
  bstat_cycles_ = bstat_rows_ = bstat_drafted_ = 0;
  bstat_wall_ = bstat_gpu_ = bstat_gap_ = 0;
}

bool Engine::make_room(Slot& keep, size_t bytes) {
  Gpu& g = Gpu::get();
  auto fits = [&] { return g.allocated_bytes() + bytes <= budget_; };
  if (fits()) return true;
  std::vector<Slot*> others;  // idle slots
  for (auto& o : slots_)
    if (&o != &keep && !o.busy && o.seq->cap > 0) others.push_back(&o);
  std::sort(others.begin(), others.end(), [](const Slot* a, const Slot* b) { return a->last_used < b->last_used; });
  for (Slot* o : others) {
    logi("memory: dropping the KV cache of slot %d (%d tokens cached) for an image", int(o - slots_.data()),
         int(o->tokens.size()));
    disk_leaving(*o, 0);
    drop_slot(*o);
    if (fits()) return true;
  }
  if (model_->ane_release()) {
    if (ane_ck_) {
      for (auto* v : {&ane_ck_->conv, &ane_ck_->rec})
        for (auto& b : *v) Gpu::get().release(b);
      ane_ck_.reset();
    }
    if (fits()) return true;
  }
  return false;
}

size_t Engine::common_rows(const Slot& slot, const std::vector<int>& keys, const vision::Prompt* mm) {
  if (mm) return vision::common_prefix(slot.tokens, slot.img_spans, slot.img_digests, *mm);
  const size_t n = std::min(slot.tokens.size(), keys.size());
  size_t i = 0;
  while (i < n && slot.tokens[i] == keys[i]) ++i;
  return i;
}

// Image and video parts become a short note (a model without an image encoder, or render_chat's callers that do not
// handle images).
static ojson without_images(const ojson& messages, int* dropped) {
  ojson out = messages;
  for (auto& m : out) {
    if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
    for (auto& p : m["content"]) {
      if (!p.is_object() || (p.contains("text") && p["text"].is_string())) continue;
      const std::string t = p.contains("type") && p["type"].is_string() ? p["type"].get<std::string>() : "";
      if (t.find("image") != std::string::npos || t.find("video") != std::string::npos || p.contains("image") ||
          p.contains("image_url") || p.contains("video")) {
        p = {{"type", "text"}, {"text", "\n[image omitted: this model can't read images]\n"}};
        ++*dropped;
      }
    }
  }
  return out;
}

// The text of a message's content (a string, or the text of its parts), trimmed as the templates trim it.
static std::string content_text(const ojson& c) {
  std::string s;
  if (c.is_string()) s = c.get<std::string>();
  else if (c.is_array())
    for (auto& p : c)
      if (p.is_object() && p.contains("text") && p["text"].is_string()) s += p["text"].get<std::string>();
  const size_t a = s.find_first_not_of(" \n\r\t"), b = s.find_last_not_of(" \n\r\t");
  return a == std::string::npos ? "" : s.substr(a, b + 1 - a);
}

std::string chat_template_source(const std::string& model_dir) {
  std::string src = read_file(model_dir + "/chat_template.jinja");
  if (src.empty()) {
    auto cfg = nlohmann::json::parse(read_file(model_dir + "/tokenizer_config.json"), nullptr, false);
    if (cfg.is_object() && cfg.contains("chat_template") && cfg["chat_template"].is_string()) src = cfg["chat_template"];
  }
  return src;
}

// (the template's own enable_thinking, ns.enable_thinking or namespace(enable_thinking=...), is not the kwarg)
bool think_flag_template(const std::string& src) {
  if (src.find("/no_think") == std::string::npos) return false;
  for (size_t at = src.find("enable_thinking"); at != std::string::npos; at = src.find("enable_thinking", at + 1))
    if (at == 0 || (src[at - 1] != '.' && src[at - 1] != '(')) return false;  // reads the kwarg
  return true;
}

// Thinking is on unless a system or user message says "/no_think" (the last flag wins), and the template strips the
// flags from the text it renders: enable_thinking becomes the flag after the last user message's text
ChatRequest think_flag_request(const ChatRequest& r) {
  const auto it = r.template_kwargs.find("enable_thinking");
  if (it == r.template_kwargs.end() || !it->is_boolean()) return r;
  ChatRequest q = r;
  const std::string flag = it->get<bool>() ? "/think" : "/no_think";
  ojson* target = nullptr;
  for (auto& m : q.messages)
    if (role_of(m) == "user") target = &m;
  if (!target && !q.messages.empty() && role_of(q.messages[0]) == "system") target = &q.messages[0];
  if (target) {
    ojson& c = (*target)["content"];
    if (c.is_array()) c.push_back({{"type", "text"}, {"text", flag}});
    else c = (c.is_string() ? c.get<std::string>() : content_text(c)) + "\n" + flag;
  }
  return q;
}

TextTags text_tags_for(const Tokenizer& tok, const std::string& src) {
  TextTags t;
  if (tok.find("<think>") < 0 && tok.find("</think>") < 0 && src.find("<think>") != std::string::npos) t.think = true;
  if (tok.find("<tool_call>") < 0 && src.find("<TOOLCALL>") != std::string::npos) {
    t.tc_open = "<TOOLCALL>";
    t.tc_close = "</TOOLCALL>";
    t.json_calls = true;
  }
  return t;
}

std::string Engine::render(const ChatRequest& r, bool add_generation_prompt) const {
  const bool images = vision_ != nullptr;  // image parts render as the model's placeholder, else as a note
  // Nemotron Nano 9B v2's template: the request's enable_thinking (every API's switch) as its "/think" / "/no_think"
  if (think_flags_ && r.template_kwargs.contains("enable_thinking"))
    return render_chat(*tmpl_, think_flag_request(r), add_generation_prompt, images);
  if (nh_ || dory_) {
    // Nemotron 3.5's (and Dory's) template names two switches its own way: medium_effort (shorter reasoning) for the low and medium
    // reasoning efforts, and truncate_history_thinking = false for preserve_thinking (every turn's reasoning kept, as
    // the Anthropic API's keep-all asks: the next request's prompt then continues what was generated)
    const auto& k = r.template_kwargs;
    const bool effort = k.contains("reasoning_effort") && k["reasoning_effort"].is_string() &&
                        (k["reasoning_effort"] == "low" || k["reasoning_effort"] == "medium") && !k.contains("medium_effort");
    const bool keep = k.contains("preserve_thinking") && k["preserve_thinking"] == true &&
                      !k.contains("truncate_history_thinking");
    if (effort || keep) {
      ChatRequest q = r;
      if (effort) q.template_kwargs["medium_effort"] = true;
      if (keep) q.template_kwargs["truncate_history_thinking"] = false;
      return render_chat(*tmpl_, q, add_generation_prompt, images);
    }
  }
  return render_chat(*tmpl_, r, add_generation_prompt, images, flint_ != nullptr);
}

// Flint's chat template is written for Jinja2, and minja (third_party, not edited) differs from it in three places the
// template uses: no replace filter (strings have the replace method), "\x00" read as the text "x00" (the template's
// removal of NUL characters would delete that text from every message instead; dropped), and no ensure_ascii keyword
// on tojson (whose output keeps non-ASCII text as ensure_ascii=False does). The source rewritten for minja.
std::string flint_template(std::string src) {
  auto replace_all = [&](const std::string& a, const std::string& b) {
    for (size_t at = 0; (at = src.find(a, at)) != std::string::npos; at += b.size()) src.replace(at, a.size(), b);
  };
  replace_all("tojson(ensure_ascii=False)", "tojson");
  replace_all(" | replace(\"\\x00\", \"\")", "");
  replace_all(" | replace(", ".replace(");
  return src;
}

// Whether a chat template reads message content as a string only (Dory's: a content array would render as its repr;
// Nemotron Nano 9B v2's calls string methods on it, which fail on an array): a user message of one text part, rendered,
// must show the text and nothing of the part's markup. Cached per template.
static bool string_content_template(const minja::chat_template& tmpl) {
  static std::mutex mu;
  static std::map<const void*, bool> known;
  std::lock_guard<std::mutex> lk(mu);
  if (auto it = known.find(&tmpl); it != known.end()) return it->second;
  static const std::string needle = "\x01kapitsa-typed-content\x01";
  bool str_only = false;
  try {
    minja::chat_template_inputs in;
    in.messages = ojson::array({{{"role", "user"}, {"content", ojson::array({{{"type", "text"}, {"text", needle}}})}}});
    in.add_generation_prompt = false;
    minja::chat_template_options opt;
    opt.apply_polyfills = false;
    const std::string out = tmpl.apply(in, opt);
    str_only = out.find(needle) == std::string::npos || out.find("'type'") != std::string::npos ||
               out.find("\"type\"") != std::string::npos;
  } catch (...) {
    str_only = true;  // (a template that cannot render a content array at all)
  }
  known[&tmpl] = str_only;
  return str_only;
}

// A content array as one string: its text parts joined by newlines (vLLM's "string" content format)
static ojson flatten_content(const ojson& messages) {
  ojson out = messages;
  for (auto& m : out) {
    if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
    std::string s;
    bool first = true;
    for (const auto& p : m["content"])
      if (p.is_object() && p.contains("text") && p["text"].is_string()) {
        if (!first) s += "\n";
        s += p["text"].get<std::string>();
        first = false;
      }
    m["content"] = s;
  }
  return out;
}

std::string render_chat(const minja::chat_template& tmpl, const ChatRequest& r, bool add_generation_prompt,
                        bool images, bool native) {
  minja::chat_template_inputs in;
  int dropped = 0;
  in.messages = images ? vision::without_image_data(r.messages) : without_images(r.messages, &dropped);
  if (!images && string_content_template(tmpl)) in.messages = flatten_content(in.messages);
  if (dropped && add_generation_prompt) logi("chat: %d image(s) left out: this model can't read images", dropped);
  // A system message after the first message (an agent's hook output, a mode change) is a system turn where it
  // stands: the Qwen3.5+ templates raise on it. It goes through the template as a marker tool response (not a user
  // query, so which assistant turns keep their reasoning does not change), and the marker's turn is then replaced.
  std::vector<std::string> later;
  static const std::string mark = "\x02kapitsa-system-turn-";
  for (size_t i = 1; i < in.messages.size() && !native; ++i) {
    ojson& m = in.messages[i];
    if (role_of(m) != "system") continue;
    later.push_back(content_text(m.contains("content") ? m["content"] : ojson()));
    m = {{"role", "user"}, {"content", "<tool_response>" + mark + std::to_string(later.size() - 1) + "\x02</tool_response>"}};
  }
  in.tools = r.tools.empty() ? ojson() : r.tools;
  in.add_generation_prompt = add_generation_prompt;
  in.extra_context = r.template_kwargs;
  minja::chat_template_options opt;
  opt.apply_polyfills = !native;
  std::string text = tmpl.apply(in, opt);
  for (size_t k = 0; k < later.size(); ++k) {
    const std::string inner = "<tool_response>" + mark + std::to_string(k) + "\x02</tool_response>";
    const std::string turn = "<|im_start|>user\n" + inner + "<|im_end|>\n";
    if (size_t at = text.find(turn); at != std::string::npos)
      text.replace(at, turn.size(), "<|im_start|>system\n" + later[k] + "<|im_end|>\n");
    else if (size_t at2 = text.find(inner); at2 != std::string::npos)  // another template's markup: its text in place
      text.replace(at2, inner.size(), later[k]);
  }
  return text;
}

// The request's images, prepared (fetched, decoded, resized; data: URLs remembered by content), for a model with an
// image encoder (without one, render() turns image parts into a note). Throws vision::Error (video parts too).
static std::vector<vision::Prepared> request_images(vision::Vision* vis, std::vector<vision::Source>* src,
                                                   const ojson& messages) {
  if (!vis) return {};
  *src = vision::image_sources(messages);
  return src->empty() ? std::vector<vision::Prepared>() : vis->prepare(*src);
}

vision::Prompt Engine::expand_images(const std::vector<int>& ids, const std::vector<vision::Prepared>& images,
                                     int* index) const {
  return vision::expand(ids, vision_->encoder().placeholder(), images, index, img_pre_, img_post_);
}

int Engine::count_tokens(const ChatRequest& r) const {
  std::vector<vision::Source> src;
  const std::vector<vision::Prepared> images = request_images(vision_.get(), &src, r.messages);
  const std::vector<int> ids = tok_.encode(render(r, true));
  if (images.empty()) return int(ids.size());
  return int(expand_images(ids, images).ids.size());  // the images' rows
}

Engine::Slot* Engine::pick_slot(const std::vector<int>& prompt, uint64_t arrival, const vision::Prompt* mm, int loops) {
  // (a looped model's cached state is only valid for the loop count that computed it)
  auto lcp = [&](const Slot& s) { return dory_ && s.loops != loops ? size_t(0) : common_rows(s, prompt, mm); };
  // What a slot can actually reuse: its live state if the prompt extends it, else its newest
  // checkpoint inside the common prefix. A shared prefix without a usable state is worthless
  // and choosing that slot would only destroy its conversation.
  auto usable = [&](const Slot& s) -> int {
    size_t l = lcp(s);
    if (l == s.tokens.size() && l < prompt.size()) return int(l);
    int u = 0;
    for (auto& c : s.ckpts)
      if (!c.prefix && c.pos > u && c.pos <= int(l) && c.pos < int(prompt.size())) u = c.pos;
    return u;
  };
  // A shared system prompt's checkpoint is no reason to take a slot from its conversation (the main agent's, while a
  // subagent starts): it only breaks ties among the slots a new conversation may take anyway.
  auto has_prefix = [&](const Slot& s) {
    size_t l = lcp(s);
    for (auto& c : s.ckpts)
      if (c.prefix && c.pos >= 64 && c.pos <= int(l) && c.pos < int(prompt.size())) return true;
    return false;
  };
  Slot* best = nullptr;
  int best_lcp = 0;
  for (auto& s : slots_) {
    if (s.busy || int(prompt.size()) + 64 > s.seq->limit) continue;
    int u = usable(s);
    if (!best || u > best_lcp) {
      best = &s;
      best_lcp = u;
    }
  }
  if (!best) {  // a prompt within 64 tokens of the context: the first free slot
    for (auto& s : slots_)
      if (!s.busy) return &s;
    return nullptr;
  }
  if (best_lcp < 64) {
    // Nothing worth reusing: a new conversation. It takes an empty slot, else one holding a one-off
    // (a conversation never continued, like an agent's title or commit-message request), and only
    // then a continued conversation; least recently used first within each. So one-offs never evict
    // the conversation in progress, however long an older one-off was, and a new conversation
    // replaces the conversation idle longest. A conversation whose request was still running when this
    // one arrived counts as continued: an agent sends its title request beside its first request
    // (opencode does), and that first conversation goes on with the tools' results.
    auto rank = [&](const Slot& s) { return s.tokens.empty() ? 0 : s.turns <= 1 && arrival > s.done_at ? 1 : 2; };
    Slot* pick = nullptr;
    for (auto& s : slots_) {
      if (s.busy || int(prompt.size()) + 64 > s.seq->limit) continue;
      if (!pick || rank(s) < rank(*pick)) {
        pick = &s;
      } else if (rank(s) == rank(*pick)) {
        const bool hp = has_prefix(s), pp = has_prefix(*pick);
        if (hp != pp ? hp : s.last_used < pick->last_used) pick = &s;
      }
    }
    if (pick) {
      pick->turns = 1;
      return pick;
    }
  }
  ++best->turns;
  return best;
}

ChatResult Engine::chat(const ChatRequest& r, const std::function<bool(const ChatDelta&)>& on_delta,
                        const std::function<void(const ChatResult&)>& on_prompt,
                        const std::function<bool()>& cancelled,
                        const std::function<bool(const ChatResult&)>& on_accepted) {
  const uint64_t arrival = ++requests_;
  // images: fetched, decoded and resized before the GPU turn (a slow download holds up no other request)
  std::vector<vision::Source> img_src;
  std::vector<vision::Prepared> images;
  const double t_img = now_ms();
  try {
    images = request_images(vision_.get(), &img_src, r.messages);
  } catch (const std::exception& ex) {  // vision::Error: the client's to fix; anything else: say what it was
    ChatResult res;
    res.finish_reason = "error";
    res.invalid_request = dynamic_cast<const vision::Error*>(&ex) != nullptr;
    res.error = res.invalid_request ? ex.what() : std::string("images: ") + ex.what();
    return res;
  }
  const double img_prep_ms = now_ms() - t_img;
  // the prompt: rendered and tokenized before the GPU turn too (CPU only: a request in flight decodes meanwhile)
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
  // images: each placeholder becomes the image's rows; prefix reuse compares keys that tell images apart by content
  vision::Prompt mm;
  const vision::Prompt* mmp = nullptr;
  if (!images.empty()) {
    try {
      mm = expand_images(prompt, images, &boundary);
    } catch (const vision::Error& ex) {
      res.finish_reason = "error";
      res.invalid_request = true;
      res.error = ex.what();
      return res;
    }
    prompt = mm.ids;
    mmp = &mm;
  }
  const std::vector<int>& keys = mmp ? mm.keys : prompt;
  const int n = int(prompt.size());
  res.prompt_tokens = n;
  // The prefix checkpoint's row: the end of the system block (where the first user turn starts) when it is at least 64
  // rows in and a strict prefix of the prompt. A new question after the same system prompt and tools (a subagent's
  // task, an agent's next conversation) then reuses the block however early it diverges.
  int pbound = -1;
  if (prefix_ckpt_ && images.empty()) {
    const size_t off = text.find(flint_ ? "<|user|>" : "<|im_start|>user");
    if (off != std::string::npos && off > 0) {
      const std::vector<int> pt = tok_.encode(text.substr(0, off));
      if (pt.size() >= 64 && int(pt.size()) < n && (boundary < 0 || int(pt.size()) < boundary) &&
          std::equal(pt.begin(), pt.end(), prompt.begin()))
        pbound = int(pt.size());
    }
  }
  const int loops = request_loops(r);  // looped models (0 otherwise)
  res.loops = loops;
  if (n + 1 >= opts_.context) {  // every slot holds opts_.context positions
    res.finish_reason = "error";
    res.error = "prompt (" + std::to_string(n) + " tokens) exceeds the context window (" + std::to_string(opts_.context) + ")";
    res.prompt_limit = opts_.context - 2;
    return res;
  }
  // the reply starts inside the reasoning (the prompt ends with the think tag)
  const bool think = !text.empty() && text.size() >= 8 && text.compare(text.size() - 8, 8, "<think>\n") == 0;
  // Structured output: the grammar's matcher on this model's vocabulary (the vocabulary's trie is built by the first
  // such request, here, before the GPU turn). The grammar holds the reply from its first token, or after </think>.
  std::unique_ptr<GrammarMatcher> gm;
  if (r.grammar) {
    std::string why;
    if (grammar_calls(*r.grammar) && (tc_open_ < 0 || tc_close_ < 0))
      why = "this model has no <tool_call> / </tool_call> tokens: its replies cannot be held to tool calls";
    else if (grammar_calls(*r.grammar) && flint_)
      why = "Flint writes its tool calls in GLM's markup, which the tool-call grammar does not hold yet: send "
            "tool_choice \"auto\" and non-strict tools";
    else if (think && think_close_ < 0)
      why = "this model writes its reasoning's end as text: structured output needs reasoning off with it";
    if (!why.empty()) {
      res.finish_reason = "error";
      res.invalid_request = true;
      res.error = why;
      return res;
    }
    gm = std::make_unique<GrammarMatcher>(r.grammar, token_table());
  }
  const double tok_wall = now_ms() - t_req;  // template + tokenizer
  if (on_accepted && !on_accepted(res)) {  // e.g. a stream whose client is gone before its first event
    res.finish_reason = "cancelled";
    return res;
  }

  // ---- the GPU turn, then a free slot (first come first served when every slot is busy) ----
  std::unique_lock<TurnLock> turn(turn_);
  // the image embeddings this request holds (Vision::embeddings): given back once it is done
  struct EndImages {
    vision::Vision* v;
    std::vector<vision::Digest> used;
    ~EndImages() {
      if (v && !used.empty()) v->end_request(used);
    }
  } end_images{vision_.get(), {}};
  // NI_ANE_ASYNC: switch the prefill FFN split on once its program is compiled (with the turn: it runs a startup check
  // through the model's buffers, and no request in flight keeps anything there between its steps)
  model_->ane_poll();
  const double t_slot = now_ms();
  Slot* picked = nullptr;
  bool queued = false;
  for (;;) {
    if (cancelled && cancelled()) {  // before a slot is picked: a client gone while its request waited takes none
      if (queued) {
        slot_queue_.erase(std::find(slot_queue_.begin(), slot_queue_.end(), arrival));
        turn_.notify();
      }
      res.finish_reason = "cancelled";
      return res;
    }
    if (slot_queue_.empty() || slot_queue_.front() == arrival) picked = pick_slot(keys, arrival, mmp, loops);
    if (picked) break;
    if (!queued) {
      slot_queue_.push_back(arrival);
      queued = true;
      logi("chat: all %d slots are busy: a request waits for one (%zu waiting; --slots sets how many requests run "
           "at once)", int(slots_.size()), slot_queue_.size());
    }
    turn_.wait_event(turn_.events(), cancelled);  // a request ending frees its slot
  }
  if (queued) {
    slot_queue_.erase(std::find(slot_queue_.begin(), slot_queue_.end(), arrival));
    turn_.notify();  // the next one in line may find another slot free
    res.queue_ms = now_ms() - t_slot;
  }
  Slot& slot = *picked;
  // the slot is this request's until it returns: no other request picks it, and its KV cache goes to another request
  // only while this one waits for memory before its prefill (reserve_kv)
  struct SlotUse {
    Engine& en;
    Slot& s;
    ~SlotUse() {
      s.busy = false;
      s.mem_wait = Slot::MemWait::none;
      --en.active_;
      en.turn_.notify();
    }
  };
  slot.busy = true;
  slot.arrival = arrival;
  ++active_;
  SlotUse slot_use{*this, slot};
  const double t_turn = now_ms();
  Seq& s = *slot.seq;
  slot.last_used = ++clock_;
  ++slot.gen;

  // ---- prefix reuse ----
  // (a looped model's KV holds the loop count that computed it: another count reads the prompt from the start)
  size_t common = dory_ && slot.loops != loops ? 0 : common_rows(slot, keys, mmp);
  if (dory_ && slot.loops != loops && !slot.tokens.empty())
    logi("chat: %d loops on slot %d, whose cache holds %d-loop KV: the prompt is read from the start", loops,
         int(&slot - slots_.data()), slot.loops);
  // SSD prefix cache (engine_disk.cpp): a stored state deeper than the slot's own is read into it once its KV cache is
  // reserved (below); the conversation the slot held goes to the SSD first, if the SSD lacks it and this request
  // leaves it behind (another conversation; with the loop count it was computed with)
  pcache::Hit disk_hit;
  if (disk_) {
    const int ram = reuse_start(slot, common, n);
    disk_hit = disk_lookup(keys, n, ram, dory_ ? uint32_t(loops) : 0, mmp);
    disk_leaving(slot, disk_hit ? 0 : ram);
    if (disk_hit) {  // the slot's own state is not reused
      common = 0;
      slot.tokens.clear();
      slot.fills.clear();
      for (auto& c : slot.ckpts) c.pos = -1;
    }
  }
  ++slot.version;
  if (dory_) slot.loops = s.loops = loops;

  // ---- images: the embeddings of those in the rows to prefill (from the live state's end or the newest usable
  // checkpoint, as below), before any state changes, so a failure leaves the slot as it was ----
  std::vector<vision::EmbedRun> runs;
  int img_encoded = 0, img_reused = 0;  // images encoded now, from the embeddings of earlier requests
  double img_ms = 0;
  auto image_runs = [&](int from) {
    const double t0 = now_ms();
    for (size_t k = 0; k < mm.spans.size(); ++k) {
      const vision::ImageSpan& sp = mm.spans[k];
      if (sp.pos + sp.n <= from) continue;
      bool have = false;
      for (const vision::EmbedRun& run : runs) have = have || run.pos == sp.pos;
      if (have) continue;
      bool encoded = false;
      const Buffer emb =
          vision_->embeddings(images[k], img_src[k], [&](size_t b) { return make_room(slot, b); }, &encoded);
      end_images.used.push_back(images[k].digest);
      runs.push_back(vision::EmbedRun{sp.pos, sp.n, emb, 0, vision_->encoder().out_dim()});
      ++(encoded ? img_encoded : img_reused);
    }
    img_ms += now_ms() - t0;
    // what one conversation can hold now that the encoder's weights take their share of the budget (/v1/models)
    const size_t fixed = fixed_bytes_ + (vision_->encoder().loaded() ? vision_->encoder().weight_bytes() : 0);
    context_fit_ = int(std::min<size_t>(size_t(opts_.context), (budget_ > fixed ? budget_ - fixed : 0) /
                                                                   model_->kv_bytes_per_pos()));
  };
  if (mmp) {
    int from = 0;
    if (disk_hit) from = disk_hit.pos;  // (the images before it come with the state read from the SSD)
    else if (common == slot.tokens.size() && int(common) < n) from = int(common);
    else
      for (auto& c : slot.ckpts)
        if (c.pos > 0 && c.pos <= int(common) && c.pos < n) from = std::max(from, c.pos);
    try {
      image_runs(from);
    } catch (const vision::Error& ex) {
      res.finish_reason = "error";
      res.error = ex.what();
      return res;
    }
  }

  // ---- KV cache for the prompt and the first part of the answer (it grows further while decoding) ----
  // other slots' caches are given up only when the prompt itself does not fit: before them this slot's own, when
  // the prompt reuses none of it (no live state to continue, no checkpoint inside the common prefix: the prefill
  // starts at 0), so the new cache starts empty: nothing to copy, no transient second copy. Requests in flight keep
  // theirs: this one waits for them to end when only their memory would make room (while it waits, a request that
  // has started, or an earlier one, may take this slot's cache: the prefill then starts at 0, as below).
  const int kv_min = std::min(s.limit, n + 32), kv_want = std::min(s.limit, n + std::clamp(r.max_tokens, 0, 1024) + 32);
  auto fit_kv = [&] { return model_->reserve(s, kv_want) || model_->reserve(s, kv_min); };
  bool kv_ok = fit_kv();
  bool reuses = common > 0 && common == slot.tokens.size() && int(common) < n;
  for (auto& c : slot.ckpts) reuses = reuses || (c.pos > 0 && c.pos <= int(common) && c.pos < n);
  if (!kv_ok && !reuses && s.cap > 0) {
    drop_slot(slot);
    kv_ok = fit_kv();
  }
  if (!kv_ok && !reserve_kv_wait(slot, kv_min, cancelled, /*started=*/false)) {
    if (cancelled && cancelled()) {
      res.finish_reason = "cancelled";
      return res;
    }
    const size_t per = model_->kv_bytes_per_pos(), used = g.allocated_bytes();
    const size_t fit = size_t(s.cap) + (budget_ > used ? budget_ - used : 0) / per;
    res.finish_reason = "error";
    res.error = "insufficient memory: a " + std::to_string(n) + "-token prompt needs more KV cache than this Mac's "
                "memory budget for Kapitsa holds with this model (" + std::to_string(fit) + " tokens)";
    res.prompt_limit = std::max(0, int(fit) - 32);
    return res;
  }
  double disk_ms = 0;  // (the SSD read: its own log line, not the template's time)
  if (disk_hit) {
    const double t_disk = now_ms();
    if (disk_restore(slot, disk_hit)) common = slot.tokens.size();  // it is the live state now
    else model_->reset(s);                                         // (no tokens: the prefill starts at 0)
    disk_ms = now_ms() - t_disk;
  }
  int start = 0;
  auto e = g.begin();
  if (common > 0 && common == slot.tokens.size() && int(common) < n) {
    start = int(common);  // live state continues (an empty slot has none: it may copy a prefix checkpoint below)
  } else {
    const TurnCheckpoint* best = nullptr;
    for (auto& c : slot.ckpts)
      if (c.pos > 0 && c.pos <= int(common) && c.pos < n && (!best || c.pos > best->pos)) best = &c;
    // A prefix checkpoint another idle slot holds beyond what this slot can restore: its attention cache rows
    // [0, pos) are copied here and its recurrent state restored, then saved as this slot's own prefix checkpoint
    // (a new conversation takes an empty slot, but the shared system prompt need not be read again).
    TurnCheckpoint* donor_ck = nullptr;
    Slot* donor = nullptr;
    if (prefix_ckpt_ && prefix_copy_ && !mmp)
      for (auto& o : slots_) {
        if (&o == &slot || o.busy || o.seq->cap == 0) continue;
        const size_t l = common_rows(o, keys, nullptr);
        for (auto& c : o.ckpts)
          if (c.prefix && c.pos >= 64 && c.pos <= int(l) && c.pos < n && c.pos <= o.seq->cap && c.pos <= s.cap &&
              c.pos > (best ? best->pos : 0) && (!donor_ck || c.pos > donor_ck->pos))
            donor_ck = &c, donor = &o;
      }
    if (donor_ck && model_->copy_prefix(*e, *donor->seq, s, donor_ck->pos)) {
      model_->restore(*e, s, *donor_ck);
      for (auto& c : slot.ckpts) {
        c.pos = -1;
        c.query = false;
      }
      for (auto& c : slot.ckpts)
        if (c.prefix) model_->save(*e, s, c, donor_ck->pos);
      start = donor_ck->pos;
      if (dfl_) DFlash2::reset(*slot.dseq);
      logd("checkpoints: copied the prefix checkpoint at %d from slot %d to slot %d", start,
           int(donor - slots_.data()), int(&slot - slots_.data()));
    } else if (best) {
      model_->restore(*e, s, *best);
      start = best->pos;
      // the drafter ring as it was at that checkpoint (the live ring may have overwritten its window since); the
      // prefix checkpoint has none: the ring starts empty (rewind below marks [0, start) unusable)
      if (dfl_ && best->prefix) DFlash2::reset(*slot.dseq);
      else if (dfl_) dfl_->encode_ring_copy(*e, *slot.dckpts[size_t(best - slot.ckpts.data())], *slot.dseq);
      if (lili_) lili_->encode_copy(*e, *slot.lckpts[size_t(best - slot.ckpts.data())], *slot.lseq);
    } else {
      model_->reset(s);
    }
    logd("checkpoints: prompt diverges at %zu of %d, %s", common, n,
         start > 0 && (!best || start != best->pos) ? "copied from another slot"
         : best ? (std::string("restored ") + (best->prefix ? "prefix" : best->query ? "query" : "tool") + " checkpoint at " +
                 std::to_string(best->pos)).c_str()
              : "none usable, prefill from 0");
  }
  if (dfl_) {
    // the drafter ring keeps the entries of the reused prefix [0, start) that are still in its window
    if (start == 0) DFlash2::reset(*slot.dseq);
    else DFlash2::rewind(*slot.dseq, start);
  }
  if (lili_) {
    if (start == 0) NhDrafter::reset(*slot.lseq);
    else NhDrafter::rewind(*slot.lseq, start);
  }
  for (auto& c : slot.ckpts)
    if (c.pos > start) c.pos = -1;
  res.cached_tokens = start;
  LiveDecode* const live = r.live.get();  // progress for a display: relaxed stores, between cycles only
  if (live) live->prompt.store(n, std::memory_order_relaxed), live->cached.store(start, std::memory_order_relaxed),
      live->read.store(start, std::memory_order_relaxed), live->phase.store(1, std::memory_order_relaxed);
  if (on_prompt) on_prompt(res);
  std::copy(prompt.begin(), prompt.end(), s.tokens());
  // images: their positions (M-RoPE; none: the text-only path) and the rows that take their embeddings
  model_->set_rope_positions(s, n, mm.spans);
  if (mmp) {
    try {
      image_runs(start);  // normally nothing left: the rows were known before
    } catch (const vision::Error& ex) {
      drop_slot(slot);  // the state is no longer what slot.tokens says
      res.finish_reason = "error";
      res.error = ex.what();
      return res;
    }
  }
  s.embeds = runs.empty() ? nullptr : &runs;

  // ---- prefill ----
  // template + tokenizer (before the turn) + slot pick (image encoding and an SSD read: logged apart)
  const double tok_ms = tok_wall + (now_ms() - t_turn) - img_ms - disk_ms;
  const AneStats ane0 = model_->ane_stats();  // the Neural Engine's busy time during this prefill (log)
  SamplingParams sp;
  sp.temperature = r.temperature;
  sp.top_p = r.top_p;
  sp.top_k = r.top_k;
  sp.seed = r.seed ? r.seed : uint32_t(std::random_device{}());
  // Structured output: the matcher's state. A reply held from its first token samples that token (in the prompt's last
  // chunk) under the grammar's first mask; with reasoning on the grammar starts after </think>.
  GrammarState gst;
  bool g_on = false;
  if (gm) {
    gst = gm->start();
    g_on = !think;
    if (!s.mask) {
      s.mask_words = gm->table()->words();
      s.mask = g.alloc(size_t(kf::kMaxRows) * size_t(s.mask_words) * 4, "grammar_mask");
    }
    if (g_on) {
      gm->fill_mask(gst, s.mask.as<uint32_t>());
      s.mask_rows = 1;
      s.mask_event = nullptr;
    }
  }
  struct MaskOff {  // no later pass of the slot reads a mask of this request's
    Seq& s;
    ~MaskOff() {
      s.mask_rows = 0;
      s.mask_event = nullptr;
    }
  } mask_off{s};
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
  // the prefix checkpoint, saved while this prompt is read through pbound (not when the prefill starts at or past
  // it: the checkpoint then holds it already, or the state at pbound is gone)
  auto prefix_slot = [&]() {
    TurnCheckpoint* c = nullptr;
    for (auto& k : slot.ckpts)
      if (k.prefix) c = &k;
    logd("checkpoints: prefix boundary at %d", pbound);
    return c;
  };
  auto victim = [&]() {
    auto rank = [&](const TurnCheckpoint& k) { return k.pos < 0 ? 0 : keep_query && k.query ? 2 : 1; };
    TurnCheckpoint* c = nullptr;
    for (auto& k : slot.ckpts)
      if (!k.prefix && (!c || rank(k) < rank(*c) || (rank(k) == rank(*c) && k.pos < c->pos))) c = &k;
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
  if (lili_) lili_->begin_prefill(n, start);
  // Other requests take their turns between chunks up to the first row the drafter captures (its capture buffer is
  // shared); from there the prefill runs to its end (at most two chunks).
  const int capture_from = dfl_ ? std::max(start, n - DFlash2::kCtxKeep)
                         : lili_ ? std::max(start, n - lili_->window())
                                 : n;
  double t_enc = 0, h_commit = 0;  // the last chunk's commit: ms since t0, host clock
  // The last two chunks' command buffers (committed): at most two chunks run ahead of the host, so a request that
  // arrives during a long prefill is seen within a chunk (encoding is far faster than the GPU: a 20K prompt's chunks
  // would all be queued in milliseconds). The GPU still always has the next chunk queued.
  std::unique_ptr<Encoder> last, last2;
  while (p < n) {
    if (last2) {
      last2->wait();
      last2.reset();
    }
    // a chunk is long: the turn goes to a waiting request at once (no NI_TURN_MS)
    if (p > start && p <= capture_from && contended(false)) {
      if (last) last->wait();  // the chunks so far (e holds nothing yet)
      if (dfl_) dfl_->end_prefill();
      if (lili_) lili_->end_prefill();
      yield_turn(res);
      if (dfl_) dfl_->begin_prefill(n, start);
      if (lili_) lili_->begin_prefill(n, start);
    }
    int T = prefill_chunk_rows(C, p, n, prefill_even_chunks() && !model_->ane_split_on());
    Checkpoint* snap = nullptr;
    int snap_at = boundary;
    if (pbound > p && pbound < p + T) {
      // the prefix checkpoint comes first; a chunk that holds the turn boundary too is cut at it (one snapshot per pass)
      if (in_pass && !model_->ane_splits(T) && !(boundary > p && boundary < p + T)) {
        snap = prefix_slot();
        snap_at = pbound;
      } else {
        T = pbound - p;
      }
    } else if (boundary > p && boundary < p + T) {
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
    model_->encode_prefill(*e, s, p, T, sp, 0, p + T == n, snap, snap ? snap_at - p : 0);
    // MTP history: pairs (hidden_i, token_{i+1}) for prompt rows whose next token is known
    const bool defer = spec_ && mtp_defer && !split && p + T == n;
    if (spec_ && !defer) q35_->encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
    if (defer) {
      mtp_e = g.begin();
      q35_->encode_mtp_prefill(*mtp_e, s, p, std::min(p + T, n - 1) - p, 0);
    }
    if (split) {
      wait_split(*e);
      if (!model_->ane_finish_chunk()) {
        // rerun on the GPU alone; it captures the chunk's rows for the drafter again (the hook is armed)
        e = g.begin();
        model_->restore(*e, s, *ane_ck_);
        model_->encode_prefill(*e, s, p, T, sp, 0, p + T == n);
        if (spec_) q35_->encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
        e->commit_and_wait();
      }
      e = g.begin();
    }
    last_chunk_row = n - 1 - p;
    p += T;
    if (p == pbound) model_->save(*e, s, *prefix_slot(), pbound);
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
    // LiLiCorr: the drafter's context, the captured prompt rows projected into its ring, and the snapshot of the
    // checkpoint this prefill saved: behind the trunk's command buffer as for DFlash2 (the first token does not
    // wait for it; the first drafting pass is queued after it), or in it with NI_MTP_DEFER=0
    const bool lili_commit = lili_ && p == n;
    auto commit_lili = [&](Encoder& en) {
      lili_->encode_prefill_commit(en, *slot.lseq);
      if (saved_ckpt >= 0) lili_->encode_copy(en, *slot.lseq, *slot.lckpts[size_t(saved_ckpt)]);
    };
    if (lili_commit) lili_->end_prefill();
    if (lili_commit && !mtp_defer) commit_lili(*e);
    if (live) e->on_completed([lv = r.live, done = p] { lv->read.store(done, std::memory_order_relaxed); });
    e->commit();
    if (p == n) {
      t_enc = now_ms() - t0;  // until the last chunk's commit
      h_commit = Encoder::host_ms();
    }
    if (dfl_commit && mtp_defer) {
      mtp_e = g.begin();
      commit_ring(*mtp_e);
    }
    if (lili_commit && mtp_defer) {
      mtp_e = g.begin();
      commit_lili(*mtp_e);
    }
    if (mtp_e) mtp_e->commit();  // queued behind the trunk; waited for by the first spec cycle's buffer
    if (p < n) {
      last2 = std::move(last);
      last = std::move(e);
      e = g.begin();
    }
  }
  e->wait();
  s.embeds = nullptr;
  s.mask_rows = 0;  // (the first token's mask, if any, was the last chunk's)
  const double h_woke = Encoder::host_ms();
  res.prefill_ms = now_ms() - t0;
  const std::string store_prompt = store_note(model_.get(), "prompt", n - start);
  // Neural Engine split: how busy the ANE was over the prefill. Near 100% it, not the GPU, is the bottleneck (a
  // smaller share would be faster: ane_calib --wave --write); the share is set for the GPU to be the bottleneck.
  if (const AneStats& a1 = model_->ane_stats(); a1.chunks > ane0.chunks)
    logi("prefill: Neural Engine busy %.0f%% of %.0f ms over %d split chunks (%d evaluations)",
         100 * (a1.ane_ms - ane0.ane_ms) / std::max(res.prefill_ms, 1.0), res.prefill_ms, a1.chunks - ane0.chunks,
         a1.evals - ane0.evals);
  const double gpu_last = e->gpu_ms();
  // last chunk: commit -> GPU start, GPU end -> this thread awake (host clock)
  const double lat_start = e->gpu_start_host_ms() - h_commit, lat_wake = h_woke - e->gpu_end_host_ms();
  slot.tokens = keys;
  slot.img_spans = mm.spans;
  slot.img_digests = mm.digests;
  const int yields_prefill = res.yields;

  // ---- decode ----
  // Between two cycles (steps) the request gives its turn to the requests waiting for one (contended()): it parks
  // first, so that nothing of its own is in flight and nothing it needs next is left in the model's or drafter's
  // shared buffers (another request's passes reuse them): DFlash2 and LiLiCorr commit the rows the cycle kept to the
  // drafter's ring (the next drafting half would have), MTP keeps the hidden rows its next pass reads in the slot.
  // A request alone never parks: its cycles are exactly those of a single request.
  // A tool call written inside the reasoning, with nothing after it but the end of the turn, is returned as a
  // tool call (Qwen3.6 sometimes stops without closing its think block); see OutputParser. NI_THINK_TOOLCALLS=0
  // leaves it in the reasoning, as before.
  static const bool think_tools = [] {
    const char* v = getenv("NI_THINK_TOOLCALLS");
    return !(v && v[0] == '0' && v[1] == 0);
  }();
  // (Nemotron-H, Dory and Flint: the whitespace before </think> stays in the reasoning, as their templates render
  // returned turns, unless the template writes a newline of its own there: think_ws_; the Nano 9B v2's tags as text;
  // Flint's calls in GLM's markup)
  OutputParser parser(tok_, think_open_, think_close_, tc_open_, tc_close_, think, !r.tools.empty(), r.tools,
                      think_tools, (nh_ != nullptr || dory_ != nullptr || flint_ != nullptr) && think_ws_, text_tags_,
                      flint_ ? CallFormat::Glm : CallFormat::QwenXml);
  const auto& eos = stop_ids_;
  double t1 = now_ms();
  int gen = 0;
  std::string content_so_far;
  std::string finish;
  // structured output: </think> ends a speculative loop running the reasoning (g_in_spec), whose drafts and verify
  // know no grammar; the constrained loop below takes the reply from there (g_break)
  bool g_in_spec = false, g_break = false;
  // Handles one generated token; returns false when generation must stop (finish is set), or when a speculative loop
  // gives the reply to the constrained loop (g_break).
  auto handle = [&](int t) -> bool {
    ++gen;
    if (live) live->tokens.store(gen, std::memory_order_relaxed);
    if (std::find(eos.begin(), eos.end(), t) != eos.end()) {
      finish = "stop";
      return false;
    }
    if (gm) {
      if (g_on) {
        if (!gm->advance(gst, t)) {  // (the masks allow only what the grammar does: a bug if it happens)
          finish = "error";
          res.error = "structured output: a token outside the grammar was sampled";
          logi("chat: %s (token %d)", res.error.c_str(), t);
          return false;
        }
      } else if (t == think_close_) {
        g_on = true;
        g_break = g_in_spec;
      }
    }
    if (parser.in_think()) ++res.reasoning_tokens;
    ChatDelta d = parser.push(t);
    if (!d.content.empty()) {
      content_so_far += d.content;
      for (auto& st : r.stop) {
        size_t at = st.empty() ? std::string::npos : content_so_far.find(st);
        if (at != std::string::npos) {
          size_t over = content_so_far.size() - at;
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
    if (finish.empty() && gen >= r.max_tokens) finish = "length";
    return finish.empty() && !g_break;
  };
  // KV room for `want` positions between two cycles, with nothing of this request in flight: as reserve_kv, but
  // when other requests' caches hold the memory, park and wait for them to end (park() makes the request's state its
  // slot's alone)
  auto grow_kv = [&](int want, const std::function<void()>& park) {
    if (reserve_kv(slot, want, true)) return true;
    if (active_ <= 1) return false;
    park();
    return reserve_kv_wait(slot, want, cancelled, true);
  };
  int fed_until = n;  // exclusive: tokens whose effect is in the live state
  int req_cycles = 0;  // DFlash2 cycles of this request
  const int depth = spec_ ? opts_.spec_depth : 0;
  auto cycle_done = [live](int m) {  // a decode cycle wrote m tokens (for a display)
    if (!live) return;
    live->last.store(m, std::memory_order_relaxed);
    live->cycles.fetch_add(1, std::memory_order_relaxed);
  };
  // NI_TRACE_TOKENS=<file> (tests: batched against single-stream outputs): each speculative cycle's tokens and the
  // margin between the two largest logits of the row that chose each, one JSON line per request at its end
  static const char* trace_path = getenv("NI_TRACE_TOKENS") && *getenv("NI_TRACE_TOKENS") ? getenv("NI_TRACE_TOKENS")
                                                                                         : nullptr;
  std::vector<int> tr_tok;
  std::vector<float> tr_margin;
  const std::function<void(const float*, int, const int*)> trace_fn = [&](const float* rows, int m, const int* toks) {
    const int V = model_->vocab();
    for (int j = 0; j < m; ++j) {
      const float* l = rows + size_t(j) * size_t(V);
      float a = -INFINITY, b = -INFINITY;
      for (int v = 0; v < V; ++v)
        if (l[v] > a) b = a, a = l[v];
        else if (l[v] > b) b = l[v];
      tr_tok.push_back(toks[j]);
      tr_margin.push_back(a - b);
    }
  };
  if (live) live->spec.store(lili_ ? 3 : dfl_ ? 2 : depth > 0 ? 1 : 0, std::memory_order_relaxed),
      live->phase.store(2, std::memory_order_relaxed);
  // batched decode: this request decodes (other requests' batch_wanted() counts it) until its loop ends
  struct Decoding {
    Slot& s;
    explicit Decoding(Slot& sl) : s(sl) { s.decoding = true; }
    ~Decoding() { s.decoding = false; }
  } decoding{slot};
  const std::function<bool(int)> handle_fn = handle;
  const std::function<void(int)> cycle_fn = cycle_done;
  // joins the batch at a cycle boundary (parked) and returns there once the request stopped (true) or is back in its
  // own loop at P (false)
  auto batch = [&](int& P) {
    BatchJob job;
    job.slot = &slot;
    job.sp = sp;
    job.P = P;
    job.handle = &handle_fn;
    job.cycle_done = &cycle_fn;
    job.trace = trace_path ? &trace_fn : nullptr;
    if (gm) job.gm = gm.get(), job.gst = &gst, job.g_on = &g_on;
    const bool done = join_batch(job);
    P = job.P;
    req_cycles += job.cycles;
    res.batched += job.cycles;
    return done;
  };
  // Structured output. DFlash2 (NI_GRAMMAR_SPEC, default on) verifies its cycles under the grammar's masks (below).
  // Other speculative loops run the reasoning alone and give the reply to the constrained loop (plain steps) at
  // </think>; a reply held from its first token, or a model without a speculative loop, takes that loop alone.
  static const bool grammar_spec = [] {
    const char* v = getenv("NI_GRAMMAR_SPEC");
    return !(v && v[0] == '0');
  }();
  const bool spec_loop = lili_ || dfl_ || dory_spec_ > 0 || nh_ngram_ > 0 || flint_ngram_ > 0 || depth > 0;
  const bool g_dfl = gm && dfl_ && grammar_spec;
  const bool g_direct = gm && !g_dfl && (g_on || !spec_loop);
  g_in_spec = gm && !g_direct && !g_on;
  double mask_ms = 0;  // the masks this request computed (host ms) and how many (the log line at its end)
  long mask_n = 0;
  // A DFlash2 cycle's masks (a tree's row r: tokens[P + r] under its parent; a chain's: tokens[P + 1 .. P + r])
  auto spec_masks = [&](int P, int rows, bool tree) {
    const int* t = tree ? dfl_->tree_desc() : nullptr;
    verify_masks(*gm, gst, s, P, rows, t ? t + kf::kPar : nullptr, t ? std::clamp(t[kf::kN], 1, rows) : rows);
  };
  if (g_direct) {
    // (the constrained loop below)
  } else if (lili_) {
    // LiLiCorr / DSpark: per cycle the drafter's context commit of the previous cycle's kept rows and the drafting pass
    // (7 proposals after the anchor tokens[P]), then the target's verify of the 8 rows (capturing the drafter's
    // features), per-row sampling and the acceptance on the GPU; the host reads the accepted count.
    // A drafter that chains() (DSpark) is double-buffered as DFlash2 below: the next cycle's drafting half (the commit
    // of the rows the acceptance keeps, the drafting pass from the anchor it wrote; both read the acceptance on the
    // GPU) is committed right behind each verify, so the GPU does not wait for the host between cycles (Lightning,
    // M5 Max, nh_eval: 14.4-14.5 -> 14.1-14.2 ms per cycle, the wall time down to the GPU time). It is left out when
    // the request parks after the verify (another request waits for the GPU, or the batch wants it) or the KV cache
    // must grow first: the next cycle then drafts from the host's P as without chaining (LiLiCorr always does).
    LiliSeq& d = *slot.lseq;
    constexpr int B = NhDrafter::kB;
    const bool chain = lili_->chains();
    int P = n, m_prev = 0;
    req_cycles = 0;
    // KFlash: the drafting pass builds a tree under the budget, the verify checks it as a tree and the acceptance keeps
    // the longest root path agreeing with the rows' own samples; the kept rows are that path's (Seq::pend_path)
    const bool kft = kf_budget_ != nullptr;
    auto commit_kept = [&](Encoder& e) {
      if (m_prev <= 0) return;
      if (kft) lili_->encode_commit_path(e, d, m_prev, s.pend_path);
      else lili_->encode_commit(e, d, m_prev);
    };
    std::unique_ptr<Encoder> next;  // the coming cycle's committed drafting half
    // parked: the rows the last cycle kept go into the ring now (the verify's captured features are shared)
    auto park = [&] {
      if (m_prev > 0) {
        auto ep = g.begin();
        commit_kept(*ep);
        ep->commit_and_wait();
        m_prev = 0;
      }
    };
    bool go = handle(s.tokens()[n]);
    while (go) {
      if (!next) {  // the drafting half from the host's P (nothing of this request in flight)
        if (P + 2 * B + 2 > s.cap && s.cap < s.limit) {
          if (mtp_e) mtp_e->wait();
          grow_kv(std::min(s.limit, P + 2 * B + 2), park);
        }
        if (P + B + 1 >= s.cap) {
          finish = "length";
          break;
        }
        next = g.begin();
        commit_kept(*next);
        m_prev = 0;
        if (kft) {
          kf_budget_->write(*kf_params_.as<KfParams>());  // (the GPU is done with it: the last cycle completed)
          lili_->encode_draft_tree(*next, s, d, P, kf_params_);
        } else {
          lili_->encode_draft(*next, s, d, P);
        }
        next->commit();
      }
      // the drafting half starts on the GPU while the host encodes the verify half (a KFlash cycle's: no chaining, so
      // the budget reads its GPU time with the verify's)
      const auto draft = std::move(next);
      auto e2 = nh_->grouped_passes() ? g.begin_grouped() : g.begin();
      lili_->arm_verify(P);
      nh_->set_dynamic_rows(true);  // the drafter's pass chose how many of the block's rows to verify
      nh_->set_tree(kft);
      nh_->encode_verify(*e2, s, P, B);
      lili_->disarm();
      if (kft) nh_->encode_accept_tree(*e2, s, P, sp);
      else nh_->encode_accept(*e2, s, P, B, B - 1, sp);
      nh_->set_tree(false);
      nh_->set_dynamic_rows(false);
      e2->commit();
      // parks after this verify (no chained drafting half): another request waits for the GPU or the batch wants this
      // one; the next anchor is at most P + 8, whose cycle must fit without growing the KV cache. (KFlash trees: the
      // drafting pass builds the next cycle's tree from the host's accepted path, so they never chain.)
      const bool join = batch_wanted(), turn = !join && contended();
      if (chain && !kft && !join && !turn && P + 8 + 2 * B + 2 <= s.cap) {
        next = g.begin();
        lili_->encode_draft_chained(*next, s, d, P);
        next->commit();
      }
      e2->wait();
      // KFlash: the budget reads the cycle's outcome and its GPU time, the drafting half's and the verify's (as every
      // backend and tool measures it)
      if (kft) kf_budget_->observe(nh_->tree_buf().as<int32_t>(), draft->gpu_ms() + e2->gpu_ms());
      const int a = nh_->accepted();
      const int* tk = s.tokens();
      int m = 0;
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
      cycle_done(m);
      if (trace_path && kft) {  // (a tree's rows: the accepted path's, in order)
        const int V = nh_->vocab();
        const int32_t* t = nh_->tree_buf().as<int32_t>();
        std::vector<float> rows(size_t(m) * size_t(V));
        for (int j = 0; j < m; ++j)
          memcpy(rows.data() + size_t(j) * V, nh_->logits() + size_t(t[kf::kPath + j]) * V, size_t(V) * 4);
        trace_fn(rows.data(), m, tk + P + 1);
      } else if (trace_path) {
        trace_fn(nh_->logits(), m, tk + P + 1);
      }
      if (kft) nh_->commit_rows_tree(s, m);  // the accepted path's first m rows are fed
      else nh_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new anchor
      if (next) {  // the chained half commits all a + 1 rows to the ring (m < a + 1: a stop inside them, the end)
        d.ctx = P + m;
        d.hi = std::max(d.hi, P + a + 1);
      } else {
        m_prev = m;
      }
      P += m;
      if (go && !next && (join || turn)) {
        park();
        if (!join) yield_turn(res);
        else if (batch(P)) go = false;
      }
    }
    auto e3 = g.begin();  // (behind a chained drafting half the loop ended with, if any)
    commit_kept(*e3);
    nh_->encode_flush(*e3, s);
    e3->commit_and_wait();
    fed_until = P;
    if (kft && req_cycles > 0) {
      logi("%s", kf_budget_->summary().c_str());
      kf_budget_->save(KfBudget::kSaveEvery);
    }
  } else if (dfl_) {
    // DFlash2, double-buffered: each cycle's verify (8-row target forward with captures + on-GPU
    // acceptance) is committed, then right behind it the NEXT cycle's drafting half (ring commit of the
    // rows this acceptance keeps + drafting pass from the anchor it picks; both read the GPU-side state
    // the acceptance writes), and only then does the host wait and read the accepted count. The GPU never
    // waits for the host between cycles; the host encodes each verify while the drafting half runs.
    // With other requests waiting for the GPU a cycle commits no drafting half behind its verify: the request parks
    // (the ring commit of the kept rows alone) and starts the next cycle from the anchor once its turn comes back.
    // Structured output: once the reasoning is over (g_on) each verify runs under the grammar's masks. The host computes
    // every verify row's mask from the drafts (the drafting half has run by then) while the verify's layers run, and
    // signals the verify's sampler; the acceptance then reads the masked samples and distributions, so the reply is
    // masked plain decoding's. Reasoning first runs unmasked and ends its pass at </think> (that cycle's later tokens
    // were verified unmasked); a second pass goes on from it, masked. In a batch (batch_cycle, batch_plain_run) the
    // runner computes the masked members' masks the same way, from their chains' drafts.
    DFlashSeq& d = *slot.dseq;
    const int K = dfl_->verify_depth();
    int P = n;
    req_cycles = 0;
    bool go = handle(s.tokens()[n]);
    // the KV room a cycle wants: the next anchor (at most P + 8) and its verify's rows (KFlash trees: up to 32; with
    // less room the drafting pass verifies fewer, DFlash2::tree_cap_)
    const int Bk = dfl_->block();  // the drafting block (8, or 16 rows: NI_DFLASH_BLOCK, or one slot's default)
    const int vrows = dfl_->tree() ? std::max(Bk, dfl_->tree_rows()) : Bk;
    for (;;) {
      int m = 0, a_last = 0;
      const bool masked = gm && g_on;
      const int mrows = dfl_->tree() ? vrows : K + 1;  // (a wide tree verify's rows; the mask buffer holds kMaxRows)
      if (masked && !mask_event_) mask_event_ = g.new_shared_event();
      std::unique_ptr<Encoder> next_draft;  // committed drafting half of the coming cycle
      // The drafting half from the anchor P with nothing in flight (after the prefill, and when the request's turn
      // comes back after parking), with KV cache room for the first cycles (it grows between cycles below): nothing may
      // use the old buffers, so the deferred prompt work completes first
      auto start_cycles = [&] {
        if (P + Bk + Bk + vrows + 3 > s.cap && s.cap < s.limit) {
          if (mtp_e) mtp_e->wait();
          grow_kv(std::min(s.limit, P + Bk + Bk + vrows + 3), [] {});
        }
        if (P + Bk + Bk + 2 < s.cap) {
          // host write of the drafter state: nothing in flight reads it (the prefill is done; its deferred
          // prompt commit, if still running, is host-driven; a request taking its turn has nothing in flight)
          dfl_->gpu_state_reset(P);
          next_draft = g.begin();
          dfl_->encode_cycle_draft_gpu(*next_draft, s, d, P, sp);
          next_draft->commit();
        }
      };
      if (go) start_cycles();
      bool parked = false;     // the request parked after the last cycle (no drafting half committed)
      bool park_last = false;  // the last cycle committed no drafting half though there was room: it meant to park
      while (go) {
        if (parked) {
          parked = false;
          start_cycles();
        }
        if (!next_draft) {  // no room for another drafting pass
          finish = "length";
          res.context_full = true;
          break;
        }
        const auto draft = std::move(next_draft);  // this cycle's drafting half (the budget reads its GPU time)
        auto e2 = g.begin();
        uint64_t pending = 0;  // masked: the value the verify's sampler waits for
        if (masked) {
          s.mask_rows = mrows;
          s.mask_event = mask_event_;
          s.mask_value = pending = ++mask_value_;
        }
        dfl_->encode_cycle_verify(*e2, s, P, K, sp);
        s.mask_rows = 0;
        s.mask_event = nullptr;
        e2->commit();
        if (masked) {  // this cycle's masks, from its drafts, while the verify's layers run
          draft->wait();
          const double tm = now_ms();
          spec_masks(P, mrows, dfl_->tree());
          mask_ms += now_ms() - tm;
          mask_n += mrows;
          Gpu::signal_event_cpu(mask_event_, pending);
        }
        bool kv_wait = false;  // no KV room for the next cycle while other requests' caches hold the memory
        if (P + Bk + Bk + vrows + 3 > s.cap && s.cap < s.limit) {
          // the KV cache grows here: the verify in flight (and the drafting half before it) complete first, so
          // this cycle's drafting half is not overlapped with it
          e2->wait();
          kv_wait = !reserve_kv(slot, std::min(s.limit, P + Bk + Bk + vrows + 3), true) && active_ > 1;
        }
        // another request waits for the GPU, this one for memory (start_cycles waits for it), or it joins the batch:
        // no drafting half behind this verify, park after it
        const bool turn = contended();
        const bool join = !kv_wait && batch_wanted();  // (a request waiting for memory would leave the batch at once)
        const bool park = turn || kv_wait || join;
        const bool room = P + Bk + Bk + Bk + 2 < s.cap;  // the next anchor is at most P + Bk
        park_last = park && room;
        if (!park && room) {
          next_draft = g.begin();
          dfl_->encode_cycle_draft_gpu(*next_draft, s, d, P + Bk, sp);
          next_draft->commit();
        }
        e2->wait();
        // KFlash: the budget reads the cycle's outcome and its GPU time, the drafting half's (done before the verify
        // started) and the verify's, as ni_run measures it. Not a masked cycle's: the grammar rejects drafts the drafter
        // was right to propose, which would teach the calibration (saved, shared) a lower acceptance than free replies
        // see, and change their trees.
        if (!masked) dfl_->observe(draft->gpu_ms() + e2->gpu_ms());
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
        cycle_done(m);
        if (trace_path && dfl_->tree()) {  // the accepted path's verify rows (tree slots), in order
          const size_t V = size_t(model_->vocab());
          std::vector<float> rows(size_t(m) * V);
          for (int j = 0; j < m; ++j)
            memcpy(rows.data() + size_t(j) * V, q35_->logits() + size_t(s.pend_path.as<int>()[j]) * V, V * 4);
          trace_fn(rows.data(), m, tk + P + 1);
        } else if (trace_path) {
          trace_fn(q35_->logits(), m, tk + P + 1);
        }
        q35_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new anchor
        P += m;
        d.ctx = P;  // the committed drafting half (if any) puts these rows into the ring
        if (go && park) {
          // park: the rows this cycle kept go into the ring as the drafting half would have put them (from the GPU
          // state the acceptance wrote, reading the verify's captures before any other request's passes)
          auto ep = g.begin();
          dfl_->encode_commit(*ep, d, -1);
          ep->commit_and_wait();
          m = 0;
          park_last = false;
          if (join) {
            if (batch(P)) go = false;
            d.ctx = P;  // (the batch committed every row before P to the ring)
          } else if (turn) {
            yield_turn(res);
          }
          parked = true;
        }
      }
      // finish: the ring rows of the last cycle (unless the committed drafting half already wrote them),
      // then fold the tape into the recurrent state
      auto e3 = g.begin();
      if (!next_draft && m > 0) {
        if (park_last) {
          // the drafting half this cycle left out to park would have committed the acceptance's rows: so does this
          dfl_->encode_commit(*e3, d, -1);
          if (m < a_last + 1) d.valid_lo = std::max(d.valid_lo, P - m + a_last + 1 - DFlash2::kWin);
        } else {
          d.ctx = P - m;
          dfl_->encode_commit(*e3, d, m);
        }
      }
      // The committed drafting half wrote all a_last + 1 rows of the last acceptance (positions P - m ..
      // P - m + a_last) to the ring. When the host kept fewer (a stop token or the budget inside the
      // accepted chain), the rows past d.ctx overwrote the slots of the window's oldest positions, which a
      // later rewind (prefix reuse) would otherwise still treat as valid context.
      if (next_draft && m < a_last + 1) d.valid_lo = std::max(d.valid_lo, P - m + a_last + 1 - DFlash2::kWin);
      q35_->encode_flush(*e3, s);
      e3->commit_and_wait();
      fed_until = P;
      if (dfl_->tree() && dfl_->budget() && req_cycles > 0) {
        logi("%s", dfl_->budget()->summary().c_str());
        dfl_->budget()->save(KfBudget::kSaveEvery);
      }
      // the reasoning ended (</think>, tokens[P], handled): the reply goes on from it, masked
      if (!(g_break && finish.empty())) break;
      g_break = g_in_spec = false;
      go = true;
    }
  } else if (dory_spec_ > 0) {
    // Dory speculative decoding (per-loop KV). Prompt lookup: when the last tokens occurred earlier in the
    // conversation, the tokens that followed them are the drafts, and one pass of the full loop count verifies them
    // with the pending token. Self-speculation (on by default; NI_DORY_SPEC=ngram turns it off; requests of 2+
    // loops): otherwise the model with one loop drafts K tokens (its input layers and loop 0 are the full model's own: their KV is kept), and the
    // verify runs loops 1.., the output layers and the head of the K + 1 rows. The samples taken at each row's position
    // accept the drafts that agree with them, so the reply is the one plain decoding writes (up to near-ties: the verify
    // pass's products read half rows).
    const std::string spec_kind = dory_spec_kind();
    const bool use_ngram = spec_kind != "self", use_self = spec_kind != "ngram" && loops > 1;
    const int K = dory_spec_auto_ ? (loops >= 3 ? 3 : 2) : dory_spec_;
    int P = n;
    req_cycles = 0;
    bool go = handle(s.tokens()[n]);
    std::vector<int> nd;
    while (go) {
      if (P + K + 2 >= s.cap) {
        if (mtp_e) mtp_e->wait();
        if (P + K + 2 >= s.limit || !grow_kv(std::min(s.limit, P + K + 3), [] {})) {
          finish = "length";
          res.context_full = true;
          break;
        }
      }
      if (mtp_e) mtp_e->wait();  // (nothing reads the token buffer before the drafts are written)
      nd.clear();
      if (use_ngram) ngram_drafts(s.tokens(), P, K, nd);
      auto e2 = g.begin();
      if (!nd.empty()) {
        int* tw = s.tokens();
        for (size_t j = 0; j < nd.size(); ++j) tw[P + 1 + int(j)] = nd[j];
        dory_->encode_verify(*e2, s, P, int(nd.size()) + 1);
        dory_->encode_accept(*e2, s, P, int(nd.size()) + 1, int(nd.size()), sp);
      } else if (use_self) {
        for (int j = 0; j < K; ++j) dory_->encode_draft(*e2, s, P + j, j);
        dory_->encode_verify_drafted(*e2, s, P, K + 1);
        dory_->encode_accept(*e2, s, P, K + 1, K, sp);
      } else {
        dory_->encode_verify(*e2, s, P, 1);
        dory_->encode_accept(*e2, s, P, 1, 0, sp);
      }
      e2->commit_and_wait();
      const int a = dory_->accepted();
      const int* tk = s.tokens();
      int m = 0;
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
      cycle_done(m);
      P += m;
      if (go && contended()) yield_turn(res);
    }
    fed_until = P;
  } else if (nh_ngram_ > 0) {
    // Nemotron-H without a block drafter (NI_NH_NGRAM): prompt lookup. When the last tokens occurred earlier in the
    // conversation, the tokens that followed them (up to nh_ngram_) are drafts, verified with the pending token in one
    // pass; otherwise the pass is the pending token's row alone. The samples taken at each row's position accept the
    // drafts that agree with them, so the reply is the one plain decoding writes (up to near-ties: a pass of 2+ rows
    // reads half rows). The accepted rows stay pending in the sequence until the next pass (or the flush) commits them.
    int P = n;
    req_cycles = 0;
    bool go = handle(s.tokens()[n]);
    std::vector<int> nd;
    while (go) {
      constexpr int B = NemotronH::kMaxRows;
      if (P + 2 * B + 2 > s.cap && s.cap < s.limit) {
        if (mtp_e) mtp_e->wait();
        grow_kv(std::min(s.limit, P + 2 * B + 2), [] {});
      }
      if (P + B + 1 >= s.cap) {
        finish = "length";
        res.context_full = true;
        break;
      }
      if (mtp_e) mtp_e->wait();  // (nothing reads the token buffer before the drafts are written)
      ngram_drafts(s.tokens(), P, nh_ngram_, nd);
      int* tw = s.tokens();
      for (size_t j = 0; j < nd.size(); ++j) tw[P + 1 + int(j)] = nd[j];
      const int R = int(nd.size()) + 1;
      auto e2 = nh_->grouped_passes() ? g.begin_grouped() : g.begin();
      nh_->encode_verify(*e2, s, P, R);
      nh_->encode_accept(*e2, s, P, R, R - 1, sp);
      e2->commit_and_wait();
      const int a = nh_->accepted();
      const int* tk = s.tokens();
      int m = 0;
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
      cycle_done(m);
      nh_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new pending token
      P += m;
      if (go && contended()) yield_turn(res);  // (the pending rows are the sequence's own)
    }
    auto e3 = g.begin();
    nh_->encode_flush(*e3, s);
    e3->commit_and_wait();
    fed_until = P;
  } else if (flint_ngram_ > 0) {
    // Flint: prompt lookup. When the last tokens occurred earlier in the conversation, the tokens that followed them (up
    // to flint_ngram_) are drafts, verified with the pending token in one pass; otherwise the pass is the pending
    // token's row alone. Each row samples at its own position, accepting the drafts that agree. Rows past the accepted
    // ones leave keys at positions the next pass writes again before any row reads them (a ring's included: its 8
    // spare rows hold them), so nothing is committed or flushed.
    int P = n;
    req_cycles = 0;
    bool go = handle(s.tokens()[n]);
    std::vector<int> nd;
    while (go) {
      constexpr int B = Flint::kMaxRows;
      if (P + 2 * B + 2 > s.cap && s.cap < s.limit) {
        if (mtp_e) mtp_e->wait();
        grow_kv(std::min(s.limit, P + 2 * B + 2), [] {});
      }
      if (P + B + 1 >= s.cap) {
        finish = "length";
        res.context_full = true;
        break;
      }
      if (mtp_e) mtp_e->wait();  // (nothing reads the token buffer before the drafts are written)
      ngram_drafts(s.tokens(), P, flint_ngram_, nd);
      int* tw = s.tokens();
      for (size_t j = 0; j < nd.size(); ++j) tw[P + 1 + int(j)] = nd[j];
      const int R = int(nd.size()) + 1;
      auto e2 = g.begin();
      flint_->encode_verify(*e2, s, P, R);
      flint_->encode_accept(*e2, s, P, R, R - 1, sp);
      e2->commit_and_wait();
      const int a = flint_->accepted();
      const int* tk = s.tokens();
      int m = 0;
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
      cycle_done(m);
      P += m;
      if (go && contended()) yield_turn(res);
    }
    fed_until = P;
  } else if (depth > 0) {
    // Speculative decoding: the MTP head drafts `depth` tokens, the trunk verifies them together
    // with the pending token, accepted rows are committed through the DeltaNet tape.
    int P = n, c = 1, h_row0 = last_chunk_row;
    // the trunk hidden rows the next MTP pass reads: the model's (the last pass's rows), or the slot's copy of them
    // while parked (other requests' passes reuse the model's)
    Buffer hsrc = q35_->hidden_rows();
    auto park = [&] {
      if (hsrc.mtl != q35_->hidden_rows().mtl) return;  // parked already
      const size_t row = q35_->hidden_row_bytes();
      if (!slot.park_h) slot.park_h = g.alloc(size_t(Qwen35::kMaxVerify) * row, "park_hidden");
      auto ep = g.begin();
      q35_->encode_copy(*ep, hsrc.view(size_t(h_row0) * row, size_t(c) * row), slot.park_h.view(0, size_t(c) * row));
      ep->commit_and_wait();
      hsrc = slot.park_h;
      h_row0 = 0;
    };
    bool go = handle(s.tokens()[n]);
    while (go) {
      if (P + depth + 2 >= s.cap) {
        // the KV cache grows between cycles; only the prompt's deferred MTP history can still be in flight
        if (mtp_e) mtp_e->wait();
        if (P + depth + 2 >= s.limit || !grow_kv(std::min(s.limit, P + depth + 3), park)) {
          finish = "length";
          res.context_full = true;
          break;
        }
      }
      auto e2 = g.begin();
      q35_->encode_mtp(*e2, s, P - c, c, P - c + 1, hsrc, h_row0, P + 1);
      for (int k = 2; k <= depth; ++k)
        q35_->encode_mtp(*e2, s, P + k - 2, 1, P + k - 1, q35_->mtp_post(), k == 2 ? c - 1 : 0, P + k, false);
      q35_->encode_verify(*e2, s, P, depth + 1, sp, uint32_t(P));
      e2->commit_and_wait();
      const int* y = q35_->samples();
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
      cycle_done(m);
      q35_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new pending token
      c = m;
      h_row0 = 0;
      hsrc = q35_->hidden_rows();
      P += m;
      if (go && contended()) {
        park();
        yield_turn(res);
      }
    }
    // finish: history rows for the last committed rows, then fold the tape into the state
    auto e3 = g.begin();
    if (c > 0 && P - c >= 0) q35_->encode_mtp(*e3, s, P - c, c, P - c + 1, hsrc, h_row0, -1);
    q35_->encode_flush(*e3, s);
    e3->commit_and_wait();
    fed_until = P;
  } else {
    std::unique_ptr<Encoder> next;  // step feeding the newest sampled token
    int pos = n;                    // position of the newest sampled token
    auto encode_at = [&](int q) {
      auto x = model_->grouped_passes() ? g.begin_grouped() : g.begin();
      model_->encode_step(*x, s, q, sp, uint32_t(q), true);
      x->commit();
      return x;
    };
    // parked: the step in flight completes (it writes only the slot's state and its next token)
    auto park = [&] {
      if (next) next->wait();
    };
    if (r.max_tokens > 1) next = encode_at(pos);
    while (true) {
      bool go = handle(s.tokens()[pos]);  // available: produced by the previous command buffer
      if (go && pos + 2 >= s.cap) {
        // the KV cache grows between steps: the step in flight writes it, so it completes first
        if (next) next->wait();
        if (pos + 2 >= s.limit || !grow_kv(std::min(s.limit, pos + 3), park)) {
          finish = "length";
          res.context_full = true;
          go = false;
        }
      }
      if (!go) break;
      if (contended()) {
        park();
        yield_turn(res);
      }
      // the step feeding token `pos` is in flight; queue the one after it, then wait
      auto cur = std::move(next);
      next = encode_at(pos + 1);
      cur->wait();
      ++pos;
      cycle_done(1);
    }
    // Drain: the step feeding the last token (if encoded) completes, so the live state covers
    // tokens [0, pos] when it ran, else [0, pos).
    fed_until = pos;
    if (next) {
      next->wait();
      fed_until = pos + 1;
    }
  }
  const bool switched = g_break;  // a speculative loop ended at </think>: tokens[fed_until] is it, handled
  g_break = g_in_spec = false;
  if (gm && finish.empty() && (g_direct || switched)) {
    // ---- structured output: plain steps, each token sampled under the grammar's mask ----
    // A step's layers run while the host computes the mask of its sample (the grammar's state after the token the step
    // feeds) and signals the event its sampler waits on (Seq::mask_event); the next step is queued before the host reads
    // this one's token (its embedding reads it on the GPU), as in the plain loop above, so the GPU waits for the host
    // only when a mask takes longer than a step's layers. Before </think> (reasoning on, no speculative loop) the masks
    // allow every token. Not batched: other requests' batches do not wait for it (slot.decoding off).
    slot.decoding = false;
    if (!mask_event_) mask_event_ = g.new_shared_event();
    s.mask_rows = 1;
    s.mask_event = mask_event_;
    uint32_t* const mw = s.mask.as<uint32_t>();
    const size_t mbytes = size_t(s.mask_words) * 4;
    int pos = switched ? fed_until : n;  // the newest sampled token (not fed yet)
    uint64_t pending = 0;               // the value the queued step's sampler waits for, until signalled
    auto signal = [&] {
      if (!pending) return;
      Gpu::signal_event_cpu(mask_event_, pending);
      pending = 0;
    };
    auto encode_at = [&](int q) {  // the step feeding tokens[q]; it samples tokens[q + 1] once its mask is signalled
      s.mask_value = ++mask_value_;
      pending = s.mask_value;
      auto x = model_->grouped_passes() ? g.begin_grouped() : g.begin();
      model_->encode_step(*x, s, q, sp, uint32_t(q), true);
      x->commit();
      return x;
    };
    // KV room for positions up to `need` (nothing of this request in flight)
    auto room = [&](int need) {
      if (need < s.cap) return true;
      if (mtp_e) mtp_e->wait();
      return need < s.limit && grow_kv(std::min(s.limit, need + 1), [] {}) && need < s.cap;
    };
    std::unique_ptr<Encoder> next;  // the step feeding tokens[pos]
    bool go = switched || handle(s.tokens()[pos]);
    while (go) {
      if (!next) {
        if (!room(pos + 1)) {
          finish = "length";
          res.context_full = true;
          break;
        }
        next = encode_at(pos);
      }
      const double tm = now_ms();
      if (g_on) gm->fill_mask(gst, mw);
      else memset(mw, 0xff, mbytes);
      mask_ms += now_ms() - tm;
      ++mask_n;
      signal();
      if (pos + 2 >= s.cap) {  // the cache grows between steps: the step in flight writes it, so it completes first
        next->wait();
        if (!room(pos + 2)) {
          finish = "length";
          res.context_full = true;
          break;
        }
      }
      if (contended()) {  // park: the step in flight completes (its mask is signalled)
        next->wait();
        yield_turn(res);
      }
      auto cur = std::move(next);
      next = encode_at(pos + 1);
      cur->wait();
      ++pos;
      cycle_done(1);
      go = handle(s.tokens()[pos]);
    }
    signal();  // a step still waiting for its mask: any lets it end (its sample is not used)
    fed_until = pos;
    if (next) {
      next->wait();
      fed_until = pos + 1;
    }
    s.mask_rows = 0;
    s.mask_event = nullptr;
  }
  if (gm && mask_n > 0)
    logi("chat: structured output (%s), %s: %ld masks (batched passes' not counted), %.1f us each",
         grammar_summary(*r.grammar).c_str(), g_dfl ? "DFlash2 cycles under masks" : "plain steps", mask_n,
         1000 * mask_ms / double(mask_n));
  slot.decoding = false;  // (a batch forming no longer waits for this request)
  if (live) live->phase.store(3, std::memory_order_relaxed);
  res.slot = int(&slot - slots_.data());
  res.slot_gen = slot.gen;
  res.req_seq = requests_.load();
  slot.done_at = res.req_seq;
  if (!images.empty()) {  // for prefill_history: the images without their pixels
    for (vision::Prepared& p : images) std::vector<uint8_t>().swap(p.pixels);
    res.images = std::make_shared<const std::vector<vision::Prepared>>(std::move(images));
  }
  ChatDelta tail = parser.finish(finish == "stop");
  res.reasoning += tail.reasoning;
  res.content += tail.content;
  for (auto& tc : tail.tool_calls) res.tool_calls.push_back(tc);
  if (!tail.reasoning.empty() || !tail.content.empty() || !tail.tool_calls.empty()) on_delta(tail);
  // Dory (NVIDIA's nemotron_v3 reasoning parser): with force_nonempty_content, on by default here (the card asks
  // coding agents to set it), a reply that put everything in its reasoning and left the content empty returns that
  // text as its content too (a last delta), so agents that read the content alone get the answer
  if (dory_ && res.tool_calls.empty() && content_text(ojson(res.content)).empty() && !res.reasoning.empty()) {
    const auto f = r.template_kwargs.find("force_nonempty_content");
    const bool force = f == r.template_kwargs.end() || !f->is_boolean() || f->get<bool>();
    if (force) {
      ChatDelta d;
      d.content = res.reasoning;
      res.content = res.reasoning;
      res.reasoning.clear();
      on_delta(d);
    }
  }
  if (parser.promoted() > 0)
    logi("chat: %d tool call(s) written inside the reasoning returned as tool calls", parser.promoted());
  slot.tokens.assign(s.tokens(), s.tokens() + fed_until);
  if (mmp) std::copy(keys.begin(), keys.end(), slot.tokens.begin());  // image rows: their keys
  res.completion_tokens = gen;
  if ((dfl_ || lili_) && spec_cycles_ > 0)
    logd("%s: %.2f tokens/cycle over %ld cycles", lili_ ? lili_->name() : "dflash", double(spec_accepted_) / spec_cycles_,
         spec_cycles_);
  else if (depth > 0 && spec_cycles_ > 0)
    logd("spec: %.2f accepted/drafted over %ld cycles", double(spec_accepted_) / (spec_cycles_ * depth), spec_cycles_);
  res.decode_ms = now_ms() - t1;
  if (trace_path) {  // one line: the prompt (FNV-1a of its ids), the first token, then each cycle's tokens and margins
    uint64_t h = 1469598103934665603ull;
    for (int t : prompt) h = (h ^ uint64_t(uint32_t(t))) * 1099511628211ull;
    nlohmann::json j{{"prompt_hash", std::to_string(h)}, {"prompt_tokens", n}, {"temperature", r.temperature},
                     {"first", s.tokens()[n]}, {"tokens", tr_tok}, {"margins", tr_margin}, {"batched", res.batched}};
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (FILE* f = fopen(trace_path, "a")) {
      fprintf(f, "%s\n", j.dump().c_str());
      fclose(f);
    }
  }
  if (finish == "stop" && !res.tool_calls.empty()) finish = "tool_calls";
  res.finish_reason = finish;
  // rows [start, fed_until) are this request's; after tool calls the next request continues the live state as it is
  // (after a final answer the history prefill makes it so)
  disk_filled(slot, start, fed_until);
  slot.live_ok = finish == "tool_calls";
  ++slot.version;  // (a state the SSD prefix cache does not have yet)
  char cyc[64] = "";
  if ((dfl_ || lili_ || dory_spec_ || nh_ngram_ || flint_ngram_) && req_cycles > 0)
    snprintf(cyc, sizeof cyc, ", %.2f tok/cycle", double(gen - 1) / req_cycles);
  // requests in flight at once: the slot, the wait for it, the turns given to the others (prefill, decode)
  char conc[256] = "";
  if (res.yields > 0 || res.queue_ms > 0 || res.batched > 0)
    snprintf(conc, sizeof conc, "; slot %d, waited %.0f ms for it, turns given to other requests: %d in the prefill, %d "
             "while decoding; %d cycles batched with other requests", res.slot, res.queue_ms, yields_prefill,
             res.yields - yields_prefill, res.batched);
  if (mmp)
    logi("chat: %zu image(s), %d tokens: %d encoded in %.0f ms, %d from earlier embeddings, %d in the cached prefix "
         "(+%.0f ms decode/resize before the request)", mm.spans.size(),
         [&] { int t = 0; for (auto& sp : mm.spans) t += sp.n; return t; }(), img_encoded, img_ms, img_reused,
         int(mm.spans.size()) - img_encoded - img_reused, img_prep_ms);
  char lp[32] = "";
  if (loops > 0) snprintf(lp, sizeof lp, ", %d loop%s", loops, loops > 1 ? "s" : "");
  logi("chat: prompt %d (cached %d) in %.0f ms (+%.1f ms template/tokenize; last chunk: encoded by %.1f ms, GPU start "
       "+%.1f ms, GPU %.1f ms, wake +%.1f ms), %d tokens in %.0f ms (%.1f tok/s), %s%s%s%s", n, start, res.prefill_ms,
       tok_ms, t_enc, lat_start, gpu_last, lat_wake, gen, res.decode_ms,
       gen > 1 ? (gen - 1) / (res.decode_ms / 1000) : 0.0, finish.c_str(), cyc, conc, lp);
  if (!store_prompt.empty()) logi("expert store: %s; %s", store_prompt.c_str(), store_note(model_.get(), "decode", gen).c_str());
  return res;
}

int Engine::request_loops(const ChatRequest& r) const {
  if (!dory_) return 0;
  int l = 0;
  if (r.loops > 0) {
    l = r.loops;
  } else if (!r.effort.empty()) {
    // OpenAI's reasoning effort: minimal / low 1 loop, medium 2, high / xhigh 4 ("none" only turns the visible
    // thinking off, as benchmarks and agents send it with any model: the server's default loops)
    const std::string& e = r.effort;
    if (e == "minimal" || e == "low") l = 1;
    else if (e == "medium") l = 2;
    else if (e == "high" || e == "xhigh" || e == "max") l = 4;
  } else if (r.thinking_budget >= 0) {
    // Anthropic's thinking: disabled 1 loop, a budget by its size (under 4096 tokens 1, under 16384 2, else 4)
    l = r.thinking_budget < 4096 ? 1 : r.thinking_budget < 16384 ? 2 : 4;
  }
  if (l <= 0) l = dory_->default_loops();
  return std::clamp(l, 1, Dory::kMaxLoops);
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
  std::unique_lock<TurnLock> lk(turn_, std::try_to_lock);  // nobody holds the GPU turn or waits for it
  if (!lk.owns_lock() || requests_.load() != res.req_seq || active_ > 0) return;
  Slot& slot = slots_[size_t(res.slot)];
  if (slot.gen != res.slot_gen) return;
  const double t0 = now_ms();
  // the answer as the client sends it back ("stop": no tool calls), then a placeholder user turn: the
  // template renders answers before the last user query without their think block, so render it that
  // way and keep the text up to the placeholder turn's start
  ChatRequest r2 = r;
  r2.messages.push_back({{"role", "assistant"}, {"content", res.content}});
  // with its reasoning when the client sends reasoning back (it asks to keep every turn's thinking, or an earlier
  // assistant turn came back with it): templates that render past reasoning (Qwen3.8; preserve_thinking) then render
  // this turn as the live state holds it, and there is nothing to prefill
  const auto pt = r.template_kwargs.find("preserve_thinking");
  bool returns_reasoning = pt != r.template_kwargs.end() && pt->is_boolean() && pt->get<bool>();
  for (const auto& m : r.messages)
    returns_reasoning = returns_reasoning || (role_of(m) == "assistant" && m.contains("reasoning_content") &&
                                              m["reasoning_content"].is_string() &&
                                              !m["reasoning_content"].get<std::string>().empty());
  if (returns_reasoning) r2.messages.back()["reasoning_content"] = res.reasoning;
  static const std::string marker = "\x01nace-history-marker\x01";
  r2.messages.push_back({{"role", "user"}, {"content", marker}});
  std::vector<vision::Prepared> images;  // the request's, as chat() prepared them (no download again)
  if (res.images) images = *res.images;
  std::string text;
  try {
    if (vision_ && vision::image_sources(r2.messages).size() != images.size()) return;
    text = render(r2, false);
  } catch (...) {
    return;
  }
  const size_t mpos = text.rfind(marker);
  const size_t cut = mpos == std::string::npos ? std::string::npos : text.rfind(flint_ ? "<|user|>" : "<|im_start|>", mpos);
  if (cut == std::string::npos) return;
  text.resize(cut);
  std::vector<int> t2 = tok_.encode(text);
  vision::Prompt mm;
  const vision::Prompt* mmp = nullptr;
  if (!images.empty()) {
    try {
      mm = expand_images(t2, images);
    } catch (...) {
      return;
    }
    t2 = mm.ids;
    mmp = &mm;
  }
  const std::vector<int>& k2 = mmp ? mm.keys : t2;
  // prefill all but the last token: the follow-up's own prefill then starts with it, so its MTP
  // history pass covers the pair (hidden, next token) of this turn's last row
  const int n = int(t2.size()) - 1;
  const size_t common = common_rows(slot, k2, mmp);
  // nothing to add, or the live state matches it as far as it goes (the follow-up continues it)
  if (n > 0 && common == slot.tokens.size() && !slot.live_ok) {
    slot.live_ok = true;  // (the SSD prefix cache saves the live state too)
    ++slot.version;
  }
  if (n <= 0 || int(common) >= n || common == slot.tokens.size()) return;
  const TurnCheckpoint* best = nullptr;
  for (auto& c : slot.ckpts)
    if (!c.prefix && c.pos > 0 && c.pos <= int(common) && c.pos < n && (!best || c.pos > best->pos)) best = &c;
  Seq& s = *slot.seq;
  if (!best || n + 64 >= s.limit) return;
  for (const vision::ImageSpan& sp : mm.spans)  // an image to encode: left to the next request
    if (sp.pos + sp.n > best->pos) return;
  if (!model_->reserve(s, n + 64)) return;
  Gpu& g = Gpu::get();
  auto e = g.begin();
  model_->restore(*e, s, *best);
  const int start = best->pos;
  for (auto& c : slot.ckpts)
    if (c.pos > start) c.pos = -1;
  std::copy(t2.begin(), t2.end(), s.tokens());
  model_->set_rope_positions(s, n, mm.spans);
  // DFlash2: the drafter's context ring goes back to the checkpoint with the target state, and the rows
  // prefilled here enter it like prompt rows (captured by every pass, projected after the last chunk)
  if (dfl_) {
    dfl_->encode_ring_copy(*e, *slot.dckpts[size_t(best - slot.ckpts.data())], *slot.dseq);
    DFlash2::rewind(*slot.dseq, start);
    dfl_->begin_prefill(n, start);
  }
  if (lili_) {
    lili_->encode_copy(*e, *slot.lckpts[size_t(best - slot.ckpts.data())], *slot.lseq);
    NhDrafter::rewind(*slot.lseq, start);
    lili_->begin_prefill(n, start);
  }
  SamplingParams sp;
  const int C = model_->prefill_chunk();
  const bool even = prefill_even_chunks() && !model_->ane_split_on();
  for (int p = start; p < n;) {
    const int T = prefill_chunk_rows(C, p, n, even);
    // as in chat(): a chunk the NI_ANE split runs is checkpointed and waited for, and rerun on the GPU
    // alone if the split failed (the rerun captures its rows for the drafter again)
    const bool split = model_->ane_splits(T);
    if (split) {
      if (!ane_ck_) ane_ck_ = std::make_unique<Checkpoint>(model_->new_checkpoint());
      model_->save(*e, s, *ane_ck_, p);
    }
    model_->encode_prefill(*e, s, p, T, sp, 0, false);
    if (spec_) q35_->encode_mtp_prefill(*e, s, p, T, 0);  // next tokens known up to t2[n]
    if (split) {
      wait_split(*e);
      if (!model_->ane_finish_chunk()) {
        e = g.begin();
        model_->restore(*e, s, *ane_ck_);
        model_->encode_prefill(*e, s, p, T, sp, 0, false);
        if (spec_) q35_->encode_mtp_prefill(*e, s, p, T, 0);
        e->commit_and_wait();
      }
      e = g.begin();
    }
    p += T;
    if (lili_ && p == n) {
      lili_->end_prefill();
      lili_->encode_prefill_commit(*e, *slot.lseq);
    }
    if (dfl_ && p == n) {
      dfl_->end_prefill();
      dfl_->encode_prefill_commit(*e, *slot.dseq);
    }
    e->commit();
    if (p < n) e = g.begin();
  }
  e->wait();
  slot.tokens.assign(k2.begin(), k2.begin() + n);
  slot.img_spans = mm.spans;
  slot.img_digests = mm.digests;
  disk_filled(slot, start, n);
  slot.live_ok = true;  // the follow-up continues this state
  ++slot.version;
  logi("history: prefilled %d tokens of the rendered reply (from %d) in %.0f ms", n - start, start, now_ms() - t0);
}

} // namespace ni
