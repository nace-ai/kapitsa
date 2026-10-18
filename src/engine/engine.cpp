#include "engine/engine.h"
#include "engine/output_parser.h"
#include "util/log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <sys/stat.h>

#include <algorithm>
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

Engine::Engine(const EngineOptions& o) : opts_(o) {
  if (opts_.model_id.empty()) {
    std::string d = opts_.model_dir;
    while (!d.empty() && d.back() == '/') d.pop_back();
    opts_.model_id = d.substr(d.find_last_of('/') + 1);
  }
  // Memory: fail before reading any weights when the model, its buffers and the slots' state with a minimal
  // KV cache cannot fit the budget; the KV caches then grow as conversations fill (Qwen35::reserve).
  budget_ = opts_.mem_budget_gb > 0 ? size_t(opts_.mem_budget_gb * 1e9) : default_mem_budget();
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
      if (opts_.prefill_chunk != 2048)
        logi("memory: %d-row prefill chunks (2048 would leave less than one %d-token conversation in the %.1f GB budget)",
             opts_.prefill_chunk, opts_.context, budget_ / 1e9);
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
  if (NemotronH::is_model(opts_.model_dir)) {
    auto m = std::make_unique<NemotronH>(opts_.model_dir, opts_.prefill_chunk);
    nh_ = m.get();
    model_ = std::move(m);
    // the publisher's sampling settings (generation_config.json; Nemotron 3.5: temperature 1.0, top-p 0.95, no
    // top-k: the sampler's 32)
    auto gc = nlohmann::json::parse(read_file(opts_.model_dir + "/generation_config.json"), nullptr, false);
    if (!gc.is_discarded() && gc.is_object()) {
      if (gc.contains("temperature") && gc["temperature"].is_number()) defaults_.temperature = gc["temperature"];
      if (gc.contains("top_p") && gc["top_p"].is_number()) defaults_.top_p = gc["top_p"];
      defaults_.top_k = gc.contains("top_k") && gc["top_k"].is_number() ? gc["top_k"].get<int>() : 0;
    }
  } else {
    auto m = std::make_unique<Qwen35>(opts_.model_dir, opts_.prefill_chunk);
    q35_ = m.get();
    model_ = std::move(m);
  }
  tok_.load(opts_.model_dir + "/tokenizer.json");
  model_->set_mem_budget(budget_);
  const std::string src = chat_template_source(opts_.model_dir);
  if (src.empty()) fatal("no chat template in %s", opts_.model_dir.c_str());
  tmpl_ = std::make_unique<minja::chat_template>(src, "", "<|im_end|>");
  think_open_ = tok_.find("<think>");
  think_close_ = tok_.find("</think>");
  tc_open_ = tok_.find("<tool_call>");
  tc_close_ = tok_.find("</tool_call>");
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
  if (opts_.spec_depth < 0) opts_.spec_depth = q35_ ? q35_->default_spec_depth() : 0;
  spec_ = q35_ && opts_.spec_depth > 0 && q35_->has_mtp();
  if (opts_.spec_depth > Qwen35::kMaxMtpRows - 1) opts_.spec_depth = Qwen35::kMaxMtpRows - 1;
  if (q35_ && !lp.dflash_off && lp.dflash_left_out.empty()) {
    const std::string& dir = lp.dflash;
    if (dir.empty()) {
      logi("DFlash2: no drafter found for %s (looked next to the model and in ~/.cache/nace-infer-models/drafters)",
           opts_.model_dir.c_str());
    } else if (lp.dflash_auto && !DFlash2::supports(*q35_)) {
      logi("DFlash2: %s does not fit this model's embedding / LM head format; not used", dir.c_str());
    } else {
      const size_t a0 = Gpu::get().allocated_bytes();
      dfl_ = std::make_unique<DFlash2>(*q35_, dir);
      DFlash2::prewarm_pipelines();
      spec_ = false;
      // the plan against what loading took (rings and the first prefill's buffers come later)
      size_t per_slot = 0, at_prefill = 0;
      const size_t planned = DFlash2::plan_bytes(dir, q35_->config(), Qwen35::plan(opts_.model_dir, 1024).head_row_bytes,
                                                 &per_slot, &at_prefill);
      logi("DFlash2: %.3f GB allocated at load (planned %.3f GB); %.3f GB more for %d slots' rings and %.3f GB at the "
           "first prefill", (Gpu::get().allocated_bytes() - a0) / 1e9, (planned - at_prefill - (size_t(4) << 20)) / 1e9,
           3.0 * per_slot * std::max(1, opts_.slots) / 1e9, std::max(1, opts_.slots), at_prefill / 1e9);
    }
  }

  if (nh_ && !lp.dflash_off && lp.dflash_left_out.empty()) {
    if (lp.dflash.empty()) {
      logi("LiLiCorr: no drafter found next to %s", opts_.model_dir.c_str());
    } else {
      lili_ = std::make_unique<LiliCorr>(*nh_, lp.dflash);
    }
  }

  auto add_slot = [&](int limit) {
    Slot s;
    s.seq = model_->new_seq(limit, 0);  // KV cache on first use, growing with the conversation
    if (dfl_) s.dseq = dfl_->new_seq();
    if (lili_) s.lseq = lili_->new_seq();
    s.ckpts.push_back({model_->new_checkpoint()});
    s.ckpts.push_back({model_->new_checkpoint()});
    if (dfl_)
      for (size_t i = 0; i < s.ckpts.size(); ++i) s.dckpts.push_back(dfl_->new_seq());
    if (lili_)
      for (size_t i = 0; i < s.ckpts.size(); ++i) s.lckpts.push_back(lili_->new_seq());
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
  const std::string mode = lili_ ? std::string("LiLiCorr speculative decoding (7-token blocks)")
                         : dfl_ ? "DFlash2 speculative decoding (verify " + std::to_string(DFlash2::verify_depth()) + " of 7)"
                         : spec_ ? "MTP speculative decoding depth " + std::to_string(opts_.spec_depth)
                                 : std::string("plain decoding");
  logi("engine ready: %s, context %d, %d slots, %s%s", opts_.model_id.c_str(), opts_.context, std::max(1, opts_.slots),
       mode.c_str(), vision_ ? ", images on" : "");
}

Engine::~Engine() = default;

size_t default_mem_budget() {
  if (const char* v = getenv("NI_MEM_BUDGET"); v && *v) return size_t(atof(v) * 1e9);
  const size_t rws = Gpu::get().recommended_working_set();
  return rws - std::max<size_t>(size_t(512) << 20, rws / 25);
}

// The model's memory plan from its headers: a Nemotron-H or a Qwen3.5-family folder.
static MemPlan model_plan(const std::string& model_dir, int prefill_chunk) {
  return NemotronH::is_model(model_dir) ? NemotronH::plan(model_dir, prefill_chunk) : Qwen35::plan(model_dir, prefill_chunk);
}
static size_t model_ane_bytes(const std::string& model_dir, const MemPlan& p) {
  return NemotronH::is_model(model_dir) ? p.ane : Qwen35::ane_plan(model_dir, p).bytes;
}

int auto_prefill_chunk(const std::string& model_dir, int context, int slots, size_t budget) {
  // 1024-row buffers need ~45% less than 2048-row ones (35B: 0.75 -> 0.38 GB, 27B: 0.94 -> 0.52) for a few % of
  // prefill speed
  const MemPlan p = model_plan(model_dir, 2048);
  const size_t ane = model_ane_bytes(model_dir, p);
  return p.fixed(std::max(1, slots)) + ane + p.kv_per_pos * size_t(context) <= budget ? 2048 : 1024;
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
  if (NemotronH::is_model(model_dir)) {
    // the LiLiCorr drafter in the model's dflash/ folder (auto), or the one named
    if (!lp.dflash_off) {
      lp.dflash = lp.dflash_auto ? LiliCorr::find(model_dir) : want;
      if (!lp.dflash.empty()) {
        size_t per_slot = 0;
        lp.dflash_bytes = lp.dflash_found_bytes = LiliCorr::plan_bytes(lp.dflash, &per_slot) + 3 * per_slot * size_t(slots);
        if (lp.dflash_auto &&
            !conversation_fits(model_dir, context, slots, budget > lp.dflash_bytes ? budget - lp.dflash_bytes : 0)) {
          lp.dflash_left_out = lp.dflash;
          lp.dflash.clear();
          lp.dflash_bytes = 0;
        }
      }
    }
  } else if (!lp.dflash_off) {
    lp.dflash = lp.dflash_auto ? DFlash2::find(model_dir, Qwen35::read_config(model_dir)) : want;
    if (!lp.dflash.empty()) {
      // auto: a drafter that would keep one conversation from its full context is left out (MTP or plain decoding
      // instead); the model with 1024-row prefill buffers, the slots' state, the Neural Engine split and one full
      // conversation come first
      const Qwen35::MemPlan p = Qwen35::plan(model_dir, 1024);
      size_t per_slot = 0;  // the ring of the live state and of each of the slot's two checkpoints
      lp.dflash_bytes = lp.dflash_found_bytes =
          DFlash2::plan_bytes(lp.dflash, Qwen35::read_config(model_dir), p.head_row_bytes, &per_slot) +
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

bool conversation_fits(const std::string& model_dir, int context, int slots, size_t budget) {
  const MemPlan p = model_plan(model_dir, 1024);
  return p.fixed(std::max(1, slots)) + model_ane_bytes(model_dir, p) + p.kv_per_pos * size_t(context) <= budget;
}

int variant_fits(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget) {
  try {
    const std::string dir = hub::plan_dir(ref);
    if (dir.empty()) return -1;
    if (!conversation_fits(dir, opt.context, opt.slots, budget)) return 0;
    const LoadPlan lp =
        plan_load(dir, opt.dflash, opt.spec_depth == 0, opt.context, opt.slots, budget, opt.prefill_chunk);
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
}

bool Engine::reserve_kv(Slot& slot, int need) {
  Seq& s = *slot.seq;
  if (model_->reserve(s, need)) return true;
  if (vision_ && vision_->release(false) && model_->reserve(s, need)) return true;  // earlier requests' image embeddings
  std::vector<Slot*> others;
  for (auto& o : slots_)
    if (&o != &slot && o.seq->cap > 0) others.push_back(&o);
  std::sort(others.begin(), others.end(), [](const Slot* a, const Slot* b) { return a->last_used < b->last_used; });
  for (Slot* o : others) {
    logi("memory: dropping the KV cache of slot %d (%d tokens cached) for a %d-token context in slot %d",
         int(o - slots_.data()), int(o->tokens.size()), need, int(&slot - slots_.data()));
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

bool Engine::make_room(Slot& keep, size_t bytes) {
  Gpu& g = Gpu::get();
  auto fits = [&] { return g.allocated_bytes() + bytes <= budget_; };
  if (fits()) return true;
  std::vector<Slot*> others;
  for (auto& o : slots_)
    if (&o != &keep && o.seq->cap > 0) others.push_back(&o);
  std::sort(others.begin(), others.end(), [](const Slot* a, const Slot* b) { return a->last_used < b->last_used; });
  for (Slot* o : others) {
    logi("memory: dropping the KV cache of slot %d (%d tokens cached) for an image", int(o - slots_.data()),
         int(o->tokens.size()));
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

std::string Engine::render(const ChatRequest& r, bool add_generation_prompt) const {
  const bool images = vision_ != nullptr;  // image parts render as the model's placeholder, else as a note
  if (nh_) {
    // Nemotron 3.5's template names two switches its own way: medium_effort (shorter reasoning) for the low and medium
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
  return render_chat(*tmpl_, r, add_generation_prompt, images);
}

std::string render_chat(const minja::chat_template& tmpl, const ChatRequest& r, bool add_generation_prompt,
                        bool images) {
  minja::chat_template_inputs in;
  int dropped = 0;
  in.messages = images ? vision::without_image_data(r.messages) : without_images(r.messages, &dropped);
  if (dropped && add_generation_prompt) logi("chat: %d image(s) left out: this model can't read images", dropped);
  // A system message after the first message (an agent's hook output, a mode change) is a system turn where it
  // stands: the Qwen3.5+ templates raise on it. It goes through the template as a marker tool response (not a user
  // query, so which assistant turns keep their reasoning does not change), and the marker's turn is then replaced.
  std::vector<std::string> later;
  static const std::string mark = "\x02kapitsa-system-turn-";
  for (size_t i = 1; i < in.messages.size(); ++i) {
    ojson& m = in.messages[i];
    if (role_of(m) != "system") continue;
    later.push_back(content_text(m.contains("content") ? m["content"] : ojson()));
    m = {{"role", "user"}, {"content", "<tool_response>" + mark + std::to_string(later.size() - 1) + "\x02</tool_response>"}};
  }
  in.tools = r.tools.empty() ? ojson() : r.tools;
  in.add_generation_prompt = add_generation_prompt;
  in.extra_context = r.template_kwargs;
  minja::chat_template_options opt;
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

Engine::Slot& Engine::pick_slot(const std::vector<int>& prompt, uint64_t arrival, const vision::Prompt* mm) {
  auto lcp = [&](const Slot& s) { return common_rows(s, prompt, mm); };
  // What a slot can actually reuse: its live state if the prompt extends it, else its newest
  // checkpoint inside the common prefix. A shared prefix without a usable state is worthless
  // and choosing that slot would only destroy its conversation.
  auto usable = [&](const Slot& s) -> int {
    size_t l = lcp(s);
    if (l == s.tokens.size() && l < prompt.size()) return int(l);
    int u = 0;
    for (auto& c : s.ckpts)
      if (c.pos > u && c.pos <= int(l) && c.pos < int(prompt.size())) u = c.pos;
    return u;
  };
  Slot* best = nullptr;
  int best_lcp = 0;
  for (auto& s : slots_) {
    if (int(prompt.size()) + 64 > s.seq->limit) continue;
    int u = usable(s);
    if (!best || u > best_lcp) {
      best = &s;
      best_lcp = u;
    }
  }
  if (!best) return slots_[0];
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
      if (int(prompt.size()) + 64 > s.seq->limit) continue;
      if (!pick || rank(s) < rank(*pick) || (rank(s) == rank(*pick) && s.last_used < pick->last_used)) pick = &s;
    }
    if (pick) {
      pick->turns = 1;
      return *pick;
    }
  }
  ++best->turns;
  return *best;
}

ChatResult Engine::chat(const ChatRequest& r, const std::function<bool(const ChatDelta&)>& on_delta,
                        const std::function<void(const ChatResult&)>& on_prompt) {
  const uint64_t arrival = ++requests_;
  // images: fetched, decoded and resized before the engine lock (a slow download holds up no other request)
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
  std::lock_guard<std::mutex> lk(mu_);
  // the cache of image embeddings goes back to its size once this request is done with them
  struct EndImages {
    vision::Vision* v;
    ~EndImages() {
      if (v) v->end_request();
    }
  } end_images{images.empty() ? nullptr : vision_.get()};
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
  Slot& slot = pick_slot(keys, arrival, mmp);
  Seq& s = *slot.seq;
  slot.last_used = ++clock_;
  ++slot.gen;
  if (n + 1 >= s.limit) {
    res.finish_reason = "error";
    res.error = "prompt (" + std::to_string(n) + " tokens) exceeds the context window (" + std::to_string(s.limit) + ")";
    res.prompt_limit = s.limit - 2;
    return res;
  }

  // ---- prefix reuse ----
  const size_t common = common_rows(slot, keys, mmp);

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
    if (common == slot.tokens.size() && int(common) < n) from = int(common);
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
  // other slots' caches are given up only when the prompt itself does not fit; before them this slot's own, when
  // the prompt reuses none of it (no live state to continue, no checkpoint inside the common prefix: the prefill
  // starts at 0), so the new cache starts empty: nothing to copy, no transient second copy
  const int kv_min = std::min(s.limit, n + 32), kv_want = std::min(s.limit, n + std::clamp(r.max_tokens, 0, 1024) + 32);
  auto fit_kv = [&] { return model_->reserve(s, kv_want) || model_->reserve(s, kv_min); };
  bool kv_ok = fit_kv();
  bool reuses = common > 0 && common == slot.tokens.size() && int(common) < n;
  for (auto& c : slot.ckpts) reuses = reuses || (c.pos > 0 && c.pos <= int(common) && c.pos < n);
  if (!kv_ok && !reuses && s.cap > 0) {
    drop_slot(slot);
    kv_ok = fit_kv();
  }
  if (!kv_ok && !reserve_kv(slot, kv_min)) {
    const size_t per = model_->kv_bytes_per_pos(), used = g.allocated_bytes();
    const size_t fit = size_t(s.cap) + (budget_ > used ? budget_ - used : 0) / per;
    res.finish_reason = "error";
    res.error = "insufficient memory: a " + std::to_string(n) + "-token prompt needs more KV cache than this Mac's "
                "memory budget for Kapitsa holds with this model (" + std::to_string(fit) + " tokens)";
    res.prompt_limit = std::max(0, int(fit) - 32);
    return res;
  }
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
      if (lili_) lili_->encode_copy(*e, *slot.lckpts[size_t(best - slot.ckpts.data())], *slot.lseq);
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
  if (lili_) {
    if (start == 0) LiliCorr::reset(*slot.lseq);
    else LiliCorr::rewind(*slot.lseq, start);
  }
  for (auto& c : slot.ckpts)
    if (c.pos > start) c.pos = -1;
  res.cached_tokens = start;
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
  const double tok_ms = now_ms() - t_req - img_ms;  // template + tokenizer + slot pick (image encoding: logged apart)
  const AneStats ane0 = model_->ane_stats();  // the Neural Engine's busy time during this prefill (log)
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
  if (lili_) lili_->begin_prefill(n, start);
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
    if (p < n) e = g.begin();
  }
  e->wait();
  s.embeds = nullptr;
  const double h_woke = Encoder::host_ms();
  res.prefill_ms = now_ms() - t0;
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
    return finish.empty();
  };
  int fed_until = n;  // exclusive: tokens whose effect is in the live state
  int req_cycles = 0;  // DFlash2 cycles of this request
  const int depth = spec_ ? opts_.spec_depth : 0;
  if (lili_) {
    // LiLiCorr: per cycle the drafter's context commit of the previous cycle's kept rows and the drafting pass (7
    // proposals after the anchor tokens[P]), then the target's verify of the 8 rows (capturing the drafter's
    // features), per-row sampling and the acceptance on the GPU; the host reads the accepted count.
    LiliSeq& d = *slot.lseq;
    constexpr int B = LiliCorr::kB;
    int P = n, m_prev = 0;
    req_cycles = 0;
    bool go = handle(s.tokens()[n]);
    while (go) {
      if (P + 2 * B + 2 > s.cap && s.cap < s.limit) {
        if (mtp_e) mtp_e->wait();
        reserve_kv(slot, std::min(s.limit, P + 2 * B + 2));
      }
      if (P + B + 1 >= s.cap) {
        finish = "length";
        break;
      }
      // two command buffers: the drafting half starts on the GPU while the host encodes the verify half
      auto e1 = g.begin();
      if (m_prev > 0) lili_->encode_commit(*e1, d, m_prev);
      lili_->encode_draft(*e1, s, d, P);
      e1->commit();
      auto e2 = nh_->grouped_passes() ? g.begin_grouped() : g.begin();
      lili_->arm_verify(P);
      nh_->set_dynamic_rows(true);  // the drafter's pass chose how many of the block's rows to verify
      nh_->encode_verify(*e2, s, P, B);
      lili_->disarm();
      nh_->encode_accept(*e2, s, P, B, B - 1, sp);
      nh_->set_dynamic_rows(false);
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
      nh_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new anchor
      m_prev = m;
      P += m;
    }
    auto e3 = g.begin();
    if (m_prev > 0) lili_->encode_commit(*e3, d, m_prev);
    nh_->encode_flush(*e3, s);
    e3->commit_and_wait();
    fed_until = P;
  } else if (dfl_) {
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
    // KV cache room for the first cycles (it grows between cycles below): nothing may use the old buffers,
    // so the deferred prompt work completes first
    if (go && P + 8 + 8 + DFlash2::kB + 3 > s.cap && s.cap < s.limit) {
      if (mtp_e) mtp_e->wait();
      reserve_kv(slot, std::min(s.limit, P + 8 + 8 + DFlash2::kB + 3));
    }
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
        res.context_full = true;
        break;
      }
      next_draft.reset();
      auto e2 = g.begin();
      dfl_->encode_cycle_verify(*e2, s, P, K, sp);
      e2->commit();
      if (P + 8 + 8 + DFlash2::kB + 3 > s.cap && s.cap < s.limit) {
        // the KV cache grows here: the verify in flight (and the drafting half before it) complete first, so
        // this cycle's drafting half is not overlapped with it
        e2->wait();
        reserve_kv(slot, std::min(s.limit, P + 8 + 8 + DFlash2::kB + 3));
      }
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
      q35_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new anchor
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
    q35_->encode_flush(*e3, s);
    e3->commit_and_wait();
    fed_until = P;
  } else if (depth > 0) {
    // Speculative decoding: the MTP head drafts `depth` tokens, the trunk verifies them together
    // with the pending token, accepted rows are committed through the DeltaNet tape.
    int P = n, c = 1, h_row0 = last_chunk_row;
    bool go = handle(s.tokens()[n]);
    while (go) {
      if (P + depth + 2 >= s.cap) {
        // the KV cache grows between cycles; only the prompt's deferred MTP history can still be in flight
        if (mtp_e) mtp_e->wait();
        if (P + depth + 2 >= s.limit || !reserve_kv(slot, std::min(s.limit, P + depth + 3))) {
          finish = "length";
          res.context_full = true;
          break;
        }
      }
      auto e2 = g.begin();
      q35_->encode_mtp(*e2, s, P - c, c, P - c + 1, q35_->hidden_rows(), h_row0, P + 1);
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
      q35_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new pending token
      c = m;
      h_row0 = 0;
      P += m;
    }
    // finish: history rows for the last committed rows, then fold the tape into the state
    auto e3 = g.begin();
    if (c > 0 && P - c >= 0) q35_->encode_mtp(*e3, s, P - c, c, P - c + 1, q35_->hidden_rows(), h_row0, -1);
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
    if (r.max_tokens > 1) next = encode_at(pos);
    while (true) {
      bool go = handle(s.tokens()[pos]);  // available: produced by the previous command buffer
      if (go && pos + 2 >= s.cap) {
        // the KV cache grows between steps: the step in flight writes it, so it completes first
        if (next) next->wait();
        if (pos + 2 >= s.limit || !reserve_kv(slot, std::min(s.limit, pos + 3))) {
          finish = "length";
          res.context_full = true;
          go = false;
        }
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
  if (parser.promoted() > 0)
    logi("chat: %d tool call(s) written inside the reasoning returned as tool calls", parser.promoted());
  slot.tokens.assign(s.tokens(), s.tokens() + fed_until);
  if (mmp) std::copy(keys.begin(), keys.end(), slot.tokens.begin());  // image rows: their keys
  res.completion_tokens = gen;
  if ((dfl_ || lili_) && spec_cycles_ > 0)
    logd("%s: %.2f tokens/cycle over %ld cycles", lili_ ? "lilicorr" : "dflash", double(spec_accepted_) / spec_cycles_,
         spec_cycles_);
  else if (depth > 0 && spec_cycles_ > 0)
    logd("spec: %.2f accepted/drafted over %ld cycles", double(spec_accepted_) / (spec_cycles_ * depth), spec_cycles_);
  res.decode_ms = now_ms() - t1;
  if (finish == "stop" && !res.tool_calls.empty()) finish = "tool_calls";
  res.finish_reason = finish;
  char cyc[64] = "";
  if ((dfl_ || lili_) && req_cycles > 0) snprintf(cyc, sizeof cyc, ", %.2f tok/cycle", double(gen - 1) / req_cycles);
  if (mmp)
    logi("chat: %zu image(s), %d tokens: %d encoded in %.0f ms, %d from earlier embeddings, %d in the cached prefix "
         "(+%.0f ms decode/resize before the request)", mm.spans.size(),
         [&] { int t = 0; for (auto& sp : mm.spans) t += sp.n; return t; }(), img_encoded, img_ms, img_reused,
         int(mm.spans.size()) - img_encoded - img_reused, img_prep_ms);
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
  const size_t cut = mpos == std::string::npos ? std::string::npos : text.rfind("<|im_start|>", mpos);
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
  if (n <= 0 || int(common) >= n || common == slot.tokens.size()) return;
  const TurnCheckpoint* best = nullptr;
  for (auto& c : slot.ckpts)
    if (c.pos > 0 && c.pos <= int(common) && c.pos < n && (!best || c.pos > best->pos)) best = &c;
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
    LiliCorr::rewind(*slot.lseq, start);
    lili_->begin_prefill(n, start);
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
  logi("history: prefilled %d tokens of the rendered reply (from %d) in %.0f ms", n - start, start, now_ms() - t0);
}

} // namespace ni
