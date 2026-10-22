// Chat engine: prompt rendering (Jinja chat template), prefix reuse across turns, generation,
// and streaming output parsing (reasoning / content / tool calls) for Qwen3.5+ models (MLX safetensors, Qwen3.8-Flash-Next
// included), Nemotron-H models (NVIDIA's safetensors) and Recursive's Flint (MLX safetensors).
#pragma once
#include "engine/grammar.h"
#include "engine/prefix_cache.h"
#include "hub/hub.h"
#include "model/dflash2.h"
#include "model/dory.h"
#include "model/flint.h"
#include "model/kflash_budget.h"
#include "model/lilicorr.h"
#include "model/nemotron_h.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "tokenizer/tokenizer.h"
#include "vision/encoder.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace minja {
class chat_template;
}

namespace ni {

using ojson = nlohmann::ordered_json;

struct EngineOptions {
  std::string model_dir;
  std::string model_id;        // id reported by /v1/models (default: directory name)
  int context = 32768;         // capacity of each slot (tokens)
  // requests that run at once, each on a slot with its own KV cache and state that keeps its conversation warm
  // (more wait for a free one); 0 = auto_slots()
  int slots = 0;
  int prefill_chunk = 0;       // rows per prefill pass; 0 = 2048, or 1024 when that leaves too little memory
  int spec_depth = -1;         // MTP draft tokens per cycle (0 = no speculative decoding, -1 = model default)
  double mem_budget_gb = 0;    // GPU memory Kapitsa may hold (GB); 0 = default_mem_budget()
  // DFlash2 block drafter (replaces MTP when loaded): "auto" = DFlash2::find() next to the model / in
  // ~/.cache/nace-infer-models/drafters (MTP when none matches), "none" = MTP, or a drafter directory.
  // "" = NI_DFLASH if set (auto | none | 0 | <dir>), else auto. spec_depth 0 turns "auto" off too.
  std::string dflash;
  // Images (models with a vision tower): at most this many pixels per image (fewer tokens, faster; resized as the
  // model's processor resizes); vision::kModelDefaultPixels = the encoder's default (1.2 million; Nemotron 3.5 Bolt
  // and Super-VL 1.4 million), 0 = the model's own limit (Qwen3.5-4B, Qwen3.6, Qwen3.8: 16.7 million, Bolt and
  // Super-VL: 13312 patches of 16x16).
  long image_max_pixels = vision::kModelDefaultPixels;
  // Looped models (Dory): the recurrent block's loop count when a request names none (0: the model's default, 2 for
  // Dory 1.0; requests choose 1-4 with "loops" or a reasoning effort), and the recurrent layers' KV ("": last-loop)
  int loops = 0;
  std::string loop_kv;
  // SSD prefix cache (src/engine/engine_disk.cpp): conversations' states kept on the SSD across restarts. "auto" (`kapitsa
  // serve`'s default: a quarter of the free space beyond 10 GiB, at most 32 GiB), "0" off, or a size ("16G"). "" =
  // NI_DISK_CACHE if set, else off (tools that construct an engine write nothing unless asked).
  std::string disk_cache;
};

// The model's memory plan from its headers (config.json and the safetensors headers): Qwen3.5 family, Qwen3.8-Flash-Next,
// Nemotron-H or Dory.
MemPlan model_plan(const std::string& model_dir, int prefill_chunk);

// Slots when none are asked for (EngineOptions::slots 0): kAutoSlots, so that an agent's conversation and its parallel
// subagents or side requests each run on a warm slot of their own, or as many (down to 2, the old default) as the
// model's memory plan, the engine's own, holds without costing anything two slots have: one full --ctx conversation
// beside them, the same decoding (drafter) and the same prefill chunk; 1 where two slots cannot start (a conversation
// of the context, up to 8K tokens, with 256-row prefill buffers) and one can. From the safetensors headers alone (the engine,
// `kapitsa serve` and `kapitsa doctor` share it).
constexpr int kAutoSlots = 4;
int auto_slots(const std::string& model_dir, const EngineOptions& opt, size_t budget);
// The slots a fit check counts before the model is known: the ones asked for, else 2 (auto never takes fewer).
inline int fit_slots(int slots) { return slots > 0 ? slots : 2; }

// GPU memory Kapitsa holds at most by default: Metal's recommended working set for this GPU
// (recommendedMaxWorkingSetSize, which macOS sizes from the RAM) minus a margin (4%, at least 512 MB) for
// the Neural Engine service's buffers and other GPU users. NI_MEM_BUDGET=<GB> overrides it.
size_t default_mem_budget();
// The prefill chunk (rows) the engine picks when none is given: 2048, or 1024 when 2048-row buffers would leave
// less than one conversation of `context` tokens in the budget (activations scale with the chunk), less (a multiple of
// 64, at least 256) when 1024-row ones would leave less than the tokens the engine starts with (the context, up to 8K)
int auto_prefill_chunk(const std::string& model_dir, int context, int slots, size_t budget);
// Rows of the next prefill chunk with `p` of `n` rows done: min(chunk, n - p), or with `even` those rows split evenly
// over the chunks that rule would take (a 4130-token prompt at a 2048 chunk is 2048+2048+34 plain, 2065+2065 even), so
// a prompt a few rows past a multiple of the chunk pays no extra pass. Never more than `chunk` rows, so the planned
// buffers still hold every chunk; `even` is off for a model whose prefill a Neural Engine split runs (it needs chunks
// of at most FfnSplit::kMaxRows rows), see Model::ane_split_on.
int prefill_chunk_rows(int chunk, int64_t p, int64_t n, bool even);
// Rows the prefill buffers plan for: `chunk` plus its tail room (2112 for 2048), so an even chunk still fits them
int prefill_chunk_room(int chunk);
// NI_BAL (default on; 0 = the chunking before tail room and even chunks), read once
bool prefill_even_chunks();

// What the engine settles before it loads a model, from the safetensors headers alone (its constructor and
// `kapitsa doctor` share it): the DFlash2 drafter and the prefill chunk.
struct LoadPlan {
  bool dflash_auto = false;       // the drafter is looked for (--dflash auto)
  bool dflash_off = false;        // --dflash none, or auto with --spec 0
  std::string dflash;             // the drafter to load ("" = none: MTP or plain decoding)
  std::string dflash_left_out;    // the drafter auto found and leaves out (it would cost context, below)
  size_t dflash_bytes = 0;        // the loaded drafter's weights and rings (live + one per checkpoint, every slot)
  size_t dflash_found_bytes = 0;  // the same for the drafter found, loaded or left out
  int prefill_chunk = 2048;
};
// dflash: "auto", "none" / "0" / "off", a drafter directory, or "" (NI_DFLASH if set, else auto); spec_off: --spec 0.
// auto leaves a drafter out where one conversation of `context` tokens would not fit the budget beside it (context
// before speed); prefill_chunk 0 = auto_prefill_chunk with the drafter's memory set aside.
LoadPlan plan_load(const std::string& model_dir, const std::string& dflash, bool spec_off, int context, int slots,
                   size_t budget, int prefill_chunk);
// What the engine needs to start a model (it refuses to load it past the budget): the weights, the MTP head, the
// drafter plan_load keeps, prefill buffers for `prefill_chunk` rows, the slots' state and an 8k-token conversation
// (min(context, 8192)). From config.json and the safetensors headers alone, so `kapitsa serve` can check a model
// before downloading it (hub::plan_dir).
struct StartNeed {
  size_t bytes = 0;
  std::string parts;  // what each part needs, for the "insufficient memory" message
  size_t weights = 0;   // the weights alone
  std::string formats;  // their formats ("bf16", "4-bit g64 + 8-bit g64" ...)
};
StartNeed start_need(const std::string& model_dir, const LoadPlan& lp, int prefill_chunk, int context, int slots);
// One conversation of `context` tokens fits within `budget` beside the model (1024-row prefill buffers, the slots'
// state, the Neural Engine split): plan_load's test before it keeps a drafter, and the test `kapitsa serve` without
// --model applies to pick the best model for this Mac.
bool conversation_fits(const std::string& model_dir, int context, int slots, size_t budget);
// Where conversation_fits does not hold, whether the conversation fits the way the engine then starts the model with
// `slots` (0: the slots it picks; more than one: never): one slot (auto_slots) and the prefill chunk auto_prefill_chunk
// takes below 1024 rows. Flint's 16gb on a 16 GB Mac: 448 rows and an 8K conversation.
bool small_start_fits(const std::string& model_dir, int context, int slots, size_t budget);
// A catalog variant on this Mac, from the engine's plan of its headers (its folder when downloaded, else
// hub::plan_dir's): 1 when one conversation of opt.context tokens fits beside it within `budget` and its DFlash2
// drafter (if it has one) is kept, 0 when not, -1 when it cannot be planned (not downloaded and offline).
int variant_fits(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget);
// The variant `--model <name>` takes on this Mac: the catalog's default where it fits, else the first of the model's
// fallbacks that fits, else the smallest that could be planned (the engine then says what it needs); offline with
// nothing planned, a downloaded one or the default. *fits: whether the returned variant fits.
hub::ModelRef pick_variant(const hub::CatalogModel& m, const EngineOptions& opt, size_t budget, bool* fits = nullptr);

// What `--context` takes: tokens ("131072"), k or m of them as model cards count them (1024 and 1,048,576 tokens:
// "128k", "1m"), or max (kContextMax): the model's own limit. 0, with *why, when it is none of those.
constexpr int kContextMax = -1;
int parse_context(const std::string& s, std::string* why);
// The longest context the model was trained for: max_position_embeddings in its config.json (262,144 tokens for the
// Qwen3.5 family, Dory and Bolt, 1,048,576 for Lightning and Super-VL); 0 when it does not say.
int model_context_limit(const std::string& model_dir);
// `context` within the model's limit, if it states one
int context_within_limit(const std::string& model_dir, int context);
// The context `kapitsa serve` and `kapitsa claude|codex|opencode|pi` take when none is given: kDefaultContext (128K),
// or the model's limit where that is less, or less where this Mac's memory is tight: the largest, in 8K steps down to
// kMinDefaultContext (32K), that keeps the drafter the model has at 32K, with one full conversation beside it (below
// 32K, in 8K steps and 1K steps below 8K, only for a model one slot cannot hold 32K tokens in: Flint with 2-bit experts
// on 16 GB, 8K). The
// model and its precision are picked at 32K, and a context costs no memory until a conversation fills it (the KV
// cache grows with it). *why, if given: why it is less than kDefaultContext.
constexpr int kDefaultContext = 131072, kMinDefaultContext = 32768;
enum class ContextWhy { kDefault, kModelLimit, kMemory, kMemoryWithDrafter };
int default_context(const std::string& model_dir, const EngineOptions& opt, size_t budget, ContextWhy* why = nullptr);

struct ChatRequest;
// The model's chat template: <dir>/chat_template.jinja, else tokenizer_config.json's chat_template ("" if none)
std::string chat_template_source(const std::string& model_dir);
// A request's prompt by a chat template (Engine::render; CPU only, no model needed: tools/api_test checks it).
// `images`: image parts render as images (their data left out; the engine puts the images' rows in place), else image
// and video parts become a short note. `native`: the template renders every role and tool call itself (Flint's), so
// no minja polyfills and system messages after the first stay where they are.
std::string render_chat(const minja::chat_template& tmpl, const ChatRequest& r, bool add_generation_prompt,
                        bool images = false, bool native = false);
// Flint's chat template rewritten for minja (engine.cpp)
std::string flint_template(std::string src);

// One request's progress while it runs, for a live display (`kapitsa serve`'s dashboard in a terminal): the caller
// passes it in ChatRequest::live and reads it from any thread; the engine writes it with relaxed atomic stores between
// decode cycles, never on the GPU's path.
struct LiveDecode {
  std::atomic<int> phase{0};               // 0 waiting for the engine, 1 reading the prompt, 2 writing, 3 done
  std::atomic<int> prompt{0}, cached{0};   // prompt tokens, of which reused from the cache
  std::atomic<int> read{0};                // prompt tokens read so far: the cached ones, then each prefill chunk the
                                           // GPU has finished
  std::atomic<int> tokens{0};              // tokens written
  std::atomic<int> cycles{0};              // decode cycles: a speculative cycle's verify, or a plain step
  std::atomic<int> last{0};                // tokens the last cycle wrote (its accepted drafts + 1)
  std::atomic<int> spec{0};                // 0 plain decoding, 1 MTP, 2 DFlash2, 3 LiLiCorr
};

struct ChatRequest {
  ojson messages = ojson::array();
  ojson tools = ojson::array();
  ojson template_kwargs = ojson::object(); // e.g. enable_thinking, reasoning_effort
  int max_tokens = 8192;
  float temperature = 0.6f;
  float top_p = 0.95f;
  int top_k = 20;
  uint32_t seed = 0;
  std::vector<std::string> stop;
  std::shared_ptr<LiveDecode> live;  // the request's progress for a display, or null
  // Looped models (Dory): the loop count asked for ("loops" in the body, 0: none), else derived from the reasoning
  // effort (OpenAI's reasoning_effort / reasoning.effort: minimal and low 1, medium 2, high and xhigh 4; none, which
  // only turns thinking off, the default) or Anthropic's thinking (disabled 1, a budget by its size); "" / -1: the
  // server's default
  int loops = 0;
  std::string effort;
  int thinking_budget = -1;     // Anthropic thinking.budget_tokens; 0: thinking disabled
  // Structured output (src/engine/grammar.h: a JSON schema, any JSON object, tool calls the request forces or strict
  // tools): what the reply must follow after its reasoning, or null. The engine samples each token under its mask.
  std::shared_ptr<const Grammar> grammar;
};

struct ToolCall {
  std::string id, name;
  std::string arguments; // JSON object text
};

// Tags the model writes as text (a tokenizer without <think> / tool-call tokens: Nemotron Nano 9B v2's), which the
// output parser matches in the decoded text instead of by token id.
struct TextTags {
  bool think = false;             // <think> / </think>
  std::string tc_open, tc_close;  // a tool call's tags, e.g. <TOOLCALL> / </TOOLCALL> (empty: none)
  bool json_calls = false;        // the call's body: a JSON list of {"name": ..., "arguments": ...} (Nano 9B v2's)
  bool any() const { return think || !tc_open.empty(); }
};

// Nemotron Nano 9B v2's chat conventions (CPU only; tools/nh_chat_check checks them): a template that switches
// thinking with "/think" / "/no_think" in a message and reads no enable_thinking; the request with its enable_thinking
// as that flag at the end of the last user message (else the first system message), or as it is without one; the tags
// a tokenizer has no tokens for, found in the template (<think>, <TOOLCALL>), which the output parser then matches in
// the text.
bool think_flag_template(const std::string& template_source);
ChatRequest think_flag_request(const ChatRequest& r);
TextTags text_tags_for(const Tokenizer& tok, const std::string& template_source);

struct ChatDelta {
  std::string reasoning;
  std::string content;
  std::vector<ToolCall> tool_calls; // complete calls, emitted when their closing tag arrives
};

struct ChatResult {
  std::string reasoning, content;
  std::vector<ToolCall> tool_calls;
  std::string finish_reason;  // stop | length | tool_calls | cancelled | error
  std::string error;
  std::string stop_sequence;  // "stop" at one of the request's stop strings: that string
  bool context_full = false;  // "length" because the context window (or the memory for it) ran out, not max_tokens
  int prompt_limit = 0;       // "error" because the prompt does not fit: the most prompt tokens that would
  bool invalid_request = false;  // "error" because of the request's content (an image that cannot be read, a video)
  int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
  int reasoning_tokens = 0;   // completion tokens inside the think block
  double prefill_ms = 0, decode_ms = 0;
  // engine bookkeeping for prefill_history: the slot, its use count and the request count at the reply, the
  // request's images (what the processor made of them, without pixels)
  int slot = -1;
  uint64_t slot_gen = 0, req_seq = 0;
  std::shared_ptr<const std::vector<vision::Prepared>> images;
  // concurrency: time this request waited for a free slot, and the GPU turns it gave to other requests
  double queue_ms = 0;
  int yields = 0;
  int batched = 0;  // decode cycles it ran in one pass with other requests' (batched decode)
  int loops = 0;    // looped models: the loop count the request ran (0: not a looped model)
};

// The engine's GPU turn: one request at a time encodes and runs model work. FIFO (a ticket lock): a request that gives
// up its turn between two steps of its own (prefill chunks, decode cycles) goes behind every request waiting, so the
// requests in flight take turns step by step. Events (a slot or memory freed, a request ended) wake requests that wait
// for one without holding the turn.
class TurnLock {
 public:
  void lock();
  void unlock();
  bool try_lock();  // the turn now when nobody holds it or waits for it (idle-time work: history prefill)
  // requests waiting for the turn (read by its holder between steps)
  int waiting() const { return waiting_.load(std::memory_order_relaxed); }
  // how long the holder has held the turn (ms; read by the holder)
  double held_ms() const;
  // events: the count (read while holding the turn); notify() adds one (while holding it)
  uint64_t events() const;
  void notify();
  // Gives up the turn until events() != seen, or until `stop` (polled every second without the turn) says so, then
  // waits for it again (behind the requests already waiting). True if an event happened.
  bool wait_event(uint64_t seen, const std::function<bool()>& stop = nullptr);

 private:
  mutable std::mutex m_;
  std::condition_variable cv_, ev_cv_;
  uint64_t next_ = 0, serving_ = 0, events_ = 0;
  std::atomic<int> waiting_{0};
  std::atomic<double> since_{0};  // when the holder took the turn (host ms)
};

class Engine {
 public:
  explicit Engine(const EngineOptions& o);
  ~Engine();
  const std::string& model_id() const { return opts_.model_id; }
  int context_length() const { return opts_.context; }
  int slots() const { return int(slots_.size()); }  // requests that run at once
  // Positions one conversation can hold in memory (the context, or less when the memory budget ends first)
  int context_fit() const { return context_fit_; }
  // The sampling settings a request gets when it names none: the model's (Nemotron-H: generation_config.json),
  // else Qwen's recommended 0.6 / 0.95 / 20.
  ChatRequest request_defaults() const { return defaults_; }
  // Renders the prompt for a request (for debugging / token counting). With an image encoder, image parts render as
  // the model's image placeholder (one token per image: chat() expands it to the image's rows); without one, image and
  // video parts become a short note. A system message after the first message becomes a system turn in place (the
  // Qwen3.5+ templates accept one only at the start).
  std::string render(const ChatRequest& r, bool add_generation_prompt = true) const;
  // The prompt's length in tokens, as chat() would prefill it (images: their rows; they are fetched and decoded for
  // their size, and a bad one throws vision::Error). Does not wait for a running request.
  int count_tokens(const ChatRequest& r) const;
  // Runs one request. `on_delta` returns false to cancel. Requests in flight at once each take a slot of their own and
  // take turns on the GPU between prefill chunks and decode cycles (one request alone runs as if it were the only
  // one); with every slot busy a request waits for one, first come first served. `on_prompt`, if given, is called once
  // the prompt is tokenized and matched against the slot's cache, before the prefill, with prompt_tokens and
  // cached_tokens set (not when the request fails before that). `cancelled`, if given, is polled while the request
  // waits for a slot or for memory (a client gone before its first token). `on_accepted`, if given, is called earlier,
  // once the prompt is rendered, tokenized and within the context (prompt_tokens set), before the request waits for the
  // GPU or a slot; returning false gives the request up ("cancelled").
  ChatResult chat(const ChatRequest& r, const std::function<bool(const ChatDelta&)>& on_delta,
                  const std::function<void(const ChatResult&)>& on_prompt = nullptr,
                  const std::function<bool()>& cancelled = nullptr,
                  const std::function<bool(const ChatResult&)>& on_accepted = nullptr);
  // History prefill (NI_HISTORY_PREFILL): after a reply, prefills the conversation as the next request
  // will render it (the template drops this turn's think block, so the generated tokens in the live
  // state do not match it) from the turn-boundary checkpoint, so that a follow-up prefills only its
  // own new messages. Call it from its own thread once the reply has been sent: it first waits until
  // the server has been idle for NI_HISTORY_IDLE_MS (default 250 ms) and gives up if any request
  // arrived meanwhile. Only replies that end with "stop" (a user query follows; after tool calls the
  // template keeps the think block and the live state already matches the next request).
  // With DFlash2 the prefilled rows also enter the slot's drafter context ring, as prompt rows do.
  void prefill_history(const ChatRequest& r, const ChatResult& res);
  // KFlash: the budgets' cost tables and calibrations (KfBudget) to their cache files now. They are saved when the
  // engine ends and every KfBudget::kSaveEvery cycles between requests; a server stopped by a signal ends no
  // destructor, so it calls this. From any thread, with requests running.
  void save_kflash();
  // SSD prefix cache: the conversations of idle slots that are not on the SSD yet, written now (the newest first, for
  // at most budget_ms), as the server stops; the cache's files are then closed. From any thread; once.
  void save_disk_cache(double budget_ms = 8000);

 private:
  // Recurrent-state snapshot at a turn boundary. `query`: the request that saved it ended with a user
  // query (not a tool response); those are kept through tool loops (see Engine::chat).
  struct TurnCheckpoint : Checkpoint {
    bool query = false;
    // The slot's prefix checkpoint (the last one of ckpts): the state at the end of the system block, saved while a
    // prompt is read through it and never overwritten by a turn boundary, so a new question that shares the system
    // prompt and tools restores it however early it diverges. It has no drafter snapshot (the ring starts empty).
    bool prefix = false;
  };
  struct Slot {
    std::unique_ptr<Seq> seq;
    std::unique_ptr<DFlashSeq> dseq;  // drafter context ring (DFlash2 only)
    std::unique_ptr<LiliSeq> lseq;    // drafter context (LiLiCorr only)
    std::vector<std::unique_ptr<LiliSeq>> lckpts;   // LiLiCorr: the drafter context as it was at ckpts[i]
    std::vector<int> tokens;             // tokens whose effect is in the live state (image rows: their keys)
    std::vector<vision::ImageSpan> img_spans;  // the images among them
    std::vector<vision::Digest> img_digests;
    std::vector<TurnCheckpoint> ckpts;   // recurrent-state snapshots
    std::vector<std::unique_ptr<DFlashSeq>> dckpts;  // DFlash2: the drafter ring as it was at ckpts[i]
    uint64_t last_used = 0;
    uint64_t gen = 0;                 // requests run on this slot
    int turns = 0;                    // requests of the conversation it holds (1: a one-off, never continued)
    uint64_t done_at = 0;             // requests_ when its last request ended: one that arrived by then was sent beside it
    int loops = 0;                    // looped models: the loop count its cached state (KV) was computed with
    bool busy = false;                // a request runs on it (its state and KV cache are that request's)
    uint64_t arrival = 0;             // that request's number (requests_ when it arrived)
    // that request waits for memory other requests' KV caches hold: before its prefill (nothing of it on the GPU
    // yet: a request that has started may take its cache, as may an earlier one; it then reads its prompt from the
    // start), or for its KV cache to grow while decoding (its cache stays)
    enum class MemWait : uint8_t { none, before_prefill, growth } mem_wait = MemWait::none;
    bool decoding = false;            // that request is in its decode loop
    Buffer park_h;                    // MTP decoding: the hidden rows the next cycle reads, while other requests run
    // SSD prefix cache (engine_disk.cpp). fills: which computation wrote the KV rows, rows [previous end, end) by the
    // request (or history prefill) `id`, so that what the SSD has of a conversation is found again by its key;
    // version: bumped by every change of the state; disk_version: the version the SSD has (or nothing worth saving);
    // live_ok: a follow-up continues the live state as it is (after tool calls, or once the history is prefilled)
    std::vector<std::pair<int32_t, uint64_t>> fills;
    uint64_t version = 0, disk_version = 0;
    bool live_ok = false;
  };
  // A free slot for a request (nullptr: every slot is busy). `arrival`: the request's number (requests_ when it
  // arrived); mm: the prompt's images (null for text), when `prompt` holds its prefix-reuse keys
  Slot* pick_slot(const std::vector<int>& prompt, uint64_t arrival, const vision::Prompt* mm = nullptr, int loops = 0);
  // Rows of `keys` (a prompt's prefix-reuse keys; mm: its images, null for text) that slot's tokens share as a
  // reusable prefix: the common prefix, cut before any image that is not the same image in both
  static size_t common_rows(const Slot& slot, const std::vector<int>& keys, const vision::Prompt* mm);
  // Frees memory until `bytes` more fit the budget: idle slots' KV caches (least recently used first), then the
  // Neural Engine split. False if that is not enough.
  bool make_room(Slot& keep, size_t bytes);
  // Grows slot's KV cache to `need` positions; when the memory budget is short, frees the caches of idle
  // slots (least recently used first: their conversations are prefilled again when they come back), then those of
  // requests waiting for memory before their prefill (the newest first): every such request's when `started` (this
  // slot's request is decoding: an answer in progress comes first, as when requests ran one at a time and the
  // others waited without a slot), later ones' otherwise. Nothing using those slots may be in flight. False if even
  // that leaves too little memory.
  bool reserve_kv(Slot& slot, int need, bool started);
  // reserve_kv, and while that fails, waits (without the turn) for memory that other requests in flight hold: a
  // request that has started (`started`; the caller has parked its work: nothing of its own in flight, nothing it
  // still needs in the model's shared buffers) while another one runs; one before its prefill also while a started or
  // an earlier request waits for memory. False when it cannot fit (for one before its prefill: not even alone, as the
  // others are later requests whose caches it took), or cancelled.
  bool reserve_kv_wait(Slot& slot, int need, const std::function<bool()>& cancelled, bool started);
  // Time to give the GPU turn away: another request waits for it and (`quantum`) this one has held it for NI_TURN_MS
  // (default 30: two of the 35B's ~19 ms decode cycles, one of the 27B's or Bolt's; each turn given away costs ~0.6 ms
  // of an idle GPU). NI_PARK_TEST=1: at every step, even alone (the parked path's exactness check).
  bool contended(bool quantum = true) const;
  // Gives the GPU turn to the requests waiting for it and waits for it again. The caller has drained its own GPU work
  // and keeps nothing it needs in the model's shared buffers (its state is in its slot).
  void yield_turn(ChatResult& res);
  // gives up a slot's conversation and its KV cache: the recurrent state back to zero (a later prompt on the slot
  // starts at position 0 from the live state), no tokens, no checkpoints
  void drop_slot(Slot& slot);

  // ---- SSD prefix cache (engine_disk.cpp) ----
  struct Disk;      // the store, its writer thread and what it was handed
  struct DiskPlan;  // what saving one slot writes
  struct DiskFree {
    void operator()(Disk* d) const;
  };
  std::unique_ptr<Disk, DiskFree> disk_;  // null: off (or the model's states are not cached on the SSD)
  void disk_start();
  // The rows [0, start) of slot's state a request reuses from the slot itself (live state or checkpoint), as chat()
  // decides it below the lookup
  static int reuse_start(const Slot& slot, size_t common, int n);
  // A request is about to replace the slot's state from row `start` (another conversation, or memory for another):
  // what it would lose and the SSD lacks is copied now and written in the background
  void disk_leaving(Slot& slot, int start);
  // The deepest stored state matching keys[0, pos), pos < n, deeper than `ram` (null hit: none); tag: what else the
  // state must match (Dory: the request's loop count); mm: the prompt's images (null for text), the same as stored
  pcache::Hit disk_lookup(const std::vector<int>& keys, int n, int ram, uint32_t tag, const vision::Prompt* mm);
  // Reads a stored state into slot (KV rows, recurrent state, drafter ring) and makes it the slot's live state; false
  // (the slot then holds nothing) when it cannot. With the GPU turn held (given up while reading).
  bool disk_restore(Slot& slot, const pcache::Hit& hit);
  // rows [start, end) of slot were computed now (by a request or a history prefill, which is done)
  void disk_filled(Slot& slot, int start, int end);
  std::unique_ptr<DiskPlan> disk_plan(const Slot& slot) const;
  // Writes slot i's state if it changed since it was last saved. held: the caller holds the GPU turn throughout;
  // otherwise each chunk is copied under a turn taken while the engine is idle, and the save stops when a request
  // arrives (what it wrote stays for the next save to reuse). False: given up.
  bool disk_save_slot(int i, bool held);
  void disk_loop();

  // ---- batched decode: the decode cycles of several requests in one forward pass ----
  // A decoding request (DFlash2 or LiLiCorr) joins the batch at a cycle boundary, parked, when another request decodes
  // and batch_rows() says a pass over their rows pays; one member's thread (the runner, holding the GPU turn) runs the
  // batch's cycles for all of them: each member's staged ring rows committed, its drafting pass (rows > 1), then one
  // verify pass of every member's rows (Qwen35 / NemotronH::encode_verify_multi), the acceptance (greedy and LiLiCorr
  // members keep a draft row while it equals the target's sample at the row before it; sampled DFlash2 members use the
  // single-stream cycle's exact speculative acceptance) and each member's tokens through its own handle(). A
  // member that stops leaves (its thread runs its request's end); when one member is left it goes back to its own loop
  // (returned). With a single request decoding none of this runs: its path is the single-stream one.
  struct BatchJob {
    Slot* slot = nullptr;
    SamplingParams sp;
    int P = 0;  // anchor: tokens[P] is the next row to feed (the rows before it are in the state or pending)
    const std::function<bool(int)>* handle = nullptr;    // the request's handle(): false once it stops
    const std::function<void(int)>* cycle_done = nullptr;
    // NI_TRACE_TOKENS: the logit rows of the tokens a cycle wrote (rows, count, the tokens); null when off
    const std::function<void(const float*, int, const int*)>* trace = nullptr;
    enum class State : uint8_t { active, done, returned } state = State::active;
    bool runner = false;
    bool stopped = false;  // its handle() said stop in the last cycle (it leaves as done)
    int cycles = 0, tokens = 0;
    // structured output: the request's grammar and its state there (its chat()'s, advanced by its handle()); while
    // *g_on its rows sample under masks the runner computes (null: no grammar)
    const GrammarMatcher* gm = nullptr;
    GrammarState* gst = nullptr;
    const bool* g_on = nullptr;
    bool masked() const { return gm && *g_on; }
  };
  // Structured output: a verify's masks, Seq::mask rows 0 .. rows - 1, from the grammar's state after the anchor
  // tokens[P]: row r's is the state after row r's path (parent[r]: a tree's, null: a chain's, r - 1; rows from `live` on
  // are never accepted). A row whose path the grammar does not allow is never accepted either (its parent's masked
  // sample is not its token): any mask.
  static void verify_masks(const GrammarMatcher& gm, const GrammarState& at, Seq& s, int P, int rows, const int* parent,
                           int live);
  // rows per member in a batch of n members (sum <= 8): 1 = plain rows, 1 + k = the anchor and k drafted tokens; 0:
  // n requests take turns instead (one pass over their rows does not pay, or the drafter was not measured batched);
  // kBatchBudget: KFlash's batch budget (KfBatch, DFlash2) chooses each member's rows every cycle from the members'
  // drafts. NI_BATCH=0 turns batching off, NI_BATCH=1 on with every drafter; NI_BATCH_ROWS="n:rows,..." overrides
  // the table and the budget; NI_BATCH_KF=0 keeps the table where the budget would choose.
  static constexpr int kBatchBudget = 9;
  int batch_rows(int n) const;
  // at a decode cycle boundary: join the batch (one is running, or another request decodes and batching pays)
  bool batch_wanted() const;
  // the caller holds the turn and has parked (nothing in flight, its drafter's ring holds the rows before job.P); returns
  // with the turn once the job is done (true: the request stopped) or returned (false: back to its own loop at job.P)
  bool join_batch(BatchJob& job);
  void run_batch(BatchJob& me);
  // one cycle of every member: rows per member, or kBatchBudget (each member drafts its chain, the budget then gives the
  // members their rows: the verify is encoded once the drafting passes are done)
  void batch_cycle(int rows);
  // plain rows, pipelined: up to 8 passes, each encoded and committed before the host reads the one before it
  void batch_plain_run();
  void batch_stats();
  // j leaves the batch: its staged rows into its drafter's ring, its state, its thread woken
  void batch_leave(BatchJob& j, BatchJob::State st);
  void wait_split(Encoder& e);

  EngineOptions opts_;
  std::unique_ptr<Model> model_;
  Qwen35* q35_ = nullptr;    // model_ when it is a Qwen3.5-family model (MTP, DFlash2, Neural Engine split; Qwen4Exp too)
  NemotronH* nh_ = nullptr;  // model_ when it is a Nemotron-H model
  Dory* dory_ = nullptr;     // model_ when it is a Dory model
  Flint* flint_ = nullptr;   // model_ when it is a Flint model
  int flint_ngram_ = 0;      // Flint: prompt-lookup drafts per cycle (0: plain decoding)
  int dory_spec_ = 0;        // Dory speculative decoding (per-loop KV): drafts per cycle (0: plain decoding)
  bool dory_spec_auto_ = false;  // drafts per cycle by the request's loops (no --spec)
  std::unique_ptr<Checkpoint> ane_ck_;  // NI_ANE: recurrent state before a split chunk (rerun on failure)
  std::unique_ptr<DFlash2> dfl_;
  std::string dfl_dir_;              // its folder
  std::unique_ptr<NhDrafter> lili_;  // Nemotron-H: the block drafter (LiLiCorr, DSpark)
  // KFlash on the Nemotron-H block drafters (on dense targets; MoE targets keep the chain unless NI_KFLASH=1; src/model/kflash_budget.h): best-first trees of the drafter's
  // lattice under a measured budget instead of its walk's chain
  std::unique_ptr<KfBudget> kf_budget_;
  Buffer kf_params_;  // KfParams the budget writes for the GPU tree builder
  std::unique_ptr<vision::Vision> vision_;  // the image encoder, for models that take images
  std::vector<int> img_pre_, img_post_;     // the tokens its processor puts around an image's rows (Nemotron: <img>)
  // `ids` with each image placeholder expanded to its image's rows (vision::expand with the encoder's marks)
  vision::Prompt expand_images(const std::vector<int>& ids, const std::vector<vision::Prepared>& images,
                               int* index = nullptr) const;
  Tokenizer tok_;
  std::unique_ptr<minja::chat_template> tmpl_;
  std::vector<Slot> slots_;
  TurnLock turn_;            // the GPU turn: engine state and the model are used by its holder only
  int active_ = 0;           // requests holding a slot (under the turn)
  int decoding_ = 0;         // of those, requests in their decode loop (under the turn)
  std::vector<BatchJob*> batch_;  // the batch's members (under the turn)
  bool batch_runner_ = false;     // a member's thread runs the batch
  bool batch_measured_ = false;   // batched decode measured with this model's drafter (batch_rows' default)
  double batch_backoff_ = 0;      // now_ms() before which no batch forms (the last one's members did not all come)
  // batch statistics since the last log line: cycles, rows, wall and GPU ms of the cycles, the GPU's idle time between
  // the drafting passes and the verify (cycles with drafting passes)
  long bstat_cycles_ = 0, bstat_rows_ = 0, bstat_drafted_ = 0;
  double bstat_wall_ = 0, bstat_gpu_ = 0, bstat_gap_ = 0;
  std::mutex bmu_;                // a waiting member's job state / runner flag
  std::condition_variable bcv_;
  std::deque<uint64_t> slot_queue_;  // requests waiting for a free slot, first come first served (under the turn)
  uint64_t clock_ = 0;
  std::atomic<uint64_t> requests_{0};  // requests started (prefill_history waits for an idle server)
  int think_open_ = -1, think_close_ = -1, tc_open_ = -1, tc_close_ = -1;
  TextTags text_tags_;        // the tags written as text (no such tokens; Nemotron Nano 9B v2)
  bool think_flags_ = false;  // the template switches thinking by "/think" / "/no_think" in a message (Nano 9B v2)
  bool think_ws_ = true;      // the whitespace before </think> stays in the reasoning (Nemotron-H, Dory)
  int nh_ngram_ = 0;          // a dense Nemotron-H model without a block drafter: prompt-lookup drafts per cycle
  std::vector<int> stop_ids_;  // the config's eos ids plus the tokenizer's eos token (chat turn end)
  bool spec_ = false;
  size_t budget_ = 0;    // GPU memory budget (bytes)
  int context_fit_ = 0;  // see context_fit()
  bool prefix_copy_ = true;   // a slot without the prefix checkpoint copies another's (NI_PREFIX_COPY=0: reads it)
  bool prefix_ckpt_ = false;  // slots keep a prefix checkpoint (TurnCheckpoint::prefix)
  size_t fixed_bytes_ = 0;  // allocated at start (model, buffers, slots' state): context_fit_'s base
  long spec_cycles_ = 0, spec_accepted_ = 0;
  ChatRequest defaults_;
  // Structured output: the vocabulary as grammars read it (built by the first request with a grammar, before its GPU
  // turn), and the event a pass's sampler waits on until the host has written the pass's masks (one value per pass,
  // in the order the passes run; under the turn)
  std::shared_ptr<const TokenTable> token_table();
  std::shared_ptr<const TokenTable> table_;
  std::mutex table_mu_;
  void* mask_event_ = nullptr;
  uint64_t mask_value_ = 0;

 public:
  // Looped models: the loop count a request runs (its "loops", else its effort or thinking budget mapped, else the
  // server's default); 0 for other models
  int request_loops(const ChatRequest& r) const;
};

} // namespace ni
