// Chat engine: prompt rendering (Jinja chat template), prefix reuse across turns, generation,
// and streaming output parsing (reasoning / content / tool calls) for Qwen3.5+ models.
#pragma once
#include "hub/hub.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "tokenizer/tokenizer.h"
#include "vision/encoder.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <memory>
#include <atomic>
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
  int slots = 2;               // conversations kept warm at once (each has its own KV/state)
  int prefill_chunk = 0;       // rows per prefill pass; 0 = 2048, or 1024 when that leaves too little memory
  int spec_depth = -1;         // MTP draft tokens per cycle (0 = no speculative decoding, -1 = model default)
  double mem_budget_gb = 0;    // GPU memory Kapitsa may hold (GB); 0 = default_mem_budget()
  // DFlash2 block drafter (replaces MTP when loaded): "auto" = DFlash2::find() next to the model / in
  // ~/.cache/nace-infer-models/drafters (MTP when none matches), "none" = MTP, or a drafter directory.
  // "" = NI_DFLASH if set (auto | none | 0 | <dir>), else auto. spec_depth 0 turns "auto" off too.
  std::string dflash;
  // Images (models with a vision tower): at most this many pixels per image (fewer tokens, faster; resized as the
  // model's processor resizes); 0 = the model's own limit.
  long image_max_pixels = 0;
};

// GPU memory Kapitsa holds at most by default: Metal's recommended working set for this GPU
// (recommendedMaxWorkingSetSize, which macOS sizes from the RAM) minus a margin (4%, at least 512 MB) for
// the Neural Engine service's buffers and other GPU users. NI_MEM_BUDGET=<GB> overrides it.
size_t default_mem_budget();
// The prefill chunk (rows) the engine picks when none is given: 2048, or 1024 when 2048-row buffers would leave
// less than one conversation of `context` tokens in the budget (activations scale with the chunk)
int auto_prefill_chunk(const std::string& model_dir, int context, int slots, size_t budget);

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
// A catalog variant on this Mac, from the engine's plan of its headers (its folder when downloaded, else
// hub::plan_dir's): 1 when one conversation of opt.context tokens fits beside it within `budget` and its DFlash2
// drafter (if it has one) is kept, 0 when not, -1 when it cannot be planned (not downloaded and offline).
int variant_fits(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget);
// The variant `--model <name>` takes on this Mac: the catalog's default where it fits, else the first of the model's
// fallbacks that fits, else the smallest that could be planned (the engine then says what it needs); offline with
// nothing planned, a downloaded one or the default. *fits: whether the returned variant fits.
hub::ModelRef pick_variant(const hub::CatalogModel& m, const EngineOptions& opt, size_t budget, bool* fits = nullptr);

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
};

struct ToolCall {
  std::string id, name;
  std::string arguments; // JSON object text
};

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
  int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
  double prefill_ms = 0, decode_ms = 0;
  // engine bookkeeping for prefill_history: the slot, its use count and the request count at the reply, the
  // request's images (what the processor made of them, without pixels)
  int slot = -1;
  uint64_t slot_gen = 0, req_seq = 0;
  std::shared_ptr<const std::vector<vision::Prepared>> images;
};

class Engine {
 public:
  explicit Engine(const EngineOptions& o);
  ~Engine();
  const std::string& model_id() const { return opts_.model_id; }
  int context_length() const { return opts_.context; }
  // Positions one conversation can hold in memory (the context, or less when the memory budget ends first)
  int context_fit() const { return context_fit_; }
  // Renders the prompt for a request (for debugging / token counting).
  std::string render(const ChatRequest& r, bool add_generation_prompt = true) const;
  // Runs one request. `on_delta` returns false to cancel. Serialized across threads.
  ChatResult chat(const ChatRequest& r, const std::function<bool(const ChatDelta&)>& on_delta);
  // History prefill (NI_HISTORY_PREFILL): after a reply, prefills the conversation as the next request
  // will render it (the template drops this turn's think block, so the generated tokens in the live
  // state do not match it) from the turn-boundary checkpoint, so that a follow-up prefills only its
  // own new messages. Call it from its own thread once the reply has been sent: it first waits until
  // the server has been idle for NI_HISTORY_IDLE_MS (default 250 ms) and gives up if any request
  // arrived meanwhile. Only replies that end with "stop" (a user query follows; after tool calls the
  // template keeps the think block and the live state already matches the next request).
  // With DFlash2 the prefilled rows also enter the slot's drafter context ring, as prompt rows do.
  void prefill_history(const ChatRequest& r, const ChatResult& res);

 private:
  // Recurrent-state snapshot at a turn boundary. `query`: the request that saved it ended with a user
  // query (not a tool response); those are kept through tool loops (see Engine::chat).
  struct TurnCheckpoint : Checkpoint {
    bool query = false;
  };
  struct Slot {
    std::unique_ptr<Seq> seq;
    std::unique_ptr<DFlashSeq> dseq;  // drafter context ring (DFlash2 only)
    std::vector<int> tokens;             // tokens whose effect is in the live state (image rows: their keys)
    std::vector<vision::ImageSpan> img_spans;  // the images among them
    std::vector<vision::Digest> img_digests;
    std::vector<TurnCheckpoint> ckpts;   // recurrent-state snapshots
    std::vector<std::unique_ptr<DFlashSeq>> dckpts;  // DFlash2: the drafter ring as it was at ckpts[i]
    uint64_t last_used = 0;
    uint64_t gen = 0;                 // requests run on this slot
    int turns = 0;                    // requests of the conversation it holds (1: a one-off, never continued)
  };
  Slot& pick_slot(const std::vector<int>& prompt, const vision::Prompt* mm = nullptr);
  // Rows of `keys` (a prompt's prefix-reuse keys; mm: its images, null for text) that slot's tokens share as a
  // reusable prefix: the common prefix, cut before any image that is not the same image in both
  static size_t common_rows(const Slot& slot, const std::vector<int>& keys, const vision::Prompt* mm);
  // Frees memory until `bytes` more fit the budget: other slots' KV caches (least recently used first), then the
  // Neural Engine split. False if that is not enough.
  bool make_room(Slot& keep, size_t bytes);
  // Grows slot's KV cache to `need` positions; when the memory budget is short, frees the caches of other
  // slots (least recently used first: their conversations are prefilled again when they come back).
  // Nothing using the slots may be in flight. False if even that leaves too little memory.
  bool reserve_kv(Slot& slot, int need);
  // gives up a slot's conversation and its KV cache: the recurrent state back to zero (a later prompt on the slot
  // starts at position 0 from the live state), no tokens, no checkpoints
  void drop_slot(Slot& slot);

  EngineOptions opts_;
  std::unique_ptr<Qwen35> model_;
  std::unique_ptr<Checkpoint> ane_ck_;  // NI_ANE: recurrent state before a split chunk (rerun on failure)
  std::unique_ptr<DFlash2> dfl_;
  std::unique_ptr<vision::Vision> vision_;  // the image encoder, for models that take images
  Tokenizer tok_;
  std::unique_ptr<minja::chat_template> tmpl_;
  std::vector<Slot> slots_;
  std::mutex mu_;
  uint64_t clock_ = 0;
  std::atomic<uint64_t> requests_{0};  // requests started (prefill_history waits for an idle server)
  int think_open_ = -1, think_close_ = -1, tc_open_ = -1, tc_close_ = -1;
  std::vector<int> stop_ids_;  // the config's eos ids plus the tokenizer's eos token (chat turn end)
  bool spec_ = false;
  size_t budget_ = 0;    // GPU memory budget (bytes)
  int context_fit_ = 0;  // see context_fit()
  size_t fixed_bytes_ = 0;  // allocated at start (model, buffers, slots' state): context_fit_'s base
  long spec_cycles_ = 0, spec_accepted_ = 0;
};

} // namespace ni
