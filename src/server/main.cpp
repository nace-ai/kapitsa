// kapitsa: Kapitsa's command line, and its server: OpenAI chat completions and Responses, Anthropic Messages
// (streaming, reasoning, tools; the translation is in src/api/), and a chat page at /.
//   kapitsa serve [--model NAME | org/name[@rev] | DIR] [--model-id ID] [--host 127.0.0.1] [--port 8000]
//                 [--ctx 32768] [--slots N] [--spec N] [--dflash auto|none|<drafter dir>] [--prefill-chunk 2048]
//                 [--api-key K] [--mem-budget GB] [--image-max-pixels N]
//                 no --model: the best catalog model for this Mac's memory
//   kapitsa claude|codex|opencode [--model M] [--port P] [--ctx N] [--api-key K] [client args]   (launch.cpp)
//   kapitsa doctor [--model MODEL]... [--ctx N] [--slots N] [--dflash auto|none|<dir>] [--json]   this Mac, what fits
//   kapitsa pull MODEL... [--from DIR]... [--dry-run [--json]]
//   kapitsa list [--json]
//   kapitsa rm MODEL...
//   kapitsa mtp-sidecar SRC OUT [--revision REV] [--hidden post_norm|pre_norm]   (tools/mtp_sidecar.py, same bytes)
//   kapitsa import FILE.gguf [--name NAME] [--mtp FILE] [--threads N] [--force]   (src/gguf/import.h)
//   kapitsa bench [--model M] [--quick] [--json] [--submit [--dry-run]] [--out FILE]; kapitsa bench submit [FILE]; kapitsa bench compare
//                 (bench_cmd.cpp: the harness's protocol on this Mac, its result, the community results)
//   kapitsa update [--check] [--version X.Y.Z] [--rollback] [--auto on|off]   (update_cmd.cpp; and once a day, in the
//                 background, the automatic update)
// ("serve" is optional for options-only command lines; build/nace-infer-server is a symlink kept for old scripts.)
#include <cstring>
#include "api/anthropic.h"
#include "api/chat_completions.h"
#include "api/responses.h"
#include "engine/doctor.h"
#include "engine/engine.h"
#include "gguf/gguf.h"
#include "gguf/import.h"
#include "hub/hub.h"
#include "metal/gpu.h"
#include "server/cli.h"
#include "server/model_manager.h"
#include "server/picker.h"
#include "server/ui.h"
#include "util/log.h"
#include "util/term.h"

#include <httplib/httplib.h>

#include <fcntl.h>
#include <netdb.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>
#include <csignal>
#include <ctime>

#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.2.7"
#endif
#ifndef KAPITSA_GIT
#define KAPITSA_GIT ""
#endif

namespace ni {
extern const char* const kChatPage;  // src/server/chat.html, embedded at build time
}

const char* const ni::kVersion = KAPITSA_VERSION;

using namespace ni;
using json = nlohmann::ordered_json;

static json normalize_messages(const json& in) {
  json out = json::array();
  for (auto m : in) {
    if (m.value("role", "") == "developer") m["role"] = "system";
    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
      for (auto& tc : m["tool_calls"]) {
        if (tc.contains("function") && tc["function"].contains("arguments") && tc["function"]["arguments"].is_string()) {
          try {
            tc["function"]["arguments"] = json::parse(tc["function"]["arguments"].get<std::string>());
          } catch (...) {
          }
        }
      }
    }
    if (m.contains("content") && m["content"].is_null()) m["content"] = "";
    out.push_back(m);
  }
  return out;
}

static ChatRequest parse_request(const json& b, const ChatRequest& defaults) {
  ChatRequest r = defaults;
  r.messages = normalize_messages(b.at("messages"));
  if (b.contains("tools") && b["tools"].is_array()) r.tools = b["tools"];
  if (b.contains("tool_choice") && b["tool_choice"].is_string() && b["tool_choice"] == "none") r.tools = json::array();
  r.max_tokens = b.value("max_completion_tokens", b.value("max_tokens", 16384));
  if (b.contains("temperature") && b["temperature"].is_number()) r.temperature = b["temperature"];
  if (b.contains("top_p") && b["top_p"].is_number()) r.top_p = b["top_p"];
  if (b.contains("top_k") && b["top_k"].is_number()) r.top_k = b["top_k"];
  if (b.contains("seed") && b["seed"].is_number()) r.seed = uint32_t(b["seed"].get<int64_t>());
  if (b.contains("stop")) {
    if (b["stop"].is_string()) r.stop.push_back(b["stop"]);
    else if (b["stop"].is_array())
      for (auto& s : b["stop"]) r.stop.push_back(s);
  }
  if (b.contains("chat_template_kwargs") && b["chat_template_kwargs"].is_object())
    r.template_kwargs = b["chat_template_kwargs"];
  if (b.contains("reasoning_effort") && b["reasoning_effort"].is_string()) {
    // none/minimal: thinking off; low/medium: the template's level; high/xhigh/max: its default (Qwen3.8's highest;
    // it raises on "high")
    const std::string e = b["reasoning_effort"];
    if (e == "none" || e == "minimal" || !r.template_kwargs.contains("reasoning_effort")) api::apply_effort(e, r.template_kwargs);
    r.effort = e;  // looped models: the loop count (Engine::request_loops)
  }
  r.loops = api::request_loops_field(b);
  // structured output: response_format, tool_choice required or named, strict tools (src/api/chat_completions.cpp)
  r.grammar = api::chat_grammar(b, r.tools);
  return r;
}

// ---------------------------------------------------------------- command line

[[noreturn]] void ni::usage_error(const std::string& msg, const char* help) {
  if (term::fancy()) term::error(msg, std::string(help) + " shows how to use it.");
  else fprintf(stderr, "kapitsa: %s\n(%s shows how to use it.)\n", msg.c_str(), help);
  exit(2);
}

int ni::fail(const std::string& msg, int code) {
  if (term::fancy()) term::error(msg);
  else fprintf(stderr, "kapitsa: %s\n", msg.c_str());
  return code;
}

// A download stopped by Ctrl+C: what to run to continue it
static int stopped(const hub::Stopped& e) {
  if (term::fancy()) term::status(""), ui::warning(e.what());
  else fprintf(stderr, "\n%s\n", e.what());
  return 130;
}

int ni::to_int(const std::string& opt, const std::string& v, const char* help) {
  char* end = nullptr;
  errno = 0;
  const long x = strtol(v.c_str(), &end, 10);
  if (v.empty() || *end || errno || x < 0 || x > INT32_MAX) usage_error(opt + " needs a whole number, not \"" + v + "\"", help);
  return int(x);
}

std::string ni::expand_home(const std::string& p) {
  const char* h = getenv("HOME");
  if (h && (p == "~" || p.rfind("~/", 0) == 0)) return h + p.substr(1);
  return p;
}

double ni::ram_gb() {
  uint64_t m = 0;
  size_t n = sizeof m;
  sysctlbyname("hw.memsize", &m, &n, nullptr, 0);
  return double(m) / 1073741824.0;
}

static double to_gb(const std::string& opt, const std::string& v, const char* help) {
  char* end = nullptr;
  const double x = strtod(v.c_str(), &end);
  if (v.empty() || *end || !(x > 0)) usage_error(opt + " needs a size in GB, not \"" + v + "\"", help);
  return x;
}

// The order best_model tries the catalog in: the most capable model first (the catalog itself lists the models from
// small to large, Qwen first, as the README and the picker show them); catalog models not named here come last.
static std::vector<const hub::CatalogModel*> by_preference() {
  static const char* const kOrder[] = {"qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.5-9b", "qwen3.5-4b", "nemotron-3.5-bolt",
                                       "nemotron-3.5-super-vl", "nemotron-3.5-lightning"};
  std::vector<const hub::CatalogModel*> out;
  for (const char* n : kOrder)
    if (const hub::CatalogModel* m = hub::find_model(n)) out.push_back(m);
  for (const auto& m : hub::catalog())
    if (std::find(out.begin(), out.end(), &m) == out.end()) out.push_back(&m);
  return out;
}

// The best catalog model for this Mac: the first in by_preference's order with a variant whose memory plan, the
// engine's own, leaves room for one conversation of opt.context tokens (at most the model's limit) within the budget
// and keeps its drafter (each model takes its default variant, else the first of its fallbacks that fits:
// pick_variant), or failing that whose smallest variant holds the conversation without the drafter (MTP decoding).
// A model not downloaded yet is planned from its headers (hub::plan_dir: a few hundred KB, fetched once); one that
// cannot be planned (offline) cannot be downloaded either and is skipped. False if none can be planned.
bool ni::best_model(const EngineOptions& opt, size_t budget, hub::ModelRef* out, std::string* why) {
  bool have = false;
  for (const hub::CatalogModel* mp : by_preference()) {
    const hub::CatalogModel& m = *mp;
    bool fits = false;
    hub::ModelRef r = pick_variant(m, opt, budget, &fits);
    if (fits) {
      *out = r;
      return true;
    }
    if (variant_fits(r, opt, budget) != 0) continue;  // not plannable (offline)
    // its smallest variant without the drafter: one conversation still fits (MTP decoding instead)
    try {
      const std::string d = hub::plan_dir(r);
      if (conversation_fits(d, context_within_limit(d, opt.context), fit_slots(opt.slots), budget)) {
        *out = r;
        return true;
      }
    } catch (const std::exception&) {
    }
    *out = r, have = true;  // too big: the smallest such is the last resort
  }
  if (!have && why) *why = "Can't reach huggingface.co to see which model fits this Mac (offline?), and none is "
                           "downloaded. Connect to the internet, or pass --model with a model folder.";
  return have;
}

// --model <name> without a variant: the one for this Mac (pick_variant); anything else as resolved.
hub::ModelRef ni::for_this_mac(hub::ModelRef ref, const EngineOptions& opt, size_t budget) {
  if (ref.kind != hub::ModelRef::kCatalog || ref.variant_given) return ref;
  return pick_variant(*ref.model, opt, budget);
}

__attribute__((format(printf, 1, 2))) static std::string strf(const char* fmt, ...) {
  va_list ap, aq;
  va_start(ap, fmt);
  va_copy(aq, ap);
  std::string s(size_t(std::max(0, vsnprintf(nullptr, 0, fmt, aq))), '\0');
  va_end(aq);
  vsnprintf(s.data(), s.size() + 1, fmt, ap);
  va_end(ap);
  return s;
}

// A model's name in a note of the terminal UI
static std::string bold_name(const std::string& s) { return term::bold(s); }

// A command's help: as it is, or styled in a terminal
static void show_help(const std::string& text) {
  if (term::fancy()) ui::help(text);
  else fputs(text.c_str(), stdout);
}

static std::string models_table() {
  std::string s;
  char line[256];
  for (const auto& m : hub::catalog()) {
    const hub::Variant* d = m.variant();
    snprintf(line, sizeof line, "  %-16s %-6s %5.1f GB  %s\n", m.name.c_str(), ("(" + hub::short_name(m) + ")").c_str(),
             d ? d->download_gb : 0.0, m.about.c_str());
    s += line;
    std::string more;
    for (const auto& v : m.variants)
      if (&v != d) {
        snprintf(line, sizeof line, "%s:%s %.1f GB", more.empty() ? "" : ", ", v.name.c_str(), v.download_gb);
        more += line;
      }
    if (d) s += "                           " + d->name + " (" + d->quant + ")";
    if (!m.fallbacks.empty()) {
      std::string fb;
      for (const auto& f : m.fallbacks) fb += (fb.empty() ? "" : ", then ") + f;
      s += "; smaller Macs: " + fb;
    }
    if (d) s += "\n";
    if (!more.empty()) s += "                           also " + more + "\n";
  }
  return s;
}

static void print_help() {
  if (term::fancy()) {
    ui::quiet();
    return ui::home(KAPITSA_VERSION);
  }
  printf("Kapitsa %s: fast local Qwen on Apple silicon.\n\n"
         "  kapitsa serve                 start the best model for this Mac (downloads it the first time)\n"
         "  kapitsa serve --model 27b     start a given model\n\n"
         "Then open http://127.0.0.1:8000 to chat, or point any OpenAI- or Anthropic-compatible app at\n"
         "http://127.0.0.1:8000/v1. Or start a coding agent on it in one command:\n\n"
         "  kapitsa claude                Claude Code\n"
         "  kapitsa codex                 Codex\n"
         "  kapitsa opencode              opencode\n"
         "  kapitsa pi                    Pi\n\n"
         "Models (with the files that speed them up; <model>:<variant> picks another precision):\n%s"
         "  or a Hugging Face repo of a Qwen3.5-family model in MLX format, bf16 or 3-8-bit (org/name[@revision]),\n"
         "  or a model folder, or a .gguf file (converted once into a model folder).\n\n"
         "Commands:\n"
         "  serve        start the server: chat page at /, OpenAI and Anthropic APIs at /v1 (kapitsa serve --help)\n"
         "  claude, codex, opencode, pi [--model M] [--port P] [--ctx N] [--api-key K] [agent args]\n"
         "               start that coding agent on Kapitsa: on the server already running on the port, else on one\n"
         "               started for it (with a context sized for agents; kapitsa claude --help)\n"
         "  pull MODEL   download a model without starting it (kapitsa pull --help)\n"
         "  list         downloaded models, their size and what they include\n"
         "  rm MODEL     delete a downloaded model\n"
         "  import FILE  convert a GGUF (LM Studio, Ollama, llama.cpp) into a model (kapitsa import --help)\n"
         "  doctor       this Mac's GPU and memory, and which models fit it (kapitsa doctor --help)\n"
         "  bench        benchmark Kapitsa (and other engines: bench compare) on this Mac (kapitsa bench --help)\n"
         "  update       update Kapitsa to its latest release; it also updates itself (kapitsa update --help)\n\n"
         "Models are stored in %s (KAPITSA_HOME moves them). HF_TOKEN is used when set.\n",
         KAPITSA_VERSION, models_table().c_str(), hub::tilde(hub::models_dir()).c_str());
}

static void print_serve_help() {
  std::string h = strf("kapitsa serve [--model MODEL] [options]\n\n"
         "  --model MODEL          35b, 27b, 4b (full names: kapitsa --help; <model>:<variant> for another\n"
         "                         precision), org/name[@revision], a model folder, or a .gguf file (converted\n"
         "                         once: kapitsa import). Default: the best model for this Mac. Downloaded on first use.\n"
         "  --host HOST            listen address (default 127.0.0.1; 0.0.0.0 serves your network)\n"
         "  --port PORT            default 8000\n"
         "  --api-key KEY          require \"Authorization: Bearer KEY\" or \"x-api-key: KEY\" (the chat page asks for\n"
         "                         it once)\n"
         "  --allow-switch         a request naming another model, or POST /v1/models/switch, loads it (may download\n"
         "                         gigabytes; 503 + Retry-After meanwhile). Default off: the \"model\" field is ignored.\n"
         "                         On a network address it needs --api-key\n"
         "  --model-id ID          the model's name in the API (default: the model's name)\n"
         "  --context N            context per conversation: tokens, k or m of them (128k, 1m), or max (the\n"
         "                         model's limit). Default: 128K, or less where this Mac's memory is tight (the\n"
         "                         most that keeps the draft model). --ctx N works too\n"
         "  --slots N              requests that run at once, each keeping its conversation ready for instant\n"
         "                         follow-ups; more wait for a free slot (default: up to 4, as memory allows)\n"
         "  --dflash auto|none|DIR DFlash2 drafter: auto (default) takes the one downloaded with the model;\n"
         "                         none decodes with the model's MTP head\n"
         "  --spec N               MTP draft depth without a drafter (default per model; 0 = no speculation);\n"
         "                         Dory with --loop-kv per-loop: drafts per cycle (default 2 at 1-2 loops, 3 at 3-4)\n"
         "  --mem-budget GB        GPU memory Kapitsa may hold (default: what macOS recommends for this GPU, a\n"
         "                         little less; kapitsa doctor shows it)\n"
         "  --prefill-chunk N      prompt tokens per prefill pass (default 2048, 1024 when memory is short)\n"
         "  --disk-cache SIZE      SSD prefix cache: conversations kept on the SSD, also across restarts, in\n"
         "                         %s/prefix-cache. auto (default): a quarter of the free space beyond\n"
         "                         10 GiB, at most 32 GiB; a size (16G) caps it; 0 turns it off\n"
         "  --loops N              Dory: the shared block's loops per token, 1-4 (default 2, NVIDIA's recommendation;\n"
         "                         more is latent thinking: slower, more accurate). Requests choose their own with\n"
         "                         \"loops\" or a reasoning effort (low 1, medium 2, high 4)\n"
         "  --loop-kv MODE         Dory: last-loop (default: one cache per layer, the size of a 1-loop model's) or\n"
         "                         per-loop (a cache per loop, as NVIDIA's vLLM commands run it; more memory)\n"
         "  --image-max-pixels N   images: resize larger ones to about N pixels (default 1200000, about what\n"
         "                         Anthropic's API sends Claude: a 2880x1800 screenshot becomes 1376x864, 1.2K\n"
         "                         tokens; Nemotron 3.5 Bolt and Super-VL 1400000, which small text needs there;\n"
         "                         more keeps finer detail and takes longer; 0 = the model's own limit)\n\n"
         "Endpoints: GET / (chat page), POST /v1/chat/completions (OpenAI), POST /v1/responses (OpenAI Responses),\n"
         "POST /v1/messages and /v1/messages/count_tokens (Anthropic), GET /v1/models, GET /health.\n"
         "Images (every catalog model reads them): OpenAI image_url parts, Anthropic image blocks and Responses\n"
         "input_image, as data: (base64) or http(s) URLs. Video is not supported.\n"
         "Tuning switches (NI_* variables): docs/usage.md.\n", hub::tilde(hub::home_dir()).c_str());
  if (term::fancy()) {  // in a terminal the engine's log goes to a file, unless --verbose; without --model it asks
    h.insert(h.find("\n\nEndpoints:"), "\n  --verbose              the engine's log on screen too (in a terminal it goes to " +
                                            hub::tilde(hub::home_dir()) + "/logs/serve-PORT.log)");
    h.insert(h.find("\n\nEndpoints:"), "\n  --yes                  don't ask which model to start (in a terminal, without --model): the best for\n"
                                         "                         this Mac");
  }
  show_help(h);
}

static void print_pull_help() {
  show_help(strf("kapitsa pull MODEL... [--from DIR]...\n\n"
         "Downloads a model with the files that speed it up (MTP head, DFlash2 drafter) into\n"
         "%s/<name>-<variant>, verifies every file, and continues an interrupted download.\n\n"
         "  --from DIR   take files that are already on this Mac from DIR (a folder from `hf download --local-dir`,\n"
         "               another app's models): only exact matches of the pinned checksums, as APFS clones\n"
         "  --dry-run    show what MODEL resolves to, without downloading (--json for programs)\n\n"
         "Files already in the Hugging Face cache (~/.cache/huggingface/hub) are reused the same way.\n",
         hub::tilde(hub::models_dir()).c_str()));
}

// Why the server could not listen on host:port, or "" (a test bind, as the server binds: SO_REUSEADDR only).
std::string ni::port_problem(const std::string& host, int port) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  const std::string p = std::to_string(port);
  if (getaddrinfo(host.c_str(), p.c_str(), &hints, &res) != 0 || !res) return "--host " + host + " is not an address of this Mac.";
  std::string why;
  const int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd >= 0) {
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(fd, res->ai_addr, res->ai_addrlen) != 0) {
      if (errno == EADDRINUSE)
        why = "Port " + p + " is already in use (another kapitsa or another server). Stop it, or pass --port " +
              std::to_string(port + 1) + ".";
      else if (errno == EACCES)
        why = "Not allowed to listen on port " + p + ". Pass a port above 1024, e.g. --port 8000.";
      else if (errno == EADDRNOTAVAIL)
        why = "--host " + host + " is not an address of this Mac.";
      else
        why = "Can't listen on " + host + ":" + p + " (" + strerror(errno) + ").";
    }
    close(fd);
  }
  freeaddrinfo(res);
  return why;
}

// One streamed reply. The engine runs in a thread of its own and hands the writer's events to the connection's thread
// through a queue. So a reply is still a plain HTTP error when the request fails before its stream starts (a prompt the
// context cannot hold: Claude Code, Codex and opencode recover from those errors by compacting). An Anthropic or
// Responses stream starts once the prompt is matched against its slot's cache (its first event carries the cached
// tokens); a Chat Completions stream as soon as the prompt is known to fit, before the request waits for a slot, and
// the engine takes a slot only once that first chunk is out, so a client gone by then costs no prefill. From then on
// the connection gets keepalive events while the engine is quiet (a wait for a slot or for memory, a long prefill:
// clients drop a stream that is idle for minutes), and a client that goes away is noticed within a second.
struct StreamJob {
  std::mutex mu;  // the queue, the flags and every call into the writer
  std::condition_variable cv;
  std::deque<std::string> out;
  bool started = false, sent = false, done = false;  // sent: the first event is out
  std::atomic<bool> cancelled{false};
  ChatResult result;
  std::thread worker;
  void cancel() {  // the client is gone
    std::lock_guard<std::mutex> lk(mu);
    cancelled = true;
    cv.notify_all();
  }
  ~StreamJob() {
    cancel();
    if (worker.joinable()) worker.join();
  }
};

// Runs `r` and answers it as a stream of `w`'s events, or with failure(result) when it fails before the stream starts:
// once the prompt is matched against the slot's cache, or `early`, as soon as the request is accepted. Then calls
// after(result) (the reply is out) from the connection's thread.
static void stream_reply(std::shared_ptr<RequestGuard> guard, const ChatRequest& r, std::shared_ptr<api::StreamWriter> w, httplib::Response& res,
                         std::function<std::pair<int, json>(const ChatResult&)> failure,
                         std::function<void(const ChatResult&)> after, std::shared_ptr<ui::Meter> meter,
                         bool early = false) {
  auto job = std::make_shared<StreamJob>();
  job->worker = std::thread([guard, r, w, meter, early, raw = job.get()] {
    auto start = [&](const ChatResult& prompt) {  // with raw->mu held
      raw->out.push_back(w->begin(prompt));
      raw->started = true;
      raw->cv.notify_all();
    };
    ChatResult out = guard->engine().chat(
        r,
        [&](const ChatDelta& d) {
          ui::token(meter);
          std::lock_guard<std::mutex> lk(raw->mu);
          std::string s = w->delta(d);
          if (!s.empty()) raw->out.push_back(std::move(s));
          raw->cv.notify_all();
          return !raw->cancelled.load();
        },
        [&](const ChatResult& prompt) {
          if (early) return;
          std::lock_guard<std::mutex> lk(raw->mu);
          start(prompt);
        },
        [raw] { return raw->cancelled.load(); },
        [&](const ChatResult& prompt) {
          if (!early) return true;
          std::unique_lock<std::mutex> lk(raw->mu);
          start(prompt);
          // the engine goes on to a slot once the first event is out: no prefill for a client already gone
          raw->cv.wait(lk, [&] { return raw->sent || raw->cancelled.load(); });
          return !raw->cancelled.load();
        });
    std::lock_guard<std::mutex> lk(raw->mu);
    if (raw->started && out.finish_reason != "cancelled") raw->out.push_back(w->end(out));
    raw->result = std::move(out);
    raw->done = true;
    raw->cv.notify_all();
  });
  {
    std::unique_lock<std::mutex> lk(job->mu);
    job->cv.wait(lk, [&] { return job->started || job->done; });
  }
  if (!job->started) {
    job->worker.join();
    ui::done(meter, job->result);
    auto [status, body] = failure(job->result);
    res.status = status;
    res.set_content(api::dump(body), "application/json");
    return;
  }
  res.set_header("Cache-Control", "no-cache");
  res.set_chunked_content_provider("text/event-stream", [guard, job, w, after, meter](size_t, httplib::DataSink& sink) {
    // the client is gone: the engine stops at its next token, or gives the request up before it takes a slot
    auto gone = [&] {
      job->cancel();
      job->worker.join();
      ui::done(meter, job->result);
      return false;
    };
    auto last = std::chrono::steady_clock::now();  // the last write
    bool first = true;
    for (;;) {
      std::deque<std::string> batch;
      bool done;
      {
        std::unique_lock<std::mutex> lk(job->mu);
        // a second at a time while nothing comes, to see whether the client is still there
        job->cv.wait_for(lk, std::chrono::seconds(1), [&] { return !job->out.empty() || job->done; });
        if (job->out.empty() && !job->done && std::chrono::steady_clock::now() - last >= std::chrono::seconds(15))
          job->out.push_back(w->keepalive());  // 15 s without an event
        batch.swap(job->out);
        done = job->done;
      }
      if (batch.empty() && !sink.is_writable()) return gone();
      for (const auto& e : batch)
        if (!sink.write(e.data(), e.size())) return gone();
      if (!batch.empty()) last = std::chrono::steady_clock::now();
      if (!batch.empty() && first) {  // an early stream's engine goes on to a slot now
        first = false;
        std::lock_guard<std::mutex> lk(job->mu);
        job->sent = true;
        job->cv.notify_all();
      }
      if (done) break;
    }
    sink.done();
    job->worker.join();
    ui::done(meter, job->result);
    after(job->result);
    return true;
  });
}

// What the server keeps when it ends: the KFlash budgets' measurements (Engine::save_kflash: this Mac's cost per verify
// row count, the drafter's calibration), which only the engine's destructor would write, and a server never ends by
// returning: a signal stops it (Ctrl+C, SIGTERM from `kapitsa claude` or a service manager, SIGHUP), or its launcher's
// exit (exit_with_parent). The engine also saves them between requests (KfBudget::kSaveEvery), for kill -9.
static std::mutex g_stop_mu;
static ModelManager* g_stop_mgr = nullptr;  // while serve() serves (SaveOnStop)

static void save_before_exit() {
  std::lock_guard<std::mutex> lk(g_stop_mu);
  if (g_stop_mgr) g_stop_mgr->save_state();  // the engine now serving: KFlash state, the SSD prefix cache's conversations
}

// A stop signal wakes a thread that saves, then takes its course as before: the terminal UI's handler (the status
// block off the screen, term::guard_signals), else the default (the process ends as killed by the signal). A second
// one while it saves ends the process at once. Signals ignored when the server started (nohup) stay ignored.
static const int kStopSignals[] = {SIGINT, SIGTERM, SIGHUP};
static struct sigaction g_stop_prev[3];
static int g_stop_pipe[2] = {-1, -1};
static volatile sig_atomic_t g_stopping = 0;

static void pass_signal_on(int sig) {
  for (int i = 0; i < 3; ++i)
    if (kStopSignals[i] == sig) sigaction(sig, &g_stop_prev[i], nullptr);
  raise(sig);  // from a handler: delivered when it returns, to the previous one
}

static void on_stop_signal(int sig) {
  const char c = char(sig);
  const bool again = g_stopping;
  g_stopping = 1;
  if (again || write(g_stop_pipe[1], &c, 1) != 1) pass_signal_on(sig);
}

// The stop handlers while it lives (serve(), from the engine's start); the previous ones come back after
struct SaveOnStop {
  explicit SaveOnStop(ModelManager& mgr) {
    {
      std::lock_guard<std::mutex> lk(g_stop_mu);
      g_stop_mgr = &mgr;
    }
    if (pipe(g_stop_pipe) != 0) return;
    for (int fd : g_stop_pipe) fcntl(fd, F_SETFD, FD_CLOEXEC);
    std::thread([] {
      char c = 0;
      ssize_t r;
      while ((r = read(g_stop_pipe[0], &c, 1)) < 0 && errno == EINTR) {
      }
      if (r != 1) return;
      save_before_exit();
      pass_signal_on(c);  // on this thread: the previous handler runs here, or the default ends the process
    }).detach();
    struct sigaction sa {};
    sa.sa_handler = on_stop_signal;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    for (int i = 0; i < 3; ++i) {
      sigaction(kStopSignals[i], nullptr, &g_stop_prev[i]);
      if (g_stop_prev[i].sa_handler != SIG_IGN) sigaction(kStopSignals[i], &sa, nullptr);
    }
    handlers_ = true;
  }
  ~SaveOnStop() {
    if (handlers_)
      for (int i = 0; i < 3; ++i)
        if (g_stop_prev[i].sa_handler != SIG_IGN) sigaction(kStopSignals[i], &g_stop_prev[i], nullptr);
    std::lock_guard<std::mutex> lk(g_stop_mu);
    g_stop_mgr = nullptr;
  }
  SaveOnStop(const SaveOnStop&) = delete;
  SaveOnStop& operator=(const SaveOnStop&) = delete;

 private:
  bool handlers_ = false;
};

// Started by `kapitsa claude|codex|opencode` (KAPITSA_PARENT_PID): exits when the launcher does, however it ends (kill -9
// included), instead of holding the model's memory with nobody left to stop it.
static void exit_with_parent() {
  const char* v = getenv("KAPITSA_PARENT_PID");
  const pid_t parent = v && *v ? pid_t(atoi(v)) : 0;
  if (parent <= 1) return;
  std::thread([parent] {
    const int kq = kqueue();
    struct kevent ev;
    EV_SET(&ev, parent, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0, nullptr);
    if (kq < 0 || kevent(kq, &ev, 1, nullptr, 0, nullptr) < 0) {
      if (errno == ESRCH) _exit(0);  // already gone
      return;
    }
    kevent(kq, nullptr, 0, &ev, 1, nullptr);
    save_before_exit();
    _exit(0);
  }).detach();
}

// While a model switch loads the new model, a request gets 503 and Retry-After (it does not queue behind the load)
static bool reject_while_switching(const ModelManager& mgr, httplib::Response& res, const std::string& body) {
  if (!mgr.switching()) return false;
  res.status = 503;
  res.set_header("Retry-After", "30");
  res.set_content(body, "application/json");
  return true;
}

static int serve(int argc, char** argv, int first) {
  exit_with_parent();
  EngineOptions opt;
  int context_req = 0;  // --context: 0 none (default_context), kContextMax, or tokens
  std::string host = "127.0.0.1", api_key, model;
  int port = 8000;
  bool verbose = false, yes = false, allow_switch = false;
  const char* help = "kapitsa serve --help";
  for (int i = first; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage_error(a + " needs a value", help);
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      print_serve_help();
      return 0;
    } else if (a == "--model" || a == "-m") model = next();
    else if (a == "--model-id") opt.model_id = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = to_int(a, next(), help);
    else if (a == "--context" || a == "--ctx") {
      std::string why;
      context_req = parse_context(next(), &why);
      if (!context_req) usage_error(why, help);
    }
    else if (a == "--slots") opt.slots = to_int(a, next(), help);
    else if (a == "--spec") opt.spec_depth = to_int(a, next(), help);
    else if (a == "--dflash") opt.dflash = next();
    else if (a == "--prefill-chunk") opt.prefill_chunk = to_int(a, next(), help);
    else if (a == "--api-key") api_key = next();
    else if (a == "--allow-switch") allow_switch = true;
    else if (a == "--mem-budget") opt.mem_budget_gb = to_gb(a, next(), help);
    else if (a == "--image-max-pixels") opt.image_max_pixels = to_int(a, next(), help);
    else if (a == "--disk-cache") {
      opt.disk_cache = next();
      uint64_t bytes = 0;
      if (!pcache::parse_size(opt.disk_cache, &bytes))
        usage_error("--disk-cache takes auto, 0 (off) or a size such as 16G, not \"" + opt.disk_cache + "\"", help);
    }
    else if (a == "--loops") {
      opt.loops = to_int(a, next(), help);
      if (opt.loops < 1 || opt.loops > Dory::kMaxLoops) usage_error("--loops: 1 to 4", help);
    } else if (a == "--loop-kv") {
      LoopKv kv;
      opt.loop_kv = next();
      if (!parse_loop_kv(opt.loop_kv, &kv)) usage_error("--loop-kv: last-loop or per-loop", help);
    }
    else if (a == "--verbose") verbose = true;  // the terminal UI's: the engine's log on screen too
    else if (a == "--yes") yes = true;          // the terminal UI's: no question, the best model for this Mac
    else if (i == first && first == 2 && !a.empty() && a[0] != '-') model = a;  // kapitsa serve 35b
    else usage_error("unknown option " + a, help);
  }
  if (port < 1 || port > 65535) usage_error("--port must be 1-65535", help);
  if (opt.disk_cache.empty()) {  // the SSD prefix cache: on by default for the server (NI_DISK_CACHE as the option)
    const char* v = getenv("NI_DISK_CACHE");
    opt.disk_cache = v && *v ? v : "auto";
  }
  // the model and its precision are picked for a conversation of the context asked for, else of 32K (the default
  // context then grows from there where it costs nothing: default_context)
  opt.context = context_req > 0 ? context_req : kMinDefaultContext;
  {  // looped models: the memory plans before the model loads count what it will run
    LoopKv kv = LoopKv::last_loop;
    if (!opt.loop_kv.empty()) parse_loop_kv(opt.loop_kv, &kv);
    Dory::set_serving(opt.loops, kv);
  }
  if (allow_switch && api_key.empty() && host != "127.0.0.1" && host != "localhost" && host != "::1")
    usage_error("--allow-switch on a network address needs --api-key (any client could load models)", help);
  const bool tui = term::fancy();  // a terminal: the styled screen, the engine's log in a file
  if (tui) ui::serve_begin(verbose);
  std::vector<std::string> notes;  // the terminal UI's, shown under its header once the server will start
  if (!model.empty() && gguf::is_gguf_file(expand_home(model))) {  // a GGUF: its conversion, made on first use
    if (const std::string why = port_problem(host, port); !why.empty()) return fail(why);  // before a long conversion
    try {
      gguf::ImportOptions io;
      io.file = expand_home(model);
      model = gguf::import_model(io);
    } catch (const gguf::Error& e) {
      return fail(e.what(), 2);
    }
  }

  // the model: a name, a repo or a folder; downloaded (or completed) first if needed
  const size_t budget = opt.mem_budget_gb > 0 ? size_t(opt.mem_budget_gb * 1e9) : default_mem_budget();
  // none named, in a terminal: which one, asked under the header (picker.h); a busy port fails before the question
  bool header = false;
  if (model.empty() && ui::can_prompt(yes)) {
    if (const std::string why = port_problem(host, port); !why.empty()) return fail(why);
    ui::serve_header(KAPITSA_VERSION);
    header = true;
    const auto pick = ui::pick_model(opt, budget, "kapitsa serve", context_req);
    if (!pick) return 130;
    model = pick->model;
    context_req = pick->context;
    opt.context = context_req > 0 ? context_req : kMinDefaultContext;
  }
  const bool chosen = !model.empty();  // named or picked: kept for the next question
  hub::ModelRef ref;
  bool last_fallback = false;
  try {
    if (model.empty()) {
      std::string why;
      if (!best_model(opt, budget, &ref, &why)) return fail(why);
      if (tui)
        notes.push_back(bold_name(ref.name) + " is the best fit for this Mac (" + Gpu::get().name() + ", " +
                        strf("%.0f GB", ram_gb()) + "). --model picks another.");
      else
        fprintf(stderr, "No --model given: %s is the best fit for this Mac (%s, %.0f GB). --model picks another.\n",
                ref.name.c_str(), Gpu::get().name().c_str(), ram_gb());
    } else {
      const hub::ModelRef named = hub::resolve(expand_home(model));
      ref = named;
      if (named.kind == hub::ModelRef::kCatalog && !named.variant_given) {
        bool fits = false;
        ref = pick_variant(*named.model, opt, budget, &fits);
        if (fits && ref.variant != named.variant && tui)
          notes.push_back(bold_name(ref.name) + ", the largest variant of " + model + " that fits this Mac (" +
                          strf("%.0f GB", ram_gb()) + "; the default, " + named.variant->name + ", does not). --model " +
                          hub::short_name(*ref.model) + ":<variant> picks another.");
        else if (fits && ref.variant != named.variant)
          fprintf(stderr, "%s on this Mac (%.0f GB): %s, the largest of its variants that fits (the default, %s, does "
                  "not). --model %s:<variant> picks another.\n", model.c_str(), ram_gb(), ref.name.c_str(),
                  named.variant->name.c_str(), hub::short_name(*ref.model).c_str());
        // none holds a whole conversation here: the last fallback, said below if it starts
        last_fallback = !fits && ref.variant != named.variant;
      }
    }
  } catch (const hub::Error& e) {
    return fail(e.what(), 2);
  }
  if (ref.kind == hub::ModelRef::kCatalog && !hub::is_current(ref)) {
    // before a long download: the engine's own start check on the model's headers; a model that would not start
    // here is refused, with the one that fits
    if (const std::string dir = hub::plan_dir(ref); !dir.empty()) {
      EngineOptions po = opt;  // a context past the model's limit runs at the limit
      po.context = context_within_limit(dir, opt.context);
      const int slots = po.slots > 0 ? po.slots : auto_slots(dir, po, budget);
      const LoadPlan lp = plan_load(dir, po.dflash, po.spec_depth == 0, po.context, slots, budget, po.prefill_chunk);
      const StartNeed need = start_need(dir, lp, lp.prefill_chunk, po.context, slots);
      if (need.bytes > budget) {
        // a variant of the same model that fits, else the best model that does
        bool fits = false;
        hub::ModelRef alt = pick_variant(*ref.model, opt, budget, &fits);
        if (!fits && !(best_model(opt, budget, &alt, nullptr) && variant_fits(alt, opt, budget) == 1)) alt = {};
        const bool other = alt.variant && alt.variant != ref.variant;
        char msg[768];
        if (need.weights > budget)  // nothing but another variant helps
          snprintf(msg, sizeof msg, "%s does not fit this Mac: its weights alone (%.1f GB, %s) exceed the %.1f GB of GPU "
                   "memory Kapitsa may use here (%.0f GB of RAM).%s%s", ref.name.c_str(), need.weights / 1e9,
                   need.formats.c_str(), budget / 1e9, ram_gb(), other ? " This one fits: kapitsa serve --model " : "",
                   other ? hub::short_name(*alt.model, *alt.variant).c_str() : "");
        else
          snprintf(msg, sizeof msg, "%s needs %.1f GB of GPU memory with %s, and this Mac (%.0f GB) allows %.1f GB, so it "
                   "would not start.%s%s", ref.name.c_str(), need.bytes / 1e9, need.parts.c_str(), ram_gb(),
                   budget / 1e9, other ? " This one fits: kapitsa serve --model " : "",
                   other ? hub::short_name(*alt.model, *alt.variant).c_str() : "");
        return fail(msg);
      }
    }
  }
  if (last_fallback && tui)
    notes.push_back(bold_name(ref.name) + ", the smallest variant of " + model + " (the default, " +
                    ref.model->default_variant + ", does not fit this Mac's " + strf("%.0f GB", ram_gb()) +
                    "; this one starts, with less room for a conversation). --model " + hub::short_name(*ref.model) +
                    ":<variant> picks another.");
  else if (last_fallback)
    fprintf(stderr, "%s on this Mac (%.0f GB): %s, the smallest of its variants (the default, %s, does not fit; this "
            "one starts, with less room for a conversation). --model %s:<variant> picks another.\n", model.c_str(),
            ram_gb(), ref.name.c_str(), ref.model->default_variant.c_str(), hub::short_name(*ref.model).c_str());
  if (ref.kind == hub::ModelRef::kLocal) {
    struct stat st;
    if (stat((ref.dir + "/config.json").c_str(), &st) != 0)
      return fail(ref.dir + " has no config.json. --model takes a model folder (config.json, *.safetensors, "
                  "tokenizer.json), a model name (kapitsa --help) or a Hugging Face repo.", 2);
  }
  if (chosen) ui::remember(ref);
  // fail now, not after loading the model, when the port is taken
  if (const std::string why = port_problem(host, port); !why.empty()) return fail(why);
  const std::string log_path = tui ? ui::serve_log_file(port) : "";
  if (tui) {
    if (!header) ui::serve_header(KAPITSA_VERSION);
    for (const auto& n : notes) ui::note(n);
  }
  try {
    hub::PullOptions po;
    if (ui::can_prompt(yes)) po.make_room = ui::make_room;  // out of disk space: which models to delete, asked
    opt.model_dir = hub::ensure(ref, po);
  } catch (const hub::Stopped& e) {
    return stopped(e);
  } catch (const hub::Error& e) {
    return fail(e.what());
  }
  // the context: the one asked for, within what the model was trained for, else the default for this Mac
  {
    const int limit = model_context_limit(opt.model_dir);
    std::string say;
    if (context_req == kContextMax) {
      if (limit <= 0)
        return fail(ref.name + " does not state the context it was trained for (max_position_embeddings in its "
                    "config.json), so --context max has no size here: give it in tokens (--context 128k).", 2);
      opt.context = limit;
    } else if (context_req > 0) {
      opt.context = context_req;
      if (limit > 0 && context_req > limit) {
        opt.context = limit;
        say = "--context " + term::context(context_req) + " is more than " + ref.name + " was trained for: it runs with " +
              term::context(limit) + ".";
      }
    } else {
      ContextWhy why = ContextWhy::kDefault;
      opt.context = default_context(opt.model_dir, opt, budget, &why);
      const std::string c = "Context: " + term::context(opt.context) + " tokens per conversation";
      const std::string max = limit > opt.context ? " (--context max: " + term::context(limit) + ")" : "";
      switch (why) {
        case ContextWhy::kDefault: say = c + max + "."; break;
        case ContextWhy::kModelLimit: say = c + ", all that " + ref.name + " takes."; break;
        case ContextWhy::kMemoryWithDrafter:
          say = c + ", the most this Mac holds with the draft model (a longer --context runs without it, slower).";
          break;
        case ContextWhy::kMemory: say = c + ", the most this Mac's memory holds."; break;
      }
    }
    if (!say.empty()) {
      if (tui) ui::note(say);
      else fprintf(stderr, "%s\n", say.c_str());
    }
  }
  if (opt.model_id.empty()) opt.model_id = ref.id;
  // the server compiles the Neural Engine program in the background (no 40 s start after the ANE
  // daemon evicted it); ni_run and the benchmarks build it synchronously
  setenv("NI_ANE_ASYNC", "1", 0);
  const time_t started = std::time(nullptr);
  char started_at[32];
  strftime(started_at, sizeof started_at, "%Y-%m-%dT%H:%M:%SZ", gmtime(&started));
  auto model_mgr = std::make_shared<ModelManager>();
  model_mgr->set_allow_switch(allow_switch);
  model_mgr->init(opt, ref, context_req, budget, tui, started, started_at);
  if (tui) term::guard_signals();  // the status block's signal handlers, which the stop handlers pass the signal on to
  SaveOnStop save_on_stop(*model_mgr);

  api::ResponseStore responses;  // recent responses, for previous_response_id
  httplib::Server srv;
  // conversations carry their images as base64 every turn (an agent's screenshots: 1-4 MB each)
  srv.set_payload_max_length(256u << 20);
  // Stream events go out as written: with Nagle's algorithm a small SSE chunk waits for the ACK of
  // the previous one (the client's delayed ACK), which can hold the first token of a fast request.
  srv.set_tcp_nodelay(!getenv("NI_TCP_NODELAY") || env_on("NI_TCP_NODELAY"));
  srv.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
    // the chat page needs no key (it asks for one and sends it with its API calls), nor does /health
    if (api_key.empty() || req.path == "/health" || (req.method == "GET" && (req.path == "/" || req.path == "/favicon.ico")))
      return httplib::Server::HandlerResponse::Unhandled;
    // OpenAI clients send "Authorization: Bearer KEY", Anthropic ones "x-api-key: KEY" (Claude Code may send both,
    // with an Anthropic key of its own in x-api-key)
    if (req.get_header_value("Authorization") == "Bearer " + api_key || req.get_header_value("x-api-key") == api_key)
      return httplib::Server::HandlerResponse::Unhandled;
    res.status = 401;
    if (req.path.rfind("/v1/messages", 0) == 0)
      res.set_content(api::dump(api::messages_error("authentication_error", "invalid api key")), "application/json");
    else
      res.set_content(R"({"error":{"message":"invalid api key","type":"auth_error"}})", "application/json");
    return httplib::Server::HandlerResponse::Handled;
  });
  srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
  });
  // the chat page: one embedded file, no requests anywhere but this server's API
  srv.Get("/", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Content-Security-Policy",
                   "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; "
                   "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'");
    res.set_header("X-Content-Type-Options", "nosniff");
    res.set_header("Referrer-Policy", "no-referrer");
    res.set_header("Cache-Control", "no-cache");
    res.set_content(kChatPage, "text/html; charset=utf-8");
  });
  srv.Get("/favicon.ico", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
  srv.Get("/v1/models", [model_mgr](const httplib::Request&, httplib::Response& res) {
    res.set_content(model_mgr->list_models_json().dump(), "application/json");
  });
  srv.Get(R"(/v1/models/(.+))", [model_mgr](const httplib::Request& req, httplib::Response& res) {
    const std::string mid = req.matches[1].str();
    auto m = model_mgr->get_model_json(mid);
    if (m.is_object() && !m.empty()) {
      res.set_content(m.dump(), "application/json");
    } else {
      res.status = 404;
      res.set_content(R"({"error":{"message":"model not found","type":"invalid_request_error"}})", "application/json");
    }
  });
  srv.Post("/v1/models/switch", [model_mgr](const httplib::Request& req, httplib::Response& res) {
    if (!model_mgr->allow_switch()) {
      res.status = 403;
      res.set_content(R"j({"error":{"message":"model switching is off (start the server with --allow-switch)","type":"invalid_request_error"}})j", "application/json");
      return;
    }
    if (model_mgr->switching()) {
      res.status = 503;
      res.set_header("Retry-After", "30");
      res.set_content(R"({"error":{"message":"another model switch is in progress","type":"server_error"}})", "application/json");
      return;
    }
    std::string target;
    try {
      json b = json::parse(req.body);
      target = b.value("model", "");
    } catch (...) {
      target = req.get_param_value("model");
    }
    if (target.empty()) {
      res.status = 400;
      res.set_content(R"({"error":{"message":"Missing 'model' parameter","type":"invalid_request_error"}})", "application/json");
      return;
    }
    const std::string prev = model_mgr->current_model_id();
    std::string err;
    auto eng = model_mgr->switch_to(target, &err);
    if (!eng) {
      res.status = 400;
      json eb{{"error", {{"message", err.empty() ? "Failed to switch model" : err}, {"type", "invalid_request_error"}}}};
      res.set_content(eb.dump(), "application/json");
      return;
    }
    json j{
      {"status", "ok"},
      {"model", eng->model_id()},
      {"previous_model", prev},
      {"context_length", eng->context_length()},
      {"context_fit", eng->context_fit()},
      {"slots", eng->slots()}
    };
    res.set_content(j.dump(), "application/json");
  });

  // After a reply: prefill the conversation as its next request will render it (NI_HISTORY_PREFILL), in its own thread
  auto prefill_after = [model_mgr](const ChatRequest& r, const ChatResult& out) {
    model_mgr->prefill_after(r, out);
  };

  srv.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (const std::exception& ex) {
      res.status = 400;
      res.set_content(api::dump(api::chat_error(std::string("bad request: ") + ex.what())), "application/json");
      return;
    }
    const std::string req_model = body.value("model", "");
    std::string err;
    if (reject_while_switching(*model_mgr, res, api::dump(api::chat_error("The server is switching models; retry shortly.")))) return;
    auto guard = model_mgr->acquire_request(req_model, &err);
    if (!guard) {
      res.status = 400;
      res.set_content(api::dump(api::chat_error("Model error: " + err)), "application/json");
      return;
    }
    ChatRequest r;
    try {
      r = parse_request(body, guard->engine().request_defaults());
    } catch (const api::ApiError& e) {
      res.status = e.status;
      res.set_content(api::dump(api::chat_error(e.what(), e.type, e.code, e.param)), "application/json");
      return;
    } catch (const std::exception& ex) {
      res.status = 400;
      res.set_content(api::dump(api::chat_error(std::string("bad request: ") + ex.what())), "application/json");
      return;
    }
    const bool stream = body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();
    const std::string id = "chatcmpl-" + std::to_string(std::time(nullptr)) + std::to_string(rand() % 100000);
    const long created = long(std::time(nullptr));
    const std::string model = guard->engine().model_id();
    const auto meter = ui::meter(req, r);
    r.live = ui::live_of(meter);
    if (!stream) {
      ChatResult out = guard->engine().chat(r, [&](const ChatDelta&) {
        ui::token(meter);
        return true;
      });
      ui::done(meter, out);
      prefill_after(r, out);
      if (out.finish_reason == "error") {
        auto [status, fail_err] = api::chat_failure(out);
        res.status = status;
        res.set_content(api::dump(fail_err), "application/json");
        return;
      }
      json msg{{"role", "assistant"}, {"content", out.content}};
      if (!out.reasoning.empty()) msg["reasoning_content"] = out.reasoning;
      if (!out.tool_calls.empty()) msg["tool_calls"] = api::chat_tool_calls(out.tool_calls, 0, false);
      json j{{"id", id},
             {"object", "chat.completion"},
             {"created", created},
             {"model", model},
             {"choices", json::array({json{{"index", 0}, {"message", msg}, {"finish_reason", out.finish_reason}}})},
             {"usage", api::chat_usage(out)}};
      res.set_content(api::dump(j), "application/json");
      return;
    }
    // the stream (src/api/chat_completions.cpp) starts as soon as the prompt is known to fit, before the slot
    stream_reply(guard, r, std::make_shared<api::ChatCompletionsStream>(id, model, created), res, api::chat_failure,
                 [&, r](const ChatResult& out) { prefill_after(r, out); }, meter, true);
  });

  // Anthropic Messages (src/api/anthropic.cpp)
  srv.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(api::dump(api::messages_error("invalid_request_error", std::string("bad request: ") + e.what())),
                      "application/json");
      return;
    }
    const std::string req_model = body.value("model", "");
    std::string err;
    if (reject_while_switching(*model_mgr, res, api::dump(api::messages_error("overloaded_error", "The server is switching models; retry shortly.")))) return;
    auto guard = model_mgr->acquire_request(req_model, &err);
    if (!guard) {
      res.status = 400;
      res.set_content(api::dump(api::messages_error("invalid_request_error", "Model error: " + err)), "application/json");
      return;
    }
    api::MessagesRequest q;
    try {
      q = api::messages_request(body);
    } catch (const api::ApiError& e) {
      res.status = e.status;
      res.set_content(api::dump(api::messages_error(e.type, e.what())), "application/json");
      return;
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(api::dump(api::messages_error("invalid_request_error", std::string("bad request: ") + e.what())),
                      "application/json");
      return;
    }
    const std::string id = api::new_id("msg_"), model = guard->engine().model_id();
    const auto meter = ui::meter(req, q.chat);
    q.chat.live = ui::live_of(meter);
    if (!q.stream) {
      ChatResult out = guard->engine().chat(q.chat, [&](const ChatDelta&) {
        ui::token(meter);
        return true;
      });
      ui::done(meter, out);
      if (out.finish_reason == "error") {
        auto [status, fail_body] = api::messages_failure(out);
        res.status = status;
        res.set_content(api::dump(fail_body), "application/json");
        return;
      }
      res.set_content(api::dump(api::messages_response(id, model, out, q.show_thinking)), "application/json");
      prefill_after(q.chat, out);
      return;
    }
    stream_reply(guard, q.chat, std::make_shared<api::MessagesStream>(id, model, q.show_thinking), res,
                 api::messages_failure, [&, r = q.chat](const ChatResult& out) { prefill_after(r, out); }, meter);
  });
  srv.Post("/v1/messages/count_tokens", [&](const httplib::Request& req, httplib::Response& res) {
    try {
      json b = json::parse(req.body);
      const std::string req_model = b.value("model", "");
      if (reject_while_switching(*model_mgr, res, api::dump(api::messages_error("overloaded_error", "The server is switching models; retry shortly.")))) return;
      auto guard = model_mgr->acquire_request(req_model);
      if (!guard) {
        res.status = 400;
        res.set_content(api::dump(api::messages_error("invalid_request_error", "Model error")), "application/json");
        return;
      }
      const ChatRequest r = api::messages_prompt(b);
      res.set_content(json{{"input_tokens", guard->engine().count_tokens(r)}}.dump(), "application/json");
    } catch (const api::ApiError& e) {
      res.status = e.status;
      res.set_content(api::dump(api::messages_error(e.type, e.what())), "application/json");
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(api::dump(api::messages_error("invalid_request_error", std::string("bad request: ") + e.what())),
                      "application/json");
    }
  });

  // OpenAI Responses (src/api/responses.cpp)
  srv.Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(api::dump(api::responses_error(std::string("bad request: ") + e.what(), "invalid_request_error")),
                      "application/json");
      return;
    }
    const std::string req_model = body.value("model", "");
    std::string err;
    if (reject_while_switching(*model_mgr, res, api::dump(api::responses_error("The server is switching models; retry shortly.", "server_error")))) return;
    auto guard = model_mgr->acquire_request(req_model, &err);
    if (!guard) {
      res.status = 400;
      res.set_content(api::dump(api::responses_error("Model error: " + err, "invalid_request_error")), "application/json");
      return;
    }
    auto q = std::make_shared<api::ResponsesRequest>();
    try {
      *q = api::responses_request(body, responses);
    } catch (const api::ApiError& e) {
      res.status = e.status;
      res.set_content(api::dump(api::responses_error(e.what(), e.type, e.code, e.param)), "application/json");
      return;
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(api::dump(api::responses_error(std::string("bad request: ") + e.what(), "invalid_request_error")),
                      "application/json");
      return;
    }
    const std::string id = api::new_id("resp_"), model = guard->engine().model_id();
    const long created = long(std::time(nullptr));
    // the conversation through this response, for a later previous_response_id
    auto keep = [&responses, q, id](const ojson& output) {
      if (!q->store) return;
      ojson items = q->items;
      for (const auto& it : output) items.push_back(it);
      responses.put(id, std::move(items));
    };
    const auto meter = ui::meter(req, q->chat);
    q->chat.live = ui::live_of(meter);
    if (!q->stream) {
      ChatResult out = guard->engine().chat(q->chat, [&](const ChatDelta&) {
        ui::token(meter);
        return true;
      });
      ui::done(meter, out);
      if (out.finish_reason == "error") {
        auto [status, fail_body] = api::responses_failure(out);
        res.status = status;
        res.set_content(api::dump(fail_body), "application/json");
        return;
      }
      ojson output;
      res.set_content(api::dump(api::responses_response(*q, id, model, created, out, &output)), "application/json");
      keep(output);
      prefill_after(q->chat, out);
      return;
    }
    auto w = std::make_shared<api::ResponsesStream>(*q, id, model, created);
    stream_reply(
        guard, q->chat, w, res, api::responses_failure,
        [&, q, w, keep](const ChatResult& out) {
          if (out.finish_reason != "error") keep(w->output());
          prefill_after(q->chat, out);
        },
        meter);
  });

  // A busy port is an error: httplib's default SO_REUSEPORT would let a second server bind it too.
  srv.set_socket_options([](socket_t sock) {
    int one = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  });
  if (!srv.bind_to_port(host, port))
    return fail("Can't listen on " + host + ":" + std::to_string(port) + ": " + port_problem(host, port) +
                " Pass another --port.");
  logi("listening on http://%s:%d", host.c_str(), port);
  const std::string shown = host == "0.0.0.0" || host == "::" ? "127.0.0.1" : host;
  const std::string base =
      "http://" + (shown.find(':') != std::string::npos ? "[" + shown + "]" : shown) + ":" + std::to_string(port);
  if (tui) {
    ui::Ready card;
    card.model = model_mgr->current_model_id();
    auto cur_ref = model_mgr->current_ref();
    if (cur_ref.variant) card.quant = cur_ref.variant->quant;
    card.base = base;
    card.api_key_note = api_key.empty() ? "" : "required (--api-key)";
    card.log = log_path;
    card.budget = budget;
    card.used = Gpu::get().allocated_bytes();
    card.context = model_mgr->current_context_length();
    card.context_fit = model_mgr->current_context_fit();
    card.slots = model_mgr->current_slots();  // as the engine started (auto_slots when --slots is not given)
    ui::ready(card);
  } else {
    printf("Ready: chat at %s, API at %s/v1 (model \"%s\"%s). Ctrl+C stops.\n", base.c_str(), base.c_str(),
           model_mgr->current_model_id().c_str(), api_key.empty() ? "" : ", API key required");
  }
  fflush(stdout);
  srv.listen_after_bind();
  return 0;
}

static void print_plan(const hub::ModelRef& r, bool as_json) {
  static const char* kinds[] = {"catalog", "repo", "local"};
  const bool local = r.kind == hub::ModelRef::kLocal;
  json j{{"model", r.name}, {"kind", kinds[r.kind]}, {"id", r.id}, {"dir", r.dir}};
  if (!local) {
    j["repo"] = r.repo;
    j["revision"] = r.revision.empty() ? "main" : r.revision;
    j["downloaded"] = hub::is_current(r);
  }
  json comp = json::array();
  // a GGUF variant: its file, then the import's size on disk (its plan, null when it cannot be planned: offline)
  const bool gguf = r.variant && !r.variant->gguf.path.empty();
  hub::ImportSize imp;
  const bool sized = gguf && hub::import_size(r, &imp);
  if (r.variant) {
    j["variant"] = r.variant->name;
    j["quant"] = r.variant->quant;
    if (gguf) {
      j["file"] = r.variant->gguf.path;
      j["sha256"] = r.variant->gguf.sha256;
      j["download_bytes"] = r.variant->gguf.size;
      j["imported_bytes"] = sized ? json(imp.disk) : json(nullptr);
    }
    for (const auto& c : r.variant->companions)
      comp.push_back(json{{"kind", c.kind}, {"repo", c.repo}, {"revision", c.revision}, {"into", c.subdir.empty() ? "." : c.subdir}});
  }
  j["companions"] = comp;
  if (as_json) {
    printf("%s\n", j.dump().c_str());
    return;
  }
  if (term::fancy()) return ui::plan(r, sized ? imp.disk : 0);
  printf("%s\n", r.name.c_str());
  struct stat st {};
  const bool here = !local && !hub::is_current(r) && stat((r.dir + "/config.json").c_str(), &st) == 0;
  printf("  folder:     %s%s\n", hub::tilde(r.dir).c_str(),
         local                ? " (yours, used as is)"
         : hub::is_current(r) ? " (downloaded)"
         : here ? " (files here, not checked yet: `kapitsa pull` checks them against the pinned checksums and fetches "
                  "only what differs)"
                : " (not downloaded yet)");
  if (!local) printf("  weights:    %s @ %s\n", r.repo.c_str(), r.revision.empty() ? "main" : r.revision.c_str());
  if (gguf) {
    const std::string size = hub::fmt_bytes(r.variant->gguf.size);
    printf("  file:       %s%s%s\n", r.variant->gguf.path.c_str(),
           (hub::is_current(r) ? " (" + size + "), imported" : ", " + size + " to download, then imported (kapitsa import)").c_str(),
           sized ? (": " + hub::fmt_bytes(imp.disk) + " on disk").c_str() : "");
  }
  if (r.variant) printf("  variant:    %s (%s)%s\n", r.variant->name.c_str(), r.variant->quant.c_str(),
                        r.variant->name == r.model->default_variant ? ", the default" : "");
  if (r.variant)
    for (const auto& c : r.variant->companions)
      printf("  %-11s %s @ %s%s\n", (c.drafter() ? "drafter:" : "MTP head:"), c.repo.c_str(), c.revision.c_str(),
             c.kind == "mtp-sidecar" ? " (its mtp.* tensors, made into mtp.safetensors)" : "");
  printf("  API name:   %s\n", r.id.c_str());
}

static int pull(int argc, char** argv) {
  std::vector<std::string> models;
  hub::PullOptions po;
  bool dry = false, as_json = false;
  const char* help = "kapitsa pull --help";
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      print_pull_help();
      return 0;
    } else if (a == "--from") {
      if (i + 1 >= argc) usage_error("--from needs a folder", help);
      po.from.push_back(expand_home(argv[++i]));
    } else if (a == "--dry-run") {
      dry = true;
    } else if (a == "--json") {
      as_json = true;
    } else if (!a.empty() && a[0] == '-') {
      usage_error("unknown option " + a, help);
    } else {
      models.push_back(a);
    }
  }
  if (models.empty()) usage_error("which model? e.g. kapitsa pull 35b", help);
  if (!as_json && ui::can_prompt(false)) po.make_room = ui::make_room;  // out of disk space: which models to delete, asked
  EngineOptions eo;  // serve's defaults: one 32k conversation, the slots auto_slots picks (fit checks count 2)
  for (const auto& name : models) {
    hub::ModelRef ref;
    try {
      ref = for_this_mac(hub::resolve(expand_home(name)), eo, default_mem_budget());
    } catch (const hub::Error& e) {
      return fail(e.what(), 2);
    }
    if (dry) {
      print_plan(ref, as_json);
      continue;
    }
    if (ref.kind == hub::ModelRef::kLocal) {
      if (term::fancy()) ui::note(hub::tilde(ref.dir) + " is a folder on this Mac: nothing to download.", true);
      else printf("%s is a folder on this Mac: nothing to download.\n", ref.dir.c_str());
      continue;
    }
    const bool had = hub::is_current(ref);
    try {
      hub::ensure(ref, po);
    } catch (const hub::Stopped& e) {
      return stopped(e);
    } catch (const hub::Error& e) {
      return fail(e.what());
    }
    const std::string start = ref.model ? hub::short_name(*ref.model, *ref.variant) : ref.name;
    if (term::fancy()) {
      ui::ok(term::clip(term::bold(ref.name) + (had ? " is already downloaded" : " is ready") + term::dim("  " + hub::tilde(ref.dir)),
                        term::columns() - 5),
             true);
      ui::note("kapitsa serve --model " + start + " starts it", true);
    } else {
      printf("%s %s in %s. Start it: kapitsa serve --model %s\n", ref.name.c_str(),
             had ? "is already downloaded" : "is ready", hub::tilde(ref.dir).c_str(), start.c_str());
    }
  }
  return 0;
}

static int list(int argc, char** argv) {
  bool as_json = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") as_json = true;
    else if (a == "--help" || a == "-h") {
      show_help(strf("kapitsa list [--json]: the models in %s, their size on disk and what they include.\n",
                     hub::tilde(hub::models_dir()).c_str()));
      return 0;
    } else usage_error("unknown option " + a);
  }
  if (!as_json && term::fancy()) {
    ui::list();
    return 0;
  }
  const auto v = hub::installed();
  if (as_json) {
    json a = json::array();
    for (const auto& m : v)
      a.push_back(json{{"name", m.name}, {"variant", m.variant}, {"dir", m.dir}, {"repo", m.repo}, {"commit", m.commit},
                       {"complete", m.complete}, {"bytes", m.bytes}, {"includes", m.extras}});
    printf("%s\n", a.dump(2).c_str());
    return 0;
  }
  if (v.empty()) {
    printf("No models downloaded yet (%s).\n", hub::tilde(hub::models_dir()).c_str());
  } else {
    printf("Models in %s:\n", hub::tilde(hub::models_dir()).c_str());
    for (const auto& m : v) {
      std::string what = "weights";
      for (const auto& e : m.extras) what += " + " + e;
      if (!m.complete) {
        // a catalog model's folder that Kapitsa did not download (no .kapitsa/): `pull` checks and adopts its files
        std::string cat;
        struct stat st {};
        if (stat((m.dir + "/.kapitsa").c_str(), &st) != 0)
          for (const auto& c : hub::catalog())
            for (const auto& var : c.variants)
              if (m.name == c.name + "-" + var.name) cat = hub::short_name(c, var);
        what = !cat.empty() ? "not checked yet: `kapitsa pull " + cat + "` checks it against the pinned checksums"
                            : "incomplete: `kapitsa pull " + m.name + "` continues it";
      }
      printf("  %-18s %9s  %s\n", m.name.c_str(), hub::fmt_bytes(m.bytes).c_str(), what.c_str());
    }
  }
  std::string more;
  for (const auto& c : hub::catalog())
    for (const auto& var : c.variants) {
      bool have = false;
      for (const auto& m : v) have = have || (m.name == hub::full_name(c, var) && m.complete);
      if (have) continue;
      char b[200];
      snprintf(b, sizeof b, "%s%s (%s, %.1f GB)", more.empty() ? "" : ", ", hub::full_name(c, var).c_str(),
               hub::short_name(c, var).c_str(), var.download_gb);
      more += b;
    }
  if (!more.empty())
    printf("\n%s: %s.\n`kapitsa pull MODEL` downloads one; `kapitsa serve --model MODEL` downloads and starts it.\n",
           v.empty() ? "Available" : "Also available", more.c_str());
  return 0;
}

static int rm(int argc, char** argv) {
  if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
    show_help(strf("kapitsa rm MODEL...: deletes downloaded models from %s.\n", hub::tilde(hub::models_dir()).c_str()));
    return argc < 3 ? 2 : 0;
  }
  for (int i = 2; i < argc; ++i) {
    hub::ModelRef ref;
    try {
      ref = hub::resolve(argv[i]);
    } catch (const hub::Error& e) {
      return fail(e.what(), 2);
    }
    struct stat st;
    if (ref.kind == hub::ModelRef::kCatalog && !ref.variant_given && stat(ref.dir.c_str(), &st) != 0) {
      // no variant named and the default is not here: the model's one downloaded variant
      std::vector<hub::ModelRef> have;
      for (const auto& v : ref.model->variants) {
        hub::ModelRef r = hub::resolve(ref.model->name + ":" + v.name);
        if (stat(r.dir.c_str(), &st) == 0) have.push_back(r);
      }
      if (have.size() == 1) ref = have[0];
      else if (have.size() > 1) {
        std::string names;
        for (const auto& r : have) names += (names.empty() ? "" : ", ") + hub::short_name(*r.model, *r.variant);
        return fail("Several variants of " + ref.model->name + " are downloaded: " + names + ". Name one.", 2);
      }
    }
    if (ref.kind == hub::ModelRef::kLocal && ref.dir.rfind(hub::models_dir() + "/", 0) != 0)
      return fail(ref.dir + " is a folder of yours, not a downloaded model: kapitsa rm leaves it alone.", 2);
    if (stat(ref.dir.c_str(), &st) != 0) {
      if (term::fancy()) ui::note(term::bold(ref.name) + " is not downloaded.", true);
      else printf("%s is not downloaded.\n", ref.name.c_str());
      continue;
    }
    const std::string root = hub::models_dir() + "/";
    if (ref.dir.compare(0, root.size(), root) != 0 || ref.dir.size() <= root.size())
      return fail("refusing to delete " + ref.dir + " (not in " + root + ")");
    // what it gives back: files cloned from another copy (Nace Agent's, the Hugging Face cache) free nothing
    const uint64_t bytes = hub::freeable_bytes(ref.dir);
    std::error_code ec;
    std::filesystem::remove_all(ref.dir, ec);
    if (ec) return fail("Can't delete " + hub::tilde(ref.dir) + ": " + ec.message());
    const std::string freed = bytes < (1u << 20) ? "nothing freed: its files are shared with another copy"
                                                 : hub::fmt_bytes(bytes) + " freed";
    if (term::fancy()) ui::ok("Deleted " + term::bold(ref.name) + term::dim("  " + freed), true);
    else printf("Deleted %s (%s).\n", ref.name.c_str(), freed.c_str());
  }
  return 0;
}

static int mtp_sidecar(int argc, char** argv) {
  std::string src, out, rev, hidden = "post_norm";
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--revision" && i + 1 < argc) rev = argv[++i];
    else if (a == "--hidden" && i + 1 < argc) hidden = argv[++i];
    else if (!a.empty() && a[0] != '-' && src.empty()) src = a;
    else if (!a.empty() && a[0] != '-' && out.empty()) out = a;
    else src.clear(), out.clear(), i = argc;
  }
  if (src.empty() || out.empty() || (hidden != "post_norm" && hidden != "pre_norm")) {
    fprintf(stderr,
            "usage: kapitsa mtp-sidecar <hf repo | local dir of HF safetensors> <out mtp.safetensors> [--revision REV]\n"
            "                           [--hidden post_norm|pre_norm]\n"
            "Builds Kapitsa's MTP sidecar from an official Qwen3.5 checkpoint's mtp.* tensors, byte for byte what\n"
            "tools/mtp_sidecar.py writes (a repo is read at REV, default main, by range requests).\n");
    return 2;
  }
  try {
    hub::build_mtp_sidecar(src, rev, out, hidden);
  } catch (const hub::Stopped& e) {
    return stopped(e);
  } catch (const hub::Error& e) {
    return fail(e.what());
  }
  return 0;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  // in a terminal the engine's log lines stay out of the screens (`kapitsa serve` keeps them in its log file)
  if (term::fancy() && cmd != "serve" && (cmd.empty() || cmd[0] != '-' || cmd == "--help" || cmd == "-h")) ui::quiet();
  // the last automatic update's line, and once a day the next check, in the background (update_cmd.cpp)
  auto_update(argc, argv);
  if (cmd.empty() || cmd == "--help" || cmd == "-h" || cmd == "help") {
    if (argc > 2 && !strcmp(argv[2], "serve")) print_serve_help();
    else if (argc > 2 && !strcmp(argv[2], "pull")) print_pull_help();
    else if (argc > 2 && !strcmp(argv[2], "update")) {
      char h[] = "--help";
      char* a[] = {argv[0], argv[2], h, nullptr};
      return run_update(3, a, 2);
    }
    else print_help();
    return 0;
  }
  if (cmd == "--version" || cmd == "-v" || cmd == "version") {
    printf("kapitsa %s%s%s%s\n", kVersion, *KAPITSA_GIT ? " (" : "", KAPITSA_GIT, *KAPITSA_GIT ? ")" : "");
    return 0;
  }
  if (cmd == "serve") return serve(argc, argv, 2);
  if (cmd == "pull") return pull(argc, argv);
  if (cmd == "list" || cmd == "ls") return list(argc, argv);
  if (cmd == "rm") return rm(argc, argv);
  if (cmd == "mtp-sidecar") return mtp_sidecar(argc, argv);
  if (cmd == "import") return gguf::run_import(argc, argv);
  if (cmd == "doctor") return ni::run_doctor(argc - 2, argv + 2);
  if (cmd == "claude" || cmd == "codex" || cmd == "opencode" || cmd == "pi") return launch_agent(cmd, argc, argv, 2);
  if (cmd == "bench") return run_bench(argc, argv, 2);
  if (cmd == "update") return run_update(argc, argv, 2);
  if (cmd[0] == '-') return serve(argc, argv, 1);  // options only: `kapitsa --model DIR ...` (older scripts)
  usage_error("unknown command \"" + cmd + "\"");
}
