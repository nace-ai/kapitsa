// `kapitsa bench`'s core (the command and its screens are src/server/bench_cmd.cpp): the published benchmark's protocol
// for one Mac, its conditions, the result file and its submission.
//
// - protocol.cpp: bench/harness/bench.py's measurements, ported request for request: the same prompts (the frozen corpus
//   is embedded), sampling, checks and rates, as a client of the server over its local API, like the harness. The
//   warm-up and the quiet-GPU wait are compare_engines.py's.
// - system.mm: the Mac (identifier, Apple's name for it and its year from bench/community/mac-models.json, chip, cores,
//   memory, macOS) and its conditions (power source, Low Power Mode, power mode, thermal state, swap), GPU time per
//   process (IOKit, what bench/harness/gpu_hogs.py reads with ioreg) and memory per process (proc_pid_rusage: the
//   numbers `footprint` prints).
// - monitor.cpp: gpu_hogs.py's contention monitor and quiet wait, plus who uses the GPU, the thermal state and the
//   power source per sample.
// - result.cpp: the result file (bench/community/schema.json), its name, and the texts made from it.
// - submit.cpp: a pull request from the user's fork, made with their own `gh` (there is no other way in).
#pragma once
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace ni::bench {

using ojson = nlohmann::ordered_json;

// ---------------------------------------------------------------- the protocol (protocol.cpp)

constexpr int kSchema = 1;                      // the result file's schema (bench/community/schema.json)
extern const char* const kCorpusSha256;         // bench/harness/corpus/NOTICE.md
const std::string& corpus();                    // bench/harness/corpus/bench_corpus.txt, embedded at build time
std::string sha256_hex(const std::string& data);
// bench.py build_context(): the corpus' characters (code points, as Python slices a str) from `offset` on, as many as
// int(target_tokens * 4.25)
std::string build_context(int target_tokens, int offset_chars = 0);

extern const char* const kDecodeUser;
extern const char* const kDecodeSys;
extern const char* const kSys;
extern const char* const kFollowups[3];
extern const char* const kToolResult;     // the round trip's tool result (a main.py)
ojson tools();                            // bench.py TOOLS
ojson prefill_messages(const std::string& ctx, const std::string& nonce, const std::string& question);
ojson decode_messages(const std::string& run_nonce);  // run_nonce: "0.123456"
ojson tools_messages();
constexpr const char* kPrefillQuestion = "In one sentence, what is the first file in the repository context about?";
constexpr const char* kWarmQuestion = "Summarize the first file in the repository context.";
constexpr int kWarmTokens = 2700, kWarmOffsetChars = 100000, kColdCacheMax = 16;
constexpr double kPromptTol = 0.01;

// bench.py's arguments, at its defaults
struct Settings {
  std::string host = "127.0.0.1";
  int port = 0;
  std::string model;  // the requests' model id
  int decode_tokens = 512, decode_runs = 5, greedy_runs = 5, window_head = 16, window_tail = 4;
  double rep_threshold = 0.5;
  std::vector<int> sizes{4000, 16000};
  int cached_samples = 3;
  double turn_gap = 1.0;
  double temperature = 0.6, top_p = 0.95;
  int top_k = 20;
  double min_p = 0, presence_penalty = 0, frequency_penalty = 0, repeat_penalty = 1.0;
  std::string reasoning_effort = "low";  // the thinking-on tool test
  std::map<int, int> ref_prompt_tokens;  // size -> the cold prompt's usage.prompt_tokens
  // the model reasons before every answer with no switch for it (Flint; bench.py --always-reasons): its reasoning is
  // not a leak, and the prefill, follow-up and tool turns get room to finish it
  bool always_reasons = false;
};
// The reference prompt_tokens of the cold prompts for a model's tokenizer family (bench.py REF_PROMPT_TOKENS, and
// configs/bolt.json's for Nemotron): empty when unknown (the check is then reported as unchecked)
std::map<int, int> reference_prompt_tokens(const std::string& family);  // "qwen3.5" | "nemotron-h" | "flint"
ojson sampling(const Settings& s, bool greedy);
ojson no_think();
ojson think_on(const Settings& s);
ojson body_of(const Settings& s, const ojson& msgs, int max_tokens, bool greedy = false, bool think = false,
              const ojson& extra = ojson::object());

// Random nonces as bench.py makes them
struct Nonces {
  std::mt19937_64 rng{std::random_device{}()};
  // tests: these instead of random ones (the run nonce, and the bits every hex nonce takes its low bits from)
  bool fixed = false;
  std::string fixed_run = "0.123456";
  uint64_t fixed_bits = 0x0123456789abcdefull;
  std::string run();              // f"{random.random():.6f}"
  std::string hex(int bits);      // f"{random.getrandbits(bits):0{bits/4}x}"
};

// One streamed chat completion (bench.py stream_chat): times in seconds since the request, -1 = none
struct ToolCallAcc {
  std::string id, name, arguments;
  bool has_id = false;
};
struct Stream {
  double ttft = -1, gen_time = -1, total = 0;
  ojson usage, timings;  // null when the stream sent none
  std::string content, reasoning, finish;
  std::vector<ToolCallAcc> tool_calls;
  int n_chunks = 0;
  std::vector<double> ev_t;  // each content-bearing event's arrival
  std::vector<int> ev_c;     // its characters (code points, as Python's len)
};
// Throws std::runtime_error ("HTTP 400: ...", or the connection's error)
Stream stream_chat(const Settings& s, const ojson& body, const std::string& session = "", int timeout_s = 1800);
ojson post_json(const Settings& s, const ojson& body, int timeout_s = 1800);  // one non-streamed reply

// The checks and rates (bench.py)
std::vector<std::string> thinking_leak(const Stream& r);
std::optional<double> distinct4(const std::string& text);
std::optional<int> cached_of(const ojson& usage, const ojson& timings);
struct Rate {
  std::optional<double> mid, e2e;
  ojson info;
};
Rate decode_rates(const std::vector<double>& t, const std::vector<int>& c, int ct, int head, int tail);
double py_round(double x, int digits);  // Python's round(x, digits)
double median(std::vector<double> v);   // statistics.median

// The measurements (bench.py t_decode, t_prefill, t_tools, _summary, tools_pass; compare_engines.py's warm-up). Each
// returns bench.py's record of it. `progress` (may be null) hears what is happening, for the screen.
using Progress = std::function<void(const std::string&)>;
ojson t_decode(const Settings& s, bool greedy, Nonces& n);
ojson t_prefill(const Settings& s, int size, Nonces& n, const Progress& progress = nullptr,
                const std::function<bool()>& stopped = nullptr);
ojson t_tools(const Settings& s, bool think);
bool tools_pass(const ojson& tools);
ojson summary(const ojson& runs, const char* key = "decode_tps");
ojson warmup(const Settings& s, Nonces& n, const Progress& progress = nullptr);

// ---------------------------------------------------------------- the Mac (system.mm)

struct Mac {
  std::string identifier;        // hw.model: "Mac16,7"
  std::string name;              // Apple's name: "MacBook Pro (16-inch, 2024)"; "" when the identifier is not known
  int year = 0;                  // 0 when not known
  std::string chip;              // machdep.cpu.brand_string: "Apple M4 Pro"
  int cpu_cores = 0, cpu_performance_cores = 0, cpu_efficiency_cores = 0;
  int gpu_cores = 0;             // IOKit (AGXAccelerator gpu-core-count); 0 when not found
  uint64_t memory_bytes = 0;     // hw.memsize
  std::string macos, macos_build;
};
Mac this_mac();
// Apple's name and year for an identifier, from the embedded table (false when it does not list it)
bool mac_model(const std::string& identifier, std::string* name, int* year);
// The table itself (bench/community/mac-models.json)
const ojson& mac_models();
// "MacBook Pro (16-inch, 2024)": Apple's name, or for a name with the chip instead of a year (the M5 Macs' "MacBook Pro
// (14-inch, M5)") the year instead of the chip, "MacBook Pro (14-inch, 2025)", as the chip is shown beside it; the
// identifier when not known
std::string mac_display(const std::string& name, int year, const std::string& chip, const std::string& identifier);
std::string chip_short(const std::string& chip);  // "Apple M4 Pro" -> "M4 Pro"

struct Power {
  std::string source;        // "ac", "battery", "ups", "" (unknown)
  int battery_percent = -1;  // -1: no battery, or unknown
  bool low_power_mode = false;
  std::string power_mode;    // "automatic", "low", "high", "" (pmset has none)
};
// with_mode: also the energy mode (pmset; a subprocess, so not for every sample)
Power power(bool with_mode = true);
std::string thermal_state();  // "nominal", "fair", "serious", "critical"
int thermal_rank(const std::string& state);
struct Vm {
  double swap_used_mb = -1;
  int64_t swapouts = -1, swapins = -1, pageouts = -1;
  int memorystatus_level = -1;
};
Vm vm();

// GPU time per process: (pid, its 16-character name) -> accumulated GPU time in ns, summed over its GPU clients
using GpuSnap = std::map<std::pair<int, std::string>, uint64_t>;
GpuSnap gpu_snapshot();
// GPU memory in use (AGXAccelerator "In use system memory"), bytes; -1 when not found
int64_t gpu_memory_in_use();
// The processes of the trees under `roots` (roots included even when gone)
std::set<int> process_tree(const std::vector<int>& roots);
// This process's ancestors (the terminal that runs it, among them)
std::set<int> ancestors();
// A process's app for messages: the outermost .app of its executable ("Google Chrome" for Chrome's GPU helper), else
// the executable's name, else `comm`. Never a path.
std::string app_name(int pid, const std::string& comm);
// Memory of a process tree (proc_pid_rusage, summed: the phys_footprint and phys_footprint_peak `footprint` prints)
struct Footprint {
  uint64_t phys = 0, peak = 0;
  int64_t mapped_clean = -1;  // resident clean mapped files (`footprint -w`), -1 when not read
};
Footprint footprint(const std::set<int>& pids, bool mapped = false);
// Other Kapitsa processes running (not this one, not `except`)
std::vector<int> other_kapitsas(int except = 0);

// The harness's GPU lock (bench/harness/gpu_hogs.py GpuLock): an exclusive flock on BENCH_GPU_LOCK, else
// /tmp/kapitsa-gpu.lock, the lock `lockf -k <file> <cmd>` takes, so a benchmark never runs beside another GPU job that
// takes it. Opened read-only (the file may be immutable) with close-on-exec (the server does not inherit it), created
// if missing, never removed.
class GpuLock {
 public:
  static std::string path();
  ~GpuLock() { release(); }
  // Takes it: at once, or after calling waiting() with who holds it and blocking until they release it. False (errno
  // set) when the file can't be opened.
  bool acquire(const std::function<void(const std::string& holders)>& waiting = nullptr);
  void release();
  bool held() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
};
// Who holds a lock file: "lockf (pid 123), Python (pid 456)"
std::string lock_holders(const std::string& path);

// ---------------------------------------------------------------- GPU contention (monitor.cpp: gpu_hogs.py)

constexpr double kForeignMaxPct = 5.0;  // other processes' GPU load above this, sustained, flags a phase
constexpr int kSustainSamples = 3;      // samples (2 s each) in a row that count as sustained
// Ambient load: never waited for, never flags, recorded apart (gpu_hogs.py's AMBIENT plus BENCH_GPU_AMBIENT)
std::vector<std::string> ambient_names();
bool is_ambient(const std::string& comm);
bool is_ignored(const std::string& comm);  // WindowServer
struct UseRow {
  double pct;
  int pid;
  std::string name;  // 16-character name
};
// Processes using the GPU between two snapshots (more than 0.5%), highest first, without WindowServer, ambient ones and
// `exclude`
std::vector<UseRow> diff_rows(const GpuSnap& a, const GpuSnap& b, double secs, const std::set<int>& exclude);

struct Sample {
  double t = 0;              // end of the sample (seconds, steady clock)
  double foreign = 0;        // other processes' GPU load, % of the GPU
  std::string top;           // the busiest of them (16-character name), "-" when none
  double top_pct = 0;
  double ambient = 0;
  std::vector<std::pair<std::string, double>> users;  // other processes over 0.5%, by app name
  std::string thermal;
  std::string power_source;
  bool low_power = false;
};
// Samples every 2 s until stopped; `own()` is the server's tree (never foreign), recomputed per sample. The processes
// in `ambient_pids` (this process's terminal) count as ambient.
class Monitor {
 public:
  Monitor(std::function<std::set<int>()> own, std::set<int> ambient_pids, double interval = 2.0);
  ~Monitor();
  void start();
  void stop();
  std::vector<Sample> samples();
  static double now();  // the samples' clock (seconds)

 private:
  void run();
  std::function<std::set<int>()> own_;
  std::set<int> ambient_pids_;
  double interval_;
  std::mutex mu_;
  std::vector<Sample> samples_;
  std::atomic<bool> stop_{false};
  std::thread th_;
};
// gpu_hogs.summarize_samples over the samples that end in [t0, t1 + 2.5], with who used the GPU, the worst thermal
// state and the power seen
ojson summarize_samples(const std::vector<Sample>& samples, double t0, double t1);
// gpu_hogs.wait_quiet with own_max: until no other process uses more than `maxpct` of the GPU, and the server's tree
// no more than `maxpct`, for `quiet_s` seconds; false after `timeout_s`. `busy` gets what was busy last. `tick` hears
// each sample's busy list ("" when quiet) and may return false to stop waiting.
struct Busy {
  double pct;
  std::string name;  // an app name, or "the server (still busy after its warm-up)"
};
bool wait_quiet(double maxpct, double quiet_s, double timeout_s, const std::function<std::set<int>()>& own,
                const std::set<int>& ambient_pids, double* waited, std::vector<Busy>* busy,
                const std::function<bool(const std::vector<Busy>&, double)>& tick = nullptr);

// ---------------------------------------------------------------- the result (result.cpp)

// What the command learned about the model it ran
struct ModelInfo {
  std::string name;        // catalog name, or the folder's name
  std::string title;       // "Qwen3.6-35B-A3B"
  std::string variant;     // catalog variant ("4bit"), "" when not a catalog model
  std::string precision;   // "4-bit", "bf16", "NVIDIA 24 GB"
  std::string quant;       // "MLX affine 4-bit, group 64"
  bool catalog = false;
  std::string repo, revision;
  std::string decoding;        // "DFlash2", "LiLiCorr", "DSpark", "MTP", "plain" (from the server's "engine ready" line)
  std::string decoding_detail; // that line's decoding part
  std::string drafter;         // "incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6", "" when none
};
// Everything a run produced, for its result file
struct RunInputs {
  Mac mac;
  ModelInfo model;
  Settings settings;
  bool quick = false;
  int context = 32768, slots = 1;
  std::string version, commit;  // Kapitsa's
  std::string time_utc;         // "2026-10-18T09:12:44Z"
  // bench.py's record of the run: decode, decode_summary, decode_greedy, decode_greedy_summary, prefill, tools,
  // tools_pass; phases {name: [t0, t1]} (Monitor::now) and memory {name: {phys_footprint, phys_footprint_peak,
  // mapped_file_clean, swapouts_delta, swap_used_mb, memorystatus_level}} per phase
  ojson raw;
  std::vector<Sample> samples;
  Power power_start, power_end;
  Vm vm_start, vm_end;
  std::string thermal_start;
  double quiet_wait_s = 0;
  bool quiet = true;
  std::vector<Busy> quiet_busy;
  bool external_server = false;  // tests: a server kapitsa bench did not start (its result is invalid)
};
// The result file's content (bench/community/schema.json)
ojson make_result(const RunInputs& in);
// A model's title as the README writes it: "Qwen3.6 35B-A3B"
std::string model_display(const std::string& title);
// "kapitsa-bench-2026-10-18-m4-pro-qwen3.6-35b-a3b-4bit.json" (the date in UTC)
std::string result_file_name(const ojson& result);
// The pull request's: "Benchmark: MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB · Qwen3.6 35B-A3B 4-bit"
std::string result_title(const ojson& result);
// "MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB"
std::string result_mac_line(const ojson& result);
// The pull request's body: the summary table and the conditions, in Markdown
std::string result_markdown(const ojson& result);
// The checks a result must pass to be submitted ("" when it may be)
std::string submit_problem(const ojson& result);
// Formats for the screens and texts: "153.7 tok/s", "1,169 tok/s", "0.06 s", "21.1 GB"
std::string fmt_tps(double v, bool integer = false);
std::string fmt_secs(double v);
std::string fmt_gb(double bytes);
std::string thousands(long v);

// ---------------------------------------------------------------- submitting (submit.cpp)

constexpr const char* kRepo = "nace-ai/kapitsa";  // KAPITSA_BENCH_REPO overrides it (tests)
std::string target_repo();
struct GhStatus {
  bool installed = false, logged_in = false;
  std::string path, login;
};
GhStatus gh_status();
// The steps of a pull request from `login`'s fork, as the gh commands they run (for --dry-run and the confirmation)
struct PrPlan {
  std::string repo, fork, branch, path, title, body, file_name, content;
  std::vector<std::string> commands;  // shell-quoted, for the screen
};
PrPlan pr_plan(const std::string& repo, const std::string& login, const ojson& result, const std::string& content);
// Runs it with gh: the pull request's URL, or throws std::runtime_error with what failed. `step` hears each step.
std::string open_pr(const PrPlan& plan, const std::string& gh, const Progress& step);
std::string shell_quote(const std::string& s);

}  // namespace ni::bench
