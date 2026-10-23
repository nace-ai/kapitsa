// kapitsa bench: Kapitsa's benchmark on this Mac, with the published protocol (src/bench/: bench/harness's bench.py and
// compare_engines.py, request for request). It starts the model's server as compare_engines.py does (one slot, a
// 32K context, NI_* variables cleared), warms it up, waits for a quiet GPU, measures it over the local API while a
// monitor samples the other processes' GPU time, the thermal state and the power source, then stops it. The result
// (bench/community/schema.json) is saved and, when asked, submitted to the community results: a pull request with
// the user's own gh login, never without an answer at the terminal (one question; Enter is a yes).
#include "bench/bench.h"
#include "engine/engine.h"
#include "gguf/gguf.h"
#include "gguf/import.h"
#include "hub/hub.h"
#include "model/flint.h"
#include "model/nemotron_h.h"
#include "server/beam.h"
#include "server/cli.h"
#include "server/picker.h"
#include "server/ui.h"
#include "update/update.h"
#include "util/log.h"
#include "util/term.h"

#include <httplib/httplib.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <thread>

#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.2.11"
#endif
#ifndef KAPITSA_GIT
#define KAPITSA_GIT ""
#endif

extern char** environ;

namespace ni {

using namespace bench;
using term::accent;
using term::bold;
using term::dim;

namespace {

const char* kHelp =
    "kapitsa bench [--model MODEL] [--quick] [--json] [--submit] [--out FILE]\n\n"
    "Benchmarks Kapitsa on this Mac with the protocol of its published benchmarks (bench/harness): writing speed\n"
    "(512-token coding answers at temperature 0.6, Flint's at 1.0, and greedy, median of 5 runs each), reading a 4K-\n"
    "and a 16K-token prompt, the time to the first token of a follow-up in each conversation, peak memory and tool\n"
    "calls. It starts the model's server for the run and measures it over the local API, as the harness does, and\n"
    "records this Mac and what could skew the numbers (other apps on the GPU, power source and mode, thermal state,\n"
    "swap). The result is saved as JSON. A run takes 3 to 15 minutes; close other apps for a clean one.\n\n"
    "  --model MODEL   35b, 27b, 4b, bolt or flint (<model>:<variant> for another precision), a model folder or a\n"
    "                  .gguf file.\n"
    "                  Default: the best model for this Mac (in a terminal it asks)\n"
    "  --quick         a shorter run: 3 runs per writing test, the 4K prompt only (marked quick in the result)\n"
    "  --json          the result as JSON on stdout (progress on stderr)\n"
    "  --out FILE      where to save the result (default: %s/kapitsa-bench-<date>-<chip>-<model>.json)\n"
    "  --submit        add the result to the community results: asks once (Enter is yes), then opens a pull request\n"
    "                  from your fork with your gh login (it needs gh, logged in)\n"
    "  --dry-run       with --submit: print what it would send, the gh commands and the pull request; nothing is sent\n"
    "  --yes           don't ask which model to run: the best for this Mac (submitting still asks)\n\n"
    "  kapitsa bench submit [FILE] [--dry-run]   submit a saved result (default: the newest)\n"
    "  kapitsa bench compare [--model M] ...      Kapitsa against the other engines installed here, one table\n"
    "                                            (kapitsa bench compare --help)\n\n"
    "Community results, and how they are checked: https://github.com/nace-ai/kapitsa/blob/main/bench/community\n";

volatile sig_atomic_t g_stop = 0;    // a stop signal
volatile pid_t g_server = 0;         // the server to stop with it

void on_stop(int sig) {
  g_stop = sig;
  if (g_server > 0) kill(g_server, SIGTERM);
}

bool stopping() { return g_stop != 0; }

struct Stopped {};

std::string bench_dir() { return hub::home_dir() + "/bench"; }

std::string self_path() {
  char buf[PATH_MAX];
  uint32_t n = sizeof buf;
  if (_NSGetExecutablePath(buf, &n) != 0) return "kapitsa";
  char real[PATH_MAX];
  return realpath(buf, real) ? real : buf;
}

int free_port() {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 0;
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port = 0;
  socklen_t len = sizeof a;
  int port = 0;
  if (bind(fd, (sockaddr*)&a, sizeof a) == 0 && getsockname(fd, (sockaddr*)&a, &len) == 0) port = ntohs(a.sin_port);
  close(fd);
  return port;
}

int http_status(int port, const std::string& path, int timeout_s) {
  httplib::Client c("127.0.0.1", port);
  c.set_connection_timeout(timeout_s, 0);
  c.set_read_timeout(timeout_s, 0);
  auto r = c.Get(path);
  return r ? r->status : -1;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// The last line of a log, without the "[kapitsa] " prefix
std::string last_line(const std::string& path) {
  std::string s = read_file(path);
  if (s.size() > 4096) s = s.substr(s.size() - 4096);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  const size_t a = s.find_last_of("\n\r");
  if (a != std::string::npos) s = s.substr(a + 1);
  if (s.rfind("[kapitsa] ", 0) == 0) s = s.substr(10);
  return s;
}

std::string tail(const std::string& path, int lines) {
  const std::string s = read_file(path);
  size_t at = s.size();
  for (int n = 0; n <= lines && at > 0;) {
    at = s.rfind('\n', at - 1);
    if (at == std::string::npos) {
      at = 0;
      break;
    }
    ++n;
  }
  return s.substr(at == 0 ? 0 : at + 1);
}

// ---------------------------------------------------------------- the screen

bool g_tui = false;   // the styled screen (a terminal, not --json)
bool g_plain = true;  // progress as plain lines on stderr

void say(const std::string& plain) {
  if (!g_tui) fprintf(stderr, "%s\n", plain.c_str());
}

// A status line redrawn 4 times a second while a phase runs (the terminal's own GPU work stays small)
class Ticker {
 public:
  explicit Ticker(std::string label) : label_(std::move(label)) {
    if (!g_tui || !term::live()) return;
    th_ = std::thread([this] {
      static const char* const frames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
      for (int i = 0; !stop_; ++i) {
        std::string s;
        {
          std::lock_guard<std::mutex> lk(mu_);
          s = "  " + accent(frames[i % 10]) + " " + label_ + "  " + dim(term::duration(secs()));
          if (!detail_.empty()) s += "  " + dim(detail_);
        }
        term::status(term::clip(s, term::columns() - 1));
        for (int k = 0; k < 25 && !stop_; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }
  ~Ticker() { done(); }
  void label(const std::string& s) {
    std::lock_guard<std::mutex> lk(mu_);
    label_ = s;
  }
  void detail(const std::string& s) {
    std::lock_guard<std::mutex> lk(mu_);
    detail_ = s;
  }
  double secs() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
  void done() {
    stop_ = true;
    if (th_.joinable()) {
      th_.join();
      term::status("");
    }
  }

 private:
  std::mutex mu_;
  std::string label_, detail_;
  std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
  std::atomic<bool> stop_{false};
  std::thread th_;
};

// "  ✓ Writing, T = 0.6     153.7 tok/s   146.6–165.8, 5 runs"
void phase_line(const std::string& what, const std::string& value, const std::string& detail, bool ok = true) {
  if (!g_tui) return;
  std::string l = "  " + (ok ? accent(term::kOk) : term::warn(term::kWarn)) + " " + term::left(what, 18) + "  " +
                  term::left(bold(value), 13) + (detail.empty() ? "" : "  " + dim(detail));
  term::line(term::clip(l, term::columns() - 1), stderr);
}

void warn_line(const std::string& s) {
  if (g_tui) ui::warning(s);
  else fprintf(stderr, "warning: %s\n", s.c_str());
}

std::string repeat(const char* g, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) s += g;
  return s;
}

void card(const std::vector<std::string>& rows, int inner, FILE* f) {
  std::vector<std::string> o;
  o.push_back("  " + dim("╭" + repeat("─", inner + 4) + "╮"));
  for (const auto& r : rows) o.push_back("  " + dim("│") + "  " + term::left(term::clip(r, inner), inner) + "  " + dim("│"));
  o.push_back("  " + dim("╰" + repeat("─", inner + 4) + "╯"));
  for (const auto& l : o) term::line(l, f);
}

// One question at the terminal, Enter for yes: true for an empty answer, "y" or "yes", false for anything else and at
// the end of input. A long question is wrapped to the terminal's width, its last line asked.
bool ask_yes(const std::string& question) {
  const int fd = open("/dev/tty", O_RDWR | O_CLOEXEC);
  if (fd < 0) return false;
  size_t ind = 0;
  while (ind < question.size() && question[ind] == ' ') ++ind;
  const std::string pad(ind, ' ');
  const auto lines = term::wrap(question.substr(ind), std::max(30, std::min(term::columns(), 100) - int(ind) - 7));
  for (size_t i = 0; i + 1 < lines.size(); ++i) term::line(pad + lines[i], stderr);
  const std::string q = pad + lines.back() + " [Y/n] ";
  if (write(fd, q.data(), q.size()) < 0) {
    close(fd);
    return false;
  }
  std::string ans;
  bool line = false;
  for (char c; read(fd, &c, 1) == 1;) {
    if (c == '\n') {
      line = true;
      break;
    }
    ans += c;
  }
  close(fd);
  if (!line) return false;  // the end of input (Ctrl-D) is a no
  for (auto& ch : ans) ch = char(tolower(uint8_t(ch)));
  while (!ans.empty() && isspace(uint8_t(ans.back()))) ans.pop_back();
  while (!ans.empty() && isspace(uint8_t(ans.front()))) ans.erase(ans.begin());
  return ans.empty() || ans == "y" || ans == "yes";
}

bool can_confirm() {
  const int fd = open("/dev/tty", O_RDWR | O_CLOEXEC);
  if (fd < 0) return false;
  close(fd);
  return isatty(0) != 0;
}

// ---------------------------------------------------------------- the model

std::string precision_of(const hub::Variant& v) {
  if (!v.label.empty()) return v.label.substr(0, v.label.find(" ("));  // "NVIDIA 24 GB"
  std::string s = v.name;
  if (s.size() > 3 && s.compare(s.size() - 3, 3, "bit") == 0) s.insert(s.size() - 3, "-");
  return s;
}

// A folder's precision from its config.json ("4-bit", "bf16")
std::string folder_precision(const std::string& dir, std::string* quant) {
  const ojson c = ojson::parse(read_file(dir + "/config.json"), nullptr, false);
  if (!c.is_discarded() && c.is_object()) {
    const ojson q = c.contains("quantization") ? c["quantization"] : c.contains("quantization_config") ? c["quantization_config"] : ojson();
    if (q.is_object() && q.contains("bits") && q["bits"].is_number()) {
      const int bits = q["bits"];
      const int g = q.value("group_size", 64);
      *quant = "MLX affine " + std::to_string(bits) + "-bit, group " + std::to_string(g);
      return std::to_string(bits) + "-bit";
    }
  }
  *quant = "bf16";
  return "bf16";
}

ModelInfo model_info(const hub::ModelRef& ref) {
  ModelInfo m;
  if (ref.kind == hub::ModelRef::kCatalog && ref.model && ref.variant) {
    m.catalog = true;
    m.name = ref.model->name;
    m.title = ref.model->title;
    m.variant = ref.variant->name;
    m.precision = precision_of(*ref.variant);
    m.quant = ref.variant->quant;
    m.repo = ref.variant->repo;
    m.revision = ref.variant->revision;
    return m;
  }
  m.name = ref.dir.substr(ref.dir.find_last_of('/') + 1);
  m.title = m.name;
  m.precision = folder_precision(ref.dir, &m.quant);
  m.repo = ref.repo;
  m.revision = ref.revision;
  return m;
}

// The drafter a catalog variant comes with, "repo@rev7" (when the server uses one)
std::string drafter_of(const hub::ModelRef& ref) {
  if (!ref.variant) return "";
  for (const auto& c : ref.variant->companions)
    if (c.drafter()) return c.repo + "@" + c.revision.substr(0, 7);
  return "";
}

// ---------------------------------------------------------------- the server

struct Server {
  pid_t pid = 0;
  int port = 0;
  std::string log;
  int64_t gpu_mem_before = -1;
};

void stop_server(Server& s) {
  if (s.pid <= 0) return;
  kill(s.pid, SIGTERM);
  int st;
  for (int i = 0; i < 150 && waitpid(s.pid, &st, WNOHANG) == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (waitpid(s.pid, &st, WNOHANG) == 0) {
    kill(s.pid, SIGKILL);
    waitpid(s.pid, &st, 0);
  }
  g_server = 0;
  s.pid = 0;
  // its memory back before the GPU lock goes (compare_engines.py: GPU memory within 1 GB of the baseline)
  if (s.gpu_mem_before >= 0)
    for (int i = 0; i < 60; ++i) {
      const int64_t now = gpu_memory_in_use();
      if (now < 0 || now <= s.gpu_mem_before + (int64_t(1) << 30)) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// ---------------------------------------------------------------- submitting

std::string pretty(const ojson& j) { return j.dump(2) + "\n"; }

// A no: the result stays where it was saved
int not_sent(const std::string& file) {
  term::line("  " + dim("Not sent. kapitsa bench submit " + hub::tilde(file) + " sends it later."), stderr);
  return 0;
}

// Adds a result (saved as `file`) to the community results after one question at the terminal, Enter for yes: a pull
// request from the user's fork with their gh login. Without gh there is no other way in (no issue, no browser form): it
// says how to get gh, or how to open the pull request by hand. `offered` is the question at the end of a run, where a
// Mac without gh is not an error.
int submit(const ojson& result, const std::string& content, const std::string& file, bool dry, bool offered = false) {
  if (const std::string why = submit_problem(result); !why.empty()) return fail(why);
  const std::string repo = target_repo();
  const GhStatus gh = gh_status();
  if (!gh.logged_in) {
    const PrPlan plan = pr_plan(repo, "<your gh login>", result, content);
    const std::string why = gh.installed ? "gh is not logged in" : "gh is not installed";
    if (dry) {
      printf("Dry run: nothing is sent. %s, and a result is added only as a pull request from your fork. With gh logged in it "
             "would open one on %s:\n\n",
             why.c_str(), repo.c_str());
      for (const auto& c : plan.commands) printf("  %s\n", c.c_str());
      printf("\nFile %s:\n%s\nPull request \"%s\":\n\n%s\n", plan.path.c_str(), content.c_str(), plan.title.c_str(),
             plan.body.c_str());
      return 0;
    }
    const std::string how = std::string("Results are added as a pull request from your fork, and ") + why + ". " +
                            (gh.installed ? "Log in with gh auth login" : "Install it (brew install gh), log in with gh auth login") +
                            ", then kapitsa bench submit " + hub::tilde(file) + ". Or by hand: fork " + repo + ", add " +
                            hub::tilde(file) + " as " + plan.path + " and open a pull request.";
    if (offered) {
      term::line("  " + dim(how), stderr);
      return 0;
    }
    return fail(how);
  }
  const PrPlan plan = pr_plan(repo, gh.login, result, content);
  if (dry) {
    printf("Dry run: nothing is sent. With your gh login (@%s) this would open a pull request on %s:\n\n", gh.login.c_str(),
           repo.c_str());
    for (const auto& c : plan.commands) printf("  %s\n", c.c_str());
    printf("\nFile %s:\n%s\nPull request \"%s\":\n\n%s\n", plan.path.c_str(), content.c_str(), plan.title.c_str(),
           plan.body.c_str());
    return 0;
  }
  if (!can_confirm()) return fail("Submitting asks first, at a terminal. Run it in one: kapitsa bench submit FILE");
  if (!ask_yes("  Add it to the community results? It opens a pull request on " + repo + " as @" + gh.login + "."))
    return not_sent(file);
  try {
    Ticker t("Submitting");
    const std::string url = open_pr(plan, gh.path, [&](const std::string& step) {
      t.detail(step);
      if (!g_tui) fprintf(stderr, "%s\n", step.c_str());
    });
    t.done();
    if (g_tui) ui::ok("Pull request opened: " + bold(url));
    else fprintf(stderr, "Pull request opened: %s\n", url.c_str());
    return 0;
  } catch (const std::exception& e) {
    return fail(std::string(e.what()) + " Nothing else was changed; kapitsa bench submit FILE tries again.");
  }
}

std::string newest_result() {
  std::string best;
  time_t when = 0;
  if (DIR* d = opendir(bench_dir().c_str())) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.rfind("kapitsa-bench-", 0) != 0 || n.size() < 5 || n.compare(n.size() - 5, 5, ".json") != 0) continue;
      struct stat st;
      const std::string p = bench_dir() + "/" + n;
      if (stat(p.c_str(), &st) == 0 && st.st_mtime >= when) when = st.st_mtime, best = p;
    }
    closedir(d);
  }
  return best;
}

void print_summary(const ojson& r, const std::string& path);

int submit_saved(const std::vector<std::string>& args) {
  std::string file;
  bool dry = false;
  for (const auto& a : args) {
    if (a == "--dry-run") dry = true;
    else if (a == "--help" || a == "-h") {
      printf("kapitsa bench submit [FILE] [--dry-run]: adds a saved kapitsa bench result to the community results (default:\n"
             "the newest in %s). It shows the result and asks once (Enter is yes); --dry-run shows what it would send.\n",
             hub::tilde(bench_dir()).c_str());
      return 0;
    } else if (!a.empty() && a[0] == '-') usage_error("unknown option " + a, "kapitsa bench --help");
    else file = expand_home(a);
  }
  if (file.empty()) file = newest_result();
  if (file.empty()) return fail("No saved result in " + hub::tilde(bench_dir()) + ". Run kapitsa bench first, or name the file.", 2);
  const std::string content = read_file(file);
  const ojson r = ojson::parse(content, nullptr, false);
  if (content.empty() || r.is_discarded() || !r.is_object()) return fail(file + " is not a kapitsa bench result.", 2);
  g_tui = term::fancy();
  // what it sends, as the run showed it
  if (!dry && submit_problem(r).empty()) {
    if (g_tui) print_summary(r, "");
    else fprintf(stderr, "%s\n%s", result_title(r).c_str(), result_markdown(r).c_str());
  }
  if (g_tui) term::line("", stderr);
  return submit(r, content, file, dry);
}

// ---------------------------------------------------------------- the summary

std::string conditions_text(const ojson& r) {
  const ojson& c = r["conditions"];
  std::vector<std::string> v;
  const std::string src = c["power_source"].is_string() ? c["power_source"].get<std::string>() : "";
  v.push_back(src == "ac" ? "AC power" : src == "battery" ? "on battery" : src == "ups" ? "UPS power" : "power unknown");
  if (c["power_mode"].is_string()) {
    const std::string m = c["power_mode"];
    v.push_back(m == "high" ? "High Power" : m == "low" ? "Low Power" : "Automatic power mode");
  }
  if (c["thermal_state"].is_string()) v.push_back("thermal " + c["thermal_state"].get<std::string>());
  const long swapped = c.value("swapouts", 0L);
  v.push_back(swapped > 1000 ? thousands(swapped) + " pages swapped" : "no swap");
  std::string s;
  for (const auto& x : v) s += (s.empty() ? "" : "  " + std::string(term::kDot) + "  ") + x;
  return s;
}

// The card's GPU line: who else used the GPU when that disturbed a measurement (its numbers carry ⚠)
std::string gpu_text(const ojson& r) {
  const ojson& c = r["conditions"];
  char b[200];
  if (!c["contention"].empty()) {
    std::vector<std::string> who;
    for (const auto& u : c["gpu_users"]) who.push_back(u.get<std::string>());
    std::string names;
    for (size_t i = 0; i < who.size(); ++i) names += (i == 0 ? "" : i + 1 == who.size() ? " and " : ", ") + who[i];
    snprintf(b, sizeof b, " used the GPU: %.0f%% on average, up to %.0f%%. Close %s for a clean run.",
             c.value("gpu_other_mean_pct", 0.0), c.value("gpu_other_max_pct", 0.0), who.size() == 1 ? "it" : "them");
    return "⚠ " + (names.empty() ? std::string("Other apps") : names) + b;
  }
  snprintf(b, sizeof b, "other apps under 5%% of the GPU (at most %.1f%%)", c.value("gpu_other_max_pct", 0.0));
  return b;
}

void print_summary(const ojson& r, const std::string& path) {
  const ojson& res = r["results"];
  const ojson& m = r["model"];
  const ojson& mac = r["mac"];
  const bool quick = r.value("mode", "") == "quick";
  const int W = std::min(term::columns(), 100);
  const int inner = std::max(40, std::min(W - 8, 72));
  const std::string dot = dim("   " + std::string(term::kDot) + "   ");
  std::set<std::string> busy;
  for (const auto& p : r["conditions"]["contention"]) busy.insert(p.get<std::string>());
  auto mark = [&](const std::string& phase) { return busy.count(phase) ? " " + term::warn("⚠") : std::string(); };
  std::vector<std::string> rows;
  auto row = [&](const std::string& k, const std::string& v) { rows.push_back(term::left(dim(k), 13) + v); };
  {
    const std::string name = bold(model_display(m.value("title", m.value("name", std::string())))) +
                             dim("  " + std::string(term::kDot) + "  ") + m.value("precision", std::string());
    const std::string tag = quick ? term::warn(term::kOn) + " quick" : accent(term::kOn) + " full run";
    const int gap = inner - term::width(name) - term::width(tag);
    rows.push_back(gap >= 2 ? name + std::string(size_t(gap), ' ') + tag : name);
  }
  {
    const std::string d = "  " + std::string(term::kDot) + "  ";
    std::string l = mac_display(mac["name"].is_string() ? mac["name"].get<std::string>() : "", mac["year"].is_number() ? mac["year"].get<int>() : 0,
                                mac.value("chip", std::string()), mac.value("identifier", std::string())) +
                    d + chip_short(mac.value("chip", std::string()));
    if (mac["gpu_cores"].is_number()) l += " " + std::to_string(mac["gpu_cores"].get<int>()) + "-core GPU";
    l += d + std::to_string(mac.value("memory_gb", 0)) + " GB";
    rows.push_back(dim(l));
  }
  rows.push_back("");
  if (res["decode"].is_object() && res["decode"]["median_tps"].is_number()) {
    std::string v = bold(fmt_tps(res["decode"]["median_tps"])) + mark("decode") + dim("  T = " + fmt_temp(writing_temperature(r)));
    if (res["decode_greedy"].is_object() && res["decode_greedy"]["median_tps"].is_number())
      v += dot + bold(fmt_tps(res["decode_greedy"]["median_tps"])) + mark("decode_greedy") + dim("  greedy");
    row("Writing", v);
  }
  std::string reading, follow;
  for (const auto& p : res["prefill"]) {
    const std::string k = std::to_string(p["size"].get<int>() / 1000) + "K", ph = "prefill_" + std::to_string(p["size"].get<int>());
    if (p["tps"].is_number()) reading += (reading.empty() ? "" : dot) + bold(fmt_tps(p["tps"], true)) + mark(ph) + dim("  " + k);
    if (p["followup_median_s"].is_number())
      follow += (follow.empty() ? "" : dot) + bold(fmt_secs(p["followup_median_s"])) + mark(ph) + dim("  at " + k) +
                (p.contains("followup_suffix_tps") && p["followup_suffix_tps"].is_number()
                     ? dim(" (new tokens " + fmt_tps(p["followup_suffix_tps"].get<double>(), true) + ")")
                     : std::string());
  }
  if (!reading.empty()) row("Reading", reading);
  if (!follow.empty()) row("Follow-up", follow);
  if (res["peak_memory_gb"].is_number()) {
    char b[48];
    snprintf(b, sizeof b, "%.1f GB", res["peak_memory_gb"].get<double>());
    std::string v = bold(b) + dim(" peak");
    if (res.contains("mapped_gb")) snprintf(b, sizeof b, " + %.1f GB mapped", res["mapped_gb"].get<double>()), v += dim(b);
    row("Memory", v);
  }
  row("Tool calls", res.value("tool_calls", std::string()) == "pass" ? accent("pass") : term::red("fail"));
  row("Decoding", m.value("decoding_detail", m.value("decoding", std::string())));
  rows.push_back("");
  row("Conditions", conditions_text(r));
  for (const auto& l : term::wrap(gpu_text(r), inner - 13))
    rows.push_back(term::left("", 13) + (busy.empty() ? dim(l) : term::warn(l)));
  term::line("", stderr);
  card(rows, inner, stderr);
  for (const auto& f : r["flags"])  // contention is on the card, in words
    if (f.get<std::string>().rfind("CONTENTION", 0) != 0) ui::warning(f.get<std::string>());
  for (const auto& f : r["invalid"]) term::line("  " + term::red(term::kFail) + " " + f.get<std::string>(), stderr);
  if (!path.empty()) ui::note("Saved " + hub::tilde(path));
}

// ---------------------------------------------------------------- the run

struct Options {
  std::string model, out;
  bool quick = false, json = false, submit = false, dry = false, yes = false;
};

std::string utc_now() {
  const time_t t = time(nullptr);
  char b[32];
  strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
  return b;
}

int run(const Options& o) {
  // ---- the model
  EngineOptions eo;  // serve's defaults: the model `kapitsa serve` would start
  const size_t budget = default_mem_budget();
  std::string model = o.model;
  if (!model.empty() && gguf::is_gguf_file(expand_home(model))) {
    try {
      gguf::ImportOptions io;
      io.file = expand_home(model);
      model = gguf::import_model(io);
    } catch (const gguf::Error& e) {
      return fail(e.what(), 2);
    }
  }
  if (g_tui) term::line("\n  " + term::badge(" kapitsa ") + "  " + dim(std::string(KAPITSA_VERSION) + "  " + term::kDot + "  bench"));
  if (model.empty() && !o.json && ui::can_prompt(o.yes)) {
    const auto pick = ui::pick_model(eo, budget, "kapitsa bench", 0, false);  // the benchmark runs its own context
    if (!pick) return 130;
    model = pick->model;
  }
  hub::ModelRef ref;
  try {
    if (model.empty()) {
      std::string why;
      if (!best_model(eo, budget, &ref, &why)) return fail(why);
      ref = for_this_mac(ref, eo, budget);
    } else {
      ref = for_this_mac(hub::resolve(expand_home(model)), eo, budget);
    }
  } catch (const hub::Error& e) {
    return fail(e.what(), 2);
  }
  if (ref.kind == hub::ModelRef::kLocal) {
    struct stat st;
    if (stat((ref.dir + "/config.json").c_str(), &st) != 0)
      return fail(ref.dir + " has no config.json. --model takes a model folder, a model name (kapitsa --help) or a Hugging Face repo.", 2);
  }
  ModelInfo mi = model_info(ref);
  const Mac mac = this_mac();
  const std::string mac_name = mac_display(mac.name, mac.year, mac.chip, mac.identifier);
  {
    char b[64];
    snprintf(b, sizeof b, "%.0f GB", double(mac.memory_bytes) / 1073741824.0);
    const std::string dot = "  " + std::string(term::kDot) + "  ";
    if (g_tui) {
      term::line("");
      term::line("  " + mac_name + dim(dot) + chip_short(mac.chip) + dim(dot) + std::to_string(mac.gpu_cores) + "-core GPU" + dim(dot) + b +
                 dim(dot + "macOS " + mac.macos));
      term::line("  " + bold(model_display(mi.title)) + dim(dot) + mi.precision +
                 (o.quick ? dim(dot + "quick: 3 runs per writing test, the 4K prompt only") : std::string()));
      term::line("");
    } else {
      fprintf(stderr, "kapitsa bench: %s %s on %s: %s, %d-core GPU, %s, macOS %s%s\n", model_display(mi.title).c_str(),
              mi.precision.c_str(), mac_name.c_str(), mac.chip.c_str(), mac.gpu_cores, b, mac.macos.c_str(), o.quick ? ", quick" : "");
    }
  }
  if (sha256_hex(corpus()) != kCorpusSha256)
    return fail("This build's prompt corpus is not the frozen one (bench/harness/corpus): its results would not compare.");
  for (int p : other_kapitsas())
    warn_line("Another Kapitsa is running (pid " + std::to_string(p) + "): its model's memory and any requests it serves skew the numbers.");

  // tests (tests/test_bench.py): KAPITSA_BENCH_SERVER=<port> measures a server already listening on 127.0.0.1:<port>
  // instead of starting one (no download, lock or quiet wait); its result is invalid, as nothing about that server is known
  const char* ext_env = getenv("KAPITSA_BENCH_SERVER");
  const int external = ext_env ? atoi(ext_env) : 0;

  // ---- the download, before the GPU lock (it needs no GPU)
  try {
    if (ref.kind != hub::ModelRef::kLocal && !external) {
      hub::PullOptions po;
      if (!o.json && ui::can_prompt(o.yes)) po.make_room = ui::make_room;  // out of disk space: what to delete, asked
      ref.dir = hub::ensure(ref, po);
    }
  } catch (const hub::Stopped& e) {
    if (g_tui) term::status(""), ui::warning(e.what());
    else fprintf(stderr, "\n%s\n", e.what());
    return 130;
  } catch (const hub::Error& e) {
    return fail(e.what());
  }

  Settings s;
  s.model = ref.id.empty() ? mi.name : ref.id;
  if (o.quick) s.decode_runs = 3, s.greedy_runs = 3, s.sizes = {4000};
  const bool flint = Flint::is_model(ref.dir);
  s.ref_prompt_tokens = reference_prompt_tokens(flint ? "flint"
                                                : NemotronH::is_model(ref.dir) || Dory::is_model(ref.dir) ? "nemotron-h"
                                                                                                          : "qwen3.5");
  if (flint) {  // as configs/flint.json: its card's temperature, and its reasoning before every answer
    s.temperature = 1.0;
    s.always_reasons = true;
  }

  // ---- the GPU lock: the harness's (gpu_hogs.GpuLock), held from the server's start until its memory is back, unless
  // the caller holds it (NACE_GPU_LOCK_HELD, as for bench.py)
  GpuLock lock;
  if (!env_on("NACE_GPU_LOCK_HELD") && !external) {
    std::unique_ptr<Ticker> t;
    const bool ok = lock.acquire([&](const std::string& who) {
      say("Waiting for the GPU lock " + GpuLock::path() + (who.empty() ? "" : ", held by " + who));
      t = std::make_unique<Ticker>("Waiting for the GPU lock");
      if (!who.empty()) t->detail("held by " + who);
    });
    if (t) {
      t->done();
      if (g_tui && ok) ui::ok("GPU lock taken after " + term::duration(t->secs()));
    }
    if (!ok) warn_line("Can't take the GPU lock " + GpuLock::path() + " (" + strerror(errno) + "): running without it.");
  }

  // ---- the server: compare_engines.py's start (its own session, NI_* cleared, one slot, 32K context)
  Server srv;
  srv.port = external ? external : free_port();
  if (srv.port <= 0) return fail("Can't find a free port for the server.");
  struct sigaction act {}, old_int, old_term, old_hup;
  act.sa_handler = on_stop;
  sigemptyset(&act.sa_mask);
  sigaction(SIGINT, &act, &old_int);
  sigaction(SIGTERM, &act, &old_term);
  sigaction(SIGHUP, &act, &old_hup);
  if (!external) {  // start one
    mkdir(hub::home_dir().c_str(), 0755);
    mkdir((hub::home_dir() + "/logs").c_str(), 0755);
    srv.log = hub::home_dir() + "/logs/bench-" + std::to_string(srv.port) + ".log";
    srv.gpu_mem_before = gpu_memory_in_use();
    // (no SSD prefix cache: a benchmark writes nothing to the disk, and no run reads what an earlier one left there)
    std::vector<std::string> sargv = {self_path(), "serve", "--model", ref.dir, "--model-id", s.model, "--host", "127.0.0.1",
                                      "--port", std::to_string(srv.port), "--ctx", "32768", "--slots", "1",
                                      "--disk-cache", "0"};
    std::vector<std::string> senv, cleared;
    for (char** e = environ; *e; ++e) {
      const std::string kv = *e;
      if (kv.rfind("NI_", 0) == 0) {
        cleared.push_back(kv.substr(0, kv.find('=')));
        continue;
      }
      if (kv.rfind("KAPITSA_PARENT_PID=", 0) == 0 || kv.rfind("KAPITSA_NO_PROMPT=", 0) == 0) continue;
      senv.push_back(kv);
    }
    if (!cleared.empty()) {
      std::string names;
      for (const auto& c : cleared) names += (names.empty() ? "" : ", ") + c;
      if (g_tui) ui::note("Left out of the server, as the harness does: " + names);
      else say("Left out of the server, as the harness does: " + names);
    }
    senv.push_back("KAPITSA_PARENT_PID=" + std::to_string(getpid()));  // it exits with this process, however it ends
    senv.push_back("KAPITSA_NO_PROMPT=1");
    senv.push_back("KAPITSA_NO_AUTO_UPDATE=1");  // no update check running beside the measurements
    std::vector<char*> sa, se;
    for (auto& x : sargv) sa.push_back(x.data());
    sa.push_back(nullptr);
    for (auto& x : senv) se.push_back(x.data());
    se.push_back(nullptr);
    const int lfd = open(srv.log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (lfd < 0) return fail("Can't write the server's log " + hub::tilde(srv.log) + ": " + strerror(errno));
    const pid_t child = fork();
    if (child < 0) return fail(std::string("Can't start the server: ") + strerror(errno));
    if (child == 0) {
      setsid();
      const int null = open("/dev/null", O_RDONLY);
      dup2(null, 0);
      dup2(lfd, 1);
      dup2(lfd, 2);
      execve(sa[0], sa.data(), se.data());
      _exit(127);
    }
    close(lfd);
    srv.pid = child;
    g_server = child;
  }
  auto own = [&] { return srv.pid > 0 ? process_tree({int(srv.pid)}) : std::set<int>{}; };
  auto cleanup = [&] {
    stop_server(srv);
    lock.release();
    sigaction(SIGINT, &old_int, nullptr);
    sigaction(SIGTERM, &old_term, nullptr);
    sigaction(SIGHUP, &old_hup, nullptr);
  };

  RunInputs in;
  in.mac = mac;
  in.quick = o.quick;
  in.version = KAPITSA_VERSION;
  in.commit = KAPITSA_GIT;
  in.time_utc = utc_now();
  in.external_server = external != 0;
  in.power_start = power();
  in.vm_start = vm();
  in.thermal_start = thermal_state();
  ojson& raw = in.raw;
  raw["phases"] = ojson::object();
  raw["memory"] = ojson::object();
  Nonces nonces;
  int code = 0;
  std::string out_path;
  try {
    // wait for it, showing what it does
    if (external) {
      mi.decoding = "unknown";
      mi.decoding_detail = "a server kapitsa bench did not start (KAPITSA_BENCH_SERVER)";
      say("Measuring the server on 127.0.0.1:" + std::to_string(external) + " (KAPITSA_BENCH_SERVER)");
    } else {
      const auto t0 = std::chrono::steady_clock::now();
      auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
      const std::string what = bold(model_display(mi.title)) + dim("  " + std::string(term::kDot) + "  ") + mi.precision;
      std::unique_ptr<ui::Loader> wobble;
      std::unique_ptr<Ticker> ticker;
      if (g_tui && ui::can_animate())
        wobble = std::make_unique<ui::Loader>(
            [&] {
              return std::vector<std::string>{"Loading " + what + "  " + dim(term::duration(secs())),
                                              dim(term::clip(last_line(srv.log), std::max(20, term::columns() - ui::kLogoCols - 8)))};
            },
            2);
      else ticker = std::make_unique<Ticker>("Loading " + what);
      say("Starting the server on 127.0.0.1:" + std::to_string(srv.port) + " (log " + hub::tilde(srv.log) + ")");
      bool ready = false;
      while (!stopping() && secs() < 900) {
        int st;
        if (waitpid(srv.pid, &st, WNOHANG) == srv.pid) {
          srv.pid = 0;
          g_server = 0;
          if (wobble) wobble->done({term::red(term::kFail) + " The server stopped while loading", ""});
          ticker.reset();
          term::status("");
          fprintf(stderr, "%s", tail(srv.log, 15).c_str());
          throw std::runtime_error("The server stopped while loading the model. Its log: " + hub::tilde(srv.log));
        }
        if (http_status(srv.port, "/health", 1) == 200 && http_status(srv.port, "/v1/models", 5) == 200) {
          ready = true;
          break;
        }
        if (ticker) ticker->detail(term::clip(last_line(srv.log), 60));
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
      if (stopping()) throw Stopped{};
      if (!ready) throw std::runtime_error("The server did not answer within 15 minutes. Its log: " + hub::tilde(srv.log));
      // what it decodes with: its "engine ready" line
      std::string detail;
      {
        const std::string log = read_file(srv.log);
        const size_t p = log.rfind("engine ready: ");
        if (p != std::string::npos) {
          std::string l = log.substr(p, log.find('\n', p) - p);
          if (const size_t k = l.find(" slots, "); k != std::string::npos) l = l.substr(k + 8);
          if (const size_t k = l.find(", images on"); k != std::string::npos) l.resize(k);
          detail = l;
        }
      }
      mi.decoding_detail = detail;
      mi.decoding = detail.rfind("DFlash2", 0) == 0 ? "DFlash2" : detail.rfind("LiLiCorr", 0) == 0 ? "LiLiCorr"
                    : detail.rfind("DSpark", 0) == 0 ? "DSpark" : detail.rfind("MTP", 0) == 0 ? "MTP"
                    : detail.rfind("plain", 0) == 0 ? "plain" : "unknown";
      if (mi.decoding == "DFlash2" || mi.decoding == "LiLiCorr" || mi.decoding == "DSpark") mi.drafter = drafter_of(ref);
      const std::string done = accent(term::kOk) + " Loaded " + what + dim(" in " + term::duration(secs()));
      // beside the logo, each line within the terminal's width: a wrapped line would misplace the logo's last frame
      const int room = std::max(20, term::columns() - ui::kLogoCols - 6);
      if (wobble) wobble->done({term::clip(done, room), dim(term::clip("  " + detail, room))});
      if (ticker) ticker->done(), term::line(term::clip("  " + done + dim("  " + std::string(term::kDot) + "  " + detail), term::columns() - 1), stderr);
      say("Ready after " + term::duration(secs()) + ": " + detail);
    }
    s.port = srv.port;

    // warm-up (compare_engines.py), then a quiet GPU
    {
      Ticker t("Warming up");
      const double t0 = Monitor::now();
      raw["warmup"] = warmup(s, nonces, [&](const std::string& step) { t.detail(step); });
      raw["phases"]["warmup"] = {t0, Monitor::now()};
      t.done();
      if (stopping()) throw Stopped{};
      bool any = false;
      for (const auto& x : raw["warmup"]["steps"]) any = any || x["ok"].get<bool>();
      if (!any) throw std::runtime_error("The warm-up failed: " + raw["warmup"]["steps"][0].value("error", std::string("no answer")));
      phase_line("Warmed up", term::duration(t.secs()), "a 2.9K-token prompt and a tool call");
      say("Warm-up done in " + term::duration(t.secs()));
    }
    const std::set<int> terminal = ancestors();  // the terminal that shows this run: ambient, like the harness's driving app
    if (!external) {
      Ticker t("Waiting for a quiet GPU");
      std::vector<Busy> busy;
      double waited = 0;
      in.quiet = wait_quiet(kForeignMaxPct, 15, o.quick ? 120 : 300, own, terminal, &waited, &busy,
                            [&](const std::vector<Busy>& b, double) {
                              std::string d;
                              for (const auto& x : b) {
                                char pc[32];
                                snprintf(pc, sizeof pc, " %.0f%%", x.pct);
                                d += (d.empty() ? "" : ", ") + x.name + pc;
                              }
                              t.detail(d.empty() ? "quiet" : "busy: " + d);
                              return !stopping();
                            });
      t.done();
      if (stopping()) throw Stopped{};
      in.quiet_wait_s = waited;
      in.quiet_busy = busy;
      if (in.quiet) {
        phase_line("GPU quiet", term::duration(waited), "other apps under 5% for 15 s");
        say("GPU quiet after " + term::duration(waited));
      } else {
        std::string d;
        for (const auto& x : busy) {
          char pc[32];
          snprintf(pc, sizeof pc, " %.0f%%", x.pct);
          d += (d.empty() ? "" : ", ") + x.name + pc;
        }
        phase_line("GPU busy", term::duration(waited), "measuring anyway, flagged where it skews: " + d, false);
        say("GPU still busy after " + term::duration(waited) + " (" + d + "): measuring anyway; contention will be flagged");
      }
    }

    // the phases (bench.py main), with the contention monitor (compare_engines.py) sampling every 2 s
    Monitor mon(own, terminal);
    mon.start();
    auto phase = [&](const std::string& name, const std::function<void()>& fn) {
      const Vm v0 = vm();
      const double t0 = Monitor::now();
      fn();
      const double t1 = Monitor::now();
      raw["phases"][name] = {t0, t1};
      const Footprint f = footprint(own(), true);
      const Vm v1 = vm();
      raw["memory"][name] = {{"phys_footprint", f.phys},
                             {"phys_footprint_peak", f.peak},
                             {"mapped_file_clean", f.mapped_clean},
                             {"swap_used_mb", v1.swap_used_mb},
                             {"memorystatus_level", v1.memorystatus_level},
                             {"swapouts_delta", v0.swapouts >= 0 && v1.swapouts >= 0 ? ojson(v1.swapouts - v0.swapouts) : ojson(nullptr)}};
      if (stopping()) throw Stopped{};
    };
    // the writing runs' temperature, as sampled: 0.6, Flint's 1.0 (set above)
    const std::string writing_t = fmt_temp(s.temperature), writing = "Writing, T = " + writing_t;
    auto decode_phase = [&](const char* name, bool greedy, int runs, const char* label) {
      Ticker t(label);
      ojson all = ojson::array();
      phase(name, [&] {
        for (int i = 0; i < runs && !stopping(); ++i) {
          t.detail("run " + std::to_string(i + 1) + " of " + std::to_string(runs));
          all.push_back(t_decode(s, greedy, nonces));
          const ojson& d = all.back();
          if (d["decode_tps"].is_number()) t.label(std::string(label) + "  " + fmt_tps(d["decode_tps"]));
          if (!g_tui) {
            char v[32] = "NA";
            if (d["decode_tps"].is_number()) snprintf(v, sizeof v, "%.2f", d["decode_tps"].get<double>());
            fprintf(stderr, "  decode T=%s: %s tok/s (%s tok, finish=%s)%s\n", greedy ? "0" : writing_t.c_str(), v,
                    d.value("completion_tokens", ojson()).dump().c_str(), d.value("finish", ojson()).dump().c_str(),
                    d["invalid"].empty() ? "" : (" INVALID " + d["invalid"].dump()).c_str());
          }
        }
      });
      t.done();
      const ojson sm = summary(all);
      raw[name] = all;
      raw[std::string(name) + "_summary"] = sm;
      const bool ok = sm["invalid"].empty() && sm.contains("median_tps");
      char d[96];
      if (sm.contains("median_tps"))
        snprintf(d, sizeof d, "%.1f–%.1f, %d of %d runs valid", sm["min_tps"].get<double>(), sm["max_tps"].get<double>(),
                 sm["valid_runs"].get<int>(), sm["runs"].get<int>());
      else
        snprintf(d, sizeof d, "no valid run");
      phase_line(label, sm.contains("median_tps") ? fmt_tps(sm["median_tps"]) : "–", d, ok);
      char med[32] = "NA";
      if (sm.contains("median_tps")) snprintf(med, sizeof med, "%.2f", sm["median_tps"].get<double>());
      say("decode T=" + (greedy ? std::string("0") : writing_t) + ": median " + med + " tok/s over " +
          std::to_string(sm["valid_runs"].get<int>()) + "/" + std::to_string(sm["runs"].get<int>()) + " valid");
    };
    decode_phase("decode", false, s.decode_runs, writing.c_str());
    decode_phase("decode_greedy", true, s.greedy_runs, "Writing, greedy");
    raw["prefill"] = ojson::array();
    for (int size : s.sizes) {
      const std::string k = std::to_string(size / 1000) + "K";
      Ticker t("Reading a " + k + "-token prompt");
      ojson p;
      phase("prefill_" + std::to_string(size), [&] {
        p = t_prefill(s, size, nonces, [&](const std::string& step) {
          t.label("Follow-ups at " + k);
          t.detail(step);
        }, stopping);
      });
      t.done();
      raw["prefill"].push_back(p);
      const bool ok = p["invalid"].empty();
      char d[128];
      if (p["prefill_tps"].is_number())
        snprintf(d, sizeof d, "%s tokens in %.2f s", thousands(p["prompt_tokens"].get<long>()).c_str(), p["ttft_s"].get<double>());
      else
        snprintf(d, sizeof d, "%s", p["invalid"].empty() ? "no answer" : p["invalid"][0].get<std::string>().c_str());
      phase_line("Reading " + k, p["prefill_tps"].is_number() ? fmt_tps(p["prefill_tps"], true) : "–", d, ok);
      const ojson& c = p.contains("cached") ? p["cached"] : ojson::object();
      if (c.contains("median_ttft_s")) {
        // time to first token, and the rate the new part of the prompt (not the cached prefix) was read at
        std::string d = "median of " + std::to_string(c["samples"].size()) + " turns";
        if (c["median_suffix_prefill_tps"].is_number())
          d += "; the new tokens read at " + fmt_tps(c["median_suffix_prefill_tps"].get<double>(), true) + " (cache hit, uncached part only)";
        phase_line("Follow-up at " + k, c["median_ttft_s"].is_number() ? fmt_secs(c["median_ttft_s"]) : "–", d, c["invalid"].empty());
      }
      say("prefill " + std::to_string(size) + ": " + (p["prompt_tokens"].is_number() ? std::to_string(p["prompt_tokens"].get<int>()) : "NA") +
          " tok, " + (p["prefill_tps"].is_number() ? std::to_string(std::lround(p["prefill_tps"].get<double>())) : "NA") +
          " tok/s; cached TTFT median " + (c.contains("median_ttft_s") && c["median_ttft_s"].is_number() ? fmt_secs(c["median_ttft_s"]) : "NA") +
          (c.contains("median_suffix_prefill_tps") && c["median_suffix_prefill_tps"].is_number()
               ? " (" + std::to_string(std::lround(c["median_suffix_prefill_tps"].get<double>())) + " tok/s on the uncached suffix)"
               : std::string()) +
          (p["invalid"].empty() ? "" : " INVALID " + p["invalid"].dump()));
    }
    {
      Ticker t("Tool calls");
      phase("tools", [&] {
        t.detail("thinking off");
        ojson a = ojson::array();
        a.push_back(t_tools(s, false));
        t.detail("thinking on");
        a.push_back(t_tools(s, true));
        raw["tools"] = a;
        raw["tools_pass"] = tools_pass(a);
      });
      t.done();
      phase_line("Tool calls", raw["tools_pass"].get<bool>() ? "pass" : "fail",
                 "thinking off and on: streamed, whole, round trip", raw["tools_pass"].get<bool>());
      say(std::string("tools: ") + (raw["tools_pass"].get<bool>() ? "pass" : "FAIL"));
    }
    mon.stop();
    in.samples = mon.samples();
  } catch (const Stopped&) {
    term::status("");
    cleanup();
    if (g_tui) ui::warning("Stopped: no result.");
    else fprintf(stderr, "Stopped: no result.\n");
    return 130;
  } catch (const std::exception& e) {
    term::status("");
    cleanup();
    return fail(e.what());
  }
  in.power_end = power();
  in.vm_end = vm();
  cleanup();

  in.model = mi;
  in.settings = s;
  const ojson result = make_result(in);
  const std::string content = pretty(result);
  // save it
  out_path = o.out;
  if (out_path.empty()) {
    mkdir(bench_dir().c_str(), 0755);
    std::string name = result_file_name(result), base = name.substr(0, name.size() - 5);
    out_path = bench_dir() + "/" + name;
    struct stat st;
    for (int n = 2; stat(out_path.c_str(), &st) == 0; ++n) out_path = bench_dir() + "/" + base + "-" + std::to_string(n) + ".json";
  }
  {
    std::ofstream f(out_path, std::ios::binary);
    f << content;
    if (!f) return fail("Can't write " + out_path + ".");
  }
  if (o.json) {
    fputs(content.c_str(), stdout);
    fflush(stdout);
  }
  if (g_tui) {
    print_summary(result, out_path);
  } else {
    if (!o.json) {
      printf("%s\n%s", result_title(result).c_str(), result_markdown(result).c_str());
      fflush(stdout);
    }
    for (const auto& f : result["flags"]) fprintf(stderr, "flag: %s\n", f.get<std::string>().c_str());
    for (const auto& f : result["invalid"]) fprintf(stderr, "INVALID: %s\n", f.get<std::string>().c_str());
    fprintf(stderr, "Saved %s\n", out_path.c_str());
  }
  code = result["invalid"].empty() ? 0 : 1;
  // submitting: asked for, or offered in a terminal
  if (o.submit) {
    const int rc = submit(result, content, out_path, o.dry);
    return rc ? rc : code;
  }
  if (g_tui && submit_problem(result).empty() && can_confirm()) {
    term::line("", stderr);
    return submit(result, content, out_path, false, true) ? 1 : code;
  }
  return code;
}

// `kapitsa bench compare`: the engine comparison is bench/harness/kbench.py (the harness's own scripts, one engine at
// a time under the GPU lock); this finds the harness and python3 and hands over. Nothing is fetched or installed.
std::string find_harness() {
  std::vector<std::string> dirs;
  // a harness named by the caller is the one or none: a wrong path is an error, not a quiet fall back to another copy
  if (const char* e = getenv("KAPITSA_HARNESS"); e && *e) dirs = {e};
  else {
    const std::string exe = self_path();
    const std::string bin = exe.substr(0, exe.rfind('/'));
    // beside the binary (an unpacked release), where a checkout's build/ or an install keeps it
    for (const char* rel : {"/bench/harness", "/../bench/harness", "/../../bench/harness",
                            "/../share/kapitsa/bench/harness"})
      dirs.push_back(bin + rel);
    if (char cwd[PATH_MAX]; getcwd(cwd, sizeof cwd)) dirs.push_back(std::string(cwd) + "/bench/harness");
  }
  for (const std::string& d : dirs) {
    struct stat st;
    char real[PATH_MAX];
    if (stat((d + "/kbench.py").c_str(), &st) == 0 && realpath(d.c_str(), real)) return real;
  }
  return "";
}

int run_compare(int argc, char** argv, int first) {
  const std::string dir = find_harness();
  if (dir.empty()) {
    fprintf(stderr,
            "kapitsa bench compare: the benchmark harness (bench/harness) was not found beside this kapitsa; reinstall "
            "it with install.sh or Homebrew (a release ships the harness), or set KAPITSA_HARNESS to another copy.\n");
    return 2;
  }
  const char* py = getenv("KAPITSA_PYTHON");
  std::vector<std::string> args = {py && *py ? py : "/usr/bin/python3", dir + "/kbench.py"};
  struct stat st;
  if (!(py && *py) && stat(args[0].c_str(), &st) != 0) args[0] = "python3";
  for (int i = first; i < argc; ++i) args.push_back(argv[i]);
  std::vector<char*> av;
  for (auto& a : args) av.push_back(a.data());
  av.push_back(nullptr);
  // the engines are measured against this binary, and no update may replace it mid-run
  setenv("KAPITSA_BIN", self_path().c_str(), 1);
  setenv("KAPITSA_NO_AUTO_UPDATE", "1", 1);
  if (!update::hold_for_benchmark())
    return fail("kapitsa was updated as this benchmark started. Run kapitsa bench compare again.");
  // a child, not exec: the hold on this binary (a lock on a descriptor that exec would close) lasts the whole run
  static volatile pid_t child = 0;
  const pid_t pid = fork();
  if (pid < 0) return fail(std::string("can't start python: ") + strerror(errno));
  if (pid == 0) {
    execvp(av[0], av.data());
    fprintf(stderr, "kapitsa bench compare: can't start %s (%s). It needs Python 3 (macOS: xcode-select --install).\n",
            av[0], strerror(errno));
    _exit(127);
  }
  child = pid;
  // Ctrl-C reaches python too (same process group); a SIGTERM to us is passed on, and python stops its server
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, [](int) { if (child > 0) kill(child, SIGTERM); });
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

}  // namespace

int run_bench(int argc, char** argv, int first) {
  Options o;
  const char* help = "kapitsa bench --help";
  if (first < argc && std::string(argv[first]) == "compare") return run_compare(argc, argv, first + 1);
  if (first < argc && std::string(argv[first]) == "submit")
    return submit_saved(std::vector<std::string>(argv + first + 1, argv + argc));
  for (int i = first; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage_error(a + " needs a value", help);
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      char b[4096];
      snprintf(b, sizeof b, kHelp, hub::tilde(bench_dir()).c_str());
      if (term::fancy()) ui::help(b);
      else fputs(b, stdout);
      return 0;
    } else if (a == "--model" || a == "-m") o.model = next();
    else if (a == "--quick") o.quick = true;
    else if (a == "--json") o.json = true;
    else if (a == "--submit") o.submit = true;
    else if (a == "--dry-run") o.dry = true;
    else if (a == "--out") o.out = expand_home(next());
    else if (a == "--yes") o.yes = true;
    else usage_error("unknown option " + a, help);
  }
  if (o.dry && !o.submit) usage_error("--dry-run goes with --submit", help);
  g_tui = term::fancy() && !o.json;
  g_plain = !g_tui;
  // the binary stays the same for the whole run (its servers start from it): no update replaces it meanwhile
  if (!update::hold_for_benchmark())
    return fail("kapitsa was updated as this benchmark started, so its servers would run another version. Run kapitsa "
                "bench again.");
  return run(o);
}

}  // namespace ni
