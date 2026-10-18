// The command line's screens in a terminal (ui.h).
#include "server/ui.h"

#include "api/clients.h"
#include "metal/gpu.h"
#include "server/cli.h"
#include "server/pendulum.h"
#include "util/log.h"
#include "util/term.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <mutex>

namespace ni::ui {

using namespace term;

namespace {

const char* const kRepoUrl = "github.com/nace-ai/kapitsa";

// The layout's width: the terminal's, up to 100 columns (wider lines read worse)
int layout_width() { return std::min(columns(), 100); }

void print(const std::vector<std::string>& lines, FILE* f = stdout) {
  std::string s;
  for (const auto& l : lines) s += l + "\n";
  fputs(s.c_str(), f);
  fflush(f);
}

std::string section(const std::string& title) { return "  " + bold(title); }

// Rows of "key  description": the keys in a column (styled by `key`), the descriptions wrapped beside them (styled per
// line by `desc`), or under them when the column leaves too little room.
using Style = std::string (*)(const std::string&);
std::string plain(const std::string& s) { return s; }

void two_columns(std::vector<std::string>& out, const std::vector<std::pair<std::string, std::string>>& rows, int indent,
                 int w, Style key = bold, Style desc = plain, int min_key = 0) {
  int kw = min_key;
  for (const auto& [k, d] : rows) kw = std::max(kw, width(k));
  const int dw = w - indent - kw - 3;
  const std::string pad(size_t(indent), ' ');
  for (const auto& [k, d] : rows) {
    if (d.empty()) {
      out.push_back(pad + key(k));
      continue;
    }
    if (dw < 20) {  // too narrow: the description under its key
      out.push_back(pad + key(k));
      for (const auto& l : wrap(d, w - indent - 2)) out.push_back(pad + "  " + desc(l));
      continue;
    }
    const auto lines = wrap(d, dw);
    for (size_t i = 0; i < lines.size(); ++i)
      out.push_back(pad + (i ? std::string(size_t(kw), ' ') : left(key(k), kw)) + "   " + desc(lines[i]));
  }
}

// A paragraph wrapped to the width at an indent, each line styled
void paragraph(std::vector<std::string>& out, const std::string& text, int indent, int w, Style style = plain) {
  for (const auto& l : wrap(text, w - indent)) out.push_back(std::string(size_t(indent), ' ') + style(l));
}

std::string gb(double g) {
  char b[32];
  snprintf(b, sizeof b, "%.1f GB", g);
  return b;
}

// The model `kapitsa serve` starts on this Mac (best_model), from what is on this Mac alone: nullptr when finding it
// would need the network (a model's plan not fetched yet) or when none fits, so that the screen shows no ★ rather
// than a guess.
const hub::CatalogModel* best_here() {
  const EngineOptions opt;  // serve's defaults
  const size_t budget = default_mem_budget();
  hub::set_offline(true);
  hub::ModelRef ref;
  const bool found = best_model(opt, budget, &ref, nullptr);
  bool fits = false;
  if (found && hub::offline_misses() == 0 && ref.kind == hub::ModelRef::kCatalog) {
    fits = variant_fits(ref, opt, budget) == 1;
    if (!fits) {
      try {
        const std::string d = hub::plan_dir(ref);
        fits = !d.empty() && conversation_fits(d, opt.context, opt.slots, budget);
      } catch (const std::exception&) {
      }
    }
  }
  const bool exact = hub::offline_misses() == 0;
  hub::set_offline(false);
  return fits && exact ? ref.model : nullptr;
}

// Downloaded: "" when the variant `--model <name>` takes by default is, else the name of another downloaded variant,
// else nullopt-ish ("-")
std::string downloaded(const hub::CatalogModel& m) {
  try {
    if (hub::is_current(hub::resolve(m.name))) return "";
    for (const auto& v : m.variants)
      if (v.name != m.default_variant && hub::is_current(hub::resolve(m.name + ":" + v.name))) return v.name;
  } catch (const hub::Error&) {
  }
  return "-";
}

// The command that installs a coding agent: the first command of its install hint
std::string install_command(const std::string& agent) {
  const std::string h = api::agent_install_hint(agent);
  const size_t nl = h.find('\n');
  if (nl == std::string::npos) return "";
  std::string c = h.substr(nl + 1);
  while (!c.empty() && c[0] == ' ') c.erase(c.begin());
  if (const size_t p = c.find("  "); p != std::string::npos) c.resize(p);
  return c;
}

}  // namespace

// ---------------------------------------------------------------- the home screen

// "Apple M4 Pro · 20-core GPU · 48 GB"
std::string this_mac() {
  Gpu& g = Gpu::get();
  char b[160];
  snprintf(b, sizeof b, "%s  %s  %d-core GPU  %s  %.0f GB", g.name().c_str(), kDot, g.core_count(), kDot, ram_gb());
  return b;
}

void home(const std::string& version) {
  const int W = layout_width();
  std::vector<std::string> o;
  // the header: Kapitsa's pendulum beside the wordmark (its intro first, where it may play)
  const std::vector<std::string> head = {badge(" kapitsa ") + "  " + dim(version), "fast local AI on Apple silicon",
                                         dim(this_mac())};
  const hub::CatalogModel* best = best_here();
  fputs("\n", stdout);
  fflush(stdout);
  if (W >= 60) {
    intro(head, 3);
  } else {
    o.push_back("  " + head[0] + "   " + head[1]);
    o.push_back("  " + head[2]);
  }
  o.push_back("");

  o.push_back(section("Get started"));
  two_columns(o, {{"kapitsa serve", "start the best model for this Mac" + (best ? " (" + hub::short_name(*best) + ")" : std::string())},
                  {"kapitsa serve --model 27b", "start another one (downloaded the first time)"},
                  {"kapitsa claude", "Claude Code on it, in one command"}},
              4, W);
  paragraph(o, "Chat at http://127.0.0.1:8000  " + std::string(kDot) + "  OpenAI and Anthropic APIs at /v1", 4, W, dim);
  o.push_back("");

  // models: alias, name, size, marks, purpose
  struct Row {
    std::string alias, name, size, marks, purpose;
  };
  std::vector<Row> rows;
  int aw = 0, nw = 0, sw = 0, mw = 0;
  bool any_mark = false, any_star = false;
  for (const auto& m : hub::catalog()) {
    Row r;
    r.alias = hub::short_name(m);
    r.name = m.name;
    const hub::Variant* d = m.variant();
    r.size = d ? gb(d->download_gb) : "";
    const std::string have = downloaded(m);
    std::string marks;
    if (have != "-") marks = accent(kOk) + (have.empty() ? "" : " " + dim(have)), any_mark = true;
    if (&m == best) marks += (marks.empty() ? "  " : " ") + accent(kStar), any_star = true;
    r.marks = marks;
    r.purpose = m.brief.empty() ? m.about : m.brief;
    aw = std::max(aw, width(r.alias));
    nw = std::max(nw, width(r.name));
    sw = std::max(sw, width(r.size));
    mw = std::max(mw, width(r.marks));
    rows.push_back(r);
  }
  const int indent = 4;
  const int fixed = indent + aw + 3 + nw + 3 + sw + (mw ? 2 + mw : 0) + 3;
  const int pw = W - fixed;
  const bool show_purpose = pw >= 16;
  std::vector<std::string> table;
  int tw = 0;
  for (const auto& r : rows) {
    std::string l = std::string(size_t(indent), ' ') + left(bold(r.alias), aw) + "   " + left(r.name, nw) + "   " +
                    right(dim(r.size), sw);
    if (mw) l += "  " + left(r.marks, mw);
    if (show_purpose) l += "   " + clip(r.purpose, pw);
    tw = std::max(tw, width(l));
    table.push_back(l);
  }
  std::string legend;
  if (any_mark) legend += accent(kOk) + dim(" downloaded");
  if (any_star) legend += (legend.empty() ? "" : "   ") + accent(kStar) + dim(" best for this Mac");
  const std::string title = section("Models");
  const int edge = std::max(tw, width(title) + 3 + width(legend));  // the legend ends where the table does
  o.push_back(legend.empty() || edge > W ? title : title + std::string(size_t(edge - width(title) - width(legend)), ' ') + legend);
  for (auto& l : table) o.push_back(l);
  paragraph(o, "4-bit by default; other precisions in kapitsa list, e.g. --model 35b:8bit", 4, W, dim);
  paragraph(o, "or any Qwen3.5-family MLX repo (org/name), a model folder, or a .gguf file", 4, W, dim);
  o.push_back("");

  // coding agents
  {
    o.push_back(section("Coding agents"));
    struct A {
      const char* cmd;
      const char* name;
    };
    const A agents[] = {{"claude", "Claude Code"}, {"codex", "Codex"}, {"opencode", "opencode"}};
    int cw = 0, nw2 = 0;
    for (const A& a : agents) cw = std::max(cw, width(a.cmd) + 8), nw2 = std::max(nw2, width(a.name));
    for (const A& a : agents) {
      const bool have = !find_in_path(a.cmd).empty();
      std::string l = "    " + left(bold(std::string("kapitsa ") + a.cmd), cw) + "   " + left(a.name, nw2) + "   ";
      if (have) {
        l += accent(kOn) + " installed";
        o.push_back(l);
        continue;
      }
      l += dim(std::string(kOff) + " not installed");
      const std::string c = install_command(a.cmd);
      if (!c.empty() && width(l) + 3 + width(c) <= W) {
        o.push_back(l + "   " + dim(c));
      } else {
        o.push_back(l);
        if (!c.empty()) o.push_back(std::string(size_t(4 + cw + 3), ' ') + dim(clip(c, W - 4 - cw - 3)));
      }
    }
  }
  o.push_back("");

  o.push_back(section("Commands"));
  two_columns(o, {{"serve", "start the server: chat page at /, OpenAI and Anthropic APIs at /v1"},
                  {"pull", "download a model without starting it"},
                  {"list", "downloaded models, and every precision"},
                  {"rm", "delete a downloaded model"},
                  {"import", "convert a GGUF (LM Studio, Ollama, llama.cpp) into a model"},
                  {"doctor", "this Mac's GPU and memory, and which models fit"}},
              4, W);
  paragraph(o, "kapitsa <command> --help shows its options.", 4, W, dim);
  o.push_back("");
  paragraph(o, "Models in " + hub::tilde(hub::models_dir()) + " (KAPITSA_HOME)  " + kDot + "  " + kRepoUrl, 2, W, dim);
  o.push_back("");
  print(o);
}

// ---------------------------------------------------------------- styled --help

void help(const std::string& text) {
  const int W = layout_width();
  std::vector<std::string> lines;
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find('\n', a);
    if (b == std::string::npos) b = text.size();
    lines.push_back(text.substr(a, b - a));
    a = b + 1;
  }
  while (!lines.empty() && lines.back().empty()) lines.pop_back();
  std::vector<std::string> o;
  o.push_back("");
  // the usage: "kapitsa <command> ..." up to its first ": " or the end of its paragraph
  size_t i = 0;
  {
    std::string usage;
    for (; i < lines.size() && !lines[i].empty() && lines[i][0] != ' '; ++i) usage += (usage.empty() ? "" : " ") + lines[i];
    std::string rest;
    if (const size_t c = usage.find(": "); c != std::string::npos) rest = usage.substr(c + 2), usage.resize(c);
    // the command words in bold, its options as they are
    size_t cmd_end = 0;
    for (size_t k = 0; k < usage.size();) {
      size_t e = usage.find(' ', k);
      if (e == std::string::npos) e = usage.size();
      const std::string word = usage.substr(k, e - k);
      if (word.empty() || word[0] == '[' || word[0] == '-' || isupper(uint8_t(word[0]))) break;
      cmd_end = e;
      k = e + 1;
    }
    const auto wrapped = wrap(usage, W - 4);
    for (size_t k = 0; k < wrapped.size(); ++k) {
      std::string l = wrapped[k];
      if (k == 0 && cmd_end > 0 && cmd_end <= l.size()) l = bold(l.substr(0, cmd_end)) + l.substr(cmd_end);
      o.push_back((k ? "    " : "  ") + l);
    }
    if (!rest.empty()) {
      o.push_back("");
      paragraph(o, rest, 2, W);
    }
  }
  // then: option rows ("  --flag ARG   what it does", continued by deeper indented lines), examples (other indented
  // lines) and paragraphs
  std::vector<std::pair<std::string, std::string>> opts;
  auto flush_opts = [&] {
    if (opts.empty()) return;
    two_columns(o, opts, 4, W);
    opts.clear();
  };
  std::string para;
  auto flush_para = [&] {
    if (para.empty()) return;
    paragraph(o, para, 2, W);
    para.clear();
  };
  for (; i < lines.size(); ++i) {
    const std::string& l = lines[i];
    if (l.empty()) {
      flush_opts();
      flush_para();
      o.push_back("");
      continue;
    }
    size_t ind = 0;
    while (ind < l.size() && l[ind] == ' ') ++ind;
    if (ind >= 6 && !opts.empty()) {  // an option's description, continued
      opts.back().second += " " + l.substr(ind);
      continue;
    }
    if (ind >= 2 && ind < 6 && l[ind] == '-' && l.size() > ind + 1 && l[ind + 1] != ' ') {
      flush_para();
      const size_t gap = l.find("  ", ind);
      if (gap == std::string::npos) {
        opts.push_back({l.substr(ind), ""});
      } else {
        size_t d = gap;
        while (d < l.size() && l[d] == ' ') ++d;
        opts.push_back({l.substr(ind, gap - ind), l.substr(d)});
      }
      continue;
    }
    if (ind >= 2) {  // an example: kept as is, the command in bold
      flush_opts();
      flush_para();
      o.push_back("    " + bold(l.substr(ind)));
      continue;
    }
    flush_opts();
    para += (para.empty() ? "" : " ") + l;
  }
  flush_opts();
  flush_para();
  o.push_back("");
  print(o);
}

// ---------------------------------------------------------------- lines

void quiet() {
  log_sink() = [](int level, const char* msg) {
    if (level == 0) term::error(msg);
  };
}


void ok(const std::string& s, bool out) { line("  " + accent(kOk) + " " + s, out ? stdout : stderr); }
void note(const std::string& s, bool out) { line("  " + dim(std::string(kArrow) + " " + s), out ? stdout : stderr); }
void warning(const std::string& s) { line("  " + warn(kWarn) + " " + s, stderr); }

int fail(const std::string& msg, int code) {
  term::error(msg);
  return code;
}

// ---------------------------------------------------------------- list, plan

void list() {}
void plan(const hub::ModelRef&) {}

// ---------------------------------------------------------------- serve

namespace {

// The engine's log while `kapitsa serve` runs in a terminal: a file, the lines that matter on screen
struct ServeLog {
  std::mutex mu;
  FILE* f = nullptr;
  std::string path;
  std::vector<std::string> pending;  // lines before the file opens (the port is checked first)
  bool verbose = false;
  Spinner* spinner = nullptr;
  std::string ready_line, images_line;
  std::string detail;  // the latest line
};
ServeLog g_log;

// Log lines worth seeing without the log: a drafter left out, images or tools a request carried that the model can't
// use, the Neural Engine unavailable, a conversation's cache given up for memory
bool worth_showing(const std::string& m) {
  if (m.rfind("DFlash2: no drafter found", 0) == 0) return false;
  std::string l = m;
  for (auto& ch : l) ch = char(tolower(uint8_t(ch)));
  for (const char* k : {"left out", "can't", "cannot", "failed", "unavailable", "not used", "insufficient",
                        "dropping the kv cache", "error", "corrupt", "not supported", "warning"})
    if (l.find(k) != std::string::npos) return true;
  return false;
}

void serve_sink(int level, const char* msg) {
  const std::string m = msg;
  const std::string text = std::string("[kapitsa] ") + (level == 0 ? "fatal: " : "") + m;
  {
    std::lock_guard<std::mutex> lk(g_log.mu);
    if (g_log.f) {
      fputs((text + "\n").c_str(), g_log.f);
      fflush(g_log.f);
    } else {
      g_log.pending.push_back(text);
    }
    if (m.rfind("engine ready: ", 0) == 0) g_log.ready_line = m;
    else if (m.rfind("images: ", 0) == 0) g_log.images_line = m;
    if (g_log.spinner && level > 0) g_log.spinner->detail(m);
    if (level > 0) g_log.detail = m;
  }
  if (level == 0) {
    end_status();  // the process exits next: no spinner left behind
    term::error(m, g_log.path.empty() ? "" : "The log: " + hub::tilde(g_log.path));
  } else if (g_log.verbose) {
    line(dim(text), stderr);
  } else if (worth_showing(m)) {
    warning(dim(m));
  }
}

std::string after(const std::string& s, const std::string& key) {
  const size_t p = s.find(key);
  return p == std::string::npos ? "" : s.substr(p + key.size());
}

// "DFlash2 speculative decoding (verify 7 of 7), images on" -> "DFlash2 drafter"
std::string decoding(const std::string& ready) {
  std::string mode = after(ready, " slots, ");
  if (const size_t p = mode.find(", images on"); p != std::string::npos) mode.resize(p);
  if (mode.rfind("DFlash2", 0) == 0) return "DFlash2 drafter " + dim("(speculative decoding)");
  if (mode.rfind("LiLiCorr", 0) == 0) return "LiLiCorr drafter " + dim("(speculative decoding)");
  if (mode.rfind("MTP", 0) == 0) return "MTP head " + dim("(speculative decoding, " + after(mode, "depth ") + " ahead)");
  if (mode.rfind("plain", 0) == 0) return "plain " + dim("(no speculative decoding)");
  return mode;
}

std::string repeat(const char* g, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) s += g;
  return s;
}

// A card: a rounded box of `rows` (styled), `inner` columns wide
void card(std::vector<std::string>& o, const std::vector<std::string>& rows, int inner) {
  o.push_back("  " + dim("╭" + repeat("─", inner + 4) + "╮"));
  for (const auto& r : rows) o.push_back("  " + dim("│") + "  " + left(clip(r, inner), inner) + "  " + dim("│"));
  o.push_back("  " + dim("╰" + repeat("─", inner + 4) + "╯"));
}

std::atomic<bool> g_serve_ui{false};
std::mutex g_req_mu;
bool g_req_header = false;

}  // namespace

void serve_begin(const std::string& version, bool verbose) {
  g_serve_ui = true;
  // NI_LOG asks for the log: on screen too
  g_log.verbose = verbose || getenv("NI_LOG") != nullptr;
  log_sink() = serve_sink;
  print({"", "  " + badge(" kapitsa ") + "  " + dim(version + "  " + kDot + "  serve")});
}

std::string serve_log_file(int port) {
  const std::string dir = hub::home_dir() + "/logs";
  mkdir(hub::home_dir().c_str(), 0755);
  mkdir(dir.c_str(), 0755);
  const std::string path = dir + "/serve-" + std::to_string(port) + ".log";
  std::lock_guard<std::mutex> lk(g_log.mu);
  g_log.f = fopen(path.c_str(), "w");
  if (!g_log.f) return "";
  g_log.path = path;
  const time_t now = time(nullptr);
  char at[64];
  strftime(at, sizeof at, "%Y-%m-%d %H:%M:%S", localtime(&now));
  fprintf(g_log.f, "kapitsa serve on port %d, started %s\n", port, at);
  for (const auto& l : g_log.pending) fprintf(g_log.f, "%s\n", l.c_str());
  g_log.pending.clear();
  fflush(g_log.f);
  return path;
}

namespace {

// While the model loads: Kapitsa's pendulum wobbling beside what is happening (where it may play), else a spinner
struct Loading {
  std::string what;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::unique_ptr<Wobble> wobble;
  double secs() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
std::unique_ptr<Loading> g_loading;

}  // namespace

void loading_begin(const std::string& what) {
  std::lock_guard<std::mutex> lk(g_log.mu);
  g_loading = std::make_unique<Loading>();
  g_loading->what = what;
  if (!can_animate()) {
    g_log.spinner = new Spinner("Loading " + bold(what));
    return;
  }
  line("");
  Loading* l = g_loading.get();
  l->wobble = std::make_unique<Wobble>(
      [l] {
        std::string detail;
        {
          std::lock_guard<std::mutex> lk2(g_log.mu);
          detail = g_log.detail;
        }
        // the wobble calms as the load goes on (a typical load: 5-10 s)
        l->wobble->progress(1 - std::exp(-l->secs() / 4));
        return std::vector<std::string>{"Loading " + bold(l->what) + "  " + dim(duration(l->secs())),
                                        dim(clip(detail, std::max(20, columns() - kArtCols - 8)))};
      },
      3);
}

void loading_end(const std::string& what) {
  Spinner* sp;
  std::unique_ptr<Loading> l;
  {
    std::lock_guard<std::mutex> lk(g_log.mu);
    sp = g_log.spinner;
    g_log.spinner = nullptr;
    l = std::move(g_loading);
  }
  if (sp) {
    sp->done("  " + accent(kOk) + " Loaded " + bold(what) + dim(" in " + duration(sp->seconds())));
    delete sp;
  }
  if (l && l->wobble) l->wobble->done({accent(kOk) + " Loaded " + bold(what) + dim(" in " + duration(l->secs())), ""});
}

void ready(const Ready& r) {
  const int W = layout_width();
  const int inner = std::max(36, std::min(W - 8, 68));
  std::vector<std::string> rows;
  auto row = [&](const std::string& k, const std::string& v) { rows.push_back(left(dim(k), 11) + v); };
  {
    const std::string status = accent(kOn) + " ready";
    const std::string name = bold(r.model);
    const int gap = inner - width(name) - width(status);
    rows.push_back(gap >= 2 ? name + std::string(size_t(gap), ' ') + status : name);
  }
  rows.push_back("");
  if (!r.quant.empty()) row("Precision", r.quant);
  std::string ready_line, images_line;
  {
    std::lock_guard<std::mutex> lk(g_log.mu);
    ready_line = g_log.ready_line, images_line = g_log.images_line;
  }
  if (!ready_line.empty()) row("Decoding", decoding(ready_line));
  if (!images_line.empty()) {
    std::string v = after(images_line, "images: ");
    std::string gbs;
    if (const size_t p = v.rfind(", "); p != std::string::npos) gbs = v.substr(p + 2), v.resize(p);
    if (const size_t p = v.find(" ("); p != std::string::npos) v.resize(p);
    if (const size_t p = gbs.find(" GB"); p != std::string::npos) gbs.resize(p + 3);
    row("Images", "on " + dim("(" + v + (gbs.empty() ? "" : ", +" + gbs + " with the first image") + ")"));
  }
  if (r.budget) {
    char b[64];
    snprintf(b, sizeof b, "%.1f of %.1f GB", r.used / 1e9, r.budget / 1e9);
    const int bw = std::min(24, inner - 11 - width(b) - 2);
    row("Memory", (bw >= 8 ? bar(double(r.used) / double(r.budget), bw) + "  " : std::string()) + b);
  }
  if (r.context) {
    std::string c = count(r.context) + " tokens";
    if (r.slots > 1) c += " " + dim("×") + " " + std::to_string(r.slots) + " conversations";
    if (r.context_fit > 0 && r.context_fit < r.context) c += dim(" (" + count(r.context_fit) + " fit in memory)");
    row("Context", c);
  }
  rows.push_back("");
  row("Chat", accent(r.base));
  row("OpenAI", accent(r.base + "/v1"));
  row("Anthropic", accent(r.base));
  if (!r.api_key_note.empty()) row("API key", r.api_key_note);
  std::vector<std::string> o;
  o.push_back("");
  card(o, rows, inner);
  o.push_back("  " + dim(std::string(kArrow) + " ") + bold("kapitsa claude") + dim(" in another terminal codes with it"));
  o.push_back("  " + dim(std::string(kArrow) + " " + (r.log.empty() ? "" : "Log " + hub::tilde(r.log) + "  " + kDot + "  ") +
                         "Ctrl+C stops"));
  o.push_back("");
  std::lock_guard<std::mutex> lk(g_req_mu);
  for (const auto& l : o) line(l);
}

// ---------------------------------------------------------------- the request log

struct Meter {
  double t0 = now_ms();
  std::atomic<double> first{0};
  std::string client;
  int images = 0;
};

namespace {

// The client by its User-Agent, when it is plain which: Claude Code, Codex, opencode, a browser (the chat page), an SDK
std::string client_of(const std::string& ua) {
  std::string l = ua;
  for (auto& ch : l) ch = char(tolower(uint8_t(ch)));
  auto has = [&](const char* k) { return l.find(k) != std::string::npos; };
  if (has("claude-cli") || has("claude-code")) return "claude";
  if (has("codex")) return "codex";
  if (has("opencode")) return "opencode";
  if (l.rfind("mozilla/", 0) == 0) return "browser";
  if (has("anthropic/python")) return "anthropic-py";
  if (has("anthropic/js")) return "anthropic-js";
  if (has("openai/python")) return "openai-py";
  if (has("openai/js")) return "openai-js";
  if (l.rfind("curl/", 0) == 0) return "curl";
  if (has("python-urllib") || has("python-requests") || has("aiohttp") || has("httpx")) return "python";
  if (l.empty()) return "";
  std::string w = l.substr(0, l.find_first_of("/ ("));
  return w.size() > 12 ? w.substr(0, 12) : w;
}

int images_in(const ojson& messages) {
  int n = 0;
  if (!messages.is_array()) return 0;
  for (const auto& m : messages) {
    if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
    for (const auto& p : m["content"])
      if (p.is_object()) {
        const std::string t = p.value("type", "");
        if (t == "image_url" || t == "image" || t == "input_image") ++n;
      }
  }
  return n;
}

}  // namespace

std::shared_ptr<Meter> meter(const httplib::Request& req, const ChatRequest& r) {
  if (!g_serve_ui) return nullptr;
  auto m = std::make_shared<Meter>();
  m->client = client_of(req.get_header_value("User-Agent"));
  m->images = images_in(r.messages);
  return m;
}

void token(const std::shared_ptr<Meter>& m) {
  if (m && m->first.load() == 0) m->first = now_ms();
}

void done(const std::shared_ptr<Meter>& m, const ChatResult& r) {
  if (!m) return;
  const time_t now = time(nullptr);
  char at[16];
  strftime(at, sizeof at, "%H:%M:%S", localtime(&now));
  const double first = m->first.load();
  char b[32];
  std::string ttft = "–", tps = "–", cached = "–";
  if (first > 0) snprintf(b, sizeof b, "%.2fs", (first - m->t0) / 1000), ttft = b;
  if (r.completion_tokens > 1 && r.decode_ms > 0)
    snprintf(b, sizeof b, "%.1f", (r.completion_tokens - 1) / (r.decode_ms / 1000)), tps = b;
  if (r.prompt_tokens > 0) snprintf(b, sizeof b, "%.0f%%", 100.0 * r.cached_tokens / r.prompt_tokens), cached = b;
  std::string fin = r.finish_reason;
  if (fin == "tool_calls") fin = accent("tool calls");
  else if (fin == "length") fin = warn(r.context_full ? "context full" : "length");
  else if (fin == "cancelled") fin = warn("cancelled");
  else if (fin == "error") fin = red("error") + (r.error.empty() ? "" : dim(" " + r.error));
  if (m->images > 0) fin += dim("  " + std::to_string(m->images) + (m->images == 1 ? " image" : " images"));
  const int W = columns();
  const bool with_time = W >= 72;
  std::string l = "  ";
  if (with_time) l += dim(at) + "  ";
  l += left(m->client.empty() ? dim("–") : m->client, 10) + "  " + right(r.prompt_tokens ? count(r.prompt_tokens) : "–", 6) +
       "  " + right(cached, 5) + "  " + right(ttft, 6) + "  " + right(count(r.completion_tokens), 5) + "  " + right(tps, 6) +
       "  " + fin;
  std::lock_guard<std::mutex> lk(g_req_mu);
  if (!g_req_header) {
    g_req_header = true;
    std::string h = "  ";
    if (with_time) h += left("time", 8) + "  ";
    h += left("client", 10) + "  " + right("prompt", 6) + "  " + right("cached", 5) + "  " + right("first", 6) + "  " +
         right("out", 5) + "  " + right("tok/s", 6) + "  finish";
    line(section("Requests"));
    line(dim(h));
  }
  line(clip(l, W - 1));
}

}  // namespace ni::ui
