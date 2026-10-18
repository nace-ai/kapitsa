// The command line's screens in a terminal (ui.h).
#include "server/ui.h"

#include "api/clients.h"
#include "metal/gpu.h"
#include "server/cli.h"
#include "util/log.h"
#include "util/term.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
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

void home(const std::string& version) {
  const int W = layout_width();
  std::vector<std::string> o;
  o.push_back("");
  o.push_back("  " + badge(" kapitsa ") + "  " + dim(version) + "   fast local Qwen on Apple silicon");
  o.push_back("");

  const hub::CatalogModel* best = best_here();
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

std::string serve_log(int, bool) { return ""; }
void serve_header(const std::string&) {}
void loading_begin(const std::string&) {}
void loading_end(bool) {}
void ready(const Ready&) {}

struct Meter {};
std::shared_ptr<Meter> meter(const httplib::Request&, const ojson&) { return nullptr; }
std::shared_ptr<Meter> meter(const httplib::Request&, const ChatRequest&) { return nullptr; }
void token(const std::shared_ptr<Meter>&) {}
void done(const std::shared_ptr<Meter>&, const ChatResult&) {}

}  // namespace ni::ui
