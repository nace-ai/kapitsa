// The command line's screens in a terminal (ui.h).
#include "server/ui.h"

#include "api/clients.h"
#include "metal/gpu.h"
#include "server/cli.h"
#include "server/beam.h"
#include "util/log.h"
#include "util/term.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
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
        fits = !d.empty() && conversation_fits(d, opt.context, fit_slots(opt.slots), budget);
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

// A model's precisions for the line under its row: "4-bit 22.1 GB ✓ · 5-bit 26.4 · ... · bf16 71.9", the default in
// the normal color and the others dim, ✓ where downloaded; without the sizes when they do not fit in `room`
std::string precisions_of(const hub::CatalogModel& m, int room) {
  auto make = [&](bool sizes) {
    std::string s;
    for (const auto& v : m.variants) {
      std::string item = v.label;
      if (item.empty()) {
        item = v.name;
        if (item.size() > 3 && item.compare(item.size() - 3, 3, "bit") == 0) item.insert(item.size() - 3, "-");
        if (sizes) {
          char b[32];
          snprintf(b, sizeof b, s.empty() ? " %.1f GB" : " %.1f", v.download_gb);
          item += b;
        }
      }
      bool have = false;
      try {
        have = hub::is_current(hub::resolve(m.name + ":" + v.name));
      } catch (const hub::Error&) {
      }
      s += (s.empty() ? "" : dim(std::string(" ") + kDot + " ")) + (v.name == m.default_variant ? item : dim(item)) +
           (have ? " " + accent(kOk) : "");
    }
    return s;
  };
  std::string s = make(true);
  if (width(s) > room) s = make(false);
  return clip(s, room);
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
static std::string this_mac() {
  Gpu& g = Gpu::get();
  char b[160];
  snprintf(b, sizeof b, "%s  %s  %d-core GPU  %s  %.0f GB", g.name().c_str(), kDot, g.core_count(), kDot, ram_gb());
  return b;
}

void home(const std::string& version) {
  const int W = layout_width();
  std::vector<std::string> o;
  // the header: the logo (a beam diffracted by a standing wave of light) beside the wordmark, its intro first where it
  // may play
  // the tagline (the README's), shorter where it does not fit beside the logo
  const int room = columns() - (2 + kLogoCols + 3) - 1;
  std::string tagline = "blazing fast Apple Silicon optimized inference engine built for Qwen and Nemotron";
  if (width(tagline) > room) tagline = "blazing fast Apple Silicon engine for Qwen and Nemotron";
  if (width(tagline) > room) tagline = "built for Qwen and Nemotron on Apple Silicon";
  const std::vector<std::string> head = {badge(" kapitsa ") + "  " + dim(version), tagline, dim(this_mac())};
  const hub::CatalogModel* best = best_here();
  fputs("\n", stdout);
  fflush(stdout);
  if (W >= 60) {
    intro(head, 1);
  } else {  // no room for the logo: the wordmark, then the tagline and this Mac under it
    o.push_back("  " + head[0]);
    o.push_back("  " + clip(head[1], W - 3));
    o.push_back("  " + clip(head[2], W - 3));
  }
  o.push_back("");

  o.push_back(section("Get started"));
  two_columns(o, {{"kapitsa serve", "pick a model and start it" + (best ? " (the best here: " + hub::short_name(*best) + ")" : std::string())},
                  {"kapitsa serve --model 27b", "start that one (downloaded the first time)"},
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
  const int under = indent + aw + 3;  // the precisions line starts under the name
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    std::string l = std::string(size_t(indent), ' ') + left(bold(r.alias), aw) + "   " + left(r.name, nw) + "   " +
                    right(dim(r.size), sw);
    if (mw) l += "  " + left(r.marks, mw);
    if (show_purpose) l += "   " + clip(r.purpose, pw);
    tw = std::max(tw, width(l));
    table.push_back(l);
    table.push_back(std::string(size_t(under), ' ') + precisions_of(hub::catalog()[i], W - under - 1));
  }
  std::string legend;
  if (any_mark) legend += accent(kOk) + dim(" downloaded");
  if (any_star) legend += (legend.empty() ? "" : "   ") + accent(kStar) + dim(" best for this Mac");
  const std::string title = section("Models");
  const int edge = std::max(tw, width(title) + 3 + width(legend));  // the legend ends where the table does
  o.push_back(legend.empty() || edge > W ? title : title + std::string(size_t(edge - width(title) - width(legend)), ' ') + legend);
  for (auto& l : table) o.push_back(l);
  {
    const std::string pick = dim("pick one: ") + "kapitsa serve --model 35b:6bit" + dim(std::string("  ") + kDot + "  bf16 = full precision");
    o.push_back("    " + (width(pick) <= W - 5 ? pick : dim("pick one: ") + "--model 35b:6bit"));
  }
  paragraph(o, "or any Qwen3.5-family MLX repo (org/name), a model folder, or a .gguf file", 4, W, dim);
  o.push_back("");

  // coding agents
  {
    o.push_back(section("Coding agents"));
    struct A {
      const char* cmd;
      const char* name;
    };
    const A agents[] = {{"claude", "Claude Code"}, {"codex", "Codex"}, {"opencode", "opencode"}, {"pi", "Pi"}};
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
                  {"doctor", "this Mac's GPU and memory, and which models fit"},
                  {"cluster", "the Macs on this network that run Kapitsa, and what they serve"},
                  {"bench", "benchmark this Mac and add it to the community results"},
                  {"update", "update Kapitsa to its latest release (it also updates itself)"}},
              4, W);
  paragraph(o, "kapitsa <command> --help shows its options.", 4, W, dim);
  o.push_back("");
  paragraph(o, "Models in " + hub::tilde(hub::models_dir()) + " (KAPITSA_HOME)  " + kDot + "  " + kRepoUrl, 2, W, dim);
  o.push_back("");
  print(o);
}

// ---------------------------------------------------------------- styled --help

void help(const std::string& text) { term::help(text); }

// ---------------------------------------------------------------- lines

void quiet() {
  log_sink() = [](int level, const char* msg) {
    if (level == 0) term::error(msg);
  };
}


void ok(const std::string& s, bool out) { line("  " + accent(kOk) + " " + s, out ? stdout : stderr); }
void note(const std::string& s, bool out) {
  // a long note wraps under itself (styled text is cut, not wrapped)
  const int w = std::max(30, std::min(columns(), 100) - 5);
  FILE* f = out ? stdout : stderr;
  if (width(s) <= w || s.find('\033') != std::string::npos) return line("  " + dim(std::string(kArrow) + " " + clip(s, w)), f);
  const auto ls = wrap(s, w);
  for (size_t i = 0; i < ls.size(); ++i) line((i ? "    " : "  " + dim(std::string(kArrow)) + " ") + dim(ls[i]), f);
}
void warning(const std::string& s) {
  // a long warning wraps under itself (styled text is cut, not wrapped)
  const int w = std::max(30, std::min(columns(), 100) - 5);
  if (width(s) <= w || s.find('\033') != std::string::npos) return line("  " + warn(kWarn) + " " + clip(s, w), stderr);
  const auto ls = wrap(s, w);
  for (size_t i = 0; i < ls.size(); ++i) line((i ? "    " : "  " + warn(kWarn) + " ") + ls[i], stderr);
}


// ---------------------------------------------------------------- list, plan

namespace {

// "9546051": a pinned revision, short
std::string short_rev(const std::string& r) { return r.size() > 12 ? r.substr(0, 7) : r.empty() ? "main" : r; }

// Every precision of every catalog model, aligned: the default first, then the others (sizes to download, ✓ where
// downloaded)
void precisions(std::vector<std::string>& o, int W) {
  std::vector<std::string> names;  // the columns after the first: the other precisions, in catalog order
  for (const auto& m : hub::catalog())
    for (const auto& v : m.variants)
      if (v.name != m.default_variant && std::find(names.begin(), names.end(), v.name) == names.end())
        names.push_back(v.name);
  struct Row {
    std::string alias;
    std::vector<std::string> cells;  // [0] the default, then one per `names`
  };
  std::vector<Row> rows;
  std::vector<int> cw(names.size() + 1, 0);
  int aw = 0;
  for (const auto& m : hub::catalog()) {
    Row r;
    r.alias = hub::short_name(m);
    aw = std::max(aw, width(r.alias));
    r.cells.assign(names.size() + 1, "");
    for (const auto& v : m.variants) {
      bool have = false;
      try {
        have = hub::is_current(hub::resolve(m.name + ":" + v.name));
      } catch (const hub::Error&) {
      }
      char b[32];
      snprintf(b, sizeof b, "%.1f", v.download_gb);
      const std::string cell = (v.name == m.default_variant ? bold(v.name) : v.name) + " " + (have ? accent(kOk) : dim(b));
      const size_t k = v.name == m.default_variant ? 0 : size_t(std::find(names.begin(), names.end(), v.name) - names.begin()) + 1;
      r.cells[k] = cell;
      cw[k] = std::max(cw[k], width(cell));
    }
    rows.push_back(r);
  }
  for (const auto& r : rows) {
    std::string l = "    " + left(bold(r.alias), aw) + "   ";
    const int start = width(l);
    for (size_t k = 0; k < r.cells.size(); ++k) {
      const std::string cell = left(r.cells[k], cw[k]) + (k + 1 < r.cells.size() ? "   " : "");
      if (width(l) + cw[k] > W && width(l) > start) {  // too wide: the rest on the next line, under the first cell
        while (!l.empty() && l.back() == ' ') l.pop_back();
        o.push_back(l);
        l = std::string(size_t(start), ' ');
      }
      l += cell;
    }
    while (!l.empty() && l.back() == ' ') l.pop_back();
    o.push_back(l);
  }
}

}  // namespace

void list() {
  const int W = layout_width();
  const auto v = hub::installed();
  std::vector<std::string> o;
  o.push_back("");
  if (v.empty()) {
    o.push_back(section("No models downloaded yet") + "  " + dim(hub::tilde(hub::models_dir())));
  } else {
    o.push_back(section("Downloaded") + "  " + dim(hub::tilde(hub::models_dir())));
    struct Row {
      std::string mark, alias, name, size, what;
    };
    std::vector<Row> rows;
    int aw = 0, nw = 0, sw = 0;
    for (const auto& m : v) {
      Row r;
      r.name = m.name;
      r.size = hub::fmt_bytes(m.bytes);
      std::string what;
      for (const auto& e : m.extras) what += (what.empty() ? "" : std::string(" ") + kDot + " ") + e;
      r.what = what.empty() ? dim("weights") : what;
      r.mark = accent(kOk);
      for (const auto& c : hub::catalog())  // what --model takes for it
        for (const auto& var : c.variants)
          if (m.name == hub::full_name(c, var) || m.name == c.name + "-" + var.name) r.alias = hub::short_name(c, var);
      if (!m.complete) {
        struct stat st {};
        const bool adopt = !r.alias.empty() && stat((m.dir + "/.kapitsa").c_str(), &st) != 0;
        r.mark = warn("◐");
        r.what = adopt ? dim("not checked yet: ") + "kapitsa pull " + r.alias + dim(" checks it")
                       : dim("incomplete: ") + "kapitsa pull " + (r.alias.empty() ? m.name : r.alias) + dim(" continues it");
      }
      aw = std::max(aw, width(r.alias)), nw = std::max(nw, width(r.name)), sw = std::max(sw, width(r.size));
      rows.push_back(r);
    }
    for (const auto& r : rows)
      o.push_back(clip("    " + r.mark + " " + (aw ? left(bold(r.alias), aw) + "   " : "") + left(r.name, nw) + "   " +
                           right(dim(r.size), sw) + "   " + r.what,
                       W));
  }
  o.push_back("");
  const std::string title = section("Every precision"), legend = dim("GB to download  ") + accent(kOk) + dim(" downloaded");
  const int gap = std::min(W, 78) - width(title) - width(legend);
  o.push_back(gap >= 3 ? title + std::string(size_t(gap), ' ') + legend : title);
  precisions(o, W);
  o.push_back("");
  paragraph(o, "kapitsa pull 35b:6bit downloads one  " + std::string(kDot) + "  kapitsa serve --model 35b:6bit starts it", 2,
            W, dim);
  o.push_back("");
  print(o);
}

void plan(const hub::ModelRef& r, uint64_t imported_bytes) {
  const int W = layout_width();
  const bool local = r.kind == hub::ModelRef::kLocal;
  const bool current = !local && hub::is_current(r);
  struct stat st {};
  const bool here = !local && !current && stat((r.dir + "/config.json").c_str(), &st) == 0;
  const std::string status = local     ? accent(kOn) + dim(" yours, used as is")
                             : current ? accent(kOk) + dim(" downloaded")
                             : here    ? warn("◐") + dim(" files here, not checked yet")
                                       : dim(std::string(kOff) + " not downloaded yet");
  std::vector<std::string> o;
  o.push_back("");
  const std::string title = "  " + bold(r.name);
  const int gap = std::min(W, 78) - width(title) - width(status);
  o.push_back(gap >= 3 ? title + std::string(size_t(gap), ' ') + status : title + "   " + status);
  std::vector<std::pair<std::string, std::string>> rows;
  rows.push_back({"folder", hub::tilde(r.dir)});
  if (!local) rows.push_back({"weights", r.repo + dim(" @ " + short_rev(r.revision))});
  if (r.variant && !r.variant->gguf.path.empty())  // a GGUF variant: its file, then the import
    rows.push_back({"file", r.variant->gguf.path + dim("  " + std::string(kDot) + "  " + hub::fmt_bytes(r.variant->gguf.size) +
                                                       (current ? ", imported" : " to download, then imported") +
                                                       (imported_bytes ? ": " + hub::fmt_bytes(imported_bytes) + " on disk" : ""))});
  if (r.variant)
    rows.push_back({"precision", r.variant->name + dim("  " + std::string(kDot) + "  " + r.variant->quant +
                                                        (r.variant->name == r.model->default_variant ? "  " + std::string(kDot) + "  the default" : ""))});
  if (r.variant)
    for (const auto& c : r.variant->companions)
      rows.push_back({c.drafter() ? "drafter" : "MTP head",
                      c.repo + dim(" @ " + short_rev(c.revision) + (c.kind == "mtp-sidecar" ? ", its mtp.* tensors" : ""))});
  rows.push_back({"API name", r.id});
  for (const auto& [k, v] : rows) o.push_back("    " + left(dim(k), 11) + clip(v, W - 15));
  if (here) {
    o.push_back("");
    paragraph(o, "kapitsa pull checks these files against the pinned checksums and fetches only what differs.", 2, W, dim);
  }
  o.push_back("");
  print(o);
}

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
    const int w = std::max(30, std::min(columns(), 100) - 5);
    const auto ls = wrap(m, w);
    for (size_t i = 0; i < ls.size(); ++i) line((i ? "    " : "  " + warn(kWarn) + " ") + dim(ls[i]), stderr);
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
  // KFlash wraps every decode: ", KFlash (<source>)" ends the mode (the engine's wrapper; absent with NI_KFLASH=0)
  std::string kflash;
  if (const size_t p = mode.rfind(", KFlash ("); p != std::string::npos) kflash = mode.substr(p + 2), mode.resize(p);
  if (!kflash.empty() && mode.rfind("plain", 0) == 0) return kflash + " " + dim("(1 row per cycle: plain decoding)");
  if (mode.rfind("DFlash2", 0) == 0) return "DFlash2 drafter " + dim("(speculative decoding)");
  if (mode.rfind("LiLiCorr", 0) == 0) return "LiLiCorr drafter " + dim("(speculative decoding)");
  if (mode.rfind("DSpark", 0) == 0) return "DSpark drafter " + dim("(speculative decoding)");
  if (mode.rfind("MTP", 0) == 0) return "MTP head " + dim("(speculative decoding, " + after(mode, "depth ") + " ahead)");
  if (mode.rfind("plain", 0) == 0) return "plain " + dim("(no speculative decoding)");
  return kflash.empty() ? mode : mode + ", " + kflash;
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

// The card's addresses: the chat page and the APIs on this Mac, where the other Macs reach it, the API key
void addresses(std::vector<std::string>& rows, const std::string& base, const Network& net, const std::string& key_note) {
  auto row = [&](const std::string& k, const std::string& v) { rows.push_back(left(dim(k), 11) + v); };
  rows.push_back("");
  row("Chat", accent(base));
  row("OpenAI", accent(base + "/v1"));
  row("Anthropic", accent(base));
  if (net.visible && !net.url.empty()) row("Network", accent(net.url));
  if (!key_note.empty()) row("API key", key_note);
}

// Under the card: the server on the network
void network_lines(std::vector<std::string>& o, const Network& n, int W) {
  const std::string s = network_line(n);
  if (s.empty()) return;  // --private: as before
  const auto ls = wrap(s, std::max(30, W - 5));
  for (size_t i = 0; i < ls.size(); ++i) o.push_back("  " + dim(std::string(i ? " " : kArrow) + " " + ls[i]));
}

// The card's last lines
void closing(std::vector<std::string>& o, const std::string& log) {
  o.push_back("  " + dim(std::string(kArrow) + " ") + bold("kapitsa claude") + dim(" in another terminal codes with it"));
  o.push_back("  " + dim(std::string(kArrow) + " " + (log.empty() ? "" : "Log " + hub::tilde(log) + "  " + kDot + "  ") +
                         "Ctrl+C stops"));
  o.push_back("");
}

std::atomic<bool> g_serve_ui{false};
std::mutex g_req_mu;
bool g_req_header = false, g_fwd_header = false;
bool g_show_first = true, g_show_read = true;  // --timing
std::atomic<double> g_rate{0};                 // the latest request's decode rate (recent_tok_s)

}  // namespace

void serve_begin(bool verbose) {
  g_serve_ui = true;
  // NI_LOG asks for the log: on screen too
  g_log.verbose = verbose || getenv("NI_LOG") != nullptr;
  log_sink() = serve_sink;
}

void serve_header(const std::string& version) {
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

// While the model loads: the logo as a loader beside what is happening (where it may play), else a spinner
struct Loading {
  std::string what;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::unique_ptr<Loader> wobble;
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
  l->wobble = std::make_unique<Loader>(
      [l] {
        std::string detail;
        {
          std::lock_guard<std::mutex> lk2(g_log.mu);
          detail = g_log.detail;
        }
        return std::vector<std::string>{"Loading " + bold(l->what) + "  " + dim(duration(l->secs())),
                                        dim(clip(detail, std::max(20, columns() - kLogoCols - 8)))};
      },
      2);
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
    std::string c = context(r.context) + " tokens";
    if (r.slots > 1) c += " " + dim("×") + " " + std::to_string(r.slots) + " conversations";
    if (r.context_fit > 0 && r.context_fit < r.context) c += dim(" (" + count(r.context_fit) + " fit in memory)");
    row("Context", c);
  }
  addresses(rows, r.base, r.net, r.api_key_note);
  std::vector<std::string> o;
  o.push_back("");
  card(o, rows, inner);
  network_lines(o, r.net, W);
  closing(o, r.log);
  std::lock_guard<std::mutex> lk(g_req_mu);
  for (const auto& l : o) line(l);
}

std::string network_line(const Network& n) {
  if (!n.visible) return "";
  std::string s = "Visible to Kapitsa on this network as " + n.name +
                  (n.api_key ? " (its API needs the key; --private hides it)" : " (no password; --private hides it)");
  if (n.others > 0) s += std::string(" ") + kDot + " " + std::to_string(n.others) + (n.others == 1 ? " other Mac" : " other Macs") + ": kapitsa cluster";
  else if (n.others == 0) s += std::string(" ") + kDot + " kapitsa cluster lists the Macs on it";
  return s;
}

std::string route_line(const std::string& model, const cluster::MacInfo& mac) {
  const std::string where = mac.this_mac() ? "the server on port " + std::to_string(mac.port) + " of this Mac"
                                           : cluster::describe(mac) + " " + kDot + " " + cluster::link_text(mac);
  return cluster::model_base(model) + " runs on " + where + ": this Mac sends its requests there. --local runs it here.";
}

void routed(const Routed& r) {
  const int W = layout_width();
  const int inner = std::max(36, std::min(W - 8, 68));
  std::vector<std::string> rows;
  auto row = [&](const std::string& k, const std::string& v) { rows.push_back(left(dim(k), 11) + v); };
  {
    const std::string status = accent(kOn) + " ready", name = bold(r.model);
    const int gap = inner - width(name) - width(status);
    rows.push_back(gap >= 2 ? name + std::string(size_t(gap), ' ') + status : name);
  }
  rows.push_back("");
  row("Runs on", (r.mac.this_mac() ? "this Mac, port " + std::to_string(r.mac.port) : cluster::describe(r.mac)) +
                     dim(", " + cluster::how_it_runs(r.mac)));
  if (!r.mac.this_mac()) row("Link", cluster::link_text(r.mac));
  if (r.mac.tok_s > 0) {
    char b[48];
    snprintf(b, sizeof b, "%.0f tok/s", r.mac.tok_s);
    row("Speed", std::string(b) + dim(" lately"));
  }
  addresses(rows, r.base, r.net, r.api_key_note);
  std::vector<std::string> o;
  o.push_back("");
  card(o, rows, inner);
  const auto rl = wrap(route_line(r.model, r.mac), std::max(30, W - 5));
  for (size_t i = 0; i < rl.size(); ++i) o.push_back("  " + dim(std::string(i ? " " : kArrow) + " " + rl[i]));
  network_lines(o, r.net, W);
  closing(o, r.log);
  std::lock_guard<std::mutex> lk(g_req_mu);
  for (const auto& l : o) line(l);
}

// ---------------------------------------------------------------- the request log and the dashboard

struct Meter {
  double t0 = now_ms();
  std::atomic<double> first{0};
  std::string client;
  int images = 0;
  std::shared_ptr<LiveDecode> live = std::make_shared<LiveDecode>();
  std::deque<std::array<double, 3>> samples;  // the dashboard thread's: (ms, tokens, cycles), the last 1.5 s
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

// The request lines' columns
struct Cols {
  bool time, first, read;  // first, read: the time to the first token and the prefill tok/s (--timing); read needs a wide terminal
};
Cols cols_now() { return Cols{columns() >= 76, g_show_first, g_show_read && (columns() >= 98 || !g_show_first)}; }

void request_header() {  // once, above the first request (g_req_mu)
  if (g_req_header) return;
  g_req_header = true;
  const Cols c = cols_now();
  std::string h = "  ";
  if (c.time) h += left("time", 8) + "  ";
  h += left("client", 9) + "  " + right("prompt", 6) + "  " + right("cache", 5) + "  ";
  if (c.first) h += right("first", 6) + "  ";
  if (c.read) h += right("read/s", 6) + "  ";
  h += right("out", 5) + "  " + right("tok/s", 5) + "  " + right("step", 4) + "  finish";
  line(section("Requests"));
  line(dim(h));
  if (c.read) line(dim("  read/s: prompt tokens read per second of prefill, those not cached" + std::string(c.first ? " (first: time to first token)" : "")));
}

// ---- the dashboard: while requests are written, a strip under the request lines. The newest request as a beam
// (Kapitza-Dirac: one step of the drafter and the verify comes out as the tokens it accepted), the others a line each;
// where no animation may play, one line of text.

struct Dash {
  std::mutex mu;
  std::condition_variable cv;
  // the requests being served; weak: one that ends without done() (its client gone) leaves with its last owner
  std::vector<std::weak_ptr<Meter>> active;
  bool running = false;
};
Dash& dash() {
  static Dash* d = new Dash;  // never destroyed: its thread may outlive main's statics
  return *d;
}

// tokens per second and per step over the last 1.5 s (since the first token while there is less)
std::pair<double, double> rates(Meter& m, double now) {
  const double tok = m.live->tokens.load(std::memory_order_relaxed), cyc = m.live->cycles.load(std::memory_order_relaxed);
  m.samples.push_back({now, tok, cyc});
  while (m.samples.size() > 2 && now - m.samples.front()[0] > 1500) m.samples.pop_front();
  const auto& a = m.samples.front();
  double dt = now - a[0], dtok = tok - a[1], dcyc = cyc - a[2];
  const double first = m.first.load();
  if (dt < 300 && first > 0) dt = now - first, dtok = tok - 1, dcyc = cyc;  // the start: since the first token
  return {dt > 50 && dtok > 0 ? dtok * 1000 / dt : 0, dcyc > 0 ? dtok / dcyc : 0};
}

std::string state(Meter& m, double now, bool beam) {
  const int phase = m.live->phase.load(std::memory_order_relaxed);
  if (phase == 0) return dim("waiting for the engine");
  if (phase == 1) {
    const int rd = m.live->read.load(std::memory_order_relaxed), p = m.live->prompt.load(std::memory_order_relaxed);
    std::string s = "reading the prompt";
    if (p > 4096 && rd < p) s = "reading " + count(rd) + dim(" of ") + count(p);  // a long prompt: how far it is
    return s + dim("  " + duration((now - m.t0) / 1000));
  }
  const auto [tps, step] = rates(m, now);
  char b[64];
  std::string s = "writing ";
  snprintf(b, sizeof b, "%.0f tok/s", tps);
  s += beam ? accent(b) : std::string(b);
  if (m.live->spec.load(std::memory_order_relaxed) > 0 && step > 0) {
    snprintf(b, sizeof b, "%.1f tokens/step", step);
    s += dim("  " + std::string(kDot) + "  ") + b;
  }
  return s;
}

std::string prompt_info(const Meter& m) {
  const int p = m.live->prompt.load(std::memory_order_relaxed), c = m.live->cached.load(std::memory_order_relaxed);
  if (p <= 0) return "";
  char b[32];
  snprintf(b, sizeof b, ", %.0f%% cached", 100.0 * c / p);
  return count(p) + " prompt" + dim(b);
}

std::vector<std::string> strip(const std::vector<std::shared_ptr<Meter>>& act, double t, bool art) {
  const double now = now_ms();
  // the newest request being read or written gets the beam
  std::shared_ptr<Meter> top = act.back();
  for (auto it = act.rbegin(); it != act.rend(); ++it)
    if ((*it)->live->phase.load(std::memory_order_relaxed) >= 1) {
      top = *it;
      break;
    }
  Meter& m = *top;
  const std::string who = bold(m.client.empty() ? "request" : m.client);
  std::vector<std::string> rows;
  if (art) {
    const int phase = m.live->phase.load(std::memory_order_relaxed);
    Beam b;
    b.t = t;
    b.flow = b.shimmer = true;
    const bool spec = m.live->spec.load(std::memory_order_relaxed) > 0;
    b.k = phase == 2 ? (spec ? std::clamp(m.live->last.load(std::memory_order_relaxed), 1, 8) : 1) : 0;
    b.band = 0.3;
    // reading: the beam comes in as the prompt is read; writing: it crosses the wave and fans out
    const double p = m.live->prompt.load(std::memory_order_relaxed);
    const double rd = p > 0 ? double(m.live->read.load(std::memory_order_relaxed)) / p : 0;
    b.reach = phase == 2 ? 1 : phase == 1 ? 0.06 + 0.28 * rd : 0.06;
    const int toks = m.live->tokens.load(std::memory_order_relaxed);
    const std::string pi = prompt_info(m);
    std::vector<std::string> text = {who + (pi.empty() ? "" : dim("  " + std::string(kDot) + "  ") + pi), state(m, now, true),
                                     dim(std::to_string(toks) + " tokens  " + kDot + "  " + duration((now - m.t0) / 1000))};
    rows = beside(diffraction(24, 3, b), 24, text, 0);
  } else {
    rows.push_back("  " + accent(kOn) + " " + who + dim("  " + std::string(kDot) + "  ") + state(m, now, false));
  }
  int more = 0;
  for (const auto& o : act) {
    if (o == top) continue;
    if (++more > 3) break;
    rows.push_back("    " + dim("↳ ") + bold(o->client.empty() ? "request" : o->client) + dim("  " + std::string(kDot) + "  ") +
                   state(*o, now, false));
  }
  return rows;
}

void dash_loop() {
  Dash& d = dash();
  const bool art = can_animate();
  const auto t0 = std::chrono::steady_clock::now();
  bool shown = false;
  for (;;) {
    std::vector<std::shared_ptr<Meter>> act;
    {
      std::unique_lock<std::mutex> lk(d.mu);
      d.active.erase(std::remove_if(d.active.begin(), d.active.end(), [](const auto& w) { return w.expired(); }),
                     d.active.end());
      if (d.active.empty()) {
        if (shown) status(""), shown = false;
        d.cv.wait(lk, [&] { return !d.active.empty(); });
      }
      for (const auto& w : d.active)
        if (auto m = w.lock()) act.push_back(m);
    }
    if (act.empty()) continue;
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    status(strip(act, t, art));
    shown = true;
    std::unique_lock<std::mutex> lk(d.mu);  // a frame every 50 ms (4 a second for the text line), or at once on a change
    d.cv.wait_for(lk, std::chrono::milliseconds(art ? 50 : 250));
  }
}

void dash_add(const std::shared_ptr<Meter>& m) {
  Dash& d = dash();
  std::lock_guard<std::mutex> lk(d.mu);
  d.active.push_back(m);
  if (!d.running && live()) {
    d.running = true;
    std::thread(dash_loop).detach();
  }
  d.cv.notify_all();
}

void dash_remove(const std::shared_ptr<Meter>& m) {
  Dash& d = dash();
  std::lock_guard<std::mutex> lk(d.mu);
  d.active.erase(std::remove_if(d.active.begin(), d.active.end(), [&](const auto& w) { return w.expired() || w.lock() == m; }),
                 d.active.end());
  d.cv.notify_all();
}

}  // namespace

std::shared_ptr<Meter> meter(const httplib::Request& req, const ChatRequest& r) {
  if (!g_serve_ui) return nullptr;
  auto m = std::make_shared<Meter>();
  m->client = client_of(req.get_header_value("User-Agent"));
  m->images = images_in(r.messages);
  {
    std::lock_guard<std::mutex> lk(g_req_mu);
    request_header();
  }
  dash_add(m);
  return m;
}

std::shared_ptr<LiveDecode> live_of(const std::shared_ptr<Meter>& m) { return m ? m->live : nullptr; }

void token(const std::shared_ptr<Meter>& m) {
  if (m && m->first.load() == 0) m->first = now_ms();
}

void done(const std::shared_ptr<Meter>& m, const ChatResult& r) {
  if (r.completion_tokens > 1 && r.decode_ms > 0) g_rate = (r.completion_tokens - 1) / (r.decode_ms / 1000);
  if (!m) return;
  dash_remove(m);
  const time_t now = time(nullptr);
  char at[16];
  strftime(at, sizeof at, "%H:%M:%S", localtime(&now));
  const double first = m->first.load();
  char b[32];
  std::string ttft = "–", tps = "–", cached = "–", step = "–", pf = "–";
  if (first > 0) snprintf(b, sizeof b, "%.2fs", (first - m->t0) / 1000), ttft = b;
  // prefill tok/s: the prompt tokens read (not the cached ones) over the time it took
  if (const int fresh = r.prompt_tokens - r.cached_tokens; fresh > 0 && r.prefill_exact && r.prefill_ms > 0 && r.finish_reason != "cancelled" &&
                                                           r.finish_reason != "error")
    snprintf(b, sizeof b, "%.0f", fresh / (r.prefill_ms / 1000)), pf = b;
  if (r.completion_tokens > 1 && r.decode_ms > 0)
    snprintf(b, sizeof b, "%.0f", (r.completion_tokens - 1) / (r.decode_ms / 1000)), tps = b;
  if (r.prompt_tokens > 0) snprintf(b, sizeof b, "%.0f%%", 100.0 * r.cached_tokens / r.prompt_tokens), cached = b;
  const int cyc = m->live->cycles.load(), tok = m->live->tokens.load();
  if (m->live->spec.load() > 0 && cyc > 0 && tok > 1) snprintf(b, sizeof b, "%.1f", double(tok) / cyc), step = b;
  std::string fin = r.finish_reason;
  if (fin == "tool_calls") fin = accent("tools");
  else if (fin == "length") fin = warn(r.context_full ? "context full" : "length");
  else if (fin == "cancelled") fin = warn("cancelled");
  else if (fin == "error") fin = red("error") + (r.error.empty() ? "" : dim(" " + r.error));
  if (m->images > 0) fin += dim("  " + std::to_string(m->images) + (m->images == 1 ? " image" : " images"));
  const Cols c = cols_now();
  std::string l = "  ";
  if (c.time) l += dim(at) + "  ";
  l += left(clip(m->client.empty() ? dim("–") : m->client, 9), 9) + "  " +
       right(r.prompt_tokens ? count(r.prompt_tokens) : "–", 6) + "  " + right(cached, 5) + "  ";
  if (c.first) l += right(ttft, 6) + "  ";
  if (c.read) l += right(pf, 6) + "  ";
  l += right(count(r.completion_tokens), 5) + "  " + right(tps, 5) + "  " + right(step, 4) + "  " + fin;
  std::lock_guard<std::mutex> lk(g_req_mu);
  request_header();
  line(clip(l, columns() - 1));
}

bool set_timing(const std::string& mode) {
  if (mode != "ttft" && mode != "prefill" && mode != "both") return false;
  g_show_first = mode != "prefill";
  g_show_read = mode != "ttft";
  return true;
}

void forwarded(const httplib::Request& req, int status, double secs, const std::string& mac) {
  const std::string what = req.method + " " + req.path;
  logi("forwarded %s to %s: %d in %.2f s", what.c_str(), mac.empty() ? "no Mac (none serves the model now)" : mac.c_str(),
       status, secs);
  if (!g_serve_ui) return;
  const time_t now = time(nullptr);
  char at[16];
  strftime(at, sizeof at, "%H:%M:%S", localtime(&now));
  const std::string st = status < 400 ? std::to_string(status) : status == 499 ? warn("gone") : red(std::to_string(status));
  const std::string client = client_of(req.get_header_value("User-Agent"));
  std::lock_guard<std::mutex> lk(g_req_mu);
  if (!g_fwd_header) {
    g_fwd_header = true;
    line(section("Requests") + dim("  sent on to the Mac that runs the model"));
    line(dim("  " + left("time", 8) + "  " + left("client", 9) + "  " + left("request", 28) + "  " + left("status", 6) + "  " +
             right("time", 6)));
  }
  line(clip("  " + dim(at) + "  " + left(clip(client.empty() ? dim("–") : client, 9), 9) + "  " + left(clip(what, 28), 28) +
                "  " + left(st, 6) + "  " + right(duration(secs), 6) +
                (mac.empty() ? "" : "  " + dim(std::string(kArrow) + " " + mac)),
            columns() - 1));
}

double recent_tok_s() { return g_rate.load(); }

}  // namespace ni::ui
