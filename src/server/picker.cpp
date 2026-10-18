// Which model to run, asked in the terminal (picker.h).
#include "server/picker.h"

#include "server/cli.h"
#include "util/log.h"
#include "util/term.h"

#include <nlohmann/json.hpp>

#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

namespace ni::ui {

using namespace term;
using ojson = nlohmann::ordered_json;

namespace {

int term_rows() {
  winsize ws {};
  for (int fd : {2, 1, 0})
    if (ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) return ws.ws_row;
  return 24;
}

bool in_ci() {
  const char* ci = getenv("CI");
  return ci && *ci && strcmp(ci, "0") != 0 && strcmp(ci, "false") != 0;
}

// A person runs this command: every stream a terminal, not a CI job
bool at_terminal() { return isatty(0) && isatty(1) && isatty(2) && !in_ci(); }

// ---------------------------------------------------------------- the remembered choice

std::string state_path() { return hub::home_dir() + "/state.json"; }

ojson read_state() {
  std::ifstream f(state_path());
  if (!f) return ojson::object();
  std::stringstream ss;
  ss << f.rdbuf();
  ojson j = ojson::parse(ss.str(), nullptr, false);
  return j.is_object() ? j : ojson::object();
}

// The catalog model and precision chosen last, if any (one the catalog no longer has is forgotten)
bool remembered(const hub::CatalogModel** m, const hub::Variant** v) {
  const ojson j = read_state();
  if (!j.contains("model") || !j["model"].is_string()) return false;
  try {
    const hub::ModelRef r = hub::resolve(j["model"].get<std::string>());
    if (r.kind != hub::ModelRef::kCatalog || !r.model || !r.variant) return false;
    *m = r.model, *v = r.variant;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// ---------------------------------------------------------------- what each choice is on this Mac

enum class Fit { kUnknown, kFits, kNoDrafter, kTooBig };

struct Choice {
  const hub::Variant* v = nullptr;
  hub::ModelRef ref;
  std::string label;     // "4-bit"
  bool have = false;     // downloaded
  bool present = false;  // its folder is there (a download to finish, or files to check)
  Fit fit = Fit::kUnknown;
};

struct ModelRow {
  const hub::CatalogModel* m = nullptr;
  std::vector<Choice> choices;
  size_t def = 0;           // the default precision
  Fit fit = Fit::kUnknown;  // some precision fits / the smallest is too big
  bool best = false;
};

// "4-bit", "bf16"; a model's only precision has no name of its own here
std::string precision(const hub::CatalogModel& m, const hub::Variant& v) {
  if (!v.label.empty()) return m.variants.size() == 1 ? "" : v.label;
  std::string s = v.name;
  if (s.size() > 3 && s.compare(s.size() - 3, 3, "bit") == 0) s.insert(s.size() - 3, "-");
  return s;
}

std::string gb(double g) {
  char b[32];
  snprintf(b, sizeof b, "%.1f GB", g);
  return b;
}

// The catalog as the picker lists it: what is downloaded (quick); the fits come from the Planner
std::vector<ModelRow> catalog_rows() {
  std::vector<ModelRow> out;
  for (const auto& m : hub::catalog()) {
    ModelRow row;
    row.m = &m;
    for (const auto& v : m.variants) {
      Choice c;
      c.v = &v;
      c.label = precision(m, v);
      try {
        c.ref = hub::resolve(m.name + ":" + v.name);
        c.have = hub::is_current(c.ref);
        struct stat st {};
        c.present = stat(c.ref.dir.c_str(), &st) == 0;
      } catch (const std::exception&) {
      }
      if (&v == m.variant()) row.def = row.choices.size();
      row.choices.push_back(c);
    }
    out.push_back(row);
  }
  return out;
}

void model_fit(ModelRow& row) {
  row.fit = Fit::kUnknown;
  const Choice* smallest = nullptr;
  for (const auto& c : row.choices) {
    if (c.fit == Fit::kFits || c.fit == Fit::kNoDrafter) row.fit = Fit::kFits;
    if (!smallest || c.v->download_gb < smallest->v->download_gb) smallest = &c;
  }
  if (row.fit != Fit::kFits && smallest && smallest->fit == Fit::kTooBig) row.fit = Fit::kTooBig;
}

// The model `kapitsa serve` starts without --model (best_model, from the fits): the first whose default precision
// (or a fallback) fits with its drafter, or whose smallest of those holds a conversation without it. nullptr when none
// fits, or (*unknown) when a plan it needs is not known (yet).
const hub::CatalogModel* best_of(const std::vector<ModelRow>& rows, bool* unknown) {
  *unknown = false;
  for (const auto& r : rows) {
    std::vector<std::string> order{r.m->default_variant};
    order.insert(order.end(), r.m->fallbacks.begin(), r.m->fallbacks.end());
    Fit last = Fit::kUnknown;
    for (const auto& name : order)
      for (const auto& c : r.choices)
        if (c.v->name == name) {
          if (c.fit == Fit::kUnknown) return *unknown = true, nullptr;
          if (c.fit == Fit::kFits) return r.m;
          last = c.fit;
        }
    if (last == Fit::kNoDrafter) return r.m;
  }
  return nullptr;
}

// The engine's plan of a precision's headers (hub::set_offline: only plans already on this Mac)
Fit fit_of(const hub::ModelRef& r, const EngineOptions& opt, size_t budget) {
  if (r.kind != hub::ModelRef::kCatalog) return Fit::kUnknown;
  const int f = variant_fits(r, opt, budget);
  if (f == 1) return Fit::kFits;
  if (f < 0) return Fit::kUnknown;
  try {  // a conversation fits, but not beside the drafter, or not even that
    const std::string d = hub::plan_dir(r);
    if (d.empty()) return Fit::kUnknown;
    return conversation_fits(d, opt.context, opt.slots, budget) ? Fit::kNoDrafter : Fit::kTooBig;
  } catch (const std::exception&) {
    return Fit::kUnknown;
  }
}

// Whether the plans of every model's default precision are on this Mac (without them the best model needs the network)
bool plans_here(const std::vector<ModelRow>& rows) {
  hub::set_offline(true);
  bool all = true;
  for (const auto& r : rows) {
    try {
      all = all && !hub::plan_dir(r.choices[r.def].ref).empty();
    } catch (const std::exception&) {
      all = false;
    }
  }
  hub::set_offline(false);
  return all;
}

// Plans every choice offline on a thread of its own, so that the picker shows at once and its marks fill in (a model
// with a large config.json takes most of a second): the row in focus first, then each model's default, then the rest.
// Nothing else plans while it runs (hub::set_offline is process-wide); finish() waits for the plan under way.
class Planner {
 public:
  Planner(const std::vector<ModelRow>& rows, const EngineOptions& opt, size_t budget) : opt_(opt), budget_(budget) {
    for (const auto& r : rows) {
      refs_.emplace_back();
      for (const auto& c : r.choices) refs_.back().push_back(c.ref);
      fits_.emplace_back(r.choices.size(), Fit::kUnknown);
      planned_.emplace_back(r.choices.size(), false);
      defs_.push_back(r.def);
    }
    hub::set_offline(true);
    th_ = std::thread([this] { run(); });
  }
  ~Planner() { finish(); }
  void focus(size_t row) { focus_ = int(row); }
  // The fits planned since the last call, into `rows`; whether there were any
  bool collect(std::vector<ModelRow>& rows) {
    std::lock_guard<std::mutex> lk(mu_);
    if (version_ == seen_) return false;
    seen_ = version_;
    for (size_t i = 0; i < rows.size(); ++i) {
      for (size_t k = 0; k < rows[i].choices.size(); ++k) rows[i].choices[k].fit = fits_[i][k];
      model_fit(rows[i]);
    }
    return true;
  }
  bool done() {
    std::lock_guard<std::mutex> lk(mu_);
    return done_;
  }
  void finish() {
    stop_ = true;
    if (th_.joinable()) {
      th_.join();
      hub::set_offline(false);
    }
  }

 private:
  // The next choice to plan, or false when all are
  bool next(size_t* i, size_t* k) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t f = size_t(focus_.load());
    auto open = [&](size_t a, size_t b) { return a < planned_.size() && b < planned_[a].size() && !planned_[a][b]; };
    if (f < defs_.size() && open(f, defs_[f])) return *i = f, *k = defs_[f], true;
    for (size_t b = 0; f < planned_.size() && b < planned_[f].size(); ++b)
      if (open(f, b)) return *i = f, *k = b, true;
    for (size_t a = 0; a < planned_.size(); ++a)
      if (open(a, defs_[a])) return *i = a, *k = defs_[a], true;
    for (size_t a = 0; a < planned_.size(); ++a)
      for (size_t b = 0; b < planned_[a].size(); ++b)
        if (open(a, b)) return *i = a, *k = b, true;
    done_ = true;
    ++version_;
    return false;
  }
  void run() {
    size_t i, k;
    while (!stop_ && next(&i, &k)) {
      const Fit f = fit_of(refs_[i][k], opt_, budget_);
      std::lock_guard<std::mutex> lk(mu_);
      fits_[i][k] = f;
      planned_[i][k] = true;
      ++version_;
    }
  }

  EngineOptions opt_;
  size_t budget_;
  std::vector<std::vector<hub::ModelRef>> refs_;
  std::vector<size_t> defs_;
  std::mutex mu_;
  std::vector<std::vector<Fit>> fits_;
  std::vector<std::vector<bool>> planned_;
  int version_ = 0, seen_ = 0;
  bool done_ = false;
  std::atomic<int> focus_{0};
  std::atomic<bool> stop_{false};
  std::thread th_;
};

// ---------------------------------------------------------------- keys

// The terminal in raw mode while the picker reads keys (no echo, no line editing, Ctrl+C a key); the modes come back
// at the end, and on a signal that ends the process (before the status block's own handler clears the screen)
termios g_saved;
std::atomic<bool> g_raw{false};
const int kSignals[] = {SIGINT, SIGTERM, SIGHUP, SIGQUIT};
struct sigaction g_prev[4];

void restore_mode() {
  if (g_raw.exchange(false)) tcsetattr(0, TCSANOW, &g_saved);
}

void on_signal(int sig) {
  restore_mode();
  for (int i = 0; i < 4; ++i)
    if (kSignals[i] == sig) sigaction(sig, &g_prev[i], nullptr);
  raise(sig);  // delivered when this handler returns, to the previous one
}

class Raw {
 public:
  Raw() {
    if (tcgetattr(0, &g_saved) != 0) return;
    termios raw = g_saved;
    raw.c_lflag &= ~tcflag_t(ICANON | ECHO | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    struct sigaction sa {};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    for (int i = 0; i < 4; ++i) {
      sigaction(kSignals[i], nullptr, &g_prev[i]);
      if (g_prev[i].sa_handler != SIG_IGN) sigaction(kSignals[i], &sa, nullptr);
    }
    handlers_ = true;
    if (tcsetattr(0, TCSANOW, &raw) == 0) g_raw = true;
  }
  ~Raw() { restore(); }
  // The terminal's modes and the signal handlers as they were
  void restore() {
    restore_mode();
    if (!handlers_) return;
    handlers_ = false;
    for (int i = 0; i < 4; ++i)
      if (g_prev[i].sa_handler != SIG_IGN) sigaction(kSignals[i], &g_prev[i], nullptr);
  }
  bool on() const { return g_raw; }

 private:
  bool handlers_ = false;
};

enum class Key { kNone, kUp, kDown, kLeft, kRight, kEnter, kEsc, kCancel, kFirst, kLast, kDigit };
struct Press {
  Key key = Key::kNone;
  int digit = 0;
};

// A byte from the terminal within `ms`: -1 none, -2 the terminal is gone; one can be put back
int g_back = -1;
int byte_within(int ms) {
  if (g_back >= 0) {
    const int c = g_back;
    g_back = -1;
    return c;
  }
  pollfd p{0, POLLIN, 0};
  const int n = poll(&p, 1, ms);
  if (n <= 0) return -1;
  if (!(p.revents & POLLIN)) return -2;
  unsigned char c;
  return read(0, &c, 1) == 1 ? c : -2;
}

Press read_key(int ms) {
  const int c = byte_within(ms);
  if (c == -1) return {};
  if (c == -2) return {Key::kCancel};
  if (c == 27) {
    // Esc alone, or the start of a key's sequence (ESC [ A, ESC O A: the arrows; ESC [ 1 ~ ...)
    const int b = byte_within(40);
    if (b == 27) g_back = b;  // Esc, then another key
    if (b == -1 || b == 27) return {Key::kEsc};
    if (b != '[' && b != 'O') return {};  // Alt + a key
    std::string params;
    for (int i = 0; i < 16; ++i) {
      const int x = byte_within(40);
      if (x < 0) return {};
      if (x >= 0x40 && x <= 0x7E) {
        switch (x) {
          case 'A': return {Key::kUp};
          case 'B': return {Key::kDown};
          case 'C': return {Key::kRight};
          case 'D': return {Key::kLeft};
          case 'H': return {Key::kFirst};
          case 'F': return {Key::kLast};
          case '~':
            if (params == "1" || params == "7" || params == "5") return {Key::kFirst};
            if (params == "4" || params == "8" || params == "6") return {Key::kLast};
            return {};
          default: return {};
        }
      }
      params += char(x);
    }
    return {};
  }
  if (c == '\r' || c == '\n') return {Key::kEnter};
  if (c == 3 || c == 4 || c == 'q') return {Key::kCancel};  // Ctrl+C, Ctrl+D
  if (c == 'k' || c == 16) return {Key::kUp};                // Ctrl+P
  if (c == 'j' || c == 14) return {Key::kDown};              // Ctrl+N
  if (c == 'h') return {Key::kLeft};
  if (c == 'l') return {Key::kRight};
  if (c >= '1' && c <= '9') return {Key::kDigit, c - '0'};
  return {};
}

// ---------------------------------------------------------------- the screens

int layout_width() { return std::min(columns(), 100); }

std::string keys_line(const char* enter, const char* esc) {
  const std::string sep = dim("  " + std::string(kDot) + "  ");
  return "  ↑↓" + dim(" move") + sep + "enter" + dim(std::string(" ") + enter) + sep + "esc" + dim(std::string(" ") + esc);
}

// "! text", wrapped under itself
void warning_lines(std::vector<std::string>& o, const std::string& text, int w) {
  const auto ls = wrap(text, std::max(20, w - 5));
  for (size_t i = 0; i < ls.size(); ++i) o.push_back((i ? "    " : "  " + warn(kWarn) + " ") + ls[i]);
}

// "title ... legend", the legend at the right edge of the rows below when it fits
std::string title_line(const std::string& title, const std::string& legend, int edge) {
  const int gap = edge - width(title) - width(legend);
  return legend.empty() || gap < 3 ? title : title + std::string(size_t(gap), ' ') + legend;
}

std::vector<std::string> models_screen(const std::vector<ModelRow>& rows, size_t cur, size_t budget) {
  const int W = layout_width();
  std::vector<std::string> sizes, marks;
  int nw = 0, sw = 0, mw = 3;  // the marks' column keeps room for "✓ ★": it does not move once the ★ is planned
  bool any_have = false, any_best = false;
  for (const auto& r : rows) {
    const Choice& d = r.choices[r.def];
    std::string mk;
    if (d.have) mk = accent(kOk);
    else
      for (const auto& c : r.choices)
        if (c.have) {
          mk = accent(kOk) + (c.label.empty() ? "" : " " + dim(c.label));
          break;
        }
    any_have = any_have || !mk.empty();
    if (r.best) mk += (mk.empty() ? "  " : " ") + accent(kStar), any_best = true;
    sizes.push_back(gb(d.v->download_gb));
    marks.push_back(mk);
    nw = std::max(nw, width(r.m->name));
    sw = std::max(sw, width(sizes.back()));
    mw = std::max(mw, width(mk));
  }
  const int pw = W - 1 - (4 + nw + 3 + sw + 2 + mw + 3);
  std::vector<std::string> o = {""};
  std::vector<std::string> table;
  int tw = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const ModelRow& r = rows[i];
    const bool sel = i == cur, big = r.fit == Fit::kTooBig;
    const std::string name = sel ? accent(bold(r.m->name)) : big ? dim(r.m->name) : r.m->name;
    std::string l = "  " + (sel ? accent(kArrow) : std::string(" ")) + " " + left(name, nw) + "   " +
                    right(dim(sizes[i]), sw) + "  " + left(marks[i], mw);
    if (pw >= 12) {
      const std::string what = big ? "too big for this Mac" : r.m->brief.empty() ? r.m->about : r.m->brief;
      l += "   " + (sel ? (big ? warn(clip(what, pw)) : clip(what, pw)) : dim(clip(what, pw)));
    }
    while (!l.empty() && l.back() == ' ') l.pop_back();
    tw = std::max(tw, width(l));
    table.push_back(l);
  }
  std::string legend;
  if (any_have) legend += accent(kOk) + dim(" downloaded");
  if (any_best) legend += (legend.empty() ? "" : "   ") + accent(kStar) + dim(" best for this Mac");
  o.push_back(title_line("  " + bold("Pick a model"), legend, std::min(std::max(tw, 60), W - 1)));
  for (auto& l : table) o.push_back(l);
  if (rows[cur].fit == Fit::kTooBig) {
    char b[160];
    snprintf(b, sizeof b, "no precision of it fits the %.1f GB of GPU memory on this Mac", budget / 1e9);
    warning_lines(o, b, W);
  }
  o.push_back("");
  o.push_back(keys_line(rows[cur].choices.size() > 1 ? "next" : "start", "cancel"));
  return o;
}

std::vector<std::string> precisions_screen(const ModelRow& r, size_t cur, const std::string& command, size_t budget) {
  const int W = layout_width();
  int lw = 0, sw = 0;
  bool any_have = false;
  std::vector<std::string> fits;
  for (const auto& c : r.choices) {
    lw = std::max(lw, width(c.label));
    sw = std::max(sw, width(gb(c.v->download_gb)));
    any_have = any_have || c.have;
    fits.push_back(c.fit == Fit::kFits        ? "fits"
                   : c.fit == Fit::kNoDrafter ? "fits, without the drafter (slower)"
                   : c.fit == Fit::kTooBig    ? "too big for this Mac"
                                              : "");
  }
  // the fit column as wide as its longest text, so that the notes beside it stay put as the fits are planned
  int fw = width("too big for this Mac");
  for (const auto& f : fits) fw = std::max(fw, width(f));
  // the notes ("the default", "full precision") where the line has room for them
  const bool notes = 4 + lw + 3 + sw + 2 + 1 + 2 + fw + 3 + width("full precision") <= W - 1;
  std::vector<std::string> o = {""};
  std::vector<std::string> table;
  int tw = 0;
  for (size_t i = 0; i < r.choices.size(); ++i) {
    const Choice& c = r.choices[i];
    const bool sel = i == cur, big = c.fit == Fit::kTooBig;
    const std::string label = sel ? accent(bold(c.label)) : big ? dim(c.label) : c.label;
    std::string l = "  " + (sel ? accent(kArrow) : std::string(" ")) + " " + left(label, lw) + "   " +
                    right(dim(gb(c.v->download_gb)), sw) + "  " + (c.have ? accent(kOk) : std::string(" ")) + "  " +
                    left(sel && big ? warn(fits[i]) : dim(fits[i]), fw);
    const std::string note = i == r.def ? "the default" : c.v->name == "bf16" ? "full precision" : "";
    if (notes && !note.empty()) l += "   " + dim(note);
    while (!l.empty() && l.back() == ' ') l.pop_back();
    tw = std::max(tw, width(l));
    table.push_back(clip(l, W - 1));
  }
  o.push_back(title_line("  " + bold(r.m->name) + dim("  " + std::string(kDot) + "  precision"),
                         any_have ? accent(kOk) + dim(" downloaded") : "", std::min(std::max(tw, 60), W - 1)));
  for (auto& l : table) o.push_back(l);
  if (r.choices[cur].fit == Fit::kTooBig) {
    char b[160];
    snprintf(b, sizeof b, "it may not start, or only with a short context (%.1f GB of GPU memory here)", budget / 1e9);
    warning_lines(o, b, W);
  }
  o.push_back("");
  o.push_back(keys_line("start", "back"));
  const std::string hint = command + " --model " + hub::short_name(*r.m, *r.choices[cur].v) + " starts it without asking";
  if (width(hint) + 2 <= W - 1) o.push_back("  " + dim(hint));
  return o;
}

}  // namespace

// ---------------------------------------------------------------- the picker

bool can_prompt(bool yes) {
  if (yes || env_on("KAPITSA_NO_PROMPT") || in_ci()) return false;
  // live(): not TERM=dumb. A brew service (a launchd job: stdin /dev/null, output to its log) has no terminal.
  if (!isatty(0) || !isatty(1) || !isatty(2) || !live()) return false;
  if (tcgetpgrp(0) != getpgrp()) return false;  // a background job: the keyboard is someone else's
  termios t {};
  if (tcgetattr(0, &t) != 0) return false;
  return columns() >= 40 && term_rows() >= 14;
}

std::optional<std::string> pick_model(const EngineOptions& opt, size_t budget, const std::string& command) {
  (void)accent(" ");  // the terminal's colors are asked for before the keys are read
  guard_signals();    // the status block's signal handlers, which ours pass the signal on to
  Raw raw;            // from now on, keys wait for the picker (unechoed), even those typed while it gets ready
  std::vector<ModelRow> rows = catalog_rows();
  const hub::CatalogModel* last_m = nullptr;
  const hub::Variant* last_v = nullptr;
  const bool has_last = remembered(&last_m, &last_v);
  if (!has_last && !plans_here(rows)) {
    // the first run: the plans the best model needs, fetched as `kapitsa serve` without --model fetches them
    Spinner sp("Checking which models fit this Mac");
    hub::ModelRef ref;
    best_model(opt, budget, &ref, nullptr);
    sp.done();
  }
  // the first choice: the last one, else the best for this Mac once it is planned (each model at its default
  // precision, the last one's at its last precision)
  size_t cur = 0;
  std::vector<size_t> vcur(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    vcur[i] = rows[i].def;
    if (has_last && rows[i].m == last_m) {
      cur = i;
      for (size_t k = 0; k < rows[i].choices.size(); ++k)
        if (rows[i].choices[k].v == last_v) vcur[i] = k;
    }
  }
  Planner planner(rows, opt, budget);
  planner.focus(cur);
  bool moved = has_last;  // the cursor no longer follows the best model
  int step = 1;
  auto plan = [&] {  // the fits planned since: the marks, the ★, and the cursor on it until the person moves it
    if (!planner.collect(rows)) return false;
    bool unknown = false;
    const hub::CatalogModel* best = best_of(rows, &unknown);
    for (size_t i = 0; i < rows.size(); ++i) {
      rows[i].best = rows[i].m == best;
      if (best && !moved && rows[i].m == best && cur != i) cur = i, planner.focus(i);
    }
    return true;
  };
  auto draw = [&] {
    status(step == 1 ? models_screen(rows, cur, budget) : precisions_screen(rows[cur], vcur[cur], command, budget));
  };
  plan();
  draw();
  bool chosen = false;
  {
    if (!raw.on()) {
      chosen = true;  // no keys to read: the first choice
    } else {
      int cols = columns();
      for (;;) {
        const Press p = read_key(planner.done() ? 250 : 30);
        bool redraw = plan();
        if (columns() != cols) cols = columns(), redraw = true;  // resized
        if (p.key == Key::kNone) {
          if (redraw) draw();
          continue;
        }
        const size_t n = step == 1 ? rows.size() : rows[cur].choices.size();
        size_t& at = step == 1 ? cur : vcur[cur];
        if (p.key == Key::kCancel || (p.key == Key::kEsc && step == 1)) break;
        if (p.key == Key::kUp) at = (at + n - 1) % n, moved = true;
        else if (p.key == Key::kDown) at = (at + 1) % n, moved = true;
        else if (p.key == Key::kFirst) at = 0, moved = true;
        else if (p.key == Key::kLast) at = n - 1, moved = true;
        else if (p.key == Key::kDigit && size_t(p.digit) <= n) at = size_t(p.digit) - 1, moved = true;
        else if ((p.key == Key::kEsc || p.key == Key::kLeft) && step == 2) step = 1;
        else if (p.key == Key::kEnter || (p.key == Key::kRight && step == 1)) {
          moved = true;
          if (step == 1 && rows[cur].choices.size() > 1) {
            step = 2;
          } else {
            chosen = true;
            break;
          }
        }
        planner.focus(cur);
        draw();
      }
    }
    status("");  // the picker off the screen, then the terminal's modes back
    raw.restore();
  }
  std::optional<std::string> out;
  if (chosen) {
    const ModelRow& r = rows[cur];
    const Choice& c = r.choices[vcur[cur]];
    std::string l = "  " + accent(kOk) + " " + bold(r.m->name) + (c.label.empty() ? "" : dim(" " + std::string(kDot) + " ") + c.label);
    if (!c.have)
      l += dim("  " + std::string(kDot) + "  " + (c.present ? "completes its download" : "downloads " + gb(c.v->download_gb)));
    line(l, stderr);
    out = r.m->name + ":" + c.v->name;
  }
  planner.finish();  // the plan under way, before anything else plans
  return out;
}

void remember(const hub::ModelRef& r) {
  if (r.kind != hub::ModelRef::kCatalog || !r.model || !r.variant || !at_terminal()) return;
  ojson j = read_state();
  const std::string name = hub::full_name(*r.model, *r.variant);
  if (j.contains("model") && j["model"] == name) return;
  j["model"] = name;
  // written whole, then renamed over the old one; a home that can't be written keeps no choice
  mkdir(hub::home_dir().c_str(), 0755);
  const std::string path = state_path(), tmp = path + ".tmp." + std::to_string(getpid());
  {
    std::ofstream f(tmp);
    if (!f) return;
    f << j.dump(2) << "\n";
    if (!f) {
      f.close();
      unlink(tmp.c_str());
      return;
    }
  }
  if (rename(tmp.c_str(), path.c_str()) != 0) unlink(tmp.c_str());
}

std::string model_label(const std::string& id) {
  try {
    const hub::ModelRef r = hub::resolve(id);
    if (r.kind == hub::ModelRef::kCatalog && r.model && r.variant) {
      const std::string p = precision(*r.model, *r.variant);
      return r.model->name + (p.empty() ? "" : " " + std::string(kDot) + " " + p);
    }
  } catch (const std::exception&) {
  }
  return id;
}

}  // namespace ni::ui
