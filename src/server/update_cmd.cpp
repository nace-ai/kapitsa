// kapitsa update, and the automatic update: the command; the check a command starts in the background once a day
// (`kapitsa update --background`, a detached low-priority child with its own time limit); and the line the next run in
// a terminal shows about it. The work itself (versions, kinds of install, downloading, checking, switching) is in
// src/update/update.h.
#include "server/cli.h"
#include "server/ui.h"
#include "update/update.h"
#include "util/log.h"
#include "util/term.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <ctime>
#include <thread>

extern char** environ;

namespace ni {

namespace {

using term::bold;
using term::dim;
using update::Failure;
using update::Install;
using update::Kind;

const long long kDay = 24 * 3600;
const char* const kHelpRef = "kapitsa update --help";

const char* kHelp =
    "kapitsa update [--check] [--version X.Y.Z] [--rollback] [--auto on|off]: updates Kapitsa to its latest release.\n\n"
    "  --check            only say whether a newer release is out (exit 0: up to date, 100: one is, 1: can't tell)\n"
    "  --version X.Y.Z    install that release instead, an older one too\n"
    "  --rollback         go back to the version the last update replaced\n"
    "  --auto on|off      updates by themselves, on by default (below)\n\n"
    "Installed with install.sh, or copied, kapitsa updates itself: it downloads the release from github.com, checks it\n"
    "against the release's sha256, checks its code signature and its version, and puts it in place in one step. The\n"
    "version it replaced is kept for --rollback, as kapitsa.previous beside the license files (../share/kapitsa from\n"
    "the binary's folder). A release that carries them also installs kapitsa-systemone (Drex decision models,\n"
    "docs/systemone.md) and kapitsa-drex-pointer beside the binary, the server only when it is signed as Kapitsa's.\n"
    "Running servers keep the old version until they restart.\n"
    "Installed with Homebrew, kapitsa update runs brew upgrade nace-ai/tap/kapitsa. In an app such as Nace Agent, the\n"
    "app updates it. Built from source, it says how: git pull, then cmake --build build.\n\n"
    "Once a day a kapitsa command checks for a new release in the background, without slowing the command down, and\n"
    "installs it where kapitsa updates itself; for Homebrew and source builds it only tells you, in a terminal. The\n"
    "next run in a terminal says what happened. The check asks github.com alone and sends nothing about this Mac.\n"
    "KAPITSA_NO_AUTO_UPDATE=1, kapitsa update --auto off or CI=true turn it off.\n";

bool in_ci() {
  const char* ci = getenv("CI");
  return ci && *ci && strcmp(ci, "0") != 0 && strcmp(ci, "false") != 0;
}

// What in the environment keeps automatic updates off ("" when nothing does)
std::string env_off() {
  if (env_on("KAPITSA_NO_AUTO_UPDATE")) return "KAPITSA_NO_AUTO_UPDATE";
  if (in_ci()) return "CI";
  return "";
}

// A person at a terminal: brew may run there
bool at_terminal() { return isatty(0) && isatty(1) && isatty(2) && !in_ci(); }

update::Version current() { return update::Version::parse(kVersion); }
// This kapitsa is older than release `v`
bool older_than(const std::string& v) {
  const update::Version x = update::Version::parse(v);
  return x.ok && update::compare(current(), x) < 0;
}

// "sudo /usr/local/bin/kapitsa update": an install in a folder only its owner can write
std::string sudo_command(const Install& in) { return "sudo " + in.path + " update"; }

// What updates this kapitsa to a release that is out, as a command: brew, a rebuild, sudo, or kapitsa update itself
std::string how(const Install& in) {
  switch (in.kind) {
    case Kind::kHomebrew: return update::brew_command(in);
    case Kind::kSource: return update::source_command(in);
    case Kind::kApp: return "update the app";
    case Kind::kStandalone: return update::unwritable(in).empty() ? "kapitsa update" : sudo_command(in);
  }
  return "kapitsa update";
}

// Automatic updates only tell this kapitsa about a release (they don't install it)
bool notify_only(const Install& in) {
  return in.kind != Kind::kStandalone || !in.release || !update::unwritable(in).empty();
}

void record(const std::string& path, const std::function<void(update::Record&)>& f) {
  update::update_state([&](update::State& s) { f(s.bins[path]); });
}

// A result line: styled on stdout in a terminal, else plain on stdout
void result(const std::string& styled, const std::string& plain) {
  if (term::fancy()) ui::ok(styled, true);
  else printf("%s\n", plain.c_str());
}
void info(const std::string& styled, const std::string& plain) {
  if (term::fancy()) ui::note(styled, true);
  else printf("%s\n", plain.c_str());
}

std::string app_message(const Install& in) {
  return "This kapitsa is part of " + hub::tilde(in.app) + ", which keeps its own Kapitsa up to date: update the app.";
}
std::string brew_message(const Install& in) {
  return "This kapitsa is Homebrew's, so Homebrew updates it.\n  " + update::brew_command(in);
}
std::string source_message(const Install& in) {
  return "This kapitsa is a build from source" + (in.source.empty() ? "" : " in " + hub::tilde(in.source)) +
         ": update the source and build it again.\n  " + update::source_command(in);
}

// A failed install or rollback, as an error message
std::string failure_message(const update::Done& d, const Install& in, const std::string& what) {
  switch (d.failure) {
    case Failure::kNetwork: return "Can't download " + what + ": " + d.why + ". Nothing was changed.";
    case Failure::kRefused: return what + " was not installed: " + d.why + ". Nothing was changed.";
    case Failure::kBusy: return "Not now: " + d.why + ", and its binary must not change under it. Update after it ends.";
    case Failure::kChanged: return d.why + ": nothing was replaced. Run the kapitsa there to update it.";
    case Failure::kLocal:
      if (!update::unwritable(in).empty())
        return "Can't write " + hub::tilde(update::unwritable(in)) + ", where this kapitsa is. Update it as the folder's "
               "owner:\n  " + sudo_command(in);
      return d.why + ".";
    case Failure::kNone: break;
  }
  return d.why;
}

// ---------------------------------------------------------------- kapitsa update --check

int check(const Install& in) {
  if (in.kind == Kind::kApp) return fail(app_message(in));
  std::unique_ptr<term::Spinner> sp;
  if (term::live()) sp = std::make_unique<term::Spinner>("Looking for a new release");
  const update::Latest L = update::latest();
  if (sp) sp->done();
  if (L.failure != Failure::kNone) return fail("Can't check for a new release: " + L.why + ".");
  const int cmp = update::compare(current(), update::Version::parse(L.version));
  int code = 0;
  if (cmp == 0) {
    result("Kapitsa " + bold(kVersion) + " is up to date " + dim("(the latest release)"),
           std::string("kapitsa ") + kVersion + " is up to date (the latest release).");
  } else if (cmp > 0) {
    result("Kapitsa " + bold(kVersion) + " is up to date " + dim("(newer than the latest release, " + L.version + ")"),
           std::string("kapitsa ") + kVersion + " is up to date (newer than the latest release, " + L.version + ").");
  } else {
    code = 100;
    if (term::fancy()) {
      ui::ok("Kapitsa " + bold(L.version) + " is available " + dim(std::string("(this is ") + kVersion + ")"), true);
      ui::note(how(in) + (in.kind == Kind::kStandalone && update::unwritable(in).empty() ? " installs it" : " updates it"), true);
    } else {
      printf("kapitsa %s is available (this is %s). Update: %s\n", L.version.c_str(), kVersion, how(in).c_str());
    }
  }
  // what happens by itself
  std::string automatic;
  if (const std::string off = env_off(); !off.empty()) automatic = "Automatic updates are off (" + off + " is set).";
  else if (!update::read_state().automatic) automatic = "Automatic updates are off (kapitsa update --auto on turns them on).";
  else if (notify_only(in)) automatic = "Automatic updates: a run in a terminal says when a release is out.";
  else automatic = "Automatic updates are on: a new release installs itself within a day.";
  info(automatic, automatic);
  return code;
}

// ---------------------------------------------------------------- kapitsa update [--version X]

// Runs brew upgrade for a Homebrew install, in the terminal: its exit code
int run_brew(const Install& in) {
  if (access(in.brew.c_str(), X_OK) != 0)
    return fail("This kapitsa is Homebrew's, and there is no " + hub::tilde(in.brew) + " to update it with. Update it with "
                "your brew:\n  " + update::brew_command(in));
  std::vector<std::string> args = {in.brew, "upgrade"};
  if (in.head) args.push_back("--fetch-HEAD");
  args.push_back("nace-ai/tap/kapitsa");
  info("This kapitsa is Homebrew's: " + bold(update::brew_command(in)), "This kapitsa is Homebrew's: " + update::brew_command(in));
  fflush(stdout);
  std::vector<char*> a;
  for (auto& s : args) a.push_back(s.data());
  a.push_back(nullptr);
  // Ctrl+C is brew's to handle while it runs: ignored here, its default there
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  sigset_t def;
  sigemptyset(&def);
  sigaddset(&def, SIGINT), sigaddset(&def, SIGQUIT);
  posix_spawnattr_setsigdefault(&at, &def);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGDEF);
  void (*old_int)(int) = signal(SIGINT, SIG_IGN);
  void (*old_quit)(int) = signal(SIGQUIT, SIG_IGN);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, a[0], nullptr, &at, a.data(), environ);
  posix_spawnattr_destroy(&at);
  int st = 0;
  if (rc == 0)
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
  signal(SIGINT, old_int);
  signal(SIGQUIT, old_quit);
  if (rc != 0) return fail("Can't run " + in.brew + " (" + strerror(rc) + ").");
  return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

// Installs release `version` (explicit: any release; else the latest, when newer) over a standalone kapitsa
int install(const Install& in, const std::string& version, bool asked) {
  const std::string was = kVersion;
  if (const std::string w = update::unwritable(in); !w.empty())
    return fail("Can't write " + hub::tilde(w) + ", where this kapitsa is. Update it as the folder's owner:\n  " +
                sudo_command(in));
  std::unique_ptr<term::Spinner> sp;
  if (term::live()) sp = std::make_unique<term::Spinner>("Updating to " + bold(version));
  const update::Done d = update::install(in, version, [&](const std::string& stage) {
    if (sp) sp->label(stage);
    else fprintf(stderr, "%s...\n", stage.c_str());
  });
  if (sp) sp->done();
  const long long now = time(nullptr);
  if (d.failure != Failure::kNone) {
    record(in.path, [&](update::Record& r) {
      r.checked = now;
      r.result = d.failure == Failure::kNetwork ? "network" : d.failure == Failure::kRefused ? "refused" : "error";
      r.why = d.why, r.notice = false;
    });
    return fail(failure_message(d, in, "Kapitsa " + version));
  }
  // moving to an older release (asked for): automatic updates leave the one moved away from
  const bool down = update::compare(update::Version::parse(version), current()) < 0;
  record(in.path, [&](update::Record& r) {
    r.checked = now, r.result = "updated", r.from = was, r.to = version, r.why = d.warning, r.notice = false;
    if (!asked) r.latest = version;
    r.skip = down ? was : "";
  });
  const std::string prev = hub::tilde(update::previous_path(in));
  if (term::fancy()) {
    ui::ok("Updated to " + bold(version) + dim(" (from " + was + ")  " + hub::tilde(in.path)), true);
    ui::note("Running servers keep " + was + " until they restart  " + term::kDot + "  kapitsa update --rollback goes back", true);
    if (!d.warning.empty()) ui::warning(d.warning);
  } else {
    printf("Updated kapitsa %s -> %s: %s\n", was.c_str(), version.c_str(), in.path.c_str());
    printf("Running servers keep %s until they restart. The previous version is kept in %s: kapitsa update --rollback "
           "goes back to it.\n", was.c_str(), prev.c_str());
    if (!d.warning.empty()) fprintf(stderr, "kapitsa: %s\n", d.warning.c_str());
  }
  return 0;
}

int update_now(const Install& in, const std::string& version) {
  if (in.kind == Kind::kApp) return fail(app_message(in));
  if (!version.empty() && in.kind != Kind::kStandalone)
    return fail(std::string("Only a kapitsa installed with install.sh (or copied) installs a given release. ") +
                (in.kind == Kind::kHomebrew ? "Homebrew installs the release its formula names:\n  " + update::brew_command(in)
                                            : "Check out that release's tag and build it:\n  " + update::source_command(in)));
  if (!version.empty()) {
    if (!update::valid_version(version)) usage_error("--version needs a release, e.g. " + std::string(kVersion), kHelpRef);
    if (version == kVersion) {
      result("Kapitsa " + bold(version) + " is already installed", "kapitsa " + version + " is already installed.");
      return 0;
    }
    return install(in, version, true);
  }
  // a --HEAD install follows main: brew knows whether there is something newer
  if (in.kind == Kind::kHomebrew && in.head) return at_terminal() ? run_brew(in) : fail(brew_message(in));
  std::unique_ptr<term::Spinner> sp;
  if (term::live()) sp = std::make_unique<term::Spinner>("Looking for a new release");
  const update::Latest L = update::latest();
  if (sp) sp->done();
  if (L.failure != Failure::kNone) return fail("Can't check for a new release: " + L.why + ".");
  if (!older_than(L.version)) {
    const bool newer = update::compare(current(), update::Version::parse(L.version)) > 0;
    const std::string what = newer ? "newer than the latest release, " + L.version : "the latest release";
    result("Kapitsa " + bold(kVersion) + " is up to date " + dim("(" + what + ")"),
           std::string("kapitsa ") + kVersion + " is up to date (" + what + ").");
    if (in.kind == Kind::kSource)
      info("A build from source follows main: " + update::source_command(in),
           "A build from source follows main: " + update::source_command(in));
    if (in.kind == Kind::kStandalone)
      record(in.path, [&](update::Record& r) {
        r.checked = time(nullptr), r.result = "up to date", r.latest = L.version, r.why.clear(), r.notice = false;
      });
    return 0;
  }
  switch (in.kind) {
    case Kind::kHomebrew: return at_terminal() ? run_brew(in) : fail(brew_message(in));
    case Kind::kSource: return fail("Kapitsa " + L.version + " is out. " + source_message(in));
    case Kind::kApp: return fail(app_message(in));
    case Kind::kStandalone: break;
  }
  return install(in, L.version, false);
}

// ---------------------------------------------------------------- kapitsa update --rollback

int roll_back(const Install& in) {
  if (in.kind == Kind::kApp) return fail(app_message(in));
  if (in.kind == Kind::kHomebrew) return fail("This kapitsa is Homebrew's, which keeps no previous version to go back to.");
  if (in.kind == Kind::kSource) return fail("This kapitsa is a build from source: check out the commit you want and build it.");
  std::string v;
  const update::Done d = update::rollback(in, &v);
  if (d.failure != Failure::kNone) {
    if (d.failure == Failure::kLocal && update::unwritable(in).empty())
      return fail(d.why + ". kapitsa update --version X.Y.Z installs a given release.");
    return fail(failure_message(d, in, "The previous version"));
  }
  const std::string was = kVersion;
  // automatic updates leave the release rolled back from (and older ones); a newer one installs as usual
  const bool back = update::compare(update::Version::parse(v), current()) < 0;
  record(in.path, [&](update::Record& r) {
    r.checked = time(nullptr), r.result = "rolled back", r.from = was, r.to = v, r.why.clear(), r.notice = false;
    r.skip = back ? was : "";
  });
  if (term::fancy()) {
    ui::ok("Back to " + bold(v) + dim(" (from " + was + ")  " + hub::tilde(in.path)), true);
    ui::note("Running servers keep " + was + " until they restart" +
             (back ? "  " + std::string(term::kDot) + "  automatic updates skip " + was : std::string()), true);
  } else {
    printf("Rolled kapitsa back %s -> %s: %s\n", was.c_str(), v.c_str(), in.path.c_str());
    printf("Running servers keep %s until they restart.%s kapitsa update --rollback again returns to %s.\n", was.c_str(),
           back ? (" Automatic updates skip " + was + " from now on.").c_str() : "", was.c_str());
  }
  return 0;
}

// ---------------------------------------------------------------- kapitsa update --auto on|off

int set_auto(const Install& in, bool on) {
  if (!update::update_state([&](update::State& s) { s.automatic = on; }))
    return fail("Can't write " + hub::tilde(update::state_path()) + ".");
  const std::string what = on ? "Automatic updates are on" : "Automatic updates are off";
  result(what, what + ".");
  std::string more;
  if (const std::string off = env_off(); on && !off.empty()) more = off + " is set, which keeps them off here.";
  else if (!on) more = "kapitsa update updates by hand; kapitsa update --auto on turns them on again.";
  else if (in.kind == Kind::kApp) more = "This kapitsa is part of an app, which updates it.";
  else if (notify_only(in)) more = "For this kapitsa a run in a terminal says when a release is out: " + how(in) + ".";
  else more = "A new release installs itself within a day; the next run in a terminal says so.";
  info(more, more);
  return 0;
}

// ---------------------------------------------------------------- the background check (kapitsa update --background)

int background() {
  // below everything else on this Mac: CPU, disk and network
  setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG);
  const Install in = update::detect(update::self_path());
  if (in.kind == Kind::kApp || !env_off().empty()) return 0;
  // its own time limit: a check that hangs ends like a network failure (tried again the next day); once the binary is
  // being switched, the update finishes (milliseconds)
  double limit = 300;
  if (const char* t = getenv("KAPITSA_UPDATE_TIMEOUT_S"); t && atof(t) > 0) limit = atof(t);
  std::thread([limit, path = in.path] {
    std::this_thread::sleep_for(std::chrono::duration<double>(limit));
    if (!update::stop_before_switch()) return;
    record(path, [&](update::Record& r) {
      r.result = "network", r.why = "the time limit (" + std::to_string(int(limit)) + " s)", r.notice = false;
    });
    update::exit_now(124);
  }).detach();
  // one check at a time
  const int fd = open((hub::home_dir() + "/update.run.lock").c_str(), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0) return 0;

  const update::Latest L = update::latest();
  if (L.failure == Failure::kNetwork) {  // silent: tried again the next day
    record(in.path, [&](update::Record& r) { r.result = "network", r.why = L.why, r.notice = false; });
    return 0;
  }
  if (L.failure != Failure::kNone) {  // a setting that can't work (KAPITSA_UPDATE_URL): said once
    record(in.path, [&](update::Record& r) { r.result = "error", r.why = L.why, r.latest.clear(), r.notice = true; });
    return 0;
  }
  if (!older_than(L.version)) {
    record(in.path, [&](update::Record& r) { r.result = "up to date", r.latest = L.version, r.why.clear(), r.notice = false; });
    return 0;
  }
  const update::State st = update::read_state();
  const auto it = st.bins.find(in.path);
  const std::string skip = it == st.bins.end() ? "" : it->second.skip;
  if (!skip.empty() && update::compare(update::Version::parse(L.version), update::Version::parse(skip)) <= 0) {
    record(in.path, [&](update::Record& r) { r.result = "skipped", r.latest = L.version, r.notice = false; });
    return 0;
  }
  if (notify_only(in)) {
    record(in.path, [&](update::Record& r) { r.result = "available", r.latest = L.version, r.why.clear(), r.notice = true; });
    return 0;
  }
  const update::Done d = update::install(in, L.version, nullptr);
  const long long now = time(nullptr);
  record(in.path, [&](update::Record& r) {
    r.latest = L.version, r.why = d.why;
    switch (d.failure) {
      case Failure::kNone:
        r.result = "updated", r.from = kVersion, r.to = L.version, r.why = d.warning, r.skip.clear(), r.notice = true;
        break;
      case Failure::kNetwork: r.result = "network", r.notice = false; break;
      case Failure::kBusy: r.result = "busy", r.checked = now - kDay + 3600, r.notice = false; break;  // again in an hour
      case Failure::kRefused: r.result = "refused", r.notice = true; break;
      case Failure::kLocal:
      case Failure::kChanged: r.result = "error", r.notice = true; break;
    }
  });
  return 0;
}

// Starts the background check: this binary, detached (a session of its own, no terminal, /dev/null for its streams);
// reaped by a thread when this process outlives it
void spawn_background(const std::string& self) {
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  sigset_t none, all;
  sigemptyset(&none);
  sigfillset(&all);
  posix_spawnattr_setsigmask(&at, &none);
  posix_spawnattr_setsigdefault(&at, &all);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
  const char* args[] = {self.c_str(), "update", "--background", nullptr};
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, self.c_str(), &fa, &at, const_cast<char* const*>(args), environ);
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  if (rc == 0)
    std::thread([pid] {
      int st;
      while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
      }
    }).detach();
}

// The line about the last check, for a run in a terminal; "" when there is nothing (left) to say
void show_notice(const update::Record& r, const Install& in) {
  if (r.result == "updated") {
    if (r.to != kVersion) return;  // this binary is not the one installed then
    ui::ok("Updated Kapitsa to " + bold(r.to) + dim(" (from " + r.from + ")") + "; restart running servers to use it");
  } else if (r.result == "available") {
    if (!older_than(r.latest)) return;
    // the command whole (a note wraps long text at any character): on a line of its own when it does not fit
    const std::string head = "Kapitsa " + r.latest + " is available:", cmd = how(in);
    if (term::width(head + " " + cmd) <= std::max(30, std::min(term::columns(), 100) - 5)) return ui::note(head + " " + cmd);
    ui::note(head);
    term::line("    " + dim(cmd), stderr);
  } else if (r.result == "refused" || r.result == "error") {
    if (!r.latest.empty() && !older_than(r.latest)) return;
    ui::warning((r.latest.empty() ? std::string("Kapitsa can't check for updates") : "Kapitsa " + r.latest + " was not installed") +
                ": " + r.why + (r.latest.empty() ? "" : "; kapitsa update tries again"));
  }
}

}  // namespace

int run_update(int argc, char** argv, int first) {
  bool check_only = false, roll = false, bg = false;
  std::string version, autov;
  for (int i = first; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* opt) -> std::string {
      const std::string o = opt;
      if (a == o) {
        if (i + 1 >= argc) usage_error(o + " needs a value", kHelpRef);
        return argv[++i];
      }
      return a.substr(o.size() + 1);  // --opt=value
    };
    if (a == "--help" || a == "-h") {
      if (term::fancy()) ui::help(kHelp);
      else fputs(kHelp, stdout);
      return 0;
    } else if (a == "--check") check_only = true;
    else if (a == "--rollback") roll = true;
    else if (a == "--version" || a.rfind("--version=", 0) == 0) version = value("--version");
    else if (a == "--auto" || a.rfind("--auto=", 0) == 0) autov = value("--auto");
    else if (a == "--background") bg = true;  // the automatic check's child (not for people)
    else usage_error("unknown option " + a, kHelpRef);
  }
  if (bg) return background();
  if (int(check_only) + int(roll) + int(!autov.empty()) + int(!version.empty()) > 1)
    usage_error("--check, --version, --rollback and --auto go one at a time", kHelpRef);
  if (!autov.empty() && autov != "on" && autov != "off") usage_error("--auto takes on or off, not \"" + autov + "\"", kHelpRef);
  if (!version.empty() && version[0] == 'v') version.erase(0, 1);
  if (!version.empty() && !update::valid_version(version))
    usage_error("--version needs a release, e.g. " + std::string(kVersion) + ", not \"" + version + "\"", kHelpRef);
  const Install in = update::detect(update::self_path());
  if (!autov.empty()) return set_auto(in, autov == "on");
  if (check_only) return check(in);
  if (roll) return roll_back(in);
  return update_now(in, version);
}

void auto_update(int argc, char** argv) {
  // a command, not help, the version or the update itself; never a benchmark (its numbers come from one binary, with
  // nothing else running)
  if (argc < 2) return;
  const std::string cmd = argv[1];
  bool runs = cmd.size() > 1 && cmd[0] == '-' && cmd != "--help" && cmd != "-h" && cmd != "--version" && cmd != "-v";
  for (const char* c : {"serve", "pull", "list", "ls", "rm", "import", "doctor", "claude", "codex", "opencode", "pi", "mtp-sidecar"})
    runs = runs || cmd == c;
  if (!runs) return;
  for (int i = 2; i < argc; ++i)
    if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h") || !strcmp(argv[i], "--version")) return;
  if (geteuid() == 0 || !env_off().empty()) return;
  const std::string self = update::self_path();
  if (self.empty()) return;
  const Install in = update::detect(self);
  if (in.kind == Kind::kApp) return;
  const update::State st = update::read_state();
  if (!st.automatic) return;
  const auto it = st.bins.find(in.path);
  const update::Record rec = it == st.bins.end() ? update::Record{} : it->second;
  const long long now = time(nullptr);
  const bool tty = term::fancy();
  // a check is due once a day (or when the clock went back); where it would only tell, only for a run in a terminal
  auto due = [&](const update::Record& r) { return r.checked <= 0 || now - r.checked >= kDay || r.checked > now + 3600; };
  const bool want_check = (tty || !notify_only(in)) && due(rec);
  const bool want_notice = tty && rec.notice;
  if (!want_check && !want_notice) return;
  bool spawn = false;
  update::Record shown;
  update::update_state([&](update::State& s) {
    update::Record& r = s.bins[in.path];
    if (want_notice && r.notice) shown = r, r.notice = false;
    if (want_check && due(r)) r.checked = now, spawn = true;  // claimed: no other run starts one today
  });
  if (shown.notice) show_notice(shown, in);
  if (spawn) spawn_background(self);
}

}  // namespace ni
