// Kapitsa updating itself (`kapitsa update` and the automatic check): which kind of install this binary is, the latest
// release, and installing one: downloaded into a folder beside the binary, checked (its sha256, its code signature,
// its --version) and switched in with one rename, the previous binary kept for a rollback. The command and the
// background check that use it are in src/server/update_cmd.cpp.
//
// It contacts only the releases page (github.com/nace-ai/kapitsa/releases, or KAPITSA_UPDATE_URL) and, for a
// download, the host GitHub keeps release files on (*.githubusercontent.com): no other redirect is followed. Requests
// carry no cookies and nothing about the Mac: a fixed User-Agent and Accept-Language.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ni::update {

// ---------------------------------------------------------------- versions

// A version, ordered by semantic versioning's precedence: 0.1.9 < 0.1.10 < 0.2.0-rc.1 < 0.2.0; build metadata (+...)
// is ignored. Takes "0.1.33" and "v0.1.33"; ok is false for anything else.
struct Version {
  bool ok = false;
  long major = 0, minor = 0, patch = 0;
  std::vector<std::string> pre;  // the prerelease's identifiers: "rc", "1"
  static Version parse(const std::string& s);
};
// <0, 0, >0 as a is older than, the same as, newer than b
int compare(const Version& a, const Version& b);
// A version as release names and URLs carry it: X.Y.Z, optionally -PRERELEASE (letters, digits, dots, hyphens)
bool valid_version(const std::string& s);

// ---------------------------------------------------------------- the install

enum class Kind {
  kStandalone,  // install.sh, or a copied binary: updated in place
  kHomebrew,    // in a Homebrew Cellar: brew upgrades it
  kApp,         // inside an app bundle: the app keeps its own Kapitsa
  kSource,      // in the build folder of a source tree: git pull and rebuild
};

struct Install {
  Kind kind = Kind::kStandalone;
  std::string path;     // the executable, resolved (symlinks followed)
  std::string dir;      // its folder
  std::string notices;  // kStandalone: where LICENSE, THIRD_PARTY_NOTICES and the rollback copy go (as install.sh)
  std::string app;      // kApp: the .app
  std::string brew;     // kHomebrew: that Homebrew's brew command
  bool head = false;    // kHomebrew: installed with --HEAD
  std::string source, build;  // kSource: the source tree and the build folder
  bool git = false;           // kSource: the source tree is a git work tree
  // Signed as release binaries are (identifier ai.nace.kapitsa). A binary built from source and copied here (./install.sh
  // in a clone) is not: the automatic update leaves it to its owner.
  bool release = false;
};

// This process's executable, resolved
std::string self_path();
// Which kind of install the executable at `path` (resolved) is; `release` is this process's own signature, so `path`
// should be self_path()
Install detect(const std::string& path);
// What updates a Homebrew install: "brew upgrade nace-ai/tap/kapitsa" (--fetch-HEAD for a --HEAD install)
std::string brew_command(const Install& in);
// What updates a source build: "git -C SRC pull && cmake --build BUILD" (without the pull outside a git work tree)
std::string source_command(const Install& in);
// The folder an update of a standalone install needs to write and can't, or ""
std::string unwritable(const Install& in);
// The rollback copy of a standalone install: NOTICES/kapitsa.previous
std::string previous_path(const Install& in);

// ---------------------------------------------------------------- releases

enum class Failure {
  kNone,
  kNetwork,  // no answer, an HTTP error, the time limit: tried again later (silently, in the background)
  kRefused,  // the release did not pass a check (sha256, signature, version): nothing was changed
  kLocal,    // this Mac: a folder that can't be written, a full disk
  kBusy,     // a kapitsa bench is running from this binary, which must not change under it
  kChanged,  // the file at the binary's path is no longer this kapitsa (something else replaced it)
};

// The releases page: KAPITSA_UPDATE_URL, else https://github.com/nace-ai/kapitsa/releases. "" with *why when the
// override is neither https nor http to this Mac (127.0.0.1, localhost, [::1]).
std::string releases_url(std::string* why);

struct Latest {
  std::string version;  // "0.1.33"
  Failure failure = Failure::kNone;
  std::string why;
};
// The latest release: where the releases page's /latest redirects (/tag/vX.Y.Z), as install.sh finds it (a HEAD
// request; no API, so no rate limit)
Latest latest();

// The stages of an install, for a spinner or a log line: "Downloading kapitsa-0.1.33-darwin-arm64.tar.gz", ...
using Stage = std::function<void(const std::string&)>;

struct Done {
  Failure failure = Failure::kNone;
  std::string why;
  std::string warning;  // done, but something beside it was not (the license files)
};
// Installs release `version` over the standalone kapitsa `in`, which must be this process's binary: downloads the
// tarball and its .sha256 into a new folder in the binary's own folder (so the switch is a rename on one volume) and
// checks the sha256; unpacks it and checks the new binary: a valid code signature, the release identifier, the same
// Developer ID team as this binary when it has one, and `--version` printing `version`; then renames it over the old
// one, after checking that the file there is still this kapitsa and that no kapitsa bench holds it. The old binary
// stays as the rollback copy (previous_path), the license files go beside it as install.sh puts them. A running
// process of the old binary keeps it (its inode) until it exits.
Done install(const Install& in, const std::string& version, const Stage& stage);
// Puts the rollback copy back (after checking its signature and --version) and keeps the binary it replaces as the
// next rollback copy. Its version in *version.
Done rollback(const Install& in, std::string* version);
// The version a binary reports (`PATH --version`), "" when it does not run or says something else
std::string version_of(const std::string& path);

// For kapitsa bench: no update replaces this binary until the process exits (a shared lock on it, which install()
// asks for exclusively). False when the file at this process's path is no longer its binary (just replaced).
bool hold_for_benchmark();

// For a time limit: true when no binary switch has begun, and then none will (the caller records why and calls
// exit_now()); false once one has begun (what is left of the update takes milliseconds: let it finish).
bool stop_before_switch();
// Ends the process at once (exit code `code`), after removing the temporary folder of an install under way. After
// stop_before_switch().
[[noreturn]] void exit_now(int code);

// ---------------------------------------------------------------- the state (KAPITSA_HOME/update.json)

// One binary's record, by its path
struct Record {
  long long checked = 0;   // the last check that ran or started (unix seconds)
  std::string latest;      // the latest release it found
  std::string result;      // "up to date", "updated", "available", "network", "refused", "error", "busy", "skipped"
  std::string from, to;    // "updated": the versions
  std::string why;         // what went wrong
  std::string skip;        // a release rolled back from: automatic updates leave it (kapitsa update installs it)
  bool notice = false;     // a line for this binary's next run in a terminal
};
struct State {
  bool automatic = true;  // kapitsa update --auto on|off
  std::map<std::string, Record> bins;
};
std::string state_path();
State read_state();
// Reads the state, lets `change` change it and writes it back, atomically, under a lock (records of binaries that are
// gone are dropped). False when it could not be written (no lock within a second, a folder that can't be written).
bool update_state(const std::function<void(State&)>& change);

}  // namespace ni::update
