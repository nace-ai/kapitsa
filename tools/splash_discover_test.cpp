// src/splash/discover.h on a throwaway Splash tree: what is found, and what each error names. CPU only, no Splash needed.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "splash/discover.h"

namespace fs = std::filesystem;
using namespace ni::splash;

static int failures = 0;
static void check(bool ok, const std::string& what, const std::string& detail = "") {
  if (!ok) {
    ++failures;
    printf("FAIL: %s %s\n", what.c_str(), detail.c_str());
  }
}
static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

static void touch(const fs::path& p, const std::string& body = "x") {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << body;
}

static void exe(const fs::path& p) {
  touch(p, "#!/bin/sh\n");
  fs::permissions(p, fs::perms::owner_all, fs::perm_options::add);
}

static void model(const std::string& home, const std::string& name, const std::string& family, bool tokenizer = true,
                  const std::string& source = "") {
  fs::path d = fs::path(home) / "Library/Application Support/Splash/models/.resolved" / name;
  touch(d / "model.json", "{\"family\":\"" + family + "\"" + (source.empty() ? "" : ",\"model\":\"" + source + "\"") + "}");
  if (tokenizer) touch(d / "tokenizer/tokenizer.json", "{}");
}

// Splash's "resolved <hours_ago> hours ago": ../.selections/<hash> -> .resolved/<name>, a symlink whose own time is that
static void selected(const std::string& home, const std::string& name, int hours_ago, const std::string& tag = "") {
  fs::path sel = fs::path(home) / "Library/Application Support/Splash/models/.selections";
  fs::create_directories(sel);
  fs::create_directory_symlink(fs::path(home) / "Library/Application Support/Splash/models/.resolved" / name, sel / ("sel" + tag + "-" + name));
  timespec t[2];
  t[0].tv_sec = t[1].tv_sec = time(nullptr) - hours_ago * 3600;
  t[0].tv_nsec = t[1].tv_nsec = 0;
  utimensat(AT_FDCWD, (sel / ("sel" + tag + "-" + name)).c_str(), t, AT_SYMLINK_NOFOLLOW);
}

int main() {
  fs::path tmp = fs::temp_directory_path() / ("splash_discover_" + std::to_string(getpid()));
  fs::remove_all(tmp);
  const std::string home = (tmp / "home").string();
  const std::vector<std::string> none;  // no system-wide root: the Mac the test runs on must not decide the result

  // nothing installed: every place looked at is named, and the flag that replaces them
  Found b = find_binary(home, "", "", none);
  check(!b.ok() && has(b.error, home + "/Applications/Splash/engine/splash") && has(b.error, "--splash-binary") && has(b.error, "SPLASH_HOME"), "binary: nothing", b.error);
  Found m = find_model(home, "");
  check(!m.ok() && has(m.error, "no Splash model found") && has(m.error, "--splash-model"), "model: nothing", m.error);

  // SPLASH_HOME, then PATH (through a symlink), then the default folders
  exe(tmp / "root/engine/splash");
  check(find_binary(home, (tmp / "root").string(), "").path == (tmp / "root/engine/splash").string(), "binary: SPLASH_HOME");
  exe(tmp / "other/engine/splash");
  fs::create_directories(tmp / "bin");
  fs::create_symlink(tmp / "other/engine/splash", tmp / "bin/splash");
  check(find_binary(home, "", (tmp / "bin").string(), none).path == fs::canonical(tmp / "other/engine/splash").string(), "binary: PATH symlink");
  check(find_binary(home, (tmp / "root").string(), (tmp / "bin").string()).path == (tmp / "root/engine/splash").string(), "binary: SPLASH_HOME first");
  // a file that cannot be run is not the engine
  touch(tmp / "plain/engine/splash");
  fs::permissions(tmp / "plain/engine/splash", fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::remove);
  Found np = find_binary(home, (tmp / "plain").string(), "");
  check(!np.ok() && has(np.error, "not an executable file"), "binary: not executable", np.error);
  // a SPLASH_HOME that is wrong is an error naming it, not another install found on PATH or under ~/Applications
  exe(fs::path(home) / "Applications/Splash/engine/splash");
  Found w = find_binary(home, (tmp / "nope").string(), (tmp / "bin").string());
  check(!w.ok() && has(w.error, (tmp / "nope/engine/splash").string()) && !has(w.error, "Applications"), "binary: SPLASH_HOME wrong", w.error);
  exe(fs::path(home) / "Applications/Splash/engine/splash");
  check(find_binary(home, "", "", none).ok(), "binary: ~/Applications/Splash");
  // a system-wide install is looked at only where the caller says (the test never sees the real /Applications)
  fs::remove(fs::path(home) / "Applications/Splash/engine/splash");
  exe(tmp / "sys/engine/splash");
  check(!find_binary(home, "", "", none).ok() && find_binary(home, "", "", {(tmp / "sys").string()}).path == (tmp / "sys/engine/splash").string(), "binary: system root");

  // models: one is the answer; a folder without a tokenizer is not one
  model(home, "aaaaaa1111", "Qwen3.8-Flash-Next");
  model(home, "zzzzzz9999", "Qwen3.8-27B", false);
  m = find_model(home, "");
  check(m.ok() && has(m.path, "/.resolved/aaaaaa1111"), "model: the only one", m.error);
  // several, no name: the most recently resolved, and it says which and how to change it; a prefix picks another
  model(home, "aaaaaa2222", "Qwen3.8-Flash-Next");
  model(home, "bbbbbb3333", "Qwen3.6-35B-A3B");
  m = find_model(home, "");
  selected(home, "aaaaaa1111", 30);
  selected(home, "aaaaaa2222", 20);
  selected(home, "bbbbbb3333", 10);
  m = find_model(home, "");
  check(m.ok() && has(m.path, "/bbbbbb3333") && has(m.note, "Qwen3.6-35B-A3B bbbbbb3333") && has(m.note, "most recently resolved of 3") &&
            has(m.note, "--splash-model") && !has(m.note, "zzzzzz9999"),
        "model: several, no flag: the latest resolved, and it says so", m.error + m.note);
  fs::remove(fs::path(home) / "Library/Application Support/Splash/models/.selections/sel-bbbbbb3333");
  selected(home, "bbbbbb3333", 50);
  m = find_model(home, "");
  check(m.ok() && has(m.path, "/aaaaaa2222"), "model: several: a later selection wins, not folder names or folder times", m.path);
  // two links to one folder (Splash hashes revision, language_only and draft_model): its latest counts, in any directory order
  selected(home, "aaaaaa2222", 100, "-old");
  for (int i = 0; i < 3; i++) {
    m = find_model(home, "");
    check(m.ok() && has(m.path, "/aaaaaa2222"), "model: several links to one folder: the latest link counts", m.path);
  }
  // a folder with no selection link falls back to its own time; all equal: the first by name, every time
  fs::remove_all(fs::path(home) / "Library/Application Support/Splash/models/.selections");
  m = find_model(home, "");
  check(m.ok() && !m.note.empty(), "model: several, no selections: still a pick", m.error);
  // the usual install: `splash serve --model OWNER/REPO:VARIANT` links models/OWNER/REPO:VARIANT, not .selections/<hash>
  {
    const fs::path models = fs::path(home) / "Library/Application Support/Splash/models";
    fs::create_directories(models / "org");
    fs::create_directory_symlink(models / ".resolved/aaaaaa1111", models / "org/Repo:Q2_0");
    timespec t[2];
    t[0].tv_sec = t[1].tv_sec = time(nullptr) + 3600;  // newer than every folder
    t[0].tv_nsec = t[1].tv_nsec = 0;
    utimensat(AT_FDCWD, (models / "org/Repo:Q2_0").c_str(), t, AT_SYMLINK_NOFOLLOW);
    m = find_model(home, "");
    check(m.ok() && has(m.path, "/aaaaaa1111"), "model: several: a plain OWNER/REPO link counts as a selection", m.path + m.error);
    fs::remove_all(models / "org");
  }
  fs::remove_all(fs::path(home) / "Library/Application Support/Splash/models/.selections");
  selected(home, "aaaaaa1111", 30);
  selected(home, "aaaaaa2222", 20);
  selected(home, "bbbbbb3333", 10);
  m = find_model(home, "bbbbbb");
  check(m.ok() && has(m.path, "/bbbbbb3333"), "model: prefix", m.error);
  m = find_model(home, "aaaaaa");
  check(!m.ok() && has(m.error, "matches 2"), "model: ambiguous prefix", m.error);
  m = find_model(home, "aaaaaa1");
  check(m.ok() && has(m.path, "/aaaaaa1111"), "model: longer prefix", m.error);
  m = find_model(home, "qq");
  check(!m.ok() && has(m.error, "family or source name"), "model: short and no family", m.error);
  m = find_model(home, "cccccc");
  check(!m.ok() && has(m.error, "starting with it"), "model: no match", m.error);
  check(list_models(home).size() == 3, "list_models");

  // a family or source name, any case: one model, or the newest of one family and source; else the candidates
  m = find_model(home, "35b");
  check(m.ok() && has(m.path, "/bbbbbb3333"), "family: one model", m.error);
  m = find_model(home, "FLASH-next");
  check(!m.ok() && has(m.error, "2 different models") && has(m.error, "aaaaaa1111") && has(m.error, "aaaaaa2222"), "family: no source, several", m.error);
  const fs::path res = fs::path(home) / "Library/Application Support/Splash/models/.resolved";
  model(home, "dddddd4444", "Qwen3.8-Flash-Next", true, "org/Flash-Next-GGUF:Q2_0");
  model(home, "eeeeee5555", "Qwen3.8-Flash-Next", true, "org/Flash-Next-GGUF:Q2_0");
  model(home, "ffffff6666", "Qwen3.8-Flash-Next", true, "org/Other-GGUF:Q4_0");
  fs::remove_all(res / "aaaaaa1111");
  fs::remove_all(res / "aaaaaa2222");
  const auto now = fs::file_time_type::clock::now();
  fs::last_write_time(res / "dddddd4444", now - std::chrono::hours(2));
  fs::last_write_time(res / "eeeeee5555", now - std::chrono::hours(1));
  m = find_model(home, "q2_0");
  check(m.ok() && has(m.path, "/eeeeee5555"), "source: the newest of one source", m.error);
  selected(home, "dddddd4444", 0);  // Splash resolved it again just now: that beats the folder's older time
  m = find_model(home, "q2_0");
  check(m.ok() && has(m.path, "/dddddd4444"), "source: a later selection beats the folder time", m.error);
  m = find_model(home, "flash-next");
  check(!m.ok() && has(m.error, "3 different models") && has(m.error, "Other-GGUF"), "family: two sources", m.error);
  m = find_model(home, "other-gguf");
  check(m.ok() && has(m.path, "/ffffff6666"), "source: a part of the source name", m.error);
  m = find_model(home, "nothing-like-it");
  check(!m.ok() && has(m.error, "family or source name") && has(m.error, "Qwen3.6-35B-A3B  bbbbbb3333"), "family: no match", m.error);

  // the last run's binary and model: tried after the rest, and the way out of "several"
  const std::string lastf = (tmp / "kapitsa-home/splash.json").string();
  check(read_last(lastf).binary.empty(), "last: missing file");
  fs::remove_all(fs::path(home) / "Applications");
  exe(tmp / "installed/engine/splash");
  const std::string lbin = (tmp / "installed/engine/splash").string();
  check(!find_binary(home, "", "", none).ok() && find_binary(home, "", "", none, lbin).path == lbin, "last: binary");
  check(find_binary(home, (tmp / "root").string(), "", none, lbin).path == (tmp / "root/engine/splash").string(), "last: SPLASH_HOME first");
  Found lost = find_binary(home, "", "", none, (tmp / "gone/engine/splash").string());
  check(!lost.ok() && has(lost.error, "gone/engine/splash  (the last run's)"), "last: binary gone", lost.error);
  write_last(lastf, {lbin, res.string() + "/ffffff6666"});
  Last got = read_last(lastf);
  check(got.binary == lbin && got.model == res.string() + "/ffffff6666", "last: round trip");
  write_last(lastf, {"", ""});
  check(read_last(lastf).binary == lbin, "last: empty write keeps it");
  write_last(lastf, {"", res.string() + "/aaaaaa1111"});  // only one field known: the other stays
  got = read_last(lastf);
  check(got.binary == lbin && got.model == res.string() + "/aaaaaa1111", "last: a partial write keeps the other field");
  touch(fs::path(lastf), "not json");
  check(read_last(lastf).binary.empty() && read_last(lastf).model.empty(), "last: bad file");
  m = find_model(home, "", res.string() + "/ffffff6666");
  check(m.ok() && has(m.path, "/ffffff6666"), "last: model among several", m.error);
  m = find_model(home, "", res.string() + "/gone");
  check(m.ok() && has(m.note, "most recently resolved"), "last: model gone: the most recently resolved instead", m.error + m.note);
  m = find_model(home, "", res.string() + "/dddddd4444");
  check(m.ok() && has(m.path, "/dddddd4444") && has(m.note, "the one the last run used"), "last: model beats newest", m.error + m.note);
  m = find_model(home, "other-gguf", res.string() + "/eeeeee5555");
  check(m.ok() && has(m.path, "/ffffff6666"), "last: a name beats it", m.error);

  // --check: what is found, what is missing, the fix; nothing is started
  Check c = check(home, "", "", "", "", {}, none);
  check(!c.ok && has(c.text, "Splash engine: NOT FOUND") && has(c.text, "SPLASH_HOME") && has(c.text, "Model to run: ") &&
            has(c.text, "Splash model: ") && !has(c.text, "NOT DECIDED"),
        "check: nothing found, several models: it still names the pick", c.text);
  c = check(home, "", "", "", "no-such-model", {}, none);
  check(!c.ok && has(c.text, "NOT DECIDED"), "check: a name that matches nothing", c.text);
  c = check(home, (tmp / "root").string(), "", "", "other-gguf", {}, none);
  check(c.ok && has(c.text, "Splash engine: " + (tmp / "root/engine/splash").string()) && has(c.text, "Model to run: ") && has(c.text, "/ffffff6666") &&
            has(c.text, "Ready: run the same command without --check"),
        "check: ready", c.text);
  c = check(home, "", "", (tmp / "nope").string(), "", {}, none);
  check(!c.ok && has(c.text, "--splash-binary") && has(c.text, "no such file"), "check: bad --splash-binary", c.text);
  touch(tmp / "plain/splash", "x");  // a file, not executable
  c = check(home, "", "", (tmp / "plain/splash").string(), "", {}, none);
  check(!c.ok && has(c.text, "not an executable file") && !has(c.text, "Ready"), "check: --splash-binary that is not executable", c.text);
  c = check(home, (tmp / "root").string(), "", "", (tmp / "no/such/folder").string(), {}, none);  // a path that is no folder is a name: no match, not Ready
  check(!c.ok && has(c.text, "NOT DECIDED") && !has(c.text, "Ready"), "check: --splash-model path that is not a folder", c.text);
  check(!explicit_binary((tmp / "nope").string()).ok() && !explicit_binary((tmp / "plain/splash").string()).ok() && explicit_binary(lbin).ok(), "explicit binary: same rule as check");
  c = check(home, "", "", lbin, "", {lbin, res.string() + "/ffffff6666"}, none);
  check(c.ok && has(c.text, "/ffffff6666"), "check: the last run's model", c.text);

  fs::remove_all(tmp);
  if (failures) return 1;
  printf("splash_discover_test: ok\n");
}
