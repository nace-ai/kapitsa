// src/splash/discover.h on a throwaway Splash tree: what is found, and what each error names. CPU only, no Splash needed.
#include <unistd.h>

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

static void model(const std::string& home, const std::string& name, const std::string& family, bool tokenizer = true) {
  fs::path d = fs::path(home) / "Library/Application Support/Splash/models/.resolved" / name;
  touch(d / "model.json", "{\"family\":\"" + family + "\"}");
  if (tokenizer) touch(d / "tokenizer/tokenizer.json", "{}");
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
  // several: an error that lists them, and a prefix picks one
  model(home, "aaaaaa2222", "Qwen3.8-Flash-Next");
  model(home, "bbbbbb3333", "Qwen3.6-35B-A3B");
  m = find_model(home, "");
  check(!m.ok() && has(m.error, "3 resolved models") && has(m.error, "Qwen3.6-35B-A3B  bbbbbb3333") && !has(m.error, "zzzzzz9999"), "model: several", m.error);
  m = find_model(home, "bbbbbb");
  check(m.ok() && has(m.path, "/bbbbbb3333"), "model: prefix", m.error);
  m = find_model(home, "aaaaaa");
  check(!m.ok() && has(m.error, "matches 2"), "model: ambiguous prefix", m.error);
  m = find_model(home, "aaaaaa1");
  check(m.ok() && has(m.path, "/aaaaaa1111"), "model: longer prefix", m.error);
  m = find_model(home, "qq");
  check(!m.ok() && has(m.error, "at least 6"), "model: short prefix", m.error);
  m = find_model(home, "cccccc");
  check(!m.ok() && has(m.error, "starts with it"), "model: no match", m.error);
  check(list_models(home).size() == 3, "list_models");

  fs::remove_all(tmp);
  if (failures) return 1;
  printf("splash_discover_test: ok\n");
}
