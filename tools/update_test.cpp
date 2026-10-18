// The updater's pieces on their own, on the CPU (tests/test_update.py runs it; the whole command is tested there):
// version order by semantic versioning's precedence, which strings are release versions, and which kind of install a
// path is (folders made in a temporary directory).
//   update_test    prints the failures; exit 1 when there is one
#include "update/update.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ni::update;

namespace {

int fails = 0, checks = 0;

void check(bool ok, const std::string& name, const std::string& got = "") {
  ++checks;
  if (!ok) ++fails, printf("FAIL %s%s\n", name.c_str(), got.empty() ? "" : ("  (got: " + got + ")").c_str());
}

int cmp(const std::string& a, const std::string& b) { return compare(Version::parse(a), Version::parse(b)); }

void touch(const std::string& path, const std::string& text = "") {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream(path) << text;
}

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::kStandalone: return "standalone";
    case Kind::kHomebrew: return "homebrew";
    case Kind::kApp: return "app";
    case Kind::kSource: return "source";
  }
  return "?";
}

}  // namespace

int main() {
  // ---- order: each older than the next (semver.org's own example list, then Kapitsa's numbers)
  const std::vector<std::string> order = {
      "0.0.1",         "0.1.0-alpha", "0.1.0-alpha.1", "0.1.0-alpha.beta", "0.1.0-beta", "0.1.0-beta.2", "0.1.0-beta.11",
      "0.1.0-rc.1",    "0.1.0",       "0.1.9",         "0.1.10",           "0.1.32",     "0.1.33-rc.1",  "0.1.33",
      "0.1.100",       "0.2.0",       "0.10.0",        "1.0.0",            "10.0.0"};
  for (size_t i = 0; i < order.size(); ++i)
    for (size_t j = 0; j < order.size(); ++j) {
      const int want = i < j ? -1 : i > j ? 1 : 0, got = cmp(order[i], order[j]);
      check((got > 0) - (got < 0) == want, order[i] + " vs " + order[j], std::to_string(got));
    }
  check(cmp("v0.1.32", "0.1.32") == 0, "a leading v is the same version");
  check(cmp("0.1.32+e7da0cc", "0.1.32") == 0, "build metadata is ignored");
  check(cmp("1.0.0-rc.011", "1.0.0-rc.11") == 0, "numeric identifiers compare as numbers");
  check(cmp("1.0.0-rc.99999999999999999999", "1.0.0-rc.100000000000000000000") < 0, "long numeric identifiers too");
  for (const char* bad : {"", "0.1", "0.1.x", "1.2.3.4", "0.1.32-", "0.1.32-a..b", "0.1.32-a b", "x0.1.32", "-0.1.32",
                          "0.1.32 ", "1234567890.0.0"})
    check(!Version::parse(bad).ok, std::string("\"") + bad + "\" is not a version");

  // ---- what goes into a release URL
  for (const char* ok : {"0.1.33", "0.1.33-rc.1", "1.0.0-beta-2"}) check(valid_version(ok), std::string(ok) + " is a release");
  for (const char* bad : {"v0.1.33", "0.1.33+b", "../0.1.33", "0.1.33/../../x", "0.1.33?x", "0.1", ""})
    check(!valid_version(bad), std::string("\"") + bad + "\" is not a release");

  // ---- kinds of install, by path
  char tmpl[] = "/tmp/kapitsa-update-test-XXXXXX";
  const std::string T = mkdtemp(tmpl);
  {
    const std::string p = T + "/a/Nace Agent.app/Contents/Resources/kapitsa/kapitsa";
    touch(p);
    const Install in = detect(p);
    check(in.kind == Kind::kApp && in.app == T + "/a/Nace Agent.app", "inside an app bundle", kind_name(in.kind) + (" " + in.app));
  }
  {
    const std::string p = T + "/hb/Cellar/kapitsa/0.1.32/bin/kapitsa";
    touch(p);
    const Install in = detect(p);
    check(in.kind == Kind::kHomebrew && in.brew == T + "/hb/bin/brew" && !in.head, "in a Homebrew Cellar",
          kind_name(in.kind) + (" " + in.brew));
    check(brew_command(in) == "brew upgrade nace-ai/tap/kapitsa", "brew upgrades it", brew_command(in));
    const std::string h = T + "/hb/Cellar/kapitsa/HEAD-abc1234/bin/kapitsa";
    touch(h);
    const Install head = detect(h);
    check(head.kind == Kind::kHomebrew && head.head && brew_command(head) == "brew upgrade --fetch-HEAD nace-ai/tap/kapitsa",
          "a --HEAD install fetches HEAD", brew_command(head));
  }
  {
    const std::string src = T + "/my src";
    touch(src + "/.git/HEAD");
    touch(src + "/build/CMakeCache.txt", "# cache\nCMAKE_HOME_DIRECTORY:INTERNAL=" + src + "\n");
    touch(src + "/build/kapitsa");
    const Install in = detect(src + "/build/kapitsa");
    check(in.kind == Kind::kSource && in.source == src && in.build == src + "/build" && in.git, "a build folder of a git clone",
          kind_name(in.kind) + (" " + in.source));
    const std::string want = "git -C '" + src + "' pull && cmake --build '" + src + "/build'";
    check(source_command(in) == want, "its update command, quoted", source_command(in));
    touch(T + "/s2/b/CMakeCache.txt", "CMAKE_HOME_DIRECTORY:INTERNAL=" + T + "/s2\n");
    touch(T + "/s2/b/bin/kapitsa");
    const Install sub = detect(T + "/s2/b/bin/kapitsa");
    check(sub.kind == Kind::kSource && !sub.git && source_command(sub) == "cmake --build " + T + "/s2/b",
          "a binary one folder below the build folder; no git", source_command(sub));
  }
  {
    touch(T + "/sa/bin/kapitsa");
    std::filesystem::create_directories(T + "/sa/share/kapitsa");
    const Install in = detect(T + "/sa/bin/kapitsa");
    check(in.kind == Kind::kStandalone && in.notices == T + "/sa/share/kapitsa" && !in.release,
          "an install.sh install: its notices in ../share/kapitsa (an unsigned file is not a release build)",
          kind_name(in.kind) + (" " + in.notices));
    check(previous_path(in) == T + "/sa/share/kapitsa/kapitsa.previous", "the rollback copy beside them", previous_path(in));
    const std::string x = T + "/dl/kapitsa-0.1.32-darwin-arm64";
    touch(x + "/kapitsa"), touch(x + "/LICENSE"), touch(x + "/THIRD_PARTY_NOTICES");
    const Install un = detect(x + "/kapitsa");
    check(un.kind == Kind::kStandalone && un.notices == x, "an unpacked release: its license files beside it", un.notices);
    touch(T + "/copied/tools/kapitsa");
    const Install cp = detect(T + "/copied/tools/kapitsa");
    check(cp.kind == Kind::kStandalone && cp.notices == T + "/copied/share/kapitsa",
          "a copied binary: install.sh's place for that folder", cp.notices);
  }
  std::filesystem::remove_all(T);

  printf("%s: %d checks, %d failed\n", fails ? "FAILED" : "ok", checks, fails);
  return fails ? 1 : 0;
}
