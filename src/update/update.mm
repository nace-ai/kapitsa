// Kapitsa updating itself (update.h).
#import <Foundation/Foundation.h>
#import <Security/Security.h>

#include "update/update.h"

#include "hub/http.h"
#include "hub/hub.h"

#include <nlohmann/json.hpp>

#include <copyfile.h>
#include <dirent.h>
#include <fcntl.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

extern char** environ;

namespace ni::update {

using json = nlohmann::ordered_json;

namespace {

const char* const kDefaultReleases = "https://github.com/nace-ai/kapitsa/releases";
// packaging/release/build.sh signs every release with this identifier
const char* const kIdentifier = "ai.nace.kapitsa";
// Signed with a Developer ID Application certificate: the requirement codesign writes for one
const char* const kDeveloperId =
    "anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists and "
    "certificate leaf[field.1.2.840.113635.100.6.1.13] exists";
const uint64_t kMaxDownload = 256ull << 20;  // a release tarball is a few MB
const char* const kTmpPrefix = ".kapitsa-update-";
// What a request says about itself. NSURLSession's defaults would name this Mac's macOS version (User-Agent) and the
// user's languages (Accept-Language).
const hub::Headers kHeaders = {{"User-Agent", "kapitsa"}, {"Accept-Language", "en"}};

bool is_dir(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
// A regular file, not a symlink
bool is_file(const std::string& p) {
  struct stat st;
  return lstat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
bool exists(const std::string& p) {
  struct stat st;
  return lstat(p.c_str(), &st) == 0;
}
std::string parent(const std::string& p) {
  const size_t s = p.rfind('/');
  if (s == std::string::npos) return ".";
  return s == 0 ? "/" : p.substr(0, s);
}
std::string join(const std::string& dir, const std::string& name) { return dir == "/" ? "/" + name : dir + "/" + name; }
bool all_digits(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s)
    if (!isdigit(uint8_t(c))) return false;
  return true;
}
std::string lower(std::string s) {
  for (auto& c : s) c = char(tolower(uint8_t(c)));
  return s;
}

// ---------------------------------------------------------------- code signatures (the Security framework)

std::string cf_string(CFTypeRef v) {
  if (!v || CFGetTypeID(v) != CFStringGetTypeID()) return "";
  char b[1024];
  return CFStringGetCString(CFStringRef(v), b, sizeof b, kCFStringEncodingUTF8) ? b : "";
}

std::string os_error(OSStatus s) {
  CFStringRef m = SecCopyErrorMessageString(s, nullptr);
  std::string r = m ? cf_string(m) : "";
  if (m) CFRelease(m);
  return r.empty() ? "error " + std::to_string(int(s)) : r;
}

struct Signing {
  std::string identifier, team;
};

Signing signing_of(SecStaticCodeRef code, SecCSFlags flags) {
  Signing g;
  CFDictionaryRef d = nullptr;
  if (SecCodeCopySigningInformation(code, flags, &d) != errSecSuccess || !d) return g;
  g.identifier = cf_string(CFDictionaryGetValue(d, kSecCodeInfoIdentifier));
  g.team = cf_string(CFDictionaryGetValue(d, kSecCodeInfoTeamIdentifier));
  CFRelease(d);
  return g;
}

SecStaticCodeRef static_code(const std::string& path) {
  CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, reinterpret_cast<const UInt8*>(path.c_str()),
                                                         CFIndex(path.size()), false);
  if (!url) return nullptr;
  SecStaticCodeRef c = nullptr;
  const OSStatus s = SecStaticCodeCreateWithPath(url, kSecCSDefaultFlags, &c);
  CFRelease(url);
  return s == errSecSuccess ? c : nullptr;
}

SecRequirementRef requirement(const std::string& text) {
  CFStringRef s = CFStringCreateWithCString(nullptr, text.c_str(), kCFStringEncodingUTF8);
  if (!s) return nullptr;
  SecRequirementRef r = nullptr;
  if (SecRequirementCreateWithString(s, kSecCSDefaultFlags, &r) != errSecSuccess) r = nullptr;
  CFRelease(s);
  return r;
}

// The file's signing information, read without checking it ("" fields when it has none)
Signing file_signing(const std::string& path) {
  Signing g;
  if (SecStaticCodeRef c = static_code(path)) {
    g = signing_of(c, kSecCSDefaultFlags);
    CFRelease(c);
  }
  return g;
}

// This process's signature: its identifier, and the Developer ID team when Apple's certificate chain says it has one
// (a team field alone could be written into an ad hoc signature). The Security framework reads it from the file at the
// process's path: is_running_file() first.
struct Self {
  std::string identifier, team;
};
Self self_signing() {
  Self me;
  SecCodeRef code = nullptr;
  if (SecCodeCopySelf(kSecCSDefaultFlags, &code) != errSecSuccess || !code) return me;
  const Signing g = signing_of(SecStaticCodeRef(code), kSecCSSigningInformation);
  me.identifier = g.identifier;
  if (!g.team.empty())
    if (SecRequirementRef r = requirement(kDeveloperId)) {
      if (SecCodeCheckValidity(code, kSecCSDefaultFlags, r) == errSecSuccess) me.team = g.team;
      CFRelease(r);
    }
  CFRelease(code);
  return me;
}

// The file at `path` is the one this process runs: the same file (device, inode) as the mapping of its executable, which
// the kernel keeps when something else is renamed over the path (a signature read from the path would be that other
// file's)
bool is_running_file(const std::string& path) {
  proc_regionwithpathinfo info{};
  const uint64_t at = uint64_t(uintptr_t(_dyld_get_image_header(0)));  // the main executable's first page
  if (proc_pidinfo(getpid(), PROC_PIDREGIONPATHINFO, at, &info, sizeof info) != int(sizeof info)) return false;
  struct stat st;
  return stat(path.c_str(), &st) == 0 && uint64_t(st.st_ino) == info.prp_vip.vip_vi.vi_stat.vst_ino &&
         uint32_t(st.st_dev) == uint32_t(info.prp_vip.vip_vi.vi_stat.vst_dev);
}

// Checks the file's code signature as `codesign --verify --strict` does, and against `req` when it is not empty.
// "" when it passes, else why not.
std::string check_signature(const std::string& path, const std::string& req) {
  SecStaticCodeRef c = static_code(path);
  if (!c) return "its code signature can't be read";
  SecRequirementRef r = nullptr;
  if (!req.empty() && !(r = requirement(req))) {
    CFRelease(c);
    return "its signature can't be checked (bad requirement)";
  }
  const OSStatus s = SecStaticCodeCheckValidity(c, kSecCSStrictValidate | kSecCSCheckAllArchitectures, r);
  if (r) CFRelease(r);
  CFRelease(c);
  if (s == errSecSuccess) return "";
  if (s == errSecCSUnsigned) return "it is not signed";
  if (s == errSecCSReqFailed) return "it is not signed by the same developer as this kapitsa";
  return "its code signature is not valid (" + os_error(s) + ")";
}

// ---------------------------------------------------------------- programs

// Runs `args` (no shell) with stdin from /dev/null, within `limit_s`: its exit status, or -1 when it could not start,
// was killed at the limit or died of a signal. Its output (stdout and stderr, the first 64 KB) in *out.
int run(const std::vector<std::string>& args, double limit_s, std::string* out, const std::vector<std::string>& env_add) {
  int fd[2];
  if (pipe(fd) != 0) return -1;
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
  posix_spawn_file_actions_adddup2(&fa, fd[1], 2);
  posix_spawn_file_actions_addclose(&fa, fd[0]);
  posix_spawn_file_actions_addclose(&fa, fd[1]);
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_CLOEXEC_DEFAULT);
  std::vector<std::string> env;
  for (char** e = environ; e && *e; ++e) env.push_back(*e);
  for (const auto& kv : env_add) env.push_back(kv);
  std::vector<char*> a, e;
  for (const auto& s : args) a.push_back(const_cast<char*>(s.c_str()));
  a.push_back(nullptr);
  for (const auto& s : env) e.push_back(const_cast<char*>(s.c_str()));
  e.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, a[0], &fa, &at, a.data(), e.data());
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  close(fd[1]);
  if (rc != 0) {
    close(fd[0]);
    return -1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto left_ms = [&] {
    return int(limit_s * 1000) -
           int(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
  };
  std::string o;
  bool killed = false;
  for (;;) {
    const int left = left_ms();
    if (left <= 0) {
      kill(pid, SIGKILL), killed = true;
      break;
    }
    pollfd p{fd[0], POLLIN, 0};
    const int n = poll(&p, 1, left);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) continue;  // the limit, checked above
    char buf[4096];
    const ssize_t k = read(fd[0], buf, sizeof buf);
    if (k <= 0) break;  // EOF: it closed its output (it exits, or handed it to a child)
    if (o.size() < 65536) o.append(buf, size_t(k));
  }
  close(fd[0]);
  int st = 0;
  for (;;) {
    const pid_t w = waitpid(pid, &st, killed ? 0 : WNOHANG);
    if (w == pid) break;
    if (w < 0 && errno != EINTR) return -1;
    if (w == 0) {
      if (left_ms() <= 0) kill(pid, SIGKILL), killed = true;
      else std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  if (out) *out = o;
  if (killed || !WIFEXITED(st)) return -1;
  return WEXITSTATUS(st);
}

// ---------------------------------------------------------------- files

// The switch of a binary (held through it); g_switched once one has begun, after which a time limit lets the update
// finish. The folder of an install under way, which exit_now() removes.
std::mutex g_switch;
bool g_switched = false;
std::mutex g_tmp_mu;
std::string g_tmp;

// A new folder in the binary's folder for an install: removed when done (or by exit_now)
struct Temp {
  std::string path;
  explicit Temp(const std::string& dir) {
    std::string t = join(dir, std::string(kTmpPrefix) + "XXXXXX");
    if (mkdtemp(t.data())) path = t;
    std::lock_guard<std::mutex> lk(g_tmp_mu);
    g_tmp = path;
  }
  ~Temp() {
    {
      std::lock_guard<std::mutex> lk(g_tmp_mu);
      g_tmp.clear();
    }
    if (!path.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }
  }
};

// Folders of installs that ended without cleaning up (killed), an hour old or more: ours by their name only
void clean_stale(const std::string& dir) {
  DIR* d = opendir(dir.c_str());
  if (!d) return;
  std::vector<std::string> old;
  const time_t now = time(nullptr);
  while (dirent* e = readdir(d)) {
    if (strncmp(e->d_name, kTmpPrefix, strlen(kTmpPrefix)) != 0) continue;
    const std::string p = join(dir, e->d_name);
    struct stat st;
    if (lstat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && now - st.st_mtime > 3600) old.push_back(p);
  }
  closedir(d);
  for (const auto& p : old) {
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
  }
}

// Moves `from` to `to` (replacing it): a rename, or across volumes a copy that replaces `to` in one rename
bool move(const std::string& from, const std::string& to, std::string* why) {
  if (rename(from.c_str(), to.c_str()) == 0) return true;
  if (errno != EXDEV) {
    *why = "Can't write " + hub::tilde(to) + " (" + strerror(errno) + ")";
    return false;
  }
  const std::string tmp = to + ".tmp." + std::to_string(getpid());
  if (copyfile(from.c_str(), tmp.c_str(), nullptr, COPYFILE_ALL) != 0 || rename(tmp.c_str(), to.c_str()) != 0) {
    *why = "Can't write " + hub::tilde(to) + " (" + strerror(errno) + ")";
    unlink(tmp.c_str());
    return false;
  }
  unlink(from.c_str());
  return true;
}

// The folder for LICENSE, THIRD_PARTY_NOTICES and the rollback copy: install.sh's (../share/kapitsa from the binary's
// folder) when it is there; the binary's own folder when it is an unpacked release (the license files beside it);
// else install.sh's, created as install.sh would for this folder.
std::string notices_dir(const std::string& bindir) {
  const std::string up = parent(bindir);
  const std::string share = join(join(up, "share"), "kapitsa");
  if (is_dir(share)) return share;
  if (is_file(join(bindir, "LICENSE")) && is_file(join(bindir, "THIRD_PARTY_NOTICES"))) return bindir;
  return share;
}

// The source tree a CMakeCache.txt names (CMAKE_HOME_DIRECTORY), "" if none
std::string cmake_home(const std::string& cache) {
  std::ifstream f(cache);
  std::string l;
  const std::string key = "CMAKE_HOME_DIRECTORY:";
  while (std::getline(f, l))
    if (l.rfind(key, 0) == 0) {
      const size_t eq = l.find('=');
      return eq == std::string::npos ? "" : l.substr(eq + 1);
    }
  return "";
}

// A path as a shell takes it: ~/ for the home folder, quoted when it needs to be
std::string shell_path(const std::string& p) {
  auto quote = [](const std::string& s) {
    bool plain = !s.empty();
    for (char c : s) plain = plain && (isalnum(uint8_t(c)) || strchr("_./+-:@%,=", c));
    if (plain) return s;
    std::string q = "'";
    for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
  };
  const std::string t = hub::tilde(p);
  if (t.rfind("~/", 0) == 0) return "~/" + quote(t.substr(2));
  return quote(p);
}

// ---------------------------------------------------------------- the network

struct Where {
  std::string scheme, host;
  int port = 0;  // the scheme's default when the URL names none
};
Where where(const std::string& url) {
  Where w;
  @autoreleasepool {
    NSURL* u = [NSURL URLWithString:@(url.c_str())];
    if (u && u.scheme) w.scheme = lower(u.scheme.UTF8String);
    if (u && u.host) w.host = lower(u.host.UTF8String);
    w.port = u && u.port ? u.port.intValue : w.scheme == "https" ? 443 : w.scheme == "http" ? 80 : 0;
  }
  return w;
}

bool loopback(const std::string& host) { return host == "127.0.0.1" || host == "localhost" || host == "::1"; }

// A client that follows redirects only within the releases page's site and to GitHub's (github.com, and
// *.githubusercontent.com where release files are), over https
void restrict_redirects(hub::HttpClient& http, const std::string& base) {
  const Where b = where(base);
  http.allow_redirects([b](const std::string& to) {
    const Where w = where(to);
    if (w.scheme == b.scheme && w.host == b.host && w.port == b.port) return true;
    if (w.scheme != "https") return false;
    const std::string gh = ".githubusercontent.com";
    return w.host == "github.com" ||
           (w.host.size() > gh.size() && w.host.compare(w.host.size() - gh.size(), gh.size(), gh) == 0);
  });
}

std::string http_problem(const hub::HttpResult& r, const std::string& url) {
  if (!r.refused.empty()) return url + " redirects to " + r.refused + ", which is not GitHub";
  if (!r.error.empty()) return "can't reach " + where(url).host + " (" + r.error + ")";
  if (r.status == 404) return "there is no " + url;
  return url + " answered HTTP " + std::to_string(r.status);
}

// GETs `url` into the file `path`
Failure download(hub::HttpClient& http, const std::string& url, const std::string& path, std::string* why) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    *why = "Can't write " + hub::tilde(path) + " (" + strerror(errno) + ")";
    return Failure::kLocal;
  }
  uint64_t got = 0;
  bool too_big = false, write_failed = false;
  int write_errno = 0;
  const hub::HttpResult r = http.stream(url, kHeaders, 0, 0, [&](const uint8_t* p, size_t n) {
    got += n;
    if (got > kMaxDownload) {
      too_big = true;
      return false;
    }
    if (fwrite(p, 1, n, f) != n) {
      write_failed = true, write_errno = errno;
      return false;
    }
    return true;
  });
  if (fclose(f) != 0 && !write_failed) write_failed = true, write_errno = errno;
  if (write_failed) {
    *why = "Can't write " + hub::tilde(path) + " (" + strerror(write_errno) + ")";
    return Failure::kLocal;
  }
  if (too_big) {
    *why = url + " is larger than a release can be";
    return Failure::kRefused;
  }
  if (!r.ok()) {
    *why = http_problem(r, url);
    return Failure::kNetwork;
  }
  return Failure::kNone;
}

// The sha256 a .sha256 file holds (`shasum -a 256` output: the hash first), "" if none
std::string listed_sha256(const std::string& path) {
  std::ifstream f(path);
  std::string h;
  f >> h;
  h = lower(h);
  if (h.size() != 64) return "";
  for (char c : h)
    if (!isxdigit(uint8_t(c))) return "";
  return h;
}

// No kapitsa bench holds the binary (hold_for_benchmark): an exclusive lock on it is free
bool bench_running(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  const bool busy = flock(fd, LOCK_EX | LOCK_NB) != 0 && errno == EWOULDBLOCK;
  close(fd);  // releases it
  return busy;
}

// Switches the binary at `in.path` to the file `next` (in `tmp`, the same folder): after checking that no kapitsa bench
// holds it and that it is still this kapitsa, links it to tmp/previous (kept: the rollback copy) and renames `next`
// over it. *previous: where the old binary is now.
Done switch_to(const Install& in, const std::string& next, const std::string& tmp, std::string* previous) {
  Done d;
  std::lock_guard<std::mutex> lk(g_switch);
  const int fd = open(in.path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    d.failure = Failure::kChanged;
    d.why = hub::tilde(in.path) + " is gone";
    return d;
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    close(fd);
    d.failure = Failure::kBusy;
    d.why = "a kapitsa bench (or another update) is using " + hub::tilde(in.path);
    return d;
  }
  // the file at the path is the one this process runs: never replace anything else (another install, or a binary
  // something else put there since this kapitsa started)
  if (!is_running_file(in.path)) {
    close(fd);
    d.failure = Failure::kChanged;
    d.why = hub::tilde(in.path) + " changed since this kapitsa started";
    return d;
  }
  g_switched = true;
  const std::string keep = join(tmp, "previous");
  if (link(in.path.c_str(), keep.c_str()) != 0 &&
      copyfile(in.path.c_str(), keep.c_str(), nullptr, COPYFILE_ALL | COPYFILE_CLONE) != 0) {
    d.failure = Failure::kLocal;
    d.why = "Can't keep a copy of " + hub::tilde(in.path) + " (" + strerror(errno) + ")";
  } else if (rename(next.c_str(), in.path.c_str()) != 0) {
    d.failure = Failure::kLocal;
    d.why = "Can't replace " + hub::tilde(in.path) + " (" + strerror(errno) + ")";
  } else {
    *previous = keep;
  }
  close(fd);
  return d;
}

}  // namespace

// ---------------------------------------------------------------- versions

Version Version::parse(const std::string& in) {
  Version v;
  std::string s = in;
  if (!s.empty() && s[0] == 'v') s.erase(0, 1);
  if (const size_t plus = s.find('+'); plus != std::string::npos) s.resize(plus);
  std::string core = s, pre;
  const size_t dash = s.find('-');
  if (dash != std::string::npos) core = s.substr(0, dash), pre = s.substr(dash + 1);
  long n[3] = {0, 0, 0};
  size_t at = 0;
  for (int i = 0; i < 3; ++i) {
    size_t e = at;
    while (e < core.size() && isdigit(uint8_t(core[e]))) ++e;
    if (e == at || e - at > 9) return v;
    n[i] = atol(core.substr(at, e - at).c_str());
    if (i < 2) {
      if (e >= core.size() || core[e] != '.') return v;
      at = e + 1;
    } else if (e != core.size()) {
      return v;
    }
  }
  if (dash != std::string::npos) {
    size_t a = 0;
    for (;;) {
      size_t b = pre.find('.', a);
      if (b == std::string::npos) b = pre.size();
      const std::string id = pre.substr(a, b - a);
      if (id.empty()) return v;
      for (char c : id)
        if (!isalnum(uint8_t(c)) && c != '-') return v;
      v.pre.push_back(id);
      if (b == pre.size()) break;
      a = b + 1;
    }
  }
  v.major = n[0], v.minor = n[1], v.patch = n[2];
  v.ok = true;
  return v;
}

int compare(const Version& a, const Version& b) {
  for (const auto& [x, y] : {std::pair{a.major, b.major}, {a.minor, b.minor}, {a.patch, b.patch}})
    if (x != y) return x < y ? -1 : 1;
  // a release is newer than its prereleases
  if (a.pre.empty() != b.pre.empty()) return a.pre.empty() ? 1 : -1;
  for (size_t i = 0; i < a.pre.size() && i < b.pre.size(); ++i) {
    const std::string &x = a.pre[i], &y = b.pre[i];
    const bool xn = all_digits(x), yn = all_digits(y);
    if (xn && yn) {  // numerically, without overflow: the longer number is larger
      const std::string xs = x.substr(std::min(x.find_first_not_of('0'), x.size() - 1));
      const std::string ys = y.substr(std::min(y.find_first_not_of('0'), y.size() - 1));
      if (xs.size() != ys.size()) return xs.size() < ys.size() ? -1 : 1;
      if (xs != ys) return xs < ys ? -1 : 1;
    } else if (xn != yn) {
      return xn ? -1 : 1;  // numeric identifiers come before alphanumeric ones
    } else if (x != y) {
      return x < y ? -1 : 1;
    }
  }
  if (a.pre.size() != b.pre.size()) return a.pre.size() < b.pre.size() ? -1 : 1;
  return 0;
}

bool valid_version(const std::string& s) {
  if (s.empty() || !isdigit(uint8_t(s[0])) || s.find('+') != std::string::npos) return false;
  return Version::parse(s).ok;
}

// ---------------------------------------------------------------- the install

std::string self_path() {
  char buf[PATH_MAX];
  uint32_t n = sizeof buf;
  if (_NSGetExecutablePath(buf, &n) != 0) return "";
  char real[PATH_MAX];
  return realpath(buf, real) ? real : buf;
}

Install detect(const std::string& path) {
  Install in;
  in.path = path;
  in.dir = parent(path);
  // inside an app bundle (the Nace Agent app ships its own, pinned and signed with the app)
  if (const size_t a = path.find(".app/Contents/"); a != std::string::npos) {
    in.kind = Kind::kApp;
    in.app = path.substr(0, a + 4);
    return in;
  }
  // in a Homebrew Cellar: HOMEBREW_PREFIX/Cellar/kapitsa/VERSION/bin/kapitsa (HEAD-<commit> for --HEAD)
  const std::string cellar = "/Cellar/kapitsa/";
  if (const size_t c = path.find(cellar); c != std::string::npos) {
    in.kind = Kind::kHomebrew;
    in.brew = path.substr(0, c) + "/bin/brew";
    in.head = path.compare(c + cellar.size(), 4, "HEAD") == 0;
    return in;
  }
  // a build folder: CMakeCache.txt beside the binary or in the folder above it
  for (const std::string& d : {in.dir, parent(in.dir)})
    if (is_file(join(d, "CMakeCache.txt"))) {
      in.kind = Kind::kSource;
      in.build = d;
      in.source = cmake_home(join(d, "CMakeCache.txt"));
      in.git = !in.source.empty() && exists(join(in.source, ".git"));
      return in;
    }
  in.kind = Kind::kStandalone;
  in.notices = notices_dir(in.dir);
  in.release = file_signing(path).identifier == kIdentifier;
  return in;
}

std::string brew_command(const Install& in) {
  return std::string("brew upgrade ") + (in.head ? "--fetch-HEAD " : "") + "nace-ai/tap/kapitsa";
}

std::string source_command(const Install& in) {
  const std::string build = "cmake --build " + shell_path(in.build);
  return in.git ? "git -C " + shell_path(in.source) + " pull && " + build : build;
}

std::string unwritable(const Install& in) {
  if (access(in.dir.c_str(), W_OK) != 0) return in.dir;
  std::string d = in.notices;  // the folder that exists, where it would be created
  while (!is_dir(d) && d != "/" && d != ".") d = parent(d);
  if (access(d.c_str(), W_OK) != 0) return d;
  return "";
}

std::string previous_path(const Install& in) { return join(in.notices, "kapitsa.previous"); }

// ---------------------------------------------------------------- releases

std::string releases_url(std::string* why) {
  const char* e = getenv("KAPITSA_UPDATE_URL");
  std::string u = e && *e ? e : kDefaultReleases;
  while (!u.empty() && u.back() == '/') u.pop_back();
  const Where w = where(u);
  if (w.host.empty() || !(w.scheme == "https" || (w.scheme == "http" && loopback(w.host)))) {
    if (why) *why = "KAPITSA_UPDATE_URL must be an https URL (or http to this Mac), not \"" + u + "\"";
    return "";
  }
  return u;
}

Latest latest() {
  Latest L;
  const std::string base = releases_url(&L.why);
  if (base.empty()) {
    L.failure = Failure::kLocal;
    return L;
  }
  hub::HttpClient http(30);
  restrict_redirects(http, base);
  const std::string url = base + "/latest";
  const hub::HttpResult r = http.head(url, kHeaders);
  if (!r.ok()) {
    L.failure = Failure::kNetwork;
    L.why = http_problem(r, url);
    return L;
  }
  // .../releases/tag/vX.Y.Z (a repository with no release redirects to its releases page instead)
  std::string tag = r.url;
  if (const size_t q = tag.find_first_of("?#"); q != std::string::npos) tag.resize(q);
  const size_t t = tag.rfind("/tag/");
  tag = t == std::string::npos ? "" : tag.substr(t + 5);
  if (tag.size() < 2 || tag[0] != 'v' || !valid_version(tag.substr(1))) {
    L.failure = Failure::kNetwork;
    L.why = "found no release at " + base;
    return L;
  }
  L.version = tag.substr(1);
  return L;
}

std::string version_of(const std::string& path) {
  std::string out;
  // the new binary's own automatic update stays off: it is only asked its version
  if (run({path, "--version"}, 15, &out, {"KAPITSA_NO_AUTO_UPDATE=1"}) != 0) return "";
  if (out.rfind("kapitsa ", 0) != 0) return "";
  std::string v = out.substr(8);
  if (const size_t e = v.find_first_of(" \n"); e != std::string::npos) v.resize(e);
  return valid_version(v) ? v : "";
}

// The benchmark harness beside the notices, where find_harness() (bench_cmd.cpp) looks for it, so `kapitsa bench
// compare` runs it from an install with no clone. A release carries it in the tarball (bench/harness), so an update
// installs it too: an install made before releases shipped it gets it here, and an older one is replaced whole
// (staged beside it, then renamed over it) so a half copy never leaves a harness that does not match this binary.
// A release without one (all of them before it shipped) keeps whatever is installed.
bool install_harness(const std::string& from, const Install& in, std::string* why) {
  if (!is_file(join(from, "kbench.py"))) return true;
  const std::string bench = join(in.notices, "bench");
  std::error_code ec;
  std::filesystem::create_directories(bench, ec);
  const std::string dest = join(bench, "harness"), pid = std::to_string(getpid());
  const std::string fresh = dest + ".new." + pid, old = dest + ".old." + pid;
  std::filesystem::remove_all(fresh, ec);
  std::filesystem::copy(from, fresh,
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec);
  if (ec) {
    *why = "Can't write " + hub::tilde(fresh) + " (" + ec.message() + ")";
    std::filesystem::remove_all(fresh, ec);
    return false;
  }
  // local.env is the user's own per-machine settings (OLLAMA_MODELS, LLAMA_SERVER, ...): it goes into the new folder,
  // and if it can't be read the old harness (with it) stays as it is rather than the update silently dropping it
  if (is_file(join(dest, "local.env"))) {
    std::error_code keep;
    std::filesystem::copy_file(join(dest, "local.env"), join(fresh, "local.env"),
                               std::filesystem::copy_options::overwrite_existing, keep);
    if (keep) {
      *why = "Can't keep " + hub::tilde(join(dest, "local.env")) + " (" + keep.message() + ")";
      std::filesystem::remove_all(fresh, ec);
      return false;
    }
  }
  ec.clear();
  std::filesystem::rename(dest, old, ec);
  const bool had = !ec;
  ec.clear();
  std::filesystem::rename(fresh, dest, ec);
  if (ec) {
    *why = "Can't write " + hub::tilde(dest) + " (" + ec.message() + ")";
    std::filesystem::remove_all(fresh, ec);
    if (had) {
      std::error_code back;
      std::filesystem::rename(old, dest, back);
    }
    return false;
  }
  std::filesystem::remove_all(old, ec);
  return true;
}

Done install(const Install& in, const std::string& version, const Stage& stage) {
  Done d;
  auto failed = [&](Failure f, const std::string& why) {
    d.failure = f, d.why = why;
    return d;
  };
  if (in.kind != Kind::kStandalone) return failed(Failure::kLocal, "this kapitsa is not one it can update");
  if (!valid_version(version)) return failed(Failure::kRefused, "\"" + version + "\" is not a version");
  std::string why;
  const std::string base = releases_url(&why);
  if (base.empty()) return failed(Failure::kLocal, why);
  if (const std::string w = unwritable(in); !w.empty()) return failed(Failure::kLocal, "Can't write " + hub::tilde(w));
  if (bench_running(in.path)) return failed(Failure::kBusy, "a kapitsa bench is running from " + hub::tilde(in.path));
  if (!is_running_file(in.path)) return failed(Failure::kChanged, hub::tilde(in.path) + " changed since this kapitsa started");
  const Self me = self_signing();
  clean_stale(in.dir);
  Temp tmp(in.dir);
  if (tmp.path.empty()) return failed(Failure::kLocal, "Can't write " + hub::tilde(in.dir) + " (" + strerror(errno) + ")");

  // the tarball and its checksum, as install.sh fetches them
  const std::string name = "kapitsa-" + version + "-darwin-arm64.tar.gz";
  const std::string url = base + "/download/v" + version + "/" + name;
  const std::string tarball = join(tmp.path, name), sums = tarball + ".sha256";
  if (stage) stage("Downloading " + name);
  hub::HttpClient http(30);
  restrict_redirects(http, base);
  if (const Failure f = download(http, url, tarball, &why); f != Failure::kNone) return failed(f, why);
  if (const Failure f = download(http, url + ".sha256", sums, &why); f != Failure::kNone) return failed(f, why);
  if (stage) stage("Checking it");
  const std::string want = listed_sha256(sums);
  if (want.empty()) return failed(Failure::kRefused, name + ".sha256 holds no sha256");
  std::string got;
  try {
    got = hub::sha256_file(tarball);
  } catch (const std::exception& e) {
    return failed(Failure::kLocal, e.what());
  }
  if (got != want) return failed(Failure::kRefused, "its sha256 is " + got + ", not the published " + want);

  // unpacked (by macOS's tar) into a folder of its own
  const std::string x = join(tmp.path, "x");
  if (mkdir(x.c_str(), 0700) != 0) return failed(Failure::kLocal, "Can't write " + hub::tilde(x) + " (" + strerror(errno) + ")");
  std::string out;
  if (run({"/usr/bin/tar", "-xzf", tarball, "-C", x, "--strip-components", "1"}, 120, &out, {}) != 0)
    return failed(Failure::kRefused, "can't unpack " + name + (out.empty() ? "" : ": " + out.substr(0, out.find('\n'))));
  const std::string next = join(x, "kapitsa");
  for (const char* f : {"kapitsa", "LICENSE", "THIRD_PARTY_NOTICES"})
    if (!is_file(join(x, f))) return failed(Failure::kRefused, name + " is not a Kapitsa release (no " + f + " file)");
  chmod(next.c_str(), 0755);

  // the new binary: signed, as Kapitsa, by the same developer when this one is Developer ID signed; the version asked
  const std::string req = me.team.empty() ? "" : std::string(kDeveloperId) + " and certificate leaf[subject.OU] = \"" + me.team + "\"";
  if (const std::string bad = check_signature(next, req); !bad.empty())
    return failed(Failure::kRefused, "the kapitsa in " + name + ": " + bad);
  if (const std::string id = file_signing(next).identifier; id != kIdentifier)
    return failed(Failure::kRefused, "the kapitsa in " + name + " is signed as \"" + id + "\", not as Kapitsa (" + kIdentifier + ")");
  const std::string says = version_of(next);
  if (says != version)
    return failed(Failure::kRefused, "the kapitsa in " + name + (says.empty() ? " does not run" : " is version " + says));

  // the switch, then the rollback copy, the license files and the benchmark harness beside each other (where
  // install.sh puts the notices)
  if (stage) stage("Installing it");
  std::string kept;
  if (Done s = switch_to(in, next, tmp.path, &kept); s.failure != Failure::kNone) return s;
  std::error_code ec;
  std::filesystem::create_directories(in.notices, ec);
  std::string warn;
  if (!move(kept, previous_path(in), &warn)) d.warning = warn + ": no copy of the previous version is kept";
  for (const char* f : {"LICENSE", "THIRD_PARTY_NOTICES"}) {
    chmod(join(x, f).c_str(), 0644);
    if (!move(join(x, f), join(in.notices, f), &warn) && d.warning.empty()) d.warning = warn;
  }
  if (!install_harness(join(join(x, "bench"), "harness"), in, &warn) && d.warning.empty()) d.warning = warn;
  return d;
}

Done rollback(const Install& in, std::string* version) {
  Done d;
  auto failed = [&](Failure f, const std::string& why) {
    d.failure = f, d.why = why;
    return d;
  };
  if (in.kind != Kind::kStandalone) return failed(Failure::kLocal, "this kapitsa is not one it can update");
  const std::string prev = previous_path(in);
  if (!is_file(prev)) return failed(Failure::kLocal, "No previous version is kept (" + hub::tilde(prev) + ")");
  if (const std::string w = unwritable(in); !w.empty()) return failed(Failure::kLocal, "Can't write " + hub::tilde(w));
  if (bench_running(in.path)) return failed(Failure::kBusy, "a kapitsa bench is running from " + hub::tilde(in.path));
  // the binary the last update replaced: signed (by anyone: it ran here before) and a kapitsa that runs
  if (const std::string bad = check_signature(prev, ""); !bad.empty())
    return failed(Failure::kRefused, hub::tilde(prev) + ": " + bad);
  *version = version_of(prev);
  if (version->empty()) return failed(Failure::kRefused, hub::tilde(prev) + " is not a kapitsa that runs");
  clean_stale(in.dir);
  Temp tmp(in.dir);
  if (tmp.path.empty()) return failed(Failure::kLocal, "Can't write " + hub::tilde(in.dir) + " (" + strerror(errno) + ")");
  const std::string next = join(tmp.path, "kapitsa");
  if (link(prev.c_str(), next.c_str()) != 0 && copyfile(prev.c_str(), next.c_str(), nullptr, COPYFILE_ALL | COPYFILE_CLONE) != 0)
    return failed(Failure::kLocal, "Can't write " + hub::tilde(next) + " (" + strerror(errno) + ")");
  std::string kept;
  if (Done s = switch_to(in, next, tmp.path, &kept); s.failure != Failure::kNone) return s;
  // the binary just replaced is the next rollback copy
  std::string warn;
  if (!move(kept, prev, &warn)) d.warning = warn;
  return d;
}

bool hold_for_benchmark() {
  const std::string p = self_path();
  const int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return true;
  flock(fd, LOCK_SH);  // an update switching it right now finishes first (a moment); the lock lasts until exit
  return is_running_file(p);
}

bool stop_before_switch() {
  g_switch.lock();  // held until exit_now(): no switch begins after this
  if (!g_switched) return true;
  g_switch.unlock();
  return false;
}

void exit_now(int code) {
  std::string tmp;
  {
    std::lock_guard<std::mutex> lk2(g_tmp_mu);
    tmp = g_tmp;
  }
  if (!tmp.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
  }
  _exit(code);
}

// ---------------------------------------------------------------- the state

std::string state_path() { return hub::home_dir() + "/update.json"; }

State read_state() {
  State s;
  std::ifstream f(state_path());
  if (!f) return s;
  std::stringstream ss;
  ss << f.rdbuf();
  const json j = json::parse(ss.str(), nullptr, false);
  if (!j.is_object()) return s;
  if (j.contains("automatic") && j["automatic"].is_boolean()) s.automatic = j["automatic"].get<bool>();
  if (j.contains("binaries") && j["binaries"].is_object())
    for (const auto& [path, r] : j["binaries"].items()) {
      if (!r.is_object()) continue;
      Record& x = s.bins[path];
      auto str = [&](const char* k) { return r.contains(k) && r[k].is_string() ? r[k].get<std::string>() : std::string(); };
      if (r.contains("checked") && r["checked"].is_number_integer()) x.checked = r["checked"].get<long long>();
      x.latest = str("latest"), x.result = str("result"), x.from = str("from"), x.to = str("to"), x.why = str("why");
      x.skip = str("skip");
      x.notice = r.contains("notice") && r["notice"].is_boolean() && r["notice"].get<bool>();
    }
  return s;
}

bool update_state(const std::function<void(State&)>& change) {
  // under sudo the files would be root's, in the user's folder: the user's state is left alone
  if (geteuid() == 0 && getenv("SUDO_UID")) return false;
  const std::string home = hub::home_dir();
  std::error_code ec;
  std::filesystem::create_directories(home, ec);
  // the lock file stays (removing it would let two writers lock different files)
  const int fd = open((home + "/update.lock").c_str(), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return false;
  bool locked = false;
  for (int i = 0; i < 50 && !(locked = flock(fd, LOCK_EX | LOCK_NB) == 0); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (!locked) {
    close(fd);
    return false;
  }
  State s = read_state();
  change(s);
  json j = json::object();
  j["automatic"] = s.automatic;
  json bins = json::object();
  for (const auto& [path, r] : s.bins) {
    if (!exists(path)) continue;  // a binary that is gone
    json x = json::object();
    x["checked"] = r.checked;
    for (const auto& [k, v] : {std::pair<const char*, const std::string*>{"latest", &r.latest}, {"result", &r.result},
                               {"from", &r.from}, {"to", &r.to}, {"why", &r.why}, {"skip", &r.skip}})
      if (!v->empty()) x[k] = *v;
    if (r.notice) x["notice"] = true;
    bins[path] = x;
  }
  j["binaries"] = bins;
  const std::string path = state_path(), tmp = path + ".tmp." + std::to_string(getpid());
  bool ok = false;
  {
    std::ofstream f(tmp);
    if (f) {
      f << j.dump(2) << "\n";
      f.close();
      ok = bool(f);
    }
  }
  ok = ok && rename(tmp.c_str(), path.c_str()) == 0;
  if (!ok) unlink(tmp.c_str());
  close(fd);
  return ok;
}

}  // namespace ni::update
