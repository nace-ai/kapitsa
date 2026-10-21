// Submitting a `kapitsa bench` result (bench.h): a pull request that adds the file under bench/community/results/, made
// with the user's own `gh` login (its REST calls through `gh api`: a fork, a branch on it, the file, the pull request).
// No other credentials are read or used, and without gh nothing is sent: there is no issue or browser path.
#include "bench/bench.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

extern char** environ;

namespace ni::bench {

std::string target_repo() {
  const char* r = getenv("KAPITSA_BENCH_REPO");
  return r && *r ? r : kRepo;
}

std::string shell_quote(const std::string& s) {
  if (!s.empty() && s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./:=@,+%") == std::string::npos)
    return s;
  std::string q = "'";
  for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return q + "'";
}

namespace {

std::string find_exe(const std::string& name) {
  const char* p = getenv("PATH");
  const std::string path = p ? p : "/usr/bin:/bin";
  for (size_t a = 0; a <= path.size();) {
    size_t b = path.find(':', a);
    if (b == std::string::npos) b = path.size();
    const std::string f = (b > a ? path.substr(a, b - a) : ".") + "/" + name;
    struct stat st;
    if (stat(f.c_str(), &st) == 0 && S_ISREG(st.st_mode) && access(f.c_str(), X_OK) == 0) return f;
    a = b + 1;
  }
  return "";
}

struct Ran {
  int code = -1;
  std::string out, err;
};

// Runs argv (argv[0] a full path) and collects its output; nothing on the terminal
Ran run(const std::vector<std::string>& argv) {
  Ran r;
  int o[2], e[2];
  if (pipe(o) != 0) return r;
  if (pipe(e) != 0) {
    close(o[0]), close(o[1]);
    return r;
  }
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&fa, o[1], 1);
  posix_spawn_file_actions_adddup2(&fa, e[1], 2);
  posix_spawn_file_actions_addclose(&fa, o[0]);
  posix_spawn_file_actions_addclose(&fa, e[0]);
  std::vector<char*> args;
  for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
  args.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, args[0], &fa, nullptr, args.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  close(o[1]), close(e[1]);
  if (rc == 0) {
    // both pipes, without blocking on one while the other fills
    fcntl(o[0], F_SETFL, O_NONBLOCK);
    fcntl(e[0], F_SETFL, O_NONBLOCK);
    bool oo = true, eo = true;
    char b[8192];
    while (oo || eo) {
      bool any = false;
      for (auto [fd, open, dst] : {std::tuple<int, bool*, std::string*>{o[0], &oo, &r.out}, {e[0], &eo, &r.err}}) {
        if (!*open) continue;
        const ssize_t n = read(fd, b, sizeof b);
        if (n > 0) dst->append(b, size_t(n)), any = true;
        else if (n == 0) *open = false;
        else if (errno != EAGAIN && errno != EINTR) *open = false;
      }
      if (!any) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    int st = 0;
    waitpid(pid, &st, 0);
    r.code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
  }
  close(o[0]), close(e[0]);
  return r;
}

std::string trim(std::string s) {
  while (!s.empty() && isspace(uint8_t(s.back()))) s.pop_back();
  size_t a = 0;
  while (a < s.size() && isspace(uint8_t(s[a]))) ++a;
  return s.substr(a);
}

std::string base64(const std::string& in) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  size_t i = 0;
  for (; i + 2 < in.size(); i += 3) {
    const uint32_t v = (uint32_t(uint8_t(in[i])) << 16) | (uint32_t(uint8_t(in[i + 1])) << 8) | uint8_t(in[i + 2]);
    out += t[v >> 18], out += t[(v >> 12) & 63], out += t[(v >> 6) & 63], out += t[v & 63];
  }
  if (i + 1 == in.size()) {
    const uint32_t v = uint32_t(uint8_t(in[i])) << 16;
    out += t[v >> 18], out += t[(v >> 12) & 63], out += "==";
  } else if (i + 2 == in.size()) {
    const uint32_t v = (uint32_t(uint8_t(in[i])) << 16) | (uint32_t(uint8_t(in[i + 1])) << 8);
    out += t[v >> 18], out += t[(v >> 12) & 63], out += t[(v >> 6) & 63], out += '=';
  }
  return out;
}

// A JSON request body for `gh api --input FILE`, in a file of its own (removed after the call)
struct InputFile {
  std::string path;
  explicit InputFile(const ojson& body) {
    const char* dir = getenv("TMPDIR");  // the user's own temporary folder on macOS
    std::string tmpl = std::string(dir && *dir ? dir : "/tmp") + "/kapitsa-bench-XXXXXX";
    const int fd = mkstemp(tmpl.data());
    if (fd < 0) throw std::runtime_error("can't write a temporary file for gh");
    const std::string s = body.dump();
    const bool ok = write(fd, s.data(), s.size()) == ssize_t(s.size());
    close(fd);
    path = tmpl;
    if (!ok) throw std::runtime_error("can't write a temporary file for gh");
  }
  ~InputFile() { unlink(path.c_str()); }
};

ojson gh_json(const std::string& gh, std::vector<std::string> args, const std::string& what) {
  args.insert(args.begin(), gh);
  const Ran r = run(args);
  if (r.code != 0) {
    std::string e = trim(r.err.empty() ? r.out : r.err);
    if (e.size() > 400) e.resize(400);
    throw std::runtime_error(what + " failed: " + e);
  }
  const ojson j = ojson::parse(r.out, nullptr, false);
  return j.is_discarded() ? ojson(trim(r.out)) : j;
}

}  // namespace

GhStatus gh_status() {
  GhStatus s;
  s.path = find_exe("gh");
  s.installed = !s.path.empty();
  if (!s.installed) return s;
  // the login gh uses for github.com (no other account, no token of ours)
  const Ran r = run({s.path, "api", "--hostname", "github.com", "user", "--jq", ".login"});
  if (r.code == 0 && !trim(r.out).empty()) s.logged_in = true, s.login = trim(r.out);
  return s;
}

PrPlan pr_plan(const std::string& repo, const std::string& login, const ojson& result, const std::string& content) {
  PrPlan p;
  p.repo = repo;
  const std::string owner = repo.substr(0, repo.find('/')), name = repo.substr(repo.find('/') + 1);
  p.fork = login == owner ? repo : login + "/" + name;
  const std::string tag = sha256_hex(content).substr(0, 6);
  std::string file = result_file_name(result);
  p.file_name = file.substr(0, file.size() - 5) + "-" + tag + ".json";
  p.path = "bench/community/results/" + p.file_name;
  p.branch = "bench-" + result.value("time_utc", std::string("result")).substr(0, 10) + "-" + tag;
  p.title = result_title(result);
  p.body = result_markdown(result);
  p.content = content;
  const std::string head = login == owner ? p.branch : login + ":" + p.branch;
  if (login != owner) p.commands.push_back("gh api -X POST repos/" + repo + "/forks -F default_branch_only=true");
  p.commands.push_back("gh api repos/" + repo + " --jq .default_branch");
  p.commands.push_back("gh api repos/" + repo + "/git/ref/heads/main --jq .object.sha");
  p.commands.push_back("gh api -X POST repos/" + p.fork + "/git/refs -f ref=refs/heads/" + p.branch + " -f sha=<that sha>");
  p.commands.push_back("gh api -X PUT repos/" + p.fork + "/contents/" + p.path + " --input <{message: " +
                       shell_quote(p.title) + ", branch: " + p.branch + ", content: <the JSON in base64>}>");
  p.commands.push_back("gh api -X POST repos/" + repo + "/pulls --input <{title: " + shell_quote(p.title) + ", head: " + head +
                       ", base: main, body: <the summary below>, maintainer_can_modify: true}>");
  return p;
}

std::string open_pr(const PrPlan& p, const std::string& gh, const Progress& step) {
  std::string fork = p.fork;
  if (fork != p.repo) {
    if (step) step("forking " + p.repo);
    // returns the fork, the existing one too (its name may differ from the repo's)
    const ojson f = gh_json(gh, {"api", "-X", "POST", "repos/" + p.repo + "/forks", "-F", "default_branch_only=true"}, "Forking " + p.repo);
    if (f.is_object() && f.contains("full_name") && f["full_name"].is_string()) fork = f["full_name"];
  }
  if (step) step("reading " + p.repo + "'s main branch");
  const ojson info = gh_json(gh, {"api", "repos/" + p.repo}, "Reading " + p.repo);
  const std::string base = info.value("default_branch", std::string("main"));
  const ojson ref = gh_json(gh, {"api", "repos/" + p.repo + "/git/ref/heads/" + base}, "Reading " + p.repo + "'s " + base);
  const std::string sha = ref.contains("object") ? ref["object"].value("sha", "") : "";
  if (sha.empty()) throw std::runtime_error("Reading " + p.repo + "'s " + base + " failed: no commit");
  // the branch on the fork, at upstream's head (a fork shares its parent's commits; a new fork needs a few seconds)
  if (step) step("creating the branch " + p.branch + " on " + fork);
  std::string last;
  bool made = false;
  for (int attempt = 0; attempt < 12 && !made; ++attempt) {
    try {
      gh_json(gh, {"api", "-X", "POST", "repos/" + fork + "/git/refs", "-f", "ref=refs/heads/" + p.branch, "-f", "sha=" + sha},
              "Creating the branch");
      made = true;
    } catch (const std::exception& e) {
      last = e.what();
      if (last.find("Reference already exists") != std::string::npos) made = true;  // a retry of the same result
      else std::this_thread::sleep_for(std::chrono::seconds(attempt < 4 ? 2 : 5));
    }
  }
  if (!made) throw std::runtime_error(last);
  if (step) step("adding " + p.path);
  {
    InputFile in(ojson{{"message", p.title}, {"branch", p.branch}, {"content", base64(p.content)}});
    gh_json(gh, {"api", "-X", "PUT", "repos/" + fork + "/contents/" + p.path, "--input", in.path}, "Adding the result file");
  }
  if (step) step("opening the pull request");
  const std::string fork_owner = fork.substr(0, fork.find('/'));
  InputFile in(ojson{{"title", p.title},
                     {"head", fork == p.repo ? p.branch : fork_owner + ":" + p.branch},
                     {"base", base},
                     {"body", p.body},
                     {"maintainer_can_modify", fork != p.repo}});
  const ojson pr = gh_json(gh, {"api", "-X", "POST", "repos/" + p.repo + "/pulls", "--input", in.path}, "Opening the pull request");
  return pr.is_object() ? pr.value("html_url", std::string()) : std::string();
}

}  // namespace ni::bench
