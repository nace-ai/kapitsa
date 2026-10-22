// Model downloads: Hugging Face listing, reuse of files already on the Mac, parallel resumable range downloads,
// size + checksum verification, the MTP sidecar builder, GGUF variants' import, the manifest and the store listing
// (see hub.h).
//
// On disk, per model folder:
//   <dir>/<file>                           finished, verified files (companions in <dir>/<subdir>/)
//   <dir>/.kapitsa/parts/<file>.part       a file being downloaded, written at its byte offsets
//   <dir>/.kapitsa/parts/<file>.part.json  which byte ranges of the part are written (resume)
//   <dir>/.kapitsa/manifest.json           written last: the model is complete
// A GGUF variant's folder holds its .gguf file until the import replaces the folder with the conversion.
#include "gguf/gguf.h"
#include "gguf/import.h"
#include "hub/hub.h"
#include "hub/http.h"
#include "util/term.h"

#include <CommonCrypto/CommonDigest.h>
#include <nlohmann/json.hpp>

#include <copyfile.h>
#include <dirent.h>
#include <fcntl.h>
#include <pwd.h>
#include <signal.h>
#include <sys/attr.h>
#include <sys/clonefile.h>
#include <sys/file.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.2.9"
#endif

namespace ni::hub {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

namespace {

constexpr uint64_t kChunk = 64ull << 20;  // one range request; workers take the next one when done
constexpr uint64_t kTailChunk = 8ull << 20, kTail = 512ull << 20;  // the last 512 MB in 8 MB ranges: no lone tail
constexpr int kWorkers = 8;               // parallel connections (8 measured ~4x one stream on this network)
constexpr int kAttempts = 8;              // per range, with backoff up to 15 s, resuming where it stopped

std::atomic<bool> g_stop{false};  // Ctrl+C during a download

// ---------------------------------------------------------------- small helpers

std::string env_str(const char* k) {
  const char* v = getenv(k);
  return v && *v ? v : "";
}

std::string user_home() {
  if (const char* h = getenv("HOME"); h && *h) return h;
  if (const passwd* pw = getpwuid(getuid()); pw && pw->pw_dir) return pw->pw_dir;
  return "/tmp";
}

std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  return s;
}

std::string read_text(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool file_size(const std::string& p, uint64_t& size) {
  struct stat st;
  if (stat(p.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
  size = uint64_t(st.st_size);
  return true;
}

bool is_dir(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string dirname_of(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? "." : s == 0 ? "/" : p.substr(0, s);
}

void mkdirs(const std::string& d) {
  if (d.empty() || is_dir(d)) return;
  mkdirs(dirname_of(d));
  if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST)
    throw Error("Can't create the folder " + tilde(d) + ": " + strerror(errno) + ".");
}

void write_atomic(const std::string& path, const std::string& data) {
  mkdirs(dirname_of(path));
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f << data;
    if (!f) throw Error("Can't write " + tilde(path) + ": " + strerror(errno) + ".");
  }
  if (rename(tmp.c_str(), path.c_str()) != 0) throw Error("Can't write " + tilde(path) + ": " + strerror(errno) + ".");
}

}  // namespace

std::string fmt_bytes(uint64_t b) {
  char s[32];
  if (b >= 999500000ull) snprintf(s, sizeof s, "%.2f GB", double(b) / 1e9);
  else if (b >= 999500ull) snprintf(s, sizeof s, "%.0f MB", double(b) / 1e6);
  else snprintf(s, sizeof s, "%.0f KB", std::ceil(double(b) / 1e3));
  return s;
}

namespace {

std::string fmt_duration(double s) {
  char b[32];
  if (s < 0 || !std::isfinite(s)) return "?";
  if (s < 60) snprintf(b, sizeof b, "%.0fs", std::max(1.0, s));
  else if (s < 3600) snprintf(b, sizeof b, "%dm %02ds", int(s) / 60, int(s) % 60);
  else snprintf(b, sizeof b, "%dh %02dm", int(s) / 3600, int(s) % 3600 / 60);
  return b;
}

std::string endpoint() {
  std::string e = env_str("HF_ENDPOINT");
  if (e.empty()) e = "https://huggingface.co";
  while (!e.empty() && e.back() == '/') e.pop_back();
  return e;
}

std::string endpoint_host() {
  const std::string e = endpoint();
  const size_t p = e.find("://");
  return p == std::string::npos ? e : e.substr(p + 3);
}

// HF_TOKEN, else the token `hf auth login` saved
std::string hf_token() {
  for (const char* k : {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})
    if (std::string v = env_str(k); !v.empty()) return trim(v);
  std::string path = env_str("HF_TOKEN_PATH");
  if (path.empty()) {
    const std::string hf_home = env_str("HF_HOME");
    path = (hf_home.empty() ? user_home() + "/.cache/huggingface" : hf_home) + "/token";
  }
  return trim(read_text(path));
}

Headers hub_headers() {
  Headers h{{"User-Agent", "kapitsa/" KAPITSA_VERSION}};
  if (const std::string t = hf_token(); !t.empty()) h.push_back({"Authorization", "Bearer " + t});
  return h;
}

std::string url_path(const std::string& p) {
  static const char* hexd = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : p) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') {
      o += char(c);
    } else {
      o += '%';
      o += hexd[c >> 4];
      o += hexd[c & 15];
    }
  }
  return o;
}

std::string file_url(const std::string& repo, const std::string& commit, const std::string& path) {
  return endpoint() + "/" + repo + "/resolve/" + commit + "/" + url_path(path);
}

std::string hf_cache_root() {
  for (const char* k : {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"})
    if (std::string v = env_str(k); !v.empty()) return v;
  if (std::string h = env_str("HF_HOME"); !h.empty()) return h + "/hub";
  return user_home() + "/.cache/huggingface/hub";
}

std::string hex(const unsigned char* d, size_t n) {
  static const char* x = "0123456789abcdef";
  std::string s(2 * n, '0');
  for (size_t i = 0; i < n; ++i) {
    s[2 * i] = x[d[i] >> 4];
    s[2 * i + 1] = x[d[i] & 15];
  }
  return s;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// sha256 and/or the git blob id (sha1 of "blob <size>\0" + content, Hugging Face's id of a small file)
bool hash_file(const std::string& path, std::string* sha256, std::string* git_sha1) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  fcntl(fd, F_NOCACHE, 1);
  struct stat st;
  fstat(fd, &st);
  CC_SHA256_CTX s;
  CC_SHA1_CTX g;
  if (sha256) CC_SHA256_Init(&s);
  if (git_sha1) {
    CC_SHA1_Init(&g);
    const std::string h = "blob " + std::to_string(uint64_t(st.st_size));
    CC_SHA1_Update(&g, h.c_str(), CC_LONG(h.size() + 1));  // with its '\0'
  }
  std::vector<unsigned char> buf(8u << 20);
  ssize_t n;
  while ((n = read(fd, buf.data(), buf.size())) > 0) {
    if (sha256) CC_SHA256_Update(&s, buf.data(), CC_LONG(n));
    if (git_sha1) CC_SHA1_Update(&g, buf.data(), CC_LONG(n));
  }
  close(fd);
  if (n < 0) return false;
  if (sha256) {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(d, &s);
    *sha256 = hex(d, sizeof d);
  }
  if (git_sha1) {
    unsigned char d[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1_Final(d, &g);
    *git_sha1 = hex(d, sizeof d);
  }
  return true;
}
#pragma clang diagnostic pop

bool pwrite_all(int fd, const void* data, size_t n, uint64_t off) {
  const char* p = static_cast<const char*>(data);
  while (n) {
    const ssize_t w = pwrite(fd, p, n, off_t(off));
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w;
    n -= size_t(w);
    off += uint64_t(w);
  }
  return true;
}

std::string disk_error(int err, const std::string& path) {
  if (err == ENOSPC) return "The disk is full while writing " + tilde(path) + ". Free some space and run the same command again (finished parts are kept).";
  return "Can't write " + tilde(path) + ": " + strerror(err) + ".";
}

// sleeps `s` seconds unless Ctrl+C
void nap(double s) {
  for (double t = 0; t < s && !g_stop; t += 0.1) std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// ---------------------------------------------------------------- Hugging Face listing

struct RemoteFile {
  std::string path;
  uint64_t size = 0;
  std::string sha256;  // LFS files
  std::string blob;    // git blob id (the content's id for a non-LFS file)
};

struct Listing {
  std::string repo, revision, commit;
  std::vector<RemoteFile> files;
  const RemoteFile* find(const std::string& p) const {
    for (const auto& f : files)
      if (f.path == p) return &f;
    return nullptr;
  }
};

std::string failure(const HttpResult& r, const std::string& repo, const std::string& rev, const std::string& what) {
  const std::string host = endpoint_host();
  if (r.offline)
    return "Can't reach " + host + ": this Mac looks offline. Connect to the internet and run the same command again "
           "(a download picks up where it stopped).";
  if (!r.error.empty())
    return "Can't download " + what + " from " + host + " (" + r.error + "). Run the same command again; finished files are kept.";
  const bool token = !hf_token().empty();
  std::string body = r.body;
  for (auto& c : body) c = char(std::tolower(static_cast<unsigned char>(c)));
  if (r.status == 401 || r.status == 403) {
    if (body.find("not found") != std::string::npos)
      return repo + " was not found on Hugging Face" + (token ? "" : " (or it is private: then set HF_TOKEN)") + ". Check the name.";
    return "Hugging Face refused access to " + repo + " (HTTP " + std::to_string(r.status) + "). " +
           (token ? "Your token (HF_TOKEN) has no access to it: accept its terms on https://huggingface.co/" + repo + "."
                  : "It is gated or private: accept its terms on https://huggingface.co/" + repo +
                        ", then set HF_TOKEN to a token with access (or run `hf auth login`).");
  }
  if (r.status == 404) {
    if (body.find("revision") != std::string::npos || body.find("invalid rev") != std::string::npos)
      return repo + " has no revision \"" + rev + "\" on Hugging Face.";
    return repo + " was not found on Hugging Face (" + what + "). Check the name.";
  }
  if (r.status == 429)
    return std::string("Hugging Face is rate-limiting this Mac (HTTP 429). Wait a minute and run the same command again") +
           (token ? "." : "; a token (HF_TOKEN) raises the limit.");
  if (r.range_ignored) return "The download server ignored a byte-range request for " + what + ". Run the same command again.";
  return "Hugging Face answered HTTP " + std::to_string(r.status) + " for " + what + ". Try again in a minute.";
}

bool retryable(const HttpResult& r) {
  if (!r.error.empty() || r.range_ignored) return true;  // transport errors (offline included: it may come back)
  return r.status == 429 || r.status >= 500 || r.status == 0 || r.status == 200 || r.status == 206;
}

// A range download that keeps failing.
std::string chunk_failure(const HttpResult& r, const std::string& label) {
  if (r.offline || r.status == 401 || r.status == 403 || r.status == 429) return failure(r, label, "", label);
  if (r.range_ignored) return "The download server ignored a byte-range request for " + label + ". Run the same command again.";
  if (!r.error.empty())
    return "The download of " + label + " keeps failing (" + r.error + "). Run the same command again; it continues where it stopped.";
  if (r.status == 404 || r.status == 410) return label + " is no longer on Hugging Face (HTTP " + std::to_string(r.status) + ").";
  return "Hugging Face answered HTTP " + std::to_string(r.status) + " for " + label +
         ". Run the same command again in a minute; it continues where it stopped.";
}

HttpResult get_retry(HttpClient& c, const std::string& url, uint64_t a = 0, uint64_t b = 0, int attempts = 4) {
  for (int i = 1;; ++i) {
    HttpResult r = c.get(url, hub_headers(), a, b);
    if (r.ok()) {
      if (b <= a || (r.status == 206 && r.body.size() == b - a)) return r;
      r.error = "short read";
    }
    // offline: one more try a second later (Wi-Fi waking up), then say so rather than wait
    if (!retryable(r) || i >= attempts || g_stop || (r.offline && i >= 2)) return r;
    nap(0.5 * (1 << i));
  }
}

Listing list_repo(HttpClient& c, const std::string& repo, const std::string& revision) {
  const std::string rev = revision.empty() ? "main" : revision;
  const std::string url = endpoint() + "/api/models/" + repo + "/revision/" + url_path(rev) + "?blobs=true";
  HttpResult r = get_retry(c, url);
  if (g_stop) throw Stopped("Download stopped. Run the same command again to continue where it stopped.");
  if (!r.ok()) throw Error(failure(r, repo, rev, "the file list of " + repo));
  Listing L;
  L.repo = repo;
  L.revision = rev;
  try {
    const json j = json::parse(r.body);
    L.commit = j.value("sha", rev);
    for (const auto& s : j.at("siblings")) {
      RemoteFile f;
      f.path = s.at("rfilename").get<std::string>();
      if (!s.contains("size")) throw std::runtime_error("no sizes");
      f.size = s.at("size").get<uint64_t>();
      f.blob = s.value("blobId", "");
      if (s.contains("lfs") && s["lfs"].is_object()) f.sha256 = s["lfs"].value("sha256", "");
      L.files.push_back(f);
    }
  } catch (const std::exception&) {
    throw Error("Unexpected answer from Hugging Face for " + repo + " (its file list). Try again later.");
  }
  return L;
}

// A GGUF variant's one file, as list_repo would give it: pinned, so no request (a revision is a commit in the catalog)
Listing gguf_listing(const ModelRef& ref) {
  const GgufFile& g = ref.variant->gguf;
  Listing L;
  L.repo = ref.repo;
  L.revision = L.commit = ref.revision;
  L.files.push_back(RemoteFile{g.path, g.size, g.sha256, ""});
  return L;
}

// ---------------------------------------------------------------- download engine

struct Ranges {
  std::vector<std::pair<uint64_t, uint64_t>> v;  // sorted, disjoint
  void add(uint64_t a, uint64_t b) {
    if (a >= b) return;
    v.emplace_back(a, b);
    std::sort(v.begin(), v.end());
    std::vector<std::pair<uint64_t, uint64_t>> m;
    for (const auto& r : v) {
      if (!m.empty() && r.first <= m.back().second) m.back().second = std::max(m.back().second, r.second);
      else m.push_back(r);
    }
    v.swap(m);
  }
  uint64_t covered() const {
    uint64_t n = 0;
    for (const auto& r : v) n += r.second - r.first;
    return n;
  }
  std::vector<std::pair<uint64_t, uint64_t>> missing(uint64_t size) const {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    uint64_t c = 0;
    for (const auto& r : v) {
      if (r.first > c) out.emplace_back(c, std::min(r.first, size));
      c = std::max(c, r.second);
    }
    if (c < size) out.emplace_back(c, size);
    return out;
  }
};

// Bytes [dst, dst + len) of a target come from bytes [src, src + len) at `url`.
struct Piece {
  std::string url;
  uint64_t src = 0, dst = 0, len = 0;
  bool plus_one = false;  // bf16 values + 1 (the sidecar's norm weights): fetched whole, converted, written
};

// One file to produce.
struct Target {
  std::string label, final_path, part_path, state_path;
  uint64_t size = 0;
  std::string sha256, blob;  // expected content (either; both empty: size only)
  std::string id;            // identifies the content a part file holds (a changed listing restarts it)
  std::string head;          // bytes written at offset 0 before any piece (the sidecar's header)
  std::vector<Piece> pieces;
  std::string hf_cached;     // the same file in the Hugging Face cache, if it is there
  // download state
  int fd = -1;
  std::mutex mu;
  Ranges done;
  uint64_t left = 0;
  bool finished = false;
};

void save_state(Target& t) {
  json j;
  {
    std::lock_guard<std::mutex> l(t.mu);
    if (t.finished) return;
    j = json{{"id", t.id}, {"size", t.size}, {"done", t.done.v}};
  }
  try {
    write_atomic(t.state_path, j.dump());
  } catch (const Error&) {
  }
}

uint64_t part_bytes(const Target& t) {  // already downloaded bytes of an unfinished part (no open)
  uint64_t sz;
  if (!file_size(t.part_path, sz) || sz != t.size) return 0;
  try {
    const json j = json::parse(read_text(t.state_path));
    if (j.value("id", "") != t.id || j.value("size", uint64_t(0)) != t.size) return 0;
    Ranges r;
    for (const auto& p : j.at("done")) r.add(p.at(0).get<uint64_t>(), p.at(1).get<uint64_t>());
    return r.covered();
  } catch (...) {
    return 0;
  }
}

void prepare(Target& t) {
  mkdirs(dirname_of(t.part_path));
  t.done = Ranges();
  uint64_t sz = 0;
  bool keep = file_size(t.part_path, sz) && sz == t.size;
  if (keep) {
    try {
      const json j = json::parse(read_text(t.state_path));
      keep = j.value("id", "") == t.id && j.value("size", uint64_t(0)) == t.size;
      if (keep)
        for (const auto& p : j.at("done")) t.done.add(p.at(0).get<uint64_t>(), p.at(1).get<uint64_t>());
    } catch (...) {
      keep = false;
    }
  }
  if (!keep) {
    unlink(t.part_path.c_str());
    t.done = Ranges();
  }
  t.fd = open(t.part_path.c_str(), O_RDWR | O_CREAT, 0644);
  if (t.fd < 0) throw Error(disk_error(errno, t.part_path));
  if (ftruncate(t.fd, off_t(t.size)) != 0) throw Error(disk_error(errno, t.part_path));
  if (!t.head.empty() && t.done.covered() < t.head.size()) {
    if (!pwrite_all(t.fd, t.head.data(), t.head.size(), 0)) throw Error(disk_error(errno, t.part_path));
    t.done.add(0, t.head.size());
  }
  t.left = t.size - t.done.covered();
  t.finished = false;
}

// bf16 values + 1, exactly as tools/mtp_sidecar.py's bf16_plus_one: bf16 -> float32, + 1 in double precision, back
// to float32 (round to nearest even), then to bf16 rounding to nearest even.
std::string plus_one(const std::string& raw) {
  std::string out(raw.size(), '\0');
  for (size_t i = 0; i + 1 < raw.size(); i += 2) {
    const uint32_t h = uint32_t(uint8_t(raw[i])) | uint32_t(uint8_t(raw[i + 1])) << 8;
    const uint32_t bits = h << 16;
    float f;
    memcpy(&f, &bits, 4);
    const float g = float(double(f) + 1.0);
    uint32_t u;
    memcpy(&u, &g, 4);
    const uint64_t v = (uint64_t(u) + 0x7FFF + ((u >> 16) & 1)) >> 16;
    out[i] = char(v & 0xFF);
    out[i + 1] = char((v >> 8) & 0xFF);
  }
  return out;
}

// Verifies a finished part and moves it into place.
bool finalize(Target& t, std::string& err) {
  if (t.fd >= 0) close(t.fd);
  t.fd = -1;
  uint64_t sz = 0;
  if (!file_size(t.part_path, sz) || sz != t.size) {
    err = t.label + ": size mismatch after download. Run the same command again.";
    return false;
  }
  if (!t.sha256.empty() || !t.blob.empty()) {
    std::string s256, s1;
    const bool ok = hash_file(t.part_path, t.sha256.empty() ? nullptr : &s256, t.sha256.empty() ? &s1 : nullptr);
    if (!ok || (!t.sha256.empty() ? s256 != t.sha256 : s1 != t.blob)) {
      unlink(t.part_path.c_str());
      unlink(t.state_path.c_str());
      err = t.label + " arrived corrupted (checksum mismatch) and was deleted. Run the same command again to download it again.";
      return false;
    }
  }
  try {
    mkdirs(dirname_of(t.final_path));
  } catch (const Error& e) {
    err = e.what();
    return false;
  }
  if (rename(t.part_path.c_str(), t.final_path.c_str()) != 0) {
    err = disk_error(errno, t.final_path);
    return false;
  }
  unlink(t.state_path.c_str());
  return true;
}

struct Progress {
  std::string title;
  uint64_t total = 0;
  std::atomic<uint64_t> done{0};
  bool show = true, tty = false;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::deque<std::pair<double, uint64_t>> samples;
  double last_line = -1e9;
  double secs() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
  void print(bool final) {
    if (!show) return;
    if (term::fancy()) return draw(final);
    const double now = secs();
    const uint64_t d = std::min(done.load(), total);
    samples.emplace_back(now, d);
    while (samples.size() > 2 && now - samples.front().first > 8) samples.pop_front();
    const double dt = now - samples.front().first;
    const double rate = dt > 0.5 ? double(d - samples.front().second) / dt : 0;
    const double eta = rate > 0 ? double(total - d) / rate : -1;
    char line[160];
    snprintf(line, sizeof line, "  %s of %s  %3d%%  %5.1f MB/s  %s left", fmt_bytes(d).c_str(),
             fmt_bytes(total).c_str(), total ? int(100.0 * double(d) / double(total)) : 100, rate / 1e6,
             final ? "0s" : fmt_duration(eta).c_str());
    if (tty) {
      fprintf(stderr, "\r\033[K%s", line);
      if (final) fprintf(stderr, "\n");
    } else if (final || now - last_line >= 15) {
      fprintf(stderr, "%s\n", line);
      last_line = now;
    }
    fflush(stderr);
  }
  // In a terminal: a bar with the share done, the bytes, the speed and the time left, on the status line
  void draw(bool final) {
    if (final) return term::status("");
    const double now = secs();
    const uint64_t d = std::min(done.load(), total);
    samples.emplace_back(now, d);
    while (samples.size() > 2 && now - samples.front().first > 8) samples.pop_front();
    const double dt = now - samples.front().first;
    const double rate = dt > 0.5 ? double(d - samples.front().second) / dt : 0;
    const double frac = total ? double(d) / double(total) : 1;
    char pct[16], sp[32];
    snprintf(pct, sizeof pct, "%3d%%", int(100 * frac));
    snprintf(sp, sizeof sp, "%.1f MB/s", rate / 1e6);
    std::string tail = std::string(pct) + "  " + fmt_bytes(d) + term::dim(" of " + fmt_bytes(total));
    const int w = term::columns();
    if (w >= 70) tail += "  " + term::dim(sp) + "  " + term::dim(rate > 0 ? fmt_duration(double(total - d) / rate) + " left" : "");
    const int bw = std::clamp(w - 6 - term::width(tail) - 3, 8, 40);
    term::status("    " + term::bar(frac, bw) + "  " + tail);
  }
};

struct Chunk {
  Target* t;
  const Piece* p;
  uint64_t off, len;  // [off, off + len) of the piece
};

// One range, retried from where it stopped. False with `err` set when it cannot be completed.
bool fetch_chunk(HttpClient& c, const Headers& h, const Chunk& k, Progress& prog, const std::atomic<bool>& failed,
                 std::string& err) {
  Target& t = *k.t;
  const Piece& p = *k.p;
  uint64_t got = 0;
  for (int attempt = 1;; ++attempt) {
    if (g_stop || failed) return false;
    const uint64_t a = p.src + k.off + got, b = p.src + k.off + k.len;
    HttpResult r;
    std::string werr;
    if (p.plus_one) {
      r = c.get(p.url, h, a, b);
      if (r.ok() && r.status == 206 && r.body.size() == k.len) {
        const std::string v = plus_one(r.body);
        if (!pwrite_all(t.fd, v.data(), v.size(), p.dst + k.off)) {
          err = disk_error(errno, t.final_path);
          return false;
        }
        {
          std::lock_guard<std::mutex> l(t.mu);
          t.done.add(p.dst + k.off, p.dst + k.off + k.len);
          t.left -= k.len;
        }
        prog.done += k.len;
        return true;
      }
    } else {
      uint64_t pos = p.dst + k.off + got;
      const uint64_t end = p.dst + k.off + k.len;
      auto sink = [&](const uint8_t* d, size_t n) -> bool {
        if (g_stop || failed) return false;
        const size_t m = size_t(std::min<uint64_t>(n, end - pos));
        if (m && !pwrite_all(t.fd, d, m, pos)) {
          werr = disk_error(errno, t.final_path);
          return false;
        }
        {
          std::lock_guard<std::mutex> l(t.mu);
          t.done.add(pos, pos + m);
          t.left -= m;
        }
        pos += m;
        got += m;
        prog.done += m;
        return pos < end;
      };
      r = c.stream(p.url, h, a, b, sink);
      if (!werr.empty()) {
        err = werr;
        return false;
      }
      if (got == k.len) return true;
    }
    if (g_stop || failed) return false;
    if (!retryable(r) || attempt >= kAttempts) {
      err = chunk_failure(r, t.label);
      return false;
    }
    nap(std::min(15.0, 0.5 * double(1 << attempt)));
  }
}

// Downloads every target (8 connections, 64 MB ranges, small files first), verifying each as it completes.
void download(std::vector<Target*>& todo, Progress& prog) {
  std::deque<Chunk> q;
  std::stable_sort(todo.begin(), todo.end(), [](const Target* a, const Target* b) { return a->size < b->size; });
  uint64_t total = 0;
  for (Target* t : todo) {
    prepare(*t);
    total += t->left;
  }
  // In queue order (small files first), 64 MB ranges, then 8 MB ones for the last 512 MB: one connection alone on a
  // big last range would hold the end of the download (a few MB/s per connection on a high-latency link).
  uint64_t queued = 0;
  for (Target* t : todo) {
    const auto miss = t->done.missing(t->size);
    for (const Piece& p : t->pieces) {
      bool whole = false;
      for (const auto& [a, b] : miss) {
        const uint64_t lo = std::max(a, p.dst), hi = std::min(b, p.dst + p.len);
        if (lo >= hi) continue;
        if (p.plus_one) {  // small: always the whole piece, written at once
          whole = true;
          break;
        }
        for (uint64_t x = lo; x < hi;) {
          const uint64_t step = queued + kChunk + kTail > total ? kTailChunk : kChunk;
          const uint64_t n = std::min(step, hi - x);
          q.push_back({t, &p, x - p.dst, n});
          x += n;
          queued += n;
        }
      }
      if (whole) {  // written at once, so never partly done
        q.push_back({t, &p, 0, p.len});
        queued += p.len;
      }
    }
  }
  prog.total = total;  // what is fetched now (a part's header written above and resumed ranges are not)
  std::mutex qm;
  std::atomic<bool> failed{false};
  std::string first_error;
  auto fail = [&](const std::string& e) {
    std::lock_guard<std::mutex> l(qm);
    if (!failed) first_error = e;
    failed = true;
  };
  auto worker = [&] {
    HttpClient c(60);
    const Headers h = hub_headers();
    for (;;) {
      Chunk k;
      {
        std::lock_guard<std::mutex> l(qm);
        if (q.empty() || failed || g_stop) return;
        k = q.front();
        q.pop_front();
      }
      std::string err;
      if (!fetch_chunk(c, h, k, prog, failed, err)) {
        if (!err.empty()) fail(err);
        return;
      }
      bool last = false;
      {
        std::lock_guard<std::mutex> l(k.t->mu);
        last = k.t->left == 0 && !k.t->finished;
        if (last) k.t->finished = true;
      }
      if (last && !finalize(*k.t, err)) {
        fail(err);
        return;
      }
    }
  };
  // a target with nothing left to fetch (a resumed part that was complete)
  for (Target* t : todo)
    if (t->left == 0 && !t->finished) {
      t->finished = true;
      std::string err;
      if (!finalize(*t, err)) throw Error(err);
    }
  std::vector<std::thread> th;
  const int n = int(std::min<size_t>(kWorkers, std::max<size_t>(1, q.size())));
  std::atomic<int> running{n};
  for (int i = 0; i < n; ++i)
    th.emplace_back([&] {
      worker();
      --running;
    });
  double last_save = prog.secs();
  while (running > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    prog.print(false);
    if (prog.secs() - last_save > 2) {
      for (Target* t : todo) save_state(*t);
      last_save = prog.secs();
    }
  }
  for (auto& x : th) x.join();
  for (Target* t : todo) {
    save_state(*t);
    if (t->fd >= 0) close(t->fd);
    t->fd = -1;
  }
  if (g_stop) throw Stopped("Download stopped. Run the same command again to continue where it stopped.");
  if (failed) throw Error(first_error);
  prog.print(true);
}

// Ctrl+C stops a download cleanly (parts and their state are kept); a second one quits at once.
struct SignalGuard {
  struct sigaction old_int {}, old_term {};
  SignalGuard() {
    g_stop = false;
    struct sigaction sa {};
    sa.sa_handler = [](int) {
      if (g_stop) _exit(130);
      g_stop = true;
    };
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, &old_int);
    sigaction(SIGTERM, &sa, &old_term);
  }
  ~SignalGuard() {
    sigaction(SIGINT, &old_int, nullptr);
    sigaction(SIGTERM, &old_term, nullptr);
  }
};

// One download at a time per model folder.
struct FolderLock {
  int fd = -1;
  explicit FolderLock(const std::string& dir) {
    mkdirs(dir + "/.kapitsa");
    fd = open((dir + "/.kapitsa/lock").c_str(), O_RDWR | O_CREAT, 0644);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) != 0)
      throw Error("Another kapitsa is downloading into " + tilde(dir) + ". Wait for it to finish, or stop it first.");
  }
  ~FolderLock() {
    if (fd >= 0) close(fd);
  }
};

// ---------------------------------------------------------------- reuse of files already on this Mac

struct FromIndex {
  std::multimap<uint64_t, std::string> by_size;
  void scan(const std::string& dir, int depth) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] == '.') continue;
      const std::string p = dir + "/" + e->d_name;
      struct stat st;
      if (stat(p.c_str(), &st) != 0) continue;
      if (S_ISDIR(st.st_mode)) {
        if (depth > 0) scan(p, depth - 1);
      } else if (S_ISREG(st.st_mode)) {
        by_size.emplace(uint64_t(st.st_size), p);
      }
    }
    closedir(d);
  }
};

// The path of a file with the target's content, from the Hugging Face cache or the --from folders; "" if none.
std::string reusable(const Target& t, const FromIndex& from, bool& from_hf) {
  from_hf = false;
  if (!t.hf_cached.empty()) {
    struct stat st;
    if (lstat(t.hf_cached.c_str(), &st) == 0) {
      uint64_t sz = 0;
      if (file_size(t.hf_cached, sz) && sz == t.size) {
        // the cache names each blob by its sha256 (LFS) or git blob id: the link proves the content
        bool ok = false;
        if (S_ISLNK(st.st_mode)) {
          char buf[4096];
          const ssize_t n = readlink(t.hf_cached.c_str(), buf, sizeof buf - 1);
          if (n > 0) {
            buf[n] = 0;
            const std::string target = buf, base = target.substr(target.find_last_of('/') + 1);
            ok = base == (t.sha256.empty() ? t.blob : t.sha256);
          }
        }
        if (!ok) {
          std::string s256, s1;
          ok = hash_file(t.hf_cached, t.sha256.empty() ? nullptr : &s256, t.sha256.empty() ? &s1 : nullptr) &&
               (t.sha256.empty() ? s1 == t.blob : s256 == t.sha256);
        }
        if (ok) {
          from_hf = true;
          return t.hf_cached;
        }
      }
    }
  }
  if (t.sha256.empty() && t.blob.empty()) return "";  // nothing to check a candidate against
  const auto range = from.by_size.equal_range(t.size);
  for (auto it = range.first; it != range.second; ++it) {
    std::string s256, s1;
    if (hash_file(it->second, t.sha256.empty() ? nullptr : &s256, t.sha256.empty() ? &s1 : nullptr) &&
        (t.sha256.empty() ? s1 == t.blob : s256 == t.sha256))
      return it->second;
  }
  return "";
}

dev_t device_of(const std::string& path) {
  std::string p = path;
  struct stat st;
  while (stat(p.c_str(), &st) != 0) {
    if (p == "/" || p == ".") return 0;
    p = dirname_of(p);
  }
  return st.st_dev;
}

// An APFS clone (no extra disk) of `src` at `dst`, or a copy across volumes.
void place(const std::string& src, const std::string& dst) {
  char real[PATH_MAX];
  const std::string from = realpath(src.c_str(), real) ? std::string(real) : src;
  mkdirs(dirname_of(dst));
  const std::string tmp = dst + ".kapitsa-tmp";
  unlink(tmp.c_str());
  if (clonefile(from.c_str(), tmp.c_str(), 0) != 0 && copyfile(from.c_str(), tmp.c_str(), nullptr, COPYFILE_DATA) != 0) {
    const int e = errno;
    unlink(tmp.c_str());
    throw Error(disk_error(e, dst));
  }
  if (rename(tmp.c_str(), dst.c_str()) != 0) throw Error(disk_error(errno, dst));
}

// The bytes of a file no APFS clone shares (getattrlist's ATTR_CMNEXT_PRIVATESIZE), or -1 where the volume does not say
static int64_t private_size(const std::string& path) {
  struct attrlist al {};
  al.bitmapcount = ATTR_BIT_MAP_COUNT;
  al.forkattr = ATTR_CMNEXT_PRIVATESIZE;
  struct __attribute__((packed)) {
    uint32_t length;
    off_t size;
  } buf {};
  if (getattrlist(path.c_str(), &al, &buf, sizeof buf, FSOPT_NOFOLLOW | FSOPT_ATTR_CMN_EXTENDED) != 0) return -1;
  return buf.size;
}

uint64_t free_bytes(const std::string& path) {
  std::string p = path;
  struct statfs s;
  while (statfs(p.c_str(), &s) != 0) {
    if (p == "/" || p == ".") return UINT64_MAX;
    p = dirname_of(p);
  }
  const uint64_t f = uint64_t(s.f_bavail) * uint64_t(s.f_bsize);
  // tests: this many bytes of the free space count as taken, to reach the out-of-space paths without filling a disk
  const char* taken = getenv("KAPITSA_TEST_DISK_TAKEN");
  const uint64_t t = taken && *taken ? strtoull(taken, nullptr, 10) : 0;
  return f > t ? f - t : 0;
}

// ---------------------------------------------------------------- MTP sidecar (tools/mtp_sidecar.py)

uint64_t le64(const std::string& b) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | uint8_t(b[size_t(i)]);
  return v;
}

std::string le64_bytes(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[size_t(i)] = char((v >> (8 * i)) & 0xFF);
  return s;
}

bool is_norm(const std::string& k) {
  auto ends = [&](const char* s) {
    const size_t n = strlen(s);
    return k.size() >= n && k.compare(k.size() - n, n, s) == 0;
  };
  return ends("norm.weight") || ends("norm_embedding.weight") || ends("norm_hidden.weight");
}

// Where the checkpoint's bytes come from: a Hugging Face repo at a revision, or a local folder.
struct SidecarSource {
  bool local = false;
  std::string src, rev;
  HttpClient* http = nullptr;
  std::string url(const std::string& file) const {
    return local ? src + "/" + file : file_url(src, rev, file);
  }
  std::string read(const std::string& file, uint64_t a, uint64_t b) const {
    if (local) {
      std::ifstream f(src + "/" + file, std::ios::binary);
      std::string out(b - a, '\0');
      f.seekg(std::streamoff(a));
      if (!f.read(out.data(), std::streamsize(b - a))) throw Error("Can't read " + src + "/" + file + ".");
      return out;
    }
    HttpResult r = get_retry(*http, url(file), a, b);
    if (g_stop) throw Stopped("Download stopped. Run the same command again to continue where it stopped.");
    if (!r.ok() || r.status != 206) throw Error(failure(r, src, rev, file + " of " + src));
    return r.body;
  }
  json header(const std::string& file, uint64_t& data0) const {
    const uint64_t n = le64(read(file, 0, 8));
    if (n > (64u << 20)) throw Error(file + " of " + src + " is not a safetensors file.");
    data0 = 8 + n;
    return json::parse(read(file, 8, 8 + n));
  }
  std::map<std::string, std::string> weight_map() const {
    std::string text;
    if (local) {
      text = read_text(src + "/model.safetensors.index.json");
      if (text.empty()) {  // no index: every *.safetensors file's tensors
        std::map<std::string, std::string> wm;
        if (DIR* d = opendir(src.c_str())) {
          while (dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) {
              uint64_t d0;
              const json h = header(n, d0);  // named: a temporary would die before the loop runs
              for (const auto& [k, v] : h.items()) wm[k] = n;
            }
          }
          closedir(d);
        }
        return wm;
      }
    } else {
      HttpResult r = get_retry(*http, url("model.safetensors.index.json"));
      if (g_stop) throw Stopped("Download stopped. Run the same command again to continue where it stopped.");
      if (!r.ok()) throw Error(failure(r, src, rev, "the weight index of " + src));
      text = r.body;
    }
    std::map<std::string, std::string> wm;
    const json index = json::parse(text);
    for (const auto& [k, v] : index.at("weight_map").items()) wm[k] = v.get<std::string>();
    return wm;
  }
};

struct SidecarPlan {
  std::string head;  // 8-byte length + JSON header
  std::vector<Piece> pieces;
  uint64_t size = 0;
};

SidecarPlan plan_sidecar(const SidecarSource& s, const std::string& source_name, const std::string& hidden) {
  std::map<std::string, std::vector<std::string>> shards;  // file -> mtp.* tensors, files in name order
  for (const auto& [k, f] : s.weight_map())
    if (k.rfind("mtp.", 0) == 0) shards[f].push_back(k);
  if (shards.empty()) throw Error(source_name + " has no mtp.* tensors.");
  struct T {
    std::string file;
    json shape;
    uint64_t a, b;
  };
  std::map<std::string, T> tensors;  // by name: the file's order
  for (auto& [file, keys] : shards) {
    uint64_t data0 = 0;
    const json hdr = s.header(file, data0);
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys) {
      const json& t = hdr.at(k);
      if (t.at("dtype") != "BF16") throw Error(k + ": expected bf16, found " + t.at("dtype").get<std::string>() + ".");
      const uint64_t a = t.at("data_offsets").at(0), b = t.at("data_offsets").at(1);
      tensors[k] = T{file, t.at("shape"), data0 + a, data0 + b};
    }
  }
  ojson meta;
  meta["__metadata__"] = ojson{{"format", "mlx"}, {"source", source_name}, {"norms", "mlx (+1)"}, {"base_hidden_variant", hidden}};
  uint64_t off = 0;
  for (const auto& [k, t] : tensors) {
    meta[k] = ojson{{"dtype", "BF16"}, {"shape", t.shape}, {"data_offsets", ojson::array({off, off + (t.b - t.a)})}};
    off += t.b - t.a;
  }
  std::string h = meta.dump(-1, ' ', true);  // Python's json.dumps(separators=(",", ":")), ASCII
  h.append((8 - h.size() % 8) % 8, ' ');
  SidecarPlan p;
  p.head = le64_bytes(h.size()) + h;
  uint64_t dst = p.head.size();
  for (const auto& [k, t] : tensors) {
    p.pieces.push_back(Piece{s.url(t.file), t.a, dst, t.b - t.a, is_norm(k)});
    dst += t.b - t.a;
  }
  p.size = dst;
  return p;
}

std::string plan_id(const SidecarPlan& p) {
  std::string all = p.head;
  for (const auto& x : p.pieces) all += x.url + ":" + std::to_string(x.src) + ":" + std::to_string(x.len) + (x.plus_one ? "+" : "") + "\n";
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(all.data(), CC_LONG(all.size()), d);
  return hex(d, sizeof d);
}

// ---------------------------------------------------------------- manifest

std::string manifest_path(const std::string& dir) { return dir + "/.kapitsa/manifest.json"; }

std::string now_iso() {
  char b[32];
  const time_t t = time(nullptr);
  struct tm g;
  gmtime_r(&t, &g);
  strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &g);
  return b;
}

// A Hugging Face repo that Kapitsa can run: a Qwen3.5-family hybrid model in MLX format, unquantized (bf16) or MLX
// affine 3/4/5/6/8-bit with groups of 64 (4-bit also 32), one format or mixed per module (the engine's
// format_supported, model/qwen35.cpp); or Recursive's Flint in its MLX 4-bit export (model/flint.h).
void check_supported(HttpClient& c, const Listing& L) {
  const RemoteFile* cfg = L.find("config.json");
  bool st = false;
  for (const auto& f : L.files)
    if (f.path.size() > 12 && f.path.compare(f.path.size() - 12, 12, ".safetensors") == 0) st = true;
  const std::string runs = " Kapitsa runs Qwen3.5-family models (Qwen3.5, 3.6, 3.8) in MLX format: bf16, or MLX affine "
                           "3/4/5/6/8-bit.";
  if (!cfg || !st) throw Error(L.repo + " has no " + (cfg ? "safetensors weights" : "config.json") + "." + runs);
  HttpResult r = get_retry(c, file_url(L.repo, L.commit, "config.json"));
  if (g_stop) throw Stopped("Download stopped. Run the same command again to continue where it stopped.");
  if (!r.ok()) throw Error(failure(r, L.repo, L.revision, "config.json of " + L.repo));
  json j;
  try {
    j = json::parse(r.body);
  } catch (...) {
    throw Error(L.repo + ": its config.json is not valid JSON.");
  }
  const json t = j.contains("text_config") ? j["text_config"] : j;
  if (j.value("model_type", std::string()) == "flint") {
    const json q = j.value("quantization", json::object());
    if (!q.is_object() || q.value("bits", 0) != 4 || q.value("group_size", 0) != 64)
      throw Error(L.repo + " is a Flint checkpoint Kapitsa does not run: it runs Flint's MLX 4-bit export, "
                           "RecursiveSI/flint-int4 (`kapitsa pull flint`).");
    return;
  }
  if (!t.contains("linear_num_value_heads") || !t.contains("num_hidden_layers"))
    throw Error(L.repo + " is a " + j.value("model_type", std::string("different")) + " model." + runs);
  const json q = j.value("quantization", json::object());
  if (!q.is_object() || q.empty()) return;  // unquantized (bf16): runs as is
  // the default format and every per-module one (mixed recipes)
  auto check = [&](const json& e) {
    const int bits = e.value("bits", q.value("bits", 0)), gs = e.value("group_size", q.value("group_size", 64));
    const std::string mode = e.value("mode", q.value("mode", std::string("affine")));
    const bool ok = mode == "affine" && ((gs == 64 && (bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8)) ||
                                         (gs == 32 && bits == 4));
    if (!ok)
      throw Error(L.repo + " is quantized as " + mode + " " + std::to_string(bits) + "-bit, group " + std::to_string(gs) +
                  "." + runs + " Try its MLX 4-bit, 6-bit, 8-bit or bf16 conversion.");
  };
  check(q);
  for (auto& [k, v] : q.items())
    if (v.is_object()) check(v);
}

bool keep_file(const std::string& p) {  // files of a generic repo worth fetching: the top level, no other formats
  if (p.find('/') != std::string::npos || p == ".gitattributes") return false;
  for (const char* x : {".gguf", ".bin", ".pt", ".pth", ".onnx", ".h5", ".msgpack", ".ckpt", ".png", ".jpg", ".jpeg",
                        ".gif", ".webp", ".mp4"}) {
    const size_t n = strlen(x);
    if (p.size() >= n && p.compare(p.size() - n, n, x) == 0) return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------- public

std::string sha256_file(const std::string& path) {
  std::string s;
  if (!hash_file(path, &s, nullptr)) throw Error("Can't read " + tilde(path) + ".");
  return s;
}

bool is_complete(const std::string& dir) {
  try {
    const json m = json::parse(read_text(manifest_path(dir)));
    for (const auto& f : m.at("files")) {
      uint64_t sz;
      if (!file_size(dir + "/" + f.at("path").get<std::string>(), sz) || sz != f.at("size").get<uint64_t>()) return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool is_current(const ModelRef& ref) {
  if (ref.kind == ModelRef::kLocal) return true;
  if (!is_complete(ref.dir)) return false;
  if (ref.kind != ModelRef::kCatalog) return true;
  // a catalog model: the files of the revisions this Kapitsa pins (a newer Kapitsa may pin newer ones)
  try {
    const json m = json::parse(read_text(manifest_path(ref.dir)));
    std::vector<std::string> have, want;
    for (const auto& c : m.at("companions")) have.push_back(c.at("kind").get<std::string>() + " " + c.at("repo").get<std::string>() + "@" + c.at("revision").get<std::string>());
    for (const auto& c : ref.variant->companions) want.push_back(c.kind + " " + c.repo + "@" + c.revision);
    std::sort(have.begin(), have.end());
    std::sort(want.begin(), want.end());
    // a GGUF variant: the conversion of the pinned file
    const std::string& gguf = ref.variant->gguf.sha256;
    if (!gguf.empty() && (!m.contains("gguf") || m["gguf"].value("sha256", "") != gguf)) return false;
    return m.at("revision").get<std::string>() == ref.revision && have == want;
  } catch (...) {
    return false;
  }
}

namespace {

// After a failed first attempt: drop a model folder that holds nothing yet (a mistyped repo leaves no trace).
void remove_if_empty(const std::string& dir) {
  bool content = false;
  std::vector<std::string> stack{dir};
  while (!stack.empty() && !content) {
    const std::string d = stack.back();
    stack.pop_back();
    if (DIR* dd = opendir(d.c_str())) {
      while (dirent* e = readdir(dd)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        const std::string p = d + "/" + n;
        if (is_dir(p)) stack.push_back(p);
        else if (!(d == dir + "/.kapitsa" && n == "lock")) content = true;
      }
      closedir(dd);
    }
  }
  if (!content) {
    unlink((dir + "/.kapitsa/lock").c_str());
    rmdir((dir + "/.kapitsa/parts").c_str());
    rmdir((dir + "/.kapitsa").c_str());
    rmdir(dir.c_str());
  }
}

std::string ensure_impl(const ModelRef& ref, const PullOptions& opt) {
  for (const auto& d : opt.from) {
    uint64_t sz;
    if (!is_dir(d) && !file_size(d, sz)) throw Error("--from " + d + ": no such folder or file.");
  }
  SignalGuard sig;
  // A folder Kapitsa did not download (no .kapitsa/: put there by hand or by another tool, e.g. a bundle that carries
  // the same weights): its files are checked against the pinned checksums and kept where they match, so only the files
  // that differ are fetched.
  const bool adopt = is_dir(ref.dir) && !is_dir(ref.dir + "/.kapitsa");
  mkdirs(ref.dir);
  FolderLock lock(ref.dir);
  if (is_current(ref)) return ref.dir;  // another kapitsa just finished it
  HttpClient api(30);
  // a GGUF variant: its pinned file, then the import (no file list to fetch)
  const bool gguf = ref.kind == ModelRef::kCatalog && !ref.variant->gguf.path.empty();
  if (gguf && !ref.variant->companions.empty()) throw Error("internal: " + ref.name + " is a GGUF variant with companions");
  // in a terminal: a spinner while the file lists come and files already here are checked
  std::unique_ptr<term::Spinner> spin;
  if (opt.progress && term::fancy())
    spin = std::make_unique<term::Spinner>(gguf ? "Preparing " + term::bold(ref.name) + term::dim(" (" + ref.variant->gguf.path + ")")
                                                : "Fetching the file list of " + term::bold(ref.name) + term::dim(" from " + endpoint_host()));
  else if (opt.progress && !gguf) fprintf(stderr, "Fetching the file list of %s from %s...\n", ref.name.c_str(), endpoint_host().c_str());

  const Listing main = gguf ? gguf_listing(ref) : list_repo(api, ref.repo, ref.revision);
  if (ref.kind == ModelRef::kRepo) check_supported(api, main);

  std::vector<std::unique_ptr<Target>> targets;
  ojson mfiles = ojson::array(), mcomp = ojson::array();
  uint64_t weights_bytes = 0;
  auto add = [&](const Listing& L, const RemoteFile& f, const std::string& subdir) {
    auto t = std::make_unique<Target>();
    const std::string rel = subdir.empty() ? f.path : subdir + "/" + f.path;
    t->label = rel;
    t->size = f.size;
    t->sha256 = f.sha256;
    t->blob = f.sha256.empty() ? f.blob : "";
    t->id = !f.sha256.empty() ? f.sha256 : f.blob;
    t->final_path = ref.dir + "/" + rel;
    t->part_path = ref.dir + "/.kapitsa/parts/" + rel + ".part";
    t->state_path = t->part_path + ".json";
    t->pieces.push_back(Piece{file_url(L.repo, L.commit, f.path), 0, 0, f.size, false});
    std::string cache_repo = L.repo;
    for (size_t p; (p = cache_repo.find('/')) != std::string::npos;) cache_repo.replace(p, 1, "--");
    t->hf_cached = hf_cache_root() + "/models--" + cache_repo + "/snapshots/" + L.commit + "/" + f.path;
    ojson e{{"path", rel}, {"size", f.size}};
    if (!f.sha256.empty()) e["sha256"] = f.sha256;
    mfiles.push_back(e);
    targets.push_back(std::move(t));
  };
  for (const auto& f : main.files) {
    if (ref.kind == ModelRef::kCatalog ? f.path == ".gitattributes" : !keep_file(f.path)) continue;
    add(main, f, "");
    weights_bytes += f.size;
  }
  std::vector<std::string> extras;
  if (ref.kind == ModelRef::kCatalog) {
    for (const Companion& c : ref.variant->companions) {
      if (c.kind == "mtp-sidecar") {
        SidecarSource s;
        s.src = c.repo;
        s.rev = c.revision;
        s.http = &api;
        const SidecarPlan p = plan_sidecar(s, c.repo, "post_norm");
        auto t = std::make_unique<Target>();
        const std::string rel = c.subdir.empty() ? "mtp.safetensors" : c.subdir + "/mtp.safetensors";
        t->label = rel + " (MTP head built from " + c.repo + ")";
        t->size = p.size;
        t->sha256 = c.sha256;
        t->id = c.sha256.empty() ? plan_id(p) : c.sha256;
        t->head = p.head;
        t->pieces = p.pieces;
        t->final_path = ref.dir + "/" + rel;
        t->part_path = ref.dir + "/.kapitsa/parts/" + rel + ".part";
        t->state_path = t->part_path + ".json";
        if (c.size && c.size != p.size)
          throw Error("The MTP head of " + ref.name + " would be " + std::to_string(p.size) + " bytes, not the expected " +
                      std::to_string(c.size) + ": " + c.repo + " changed. Please report this.");
        ojson e{{"path", rel}, {"size", p.size}};
        if (!c.sha256.empty()) e["sha256"] = c.sha256;
        mfiles.push_back(e);
        mcomp.push_back(ojson{{"kind", c.kind}, {"repo", c.repo}, {"revision", c.revision}, {"files", ojson::array({rel})}});
        targets.push_back(std::move(t));
        extras.push_back("MTP head " + fmt_bytes(p.size));
        continue;
      }
      const Listing L = list_repo(api, c.repo, c.revision);
      ojson cf = ojson::array();
      uint64_t cbytes = 0;
      for (const auto& name : c.files) {
        const RemoteFile* f = L.find(name);
        if (!f) throw Error(c.repo + " has no " + name + " at " + c.revision.substr(0, 7) + ". Please report this.");
        add(L, *f, c.subdir);
        cf.push_back(c.subdir.empty() ? name : c.subdir + "/" + name);
        cbytes += f->size;
      }
      mcomp.push_back(ojson{{"kind", c.kind}, {"repo", c.repo}, {"revision", c.revision}, {"commit", L.commit}, {"files", cf}});
      extras.push_back((c.kind == "dflash2"    ? "DFlash2 drafter "
                        : c.kind == "dflash"   ? "DFlash drafter "
                        : c.kind == "lilicorr" ? "LiLiCorr drafter "
                        : c.kind == "dspark"   ? "DSpark drafter "
                                               : "MTP head ") +
                       fmt_bytes(cbytes));
    }
  }

  // what is already here: finished files, the Hugging Face cache, --from folders (or files), and the Nace Agent
  // app's copy of a catalog model (~/.nace/models/<key>, companions included): exact checksum matches only
  FromIndex from;
  for (const auto& d : opt.from) {
    uint64_t sz;
    if (is_dir(d)) from.scan(d, 3);
    else if (file_size(d, sz)) from.by_size.emplace(sz, d);
  }
  if (ref.kind == ModelRef::kCatalog) {
    const std::string app = user_home() + "/.nace/models/" + ref.model->name + "-" + ref.variant->name;
    if (is_dir(app)) {
      from.scan(app, 3);
      if (spin) spin->label("Checking Nace Agent's copy in " + tilde(app));
      else if (opt.progress) fprintf(stderr, "Checking Nace Agent's copy in %s...\n", tilde(app).c_str());
    }
  }
  if (spin && !opt.from.empty())
    spin->label("Checking the files in " + (opt.from.size() == 1 ? tilde(opt.from[0]) : std::string("the --from folders")) +
                " against the pinned checksums");
  else if (opt.progress && !opt.from.empty()) fprintf(stderr, "Checking the files in %s against the pinned checksums...\n",
                                                  opt.from.size() == 1 ? tilde(opt.from[0]).c_str() : "the --from folders");
  const dev_t dest_dev = device_of(ref.dir);
  std::vector<Target*> todo;
  std::vector<std::pair<std::string, Target*>> reuse;
  uint64_t need = 0, total = 0, have = 0;
  int n_hf = 0, n_from = 0;
  // A manifest that is no longer current (a file went missing, or this Kapitsa pins newer revisions): drop it now
  // that the new file list is in, check the files that are here against it, and write it again at the end. The
  // files of an adopted folder are checked the same way. The checks run in parallel (8 files at a time).
  const bool recheck = unlink(manifest_path(ref.dir).c_str()) == 0;
  std::vector<char> matches(targets.size(), 1);
  if (recheck || adopt) {
    std::vector<size_t> check;
    uint64_t check_bytes = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
      const Target& t = *targets[i];
      uint64_t sz;
      if (file_size(t.final_path, sz) && sz == t.size && (!t.sha256.empty() || !t.blob.empty())) {
        check.push_back(i);
        check_bytes += t.size;
      }
    }
    if (spin && !check.empty())
      spin->label("Checking the " + std::to_string(check.size()) + " files already in " + tilde(ref.dir) + term::dim(" (" + fmt_bytes(check_bytes) + ")") +
                  " against the pinned checksums");
    else if (opt.progress && !check.empty())
      fprintf(stderr, "Checking the %zu files already in %s (%s) against the pinned checksums...\n", check.size(),
              tilde(ref.dir).c_str(), fmt_bytes(check_bytes).c_str());
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int w = 0; w < kWorkers; ++w)
      th.emplace_back([&] {
        for (size_t k; (k = next.fetch_add(1)) < check.size();) {
          const Target& t = *targets[check[k]];
          std::string s256, s1;
          matches[check[k]] = hash_file(t.final_path, t.sha256.empty() ? nullptr : &s256, t.sha256.empty() ? &s1 : nullptr) &&
                              (t.sha256.empty() ? s1 == t.blob : s256 == t.sha256);
        }
      });
    for (auto& x : th) x.join();
  }
  int n_differ = 0;
  for (size_t i = 0; i < targets.size(); ++i) {
    auto& t = targets[i];
    total += t->size;
    uint64_t sz;
    if (file_size(t->final_path, sz) && sz == t->size) {
      if (matches[i]) {
        have += t->size;
        continue;
      }
      if (recheck) unlink(t->final_path.c_str());  // (an adopted folder's own file stays until the pinned one replaces it)
      ++n_differ;
    } else if (adopt && file_size(t->final_path, sz)) {
      ++n_differ;
    }
    if (t->size == 0) {
      write_atomic(t->final_path, "");
      continue;
    }
    bool hf = false;
    const std::string src = reusable(*t, from, hf);
    if (!src.empty()) {
      reuse.emplace_back(src, t.get());
      (hf ? n_hf : n_from)++;
      if (device_of(src) != dest_dev) need += t->size;
      continue;
    }
    const uint64_t part = part_bytes(*t);
    have += part;
    need += t->size - part;
    todo.push_back(t.get());
  }
  // a GGUF variant: its import then writes the trunk beside the file (the experts are an APFS clone of it) and, as
  // `kapitsa import` does, leaves 6 GB free
  uint64_t margin = 1ull << 30;
  if (gguf) {
    if (spin) spin->label("Planning the import of " + term::bold(ref.variant->gguf.path) + term::dim(" (its header)"));
    ImportSize is;
    if (import_size(ref, &is)) need += is.write;
    margin = 6ull << 30;
  }
  uint64_t free = free_bytes(ref.dir);
  if (need + margin > free && opt.make_room) {  // a person at a terminal: which models to delete, asked
    if (spin) spin->done();
    spin.reset();
    if (opt.make_room(SpaceShort{ref.name, ref.dir, need, free > margin ? free - margin : 0})) free = free_bytes(ref.dir);
  }
  if (need + margin > free) {
    // the downloaded models that would make room, by what deleting each gives back
    std::vector<std::pair<uint64_t, std::string>> rm;
    for (const auto& m : installed())
      if (m.dir != ref.dir)
        if (const uint64_t b = freeable_bytes(m.dir); b >= (100ull << 20)) rm.emplace_back(b, m.name);
    std::sort(rm.rbegin(), rm.rend());
    std::string hint;
    for (size_t i = 0; i < rm.size() && i < 3; ++i)
      hint += std::string(i ? ", " : "") + "kapitsa rm " + rm[i].second + " (" + fmt_bytes(rm[i].first) + ")";
    throw Error("Not enough disk space for " + ref.name + ": it needs " + fmt_bytes(need) + " more and " +
                fmt_bytes(free > margin ? free - margin : 0) + " is free (keeping " + (gguf ? "6" : "1") +
                " GB spare) on the disk of " +
                tilde(models_dir()) + "." + (hint.empty() ? " Free some space," : " Deleting a model frees space: " + hint + ";") +
                " or set KAPITSA_HOME to a folder on a bigger disk.");
  }
  for (auto& [src, t] : reuse) place(src, t->final_path);
  const bool tui = opt.progress && term::fancy();
  if (spin) spin->done();
  if (tui && adopt && n_differ)
    term::line("  " + term::warn(term::kWarn) + " " + std::to_string(n_differ) + (n_differ == 1 ? " file here differs" : " files here differ") +
                   " from the pinned ones and will be replaced", stderr);
  else if (opt.progress && adopt && n_differ)
    fprintf(stderr, "%d file%s here differ%s from the pinned ones and will be replaced.\n", n_differ, n_differ == 1 ? "" : "s",
            n_differ == 1 ? "s" : "");
  if (tui && (n_hf || n_from))
    term::line("  " + term::accent(term::kOk) + " Reused " + std::to_string(n_hf + n_from) + (n_hf + n_from == 1 ? " file" : " files") +
                   " already on this Mac" +
                   term::dim(std::string(n_hf ? (n_from ? " (the Hugging Face cache and other folders" : " (the Hugging Face cache")
                                              : " (other folders") +
                             "; APFS clones take no extra space)"),
               stderr);
  else if (opt.progress && (n_hf || n_from))
    fprintf(stderr, "Reused %d file%s already on this Mac (%schecksums match; APFS clones take no extra space).\n",
            n_hf + n_from, n_hf + n_from == 1 ? "" : "s",
            n_hf ? (n_from ? "from the Hugging Face cache and other folders; " : "from the Hugging Face cache; ") : "");
  if (!todo.empty()) {
    Progress prog;  // its total is set by download(): what is fetched now
    prog.show = opt.progress;
    prog.tty = isatty(2);
    std::string what = (gguf ? ref.variant->gguf.path + " " : "weights ") + fmt_bytes(weights_bytes);
    if (tui) {
      for (const auto& e : extras) what += std::string(" ") + term::kDot + " " + e;
      term::line("  " + term::accent("↓") + " Downloading " + term::bold(ref.name) + "  " + term::dim(what), stderr);
      term::line("    " + term::dim(term::clip("into " + tilde(ref.dir) + (have ? " (" + fmt_bytes(have) + " already here)" : ""),
                                         term::columns() - 5)),
                 stderr);
    } else if (opt.progress) {
      for (const auto& e : extras) what += ", " + e;
      fprintf(stderr, "Downloading %s (%s) into %s%s\n", ref.name.c_str(), what.c_str(), tilde(ref.dir).c_str(),
              have ? (", " + fmt_bytes(have) + " already here").c_str() : "");
    }
    download(todo, prog);
    if (tui)
      term::line("  " + term::accent(term::kOk) + " Downloaded " + fmt_bytes(prog.total) +
                     term::dim(" in " + fmt_duration(prog.secs()) + " (" +
                               std::to_string(int(double(prog.total) / 1e6 / std::max(0.001, prog.secs()))) + " MB/s)"),
                 stderr);
    else if (opt.progress)
      fprintf(stderr, "Downloaded %s in %s (%.0f MB/s).\n", fmt_bytes(prog.total).c_str(), fmt_duration(prog.secs()).c_str(),
              double(prog.total) / 1e6 / std::max(0.001, prog.secs()));
  }
  ojson m{{"name", ref.name}, {"repo", main.repo}, {"revision", main.revision}, {"commit", main.commit}};
  if (ref.variant) {
    m["variant"] = ref.variant->name;
    m["quant"] = ref.variant->quant;
  }
  if (gguf) {
    // `kapitsa import` over the folder (--force): the conversion replaces it, the downloaded file and the parts
    // folder with it (the experts' clone keeps the file's blocks); its manifest lists the files, ours replaces it
    const GgufFile& g = ref.variant->gguf;
    gguf::ImportOptions io;
    io.file = ref.dir + "/" + g.path;
    io.name = ref.dir.substr(ref.dir.find_last_of('/') + 1);
    io.force = true;
    io.progress = opt.progress;
    if (models_dir() + "/" + io.name != ref.dir) throw Error("internal: " + tilde(ref.dir) + " is not in " + tilde(models_dir()));
    try {
      gguf::import_model(io);
    } catch (const gguf::Error& e) {
      // Ctrl+C: the downloaded file stays, and the same command imports it again
      if (strncmp(e.what(), "Stopped", 7) == 0)
        throw Stopped("Import stopped. The download is kept: run the same command again to import it.");
      throw Error(e.what());
    } catch (const std::exception& e) {
      throw Error("The import of " + g.path + " failed: " + e.what());
    }
    ojson conv = ojson::object();
    try {
      const ojson im = ojson::parse(read_text(manifest_path(ref.dir)));
      mfiles = im.at("files");
      total = im.at("total_bytes").get<uint64_t>();
      if (im.contains("gguf") && im["gguf"].contains("conversions")) conv = im["gguf"]["conversions"];
    } catch (const std::exception&) {
      throw Error("The import of " + g.path + " left no manifest in " + tilde(ref.dir) + ". Please report this.");
    }
    m["gguf"] = ojson{{"path", g.path}, {"sha256", g.sha256}, {"size", g.size}, {"conversions", conv}};
  }
  m["files"] = mfiles;
  m["companions"] = mcomp;
  m["total_bytes"] = total;
  m["completed_at"] = now_iso();
  m["kapitsa"] = KAPITSA_VERSION;
  write_atomic(manifest_path(ref.dir), m.dump(2) + "\n");
  for (const auto& t : targets)  // the companions' part folders (dflash/ ...), then the parts folder: empty now
    for (std::string d = dirname_of(t->part_path); d.size() > ref.dir.size() + 15 && rmdir(d.c_str()) == 0;) d = dirname_of(d);
  rmdir((ref.dir + "/.kapitsa/parts").c_str());  // (fails harmlessly if something is left)
  return ref.dir;
}

}  // namespace

std::string ensure(const ModelRef& ref, const PullOptions& opt) {
  if (ref.kind == ModelRef::kLocal || is_current(ref)) return ref.dir;
  const bool existed = is_dir(ref.dir);
  try {
    return ensure_impl(ref, opt);
  } catch (const Stopped&) {
    throw;
  } catch (const Error& e) {
    // a finished download of older pinned revisions still runs: better than nothing while offline
    if (is_complete(ref.dir)) {
      if (term::fancy())
        term::line("  " + term::warn(term::kWarn) + " Using the copy of " + term::bold(ref.name) + " already here " +
                       term::dim(std::string("(it could not be updated: ") + e.what() + ")"),
                   stderr);
      else
        fprintf(stderr, "Using the copy of %s already here (it could not be updated: %s)\n", ref.name.c_str(), e.what());
      return ref.dir;
    }
    if (!existed) remove_if_empty(ref.dir);
    throw;
  } catch (...) {
    if (!existed) remove_if_empty(ref.dir);
    throw;
  }
}

void build_mtp_sidecar(const std::string& src, const std::string& revision, const std::string& out,
                       const std::string& hidden) {
  SignalGuard sig;
  HttpClient http(30);
  SidecarSource s;
  s.local = is_dir(src);
  s.src = src;
  s.rev = revision.empty() ? "main" : revision;
  s.http = &http;
  if (!s.local) {  // pin the revision's commit, like a download
    const Listing L = list_repo(http, src, s.rev);
    s.rev = L.commit;
  }
  const SidecarPlan p = plan_sidecar(s, src, hidden);
  if (s.local) {  // straight from the files
    std::string data = p.head;
    for (const Piece& x : p.pieces) {
      std::ifstream f(x.url, std::ios::binary);
      std::string b(x.len, '\0');
      f.seekg(std::streamoff(x.src));
      if (!f.read(b.data(), std::streamsize(x.len))) throw Error("Can't read " + x.url + ".");
      data += x.plus_one ? plus_one(b) : b;
    }
    write_atomic(out, data);
  } else {
    Target t;
    t.label = out;
    t.size = p.size;
    t.id = plan_id(p);
    t.head = p.head;
    t.pieces = p.pieces;
    t.final_path = out;
    t.part_path = out + ".part";
    t.state_path = out + ".part.json";
    Progress prog;
    prog.tty = isatty(2);
    std::vector<Target*> todo{&t};
    download(todo, prog);
  }
  fprintf(stderr, "wrote %s: %zu tensors, %.1f MB, base_hidden_variant %s\n", out.c_str(), p.pieces.size(),
          double(p.size - p.head.size()) / 1e6, hidden.c_str());
}

static std::atomic<bool> g_offline{false};
static std::atomic<int> g_offline_misses{0};

void set_offline(bool on) {
  g_offline = on;
  g_offline_misses = 0;
}

int offline_misses() { return g_offline_misses; }

// A GGUF variant's plan (plan_dir): `kapitsa import` plans the conversion from the file's header into `dir`, with
// `mark` last. The header is read from the downloaded file when that is complete, else fetched: the file's first 16 MB
// (64 MB if its header is longer) by a range request, written into a sparse file of the file's size, which the import
// reads as the file (it reads no tensor data to plan). That file sits in the plan's folder, on the models' disk where
// it is APFS (so that the import plans the clone it will make there), else in the temporary folder; it goes after.
// The plan is written beside `dir` and renamed into place.
static bool plan_gguf(const ModelRef& ref, const std::string& dir, const std::string& mark_text) {
  const GgufFile& g = ref.variant->gguf;
  const std::string out = dir + ".planning-" + std::to_string(getpid());
  std::error_code ec;
  std::filesystem::remove_all(out, ec);
  try {
    mkdirs(out);
  } catch (const Error&) {
    return false;
  }
  std::string src = ref.dir + "/" + g.path, tmp, tmp_dir;
  uint64_t sz = 0;
  if (!file_size(src, sz) || sz != g.size) {
    struct statfs s;
    std::string where = out;
    if (statfs(where.c_str(), &s) != 0 || strcmp(s.f_fstypename, "apfs") != 0) {
      std::string t = std::filesystem::temp_directory_path(ec).string() + "/kapitsa-plan-XXXXXX";
      if (ec || !mkdtemp(t.data()) || statfs(t.c_str(), &s) != 0 || strcmp(s.f_fstypename, "apfs") != 0) {
        if (!ec) rmdir(t.c_str());
        std::filesystem::remove_all(out, ec);
        return false;  // no sparse files there: not planned
      }
      where = tmp_dir = t;
    }
    tmp = src = where + "/" + g.path.substr(g.path.find_last_of('/') + 1);
    HttpClient http(30);
    const std::string url = file_url(ref.repo, ref.revision, g.path);
    std::string head;
    bool whole = false;  // the header, all of it
    for (uint64_t n = 16ull << 20; !whole; n *= 4) {
      n = std::min(n, g.size);
      HttpResult r = get_retry(http, url, head.size(), n);
      if (!r.ok() || r.status != 206) break;
      head += r.body;
      const int fd = open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
      if (fd < 0) break;
      const bool wrote = ftruncate(fd, off_t(g.size)) == 0 && pwrite_all(fd, head.data(), head.size(), 0);
      close(fd);
      if (!wrote) break;
      try {  // a header cut short reads as zeros past its end, which the parser refuses, or as tensor data after it
        const gguf::File f(tmp);
        uint64_t data0 = UINT64_MAX;
        for (const auto& t : f.tensors()) data0 = std::min(data0, t.offset);
        whole = data0 <= head.size();
      } catch (const gguf::Error&) {
      }
      if (n >= g.size || n >= (64ull << 20)) break;
    }
    if (!whole) src.clear();
  }
  bool planned = false;
  if (!src.empty()) {
    try {
      gguf::ImportOptions io;
      io.file = src;
      io.name = ref.dir.substr(ref.dir.find_last_of('/') + 1);
      io.force = true;  // (it writes nothing in the models dir)
      io.progress = false;
      io.plan_dir = out;
      gguf::import_model(io);
      planned = true;
    } catch (const std::exception&) {
    }
  }
  if (!tmp.empty()) unlink(tmp.c_str());
  if (!tmp_dir.empty()) rmdir(tmp_dir.c_str());
  bool ok = false;
  if (planned) {
    try {
      write_atomic(out + "/.complete", mark_text);
      std::filesystem::remove_all(dir, ec);
      ok = rename(out.c_str(), dir.c_str()) == 0;
    } catch (const Error&) {
    }
  }
  if (!ok) std::filesystem::remove_all(out, ec);
  return ok;
}

std::string plan_dir(const ModelRef& ref) {
  if (ref.kind == ModelRef::kLocal || is_current(ref)) return ref.dir;
  if (ref.kind != ModelRef::kCatalog) return "";
  const Variant& m = *ref.variant;
  const std::string dir = home_dir() + "/plans/" + ref.model->name + "-" + m.name, mark = dir + "/.complete";
  std::string want = m.repo + "@" + m.revision;
  // a GGUF variant's plan is the import's: planned again by another Kapitsa, whose conversion may differ
  if (!m.gguf.path.empty()) want += "\ngguf " + m.gguf.path + " " + m.gguf.sha256 + "\nkapitsa " KAPITSA_VERSION;
  for (const auto& c : m.companions) want += "\n" + c.kind + " " + c.repo + "@" + c.revision;
  if (read_text(mark) == want) return dir;
  if (g_offline) {
    ++g_offline_misses;
    return "";
  }
  unlink(mark.c_str());
  if (!m.gguf.path.empty()) return plan_gguf(ref, dir, want) ? dir : "";
  // the repos' file lists, then config.json whole and each safetensors file's header, all in parallel (a few hundred
  // small requests' worth of latency otherwise)
  struct Repo {
    std::string repo, revision, sub;
    std::vector<std::string> only;
    Listing L;
  };
  std::vector<Repo> repos{{m.repo, m.revision, "", {}, {}}};
  const Companion* sidecar = nullptr;
  for (const Companion& c : m.companions) {
    if (c.kind == "mtp-sidecar") sidecar = &c;
    else repos.push_back({c.repo, c.revision, c.subdir.empty() ? "" : c.subdir + "/", c.files, {}});
  }
  std::atomic<bool> bad{false};
  auto parallel = [&](size_t n, const std::function<void(size_t, HttpClient&)>& job) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (size_t t = 0; t < std::min<size_t>(n, 8); ++t)
      th.emplace_back([&] {
        HttpClient http(20);
        for (size_t i; !bad && (i = next++) < n;) {
          try {
            job(i, http);
          } catch (...) {
            bad = true;
          }
        }
      });
    for (auto& x : th) x.join();
  };
  std::string head4b;
  parallel(repos.size() + (sidecar ? 1 : 0), [&](size_t i, HttpClient& http) {
    if (i < repos.size()) {
      repos[i].L = list_repo(http, repos[i].repo, repos[i].revision);
      return;
    }
    SidecarSource s;
    s.src = sidecar->repo;
    s.rev = sidecar->revision;
    s.http = &http;
    head4b = plan_sidecar(s, sidecar->repo, "post_norm").head;
  });
  if (bad) return "";
  struct Job {
    std::string url, dest;
    bool header;
  };
  std::vector<Job> jobs;
  for (const Repo& r : repos)
    for (const auto& f : r.L.files) {
      if (!r.only.empty() && std::find(r.only.begin(), r.only.end(), f.path) == r.only.end()) continue;
      const bool st = f.path.size() > 12 && f.path.compare(f.path.size() - 12, 12, ".safetensors") == 0;
      if (st || f.path == "config.json") jobs.push_back({file_url(r.repo, r.L.commit, f.path), dir + "/" + r.sub + f.path, st});
    }
  parallel(jobs.size(), [&](size_t i, HttpClient& http) {
    const Job& j = jobs[i];
    if (!j.header) {
      HttpResult r = get_retry(http, j.url);
      if (!r.ok()) throw Error(r.error);
      write_atomic(j.dest, r.body);
      return;
    }
    HttpResult r = get_retry(http, j.url, 0, 8);
    if (!r.ok() || r.status != 206) throw Error(r.error);
    const uint64_t n = le64(r.body);
    if (n > (64u << 20)) throw Error("not a safetensors file");
    HttpResult h = get_retry(http, j.url, 8, 8 + n);
    if (!h.ok() || h.status != 206) throw Error(h.error);
    write_atomic(j.dest, r.body + h.body);
  });
  if (bad) return "";
  try {
    if (sidecar) write_atomic(dir + "/" + (sidecar->subdir.empty() ? "" : sidecar->subdir + "/") + "mtp.safetensors", head4b);
    write_atomic(mark, want);
  } catch (const Error&) {
    return "";
  }
  return dir;
}

bool import_size(const ModelRef& ref, ImportSize* out) {
  if (ref.kind != ModelRef::kCatalog || ref.variant->gguf.path.empty()) return false;
  const std::string d = plan_dir(ref);
  if (d.empty()) return false;
  try {
    if (d == ref.dir) {  // imported already: the folder as its manifest counts it
      *out = ImportSize{0, json::parse(read_text(manifest_path(d))).at("total_bytes").get<uint64_t>()};
      return true;
    }
    const json j = json::parse(read_text(d + "/import.json"));
    *out = ImportSize{j.at("write").get<uint64_t>(), j.at("disk").get<uint64_t>()};
    return true;
  } catch (...) {
    return false;
  }
}

uint64_t freeable_bytes(const std::string& dir) {
  uint64_t total = 0;
  std::vector<std::string> stack{dir};
  while (!stack.empty()) {
    const std::string d = stack.back();
    stack.pop_back();
    DIR* dd = opendir(d.c_str());
    if (!dd) continue;
    while (dirent* e = readdir(dd)) {
      const std::string fn = e->d_name;
      if (fn == "." || fn == "..") continue;
      const std::string p = d + "/" + fn;
      struct stat st;
      if (lstat(p.c_str(), &st) != 0) continue;
      if (S_ISDIR(st.st_mode)) stack.push_back(p);
      else if (S_ISREG(st.st_mode)) {
        const int64_t own = private_size(p);
        total += own >= 0 ? uint64_t(own) : uint64_t(st.st_blocks) * 512;
      }
    }
    closedir(dd);
  }
  return total;
}

std::vector<Installed> installed() {
  std::vector<Installed> out;
  DIR* d = opendir(models_dir().c_str());
  if (!d) return out;
  std::vector<std::string> names;
  while (dirent* e = readdir(d))
    if (e->d_name[0] != '.' && is_dir(models_dir() + "/" + e->d_name)) names.push_back(e->d_name);
  closedir(d);
  std::sort(names.begin(), names.end());
  for (const auto& n : names) {
    Installed m;
    m.dir = models_dir() + "/" + n;
    m.name = n;
    if (!is_dir(m.dir + "/.kapitsa") && !is_dir(m.dir + "/dflash2")) {  // a folder put here by hand
      uint64_t sz;
      if (!file_size(m.dir + "/config.json", sz)) continue;
    }
    try {
      const json j = json::parse(read_text(manifest_path(m.dir)));
      m.name = j.value("name", n);
      m.variant = j.value("variant", "");
      m.repo = j.value("repo", "");
      m.commit = j.value("commit", "");
      m.total = j.value("total_bytes", uint64_t(0));
    } catch (...) {
    }
    m.complete = is_complete(m.dir);
    // bytes on disk: finished files and the written ranges of parts
    std::vector<std::string> stack{m.dir};
    while (!stack.empty()) {
      const std::string dir = stack.back();
      stack.pop_back();
      if (DIR* dd = opendir(dir.c_str())) {
        while (dirent* e = readdir(dd)) {
          const std::string fn = e->d_name;
          if (fn == "." || fn == "..") continue;
          const std::string p = dir + "/" + fn;
          struct stat st;
          if (lstat(p.c_str(), &st) != 0) continue;
          if (S_ISDIR(st.st_mode)) {
            stack.push_back(p);
          } else if (fn.size() > 10 && fn.compare(fn.size() - 10, 10, ".part.json") == 0) {
            try {
              Ranges r;
              const json state = json::parse(read_text(p));
              for (const auto& x : state.at("done")) r.add(x.at(0).get<uint64_t>(), x.at(1).get<uint64_t>());
              m.bytes += r.covered();
            } catch (...) {
            }
          } else if (S_ISREG(st.st_mode) && !(fn.size() > 5 && fn.compare(fn.size() - 5, 5, ".part") == 0) &&
                     fn != "manifest.json" && fn != "lock") {
            m.bytes += uint64_t(st.st_size);
          } else if (S_ISLNK(st.st_mode)) {
            uint64_t sz;
            if (file_size(p, sz)) m.bytes += sz;
          }
        }
        closedir(dd);
      }
    }
    uint64_t sz;
    if (file_size(m.dir + "/mtp.safetensors", sz)) m.extras.push_back("MTP head");
    if (file_size(m.dir + "/dflash2/model.safetensors", sz)) {
      bool v1 = false;  // a DFlash (v1) drafter in the same folder (Qwen3.5-4B's)
      try {
        v1 = json::parse(read_text(m.dir + "/dflash2/config.json")).value("architectures", json::array()) ==
             json::array({"DFlashDraftModel"});
      } catch (...) {
      }
      m.extras.push_back(v1 ? "DFlash drafter" : "DFlash2 drafter");
    }
    if (file_size(m.dir + "/dflash/model.safetensors", sz)) {
      // Nemotron 3.5's block drafters: DSpark (Lightning) or LiLiCorr (Bolt, Super-VL)
      std::ifstream f(m.dir + "/dflash/config.json");
      const std::string c((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      m.extras.push_back(c.find("\"dspark_fixup_head_type\"") != std::string::npos ? "DSpark drafter" : "LiLiCorr drafter");
    }
    out.push_back(m);
  }
  return out;
}

}  // namespace ni::hub
