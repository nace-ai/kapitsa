#include "util/canonical_plan_memo.h"

#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ni::canonical_plan_memo {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
[[noreturn]] void fail(const std::string& reason) {
  throw std::runtime_error("KFlash: canonical plan memo: " + reason);
}
struct FD {
  int value;
  explicit FD(const fs::path& path) : value(open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
    if (value < 0) fail("cannot open " + path.string());
  }
  ~FD() { close(value); }
};
struct Stamp {
  uint64_t device, inode, size;
  int64_t mtime_s, mtime_ns, ctime_s, ctime_ns;
  bool operator==(const Stamp&) const = default;
};
Stamp stamp(int fd, const fs::path& path) {
  struct stat st {};
  if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) fail("invalid file " + path.string());
  return {uint64_t(st.st_dev), uint64_t(st.st_ino), uint64_t(st.st_size), st.st_mtimespec.tv_sec,
          st.st_mtimespec.tv_nsec, st.st_ctimespec.tv_sec, st.st_ctimespec.tv_nsec};
}
void read_exact(int fd, uint64_t offset, void* data, size_t bytes, const fs::path& path) {
  auto* out = static_cast<uint8_t*>(data);
  for (size_t done = 0; done < bytes;) {
    const ssize_t got = pread(fd, out + done, bytes - done, off_t(offset + done));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) fail("short identity read " + path.string());
    done += size_t(got);
  }
}
using Digest = std::array<unsigned char, CC_SHA256_DIGEST_LENGTH>;
Digest digest(const std::vector<uint8_t>& bytes) {
  Digest value{};
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  CC_SHA256(bytes.data(), CC_LONG(bytes.size()), value.data());
#pragma clang diagnostic pop
  return value;
}
struct File {
  std::string name, target;
  Stamp state;
  Digest contents;
  bool operator==(const File&) const = default;
};
struct Identity {
  std::string directory;
  File config;
  std::vector<File> shards;
  bool operator==(const Identity&) const = default;
};
std::vector<std::string> shard_names(const fs::path& dir) {
  std::vector<std::string> result;
  std::error_code ec;
  fs::directory_iterator it(dir, ec), end;
  if (ec) fail("cannot list " + dir.string());
  for (; it != end; it.increment(ec)) {
    if (ec) fail("cannot list " + dir.string());
    if (it->path().extension() == ".safetensors") result.push_back(it->path().filename().string());
  }
  if (ec) fail("cannot list " + dir.string());
  std::sort(result.begin(), result.end());
  if (result.empty()) fail("no safetensors in " + dir.string());
  return result;
}
std::string resolved(const fs::path& path) {
  std::error_code ec;
  auto value = fs::canonical(path, ec);
  if (ec) fail("cannot resolve " + path.string());
  return value.string();
}
File identify(const fs::path& dir, const std::string& name, bool header_only) {
  const fs::path path = dir / name;
  FD fd(path);
  const Stamp before = stamp(fd.value, path);
  const std::string target = resolved(path);
  uint64_t offset = 0, bytes = before.size;
  if (header_only) {
    if (before.size < 10) fail("invalid safetensors header " + path.string());
    read_exact(fd.value, 0, &bytes, 8, path);
    offset = 8;
    if (bytes < 2 || bytes > before.size - 8) fail("invalid safetensors header " + path.string());
  }
  if (bytes > (uint64_t(256) << 20)) fail("identity metadata exceeds 256 MiB: " + path.string());
  std::vector<uint8_t> data(static_cast<size_t>(bytes));
  read_exact(fd.value, offset, data.data(), data.size(), path);
  File result{name, target, before, digest(data)};
  if (!(stamp(fd.value, path) == before)) fail("file changed while hashing " + path.string());
  FD current(path);
  if (!(stamp(current.value, path) == before) || resolved(path) != target)
    fail("file replaced while hashing " + path.string());
  return result;
}
Identity snapshot(const std::string& dir) {
  Identity result;
  result.directory = resolved(dir);
  const auto names = shard_names(dir);
  result.config = identify(dir, "config.json", false);
  for (const auto& name : names) result.shards.push_back(identify(dir, name, true));
  // Catch a change to an earlier file while a later file's header was read.
  auto recheck = [&](const File& file) {
    const fs::path path = fs::path(dir) / file.name;
    FD fd(path);
    if (!(stamp(fd.value, path) == file.state) || resolved(path) != file.target)
      fail("file changed during identity snapshot " + path.string());
  };
  recheck(result.config);
  for (const auto& file : result.shards) recheck(file);
  if (names != shard_names(dir) || resolved(dir) != result.directory) fail("model directory changed during identity snapshot");
  return result;
}
thread_local Scope* active = nullptr;
}  // namespace

struct Scope::Impl {
  std::string requested;
  Identity initial;
  std::string key;
  std::optional<Counts> value;
};

Scope::Scope(const std::string& dir, bool enabled) {
  if (!enabled) return;
  const auto start = Clock::now();
  auto state = std::make_unique<Impl>();
  // Preserve the caller's spelling: lexical normalization through symlinks/.. can name another object.
  state->requested = dir;
  state->initial = snapshot(dir);
  stats_.identity_ms += elapsed(start);
  impl_ = std::move(state);
  previous_ = active;
  active = this;
}
Scope::~Scope() { close(); }
void Scope::close() noexcept {
  if (!impl_) return;
  if (active == this) active = previous_;
  else {
    // Normal scopes are stack ordered. Unlink defensively if an owning caller destroys an outer scope first.
    for (auto* it = active; it; it = it->previous_)
      if (it->previous_ == this) { it->previous_ = previous_; break; }
  }
  impl_.reset();  // release all identities and the three-count entry on success or unwind
  previous_ = nullptr;
}
bool Scope::has_entry() const { return impl_ && impl_->value.has_value(); }
Scope* Scope::current_for(const std::string& dir) {
  if (!active || !active->impl_) return nullptr;
  // A different alias gets its own ephemeral scope, so its build is checked against that caller path.
  // The same spelling remains in this scope after retargeting; get() then rejects the changed identity.
  return dir == active->impl_->requested ? active : nullptr;
}
Counts Scope::get(const std::string& layout_key, const std::function<Counts()>& build) {
  if (!impl_) fail("entry requested from a closed scope");
  if (active != this) fail("entry requested from a non-current scope");
  auto checked_identity = [&] {
    const auto start = Clock::now();
    try {
      Identity now = snapshot(impl_->requested);
      stats_.identity_ms += elapsed(start);
      if (!(now == impl_->initial)) fail("model files/config changed during this load");
      return now;
    } catch (...) {
      impl_->value.reset(); impl_->key.clear();
      throw;
    }
  };
  const auto before = checked_identity();
  if (impl_->value && impl_->key == layout_key) { ++stats_.hits; return *impl_->value; }
  impl_->value.reset(); impl_->key.clear();
  ++stats_.misses;
  const auto start = Clock::now();
  Counts value;
  try { value = build(); }
  catch (...) { stats_.compute_ms += elapsed(start); ++stats_.failed_builds; throw; }
  stats_.compute_ms += elapsed(start);
  if (!(checked_identity() == before)) fail("model changed during exact scan");
  impl_->key = layout_key;
  impl_->value = value;
  return value;
}
void Scope::finish() {
  if (!impl_) return;
  if (active != this) fail("nested scope must finish first");
  const auto start = Clock::now();
  try {
    const auto now = snapshot(impl_->requested);
    stats_.identity_ms += elapsed(start);
    if (!(now == impl_->initial)) fail("model files/config changed before engine ready");
  } catch (...) { close(); throw; }
  close();
}
Counts evaluate(const std::string& dir, const std::string& layout_key, const std::function<Counts()>& build) {
  if (auto* scope = Scope::current_for(dir)) return scope->get(layout_key, build);
  Scope temporary(dir);
  const Counts result = temporary.get(layout_key, build);
  temporary.finish();
  return result;
}
}  // namespace ni::canonical_plan_memo
