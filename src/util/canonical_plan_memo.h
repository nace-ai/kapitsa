#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace ni::canonical_plan_memo {

struct Counts {
  size_t deferred_bytes = 0, image_pages = 0, image_bytes = 0;
  bool operator==(const Counts&) const = default;
};
struct Stats {
  size_t hits = 0, misses = 0, failed_builds = 0;
  double identity_ms = 0, compute_ms = 0;
};

// One synchronous Engine load only. The entry contains counts and small file metadata,
// never coefficients/payloads. finish() validates the pinned inputs and releases the entry.
class Scope {
 public:
  explicit Scope(const std::string& dir, bool enabled = true);
  ~Scope();
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Scope(Scope&&) = delete;
  Scope& operator=(Scope&&) = delete;
  Counts get(const std::string& layout_key, const std::function<Counts()>& build);
  void finish();
  const Stats& stats() const { return stats_; }
  bool has_entry() const;
  bool enabled() const { return bool(impl_); }
  static Scope* current_for(const std::string& dir);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Scope* previous_ = nullptr;
  Stats stats_;
  void close() noexcept;
};

// Outside a load scope this is a fresh, identity-checked scan, with no retained memo.
Counts evaluate(const std::string& dir, const std::string& layout_key, const std::function<Counts()>& build);

}  // namespace ni::canonical_plan_memo
