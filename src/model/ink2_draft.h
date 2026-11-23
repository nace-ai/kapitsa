#pragma once
#include "loader/safetensors.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

struct Ink2Mat;

class Ink2DraftSeq {
 public:
  ~Ink2DraftSeq();
  Ink2DraftSeq(Ink2DraftSeq&&) noexcept;
  Ink2DraftSeq& operator=(Ink2DraftSeq&&) noexcept;
  Ink2DraftSeq(const Ink2DraftSeq&) = delete;
  Ink2DraftSeq& operator=(const Ink2DraftSeq&) = delete;

  int position() const;
  size_t bytes() const;

 private:
  friend class Ink2Draft;
  Ink2DraftSeq();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct Ink2DraftMemory {
  size_t shared_bytes;
  size_t empty_seq_bytes;
};

class Ink2Draft {
 public:
  Ink2Draft(const std::string& model_dir, const Tensor& embedding, const Tensor& embed_norm,
            const Ink2Mat& unembed);
  ~Ink2Draft();
  Ink2Draft(const Ink2Draft&) = delete;
  Ink2Draft& operator=(const Ink2Draft&) = delete;

  // CPU-only. Missing mtp.safetensors returns zeroes; invalid present weights/config throw.
  static Ink2DraftMemory memory_plan(const std::string& dir);
  // New allocations only. Existing active and retired buffers are already included in bytes().
  size_t observe_allocation_bytes(const Ink2DraftSeq& seq, int rows) const;
  std::unique_ptr<Ink2DraftSeq> new_seq();
  // Observe only committed rows, in order. captures is [4, rows, 2304] float32
  // at visits 5/23/41 and final norm. Shared scratch requires serialized calls.
  // Commit observe before propose; finish GPU work before destroying or replacing either owner.
  void observe(Encoder& e, Ink2DraftSeq& seq, const Buffer& captures, int rows, int pos0);
  std::vector<int> propose(Ink2DraftSeq& seq, int anchor_token, int count = 7);
  size_t bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ni
