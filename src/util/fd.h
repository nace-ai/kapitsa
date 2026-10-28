#pragma once
#include <unistd.h>
#include <utility>

namespace ni {
// File ownership must survive a loader throwing before its constructor finishes.
class Fd {
 public:
  explicit Fd(int fd = -1) : fd_(fd) {}
  ~Fd() { if (fd_ >= 0) close(fd_); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) { if (fd_ >= 0) close(fd_); fd_ = std::exchange(other.fd_, -1); }
    return *this;
  }
  int get() const { return fd_; }
 private:
  int fd_;
};
}  // namespace ni
