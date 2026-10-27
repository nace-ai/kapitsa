#pragma once

#include "model/expert_store.h"

namespace ni::q20 {

struct ExpertCacheTestAccess {
  enum class Point { BeforeWorkerStart, BeforeRead, ReadDone, ReadersClosed, ReadersJoined };
  static void observe(const ExpertCache& cache, Point point);

  static std::unique_lock<std::mutex> hold_io(ExpertCache& c) { return std::unique_lock<std::mutex>(c.io_mu_); }
  static int pending(const ExpertCache& c, size_t key) { return c.pending_[key].load(std::memory_order_acquire); }
  static bool stopping(const ExpertCache& c) { return c.turn_stop_.load(); }
  static bool all_read(const ExpertCache& c) {
    for (size_t k = 0; k < size_t(c.L_) * c.E_; ++k)
      if (pending(c, k) != 0) return false;
    return true;
  }
};

}
