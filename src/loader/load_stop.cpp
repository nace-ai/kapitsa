#include "loader/load_stop.h"

#include <atomic>
#include <cstdlib>
#include <unistd.h>

namespace ni {

namespace {
std::atomic<LoadStopAction> g_action{nullptr};
std::atomic<bool> g_asked{false};
std::atomic<bool> g_acting{false};
}  // namespace

void set_load_stop_action(LoadStopAction action) { g_action.store(action); }

void ask_load_stop() { g_asked.store(true); }

void load_stop_point() {
  if (!g_asked.load(std::memory_order_relaxed)) return;  // (the one cost on every read: a relaxed load)
  const LoadStopAction action = g_action.load();
  if (!action) return;
  if (!g_acting.exchange(true)) {
    action();
    std::abort();  // (an action that returned: a bug, not a stop)
  }
  for (;;) pause();  // another reader is ending the process
}

}  // namespace ni
