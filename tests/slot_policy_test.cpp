#include "engine/slot_policy.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); \
} } while (false)

int main() {
  // A short prompt extending every row in the best slot keeps that conversation.
  CHECK(!ni::should_allocate_new_conversation(12, 12, 19));

  // A short shared prefix that does not cover the slot's whole live state starts a new conversation.
  CHECK(ni::should_allocate_new_conversation(12, 20, 27));

  // The existing reuse threshold continues to apply to longer prefixes.
  CHECK(!ni::should_allocate_new_conversation(64, 64, 70));

  // A prompt that adds no rows is not a continuation.
  CHECK(ni::should_allocate_new_conversation(12, 12, 12));
}
