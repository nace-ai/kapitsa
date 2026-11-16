// Flint refuses NI_KV_I8 (#1257): the refusal's text, from the switch's own reader and from a given answer.
// No model, no GPU.
#include "model/flint_kv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;
#define CHECK(condition, what)                                                                       \
  do {                                                                                               \
    if (!(condition)) {                                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, what, #condition);          \
      ++failures;                                                                                    \
    }                                                                                                \
  } while (false)

int main() {
  static const char* kMsg = "int8 KV cache isn't supported with Flint models yet; unset NI_KV_I8";
  CHECK(ni::flint_kv_i8_refusal(false) == nullptr, "fp16 KV is not refused");
  const char* r = ni::flint_kv_i8_refusal(true);
  CHECK(r && std::strcmp(r, kMsg) == 0, "int8 KV is refused with the one message");

  // the switch as the loader reads it (kv_i8 reads the environment once)
  unsetenv("NI_KV_I8");
  setenv("NI_KV_I8", "1", 1);
  const char* e = ni::flint_kv_i8_refusal();
  CHECK(e && std::strcmp(e, kMsg) == 0, "NI_KV_I8=1 is refused");

  if (failures) return 1;
  std::puts("flint_kv_test: ok");
  return 0;
}
