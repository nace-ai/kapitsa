// Integer prefix selection only. GPU verification and request accounting have separate gates.
#include "model/ink2_accept.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <span>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); \
} } while (false)

using namespace ni;

int main() {
  int cases = 0;
  for (const int base : {0, 17000, 2147483000}) {
    for (int proposals = 0; proposals <= 7; ++proposals) {
      for (unsigned mask = 0; mask < (1u << proposals); ++mask) {
        std::array<int, 7> draft{};
        std::array<int, 8> target{};
        int first_reject = proposals;
        for (int i = 0; i < proposals; ++i) {
          draft[i] = base + 17 * i;
          const bool agrees = (mask & (1u << i)) != 0;
          target[i] = agrees ? draft[i] : base + 200 + 13 * i;
          if (!agrees && first_reject == proposals) first_reject = i;
        }
        target[proposals] = base + 500;
        const auto choice = ink2_choose(std::span<const int>(draft.data(), proposals),
                                       std::span<const int>(target.data(), proposals + 1));
        CHECK(choice.accepted == first_reject);
        CHECK(choice.size() == first_reject + 1);
        CHECK(choice.size() >= 1 && choice.size() <= 8);
        for (int i = 0; i < first_reject; ++i) CHECK(choice.tokens[i] == draft[i]);
        CHECK(choice.tokens[first_reject] == target[first_reject]);
        // The returned correction/bonus must not alias the sampled rows or draft storage.
        const int correction = target[first_reject];
        target.fill(-1);
        draft.fill(-1);
        CHECK(choice.tokens[first_reject] == correction);
        ++cases;
      }
    }
  }
  {
    const std::array<int, 3> draft{11, 22, 33};
    const std::array<int, 4> target{11, 99, 33, 44};
    const auto choice = ink2_choose(draft, target);
    CHECK(choice.accepted == 1 && choice.size() == 2);
    CHECK(choice.tokens[0] == 11 && choice.tokens[1] == 99);
    ++cases;
  }
  std::printf("ink2_accept_test: %d cases passed\n", cases);
}
