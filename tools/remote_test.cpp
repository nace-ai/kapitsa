// CPU test of a split model's reserve on its other Macs (src/model/remote.h, remote_reserve): no GPU, no model. A fake
// Remote records each grow() and answers as a stage does, rounding its growth up by its own rule (or failing, or lost).
// Reserves as the engine makes them (2048 -> 2049 -> 3000 -> ...) must ask the other Macs only past what they hold,
// take what they hold from their answer (not this Mac's cap, which rounds by this Mac's rule), and ask again after a
// failure.
//   remote_test
#include "model/remote.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace ni;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                      \
  do {                                                     \
    if (!(c)) {                                            \
      ++g_fail;                                            \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
  } while (0)

// A stage: its cache grows to `need` rounded up to `step` positions; fail_next: the next growth lacks the memory
struct FakeStage : Remote {
  std::vector<int> asked;
  int cap = 0, step = 256;
  bool fail_next = false, gone = false;
  bool pass(const Pass&, const Rows&) override { return true; }
  int new_seq(int) override { return 0; }
  int grow(int, int need) override {
    asked.push_back(need);
    if (gone) return -1;
    if (fail_next) {
      fail_next = false;
      return -1;
    }
    if (need > cap) cap = (need + step - 1) / step * step;
    return cap;
  }
  void drop_kv(int) override {}
  void reset(int) override {}
  int new_checkpoint() override { return 0; }
  void save(int, int, int) override {}
  void restore(int, int) override {}
  void flush(int, int) override {}
  void rope(int, int, int, const int32_t*) override {}
  std::string lost() const override { return gone ? "gone" : ""; }
};

std::string list(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
  return s;
}

}  // namespace

int main() {
  {  // the engine's growth: asked only past what the stage holds, which its answers set
    FakeStage r;
    Seq s;
    s.limit = 8192;
    for (int need : {2048, 2049, 2300, 3000, 3072}) CHECK(remote_reserve(r, s, need), "reserve %d", need);
    CHECK(list(r.asked) == "2048,2049,3000", "asked %s", list(r.asked).c_str());
    CHECK(s.rcap == 3072, "the stage holds %d", s.rcap);
  }
  {  // a stage that grows exactly what it is asked: a head whose own cap rounded further (3072) still asks for 2500
    FakeStage r;
    r.step = 1;
    Seq s;
    s.limit = 8192;
    s.cap = 3072;  // (this Mac's own rounding)
    CHECK(remote_reserve(r, s, 2049) && remote_reserve(r, s, 2500), "reserves");
    CHECK(list(r.asked) == "2049,2500", "asked %s", list(r.asked).c_str());
    CHECK(s.rcap == 2500, "the stage holds %d", s.rcap);
  }
  {  // a growth the stage cannot make: false, what it holds unchanged, and asked again at the next reserve
    FakeStage r;
    Seq s;
    s.limit = 8192;
    CHECK(remote_reserve(r, s, 1000), "first");
    r.fail_next = true;
    CHECK(!remote_reserve(r, s, 4000), "no memory there");
    CHECK(s.rcap == 1024, "still %d", s.rcap);
    CHECK(remote_reserve(r, s, 4000), "asked again");
    CHECK(list(r.asked) == "1000,4000,4000" && s.rcap == 4096, "asked %s, holds %d", list(r.asked).c_str(), s.rcap);
  }
  {  // past the conversation's limit: nothing asked (the head's own reserve refuses it); lost Macs: false
    FakeStage r;
    Seq s;
    s.limit = 4096;
    CHECK(remote_reserve(r, s, 5000) && r.asked.empty(), "past the limit");
    r.gone = true;
    CHECK(!remote_reserve(r, s, 100), "lost");
  }
  printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
