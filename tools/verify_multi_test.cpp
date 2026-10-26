// The batched-verify guard's contract (Qwen35::verify_multi_ok), CPU only: no GPU, no model folder, no Encoder.
//   verify_multi_test   CPU: what Engine::batch_rows routes on (src/engine/engine.cpp:1183): the virtual's
//                         default and the opt-out, read through a base pointer as the engine reads it, on Qwen35
//                         subclasses built with the protected NoLoad constructor (src/model/qwen35.cpp:888:
//                         members only, nothing read, no GPU allocation; ~Qwen35 releases GPU state only when
//                         the NI_QMT_I8 ANE ring ran, which it does not here)
//
// What a CPU cannot test (each needs a model folder and a GPU; documented, not asserted):
//   - Qwen4Exp::verify_multi_ok() with the DFlash2 path bound (NI_DFLASH_FN=1): the switch itself is CPU-testable
//     and tools/fn_switch_test.cpp asserts it both ways on the real class; a *loaded* Qwen4Exp would need the
//     checkpoint its constructor reads from model_dir.
//   - Engine::batch_rows(n) == 0 through the guard (src/engine/engine.cpp:1183): Engine's constructor builds the
//     model from opts_.model_dir and allocates GPU buffers.
//   - The fatal() backstop in encode_verify_multi (src/model/qwen35_v8.cpp:1279): reaching it needs a live
//     Encoder (a Metal command buffer, so a Gpu instance) and a valid VerifyGroup, and fatal() exits the process
//     (src/util/log.h), so it would take a death-test subprocess on top of the GPU state.
// The subclass below states the same contract those three exercise: a trunk whose single-sequence verify is
// overridden opts out, and the engine's guard reads that through the base pointer.
#include "model/qwen35.h"

#include <cstdio>
#include <string>

using namespace ni;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

// A Qwen3.5-family trunk as the engine builds one (src/engine/engine.cpp, the q35_ branch), minus the load
struct Trunk : Qwen35 {
  Trunk() : Qwen35("", 1, NoLoad{}) {}
};

// The opt-out: what Qwen4Exp states (src/model/qwen4_exp.h:142) because its single-sequence verify is overridden
// (4-stream GDN state, the indexer's KV) and Qwen35::encode_verify_multi walks neither
struct OverriddenVerify : Qwen35 {
  OverriddenVerify() : Qwen35("", 1, NoLoad{}) {}
  bool verify_multi_ok() const override { return false; }
};

}  // namespace

int main() {
  const Trunk trunk;
  const OverriddenVerify overridden;
  // through base pointers, as Engine::batch_rows reads it (src/engine/engine.cpp:1183)
  const Qwen35& base_trunk = trunk;
  const Qwen35& base_overridden = overridden;
  check(base_trunk.verify_multi_ok(),
        "a Qwen3.5-family trunk keeps the default: verify_multi_ok() true (qwen35.h:476)");
  check(!base_overridden.verify_multi_ok(),
        "a trunk whose single-sequence verify is overridden opts out through the base pointer, as the engine reads it");
  check(trunk.verify_multi_ok() != overridden.verify_multi_ok(),
        "the virtual dispatches per subclass: the guard tells the two trunks apart");
  std::printf("%s\n", g_fail ? "FAILED" : "all passed");
  return g_fail ? 1 : 0;
}
