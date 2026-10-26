// The NI_DFLASH_FN switch's contract (Qwen4Exp::dflash_fn), CPU only: no GPU, no model folder, no Encoder.
//   fn_switch_test   CPU: the #608 review ruling's switch, proven both ways —
//                          switch off is main before DFlash2 support for this class (block_drafter_ok()'s
//                          literal false, the verify flags Qwen35's base values, so plan_load's Qwen4Exp
//                          exclusion keeps a found drafter from binding), and switch on is the fn8t evidence
//                          binaries' hardcoded flag set (78416f58: the block drafter binds, both verifies
//                          false). dflash_fn() reads the environment once at startup, so the on side is a
//                          re-exec of this binary with NI_DFLASH_FN=1, not a setenv between assertions.
//                          Both sides build the real Qwen4Exp through its protected NoLoad constructor
//                          (qwen4_exp.h: members only, nothing read, no GPU allocation; ~Qwen4Exp releases
//                          only events the load creates, and it did not run), so the virtual dispatch
//                          asserted is the real class's, read as the engine reads it.
//
// What a CPU cannot test (documented, not asserted):
//   - plan_load's Qwen4Exp exclusion (src/engine/engine.cpp): with the switch off the condition carries
//     `!Qwen4Exp::is_model(model_dir)` again, main's literal term; exercising the found-drafter branch needs a
//     Flash-Next folder whose headers Qwen4Exp::plan reads.
//   - The engine's load routing (src/engine/engine.cpp:390): it reads block_drafter_ok() through a live model.
#include "model/qwen4_exp.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

using namespace ni;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

// The real Qwen4Exp minus the load (members only)
struct Probe : Qwen4Exp {
  Probe() : Qwen4Exp(1, NoLoad{}) {}
};

// A Qwen3.5-family trunk on the same NoLoad config: what the verify flags fall back to with the switch off
struct Trunk : Qwen35 {
  Trunk() : Qwen35("", 1, NoLoad{}) {}
};

int run_off() {
  std::printf("== switch off (default) ==\n");
  const Probe exp;
  const Trunk trunk;
  const Qwen35& base = exp;  // through a base pointer, as the engine reads the flags (engine.cpp:390)
  check(!Qwen4Exp::dflash_fn(), "NI_DFLASH_FN unset: dflash_fn() false (default off, read once at startup)");
  check(!base.block_drafter_ok(),
        "block_drafter_ok() false: main's literal value for this class, so no block drafter binds");
  // main had no verify overrides on Qwen4Exp: both flags were Qwen35's base values (verify_tree_ok reads the
  // config, so the comparison is against the trunk's, not a hardcoded constant)
  check(exp.verify_tree_ok() == trunk.verify_tree_ok(),
        "verify_tree_ok() is Qwen35's base value (main had no override)");
  check(trunk.verify_multi_ok() && exp.verify_multi_ok() == trunk.verify_multi_ok(),
        "verify_multi_ok() is Qwen35's base true (main had no override)");
  const Probe second;  // the read-once guard: a later instance in the same process reads the same value
  check(second.block_drafter_ok() == exp.block_drafter_ok(), "the switch reads once: a second instance agrees");
  std::printf("%s\n", g_fail ? "FAILED" : "off side passed");
  return g_fail ? 1 : 0;
}

int run_on() {
  std::printf("== switch on (NI_DFLASH_FN=1) ==\n");
  const Probe exp;
  const Qwen35& base = exp;
  check(Qwen4Exp::dflash_fn(), "NI_DFLASH_FN=1: dflash_fn() true");
  check(base.block_drafter_ok(), "block_drafter_ok() true: the evidence binaries' value (78416f58)");
  check(!exp.verify_tree_ok(), "verify_tree_ok() false: the evidence binaries' value (chain-only verify)");
  check(!exp.verify_multi_ok(), "verify_multi_ok() false: the evidence binaries' value (no batched verify)");
  std::printf("%s\n", g_fail ? "FAILED" : "on side passed");
  return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (getenv("FN_SWITCH_CHILD")) return run_on();
  unsetenv("NI_DFLASH_FN");  // the off side owns its environment: a preset value cannot fake an off side
  if (run_off()) return 1;
  // dflash_fn() reads once at startup: the on side is a re-exec with the switch in the environment
  std::fflush(nullptr);  // execv replaces the image: unflushed stdio buffers would be lost
  setenv("NI_DFLASH_FN", "1", 1);
  setenv("FN_SWITCH_CHILD", "1", 1);
  execv(argv[0], argv);
  std::printf("FAIL  re-exec failed: %s\n", strerror(errno));
  (void)argc;
  return 1;
}
