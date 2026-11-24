#!/usr/bin/env python3
"""Compile the production planning selector with a simulated OS and forbidden GPU access.

Only the two profile helpers and the planner's scratch-selection expression are
needed. Link the real profile policy, not Metal, so an older-OS case can run on
any CI host and fail immediately if argument evaluation reaches current_device.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InklingPlanProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="inkling-plan-profile-")
        cls.addClassCleanup(cls.temp.cleanup)
        source = (ROOT / "src/model/inkling2.cpp").read_text()
        helpers = source.split("static Profile ink2_profile()", 1)[1].split(
            "\nbool Inkling2::fast_profile_available()", 1
        )[0]
        helpers = "static Profile ink2_profile()" + helpers
        plan = source.split("MemPlan Inkling2::plan(", 1)[1].split("\nInkling2::Inkling2(", 1)[0]
        selector = re.search(r"const size_t moe_tmp_bytes =\s*(.*?)\s*\?", plan, re.S).group(1)
        harness = r'''
#include "engine/device_profile.h"
#include "engine/profile.h"
#include <cstdio>
#include <cstdlib>
#include <string>

namespace ni {
static bool tensor_api_available = false;
static int device_reads = 0;
const DeviceProfile& current_device() {
  ++device_reads;
  if (!tensor_api_available) {
    std::fputs("FAIL: live-device access forbidden on older macOS\n", stderr);
    std::exit(42);
  }
  static const auto device = resolve_device({.apple_family = 10, .cores = 20});
  return device;
}
bool default_on(Knob k, const ModelFacts& m) { return default_on(k, current_device(), m); }
#define __builtin_available(...) tensor_api_available
''' + helpers + r'''
#undef __builtin_available
int check(bool available, bool expected) {
  tensor_api_available = available;
  const bool selected = ''' + selector + r''';
  if (selected != expected || (!available && device_reads != 0)) return 1;
  std::printf("PASS: tensor_api=%d selected=%d device_reads=%d\n", available, selected, device_reads);
  return 0;
}
}  // namespace ni
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  return ni::check(std::string(argv[1]) == "new", std::string(argv[2]) == "1");
}
'''
        cpp = Path(cls.temp.name) / "selector.cpp"
        cpp.write_text(harness)
        cls.binary = Path(cls.temp.name) / "selector"
        compiler = shutil.which("c++")
        if not compiler:
            raise RuntimeError("C++ compiler required for the planning regression")
        subprocess.run([
            compiler, "-std=c++20", "-O0", "-I", str(ROOT / "src"), str(cpp),
            str(ROOT / "src/engine/profile.cpp"),
            str(ROOT / "src/engine/device_profile.cpp"),
            str(ROOT / "src/ane/default_share.cpp"), "-o", str(cls.binary),
        ], check=True, capture_output=True, text=True, timeout=60)

    def check_selector(self, os_version, profile, tensor, expected):
        env = dict(os.environ)
        for key, value in (("KAPITSA_PROFILE", profile), ("NI_INK2_TENSOR", tensor)):
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        result = subprocess.run([
            str(self.binary), os_version, str(int(expected)),
        ], env=env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS:", result.stdout)
        if os_version == "old":
            self.assertIn("device_reads=0", result.stdout)

    def test_older_macos_never_reads_live_device(self):
        for profile in (None, "", "precise", "fast", "invalid"):
            for tensor in (None, "", "0", "1"):
                with self.subTest(profile=profile, tensor=tensor):
                    self.check_selector("old", profile, tensor, False)

    def test_available_api_keeps_profile_and_member_overrides(self):
        for profile, tensor, expected in (
            (None, None, True), ("fast", None, True), ("precise", None, False),
            ("invalid", None, False), ("fast", "0", False), ("precise", "1", True),
        ):
            with self.subTest(profile=profile, tensor=tensor):
                self.check_selector("new", profile, tensor, expected)


if __name__ == "__main__":
    unittest.main()
