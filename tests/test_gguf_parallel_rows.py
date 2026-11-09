#!/usr/bin/env python3
"""CPU regression for the importer helper, compiled verbatim without loading a model.

Inject bad_alloc at each caller-thread allocation during worker launch. No resource
exhaustion or production fault hook is needed. --source can select a reference tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

PREAMBLE = r"""
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>
static thread_local bool counting = false;
static thread_local int allocations = 0, fail_at = 0;
static std::atomic<int> started{0}, finished{0};
void* operator new(std::size_t n) {
  if (counting && ++allocations == fail_at) throw std::bad_alloc();
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
"""

DRIVER = r"""
static void check(bool value) { if (!value) std::exit(1); }
static void controls() {
  int cases = 0;
  for (uint64_t n : {0u, 1u, 2u, 3u, 7u, 64u, 101u}) {
    for (int t : {-2, 0, 1, 2, 4, 8, 32}) {
      std::vector<std::atomic<int>> seen(n);
      std::atomic<int> calls{0};
      parallel_rows(n, t, [&](uint64_t a, uint64_t b) {
        check(a <= b && b <= n);
        ++calls;
        for (uint64_t i = a; i < b; ++i) ++seen[i];
      });
      for (auto& x : seen) check(x == 1);
      check(calls >= 1);
      if (n == 0 || t <= 1) check(calls == 1);
      ++cases;
    }
  }
  for (int t : {1, 4}) {
    std::atomic<int> exits{0};
    bool caught = false;
    try {
      parallel_rows(64, t, [&](uint64_t a, uint64_t) {
        struct Exit { std::atomic<int>& n; ~Exit() { ++n; } } exit{exits};
        if (a == 0) throw 1234;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      });
    } catch (int value) { caught = value == 1234; }
    check(caught && exits == t);
    ++cases;
  }
  std::printf("{\"controls\":%d}\n", cases);
}
int main(int argc, char** argv) {
  if (argc == 1) { controls(); return 0; }
  std::set_terminate([] {
    std::fprintf(stderr, "terminate: allocations=%d started=%d finished=%d\n",
                 allocations, started.load(), finished.load());
    std::_Exit(86);
  });
  fail_at = std::atoi(argv[1]);
  const bool worker_error = argc > 2;
  int caught = 0;
  counting = true;
  try {
    parallel_rows(64, 4, [&](uint64_t a, uint64_t) {
      ++started;
      struct Exit { ~Exit() { ++finished; } } exit;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (worker_error && a == 0) throw 1234;
    });
  } catch (const std::bad_alloc&) { caught = 1; }
    catch (int value) { caught = value == 1234 ? 2 : 3; }
  counting = false;
  std::printf("{\"caught\":%d,\"allocations\":%d,\"started\":%d,\"finished\":%d}\n",
              caught, allocations, started.load(), finished.load());
  const bool injected = fail_at > 0 && allocations >= fail_at;
  check(caught == (injected ? 1 : worker_error ? 2 : 0));
  check(started == finished);
  if (!injected) check(finished == 4);
}
"""


def extract(source):
    start = source.index("template <class F>\nvoid parallel_rows(")
    opening = source.index("{", start)
    depth = 1
    for i in range(opening + 1, len(source)):
        depth += (source[i] == "{") - (source[i] == "}")
        if not depth:
            return source[start:i + 1]
    raise ValueError("unterminated parallel_rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "src/gguf/import.cpp")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    function = extract(args.source.read_text())
    print("helper_sha256=" + hashlib.sha256(function.encode()).hexdigest(), flush=True)
    with tempfile.TemporaryDirectory(prefix="gguf-rows-test-") as directory:
        cpp = Path(directory) / "test.cpp"
        binary = Path(directory) / "test"
        cpp.write_text(PREAMBLE + function + "\n" + DRIVER)
        command = [os.environ.get("CXX", "clang++"), "-std=c++20", "-O3", "-pthread"]
        if args.sanitize:
            command += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
        subprocess.run(command + [str(cpp), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=30)
        failures = []
        for worker_error in (False, True):
            suffix = ["worker-error"] if worker_error else []
            control = subprocess.run([str(binary), "0", *suffix], capture_output=True, text=True, timeout=30)
            control.check_returncode()
            count = json.loads(control.stdout)["allocations"]
            if count < 4:
                raise AssertionError("launch allocation injection did not engage")
            print(json.dumps({"fail_at": 0, "worker_error": worker_error,
                              "rc": control.returncode, "result": json.loads(control.stdout)}), flush=True)
            for nth in range(1, count + 2):
                result = subprocess.run([str(binary), str(nth), *suffix],
                                        capture_output=True, text=True, timeout=30)
                print(json.dumps({"fail_at": nth, "worker_error": worker_error,
                                  "rc": result.returncode, "stdout": result.stdout.strip(),
                                  "stderr": result.stderr.strip()}), flush=True)
                if result.returncode:
                    failures.append((worker_error, nth, result.returncode))
        if failures:
            raise AssertionError(f"launch/unwind failures: {failures}")
    print("PASS row coverage, worker errors, launch allocation failures and joined lifetime")


if __name__ == "__main__":
    main()
