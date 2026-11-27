#!/usr/bin/env python3
"""CPU-only failure controls for the sampled-token runner, using fake ni_run executables."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

RUNNER = Path(__file__).with_name("test_sampled_tokens.py")


class HarnessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{}')
        tokenizer = {"model": {"type": "BPE", "vocab": {"x": 0, "assistantĊ": 2, "a": 3, "b": 4, "c": 5, "Ġ": 8}},
                     "decoder": {"type": "ByteLevel"}, "added_tokens": [
                         {"id": 1, "content": "<|im_start|>"}, {"id": 6, "content": "<|im_end|>"},
                         {"id": 7, "content": "<|endoftext|>"}]}
        (self.model / "tokenizer.json").write_text(json.dumps(tokenizer))
        self.prompt = self.root / "prompt.json"
        self.prompt.write_text(json.dumps({"ids": [0, 1, 2], "text": "x<|im_start|>assistant\n"}))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "NACE_GPU_", "BENCH_GPU_"))}
        self.env["BENCH_GPU_LOCK"] = str(self.root / "gpu.lock")
        self.out = self.root / "out"
        self.command = [sys.executable, str(RUNNER), "--reference", str(self.root / "reference"),
                        "--candidate", str(self.root / "candidate"), "--model", str(self.model),
                        "--prompt", str(self.prompt), "--out", str(self.out), "--tokens", "3"]
        for arm in ("reference", "candidate"):
            self.driver(arm)

    def driver(self, arm, output="ids: 3 4 5", code=""):
        path = self.root / arm
        path.write_text(f"#!{sys.executable}\n# {arm}\nimport os, sys, time\n"
                        "assert os.environ['NI_LOCK_HELD'] == '1'\n"
                        "assert os.environ['NI_SEED'] == '42'\n"
                        "assert sys.argv[3:6] == ['3', '0.7', '--spec']\n"
                        "assert sys.argv[6] in ('0', '1') and sys.argv[7:] == ['--dflash', 'none']\n"
                        + code + f"\nprint({output!r})\n")
        path.chmod(0o755)

    def run_check(self, error=None, extra=()):
        run = subprocess.run(self.command + list(extra), env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 1 if error else 0, run.stdout + run.stderr)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["passed"], error is None)
        if error:
            self.assertIn(error, report["error"])
        return report

    def test_success_records_actual_ids_and_identity(self):
        report = self.run_check()
        self.assertEqual(report["prompt_tokens"], 3)
        self.assertEqual(report["sampling"]["top_k"], 20)
        self.assertEqual(report["arms"]["candidate"]["eos_count"], 0)
        self.assertEqual(json.loads((self.out / "candidate.ids.json").read_text()), [3, 4, 5])
        self.assertEqual((self.out / "reference.txt").read_text(), "abc")
        self.assertNotEqual(report["arms"]["reference"]["binary_sha256"], report["arms"]["candidate"]["binary_sha256"])
        self.assertIn("tokenizer", report["inputs"])

    def test_first_mismatch_fails(self):
        self.driver("candidate", "ids: 3 5 4")
        self.run_check("mismatch at index 1: reference=4, candidate=5")

    def test_empty_missing_duplicate_malformed_ids_fail(self):
        for i, output in enumerate(("", "ids:", "ids: 3 4 x", "ids: -1 4 5", "ids: 3 4 5\nids: 3 4 5")):
            with self.subTest(output=output):
                self.out = self.root / f"out-{i}"
                self.driver("candidate", output)
                self.run_check("valid ids: line", ("--out", str(self.out)))

    def test_equal_truncation_is_not_success(self):
        for arm in ("reference", "candidate"):
            self.driver(arm, "ids: 3 4")
        self.run_check("expected 3 tokens")

    def test_equal_empty_text_is_not_success(self):
        for arm in ("reference", "candidate"):
            self.driver(arm, "ids: 8 8 8")
        self.run_check("nonempty text")

    def test_eos_at_last_position_is_not_success(self):
        self.driver("candidate", "ids: 3 4 6")
        self.run_check("no EOS")

    def test_unknown_token_fails(self):
        self.driver("candidate", "ids: 3 4 1000")
        self.run_check("invalid token ID")

    def test_nonzero_exit_does_not_pass_with_matching_ids(self):
        self.driver("candidate", code="print('ids: 3 4 5'); sys.exit(9)")
        report = self.run_check("ni_run exited 9")
        self.assertEqual(report["arms"]["candidate"]["returncode"], 9)

    def test_timeout_stops_child(self):
        pidfile = self.root / "child.pid"
        self.driver("candidate", code=f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(30)")
        self.run_check("timed out", ("--timeout", "1"))
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)

    def test_same_binary_bytes_rejected(self):
        (self.root / "candidate").write_bytes((self.root / "reference").read_bytes())
        self.run_check("different builds")

    def test_wrong_tokenizer_for_frozen_prompt_rejected(self):
        self.prompt.write_text(json.dumps({"ids": [3, 4], "text": "x<|im_start|>assistant\n"}))
        self.run_check("prompt must decode exactly")

    def test_tokenizer_mutation_rejected(self):
        path = self.model / "tokenizer.json"
        self.driver("reference", code=f"open({str(path)!r}, 'a').write(' ')")
        self.run_check("tokenizer changed during run")

    def test_override_rejected(self):
        self.env["NI_QWEN_SAMPLE_SG"] = "1"
        self.run_check("unset NI_* overrides")

    def test_mtp_fallback_does_not_count_as_speculative_coverage(self):
        self.run_check("MTP head / accepted speculative tokens not observed", ("--spec", "1"))

    def test_mtp_requires_loaded_head_and_accepted_tokens(self):
        for arm in ("reference", "candidate"):
            self.driver(arm, "MTP head loaded from model\nspec decode D=1: 2 tok in 1 ms (2000 tok/s), "
                        "2 cycles, 1.00 tok/cycle, accepted/drafted 0.50\nids: 3 4 5")
        report = self.run_check(extra=("--spec", "1"))
        self.assertEqual(report["arms"]["candidate"]["cycles"], 2)

    def test_existing_output_is_not_overwritten(self):
        self.out.mkdir()
        marker = self.out / "report.json"
        marker.write_text("old evidence")
        run = subprocess.run(self.command, env=self.env, capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 1)
        self.assertEqual(marker.read_text(), "old evidence")

    def test_held_lock_blocks_execution(self):
        with open(self.env["BENCH_GPU_LOCK"], "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            child = subprocess.Popen(self.command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while not (self.out / "prompt.ids").exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((self.out / "prompt.ids").exists())
                self.assertIsNone(child.poll())
                self.assertFalse((self.out / "reference.log").exists())
                fcntl.flock(lock, fcntl.LOCK_UN)
                stdout, stderr = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stdout + stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()

    def test_caller_lock_marker_avoids_nested_lock(self):
        for marker in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD"):
            with self.subTest(marker=marker), open(self.env["BENCH_GPU_LOCK"], "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.env[marker] = "1"
                self.out = self.root / marker
                self.run_check(extra=("--out", str(self.out)))
                del self.env[marker]


if __name__ == "__main__":
    unittest.main()
