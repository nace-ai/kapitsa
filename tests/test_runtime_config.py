#!/usr/bin/env python3
"""Native helper configuration validation. --check never loads a model or uses Metal."""
import json
import subprocess
import sys
import unittest

HELPER = sys.argv.pop(1) if len(sys.argv) > 1 else "build/kapitsa-runtime"


def check_config(value):
    payload = value if isinstance(value, str) else json.dumps(value)
    return subprocess.run([HELPER, "--check"], input=payload, text=True,
                          capture_output=True, timeout=10)


class RuntimeConfigurationTests(unittest.TestCase):
    def test_defaults_and_explicit_launch_options(self):
        for configuration in [
            {"protocol": 1, "model": "missing-test-model"},
            {"protocol": 1, "instance_id": "code-1", "name": "Code", "model": "missing-test-model",
             "port": 8100, "context": 32768, "slots": 2, "mem_budget_gb": 16,
             "spec": 2, "prefill_chunk": 512, "dflash": "/missing/draft", "disk_cache": "2G"},
        ]:
            with self.subTest(configuration=configuration):
                result = check_config(configuration)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), {"protocol": 1, "status": "valid"})

    def test_invalid_options_are_rejected(self):
        invalid = [
            ("protocol", 2), ("protocol", True), ("instance_id", "../escape"),
            ("name", ""), ("model", ""), ("model", 42), ("port", 0), ("port", 65536),
            ("port", 1.5), ("slots", 0), ("slots", 5), ("context", 0),
            ("context", 1048577), ("prefill_chunk", 8193), ("spec", -2),
            ("mem_budget_gb", 0), ("mem_budget_gb", 4097), ("disk_cache", "bad"),
            ("host", "0.0.0.0"), ("api_key", "unexpected"),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                result = check_config({"protocol": 1, "model": "missing-test-model", key: value})
                self.assertEqual(result.returncode, 2)
                self.assertFalse(result.stdout)
                self.assertTrue(result.stderr.startswith("kapitsa-runtime:"))

    def test_malformed_missing_and_oversized_configuration(self):
        for value in ["{", "[]", {"protocol": 1}, " " * 16385]:
            with self.subTest(value=str(value)[:80]):
                result = check_config(value)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(result.stdout)


if __name__ == "__main__":
    unittest.main()
