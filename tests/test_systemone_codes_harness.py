#!/usr/bin/env python3
"""Check script failure modes with stand-in results, not a tokenizer or encoder."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("systemone_codes", Path(__file__).with_name("test_systemone_codes.py"))
CODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CODES)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tokenizer = Path(temporary.name) / "tokenizer.json"
        self.contents = json.dumps({"model": {"type": "BPE", "ignore_merges": True},
                                    "pre_tokenizer": {"type": "ByteLevel"}}).encode()
        self.tokenizer.write_bytes(self.contents)
        self.names = list(CODES.corpus())
        self.reference = {name: {"error": "BadRequest", "message": "stand-in rejection"} for name in self.names}
        pair = {"state": [], "branches": [{"tokens": [42, 43], "decide": 1, "options": [0, 1]}],
                "meta": [{"id": "q", "type": "choice", "keys": ["key0", "key1"], "legend": []}],
                "total_tokens": 2, "max_branch": 2}
        for name in ("pair", "fresh_after_wide"):
            self.reference[name] = copy.deepcopy(pair)
        self.candidate = copy.deepcopy(self.reference)
        self.calls = []
        self.output = io.StringIO()
        patches = [mock.patch.object(CODES, "TOKENIZER", self.tokenizer),
                   mock.patch.object(CODES, "REFERENCE", Path("reference")),
                   mock.patch.object(CODES, "CANDIDATE", Path("candidate")),
                   mock.patch.object(CODES.CodesTests, "run_driver", side_effect=self.driver),
                   mock.patch.object(CODES.CodesTests, "golden", side_effect=AssertionError("synthetic golden")),
                   mock.patch.object(CODES.tempfile, "TemporaryDirectory", side_effect=AssertionError("synthetic creation")),
                   mock.patch.object(CODES.importlib.util, "spec_from_file_location",
                                     side_effect=AssertionError("synthetic fixture import")),
                   mock.patch.object(CODES.sys, "stderr", self.output)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.checker = CODES.CodesTests("test_real_tokenizer_differential")

    def driver(self, driver, tokenizer, cases):
        self.calls.append((driver, tokenizer, list(cases)))
        return copy.deepcopy(self.reference if driver == Path("reference") else self.candidate)

    def test_real_mode_compares_all_cases_and_reports_identity_without_synthetic_work(self):
        self.checker.test_real_tokenizer_differential()
        self.assertEqual(self.calls, [(Path("reference"), self.tokenizer, self.names),
                                     (Path("candidate"), self.tokenizer, self.names)])
        self.assertEqual(len(self.names), 104)
        self.assertEqual(self.tokenizer.read_bytes(), self.contents)
        lines = self.output.getvalue().splitlines()
        identity = json.loads(lines[0].removeprefix("real-vocab: "))
        self.assertEqual(identity, {"tokenizer": str(self.tokenizer.resolve()),
                                    "tokenizer_sha256": hashlib.sha256(self.contents).hexdigest(),
                                    "model_dir": str(self.tokenizer.resolve().parent), "model_type": "BPE",
                                    "ignore_merges": True, "pre_tokenizer": {"type": "ByteLevel"}})
        for role in ("reference", "candidate"):
            self.assertIn(f"real-vocab/{role}: 104 requests, 2 accepted, 102 rejected", lines)

    def test_equal_all_errors_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.reference[name] = {"error": "BadRequest", "message": "no usable codes"}
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "known-valid two-option"):
            self.checker.test_real_tokenizer_differential()
        self.assertIn("104 requests, 0 accepted, 104 rejected", self.output.getvalue())

    def test_equal_empty_tokens_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.reference[name]["branches"][0]["tokens"] = []
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "must produce tokens"):
            self.checker.test_real_tokenizer_differential()

    def test_equal_negative_token_ids_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.reference[name]["branches"][0]["tokens"] = [-1, -1]
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaises(AssertionError):
            self.checker.test_real_tokenizer_differential()

    def test_equal_wrong_option_count_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.reference[name]["branches"][0]["options"] = [0]
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaises(AssertionError):
            self.checker.test_real_tokenizer_differential()

    def test_different_rejection_message_is_not_success(self):
        self.candidate["seeded35"]["message"] = "different failure"
        with self.assertRaises(AssertionError):
            self.checker.test_real_tokenizer_differential()

    def test_different_token_id_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.candidate[name]["branches"][0]["tokens"] = [42, 44]
        with self.assertRaises(AssertionError):
            self.checker.test_real_tokenizer_differential()

    def test_equal_inconsistent_token_count_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.reference[name]["total_tokens"] = 0
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaises(AssertionError):
            self.checker.test_real_tokenizer_differential()

    def test_tokenizer_change_stops_before_second_driver(self):
        def changing_driver(driver, tokenizer, cases):
            result = self.driver(driver, tokenizer, cases)
            tokenizer.write_bytes(b"changed stand-in metadata")
            return result
        with mock.patch.object(CODES.CodesTests, "run_driver", side_effect=changing_driver):
            with self.assertRaisesRegex(AssertionError, "tokenizer changed during reference run"):
                self.checker.test_real_tokenizer_differential()
        self.assertEqual(len(self.calls), 1)

    def test_missing_tokenizer_is_not_created(self):
        missing = self.tokenizer.with_name("missing.json")
        with mock.patch.object(CODES, "TOKENIZER", missing):
            with self.assertRaises(FileNotFoundError):
                self.checker.test_real_tokenizer_differential()
        self.assertFalse(missing.exists())
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
