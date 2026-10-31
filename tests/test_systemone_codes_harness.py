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
RUN_DRIVER = CODES.CodesTests.run_driver


class SyntheticPathUsed(RuntimeError):
    pass


def stand_in_result(widths):
    branches = [{"tokens": [42, 43], "decide": 1, "options": list(range(width))} for width in widths.values()]
    return {"state": [], "branches": branches,
            "meta": [{"id": name, "type": "choice", "keys": [f"key{i}" for i in range(width)], "legend": []}
                     for name, width in widths.items()],
            "total_tokens": 2 * len(branches), "max_branch": 2}


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
        for name in ("pair", "fresh_after_wide"):
            self.reference[name] = stand_in_result({"q": 2})
        for width in (28, 40, 255):
            self.reference[f"width{width}"] = stand_in_result({"q": width})
        self.reference["mixed"] = stand_in_result({"small": 2, "wide": 40, "small_again": 3})
        self.candidate = copy.deepcopy(self.reference)
        self.calls = []
        self.output = io.StringIO()
        patches = [mock.patch.object(CODES, "TOKENIZER", self.tokenizer),
                   mock.patch.object(CODES, "REFERENCE", Path("reference")),
                   mock.patch.object(CODES, "CANDIDATE", Path("candidate")),
                   mock.patch.object(CODES.CodesTests, "run_driver", side_effect=self.driver),
                   mock.patch.object(CODES.CodesTests, "golden", side_effect=SyntheticPathUsed("synthetic golden")),
                   mock.patch.object(CODES.tempfile, "TemporaryDirectory", side_effect=SyntheticPathUsed("synthetic creation")),
                   mock.patch.object(CODES.importlib.util, "spec_from_file_location",
                                     side_effect=SyntheticPathUsed("synthetic fixture import")),
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
        self.assertFalse(issubclass(SyntheticPathUsed, AssertionError))
        self.assertEqual(self.tokenizer.read_bytes(), self.contents)
        lines = self.output.getvalue().splitlines()
        self.assertIn("real-vocab: identity NOT established unless tokenizer_sha256 matches the approved Drex Max v1 release record",
                      lines)
        identity = json.loads(lines[0].removeprefix("real-vocab: "))
        self.assertEqual(identity, {"tokenizer": str(self.tokenizer.resolve()),
                                    "tokenizer_sha256": hashlib.sha256(self.contents).hexdigest(),
                                    "model_dir": str(self.tokenizer.resolve().parent), "model_type": "BPE",
                                    "ignore_merges": True, "pre_tokenizer": {"type": "ByteLevel"}})
        accepted = ["width28", "width40", "width255", "mixed", "pair", "fresh_after_wide"]
        rejected = [name for name in self.names if name not in accepted]
        for role in ("reference", "candidate"):
            self.assertIn(f"real-vocab/{role}: 104 requests, 6 accepted, 98 rejected", lines)
            self.assertIn(f"real-vocab/{role}: accepted cases {json.dumps(accepted)}", lines)
            self.assertIn(f"real-vocab/{role}: rejected cases {json.dumps(rejected)}", lines)

    def test_required_real_cases_are_preregistered(self):
        self.assertEqual(CODES.REAL_REQUIRED_CASES,
                         ("pair", "fresh_after_wide", "width28", "width40", "width255", "mixed"))

    def test_only_two_accepted_cases_is_not_success(self):
        for name in ("width28", "width40", "width255", "mixed"):
            self.reference[name] = {"error": "BadRequest", "message": "wide codes unavailable"}
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "reference: required real-vocabulary case width28 was rejected"):
            self.checker.test_real_tokenizer_differential()
        for role in ("reference", "candidate"):
            self.assertIn(f"real-vocab/{role}: 104 requests, 2 accepted, 102 rejected", self.output.getvalue())

    def test_any_required_wide_case_rejection_or_wrong_width_is_not_success(self):
        baseline = copy.deepcopy(self.reference)
        for role in ("reference", "candidate"):
            for name in ("width28", "width40", "width255", "mixed"):
                for failure in ("rejection", "meta_width"):
                    with self.subTest(driver=role, case=name, failure=failure):
                        self.reference = copy.deepcopy(baseline)
                        self.candidate = copy.deepcopy(baseline)
                        results = self.reference if role == "reference" else self.candidate
                        if failure == "rejection":
                            results[name] = {"error": "BadRequest", "message": "wide codes unavailable"}
                            reason = "was rejected"
                        else:
                            row = 1 if name == "mixed" else 0
                            results[name]["meta"][row]["keys"].pop()
                            results[name]["branches"][row]["options"].pop()
                            reason = "has wrong meta key counts"
                        with self.assertRaisesRegex(AssertionError, f"{role}: required real-vocabulary case {name} {reason}"):
                            self.checker.test_real_tokenizer_differential()

    def test_equal_all_errors_is_not_success(self):
        self.reference = {name: {"error": "BadRequest", "message": "no usable codes"} for name in self.names}
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "reference: required real-vocabulary case pair was rejected"):
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
        with self.assertRaisesRegex(AssertionError, "pair: nonnegative integer token IDs required"):
            self.checker.test_real_tokenizer_differential()

    def test_equal_wrong_option_count_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.reference[name]["branches"][0]["options"] = [0]
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "pair must expose exactly two options"):
            self.checker.test_real_tokenizer_differential()

    def test_different_rejection_message_is_not_success(self):
        self.candidate["seeded35"]["message"] = "different failure"
        with self.assertRaisesRegex(AssertionError, "seeded35: candidate differs from reference"):
            self.checker.test_real_tokenizer_differential()

    def test_different_token_id_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.candidate[name]["branches"][0]["tokens"] = [42, 44]
        with self.assertRaisesRegex(AssertionError, "pair: candidate differs from reference"):
            self.checker.test_real_tokenizer_differential()

    def test_equal_inconsistent_token_count_is_not_success(self):
        for name in ("pair", "fresh_after_wide"):
            self.reference[name]["total_tokens"] = 0
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "pair: total_tokens does not match branches"):
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
            with self.assertRaisesRegex(FileNotFoundError, r"missing\.json"):
                self.checker.test_real_tokenizer_differential()
        self.assertFalse(missing.exists())
        self.assertEqual(self.calls, [])

    def test_missing_pair_is_not_success(self):
        del self.reference["pair"]
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "reference: case names do not match requests"):
            self.checker.test_real_tokenizer_differential()

    def test_pair_different_from_fresh_is_not_success(self):
        self.reference["fresh_after_wide"]["branches"][0]["tokens"] = [42, 44]
        self.candidate = copy.deepcopy(self.reference)
        with self.assertRaisesRegex(AssertionError, "pair differs from fresh_after_wide"):
            self.checker.test_real_tokenizer_differential()

    def test_nonzero_driver_exit_is_not_success(self):
        proc = CODES.subprocess.CompletedProcess([], 7, stdout="", stderr="stand-in failure")
        with mock.patch.object(CODES.subprocess, "run", return_value=proc) as run:
            with self.assertRaisesRegex(AssertionError, "driver exited nonzero: stand-in failure"):
                RUN_DRIVER(self.checker, Path("reference"), self.tokenizer, CODES.corpus())
        run.assert_called_once()

    def test_wrong_driver_line_count_is_not_success(self):
        proc = CODES.subprocess.CompletedProcess([], 0, stdout="{}\n", stderr="")
        with mock.patch.object(CODES.subprocess, "run", return_value=proc) as run:
            with self.assertRaisesRegex(AssertionError, "driver output line count does not match requests"):
                RUN_DRIVER(self.checker, Path("reference"), self.tokenizer, CODES.corpus())
        run.assert_called_once()

    def test_missing_or_invalid_model_type_refuses_before_drivers(self):
        for data in ({}, {"model": {}}, {"model": None}, {"model": {"type": None}}, {"model": {"type": " "}}):
            with self.subTest(metadata=data):
                self.tokenizer.write_text(json.dumps(data))
                with self.assertRaisesRegex(AssertionError, r"tokenizer metadata requires nonempty model\.type"):
                    self.checker.test_real_tokenizer_differential()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
