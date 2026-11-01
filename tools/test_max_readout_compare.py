#!/usr/bin/env python3

import contextlib
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import uuid
from unittest import mock

import logit_compare
import max_readout_compare as compare

CANDIDATE_COMMITS = ("24265bd0a5282a110a4c280cae66273be45b0670", "7442da53ec66e4fde46c01b96716a2eb48a58950")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class MaxReadoutCompareTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        offset = 0
        for index, width in enumerate((2, 3)):
            keys = [f"option-{x}" for x in range(width)]
            self.rows.append({"request_index": 0, "branch_index": index,
                              "question": {"id": f"q{index}", "type": "choice", "keys": keys, "legend": []},
                              "option_order": keys, "tokens": [4, 8, 15, 16, 23, 42], "decide": 5,
                              "width": width, "byte_offset": offset})
            offset += width * 4
        request = self.root / "request.json"
        layout = self.root / "layout.json"
        write_json(request, {"frozen": "CPU fixture"})
        write_json(layout, {"rows": self.rows})
        self.provenance = {key: "a" * 64 for key in compare.HASH_KEYS}
        self.provenance.update(production_commit=compare.BASELINE_COMMIT, measurement_commit="b" * 40,
                               request_sha256=compare.sha256(request.read_bytes()),
                               layout_sha256=compare.sha256(layout.read_bytes()), readout_temperature=0.75,
                               host={"machine": "test", "chip": "CPU", "os": "test", "build": "test"},
                               environment=dict(compare.NATIVE_ENVIRONMENT, NI_TEST="1"), argv=["systemone", "--once", str(request)],
                               build_flags="-O3", run_id=str(uuid.uuid4()), process_pid=100,
                               process_started_at="2026-11-01T10:00:00Z")
        self.protocol = dict(compare.POLICY, candidate_commits=list(CANDIDATE_COMMITS), context=8192, model_dir="/unused/cpu-fixture",
                             corpus_sha256=self.provenance["request_sha256"],
                             layout_sha256=self.provenance["layout_sha256"], request_file=str(request),
                             layout_file=str(layout), comparator_sha256=compare.numeric_runtime()["logit_compare_sha256"],
                             metric_definition="max(0, row_metrics.kl)")
        self.protocol_path = self.root / "protocol.json"
        write_json(self.protocol_path, self.protocol)

    def capture(self, name, vectors=None, **changes):
        path = self.root / name
        path.mkdir()
        vectors = vectors or ([0.0, 1.0], [0.0, 1.0, 2.0])
        payload = b"".join(struct.pack("<" + "f" * len(row), *row) for row in vectors)
        (path / "logits.f32").write_bytes(payload)
        write_json(path / "manifest.json", {"schema_version": 1, "complete": True,
                                            "payload_bytes": len(payload), "rows": self.rows})
        provenance = dict(self.provenance)
        provenance.update(changes)
        provenance["manifest_sha256"] = compare.sha256((path / "manifest.json").read_bytes())
        provenance["payload_sha256"] = compare.sha256(payload)
        write_json(path / "provenance.json", provenance)
        return path

    def repeat(self, name="repeat", vectors=None, **changes):
        provenance = dict(run_id=str(uuid.uuid4()), process_pid=101, process_started_at="2026-11-01T10:01:00Z")
        provenance.update(changes)
        return self.capture(name, vectors, **provenance)

    def candidate(self, name="candidate", vectors=None):
        return self.capture(name, vectors, production_commit=CANDIDATE_COMMITS[0], binary_sha256="c" * 64)

    def floor(self, baseline=None, repeat=None):
        a = compare.read_capture(baseline or self.capture("base"))
        b = compare.read_capture(repeat or self.repeat())
        return a, compare.make_floor(a, b, self.protocol_path)

    def mutate_manifest(self, path, change):
        manifest, _ = compare.load_json(path / "manifest.json")
        change(manifest)
        write_json(path / "manifest.json", manifest)
        provenance, _ = compare.load_json(path / "provenance.json")
        provenance["manifest_sha256"] = compare.sha256((path / "manifest.json").read_bytes())
        write_json(path / "provenance.json", provenance)

    def test_mixed_width_stock_math_and_no_nll(self):
        a = compare.read_capture(self.capture("base"))
        b = compare.read_capture(self.repeat(vectors=([0.0, 1.125], [0.0, 1.0, 1.875])))
        report = compare.compare(a, b)
        self.assertEqual([len(row[1]) for row in a["values"]], [2, 3])
        for result, left, right in zip(report["rows"], a["values"], b["values"]):
            expected = logit_compare.row_metrics(left, right, None)
            self.assertEqual(result["kl"], expected["kl"])
            self.assertEqual(result["gap"], max(0, expected["kl"]))
            self.assertNotIn("nll_change", result)
        self.assertNotEqual(report["rows"][0]["kl"], logit_compare.row_metrics(b["values"][0], a["values"][0], None)["kl"])
        self.assertEqual(report["global_gap"], max(row["gap"] for row in report["rows"]))

    def test_reject_unknown_candidate_baseline_capture_and_baseline_binary(self):
        a, floor = self.floor()
        cases = (("unknown", {"production_commit": "c" * 40, "binary_sha256": "c" * 64}),
                 ("baseline", {}),
                 ("same-binary", {"production_commit": CANDIDATE_COMMITS[0]}))
        for name, changes in cases:
            with self.subTest(name=name):
                candidate = compare.read_capture(self.capture(name, **changes))
                with self.assertRaises(ValueError):
                    compare.check(a, candidate, floor)

    def test_both_preregistered_candidates_with_distinct_binary_are_accepted(self):
        a, floor = self.floor()
        for index, commit in enumerate(CANDIDATE_COMMITS):
            candidate = compare.read_capture(self.capture("approved-" + str(index), production_commit=commit,
                                                         binary_sha256="c" * 64))
            self.assertTrue(compare.check(a, candidate, floor)["passed"])

    def test_global_floor_applies_to_every_row(self):
        a, floor = self.floor(repeat=self.repeat(vectors=([0.0, 1.5], [0.0, 1.0, 2.0])))
        candidate = compare.read_capture(self.candidate(vectors=([0.0, 1.0], [0.0, 1.125, 2.0])))
        report = compare.check(a, candidate, floor)
        self.assertGreater(report["rows"][1]["gap"], 0)
        self.assertEqual(floor["baseline_report"]["rows"][1]["gap"], 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["threshold"], 2 * floor["global_floor"])

    def test_zero_floor_does_not_substitute_bitwise_gate(self):
        a, floor = self.floor()
        candidate = compare.read_capture(self.candidate("signed-zero", vectors=([-0.0, 1.0], [-0.0, 1.0, 2.0])))
        report = compare.check(a, candidate, floor)
        self.assertEqual(report["threshold"], 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["bitwise_rows"], 0)
        self.assertEqual(report["maxdiff"], 0)

    def test_fixed_failing_gate_and_no_recalibration(self):
        a, floor = self.floor()
        candidate = compare.read_capture(self.candidate("changed", vectors=([3.0, 1.0], [0.0, 1.0, 2.0])))
        original = copy.deepcopy(floor)
        report = compare.check(a, candidate, floor)
        self.assertFalse(report["passed"])
        self.assertGreater(report["rows"][0]["gap"], 0)
        self.assertEqual(floor, original)

    def test_exact_threshold_boundary_without_epsilon(self):
        a, floor = self.floor()
        metrics = {"kl": 0.0, "bitwise": False, "top_ref": 0, "top_cand": 1,
                   "maxdiff": 1.0, "margin": 1.0, "margin2": 1.0, "margin_change": 1.0, "nll_change": None}
        b = compare.read_capture(self.candidate())
        with mock.patch.object(compare, "row_metrics", side_effect=lambda *_: dict(metrics)):
            self.assertTrue(compare.check(a, b, floor)["passed"])
        metrics["kl"] = 5e-324
        with mock.patch.object(compare, "row_metrics", side_effect=lambda *_: dict(metrics)):
            self.assertFalse(compare.check(a, b, floor)["passed"])

    def test_negative_raw_kl_clamps_only_gap(self):
        a = compare.read_capture(self.capture("base"))
        metrics = {"kl": -1e-16, "bitwise": True, "top_ref": 0, "top_cand": 0, "maxdiff": 0, "nll_change": None}
        with mock.patch.object(compare, "row_metrics", side_effect=lambda *_: dict(metrics)):
            report = compare.compare(a, a)
        self.assertEqual(report["rows"][0]["kl"], -1e-16)
        self.assertEqual(report["global_gap"], 0)

    def test_reject_bad_manifest_shapes(self):
        mutations = {
            "partial": lambda m: m.update(complete=False),
            "width": lambda m: m["rows"][0].update(width=3),
            "gap": lambda m: m["rows"][1].update(byte_offset=12),
            "overlap": lambda m: m["rows"][1].update(byte_offset=4),
            "order": lambda m: m["rows"].reverse(),
            "duplicate": lambda m: m["rows"][1].update(branch_index=0),
            "decide": lambda m: m["rows"][0].update(decide=6),
            "tokens": lambda m: m["rows"][0].update(tokens=[True]),
            "options": lambda m: m["rows"][0]["option_order"].reverse(),
            "extra_field": lambda m: m.update(vocab=42),
        }
        for name, change in mutations.items():
            with self.subTest(name=name):
                path = self.capture(name)
                self.mutate_manifest(path, change)
                with self.assertRaises(ValueError):
                    compare.read_capture(path)

    def test_reject_truncated_extra_and_nonfinite_payload(self):
        for name, raw in (("truncated", b"\0" * 19), ("extra", b"\0" * 21),
                          ("nan", struct.pack("<5f", 0, 1, 0, 1, float("nan"))),
                          ("infinite", struct.pack("<5f", 0, 1, 0, 1, float("inf")))):
            with self.subTest(name=name):
                path = self.capture(name)
                (path / "logits.f32").write_bytes(raw)
                provenance, _ = compare.load_json(path / "provenance.json")
                provenance["payload_sha256"] = compare.sha256(raw)
                write_json(path / "provenance.json", provenance)
                with self.assertRaises(ValueError):
                    compare.read_capture(path)

    def test_reject_cross_capture_tokens_metadata_options_and_missing_rows(self):
        a = compare.read_capture(self.capture("base"))
        for name, change in (("tokens", lambda r: r[0]["tokens"].append(99)),
                             ("metadata", lambda r: r[0]["question"].update(id="different")),
                             ("missing", lambda r: r.pop()),
                             ("options", lambda r: (r[0]["option_order"].reverse(), r[0]["question"]["keys"].reverse()))):
            with self.subTest(name=name):
                b = compare.read_capture(self.capture(name))
                change(b["rows"])
                with self.assertRaises(ValueError):
                    compare.compare(a, b)

    def test_reject_identity_mismatch(self):
        a = compare.read_capture(self.capture("base"))
        for key in compare.IDENTITIES:
            with self.subTest(key=key):
                b = compare.read_capture(self.capture("id-" + key))
                b["provenance"][key] = "different"
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    compare.compare(a, b)

    def test_reject_bad_floor_processes_and_production(self):
        a = compare.read_capture(self.capture("base"))
        cases = ({"process_pid": 100}, {"run_id": self.provenance["run_id"]},
                 {"process_started_at": self.provenance["process_started_at"]},
                 {"production_commit": "c" * 40}, {"binary_sha256": "c" * 64}, {"patch_sha256": "c" * 64})
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes):
                b = compare.read_capture(self.repeat("badbase" + str(index), **changes))
                with self.assertRaises(ValueError):
                    compare.make_floor(a, b, self.protocol_path)

    def test_reject_protocol_policy_and_frozen_layout_mismatch(self):
        a = compare.read_capture(self.capture("base"))
        b = compare.read_capture(self.repeat())
        for key, wrong in (("metric", "maxdiff"), ("direction", "candidate||reference"), ("aggregation", "per_prompt"),
                           ("threshold_factor", 2.001), ("baseline_commit", "c" * 40), ("comparator_sha256", "c" * 64)):
            with self.subTest(key=key):
                protocol = dict(self.protocol)
                protocol[key] = wrong
                write_json(self.protocol_path, protocol)
                with self.assertRaises(ValueError):
                    compare.make_floor(a, b, self.protocol_path)
        write_json(self.protocol_path, self.protocol)
        a["rows"][0]["tokens"].append(99)
        with self.assertRaisesRegex(ValueError, "frozen encoded layout"):
            compare.make_floor(a, b, self.protocol_path)

    def test_reject_tampered_floor_and_wrong_reference(self):
        a, floor = self.floor()
        b = compare.read_capture(self.candidate())
        for change in (lambda f: f.update(global_floor=0.1), lambda f: f.update(comparator_sha256="c" * 64),
                       lambda f: f["baseline_provenance"][1].update(process_pid=100)):
            altered = copy.deepcopy(floor)
            change(altered)
            with self.assertRaises(ValueError):
                compare.check(a, b, altered)
        a["provenance_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            compare.check(a, b, floor)

    def test_backend_mismatch_and_nonfinite_metric_rejected(self):
        a, floor = self.floor()
        b = compare.read_capture(self.candidate())
        altered = copy.deepcopy(floor)
        altered["numeric_runtime"]["backend_version"] = "different"
        with self.assertRaisesRegex(ValueError, "numeric backend"):
            compare.check(a, b, altered)
        metrics = {"kl": 0.0, "bitwise": True, "top_ref": 0, "top_cand": 0,
                   "maxdiff": float("inf"), "nll_change": None}
        with mock.patch.object(compare, "row_metrics", side_effect=lambda *_: dict(metrics)):
            with self.assertRaisesRegex(ValueError, "nonfinite comparison metric"):
                compare.compare(a, b)

    def test_duplicate_json_keys_and_hashes_rejected(self):
        path = self.capture("badjson")
        (path / "manifest.json").write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            compare.read_capture(path)
        path = self.capture("badhash")
        (path / "logits.f32").write_bytes(b"\0" * 20)
        with self.assertRaisesRegex(ValueError, "payload hash"):
            compare.read_capture(path)

    def test_stdlib_decoding_uses_stock_math(self):
        with mock.patch.object(logit_compare, "np", None):
            a = compare.read_capture(self.capture("stdlib-a"))
            b = compare.read_capture(self.repeat("stdlib-b", ([0, 1.125], [0, 1, 2])))
            report = compare.compare(a, b)
            self.assertEqual(report["rows"][0]["kl"], logit_compare.row_metrics(a["values"][0], b["values"][0], None)["kl"])

    def test_cli_floor_check_failure_exit_and_exclusive_outputs(self):
        a = self.capture("base")
        b = self.repeat()
        candidate = self.candidate(vectors=([4, 1], [0, 1, 2]))
        floor_path = self.root / "floor.json"
        out = self.root / "report.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(["floor", str(a), str(b), "--protocol", str(self.protocol_path), "--out", str(floor_path)]), 0)
            self.assertEqual(compare.main(["check", str(a), str(candidate), "--floor", str(floor_path), "--out", str(out)]), 1)
            frozen = out.read_bytes()
            self.assertEqual(compare.main(["check", str(a), str(candidate), "--floor", str(floor_path), "--out", str(out)]), 2)
            self.assertEqual(out.read_bytes(), frozen)
            bad_floor = self.root / "bad-floor.json"
            write_json(bad_floor, {"schema_version": 1})
            self.assertEqual(compare.main(["check", str(a), str(self.root / "does-not-exist"), "--floor", str(bad_floor),
                                           "--out", str(self.root / "invalid.json")]), 2)
            self.assertFalse((self.root / "invalid.json").exists())


if __name__ == "__main__":
    unittest.main()
