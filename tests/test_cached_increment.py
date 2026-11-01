#!/usr/bin/env python3
"""Cached-increment workload against a scripted HTTP server; no model or GPU, stdlib only."""
import fcntl
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench/harness"))
import bench  # noqa: E402
import cached_increment as ci  # noqa: E402
import compare_engines  # noqa: E402


class Fake(http.server.BaseHTTPRequestHandler):
    mode = "normal"
    records = []
    sessions = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        sid = self.headers.get("X-Session-Id")
        previous = self.sessions.get(sid)
        pt = 32 + sum(len(m["content"]) for m in body["messages"]) // 4
        cached = previous["prompt_tokens"] if previous else 0
        turn = (len(body["messages"]) - 2) // 2
        answer = f"ACK {len(self.records)}: α\n"
        self.records.append({"body": body, "session": sid, "answer": answer, "path": self.path})
        self.sessions[sid] = {"prompt_tokens": pt, "messages": body["messages"]}
        if self.mode == "http_error":
            self.send_error(503, "scripted failure")
            return
        if self.mode == "no_cache":
            cached = 0
        if previous and self.mode == "excess_cache":
            cached = pt + 1
        if previous and self.mode == "truncated_prompt":
            pt = previous["prompt_tokens"]
        usage = {"prompt_tokens": pt, "completion_tokens": 6}
        if self.mode != "missing_cache":
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        finish = "length" if self.mode == "length" or (self.mode == "late_length" and turn) else "stop"
        content = "" if self.mode == "empty" else answer
        events = [{"choices": [{"delta": {"role": "assistant"}}]}]
        if self.mode == "thinking":
            events.append({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
        events += [{"choices": [{"delta": {"content": content[:4]}}]},
                   {"choices": [{"delta": {"content": content[4:]}}]},
                   {"choices": [{"delta": {}, "finish_reason": finish}]},
                   {"choices": [], "usage": usage}]
        wire = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
        encoded = wire.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class IncrementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Fake.mode, Fake.records, Fake.sessions = "normal", [], {}

    def argv(self, *extra):
        return ["--base-url", self.base, "--model", "scripted", "--label", "smoke-increment",
                "--cached-increment", "--increment-prefixes", "256,512", "--increment-tokens", "128",
                "--increment-turns", "2", "--increment-repeats", "2", "--max-ctx", "4096",
                "--turn-gap", "0", "--warmup", "none", "--skip", "decode,greedy,prefill,tools,agent", *extra]

    def run_phase(self, *extra):
        a = bench.parse_args(self.argv(*extra))
        return ci.run(a, log=lambda m: None)

    def test_growth_and_replay(self):
        r = self.run_phase()
        self.assertFalse(r["invalid"])
        self.assertFalse(r["flags"])
        self.assertEqual([s["prefix_target"] for s in r["series"]], [256, 512, 512, 256])
        self.assertTrue(all(s["complete"] for s in r["series"]))
        self.assertEqual(len(Fake.records), 12)
        self.assertEqual(len({x["session"] for x in Fake.records}), 4)
        corpus = bench.load_corpus()
        for i, seq in enumerate(r["series"]):
            requests = Fake.records[3 * i:3 * i + 3]
            previous_messages = None
            for j, (wire, rec) in enumerate(zip(requests, seq["requests"])):
                body, msgs = wire["body"], wire["body"]["messages"]
                self.assertEqual(wire["path"], "/v1/chat/completions")
                self.assertEqual(wire["session"], seq["session"])
                self.assertTrue(msgs[0]["content"].startswith("[" + seq["session"] + "]"))
                self.assertEqual(body["temperature"], 0)
                self.assertEqual((body["top_k"], body["top_p"], body["min_p"]), (1, 1, 0))
                self.assertEqual(body["reasoning_effort"], "none")
                self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
                self.assertTrue(body["stream_options"]["include_usage"])
                self.assertEqual(body["max_tokens"], ci.ANSWER_TOKENS)
                off, size = rec["corpus_offset"], rec["corpus_chars"]
                self.assertEqual(msgs[-1]["content"], ci.ACK + corpus[off:off + size])
                self.assertEqual(rec["answer"], wire["answer"])
                if previous_messages:
                    self.assertEqual(msgs[:-2], previous_messages)
                    self.assertEqual(msgs[-2], {"role": "assistant", "content": requests[j - 1]["answer"]})
                    self.assertEqual(rec["cached_tokens"], seq["requests"][j - 1]["prompt_tokens"])
                    self.assertEqual(rec["cache_status"], "reused")
                previous_messages = msgs
        self.assertTrue(all(s["verified_samples"] == 2 and s["median_ttft_s"] > 0 for s in r["summary"]))

    def test_unknown_cache_is_not_zero(self):
        Fake.mode = "missing_cache"
        r = self.run_phase("--increment-prefixes", "256", "--increment-repeats", "1")
        self.assertFalse(r["invalid"])
        self.assertTrue(r["flags"])
        for rec in r["series"][0]["requests"]:
            self.assertIsNone(rec["cached_tokens"])
            self.assertIsNone(rec["uncached_tokens"])
            self.assertIsNone(rec["uncached_tokens_per_ttft_s"])
            self.assertGreater(rec["ttft_s"], 0)
        self.assertTrue(all(s["verified_samples"] == 0 and s["median_ttft_s"] is None for s in r["summary"]))

    def test_cache_misses_are_retained_and_excluded_from_cached_summary(self):
        Fake.mode = "no_cache"
        r = self.run_phase("--increment-prefixes", "256", "--increment-repeats", "1")
        self.assertTrue(r["series"][0]["complete"])
        self.assertFalse(r["invalid"])
        for rec in r["series"][0]["requests"][1:]:
            self.assertEqual(rec["cached_tokens"], 0)
            self.assertEqual(rec["uncached_tokens"], rec["prompt_tokens"])
            self.assertEqual(rec["cache_status"], "miss")
        self.assertTrue(all(s["samples"] == 1 and s["verified_samples"] == 0 for s in r["summary"]))

    def test_bad_responses_stop_the_series(self):
        for mode, expected in (("length", 1), ("empty", 1), ("thinking", 1), ("http_error", 1),
                               ("late_length", 2), ("excess_cache", 2), ("truncated_prompt", 2)):
            with self.subTest(mode=mode):
                self.setUp()
                Fake.mode = mode
                r = self.run_phase("--increment-prefixes", "256", "--increment-repeats", "1")
                self.assertTrue(r["invalid"])
                self.assertFalse(r["series"][0]["complete"])
                self.assertEqual(len(Fake.records), expected)
                self.assertTrue(all(s["verified_samples"] == 0 for s in r["summary"]))

    def test_rate_uses_uncached_tokens_and_actual_growth(self):
        response = {"usage": {"prompt_tokens": 150, "prompt_tokens_details": {"cached_tokens": 100}},
                    "ttft": 2.0, "total": 3.0, "finish": "stop", "content": "ACK"}
        rec = ci.sample(response, 100, 1, 4096)
        self.assertFalse(rec["invalid"])
        self.assertEqual(rec["uncached_tokens"], 50)
        self.assertEqual(rec["prompt_growth_tokens"], 50)
        self.assertEqual(rec["uncached_tokens_per_ttft_s"], 25.0)
        for bad in (None, 0, -1, True):
            with self.subTest(prompt=bad):
                x = dict(response, usage={"prompt_tokens": bad})
                self.assertTrue(ci.sample(x, 100, 1, 4096)["invalid"])
        for bad in (None, 0, -1, True, float("nan"), float("inf")):
            with self.subTest(ttft=bad):
                rec = ci.sample(dict(response, ttft=bad), 100, 1, 4096)
                self.assertTrue(rec["invalid"])
                self.assertIsNone(rec["uncached_tokens_per_ttft_s"])

    def test_cache_counters_are_validated_before_combining_sources(self):
        response = {"ttft": 1.0, "total": 2.0, "finish": "stop", "content": "ACK"}
        for bad in (-1, 151, True, 100.5, "100", float("nan"), float("inf")):
            for source in ("usage", "timings"):
                with self.subTest(value=bad, source=source):
                    usage = {"prompt_tokens": 150, "prompt_tokens_details": {"cached_tokens": 100}}
                    timings = {"cache_n": 100}
                    if source == "usage":
                        usage["prompt_tokens_details"]["cached_tokens"] = bad
                    else:
                        timings["cache_n"] = bad
                    rec = ci.sample(dict(response, usage=usage, timings=timings), 100, 1, 4096)
                    self.assertIn("invalid cached-token count", rec["invalid"])
                    self.assertIsNone(rec["uncached_tokens_per_ttft_s"])
        rec = ci.sample(dict(response, usage={"prompt_tokens": 150}, timings={"cache_n": 100.0}), 100, 1, 4096)
        self.assertFalse(rec["invalid"])
        self.assertEqual(rec["uncached_tokens"], 50)

    def test_plan_rejects_invalid_or_short_scenarios_before_http(self):
        for args in (("--increment-turns", "0"), ("--increment-prefixes", "-1"),
                     ("--increment-prefixes", "256,,512"), ("--increment-repeats", "0"),
                     ("--increment-tokens", "0"), ("--max-ctx", "600"), ("--turn-gap", "-1"),
                     ("--always-reasons",), ("--extra-body", '{"messages": []}'),
                     ("--extra-body", '[]'), ("--extra-body", 'null'), ("--extra-body", '{')):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                bench.check_sizes(bench.parse_args(self.argv(*args)))
        with patch.object(bench, "load_corpus", return_value="x" * 100):
            with self.assertRaises(SystemExit):
                bench.check_sizes(bench.parse_args(self.argv()))
        self.assertEqual(Fake.records, [])

    def test_engine_runner_rejects_oversized_plan_before_start(self):
        from types import SimpleNamespace
        args = SimpleNamespace(sizes="4000,16000", skip="decode,greedy,prefill,tools,agent", agentic=False,
                               agent_trace="none", bench_args="--cached-increment --max-ctx 600")
        config = {"label": "smoke-increment", "engine": "scripted", "version": "test", "cmd": None,
                  "base_url": self.base, "model": "scripted"}
        p, _ = compare_engines.plan(config, args, "test")
        self.assertTrue(any("above --max-ctx" in e for e in p["errors"]))
        with patch.object(compare_engines.gpu_hogs, "GpuLock", side_effect=AssertionError("lock before validation")):
            compare_engines.run_config(config, args, "test")
        self.assertEqual(Fake.records, [])

    def test_cli_results_and_gpu_lock(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result.json"
            env = dict(os.environ, BENCH_GPU_LOCK=str(Path(td) / "gpu.lock"))
            for key in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD"):
                env.pop(key, None)
            args = self.argv("--increment-prefixes", "256", "--increment-repeats", "1", "--out", str(out))
            cp = subprocess.run([sys.executable, str(ROOT / "bench/harness/bench.py"), *args],
                                env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(cp.returncode, 0, cp.stderr + cp.stdout)
            r = json.loads(out.read_text())
            self.assertIn("increment", r["phases"])
            self.assertIn("increment", r["memory"])
            self.assertEqual(r["gpu_lock"]["held_by"], "bench")
            self.assertEqual(r["gpu_lock"]["path"], env["BENCH_GPU_LOCK"])
            self.assertFalse(r["invalid"])
            self.assertTrue(r["corpus"]["frozen"])
            corpus = Path(td) / "corpus.txt"
            corpus.write_text("unique supplied corpus\n" * 200)
            Fake.records.clear()
            cp = subprocess.run([sys.executable, str(ROOT / "bench/harness/bench.py"), *args,
                                 "--corpus", str(corpus)], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(cp.returncode, 0, cp.stderr + cp.stdout)
            r = json.loads(out.read_text())
            self.assertFalse(r["corpus"]["frozen"])
            for wire, rec in zip(Fake.records, r["increment"]["series"][0]["requests"]):
                off, size = rec["corpus_offset"], rec["corpus_chars"]
                self.assertEqual(wire["body"]["messages"][-1]["content"], ci.ACK + corpus.read_text()[off:off + size])
            with open(env["BENCH_GPU_LOCK"], "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # previous run released it
                bad = self.argv("--max-ctx", "600", "--out", str(out))
                cp = subprocess.run([sys.executable, str(ROOT / "bench/harness/bench.py"), *bad],
                                    env=env, capture_output=True, text=True, timeout=5)
                self.assertNotEqual(cp.returncode, 0)
                self.assertIn("above --max-ctx", cp.stderr)


if __name__ == "__main__":
    unittest.main()
