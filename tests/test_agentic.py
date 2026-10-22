#!/usr/bin/env python3
"""bench/harness/agentic.py (the --agentic metrics of bench.py) against a fake OpenAI-compatible server that has a
per-session prefix cache, plus its pure summary functions, the pmset parser and the compare_table row. No model, no
GPU, stdlib only.

usage: test_agentic.py
"""
import http.server, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "..", "bench", "harness")
sys.path.insert(0, HARNESS)
os.environ["BENCH_GPU_LOCK"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), "test-agentic-gpu.lock")
import agentic, bench, compare_engines, compare_table  # noqa: E402

FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


class Fake(http.server.BaseHTTPRequestHandler):
    """Counts prompt tokens as chars/4. Keeps the last system message per session id (cache_sessions = 1 keeps only
    the most recent conversation, like an engine with one slot)."""
    last = {}
    cache_sessions = 8
    log = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        sysm = body["messages"][0]["content"]
        sid = self.headers.get("X-Session-Id", "")
        pt = sum(len(m["content"]) for m in body["messages"]) // 4
        cached = len(sysm) // 4 if Fake.last.get(sid) == sysm else 0
        Fake.last[sid] = sysm
        while len(Fake.last) > Fake.cache_sessions:
            Fake.last.pop(next(iter(Fake.last)))
        Fake.log.append((sid, pt, cached, body.get("max_tokens")))
        n = min(body.get("max_tokens", 16), 64)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.002 if cached else 0.02)  # a cache hit prefills faster
        for i in range(n):
            ev = {"choices": [{"delta": {"content": f"w{i} "}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            time.sleep(0.001)
        fin = {"choices": [{"delta": {}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": pt, "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": cached}}}
        self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}/v1"

a = bench.parse_args(["--base-url", BASE, "--model", "fake", "--label", "t", "--decode-tokens", "64", "--turn-gap", "0",
                      "--sweep-sizes", "1000,2000,max", "--sweep-repeats", "1", "--max-ctx", "8192",
                      "--cache-tokens", "2000", "--sustain-s", "0.3", "--agentic"])

# ---- pure helpers
check(agentic.sweep_targets("1000,max", 32768) == [1000, 30896] or agentic.sweep_targets("1000,max", 32768)[0] == 1000,
      "sweep targets start at 1000")
t = agentic.sweep_targets("1000,2000,max,99999", 8192)
check(t == sorted(set(t)) and t[-1] < 8192 and len(t) == 3, "sweep targets dedupe and cap at the window", str(t))
check(agentic.hit_rate([{"follow_up": True, "prompt_tokens": 100, "cached_tokens": 90},
                        {"follow_up": True, "prompt_tokens": 100, "cached_tokens": 50},
                        {"follow_up": False, "prompt_tokens": 100, "cached_tokens": 0}]) == 0.7, "hit rate over follow-ups only")
check(agentic.hit_rate([{"follow_up": True, "prompt_tokens": 100, "cached_tokens": None}]) is None, "unreported is None, not 0")
check(agentic.hit_rate([{"follow_up": True, "prompt_tokens": 100, "cached_tokens": 0}]) == 0.0, "reported zero is 0")
check(agentic.hit_rate([]) is None, "no turns is None")
s = agentic.summarize_sustain([{"tps": x, "t_s": i * 10, "speed_limit": 100} for i, x in enumerate([100, 100, 100, 90, 80, 70])])
check(s["thermal_falloff_pct"] == 25.0 and s["first_third_tps"] == 100 and s["last_third_tps"] == 75, "falloff first vs last third", str(s))
check("thermal_falloff_pct" not in agentic.summarize_sustain([{"tps": 100, "t_s": 1}, {"tps": 50, "t_s": 2}]), "no falloff from 2 runs")
check(agentic.summarize_sustain([])["valid_runs"] == 0, "empty sustain")
sw = agentic.summarize_sweep([{"target": 1000, "decode_tps": 100.0, "prompt_tokens": 1100},
                              {"target": 8000, "decode_tps": 80.0, "prompt_tokens": 8300},
                              {"target": 30000, "error": "boom"}])
check(sw["decode_retention"] == 0.8 and sw["max_prompt_tokens"] == 8300, "retention skips failed points", str(sw))
c = agentic.summarize_cache({"cold_ttft_s": 2.0, "warm_ttft_s": 0.4, "interleave_ttft_s": 1.9})
check(c["warm_vs_cold_speedup"] == 5.0 and c["interleave_prefix_kept"] is False, "interleave evicted", str(c))
check(agentic.summarize_cache({"cold_ttft_s": 2.0, "interleave_ttft_s": 0.3})["interleave_prefix_kept"] is True, "interleave kept")
th = agentic.parse_thermal("Note: No thermal warning level has been recorded\nCPU_Scheduler_Limit = 100\nCPU_Speed_Limit = 87\n",
                           "Now drawing from 'Battery Power'\n", " lowpowermode 0\n powermode 2\n")
check(th == {"speed_limit": 87, "scheduler_limit": 100, "battery": True, "low_power": False}, "pmset parse", str(th))
check(agentic.parse_thermal("Note: No CPU power status has been recorded\n")["speed_limit"] is None, "pmset without limits")

# ---- phases against the fake server
Fake.log.clear()
cs = agentic.cold_start(a, repeats=2, target=1000)
check(cs["first"].get("ttft_s") and len(cs["repeats"]) == 2 and "first_request_penalty_s" in cs, "cold start record", str(cs)[:200])
check(not cs["invalid"], "cold start valid")
check(len({x[0] for x in Fake.log}) == 3 and all(x[2] == 0 for x in Fake.log), "cold-start requests are separate cold sessions")

Fake.log.clear()
sw = agentic.t_sweep(a, log=lambda m: None)
check([p["target"] for p in sw["points"]] == agentic.sweep_targets(a.sweep_sizes, a.max_ctx), "sweep covers the targets")
check(all(p.get("decode_tps") for p in sw["points"]) and not sw["invalid"], "sweep decode rates and valid", str(sw["invalid"]))
check(all(x[2] == 0 for x in Fake.log), "sweep requests share no cache")
tok = [p["prompt_tokens"] for p in sw["points"]]
check(tok == sorted(tok) and tok[-1] <= a.max_ctx, "sweep prompt tokens grow and fit", str(tok))

Fake.log.clear()
Fake.cache_sessions = 8
ca = agentic.t_cache(a, log=lambda m: None)
check(ca["warm_cached_tokens"] and ca["warm_hit_rate"] and ca["warm_hit_rate"] > 0.9, "warm prefix is hit", str(ca))
check(ca.get("followup_hit_rate") and ca["followup_hit_rate"] > 0.9 and ca["followup_ttft_s"], "follow-up turn is hit", str(ca))
check(ca["interleave_prefix_kept"] is True, "multi-slot engine keeps the prefix across another chat", str(ca))
Fake.last.clear()
Fake.cache_sessions = 1
ca1 = agentic.t_cache(a, log=lambda m: None)
check(ca1["interleave_prefix_kept"] is False and ca1["interleave_cached_tokens"] == 0, "one-slot engine loses it", str(ca1))
check(not ca1["invalid"], "cache valid on a clean server")
Fake.cache_sessions = 8

# ---- the slot count is recorded, labelled and flagged: --slots 1 evicts A, four keep it
a.server_slots = 1
Fake.last.clear()
Fake.cache_sessions = 1
ca1s = agentic.t_cache(a, log=lambda m: None)
check(ca1s["server_slots"] == 1, "cache records the server's slot count", str(ca1s.get("server_slots")))
check(any(t["label"] == "A after B (slots 1)" for t in ca1s["turns"]), "the A-after-B turn is labelled with the slots",
      str([t["label"] for t in ca1s["turns"]]))
check(bool(ca1s.get("interleave_note")) and any("--slots 1" in f for f in ca1s["flags"]),
      "one slot is flagged as the configuration, not as a lost prefix", str(ca1s["flags"]))
a.server_slots = 4
Fake.last.clear()
Fake.cache_sessions = 4
ca4 = agentic.t_cache(a, log=lambda m: None)
check(ca4["server_slots"] == 4 and ca4["interleave_prefix_kept"] is True
      and not ca4.get("interleave_note") and not [f for f in ca4["flags"] if f.startswith("cache:")],
      "four slots keep A and flag nothing", str(ca4["flags"]))
a.server_slots = 0
Fake.cache_sessions = 1
ca0 = agentic.t_cache(a, log=lambda m: None)
check(ca0["server_slots"] == 0 and not [f for f in ca0["flags"] if f.startswith("cache:")],
      "an unknown slot count is recorded as 0 and not flagged", str(ca0["flags"]))
Fake.cache_sessions = 8

su = agentic.t_sustain(a, log=lambda m: None)
check(su["valid_runs"] >= 2 and su["duration_s"] >= 0 and "thermal_start" in su, "sustain runs until the time is up", str(su)[:200])

# ---- failure is recorded, not raised
b = bench.parse_args(["--base-url", "http://127.0.0.1:1/v1", "--model", "x", "--label", "t", "--sweep-sizes", "1000",
                      "--sweep-repeats", "1", "--agentic"])
sw = agentic.t_sweep(b, log=lambda m: None)
check(sw["invalid"] and sw["points"][0].get("error"), "unreachable server is INVALID, not a crash")
check(agentic.cold_start(b, repeats=1)["invalid"], "unreachable cold start is INVALID")

# ---- compare_table row
d = {"label": "q35-4b-x", "schema": 2, "when": "2026-10-21 10:00:00", "display": "X", "engine": "x", "version": "1",
     "decode_summary": {"median_tps": 90}, "prefill": [], "invalid": [], "flags": [], "idle_footprint_mb": 3000,
     "peak_memory_mb": 9000, "cold_start": {"spawn_to_first_token_s": 6.5, "first_request_penalty_s": 0.4},
     "sweep": {"points": [{"target": 1000, "ttft_s": 0.5, "decode_tps": 90.0, "prompt_tokens": 1100},
                          {"target": 16000, "ttft_s": 6.0, "decode_tps": 70.0, "prompt_tokens": 16300},
                          {"target": 30000, "ttft_s": 14.0, "decode_tps": 60.0, "prompt_tokens": 30900}],
               "decode_retention": 0.667},
     "cache": {"warm_ttft_s": 0.2, "interleave_prefix_kept": False, "interleave_vs_cold": 0.9},
     "sustain": {"thermal_falloff_pct": 3.2, "duration_s": 300}, "agent": {"cache_hit_rate": 0.82}}
base = compare_table.extract(d, "x.json")
row = compare_table.extract_agentic(d, "x.json", base)
check(row and row["ttft_16k"] == 6.0 and row["dec_max"] == 60.0 and row["max_tokens"] == 30900 and row["hit_rate"] == 0.82,
      "compare_table row", str(row))
md = compare_table.agentic_table("Qwen3.5-4B", [row])
check("agent-harness metrics" in md and "82%" in md and "no (0.90x cold)" in md, "agentic table renders", md[-400:])
check(compare_table.extract_agentic({"label": "q", "schema": 2}, "y.json", base) is None, "results without --agentic add no row")
for slots, want, unwanted in ((1, "slots 1", "engine default"), (4, "slots 4", "engine default"), (0, "engine default", "slots 1")):
    ds = json.loads(json.dumps(d))
    ds["cache"]["server_slots"] = slots
    mds = compare_table.agentic_table("Qwen3.5-4B", [compare_table.extract_agentic(ds, "x.json", base)])
    check(want in mds and unwanted not in mds, f"agentic table shows slots {slots}", mds[-500:])

# ---- which live conversation count an engine command asks for
check(compare_engines.live_slots(["kapitsa", "serve", "--slots", "4"]) == 4, "live_slots reads --slots N")
check(compare_engines.live_slots(["serve", "--slots=2"]) == 2, "live_slots reads --slots=N")
check(compare_engines.live_slots(["llama-server", "-np", "3"]) == 3, "live_slots reads -np N")
check(compare_engines.live_slots(["s", "--parallel", "8"]) == 8, "live_slots reads --parallel N")
check(compare_engines.live_slots(["s", "--max-concurrency=6"]) == 6, "live_slots reads --max-concurrency=N")
check(compare_engines.live_slots(["s", "--slots"]) == 0 and compare_engines.live_slots(["s", "--slots", "auto"]) == 0
      and compare_engines.live_slots([]) == 0 and compare_engines.live_slots(None) == 0,
      "live_slots is 0 when the command leaves it to the engine")

# ---- the corpus: long enough for a 128K prompt, its first 144,500 characters the corpus before the extension
c = bench.load_corpus()
check(len(c) >= 128000 * bench.CHARS_PER_TOKEN, "the corpus holds a 128K-token prompt", str(len(c)))
check(bench.corpus_record()["frozen"], "the extended corpus is the frozen one")
import hashlib  # noqa: E402
check(hashlib.sha256(c[:144500].encode()).hexdigest() == bench.CORPUS_SHA256_V1,
      "its first 144,500 characters are byte-identical to the corpus before the extension")
check(len(bench.build_context(128000)) == 544000 and bench.corpus_short(128000) is None, "a 128K context is cut whole")
why = bench.corpus_short(200000)
check(why is not None and "200,000-token" in why and f"{len(c):,}" in why, "a size past the corpus says why", why)
try:
    bench.build_context(200000)
    check(False, "build_context refuses a size past the corpus")
except ValueError as e:
    check("needs" in str(e), "build_context refuses a size past the corpus", str(e))
for argv, ok, name in [(["--sizes", "4000,131072"], True, "a 128K prefill size passes the size check"),
                       (["--sizes", "4000,200000"], False, "a prefill size past the corpus fails before any request"),
                       (["--sizes", "200000", "--skip", "prefill"], True, "a skipped phase's sizes are not checked"),
                       (["--agentic", "--max-ctx", "131072", "--sweep-sizes", "64000,max"], True,
                        "an agentic sweep to 128K passes the size check"),
                       (["--agentic", "--max-ctx", "262144", "--sweep-sizes", "max"], False,
                        "an agentic sweep 'max' past the corpus fails before any request")]:
    try:
        bench.check_sizes(bench.parse_args(["--base-url", "http://x", "--model", "m", "--label", "t"] + argv))
        check(ok, name)
    except SystemExit as e:
        check(not ok and "corpus" in str(e), name, str(e))

srv.shutdown()
print("FAILED: " + ", ".join(FAILED) if FAILED else "all passed")
sys.exit(1 if FAILED else 0)
