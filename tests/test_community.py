#!/usr/bin/env python3
"""The community results' checks and table (tools/community_check.py, tools/community_table.py), offline.

usage: test_community.py [BUILD_DIR]   (default: build; needs BUILD_DIR/bench_test for the sample results, and
                                        jsonschema: pip install jsonschema)

Sample results come from `bench_test result` (a made-up run, as `kapitsa bench` writes it). The checker must accept
them and refuse each kind of implausible one; the labels must follow; the table must group by Mac, then model and
precision, with units and the contention marks; and the pull request check must work end to end against a stand-in
GitHub API: the files read as data at the head commit, the labels set, the report in the job summary.
"""
import copy, http.server, json, os, shutil, socketserver, subprocess, sys, tempfile, threading, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
BUILD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "build"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import community_check as cc  # noqa: E402
import community_table as ct  # noqa: E402

FAILS = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


def sample(quick=False):
    out = subprocess.run([os.path.join(BUILD, "bench_test"), "result"] + (["quick"] if quick else []), capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out)


import jsonschema  # noqa: E402

schema = json.load(open(os.path.join(REPO, "bench", "community", "schema.json")))
cat, macs, versions = cc.catalog(), cc.mac_models(), cc.known_versions(use_git=False)

# ---------------------------------------------------------------- what the checker knows
check(list(cat) == ["qwen3.5-4b", "qwen3.5-9b", "qwen3.8-27b", "qwen3.6-35b-a3b", "qwen3.8-flash-next", "nemotron-labs-dory1.0-4b",
                   "nemotron-3-nano-4b", "nemotron-nano-9b-v2", "nemotron-3.5-lightning", "nemotron-3.5-bolt", "nemotron-3.5-super-vl", "flint"], "the catalog's models, in order",
      str(list(cat)))
check(cat["qwen3.6-35b-a3b"]["variants"] == {"4bit": "4-bit", "5bit": "5-bit", "6bit": "6-bit", "8bit": "8-bit", "bf16": "bf16",
                                                   "16gb": "2-bit experts"},
      "the 35B's precisions", str(cat["qwen3.6-35b-a3b"]["variants"]))
check(cat["nemotron-3.5-bolt"]["variants"] == {"24gb": "NVIDIA 24 GB"} and cat["qwen3.5-4b"]["title"] == "Qwen3.5-4B",
      "Bolt's precision and the 4B's title")
check(macs["Mac16,7"]["name"] == "MacBook Pro (16-inch, 2024)" and len(macs) >= 58, "the Mac table")
check(cat["qwen3.8-flash-next"]["variants"] == {"4bit": "4-bit"}, "a variant whose .quant string spans several lines is still read")
check(versions, "a known version from CMakeLists.txt", str(versions))

# ---------------------------------------------------------------- accepted, refused, warned
full, quick = sample(), sample(True)
for name, r in (("full", full), ("quick", quick)):
    errs = list(jsonschema.Draft202012Validator(schema).iter_errors(r))
    e, w = cc.check_result(r, cat, macs, versions)
    check(not errs and not e, f"a {name} result passes", "; ".join([x.message for x in errs] + e))
    check(any("CONTENTION" in x for x in w), f"its contention flag is a warning, not an error ({name})")


def refused(change, why, text, schema_only=False, existing=()):
    r = copy.deepcopy(full)
    change(r)
    errs = [x.message for x in jsonschema.Draft202012Validator(schema).iter_errors(r)]
    e = [] if errs else cc.check_result(r, cat, macs, versions, existing=existing)[0]
    hit = errs if schema_only else e
    check(bool(hit) and (schema_only or any(text in x for x in e)), f"refused: {why}", str(errs + e)[:300])


other = copy.deepcopy(full)
refused(lambda r: r["kapitsa"].update(version="0.0.9"), "a version never released", "not a released version")
refused(lambda r: r["model"].update(name="llama-3-8b", catalog=False), "a model outside the catalog", "not in the catalog")
refused(lambda r: r["model"].update(variant="3bit"), "a precision the model has not", "has no variant")
refused(lambda r: r["model"].update(precision="6-bit"), "a precision that is not its variant's", "precision")
refused(lambda r: r["model"].update(title="Qwen3.6-35B"), "a wrong title", "title")
lightning = copy.deepcopy(full)  # what kapitsa bench writes for Lightning: DSpark decoding, with its drafter
lightning["model"].update(name="nemotron-3.5-lightning", title=cat["nemotron-3.5-lightning"]["title"], variant="nvfp4",
                          precision=cat["nemotron-3.5-lightning"]["variants"]["nvfp4"], decoding="DSpark",
                          drafter="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark@" + "0" * 40)
errs = [x.message for x in jsonschema.Draft202012Validator(schema).iter_errors(lightning)]
e = cc.check_result(lightning, cat, macs, versions)[0]
check(not errs and not any("decoding" in x or "drafter" in x for x in e), "DSpark decoding (Lightning) passes", str(errs + e)[:300])
refused(lambda r: r["model"].update(decoding="DSpark", drafter=None), "DSpark decoding without its drafter", "drafter")
dory = copy.deepcopy(full)
dory["model"].update(name="nemotron-labs-dory1.0-4b", title="Nemotron-Labs Dory1.0 4B", variant="nvfp4",
                     precision=cat["nemotron-labs-dory1.0-4b"]["variants"]["nvfp4"])
e = cc.check_result(dory, cat, macs, versions)[0]
check(not any("title" in x for x in e), "a title from an earlier release passes (Dory's, until 0.1.35)", str(e))
check(ct.model_name(dory["model"]) == "Nemotron Dory1.0 4B", "the table names a model as the catalog does now",
      ct.model_name(dory["model"]))


def prompts(r, sizes):
    for p, n in zip(r["results"]["prefill"], sizes):
        p.update(prompt_tokens=n, tps=round(n / p["ttft_s"], 1))


flint = copy.deepcopy(full)  # what kapitsa bench writes for Flint: its prompts as Flint's own tokenizer counts them, T = 1.0
flint["model"].update(name="flint", title=cat["flint"]["title"], variant="4bit", precision=cat["flint"]["variants"]["4bit"])
flint["protocol"]["temperature"] = 1.0
prompts(flint, (3873, 15065))
e = cc.check_result(flint, cat, macs, versions)[0]
check(not e, "Flint's prompts pass against its own reference (3,874 and 15,066 tokens)", str(e))
r = copy.deepcopy(flint)
prompts(r, (4133, 16286))
e = cc.check_result(r, cat, macs, versions)[0]
check(any("not within 1% of 3874" in x for x in e), "refused: Flint with the Qwen templates' prompt sizes", str(e))
r = copy.deepcopy(flint)
r["protocol"]["temperature"] = 0.6
e = cc.check_result(r, cat, macs, versions)[0]
check(any("protocol temperature is 0.6" in x for x in e), "refused: Flint writing at 0.6, not its 1.0", str(e))
del r["protocol"]["temperature"]
e = cc.check_result(r, cat, macs, versions)[0]
check(not e and not list(jsonschema.Draft202012Validator(schema).iter_errors(r)),
      "a result from before the temperature was recorded passes", str(e))
refused(lambda r: r["protocol"].update(temperature=1.0), "the 35B writing at 1.0, not its 0.6", "protocol temperature is 1.0")
refused(lambda r: r["mac"].update(name="MacBook Pro (14-inch, 2024)"), "a Mac name that is not its identifier's", "is MacBook Pro (16-inch, 2024)")
refused(lambda r: r["mac"].update(identifier="Mac99,9"), "a name for an identifier the table does not have", "not in mac-models.json")
refused(lambda r: r.update(invalid=["decode: reasoning text returned (12 chars)"]), "invalid measurements", "invalid measurements")
refused(lambda r: r["results"]["decode"].update(median_tps=170.0), "a median that is not its runs'", "is not the median")
refused(lambda r: r["results"]["decode"].update(runs_tps=[150.0, 151.0, 152.0], median_tps=151.0), "too few runs for the mode", "valid runs")
refused(lambda r: r["results"]["prefill"][0].update(prompt_tokens=4300, tps=round(4300 / r["results"]["prefill"][0]["ttft_s"], 1)),
        "a prompt 4% off the reference", "not within 1%")
refused(lambda r: r["results"]["prefill"][0].update(tps=5000.0), "a rate that is not tokens / time", "is not")
refused(lambda r: r["results"]["prefill"][1].update(followup_median_s=0.5), "follow-ups and a median that disagree", "disagree")
refused(lambda r: r["protocol"].update(decode_runs=3), "a protocol that is not its mode's", "protocol decode_runs")
refused(lambda r: r["protocol"].update(corpus_sha256="0" * 64), "prompts from another corpus", "frozen corpus")
refused(lambda r: r["results"].update(peak_memory_gb=200.0), "more memory than the Mac has", "peak memory")
refused(lambda r: r.update(time_utc="2031-01-01T00:00:00Z"), "a run in the future", "not plausible")
refused(lambda r: r["results"]["decode"].update(runs_tps=[3000.0] * 5, median_tps=3000.0), "an impossible rate", "outside 1-2000")
refused(lambda r: None, "a copy of a result already in the table", "already in the table",
        existing=[(os.path.join(REPO, "bench", "community", "results", "other.json"), other)])
refused(lambda r: r["mac"].update(serial="C02X"), "an extra field (a serial number)", "", schema_only=True)
refused(lambda r: r["mac"].update(chip="Intel Core i9"), "a chip that is not Apple's", "", schema_only=True)
refused(lambda r: r["conditions"].update(contention=["rm -rf /"]), "a phase name that is not a phase", "", schema_only=True)
r = copy.deepcopy(full)
r["mac"].update(identifier="Mac99,9", name=None, year=None)
e, w = cc.check_result(r, cat, macs, versions)
check(not e and any("not in mac-models.json" in x for x in w), "an unknown Mac (no name) passes, with a warning", str(e))
r = copy.deepcopy(full)
r["results"].update(tool_calls="fail", tool_calls_failed=["thinking on: round_trip"])
e, w = cc.check_result(r, cat, macs, versions)
check(not e and any("tool calls failed" in x for x in w), "a failed tool check passes, with a warning")
check(cc.labels_for(quick, []) == {"benchmark", "quick", "contention"}, "labels: a quick run with contention", str(cc.labels_for(quick, [])))
r = copy.deepcopy(full)
r["conditions"]["contention"], r["flags"] = [], ["power: on battery (a Mac on battery may run slower)"]
check(cc.labels_for(r, []) == {"benchmark", "flagged"} and cc.labels_for(r, ["x"]) == {"benchmark", "flagged", "invalid"},
      "labels: flagged, and invalid when a check fails")

# ---------------------------------------------------------------- the table
tmp = tempfile.mkdtemp(prefix="kapitsa-community-test-")
res_dir = os.path.join(tmp, "results")
os.makedirs(res_dir)
files = {"kapitsa-bench-2026-10-18-m4-pro-qwen3.6-35b-a3b-4bit-aaaaaa.json": full,
         "kapitsa-bench-2026-10-18-m4-pro-qwen3.6-35b-a3b-4bit-quick-bbbbbb.json": quick}
m5 = copy.deepcopy(full)
m5["mac"].update(identifier="Mac17,6", name="MacBook Pro (16-inch, M5 Pro or M5 Max)", year=2026, chip="Apple M5 Max", gpu_cores=40,
                 memory_gb=128)
m5["model"].update(name="qwen3.8-27b", title="Qwen3.8-27B", variant="6bit", precision="6-bit")
m5["conditions"]["contention"], m5["flags"], m5["time_utc"] = [], [], "2026-10-19T08:00:00Z"
files["kapitsa-bench-2026-10-19-m5-max-qwen3.8-27b-6bit-cccccc.json"] = m5
for n, r in files.items():
    json.dump(r, open(os.path.join(res_dir, n), "w"), indent=2)
open(os.path.join(res_dir, "notes.json"), "w").write("[]")  # not a result: skipped
out = os.path.join(tmp, "RESULTS.md")
p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "community_table.py"), "--results", res_dir, "--out", out],
                   capture_output=True, text=True)
md = open(out).read() if os.path.exists(out) else ""
check(p.returncode == 0 and "skipped" in p.stderr, "the table is written; a file that is not a result is skipped", p.stderr[-200:])
i_m5, i_m4 = md.find("## MacBook Pro (16-inch, 2026) · M5 Max 40-core GPU · 128 GB"), md.find(
    "## MacBook Pro (16-inch, 2024) · M4 Pro 20-core GPU · 48 GB")
check(0 < i_m5 < i_m4, "grouped by Mac, the newest first, an M5 Mac named with its year")
for name, year, want in (("MacBook Pro (14-inch, M5)", 2025, "MacBook Pro (14-inch, 2025)"), ("Mac mini (M6)", 2026, "Mac mini (2026)"),
                         ("MacBook Air (15-inch, M4, 2025)", 2025, "MacBook Air (15-inch, M4, 2025)"),
                         ("MacBook Pro (16-inch, M5 Pro or M5 Max)", 2026, "MacBook Pro (16-inch, 2026)")):
    check(ct.mac_name({"name": name, "year": year, "identifier": "x", "chip": "Apple M5"}) == want, f"the table names {name} {want}")
check("### Qwen3.6 35B-A3B · 4-bit" in md and "### Qwen3.8 27B · 6-bit" in md, "then by model and precision")
rows = [l for l in md.splitlines() if l.startswith("| [2026-")]
check(len(rows) == 3, "a row per run", str(len(rows)))
# the measurement columns (writing, reading, follow-ups, memory): each number carries its unit
cells = [c.strip() for l in rows for c in l.strip("|").split("|")[1:8]]
check(all(c == "–" or c.endswith(("tok/s", "tok/s ⚠", " s", " s ⚠", " GB")) or " GB + " in c for c in cells),
      "every number with its unit", str([c for c in cells if not c.endswith(("tok/s", " s", " GB", "⚠", "–"))]))
m4_rows = md[i_m4:].splitlines()
check(any("1,169 tok/s ⚠" in l and "contention (prefill_4000)" in l for l in m4_rows), "a disturbed measurement is marked, and its flag listed")
check(any(") quick |" in l for l in m4_rows), "a quick run says so")
both = ct.render([(os.path.join(res_dir, "flint.json"), flint), (os.path.join(res_dir, "qwen.json"), full)], res_dir, out)
sections = {s.split("\n", 1)[0]: s for s in both.split("\n### ")[1:]}
check("| Run | Writing, T = 1.0 |" in sections.get("Recursive Flint 40B-A2.5B · 4-bit", "")
      and "| Run | Writing, T = 0.6 |" in sections.get("Qwen3.6 35B-A3B · 4-bit", ""),
      "Flint's writing is headed T = 1.0 (its model card's), every other model's T = 0.6", str(list(sections)))
p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "community_table.py"), "--results", res_dir, "--out", out, "--check"],
                   capture_output=True, text=True)
check(p.returncode == 0, "--check: up to date")
os.remove(os.path.join(res_dir, "kapitsa-bench-2026-10-19-m5-max-qwen3.8-27b-6bit-cccccc.json"))
p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "community_table.py"), "--results", res_dir, "--out", out, "--check"],
                   capture_output=True, text=True)
check(p.returncode == 1, "--check: a result removed, the table is out of date")
empty = os.path.join(tmp, "empty")
os.makedirs(empty)
p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "community_table.py"), "--results", empty, "--out",
                    os.path.join(tmp, "E.md")], capture_output=True, text=True)
check("No results yet" in open(os.path.join(tmp, "E.md")).read(), "no results: says how to add the first")

# ---------------------------------------------------------------- the pull request check, against a stand-in API
good_name = "bench/community/results/kapitsa-bench-2026-10-18-m4-pro-qwen3.6-35b-a3b-4bit-quick-bbbbbb.json"
bad = copy.deepcopy(full)
bad["model"]["precision"] = "6-bit"
bad_name = "bench/community/results/kapitsa-bench-2026-10-18-m4-pro-qwen3.6-35b-a3b-4bit-dddddd.json"
contents = {good_name: json.dumps(quick, indent=2), bad_name: json.dumps(bad, indent=2)}
calls = []


class Api(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj, status=200, raw=False):
        body = obj if raw else json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        calls.append(("GET", u.path, self.headers.get("Accept")))
        if u.path == "/repos/nace-ai/kapitsa/pulls/7":
            return self.reply({"head": {"repo": {"full_name": "someone/kapitsa"}, "sha": "feedface"}, "labels": [{"name": "flagged"}]})
        if u.path == "/repos/nace-ai/kapitsa/pulls/7/files":
            return self.reply([{"filename": good_name, "status": "added"}, {"filename": bad_name, "status": "added"},
                               {"filename": "README.md", "status": "modified"}])
        if u.path.startswith("/repos/someone/kapitsa/contents/"):
            name = urllib.parse.unquote(u.path[len("/repos/someone/kapitsa/contents/"):])
            assert urllib.parse.parse_qs(u.query)["ref"] == ["feedface"]
            return self.reply(contents[name].encode(), raw=True)
        self.reply({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        calls.append(("POST", self.path, json.loads(self.rfile.read(n))))
        self.reply({}, 201)

    def do_DELETE(self):
        calls.append(("DELETE", self.path, None))
        self.reply({})


class T(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


srv = T(("127.0.0.1", 0), Api)
threading.Thread(target=srv.serve_forever, daemon=True).start()
summary = os.path.join(tmp, "summary.md")
env = dict(os.environ, GITHUB_API_URL=f"http://127.0.0.1:{srv.server_address[1]}", GH_TOKEN="test-token",
           GITHUB_REPOSITORY="nace-ai/kapitsa", PR_NUMBER="7", GITHUB_STEP_SUMMARY=summary)
p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "community_check.py"), "--github"], env=env, capture_output=True,
                   text=True, timeout=120)
rep = open(summary).read() if os.path.exists(summary) else ""
check(p.returncode == 1, "--github: one bad file fails the check", (p.stdout + p.stderr)[-300:])
check("✅" in rep and "❌" in rep and "precision '6-bit'" in rep, "--github: the report names the passing file and why the other fails")
check("README.md" in rep, "--github: other files the pull request changes are mentioned")
gets = [c for c in calls if c[0] == "GET" and "/contents/" in c[1]]
check(len(gets) == 2 and all(c[2] == "application/vnd.github.raw" for c in gets), "--github: the result files read as data at the head commit")
set_labels = [c for c in calls if c[0] == "POST" and c[1] == "/repos/nace-ai/kapitsa/issues/7/labels"]
check(set_labels and set(set_labels[0][2]["labels"]) == {"benchmark", "invalid", "quick", "contention"},
      "--github: labels benchmark, invalid, quick, contention", str(set_labels))
check(("DELETE", "/repos/nace-ai/kapitsa/issues/7/labels/flagged", None) in calls, "--github: a stale label it manages is removed")
srv.shutdown()

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{'FAIL' if FAILS else 'ok'}: {len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
