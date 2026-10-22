#!/usr/bin/env python3
"""Checks `kapitsa bench` results for the community table (bench/community): the schema, then whether each is plausible.

  python3 tools/community_check.py FILE.json...      check result files (exit 1 if one fails)
  python3 tools/community_check.py --github          in CI (.github/workflows/community-bench.yml): the results a pull
                                                     request adds, read as data through the API (none of its code runs),
                                                     checked, the pull request labeled, a report in the job summary

A result passes when it validates against bench/community/schema.json (needs `pip install jsonschema`; --no-schema
skips it) and is plausible: a Kapitsa version that was released (a v<version> tag) or is the current one in
CMakeLists.txt; a model and precision of the catalog (src/hub/catalog.cpp) with its title; the Mac's name and year as
bench/community/mac-models.json gives them for its identifier; the protocol of its mode; no invalid measurement; rates,
times and memory within physical bounds and consistent with each other (the medians of the runs, tokens / time to the
first token, prompt sizes within 1% of the reference, follow-ups faster than cold prompts); not a copy of a result
already in the table. Warnings (an unknown Mac, flags, a failed tool check) do not fail it. Stdlib only, but jsonschema.
"""
import argparse, datetime, glob, json, os, re, statistics, subprocess, sys, urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
COMMUNITY = os.path.join(REPO, "bench", "community")
RESULTS = os.path.join(COMMUNITY, "results")
CORPUS_SHA256 = "58a7756fd94fc48a7b7306d330b2c44663bdf22faa943f82430966c2878081b2"
CORPUS_SHA256_V1 = "1a9bb103b3f6d39ee5b9f5bf7207e304c5ce92788c273685f597508c7d2bde28"  # the first 144,500 characters of it (bench/harness/corpus/NOTICE.md): the same prompts
# usage.prompt_tokens of the cold prompts by tokenizer, as kapitsa bench checks them (src/bench/protocol.cpp): bench.py
# REF_PROMPT_TOKENS, configs/bolt.json's for Nemotron, configs/flint.json's for Flint
REFERENCE = {"qwen": {4000: 4133, 16000: 16286}, "nemotron": {4000: 4183, 16000: 16514},
             "flint": {4000: 3874, 16000: 15066}}
# the writing runs' temperature where it is not 0.6: Flint's model card's, as configs/flint.json (src/server/bench_cmd.cpp)
TEMPERATURE = {"flint": 1.0}
PROTOCOL = {"full": {"decode_tokens": 512, "decode_runs": 5, "greedy_runs": 5, "sizes": [4000, 16000]},
            "quick": {"decode_tokens": 512, "decode_runs": 3, "greedy_runs": 3, "sizes": [4000]}}
FOLLOWUPS = 3
# GPU cores of each Apple silicon tier, loosely (a warning outside them): base, Pro, Max, Ultra
GPU_TIERS = {"": (7, 12), "Pro": (14, 24), "Max": (24, 48), "Ultra": (48, 96)}
LABELS = {"benchmark": ("0E8A16", "A kapitsa bench result"), "quick": ("C5DEF5", "A quick kapitsa bench run"),
          "contention": ("FBCA04", "Other apps used the GPU during a measurement"),
          "flagged": ("FEF2C0", "A condition that may skew the numbers"), "invalid": ("D93F0B", "Fails the result checks")}


# ---------------------------------------------------------------- what the repository knows

# Titles a catalog model had in earlier releases, which the results those releases wrote carry
FORMER_TITLES = {"nemotron-labs-dory1.0-4b": ("Nemotron-Labs Dory1.0 4B",)}  # until 0.1.35


def catalog(repo=REPO):
    """{name: {"title", "variants": {variant: precision}}} from src/hub/catalog.cpp"""
    src = open(os.path.join(repo, "src", "hub", "catalog.cpp"), encoding="utf-8").read()
    body = src[src.index("const std::vector<CatalogModel>& catalog()"):]
    starts = [m for m in re.finditer(r'\{\.name = "([^"]+)",\s*\.aliases = \{[^}]*\},\s*\.title = "([^"]+)"', body)]
    out = {}
    for i, m in enumerate(starts):
        chunk = body[m.end():starts[i + 1].start() if i + 1 < len(starts) else len(body)]
        variants = {}
        for v in re.finditer(r'\{\.name = "([a-z0-9]+)",\s*\.quant = (?:"[^"]*"\s*)+,(?:\s*\.label = "([^"]*)",)?', chunk):
            name, label = v.group(1), v.group(2)
            variants[name] = label.split(" (")[0] if label else re.sub(r"bit$", "-bit", name)
        out[m.group(1)] = {"title": m.group(2), "variants": variants}
    return out


def mac_models(repo=REPO):
    with open(os.path.join(repo, "bench", "community", "mac-models.json"), encoding="utf-8") as fh:
        return {m["identifier"]: m for m in json.load(fh)["models"]}


def known_versions(repo=REPO, use_git=True):
    """The released versions (v* tags) and the one in CMakeLists.txt"""
    out = set()
    m = re.search(r'set\(KAPITSA_VERSION "([^"]+)"\)', open(os.path.join(repo, "CMakeLists.txt")).read())
    if m:
        out.add(m.group(1))
    if use_git:
        try:
            tags = subprocess.run(["git", "-C", repo, "tag", "-l", "v*"], capture_output=True, text=True, timeout=30).stdout
            out |= {t[1:] for t in tags.split()}
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- the checks

def schema_errors(r, schema):
    import jsonschema
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '(top)'}: {e.message}" for e in v.iter_errors(r)]


def median_ok(d):
    runs = d.get("runs_tps") or []
    return bool(runs) and d.get("median_tps") is not None and abs(statistics.median(runs) - d["median_tps"]) <= 0.011


def check_result(r, cat, macs, versions, existing=(), now=None):
    """(errors, warnings) of one result, a dict that already validates against the schema"""
    errs, warns = [], []
    now = now or datetime.datetime.now(datetime.timezone.utc)
    # who made it
    if r["kapitsa"]["version"] not in versions:
        errs.append(f"Kapitsa {r['kapitsa']['version']} is not a released version ({', '.join(sorted(versions)) or 'none known'})")
    t = datetime.datetime.strptime(r["time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    if t > now + datetime.timedelta(days=1) or t < datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc):
        errs.append(f"run time {r['time_utc']} is not plausible")
    # the model
    m = r["model"]
    if not m["catalog"] or m["name"] not in cat:
        errs.append(f"model {m['name']} is not in the catalog (src/hub/catalog.cpp)")
    else:
        c = cat[m["name"]]
        if m["title"] != c["title"] and m["title"] not in FORMER_TITLES.get(m["name"], ()):
            errs.append(f"model title {m['title']!r} is not the catalog's {c['title']!r}")
        if m["variant"] not in c["variants"]:
            errs.append(f"{m['name']} has no variant {m['variant']!r} ({', '.join(c['variants'])})")
        elif m["precision"] != c["variants"][m["variant"]]:
            errs.append(f"precision {m['precision']!r} is not {m['name']}:{m['variant']}'s ({c['variants'][m['variant']]!r})")
    if m["decoding"] in ("DFlash2", "LiLiCorr", "DSpark") and not m["drafter"]:
        errs.append(f"{m['decoding']} decoding without its drafter")
    # the Mac
    mac = r["mac"]
    known = macs.get(mac["identifier"])
    if known:
        if mac["name"] != known["name"] or mac["year"] != known["year"]:
            errs.append(f"{mac['identifier']} is {known['name']} ({known['year']}), not {mac['name']} ({mac['year']})")
    else:
        if mac["name"] is not None or mac["year"] is not None:
            errs.append(f"{mac['identifier']} is not in mac-models.json, so its name can't be {mac['name']!r}")
        warns.append(f"{mac['identifier']} is not in mac-models.json (a new Mac? tools/mac_models.py updates it)")
    chip = re.fullmatch(r"Apple M(\d+)(?: (Pro|Max|Ultra))?", mac["chip"])
    if not chip:
        warns.append(f"chip {mac['chip']!r} is not an M-series name")
    elif mac["gpu_cores"] is not None:
        lo, hi = GPU_TIERS[chip.group(2) or ""]
        if not lo <= mac["gpu_cores"] <= hi:
            warns.append(f"{mac['gpu_cores']} GPU cores is unusual for an {mac['chip']}")
    if mac["memory_gb"] not in (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512) and not 8 <= mac["memory_gb"] <= 2048:
        errs.append(f"{mac['memory_gb']} GB of memory is not plausible")
    # the protocol
    want = PROTOCOL[r["mode"]]
    if r["protocol"]["corpus_sha256"] not in (CORPUS_SHA256, CORPUS_SHA256_V1):
        errs.append("the prompts are not from the frozen corpus")
    for k, v in want.items():
        if r["protocol"][k] != v:
            errs.append(f"protocol {k} is {r['protocol'][k]}, a {r['mode']} run's is {v}")
    t, want_t = r["protocol"].get("temperature"), TEMPERATURE.get(m["name"], 0.6)  # older results leave it out
    if t is not None and abs(t - want_t) > 1e-9:
        errs.append(f"protocol temperature is {t}, {m['name']}'s writing runs are at {want_t}")
    if r["invalid"]:
        errs.append("invalid measurements: " + "; ".join(r["invalid"][:3]))
    # the numbers
    res = r["results"]
    for k, runs in (("decode", want["decode_runs"]), ("decode_greedy", want["greedy_runs"])):
        d = res.get(k)
        if not d or d.get("median_tps") is None:
            errs.append(f"{k}: no median")
            continue
        if len(d["runs_tps"]) != runs:
            errs.append(f"{k}: {len(d['runs_tps'])} valid runs, the protocol makes {runs}")
        if not median_ok(d):
            errs.append(f"{k}: median {d['median_tps']} is not the median of its runs {d['runs_tps']}")
        if not all(1 <= x <= 2000 for x in d["runs_tps"]):
            errs.append(f"{k}: rates {d['runs_tps']} outside 1-2000 tok/s")
    family = "flint" if m["name"] == "flint" else "nemotron" if m["name"].startswith("nemotron") else "qwen"
    sizes = [p["size"] for p in res["prefill"]]
    if sizes != want["sizes"]:
        errs.append(f"prefill sizes {sizes}, the protocol's are {want['sizes']}")
    for p in res["prefill"]:
        tag = f"prefill_{p['size']}"
        if None in (p["prompt_tokens"], p["ttft_s"], p["tps"], p["followup_median_s"]):
            errs.append(f"{tag}: a measurement is missing")
            continue
        ref = REFERENCE[family].get(p["size"])
        if ref and abs(p["prompt_tokens"] - ref) > 0.01 * ref:
            errs.append(f"{tag}: {p['prompt_tokens']} prompt tokens, not within 1% of {ref}")
        if abs(p["prompt_tokens"] / p["ttft_s"] - p["tps"]) > 0.005 * p["tps"] + 0.1:
            errs.append(f"{tag}: {p['tps']} tok/s is not {p['prompt_tokens']} tokens / {p['ttft_s']} s")
        if not 10 <= p["tps"] <= 100000:
            errs.append(f"{tag}: {p['tps']} tok/s is not plausible")
        if len(p["followup_s"]) != FOLLOWUPS or abs(statistics.median(p["followup_s"]) - p["followup_median_s"]) > 0.0002:
            errs.append(f"{tag}: the follow-ups {p['followup_s']} and their median {p['followup_median_s']} disagree")
        elif not 0.001 <= p["followup_median_s"] <= 120:
            errs.append(f"{tag}: a follow-up of {p['followup_median_s']} s is not plausible")
        elif p["followup_median_s"] >= p["ttft_s"]:
            warns.append(f"{tag}: follow-ups ({p['followup_median_s']} s) no faster than the cold prompt ({p['ttft_s']} s)")
    if res["peak_memory_gb"] is None or not 0.3 <= res["peak_memory_gb"] <= 1.5 * mac["memory_gb"]:
        errs.append(f"peak memory {res['peak_memory_gb']} GB is not plausible on {mac['memory_gb']} GB")
    if res["tool_calls"] != "pass":
        warns.append("tool calls failed: " + ", ".join(res.get("tool_calls_failed", [])))
    for f in r["flags"]:
        warns.append("flag: " + f)
    # not a copy of one in the table
    key = (r["time_utc"], mac["identifier"], m["name"], m["variant"])
    for path, other in existing:
        if (other.get("time_utc"), other.get("mac", {}).get("identifier"), other.get("model", {}).get("name"),
                other.get("model", {}).get("variant")) == key:
            errs.append(f"the same run is already in the table: {os.path.relpath(path, REPO)}")
    return errs, warns


def labels_for(r, errs):
    out = {"benchmark"}
    if errs:
        out.add("invalid")
    if r.get("mode") == "quick":
        out.add("quick")
    if r.get("conditions", {}).get("contention"):
        out.add("contention")
    if [f for f in r.get("flags", []) if not f.startswith("CONTENTION")]:
        out.add("flagged")
    return out


def existing_results(skip=()):
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        if os.path.basename(p) in skip:
            continue
        try:
            out.append((p, json.load(open(p, encoding="utf-8"))))
        except Exception:
            pass
    return out


def check_texts(items, schema, use_git=True, named=False):
    """[(name, text)] -> ([(name, result or None, errors, warnings)], all passed). named: files for
    bench/community/results/, whose names follow the rule."""
    cat, macs, versions = catalog(), mac_models(), known_versions(use_git=use_git)
    existing = existing_results(skip={os.path.basename(n) for n, _ in items})
    report, ok = [], True
    for name, text in items:
        try:
            r = json.loads(text)
        except Exception as e:
            report.append((name, None, [f"not JSON: {e}"], []))
            ok = False
            continue
        errs = schema_errors(r, schema) if schema else []
        warns = []
        if not errs:
            e2, warns = check_result(r, cat, macs, versions, existing)
            errs += e2
        if named and (not os.path.basename(name).startswith("kapitsa-bench-") or not name.endswith(".json")):
            errs.append("result files are named kapitsa-bench-<date>-<chip>-<model>-....json")
        ok = ok and not errs
        report.append((name, r, errs, warns))
    return report, ok


def markdown(report):
    lines = []
    for name, r, errs, warns in report:
        what = ""
        if r and isinstance(r, dict) and "mac" in r:
            try:
                what = f" — {r['mac'].get('name') or r['mac']['identifier']}, {r['mac']['chip']}, {r['mac']['memory_gb']} GB: " \
                       f"{r['model']['title']} {r['model']['precision']}, {r['mode']} run"
            except Exception:
                pass
        lines.append(f"### {'✅' if not errs else '❌'} `{os.path.basename(name)}`{what}\n")
        lines += [f"- ❌ {e}" for e in errs] + [f"- ⚠️ {w}" for w in warns]
        if not errs and not warns:
            lines.append("- passes every check")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- GitHub

def api(path, token, method="GET", data=None, accept="application/vnd.github+json"):
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")  # Actions sets it; tests point it elsewhere
    req = urllib.request.Request(f"{base}/{path}", method=method,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Authorization": f"Bearer {token}", "Accept": accept,
                                          "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "kapitsa-community-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    return json.loads(body) if accept.endswith("json") and body else body


def github(schema):
    """The pull request's result files (read through the API, never checked out), checked; the pull request labeled."""
    token, repo = os.environ["GH_TOKEN"], os.environ["GITHUB_REPOSITORY"]
    number = int(os.environ["PR_NUMBER"])
    pr = api(f"repos/{repo}/pulls/{number}", token)
    head_repo, head_sha = pr["head"]["repo"]["full_name"], pr["head"]["sha"]
    files, page = [], 1
    while True:
        batch = api(f"repos/{repo}/pulls/{number}/files?per_page=100&page={page}", token)
        files += batch
        if len(batch) < 100:
            break
        page += 1
    items, other = [], []
    for f in files:
        name = f["filename"]
        if not name.startswith("bench/community/results/"):
            other.append(name)
            continue
        if f["status"] == "removed":
            continue
        text = api(f"repos/{head_repo}/contents/{urllib.parse.quote(name)}?ref={head_sha}", token,
                   accept="application/vnd.github.raw").decode("utf-8")
        items.append((name, text))
    report, ok = check_texts(items, schema, named=True)
    labels = set()
    for name, r, errs, _ in report:
        labels |= labels_for(r if isinstance(r, dict) else {}, errs)
    if not items:
        labels = set()
    summary = "## Community benchmark results\n\n" + (markdown(report) if items else "No result files in this pull request.\n")
    if other and items:
        summary += "\nThis pull request also changes other files: " + ", ".join(f"`{o}`" for o in other[:10]) + \
                   " (result submissions change only bench/community/results/).\n"
    with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a", encoding="utf-8") as fh:
        fh.write(summary)
    print(summary)
    # labels: create the ones missing, then set this pull request's (only the ones this check manages)
    for name in labels:
        color, desc = LABELS[name]
        try:
            api(f"repos/{repo}/labels", token, "POST", {"name": name, "color": color, "description": desc})
        except urllib.error.HTTPError as e:
            if e.code != 422:  # 422: it exists
                print(f"can't create label {name}: {e}")
    current = {l["name"] for l in pr.get("labels", [])}
    for name in (current & set(LABELS)) - labels:
        try:
            api(f"repos/{repo}/issues/{number}/labels/{urllib.parse.quote(name)}", token, "DELETE")
        except urllib.error.HTTPError:
            pass
    if labels - current:
        api(f"repos/{repo}/issues/{number}/labels", token, "POST", {"labels": sorted(labels - current)})
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="result files to check")
    ap.add_argument("--github", action="store_true", help="check a pull request's results in CI (env GH_TOKEN, GITHUB_REPOSITORY, PR_NUMBER)")
    ap.add_argument("--no-schema", action="store_true", help="skip the schema (when jsonschema is not installed)")
    ap.add_argument("--no-git", action="store_true", help="known versions from CMakeLists.txt only, not the git tags")
    a = ap.parse_args()
    schema = None
    if not a.no_schema:
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            raise SystemExit("the schema check needs jsonschema: pip install jsonschema (or --no-schema)")
        schema = json.load(open(os.path.join(COMMUNITY, "schema.json"), encoding="utf-8"))
    if a.github:
        return github(schema)
    if not a.files:
        ap.error("name result files, or --github")
    report, ok = check_texts([(f, open(f, encoding="utf-8").read()) for f in a.files], schema, not a.no_git)
    print(markdown(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
