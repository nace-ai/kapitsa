#!/usr/bin/env python3
"""Writes bench/community/RESULTS.md, the community's `kapitsa bench` results, from bench/community/results/*.json.

  python3 tools/community_table.py [--results DIR] [--out FILE] [--check]

Grouped by Mac (Apple's name for it, the chip and its GPU cores, the memory), then by model and precision; one row per
run, newest first, every rate in tok/s and every time in seconds, so that each number stands under the Mac, the model
and the precision it was measured with. A measurement that other apps disturbed (contention flagged) carries ⚠, and
each row lists its run's flags. Quick runs say so. Files that are not results (or fail the schema's basics) are skipped
with a warning; tools/community_check.py is the full check, run on every pull request. --check: exit 1 when RESULTS.md
is not what this would write (it changes nothing).
"""
import argparse, functools, glob, json, os, re, statistics, sys

import community_check  # the catalog, from the same folder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
COMMUNITY = os.path.join(REPO, "bench", "community")
MODEL_ORDER = ["qwen3.5-4b", "qwen3.8-27b", "qwen3.6-35b-a3b", "nemotron-3.5-lightning", "nemotron-3.5-bolt",
               "nemotron-3.5-super-vl"]
VARIANT_ORDER = ["4bit", "5bit", "6bit", "8bit", "bf16", "24gb"]
WARN = "⚠"


def chip_short(chip):
    return chip[6:] if chip.startswith("Apple ") else chip


def mac_name(mac):
    """Apple's name, or for a name with the chip instead of a year (the M5 Macs' "MacBook Pro (14-inch, M5)") the year
    instead of the chip, as `kapitsa bench` shows it: the chip is shown beside it"""
    name, year = mac.get("name"), mac.get("year")
    if not name:
        return mac["identifier"]
    m = re.fullmatch(r"(.*)\((.*)\)", name)
    if not m or not year:
        return name
    parts = m.group(2).split(", ")
    if any(re.search(r"\b20\d\d\b", p) for p in parts):
        return name
    chip = re.compile(r"M\d+( (Pro|Max|Ultra))?( or M\d+( (Pro|Max|Ultra))?)*")
    parts = [p for p in parts if not chip.fullmatch(p)] + [str(year)]
    return f"{m.group(1)}({', '.join(parts)})"


@functools.lru_cache(maxsize=None)
def catalog_titles():
    return {name: m["title"] for name, m in community_check.catalog().items()}


def model_name(m):
    # the catalog's title, so a result from a release that titled the model differently sits under today's name
    t = catalog_titles().get(m["name"]) or m.get("title") or m["name"]
    return t.replace("-", " ", 1) if t.startswith("Qwen") else t


def tps(v, integer=False):
    if v is None:
        return "–"
    return f"{round(v):,} tok/s" if integer else f"{v:.1f} tok/s"


def secs(v):
    return "–" if v is None else f"{v:.2f} s"


def short_flags(r):
    """A run's flags in a few words each"""
    out = []
    for f in r.get("flags", []):
        if f.startswith("CONTENTION"):
            tag = "contention (" + ", ".join(r.get("conditions", {}).get("contention", [])) + ")"
        elif f.startswith("power: on battery"):
            tag = "on battery"
        elif f.startswith("power: Low Power"):
            tag = "Low Power Mode"
        elif f.startswith("thermal:"):
            tag = "thermal " + f.split()[1]
        elif "swapped out" in f:
            tag = "swap"
        elif "repetition loop" in f:
            tag = "repetition loop"
        elif "finish_reason" in f or "completion_tokens" in f:
            tag = "short answer"
        elif "reused only" in f:
            tag = "follow-up cache miss"
        else:
            tag = f.split(":")[0]
        if tag not in out:
            out.append(tag)
    return out


def load(results_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
            assert r.get("kind") == "kapitsa-bench" and r.get("schema") == 1
            r["mac"]["identifier"], r["model"]["name"], r["results"]["prefill"]
        except Exception as e:
            print(f"skipped {os.path.relpath(p, REPO)}: not a kapitsa bench result ({e!r})", file=sys.stderr)
            continue
        out.append((p, r))
    return out


def render(results, results_dir, out_path):
    rel = os.path.relpath(results_dir, os.path.dirname(out_path))
    lines = ["<!-- Written by tools/community_table.py from bench/community/results/*.json: do not edit by hand. -->", "",
             "# Community results", "",
             "Kapitsa measured on people's own Macs with `kapitsa bench`: the protocol of the published benchmarks "
             "([bench/harness](../harness/README.md)), as a client of the local server. Add yours with "
             "`kapitsa bench --submit`; [README.md](README.md) says how runs are checked.", "",
             "Writing: tokens per second of a 512-token coding answer, median of the runs (5, or 3 in a quick run), at "
             "temperature 0.6 (Flint at 1.0, its model card's) and greedy. Reading: a cold prompt of about 4K and 16K "
             "tokens, prompt tokens per second to its first token. Follow-up: the time to the first token of a new message "
             "in that conversation. "
             "Memory: the server's peak, GB of 2^30 bytes (a quick run reads no 16K prompt, so its peak is lower: about 0.4 GB "
             f"on the 4B). {WARN}: other apps used the GPU during that measurement "
             "(contention flagged), so it may be low. Different Macs, macOS versions and Kapitsa versions give "
             "different numbers; these are what each run measured.", ""]
    if not results:
        lines += ["No results yet. Add the first with `kapitsa bench --submit`.", ""]
        return "\n".join(lines)
    macs = {}
    for p, r in results:
        m = r["mac"]
        key = (mac_name(m), chip_short(m["chip"]), m.get("gpu_cores"), m["memory_gb"])
        macs.setdefault(key, []).append((p, r))
    def mac_sort(k):
        rs = macs[k]
        return (-max(r["mac"].get("year") or 0 for _, r in rs), k[0], k[1], k[3])
    keys = sorted(macs, key=mac_sort)
    lines += ["| Mac | Runs |", "| --- | ---: |"]
    for k in keys:
        title = f"{k[0]} · {k[1]}" + (f" {k[2]}-core GPU" if k[2] else "") + f" · {k[3]} GB"
        anchor = re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")
        lines.append(f"| [{title}](#{anchor}) | {len(macs[k])} |")
    lines.append("")
    for k in keys:
        title = f"{k[0]} · {k[1]}" + (f" {k[2]}-core GPU" if k[2] else "") + f" · {k[3]} GB"
        lines += [f"## {title}", ""]
        models = {}
        for p, r in macs[k]:
            models.setdefault((r["model"]["name"], r["model"].get("variant") or ""), []).append((p, r))
        order = lambda mk: (MODEL_ORDER.index(mk[0]) if mk[0] in MODEL_ORDER else 99,
                            VARIANT_ORDER.index(mk[1]) if mk[1] in VARIANT_ORDER else 99, mk)
        for mk in sorted(models, key=order):
            rows = sorted(models[mk], key=lambda pr: pr[1]["time_utc"], reverse=True)
            m0 = rows[0][1]["model"]
            t = community_check.TEMPERATURE.get(mk[0], 0.6)
            lines += [f"### {model_name(m0)} · {m0['precision']}", "",
                      f"| Run | Writing, T = {t:.1f} | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | "
                      "Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |"]
            for p, r in rows:
                res, busy = r["results"], set(r.get("conditions", {}).get("contention", []))
                mark = lambda phase: f" {WARN}" if phase in busy else ""
                pre = {x["size"]: x for x in res["prefill"]}
                def dec(key):
                    d = res.get(key)
                    if not d or d.get("median_tps") is None:
                        return "–"
                    return tps(d["median_tps"]) + mark(key)
                cells = [f"[{r['time_utc'][:10]}]({rel}/{os.path.basename(p)})" + (" quick" if r["mode"] == "quick" else ""),
                         dec("decode"), dec("decode_greedy")]
                for size in (4000, 16000):
                    x = pre.get(size)
                    cells.append(tps(x["tps"], True) + mark(f"prefill_{size}") if x and x.get("tps") else "–")
                for size in (4000, 16000):
                    x = pre.get(size)
                    cells.append(secs(x["followup_median_s"]) + mark(f"prefill_{size}") if x and x.get("followup_median_s") else "–")
                mem = res.get("peak_memory_gb")
                cells.append("–" if mem is None else f"{mem:.1f} GB" + (f" + {res['mapped_gb']:.1f} mapped" if res.get("mapped_gb") else ""))
                cells.append(res["tool_calls"] + mark("tools"))
                cells.append(r["model"]["decoding"])
                cells.append(r["kapitsa"]["version"])
                cells.append(r["mac"]["macos"])
                fl = short_flags(r)
                cells.append(", ".join(fl) if fl else "–")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(COMMUNITY, "results"))
    ap.add_argument("--out", default=os.path.join(COMMUNITY, "RESULTS.md"))
    ap.add_argument("--check", action="store_true", help="exit 1 when the file is not up to date; write nothing")
    a = ap.parse_args()
    results = load(a.results)
    text = render(results, os.path.abspath(a.results), os.path.abspath(a.out))
    if a.check:
        have = open(a.out, encoding="utf-8").read() if os.path.exists(a.out) else ""
        print("up to date" if have == text else f"{os.path.relpath(a.out)} is not up to date: python3 tools/community_table.py")
        return 0 if have == text else 1
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"{os.path.relpath(a.out)}: {len(results)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
